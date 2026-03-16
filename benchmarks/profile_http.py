#!/usr/bin/env python3
"""Profile where time is spent in a containerized HTTP apply call.

Instruments both client-side and server-side to measure:
  - Client: array encoding, JSON serialization, HTTP send, response wait,
            JSON deserialization, array decoding
  - Server: total request handling time (via Server-Timing header middleware)

Usage:
    python benchmarks/profile_http.py [--array-size N] [--rounds N]
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def create_test_array(size: int) -> np.ndarray:
    return np.random.default_rng(42).standard_normal(size).astype("float64")


def build_noop_image() -> str:
    image_name = "benchmark-noop:latest"
    tesseract_dir = Path(__file__).parent / "tesseract_noop"
    result = subprocess.run(
        [
            "tesseract",
            "build",
            str(tesseract_dir),
            "--config-override",
            "name=benchmark-noop",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        print(f"Failed to build noop tesseract: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    return image_name


def profile_apply(tesseract, array_size: int, rounds: int) -> None:
    """Run profiled apply calls and report timing breakdown."""
    from pydantic_core import from_json, to_json

    from tesseract_core.sdk.tesseract import (
        _decode_array,
        _encode_array,
        _tree_map,
    )

    arr = create_test_array(array_size)
    inputs = {"data": arr}
    payload = {"inputs": inputs}

    client = tesseract._client
    url = f"{client.url}/apply"
    session = client._session

    timings: dict[str, list[float]] = {
        "client_encode_arrays": [],
        "client_json_serialize": [],
        "client_http_roundtrip": [],
        "client_json_deserialize": [],
        "client_decode_arrays": [],
        "total_client": [],
        "server_total": [],
    }

    # Measure baseline Docker network latency with tiny payload (health endpoint)
    health_url = f"{client.url}/health"
    for _ in range(5):  # warmup
        session.request(method="GET", url=health_url)
    health_times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        session.request(method="GET", url=health_url)
        t1 = time.perf_counter()
        health_times.append(t1 - t0)
    timings["health_roundtrip"] = health_times

    # Warmup
    tesseract.apply(inputs)
    tesseract.apply(inputs)

    for _ in range(rounds):
        t_total_start = time.perf_counter()

        # 1. Encode arrays (base64)
        t0 = time.perf_counter()
        encoded_payload = _tree_map(
            _encode_array, payload, is_leaf=lambda x: hasattr(x, "shape")
        )
        t1 = time.perf_counter()
        timings["client_encode_arrays"].append(t1 - t0)

        # 2. JSON serialize
        t0 = time.perf_counter()
        json_body = to_json(encoded_payload)
        t1 = time.perf_counter()
        timings["client_json_serialize"].append(t1 - t0)
        json_body_size = len(json_body)

        # 3. HTTP roundtrip (send + server processing + receive)
        t0 = time.perf_counter()
        response = session.request(method="POST", url=url, data=json_body)
        t1 = time.perf_counter()
        timings["client_http_roundtrip"].append(t1 - t0)

        # Extract server-side timing from headers
        server_timing = response.headers.get("Server-Timing")
        if server_timing:
            for part in server_timing.split(","):
                part = part.strip()
                if ";" in part:
                    name, _, params = part.partition(";")
                    for param in params.split(";"):
                        param = param.strip()
                        if param.startswith("dur="):
                            dur_ms = float(param[4:])
                            key = f"server_{name.strip()}"
                            timings.setdefault(key, [])
                            timings[key].append(dur_ms / 1000.0)

        # Extract detailed server timing headers
        for hdr, key in [
            ("X-Server-Body-Receive-Ms", "server_body_receive"),
            ("X-Server-Compute-Ms", "server_compute"),
            ("X-Server-Serialize-Ms", "server_serialize"),
        ]:
            val = response.headers.get(hdr)
            if val:
                timings.setdefault(key, [])
                timings[key].append(float(val) / 1000.0)

        # 4. JSON deserialize
        t0 = time.perf_counter()
        data = from_json(response.content)
        t1 = time.perf_counter()
        timings["client_json_deserialize"].append(t1 - t0)
        response_size = len(response.content)

        # 5. Decode arrays
        t0 = time.perf_counter()

        def decode_with_path(arr_dict: dict) -> np.ndarray:
            return _decode_array(arr_dict, output_path=client._output_path)

        _tree_map(
            decode_with_path,
            data,
            is_leaf=lambda x: type(x) is dict and "shape" in x,
        )
        t1 = time.perf_counter()
        timings["client_decode_arrays"].append(t1 - t0)

        t_total_end = time.perf_counter()
        timings["total_client"].append(t_total_end - t_total_start)

    # Report
    raw_bytes = array_size * 8  # float64
    b64_bytes = (raw_bytes * 4 + 2) // 3  # base64 expansion

    print(f"\n{'=' * 70}")
    print(f"Profile: test_containerized_http [{array_size:,}]")
    print(
        f"Array: {array_size:,} float64 = {raw_bytes / 1e6:.1f} MB raw, ~{b64_bytes / 1e6:.1f} MB base64"
    )
    print(f"Rounds: {rounds}")
    print(f"{'=' * 70}")
    print(f"{'Phase':<35} {'Mean (ms)':>10} {'Std (ms)':>10} {'% of total':>10}")
    print(f"{'-' * 35} {'-' * 10} {'-' * 10} {'-' * 10}")

    total_mean = statistics.mean(timings["total_client"]) * 1000

    # Print client-side phases
    client_phases = [
        ("client_encode_arrays", "Client: encode arrays (b64)"),
        ("client_json_serialize", "Client: JSON serialize"),
        ("client_http_roundtrip", "Client: HTTP roundtrip"),
        ("client_json_deserialize", "Client: JSON deserialize"),
        ("client_decode_arrays", "Client: decode arrays (b64)"),
    ]

    for key, label in client_phases:
        vals_ms = [v * 1000 for v in timings[key]]
        mean = statistics.mean(vals_ms)
        std = statistics.stdev(vals_ms) if len(vals_ms) > 1 else 0
        pct = mean / total_mean * 100
        print(f"  {label:<33} {mean:>10.2f} {std:>10.2f} {pct:>9.1f}%")

    print(f"{'-' * 35} {'-' * 10} {'-' * 10} {'-' * 10}")
    total_std = (
        statistics.stdev([v * 1000 for v in timings["total_client"]])
        if rounds > 1
        else 0
    )
    print(
        f"  {'TOTAL (client-measured)':<33} {total_mean:>10.2f} {total_std:>10.2f} {'100.0':>9}%"
    )

    # Server-side breakdown
    has_server = "server_total" in timings
    has_detail = "server_compute" in timings and "server_serialize" in timings

    if has_server:
        print("\n  Server-side breakdown (from response headers):")
        print(f"  {'-' * 33} {'-' * 10} {'-' * 10} {'-' * 10}")

        srv_total_ms = statistics.mean(timings["server_total"]) * 1000
        srv_total_std = (
            statistics.stdev([v * 1000 for v in timings["server_total"]])
            if rounds > 1
            else 0
        )

        if has_detail:
            compute_ms = statistics.mean(timings["server_compute"]) * 1000
            compute_std = (
                statistics.stdev([v * 1000 for v in timings["server_compute"]])
                if rounds > 1
                else 0
            )
            serialize_ms = statistics.mean(timings["server_serialize"]) * 1000
            serialize_std = (
                statistics.stdev([v * 1000 for v in timings["server_serialize"]])
                if rounds > 1
                else 0
            )

            # Input handling = total - compute - serialize
            input_total_ms = srv_total_ms - compute_ms - serialize_ms

            has_body_receive = "server_body_receive" in timings
            if has_body_receive:
                body_recv_ms = statistics.mean(timings["server_body_receive"]) * 1000
                body_recv_std = (
                    statistics.stdev([v * 1000 for v in timings["server_body_receive"]])
                    if rounds > 1
                    else 0
                )
                pydantic_ms = input_total_ms - body_recv_ms

                print(
                    f"    {'Srv: body receive (HTTP)':<31} {body_recv_ms:>10.2f} {body_recv_std:>10.2f} {body_recv_ms / total_mean * 100:>9.1f}%"
                )
                print(
                    f"    {'Srv: input deser (pydantic)':<31} {pydantic_ms:>10.2f} {'~':>10} {pydantic_ms / total_mean * 100:>9.1f}%"
                )
            else:
                print(
                    f"    {'Srv: body recv + input deser':<31} {input_total_ms:>10.2f} {'~':>10} {input_total_ms / total_mean * 100:>9.1f}%"
                )

            print(
                f"    {'Srv: endpoint + apply()':<31} {compute_ms:>10.2f} {compute_std:>10.2f} {compute_ms / total_mean * 100:>9.1f}%"
            )
            print(
                f"    {'Srv: output ser (pydantic)':<31} {serialize_ms:>10.2f} {serialize_std:>10.2f} {serialize_ms / total_mean * 100:>9.1f}%"
            )
            print(f"  {'-' * 33} {'-' * 10} {'-' * 10} {'-' * 10}")

        pct = srv_total_ms / total_mean * 100
        print(
            f"    {'Srv: TOTAL':<31} {srv_total_ms:>10.2f} {srv_total_std:>10.2f} {pct:>9.1f}%"
        )

        # Network overhead = HTTP roundtrip - server total
        http_mean = statistics.mean(timings["client_http_roundtrip"]) * 1000
        net_overhead = http_mean - srv_total_ms
        net_pct = net_overhead / total_mean * 100

        # Break down network overhead using health baseline
        health_ms = statistics.mean(timings["health_roundtrip"]) * 1000
        health_std = (
            statistics.stdev([v * 1000 for v in timings["health_roundtrip"]])
            if rounds > 1
            else 0
        )
        data_transit_ms = net_overhead - health_ms

        print("\n  Network overhead breakdown:")
        print(f"  {'-' * 33} {'-' * 10} {'-' * 10} {'-' * 10}")
        print(
            f"    {'Fixed latency (health RTT)':<31} {health_ms:>10.2f} {health_std:>10.2f} {health_ms / total_mean * 100:>9.1f}%"
        )
        total_transit_mb = (json_body_size + response_size) / 1e6
        print(
            f"    {f'Data in-flight (~{total_transit_mb:.1f} MB)':<31} {data_transit_ms:>10.2f} {'~':>10} {data_transit_ms / total_mean * 100:>9.1f}%"
        )
        print(
            f"    {'TOTAL network overhead':<31} {net_overhead:>10.2f} {'':>10} {net_pct:>9.1f}%"
        )

    print()


def main():
    parser = argparse.ArgumentParser(description="Profile containerized HTTP apply")
    parser.add_argument("--array-size", type=int, default=1_000_000)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--skip-build", action="store_true", help="Skip image build")
    args = parser.parse_args()

    import tempfile

    from tesseract_core.sdk.tesseract import Tesseract

    if not args.skip_build:
        print("Building noop tesseract image...")
        image_name = build_noop_image()
    else:
        image_name = "benchmark-noop:latest"

    print(f"Starting container for {image_name}...")
    tmpdir = tempfile.mkdtemp(prefix="profile_http_")
    cm = Tesseract.from_image(image_name, output_path=tmpdir)
    tesseract = cm.__enter__()

    try:
        # Warmup container
        tesseract.health()
        print("Container ready. Profiling...")

        profile_apply(tesseract, args.array_size, args.rounds)
    finally:
        cm.__exit__(None, None, None)


if __name__ == "__main__":
    main()
