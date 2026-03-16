#!/usr/bin/env python3
"""Microbenchmark: client-side serde vs server-side serde for 1M float64 array.

Runs everything in-process (no Docker/HTTP) to isolate pure serde costs.
"""

import time

import numpy as np
import pybase64
from pydantic_core import from_json, to_json


def bench(label, func, rounds=20):
    # warmup
    for _ in range(3):
        func()
    times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        func()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    import statistics

    mean = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0
    print(f"  {label:<45} {mean:>8.2f} ms  (std {std:.2f})")
    return mean


def main():
    arr = np.random.default_rng(42).standard_normal(1_000_000).astype("float64")
    raw_bytes = arr.nbytes
    print(f"Array: 1,000,000 float64 = {raw_bytes / 1e6:.1f} MB\n")

    # === CLIENT-SIDE ENCODE ===
    print("CLIENT-SIDE (encode path):")

    def client_b64_encode():
        return pybase64.b64encode_as_string(np.ascontiguousarray(arr).data)

    bench("pybase64.b64encode_as_string", client_b64_encode)

    # Build the encoded payload once
    encoded_payload = {
        "inputs": {
            "data": {
                "shape": list(arr.shape),
                "dtype": arr.dtype.name,
                "data": {
                    "buffer": pybase64.b64encode_as_string(
                        np.ascontiguousarray(arr).data
                    ),
                    "encoding": "base64",
                },
            }
        }
    }

    def client_to_json():
        return to_json(encoded_payload)

    bench("pydantic_core.to_json (plain dict)", client_to_json)

    json_bytes = to_json(encoded_payload)

    def client_full_encode():
        b64 = pybase64.b64encode_as_string(np.ascontiguousarray(arr).data)
        payload = {
            "inputs": {
                "data": {
                    "shape": list(arr.shape),
                    "dtype": arr.dtype.name,
                    "data": {"buffer": b64, "encoding": "base64"},
                }
            }
        }
        return to_json(payload)

    bench("Full client encode (b64 + to_json)", client_full_encode)

    # === CLIENT-SIDE DECODE ===
    print("\nCLIENT-SIDE (decode path):")

    def client_from_json():
        return from_json(json_bytes)

    bench("pydantic_core.from_json", client_from_json)

    data = from_json(json_bytes)

    def client_b64_decode():
        buf = pybase64.b64decode(data["inputs"]["data"]["data"]["buffer"])
        return np.frombuffer(buf, dtype="float64").reshape((1_000_000,))

    bench("pybase64.b64decode + frombuffer", client_b64_decode)

    # === SERVER-SIDE (Pydantic model validation) ===
    print("\nSERVER-SIDE (Pydantic model_validate_json):")

    # Import the actual schemas used by the server
    from benchmarks.tesseract_noop.tesseract_api import InputSchema, OutputSchema
    from tesseract_core.runtime.schema_generation import create_apply_schema

    ApplyInputSchema, ApplyOutputSchema = create_apply_schema(InputSchema, OutputSchema)

    def server_validate_json():
        return ApplyInputSchema.model_validate_json(json_bytes)

    t_validate = bench("ApplyInputSchema.model_validate_json", server_validate_json)

    # Break down: JSON parse only vs Pydantic validation
    def server_json_parse_only():
        return from_json(json_bytes)

    t_parse = bench(
        "from_json only (JSON parse, no validation)", server_json_parse_only
    )

    print(
        f"\n  --> Pydantic overhead (validate - parse):    {t_validate - t_parse:>8.2f} ms"
    )

    # Now measure output serialization
    print("\nSERVER-SIDE (output serialization):")
    validated = ApplyInputSchema.model_validate_json(json_bytes)
    result = ApplyOutputSchema.model_validate({"result": validated.inputs.data})

    from pydantic import TypeAdapter

    ObjSchema = TypeAdapter(type(result))

    def server_dump_json():
        return ObjSchema.dump_json(
            result, context={"array_encoding": "base64"}, exclude_unset=True
        )

    bench("TypeAdapter.dump_json (base64)", server_dump_json)

    # Isolate just b64 encode for comparison
    def raw_b64_encode():
        return pybase64.b64encode_as_string(np.ascontiguousarray(arr).data)

    bench("raw pybase64.b64encode_as_string", raw_b64_encode)


if __name__ == "__main__":
    main()
