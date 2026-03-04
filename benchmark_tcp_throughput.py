#!/usr/bin/env python3
"""Benchmark TCP loopback throughput between host and Docker container.

Builds a minimal Docker image with a raw TCP echo server, then measures
round-trip throughput for payloads of increasing size.

Usage:
    python3 benchmark_tcp_throughput.py
"""

import socket
import struct
import subprocess
import sys
import textwrap
import time

HOST_PORT = 19876
CONTAINER_PORT = 9876

# (label, size_bytes, iterations)
PAYLOAD_SIZES = [
    ("1 KB", 1_000, 1000),
    ("10 KB", 10_000, 500),
    ("100 KB", 100_000, 100),
    ("1 MB", 1_000_000, 30),
    ("10 MB", 10_000_000, 10),
    ("100 MB", 100_000_000, 3),
]

IMAGE_NAME = "tcp-echo-bench"

SERVER_SCRIPT = textwrap.dedent("""\
    import socket, struct

    def recvn(sock, n):
        data = bytearray()
        while len(data) < n:
            chunk = sock.recv(min(n - len(data), 1 << 20))
            if not chunk:
                raise ConnectionError("connection closed")
            data.extend(chunk)
        return bytes(data)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", 9876))
    srv.listen(1)
    print("echo server listening", flush=True)
    conn, addr = srv.accept()
    # Set large buffer sizes
    conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
    while True:
        try:
            hdr = recvn(conn, 4)
        except ConnectionError:
            break
        length = struct.unpack("!I", hdr)[0]
        if length == 0:
            break
        payload = recvn(conn, length)
        conn.sendall(hdr + payload)
    conn.close()
    srv.close()
""")

DOCKERFILE = textwrap.dedent("""\
    FROM python:3.12-slim
    COPY server.py /server.py
    CMD ["python", "-u", "/server.py"]
""")


def build_image():
    """Build the echo server Docker image."""
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "server.py"), "w") as f:
            f.write(SERVER_SCRIPT)
        with open(os.path.join(tmpdir, "Dockerfile"), "w") as f:
            f.write(DOCKERFILE)
        subprocess.run(
            ["docker", "build", "-t", IMAGE_NAME, "-q", tmpdir],
            check=True,
            capture_output=True,
        )


def start_container():
    """Start the echo server container, return container ID."""
    result = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "-p",
            f"127.0.0.1:{HOST_PORT}:{CONTAINER_PORT}",
            "--name",
            "tcp-echo-bench-run",
            IMAGE_NAME,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def stop_container(container_id):
    # Print logs before stopping for debugging
    logs = subprocess.run(
        ["docker", "logs", container_id], capture_output=True, text=True
    )
    if logs.stdout.strip():
        print(f"\n[container stdout]: {logs.stdout.strip()}", file=sys.stderr)
    if logs.stderr.strip():
        print(f"[container stderr]: {logs.stderr.strip()}", file=sys.stderr)
    subprocess.run(["docker", "stop", container_id], capture_output=True)
    subprocess.run(["docker", "rm", container_id], capture_output=True)


def connect_to_server(timeout=15):
    """Wait until the TCP echo server is ready and return a connected socket."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            s = socket.create_connection(("127.0.0.1", HOST_PORT), timeout=2)
            # Verify the server is actually echoing (not just Docker proxy accepting)
            s.settimeout(2)
            ping = struct.pack("!I", 4) + b"ping"
            s.sendall(ping)
            resp = s.recv(8)
            if resp == ping:
                s.settimeout(None)
                return s
            s.close()
        except (ConnectionRefusedError, ConnectionError, OSError, TimeoutError):
            pass
        time.sleep(0.3)
    raise TimeoutError("echo server did not start in time")


def recvn(sock, n):
    data = bytearray()
    while len(data) < n:
        chunk = sock.recv(min(n - len(data), 1 << 20))
        if not chunk:
            raise ConnectionError("connection closed")
        data.extend(chunk)
    return bytes(data)


def run_benchmark(sock, payload_size, iterations):
    """Send payload and receive echo, return (total_bytes, elapsed_seconds)."""
    payload = bytes(range(256)) * (payload_size // 256 + 1)
    payload = payload[:payload_size]
    header = struct.pack("!I", payload_size)

    # Warmup
    sock.sendall(header + payload)
    resp_hdr = recvn(sock, 4)
    resp_len = struct.unpack("!I", resp_hdr)[0]
    recvn(sock, resp_len)

    total_bytes = 0
    start = time.perf_counter()
    for _ in range(iterations):
        sock.sendall(header + payload)
        resp_hdr = recvn(sock, 4)
        resp_len = struct.unpack("!I", resp_hdr)[0]
        recvn(sock, resp_len)
        total_bytes += payload_size * 2  # sent + received
    elapsed = time.perf_counter() - start
    return total_bytes, elapsed


def format_size(n):
    if n >= 1e9:
        return f"{n / 1e9:.1f} GB"
    if n >= 1e6:
        return f"{n / 1e6:.1f} MB"
    if n >= 1e3:
        return f"{n / 1e3:.0f} KB"
    return f"{n} B"


def main():
    print("Building echo server image...")
    build_image()

    print("Starting container...")
    container_id = start_container()

    try:
        print("Connecting to server...")
        sock = connect_to_server()
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print("Connected.")

        print()
        hdr = f"{'Payload':<12} {'Iters':>6} {'Throughput':>14} {'Avg Latency':>14} {'Total':>10}"
        print(hdr)
        print("-" * len(hdr))

        for label, size, iters in PAYLOAD_SIZES:
            total_bytes, elapsed = run_benchmark(sock, size, iters)
            throughput_mbs = (total_bytes / elapsed) / 1e6
            avg_latency_ms = (elapsed / iters) * 1000
            print(
                f"{label:<12} {iters:>6} {throughput_mbs:>11.1f} MB/s {avg_latency_ms:>11.3f} ms {format_size(total_bytes):>10}"
            )

        # Signal server to stop
        sock.sendall(struct.pack("!I", 0))
        sock.close()

    finally:
        print("\nStopping container...")
        stop_container(container_id)

    print("Done.")


if __name__ == "__main__":
    main()
