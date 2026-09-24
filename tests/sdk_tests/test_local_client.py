# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for serving Tesseracts in dedicated subprocesses.

These spawn real ``tesseract-runtime serve`` processes (but no containers), so
they exercise the actual startup / health-check / removal path.
"""

import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest
import requests

from tesseract_core import Tesseract
from tesseract_core.sdk import local_client, serving, venv_provision
from tesseract_core.sdk.api_parse import get_config
from tesseract_core.sdk.exceptions import UserError

pytestmark = pytest.mark.timeout(120)


def _process_alive(pid: int) -> bool:
    """Whether `pid` still exists, without reaping or signalling it."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_serve_and_remove(dummy_api_path):
    served = local_client.serve(dummy_api_path, python_executable=sys.executable)
    try:
        assert local_client.is_running(served)
        assert served.port != 0
    finally:
        logs = served.logs().decode()
        served.remove(force=True)

    assert not local_client.is_running(served)
    if os.name != "nt":
        # Windows refuses to delete a file a just-killed child may still hold, so
        # removal tolerates that and leaves it in the temp directory.
        assert not served.log_path.exists()
    # uvicorn announces itself on startup; if we captured nothing, log capture
    # is broken even though the health check passed.
    assert logs.strip()


def test_wait_reports_the_exit_code(dummy_api_path):
    """`wait` answers with a StatusCode dict, as a container's does."""
    served = local_client.serve(dummy_api_path, python_executable=sys.executable)
    served.remove(force=True)

    assert served.wait(timeout=5)["StatusCode"] is not None


def test_wait_times_out_on_a_running_process(dummy_api_path):
    """Waiting on a live Tesseract gives up rather than blocking for its lifetime."""
    served = local_client.serve(dummy_api_path, python_executable=sys.executable)
    try:
        with pytest.raises(TimeoutError, match="still running"):
            served.wait(timeout=0.2)
    finally:
        served.remove(force=True)


@pytest.mark.skipif(os.name == "nt", reason="POSIX signals")
def test_orphaned_tesseract_shuts_itself_down(dummy_api_path, tmp_path):
    """A Tesseract must not outlive the process that served it.

    `remove` covers an orderly exit, but nothing runs when the parent is killed
    outright -- and a served subprocess is otherwise invisible and immortal,
    unlike a container, which at least still shows up in `docker ps -a`.

    Signalling it is not an alternative: SIGKILL leaves the parent no chance to
    send anything, and a terminal's SIGINT goes to the foreground process group,
    which the Tesseract is deliberately not in. Hence the watch pipe in
    `tesseract_core.runtime.cli._exit_when_parent_closes`; stubbing
    `parent_watch_pipe` out fails this test.
    """
    helper = tmp_path / "helper.py"
    helper.write_text(
        textwrap.dedent(f"""
        import sys
        import time
        from tesseract_core.sdk import local_client

        served = local_client.serve({str(dummy_api_path)!r}, python_executable=sys.executable)
        print(served.process.pid, flush=True)
        time.sleep(600)
        """)
    )
    with subprocess.Popen(
        [sys.executable, str(helper)], stdout=subprocess.PIPE, text=True
    ) as parent:
        try:
            tesseract_pid = int(parent.stdout.readline().strip())
            assert _process_alive(tesseract_pid)

            parent.kill()  # no cleanup runs, so only the Tesseract can react
            parent.wait(timeout=10)

            deadline = time.monotonic() + 15
            while _process_alive(tesseract_pid) and time.monotonic() < deadline:
                time.sleep(0.2)
            assert not _process_alive(tesseract_pid), (
                "Tesseract outlived the process that served it"
            )
        finally:
            parent.kill()
            # If the mechanism under test is broken, the Tesseract is orphaned --
            # do not leave behind the very thing this is here to prevent.
            if _process_alive(tesseract_pid):
                os.kill(tesseract_pid, signal.SIGKILL)


def test_remove_refuses_a_running_process(dummy_api_path):
    """Unforced removal refuses a live Tesseract, as removing a container does."""
    served = local_client.serve(dummy_api_path, python_executable=sys.executable)
    try:
        with pytest.raises(RuntimeError, match="still running"):
            served.remove()
        assert local_client.is_running(served)
    finally:
        served.remove(force=True)


def test_remove_is_idempotent(dummy_api_path):
    served = local_client.serve(dummy_api_path, python_executable=sys.executable)
    served.remove(force=True)
    # Must not raise, even though the process and its log file are gone
    served.remove(force=True)
    if os.name != "nt":
        # The logs really are gone, as they are for a removed container. Except
        # on Windows, which refuses to delete a file a just-killed child may
        # still hold -- removal tolerates that and leaves it behind.
        with pytest.raises(FileNotFoundError):
            served.logs()


# Nothing listens on port 1, so the health poll fails at once rather than
# racing a Tesseract that may already have come up. These tests are about
# what happens after the poll gives up, not about the poll.
_DEAD_URL = "http://127.0.0.1:1"


@pytest.mark.skipif(
    os.name == "nt",
    reason="Windows will not unlink a file the live child still holds open",
)
def test_unreadable_logs_do_not_mask_a_subprocess_startup_failure(dummy_api_path):
    """A local Tesseract fails in its own vocabulary, not Docker's.

    The startup path used to catch only `APIError`, so the `FileNotFoundError`
    a vanished log file raises escaped and replaced the failure it was being
    read to explain.

    POSIX only, because the premise needs a log file deleted from under a
    running process, which Windows refuses -- see `TesseractProcess.remove`.
    `test_any_unreadable_log_does_not_mask_the_startup_failure` makes the same
    point everywhere, without needing the file to actually go away.
    """
    served = local_client.serve(
        dummy_api_path, python_executable=sys.executable, skip_health_check=True
    )
    try:
        # Really stopped, rather than pretended so by patching `is_running`:
        # otherwise the startup path asks a live process for its exit code and
        # waits the full health timeout to be told it has none.
        served.process.terminate()
        served.process.wait(timeout=30)
        served.log_path.unlink()

        with pytest.raises((RuntimeError, TimeoutError)) as excinfo:
            serving.wait_for_health_or_dispose(served, _DEAD_URL, timeout=0.05)
        assert "stopped running during startup" in str(excinfo.value)
    finally:
        served.remove(force=True)


def test_any_unreadable_log_does_not_mask_the_startup_failure(dummy_api_path):
    """Whatever reading the logs raises, the startup failure is what surfaces.

    The point of catching broadly rather than naming a transport's exceptions:
    reading the logs is how the failure gets reported, so it must not become the
    failure. Runs on every platform, unlike the vanished-file case above.
    """
    served = local_client.serve(
        dummy_api_path, python_executable=sys.executable, skip_health_check=True
    )
    try:
        served.process.terminate()
        served.process.wait(timeout=30)

        def unreadable():
            raise RuntimeError("log storage is on fire")

        served.logs = unreadable
        with pytest.raises((RuntimeError, TimeoutError)) as excinfo:
            serving.wait_for_health_or_dispose(served, _DEAD_URL, timeout=0.05)
        assert "stopped running during startup" in str(excinfo.value)
        assert "on fire" not in str(excinfo.value)
    finally:
        del served.logs
        served.remove(force=True)


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({}, "none"),
        ({"gpu_transport": "cuda_ipc"}, "cuda_ipc"),
        ({"runtime_config": {"gpu_transport": "cuda_ipc"}}, "cuda_ipc"),
        # An explicit value wins, including when it is the default.
        (
            {"gpu_transport": "none", "runtime_config": {"gpu_transport": "cuda_ipc"}},
            "none",
        ),
    ],
)
def test_gpu_transport_reaches_child_and_client_alike(dummy_api_path, kwargs, expected):
    """Both ends have to agree, or inputs and outputs cross differently.

    The child reads it from its environment and the client from the spawn
    config, by two separate paths, so this pins that they resolve to the same
    thing under each way of setting it. Does not exercise the transport itself,
    which needs a GPU.
    """
    tess = Tesseract.from_source(
        dummy_api_path, python_executable=sys.executable, **kwargs
    )
    runtime_config = tess._spawn_config["runtime_config"]

    child = serving.runtime_config_to_env(runtime_config).get("TESSERACT_GPU_TRANSPORT")
    client = tess._spawn_config.get("gpu_transport") or runtime_config.get(
        "gpu_transport", "none"
    )
    assert child == expected
    assert client == expected


def test_gpu_transport_rejects_an_unknown_value(dummy_api_path):
    with pytest.raises(ValueError, match="Unknown gpu_transport"):
        Tesseract.from_source(
            dummy_api_path, python_executable=sys.executable, gpu_transport="nonsense"
        )


def test_serve_rejects_binref_without_an_output_path(dummy_api_path):
    """The same error as the containerized path, and the same type.

    A caller that moves between the two engines should not have to catch two
    different exceptions for one mistake.
    """
    with pytest.raises(UserError, match=r"json\+binref"):
        local_client.serve(
            dummy_api_path,
            python_executable=sys.executable,
            output_format="json+binref",
        )


def test_serve_rejects_missing_api():
    with pytest.raises(FileNotFoundError, match="is not a file"):
        local_client.serve("/nonexistent/tesseract_api.py")


def test_serve_on_explicit_port(dummy_api_path):
    from tesseract_core.sdk.engine import get_free_port

    port = get_free_port()
    served = local_client.serve(
        dummy_api_path, python_executable=sys.executable, port=port
    )
    try:
        assert served.port == port
        assert served.url.endswith(f":{port}")
    finally:
        served.remove(force=True)


def test_endpoints_over_subprocess(dummy_api_path, sample_inputs):
    with Tesseract.from_source(
        dummy_api_path, python_executable=sys.executable
    ) as tess:
        result = tess.apply(sample_inputs)
        np.testing.assert_allclose(result["result"], [5.0, 8.0])

        assert tess.health()["status"] == "ok"

        # debug mode is on by default, mirroring the in-process path, so the
        # test endpoint must be available
        assert "test" in tess.available_endpoints


def test_runs_in_a_different_process(dummy_tesseract_package, sample_inputs):
    """The whole point: the Tesseract must not share our interpreter."""
    api_path = dummy_tesseract_package / "tesseract_api.py"
    api_path.write_text(
        api_path.read_text()
        + textwrap.dedent(
            """

            def _pid_check(inputs):
                import os
                return os.getpid()
            """
        )
    )

    tess = Tesseract.from_source(api_path)
    with tess:
        process = tess._serve_context.process
        assert process.pid != os.getpid()
        assert process.poll() is None

    assert process.poll() is not None


def test_remove_stops_the_process(dummy_api_path):
    tess = Tesseract.from_source(dummy_api_path, python_executable=sys.executable)
    tess.serve()
    served = tess._serve_context
    assert local_client.is_running(served)

    tess.teardown()

    assert not local_client.is_running(served)
    assert tess._client is None
    assert tess._serve_context is None


def test_logs_are_captured(dummy_api_path):
    tess = Tesseract.from_source(dummy_api_path, python_executable=sys.executable)
    with tess:
        assert "Uvicorn running" in tess.server_logs()

    # Logs remain available after removal
    assert "Uvicorn running" in tess.server_logs()


def test_stream_logs_without_output_path(dummy_api_path, sample_inputs):
    """Streaming must work without the caller specifying an output directory."""
    lines = []

    with Tesseract.from_source(
        dummy_api_path, python_executable=sys.executable, stream_logs=lines.append
    ) as tess:
        tess.apply(sample_inputs)

    # The dummy Tesseract logs nothing itself, so assert on the mechanism having
    # run rather than on content: a missing output path would have raised.
    assert isinstance(lines, list)


def _debug_address(served):
    """Read the debug address a served Tesseract reported binding."""
    match = re.search(r"Debugger listening on ([\d.]+):(\d+)", served.logs().decode())
    return (match.group(1), int(match.group(2))) if match else None


def test_reported_debug_address_honours_an_inherited_runtime_override(
    dummy_api_path, monkeypatch, caplog
):
    """The address we log must be the one the runtime actually binds.

    We never write DEBUGPY_HOST ourselves, so an inherited
    TESSERACT_RUNTIME_DEBUGPY_HOST is what typer hands the runtime -- and it
    outranks the TESSERACT_ spelling. Reading only the latter would send the
    user's debugger to the default while the Tesseract listened elsewhere.
    """
    monkeypatch.setenv("TESSERACT_RUNTIME_DEBUGPY_HOST", "0.0.0.0")
    caplog.set_level(logging.INFO, logger="tesseract")

    served = local_client.serve(
        dummy_api_path, python_executable=sys.executable, runtime_config={"debug": True}
    )
    try:
        bound_host, _ = _debug_address(served)
        assert bound_host == "0.0.0.0", "override did not reach the runtime"

        reported = re.search(r"Attach a debugger to ([\d.]+):(\d+)", caplog.text)
        assert reported is not None, f"no attach address logged: {caplog.text}"
        assert reported.group(1) == bound_host
    finally:
        served.remove(force=True)


def test_debugger_listens_on_loopback_by_default(dummy_api_path):
    """Debugging is the point of a non-containerized Tesseract, so it's on...

    ...but on loopback: unlike a container, there is no network namespace here,
    and debugpy is unauthenticated code execution.
    """
    served = local_client.serve(
        dummy_api_path, python_executable=sys.executable, runtime_config={"debug": True}
    )
    try:
        host, port = _debug_address(served)
        assert host == "127.0.0.1"
        assert port != served.port, "Debugger must not share the API port"
    finally:
        served.remove(force=True)


def test_two_tesseracts_get_distinct_debugpy_ports(dummy_api_path):
    """The whole reason the address is configurable: both must be debuggable."""
    first = local_client.serve(
        dummy_api_path, python_executable=sys.executable, runtime_config={"debug": True}
    )
    second = local_client.serve(
        dummy_api_path, python_executable=sys.executable, runtime_config={"debug": True}
    )
    try:
        assert _debug_address(first) != _debug_address(second)
    finally:
        first.remove(force=True)
        second.remove(force=True)


def test_debugpy_port_collision_recovers_even_with_a_pinned_api_port(
    dummy_api_path, monkeypatch
):
    """A collision on a port we chose must be retried, not surfaced.

    The API port can be pinned by the caller while the debug port is still
    picked automatically. Deciding retriability from the API port alone would
    refuse to recover from a collision of our own making.
    """
    import socket
    from contextlib import closing

    from tesseract_core.sdk.engine import get_free_port

    api_port = get_free_port()
    taken = get_free_port(exclude=(api_port,))
    calls = []

    def fake_get_free_port(within_range=None, exclude=()):
        # Hand out an already-bound port first, a usable one afterwards.
        calls.append(1)
        return taken if len(calls) == 1 else get_free_port(exclude=(api_port, taken))

    monkeypatch.setattr(local_client, "get_free_port", fake_get_free_port)

    with closing(socket.socket()) as occupied:
        occupied.bind(("127.0.0.1", taken))
        occupied.listen(1)

        served = local_client.serve(
            dummy_api_path,
            port=api_port,
            runtime_config={"debug": True},
            python_executable=sys.executable,
        )
        try:
            assert local_client.is_running(served)
            _, port = _debug_address(served)
            assert port != taken, "retried onto the port that was already in use"
            assert served.port == api_port, "pinned API port must be honoured"
        finally:
            served.remove(force=True)


def test_inherited_runtime_override_cannot_hijack_the_debugpy_port(
    dummy_api_path, monkeypatch
):
    """An exported setting must not override the port we picked per Tesseract.

    The child inherits our environment, so a TESSERACT_RUNTIME_DEBUGPY_PORT the
    caller exported is already there -- and typer resolves it as a CLI option,
    which beats the TESSERACT_DEBUGPY_PORT we set. Every Tesseract would land on
    the same port and the second would fail to start.
    """
    monkeypatch.setenv("TESSERACT_RUNTIME_DEBUGPY_PORT", "47777")

    with (
        Tesseract.from_source(
            dummy_api_path, python_executable=sys.executable
        ) as first,
        Tesseract.from_source(
            dummy_api_path, python_executable=sys.executable
        ) as second,
    ):
        ports = {_debug_address(t._serve_context)[1] for t in (first, second)}

    assert len(ports) == 2, "both Tesseracts used the inherited port"
    assert 47777 not in ports


def test_debugger_can_be_opted_out(dummy_api_path):
    """Turning off debug mode must also mean no debugger.

    Debug mode is what starts a debugger, here as everywhere else, so there is
    no separate knob to disable one -- which is what someone running dedicated
    processes for isolation rather than debugging wants anyway, since debug mode
    also exposes tracebacks and the `test` endpoint.
    """
    served = local_client.serve(
        dummy_api_path,
        python_executable=sys.executable,
        runtime_config={"debug": False},
    )
    try:
        assert _debug_address(served) is None
    finally:
        logs = served.logs().decode()
        served.remove(force=True)

    assert "Debugger listening" not in logs


def test_two_tesseracts_get_distinct_ports(dummy_api_path, sample_inputs):
    with (
        Tesseract.from_source(
            dummy_api_path, python_executable=sys.executable
        ) as first,
        Tesseract.from_source(
            dummy_api_path, python_executable=sys.executable
        ) as second,
    ):
        assert first._client.url != second._client.url
        np.testing.assert_allclose(first.apply(sample_inputs)["result"], [5.0, 8.0])
        np.testing.assert_allclose(second.apply(sample_inputs)["result"], [5.0, 8.0])


def test_runtime_config_does_not_leak_into_parent(dummy_api_path):
    """Config goes to the child as env vars, not into our own runtime config."""
    from tesseract_core.runtime.config import get_config

    before = get_config().output_format

    with Tesseract.from_source(
        dummy_api_path, python_executable=sys.executable, output_format="json"
    ):
        assert get_config().output_format == before


def test_requires_context_manager(dummy_api_path, sample_inputs):
    tess = Tesseract.from_source(dummy_api_path, python_executable=sys.executable)
    with pytest.raises(RuntimeError, match="from_source"):
        tess.apply(sample_inputs)


def test_binref_works_without_being_given_directories(dummy_api_path, sample_inputs):
    """Binref needs scratch dirs; not being told about them is not the user's problem."""
    with Tesseract.from_source(
        dummy_api_path, python_executable=sys.executable, output_format="json+binref"
    ) as tess:
        result = tess.apply(sample_inputs)

    assert result["result"].shape == sample_inputs["a"].shape


def test_binref_pool_is_available_without_a_linux_host(dummy_api_path, sample_inputs):
    """The pool is barred for containers off Linux, not for a plain subprocess.

    A container elsewhere runs in a VM, so bind mounts cross the VM boundary and
    client and server never share a page cache. Two processes on one host do.
    """
    if os.name != "posix":
        pytest.skip("the pool decodes with a read-only mmap, which needs POSIX")

    tess = Tesseract.from_source(
        dummy_api_path,
        output_format="json+binref",
        experimental_binref_pool=True,
        python_executable=sys.executable,
    )
    with tess:
        assert tess._client._binref_pool is not None
        result = tess.apply(sample_inputs)

    assert result["result"].shape == sample_inputs["a"].shape


def _scratch_tesseract(directory: Path, api: str) -> Path:
    """Write the smallest Tesseract that can be served, and return its api file."""
    (directory / "tesseract_config.yaml").write_text('name: "scratch"\n')
    api_path = directory / "tesseract_api.py"
    api_path.write_text(api)
    return api_path


def test_startup_failure_surfaces_child_traceback(tmp_path):
    api_path = _scratch_tesseract(
        tmp_path, "raise RuntimeError('kaboom at import time')\n"
    )

    tess = Tesseract.from_source(api_path)
    with pytest.raises(RuntimeError) as excinfo:
        tess.serve()

    message = str(excinfo.value)
    assert "stopped running during startup (exit code 1)" in message
    assert "kaboom at import time" in message


def test_failed_startup_leaves_no_log_file(tmp_path):
    """The captured output is read into the error, so its file has served its purpose."""
    api_path = _scratch_tesseract(
        tmp_path, "raise RuntimeError('kaboom at import time')\n"
    )

    temp_dir = Path(tempfile.gettempdir())
    before = set(temp_dir.glob("tesseract_serve_*.log"))

    with pytest.raises(RuntimeError):
        local_client.serve(api_path)

    assert not set(temp_dir.glob("tesseract_serve_*.log")) - before


def test_startup_timeout_is_reported(dummy_api_path, monkeypatch):
    """A Tesseract that never becomes healthy reports a timeout, not a crash."""

    def never_healthy(*args, **kwargs):
        raise requests.exceptions.ConnectionError("nope")

    monkeypatch.setattr(requests, "get", never_healthy)

    with pytest.raises(TimeoutError, match="did not respond to a health check"):
        local_client.serve(
            dummy_api_path, python_executable=sys.executable, startup_timeout=1.0
        )


def test_skip_health_check_returns_immediately(dummy_api_path):
    served = local_client.serve(
        dummy_api_path, python_executable=sys.executable, skip_health_check=True
    )
    try:
        assert local_client.is_running(served)
    finally:
        served.remove(force=True)


def test_foreign_interpreter_does_not_inherit_our_import_paths(monkeypatch):
    """Our sys.path must not follow a Tesseract into a different environment."""
    monkeypatch.setenv("PYTHONPATH", "/some/other/site-packages")
    monkeypatch.setenv("VIRTUAL_ENV", "/some/other/env")

    same = local_client._runtime_env(
        Path("tesseract_api.py"),
        input_path=None,
        output_path=None,
        output_format=None,
        runtime_config=None,
        environment=None,
        foreign_interpreter=False,
    )
    assert same["PYTHONPATH"] == "/some/other/site-packages"

    foreign = local_client._runtime_env(
        Path("tesseract_api.py"),
        input_path=None,
        output_path=None,
        output_format=None,
        runtime_config=None,
        environment=None,
        foreign_interpreter=True,
    )
    assert "PYTHONPATH" not in foreign
    assert "VIRTUAL_ENV" not in foreign

    # ...unless the caller insists
    explicit = local_client._runtime_env(
        Path("tesseract_api.py"),
        input_path=None,
        output_path=None,
        output_format=None,
        runtime_config=None,
        environment={"PYTHONPATH": "/deliberate"},
        foreign_interpreter=True,
    )
    assert explicit["PYTHONPATH"] == "/deliberate"


def test_missing_interpreter_is_reported(dummy_api_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        local_client.serve(dummy_api_path, python_executable="/nonexistent/bin/python")


@pytest.fixture
def sample_inputs():
    return {
        "a": np.array([1.0, 2.0], dtype=np.float32),
        "b": np.array([3.0, 4.0], dtype=np.float32),
        "s": 2,
    }


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
def test_removing_a_tesseract_does_not_kill_the_caller(dummy_api_path, tmp_path):
    """What the child's own process group is actually for.

    `remove` signals the process *group* so that uvicorn's workers go down with
    the process that spawned them. A child sharing our group puts that signal on
    us too, so disposing of a Tesseract would kill whoever created it.

    Fatal to whoever runs it, so it runs in a helper: with `popen_kwargs`
    stubbed out the helper is killed by SIGTERM at `remove`, and never reaches
    its final print. The helper is started in its own session so that the signal
    cannot reach this test runner either way.
    """
    helper = tmp_path / "helper.py"
    helper.write_text(
        textwrap.dedent(f"""
        import sys
        from tesseract_core.sdk import local_client

        if "--no-group" in sys.argv:
            local_client.popen_kwargs = lambda: {{}}

        served = local_client.serve({str(dummy_api_path)!r}, python_executable=sys.executable)
        served.remove(force=True)
        print("caller survived", flush=True)
        """)
    )

    def run(*args):
        return subprocess.run(
            [sys.executable, str(helper), *args],
            capture_output=True,
            text=True,
            start_new_session=True,
            timeout=120,
        )

    as_shipped = run()
    assert as_shipped.returncode == 0, as_shipped.stderr
    assert "caller survived" in as_shipped.stdout

    without_group = run("--no-group")
    assert without_group.returncode == -signal.SIGTERM, (
        "removing a Tesseract in the caller's own process group should have "
        f"killed the caller, but it exited {without_group.returncode}"
    )
    assert "caller survived" not in without_group.stdout


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
def test_child_runs_in_its_own_process_group(dummy_api_path):
    """The mechanism `remove` relies on to reach uvicorn's workers.

    `test_removing_a_tesseract_does_not_kill_the_caller` covers why it matters;
    this pins the mechanism itself, so a regression says which part broke.
    """
    served = local_client.serve(dummy_api_path, python_executable=sys.executable)
    try:
        assert os.getpgid(served.process.pid) != os.getpgid(os.getpid())
    finally:
        served.remove(force=True)


@pytest.mark.skipif(os.name == "nt", reason="POSIX signals")
def test_remove_escalates_to_sigkill(dummy_api_path):
    """A Tesseract that ignores SIGTERM still gets cleaned up."""
    served = local_client.serve(dummy_api_path, python_executable=sys.executable)

    # Make the child ignore SIGTERM by shortening our patience instead of
    # modifying the child: escalation must happen either way.
    monkeypatched_timeout = 0.5
    original = local_client._TERMINATE_TIMEOUT
    local_client._TERMINATE_TIMEOUT = monkeypatched_timeout
    try:
        os.killpg(os.getpgid(served.process.pid), signal.SIGSTOP)  # ignores SIGTERM
        served.remove(force=True)
    finally:
        local_client._TERMINATE_TIMEOUT = original

    assert served.process.poll() is not None


# Real Tesseracts from examples/, chosen because each breaks a different
# assumption that dummy_api_path never exercises. Deliberately not the whole
# corpus, which `test_examples.py` covers.


EXAMPLES = Path(__file__).parents[2] / "examples"


@pytest.fixture
def example_copy(tmp_path):
    """Copy an example out of the repo, so provisioning cannot pollute it.

    Resolving an environment writes a `.venv` beside the `tesseract_api.py`. A
    test that let that land in `examples/` would leave ~200 MB behind and make
    every later test in the session see a pre-built environment.
    """

    def copy(name: str) -> Path:
        destination = tmp_path / name
        # Never copy an environment: one built in a developer's checkout holds
        # absolute paths back into it, so the copy looks usable and is not.
        shutil.copytree(
            EXAMPLES / name,
            destination,
            ignore=shutil.ignore_patterns(".venv", "venv", "__pycache__"),
        )
        return destination / "tesseract_api.py"

    return copy


def test_an_environment_is_built_even_when_this_one_would_do(
    example_copy, dummy_tesseract_package
):
    """Provisioning never quietly falls back to the interpreter we are running.

    `helloworld` declares nothing, and the dummy Tesseract here declares only
    something this environment already has, so both could be served from here.
    Doing that would make behaviour depend on what is installed alongside the
    SDK: an upgrade elsewhere turns an instant constructor into a slow one, no
    environment appears where the user expected one, and a Tesseract can import
    packages it never declared. `python_executable=sys.executable` asks for
    this interpreter explicitly.

    `packaging` rather than something heavier because the point is the decision,
    not the download.
    """
    declares_nothing = example_copy("helloworld")
    declares_what_we_have = dummy_tesseract_package / "tesseract_api.py"
    (dummy_tesseract_package / "tesseract_requirements.txt").write_text("packaging\n")

    for api_path in (declares_nothing, declares_what_we_have):
        chosen = venv_provision.resolve_python_executable(api_path)

        assert chosen != Path(sys.executable)
        assert chosen == venv_provision._python_in(api_path.parent / ".venv")


def test_builds_an_environment_for_dependencies_we_do_not_have(example_copy):
    """The case that needed `python_executable` filled in by hand before.

    `localpackage` needs a local package installed (``./helloworld``) that this
    interpreter does not have, and imports a sibling module shipped as
    package_data (``goodbyeworld``) which only resolves because the runtime puts
    the API's own directory on sys.path. The greeting proves both halves.

    Its requirement is a relative path, which is also a case
    `_caller_shortfall` has to decline to judge. So this covers falling through
    to a build as well.
    """
    api_path = example_copy("localpackage")

    with Tesseract.from_source(api_path) as tess:
        result = tess.apply({"name": "World"})

    assert "Hello World!" in result["message"], "local package dependency missing"
    assert "Goodbye World!" in result["message"], "package_data sibling missing"
    assert (api_path.parent / ".venv").is_dir(), "no environment was built"


def test_an_environment_built_once_is_reused(dummy_tesseract_package):
    """The second serve must not install anything.

    Whatever provisioning costs, it should be paid once, not on every serve and
    certainly not on every endpoint call. That is why it happens before the
    process starts.

    This uses a remote requirement on purpose. A local path is always built from
    source, so uv never reports it as satisfied and a Tesseract declaring one is
    reinstalled every time by design.
    """
    api_path = dummy_tesseract_package / "tesseract_api.py"
    (dummy_tesseract_package / "tesseract_requirements.txt").write_text("cowsay\n")

    first = venv_provision.resolve_python_executable(api_path)
    assert first == venv_provision._python_in(dummy_tesseract_package / ".venv")

    # Reuse means nothing gets installed, which a wall-clock budget cannot show:
    # re-running an install against an already-satisfied environment is fast
    # enough to pass one. Failing outright if provisioning is attempted is the
    # only assertion that tells the two apart.
    def provisioned(command, what):
        raise AssertionError(f"reprovisioned a good environment: {what}")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(venv_provision, "_run", provisioned)
        assert venv_provision.resolve_python_executable(api_path) == first


def test_an_environment_without_the_runtime_is_completed(example_copy):
    """A `.venv` that cannot serve gets completed, not handed back as-is.

    ``runtime`` is an optional extra, so an environment can have the SDK and
    still be unable to serve anything. This is why ``pip install
    tesseract-core`` followed by ``from_source`` fails. We must not mistake such
    an environment for a usable one.
    """
    api_path = example_copy("localpackage")
    venv = api_path.parent / ".venv"
    # Built the way the resolver builds one, so this really is the kind of bare
    # .venv a user might already have, not an imitation of it.
    venv_provision._ensure_venv(venv)
    assert not venv_provision._can_serve(venv_provision._python_in(venv))

    with Tesseract.from_source(api_path) as tess:
        assert "Hello World!" in tess.apply({"name": "World"})["message"]

    assert venv_provision._can_serve(venv_provision._python_in(venv))


def test_python_bounds_exclude_what_the_runtime_cannot_use():
    """Only bounds are kept, and the SDK's Requires-Python sets the floor.

    We always start from the current interpreter and let uv say which way to go,
    so there is no list of preferred versions to keep up to date. The ceiling
    comes from uv, so new releases and prereleases are included automatically.
    """
    bounds = venv_provision._python_bounds()
    if bounds is None:
        pytest.skip("uv could not report which Pythons it can provide")

    floor, ceiling = bounds
    # uv offers 3.8 and 3.9, but the runtime will not install on them.
    assert floor >= 10, "the runtime does not install on end-of-life Pythons"
    assert floor <= sys.version_info.minor <= ceiling


def test_a_newer_python_is_reachable_not_just_an_older_one():
    """A Tesseract may need a Python newer than the caller's, not just older.

    Only ever searching downwards would break on 3.10, the oldest version we
    support and one this project tests on, because there is nothing below it. A
    package that ships wheels only for a newer Python without declaring a floor
    is a common build-matrix slip, so that case has to work too.
    """
    ours = sys.version_info.minor
    bounds = venv_provision._python_bounds()
    if bounds is None:
        pytest.skip("uv could not report which Pythons it can provide")
    if bounds[1] <= ours:
        pytest.skip("uv offers nothing newer than the running Python")

    # What uv reports when a package only has wheels for newer versions.
    assert venv_provision._nearest(range(ours + 1, bounds[1] + 1)) == (
        f"{sys.version_info.major}.{ours + 1}"
    )

    # Closest wins over newest, so a built environment stays near ours. Only
    # meaningful when there is something below us: on the oldest version we
    # support there is not, and `_nearest` drops it as out of bounds.
    if bounds[0] < ours:
        assert venv_provision._nearest([ours - 1, bounds[1]]) == (
            f"{sys.version_info.major}.{ours - 1}"
        )


def test_abi_tag_hint_is_read_as_the_answer():
    """Uv lists the versions a package does have wheels for.

    That is the answer itself, so we use it instead of trying versions. The tag
    for the version we asked about appears earlier in the message, and must not
    be read as one of the available ones.
    """
    hint = (
        "hint: You require CPython 3.13 (`cp313`), but we only found wheels for "
        "`jaxlib` (v0.4.28) with the following Python ABI tags: `cp39`, `cp310`, "
        "`cp311`, `cp312`"
    )

    assert venv_provision._minors_from_abi_tags(hint) == [9, 10, 11, 12]


def test_requires_python_hint_is_read_as_the_answer():
    """Uv names the Python range a dependency wants, so we need not search."""
    hint = (
        "Because the requested Python version (>=3.10) does not satisfy "
        "Python>=3.12 and numpy==2.5.1 depends on Python>=3.12, we can conclude "
        "that numpy==2.5.1 cannot be used.\n"
        "hint: The `--python-version` value (>=3.10) includes Python versions "
        "that are not supported by your dependencies (e.g., numpy==2.5.1 only "
        "supports >=3.12). Consider using a higher `--python-version` value."
    )

    assert str(venv_provision._requires_python(hint)) == ">=3.12"


def test_a_pin_gets_a_python_that_has_wheels_for_it(example_copy):
    """Whatever version is chosen, the declared pin must install on it.

    In a container the Python comes from the base image, 3.11 for the default
    `debian:bookworm-slim`, so `univariate` pinning `jax[cpu]==0.4.28` builds
    there without trouble. jaxlib 0.4.28 publishes no wheel past cp312, so on a
    newer interpreter we have to pick a different version, and on an older one
    staying put is already correct. Asserting the property rather than a
    particular version keeps this true on every Python we support.
    """
    api_path = example_copy("univariate")
    build_config = get_config(api_path.parent).build_config
    requirements_file = api_path.parent / "tesseract_requirements.txt"

    chosen = venv_provision._build_python_version(build_config, requirements_file)
    if chosen is None:
        chosen = f"{sys.version_info.major}.{sys.version_info.minor}"

    _, remote = venv_provision.parse_requirements(requirements_file)
    assert venv_provision._compile(remote, chosen, wheels_only=True) is None, (
        f"chose Python {chosen}, which has no wheels for {remote}"
    )


def test_a_local_requirement_does_not_constrain_the_python(example_copy):
    """Local paths must not be mistaken for a version constraint.

    `localpackage` declares only `./helloworld`, which is an sdist by nature and
    always buildable. Asking whether it has a wheel would reject every Python
    and say nothing, so it has to be left out of the question entirely.
    """
    api_path = example_copy("localpackage")
    build_config = get_config(api_path.parent).build_config

    assert (
        venv_provision._build_python_version(
            build_config, api_path.parent / "tesseract_requirements.txt"
        )
        is None
    )


def test_a_declared_python_version_applies_with_no_requirements(
    dummy_tesseract_package,
):
    """`python_version` is a property of the environment, not of its contents.

    A Tesseract can name the Python it wants and install nothing at all. The
    two used to be handled by separate paths, and the one for "nothing to
    install" quietly ignored the version.
    """
    (dummy_tesseract_package / "tesseract_config.yaml").write_text(
        'name: "pinned"\n'
        "build_config:\n"
        "  requirements:\n"
        "    provider: uv-pip\n"
        '    python_version: "3.11"\n'
    )
    api_path = dummy_tesseract_package / "tesseract_api.py"

    build_config, requirements_file = venv_provision._declared_requirements(api_path)

    assert requirements_file is None, "the fixture declares no requirements"
    assert venv_provision._build_python_version(build_config, None) == "3.11"


def test_a_missing_config_is_reported(dummy_tesseract_package):
    """No config means no way to tell what to install, so say so.

    The runtime itself never reads `tesseract_config.yaml`, so this could serve
    on the caller's interpreter instead. But then a `tesseract_requirements.txt`
    sitting next to the api file would be silently ignored, and `tesseract
    build` would reject the same directory anyway.
    """
    api_path = dummy_tesseract_package / "tesseract_api.py"
    (dummy_tesseract_package / "tesseract_config.yaml").unlink()

    with pytest.raises(UserError, match=r"tesseract_config\.yaml"):
        venv_provision.resolve_python_executable(api_path)


def test_a_broken_config_is_reported_not_worked_around(dummy_tesseract_package):
    """A config `tesseract build` would reject should fail here too.

    Serving from source is usually the step before building, so a malformed
    `tesseract_config.yaml` is something the user is about to hit anyway.
    Carrying on with this interpreter would hide it.
    """
    api_path = dummy_tesseract_package / "tesseract_api.py"
    (dummy_tesseract_package / "tesseract_config.yaml").write_text("name: [unclosed\n")

    with pytest.raises(UserError, match=r"tesseract_config\.yaml"):
        venv_provision.resolve_python_executable(api_path)


def test_an_explicit_interpreter_skips_resolution(dummy_tesseract_package):
    """Naming an interpreter means using it, not stating a preference.

    The requirement here is one this environment does not have, so resolving
    would build a `.venv`. Naming an interpreter has to stop that happening. An
    explicit argument should win, and this is also the escape hatch that every
    "could not provision" message points at.
    """
    api_path = dummy_tesseract_package / "tesseract_api.py"
    (dummy_tesseract_package / "tesseract_requirements.txt").write_text("cowsay\n")

    with Tesseract.from_source(api_path, python_executable=sys.executable) as tess:
        assert tess.health()["status"] == "ok"

    assert not (dummy_tesseract_package / ".venv").exists()
