# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for serving Tesseracts in dedicated subprocesses.

These spawn real ``tesseract-runtime serve`` processes (but no containers), so
they exercise the actual startup / health-check / removal path.
"""

import logging
import os
import re
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
from tesseract_core.sdk import local_client, serving
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
    served = local_client.serve(dummy_api_path)
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
    served = local_client.serve(dummy_api_path)
    served.remove(force=True)

    assert served.wait(timeout=5)["StatusCode"] is not None


def test_wait_times_out_on_a_running_process(dummy_api_path):
    """Waiting on a live Tesseract gives up rather than blocking for its lifetime."""
    served = local_client.serve(dummy_api_path)
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
    """
    helper = tmp_path / "helper.py"
    helper.write_text(
        textwrap.dedent(f"""
        import time
        from tesseract_core.sdk import local_client

        served = local_client.serve({str(dummy_api_path)!r})
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
    served = local_client.serve(dummy_api_path)
    try:
        with pytest.raises(RuntimeError, match="still running"):
            served.remove()
        assert local_client.is_running(served)
    finally:
        served.remove(force=True)


def test_remove_is_idempotent(dummy_api_path):
    served = local_client.serve(dummy_api_path)
    served.remove(force=True)
    # Must not raise, even though the process and its log file are gone
    served.remove(force=True)
    if os.name != "nt":
        # The logs really are gone, as they are for a removed container. Except
        # on Windows, which refuses to delete a file a just-killed child may
        # still hold -- removal tolerates that and leaves it behind.
        with pytest.raises(FileNotFoundError):
            served.logs()


@pytest.mark.skipif(
    os.name == "nt",
    reason="Windows will not unlink a file the live child still holds open",
)
def test_unreadable_logs_do_not_mask_a_subprocess_startup_failure(
    dummy_api_path, monkeypatch
):
    """A local Tesseract fails in its own vocabulary, not Docker's.

    The startup path used to catch only `APIError`, so the `FileNotFoundError`
    a vanished log file raises escaped and replaced the failure it was being
    read to explain.

    POSIX only, because the premise needs a log file deleted from under a
    running process, which Windows refuses -- see `TesseractProcess.remove`.
    `test_any_unreadable_log_does_not_mask_the_startup_failure` makes the same
    point everywhere, without needing the file to actually go away.
    """
    monkeypatch.setattr(serving, "is_running", lambda served: False)

    served = local_client.serve(dummy_api_path, skip_health_check=True)
    try:
        served.log_path.unlink()
        with pytest.raises((RuntimeError, TimeoutError)) as excinfo:
            serving.wait_for_health_or_dispose(served, served.url, timeout=0.05)
        assert "stopped running during startup" in str(excinfo.value)
    finally:
        served.remove(force=True)


def test_any_unreadable_log_does_not_mask_the_startup_failure(
    dummy_api_path, monkeypatch
):
    """Whatever reading the logs raises, the startup failure is what surfaces.

    The point of catching broadly rather than naming a transport's exceptions:
    reading the logs is how the failure gets reported, so it must not become the
    failure. Runs on every platform, unlike the vanished-file case above.
    """
    monkeypatch.setattr(serving, "is_running", lambda served: False)

    served = local_client.serve(dummy_api_path, skip_health_check=True)
    try:

        def unreadable():
            raise RuntimeError("log storage is on fire")

        served.logs = unreadable
        with pytest.raises((RuntimeError, TimeoutError)) as excinfo:
            serving.wait_for_health_or_dispose(served, served.url, timeout=0.05)
        assert "stopped running during startup" in str(excinfo.value)
        assert "on fire" not in str(excinfo.value)
    finally:
        del served.logs
        served.remove(force=True)


def test_serve_rejects_binref_without_an_output_path(dummy_api_path):
    """The same error as the containerized path, and the same type.

    A caller that moves between the two engines should not have to catch two
    different exceptions for one mistake.
    """
    with pytest.raises(UserError, match=r"json\+binref"):
        local_client.serve(dummy_api_path, output_format="json+binref")


def test_serve_rejects_missing_api():
    with pytest.raises(FileNotFoundError, match="is not a file"):
        local_client.serve("/nonexistent/tesseract_api.py")


def test_serve_on_explicit_port(dummy_api_path):
    from tesseract_core.sdk.engine import get_free_port

    port = get_free_port()
    served = local_client.serve(dummy_api_path, port=port)
    try:
        assert served.port == port
        assert served.url.endswith(f":{port}")
    finally:
        served.remove(force=True)


def test_endpoints_over_subprocess(dummy_api_path, sample_inputs):
    with Tesseract.from_source(dummy_api_path) as tess:
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
    tess = Tesseract.from_source(dummy_api_path)
    tess.serve()
    served = tess._serve_context
    assert local_client.is_running(served)

    tess.teardown()

    assert not local_client.is_running(served)
    assert tess._client is None
    assert tess._serve_context is None


def test_logs_are_captured(dummy_api_path):
    tess = Tesseract.from_source(dummy_api_path)
    with tess:
        assert "Uvicorn running" in tess.server_logs()

    # Logs remain available after removal
    assert "Uvicorn running" in tess.server_logs()


def test_stream_logs_without_output_path(dummy_api_path, sample_inputs):
    """Streaming must work without the caller specifying an output directory."""
    lines = []

    with Tesseract.from_source(dummy_api_path, stream_logs=lines.append) as tess:
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

    served = local_client.serve(dummy_api_path, runtime_config={"debug": True})
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
    served = local_client.serve(dummy_api_path, runtime_config={"debug": True})
    try:
        host, port = _debug_address(served)
        assert host == "127.0.0.1"
        assert port != served.port, "Debugger must not share the API port"
    finally:
        served.remove(force=True)


def test_two_tesseracts_get_distinct_debugpy_ports(dummy_api_path):
    """The whole reason the address is configurable: both must be debuggable."""
    first = local_client.serve(dummy_api_path, runtime_config={"debug": True})
    second = local_client.serve(dummy_api_path, runtime_config={"debug": True})
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
            dummy_api_path, port=api_port, runtime_config={"debug": True}
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
        Tesseract.from_source(dummy_api_path) as first,
        Tesseract.from_source(dummy_api_path) as second,
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
    served = local_client.serve(dummy_api_path, runtime_config={"debug": False})
    try:
        assert _debug_address(served) is None
    finally:
        logs = served.logs().decode()
        served.remove(force=True)

    assert "Debugger listening" not in logs


def test_two_tesseracts_get_distinct_ports(dummy_api_path, sample_inputs):
    with (
        Tesseract.from_source(dummy_api_path) as first,
        Tesseract.from_source(dummy_api_path) as second,
    ):
        assert first._client.url != second._client.url
        np.testing.assert_allclose(first.apply(sample_inputs)["result"], [5.0, 8.0])
        np.testing.assert_allclose(second.apply(sample_inputs)["result"], [5.0, 8.0])


def test_runtime_config_does_not_leak_into_parent(dummy_api_path):
    """Config goes to the child as env vars, not into our own runtime config."""
    from tesseract_core.runtime.config import get_config

    before = get_config().output_format

    with Tesseract.from_source(dummy_api_path, output_format="json"):
        assert get_config().output_format == before


def test_requires_context_manager(dummy_api_path, sample_inputs):
    tess = Tesseract.from_source(dummy_api_path)
    with pytest.raises(RuntimeError, match="from_source"):
        tess.apply(sample_inputs)


def test_binref_works_without_being_given_directories(dummy_api_path, sample_inputs):
    """Binref needs scratch dirs; not being told about them is not the user's problem."""
    with Tesseract.from_source(dummy_api_path, output_format="json+binref") as tess:
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
        dummy_api_path, output_format="json+binref", experimental_binref_pool=True
    )
    with tess:
        assert tess._client._binref_pool is not None
        result = tess.apply(sample_inputs)

    assert result["result"].shape == sample_inputs["a"].shape


def test_startup_failure_surfaces_child_traceback(tmp_path):
    api_path = tmp_path / "tesseract_api.py"
    api_path.write_text("raise RuntimeError('kaboom at import time')\n")

    tess = Tesseract.from_source(api_path)
    with pytest.raises(RuntimeError) as excinfo:
        tess.serve()

    message = str(excinfo.value)
    assert "stopped running during startup (exit code 1)" in message
    assert "kaboom at import time" in message


def test_failed_startup_leaves_no_log_file(tmp_path):
    """The captured output is read into the error, so its file has served its purpose."""
    api_path = tmp_path / "tesseract_api.py"
    api_path.write_text("raise RuntimeError('kaboom at import time')\n")

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
        local_client.serve(dummy_api_path, startup_timeout=1.0)


def test_skip_health_check_returns_immediately(dummy_api_path):
    served = local_client.serve(dummy_api_path, skip_health_check=True)
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


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
def test_child_runs_in_its_own_process_group(dummy_api_path):
    """So that a Ctrl-C in the parent's terminal doesn't race us to the child."""
    served = local_client.serve(dummy_api_path)
    try:
        assert os.getpgid(served.process.pid) != os.getpgid(os.getpid())
    finally:
        served.remove(force=True)


@pytest.mark.skipif(os.name == "nt", reason="POSIX signals")
def test_remove_escalates_to_sigkill(dummy_api_path):
    """A Tesseract that ignores SIGTERM still gets cleaned up."""
    served = local_client.serve(dummy_api_path)

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
# corpus: a third of it cannot run here at all, and which third depends on what
# happens to be installed. Once a venv is built on demand, most of the rest
# becomes reachable and this can grow.


EXAMPLES = Path(__file__).parents[2] / "examples"


def test_serves_a_tesseract_whose_dependencies_we_do_not_have(example_venv):
    """The case `python_executable` exists for.

    `localpackage` needs a local package installed (``./helloworld``) that this
    interpreter does not have, and imports a sibling module shipped as
    package_data (``goodbyeworld``) which only resolves because the runtime puts
    the API's own directory on sys.path. The greeting proves both halves.
    """
    interpreter = example_venv("localpackage")

    with Tesseract.from_source(
        EXAMPLES / "localpackage" / "tesseract_api.py",
        python_executable=interpreter,
    ) as tess:
        result = tess.apply({"name": "World"})

    assert "Hello World!" in result["message"], "local package dependency missing"
    assert "Goodbye World!" in result["message"], "package_data sibling missing"


def test_required_files_resolve_against_the_input_path(tmp_path):
    """A Tesseract that reads a file at import time, not just at apply time.

    `require_file` is resolved against the input path while `tesseract_api.py` is
    being imported, so the setting has to be in the child's environment before it
    starts -- not passed with the first request.
    """
    with Tesseract.from_source(
        EXAMPLES / "required_files" / "tesseract_api.py",
        input_path=EXAMPLES / "required_files" / "input",
    ) as tess:
        result = tess.apply({})

    # Read straight out of input/parameters1.json at import time.
    assert result == {"a": 1.0, "b": 100.0}


def test_a_container_only_tesseract_fails_legibly():
    """Some Tesseracts cannot be served this way, and must say so clearly.

    `userhandling` creates /home/tesseract-user at import time, which exists only
    in the image. No interpreter choice fixes that, so the value here is the
    diagnosis: the child's own traceback has to reach the caller instead of a
    bare "stopped running".
    """
    with pytest.raises(RuntimeError) as excinfo:
        with Tesseract.from_source(EXAMPLES / "userhandling" / "tesseract_api.py"):
            pass

    message = str(excinfo.value)
    assert "stopped running during startup" in message
    # The child's own traceback, not just our summary of it.
    assert "Traceback (most recent call last)" in message
    assert "userhandling" in message, "child traceback did not reach us"
    # Why it cannot be served differs by platform: POSIX gets as far as the
    # container-only directory, Windows has no `pwd` module to import first.
    cause = "No module named 'pwd'" if os.name == "nt" else "/home/tesseract-user"
    assert cause in message
