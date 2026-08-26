# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for serving Tesseracts in dedicated subprocesses.

These spawn real ``tesseract-runtime serve`` processes (but no containers), so
they exercise the actual startup / health-check / removal path.
"""

import gc
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import pytest
import requests

from tesseract_core import Tesseract
from tesseract_core.sdk import local_client, local_engine

pytestmark = pytest.mark.timeout(120)


def _env_without_pythonpath() -> dict[str, str]:
    """Environment safe to hand to a different interpreter.

    Importing a tesseract_api.py in-process puts this interpreter's sys.path on
    PYTHONPATH as a side effect, and any earlier test in the session may have
    done so. A 3.11 interpreter that inherits it picks up 3.13 packages.
    """
    return {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}


@pytest.fixture
def dummy_api_path(dummy_tesseract_package):
    return dummy_tesseract_package / "tesseract_api.py"


@pytest.fixture
def sample_inputs():
    return {
        "a": np.array([1.0, 2.0], dtype=np.float32),
        "b": np.array([3.0, 4.0], dtype=np.float32),
        "s": 2,
    }


def test_serve_and_remove(dummy_api_path):
    served = local_engine.serve(dummy_api_path)
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
    served = local_engine.serve(dummy_api_path)
    served.remove(force=True)

    assert served.wait(timeout=5)["StatusCode"] is not None


def test_wait_times_out_on_a_running_process(dummy_api_path):
    """Waiting on a live Tesseract gives up rather than blocking for its lifetime."""
    served = local_engine.serve(dummy_api_path)
    try:
        with pytest.raises(TimeoutError, match="still running"):
            served.wait(timeout=0.2)
    finally:
        served.remove(force=True)


def test_remove_refuses_a_running_process(dummy_api_path):
    """Unforced removal refuses a live Tesseract, as removing a container does."""
    served = local_engine.serve(dummy_api_path)
    try:
        with pytest.raises(RuntimeError, match="still running"):
            served.remove()
        assert local_client.is_running(served)
    finally:
        served.remove(force=True)


def test_remove_is_idempotent(dummy_api_path):
    served = local_engine.serve(dummy_api_path)
    served.remove(force=True)
    # Must not raise, even though the process and its log file are gone
    served.remove(force=True)
    if os.name != "nt":
        # The logs really are gone, as they are for a removed container. Except
        # on Windows, which refuses to delete a file a just-killed child may
        # still hold -- removal tolerates that and leaves it behind.
        with pytest.raises(FileNotFoundError):
            served.logs()


def test_serve_rejects_missing_api():
    with pytest.raises(FileNotFoundError, match="is not a file"):
        local_engine.serve("/nonexistent/tesseract_api.py")


def test_serve_on_explicit_port(dummy_api_path):
    from tesseract_core.sdk.engine import get_free_port

    port = get_free_port()
    served = local_engine.serve(dummy_api_path, port=port)
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


def test_garbage_collection_reaps_process(dummy_api_path):
    """A forgotten Tesseract must not leave an orphaned process behind."""
    import gc

    tess = Tesseract.from_source(dummy_api_path)
    tess.serve()
    # Hold the process, not the Tesseract, so it can still be collected.
    process = tess._serve_context.process

    del tess
    gc.collect()

    assert process.poll() is not None


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


def test_debugger_listens_on_loopback_by_default(dummy_api_path):
    """Debugging is the point of a non-containerized Tesseract, so it's on...

    ...but on loopback: unlike a container, there is no network namespace here,
    and debugpy is unauthenticated code execution.
    """
    served = local_engine.serve(dummy_api_path, runtime_config={"debug": True})
    try:
        host, port = _debug_address(served)
        assert host == "127.0.0.1"
        assert port != served.port, "Debugger must not share the API port"
    finally:
        served.remove(force=True)


def test_two_tesseracts_get_distinct_debugpy_ports(dummy_api_path):
    """The whole reason the address is configurable: both must be debuggable."""
    first = local_engine.serve(dummy_api_path, runtime_config={"debug": True})
    second = local_engine.serve(dummy_api_path, runtime_config={"debug": True})
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

    monkeypatch.setattr(local_engine, "get_free_port", fake_get_free_port)

    with closing(socket.socket()) as occupied:
        occupied.bind(("127.0.0.1", taken))
        occupied.listen(1)

        served = local_engine.serve(
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
    served = local_engine.serve(dummy_api_path, runtime_config={"debug": False})
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


def test_rejects_imported_module(dummy_tesseract_module):
    """A module cannot be handed to another process, so say so clearly.

    Tested against the helper rather than `from_source`, whose annotation lets
    typeguard reject it first under the test suite -- but nothing enforces
    annotations at runtime, so the check still has to exist.
    """
    from tesseract_core.sdk.tesseract import _subprocess_spawn_config

    with pytest.raises(ValueError, match="already imported module was given"):
        _subprocess_spawn_config(
            dummy_tesseract_module,
            input_path=None,
            output_path=None,
            output_format="json+base64",
            runtime_config=None,
            python_executable=None,
            startup_timeout=1.0,
        )


def test_binref_works_without_being_given_directories(dummy_api_path, sample_inputs):
    """Binref needs scratch dirs; not being told about them is not the user's problem."""
    with Tesseract.from_source(dummy_api_path, output_format="json+binref") as tess:
        result = tess.apply(sample_inputs)

    assert result["result"].shape == sample_inputs["a"].shape


def test_auto_created_scratch_dirs_are_purged(dummy_api_path, sample_inputs):
    """What we made, we clean up -- unlike directories the caller passed in."""
    tess = Tesseract.from_source(dummy_api_path, output_format="json+binref")
    scratch = [
        Path(tess._spawn_config["input_path"]),
        Path(tess._spawn_config["output_path"]),
    ]
    assert all(d.exists() for d in scratch)

    with tess:
        tess.apply(sample_inputs)
    del tess
    gc.collect()

    assert not any(d.exists() for d in scratch)


def test_given_scratch_dirs_are_left_alone(dummy_api_path, sample_inputs, tmp_path):
    given_in, given_out = tmp_path / "in", tmp_path / "out"
    given_in.mkdir()
    given_out.mkdir()

    tess = Tesseract.from_source(
        dummy_api_path,
        input_path=given_in,
        output_path=given_out,
        output_format="json+binref",
    )
    with tess:
        tess.apply(sample_inputs)
    del tess
    gc.collect()

    assert given_in.exists() and given_out.exists()


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


def test_container_info_unavailable(dummy_api_path):
    tess = Tesseract.from_source(dummy_api_path)
    with pytest.raises(RuntimeError, match="from_image"):
        tess.container_info()


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
        local_engine.serve(api_path)

    assert not set(temp_dir.glob("tesseract_serve_*.log")) - before


def test_startup_timeout_is_reported(dummy_api_path, monkeypatch):
    """A Tesseract that never becomes healthy reports a timeout, not a crash."""

    def never_healthy(*args, **kwargs):
        raise requests.exceptions.ConnectionError("nope")

    monkeypatch.setattr(requests, "get", never_healthy)

    with pytest.raises(TimeoutError, match="did not respond to a health check"):
        local_engine.serve(dummy_api_path, startup_timeout=1.0)


def test_skip_health_check_returns_immediately(dummy_api_path):
    served = local_engine.serve(dummy_api_path, skip_health_check=True)
    try:
        assert local_client.is_running(served)
    finally:
        served.remove(force=True)


@pytest.fixture(scope="session")
def foreign_venv(tmp_path_factory):
    """A separate environment running a different Python from this one.

    This is what makes subprocess isolation worth more than a nicety: the
    Tesseract need not be installable alongside the caller. Deliberately no
    version pins beyond the interpreter -- CI rewrites the runtime extras to
    exact pins on its oldest-dependency axis, so anything we add here can
    conflict with them.
    """
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required to build a foreign environment")

    # Any supported version that is not the one running the tests.
    ours = f"{sys.version_info.major}.{sys.version_info.minor}"
    foreign = next(v for v in ("3.12", "3.11", "3.13") if v != ours)

    venv_dir = tmp_path_factory.mktemp("foreign_venv") / "env"
    repo_root = Path(__file__).parents[2]
    env = _env_without_pythonpath()

    def run(*args):
        result = subprocess.run(args, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            pytest.skip(
                f"could not build a Python {foreign} environment: "
                f"{result.stderr.strip()[-300:]}"
            )

    run(uv, "venv", str(venv_dir), "--python", foreign)
    run(uv, "pip", "install", "--python", str(venv_dir), f"{repo_root}[runtime]")

    scripts, exe = ("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")
    return venv_dir / scripts / exe, foreign


def test_foreign_interpreter_does_not_inherit_our_import_paths(monkeypatch):
    """Our sys.path must not follow a Tesseract into a different environment."""
    monkeypatch.setenv("PYTHONPATH", "/some/other/site-packages")
    monkeypatch.setenv("VIRTUAL_ENV", "/some/other/env")

    same = local_engine._runtime_env(
        Path("tesseract_api.py"),
        input_path=None,
        output_path=None,
        output_format=None,
        runtime_config=None,
        environment=None,
        foreign_interpreter=False,
    )
    assert same["PYTHONPATH"] == "/some/other/site-packages"

    foreign = local_engine._runtime_env(
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
    explicit = local_engine._runtime_env(
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
        local_engine.serve(dummy_api_path, python_executable="/nonexistent/bin/python")


@pytest.mark.foreign_venv
def test_tesseract_in_foreign_environment(foreign_venv, dummy_api_path, sample_inputs):
    """A Tesseract runs under an interpreter the caller could not have used."""
    interpreter, foreign_version = foreign_venv

    reported = subprocess.run(
        [str(interpreter), "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
        check=True,
        capture_output=True,
        text=True,
        env=_env_without_pythonpath(),
    ).stdout.strip()

    # Guard the premise: same interpreter would prove nothing
    assert reported == foreign_version
    assert reported != f"{sys.version_info.major}.{sys.version_info.minor}"

    with Tesseract.from_source(
        dummy_api_path,
        python_executable=interpreter,
    ) as tess:
        result = tess.apply(sample_inputs)

    np.testing.assert_allclose(result["result"], [5.0, 8.0])


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
def test_child_runs_in_its_own_process_group(dummy_api_path):
    """So that a Ctrl-C in the parent's terminal doesn't race us to the child."""
    served = local_engine.serve(dummy_api_path)
    try:
        assert os.getpgid(served.process.pid) != os.getpgid(os.getpid())
    finally:
        served.remove(force=True)


@pytest.mark.skipif(os.name == "nt", reason="POSIX signals")
def test_remove_escalates_to_sigkill(dummy_api_path):
    """A Tesseract that ignores SIGTERM still gets cleaned up."""
    served = local_engine.serve(dummy_api_path)

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
