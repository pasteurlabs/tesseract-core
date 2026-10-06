# TEMP: subprocess instrumentation to attribute test wall-time to container-engine
# operations (build/run/pull/etc.) vs. everything else. Enabled only when
# TESSERACT_INSTRUMENT_SUBPROCESS=1, so local test runs are unaffected.
#
# Patches subprocess.Popen (which subprocess.run also uses). For each call,
# records (test_id, classified_verb, duration). At session end, prints a per-test
# breakdown of time spent in subprocess calls, grouped by verb.
import os
import subprocess
import threading
import time
from collections import defaultdict

_ENABLED = os.environ.get("TESSERACT_INSTRUMENT_SUBPROCESS") == "1"

_timings: dict[tuple[str, str], list[float]] = defaultdict(list)
_current_test: str | None = None
_lock = threading.Lock()


def _classify(cmd) -> str:
    """Turn a command list/string into a short verb like 'podman buildx build'."""
    if not cmd:
        return "other"
    parts = cmd if isinstance(cmd, (list, tuple)) else str(cmd).split()
    exe = str(parts[0]).rsplit("/", 1)[-1]
    if exe in ("docker", "podman") and len(parts) > 1:
        sub = str(parts[1])
        if sub == "buildx" and len(parts) > 2:
            return f"{exe} buildx {parts[2]}"
        return f"{exe} {sub}"
    return exe


if _ENABLED:
    _orig_popen_init = subprocess.Popen.__init__
    _orig_popen_wait = subprocess.Popen.wait

    def _patched_init(self, cmd, *args, **kwargs):
        self._tess_verb = _classify(cmd)
        self._tess_start = time.monotonic()
        _orig_popen_init(self, cmd, *args, **kwargs)

    def _patched_wait(self, *args, **kwargs):
        rc = _orig_popen_wait(self, *args, **kwargs)
        verb = getattr(self, "_tess_verb", "other")
        start = getattr(self, "_tess_start", None)
        if start is not None:
            with _lock:
                _timings[(_current_test or "<outside-test>", verb)].append(
                    time.monotonic() - start
                )
        return rc

    subprocess.Popen.__init__ = _patched_init
    subprocess.Popen.wait = _patched_wait


def pytest_runtest_setup(item):
    global _current_test
    _current_test = item.nodeid


def pytest_runtest_teardown(item):
    global _current_test
    _current_test = None


def pytest_terminal_summary(terminalreporter, exitstatus):
    if not _ENABLED or not _timings:
        return

    per_test: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for (test, verb), durs in _timings.items():
        per_test[test][verb] += sum(durs)

    terminalreporter.section("subprocess timing breakdown (TESSERACT_INSTRUMENT_SUBPROCESS)")

    # Top 20 tests by total subprocess time.
    ranked = sorted(per_test.items(), key=lambda kv: -sum(kv[1].values()))[:20]
    for test, verbs in ranked:
        total = sum(verbs.values())
        terminalreporter.write_line(f"{total:8.1f}s  {test}")
        for verb, dur in sorted(verbs.items(), key=lambda x: -x[1])[:6]:
            terminalreporter.write_line(f"    {dur:8.1f}s  {verb}")

    # Session-wide verb totals.
    verb_totals: dict[str, float] = defaultdict(float)
    for verbs in per_test.values():
        for verb, dur in verbs.items():
            verb_totals[verb] += dur
    terminalreporter.write_line("")
    terminalreporter.write_line("session totals by verb:")
    for verb, dur in sorted(verb_totals.items(), key=lambda x: -x[1]):
        terminalreporter.write_line(f"    {dur:8.1f}s  {verb}")
