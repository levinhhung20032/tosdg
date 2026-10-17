"""Parallel generation (parallel.py): the machine's limits, and the same files for any number of worker processes."""
import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from helpers import make_config

from tosdg import parallel
from tosdg.cli import main
from tosdg.generate import run
from tosdg.scene import GenerationContext

MACHINE = {"cpus": 8, "ram_free_mb": 10000.0, "ram_total_mb": 16000.0, "os": "Windows",
           "gpu": {"name": "Test GPU", "vram_total_mb": 6144.0, "vram_free_mb": 4000.0}}


class Limits(unittest.TestCase):
    def test_caps_by_cpu_ram_and_video_memory(self):
        self.assertEqual(parallel.max_workers("gpu", MACHINE)[0], 2)          # video memory (4000 - 512) // 1300, RAM (10000 - 2048) // 3500
        self.assertEqual(parallel.max_workers("cpu", MACHINE)[0], 3)          # (10000 - 2048) // 2200
        self.assertEqual(parallel.max_workers("proxy", MACHINE)[0], 7)        # 8 CPUs - 1 for the parent
        no_gpu = {**MACHINE, "gpu": None}
        n, limits = parallel.max_workers("gpu", no_gpu)
        self.assertEqual(n, parallel.UNKNOWN_VRAM_WORKERS)
        self.assertTrue(any("unknown" in l for l in limits))
        tiny = {**MACHINE, "cpus": 1, "ram_free_mb": 100.0}
        self.assertEqual(parallel.max_workers("cpu", tiny)[0], 1)             # never below one
        self.assertEqual(parallel.max_workers("cpu", {**MACHINE, "ram_free_mb": None})[0], 7)   # RAM unknown: not limited

    def test_plan(self):
        self.assertEqual(parallel.plan(1, "gpu", MACHINE), (1, "1 worker"))
        n, note = parallel.plan("auto", "gpu", MACHINE)
        self.assertEqual(n, 2)
        self.assertIn("auto", note)
        n, note = parallel.plan(6, "gpu", MACHINE)
        self.assertEqual(n, 2)
        self.assertIn("6 asked", note)
        self.assertEqual(parallel.plan(2, "cpu", MACHINE)[0], 2)
        with self.assertRaises(ValueError):
            parallel.plan(0, "cpu", MACHINE)

    def test_this_machine_is_read(self):
        info = parallel.machine()
        self.assertGreaterEqual(info["cpus"], 1)
        self.assertIn("gpu", info)
        if info["ram_free_mb"] is not None:
            self.assertGreater(info["ram_total_mb"], info["ram_free_mb"] * 0.99)


class SavedProfile(unittest.TestCase):
    """data/machine.json: measured on first use, read afterwards, measured again when stale or asked."""

    def setUp(self):
        self.td = Path(tempfile.mkdtemp())
        self.path = self.td / "machine.json"
        self.calls = 0
        self._measure = parallel.measure_profile

        def fake():
            self.calls += 1
            prof = self._measure()
            prof["saved_utc"] = f"2026-01-0{self.calls}T00:00:00+00:00"
            return prof
        parallel.measure_profile = fake

    def tearDown(self):
        parallel.measure_profile = self._measure
        shutil.rmtree(self.td)

    def test_measured_once_then_read(self):
        logs = []
        a = parallel.load_profile(self.path, log=logs.append)
        self.assertTrue(self.path.exists())
        self.assertEqual(self.calls, 1)
        self.assertIn("first run", logs[0])
        b = parallel.load_profile(self.path, log=logs.append)
        self.assertEqual((self.calls, b["saved_utc"]), (1, a["saved_utc"]))       # read, not measured again
        self.assertEqual(len(logs), 1)
        self.assertEqual(set(a["max_workers"]), {"gpu", "cpu", "proxy"})
        parallel.load_profile(self.path, refresh=True)
        self.assertEqual(self.calls, 2)

    def test_measured_again_when_stale_or_broken(self):
        prof = parallel.load_profile(self.path)
        prof["host"] = "another-machine"
        self.path.write_text(json.dumps(prof), encoding="utf-8")
        logs = []
        parallel.load_profile(self.path, log=logs.append)
        self.assertEqual(self.calls, 2)
        self.assertIn("another machine", logs[0])
        self.path.write_text("{not json", encoding="utf-8")
        parallel.load_profile(self.path, log=logs.append)
        self.assertEqual(self.calls, 3)
        self.assertIn("unreadable", logs[-1])

    def test_plan_uses_the_saved_caps(self):
        prof = parallel.load_profile(self.path)
        prof["max_workers"]["cpu"] = 3
        self.path.write_text(json.dumps(prof), encoding="utf-8")
        n, note = parallel.plan("auto", "cpu", profile_path=self.path)
        self.assertEqual(n, 3)
        self.assertIn("--refresh", note)
        self.assertEqual(parallel.plan(9, "cpu", profile_path=self.path)[0], 3)
        self.assertEqual(parallel.plan(1, "cpu", profile_path=self.td / "none.json"), (1, "1 worker"))   # one worker: nothing read
        self.assertFalse((self.td / "none.json").exists())


class SameFiles(unittest.TestCase):
    def setUp(self):
        self.td = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.td)

    def check(self, **over):
        cfg = make_config(self.td, num_samples=4, **over)
        ctx = GenerationContext(cfg)
        one = run(ctx, self.td / "one", log=lambda *_: None)
        lines = []
        two = parallel.run_parallel(ctx, cfg, None, self.td / "two", 2, log=lines.append)
        self.assertEqual(len(lines), 4)
        self.assertEqual([s["sha256"] for s in one["samples"]], [s["sha256"] for s in two["samples"]])
        strip = lambda m: {k: v for k, v in m.items() if k != "created_utc"}
        self.assertEqual(strip(one), strip(two))
        self.assertEqual(sorted(p.name for p in (self.td / "one").iterdir()), sorted(p.name for p in (self.td / "two").iterdir()))

    def test_proxy_backend(self):
        self.check(backend="proxy", uniform_brightness_range=[1.0, 1.0])

    def test_mesh_backend(self):
        self.check()

    def test_a_dead_worker_does_not_change_the_files(self):
        import os
        cfg = make_config(self.td, num_samples=4, backend="proxy", uniform_brightness_range=[1.0, 1.0])
        ctx = GenerationContext(cfg)
        one = run(ctx, self.td / "one", log=lambda *_: None)
        lines = []
        os.environ["TOSDG_TEST_CRASH_INDEX"] = "1"
        try:
            two = parallel.run_parallel(ctx, cfg, None, self.td / "two", 2, log=lines.append)
        finally:
            del os.environ["TOSDG_TEST_CRASH_INDEX"]
        self.assertTrue(any(l.startswith("warning:") and "lost with a worker" in l for l in lines))
        self.assertEqual([s["sha256"] for s in one["samples"]], [s["sha256"] for s in two["samples"]])

    def test_cli(self):
        import os
        os.environ["TOSDG_MACHINE_FILE"] = str(self.td / "machine.json")
        self.addCleanup(os.environ.pop, "TOSDG_MACHINE_FILE", None)
        cfg = make_config(self.td, num_samples=2, backend="proxy", uniform_brightness_range=[1.0, 1.0])
        out = io.StringIO()
        with redirect_stdout(out):
            rc = main(["generate", str(cfg), "--out", str(self.td / "cli"), "--workers", "2"])
        self.assertEqual(rc, 0, out.getvalue())
        self.assertIn("workers: 2 worker(s)", out.getvalue())
        with redirect_stdout(io.StringIO()) as rep:
            self.assertEqual(main(["reproduce", str(self.td / "cli"), "--workers", "2"]), 0)
        self.assertIn("2 identical to the manifest", rep.getvalue())
        self.assertTrue((self.td / "machine.json").exists())                       # saved by the first --workers run
        with redirect_stdout(io.StringIO()) as m:
            self.assertEqual(main(["machine"]), 0)
        self.assertIn("saved profile", m.getvalue())
        with redirect_stdout(io.StringIO()) as m:
            self.assertEqual(main(["machine", "--refresh"]), 0)
        self.assertIn("measured (asked)", m.getvalue())
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            main(["generate", str(cfg), "--out", str(self.td / "x"), "--workers", "0"])
        # --renderer is for the mesh backend: a one-line error, exit 2
        with redirect_stdout(io.StringIO()), redirect_stderr_to_string() as err:
            self.assertEqual(main(["generate", str(cfg), "--out", str(self.td / "y"), "--renderer", "cpu"]), 2)
        self.assertIn("mesh_renderer", err.getvalue())


class redirect_stderr_to_string:
    def __enter__(self):
        import sys
        self._old, self.buf = sys.stderr, io.StringIO()
        sys.stderr = self.buf
        return self.buf

    def __exit__(self, *exc):
        import sys
        sys.stderr = self._old
        return False


if __name__ == "__main__":
    unittest.main()
