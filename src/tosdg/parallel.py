"""Parallel generation (`tosdg generate --workers N`, `tosdg reproduce --workers N`).

Every sample has its own seed (H(master_seed, index)) and draws only from named streams of it, so samples can be made in
any order by any process: N worker processes, each with its own GenerationContext (and, with the GPU renderer, its own
OpenGL context), make samples by index; the parent collects the results and writes the manifest. The files are
byte-identical to a one-process run for any N (tests/test_parallel.py).

Before a run the tool reads the machine (`machine`) and caps N (`max_workers`): logical CPUs minus one for the parent,
free memory over RAM_PER_WORKER_MB, and with the GPU renderer free video memory over VRAM_PER_WORKER_MB. Measured on an
RTX 3060 Laptop (8 samples 1280x720): a CPU worker peaks at ~1.8 GB of committed memory (the 4096^2 textures' mip chains),
a GPU worker at ~3.3 GB (mostly the graphics driver) and ~1.3 GB of video memory. "Free memory" is the smaller of the free
physical memory and, on Windows, the commit still available (a machine without a page file cannot commit more than its
RAM: 11 CPU workers ran out of memory there before this was counted). Asking for more than the cap runs the cap (and says
so); "auto" runs the cap. A worker that dies (out of memory, killed) does not stop the run: its samples are made again in
the parent process, so the files are the same.

The machine is measured once: the first run with --workers saves its profile (CPUs, memory, GPU and the cap of every
renderer) to data/machine.json (git-ignored), and later runs read it instead of measuring again. It is measured again
with `tosdg machine --refresh`, or by itself when the file is from another machine (host name, CPU count), was made with
other per-worker sizes, or cannot be read. The saved caps reflect the free memory at that moment; a run that later finds
less memory still finishes (dead workers' samples are remade), and --refresh updates the caps.

Contents:
  constants: RAM_PER_WORKER_MB, VRAM_PER_WORKER_MB, RAM_RESERVE_MB, VRAM_RESERVE_MB, UNKNOWN_VRAM_WORKERS, RENDERERS, PROFILE_VERSION
  machine — CPUs, RAM and the GPU's video memory of this machine (None where it cannot be read)
  max_workers — the largest useful number of worker processes for a renderer, and why
  plan — the number of workers to run for a request (an int or "auto"), with a one-line note
  default_profile_path / measure_profile / save_profile / load_profile — the saved machine profile (data/machine.json)
  run_parallel — generate.run with N worker processes (same manifest, same files)
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

RAM_PER_WORKER_MB = {"cpu": 2200, "gpu": 3500, "proxy": 800}    # peak committed memory (+ ~20 %): CPU 1.8 GB, GPU 3.3 GB (driver)
VRAM_PER_WORKER_MB = 1300
RAM_RESERVE_MB = 2048            # left to the parent, the system and other programs
VRAM_RESERVE_MB = 512            # left to the display / other programs
UNKNOWN_VRAM_WORKERS = 2         # GPU renderer on a machine whose free video memory cannot be read
RENDERERS = ("gpu", "cpu", "proxy")
PROFILE_VERSION = "1.0"          # data/machine.json (schemas/machine.schema.json)


def _ram_mb() -> tuple[float | None, float | None]:
    """(free, total) memory in MB: free = what new processes can still use."""
    try:
        if sys.platform == "win32":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(st)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                # ullAvailPageFile = the commit still available (RAM + page file - what is committed): the real limit
                return min(st.ullAvailPhys, st.ullAvailPageFile) / 2**20, st.ullTotalPhys / 2**20
        elif Path("/proc/meminfo").exists():
            info = {l.split(":")[0]: float(l.split()[1]) / 1024 for l in Path("/proc/meminfo").read_text().splitlines() if ":" in l}
            return info.get("MemAvailable"), info.get("MemTotal")
        import psutil                                             # optional elsewhere (macOS)
        vm = psutil.virtual_memory()
        return vm.available / 2**20, vm.total / 2**20
    except Exception:
        return None, None


def _gpu() -> dict | None:
    """{"name", "vram_total_mb", "vram_free_mb"} of the first NVIDIA GPU (nvidia-smi), else None."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        if r.returncode != 0 or not r.stdout.strip():
            return None
        name, total, free = (x.strip() for x in r.stdout.strip().splitlines()[0].split(","))
        return {"name": name, "vram_total_mb": float(total), "vram_free_mb": float(free)}
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def machine() -> dict:
    """{"cpus", "ram_free_mb", "ram_total_mb", "gpu": {...} | None, "os"} of this machine."""
    cpus = (os.process_cpu_count() if hasattr(os, "process_cpu_count") else os.cpu_count()) or 1
    free, total = _ram_mb()
    return {"cpus": int(cpus), "ram_free_mb": free, "ram_total_mb": total, "gpu": _gpu(), "os": platform.system()}


def max_workers(renderer: str, info: dict | None = None) -> tuple[int, list[str]]:
    """The largest number of worker processes worth running with `renderer` ("cpu", "gpu" or "proxy") on this machine, and
    the limits that set it (one line each)."""
    info = info or machine()
    limits = [f"CPU {info['cpus']} logical - 1 for the parent = {max(1, info['cpus'] - 1)}"]
    caps = [max(1, info["cpus"] - 1)]
    per = RAM_PER_WORKER_MB.get(renderer, RAM_PER_WORKER_MB["cpu"])
    if info.get("ram_free_mb") is not None:
        n = int(max(0.0, info["ram_free_mb"] - RAM_RESERVE_MB) // per)
        caps.append(max(1, n))
        limits.append(f"RAM {info['ram_free_mb'] / 1024:.1f} GB free - {RAM_RESERVE_MB / 1024:.0f} GB reserve, {per / 1024:.1f} GB per worker = {max(1, n)}")
    else:
        limits.append("RAM: unknown (not limited)")
    if renderer == "gpu":
        g = info.get("gpu")
        if g:
            n = int(max(0.0, g["vram_free_mb"] - VRAM_RESERVE_MB) // VRAM_PER_WORKER_MB)
            caps.append(max(1, n))
            limits.append(f"GPU {g['name']}: {g['vram_free_mb'] / 1024:.1f} GB video memory free - {VRAM_RESERVE_MB / 1024:.1f} GB "
                          f"reserve, {VRAM_PER_WORKER_MB / 1024:.1f} GB per worker = {max(1, n)}")
        else:
            caps.append(UNKNOWN_VRAM_WORKERS)
            limits.append(f"GPU: video memory unknown (no nvidia-smi) = {UNKNOWN_VRAM_WORKERS}")
    return min(caps), limits


def plan(requested, renderer: str, info: dict | None = None, profile_path: str | Path | None = None, log=None) -> tuple[int, str]:
    """(workers to run, note) for a request: an int >= 1 or "auto" (= the machine's cap). The cap comes from `info` when
    given (tests), else from the saved machine profile (`load_profile`, measured and saved on first use)."""
    if requested in (None, 1, "1"):
        return 1, "1 worker"
    if info is not None:
        cap, limits = max_workers(renderer, info)
        where = ""
    else:
        prof = load_profile(profile_path, log=log)
        cap, limits = prof["max_workers"][renderer], prof["limits"][renderer]
        where = f"; machine profile of {prof['saved_utc'][:10]}, `tosdg machine --refresh` to measure again"
    if requested == "auto":
        return cap, f"{cap} worker(s) (auto, {renderer} renderer; " + "; ".join(limits) + where + ")"
    n = int(requested)
    if n < 1:
        raise ValueError("--workers must be 1 or more, or 'auto'")
    if n > cap:
        return cap, (f"{cap} worker(s): {n} asked, this machine runs at most {cap} with the {renderer} renderer ("
                     + "; ".join(limits) + where + ")")
    return n, f"{n} worker(s) ({renderer} renderer; at most {cap} on this machine{where})"


# ----------------------------------------------------------------------------- the saved machine profile
def default_profile_path() -> Path:
    """<repo>/data/machine.json (git-ignored: it belongs to this machine), or $TOSDG_MACHINE_FILE."""
    from .schema import repo_root
    env = os.environ.get("TOSDG_MACHINE_FILE")
    return Path(env) if env else repo_root() / "data" / "machine.json"


def measure_profile() -> dict:
    """Read this machine now and work out the cap of every renderer: the machine profile (schema machine 1.0)."""
    from datetime import datetime, timezone
    info = machine()
    caps, limits = {}, {}
    for r in RENDERERS:
        caps[r], limits[r] = max_workers(r, info)
    g = info["gpu"]
    return {"schema_version": PROFILE_VERSION, "saved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "host": platform.node(), "os": info["os"], "cpus": info["cpus"],
            "ram_total_mb": None if info["ram_total_mb"] is None else round(info["ram_total_mb"]),
            "ram_free_mb": None if info["ram_free_mb"] is None else round(info["ram_free_mb"]),
            "gpu": None if g is None else {"name": g["name"], "vram_total_mb": round(g["vram_total_mb"]), "vram_free_mb": round(g["vram_free_mb"])},
            "per_worker_mb": {"ram": dict(RAM_PER_WORKER_MB), "vram": VRAM_PER_WORKER_MB},
            "max_workers": caps, "limits": limits}


def save_profile(prof: dict, path: str | Path | None = None) -> Path:
    from .schema import validate_instance
    import json
    validate_instance(prof, "machine")
    p = Path(path) if path else default_profile_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(prof, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return p


def _same_machine(prof: dict) -> bool:
    """The profile was measured here: same host name and CPU count, and the per-worker sizes it used are the current ones."""
    cpus = (os.process_cpu_count() if hasattr(os, "process_cpu_count") else os.cpu_count()) or 1
    return (prof.get("host") == platform.node() and prof.get("cpus") == cpus
            and prof.get("per_worker_mb") == {"ram": dict(RAM_PER_WORKER_MB), "vram": VRAM_PER_WORKER_MB})


def load_profile(path: str | Path | None = None, refresh: bool = False, log=None) -> dict:
    """The saved machine profile; measured and saved first when there is none, when it is from another machine (host name,
    CPU count) or older per-worker sizes, when it cannot be read, or when `refresh` is asked."""
    import json
    from .schema import SchemaError, validate_instance
    p = Path(path) if path else default_profile_path()
    reason = "asked" if refresh else None
    if reason is None:
        if not p.exists():
            reason = "first run"
        else:
            try:
                prof = json.loads(p.read_text(encoding="utf-8"))
                validate_instance(prof, "machine")
                if _same_machine(prof):
                    return prof
                reason = "another machine or older settings"
            except (ValueError, OSError, SchemaError):
                reason = "unreadable"
    prof = measure_profile()
    saved = save_profile(prof, p)
    if log:
        log(f"machine profile measured ({reason}) and saved to {saved}")
    return prof


# ----------------------------------------------------------------------------- worker processes
_WORKER: dict = {}


def _init_worker(config_path: str, overrides: dict, out_dir: str) -> None:
    """Each worker builds its own context once (assets loaded, GL context on first render); its prints are dropped (the
    parent reports), errors still raise."""
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
    from .scene import GenerationContext
    _WORKER["ctx"] = GenerationContext(config_path, overrides or None)
    _WORKER["out"] = Path(out_dir)


def _work(index: int) -> dict:
    from .generate import generate_index
    if os.environ.get("TOSDG_TEST_CRASH_INDEX") == str(index):        # tests: a worker that dies abruptly
        os._exit(3)
    ctx = _WORKER["ctx"]
    return generate_index(ctx, _WORKER["out"], index, ctx.config.get("max_attempts_per_sample", 20))


def run_parallel(ctx, config_path: str | Path, overrides: dict | None, out_dir: str | Path, workers: int,
                 indices: list[int] | None = None, log=print, overwrite: bool = False) -> dict:
    """generate.run with `workers` processes. `ctx` is the parent's context (it already read and stored the library's
    people / shadows, so the workers only read); `config_path` + `overrides` rebuild the same context in each worker.
    Samples are written by the workers; the manifest (samples by index) by the parent."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp
    from .generate import build_manifest, generate_index, prepare_output_dir, run
    if workers <= 1:
        return run(ctx, out_dir, indices, log=log, overwrite=overwrite)
    out = Path(out_dir)
    prepare_output_dir(out, overwrite)
    indices = list(range(ctx.config["num_samples"])) if indices is None else list(indices)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(k, "1")                      # one thread per worker: the processes are the parallelism
    samples, failures, rejections = [], [], {}

    def take(r):
        log(r["line"])
        if r["sample"] is not None:
            samples.append(r["sample"])
        if r["failure"] is not None:
            failures.append(r["failure"])
        for k, v in r["rejections"].items():
            rejections[k] = rejections.get(k, 0) + v
    redo, why = [], None
    with ProcessPoolExecutor(max_workers=min(workers, len(indices)) or 1, mp_context=mp.get_context("spawn"),
                             initializer=_init_worker, initargs=(str(Path(config_path).resolve()), dict(overrides or {}), str(out))) as pool:
        futures = {pool.submit(_work, i): i for i in indices}
        for f in as_completed(futures):
            try:
                take(f.result())
            except Exception as e:                       # a worker died (out of memory, killed): its sample is made again below
                redo.append(futures[f])
                why = why or f"{type(e).__name__}: {e}"
    if redo:
        log(f"warning: {len(redo)} sample(s) were lost with a worker ({why}); making them in this process "
            "(use fewer --workers if this repeats)")
        max_attempts = ctx.config.get("max_attempts_per_sample", 20)
        for i in sorted(redo):
            take(generate_index(ctx, out, i, max_attempts))
    return build_manifest(ctx, out, samples, failures, rejections)
