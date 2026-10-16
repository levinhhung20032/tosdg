"""Deterministic seeding.

All randomness flows from explicit integer seeds. A sample seed is derived from
(master_seed, sample_index); each subsystem draws from its own *named stream*
so adding a new randomized dimension never shifts the values of existing ones.
No module may use global/unseeded RNG state.

Contents:
  derive_sample_seed — Seed of sample `sample_index` within a batch
  derive_attempt_seed — Seed used for resampling attempt `attempt` (0 = first try)…
  stream — Independent generator for subsystem `name`
  weighted_choice — Pick a key with probability proportional to its weight
  uniform
  (+ 1 private helper)
"""
from __future__ import annotations

import hashlib

import numpy as np

_MASK64 = (1 << 64) - 1


def _hash_to_u64(*parts: object) -> int:
    h = hashlib.sha256(":".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little") & _MASK64


def derive_sample_seed(master_seed: int, sample_index: int) -> int:
    """Seed of sample `sample_index` within a batch. Stable across platforms."""
    return _hash_to_u64("sample", int(master_seed), int(sample_index))


def derive_attempt_seed(sample_seed: int, attempt: int) -> int:
    """Seed used for resampling attempt `attempt` (0 = first try) of a sample."""
    if attempt == 0:
        return int(sample_seed)
    return _hash_to_u64("attempt", int(sample_seed), int(attempt))


def stream(seed: int, name: str) -> np.random.Generator:
    """Independent generator for subsystem `name` (e.g. 'pose', 'camera')."""
    return np.random.Generator(np.random.PCG64(_hash_to_u64("stream", int(seed), name)))


def weighted_choice(rng: np.random.Generator, weights: dict[str, float]) -> str:
    """Pick a key with probability proportional to its weight. Keys are sorted
    so the result never depends on dict insertion order."""
    keys = sorted(k for k, w in weights.items() if w > 0)
    if not keys:
        raise ValueError("weighted_choice: no positive weights")
    w = np.array([weights[k] for k in keys], dtype=float)
    return keys[int(rng.choice(len(keys), p=w / w.sum()))]


def uniform(rng: np.random.Generator, lo_hi) -> float:
    lo, hi = float(lo_hi[0]), float(lo_hi[1])
    return lo if lo == hi else float(rng.uniform(lo, hi))
