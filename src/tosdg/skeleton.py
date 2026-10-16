"""Skeleton definition, forward kinematics and anatomical checks.

A Pose maps joint -> {dof_name: degrees}; missing DOFs are 0 (rest pose).
Forward kinematics returns joint world positions/rotations in the *character*
frame (X forward, Y right, Z up, cm, origin on the ground between the feet).

Contents:
  constants: DIRECTIONS
  axis_rotation
  direction
  class Joint
  class FKResult
  class Skeleton
  segment_distance — Minimum distance between 3D segments p1q1 and p2q2
  angle_deg
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

Pose = dict[str, dict[str, float]]

_AXES = {"x": 0, "y": 1, "z": 2}
DIRECTIONS = {
    "forward": (1.0, 0.0, 0.0), "back": (-1.0, 0.0, 0.0),
    "right": (0.0, 1.0, 0.0), "left": (0.0, -1.0, 0.0),
    "up": (0.0, 0.0, 1.0), "down": (0.0, 0.0, -1.0),
}


def axis_rotation(axis: str, deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    i = _AXES[axis]
    if i == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if i == 1:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def direction(v) -> np.ndarray:
    if isinstance(v, str):
        v = DIRECTIONS[v]
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


@dataclass
class Joint:
    name: str
    parent: str | None
    offset: np.ndarray
    dofs: list[dict] = field(default_factory=list)
    virtual: bool = False

    def dof(self, name: str) -> dict | None:
        return next((d for d in self.dofs if d["name"] == name), None)


@dataclass
class FKResult:
    positions: dict[str, np.ndarray]
    rotations: dict[str, np.ndarray]
    twists: dict[str, tuple[str, float, float]] | None = None   # joint -> (dof, sign, value deg) of its last DOF when that is a roll about the bone

    def point(self, joint: str, offset=(0.0, 0.0, 0.0)) -> np.ndarray:
        return self.positions[joint] + self.rotations[joint] @ np.asarray(offset, dtype=float)

    def transformed(self, R: np.ndarray, t: np.ndarray) -> "FKResult":
        """Apply a rigid transform (e.g. character -> world)."""
        return FKResult({k: R @ p + t for k, p in self.positions.items()},
                        {k: R @ r for k, r in self.rotations.items()}, self.twists)


class Skeleton:
    def __init__(self, data: dict, source: Path | None = None):
        self.data = data
        self.id = data["id"]
        self.source = source
        self.joints: dict[str, Joint] = {}
        for j in data["joints"]:  # file order is topological (parents first)
            if j["parent"] is not None and j["parent"] not in self.joints:
                raise ValueError(f"joint {j['name']} listed before parent {j['parent']}")
            self.joints[j["name"]] = Joint(j["name"], j["parent"], np.asarray(j["offset"], float),
                                           j.get("dofs", []), j.get("virtual", False))
        self.segments = {s["name"]: s for s in data["segments"]}
        self.collision_pairs = [tuple(p) for p in data["collision_pairs"]]
        self.collision_tolerance = float(data.get("collision_tolerance", 0.85))
        self.palms = data["palms"]
        self.fingers = data["fingers"]
        self.keypoint_sets = data["keypoint_sets"]

    @classmethod
    def load(cls, path: str | Path) -> "Skeleton":
        path = Path(path)
        return cls(json.loads(path.read_text(encoding="utf-8")), path)

    # -- pose checks ----------------------------------------------------------
    def pose_errors(self, pose: Pose) -> list[str]:
        """Unknown joints/DOFs and joint-limit violations."""
        errs = []
        for jn, dofs in pose.items():
            j = self.joints.get(jn)
            if j is None:
                errs.append(f"unknown joint '{jn}'")
                continue
            for dn, v in dofs.items():
                d = j.dof(dn)
                if d is None:
                    errs.append(f"{jn}: unknown dof '{dn}'")
                elif not np.isfinite(v):
                    errs.append(f"{jn}.{dn}: non-finite value")
                elif not (d["min"] - 1e-6 <= v <= d["max"] + 1e-6):
                    errs.append(f"{jn}.{dn}={v:.1f} outside [{d['min']}, {d['max']}]")
        return errs

    # -- kinematics -----------------------------------------------------------
    def local_rotation(self, joint: Joint, values: dict[str, float]) -> np.ndarray:
        R = np.eye(3)
        for d in joint.dofs:
            v = values.get(d["name"], 0.0)
            if v:
                R = R @ axis_rotation(d["axis"], d["sign"] * v)
        return R

    def fk(self, pose: Pose) -> FKResult:
        pos, rot, twists = {}, {}, {}
        for name, j in self.joints.items():
            local = self.local_rotation(j, pose.get(name, {}))
            if j.dofs and j.dofs[-1]["axis"] == "z" and j.dofs[-1]["name"] in ("twist", "pronation"):
                d = j.dofs[-1]
                twists[name] = (d["name"], float(d["sign"]), float(pose.get(name, {}).get(d["name"], 0.0)))
            if j.parent is None:
                pos[name] = j.offset.copy()
                rot[name] = local
            else:
                pos[name] = pos[j.parent] + rot[j.parent] @ j.offset
                rot[name] = rot[j.parent] @ local
        return FKResult(pos, rot, twists)

    # -- derived geometric quantities ----------------------------------------
    def palm_normal(self, fk: FKResult, side: str) -> np.ndarray:
        p = self.palms[side]
        return fk.rotations[p["joint"]] @ np.asarray(p["normal"], float)

    def finger_curl(self, pose: Pose, side: str, finger: str) -> float:
        """Sum of flexion angles along a finger chain (degrees)."""
        return float(sum(pose.get(j, {}).get("flexion", 0.0) for j in self.fingers[side][finger]))

    def keypoints(self, fk: FKResult, set_name: str) -> tuple[list[str], np.ndarray]:
        kps = self.keypoint_sets[set_name]
        return [k["name"] for k in kps], np.array([fk.point(k["joint"], k["offset"]) for k in kps])

    def collisions(self, fk: FKResult) -> list[str]:
        """Segment pairs penetrating beyond tolerance * (r1 + r2)."""
        hits = []
        for a, b in self.collision_pairs:
            sa, sb = self.segments[a], self.segments[b]
            d = segment_distance(fk.positions[sa["from"]], fk.positions[sa["to"]],
                                 fk.positions[sb["from"]], fk.positions[sb["to"]])
            limit = self.collision_tolerance * (sa["radius"] + sb["radius"])
            if d < limit:
                hits.append(f"{a}~{b} dist={d:.1f}cm < {limit:.1f}cm")
        return hits


def segment_distance(p1, q1, p2, q2) -> float:
    """Minimum distance between 3D segments p1q1 and p2q2."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    eps = 1e-9
    if a <= eps and e <= eps:
        return float(np.linalg.norm(r))
    if a <= eps:
        s, t = 0.0, np.clip(f / e, 0, 1)
    else:
        c = d1 @ r
        if e <= eps:
            s, t = np.clip(-c / a, 0, 1), 0.0
        else:
            b = d1 @ d2
            denom = a * e - b * b
            s = np.clip((b * f - c * e) / denom, 0, 1) if denom > eps else 0.0
            t = (b * s + f) / e
            if t < 0:
                s, t = np.clip(-c / a, 0, 1), 0.0
            elif t > 1:
                s, t = np.clip((b - c) / a, 0, 1), 1.0
    return float(np.linalg.norm((p1 + d1 * s) - (p2 + d2 * t)))


def angle_deg(u, v) -> float:
    u, v = np.asarray(u, float), np.asarray(v, float)
    c = u @ v / (np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))
