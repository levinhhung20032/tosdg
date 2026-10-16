"""Pose library: load/save body poses and side-agnostic hand poses.

Body pose file  (poses/body/<id>.json):  joints use full bone names (upperarm_r).
Hand pose file  (poses/hands/<id>.json): joints use side-less names (index_01);
                                         applied to a side by appending _l/_r.
Poses are physical configurations only - no gesture semantics live here.

Contents:
  constants: POSE_SCHEMA_VERSION
  load_pose_file
  save_pose
  class PoseLibrary
  merge — Later poses override earlier ones per DOF
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from .schema import validate_instance
from .skeleton import Pose

POSE_SCHEMA_VERSION = "1.0"


def load_pose_file(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_instance(data, "pose")
    return data


def save_pose(path: str | Path, pose_id: str, joints: Pose, kind: str = "body", description: str = "") -> dict:
    data = {"schema_version": POSE_SCHEMA_VERSION, "kind": kind, "id": pose_id,
            "skeleton_family": "ue5_humanoid", "description": description,
            "joints": {j: {d: round(float(v), 4) for d, v in sorted(dofs.items())}
                       for j, dofs in sorted(joints.items()) if dofs}}
    validate_instance(data, "pose")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return data


_MIDLINE_NEGATE = ("lateral", "twist")      # a mirrored spine / neck / head bends and turns the other way; flexion stays


def mirror_pose(pose: Pose) -> Pose:
    """The left-right mirror image of a pose: _l and _r joints swap their DOF values (the skeleton's sign conventions make equal values
    on the two sides mirror images), midline joints negate lateral bend and twist."""
    out: Pose = {}
    for j, dofs in pose.items():
        if j.endswith("_l"):
            out[j[:-2] + "_r"] = dict(dofs)
        elif j.endswith("_r"):
            out[j[:-2] + "_l"] = dict(dofs)
        else:
            out[j] = {d: (-v if d in _MIDLINE_NEGATE else v) for d, v in dofs.items()}
    return out


def mirror_name(pose_id: str) -> str | None:
    """stop_bent_r -> stop_bent_l (and back); None for a pose without a side."""
    if pose_id.endswith("_l"):
        return pose_id[:-2] + "_r"
    if pose_id.endswith("_r"):
        return pose_id[:-2] + "_l"
    return None


class PoseLibrary:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.body: dict[str, dict] = {}
        self.hands: dict[str, dict] = {}
        for kind, store in (("body", self.body), ("hands", self.hands)):
            for p in sorted((self.root / kind).glob("*.json")):
                d = load_pose_file(p)
                if d["id"] != p.stem:
                    raise ValueError(f"{p}: id '{d['id']}' does not match filename")
                store[d["id"]] = d

    def body_pose(self, pose_id: str) -> Pose:
        if pose_id not in self.body:
            raise ValueError(f"unknown body pose '{pose_id}'; known: {', '.join(sorted(self.body))}")
        return copy.deepcopy(self.body[pose_id]["joints"])

    def hand_pose(self, pose_id: str, side: str) -> Pose:
        if pose_id not in self.hands:
            raise ValueError(f"unknown hand pose '{pose_id}'; known: {', '.join(sorted(self.hands))}")
        return {f"{j}_{side}": dict(v) for j, v in self.hands[pose_id]["joints"].items()}


def merge(*poses: Pose) -> Pose:
    """Later poses override earlier ones per DOF."""
    out: Pose = {}
    for p in poses:
        for j, dofs in p.items():
            out.setdefault(j, {}).update(dofs)
    return out
