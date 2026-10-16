"""Gesture semantics -> constrained, seeded pose sampling.

Pipeline (per sample):
    GestureDefinition
      -> pick pose variant (weighted) -> base body pose (pose library)
      -> pick hand variant per side   -> hand poses (pose library)
      -> controlled variation (bounded per-DOF deltas, seeded)
      -> validity: joint limits, self-collision, gesture constraints
      -> reject & redraw until valid or max_tries exhausted

A gesture is semantic; constraints are evaluated on the *resulting* geometry,
so any pose that satisfies them is a valid realisation of the gesture.

Contents:
  class PoseSamplingError
  class SampledPose
  constraint_failures
  apply_variation
  pose_validity — All reasons this pose is not a valid realisation of…
  rejection_kind
  class GestureLibrary
  approach_side — Which side of the officer an observer at `azimuth_deg` is on
  side_boundary_distance — Angular distance (deg) to the nearest…
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import rng as rngmod
from . import baton as batonmod
from .poses import PoseLibrary, merge
from .skeleton import FKResult, Pose, Skeleton, angle_deg, direction


class PoseSamplingError(RuntimeError):
    pass


@dataclass
class SampledPose:
    gesture_id: str
    pose_variant: str
    hand_variants: dict[str, str]
    signal_arms: list[str]
    pose: Pose
    tries: int
    rejections: dict[str, int] = field(default_factory=dict)
    baton: dict | None = None            # {"hand", "mode", "closure", "thumb_closure"} when a baton is held


def constraint_failures(c_list: list[dict], skeleton: Skeleton, pose: Pose, fk: FKResult) -> list[str]:
    fails = []
    for c in c_list:
        t = c["type"]
        if t == "limb_direction":
            v = fk.positions[c["to"]] - fk.positions[c["from"]]
            a = angle_deg(v, direction(c["direction"]))
            if a > c["max_angle_deg"]:
                fails.append(f"limb_direction {c['from']}->{c['to']} {a:.0f}deg > {c['max_angle_deg']}")
        elif t == "palm_facing":
            a = angle_deg(skeleton.palm_normal(fk, c["side"]), direction(c["direction"]))
            if a > c["max_angle_deg"]:
                fails.append(f"palm_facing {c['side']} {a:.0f}deg > {c['max_angle_deg']}")
        elif t in ("above", "below"):
            dz = fk.positions[c["joint"]][2] - fk.positions[c["reference"]][2]
            m = c.get("margin_cm", 0.0)
            ok = dz > m if t == "above" else dz < -m
            if not ok:
                fails.append(f"{t} {c['joint']} vs {c['reference']} dz={dz:.1f}")
        elif t == "finger_curl":
            for f in c["fingers"]:
                curl = skeleton.finger_curl(pose, c["side"], f)
                lo, hi = c.get("min_deg", -1e9), c.get("max_deg", 1e9)
                if not lo <= curl <= hi:
                    fails.append(f"finger_curl {c['side']}.{f}={curl:.0f} not in [{lo}, {hi}]")
        else:  # schema should have caught this
            raise ValueError(f"unknown constraint type {t}")
    return fails


def apply_variation(rng: np.random.Generator, pose: Pose, variation: dict[str, list[float]]) -> Pose:
    out = merge(pose)
    for key in sorted(variation):  # sorted => draw order independent of JSON key order
        joint, dof = key.split(".")
        delta = rngmod.uniform(rng, variation[key])
        out.setdefault(joint, {})[dof] = out.get(joint, {}).get(dof, 0.0) + delta
    return out


def pose_validity(skeleton: Skeleton, gesture: dict, variant: dict, pose: Pose,
                  baton: dict | None = None) -> tuple[list[str], FKResult]:
    """All reasons this pose is not a valid realisation of `gesture`/`variant` (empty = valid). `baton` = a normalised baton spec
    (see baton.py): the baton must not penetrate the body."""
    errs = skeleton.pose_errors(pose)
    fk = skeleton.fk(pose)
    if not errs:
        errs += [f"collision {h}" for h in skeleton.collisions(fk)]
        constraints = gesture.get("constraints", []) + variant.get("constraints", [])
        if baton:        # the baton hand grips the baton with a straight wrist: "open hand" (finger_curl) requirements cannot apply to it
            hand = f"hand_{baton['hand']}"          # nor can the wrist bend or the palm turn out: "hand up / palm out" requirements describe an open, bent hand
            constraints = [c for c in constraints if not (c["type"] in ("finger_curl", "palm_facing") and c["side"] == baton["hand"])
                           and not (c["type"] == "limb_direction" and c["from"] == hand)]
        errs += constraint_failures(constraints, skeleton, pose, fk)
        if baton:
            errs += [f"baton {h}" for h in batonmod.body_collisions(skeleton, fk, baton, variant.get("baton_axis", "across"))]
    return errs, fk


def rejection_kind(err: str) -> str:
    if "outside [" in err:
        return "joint_limit"
    return err.split(" ")[0]


class GestureLibrary:
    def __init__(self, data: dict, poses: PoseLibrary):
        self.data = data
        self.poses = poses
        self.gestures = {g["id"]: g for g in data["gestures"]}
        self.common_variation = data.get("common_variation", {})
        self._check_references()

    def _check_references(self) -> None:
        for g in self.gestures.values():
            for v in g["pose_variants"]:
                if v["pose"] not in self.poses.body:
                    raise ValueError(f"gesture {g['id']}: unknown body pose '{v['pose']}'")
                for side, hands in v.get("hands", {}).items():
                    for h in hands:
                        if h not in self.poses.hands:
                            raise ValueError(f"gesture {g['id']}: unknown hand pose '{h}'")

    def sample(self, gesture_id: str, skeleton: Skeleton, rng: np.random.Generator,
               max_tries: int = 200, baton: dict | None = None) -> SampledPose:
        """`baton`: a normalised baton spec (baton.normalise) or None. The grip is deterministic (no RNG draws), so enabling it never
        changes the other random dimensions of a sample."""
        g = self.gestures[gesture_id]
        rejections: dict[str, int] = {}
        variants = {v["id"]: v for v in g["pose_variants"]}
        for attempt in range(1, max_tries + 1):
            vid = rngmod.weighted_choice(rng, {k: v.get("weight", 1.0) for k, v in variants.items()})
            v = variants[vid]
            pose = self.poses.body_pose(v["pose"])
            hand_choice = {}
            for side in ("l", "r"):
                opts = v.get("hands", {}).get(side, {"relaxed": 1.0})
                hand_choice[side] = rngmod.weighted_choice(rng, opts)
                pose = merge(pose, self.poses.hand_pose(hand_choice[side], side))
            variation = {**self.common_variation, **g.get("variation", {}), **v.get("variation", {})}
            pose = apply_variation(rng, pose, variation)
            grip = None
            if baton:
                pose, grip = batonmod.apply_grip(skeleton, pose, baton, v.get("baton_axis", "across"), self.poses.hand_pose("fist", baton["hand"]))
            errs, _ = pose_validity(skeleton, g, v, pose, baton)
            if not errs:
                pose = {j: {d: round(x, 4) for d, x in dofs.items()} for j, dofs in pose.items()}
                return SampledPose(gesture_id, vid, hand_choice, list(v["signal_arms"]), pose, attempt, rejections, grip)
            kind = rejection_kind(errs[0])
            rejections[kind] = rejections.get(kind, 0) + 1
        raise PoseSamplingError(f"{gesture_id}: no valid pose in {max_tries} tries; rejections={rejections}")


def approach_side(azimuth_deg: float) -> str:
    """Which side of the officer an observer at `azimuth_deg` is on.
    Azimuth is measured in the officer frame: 0 = in front, +90 = officer's right."""
    a = (azimuth_deg + 180.0) % 360.0 - 180.0
    if -45 <= a <= 45:
        return "front"
    if 45 < a < 135:
        return "right"
    if -135 < a < -45:
        return "left"
    return "back"


def side_boundary_distance(azimuth_deg: float) -> float:
    """Angular distance (deg) to the nearest front/right/back/left boundary."""
    a = (azimuth_deg - 45.0) % 90.0
    return float(min(a, 90.0 - a))
