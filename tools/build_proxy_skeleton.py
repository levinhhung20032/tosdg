"""Generate characters/proxy_v1/skeleton.json (the proxy test character).

The skeleton is authored here so left/right sides are mirrored exactly; the
JSON file is the asset that the rest of the system reads. Re-run after edits:

    python tools/build_proxy_skeleton.py

Bone names follow the UE5 Mannequin (Manny/Quinn) convention so the Unreal
backend can map 1:1. Joints marked "virtual" have no UE bone; in Unreal they
become sockets.

Frame: X forward, Y right, Z up, centimetres (Unreal world convention).
Rest pose: standing, arms hanging, palms facing the thighs, all local rotations
identity. A joint's DOFs rotate the joint's own frame (and thus its children);
the local rotation is R = R(dof_0) @ R(dof_1) @ ... in listed order.

Contents:
  constants: OUT, SPINE, FINGERS, COCO17
  J
  S
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "characters" / "proxy_v1" / "skeleton.json"

joints: list[dict] = []
segments: list[dict] = []


def J(name, parent, offset, dofs=(), virtual=False):
    j = {"name": name, "parent": parent, "offset": [round(float(v), 3) for v in offset]}
    if dofs:
        j["dofs"] = [dict(zip(("name", "axis", "sign", "min", "max"), d)) for d in dofs]
    if virtual:
        j["virtual"] = True
    joints.append(j)


def S(name, a, b, radius, region, side=None):
    s = {"name": name, "from": a, "to": b, "radius": radius, "region": region}
    if side:
        s["side"] = side
    segments.append(s)


# ---- spine / head ---------------------------------------------------------
SPINE = [("flexion", "y", 1, -10, 25), ("lateral", "x", -1, -12, 12), ("twist", "z", 1, -15, 15)]
J("root", None, (0, 0, 0), virtual=True)
J("pelvis", "root", (0, 0, 95))
J("spine_01", "pelvis", (0, 0, 10), SPINE)
J("spine_02", "spine_01", (0, 0, 15), SPINE)
J("spine_03", "spine_02", (0, 0, 15), SPINE)
J("neck_01", "spine_03", (0, 0, 15), [("flexion", "y", 1, -30, 40), ("lateral", "x", -1, -25, 25), ("twist", "z", 1, -45, 45)])
J("head", "neck_01", (0, 0, 10), [("flexion", "y", 1, -30, 30), ("lateral", "x", -1, -15, 15), ("twist", "z", 1, -40, 40)])
J("head_end", "head", (0, 0, 12), virtual=True)
J("cap_top", "head", (0, 0, 19), virtual=True)

S("torso_lower", "pelvis", "spine_02", 15, "torso")
S("torso_upper", "spine_02", "neck_01", 16, "torso")
S("neck", "neck_01", "head", 5.5, "neck")
S("head", "head", "head_end", 10, "head")
S("cap", "head_end", "cap_top", 10.5, "headwear")
S("hips", "thigh_l", "thigh_r", 12, "pelvis")

FINGERS = {  # name: (base offset from hand [x, |y| palm-side, z], phalanx lengths, tip len)
    "index": ((3.5, 0.0, -9.0), (4.5, 2.6, 2.0)),
    "middle": ((1.2, 0.0, -9.3), (5.0, 3.0, 2.2)),
    "ring": ((-1.2, 0.0, -9.0), (4.7, 2.8, 2.1)),
    "pinky": ((-3.4, 0.0, -8.3), (3.6, 2.1, 1.9)),
}

for side, s in (("l", -1), ("r", 1)):
    # s = sign of the lateral (Y) direction for this side; palm faces -s (medial).
    other = "r" if side == "l" else "l"
    J(f"clavicle_{side}", "spine_03", (0, 3 * s, 10),
      [("elevation", "x", s, -10, 30), ("protraction", "z", -s, -15, 20)])
    J(f"upperarm_{side}", f"clavicle_{side}", (0, 17 * s, 0),
      [("flexion", "y", -1, -60, 180), ("abduction", "x", s, -30, 180), ("twist", "z", -s, -90, 90)])
    J(f"lowerarm_{side}", f"upperarm_{side}", (0, 0, -30),
      [("flexion", "y", -1, 0, 150), ("pronation", "z", -s, -80, 90)])
    J(f"hand_{side}", f"lowerarm_{side}", (0, 0, -26),
      [("flexion", "x", -s, -90, 80), ("deviation", "y", 1, -20, 30)])
    # fingers: flexion curls toward the palm (palm normal = -s * Y at rest)
    for f, (base, lens) in FINGERS.items():
        J(f"{f}_01_{side}", f"hand_{side}", base,
          [("flexion", "x", -s, -20, 90), ("abduction", "y", 1, -20, 20)])
        J(f"{f}_02_{side}", f"{f}_01_{side}", (0, 0, -lens[0]), [("flexion", "x", -s, 0, 110)])
        J(f"{f}_03_{side}", f"{f}_02_{side}", (0, 0, -lens[1]), [("flexion", "x", -s, 0, 80)])
        J(f"{f}_tip_{side}", f"{f}_03_{side}", (0, 0, -lens[2]), virtual=True)
    J(f"thumb_01_{side}", f"hand_{side}", (2.6, -1.6 * s, -2.5),
      [("flexion", "x", -s, -20, 50), ("abduction", "y", -1, -30, 40)])
    J(f"thumb_02_{side}", f"thumb_01_{side}", (1.6, -0.5 * s, -3.4), [("flexion", "x", -s, 0, 60)])
    J(f"thumb_03_{side}", f"thumb_02_{side}", (0.9, -0.3 * s, -3.0), [("flexion", "x", -s, -10, 80)])
    J(f"thumb_tip_{side}", f"thumb_03_{side}", (0.6, -0.2 * s, -2.4), virtual=True)

    J(f"thigh_{side}", "pelvis", (0, 9 * s, -5),
      [("flexion", "y", -1, -30, 120), ("abduction", "x", s, -25, 45), ("twist", "z", -s, -40, 40)])
    J(f"calf_{side}", f"thigh_{side}", (0, 0, -43), [("flexion", "y", 1, 0, 150)])
    J(f"foot_{side}", f"calf_{side}", (0, 0, -40), [("flexion", "y", 1, -30, 45)])
    J(f"ball_{side}", f"foot_{side}", (14, 0, -6))
    J(f"toe_{side}", f"ball_{side}", (6, 0, 0), virtual=True)

    S(f"shoulder_{side}", "spine_03", f"upperarm_{side}", 7, "torso", side)
    S(f"upperarm_{side}", f"upperarm_{side}", f"lowerarm_{side}", 5.5, "upper_arm", side)
    S(f"forearm_{side}", f"lowerarm_{side}", f"hand_{side}", 4.5, "forearm", side)
    S(f"palm_{side}", f"hand_{side}", f"middle_01_{side}", 3.8, "hand", side)
    for f in list(FINGERS) + ["thumb"]:
        chain = [f"{f}_01_{side}", f"{f}_02_{side}", f"{f}_03_{side}", f"{f}_tip_{side}"]
        for i in range(3):
            S(f"{f}_{i + 1}_{side}", chain[i], chain[i + 1], 0.95 if f != "thumb" else 1.1, "finger", side)
    S(f"thigh_{side}", f"thigh_{side}", f"calf_{side}", 7.5, "thigh", side)
    S(f"shin_{side}", f"calf_{side}", f"foot_{side}", 6, "shin", side)
    S(f"foot_{side}", f"foot_{side}", f"ball_{side}", 4, "foot", side)
    S(f"toes_{side}", f"ball_{side}", f"toe_{side}", 3.5, "foot", side)

# ---- self-collision pairs (whitelist; adjacent/attached segments excluded) --
pairs = []
for side in ("l", "r"):
    o = "r" if side == "l" else "l"
    for limb in (f"forearm_{side}", f"palm_{side}"):
        for body in ("torso_lower", "torso_upper", "head", "neck", "cap", f"thigh_{side}", f"thigh_{o}",
                     f"upperarm_{o}", f"forearm_{o}", f"palm_{o}"):
            pairs.append([limb, body])
    pairs.append([f"upperarm_{side}", "head"])
    pairs.append([f"upperarm_{side}", "torso_lower"])
pairs += [["thigh_l", "thigh_r"], ["shin_l", "shin_r"], ["foot_l", "foot_r"], ["shin_l", "thigh_r"], ["shin_r", "thigh_l"]]
# the legs against the upper body and the arms: a raised leg (thigh flexion up to 120) must not pass through the chest, the head or an arm.
# Hips/torso_lower are attached to the thighs (left out); the forearms and palms against the thighs are already in above.
for side in ("l", "r"):
    for leg in (f"thigh_{side}", f"shin_{side}", f"foot_{side}"):
        for other in ("torso_upper", "head", "neck", "cap", "shoulder_l", "shoulder_r", "upperarm_l", "upperarm_r"):
            pairs.append([leg, other])
        if leg != f"thigh_{side}":
            pairs += [[leg, "torso_lower"], [leg, "forearm_l"], [leg, "forearm_r"], [leg, "palm_l"], [leg, "palm_r"]]
# dedupe symmetric duplicates
seen, collision_pairs = set(), []
for a, b in pairs:
    k = tuple(sorted((a, b)))
    if k not in seen:
        seen.add(k)
        collision_pairs.append(list(k))

# ---- keypoints --------------------------------------------------------------
COCO17 = [
    ("nose", "head", (11, 0, 6)), ("left_eye", "head", (9, -3.2, 9)), ("right_eye", "head", (9, 3.2, 9)),
    ("left_ear", "head", (0, -8, 7)), ("right_ear", "head", (0, 8, 7)),
    ("left_shoulder", "upperarm_l", (0, 0, 0)), ("right_shoulder", "upperarm_r", (0, 0, 0)),
    ("left_elbow", "lowerarm_l", (0, 0, 0)), ("right_elbow", "lowerarm_r", (0, 0, 0)),
    ("left_wrist", "hand_l", (0, 0, 0)), ("right_wrist", "hand_r", (0, 0, 0)),
    ("left_hip", "thigh_l", (0, 0, 0)), ("right_hip", "thigh_r", (0, 0, 0)),
    ("left_knee", "calf_l", (0, 0, 0)), ("right_knee", "calf_r", (0, 0, 0)),
    ("left_ankle", "foot_l", (0, 0, 0)), ("right_ankle", "foot_r", (0, 0, 0)),
]
keypoint_sets = {"coco17": [{"name": n, "joint": j, "offset": list(o)} for n, j, o in COCO17]}
for side in ("l", "r"):
    hk = [{"name": "wrist", "joint": f"hand_{side}", "offset": [0, 0, 0]}]
    for f in ("thumb", "index", "middle", "ring", "pinky"):
        for k, suffix in enumerate(("01", "02", "03", "tip")):
            hk.append({"name": f"{f}_{k + 1}", "joint": f"{f}_{suffix}_{side}", "offset": [0, 0, 0]})
    keypoint_sets[f"hand21_{side}"] = hk

skeleton = {
    "schema_version": "1.0",
    "id": "proxy_v1",
    "description": "Proxy humanoid used by the proxy test backend; UE5 Mannequin bone names.",
    "units": "cm",
    "frame": "X forward, Y right, Z up (Unreal world convention); character faces +X",
    "rotation_convention": "local R = product of per-DOF axis rotations in listed order; angle in degrees times sign",
    "joints": joints,
    "palms": {
        "l": {"joint": "hand_l", "normal": [0, 1, 0]},
        "r": {"joint": "hand_r", "normal": [0, -1, 0]},
    },
    "fingers": {side: {f: [f"{f}_01_{side}", f"{f}_02_{side}", f"{f}_03_{side}"]
                       for f in ("thumb", "index", "middle", "ring", "pinky")} for side in ("l", "r")},
    "segments": segments,
    "collision_pairs": collision_pairs,
    "collision_tolerance": 0.85,
    "keypoint_sets": keypoint_sets,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(skeleton, indent=1) + "\n", encoding="utf-8")
print(f"wrote {OUT} ({len(joints)} joints, {len(segments)} segments, {len(collision_pairs)} collision pairs)")
