"""Generate characters/tripo_v1/skeleton.json: proxy_v1 with the joints placed where a Tripo auto-rig puts them.

DOFs, limits, capsules, keypoint sets and the rest-pose convention (arms hanging) are copied from proxy_v1, so poses and gestures keep
their meaning; the nose/eye/ear keypoints and the ends of the head and cap capsules are measured on the model's mesh
(import_mesh.head_landmarks). Otherwise only the joint offsets change: each is the model's own joint offset, expressed in its parent's frame in the skeleton's
bind pose (the A-pose the importer fits), averaged with the mirrored other side so the skeleton stays left/right symmetric. The importer
then has (almost) nothing to shift, so arms, hands and fingers are no longer stretched to a skeleton of other proportions.

The fit file of the model (profiles/VN/assets/<model>.fit.json) may carry the joint corrections of import_mesh.adjust_model_joints:
`shoulder_out_cm` (the auto-rig puts the shoulders inside the chest), `wrist_drop_cm` (and the wrists inside the sleeves), `forearm_scale`.

    python tools/build_tripo_skeleton.py [model.glb]     # default: the first Tripo model under profiles/VN/models

Re-run after changing the model's rig or its fit, then re-import the model (`tosdg import-model ... --character tripo_v1`).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tosdg import import_mesh as im  # noqa: E402
from tosdg.skeleton import Skeleton  # noqa: E402

DEFAULT_MODEL = ROOT / "profiles" / "VN" / "models" / "beta.glb"
ITERATIONS = 5


def build(model: Path, out: Path, fit_override: dict | None = None) -> dict:
    """Write the skeleton to `out`. `fit_override` replaces keys of the model's fit file (to try values before saving them)."""
    proxy_path = ROOT / "characters" / "proxy_v1" / "skeleton.json"
    data = json.loads(proxy_path.read_text(encoding="utf-8"))
    fit = dict(im.DEFAULT_FIT)
    fit_file = ROOT / "profiles" / "VN" / "assets" / (model.stem + ".fit.json")
    if fit_file.exists():
        fit.update({k: v for k, v in json.loads(fit_file.read_text(encoding="utf-8")).items() if k != "bind_pose"})
    fit.update(fit_override or {})
    g = im.read_static_glb(model)
    if g["joint_names"] is None:
        raise SystemExit(f"{model}: not a skinned GLB")
    sk0 = Skeleton(data, proxy_path)
    _, jp = im._to_character(g["pos"], g["joint_pos"], fit)
    mapped = im.map_model_joints(g["joint_names"], sk0, fit)
    model_pos = im.adjust_model_joints({m: jp[k] for k, m in enumerate(mapped) if m and m == g["joint_names"][k]}, fit)

    joints = {j["name"]: j for j in data["joints"]}
    err = 0.0
    for it in range(ITERATIONS):
        sk = Skeleton(data, proxy_path)
        bind_pose, err = im.fit_bind_pose(sk, model_pos)
        bind = sk.fk(bind_pose)
        canon: dict[str, np.ndarray] = {}                      # joint stem (name without _l/_r) -> offsets in the right-hand convention
        for name, j in joints.items():
            if j.get("virtual") or name not in model_pos:
                continue
            parent = j["parent"]
            if parent is None:
                loc = model_pos[name].copy()
            elif parent in model_pos:
                loc = bind.rotations[parent].T @ (model_pos[name] - model_pos[parent])
            else:
                continue
            stem, side = (name[:-2], name[-1]) if name[-2:] in ("_l", "_r") else (name, None)
            if side == "l":
                loc[1] = -loc[1]
            canon.setdefault(stem, []).append(loc)
        moved = 0.0
        for name, j in joints.items():
            stem, side = (name[:-2], name[-1]) if name[-2:] in ("_l", "_r") else (name, None)
            if stem not in canon:
                continue
            v = np.mean(canon[stem], axis=0)
            if side is None:
                v[1] = 0.0                                      # on the symmetry plane
            elif side == "l":
                v[1] = -v[1]
            new = [round(float(x), 3) for x in v]
            moved = max(moved, float(np.abs(np.array(new) - np.array(j["offset"])).max()))
            j["offset"] = new
        if moved < 0.01:
            break

    data["id"] = "tripo_v1"
    data["description"] = (f"proxy_v1 DOFs and limits with the joints of a Tripo auto-rig ({model.name}); face keypoints and head/cap "
                           f"capsules measured on its mesh; bind-pose direction error {err:.1f} deg.")
    # the face: proxy_v1's nose/eye/ear offsets belong to another head; measure them on the model imported onto this skeleton
    sk = Skeleton(data, proxy_path)
    mesh = im.import_glb(model, sk, fit)
    im.orient_outward(mesh)
    cap_radius = next(seg["radius"] for seg in data["segments"] if seg["name"] == "cap")
    marks = im.head_landmarks(mesh, sk.fk(mesh.info["bind_pose"]), cap_radius)
    for k in data["keypoint_sets"]["coco17"]:
        if k["name"] in marks:
            k["offset"] = [round(float(x), 3) for x in marks[k["name"]]]
    for name in ("head_end", "cap_top"):
        joints[name]["offset"] = [round(float(x), 3) for x in marks[name]]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    sk = Skeleton(data, out)
    bind = sk.fk(bind_pose)
    return {"bind_error_deg": round(err, 2), "bind_pose": bind_pose,
            "face_cm": {k: np.round(v, 1).tolist() for k, v in marks.items()},
            "joint_shift_cm": {n: round(float(np.linalg.norm(bind.positions[n] - model_pos[n])), 2) for n in model_pos if n in bind.positions
                               and float(np.linalg.norm(bind.positions[n] - model_pos[n])) > 0.5}}


if __name__ == "__main__":
    model = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MODEL
    res = build(model, ROOT / "characters" / "tripo_v1" / "skeleton.json")
    print("bind error (deg):", res["bind_error_deg"])
    print("joints the importer still has to shift by more than 0.5 cm:", res["joint_shift_cm"] or "none")
    print("face keypoints and head/cap capsule ends (head frame, cm):", res["face_cm"])
