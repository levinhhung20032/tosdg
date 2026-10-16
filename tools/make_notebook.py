"""Builds notebooks/yolo_finetune_compare.ipynb (run from the tosdg root: python tools/make_notebook.py). Edit the notebook's code here,
so that the committed notebook stays free of outputs."""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# YOLO trước và sau khi học dữ liệu tosdg

Notebook này:

1. xuất dataset tosdg (mặc định `out/vn300`) sang định dạng **YOLO-pose** (1 lớp `traffic_officer`, 59 keypoint: COCO 17 + 2 × 21 bàn tay), chia train / val / test;
2. **tinh chỉnh** YOLO11-pose (đã huấn luyện sẵn trên COCO) trên dữ liệu đó;
3. cho **cả hai mô hình** — *trước* (YOLO gốc, chỉ biết lớp `person`) và *sau* (đã học dữ liệu tosdg) — phát hiện trên **tập dữ liệu của bạn**, và so sánh.

**Tập dữ liệu của bạn** (`USER_DATA` ở ô cấu hình) có thể là:

| Định dạng | Cấu trúc | So sánh |
|---|---|---|
| YOLO | thư mục có `images/…` và `labels/…` (hoặc file `.txt` cạnh ảnh), dòng nhãn `cls cx cy w h [kx ky v …]` (toạ độ chuẩn hoá) | AP50, AP50–95, precision / recall / F1, phát hiện sai, OKS 17 điểm thân (nếu có keypoint) |
| COCO | một file `.json` (`images`, `annotations`, `categories`), ảnh trong cùng thư mục hoặc thư mục con | như trên |
| Chỉ có ảnh | một thư mục ảnh | so sánh bằng mắt + số lượng phát hiện |

Cách so sánh: YOLO gốc không phân biệt cảnh sát với người khác, nên mọi phát hiện `person` của nó được tính là "cảnh sát" — đó chính là điều cần đo: sau khi học, mô hình có tìm đúng **viên cảnh sát** (và ít báo nhầm người đi đường) hơn không. Hai mô hình được đánh giá bằng **cùng một đoạn code**, cùng ngưỡng.

> Dữ liệu tosdg là ảnh tổng hợp; kết quả trên tập ảnh thật của bạn là thước đo đúng nhất cho giá trị của nó.
""")

code(r"""
# ---------------------------------------------------------------- cấu hình (sửa ở đây)
import os
from pathlib import Path

TOSDG_ROOT = Path(os.environ.get("TOSDG_ROOT", Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd())).resolve()
DATASET    = TOSDG_ROOT / "out" / "vn300"          # dataset do `tosdg generate` sinh ra
YOLO_DIR   = TOSDG_ROOT / "out" / "vn300_yolo"     # bản xuất YOLO-pose (tạo ở bước 2 nếu chưa có)
RUNS_DIR   = TOSDG_ROOT / "out" / "yolo_runs"
SPLIT      = "0.8,0.1,0.1"                          # train, val, test
SPLIT_SEED = 20261016

BASE_MODEL = "yolo11s-pose.pt"                      # YOLO gốc (COCO, 17 keypoint): "trước khi học"
EPOCHS, IMGSZ, BATCH, PATIENCE = 60, 960, 8, 20

# Tập dữ liệu của bạn: thư mục YOLO, file / thư mục COCO .json, hoặc thư mục ảnh. None = chỉ chạy trên tập test tosdg.
USER_DATA = os.environ.get("USER_DATA") or None
OFFICER_CLASSES = None        # YOLO: id lớp là "cảnh sát" trong nhãn của bạn (None = mọi lớp); COCO: lọc theo tên, xem bên dưới
OFFICER_NAMES = ("officer", "police", "csgt", "traffic")   # COCO: category có tên chứa một trong các từ này (không có thì lấy mọi category)

CONF = 0.25                   # ngưỡng tin cậy cho precision / recall / F1 và hình minh hoạ (AP dùng mọi ngưỡng)
N_SHOW = 8                    # số ảnh so sánh bằng mắt

QUICK_TEST = os.environ.get("QUICK_TEST") == "1"   # chạy thử nhanh: 1 epoch, ảnh nhỏ, model nano
if QUICK_TEST:
    BASE_MODEL, EPOCHS, IMGSZ, BATCH, PATIENCE = "yolo11n-pose.pt", 1, 320, 4, 1
print("tosdg:", TOSDG_ROOT, "| dataset:", DATASET, "| your data:", USER_DATA, "| quick test:", QUICK_TEST)
""")

md("## 1. Môi trường")
code(r"""
import sys, subprocess, json, shutil
try:
    import ultralytics, torch
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
    import ultralytics, torch
DEVICE = 0 if torch.cuda.is_available() else "cpu"
print("ultralytics", ultralytics.__version__, "| torch", torch.__version__, "| device:", torch.cuda.get_device_name(0) if DEVICE == 0 else "CPU")
if DEVICE == "cpu":
    print("Lưu ý: PyTorch đang chạy CPU - huấn luyện sẽ chậm. Máy có GPU NVIDIA thì cài bản CUDA, ví dụ:\n"
          "  pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126")
WEIGHTS_DIR = TOSDG_ROOT / "data" / "models"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
""")

md("## 2. Xuất dataset tosdg sang YOLO-pose (train / val / test)")
code(r"""
if not (YOLO_DIR / "data.yaml").exists():
    if not (DATASET / "manifest.json").exists():
        raise SystemExit(f"Chưa có dataset {DATASET}: chạy `python -m tosdg generate configs/dataset_300.json --out out/vn300` trước.")
    r = subprocess.run([sys.executable, "-m", "tosdg", "export", str(DATASET), "--format", "yolo", "--out", str(YOLO_DIR),
                        "--split", SPLIT, "--seed", str(SPLIT_SEED), "--yolo-classes", "officer"],
                       cwd=TOSDG_ROOT, env={**os.environ, "PYTHONPATH": str(TOSDG_ROOT / "src")}, capture_output=True, text=True)
    print(r.stdout[-2000:], r.stderr[-2000:])
    r.check_returncode()
DATA_YAML = YOLO_DIR / "data.yaml"
print(DATA_YAML.read_text(encoding="utf-8")[:600])
for s in ("train", "val", "test"):
    print(s, len(list((YOLO_DIR / "images" / s).glob("*"))), "ảnh")
""")

md("## 3. Tinh chỉnh YOLO11-pose trên dữ liệu tosdg\n\nĐầu dò keypoint được dựng lại cho 59 điểm; phần còn lại của mạng lấy trọng số COCO.")
code(r"""
from ultralytics import YOLO
base_path = WEIGHTS_DIR / BASE_MODEL
RUN_NAME = "tosdg_officer" + ("_quick" if QUICK_TEST else "")
model = YOLO(str(base_path))                 # tải về data/models lần đầu
results = model.train(data=str(DATA_YAML), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, patience=PATIENCE, device=DEVICE,
                      project=str(RUNS_DIR), name=RUN_NAME, exist_ok=True, plots=True, verbose=False, workers=0 if os.name == "nt" else 4)
BEST = Path(model.trainer.best)
print("mô hình đã học:", BEST)
""")

md("## 4. Nạp hai mô hình và tập dữ liệu cần so sánh")
code(r"""
import numpy as np
from PIL import Image

before = YOLO(str(base_path))   # "trước": YOLO gốc - lớp 0 = person
after  = YOLO(str(BEST))        # "sau": đã học tosdg - lớp 0 = traffic_officer
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def _yolo_labels(img: Path, root: Path):
    # labels/<...>/x.txt next to images/<...>/x.jpg, or x.txt beside the image
    parts = list(img.relative_to(root).parts)
    cands = [img.with_suffix(".txt")]
    if "images" in parts:
        i = len(parts) - 1 - parts[::-1].index("images")
        cands.insert(0, root.joinpath(*parts[:i], "labels", *parts[i + 1:]).with_suffix(".txt"))
    return next((c for c in cands if c.exists()), None)

def load_yolo(root: Path, classes=None):
    items = []
    for img in sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT):
        lab = _yolo_labels(img, root)
        if lab is None:
            continue
        w, h = Image.open(img).size
        boxes, kpts = [], []
        for line in lab.read_text().split("\n"):
            v = line.split()
            if len(v) < 5 or (classes is not None and int(float(v[0])) not in classes):
                continue
            cx, cy, bw, bh = (float(x) for x in v[1:5])
            boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
            k = np.array([float(x) for x in v[5:]])
            k = k.reshape(-1, 3) if len(k) % 3 == 0 and len(k) else (np.c_[k.reshape(-1, 2), np.full(len(k) // 2, 2.0)] if len(k) else None)
            kpts.append(None if k is None else k[:17] * [w, h, 1])
        items.append({"image": img, "boxes": np.array(boxes).reshape(-1, 4), "kpts": kpts})
    return items

def load_coco(path: Path):
    # one .json, or every COCO annotation file under a folder (e.g. train / val / test), merged
    files_json = [path] if path.is_file() else [p for p in sorted(path.rglob("*.json"))
                                                 if '"annotations"' in p.read_text(encoding="utf-8", errors="ignore")]
    items = []
    for js in files_json:
        d = json.loads(js.read_text(encoding="utf-8"))
        cats = [c["id"] for c in d.get("categories", []) if any(n in c.get("name", "").lower() for n in OFFICER_NAMES)] or [c["id"] for c in d.get("categories", [])]
        files = {p.name: p for base in (js.parent, js.parent.parent) for p in base.rglob("*") if p.suffix.lower() in IMG_EXT}
        by_img = {im["id"]: {"image": files.get(Path(im["file_name"]).name), "boxes": [], "kpts": []} for im in d["images"]}
        for a in d["annotations"]:
            if a["category_id"] not in cats or a["image_id"] not in by_img:
                continue
            x, y, w, h = a["bbox"]
            by_img[a["image_id"]]["boxes"].append([x, y, x + w, y + h])
            k = np.array(a.get("keypoints") or [], float)
            by_img[a["image_id"]]["kpts"].append(k.reshape(-1, 3)[:17] if len(k) >= 51 else None)
        got = [dict(v, boxes=np.array(v["boxes"]).reshape(-1, 4)) for v in by_img.values() if v["image"] is not None]
        print(f"COCO {js.name}: {len(got)} ảnh, categories dùng: {cats}")
        items += got
    return items

def load_any(path):
    path = Path(path)
    if path.suffix.lower() == ".json" or (path.is_dir() and any(path.rglob("*.json")) and not any(path.rglob("*.txt"))):
        return load_coco(path), True
    items = load_yolo(path, OFFICER_CLASSES)
    if items:
        return items, True
    imgs = sorted(p for p in path.rglob("*") if p.suffix.lower() in IMG_EXT)
    return [{"image": p, "boxes": None, "kpts": []} for p in imgs], False

EVAL_SETS = {"tosdg test (tổng hợp)": [it for it in load_yolo(YOLO_DIR, None) if "test" in it["image"].relative_to(YOLO_DIR).parts]}
HAS_GT = {"tosdg test (tổng hợp)": True}
if USER_DATA:
    EVAL_SETS["dữ liệu của bạn"], HAS_GT["dữ liệu của bạn"] = load_any(USER_DATA)
for k, v in EVAL_SETS.items():
    print(f"{k}: {len(v)} ảnh, nhãn: {'có' if HAS_GT[k] else 'không'}" + (f", {sum(len(i['boxes']) for i in v)} viên cảnh sát" if HAS_GT[k] else ""))
""")

md("## 5. Phát hiện và đánh giá\n\nAP theo kiểu COCO (nội suy 101 điểm); một phát hiện đúng khi IoU ≥ ngưỡng với một viên cảnh sát chưa được ghép. OKS dùng 17 điểm thân (hệ số sigma của COCO) trên các cặp ghép ở IoU 0,5.")
code(r"""
SIGMAS = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0

def predict(model, items, officer_cls=0):
    out = []
    for it in items:
        r = model.predict(str(it["image"]), imgsz=IMGSZ, conf=0.001, device=DEVICE, verbose=False)[0]
        keep = (r.boxes.cls.cpu().numpy().astype(int) == officer_cls) if len(r.boxes) else np.zeros(0, bool)
        k = r.keypoints.data.cpu().numpy()[keep][:, :17] if r.keypoints is not None and len(r.boxes) else np.zeros((0, 17, 3))
        out.append({"boxes": r.boxes.xyxy.cpu().numpy()[keep], "scores": r.boxes.conf.cpu().numpy()[keep], "kpts": k})
    return out

def iou(a, b):
    x0, y0 = np.maximum(a[:, None, 0], b[None, :, 0]), np.maximum(a[:, None, 1], b[None, :, 1])
    x1, y1 = np.minimum(a[:, None, 2], b[None, :, 2]), np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)

def match(preds, items, thr, conf_min=0.0):
    # greedy by score over the whole set: (scores, is_tp, n_gt, matched pairs (image, pred, gt))
    rows = [(p["scores"][j], i, j) for i, p in enumerate(preds) for j in range(len(p["scores"])) if p["scores"][j] >= conf_min]
    rows.sort(key=lambda r: -r[0])
    used = [np.zeros(len(it["boxes"]), bool) for it in items]
    tp, pairs = [], []
    for s, i, j in rows:
        g = items[i]["boxes"]
        if len(g):
            o = iou(preds[i]["boxes"][j:j + 1], g)[0]
            o[used[i]] = -1
            k = int(np.argmax(o))
            if o[k] >= thr:
                used[i][k] = True; tp.append(True); pairs.append((i, j, k)); continue
        tp.append(False)
    return np.array([r[0] for r in rows]), np.array(tp, bool), sum(len(it["boxes"]) for it in items), pairs

def ap(tp, n_gt):
    if n_gt == 0 or not len(tp):
        return 0.0
    c = np.cumsum(tp); rec = c / n_gt; prec = c / np.arange(1, len(tp) + 1)
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    return float(np.mean([prec[rec >= t].max() if (rec >= t).any() else 0.0 for t in np.linspace(0, 1, 101)]))

def oks(preds, items, pairs):
    vals = []
    for i, j, k in pairs:
        g = items[i]["kpts"][k] if k < len(items[i]["kpts"]) else None
        if g is None or preds[i]["kpts"].shape[0] <= j:
            continue
        v = g[:, 2] > 0
        if not v.any():
            continue
        b = items[i]["boxes"][k]; area = max((b[2] - b[0]) * (b[3] - b[1]), 1.0)
        d2 = ((preds[i]["kpts"][j][:, :2] - g[:, :2]) ** 2).sum(1)
        vals.append(np.mean(np.exp(-d2[v] / (2 * area * (2 * SIGMAS[v]) ** 2))))
    return float(np.mean(vals)) if vals else float("nan")

def evaluate(preds, items):
    _, tp50, n_gt, pairs = match(preds, items, 0.5)
    s, tpc, _, _ = match(preds, items, 0.5, CONF)
    n_tp, n_det = int(tpc.sum()), len(tpc)
    p, r = n_tp / max(n_det, 1), n_tp / max(n_gt, 1)
    return {"AP50": ap(tp50, n_gt), "AP50-95": float(np.mean([ap(match(preds, items, t)[1], n_gt) for t in np.arange(0.5, 0.96, 0.05)])),
            f"P@{CONF}": p, f"R@{CONF}": r, f"F1@{CONF}": 2 * p * r / max(p + r, 1e-9), f"FP@{CONF}": n_det - n_tp,
            "OKS17": oks(preds, items, pairs), "officers": n_gt}

PRED = {}
for set_name, items in EVAL_SETS.items():
    PRED[set_name] = {"trước (YOLO gốc)": predict(before, items, 0), "sau (học tosdg)": predict(after, items, 0)}
    print(set_name, "- xong")
""")

code(r"""
import pandas as pd
rows = []
for set_name, items in EVAL_SETS.items():
    for model_name, preds in PRED[set_name].items():
        row = {"tập": set_name, "mô hình": model_name, f"số phát hiện @{CONF}": int(sum((p["scores"] >= CONF).sum() for p in preds))}
        if HAS_GT[set_name]:
            row.update(evaluate(preds, items))
        rows.append(row)
summary = pd.DataFrame(rows)
RUNS_DIR.mkdir(parents=True, exist_ok=True)
summary.to_csv(RUNS_DIR / "comparison.csv", index=False, encoding="utf-8-sig")
summary.round(3)
""")

md("## 6. So sánh bằng mắt\n\nTrái: trước (YOLO gốc, lớp `person`). Phải: sau khi học dữ liệu tosdg. Khung xanh lá: nhãn (nếu có); đỏ: phát hiện với độ tin cậy ≥ `CONF`.")
code(r"""
import matplotlib.pyplot as plt
from PIL import ImageDraw

def draw(item, pred, gt=True):
    im = Image.open(item["image"]).convert("RGB"); d = ImageDraw.Draw(im); lw = max(2, im.width // 400)
    if gt and item["boxes"] is not None:
        for b in item["boxes"]:
            d.rectangle(list(map(float, b)), outline=(40, 220, 40), width=lw)
    for b, s in zip(pred["boxes"], pred["scores"]):
        if s >= CONF:
            d.rectangle(list(map(float, b)), outline=(230, 30, 30), width=lw); d.text((float(b[0]) + 3, float(b[1]) + 3), f"{s:.2f}", fill=(255, 255, 0))
    return im

for set_name, items in EVAL_SETS.items():
    rng = np.random.default_rng(0)
    idx = rng.choice(len(items), size=min(N_SHOW, len(items)), replace=False)
    fig, axes = plt.subplots(len(idx), 2, figsize=(14, 4 * len(idx)), squeeze=False)
    for row, i in enumerate(idx):
        for col, model_name in enumerate(PRED[set_name]):
            axes[row, col].imshow(draw(items[i], PRED[set_name][model_name][i], HAS_GT[set_name])); axes[row, col].axis("off")
            n = int((PRED[set_name][model_name][i]["scores"] >= CONF).sum())
            axes[row, col].set_title(f"{model_name} - {items[i]['image'].name} - {n} phát hiện", fontsize=9)
    fig.suptitle(set_name); fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(RUNS_DIR / f"compare_{'user' if 'bạn' in set_name else 'tosdg'}.jpg", dpi=80)
    plt.show()
""")

md(r"""
## 7. Đọc kết quả

- **AP50 / AP50–95**: chất lượng khung của *viên cảnh sát* trên mọi ngưỡng tin cậy (cao hơn = tốt hơn).
- **FP@CONF**: số phát hiện sai ở ngưỡng `CONF` — YOLO gốc thường cao vì nó báo *mọi người* (người đi bộ, người đi xe máy) là người.
- **OKS17**: độ chính xác 17 keypoint thân trên các viên cảnh sát được tìm đúng (cần nhãn keypoint).
- Bảng được lưu ở `out/yolo_runs/comparison.csv`, hình ở `out/yolo_runs/compare_*.jpg`, mô hình đã học ở `out/yolo_runs/tosdg_officer/weights/best.pt`.

Mô hình đã học còn dự đoán 2 × 21 keypoint bàn tay và có thể huấn luyện để phân loại **cử chỉ** (xuất lại với `--yolo-classes gesture`, khi đó lớp = cử chỉ).
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                                "language_info": {"name": "python"}})
nbf.write(nb, "notebooks/yolo_finetune_compare.ipynb")
print("written")
