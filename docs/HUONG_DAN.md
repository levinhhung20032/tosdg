# Hướng dẫn sử dụng tosdg (tham chiếu đầy đủ)

Tài liệu tham chiếu chi tiết: cài đặt, mọi lệnh CLI, Studio, generation config, định dạng mẫu, quy ước.
Tổng quan ngắn ở [`README.md`](../README.md); kiến trúc ở [`ARCHITECTURE.md`](ARCHITECTURE.md); quyết định thiết kế ở [`DECISIONS.md`](DECISIONS.md).
Các đường dẫn tương đối trong tài liệu này (`configs/`, `src/`, …) tính từ thư mục gốc của project.

## Mục lục

1. [Cài đặt](#1-cài-đặt)
2. [Bắt đầu nhanh](#2-bắt-đầu-nhanh)
3. [Tham chiếu CLI](#3-tham-chiếu-cli)
4. [Studio: công cụ dựng cảnh tương tác](#4-studio-công-cụ-dựng-cảnh-tương-tác)
5. [Tham chiếu generation config](#5-tham-chiếu-generation-config)
6. [Một mẫu sinh ra chứa gì](#6-một-mẫu-sinh-ra-chứa-gì)
7. [Khái niệm và quy ước](#7-khái-niệm-và-quy-ước), [danh sách hiệu lệnh](#7b-danh-sách-hiệu-lệnh-vn-theo-qcvn-412019bgtvt), [model viên cảnh sát](#7a-model-viên-cảnh-sát-model-3d-của-bạn), [baton và bục](#7c-baton-và-bục-giao-thông) và [camera, ống kính](#7d-camera-ống-kính-và-nền)
8. [Các việc thường làm](#8-các-việc-thường-làm)
9. [Cấu trúc repo](#9-cấu-trúc-repo)
10. [Kiểm thử](#10-kiểm-thử)
11. [Hạn chế đã biết](#11-hạn-chế-đã-biết)

---

## 1. Cài đặt

```
git clone https://github.com/levinhhung20032/tosdg.git
cd tosdg
pip install numpy pillow jsonschema
```

Yêu cầu: Python ≥ 3.11 và `numpy`, `pillow`, `jsonschema`. Trên Windows không cần cài bằng tay: `start_studio.bat` tự kiểm tra và cài gói thiếu (xem [mục 2](#2-bắt-đầu-nhanh)).

Tuỳ chọn:

| Gói | Dùng cho |
|---|---|
| `moderngl` (`pip install -e .[gpu]`) | renderer GPU (`mesh_renderer: gpu`, Studio "mesh — GPU") |
| `torch`, `transformers`, `ultralytics` (`pip install -e .[detect]`) | phân tích ảnh nền khi nhập: mask mặt đường (SegFormer) và người / xe tham chiếu (YOLO11, trọng số tự tải vào `data/models`) |

**Ảnh nền mẫu (tuỳ chọn).** `data/backgrounds/index.json` liệt kê 50 ảnh đường phố Việt Nam (Wikimedia Commons) đã được dùng trong quá trình xây dựng và kiểm thử tool, kèm phần chú thích tay của chúng (đường chân trời, bố cục giao lộ, tag) và người / xe đã dò; `data/backgrounds/ATTRIBUTION.md` ghi nguồn, tác giả và giấy phép từng ảnh. **Bản thân các ảnh không nằm trong repo** (dung lượng, giấy phép). Tải chúng **không bắt buộc**: bạn có thể tải về để tham khảo hoặc dùng luôn, hoặc **tự upload ảnh nền của mình** (Studio → Background → upload, hoặc `tosdg backgrounds import`). Ảnh có tên trong `index.json` mà máy chưa có được `generate`, `validate-profile` và Studio bỏ qua (kèm một dòng ghi chú); `backgrounds check` chỉ ghi chú, không báo lỗi.

Muốn tải: **`fetch_backgrounds.bat`** (nhấp đúp; hoặc `python tools/fetch_backgrounds.py [--dry-run] [--force] [--no-masks]`): với mỗi ảnh có trong `index.json` mà máy chưa có, lấy trang Wikimedia Commons trong `ATTRIBUTION.md`, tải bản đúng chiều rộng đã chú thích, so sha256 với `index.json`, rồi dò lại mask mặt đường còn thiếu (cần `pip install -e .[detect]`; mask dò lại có thể lệch chút ít so với bản gốc, script báo những ảnh đó). Không bao giờ sửa `index.json`. Ảnh thêm từ nguồn khác (không có dòng trong `ATTRIBUTION.md`) phải chép tay. Chỉ muốn chạy thử: `tosdg fixtures <thư mục>` tạo một thư viện nền tổng hợp nhỏ.

Chạy mọi thứ từ thư mục gốc của project (thư mục chứa `pyproject.toml`) và trỏ
`PYTHONPATH` vào `src`:

```powershell
# PowerShell
$env:PYTHONPATH = "src"
python -m tosdg --help
```
```bash
# bash
PYTHONPATH=src python3 -m tosdg --help
```

Hoặc cài đặt (`pip install -e .`) rồi dùng lệnh `tosdg` thay cho `python -m tosdg`.
Các ví dụ bên dưới dùng `python -m tosdg` và giả định `PYTHONPATH` đã được đặt.

## 2. Bắt đầu nhanh

**Windows: chỉ cần chạy `start_studio.bat`** (nhấp đúp). File này:
1. Kiểm tra Python ≥ 3.11 và các gói numpy, pillow, jsonschema, OpenCV; gói nào thiếu hoặc cũ thì tự cài (qua `install_requirements.bat`, cũng chạy riêng được).
2. Đóng Studio đang chạy (nếu có, kể cả cửa sổ cmd của nó) rồi mở Studio mới trong trình duyệt (http://127.0.0.1:8765).
   Nếu cổng 8765 đang bị chương trình khác giữ, file báo tên chương trình đó và dừng.
3. Khi bạn đóng tab, Studio tự dừng và cửa sổ cmd hiện lại để bạn bấm **S** (tắt) hoặc **R** (chạy lại); không bấm gì thì tự tắt sau 5 giây.

Lần đầu mở có thể mất một lúc (kiểm tra và cài gói); các lần sau Studio lên sau vài giây. Mỗi workspace chỉ chạy được một Studio (khoá `data/studio/.studio.lock`), và mỗi cổng chỉ một server.

Dùng dòng lệnh (Linux/macOS hoặc không dùng Studio):

```
python -m tosdg fixtures data/backgrounds                       # 4 backplate tổng hợp để test
python -m tosdg validate-profile configs/smoke.json             # kiểm tra config + profile
python -m tosdg generate configs/smoke.json --out out/smoke     # sinh 5 mẫu (thư mục phải mới/rỗng; thêm --overwrite để thay)
python -m tosdg validate out/smoke                              # kiểm tra lại dataset
python -m tosdg reproduce out/smoke                             # chứng minh sinh lại giống hệt
python -m tosdg overlay out/smoke --zoom --out overlay.png      # nhìn các nhãn
python -m tosdg pose-preview stop_bent_r --out pose.png         # model mặc định nhìn từ 4 phía (thêm --baton r: cầm baton)
python -m tosdg import-model duong/dan/officer.glb              # dùng model 3D của bạn
python -m tosdg studio                                          # công cụ dựng cảnh tương tác
```

## 3. Tham chiếu CLI

`python -m tosdg <lệnh> [tuỳ chọn]`. Mọi lệnh thoát với mã `0` khi thành công, `1` khi kiểm tra
hợp lệ thất bại, và `2` khi tham số hoặc file đầu vào sai (một dòng `error: ...`, không có traceback; đặt
`TOSDG_DEBUG=1` để xem traceback). Dùng được trong script và CI.

### `generate` — sinh một lô
```
python -m tosdg generate CONFIG --out DIR [--num N] [--seed S] [--background-library DIR] [--image-format {png,jpg}] [--overwrite]
                         [--renderer {cpu,gpu}] [--workers N|auto]
```
| Tuỳ chọn | Ý nghĩa |
|---|---|
| `CONFIG` | JSON generation config (xem [mục 5](#5-tham-chiếu-generation-config)). |
| `--out DIR` | Thư mục xuất (bắt buộc). Phải mới hoặc rỗng: không bao giờ trộn dataset mới vào dataset cũ. |
| `--overwrite` | Thay dataset đã có trong `--out`. Chỉ xoá file của dataset; thư mục có file khác thì bị từ chối. |
| `--num N` | Ghi đè `num_samples` (≥ 1). |
| `--seed S` | Ghi đè `master_seed` (≥ 0). |
| `--background-library DIR` | Ghi đè `background_library` của config. |
| `--renderer cpu\|gpu` | Renderer mesh, ghi đè `mesh_renderer` của config (thuộc config: ảnh hai renderer khác nhau). Chỉ backend `mesh`. |
| `--workers N\|auto` | Số tiến trình sinh song song (mặc định 1). File giống từng byte với mọi N. N được giới hạn theo cấu hình máy (CPU, bộ nhớ, VRAM khi dùng GPU) đo lần đầu và lưu ở `data/machine.json`; `auto` = N tối đa; xem `tosdg machine`. Mỗi tiến trình khởi động ~5–7 s nên lô nhỏ lợi ít (D47). |

Mẫu lỗi sẽ được bốc lại (tối đa `max_attempts_per_sample` lần) hoặc ghi nhận là lỗi trong
manifest; không bao giờ được ghi ra. Cuối cùng toàn bộ dataset được kiểm tra hợp lệ.

### `validate` — kiểm tra dataset đã sinh
```
python -m tosdg validate DATASET_DIR
```
Kiểm tra schema, sự có mặt và tên file, hash (file bị sửa, thiếu hoặc thừa), ảnh render không trống,
tính nhất quán của bbox và keypoint, và tay ra hiệu nằm trong khung hình.

### `reproduce` — sinh lại và so sánh
```
python -m tosdg reproduce DATASET_DIR [--index I [I ...]] [--background-library DIR] [--workers N|auto]
```
Trước hết kiểm tra file trên đĩa so với manifest (như `validate`, in `ON DISK ...` nếu có lỗi), rồi dựng lại
các mẫu từ config và seed trong manifest và so hash. In ra `identical` hoặc `DIFFERENT` cho từng mẫu và một
dòng tổng kết. `--index` giới hạn trong các chỉ số mẫu cụ thể; `--background-library` khi thư viện nền đã
chuyển chỗ (manifest ghi đường dẫn tương đối với repo, hoặc tuyệt đối nếu nằm ngoài repo).

### `machine` — cấu hình máy và số tiến trình
```
python -m tosdg machine [--refresh] [--file PATH]
```
In số luồng CPU, RAM / bộ nhớ commit còn trống, GPU và VRAM trống (`nvidia-smi`), và `--workers auto` sẽ chạy bao nhiêu tiến trình với từng renderer (gpu / cpu / proxy) kèm giới hạn nào quyết định. Mỗi tiến trình được tính ~2,2 GB bộ nhớ (CPU) hoặc ~3,5 GB bộ nhớ + 1,3 GB VRAM (GPU).

**Cấu hình máy được lưu**: lần đầu (lệnh này, hoặc `generate` / `reproduce` với `--workers` > 1) tool đo máy và lưu vào `data/machine.json` (git bỏ qua — mỗi máy một file); các lần sau chỉ đọc file. `--refresh` đo lại (bộ nhớ trống thay đổi theo những chương trình đang mở); tool cũng tự đo lại khi file là của máy khác (tên máy, số CPU), hỏng, hoặc làm với hằng số cũ. `--file` / biến môi trường `TOSDG_MACHINE_FILE`: đổi chỗ lưu. Nếu lúc chạy máy ít bộ nhớ hơn lúc đo, tiến trình hết bộ nhớ không làm hỏng lần chạy (mẫu được làm lại) — chạy `--refresh` để giảm N.

### `validate-profile` — kiểm tra profile quốc gia
```
python -m tosdg validate-profile CONFIG [--profile CC]
```
Kiểm tra mọi schema và tham chiếu, kiểm tra pose cơ sở của từng cử chỉ có thoả ràng buộc của chính nó không,
từ chỉ dẫn có nằm trong bộ từ vựng của profile không, và các file thư viện nền có tồn tại và khớp hash không.

### `fixtures` — backplate tổng hợp
```
python -m tosdg fixtures LIBRARY_DIR
```
Tạo bốn backplate đường phố tô phẳng (ban ngày, nhiều mây, hoàng hôn, nhìn từ trên cao) với đường chân trời đã biết,
để chạy pipeline mà không cần ảnh thật. Các mục đã có được giữ nguyên.

### `backgrounds import` / `ground` / `remove` / `check` — quản lý backplate thật
```
python -m tosdg backgrounds import LIBRARY IMAGE [IMAGE ...]
        [--tag TAG]... [--category NAME] [--id ID] [--horizon FRAC]
        [--ground {detect,none}] [--people {detect,none}]
python -m tosdg backgrounds ground LIBRARY [--id ID]... [--force | --mask MASK.png | --clear]
python -m tosdg backgrounds people LIBRARY [--id ID]... [--force | --clear]
python -m tosdg backgrounds objects LIBRARY ID [--add KIND X0 Y0 X1 Y1 [--height-cm CM]] [--clear]
python -m tosdg backgrounds calibrate LIBRARY [--id ID]... [-v]
python -m tosdg backgrounds junction LIBRARY ID [--type {cross,T,Y,roundabout} --center U V --road DEG [--road DEG]... [--frac] | --clear]
python -m tosdg backgrounds set LIBRARY ID [--horizon FRAC] [--camera-height CM] [--road-yaw DEG] [--tag TAG]...
python -m tosdg backgrounds shadows LIBRARY [--id ID]... [--force | --clear | --clear-marks | --mark FU FV TU TV]
python -m tosdg backgrounds remove LIBRARY ID [ID ...]
python -m tosdg backgrounds check LIBRARY
```
| Tuỳ chọn | Ý nghĩa |
|---|---|
| `--tag` | Lặp lại được. Camera chọn backplate theo tag (`ground_level`, `day`, …). |
| `--category` | Nhóm tự do (mặc định `uncategorized`). |
| `--id` | Id của backplate (mặc định lấy từ tên file); chỉ dùng khi nhập một ảnh. |
| `--horizon` | Hàng chân trời theo tỉ lệ chiều cao ảnh (0–1). Cho phép generator căn đường chân trời của ảnh với camera. |
| `--ground` | `detect` (mặc định): dò **vùng đứng được** (mặt đường + vỉa hè) bằng SegFormer (Cityscapes) để công an không bị đặt lên tường, nhà, xe hay bồn hoa; cần `torch` + `transformers` (không có thì nhập không kèm mask và báo cảnh báo). `none`: không dò. |

`backgrounds ground` dò (hoặc dò lại với `--force`) mask cho các nền đã có trong thư viện; `--mask` dùng một ảnh tự vẽ (trắng = đứng được, cùng tỉ lệ khung với ảnh; cần một `--id`); `--clear` bỏ mask. Mask lưu ở `images/<id>.ground.png`, có sha256 trong `index.json` (thư viện 1.1); `check` kiểm tra cả mask. Khi sinh dữ liệu, các điểm trên mặt đất dưới hai bàn chân (hoặc quanh chân bục) phải rơi vào mask: nếu không, ảnh nền được cắt lệch sang ngang tới chỗ hợp lệ; không có chỗ nào thì mẫu bị loại và bốc lại (`rejection_counts.ground`). Kết quả ghi ở metadata `background.ground_check` (1.7). Nền không có mask thì không kiểm tra, như trước. Xem DECISIONS D37.

`people` dò **người tham chiếu** trong ảnh (`--people detect` khi import, mặc định): người đứng (khớp hông–gối–cổ chân thẳng, thấy cổ chân) và người ngồi trên xe máy / xe đạp, bằng YOLO11 (`ultralytics`; trọng số tải một lần vào `data/models`), cùng lượt đó dò **phương tiện không người lái** (ô tô, xe buýt, xe tải, xe máy, xe đạp; bỏ xe bị mép ảnh cắt hoặc bị vật gần hơn che > 25 %). Mỗi người / vật lưu điểm chân và chiều cao pixel trong `index.json` (thư viện 1.2, khối `people` và `objects`). Người bị mép ảnh cắt hoặc nhỏ hơn 40 px bị bỏ.
`objects` liệt kê, **đánh dấu tay** (`--add KIND X0 Y0 X1 Y1`, khung theo pixel ảnh, đáy khung = chỗ chạm đất; KIND: `car`, `bus`, `truck`, `motorbike`, `bicycle`, `barrier`; `--height-cm` khi biết chiều cao thật) hoặc xoá (`--clear`) vật tham chiếu. Rào chắn không có trong lớp của YOLO (COCO) nên chỉ đánh dấu tay được; vật đánh dấu tay được giữ khi dò lại.
`junction` đặt tay **bố cục giao lộ** của ảnh (dùng cho `officer_yaw_mode: "junction"`): loại (`cross` = ngã tư hoặc nhiều đường hơn, `T`, `Y`, `roundabout` = bùng binh), `--center` = điểm trên mặt đất nơi viên cảnh sát đứng (giữa giao lộ; pixel ảnh, hoặc tỉ lệ với `--frac`), một `--road` cho mỗi nhánh đường rời giao lộ: hướng của nhánh tính bằng độ so với hướng nhìn của ảnh (0 = đi xa camera, 90 = sang phải, 180 = về phía camera, −90 = sang trái; như `road_yaw_deg`). Không có giao lộ = đường thẳng. `--clear` bỏ.
`calibrate` in camera mà `size_reference: "auto"` dùng cho từng ảnh (xem dưới, `-v` cho biết vì sao người / vật chưa đủ).
`set` đặt tay đường chân trời (`horizon_source: manual`), độ cao máy chụp, hướng đường, tag. Đường chân trời tự dò (tương phản trời/nhà) thường là đường mái nhà chứ không phải đường chân trời ngang tầm mắt; `import --horizon` và kéo trong Studio cũng là đặt tay. Với `auto`, đường chân trời và độ cao máy chụp chỉ dùng khi ảnh không có đủ người / vật.

**Hiệu chỉnh camera của ảnh nền** (`size_reference: "auto"`, `src/tosdg/calibrate.py`, D42). Trên mặt đường phẳng, một vật cao thật H đứng ở hàng ảnh v cao `H · (v − v0) / h_c` pixel, với v0 là hàng đường chân trời và h_c là độ cao camera. Mỗi người / vật cho một phương trình tuyến tính; vật ở các khoảng cách (hàng) khác nhau tách được v0 và h_c. Thứ tự:
1. **người**: ít nhất 3 người (người đứng coi cao **165 cm**, người ngồi xe máy tính từ đầu tới đáy bánh xe coi 150 cm; hai người trên một xe tính một);
2. **người + vật**: thêm ô tô 150 cm, xe buýt 310, xe tải 300 (sai số lớn), xe máy 110, xe đạp 100, rào chắn 100 (hoặc `height_cm`), ít nhất 3;
3. **đường chân trời** của ảnh (đặt tay hoặc tự dò), độ cao camera đo từ các vật có trong ảnh ở đường chân trời đó (trung vị), nếu không có thì `camera_height_cm` / mặc định 155 cm (125 cm với tag `dashcam`).
Phép khớp là bình phương tối thiểu có trọng số trên tập đồng thuận của mọi cặp vật (loại trẻ em, người ngồi, khung dính hai xe…), và bị từ chối khi sai số chuẩn của đường chân trời > 4 % chiều cao ảnh hoặc của độ cao camera > 20 % (các vật cách camera gần như nhau), độ cao ngoài 50–1000 cm, hoặc có vật đứng trên đường chân trời. Trên 20 ảnh thư viện: 8 ảnh từ người, 6 từ người + xe, 6 dùng đường chân trời; với các ảnh có đường chân trời đặt tay, đường chân trời suy từ người lệch 0,5–5 % chiều cao ảnh.
`shadows` đọc **thời tiết và bóng** của ảnh nền (D46; dùng cho config `shadow` nguồn `photo`): trời không xanh / tag `night` / `dusk` = không có nắng; ảnh nắng thì tìm hướng nắng từ bóng của người / xe đã dò. `--force` đọc lại (giữ dấu tay), `--mark FEET_U FEET_V TIP_U TIP_V [--height-cm CM] --id ID` đánh dấu tay một bóng (pixel ảnh: chỗ vật đứng, đỉnh bóng của nó; được ưu tiên), `--clear-marks`, `--clear`. `import` tự đọc sau khi dò người / xe.
```
python -m tosdg backgrounds shadows LIBRARY [--id ID]... [--force | --clear | --clear-marks | --mark FU FV TU TV [--height-cm CM]]
```

`remove` bỏ nền khỏi thư viện (mục trong `index.json`, file ảnh, mask): `generate` không dùng nó nữa.
`check` kiểm tra mọi file trong thư viện có tồn tại và khớp SHA-256.
(Studio cũng nhập được từ trình duyệt — xem bên dưới.)

### `overlay` — xem ground truth
```
python -m tosdg overlay DATASET_DIR [--out FILE] [--max N] [--zoom]
```
Bảng ảnh có bbox, skeleton COCO và keypoint bàn tay vẽ chồng lên từng ảnh, kèm chú thích
cử chỉ, camera, hướng tiếp cận và chỉ dẫn kết quả. `--zoom` thêm một ảnh phóng to quanh viên cảnh sát.
Mặc định `--out overlay.png`, `--max 12`.

### `export` — xuất sang COCO / YOLO-pose / CVAT
```
python -m tosdg export DATASET_DIR --format coco|yolo|cvat --out DIR [--split 0.8,0.1,0.1] [--seed N] [--yolo-classes officer|gesture]
```
Ghi dataset (đã qua `validate`) sang định dạng của công cụ huấn luyện / gán nhãn, chia train / val / test bằng
một hoán vị có seed (mặc định seed của dataset), nên cùng dataset và cùng tỉ lệ luôn cho cùng cách chia. Mỗi viên
cảnh sát có 59 keypoint (COCO 17 thân, 21 tay trái, 21 tay phải, như COCO-WholeBody); điểm ngoài ảnh có visibility 0,
điểm trong ảnh 2 (chưa mô hình hoá tự che khuất). `--out` phải mới hoặc rỗng.
Visibility của keypoint: 2 = thấy, **1 = trong ảnh nhưng bị chính cơ thể viên cảnh sát (hoặc baton đang cầm) che** (metadata `occluded`), 0 = ngoài ảnh. Mọi định dạng có thêm **`labels.csv`** (UTF-8, mở được bằng Excel): mỗi ảnh một dòng — ảnh, split, có ra hiệu lệnh không, cử chỉ, `hieu_lenh` (mô tả tiếng Việt, hoặc "không ra hiệu lệnh"), chỉ dẫn cho từng phía, phía của người quan sát và chỉ dẫn cho họ.
- `coco`: `images/<split>/`, `annotations/person_keypoints_<split>.json` (bbox, mask RLE, keypoints, `gesture`).
- `yolo`: `images/<split>/`, `labels/<split>/`, `data.yaml` (`kpt_shape: [59, 3]`, `flip_idx`; `path` tuyệt đối, sửa nếu chuyển thư mục). `--yolo-classes gesture`: mỗi cử chỉ một lớp.
- `cvat`: `images/`, `annotations.xml` (CVAT for images 1.1: box + mask + 3 skeleton, thuộc tính `gesture`, `subset` = split) và `cvat_labels.json`: dán vào ô *Raw* khi tạo task, upload ảnh, rồi *Upload annotations* chọn "CVAT 1.1".

### Huấn luyện YOLO và so sánh trước / sau — `notebooks/yolo_finetune_compare.ipynb`
```
pip install -e .[train]
jupyter notebook notebooks/yolo_finetune_compare.ipynb
```
Xuất dataset (mặc định `out/vn300`) sang YOLO-pose (lớp `traffic_officer`, 59 keypoint, train / val / test), tinh chỉnh YOLO11-pose, rồi cho **YOLO gốc** (lớp `person`) và **YOLO đã học** phát hiện trên **tập dữ liệu của bạn** (`USER_DATA` ở ô cấu hình: thư mục YOLO `images/` + `labels/`, file / thư mục COCO `.json`, hoặc chỉ thư mục ảnh) và so sánh bằng cùng một đoạn code: AP50, AP50–95, precision / recall / F1, phát hiện sai, OKS 17 điểm thân, kèm hình so sánh. Kết quả ở `out/yolo_runs/` (`comparison.csv`, `compare_*.jpg`, `tosdg_officer/weights/best.pt`). Huấn luyện cần GPU để nhanh (PyTorch bản CUDA); biến môi trường `QUICK_TEST=1` chạy thử 1 epoch. Sửa code của notebook trong `tools/make_notebook.py` rồi chạy lại để notebook trong repo không chứa output.

### `pose-preview` — xem một pose
```
python -m tosdg pose-preview POSE [--hand-l NAME] [--hand-r NAME] [--character ID]
                                  [--profile CC] [--out FILE]
                                  [--renderer {mesh,gpu,proxy}] [--uniform ID] [--brightness X]
                                  [--baton {l,r}] [--baton-axis {across,along}] [--podium]
```
Render một pose từ phía trước, phải, sau và trái. In ra lỗi vượt giới hạn khớp hoặc tự va chạm.
`POSE` là tên file (không đuôi) trong `poses/body/` (ví dụ `stop_bent_r`); bàn tay mặc định là `relaxed`.
`POSE` và tên bàn tay sai thì báo lỗi kèm danh sách tên hợp lệ. `--baton l|r` cho tay đó cầm baton (tay đóng quanh baton; `--baton-axis along` giữ baton dọc cẳng tay), `--podium` đặt viên cảnh sát lên bục giao thông; mặc định baton hiện ở thắt lưng.
`--renderer` là `mesh` (model 3D, renderer CPU, mặc định), `gpu` (cùng model, renderer GPU — đẹp nhất) hoặc `proxy` (capsule; chỉ vẽ baton cầm tay). `--uniform` mặc định là uniform đầu tiên của profile
(chính là uniform đang dùng); `--brightness` nhân màu vải (chỉ mesh).

### `import-model` — dùng model 3D của bạn
```
python -m tosdg import-model MODEL.glb [--name ID] [--profile CC] [--character ID] [--no-activate]
```
Đăng ký một file GLB có texture của người mặc đồng phục làm uniform đang dùng của profile (`--no-activate`: chỉ đăng ký, giữ model hiện tại).
`--character` mặc định là `tripo_v1` (skeleton dựng từ rig Tripo). Model **không được** mang baton riêng: baton là prop do tool thêm vào (xem [mục 7c](#7c-baton-và-bục-giao-thông)).
`--name` phải có ít nhất một chữ hoặc số. Chi tiết ở [mục 7a](#7a-model-viên-cảnh-sát-model-3d-của-bạn).

### `studio` — công cụ dựng cảnh tương tác
```
python -m tosdg studio [--port 8765] [--profile VN] [--workspace DIR] [--library DIR] [--no-browser] [--exit-on-close]
```
| Tuỳ chọn | Ý nghĩa |
|---|---|
| `--port` | Cổng trên `127.0.0.1` (mặc định 8765). |
| `--profile` | Profile quốc gia cần nạp (mặc định `VN`). |
| `--workspace` | Nơi lưu upload và trạng thái (mặc định `data/studio`). |
| `--library` | Thư viện nền mà Studio hiển thị, upload vào và xoá khỏi — cũng là thư viện `generate` dùng (mặc định `data/backgrounds`). |
| `--no-browser` | Không tự mở tab trình duyệt. |
| `--exit-on-close` | Dừng server khi tab trình duyệt cuối cùng đóng (`start_studio.bat` dùng cờ này). |

Cổng bị chiếm hoặc workspace đang có Studio khác thì lệnh báo lỗi một dòng và thoát (mã 1), không chạy chồng lên nhau.

---

## 4. Studio: công cụ dựng cảnh tương tác

```
python -m tosdg studio
```
mở **http://127.0.0.1:8765**. Server chỉ lắng nghe trên máy của bạn. Mọi thứ bạn thấy
(FOV, keypoint, bbox, "generator có giữ góc nhìn này không") đều được tính bằng cùng một
động học thuận và phép chiếu camera mà generator dùng cho ground truth, nên Studio và
dataset luôn khớp nhau.

### 4.1 Bố cục

- **Cột trái** — điều khiển: Officer, Camera, Background, Pose & gesture, **Officer model (3D)**.
- **Giữa** — khung nhìn camera (thứ ống kính thấy), rồi **góc nhìn từ trên xuống (plan)** và
  **góc nhìn bên (profile)** dọc theo hướng của camera.
- **Cột phải** — các chỉ số đọc: Field of view & framing, Generator check, Gesture match, Selected
  joint, Pose warnings.
- **Thanh công cụ** phía trên khung camera bật/tắt: nền, **model 3D**, capsule, skeleton, điểm kéo, bbox, đường chia ba,
  thước FOV, đường chân trời.
- **Model 3D trong khung nhìn camera.** Khi bạn kéo, viên cảnh sát được vẽ bằng capsule (tức thì); khoảng 0,2 giây sau khi dừng
  (cộng khoảng 0,3 giây để vẽ) model 3D có skin thay thế chúng, nằm trên ảnh nền và dưới các điểm skeleton, và
  chú thích ghi `3D uniform` (hoặc `updating…`). Nó theo dõi pose, officer, camera và độ sáng. Nó được vẽ với
  một ánh sáng xem trước cố định, không phải ánh sáng ngẫu nhiên của mẫu sinh ra, và một lớp không còn khớp với cảnh thì không bao giờ được hiện. Tắt
  bằng nút **3D uniform**, hoặc chọn renderer capsule trong *Render preview options*.

**Toàn màn hình và cửa sổ riêng.** Mỗi khung trong ba khung có nút **⛶ Full screen** và **⧉ Pop out**
ở tiêu đề.
- *Full screen* cho khung đó phủ kín màn hình (Esc hoặc bấm lại nút để thoát). Khung camera giữ
  tỉ lệ khung hình và hiện một chỉ số nhỏ (FOV, tiêu cự, khoảng cách, azimuth, độ cao, chiều cao của
  viên cảnh sát tính bằng pixel); khung plan và profile dùng toàn màn hình. Việc kéo vẫn như bình thường.
- *Pop out* mở khung trong một cửa sổ trình duyệt riêng (hãy cho phép pop-up nếu trình duyệt chặn).
  Đặt khung camera ở một màn hình và khung plan ở màn hình khác. Mọi cửa sổ chia sẻ cùng một
  cảnh theo thời gian thực: thay đổi hoặc kéo ở bất kỳ cửa sổ nào hiện ngay ở các cửa sổ khác.
- **Lăn chuột trên khung camera đưa camera lại gần / ra xa viên cảnh sát** (nó đổi
  *khoảng cách*; ống kính không đổi), tiện khi ở chế độ toàn màn hình và cửa sổ riêng nơi thanh trượt
  không hiện. **Shift + lăn chuột** thì là zoom ống kính (tiêu cự). Ở khung plan và profile
  lăn chuột phóng to chính hình vẽ, và nhấp đúp để đặt lại zoom.

**Tiện ích hằng ngày** (tiêu đề và phím tắt)
- **↶ Undo / ↷ Redo** (`Ctrl+Z`, `Ctrl+Y` hoặc `Ctrl+Shift+Z`) áp dụng cho pose, officer và camera. Một lần
  kéo thanh trượt hoặc một lần kéo skeleton là một bước; các thanh trượt đi theo. (Nền, model và
  file đã lưu không thuộc undo.)
- **? Help** (hoặc `?`) liệt kê mọi thao tác chuột và phím tắt; một thanh **Quick tips** có thể đóng
  hiện các điều cần thiết lần đầu tiên.
- `F` khung camera toàn màn hình · `R` render preview · `Home` đặt lại camera · `Esc` đóng hộp thoại và menu.
- Nút **↺ reset** trong panel Officer và Camera. Các panel bạn gập lại sẽ vẫn gập ở lần mở Studio sau.
- Thông báo hiện ở góc dưới bên phải, xếp chồng và đóng được: đỏ = có vấn đề (ở lâu hơn),
  xanh lá = xong, xanh dương = thông tin. **⏳ working…** hiện ở tiêu đề khi có việc mất hơn
  nửa giây (upload, nhập model).
- Di chuột lên các nút bật/tắt và nút tiêu đề để xem giải thích một dòng.

### 4.2 Officer
`x`, `y` (cm trên mặt đất) và `yaw` (độ; 0 = hướng +X của thế giới). Di chuyển hoặc xoay viên
cảnh sát giữ camera cố định trong thế giới: Studio tự tính lại các tham số camera so với officer
(generator lưu camera theo hệ của officer).

### 4.3 Camera
| Điều khiển | Ý nghĩa |
|---|---|
| preset profile + **mid** / **random** | Nạp giá trị giữa, hoặc một lần bốc ngẫu nhiên, trong các khoảng của camera profile (dashcam, driver_eye, motorbike_rider, roadside). Chọn loại camera cũng nạp ống kính của nó (méo, viền đen), xem [mục 7d](#7d-camera-ống-kính-và-nền). |
| lens k1 / k2, black border | Méo ống kính xuyên tâm (k1 < 0 là méo thùng) và vòng tròn ảnh; viền đen tự tính theo méo (ô *auto black border*), kéo tay thì tắt tự động. |
| distance, azimuth, height, aim height | Đặt vị trí theo quỹ đạo quanh officer. **Azimuth tính trong hệ của officer: 0 = phía trước officer, +90 = bên phải officer.** |
| yaw / pitch offset, roll | Chỉnh hướng nhìn tay khỏi "nhìn vào điểm ngắm", và góc roll của camera. |
| focal (mm), sensor w (mm) | Ống kính và cảm biến. FOV suy ra từ hai giá trị này. |
| resolution | Kích thước đầu ra, cũng quyết định tỉ lệ khung của khung camera. |

Panel **Field of view & framing** hiện FOV ngang / dọc / chéo, tiêu cự tương đương 35 mm,
vị trí và góc camera, hàng chân trời, chiều cao officer tính bằng pixel, tỉ lệ tại officer (px/cm) và
camera đang ở phía nào của officer (trước / phải / sau / trái).

### 4.4 Background (nền 2D)
1. **Choose Files** — upload một hoặc nhiều ảnh PNG, JPG/JPEG, WebP, BMP, GIF hoặc TIFF (tối đa 40 MB mỗi ảnh). PNG giữ PNG, JPEG giữ JPG;
   WebP/BMP/GIF/TIFF tự chuyển sang **JPG** (nếu ảnh có vùng trong suốt thì lưu PNG để không mất). Chúng được
   kiểm tra, lưu **thẳng vào thư viện nền của generator** (`data/backgrounds`, hoặc `--library`) với tên do server đặt,
   tag `day`, `ground_level`, đường chân trời và mask mặt đường tự dò (cần `torch` + `transformers`), và hiện phía sau khung camera,
   cắt và co giãn đúng như generator sẽ làm. Metadata camera (GPS, thiết bị) bị bỏ và
   ảnh được xoay thẳng theo hướng EXIF. Studio không làm mờ mặt: chỉ dùng ảnh bạn được phép dùng.
2. Bấm vào ảnh nhỏ để chọn. **Nhấp chuột phải vào ảnh nhỏ** (hoặc focus và nhấn `Delete`) để mở menu:
   *Use as background* hoặc *Delete this background…*. Xoá sẽ hỏi xác nhận (focus bàn phím nằm ở
   Cancel nên một cú Enter lỡ tay không làm gì), rồi xoá ảnh, mask và mục của nó khỏi thư viện: `generate` không dùng
   nó nữa. Nếu đó là nền đang chọn thì một nền khác được chọn. Bạn cũng có thể **thả file ảnh vào panel** để upload.
   Studio và `generate` dùng **một thư mục ảnh nền duy nhất**: thấy gì trong Studio thì `generate` dùng đúng như vậy.
3. **Horizon** — kéo đường vàng trong ảnh xem trước nhỏ tới đường chân trời của ảnh. Generator
   dùng nó để căn ảnh với đường chân trời của camera. **clear horizon** xoá nó.
4. **save** — lưu tag, **camera height cm** và **road direction °** (hướng xe chạy so với hướng nhìn của ảnh,
   0 = dọc theo hướng nhìn) của ảnh vào thư viện. Panel cho biết số người tham chiếu đã dò trong ảnh.
   Tag quyết định camera nào được dùng nó: mọi loại camera đều tìm `ground_level`.

### 4.4b Shadow (bóng đổ)

Bóng của viên cảnh sát trên mặt đường, đúng hình model 3D ở mọi tư thế (D46). Cần model 3D.
- **cast a shadow**: bật / tắt. Tắt = không có bóng trong khung camera và Render preview.
- **direction from**: *from the photo* — theo bóng có sẵn trong ảnh (bóng đánh dấu tay trước, bóng tự dò sau); ảnh âm u / chạng vạng /
  đêm chỉ có bóng tiếp xúc mềm; ảnh chưa có hướng thì dùng vector tự chỉnh (dòng thông tin ghi rõ). *custom vector* — **direction (°)**
  so với camera (0 = ra xa, 90 = sang phải, 180 = về phía camera, −90 = sang trái) và **length × height** (dài bóng / chiều cao người;
  mặt trời cao = bóng ngắn).
- **darkness**, **softness** (viền mềm, rộng dần theo độ cao điểm đổ bóng), **contact** (bóng tiếp xúc dưới giày).
- Khung camera: lớp bóng nằm dưới model 3D, mũi tên vàng nét đứt là hướng và độ dài bóng; Render preview vẽ cùng bóng vào ảnh.
- **✎ mark a shadow**: bấm vào chân một người trong ảnh, rồi bấm vào đỉnh bóng của họ (Esc huỷ). Bóng được lưu vào thư viện (pixel ảnh)
  và quyết định hướng cho ảnh này ở Studio lẫn `generate`. **clear marks** xoá dấu tay; **↻ read photo** đọc lại ảnh.
- Random generate: ô **shadow** thêm khối `shadow` vào config:
  - *off*; *from the photo* (ảnh âm u / đêm chỉ có bóng tiếp xúc); *random sun*;
  - *from marked / found shadows*: mỗi ảnh lấy hướng từ bóng đánh dấu tay hoặc tự dò của nó. Khi bấm ▶ start, Studio đọc những ảnh
    chưa đọc rồi kiểm tra ảnh nào **chưa có hướng bóng** (không có dấu tay, không dò được); nếu có, một popup liệt kê chúng:
    **Continue** — tiếp tục, các ảnh đó nhận hướng nắng ngẫu nhiên; **Cancel** — dừng để vẽ tay (bấm một ảnh trong danh sách:
    ảnh được mở, panel Shadow bật sẵn chế độ ✎ mark a shadow). Dấu tay được lưu trong thư viện nên dùng lại cho mọi lần sinh.

### 4.5 Pose & gesture
- **apply preset** — chọn một pose thân và một pose cho mỗi bàn tay, rồi áp dụng.
  Thân: `neutral_stand` (đứng nghỉ, nền cho mọi pose), `palms_forward_both`,
  `speed_up_ear_l/r`, `speed_up_chest_l/r`, `slow_down_l/r`, `stop_bent_l/r`, `stop_handup_l/r`, `stop_forward_r`,
  `arm_forward_r`, `arm_vertical_l/r`, `arm_horizontal_l/r`, `arms_horizontal`, `hello_l/r`, `thumb_up_l/r`.
  Bàn tay: `relaxed`, `open_flat`, `open_spread`, `fist`, `thumb_up`.
- **load a gesture variant** — nạp pose và các pose bàn tay khả dĩ nhất của bất kỳ biến thể nào của bất kỳ cử chỉ nào.
- **sửa gesture và ghi đè gesture gốc** — sau khi nạp một biến thể, panel hiện ô *editing <cử chỉ> / <biến thể>*. Chỉnh tư thế
  (kéo khớp, Alt+click, Ctrl+kéo) rồi bấm **save over the gesture**: pose thân hiện tại ghi đè `poses/body/<pose>.json` của biến thể đó
  (ngón tay không thuộc pose thân). Pose phải trong giới hạn khớp và vẫn đạt các ràng buộc của cử chỉ; nếu không, Studio nêu lý do và
  hỏi có lưu dù vậy không. File cũ được giữ ở `poses/body/.backup/` (không đưa vào git). Tick **mirror to …** (mặc định bật) ghi pose
  đối xứng cho biến thể bên kia (`stop_bent_r` → `stop_bent_l`). Generator dùng pose mới ngay ở lần sinh tiếp theo; dữ liệu đã sinh
  trước đó sẽ không còn `reproduce` giống hệt. **stop editing** thoát chế độ sửa.
- **reset** — về `neutral_stand` với bàn tay thả lỏng.
- **hand keypoints** / **spine / neck handles** — hiện thêm các điểm kéo được. Chấm ngón tay luôn được vẽ nhưng chỉ bắt chuột khi bàn tay rộng ≥ 48 px trên màn hình (bàn tay nhỏ: kéo cổ tay; zoom bằng con lăn để kéo ngón).
- **Alt + nhấp một khớp** (cổ tay, gối, khuỷu ...) bật gizmo xoay: mỗi trục DOF có một vòng màu, kéo để xoay trong giới hạn khớp và không gây va chạm mới (`Esc` để thoát).
- **save to poses/** — ghi pose hiện tại thành file pose (`body`, hoặc `left/right hand`)
  với id bạn gõ (chữ, số, `_`, `-`). Nó từ chối ghi đè file có sẵn
  trừ khi tick **overwrite**, và từ chối pose vượt giới hạn khớp. Pose đã lưu
  chỉ mang tính vật lý; không có ý nghĩa cử chỉ (ý nghĩa nằm trong `gestures.json` của profile).

### 4.6 Kéo skeleton

Trong **khung nhìn camera**:

| Điểm kéo | Kéo để |
|---|---|
| chấm hồng (17 keypoint thân COCO; keypoint bàn tay nếu bật) | Di chuyển điểm đó. Bộ giải động học ngược đổi các khớp phía trên để theo con trỏ, luôn nằm trong giới hạn giải phẫu. Kéo đầu ngón tay làm ngón cong lại thay vì vung cả cánh tay. Chấm vàng là khớp đang chọn. |
| ô vuông xanh ở chân officer | Di chuyển officer trên mặt đất. |

Khung camera kéo **trong mặt phẳng màn hình** (điểm giữ nguyên khoảng cách tới camera).
Để di chuyển khớp **lại gần hoặc ra xa camera**, hãy kéo nó trong khung plan hoặc profile.

**Trục Oxyz** (toggle `axes` trên thanh công cụ): góc dưới trái khung camera vẽ ba trục của viên cảnh sát (X đỏ = phía trước,
Y xanh lá = bên phải, Z xanh dương = lên). Bấm chấm X / Y / Z để camera nhìn dọc trục đó (bấm lần nữa: nhìn từ phía đối diện;
Z = nhìn từ trên xuống, gần thẳng đứng vì camera cách officer tối thiểu 50 cm theo phương ngang). **Giữ Ctrl khi kéo** một khớp:
khớp chỉ di chuyển trong mặt phẳng vuông góc với trục đang căn (toạ độ trên trục đó giữ nguyên), thay vì trong mặt phẳng màn hình
vốn trôi tới / lùi theo góc nhìn khi camera nhìn chéo. Chưa căn trục nào thì Ctrl+kéo dùng trục gần hướng nhìn nhất.

Trong **khung plan** (từ trên xuống, +X lên, +Y phải):

| Điểm kéo | Kéo để |
|---|---|
| chấm xanh (camera) | Di chuyển camera quanh officer (đổi khoảng cách và azimuth). |
| chấm vàng ở cuối đường nét đứt | Chỉnh hướng nhìn của camera (yaw offset). |
| ô vuông xanh (officer) | Di chuyển officer. |
| chấm xanh lá ("facing") | Xoay officer. |
| chấm hồng | Di chuyển khớp theo X–Y. |

Trong **khung profile** (nhìn dọc theo hướng của camera; officer được neo ở bên phải):

| Điểm kéo | Kéo để |
|---|---|
| chấm xanh (camera) | Độ cao camera (dọc) và khoảng cách (ngang). |
| chấm vàng | Nghiêng camera (pitch offset). |
| chấm hồng | Di chuyển khớp theo độ sâu và độ cao. |

Để có giá trị chính xác, chọn một khớp trong **Selected joint** (cột phải, hoặc bấm vào một điểm kéo)
và dùng các thanh trượt của nó; mỗi thanh hiện giới hạn giải phẫu.

### 4.7 Các chỉ số đọc ở bên phải
- **Generator check** — ✓/✗ cho: officer ở phía trước camera, cả 17 keypoint thân nằm trong khung,
  bbox cao ít nhất 24 px, bbox không chạm mép ảnh. Đây là ước lượng gần đúng từ phạm vi
  capsule; **Render preview** cho bbox chính xác theo mask.
- **Gesture match** — cử chỉ nào đã định nghĩa mà pose hiện tại thoả, biến thể nào, và
  chỉ dẫn mà một người đi đường đọc được từ phía camera (ví dụ STOP / PROCEED). Nếu không có gì
  khớp, các cử chỉ gần nhất và ràng buộc không đạt được liệt kê.
- **Pose warnings** — vi phạm giới hạn khớp và tự va chạm.

### 4.8 Officer model (3D)
Panel này là nơi bạn đưa model 3D của viên cảnh sát vào dự án (xem [mục 7a](#7a-model-viên-cảnh-sát-model-3d-của-bạn)).
- **Upload model (.glb)** (hoặc thả file `.glb` vào panel) — upload GLB có texture của người mặc đồng phục. Studio nhập nó vào skeleton của dự án (dùng skeleton/skin
  có sẵn trong file nếu có, nếu không thì tự gán xương), ghi `profiles/<CC>/models/<id>.glb`, `assets/<id>.fit.json`, `assets/<id>.glb` và cập nhật `uniforms.json`.
  Model vừa upload trở thành model đang dùng ngay, và khung camera / Render preview vẽ nó. Với model lớn việc nhập có thể mất khoảng một phút.
- Panel hiện id và mô tả của model đang dùng (cho biết nó dùng skeleton của chính model hay được gán xương) và danh sách model đã lưu (bấm để dùng lại).
- **Holding a baton** — bật thì tay đã chọn (phải/trái) đóng quanh baton 3D và baton rời thắt lưng; tắt thì baton treo ở thắt lưng. Một ảnh luôn chỉ có một baton. Dòng gợi ý cho biết độ khép của nắm tay và cảnh báo nếu baton chạm thân.
- **Standing on a podium** — đặt viên cảnh sát lên bục giao thông (chỉ để xem trước; `generate` dùng bục qua config `podium`).
- Ảnh đồng phục (nhiều ảnh quanh người) rồi "mặc" lên model: **future work** (panel thử nghiệm "Uniform views" đã bị gỡ, D44).

### 4.8b Random generate
Sinh một lô ngay trong Studio, với một số thông số **khoá** theo cảnh đang dựng và phần còn lại **ngẫu nhiên** như `generate`:
cử chỉ (random / một cử chỉ), loại camera, *lock the camera to this view* (khoảng cách, độ cao, góc, tiêu cự hiện tại), ảnh nền
(random / ảnh đang chọn), hướng quay (*0 / 90 / 180 / 270° from the road*, hoặc *as in this view* khi khoá camera), baton,
bục (never / 15 % / always), *size the officer from the people in the photo* và *min height px*, số mẫu, seed, renderer.
**▶ start random generate** chạy nền (một lô mỗi lúc), hiện tiến độ và ảnh nhỏ của kết quả; dataset đầy đủ (ảnh, mask, nhãn,
manifest) ở `out/studio_random/<thời gian>/`, config đi kèm ở `<thời gian>.config.json`. Camera bị khoá thì bỏ qua *min height*.

### 4.9 Các nút ở tiêu đề
- **Render preview** (`R`) — render cảnh hiện tại trên nền đã ghép bằng renderer **mesh**
  (model 3D) hoặc proxy capsule, và hiện ảnh cùng bbox mask chính xác (và cho biết nó có
  bị cắt bởi mép ảnh không). Panel bên phải **Render preview options** chọn renderer và
  **độ sáng vải** (0,7–1,3; **🎲 random ±10 %** bốc một giá trị như generator làm). Khung camera hiện model 3D khi bạn ngừng kéo
  (xem Bố cục); Render preview vẽ ảnh đầy đủ với nền thật và bbox chính xác.
- **Save scene** — ghi một bản chụp (officer, tham số camera và metadata camera đầy đủ, pose,
  nền, các cử chỉ khớp) vào `data/studio/scenes/`. Đây là bản ghi để soạn thảo
  (`format: tosdg.studio_scene/1`); `generate` chưa đọc nó.

### 4.10 Dữ liệu được lưu ở đâu
```
data/studio/                 (git bỏ qua; đổi bằng --workspace)
├── state.json               pose, officer, camera, lựa chọn gần nhất (khôi phục khi tải lại)
└── scenes/                  các bản chụp cảnh đã lưu
```
Ảnh nền bạn upload đi thẳng vào `data/backgrounds/` (hoặc `--library`), thư viện mà `generate` dùng — thư mục ảnh nền duy nhất; nguồn và giấy phép của từng ảnh ghi ở `data/backgrounds/ATTRIBUTION.md`; một workspace cũ
còn thư mục `backgrounds/` riêng được chuyển vào thư viện một lần khi Studio khởi động. Pose bạn lưu đi vào
`poses/body/` hoặc `poses/hands/`; model bạn upload đi vào `profiles/<CC>/models/` và `profiles/<CC>/assets/`.

### 4.11 API cục bộ (để viết script)
Mọi yêu cầu `POST` cần header `X-Studio: 1` và `Host` là loopback; thân JSON bị giới hạn 4 MB,
upload ảnh 40 MB và upload model 200 MB. Độ phân giải 16–4096 px, khoảng cách camera 50–3000 cm; góc khớp ngoài ±360° bị từ chối (vượt giới hạn khớp thì được vẽ và báo ở Pose warnings).

| Endpoint | Mục đích |
|---|---|
| `GET /` , `GET /api/init` | Trang; skeleton, pose, cử chỉ, profile, nội dung workspace. |
| `POST /api/eval` | Chiếu một pose qua camera: khớp, keypoint, capsule, bbox, kiểm tra, các cử chỉ khớp. |
| `POST /api/drag` | Kéo IK (`mode: "joint"` hoặc `"officer"`; đích tính bằng pixel camera hoặc toạ độ thế giới). |
| `POST /api/pose_preset` | Ghép một pose thân + pose bàn tay từ thư viện. |
| `POST /api/backplate` | JPEG nền đã ghép cho một camera. |
| `POST /api/render` | Render preview (renderer mesh hoặc capsule, `brightness`) + bbox chính xác. |
| `POST /api/model_layer` | Riêng model 3D dưới dạng PNG RGBA (rộng tối đa 960 px, cùng trường nhìn) cho khung camera. |
| `POST /api/upload_model?name=...` | Upload GLB (thân nhị phân) làm model đang dùng; trả về tóm tắt nhập (`skin`: `model` = dùng skeleton của file, ngược lại gán xương) và danh sách uniform. |
| `POST /api/models`, `/api/model_activate` | Danh sách model đã lưu; đổi model đang dùng. |
| `POST /api/ping`, `/api/bye` | Trang báo còn mở / sắp đóng (dùng cho `--exit-on-close`). |
| `POST /api/upload_background`, `/api/background_update`, `/api/background_delete` | Thêm ảnh vào thư viện nền; đặt đường chân trời / tag / đèn; xoá ảnh khỏi thư viện. |
| `POST /api/add_to_library` | Sao chép một nền vào thư viện của generator. |
| `POST /api/save_pose`, `/api/save_scene`, `/api/state` | Lưu pose, bản chụp cảnh, trạng thái UI. |
| `POST /api/uniform_*` , `/api/upload_uniform_view` | Còn lại từ bản thử "uniform từ ảnh" trước; **không thuộc v1** (future work), giao diện của chúng đang bị ẩn. |
| `GET /bg/<path>`, `GET /files/<path>` | Phục vụ một ảnh của thư viện nền / của workspace (chỉ bên trong thư mục đó). |

---

## 5. Tham chiếu generation config

Config là một file JSON được kiểm tra theo `schemas/generation_config.schema.json`.
`configs/smoke.json` là ví dụ tối thiểu.

| Khoá | Bắt buộc | Ý nghĩa |
|---|---|---|
| `schema_version` | có | ví dụ `"1.1"` (1.1 thêm `backend: mesh` và `uniform_brightness_range`; config 1.0 vẫn hợp lệ). |
| `name` | có | Chữ, số, `_`, `-`. |
| `profile` | có | Thư mục profile quốc gia trong `profiles/` (ví dụ `VN`). |
| `character` | có | Thư mục skeleton trong `characters/`: `tripo_v1` (model hiện tại, dựng từ rig Tripo) hoặc `proxy_v1` (skeleton chuẩn dùng cho test). Mesh của mọi uniform được chọn phải khớp skeleton này. |
| `backend` | có | `mesh` (model 3D có skin, chạy được ngay) hoặc `proxy` (capsule, để test). `unreal` có trong schema nhưng chưa xây dựng (báo lỗi rõ). Với `mesh`, mọi uniform có thể được chọn (weight > 0) cần một mục `mesh` hợp lệ. |
| `master_seed` | có | Số nguyên ≥ 0. Mọi seed của mẫu đều suy ra từ nó. |
| `num_samples` | có | Số nguyên ≥ 1. |
| `resolution` | có | `[rộng, cao]`, mỗi giá trị ≥ 16. |
| `background_library` | có | Thư mục có `index.json` + `images/`; đường dẫn tương đối được tính từ file config. |
| `gesture_weights` | không | `{gesture_id: weight}`; mặc định = trọng số riêng của từng cử chỉ. |
| `camera_profile_weights` | không | `{camera_profile_id: weight}`. |
| `uniform_weights` | không | `{uniform_id: weight}`. |
| `uniform_brightness_range` | không | `[lo, hi]` (mỗi giá trị 0,3–2,0, `lo <= hi`): hệ số nhân ngẫu nhiên theo mẫu trên **vải** của model (texel không phải da), bốc từ luồng RNG riêng và ghi thành `officer.uniform_brightness`. Mặc định `[1, 1]` = tắt. Cần backend `mesh`. `configs/smoke.json` dùng `[0.9, 1.1]`. |
| `mesh_renderer` | không | `cpu` (mặc định, numpy, chạy ở mọi máy) hoặc `gpu` (OpenGL 4.3 qua `moderngl`, card đồ hoạ): dùng normal map và roughness map của model, bóng mềm của nắng, ánh sáng trời/đất theo màu ảnh nền, ambient occlusion, bóng đổ của viên cảnh sát xuống mặt đường. Cùng hình học, mask và ground truth. Chỉ backend `mesh` (config 1.4). |
| `size_reference` | không | `horizon` (mặc định): căn đường chân trời của ảnh nền với camera. **`auto`** (config 1.6, nên dùng): camera ảo **là camera đã chụp ảnh nền**, suy từ người / xe trong ảnh (xem *Hiệu chỉnh camera của ảnh nền* ở mục `backgrounds`): đặt ở độ cao đó, **pitch** sao cho đường chân trời của camera trùng đường chân trời suy ra trên khung ảnh (ảnh nền giữ nguyên khung, chỉ phóng khi viên cảnh sát không đủ `min_officer_height_px`), khoảng cách chọn trong dải mà viên cảnh sát nằm trọn trong khung và đủ cỡ (giữ vị trí tương đối của khoảng cách bốc theo camera profile), cắt ngang sao cho chân trên mặt đường và không đè lên người / xe **gần camera hơn** (luồng mới `calibration`). Khi đó cỡ viên cảnh sát bằng cỡ một người 165 cm đứng ở chỗ đó trong ảnh, với mọi khoảng cách. Studio khoá camera thì giữ camera đã khoá, chỉ căn đường chân trời suy ra. YOLO chạy một lần trước khi sinh cho ảnh chưa dò. Metadata `background.size_reference`: `mode: calibrated`, `method` (`people` / `objects` / `horizon`), độ cao, đường chân trời, số vật dùng, `pitch_deg`, `zoom`. Mẫu không làm được cỡ nhất quán bị loại (`rejection_counts.size_reference`). (Chế độ `people` — định cỡ theo một người — đã bị gỡ, D44: `auto` thay thế.) |
| `min_officer_height_px` | không | Chiều cao tối thiểu (px) của viên cảnh sát (config 1.5), **cần `size_reference: "auto"`**: viên cảnh sát đủ cỡ cùng với ảnh nền (ảnh được cắt / phóng nhất quán với camera đã hiệu chỉnh), không phải chỉ viên cảnh sát to lên trước một ảnh giữ nguyên. bbox đã render cũng phải cao ít nhất bằng giá trị này (trừ khi `validation.min_bbox_height_px` đặt khác). |
| `lens_distortion` | không | `false`: không méo ống kính, không viền đen ở mọi loại camera (config 1.6). Mặc định `true`. |
| `officer_yaw_mode` (`junction`) | không | **`junction`** (config 1.6, cần `size_reference: "auto"`): ảnh có giao lộ → viên cảnh sát **đứng giữa giao lộ** (lệch tối đa 1 m, chân trên mặt đường, không đè người / xe gần camera hơn), hướng mặt theo loại: **ngã tư / bùng binh** quay mặt về một nhánh đường; **ngã ba chữ T** quay mặt về nhánh dọc (thân chữ T) để cả ba nhánh trong tầm nhìn ±110°; **ngã ba chữ Y** quay về một nhánh hoặc sao cho ít nhất hai nhánh trong tầm nhìn. Trong các hướng đó, chọn hướng mà **các phía cử chỉ ra lệnh** (`instructions` khác `NONE` trong `gestures.json`; cử chỉ cho 3–4 phía thì phía sau được phép là tường) đều là đường, rồi ưu tiên hướng mà camera nằm trong khoảng `azimuth_deg` của camera profile. Ảnh không có giao lộ: dọc / ngang đường theo cùng quy tắc. **Bùng binh**: bục và gậy bốc ngẫu nhiên (`roundabout.podium_probability` / `baton_probability`, mặc định 0,5; bục cần backend `mesh`). Metadata `background.junction`. Xem `src/tosdg/junction.py`, D43. |
| `match_backplate_light` | không | `true` (config 1.6): viên cảnh sát tối đi theo ảnh nền quanh chỗ đứng (đêm, chiều tối, bóng râm): exposure giảm `log2(L / 0,2)` trong [−2; 0] EV, L = trung bình nhân của độ sáng trung bình-logarit (tuyến tính) quanh viên cảnh sát và của cả ảnh; ảnh tag `night` thêm: "mặt trời" thành đèn đường yếu (25 %), ánh sáng trời 60 %, ánh vàng ≤ 3400 K. Metadata `lighting.backplate_luminance`, `lighting.backplate_exposure_ev`. Studio Random generate bật sẵn. |
| `officer_yaw_mode` | không | `free` (mặc định, dùng `officer_yaw_deg`) hoặc `road` (config 1.6): viên cảnh sát quay 0 / 90 / 180 / 270° so với hướng xe chạy (hướng nhìn của ảnh nền, xoay thêm `road_yaw_deg` của ảnh) cộng lệch nhỏ `road_yaw_jitter_deg` (mặc định [-5, 5]). |
| `background_ids`, `camera_overrides` | không | Chỉ dùng các ảnh nền này; thay khoảng của camera profile bằng giá trị cố định (`distance_cm`, `height_cm`, `azimuth_deg`, `focal_length_mm`, `aim_height_cm`, `roll_deg` dạng `[lo, hi]`, `sensor_width_mm`). Studio *Random generate* dùng chúng để khoá thông số (config 1.6). |
| `image_format` | không | `png` (mặc định, không mất dữ liệu) hoặc `jpg` (nhẹ hơn): định dạng file ảnh rgb của mọi mẫu (config 1.3+). Mask luôn là PNG. Metadata ghi `image.format`. |
| `officer_yaw_deg` | không | Khoảng hướng của officer `[min, max]` (mặc định `[0, 0]`; `min ≤ max`). |
| `baton` | không | `{"enabled": true, "hand": "r"\|"l", "length_cm", "radius_cm", "handle_cm"}`. Bật: viên cảnh sát **cầm** baton (tay đóng quanh baton, pose tự điều chỉnh). Tắt/vắng: baton **treo ở thắt lưng** (vị trí theo fit của model). Xem [mục 7c](#7c-baton-và-bục-giao-thông). |
| `podium` | không | `{"probability": p}` (0–1): xác suất mỗi mẫu đứng trên bục giao thông (luồng RNG riêng `podium`; chỉ backend `mesh`). |
| `lighting` | không | Bất kỳ khoá nào trong `sun_elevation_deg`, `sun_azimuth_deg`, `sun_intensity`, `ambient_intensity`, `color_temperature_k`, `exposure_ev`, mỗi cái `[min, max]`. Mặc định: độ cao 15–75, azimuth −180–180, mặt trời 0,6–1,1, môi trường 0,25–0,5, 4500–7000 K, phơi sáng −0,5–0,5 EV. |
| `shadow` | không | Config 1.7, D46: bóng đổ của viên cảnh sát (backend `mesh`). `{"enabled": true, "source": "photo" \| "random" \| "fixed", "direction_deg": [min, max], "length_ratio": [min, max], "strength": [min, max], "softness": [min, max], "contact": 0–1, "jitter_deg": 8, "unknown_sun": "random" \| "contact" \| "off", "no_sun": "contact" \| "random" \| "off"}`. `photo`: hướng / độ dài theo bóng trong ảnh nền (đánh dấu tay trước, tự dò sau; ảnh chưa đọc được đọc khi nạp config); hướng so với camera: 0 = ra xa, 90 = phải, 180 = về phía camera, −90 = trái; `length_ratio` = dài bóng / chiều cao. Mặc định: strength 0,45–0,65, softness 0,15–0,4, contact 0,5, ảnh âm u / đêm chỉ có bóng tiếp xúc. Metadata `lighting.shadow`. |
| `validation` | không | `min_bbox_height_px` (24), `min_keypoints_in_frame_frac` (0.25), `require_signal_arm_in_frame` (false), `max_bbox_edge_touch` (true: bbox chạm mép ảnh bị loại), `min_image_std` (2.0), `side_ambiguity_margin_deg` (10.0). |
| `max_attempts_per_sample` | không | Số lần bốc lại trước khi một mẫu bị ghi nhận là lỗi. |
| `max_pose_tries` | không | Số lần bốc lại pose mỗi lần thử trước khi bỏ cuộc với một cử chỉ. |

Config sai bị từ chối ngay với thông báo một dòng: khoá lạ, `num_samples` < 1, `master_seed` không phải số nguyên ≥ 0, khoảng `[min, max]` ngược, thư viện nền không tồn tại hoặc rỗng, `backend` chưa có. Override từ dòng lệnh (`--num`, `--seed` ...) cũng được kiểm tra theo schema.

## 6. Một mẫu sinh ra chứa gì

Mỗi mẫu `<id>` tạo ra:

- `<id>.rgb.png` (hoặc `<id>.rgb.jpg` nếu `image_format` là `jpg`) — ảnh.
- `<id>.mask.png` — instance mask của viên cảnh sát.
- `<id>.json` — các nhãn:
  - id/nhãn cử chỉ, biến thể pose, pose bàn tay, (các) tay ra hiệu, và mọi góc khớp;
  - bounding box (từ mask; `xyxy`, `xywh`, `truncated`, diện tích);
  - keypoint 2D (`coco17`, `hand21_l`, `hand21_r`) kèm độ sâu và cờ trong khung hình, và keypoint 3D trong thế giới;
  - camera: `K`, `R`, vị trí, FOV, tiêu cự, cảm biến;
  - nền đã dùng và cách ghép, ánh sáng, uniform và độ sáng vải ngẫu nhiên của nó (`officer.uniform_brightness`);
  - ống kính (`camera.lens`: mô hình Brown `k1`, `k2`, vòng tròn ảnh), baton đang cầm (`officer.baton`: tay, kiểu cầm, độ khép nắm tay, đầu cán/đầu mút trong thế giới; `{"enabled": false}` khi baton ở thắt lưng) và bục (`officer.podium`), định dạng ảnh (`image.format`);
  - **chỉ dẫn cho người ở vị trí camera** (`traffic_control`): camera ở phía nào của viên cảnh sát, phía đó có mơ hồ
    không (gần ranh giới hai phía), và lệnh của cử chỉ cho phía đó (`instruction_for_observer`, ví dụ STOP; `null`
    với camera hạ tầng);
  - seed và hash của asset.
- `manifest.json` (một file cho mỗi dataset, phiên bản 1.1) — config, hash của mọi mẫu, các lỗi, số lần bị loại, thư viện nền đã dùng (đường dẫn tương đối với repo, dấu `/`, không lộ tên người dùng).

Phiên bản metadata hiện tại là **1.9**; generator **0.7.0**.

## 7. Khái niệm và quy ước

- **Gesture ≠ pose.** *Gesture* là một ý nghĩa (`profiles/<CC>/gestures.json`, có ràng buộc và
  chỉ dẫn theo từng phía). *Pose* là một cấu hình vật lý của cơ thể (`poses/`). Không bao giờ gộp chúng:
  một cử chỉ liệt kê các biến thể pose và ràng buộc, và mọi pose thoả ràng buộc đều là một cách thể hiện hợp lệ.
- **Kiến thức về quốc gia chỉ nằm trong `profiles/<CC>/*.json`** (`profile.json`, `gestures.json`,
  `uniforms.json`, `camera_profiles.json`). Không có logic quốc gia trong code. Dữ liệu VN được đánh dấu
  `verified: false` cho tới khi được rà soát theo quy định hiện hành.
- **Khả năng tái lập.** Mọi yếu tố ngẫu nhiên đều đến từ `tosdg.rng.stream(seed, "<tên>")`. Một chiều
  ngẫu nhiên mới cần một tên luồng mới; không bao giờ dùng lại hoặc đổi thứ tự các luồng có sẵn.
- **Toạ độ.** Thế giới Unreal: X tiến, Y phải, Z lên, cm. Góc quay camera là
  `FRotator` của Unreal (pitch, yaw, roll tính bằng độ). Pixel `(i, j)` phủ `[i, i+1) × [j, j+1)`. Camera lỗ kim cộng méo
  ống kính xuyên tâm tuỳ chọn theo từng loại camera (xem DECISIONS D27).
- **Azimuth camera** tính trong hệ của officer: 0 = phía trước, +90 = bên phải officer.
- **Ground truth lấy từ trạng thái cảnh** (động học thuận + phép chiếu camera + mask đã render).
- **Mẫu lỗi không bao giờ được ghi ra.**
- Skeleton: `proxy_v1` (chuẩn: 67 khớp gồm 3 đốt cho mỗi ngón tay, giới hạn giải phẫu, 52 capsule,
  45 cặp tự va chạm, các bộ keypoint `coco17` và `hand21` trái/phải) và `tripo_v1` (cùng DOF, giới hạn, capsule và cấu trúc, nhưng vị trí khớp
  lấy từ rig của model Tripo và keypoint mặt/đầu đo trên mesh của model; dựng bằng `tools/build_tripo_skeleton.py`). Pose và cử chỉ dùng được cho cả hai.

## 7b. Danh sách hiệu lệnh (VN, theo QCVN 41:2019/BGTVT)

15 cử chỉ trong `profiles/VN/gestures.json` (`verified: false`, chưa đối chiếu văn bản gốc). Hướng (trước/sau/trái/phải) tính theo cơ thể viên cảnh sát;
dựng theo infographic LuatVietnam "Cách nhận biết hiệu lệnh của cảnh sát giao thông" và mô tả đi kèm.

| Cử chỉ (`id`) | Tư thế | Người nhận chỉ dẫn |
|---|---|---|
| `VN_ARM_VERTICAL` | một tay giơ thẳng đứng (biến thể `arm_r`, `arm_l`) | mọi phía: STOP |
| `VN_ARMS_HORIZONTAL` | hai tay (hoặc một tay: `right_only`, `left_only`) dang ngang | trước và sau: STOP; trái và phải: PROCEED |
| `VN_RIGHT_ARM_FORWARD` | tay phải đưa ngang về phía trước | trước: TURN_RIGHT; sau và phải: STOP; trái: PROCEED_ANY_DIRECTION |
| `VN_STOP_FRONT` | tay phải đưa thẳng ra trước, bàn tay dựng đứng | phía trước: STOP |
| `VN_STOP_LEFT` / `VN_STOP_RIGHT` | cánh tay trái (phải) giơ ngang vai, cẳng tay gập lên, lòng bàn tay dựng đứng hướng ra ngoài (hoặc cả tay dang ngang, bàn tay bẻ lên) | bên trái (phải): STOP |
| `VN_SLOW_DOWN_LEFT` / `VN_SLOW_DOWN_RIGHT` | tay trái (phải) đưa chếch ra cạnh hông, lòng bàn tay úp xuống | bên trái (phải): **SLOW_DOWN** |
| `VN_SPEED_UP_LEFT` / `VN_SPEED_UP_RIGHT` | tay trái (phải) gập, bàn tay sát tai hoặc vắt ngang ngực | bên trái (phải): **SPEED_UP** |
| `VN_TURN_LEFT_IN_FRONT` | hai tay gập vuông góc trước ngực, hai lòng bàn tay hướng thẳng về trước | bên trái: **TURN_LEFT** |
| `VN_NO_SIGNAL` | đứng thẳng, không ra hiệu (mẫu âm khó) | không có chỉ dẫn |
| `VN_HELLO` | vẫy chào: một tay giơ lên, bàn tay mở ra trước (không phải hiệu lệnh); phân biệt với THUMB_UP bằng hình bàn tay (4 ngón duỗi) | không có chỉ dẫn; `weight` 0 — chỉ sinh khi config đặt `gesture_weights` |
| `VN_THUMB_UP` | giơ ngón cái: nắm 4 ngón, ngón cái duỗi (không phải hiệu lệnh) | như trên |
| `VN_A_POSE` | tư thế A của model gốc: đứng, hai tay buông chếch ra ~20°, bàn tay mở (không phải hiệu lệnh; weight 0, chỉ chọn qua `gesture_weights`) | không có chỉ dẫn |

Hiệu lệnh nhận chỉ dẫn `NONE` cho các phía không được nêu trong nguồn. Mỗi pose trong `poses/body/` phải được ít nhất một cử chỉ dùng (có test).

## 7a. Model viên cảnh sát (model 3D của bạn)

**Phạm vi (v1):** bạn cung cấp một GLB có texture của người *đã mặc đồng phục*; model đó là thứ xem trước và generator vẽ.
"Mặc" đồng phục lên model từ ảnh là **future work** (xem `docs/PROJECT_STATE.md`).

```
python -m tosdg import-model duong/dan/officer.glb          # hoặc Studio -> "Officer model (3D)" -> Upload model (.glb)
```
```
profiles/VN/models/<id>.glb        model nguồn của bạn (giữ nguyên như đã upload)
profiles/VN/assets/<id>.fit.json   thiết lập nhập (height_cm, ngưỡng màu da, các hiệu chỉnh bên dưới, tuỳ chọn joint_map / bind_pose) - chỉ ghi một lần, sửa tay được
profiles/VN/assets/<id>.glb        model đã đặt lên skeleton dự án (renderer đọc file này; Unreal / Blender nhập được)
profiles/VN/uniforms.json          model trở thành mục đầu tiên, weight 1 (các model trước đó vẫn được liệt kê với weight 0, và phải được nhập lại nếu bạn đổi skeleton)
```
- **Skeleton:** nếu GLB đã có skin mà các xương mang tên UE5 (`pelvis`, `spine_01`, `upperarm_l`, `index_01_r` ...; các xương phụ `*_twist_NN_x`
  được gộp vào xương gốc), skeleton và skin weights của nó được dùng nguyên - chỉ khớp lại tư thế bind (DOF của
  tay/chân để skeleton dự án vào đúng tư thế chữ A của model) và các đỉnh được dịch cùng khớp của chúng để mọi điểm xoay là của skeleton.
  Tên xương khác có thể ánh xạ bằng `fit.joint_map`. Model **không** có skeleton dùng được thì được tự gán xương theo khoảng cách tới xương.
  Kết quả cho biết đã đi theo đường nào.
- **Skeleton `tripo_v1`.** Rig tự động của Tripo có tỉ lệ khác `proxy_v1`; ép model vào `proxy_v1` kéo đỉnh lưới 4–14 cm làm nhàu ngón và méo khuỷu. Vì vậy model Tripo được dựng trên `tripo_v1`: cùng DOF và giới hạn, vị trí khớp lấy từ rig của model, keypoint mũi/mắt/tai và capsule đầu/mũ đo trên mesh.
  Đổi model hoặc fit thì chạy `python tools/build_tripo_skeleton.py` rồi `import-model` lại.
- **Hiệu chỉnh trong `.fit.json`** (đều mặc định tắt, đều chỉ dời điểm xoay hoặc sửa trọng số, không thay model gốc):

  | Khoá | Tác dụng |
  |---|---|
  | `shoulder_out_cm` | dời khớp vai ra ngoài (auto-rig đặt vai trong ngực nên tay cắm vào sườn); beta: 4 |
  | `wrist_drop_cm` | dời khớp cổ tay về phía bàn tay (rig đặt cổ tay trong tay áo); beta: 3 |
  | `forearm_scale` | co/giãn cẳng tay, bàn tay và ngón dời theo; beta: 0,9 (cổ tay ngang thắt lưng) |
  | `belt_baton` | nơi baton treo ở thắt lưng: `handle_cm`, `direction`, `length_cm`, `radius_cm`, `joint`; không có khoá = không hiện baton khi không cầm |
  | `close_hand_holes` | lấp khe hở nhỏ trên găng tay (mặc định bật); khe hở nở thành nêm trong suốt khi nắm tay |
  | `strip_stray_arm_weights` | bỏ trọng số tay khỏi bao súng/thắt lưng/gấu áo dưới nách (mặc định bật); không thì giơ tay xé bao súng |
  | `bind_pose`, `joint_map`, `height_cm`, `skin_hue_deg` ... | tư thế bind, ánh xạ tên xương, chiều cao, ngưỡng màu da |

- **Model không được mang baton riêng** (baton là prop, xem mục 7c); model có baton đeo sẵn sẽ làm một ảnh có hai baton.
- **Hình thức** là texture của chính model (bản đồ màu cơ bản của material). Màu từ ảnh không đổi màu nó; khoảng độ sáng chỉ
  nhân lên texel không phải da (chỉnh `skin_hue_deg` / `skin_saturation` trong fit nếu vải bị nhầm là da).
- **Quy ước GLB:** glTF 2.0, +Y lên, mặt trước +Z, mét. Unreal là hệ toạ độ tay trái, nên khi chuyển đổi thứ tự đỉnh tam giác bị đảo (`to_gltf` / `from_gltf` đã được test).
- **Lưới an toàn:** `generate` / `validate-profile` từ chối mesh nhập bằng một fit khác, không khớp skeleton, hoặc một
  uniform có thể được chọn mà không có mesh. Các GLB và file fit nằm trong `assets_sha256`, nên `reproduce` phát hiện thay đổi.
- **Giới hạn:** tỉ lệ skeleton cố định, nên model có chiều dài chân khác bị dịch chuyển (đế giày có thể thấp vài cm); ngón tay
  theo trọng số riêng của model (nắm chặt vẫn có nếp gấp góc cạnh ở mu bàn tay); keypoint hand21 theo vị trí khớp của rig chứ không theo giải phẫu; nguồn gốc/bản quyền của model là trách nhiệm của bạn (GLB có mặt người là dữ liệu cá nhân - kiểm tra trước khi chia sẻ dataset).

## 7c. Baton và bục giao thông

**Baton** là một prop (`profiles/VN/props/baton.glb`), không thuộc model người:
- **Tắt** (mặc định): baton treo ở thắt lưng, gắn cứng vào xương `pelvis` (đi theo thân, không đung đưa theo đùi), ở chỗ fit `belt_baton` của model chỉ định.
- **Bật** (`"baton": {"enabled": true, "hand": "r"}` trong config, ô *Holding a baton* trong Studio, `pose-preview --baton r`): baton nằm trong tay đã chọn. Tay đóng thành nắm đấm chặt quanh baton (cổ tay thẳng, ngón cái tựa phía quay, mỗi ngón khép tới mức baton cho phép); `VN_ARM_VERTICAL` và `VN_ARMS_HORIZONTAL` giữ baton dọc cánh tay (`baton_axis: along`), các cử chỉ khác giữ ngang nắm tay. Pose không hợp lệ khi baton chạm thân thì bị bốc lại.
- Một ảnh luôn chỉ có một baton; baton là một phần của officer nên nằm trong instance mask và bbox. Metadata `officer.baton` ghi tay, kiểu cầm, độ khép nắm tay và đầu cán/đầu mút trong thế giới khi cầm.
- Không có `baton.glb` thì baton là hình trụ. Backend `proxy` (capsule) chỉ vẽ baton cầm tay.

**Bục giao thông** (`profiles/VN/props/podium.glb`, đường kính 84 cm, cao khoảng 47 cm): config `podium.probability` hoặc ô *Standing on a podium* / `pose-preview --podium`. Bục gắn cứng vào xương `root` (đứng yên khi đổi pose), viên cảnh sát đứng trên mặt bục nên cao hơn mặt đất; chiều cao camera và điểm ngắm vẫn tính từ mặt đất. Bục không thuộc instance mask (bục che chân thì mask mất phần bị che).

## 7d. Camera, ống kính và nền

- **Bốn loại camera** (`profiles/VN/camera_profiles.json`): `dashcam` (cam hành trình, cao 105–130 cm), `driver_eye` (người lái), `motorbike_rider` (camera mũ/ghi đông, góc rộng, nghiêng roll ±5°), `roadside` (không méo). Camera CCTV trên cột đã bỏ (D40). Chúng khác nhau về vị trí, tiêu cự, nhiễu hướng ngắm và ống kính.
- **Ống kính** (D27): mô hình Brown xuyên tâm `k1`, `k2` (k1 < 0 là méo thùng) và vùng đen ngoài vòng tròn ảnh (`black_border_probability`); keypoint, bbox, baton và đường chân trời đều nằm trong ảnh đã méo, vẫn lấy từ trạng thái cảnh. Ảnh nền cũng bị méo cùng officer. Khoảng 30 % mẫu không có viền đen để dataset có cả hai kiểu.
- **Nền**: Studio và `backgrounds import` nhận PNG, JPG, WebP, BMP, GIF, TIFF (WebP/BMP/GIF/TIFF lưu thành JPG); đường chân trời được dò tự động (`src/tosdg/horizon.py`, ước lượng, chỉ là giá trị ban đầu) và kéo tay được; thư viện nền cục bộ `data/backgrounds` là ảnh đường phố thật (`tosdg fixtures` tạo nền tổng hợp chỉ để thử).
- **Định dạng ảnh xuất**: `image_format` `png` (không mất dữ liệu) hoặc `jpg` (chất lượng 95); mask luôn PNG.

## 8. Các việc thường làm

**Dùng ảnh nền của riêng bạn**
1. Trong Studio, upload ảnh và đặt đường chân trời, tag cho từng ảnh (ảnh đã nằm trong thư viện `data/backgrounds`),
   hoặc dùng `backgrounds import LIBRARY photo.jpg --tag ground_level --horizon 0.45`. Bỏ ảnh: xoá trong Studio hoặc
   `backgrounds remove LIBRARY ID`.
2. Trỏ `background_library` của config vào thư mục đó và chạy `generate`.

**Dùng model 3D của bạn**
`python -m tosdg import-model officer.glb`, rồi `python -m tosdg pose-preview stop_bent_r` để xem, rồi `generate`.
Model Tripo mới: chạy `python tools/build_tripo_skeleton.py officer.glb` trước (dựng `tripo_v1` từ rig của nó), xem lại ảnh cận cảnh vai, cổ tay, mặt và bàn tay nắm,
chỉnh `shoulder_out_cm`, `wrist_drop_cm`, `forearm_scale`, `belt_baton` trong `.fit.json` nếu cần, rồi `import-model` lại.

**Cho viên cảnh sát cầm baton hoặc đứng trên bục**
Thêm `"baton": {"enabled": true, "hand": "r"}` và/hoặc `"podium": {"probability": 0.3}` vào config (xem mục 5 và 7c).

**Tạo một pose mới và dùng nó**
1. Kéo skeleton trong Studio, rồi **save to poses/** với một id mới.
2. Thêm một biến thể vào một cử chỉ trong `profiles/<CC>/gestures.json` tham chiếu id pose đó, kèm
   các ràng buộc khiến nó là một thể hiện hợp lệ của cử chỉ.
3. `validate-profile configs/smoke.json`, rồi `generate`.

**Kiểm tra một thiết lập camera dùng được không trước khi sinh**
Đặt camera trong Studio và đọc **Generator check**; dùng **Render preview** để có bbox chính xác.

**Tạo dataset lớn hơn**
`python -m tosdg generate configs/smoke.json --out out/big --num 200 --seed 7`
(khoảng 0,2 giây mỗi mẫu ở 640×360 với backend mesh; proxy capsule tương tự).

**Thêm một quốc gia**
Sao chép `profiles/VN/` sang `profiles/<CC>/`, sửa bốn file JSON của nó, đặt `"profile": "<CC>"` trong config,
và chạy `validate-profile`. Không cần đổi code.

## 9. Cấu trúc repo

```
configs/        generation config (smoke.json)
profiles/<CC>/  kiến thức quốc gia: profile, gestures, uniforms, camera profiles
poses/          file pose body/ và hands/ (chỉ pose vật lý)
characters/     định nghĩa skeleton (proxy_v1 chuẩn; tripo_v1 dựng từ rig Tripo)
schemas/        JSON schema cho mọi loại file
src/tosdg/      package
  cli.py          dòng lệnh
  generate.py     sinh hàng loạt          scene.py       lấy mẫu cảnh theo seed
  gestures.py     cử chỉ → lấy mẫu pose   skeleton.py    động học, giới hạn, va chạm
  camera.py       camera lỗ kim           groundtruth.py keypoint, bbox, chỉ dẫn giao thông
  backgrounds.py  thư viện backplate      lighting.py    lấy mẫu ánh sáng
  render_proxy.py renderer capsule để test validate.py    kiểm tra mẫu/dataset
  import_mesh.py, models.py  nhập model người (GLB) lên skeleton; uniform_mesh.py  đọc/ghi GLB
  render_mesh.py  skinning + renderer z-buffer (CPU)   render_gl.py  renderer GPU (OpenGL, PBR)   mesh_renderer.py  chọn cpu/gpu
  baton.py, podium.py  prop baton (cầm tay / thắt lưng) và bục giao thông
  export.py       xuất COCO / YOLO-pose / CVAT
  lens.py         méo ống kính + viền đen   horizon.py     dò đường chân trời   imageio.py  nhập/xuất ảnh
  studio/         công cụ tương tác (core.py IK + đánh giá, server.py, workspace.py, static/index.html)
tests/          test đơn vị + end-to-end + studio
profiles/<CC>/models/  model người nguồn của bạn (GLB);  profiles/<CC>/assets/  fit JSON của chúng + GLB có skin mà renderer dùng
tools/          build_proxy_skeleton.py, build_tripo_skeleton.py (dựng lại skeleton),
                stop_studio.ps1 (đóng Studio đang chạy cho start_studio.bat), bring_to_front.ps1
docs/           PROJECT_STATE.md (tiến độ), ARCHITECTURE.md, DECISIONS.md (D1–D47), HUONG_DAN.md (tài liệu này), CHANGELOG.md
start_studio.bat, install_requirements.bat   mở Studio một lần bấm (đóng bản đang chạy trước); cài/nâng cấp gói
version/        bản lưu RAR của toàn bộ file tại các mốc (ví dụ tosdg_1st_acceptable.rar)
data/, out/     dữ liệu sinh ra (git bỏ qua); data/machine.json: cấu hình máy cho --workers
```

## 10. Kiểm thử

```
python -m unittest discover -s tests -t tests
```
324 test, khoảng 3 phút, dùng `unittest` chuẩn (không dùng pytest). Chúng bao phủ schema, RNG, động học
và quy ước camera, lấy mẫu cử chỉ, ground truth, chạy end-to-end trên cả backend mesh và proxy với
sinh lại giống từng byte, các ca kiểm tra hợp lệ thất bại, model 3D (quy ước GLB, skin weights, bộ raster chính xác, keypoint
nằm trong mask của mesh ở mọi pose cơ sở, độc lập của luồng độ sáng, phát hiện asset cũ, nhập model có skeleton / không có skeleton),
và IK của Studio, độ khớp phép chiếu với generator, an toàn khi upload và API HTTP (gồm upload model).
Từ v2.5 còn có: giải phẫu của model (tỉ lệ chi, keypoint mặt nằm trên mặt, cổ tay ngang thắt lưng, tay không lún vào thân, giơ tay không kéo bao súng), khe hở trên găng tay, baton (cầm/đeo, một baton mỗi ảnh),
CLI và vệ sinh dataset (`tests/test_cli.py`: thư mục đầu ra, tham số, lỗi một dòng, đường dẫn khả chuyển, `reproduce`), cổng và khoá workspace của Studio.

Sau khi sửa skeleton proxy: `python tools/build_proxy_skeleton.py` (hoặc `tools/build_tripo_skeleton.py` cho `tripo_v1`), rồi nhập lại model (`python -m tosdg import-model profiles/VN/models/<id>.glb`) -
một GLB cũ làm `generate`, `validate-profile` và các test bị lỗi.

## 11. Hạn chế đã biết

Danh sách đầy đủ và cập nhật ở `docs/PROJECT_STATE.md`. Tóm tắt:

- **Chưa có renderer chân thực.** Renderer GPU (`mesh_renderer: gpu`) có PBR, bóng nắng, SSAO và khớp độ sáng ảnh nền, nhưng
  không có chiếu sáng toàn cục giữa người và cảnh; renderer CPU chỉ có một mặt trời + ánh sáng môi trường; `proxy` vẽ capsule (để test).
  Unreal (renderer sản phẩm) chưa làm. Mô hình chỉ học trên ảnh tổng hợp nên được kỳ vọng cần thêm dữ liệu thật để chạy tốt trên ảnh thật.
- Dữ liệu luật VN **chưa được xác minh** (`verified: false`).
- Một model duy nhất (`beta`, Tripo, rig UE5, skeleton `tripo_v1`): một gương mặt, một dáng, một bộ đồng phục; mặt người có
  nguồn gốc / bản quyền chưa rõ — kiểm tra trước khi chia sẻ dataset hoặc repo. Keypoint bàn tay theo rig Tripo, chưa đúng vị trí giải phẫu.
- Bóng đổ (D46) giả định mặt đất phẳng: không leo lên xe / tường, không bị vật phía trước che; hướng nắng tự dò chỉ có ở ảnh nắng đủ người / xe (còn lại đánh dấu tay hoặc ngẫu nhiên).
- Che khuất chỉ do chính cơ thể / gậy (D41), chưa có xe / người trong ảnh nền che viên cảnh sát; méo ống kính chỉ xuyên tâm k1/k2.
- Hiệu chỉnh camera giả định mặt đường phẳng; bố cục giao lộ và đường chân trời đặt bằng mắt.
- Studio: IK không có tiếp xúc mặt đất; Render preview vẫn căn theo đường chân trời đã lưu; JSON cảnh của Studio chưa được `generate` đọc.
- Studio không làm mờ mặt (D24): chỉ upload ảnh bạn được phép dùng.
- **Future work — uniform từ ảnh:** upload ảnh đồng phục rồi "mặc" lên model cơ thể (baking texture / dựng lại 3D).
