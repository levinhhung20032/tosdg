# CHANGELOG

Các thay đổi theo phiên bản, mới nhất ở trên. Chi tiết thiết kế: `docs/DECISIONS.md` (D1–D43). Bỏ qua các chỉnh nhỏ (chữ, căn lề, đổi tên biến).

## Chưa phát hành

- **Gỡ tính năng ngoài phạm vi** (D44): công cụ chỉ xoay quanh việc đặt một viên cảnh sát vào bất kỳ ảnh nào với cử chỉ tuỳ chỉnh / ngẫu nhiên.
  - Trạng thái đèn giao thông và biển báo của ảnh nền (`scene`, `import --traffic-light / --sign`, ô *traffic light* trong Studio), thứ tự ưu tiên tín hiệu (`control_precedence`, profile 1.1) và các nhãn `signals` / `precedence` / `effective_instruction` / `effective_source` / `conflict`. `traffic_control` (metadata 1.8) còn: phía camera đứng và `instruction_for_observer`; `labels.csv` bỏ hai cột `effective_*`. Thư viện cũ có `scene` vẫn đọc được (bị bỏ khi nạp).
  - `size_reference: "people"` (định cỡ theo một người; `auto` thay thế) và `people.plan_from_person`.
  - Panel ẩn "Uniform views" của Studio (bộ ảnh quanh người, trích màu đồng phục từ ảnh, ảnh "ghost"), `studio/appearance.py`, các API `/api/uniform_*`, `/api/upload_uniform_view`, `/files/`.
- **Giới hạn độ sáng** (D45): khoảng `lighting` trong config có giới hạn (exposure −3..3 EV, cường độ ≥ 0; trước đó `exposure_ev` lớn làm tràn số và dừng chương trình), exposure cuối của mỗi mẫu bị kẹp trong [−2,5; 2] EV, `match_backplate_light` chỉ làm tối tới −2 EV (trước −4: viên cảnh sát gần như đen), và mẫu có viên cảnh sát quá tối / cháy sáng bị loại (`validation.min_officer_level` 12, `max_officer_level` 245, mức sRGB trung bình của các pixel viên cảnh sát).
- `tools/mirror_commit.py`: chép commit cuối của một bản tosdg (thư mục con trong repo nhóm, hoặc repo riêng) sang bản kia và commit với cùng message / tác giả; không chép lịch sử git.
- `.gitignore`: `data/` giữ cây thư mục (`.gitkeep`), chỉ bỏ qua nội dung các thư mục con.
- **Vị trí và hướng quay theo giao lộ** (D43): `officer_yaw_mode: "junction"` — viên cảnh sát đứng giữa ngã ba / ngã tư / bùng binh (lệch ≤ 1 m), quay mặt theo loại giao lộ (ngã tư, bùng binh: về một nhánh; chữ T: về thân chữ T, cả ba nhánh trong tầm nhìn; chữ Y: về một nhánh hoặc thấy ít nhất hai nhánh) và theo cử chỉ (các phía cử chỉ ra lệnh phải là đường). Bùng binh: bục và gậy cầm tay bốc ngẫu nhiên (`roundabout`, luồng `roundabout`). Bố cục giao lộ đặt tay (`backgrounds junction`, thư viện 1.2 khối `junction`). Module mới `src/tosdg/junction.py`; luồng mới `junction`. Metadata 1.8 `background.junction`.
- **Ánh sáng theo ảnh nền** (D43): `match_backplate_light: true` làm tối viên cảnh sát theo độ sáng ảnh nền quanh chỗ đứng (tới −4 EV); ảnh đêm có ánh đèn đường vàng thay nắng. Ảnh nền đêm / chiều tối được nhận vào thư viện.
- **Thư viện nền 50 ảnh** (D43): thêm 30 ảnh Wikimedia Commons (CC BY / CC BY-SA / CC0): 10 bùng binh, 12 ngã tư, 5 ngã ba chữ T, 1 ngã ba chữ Y, 2 đường phố; 4 ảnh đêm, 1 chiều tối. Mỗi ảnh có đường chân trời đặt tay, mask mặt đường, người / xe dò bằng YOLO và bố cục giao lộ; 9 ảnh cũ cũng có bố cục giao lộ. Nguồn ở `data/backgrounds/ATTRIBUTION.md`.
- Studio Random generate: mặc định *officer turn: junction*, bật `match_backplate_light`.
- **Hiệu chỉnh camera của ảnh nền từ người / xe** (D42): `size_reference: "auto"` không còn dựa vào đường chân trời. Độ cao và đường chân trời (⇒ pitch) của máy đã chụp ảnh nền được suy từ ít nhất 3 người (coi cao 165 cm), nếu không đủ thì người + xe (ô tô, xe buýt, xe tải, xe máy, xe đạp) + rào chắn đánh dấu tay, cuối cùng mới tới đường chân trời (module mới `src/tosdg/calibrate.py`). Camera ảo đặt ở độ cao đó, pitch cho đường chân trời trùng ảnh, khoảng cách chọn để viên cảnh sát nằm trong khung và đủ cỡ, không đè lên người / xe gần hơn (luồng `calibration`). Bộ dò YOLO lưu thêm phương tiện (`objects`, thư viện 1.2); lệnh mới `backgrounds objects` (đánh dấu rào chắn / xe bằng tay) và `backgrounds calibrate`; import luôn dò người / xe. Metadata 1.8 `size_reference.mode: calibrated`. Generator 0.6.0. Thư viện cục bộ đã dò lại cả 20 ảnh.
- **Keypoint bị che**: metadata `keypoints_2d.<set>.occluded` (1.8): điểm trong ảnh mà tia nhìn từ camera đi qua capsule cơ thể của chính viên cảnh sát (hoặc baton đang cầm) trước khi tới; điểm trên mặt (mũi, mắt, tai) bị che khi pháp tuyến quay khỏi camera. Xuất: COCO / YOLO visibility 1, CVAT `occluded="1"`. `visibility_model` ghi cách tính.
- **`labels.csv`** khi xuất (mọi định dạng): ảnh nào, viên cảnh sát ra hiệu lệnh gì (nếu có), nghĩa cho từng phía và cho người quan sát.
- **`size_reference: "auto"`**: đường chân trời đặt tay (`horizon_source: manual`, thư viện 1.2) được căn với camera ảo ở độ cao máy chụp ảnh (`camera_height_cm`) — sửa nguyên nhân viên cảnh sát to hơn người trong ảnh (camera hành trình 105–130 cm trước ảnh chụp ngang tầm mắt ~155 cm); chỉ ảnh có đường chân trời không chắc mới dùng người tham chiếu, và YOLO chỉ chạy cho chúng (lúc import, hoặc một lần trước khi sinh). Lệnh `backgrounds set`; Studio: kéo đường chân trời = đặt tay, ô *camera height cm*.
- **`lens_distortion: false`**: ảnh không méo, không viền đen.
- **Thư viện nền**: bỏ ảnh nhìn từ trên cao `nga4_hcm_phunhuan_elevated`; thêm 11 ảnh Wikimedia Commons (CC BY / CC BY-SA, nguồn ở `data/backgrounds/ATTRIBUTION.md`), trong đó 3 ảnh góc camera hành trình (`dash_*`, tag `dashcam`); 20 ảnh, 17 ảnh có đường chân trời đặt tay. Xoá các dataset cũ trong `out/`.
- **Cỡ viên cảnh sát theo người trong ảnh** (D40): `backgrounds people` / `import --people detect` / upload trong Studio dò người đứng và người ngồi xe máy/xe đạp (YOLO11 `ultralytics`, `src/tosdg/people.py`), lưu điểm chân và chiều cao pixel (thư viện **1.2**). Config `size_reference: "people"` (config **1.6**): ảnh nền được phóng để người đó cao bằng viên cảnh sát / k (k ∈ [0,9; 1,1], luồng `size_reference`), chân cùng hàng; viên cảnh sát tránh mọi người đã dò, chân trên mặt đường. Ảnh không có người dùng đường chân trời; không nhất quán được thì loại (`size_reference`). Metadata **1.8** `background.size_reference`. Generator 0.5.0.
- **`min_officer_height_px` cần `size_reference: "people"`**: trước đó chỉ viên cảnh sát to lên trước một ảnh nền giữ nguyên, nên to hơn hẳn người và xe trong ảnh; nay ảnh nền phóng theo.
- **Hướng quay theo đường** (D40): `officer_yaw_mode: "road"`: 0 / 90 / 180 / 270° so với hướng xe chạy (hướng nhìn của ảnh + `road_yaw_deg` của ảnh, thư viện 1.2) ± 5°, luồng `road_yaw`; `camera.placement.road_quarter`. Studio: ô *road direction °*.
- **Studio Random generate** (D40): panel mới, khoá cử chỉ / loại camera / góc nhìn hiện tại / ảnh nền / baton / bục, phần còn lại ngẫu nhiên; chạy nền, xem tiến độ và ảnh nhỏ; ra `out/studio_random/`. API `/api/random_generate`, `/api/random_status`, `GET /gen/<file>`. Config mới `background_ids`, `camera_overrides`.
- **Bỏ camera CCTV** (`cctv_elevated`) khỏi profile VN, Studio và mọi danh sách; còn 4 loại camera.
- Bộ test: 264 → 316 (`tests/test_people.py`, `tests/test_calibrate.py`, `tests/test_junction.py`, `tests/test_lighting_bounds.py`, `tests/test_mirror.py`).

## v3.0 — 2026-10-15

Từ v2.7: dataset dùng được ngay trong công cụ huấn luyện và gán nhãn (xuất COCO keypoints / YOLO-pose / CVAT, chia train/val/test), viên cảnh sát đủ lớn trong khung (`min_officer_height_px`), và Studio với `generate` dùng chung một thư mục ảnh nền.

- **Một thư mục ảnh nền** (D39): Studio đọc, upload vào và xoá khỏi thẳng thư viện của generator (`data/backgrounds`, hoặc `--library`). Trước đó Studio có thư mục riêng (`data/studio/backgrounds`) và nút *add to generator library* sao chép sang; xoá trong Studio không xoá bản sao nên `generate` vẫn dùng ảnh đã xoá. Upload nay gắn tag `day`, `ground_level`, tự dò đường chân trời và mask mặt đường; nút *save tags* lưu tag và đèn giao thông. Workspace cũ được chuyển vào thư viện một lần khi khởi động (cùng id và cùng kích thước thì coi là cùng ảnh), rồi thư mục cũ bị xoá. Bỏ `/api/add_to_library`; thêm `GET /bg/<file>`. Bỏ `data/street_photos` (bản tải gốc trùng từng byte với ảnh trong thư viện); nguồn và giấy phép ảnh chuyển sang `data/backgrounds/ATTRIBUTION.md`. CLI mới `backgrounds remove LIBRARY ID...`; `BackgroundLibrary.remove`.
- **Xuất dataset** (D38): `tosdg export DATASET --format coco|yolo|cvat --out DIR [--split 0.8,0.1,0.1] [--seed N] [--yolo-classes officer|gesture]`. 59 keypoint (COCO 17 + 2 × 21 bàn tay), bbox, mask (RLE), cử chỉ; chia train/val/test bằng hoán vị có seed (luồng `export_split`). Chỉ xuất dataset qua `validate`. Module mới `src/tosdg/export.py`. Đã đọc thử bằng `pycocotools` và Ultralytics (`YOLODataset`, 0 nhãn hỏng); file CVAT chưa thử import trên một server CVAT thật.
- **Kích cỡ tối thiểu** (D38): config `min_officer_height_px` (generation config **1.5**). Khoảng cách camera được giới hạn theo tiêu cự và chiều cao pose thay vì loại rồi bốc lại; bbox cũng phải đạt. Lô thử 12 mẫu 640×360, tối thiểu 120 px: 82 lần loại → 0, 30 s → 4,5 s. Tắt thì mọi lần bốc và ảnh như cũ.
- Bộ test: 247 → 264 test (`tests/test_export.py`, thư viện nền dùng chung).

## v2.7 — 2026-10-15

Từ v2.6: viên cảnh sát chỉ đứng trên mặt đường / vỉa hè của ảnh nền (mask vùng đứng được), 10 ảnh ngã tư Việt Nam trong thư viện cục bộ, Studio khởi động lại sạch hơn.

- **Công an không còn bị đặt lên tường** (D37): mỗi backplate có thể có mask vùng đứng được (mặt đường + vỉa hè), dò bằng SegFormer (Cityscapes) khi nhập (`backgrounds import --ground detect`, mặc định), `backgrounds ground` cho thư viện có sẵn (`--force`, `--mask` vẽ tay, `--clear`), và Studio "add to generator library". Khi sinh, chân phải nằm trên mask: ảnh nền được cắt lệch ngang (luồng RNG mới `ground`) hoặc mẫu bị loại và bốc lại. Thư viện backplate 1.1 (khối `ground`), metadata 1.7 (`background.ground_check`), generator 0.4.0. Fixture có mask chính xác (`tosdg fixtures` thêm mask cho thư viện cũ). Module mới `src/tosdg/ground.py`; `backgrounds.plan_backplate` / `crop_backplate`; `lens.canvas_camera`.
- Thư viện cục bộ `data/backgrounds`: thêm 10 ảnh ngã ba/ngã tư Việt Nam (Wikimedia Commons, CC BY / CC BY-SA; nguồn ở `data/backgrounds/ATTRIBUTION.md`) với đường chân trời chỉnh tay; cả 15 nền có mask.
- Studio: danh sách ảnh nền là khung lưới 3 cột, tối đa 3 hàng, cuộn được; giữ vị trí cuộn khi chọn ảnh.
- `start_studio.bat`: đóng mọi chương trình đang giữ cổng 8765 (không chỉ Studio); tab Studio của lần chạy trước tự đóng khi tab mới mở (`boot` trong `/api/init`, BroadcastChannel); không đóng được thì hiện thông báo và ngừng lưu / ping.
- Bộ test: 233 → 247 test (`tests/test_ground.py`), ~100 giây.
- Dọn: xoá `tools/import_static_mesh.py` (chỉ bọc lại `tosdg import-model`).

## v2.6 — 2026-10-15

Từ v2.5: renderer GPU (OpenGL/moderngl, PBR như viewer của Tripo), model đẹp hơn (normal/roughness map, nách sạch, chân đứng phẳng, tay xoắn tự nhiên), Studio sửa được gesture và ghi đè gesture gốc, trục Oxyz + Ctrl+kéo, va chạm của chân, bộ test nhanh gấp ~3 lần.

### Kiểm thử lại toàn bộ (2026-10-15) — lỗi tìm thấy và đã sửa
- Renderer CPU dựng lại chuỗi mip của ba texture 4k cho **mỗi ảnh** (mesh có baton/bục là object mới mỗi mẫu, bộ đệm theo mesh bị mất): 85 % thời gian của một khung hình. Nay bộ đệm theo mảng texture.
- Mỗi `GenerationContext` / mỗi lần đọc model giải nén lại ba PNG 4k (~1,5 s): nay bộ đệm theo hash ảnh (mảng chỉ-đọc dùng chung).
- Studio: lưu pose (`/api/save_pose`) ghi đè file nhưng thư viện gesture vẫn giữ pose cũ đến khi khởi động lại; nay nạp lại cả gesture.
- Chân không có cặp va chạm với thân trên / tay (tay xuyên bàn chân không báo) — thêm 68 cặp.
- Ctrl+kéo không khác kéo thường khi điểm đích ngoài tầm với (IK trôi tới / lùi) — IK khoá trục.
- Vòng Alt+click của cổ tay chồng nhau (cùng bán kính) và khó kéo khi nhìn chéo — bán kính riêng, kéo theo mép trước.
- Trang Studio không tải (lỗi cú pháp JS do một chú thích chèn giữa dòng) và kẹt "3D uniform updating…" khi mở — sửa, có test cú pháp bằng Node.
- Sắp xếp lại gesture không đổi dữ liệu sinh ra: kiểm chứng 12/12 ảnh, mask và gesture giống hệt với cùng seed (chỉ metadata đổi vì ghi hash của `gestures.json`).

### Tính năng mới
- **Sửa gesture và ghi đè gesture gốc** (Studio, panel Pose & gesture): nạp một biến thể, chỉnh tư thế, bấm "save over the gesture". Pose phải còn là một ví dụ hợp lệ của gesture (giới hạn khớp + ràng buộc của gesture; không đạt thì hỏi "lưu dù vậy"). File cũ được sao lưu vào `poses/body/.backup/` (không vào git); biến thể đối xứng (`stop_bent_l` ↔ `stop_bent_r`) nhận pose đối xứng (kiểm tra bằng FK). API `/api/save_gesture_pose`; `poses.mirror_pose`, `poses.mirror_name`.
- **Thứ tự gesture**: ba hiệu lệnh cơ bản (tay giơ thẳng đứng, dang ngang, tay phải đưa ra trước), rồi dừng / đi chậm / đi nhanh / rẽ trái, cuối cùng các tư thế không phải hiệu lệnh (không ra hiệu, chào, giơ ngón cái, tư thế A).
- Studio: các tick "hand keypoints", "face keypoints", "spine / neck handles" mỗi tick một dòng.
- Bộ test: 231 → 233 test (thêm 3 test sửa gesture, bỏ 1 test trùng với test theo từng biến thể), 228 s → ~70 s.

- **Va chạm của chân**: skeleton `proxy_v1` / `tripo_v1` có thêm cặp va chạm chân (đùi, ống chân, bàn chân) ↔ thân trên, cổ, đầu, mũ, vai, cánh tay (ống chân, bàn chân thêm ↔ thân dưới, cẳng tay, bàn tay): 45 → 113 cặp. Trước đó một chân giơ cao xuyên qua tay hay ngực mà Pose warnings không báo. Khớp và offset không đổi.
- Studio Ctrl+kéo: bộ giải IK khoá trục đã chọn (sai số dọc trục đó nặng gấp 8 lần), nên khi điểm đích ngoài tầm với khớp vẫn không trôi tới / lùi; khi kéo hiện "Ctrl: X locked" và trục bị khoá (nét đứt) qua khớp.
- Studio Alt+click: các vòng của một khớp có bán kính khác nhau (trước đây cùng bán kính, cắt nhau, bấm hay trúng vòng khác — vòng lệch cổ tay gần như không bấm được); vòng nhìn chéo / nhìn cạnh xoay theo chuyển động chuột dọc mép trước của vòng.

- Studio: trục Oxyz ở góc khung camera (toggle `axes`). Bấm X / Y / Z để căn camera theo trục; **Ctrl+kéo** khớp giữ nguyên toạ độ trên trục đang căn (server: `target.plane` của `/api/drag`, `core.axis_plane_target`).
- Studio: khi cầm baton, cổ tay xoay bằng Alt+click (gập, lệch) nay hiện cả trên mesh 3D và Render preview (trước đó bước nắm baton ép cổ tay thẳng lại; generator vẫn giữ cổ tay thẳng). Kéo Alt+click cũng đánh giá theo pose đã nắm baton.
- Studio: sửa lỗi khung camera kẹt ở "3D uniform updating…" khi mở trang, và lỗi cú pháp làm trang không tải (thêm test kiểm tra cú pháp script bằng Node).

- **Nách không còn vệt nâu khi giơ tay (D36)**: texture gốc của Tripo có bóng nách của tư thế A vẽ sẵn (nâu đậm trên vải kaki). Importer có bước `even_armpits` (bật trong `beta.fit.json`): các texel vải quanh hai nách (đầy đủ trong 9 cm, nhạt dần tới 15 cm) tối hơn vải xung quanh được đưa về màu vải, giữ 25 % độ tối làm chi tiết nếp; phù hiệu, da, găng giữ nguyên. Nhập lại model để áp dụng.
- **Bàn chân phẳng ở tư thế mặc định (D36)**: bind pose của bàn chân trước đây khớp theo hướng xương bàn chân → mũi, nên lệch tuỳ ý ~6°; ở tư thế 0° mũi giày chếch lên ~10° (đứng trên gót). Nay bàn chân được khớp sao cho đế phẳng; skeleton `tripo_v1` dựng lại (`tools/build_tripo_skeleton.py`), model nhập lại, `a_pose` cập nhật.
- **Xoay cẳng tay/cánh tay không còn vặn khuỷu và vai (D36)**: pronation của cẳng tay và twist của cánh tay trên/đùi được phân bố dọc theo xương khi skinning (0 ở khớp gốc → đủ ở khớp ngọn), như xương twist của UE5 mà importer đã gộp vào xương chính. Ground truth (FK) không đổi.
- Studio Alt+click: khi vòng xoay chạm giới hạn khớp, điểm neo đặt lại tại góc giới hạn — kéo ngược lại thì khớp xoay ngay (trước đó phải kéo bù phần đã vượt).

- **Renderer GPU mới (`render_gl.py`, D35)**: OpenGL 4.3 qua `moderngl` trên card đồ hoạ (máy này: RTX 3060 Laptop). PBR metal/roughness (GGX) với base colour, normal map, roughness map của model (mip-map + lọc dị hướng 16x), nắng có bóng mềm (shadow map 4096², PCF), ánh sáng môi trường trời/đất ám màu theo ảnh nền (SH9 + specular lọc sẵn), SSAO, MSAA 4x trên ảnh siêu lấy mẫu 2x, tone curve ACES như three.js (viewer của Tripo), bóng đổ của viên cảnh sát xuống mặt đường. Màu đã so với video Tripo (tỉ lệ G/R, B/R của vải trùng trong ~1 %). Vẽ ~0,2–0,3 s/ảnh 1280×720. Dùng: config `"mesh_renderer": "gpu"` (schema config **1.4**), Studio chọn "mesh — 3D uniform, GPU (best)" (mặc định khi máy có GPU), `pose-preview --renderer gpu`. Mask lấy từ cùng hình học (IoU với CPU > 0,97). `moderngl` là phụ thuộc tuỳ chọn (`install_requirements.bat` cài).
- Renderer CPU giữ làm dự phòng: siêu lấy mẫu trở lại 2x, tắt AO (nhanh lại); vẫn có normal map và ánh sáng tuyến tính.
- **Renderer `mesh` đẹp hơn, gần bản xem của Tripo**: bản nhập giữ normal map và metallic/roughness map của model gốc (`profiles/VN/assets/beta.glb` ~31 MB, phải `import-model` lại); renderer dùng chúng (túi áo, đường may, nếp vải), ánh sáng tuyến tính (sRGB → tuyến tính), Lambert + GGX, đèn phụ, môi trường trời/đất, ambient occlusion màn hình, rim nhẹ, mip-map cho texture 4k, SUPERSAMPLE 2 → 3. Chậm hơn khoảng 2–3 lần (toàn bộ test ~260 s).
- **Sửa lỗi shading**: normal skin trong khung bind của model (bind pose A-pose không phải rest) — trước đó pháp tuyến cánh tay bị xoay sai nên hình khối khác Tripo.
- Cử chỉ `VN_A_POSE` (weight 0, không phải hiệu lệnh) + pose `a_pose` = tư thế A của model gốc (bind pose của `beta`): 15 cử chỉ, 24 pose.
- Studio: tắt "face keypoints" giờ ẩn cả các điểm mặt (mũi, mắt, tai) trong khung camera (trước đó chỉ ẩn ở bản đồ mặt bằng/nghiêng).

## v2.5 — 2026-10-15

Từ v2 (commit `d68848d` → `aa60da2`) đến v2.5: model 3D Tripo thật chạy tốt (skeleton riêng, giải phẫu đúng, tay không lỗi), baton có chỗ của nó, camera có ống kính thật, CLI và Studio chắc tay hơn, 215 test.

### Tính năng mới

**Model 3D và skeleton Tripo — D28, D29, D31, D32, D34**
- Skeleton `characters/tripo_v1` dựng bằng `tools/build_tripo_skeleton.py` từ rig của model Tripo: cùng DOF, giới hạn, capsule, keypoint như `proxy_v1`, nhưng vị trí khớp theo rig (importer không còn phải kéo đỉnh lưới 4–14 cm, hết nhàu ngón và méo khuỷu); keypoint mũi/mắt/tai và capsule đầu/mũ đo trên mesh (`import_mesh.head_landmarks`).
- Bind pose khớp thêm roll cẳng tay (trước đó lòng bàn tay lệch ~75°); normal mesh tính sau khi gộp đỉnh trùng vị trí (hết cạnh gãy ở đường nối UV).
- Hiệu chỉnh trong `.fit.json`: `shoulder_out_cm`, `wrist_drop_cm`, `forearm_scale`, `belt_baton`, `close_hand_holes`, `strip_stray_arm_weights`. Model hiện tại `beta` (Tripo, 22 296 đỉnh) là model duy nhất; các model cũ đã xoá.
- Sửa mesh `beta`: lấp khe hở ở găng tay phải (nắm tay mở ra thành nêm trong suốt) và bỏ trọng số tay khỏi bao súng/thắt lưng (giơ tay phải xé bao súng).
- `import-model --no-activate`; model `--character` mặc định `tripo_v1`; model không được mang baton riêng.

**Baton luôn là prop — D18, D21, D33**
- Bật: cầm tay (tay đóng quanh baton, hai kiểu cầm `across`/`along`). Tắt: treo ở thắt lưng, gắn cứng vào `pelvis`, vị trí theo fit `belt_baton`. Một ảnh luôn một baton.
- Bỏ cơ chế biến thể "không baton" (`variant_of`, `baton_model`, schema uniforms 1.3, cờ `import-model --baton-model/--variant-of`, nút upload của Studio).

**Cử chỉ và pose — D16, D17**
- Tổng 14 cử chỉ (thêm `VN_STOP_FRONT`, `VN_ARM_VERTICAL`, `VN_ARMS_HORIZONTAL`, `VN_RIGHT_ARM_FORWARD`, `VN_HELLO`, `VN_THUMB_UP`), 23 pose thân, 5 pose bàn tay. HELLO và THUMB_UP có `weight: 0` (chỉ sinh khi config chọn), được phân biệt bằng hình bàn tay.
- Chủ thể được phép ra ngoài khung hình (metadata có `bbox.full_xyxy`, `truncated`).

**Camera và ảnh — D26, D27**
- Năm loại camera (thêm `motorbike_rider`), ống kính Brown k1/k2 và viền đen theo từng loại; keypoint/bbox/baton nằm trong ảnh đã méo; nền cũng méo cùng officer (`src/tosdg/lens.py`). Metadata 1.6 (`camera.lens`).
- Nhập ảnh PNG/JPG/WebP/BMP/GIF/TIFF (`imageio.py`), xuất `image_format` png/jpg; dò đường chân trời tự động (`horizon.py`).

**Bục giao thông — D23**: `podium.probability` trong config, ô *Standing on a podium* trong Studio, `pose-preview --podium`; gờ bục dựng lại thành vòng sạch.

**Studio**: gizmo xoay khớp (Alt + nhấp, D22); khung camera có model 3D trực tiếp; cổng độc quyền và khoá workspace; `start_studio.bat` đóng Studio đang chạy rồi mở bản mới (`tools/stop_studio.ps1`); tự dừng khi đóng tab (D25); Render preview tự cuộn tới ảnh; nút "⋯" cho ảnh nền; chấm ngón tay chỉ bắt chuột khi bàn tay đủ lớn.

### Sửa lỗi (đợt kiểm thử "khách hàng khó tính" 2026-10-14 — D30)
- `generate` từ chối thư mục `--out` không rỗng (trước đó ghi đè một phần và trộn dataset); `--overwrite` chỉ xoá file của dataset.
- Tham số CLI (`--num`, `--seed`, `--horizon`), config (`[min, max]` ngược, backend chưa có, thư viện nền không tồn tại) được kiểm tra và báo một dòng, mã thoát 2 (`TOSDG_DEBUG=1` để xem traceback).
- `reproduce` kiểm tra file trên đĩa trước khi sinh lại (dataset bị sửa không còn báo "identical"), thêm `--background-library`.
- Manifest **1.1** và `index.json` dùng dấu `/` và đường dẫn tương đối với repo (không lộ tên người dùng, mở được trên Linux). `backgrounds import --id`.
- Pose `slow_down_l/r` hợp lệ trên `tripo_v1`; HELLO và THUMB_UP không còn khớp cùng một tư thế; API render từ chối góc ngoài ±360°.

### Schema
- `sample_metadata`: 1.4 → 1.6 (podium, định dạng ảnh, ống kính). `manifest`: 1.0 → 1.1. `uniforms`: 1.1 → 1.3 (bỏ `variant_of`, `baton_model`). `gestures`: 1.2 (`weight`). `generation_config`: thêm `podium`, `image_format`. `camera_profiles`: thêm khối `lens`.

### Kiểm thử
- 124 → 215 test, ~65 giây: giải phẫu model, khe hở găng tay, baton cầm/đeo, CLI và vệ sinh dataset, cổng/khoá Studio, tái lập từng byte.
- Tối ưu thời gian: các test phát hiện biên chạy bằng backend proxy, bớt góc nhìn thừa, gộp test trùng; cả bộ test từ ~205 giây xuống ~65–135 giây (tuỳ tải máy) dù thêm ~90 test.

### Đã biết, chưa làm
- Ảnh còn nhỏ và thiếu chân thực: bàn tay rộng ~6 px ở 640×360, 35 % mẫu có > 10 % viền đen, không có bóng, nền mặc định không phải đường phố Việt Nam, chỉ một model.
- Renderer Unreal chưa có (backend `mesh` là phần mềm numpy, không chân thực). Baton đeo thắt lưng chưa ghi riêng trong metadata.

## v2 (tag `v2`, commit `aa60da2`)

Tổng hợp các thay đổi từ v1 (commit `d68848d`) đến v2. Bỏ qua các chỉnh nhỏ (chữ, căn lề, đổi tên biến).
Chi tiết thiết kế: `docs/DECISIONS.md` (D23, D24, D25).

## Tính năng mới

### Bục giao thông (podium) — D23
- Module mới `src/tosdg/podium.py`; model 3D `profiles/VN/props/podium.glb`.
- Bục được gộp vào mesh người, skin 100 % vào xương `root` nên đứng yên khi đổi pose. Officer đứng trên bục (cao thêm ≈ 43 cm).
- Config sinh dữ liệu: `podium: {"probability": p}` (chỉ backend `mesh`); luồng RNG mới `podium`, chỉ bốc khi bật nên các luồng khác không đổi.
- CLI `pose-preview --podium`.
- Instance mask/bbox chỉ gồm officer (`UniformMesh.instance_tris`); bục che chân thì mask mất phần bị che.
- Metadata schema **1.3 → 1.4**: thêm `officer.podium` (`enabled`, `diameter_cm`, `depth_cm`, `rim_cm`); `officer.world_position[2]` là chiều cao bục.
- Chiều cao camera / điểm ngắm vẫn tính từ mặt đất (`sample_camera` nhận vị trí với z = 0) nên camera cùng seed không đổi.
- `podium.glb` chỉ nằm trong `assets_sha256` của dataset có dùng bục.
- Studio: checkbox *Standing on a podium* (panel Officer model); `officer.z` do server quyết định; `core.ground_target(cam, uv, z)` kéo officer trên mặt bục.
- Test: `tests/test_podium.py`.

### Tự tắt khi đóng tab — D25
- `tosdg studio --exit-on-close`: server dừng khi tab trình duyệt cuối cùng đóng.
- Lớp `Presence` (server.py): trang ping mỗi vài giây (`/api/ping`), báo `/api/bye` khi đóng; refresh không tính là rời đi (thời gian chờ 4 s), trang im lặng 90 s coi như đã mất.
- `start_studio.bat` dùng `--exit-on-close`; sau khi studio dừng hỏi **S** (tắt) / **R** (chạy lại), đếm ngược 5 s trên một dòng: mỗi giây xoá dòng cũ (mã ANSI) rồi in dòng mới.

## Thay đổi hành vi

### Bỏ làm mờ mặt trong Studio — D24
- Xoá `src/tosdg/studio/privacy.py`, `tests/test_studio_privacy.py`, API `/api/uniform_blur`, tham số upload `blur` / `fallback`, huy hiệu privacy và trình sửa vùng mờ.
- Upload chỉ còn: kiểm tra ảnh, xoay thẳng theo EXIF (`ImageOps.exif_transpose`), bỏ metadata, mã hoá lại. Chỉ upload ảnh bạn được phép dùng.

### Camera mặc định của Studio
- Khoảng cách 800 → 1600 cm, độ cao 140 → 132.5 cm, độ cao điểm ngắm 100 → 120 cm.
- Tiêu cự 35 → 3.6 mm, sensor 36 → 6.17 mm (camera giám sát nhỏ).

### Cache mesh của Studio
- Khi một file model đổi, chỉ bỏ bản cũ của file đó, giữ các model khác trong cache.

## Sửa lỗi Studio
- Nền (backplate) không tự hiện khi mở: yêu cầu ảnh nền chạy sau 120 ms, trước khi eval đầu tiên xong (`S.ev` còn null) nên bỏ qua và không thử lại. Nay lên lịch lại ngay sau eval đầu tiên.
- Mọi panel bên cạnh (officer, camera, background, pose, model, render, FOV, check, gesture, joint, warnings) luôn thu gọn khi khởi động; không còn nhớ trạng thái mở/đóng trong `localStorage`.

## Schema
- `sample_metadata.schema.json`: 1.4 (podium); bỏ trường `privacy` của ảnh.
- `generation_config.schema.json`: thêm `podium`.

## Dọn mã
- Xoá mã không dùng: `baton.mode_of`, `import_mesh.write_import`, `Skeleton.clamp`, `uniform_mesh._matrix_from_quat`, các import/biến thừa (kể cả trong test).
- `baton.prop_path` có `Path` import đúng chỗ (trước đây annotation tham chiếu tên chưa định nghĩa).
- Các hằng số nằm giữa file được đưa lên khối hằng số ở đầu module (`baton`, `cli`, `import_mesh`, `workspace`, `uniform_mesh`).
- Mỗi file mã (src, tools, tests, `index.html`) có mục lục (`Contents:`) ở đầu file.

## Cài đặt thư viện
- `install_requirements.bat`: kiểm tra Python ≥ 3.11 và phiên bản numpy, pillow, jsonschema, opencv-python-headless; cài hoặc nâng cấp gói thiếu/cũ (`quiet` = không pause). `start_studio.bat` gọi nó khi import gói lỗi.
- OpenCV vẫn giữ (extra `studio`) để cắt nền ảnh đồng phục bằng GrabCut; mô tả cũ về nhận diện mặt trong README/pyproject đã bỏ.

## Định dạng ảnh nhập/xuất — D26
- Nhập (Studio + `backgrounds import`): PNG, JPG/JPEG, WebP, BMP, GIF, TIFF; WebP/BMP/GIF/TIFF tự lưu thành JPG (có trong suốt thì PNG). Module mới `src/tosdg/imageio.py`.
- Xuất: config `image_format` = `png` | `jpg` và `generate --image-format`; mask luôn PNG. Metadata schema 1.4 → **1.5** (`image.format`).
- Studio: nút ⬇ PNG / ⬇ JPG ở Render preview.
- Test: `tests/test_imageio.py`.

## Posture và cử chỉ mới: hello, thumb_up
- Pose thân `hello_r` / `hello_l` (giơ tay vẫy chào, lòng bàn tay mở hướng ra trước) và `thumb_up_r` / `thumb_up_l` (gập khuỷu đưa tay lên, giơ ngón cái); pose bàn tay `thumb_up` (nắm tay, ngón cái duỗi lên) trong `poses/`. Xem thử: `python -m tosdg pose-preview hello_r --hand-r open_spread`, `pose-preview thumb_up_r --hand-r thumb_up`; trong Studio chọn ở panel Pose.
- Cử chỉ `VN_HELLO`, `VN_THUMB_UP` (mỗi cử chỉ có biến thể `right` / `left`, `signalling: false`, mọi hướng `NONE`) vì quy tắc "mỗi pose phải có cử chỉ dùng nó".
- Cử chỉ có thêm trường tuỳ chọn `weight` (schema gestures 1.2): hai cử chỉ mới có `weight: 0` nên **không bị bốc theo mặc định** (mọi seed cũ cho ảnh giống hệt, đã kiểm). Muốn sinh chúng: `"gesture_weights": {"VN_HELLO": 1, "VN_THUMB_UP": 1}` trong config.
