# Trạng thái dự án

**Phiên bản:** v3.0 (2026-10-15). **Giai đoạn:** MVP — phần lõi độc lập với engine đã hoàn thành và kiểm chứng trên backend proxy (capsule) và backend mesh (model 3D có skin, phần mềm).
Backend Unreal chưa bắt đầu (môi trường build hiện không có Unreal Engine hay Blender).
**Cập nhật lần cuối:** 2026-10-15. Lịch sử thay đổi: `docs/CHANGELOG.md`; quyết định thiết kế: `docs/DECISIONS.md` (D1–D43).

## Tóm tắt một dòng
Sinh được dataset (ảnh + mask + nhãn keypoint/bbox/cử chỉ/chỉ dẫn giao thông) tái lập được từng byte, với model 3D Tripo đã được chỉnh theo giải phẫu và không còn lỗi hình học thấy được ở các tư thế đã thử; **chất lượng ảnh còn chưa đủ để huấn luyện nhận dạng cử chỉ bàn tay** (người quá nhỏ trong khung, nền không phải đường phố Việt Nam, không có bóng).

## Đã xong (đã kiểm chứng)

**Phần lõi**
- Schema (JSON Schema) cho profile, gesture, pose, uniform, camera profile, thư viện backplate (1.1), generation config, sample metadata (1.7), manifest (1.1).
- Skeleton `proxy_v1` (67 khớp, 5 ngón × 3 đốt mỗi tay, giới hạn DOF giải phẫu, 52 capsule, 45 cặp tự va chạm, keypoint `coco17` + `hand21` trái/phải) và `tripo_v1` (cùng cấu trúc, vị trí khớp theo rig Tripo, keypoint mặt đo trên mesh).
- Lấy mẫu pose có ràng buộc theo seed (cử chỉ ≠ pose; loại bỏ và bốc lại), camera profile (5 loại, có ống kính Brown + viền đen), ánh sáng, nhập/gắn tag/chọn backplate và ghép theo đường chân trời; chân công an luôn ở trên mặt đường/vỉa hè của ảnh nền (mask vùng đứng được dò bằng SegFormer khi nhập; cắt lệch ngang hoặc loại mẫu; D37).
- Ground truth: keypoint 2D/3D, bbox từ instance mask, lệnh của cử chỉ cho phía camera đứng; chủ thể được phép ra ngoài khung (`bbox.full_xyxy`).
- Kiểm tra từng mẫu và cả dataset; sinh hàng loạt; manifest có hash; `reproduce` giống từng byte và kiểm tra cả file trên đĩa; `generate` không trộn vào dataset cũ (`--overwrite`).
- Profile VN: **15 cử chỉ** (kể cả `VN_A_POSE`, weight 0) (đi nhanh/chậm trái-phải, dừng trái-phải, dừng phía trước, rẽ trái, tay thẳng đứng, dang ngang, tay phải đưa trước, không ra hiệu, vẫy chào, giơ ngón cái), 23 pose thân, 5 pose bàn tay, 4 camera profile. HELLO và THUMB_UP có `weight: 0`. Xem README mục 7b, DECISIONS D16.

**Model 3D (D15, D29, D31, D32, D34)**
- **Phạm vi v1:** viên cảnh sát là model 3D do người dùng cung cấp (`tosdg import-model model.glb` hoặc Studio → Officer model (3D) → Upload). Model hiện tại **`beta`** (Tripo, rig UE5, 22 296 đỉnh, 4096² texture) là uniform duy nhất, trên skeleton `tripo_v1`; mọi model cũ đã xoá.
- Importer dùng skeleton/skin có sẵn nếu tên xương UE5 (gộp xương twist), nếu không thì tự gán xương theo khoảng cách. Hiệu chỉnh trong `.fit.json`: `shoulder_out_cm` 4, `wrist_drop_cm` 3, `forearm_scale` 0,9, `belt_baton`, và hai bước tự động bật: lấp khe hở trên găng, bỏ trọng số tay khỏi bao súng/thắt lưng.
- Đã kiểm tra giải phẫu: tỉ lệ cẳng tay/cánh tay 0,84, cổ tay buông ngang thắt lưng (cao hơn khớp hông ~8 cm), keypoint mắt/mũi/tai nằm trên mặt, tay không lún quá 5 cm vào thân, giơ tay không kéo bao súng, nắm tay không còn hở.

**Prop**
- **Baton** (D18, D21, D33): bật thì cầm tay (tay đóng quanh baton, hai kiểu cầm), tắt thì treo ở thắt lưng (fit `belt_baton`); một ảnh luôn một baton.
- **Bục giao thông** (D23): `podium.probability` trong config; Studio có ô bật; gờ bục đã dựng lại sạch.

**Backend mesh** (D12/D14): `import_mesh.py`, `models.py`, `uniform_mesh.py`, `render_mesh.py` (skinning + z-buffer chính xác, siêu lấy mẫu 2×). Đo trên máy thử: 0,9 giây mỗi mẫu 640×360, 2,1 giây ở 1280×720, 4,0 giây ở 1920×1080 (một tiến trình; 100 000 ảnh Full HD ≈ 4,7 ngày máy). Độ sáng vải ngẫu nhiên theo mẫu (`uniform_brightness_range`).

**Renderer GPU** (D35): `render_gl.py` (OpenGL 4.3/moderngl), bật bằng config `mesh_renderer: gpu`, Studio (mặc định khi có GPU) hoặc `pose-preview --renderer gpu`. PBR với normal/roughness map của model, bóng nắng mềm, ánh sáng trời/đất theo ảnh nền, SSAO, bóng trên mặt đường. Vẽ ~0,2–0,3 s/ảnh 1280×720 trên RTX 3060 Laptop; cả mẫu (ống kính, dựng cảnh, ghi file) ~3,7 s.

**Giao lộ** (D43, `junction.py`): `officer_yaw_mode: "junction"` — đứng giữa ngã ba / ngã tư / bùng binh, hướng mặt theo loại giao lộ (T thấy cả 3 nhánh, Y thấy ≥ 2 hoặc về một nhánh, ngã tư / bùng binh về một nhánh) và theo các phía mà cử chỉ ra lệnh; bùng binh có bục và gậy ngẫu nhiên. `match_backplate_light`: model tối theo ảnh nền (ảnh đêm).

**Hiệu chỉnh camera ảnh nền** (D42, `calibrate.py`): độ cao camera + đường chân trời từ người / xe / rào chắn đánh dấu tay, khớp bền vững có kiểm tra độ tin cậy; `backgrounds calibrate`, `backgrounds objects`.

**Studio** (`tosdg studio`, `start_studio.bat`): upload nền 2D (đường chân trời tự dò, kéo tay được), camera (quỹ đạo, ống kính, méo, viền đen) với thước FOV và góc nhìn từ trên/bên, kéo skeleton (IK bình phương tối thiểu có giảm chấn trên FK thật, gizmo xoay Alt + nhấp), Gesture match, Generator check, Render preview (mesh hoặc capsule, ⬇ PNG/JPG), model 3D trực tiếp trong khung camera, baton và bục, lưu pose/cảnh, đổi model. Cổng độc quyền, khoá workspace, `start_studio.bat` đóng bản đang chạy trước khi mở bản mới, tự dừng khi đóng tab. Thư mục làm việc `data/studio/` (git bỏ qua). **Random generate** (D40): sinh một lô với các thông số khoá theo cảnh (cử chỉ, loại camera, góc nhìn, ảnh nền, baton, bục), phần còn lại ngẫu nhiên. Render preview / Generator check của Studio vẫn căn theo đường chân trời đã lưu (chưa dùng camera hiệu chỉnh).

**CLI** (`python -m tosdg`): `generate`, `validate`, `reproduce`, `validate-profile`, `fixtures`, `backgrounds import|ground|people|objects|calibrate|set|remove|check`, `overlay`, `export` (COCO / YOLO-pose / CVAT), `pose-preview`, `import-model`, `studio`. Lỗi đầu vào là một dòng `error: ...` với mã thoát 2 (`TOSDG_DEBUG=1` hiện traceback).

## Kiểm thử
`python -m unittest discover -s tests -t tests` → **321 test, tất cả đều qua** (~135 giây; renderer GPU tự bỏ qua khi máy không có), gồm:
- Smoke E2E: 3 mẫu, đủ file, kiểm tra dataset đạt, sinh lại giống từng byte (cả theo chỉ số), seed khác ⇒ dữ liệu khác.
- Các ca kiểm tra âm: ảnh trống, thiếu chủ thể, bbox diện tích 0/bị cắt, keypoint ngoài bbox, tay ra hiệu ngoài khung, vi phạm schema, tên file không khớp, file bị sửa/thiếu/thừa; chủ thể ra ngoài khung bị loại và bốc lại.
- Camera: góc quay khớp công thức đóng của FRotationMatrix; ống kính, viền đen, remap mask.
- Model: nhập model có/không có skeleton, upload qua Studio, giải phẫu (tỉ lệ chi, mặt, cổ tay, tay-thân, bao súng), khe hở găng tay, baton cầm/đeo, bục.
- CLI và vệ sinh dataset (`tests/test_cli.py`): thư mục đầu ra, tham số, lỗi một dòng, đường dẫn khả chuyển, `reproduce` kiểm tra file, `validate-profile` với skeleton của config.
- Studio: IK, độ khớp phép chiếu với generator, an toàn upload, API HTTP, cổng và khoá workspace.

Đợt kiểm thử thủ công 2026-10-14 ("khách hàng khó tính", báo cáo trong tài liệu riêng): 16 lỗi tìm thấy, 15 đã sửa (D30), 1 (hai cây baton) đã xử lý bằng thiết kế mới (D33). Lô 20 mẫu: 0 lỗi, `validate` và `reproduce` đạt.

## Vấn đề đã biết / hạn chế
**Chất lượng dữ liệu (quan trọng nhất)**
- Cỡ viên cảnh sát: với `size_reference: "auto"` (D42) camera ảo là camera đã chụp ảnh nền, suy từ ≥ 3 người (165 cm), không đủ thì người + xe, cuối cùng đường chân trời; trên 20 ảnh thư viện: 8 từ người, 6 từ người + xe, 6 từ đường chân trời. Giả định mặt đường phẳng và người lớn cao gần như nhau; ảnh có người dồn cùng một dải khoảng cách không hiệu chỉnh được; người ngồi xe máy bị xe khác che bánh thì hàng chân lệch. Config cũ (`horizon`) viên cảnh sát vẫn có thể nhỏ.
- Viền đen do ống kính: 35 % mẫu có > 10 % diện tích đen (dashcam trung bình 10 %, xe máy 9,6 %); cần hiệu chỉnh `k1`, `k2`, vòng ảnh theo ảnh thật.
- Nền: thư viện cục bộ (dùng chung cho Studio và `generate`, D39) **50 ảnh** đường phố Việt Nam (Wikimedia Commons): 10 bùng binh, 17 ngã tư (kể cả ngã sáu), 9 ngã ba chữ T, 1 chữ Y, còn lại đường thẳng / camera hành trình; 4 ảnh đêm, 1 chiều tối; 37 ảnh có bố cục giao lộ đặt tay. Chưa có mưa; chữ Y chỉ 1 ảnh; bố cục giao lộ và đường chân trời đặt bằng mắt (sai số vài %). Mask vùng đứng được không xét che khuất (công an vẫn được vẽ đè lên xe phía trước nếu chân ở sau xe).
- Ảnh ghép chưa thật: không có bóng dưới chân, người nét hơn hẳn nền, ánh sáng ngẫu nhiên không ước lượng từ ảnh nền, không có nhiễu/nén khớp nền.
- Độ đa dạng thấp: một model, một gương mặt, một dáng; không có xe/người đi đường che khuất, áo mưa, áo phản quang.
- Ảnh mesh là ảnh tổng hợp: renderer GPU có PBR, bóng và ánh sáng môi trường nhưng không có chiếu sáng toàn cục giữa người và cảnh, ánh sáng không suy từ ảnh nền thật; cần đối chiếu với ảnh thật/Unreal trước khi huấn luyện (D2/D12, D35). Vệt nâu ở nách khi giơ tay nằm sẵn trong texture gốc. Ảnh capsule (proxy) chỉ để test.

**Model và hình học**
- Rig Tripo đặt khớp ngón dồn về đầu ngón nên keypoint hand21 không đúng vị trí giải phẫu; keypoint vai nằm sâu 6,7–6,9 cm trong áo (D29, D31).
- Nắm chặt: mu bàn tay có nếp gấp góc cạnh (skinning tuyến tính, găng ít đỉnh); đứng nghiêm: vải nách lún vào thân tới ~3,5 cm nhưng nằm khuất (D31, D34).
- Model hiện tại có mặt người, nguồn gốc/bản quyền chưa rõ (Tripo): kiểm tra trước khi chia sẻ dataset hoặc repo.
- Baton đeo thắt lưng chỉ có ở backend `mesh` (capsule chỉ vẽ baton cầm tay) và chưa ghi riêng trong metadata (D33).

**Dữ liệu luật và phạm vi**
- **Dữ liệu luật VN chưa được xác minh** (`verified: false`): ý nghĩa cử chỉ mới đối chiếu với tóm tắt thứ cấp của luật 2008 và infographic LuatVietnam; một số tư thế (đi nhanh/chậm, tay nào dùng cho phía nào) được suy từ hình/mô tả nên cần người có chuyên môn kiểm tra; số điều của luật 2024 cần rà soát.
- Che khuất: có tự che khuất bởi capsule cơ thể (D41), chưa có che khuất bởi vật trong ảnh nền. Méo ống kính chỉ xuyên tâm k1/k2. Bỏ qua roll khi căn chân trời. Trạng thái đèn trên backplate được giả định quay về phía camera.
- Studio: IK không có tiếp xúc mặt đất (kéo bàn chân thì chân di chuyển chứ pelvis không đổi); JSON cảnh của Studio chỉ là bản chụp để soạn thảo, `generate` chưa đọc nó.
- Sinh dữ liệu chạy một tiến trình (chưa có `--workers`).

## Mốc tiếp theo
**Việc nên làm trước (ngắn hạn):**
1. ~~Cho viên cảnh sát to hơn trong khung~~ — đã có `min_officer_height_px` (v3.0); phân phối kích thước và tỉ lệ cử chỉ/camera/ngày-đêm theo yêu cầu để sau.
2. Thu ~200 ảnh ngã tư Việt Nam (dashcam, xe máy, người lái) kèm đường chân trời; hiệu chỉnh ống kính theo ảnh thật.
3. Ghép ảnh thật hơn: bóng tiếp xúc dưới chân theo hướng mặt trời, khớp nhiễu/độ mờ/nén, cân màu theo nền.
4. Ghi baton đeo vào metadata (1.7), sinh song song (`--workers`), cho `reproduce` so sánh với file.
5. Đo trên 100–200 ảnh cảnh sát thật: huấn luyện trên dữ liệu tổng hợp rồi đánh giá trên tập thật, để biết cải tiến nào đáng làm.

**M2 — Lát cắt dọc cho backend Unreal:** project UE + plugin nạp JSON `Scene`, nhập `profiles/VN/assets/beta.glb`, đặt tư thế cho UE5 Mannequin từ các DOF, đặt CineCamera/backplate/ánh sáng, render RGB + stencil mask cho 5 mẫu smoke, và xuất toạ độ thế giới của xương; test ở phần lõi kiểm tra keypoint chiếu của engine khớp với phần lõi trong vòng 1 px. Cần: phiên bản UE, hệ điều hành/GPU, lựa chọn nhân vật. Blender (Cycles/EEVEE) là bước trung gian có thể.

## Ưu tiên đã chốt (2026-10-15, sau danh sách yêu cầu "khách hàng khó tính")
- **Đã làm:** xuất COCO / YOLO-pose / CVAT có chia train/val/test; kích cỡ tối thiểu của viên cảnh sát.
- **Để sau:** đo trên ảnh thật (huấn luyện trên dữ liệu tổng hợp, đánh giá trên ảnh CSGT thật); điều khiển phân bố (tỉ lệ cử chỉ, góc camera, ngày/đêm) và báo cáo thống kê dataset.
- **Không cần làm thêm:** khả năng mở rộng — pipeline theo seed từng mẫu nên chạy song song được, hiện chỉ bị giới hạn phần cứng.
- **Tương lai (thời gian và phần cứng chưa cho phép):** đa dạng model (gương mặt, dáng người, đồng phục hè/đông, áo mưa, ban đêm), xe và người che khuất; backend Unreal, thêm profile quốc gia khác.
- **Chưa ưu tiên:** nhãn che khuất thật, video, pháp lý / xác minh luật, giao diện không cần lập trình.

## Việc làm sau (ngoài v1 — future work)
- **Uniform từ ảnh:** upload ảnh đồng phục rồi "mặc" lên model cơ thể (baking texture / dựng lại 3D, ví dụ photogrammetry như COLMAP/Meshroom hoặc mô hình khớp trang phục). Panel thử nghiệm "Uniform views" 2D đã bị gỡ (D44); làm lại từ đầu khi bắt đầu tính năng này.
- Nhiều model trong cùng một dataset theo trọng số (khuôn mặt, dáng người, đồng phục hè/đông, áo mưa, áo phản quang), xe và người đi đường che khuất kèm ground truth vùng bị che.
- Khớp tỉ lệ theo từng model (co giãn chi theo model thay vì dịch chuyển lên skeleton cố định), keypoint vai theo quy ước COCO.
