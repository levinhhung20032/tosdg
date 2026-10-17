# Kiến trúc

## Tổng quan

```
GenerationConfig ─┐
Profile (VN/…) ───┤                  ┌─────────── backend ───────────┐
Skeleton nhân vật ┼─► build_scene ──►│ proxy  (capsule, để test)     │──► rgb + instance mask
Thư viện pose ────┤   (theo seed)    │ mesh   (GLB có skin, phần mềm)│
                  │                  │ unreal (sản phẩm, dự kiến)    │
Backplate ────────┘                  └───────────────────────────────┘
                          │                                 │
                          ▼                                 ▼
                    Scene (dữ liệu thuần) ───► ground truth ◄── mask
                                              │
                                     validate ─┴─► ghi mẫu  /  bốc lại
```

**Phần lõi** (Python, `src/tosdg`) giữ mọi thứ phải độc lập với engine:
schema dữ liệu, profile quốc gia, ngữ nghĩa cử chỉ, lấy mẫu pose có ràng buộc,
seed, mô hình camera, suy ra ground truth, kiểm tra hợp lệ, đọc/ghi dataset.
Một **render backend** chỉ biến `Scene` thành điểm ảnh (+ instance mask). Cách chia này giữ kiến thức
về cử chỉ/quốc gia ra khỏi code engine, và cho phép kiểm tra mọi backend bằng cùng một phép chiếu.

## Mô hình dữ liệu

| Khái niệm | Nằm ở | Ghi chú |
|---|---|---|
| Profile quốc gia | `profiles/<CC>/profile.json` | cơ quan chức năng, nguồn văn bản luật, bộ từ vựng chỉ dẫn |
| Cử chỉ (gesture) | `profiles/<CC>/gestures.json` | id/nhãn ngữ nghĩa, chỉ dẫn theo từng hướng tiếp cận, biến thể pose, biến thể bàn tay, khoảng biến thiên, ràng buộc |
| Pose | `poses/body/*.json`, `poses/hands/*.json` | chỉ là cấu hình vật lý; pose bàn tay không phân trái/phải và áp dụng cho `_l`/`_r` |
| Uniform (model người) | `profiles/<CC>/uniforms.json` | model người do người dùng upload: `mesh` = GLB có skin + file fit; `proxy_colors` chỉ dùng cho backend capsule; tham chiếu bộ material Unreal |
| Model nguồn | `profiles/<CC>/models/<id>.glb` | file GLB gốc người dùng upload (có texture) |
| Asset / fit | `profiles/<CC>/assets/<id>.glb` / `.fit.json` | model đã đặt lên skeleton dự án (dùng skeleton + skin có sẵn nếu có, nếu không thì gán xương) và các thiết lập nhập (chiều cao, ngưỡng màu da, `joint_map`, `bind_pose`, hiệu chỉnh giải phẫu `shoulder_out_cm`, `wrist_drop_cm`, `forearm_scale`, vị trí baton `belt_baton`, và hai bước sửa mesh `close_hand_holes`, `strip_stray_arm_weights`; D31, D34) |
| Camera profile | `profiles/<CC>/camera_profiles.json` | các khoảng bị ràng buộc; `observer_role` = road_user / infrastructure |
| Nhân vật | `characters/<id>/skeleton.json` | các khớp (tên theo UE5 Mannequin), DOF giải phẫu + giới hạn, capsule va chạm, các bộ keypoint. `proxy_v1` là chuẩn; `tripo_v1` giữ DOF/giới hạn/capsule nhưng lấy vị trí khớp từ rig của model Tripo và keypoint mặt đo trên mesh (`tools/build_tripo_skeleton.py`, D29) |
| Prop | `profiles/<CC>/props/baton.glb`, `podium.glb` | không thuộc model người; renderer thêm vào mesh. Baton: cầm tay (gắn vào khớp tay) hoặc treo ở thắt lưng (gắn vào `pelvis`, vị trí trong fit `belt_baton`); bục gắn vào `root` (D23, D33) |
| Backplate | `<library>/index.json` | sha256, tag, hàng chân trời, người / xe tham chiếu, bố cục giao lộ, mask vùng đứng được `images/<id>.ground.png` (thư viện 1.1, `src/tosdg/ground.py`, D37) |

Mọi file đều mang `schema_version` và được kiểm tra theo `schemas/*.schema.json`. Phiên bản hiện tại: metadata 1.9, generation config 1.7, thư viện backplate 1.3, manifest 1.1, uniforms 1.3, gestures 1.2.

## Quy ước

- **Hệ toạ độ thế giới:** Unreal — X tiến, Y phải, Z lên, đơn vị cm. Nhân vật hướng +X khi yaw bằng 0.
- **Xoay khớp:** mỗi khớp liệt kê các DOF giải phẫu `{name, axis, sign, min, max}`;
  R cục bộ = Π R(axis, sign·độ) theo thứ tự liệt kê; tư thế nghỉ = ma trận đơn vị. Dấu trái/phải được
  lật đối xứng, nên ví dụ `abduction` luôn đưa tay ra xa thân.
- **Camera:** lỗ kim, pixel vuông, điểm chính ở tâm ảnh; thêm méo ống kính xuyên tâm tuỳ chọn (Brown k1, k2) và vùng đen ngoài vòng tròn ảnh, xem D27 (`lens.py`).
  Góc quay = UE `FRotator` → `FRotationMatrix` (đã test với công thức đóng).
  `fx = focal_mm · W / sensor_width_mm`. Pixel (i,j) phủ [i,i+1)×[j,j+1).
- **Azimuth:** vị trí camera tính trong hệ của viên cảnh sát; 0 = phía trước, +90 = bên phải của cảnh sát.

## Quy trình lấy mẫu pose

```
cử chỉ → biến thể pose có trọng số → pose thân cơ sở
       → biến thể bàn tay có trọng số cho từng bên → pose bàn tay
       → độ lệch đều có giới hạn theo từng DOF (chung + cử chỉ + biến thể)
       → loại nếu: vượt giới hạn khớp | capsule tự va chạm | ràng buộc cử chỉ không đạt
       → bốc lại (cùng luồng RNG) đến khi hợp lệ hoặc hết max_pose_tries
```
Các ràng buộc được tính trên hình học FK (hướng chi, hướng lòng bàn tay, trên/dưới,
độ cong ngón), nên mọi pose thoả chúng đều là một cách thể hiện hợp lệ của cử chỉ.

## Ground truth

- Keypoint 2D: điểm khớp/ổ 3D (FK, toạ độ thế giới) → phép chiếu camera. Các bộ: `coco17`, `hand21_l`, `hand21_r`. Keypoint 3D thế giới cũng được xuất.
- Khả năng nhìn thấy: `in_frame` và `occluded` (tự che khuất bởi capsule cơ thể / gậy, D41); chưa có che khuất bởi vật trong ảnh nền.
- BBox: phạm vi sát nhất của instance mask của viên cảnh sát do backend trả về (`source: instance_mask`).
- Chỉ dẫn giao thông: phía của viên cảnh sát mà camera đứng, và lệnh của cử chỉ cho phía đó (`instruction_for_observer`).
  Không chú thích đèn / biển báo trên ảnh nền (D44).

## Kiểm tra hợp lệ (từng mẫu, trước khi ghi)

kiểm tra ảnh render (kích thước, không đồng nhất, mask không rỗng) → bbox (hợp lệ, chiều cao tối thiểu, không bị cắt) →
keypoint (phía trước camera, trong khung hình, trong bbox của mask ⇒ phép chiếu và raster khớp nhau,
tay ra hiệu nhìn thấy được) → schema metadata + id/tên file nhất quán.
Mức dataset (`tosdg validate`): schema manifest, sự có mặt của file, sha256, kích thước ảnh, diện tích mask, không có file thừa.

## Khả năng tái lập

`sample_seed = H(master_seed, index)`; lần thử k dùng `H(sample_seed, k)`; mỗi hệ con
bốc từ `stream(seed, name)`. Manifest lưu config, hash của config, hash của từng asset, các seed.
`tosdg reproduce` sinh lại và so hash file (giống từng byte trên backend proxy và mesh).

## Backend mesh (phần mềm, chạy được ngay)

`import_mesh.py` / `models.py` nhập model người (GLB có texture): nếu model đã có skeleton + skin với tên xương khớp skeleton dự án
(xương `*_twist_NN_x` được gộp vào xương gốc, có thể ghi đè bằng `fit.joint_map`) thì giữ nguyên skin weights của nó, chỉ khớp tư thế bind (DOF của tay/chân
sao cho skeleton dự án vào đúng tư thế nghỉ của model) và dịch các đỉnh cùng khớp của chúng; nếu không có skeleton thì gán xương theo khoảng cách tới xương.
`uniform_mesh.py` đọc/ghi GLB (glTF 2.0, +Y lên, mặt trước +Z, đơn vị mét; phép đổi chiều tay từ Unreal cũng đảo thứ tự đỉnh của tam giác) và vẫn
chứa bộ dựng mesh cũ từ spec trang phục (mã thư viện, không còn dữ liệu đi kèm trong v1).
Sau khi nhập, importer còn sửa ba điều mà auto-rig của Tripo làm sai (D31, D34): điểm xoay vai/cổ tay/cẳng tay (chỉ dời điểm xoay, lưới giữ nguyên, hoặc co cẳng tay có chủ ý), khe hở trên găng tay (lấp bằng miếng vá cùng một texel), và phần trọng số cẳng tay chia cho bao súng/thắt lưng (bỏ). Prop (baton, bục) được thêm vào mesh lúc vẽ bằng `baton.mesh_with_officer_baton` và `podium.mesh_with_podium`. `render_mesh.py` tạo dáng mesh bằng
linear blend skinning từ cùng FK với ground truth, chiếu bằng `Camera.project`, raster bằng z-buffer chính xác
(numpy, siêu lấy mẫu 2x) và tô bóng bằng mặt trời + ánh sáng môi trường (CPU). `render_gl.py` (config `mesh_renderer: gpu`, D35) vẽ cùng mesh đã skin bằng OpenGL trên GPU: PBR với normal/roughness map của model, bóng nắng, ánh sáng trời/đất, SSAO, bóng trên mặt đường; `mesh_renderer.py` chọn một trong hai. Instance mask là phạm vi phủ chính xác
của mesh. Độ sáng vải (`uniform_brightness_range`) được bốc theo từng mẫu từ luồng RNG riêng và ghi vào
metadata; chỉ texel không phải da được nhân hệ số này. Đây là model 3D thật nhưng không phải ảnh render chân thực: không có bóng hay phản xạ, và không phải backend Unreal.

## Backend Unreal (dự kiến)

Dùng cùng `Scene` (xuất ra JSON) và có thể nhập GLB của model (cùng skeleton, tư thế bind lưu trong file): đặt góc xoay xương của skeletal mesh từ pose
(qua ánh xạ DOF→không gian xương cho từng nhân vật), bộ material, backplate là một
mặt phẳng căn theo camera/lớp composure, ánh sáng định hướng + bầu trời từ `lighting`, CineCamera từ `camera`
(tiêu cự, filmback = cảm biến). Render RGB bằng Movie Render Queue và instance mask của viên cảnh sát
bằng pass Custom Depth/Stencil. Nó phải trả lại biến đổi thế giới của các xương để phần lõi
kiểm tra phép chiếu keypoint của mình so với engine (test có dung sai).

## Bóng đổ (D46, `shadows.py`)

Ảnh 2D không cho biết mặt trời ở đâu, nên hướng nắng đến từ dữ liệu: (1) bóng đánh dấu tay hoặc tự dò của người / xe trong ảnh nền
(thư viện, khối `shadows`, vector chân → đỉnh bóng theo pixel ảnh, tính một lần khi nhập — `generate` không chạy thuật toán dò), chiếu
ngược lên mặt đất qua camera ảo của mẫu; (2) vector tự chỉnh / ngẫu nhiên so với hướng nhìn của camera. Bóng được vẽ bằng cách trượt mọi
đỉnh của mesh đã skin theo hướng mặt trời xuống z = 0 và raster qua camera pinhole (`shadows.cast`), rồi làm tối ảnh nền
(`shadows.apply`) trước khi renderer vẽ viên cảnh sát — độc lập renderer, không thuộc mask. Ánh nắng chiếu viên cảnh sát dùng cùng
mặt trời; ảnh không có nắng chỉ có bóng tiếp xúc và ánh sáng phẳng.
