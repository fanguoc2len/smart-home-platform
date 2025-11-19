# sdsa_mediapipe_fixed.py
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time
import os
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# Cấu hình (thay đổi nếu cần)
# -------------------------
SAMPLE_IMAGE = "namtran.jpg"       # ảnh gốc (hoặc crop) để làm mẫu
SAMPLE_CROP = "namtran_crop.jpg"   # ảnh crop sẽ được lưu khi nhấn S
SIM_THRESHOLD = 0.67               # ngưỡng cosine similarity (0..1)
INPUT_W, INPUT_H = 224, 224        # kích thước input cho MobileNetV2

# -------------------------
# GPU config
# -------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print("✅ GPU đang sử dụng:", gpus[0].name)
    except RuntimeError as e:
        print("⚠️ Lỗi GPU:", e)
else:
    print("⚠️ Không tìm thấy GPU — chạy CPU")

# -------------------------
# Tạo model embedding (MobileNetV2 pretrained)
# -------------------------
base_model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg",
                         input_shape=(INPUT_H, INPUT_W, 3))
model = Model(inputs=base_model.input, outputs=base_model.output)

@tf.function
def get_embedding_batch(imgs):
    emb = model(imgs, training=False)
    emb = tf.nn.l2_normalize(emb, axis=1)
    return emb

def get_embedding_from_bgr(bgr_img):
    """
    input: BGR image (face crop)
    output: 1D numpy embedding (L2-normalized)
    """
    if bgr_img is None or bgr_img.size == 0:
        raise ValueError("Empty image passed to get_embedding_from_bgr")
    img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_W, INPUT_H))
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    emb = get_embedding_batch(arr)
    return emb.numpy().flatten()

# -------------------------
# Mediapipe face detection
# -------------------------
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)
print("✅ MediaPipe ready")

# -------------------------
# Helpers: load known face (crop by mediapipe)
# -------------------------
known_embeddings = []
known_names = []

def load_known_face_from_file(path, name="Unknown"):
    """
    Load an image file, detect first face using MediaPipe, crop it and compute embedding.
    Returns True if successful.
    """
    if not os.path.exists(path):
        print(f"⚠️ File {path} không tồn tại.")
        return False
    img = cv2.imread(path)
    if img is None:
        print(f"⚠️ Không đọc được {path}")
        return False
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = face_detection.process(rgb)
    if not res.detections:
        print(f"⚠️ Không phát hiện khuôn mặt trong {path}")
        return False
    det = res.detections[0]
    bbox = det.location_data.relative_bounding_box
    x = int(bbox.xmin * w)
    y = int(bbox.ymin * h)
    bw = int(bbox.width * w)
    bh = int(bbox.height * h)
    x, y = max(0, x), max(0, y)
    bw, bh = max(1, bw), max(1, bh)
    x2 = min(w, x + bw)
    y2 = min(h, y + bh)
    face_crop = img[y:y2, x:x2]
    if face_crop.size == 0:
        print("⚠️ Crop ra ảnh rỗng")
        return False
    emb = get_embedding_from_bgr(face_crop)
    known_embeddings.append(emb)
    known_names.append(name)
    print(f"✅ Load sample '{name}' thành công (crop size: {face_crop.shape[:2]})")
    # optionally save crop for debugging
    cv2.imwrite(SAMPLE_CROP, face_crop)
    return True

# -------------------------
# Load sample nếu có
# -------------------------
if os.path.exists(SAMPLE_IMAGE):
    ok = load_known_face_from_file(SAMPLE_IMAGE, "Nam Trần")
    if not ok:
        print("⚠️ Ảnh mẫu tồn tại nhưng không thể tạo embedding.")
else:
    print("⚠️ Chưa có ảnh mẫu. Nhấn 's' khi có khuôn mặt để lưu mẫu.")

# -------------------------
# Dò camera (nếu cần)
# -------------------------
def find_camera_index(max_checks=5):
    for i in range(max_checks):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.release()
            return i
        cap.release()
    return 0

cam_index = find_camera_index()
print(f"🎥 Sử dụng camera index: {cam_index}")

cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# -------------------------
# Main loop
# -------------------------
fps_time = time.time()
frame_count = 0
last_face_crop = None   # lưu crop cuối cùng (dùng để lưu mẫu khi nhấn 's')

print("🎥 Nhấn 'q' để thoát | Nhấn 's' để lưu mẫu (crop) | Ngưỡng similarity:", SIM_THRESHOLD)

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Không đọc được frame từ camera.")
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)

    # nếu có detections, xử lý từng khuôn mặt
    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)
            x, y = max(0, x), max(0, y)
            x2 = min(w, x + bw)
            y2 = min(h, y + bh)
            face_crop = frame[y:y2, x:x2]

            # lưu crop cuối để khi nhấn 's' có dữ liệu
            last_face_crop = face_crop.copy() if face_crop is not None and face_crop.size else None

            if face_crop is None or face_crop.size == 0:
                continue
            # lấy embedding từ crop
            try:
                emb = get_embedding_from_bgr(face_crop)
            except Exception as e:
                # tránh crash nếu có input lạ
                print("⚠️ Lỗi khi lấy embedding:", e)
                continue

            # so sánh với known embeddings
            best_name = "Unknown"
            best_sim = -1.0
            for k_emb, k_name in zip(known_embeddings, known_names):
                sim = float(cosine_similarity([emb], [k_emb])[0][0])
                if sim > best_sim:
                    best_sim = sim
                    best_name = k_name if sim >= SIM_THRESHOLD else "Unknown"

            color = (0, 255, 0) if best_name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
            cv2.putText(frame, f"{best_name} ({best_sim:.2f})", (x, max(15, y-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    else:
        last_face_crop = None

    # tính FPS
    frame_count += 1
    if frame_count % 10 == 0:
        fps = 10 / (time.time() - fps_time)
        fps_time = time.time()
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 2)

    cv2.imshow("Face Recognition (MediaPipe + MobileNetV2)", frame)

    # chỉ gọi waitKey 1 lần ở cuối loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('s'):
        # lưu sample từ last_face_crop (nếu có)
        if last_face_crop is None or last_face_crop.size == 0:
            print("⚠️ Chưa có khuôn mặt để lưu. Đưa mặt vào khung rồi nhấn 's' nhé.")
        else:
            # lưu ảnh crop và ảnh gốc
            cv2.imwrite(SAMPLE_CROP, last_face_crop)
            cv2.imwrite(SAMPLE_IMAGE, last_face_crop)  # ghi sample gốc bằng crop để đơn giản
            # cập nhật embedding mới
            known_embeddings.clear()
            known_names.clear()
            try:
                emb_new = get_embedding_from_bgr(last_face_crop)
                known_embeddings.append(emb_new)
                known_names.append("Nam Trần")
                print("💾 Đã lưu sample và cập nhật embedding.")
            except Exception as e:
                print("⚠️ Lỗi khi tạo embedding cho sample mới:", e)

# cleanup
cap.release()
cv2.destroyAllWindows()
