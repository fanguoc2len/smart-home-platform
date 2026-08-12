from flask import Flask, request, jsonify, render_template
import torch, whisper
import cv2, numpy as np, base64, os, tensorflow as tf, mediapipe as mp, json, time
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
from pyngrok import ngrok
import time, re, subprocess, tempfile, requests
from pathlib import Path

# ===== NEW: scheduling & helpers (no external deps) =====
import threading, heapq, uuid
from datetime import datetime, timedelta

# ==== MẬT KHẨU ADMIN CHO ĐĂNG KÝ KHUÔN MẶT ====
# ĐỔI "123456" THÀNH MẬT KHẨU RIÊNG, HOẶC SET ENV ADMIN_REGISTER_PASSWORD
ADMIN_REGISTER_PASSWORD = os.environ.get("ADMIN_REGISTER_PASSWORD", "").strip()

# ==== MÃ PIN DỰ PHÒNG ĐĂNG NHẬP (KEYPAD) ====
# Đổi "2580" thành mã PIN bạn muốn, hoặc set biến môi trường SMART_HOME_PIN
SMART_HOME_PIN = os.environ.get("SMART_HOME_PIN", "").strip()

# ============================================
# TELEGRAM BOT CONFIG
# ============================================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

TG_SEND_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
TG_PHOTO_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"


def tg_send(text: str):
    """Gửi tin nhắn văn bản đến Telegram"""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        r = requests.post(
            TG_SEND_URL,
            data={"chat_id": TELEGRAM_CHAT_ID, "text": text},
            timeout=5
        )
        print("📨 Telegram text status:", r.status_code, r.text[:100])
    except Exception as e:
        print("⚠️ Telegram send text error:", e)


def tg_send_photo(image_bytes, caption="Ảnh Unknown"):
    """Gửi ảnh lên Telegram"""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        files = {"photo": ("unknown.jpg", image_bytes)}
        data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption}
        r = requests.post(
            TG_PHOTO_URL,
            data=data,
            files=files,
            timeout=5
        )
        print("📨 Telegram photo status:", r.status_code, r.text[:100])
    except Exception as e:
        print("⚠️ Telegram send photo error:", e)


# Optional: Google Drive upload (pydrive2)
try:
    from pydrive2.auth import GoogleAuth
    from pydrive2.drive import GoogleDrive
    PYDRIVE_AVAILABLE = True
except Exception:
    PYDRIVE_AVAILABLE = False

# ===============================
# MODEL CONFIG
# ===============================
SAMPLE_IMAGE = "namtran.jpg"
SAMPLE_CROP = "namtran_crop.jpg"

# Ngưỡng similarity cho nhận diện (0.67 hơi thấp, tăng lên cho an toàn hơn)
SIM_THRESHOLD = 0.7

INPUT_W, INPUT_H = 224, 224

# Ngưỡng chất lượng khuôn mặt
MIN_FACE_SIZE    = 80    # chiều rộng/ cao tối thiểu (pixels) mới xử lý
BLUR_THRESHOLD   = 50.0  # độ nhòe tối đa (variance of Laplacian)

# Multi-frame xác nhận (theo IP)
REQ_CONSEC_FRAMES = 3     # cần ≥ 3 frame liên tiếp chuẩn mới cho pass (đã biết mặt)
RECOG_WINDOW_SEC  = 3.0   # trong vòng 3s, nếu đổi người coi như chuỗi mới

# file to persist registered users
REGISTERED_DB = "registered.json"
UPLOADS_DIR = "registered_images"
os.makedirs(UPLOADS_DIR, exist_ok=True)

# GPU config
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass
    print("✅ GPU đang dùng:", gpus[0].name)
else:
    print("⚠️ Không có GPU, chạy CPU.")

# Model
base_model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg",
                         input_shape=(INPUT_H, INPUT_W, 3))
model = Model(inputs=base_model.input, outputs=base_model.output)

@tf.function
def get_embedding_batch(imgs):
    emb = model(imgs, training=False)
    emb = tf.nn.l2_normalize(emb, axis=1)
    return emb

def get_embedding_from_bgr(bgr_img):
    if bgr_img is None or bgr_img.size == 0:
        raise ValueError("Empty image passed to get_embedding_from_bgr")
    img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_W, INPUT_H))
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    emb = get_embedding_batch(arr)
    return emb.numpy().flatten()

mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)
print("✅ MediaPipe ready")

# ===============================
# Known faces storage (in-memory)
# ===============================
known_embeddings, known_names = [], []

# State cho multi-frame recognition (key theo IP)
# { ip: { "name": str | None, "count": int, "ts": float, "last_sim": float, "notified": bool } }
RECOG_STATE = {}

# Chống spam Telegram khi gặp Unknown face
# Mỗi IP chỉ gửi tối đa 1 cảnh báo Unknown trong khoảng này
UNKNOWN_ALERT_COOLDOWN_SEC = 60.0  # giây
UNKNOWN_ALERT_STATE = {}           # { ip: last_alert_ts }


def is_blurry(bgr_img, thresh: float = BLUR_THRESHOLD) -> bool:
    """
    Kiểm tra ảnh có quá mờ không (variance of Laplacian).
    Mờ quá thì bỏ qua frame đó, tránh nhầm.
    """
    try:
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        fm = cv2.Laplacian(gray, cv2.CV_64F).var()
        return fm < thresh
    except Exception:
        # nếu lỗi gì đó thì cứ coi như không mờ để tránh drop lung tung
        return False
    

def is_face_occluded(full_img, det, debug=False):
    """
    Ước lượng mặt có bị che mũi/miệng nhiều hay không.
    Dùng keypoints của MediaPipe: 2 = mũi, 3 = miệng.
    Nếu 2 vùng này quá phẳng / ít chi tiết so với toàn mặt → coi như bị che.
    """
    h, w = full_img.shape[:2]

    # bbox khuôn mặt
    bbox = det.location_data.relative_bounding_box
    x = int(bbox.xmin * w)
    y = int(bbox.ymin * h)
    bw = int(bbox.width * w)
    bh = int(bbox.height * h)

    x = max(0, x)
    y = max(0, y)
    x2 = min(w, x + bw)
    y2 = min(h, y + bh)

    face = full_img[y:y2, x:x2]
    if face is None or face.size == 0:
        return False  # không rõ, thôi cho qua

    gray_full = cv2.cvtColor(full_img, cv2.COLOR_BGR2GRAY)
    gray_face = gray_full[y:y2, x:x2]

    std_face = float(gray_face.std())
    mean_face = float(gray_face.mean())

    # nếu cả mặt đã quá phẳng/tối thì bước khác đã loại rồi, ở đây không xử nữa
    if std_face < 8.0 or mean_face < 40.0:
        return False

    # keypoints: 2 = mũi, 3 = miệng
    rel_kps = det.location_data.relative_keypoints
    important_idx = [2, 3]  # mũi + miệng
    occluded_count = 0

    # bán kính patch quanh keypoint (tỉ lệ theo mặt)
    patch_r = int(0.08 * min(bw, bh))
    patch_r = max(patch_r, 6)

    for idx in important_idx:
        if idx >= len(rel_kps):
            continue
        kp = rel_kps[idx]
        cx = int(kp.x * w)
        cy = int(kp.y * h)

        x1 = max(0, cx - patch_r)
        xp2 = min(w, cx + patch_r)
        y1 = max(0, cy - patch_r)
        yp2 = min(h, cy + patch_r)

        patch = gray_full[y1:yp2, x1:xp2]
        if patch is None or patch.size == 0:
            continue

        std_patch = float(patch.std())
        mean_patch = float(patch.mean())

        if debug:
            print(
                f"[OCCLUSION] idx={idx} std_patch={std_patch:.1f}, "
                f"mean_patch={mean_patch:.1f}, std_face={std_face:.1f}"
            )

        # điều kiện: patch quá phẳng so với toàn mặt hoặc bản thân quá phẳng
        if std_patch < 0.25 * std_face or std_patch < 6.0:
            occluded_count += 1

    # nếu cả MŨI + MIỆNG đều "phẳng" → khả năng che khá cao
    return occluded_count >= 2

def load_sample(path, name):
    if not os.path.exists(path):
        print(f"⚠️ load_sample: file not found {path}")
        return False

    img = cv2.imread(path)
    if img is None:
        print("⚠️ load_sample: cv2 couldn't read image")
        return False

    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = face_detection.process(rgb)
    if not res.detections:
        print("⚠️ Không phát hiện khuôn mặt trong ảnh mẫu:", path)
        return False

    # chọn detection lớn nhất
    det = max(
        res.detections,
        key=lambda d: d.location_data.relative_bounding_box.width *
                      d.location_data.relative_bounding_box.height
    )

    # điểm tin cậy của detection
    try:
        score = float(det.score[0])
    except Exception:
        score = 1.0
    if score < 0.7:
        print(f"⚠️ load_sample: score detection thấp ({score:.2f}) – từ chối.")
        return False

    bbox = det.location_data.relative_bounding_box
    x = int(bbox.xmin * w)
    y = int(bbox.ymin * h)
    bw = int(bbox.width * w)
    bh = int(bbox.height * h)

    # kích thước tối thiểu
    if bw < MIN_FACE_SIZE or bh < MIN_FACE_SIZE:
        print(f"⚠️ load_sample: mặt quá nhỏ ({bw}x{bh}px) – từ chối.")
        return False

    x = max(0, x)
    y = max(0, y)
    x2 = min(w, x + bw)
    y2 = min(h, y + bh)
    face_crop = img[y:y2, x:x2]
    if face_crop is None or face_crop.size == 0:
        print("⚠️ load_sample: crop empty")
        return False

    # kiểm tra độ tối / đồng màu (che cam / phòng tối)
    try:
        gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        mean_val = float(gray_face.mean())
        std_val = float(gray_face.std())
        if mean_val < 40.0 or std_val < 5.0:
            print(
                f"⚠️ load_sample: face quá tối/đồng màu "
                f"(mean={mean_val:.1f}, std={std_val:.1f}) – có thể camera bị che, từ chối."
            )
            return False
    except Exception as _e:
        print("ℹ️ load_sample: brightness check error:", _e)

    # kiểm tra mờ
    if is_blurry(face_crop):
        print("⚠️ load_sample: face quá mờ – từ chối.")
        return False

    # kiểm tra bị che mũi/miệng nhiều không
    if is_face_occluded(img, det, debug=False):
        print("⚠️ load_sample: mặt có vẻ bị che mũi/miệng nhiều – từ chối đăng ký.")
        return False

    # Nếu qua được hết các check ở trên → mới lấy embedding và lưu
    emb = get_embedding_from_bgr(face_crop)
    known_embeddings.append(emb)
    known_names.append(name)
    try:
        cv2.imwrite(SAMPLE_CROP, face_crop)
    except Exception:
        pass
    print(
        f"✅ Đã load sample {name} "
        f"(crop size: {face_crop.shape[:2]}, score={score:.2f})"
    )
    return True

# ===============================
# Google Drive helper (optional)
# ===============================
drive = None
gauth = None
DRIVE_AVAILABLE = False

def init_drive():
    global drive, gauth, DRIVE_AVAILABLE
    if not PYDRIVE_AVAILABLE:
        print("ℹ️ pydrive2 không cài — Drive upload sẽ bị tắt.")
        DRIVE_AVAILABLE = False
        return
    creds_path = "client_secrets.json"
    if not os.path.exists(creds_path):
        print("⚠️ client_secrets.json không tìm thấy — Drive upload tạm tắt.")
        DRIVE_AVAILABLE = False
        return
    gauth = GoogleAuth()
    creds_file = "mycreds.txt"
    try:
        gauth.LoadCredentialsFile(creds_file)
        if gauth.credentials is None:
            gauth.LocalWebserverAuth()
            gauth.SaveCredentialsFile(creds_file)
        else:
            gauth.Refresh()
    except Exception:
        try:
            gauth.LocalWebserverAuth()
            gauth.SaveCredentialsFile(creds_file)
        except Exception as e:
            print("⚠️ Drive auth failed:", e)
            DRIVE_AVAILABLE = False
            return
    from pydrive2.drive import GoogleDrive
    drive = GoogleDrive(gauth)
    DRIVE_AVAILABLE = True
    print("✅ Google Drive ready (pydrive2).")

init_drive()

def upload_file_to_drive(local_path, title=None):
    if not DRIVE_AVAILABLE:
        return None
    try:
        file_metadata = {'title': title or os.path.basename(local_path)}
        gfile = drive.CreateFile(file_metadata)
        gfile.SetContentFile(local_path)
        gfile.Upload()
        return gfile.get('id')
    except Exception as e:
        print("⚠️ Upload to Drive failed:", e)
        return None

def download_file_from_drive(file_id, local_path):
    if not DRIVE_AVAILABLE:
        return False
    try:
        f = drive.CreateFile({'id': file_id})
        f.GetContentFile(local_path)
        return True
    except Exception as e:
        print("⚠️ Download from Drive failed:", e)
        return False

# ===============================
# Persistence for registered users
# ===============================
def _load_registered_db_raw():
    if not os.path.exists(REGISTERED_DB):
        return []
    try:
        with open(REGISTERED_DB, 'r', encoding='utf-8') as f:
            db = json.load(f) or []
            if isinstance(db, list):
                return db
            return []
    except Exception as e:
        print("⚠️ Không thể đọc registered.json:", e)
        return []

def _save_registered_db_raw(db):
    try:
        with open(REGISTERED_DB, 'w', encoding='utf-8') as f:
            json.dump(db, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("⚠️ Lỗi lưu registered.json:", e)

def load_registered_db():
    """Load registered.json, download file từ Drive nếu cần, và nạp embedding vào RAM."""
    db = _load_registered_db_raw()
    if not db:
        print("ℹ️ registered.json đang rỗng hoặc không tồn tại.")
        return

    changed = False
    for entry in db:
        name = entry.get('name')
        local_file = entry.get('local_file')
        drive_id = entry.get('drive_id')

        if local_file and os.path.exists(local_file):
            load_sample(local_file, name)
            continue

        if drive_id and DRIVE_AVAILABLE:
            target = os.path.join(UPLOADS_DIR, f"{name.replace(' ','_')}_{drive_id}.jpg")
            ok = download_file_from_drive(drive_id, target)
            if ok:
                entry['local_file'] = target
                changed = True
                load_sample(target, name)

    if changed:
        _save_registered_db_raw(db)

    print("ℹ️ Đã load registered db (n=", len(known_names), ")")

def save_registered_entry(name, local_file, drive_id=None):
    db = _load_registered_db_raw()
    db.append({'name': name, 'local_file': local_file, 'drive_id': drive_id})
    _save_registered_db_raw(db)

def rebuild_known_faces():
    """Xoá toàn bộ embedding trong RAM và load lại từ sample + registered.json."""
    global known_embeddings, known_names
    known_embeddings = []
    known_names = []

    # load sample cố định (nếu có)
    if os.path.exists(SAMPLE_IMAGE):
        load_sample(SAMPLE_IMAGE, "Nam Trần")

    # load từ registered.json
    load_registered_db()
    print("ℹ️ rebuild_known_faces: done, n =", len(known_names))

# Khởi động: rebuild full
rebuild_known_faces()


# ===============================
# VOICE / WHISPER CONFIG
# ===============================
VOICE_LANGUAGE   = "vi"
VOICE_MODEL_NAME = os.environ.get("WHISPER_MODEL", "small")
INITIAL_PROMPT   = (
 "Ngữ cảnh: điều khiển nhà thông minh bằng tiếng Việt. "
    "Các câu lệnh thường dùng: "
    "'bật tất cả đèn phòng khách', "
    "'tắt hết quạt và loa', "
    "'bật chế độ party', "
    "'chuyển sang chế độ relax', "
    "'bật chế độ night', "
    "'tăng độ sáng đèn hành lang lên 50 phần trăm', "
    "'giảm độ sáng đèn bếp xuống 20 phần trăm', "
    "'tắt tất cả thiết bị trong nhà sau 5 phút', "
    "'mở cửa chính', "
    "'đóng cửa bên hông', "
    "'mở khóa cửa chính', "
    "'tắt đèn cửa', "
    "'bật đèn hành lang'. "
    "Tên phòng: phòng khách, bếp, hành lang, cửa chính, cửa bên hông. "
    "Tên chế độ: party, relax, night, focus."   
)

VOICE_EXECUTE_DEFAULT = os.environ.get("VOICE_EXECUTE_DEFAULT", "0") == "1"
VOICE_CONTROL_TOKEN   = os.environ.get("VOICE_CONTROL_TOKEN")

# Một số cụm Whisper hay nghe nhầm → sửa tay trước khi parse
# ============================
# Sửa lỗi thường gặp từ Whisper
# ============================

COMMON_FIXES = {
    # ===== PHÒNG / KHU VỰC =====
    "phòng cách": "phòng khách",
    "phòng khắc": "phòng khách",
    "phòng kháchs": "phòng khách",
    "phong khach": "phòng khách",

    "phòng ngũ": "phòng ngủ",
    "phong ngu": "phòng ngủ",

    "phòng lam việc": "phòng làm việc",
    "phòng làm việt": "phòng làm việc",
    "phong lam viec": "phòng làm việc",

    "phòng học tập": "phòng học",
    "phong hoc": "phòng học",

    # ===== THIẾT BỊ: ĐÈN =====
    "đèn tràng": "đèn trần",
    "đèn trằn": "đèn trần",
    "đèn chần": "đèn trần",
    "den tran": "đèn trần",

    "đèn hắt tường": "đèn hắt",
    "den hat": "đèn hắt",

    "đèn bàn học": "đèn bàn",
    "den ban hoc": "đèn bàn",

    "đèn ngủ nhỏ": "đèn ngủ",
    "den ngu": "đèn ngủ",

    "đèn cây đứng": "đèn cây",
    "den cay": "đèn cây",

    # ===== THIẾT BỊ: QUẠT / LOA =====
    "quạ": "quạt",
    "quạt máy": "quạt",
    "quat may": "quạt",

    "quạt trằng": "quạt trần",
    "quat tran": "quạt trần",

    "loa bluetooth": "loa",
    "loa bluetool": "loa",
    "loa blutút": "loa",

    # ===== CỬA =====
    "cửa chỉnh": "cửa chính",
    "cửa chín": "cửa chính",
    "cửa trước": "cửa chính",

    "cửa hông": "cửa bên hông",
    "cửa bên hong": "cửa bên hông",

    # ===== SCENE / CHẾ ĐỘ =====
    # Party
    "pati": "party",
    "pát ti": "party",
    "pạc ti": "party",
    "parti": "party",
    "party mode": "party",
    "chế độ party": "party",

    # Relax
    "rilát": "relax",
    "rì lắc": "relax",
    "chế độ thư giãn": "relax",
    "chế độ relax": "relax",

    # Focus
    "phô cút": "focus",
    "phô cút mode": "focus",
    "chế độ tập trung": "focus",
    "chế độ focus": "focus",

    # Night
    "nai mode": "night",
    "chế độ ban đêm": "night",
    "chế độ buổi đêm": "night",
    "chế độ night": "night",

    # ===== HÀNH ĐỘNG / CÂU MỆNH LỆNH =====
    # Bật / tắt
    "bật lên": "bật",
    "bật giùm": "bật",
    "bật giúp": "bật",

    "tắt đi": "tắt",
    "tắt giùm": "tắt",
    "tắt giúp": "tắt",

    # Mở / đóng cửa
    "mở khóa cửa": "mở cửa",
    "mở khoá cửa": "mở cửa",
    "đóng khóa cửa": "đóng cửa",
    "đóng khoá cửa": "đóng cửa",

    # ===== CỤM LỆNH "TẤT CẢ" =====
    "tắt hết tất cả đèn": "tắt tất cả đèn",
    "tắt hết đèn": "tắt tất cả đèn",
    "tắt toàn bộ đèn": "tắt tất cả đèn",

    "bật hết tất cả đèn": "bật tất cả đèn",
    "bật hết đèn": "bật tất cả đèn",
    "bật toàn bộ đèn": "bật tất cả đèn",

    "tắt hết quạt": "tắt tất cả quạt",
    "bật hết quạt": "bật tất cả quạt",

    # ===== MỘT SỐ LỖI VẶT KHÁC =====
    "phần trăm": "phần trăm",  # để lỡ Whisper viết dính vẫn coi như đúng
    "phần chăm": "phần trăm",
    "phần châm": "phần trăm",
}

FILLER_WORDS = [
    "giúp tui", "giúp tôi", "cho tui", "cho tôi",
    "làm ơn", "với", "nhé", "đi", "ạ"
]

def normalize_text_for_command(s: str) -> str:
    """Chuẩn hoá câu lệnh giọng nói cho parser.
    - Đưa về lowercase
    - Bỏ bớt từ đệm (giúp tui, làm ơn, ...)
    - Sửa một số cụm hay bị nghe nhầm
    - Gộp khoảng trắng thừa
    """
    s = (s or "").lower().strip()
    # bỏ từ đệm
    for f in FILLER_WORDS:
        if f in s:
            s = s.replace(f, " ")
    # sửa các cụm hay sai
    for wrong, right in COMMON_FIXES.items():
        if wrong in s:
            s = s.replace(wrong, right)
    # gộp khoảng trắng
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ===== NEW: Firebase base & URLs =====
DEFAULT_FB_BASE = "https://do-an-2-91a3c-default-rtdb.asia-southeast1.firebasedatabase.app"
FIREBASE_COMMAND_URL = os.environ.get("FIREBASE_COMMAND_URL", f"{DEFAULT_FB_BASE}/commands.json")
FIREBASE_COMMAND_URL_SANDBOX = os.environ.get("FIREBASE_COMMAND_URL_SANDBOX", f"{DEFAULT_FB_BASE}/commands_sandbox.json")
FIREBASE_DEVICES_URL = os.environ.get("FIREBASE_DEVICES_URL", f"{DEFAULT_FB_BASE}/devices.json")
FIREBASE_DOORS_BASE = os.environ.get("FIREBASE_DOORS_BASE", f"{DEFAULT_FB_BASE}/doors")

# Lazy init Whisper
WHISPER_MODEL = None
_whisper_device = None
_whisper_inited = False

def _init_whisper_safe():
    global WHISPER_MODEL, _whisper_device, _whisper_inited
    if _whisper_inited:
        return WHISPER_MODEL is not None
    _whisper_inited = True
    if torch.cuda.is_available():
        try:
            print("✅ GPU đang dùng:", torch.cuda.get_device_name(0))
        except Exception:
            pass
    else:
        print("⚠️ Không có GPU, chạy CPU cho Whisper.")
    print("ℹ️  PyTorch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
    try:
        _whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
        t0 = time.time()
        print(f"🎙️  Đang tải Whisper model ({VOICE_MODEL_NAME}) ...")
        WHISPER_MODEL = whisper.load_model(VOICE_MODEL_NAME, device=_whisper_device)
        print(f"✅ Whisper model: {VOICE_MODEL_NAME} on {_whisper_device} (load {time.time()-t0:.2f}s)")
        return True
    except Exception as e:
        WHISPER_MODEL = None
        print("❌ Whisper load failed:", e)
        return False

def _ensure_wav_16k_mono(pth: str) -> str:
    p = Path(pth)
    if p.suffix.lower() == ".wav":
        return str(p)
    out = p.with_suffix(".wav")
    subprocess.run(["ffmpeg", "-y", "-i", str(p), "-ac", "1", "-ar", "16000", str(out)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return str(out)

def _probe_duration_seconds(path: str) -> float:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=5
        )
        return float(r.stdout.strip())
    except Exception:
        return -1.0

# ===== NEW: text folding + fuzzy =====
from difflib import SequenceMatcher
def _fold_text(t: str) -> str:
    import unicodedata
    t = (t or "").lower().strip()
    t = ''.join(c for c in unicodedata.normalize('NFD', t) if unicodedata.category(c) != 'Mn')
    t = t.replace('đ', 'd')
    t = re.sub(r'[^a-z0-9 ]+', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()

def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

# ===== NEW: load device catalog from Firebase =====
_DEVICE_CACHE = {"data": None, "ts": 0.0}
def _load_device_catalog(force: bool = False):
    """Đọc danh sách devices từ Firebase và build catalog dùng cho voice."""
    global _DEVICE_CACHE
    ttl = 20  # giây cache

    if (not force) and _DEVICE_CACHE["data"] and (time.time() - _DEVICE_CACHE["ts"] < ttl):
        return _DEVICE_CACHE["data"]

    catalog = []
    try:
        r = requests.get(FIREBASE_DEVICES_URL, timeout=4)
        data = r.json() or {}
    except Exception:
        data = {}

    if isinstance(data, dict):
        items = data.items()
    else:
        items = []

    for key, dev in items:
        dev = dev or {}
        meta = dev.get("metadata", {}) or {}

        name = str(dev.get("name") or meta.get("name") or key)
        area = str(
            dev.get("area")
            or dev.get("room")
            or meta.get("area")
            or "Default"
        )
        dtype = str(
            dev.get("device")
            or dev.get("type")
            or meta.get("device")
            or meta.get("type")
            or "Light"
        )
        brand = str(dev.get("brand") or meta.get("brand") or "")

        base_syn = set(
            filter(
                None,
                [
                    _fold_text(name),
                    _fold_text(dtype),
                    _fold_text(brand),
                    _fold_text(name.replace(" ", "")),
                    _fold_text(f"{dtype} {area}"),
                ],
            )
        )

        dtypef = _fold_text(dtype)

        # Đèn
        if any(k in dtypef for k in ["light", "lamp", "den"]):
            base_syn.update({"den", "bong den", "light", "lamp"})

        # NeoPixel
        if any(k in dtypef for k in ["neopixel", "neo pixel", "neo"]):
            base_syn.update({"neopixel", "neo pixel", "neo"})

        # Philips Hue
        if any(k in dtypef for k in ["philips", "phillips", "hue"]):
            base_syn.update({"philips", "phills", "hue", "phillips"})

        # === NEW: quạt / fan ===
        if "fan" in dtypef or "quat" in dtypef:
            base_syn.update({"quat", "quat tran", "fan"})

        # === NEW: loa / speaker ===
        if "speaker" in dtypef or "loa" in dtypef or "audio" in dtypef:
            base_syn.update({"loa", "loa nghe nhac", "speaker", "am thanh"})

        catalog.append(
            {
                "key": key,
                "id": dev.get("id") or meta.get("id") or key,
                "name": name,
                "area": area,
                "device": dtype,
                "syn": list(base_syn),
            }
        )

    _DEVICE_CACHE = {"data": catalog, "ts": time.time()}
    return catalog

def _known_areas():
    cat = _load_device_catalog()
    return sorted(set(d["area"] for d in cat))

def _infer_area_from_text(sf: str):
    areas = _known_areas()
    m = {_fold_text(a): a for a in areas}
    for fa, orig in m.items():
        if fa and fa in sf:
            return orig
    m2 = re.search(r'\b(o|tai|o tai)\s+([a-z0-9 ]{2,})$', sf)
    if m2:
        cand = m2.group(2).strip()
        best, score = None, 0
        for fa, orig in m.items():
            s = _sim(cand, fa)
            if s > score:
                best, score = orig, s
        if score > 0.6:
            return best
    return None

def _best_device_match(text: str, area_hint: str = None):
    """
    Chọn thiết bị tốt nhất:
      - fuzzymatch với name/syn/device
      - lọc theo loại: đèn / quạt / loa / cửa (nếu câu nói có)
      - ưu tiên thiết bị có tên xuất hiện trực tiếp trong câu
      - ưu tiên khu vực (nếu xác định được)
      - giữ 1 special-case: "neopixel" (loại LED, không phụ thuộc tên cụ thể)
    """
    cat = _load_device_catalog()
    if not cat:
        return None

    sf = _fold_text(text)
    if not area_hint:
        area_hint = _infer_area_from_text(sf) or _last_area_global() or _last_area_client()
    area_hint_f = _fold_text(area_hint) if area_hint else None

    # Ý định loại thiết bị trong câu
    want_light   = any(k in sf for k in ("den", "light", "neopixel"))
    want_fan     = any(k in sf for k in ("quat", "fan"))
    want_speaker = any(k in sf for k in ("loa", "speaker", "am thanh"))
    want_door    = any(k in sf for k in ("cua", "door"))

    # từ khóa đặc biệt cho LED
    wants_neopixel = ("neopixel" in sf) or ("neo pixel" in sf)
    wants_panasonic = ("panasonic" in sf) or ("pana sonic" in sf)
    wants_phillips = ("phillips" in sf) or ("phi lips" in sf)

    best = None
    best_score = 0.0

    for d in cat:
        name_f = _fold_text(d.get("name", ""))
        dev_f  = _fold_text(d.get("device", ""))
        area_f = _fold_text(d.get("area", ""))
        syn_f  = [_fold_text(s) for s in (d.get("syn") or [])]

        profile = " ".join([name_f, dev_f, area_f, " ".join(syn_f)])

        is_light = (
            ("light" in dev_f)
            or ("den" in dev_f)
            or ("lamp" in dev_f)
            or ("den" in name_f)
        )
        is_fan = ("fan" in dev_f) or ("quat" in dev_f) or ("quat" in name_f)
        is_speaker = (
            ("speaker" in dev_f)
            or ("loa" in dev_f)
            or ("loa" in name_f)
            or ("audio" in dev_f)
        )
        is_door = ("door" in dev_f) or ("cua" in name_f)
        is_panasonic = ("panasonic" in dev_f) or ("panasonic" in name_f) or ("pana sonic" in profile)
        is_phillips = ("phillips" in dev_f) or ("phillips" in name_f) or ("phil lips" in profile)
        is_neopixel = ("neopixel" in dev_f) or ("neopixel" in name_f) or ("neo pixel" in profile)

        # Lọc theo loại nếu người dùng nói rõ
        if want_light and not (is_light or is_neopixel or is_panasonic or is_phillips):
            continue
        if want_fan and not is_fan:
            continue
        if want_speaker and not is_speaker:
            continue
        if want_door and not is_door:
            continue

        # Tập chuỗi để so sánh
        syns = syn_f + [name_f, dev_f, f"{dev_f} {area_f}"]
        syns = [s for s in syns if s]

        # Điểm fuzzy (SequenceMatcher)
        syn_score = max((_sim(sf, s) for s in syns), default=0.0)

        # Boost nếu tên/syn xuất hiện trực tiếp trong câu nói
        contain_hit = 0
        for s in syns:
            if s and s in sf:
                contain_hit = 1
                break

        score = syn_score + (0.6 if contain_hit else 0.0)

        # Ưu tiên đúng khu vực (nếu tìm được)
        if area_hint_f and area_f == area_hint_f:
            score *= 1.1

        # Nói "neopixel" → ưu tiên thiết bị NeoPixel
        if wants_neopixel and is_neopixel:
            score *= 1.5

        if wants_panasonic and is_panasonic:
            score *= 1.5

        if wants_phillips and is_phillips:
            score *= 1.5

        if score > best_score:
            best, best_score = d, score

    # Hạ ngưỡng một chút cho đỡ "không tìm thấy thiết bị"
    return best if best_score >= 0.35 else None

# ===== NEW: context memory per-client =====
_LAST_CTX = {}
def _get_client_id():
    return request.headers.get("X-Client-Id") or request.remote_addr or "anon"
def _remember_ctx(device_id: str, area: str):
    _LAST_CTX[_get_client_id()] = {"device_id": device_id, "area": area, "ts": time.time()}
def _last_area_client():
    ctx = _LAST_CTX.get(_get_client_id())
    if not ctx:
        return None
    return ctx["area"] if (time.time()-ctx["ts"] < 1800) else None

def _parse_schedule_vi(text: str):
    """
    Return (due_ts, friendly) or (None, None) if no schedule found.
    Hỗ trợ:
      - "trong/sau 5 giây"
      - "trong/sau 5 phút/giờ"
      - "sau 10p"
      - "lúc 7:30", "9 giờ tối"
      - "ngày mai lúc 6 giờ"
    """
    s = text.lower().strip()
    sf = _fold_text(s)
    now = datetime.now()

    # ===== relative seconds/minutes/hours =====
    m = re.search(r'\b(trong|sau)\s+(\d+)\s*(giay|s|phut|p|ph|minute|min|gio|h)\b', sf)
    if m:
        n = int(m.group(2))
        unit = m.group(3)
        if unit in ("giay", "s"):
            delta = timedelta(seconds=n)
            label_unit = "giây"
        elif unit in ("phut", "p", "ph", "minute", "min"):
            delta = timedelta(minutes=n)
            label_unit = "phút"
        else:
            delta = timedelta(hours=n)
            label_unit = "giờ"
        due = now + delta
        return due.timestamp(), f"hẹn sau {n} {label_unit}"

    # ===== absolute today / tomorrow =====
    is_tomorrow = "ngay mai" in sf or "mai" in sf
    m2 = re.search(r'\b(luc|vao)?\s*(\d{1,2})(?:[:h ](\d{1,2}))?\s*(sang|chieu|toi|am|pm)?\b', sf)
    if m2:
        hh = int(m2.group(2))
        mm = int(m2.group(3) or 0)
        ampm = m2.group(4)
        if ampm in ("pm", "chieu", "toi") and hh < 12:
            hh += 12
        if ampm in ("am", "sang") and hh == 12:
            hh = 0
        day = now.date() + (timedelta(days=1) if is_tomorrow else timedelta(days=0))
        due = datetime.combine(day, datetime.min.time()).replace(hour=hh, minute=mm)
        if due < now and not is_tomorrow:
            due = due + timedelta(days=1)
        label = f"hẹn lúc {due.strftime('%H:%M')}" + (" ngày mai" if is_tomorrow else "")
        return due.timestamp(), label

    return None, None


# ===============================
# AREA MEMORY (ghi nhớ khu vực cuối)
# ===============================
_LAST_AREA = None

def _remember_area(area: str):
    """Lưu khu vực cuối cùng mà giọng nói vừa nhắc tới."""
    global _LAST_AREA
    if area:
        _LAST_AREA = area

def _last_area_global():
    """Lấy khu vực đã được nhắc gần nhất (nếu có)."""
    return _LAST_AREA

# ===== Parser =====
def _parse_command_vi(text: str) -> dict:
    """
    Parse câu tiếng Việt thành command:
    - action: on/off/set
    - brightness / brightness_delta (mức % chung)
    - color (cho đèn)
    - speed/speed_op (cho quạt)
    - volume/volume_op (cho loa)
    """
    s_raw = (text or "").strip()
    s = s_raw.lower()
    sf = _fold_text(s_raw)

    # ----- Hiệu ứng đặc biệt (rainbow cho đèn) -----
    effect = None
    if "cau vong" in sf or "nhay cau vong" in sf or "rainbow" in sf:
        effect = "rainbow_on"
    if "tat cau vong" in sf or "khong cau vong" in sf or "mau binh thuong" in sf:
        effect = "rainbow_off"

    # ----- Scene intents (bật / tắt scene, trả về sớm) -----
    scene_id = None
    scene_action = "apply"  # mặc định là bật / chuyển scene

    # Các câu kiểu "tắt chế độ", "thoát scene", "về mặc định"
    if any(kw in sf for kw in (
        "tat che do",
        "tat che do hien tai",
        "tat scene",
        "tat mode",
        "thoat che do",
        "thoat scene",
        "bo che do",
        "bo scene",
        "ve mac dinh",
        "ve che do mac dinh",
        "ve trang thai mac dinh"
    )) or (
        ("thoat" in sf or "thoai" in sf or "thoat khoi" in sf) and
        any(k in sf for k in ("party", "relax", "focus", "night"))
    ) or (
        ("tat" in sf) and any(k in sf for k in ("party", "relax", "focus", "night"))
    ):
        scene_action = "clear"

    # Nhận diện tên scene
    if "party" in sf or "tiec" in sf:
        scene_id = "party"
    elif "relax" in sf or "thu gian" in sf:
        scene_id = "relax"
    elif "focus" in sf or "tap trung" in sf:
        scene_id = "focus"
    elif (
        "night" in sf
        or "ban dem" in sf
        or "di ngu" in sf
        or ("che do" in sf and "ngu" in sf)
    ):
        scene_id = "night"

    # Nếu chỉ nói "tắt chế độ / thoát scene" không nói tên
    if scene_action == "clear" and not scene_id:
        return {
            "type": "scene",
            "scene_id": "all_off",
            "action": "clear",
            "raw_text": s_raw,
        }

    if scene_id:
        cmd = {
            "type": "scene",
            "scene_id": scene_id,
            "action": scene_action,   # "apply" hoặc "clear"
            "raw_text": s_raw,
        }
        delay_sec = _extract_delay_vi(s_raw)
        if delay_sec:
            cmd["delay_seconds"] = delay_sec
        return cmd

    # ----- Door intents (cửa trước / cửa bên hông) -----
    door_id = None
    if "cua" in sf or "door" in sf:
        # cửa trước / cửa chính
        if any(k in sf for k in ("cua truoc", "cua chinh", "cua vao", "front door")):
            door_id = "frontDoor"
        # cửa bên / cửa hông / cửa phụ
        elif any(k in sf for k in ("cua ben", "cua hong", "cua ben hong", "cua phu", "side door")):
            door_id = "sideDoor"

    if door_id:
        door_action = None
        # các kiểu "mở khóa", "unlock"
        if "mo khoa" in sf or "unlock" in sf:
            door_action = "open"
        elif "khoa" in sf and "mo khoa" not in sf:
            door_action = "close"
        elif "open" in sf or "mo " in sf:
            door_action = "open"
        elif "dong" in sf or "dong lai" in sf or "close" in sf:
            door_action = "close"

        # nếu chỉ nói "cửa chính" / "cửa bên hông" → toggle
        if door_action is None:
            door_action = "toggle"

        cmd = {
            "type": "door",
            "door_id": door_id,   # 'frontDoor' hoặc 'sideDoor'
            "action": door_action,
            "raw_text": s_raw,
        }
        # hỗ trợ luôn hẹn kiểu "sau 5 giây mở cửa"
        delay_sec = _extract_delay_vi(s_raw)
        if delay_sec:
            cmd["delay_seconds"] = delay_sec
        return cmd

    # ----- Action -----
    action = None
    if re.search(r"\b(bật|bat|mo|turn on|on)\b", sf):
        action = "on"
    elif re.search(r"\b(tắt|tat|dong|turn off|off)\b", sf):
        action = "off"

    # ----- Brightness (mức %) dùng chung -----
    brightness = None
    brightness_delta = None

    m_abs = re.search(r"(\d+)\s*%?", s)
    if m_abs:
        try:
            brightness = max(0, min(100, int(m_abs.group(1))))
        except Exception:
            brightness = None

    if ("tăng" in s) or ("giảm" in s) or ("giam" in sf):
        sign = 1 if "tăng" in s else -1
        if brightness is not None:
            # ví dụ: "tăng 20%" -> delta = +20
            brightness_delta = sign * brightness
            brightness = None
        else:
            # ví dụ: "tăng đèn phòng khách" -> mặc định +/-10
            brightness_delta = sign * 10

        
        # ----- Color cho đèn -----
    # Chỉ map những màu hay dùng, KHÔNG map "den" bừa bãi
    COLORS = {
        "do": "red",
        "do tuoi": "red",

        "xanh la": "green",
        "la": "green",
        "xanh luc": "green",
        "luc": "green",

        "xanh duong": "blue",
        "xanh lam": "blue",
        "xanh nuoc bien": "blue",

        "vang": "yellow",

        "trang": "white",
        "trang am": "warm white",
        "am": "warm white",
        "trang lanh": "cool white",
        "lanh": "cool white",

        "cam": "orange",
        "tim": "purple",
        "hong": "pink",

        "xanh ngoc": "cyan",
        "ngoc": "cyan",
        "ho phach": "amber",
        "nau": "brown",
        # KHÔNG để "den": "black" ở đây,
        # để tránh nhầm "bật đèn" -> "màu đen"
    }

    color = None

    # Ưu tiên pattern "màu X" (sf đã bỏ dấu → "mau")
    m_col = re.search(r"\bmau\s+([a-z0-9 ]+)", sf)
    if m_col:
        tail = m_col.group(1).strip()
        for k, v in COLORS.items():
            if tail.startswith(k):
                color = v
                break

    # Fallback: nếu không có từ "mau" mà vẫn nhắc màu
    if color is None:
        sf_padded = f" {sf} "
        for k, v in COLORS.items():
            # tránh nhầm "độ sáng" -> "do" (màu đỏ)
            if k == "do" and "do sang" in sf_padded:
                continue
            if f" {k} " in sf_padded:
                color = v
                break

    # Nếu THẬT SỰ muốn màu đen → phải nói "màu đen"
    if color is None and "mau den" in sf:
        color = "black"

    # ----- Area hint + chọn device -----
    area_hint = _infer_area_from_text(sf) or _last_area_global() or _last_area_client()
    devmatch = _best_device_match(s_raw, area_hint)

    if devmatch:
        # match được thiết bị thật từ /devices
        device = devmatch["device"]
        area = devmatch["area"]
        device_id = devmatch.get("id") or devmatch.get("key")
    else:
        # KHÔNG match được: chọn 1 đèn có sẵn làm "default",
        # tuyệt đối không tạo Light_Default ma nữa
        cat = _load_device_catalog()
        fallback = None
        for d in cat:
            syns = " ".join(_fold_text(s) for s in (d.get("syn") or []))
            if any(k in syns for k in ("light", "den", "lamp", "neopixel", "neo pixel", "neo")):
                fallback = d
                break

        if fallback:
            device = fallback["device"]
            area = fallback["area"]
            device_id = fallback.get("id") or fallback.get("key")
        else:
            # hệ thống không có đèn nào luôn
            device = "Light"
            area = area_hint or "Default"
            device_id = None  # để _push_desired_from_voice skip

    # nhớ khu vực cuối cùng
    _remember_area(area)
    _remember_ctx(device_id, area)

    # nếu chỉnh brightness/color mà chưa có action thì xem như "set"
    if action is None and (brightness is not None or color is not None or brightness_delta is not None):
        action = "set"
    
    # Nếu chỉ nói "cho đèn X cầu vồng" mà không nói bật/tắt → mặc định bật
    if effect == "rainbow_on" and action is None:
        action = "on"

    # ----- NEW: map mức → speed/volume cho quạt & loa -----
    dev_fold = _fold_text(str(device))
    is_fan = ("fan" in dev_fold) or ("quat" in sf)
    is_speaker = ("speaker" in dev_fold) or ("loa" in sf) or ("am thanh" in sf)

    speed = None
    speed_op = None
    volume = None
    volume_op = None

    if is_fan:
        # tuyệt đối: "quạt 50%" → level 1–3
        if brightness is not None:
            if brightness <= 0:
                speed = 0
            else:
                speed = max(1, min(3, round(brightness * 3 / 100)))
        # tương đối: "tăng/giảm quạt"
        if brightness_delta is not None:
            speed_op = "inc" if brightness_delta > 0 else "dec"
            step = max(1, round(abs(brightness_delta) * 3 / 100))
            speed = step

    if is_speaker:
        # tuyệt đối: "loa Bose 30%" → 30
        if brightness is not None:
            volume = brightness
        # tương đối: "tăng/giảm âm lượng loa"
        if brightness_delta is not None:
            volume_op = "inc" if brightness_delta > 0 else "dec"
            volume = abs(brightness_delta) or 5  # mặc định bước 5

    cmd = {
        "device_id": device_id,
        "device": device,
        "area": area,
        "action": action,
        "brightness": brightness,
        "color": color,
        "brightness_delta": brightness_delta,
        "speed": speed,
        "speed_op": speed_op,
        "volume": volume,
        "volume_op": volume_op,
        "effect": effect,
        "raw_text": s_raw,
    }
    return cmd

# ===== Việt Nam delay parser =====
def _extract_delay_vi(text: str):
    t = text.lower()
    # sau 30 giây
    m = re.search(r"sau\s+(\d+)\s*(giay|giy|dây|day|giây|s)", t)
    if m:
        return int(m.group(1))

    # sau 2 phút
    m = re.search(r"sau\s+(\d+)\s*(phut|phu|phuc|phúc|phít|phit|phút|p)", t)
    if m:
        return int(m.group(1)) * 60

    # trong 5 giây
    m = re.search(r"trong\s+(\d+)\s*(giay|giy|dây|day|giây|s)", t)
    if m:
        return int(m.group(1))

    # trong 3 phút
    m = re.search(r"trong\s+(\d+)\s*(phut|phu|phuc|phúc|phít|phit|phút|p)", t)
    if m:
        return int(m.group(1)) * 60

    return None

# ===== Push command (with sandbox) =====
def _push_command(cmd: dict, sandbox=False) -> dict:
    payload = {k: v for k, v in cmd.items() if v is not None}
    url = FIREBASE_COMMAND_URL_SANDBOX if sandbox else FIREBASE_COMMAND_URL
    if not url:
        print("⚠️ FIREBASE_COMMAND_URL chưa set → DRY-RUN (không thực thi).")
        return {"target": "none", "status": "noop", "payload": payload}
    try:
        r = requests.post(url, json=payload, timeout=4)
        return {"target": "firebase", "status": "ok", "code": r.status_code, "payload": payload}
    except Exception as e:
        return {"target": "firebase", "status": "error", "message": str(e), "payload": payload}

# ===== NEW: simple in-process scheduler =====
_sched_lock = threading.Lock()
_sched_heap = []  # heap of (due_ts, job_id, job_dict)
_sched_started = False

def _scheduler_loop():
    print("⏱️  Scheduler thread started.")
    while True:
        now = time.time()
        job = None
        with _sched_lock:
            if _sched_heap and _sched_heap[0][0] <= now:
                _, _, job = heapq.heappop(_sched_heap)
        if job:
            print(f"⏰ Running scheduled job {job['id']} → push command")
            _push_command(job["cmd"], sandbox=job.get("sandbox", False))
        time.sleep(0.3)

def _start_scheduler_once():
    global _sched_started
    if _sched_started:
        return
    t = threading.Thread(target=_scheduler_loop, daemon=True)
    t.start()
    _sched_started = True

def _schedule_command(due_ts: float, cmd: dict, sandbox=False):
    _start_scheduler_once()
    job_id = str(uuid.uuid4())[:8]
    job = {"id": job_id, "cmd": cmd, "sandbox": sandbox}
    with _sched_lock:
        heapq.heappush(_sched_heap, (due_ts, job_id, job))
    return job_id

# ===============================
# FLASK APP
# ===============================
app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

# ==== API QUẢN LÝ KHUÔN MẶT ====

@app.route('/faces', methods=['GET'])
def list_faces():
    """Trả về danh sách khuôn mặt đã đăng ký (từ registered.json)."""
    db = _load_registered_db_raw()
    return jsonify(db)

@app.route('/faces/delete', methods=['POST'])
def delete_face():
    """Xóa 1 khuôn mặt theo name: xóa DB + xóa file trong registered_images + rebuild embedding."""
    try:
        data = request.get_json() or {}
        name = (data.get("name") or "").strip()
        if not name:
            return jsonify({"status": "error", "message": "Missing name"}), 400

        db = _load_registered_db_raw()
        if not db:
            return jsonify({"status": "error", "message": "registered.json rỗng hoặc không tồn tại"}), 404

        remaining = []
        deleted_files = []
        deleted = False

        for entry in db:
            if entry.get("name") == name:
                lf = entry.get("local_file")
                if lf and os.path.exists(lf):
                    try:
                        os.remove(lf)      # 🔥 xóa luôn file ảnh
                        deleted_files.append(lf)
                    except Exception as e:
                        print("⚠️ Không xóa được file:", lf, e)
                deleted = True
            else:
                remaining.append(entry)

        if not deleted:
            return jsonify({"status": "error", "message": "Không tìm thấy name cần xóa"}), 404

        _save_registered_db_raw(remaining)
        rebuild_known_faces()

        return jsonify({
            "status": "success",
            "deleted": name,
            "deleted_files": deleted_files,
            "remaining": len(remaining)
        })
    except Exception as e:
        print("⚠️ Lỗi delete_face:", e)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/execute_scene', methods=['POST'])
def execute_scene():
    """
    Áp dụng scene:
    - Ưu tiên dùng scene.actions nếu có (kiểu cũ: device + desired)
    - Nếu không có actions thì dùng scene.rules (kiểu mới: match type/section)
    """
    data = request.get_json(silent=True) or {}
    scene = data.get("scene")

    # Khi clearScene() gửi scene = null -> chỉ trả về ok, không làm gì
    if not scene:
        return jsonify({"status": "cleared"})

    import requests, time

    # ----- 1. Load config scene -----
    SCENE_URL = f"{DEFAULT_FB_BASE}/scenes/{scene}.json"
    resp_scene = requests.get(SCENE_URL)
    if resp_scene.status_code != 200:
        return jsonify({"error": "Scene not found"}), 404

    scene_data = resp_scene.json() or {}
    if not scene_data:
        return jsonify({"error": "Scene not found"}), 404

    actions = scene_data.get("actions")
    rules   = scene_data.get("rules")

    patches = []   # danh sách (device_id, desired)

    # ----- 2A. Kiểu cũ: actions (device + desired) -----
    if actions:
        for act in actions or []:
            dev_id = act.get("device")
            desired = act.get("desired") or {}
            if not dev_id or not isinstance(desired, dict):
                continue
            patches.append((dev_id, desired))

    # ----- 2B. Kiểu mới: rules (match type/section) -----
    elif rules:
        DEVICES_URL = f"{DEFAULT_FB_BASE}/devices.json"
        resp_dev = requests.get(DEVICES_URL)
        if resp_dev.status_code != 200:
            return jsonify({"error": "Failed to load devices"}), 500

        devices = resp_dev.json() or {}

        def norm(v):
            return (v or "").strip().lower()

        for dev_id, node in devices.items():
            meta   = node.get("metadata", {})
            legacy = node  # fallback
            dev_type = norm(meta.get("type") or legacy.get("type"))
            section  = norm(meta.get("section") or legacy.get("section"))

            for rule in rules:
                match = rule.get("match") or {}
                m_type    = norm(match.get("type"))
                m_section = norm(match.get("section"))

                # match type
                if m_type and dev_type != m_type:
                    continue
                # match section (nếu có)
                if m_section and section != m_section:
                    continue

                desired = dict(rule.get("desired") or {})
                if not desired:
                    continue

                # thêm metadata cho log
                desired.setdefault("ts", int(time.time() * 1000))
                desired.setdefault("updated_by", f"scene:{scene}")

                patches.append((dev_id, desired))
                # KHÔNG break: để night có thể áp dụng 2 rule
                # (vd: tắt hết light rồi bật lại light ở Entry)

    else:
        return jsonify({"error": "Scene has neither actions nor rules"}), 400

    # ----- 3. Gửi PATCH lên từng thiết bị -----
    results = {}
    for dev_id, desired in patches:
        # chuyển state(bool) -> power(on/off) nếu có
        if "state" in desired:
            state_val = desired.pop("state")
            if isinstance(state_val, bool):
                desired["power"] = "on" if state_val else "off"

        try:
            r = requests.patch(
                f"{DEFAULT_FB_BASE}/devices/{dev_id}/desired.json",
                json=desired,
                timeout=5
            )
            results[dev_id] = r.status_code
        except Exception as e:
            results[dev_id] = f"error: {e}"

    return jsonify({"status": "ok", "scene": scene, "patched": results})

@app.route('/upload', methods=['POST'])
def upload():
    """
    Nhận 1 frame (ảnh) từ client, nhận diện khuôn mặt.
      - Bỏ qua mặt quá nhỏ / quá mờ.
      - Dùng SIM_THRESHOLD = 0.70.
      - Yêu cầu N frame liên tiếp (theo IP) đều đạt ngưỡng mới cho pass.
      - Nếu Unknown liên tiếp >= 5 frame trong 3s -> gửi cảnh báo Telegram (1 lần / chuỗi).
    """
    try:
        data = request.get_json()
        img_base64 = data.get('image')
        if not img_base64:
            return jsonify({'error': 'Không có ảnh'}), 400

        # decode base64 -> BGR frame
        img_bytes = base64.b64decode(img_base64.split(',')[1])
        npimg = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'error': 'Không decode được ảnh'}), 400

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb)

        raw_name, best_sim = "Unknown", -1.0

        if results.detections:
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                bw, bh = int(bbox.width * w), int(bbox.height * h)
                x, y = max(0, x), max(0, y)
                x2, y2 = min(w, x + bw), min(h, y + bh)

                # Bỏ qua mặt quá nhỏ
                if bw < MIN_FACE_SIZE or bh < MIN_FACE_SIZE:
                    continue

                face_crop = frame[y:y2, x:x2]
                if face_crop is None or face_crop.size == 0:
                    continue

                # Bỏ qua frame quá mờ
                if is_blurry(face_crop):
                    continue

                emb = get_embedding_from_bgr(face_crop)
                for k_emb, k_name in zip(known_embeddings, known_names):
                    sim = float(cosine_similarity([emb], [k_emb])[0][0])
                    if sim > best_sim:
                        best_sim = sim
                        raw_name = k_name if sim >= SIM_THRESHOLD else "Unknown"

        # ===== Multi-frame confirm theo IP =====
        client_ip = request.remote_addr or "unknown"
        now = time.time()
        st = RECOG_STATE.get(client_ip, {
            "name": None,
            "count": 0,
            "ts": 0.0,
            "last_sim": -1.0,
            "notified": False
        })

        final_name = "Unknown"

        if raw_name != "Unknown" and best_sim >= SIM_THRESHOLD:
            # nếu còn trong window & cùng 1 tên -> tăng count
            if st["name"] == raw_name and (now - st["ts"]) <= RECOG_WINDOW_SEC:
                st["count"] += 1
            else:
                # bắt đầu chuỗi mới (đã biết mặt)
                st = {
                    "name": raw_name,
                    "count": 1,
                    "ts": now,
                    "last_sim": best_sim,
                    "notified": False
                }

            st["ts"] = now
            st["last_sim"] = best_sim
            RECOG_STATE[client_ip] = st

            if st["count"] >= REQ_CONSEC_FRAMES:
                final_name = raw_name
            else:
                final_name = "Unknown"
        else:
            # không nhận diện được / similarity thấp -> chuỗi Unknown
            if st["name"] == "Unknown" and (now - st["ts"]) <= RECOG_WINDOW_SEC:
                st["count"] += 1
            else:
                st = {
                    "name": "Unknown",
                    "count": 1,
                    "ts": now,
                    "last_sim": best_sim,
                    "notified": False
                }
            st["ts"] = now
            st["last_sim"] = best_sim
            RECOG_STATE[client_ip] = st
            final_name = "Unknown"

        streak = RECOG_STATE.get(client_ip, {}).get("count", 0)
        print(
            f"📸 Kết quả: raw={raw_name}, final={final_name}, "
            f"sim={best_sim:.2f}, streak={streak}"
        )

        # ==== TELEGRAM ALERT: Unknown liên tiếp >= 5 frame (có cooldown chống spam) ====
        st_state = RECOG_STATE.get(client_ip, st)

        # Lấy lần gửi cảnh báo gần nhất cho IP này
        last_alert_ts = UNKNOWN_ALERT_STATE.get(client_ip, 0.0)
        can_alert_time = (now - last_alert_ts) >= UNKNOWN_ALERT_COOLDOWN_SEC

        should_alert = (
            final_name == "Unknown"
            and streak >= 5
            and not st_state.get("notified")
            and can_alert_time
        )

        if should_alert:
            try:
                tg_send(f"🚨 Camera: Phát hiện khuôn mặt lạ (IP {client_ip}, streak={streak})")
                _, jpg = cv2.imencode(".jpg", frame)
                tg_send_photo(jpg.tobytes(), caption="🚨 Unknown face detected")
            except Exception as _e:
                print("⚠️ Telegram alert error:", _e)

            # đánh dấu đã báo + cập nhật thời gian alert cuối
            st_state["notified"] = True
            RECOG_STATE[client_ip] = st_state
            UNKNOWN_ALERT_STATE[client_ip] = now

        # JSON giữ format cũ để index.html không phải sửa
        return jsonify({
            "recognized": final_name,
            "name": final_name,
            "similarity": best_sim,
            "raw_name": raw_name,
            "threshold": SIM_THRESHOLD,
            "streak": streak,
            "required_streak": REQ_CONSEC_FRAMES
        })
    except Exception as e:
        print("⚠️ Lỗi xử lý:", e)
        return jsonify({'error': str(e)}), 500

@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json() or {}

        # Lấy trước name + ảnh (dù đúng hay sai mật khẩu)
        name = (data.get('name') or "Unknown").strip()
        img_base64 = data.get('image')

        # ======= CHECK MẬT KHẨU ADMIN =======
        admin_pass = (data.get("admin_password") or "").strip()
        if not ADMIN_REGISTER_PASSWORD:
            return jsonify({
                'status': 'error',
                'code': 'access_control_not_configured',
                'message': 'ADMIN_REGISTER_PASSWORD chưa được cấu hình.'
            }), 503

        if admin_pass != ADMIN_REGISTER_PASSWORD:
            # 🔒 Cảnh báo bảo mật: có người cố đăng ký nhưng sai mật khẩu admin
            try:
                ip = request.remote_addr or "unknown"
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                msg = (
                    "⚠️ CẢNH BÁO BẢO MẬT: Có người nhập SAI mật khẩu admin khi ĐĂNG KÝ KHUÔN MẶT.\n"
                    f"- Thời gian: {ts}\n"
                    f"- IP: {ip}\n"
                    f'- Tên khai báo: "{name}"'
                )
                tg_send(msg)

                # Nếu web gửi kèm ảnh thì gửi luôn ảnh đó lên Telegram
                if img_base64:
                    try:
                        b64 = img_base64.split(',')[1] if ',' in img_base64 else img_base64
                        img_bytes = base64.b64decode(b64)
                        tg_send_photo(img_bytes, caption="📸 Ảnh người nhập sai mật khẩu admin")
                    except Exception as e_img:
                        print("⚠️ Không gửi được ảnh sai mật khẩu admin:", e_img)

            except Exception as e_tg:
                print("⚠️ Không gửi được cảnh báo Telegram (register sai mật khẩu):", e_tg)

            return jsonify({
                'status': 'error',
                'message': 'Sai mật khẩu admin, không được phép đăng ký khuôn mặt mới.'
            }), 403
        # ====================================

        if not img_base64:
            return jsonify({'status': 'error', 'message': 'Không có ảnh'}), 400

        timestamp = int(time.time())
        safe_name = name.replace(' ', '_')
        filename = os.path.join(UPLOADS_DIR, f"{safe_name}_{timestamp}.jpg")
        img_bytes = base64.b64decode(img_base64.split(',')[1])
        with open(filename, 'wb') as f:
            f.write(img_bytes)

        drive_id = None
        if DRIVE_AVAILABLE:
            drive_id = upload_file_to_drive(filename, title=os.path.basename(filename))
            if drive_id:
                print(f"✅ Uploaded {filename} to Drive id={drive_id}")
            else:
                print("ℹ️ Upload to Drive failed hoặc trả về None.")

        ok = load_sample(filename, name)
        if not ok:
            return jsonify({'status': 'error', 'message': 'Không thể tạo embedding từ ảnh đã tải lên.'}), 500

        save_registered_entry(name, filename, drive_id)

        resp = {
            'status': 'success',
            'message': f'Đăng ký thành công: {name}',
            'file': filename,
            'drive_id': drive_id
        }

        return jsonify(resp)
    except Exception as e:
        print("⚠️ Lỗi register:", e)
        return jsonify({'status': 'error', 'message': str(e)}), 500
    
@app.route('/pin_login', methods=['POST'])
def pin_login():
    """
    Đăng nhập dự phòng bằng mã PIN (dùng cho keypad ở index.html).
    Front-end gửi JSON: { "pin": "2580" }
    → Nếu đúng PIN: trả về {status:"success", name:"Admin"}
      Nếu sai:      {status:"error", code:"bad_pin", ...}
    """
    try:
        data = request.get_json(silent=True) or {}
        pin = str(data.get("pin", "")).strip()

        if not pin:
            return jsonify({
                "status": "error",
                "code": "missing_pin",
                "message": "Thiếu mã PIN."
            }), 400

        if not SMART_HOME_PIN:
            return jsonify({
                "status": "error",
                "code": "access_control_not_configured",
                "message": "SMART_HOME_PIN chưa được cấu hình."
            }), 503

        if pin != SMART_HOME_PIN:
            return jsonify({
                "status": "error",
                "code": "bad_pin",
                "message": "Sai mã PIN."
            }), 403

        # Thành công: cho vào nhà với tên "Admin" (index.html sẽ gọi onRecognized(name))
        return jsonify({
            "status": "success",
            "name": "Admin"
        })
    except Exception as e:
        print("⚠️ Lỗi pin_login:", e)
        return jsonify({
            "status": "error",
            "code": "exception",
            "message": str(e)
        }), 500

@app.route('/voice-door', methods=['POST'])
def voice_door():
    data = request.json

    device = data.get("device", "").lower()
    action = data.get("action", "").lower()

    # map tên cửa
    if "front" in device or "trước" in device or "main" in device:
        door_id = "frontDoor"
    elif "side" in device or "sau" in device or "hông" in device or "phụ" in device:
        door_id = "sideDoor"
    else:
        return jsonify({"error": "Unknown door"}), 400

    # map action nghĩa rộng
    if action in ["open", "mở"]:
        state = "open"
    elif action in ["close", "đóng", "shut"]:
        state = "closed"
    elif action in ["lock", "khóa"]:
        state = "closed"
    elif action in ["unlock", "mở khóa"]:
        state = "open"
    else:
        return jsonify({"error": "Unknown action"}), 400

    # gửi lên firebase
    ref = f"{FIREBASE_DOORS_BASE}/{door_id}/state.json"
    body = f"\"{state}\""
    r = requests.patch(ref, data=body)

    return jsonify({"status": "ok", "door": door_id, "state": state})

@app.route('/voice', methods=['POST'])
def voice():
    # Lazy init Whisper
    if not _init_whisper_safe():
        return jsonify({"status": "error", "message": "Whisper model chưa sẵn sàng"}), 500

    if "audio" not in request.files and not request.get_json(silent=True):
        return jsonify({"status": "error", "message": "Thiếu audio hoặc text"}), 400

    # --- Gating (execute/token) ---
    execute = VOICE_EXECUTE_DEFAULT
    if request.headers.get("X-Execute", "").strip() == "1":
        execute = True
    if request.form.get("execute", "").strip() == "1":
        execute = True
    data_json = request.get_json(silent=True) or {}
    if str(data_json.get("execute", "0")).lower() in ("1", "true", "yes"):
        execute = True

    provided_token = (
        request.headers.get("X-Voice-Token")
        or request.form.get("token")
        or data_json.get("token")
    )
    if VOICE_CONTROL_TOKEN and provided_token != VOICE_CONTROL_TOKEN:
        print("⚠️ Token không khớp → DRY-RUN.")
        execute = False

    # Sandbox?
    sandbox = (
        request.headers.get("X-Sandbox") in ("1", "true")
        or str(data_json.get("sandbox", "0")).lower() in ("1", "true")
    )

    try:
        # --- Get text ---
        if "audio" in request.files:
            f = request.files["audio"]
            tmp = tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(f.filename).suffix or ".webm"
            )
            f.save(tmp.name)
            tmp.close()
            print(f" Nhận audio: {f.filename} -> {tmp.name}")

            t_ff = time.time()
            wav_path = _ensure_wav_16k_mono(tmp.name)
            dur = _probe_duration_seconds(wav_path)
            print(f" ffmpeg -> {wav_path} (dur={dur:.2f}s) trong {time.time()-t_ff:.2f}s")

            print(
                f" Transcribe start (device={_whisper_device}, model={VOICE_MODEL_NAME}, execute={execute})"
            )
            t_stt = time.time()
            res = WHISPER_MODEL.transcribe(
                wav_path,
                language=VOICE_LANGUAGE,
                initial_prompt=INITIAL_PROMPT,
                temperature=0,
                beam_size=3,
                condition_on_previous_text=False,
                fp16=(_whisper_device == "cuda"),
                no_speech_threshold=0.4,          # bỏ bớt đoạn im lặng / noise
                logprob_threshold=-1.0,           # tránh nhận kết quả quá tệ
                compression_ratio_threshold=2.4,  # tránh text “lặp lặp lặp”
             )
            
            raw_text = (res.get("text") or "").strip()
            print(f" Transcribe xong trong {time.time()-t_stt:.2f}s → raw='{raw_text}'")
        else:
            raw_text = (data_json.get("text") or "").strip()
            print(f" Text client gửi: '{raw_text}' (execute={execute})")
        
        text = normalize_text_for_command(raw_text)
        
        if not text:
            return jsonify(
                {"status": "error", "message": "Không có nội dung để phân tích"}
            ), 400

        # --- Parse command & schedule (mặc định 1 thiết bị) ---
        cmd = _parse_command_vi(text)
        print(" Parsed:", cmd)

         # --- Scene: trả thẳng cho client, không push /commands ---
        if isinstance(cmd, dict) and cmd.get("type") == "scene" and cmd.get("scene_id"):
            # nếu chưa có delay_seconds thì bổ sung từ câu lệnh
            if "delay_seconds" not in cmd:
                delay_sec = _extract_delay_vi(text)
                if delay_sec:
                    cmd["delay_seconds"] = delay_sec

            return jsonify(
                {
                    "status": "ok",
                    "execute": execute,
                    "text": text,
                    "intent": "scene",
                    "command": cmd,
                    "scheduled": None,
                    "sandbox": bool(sandbox),
                }
            )

        # --- xử lý riêng câu "tắt hết / bật hết tất cả đèn"/ "tắt tất cả thiết bị" ---
        sf_all = _fold_text(text)

        # ========= ALL DEVICES OFF =========
        if (
            "tat tat ca thiet bi" in sf_all
            or "tat het tat ca thiet bi" in sf_all
            or "tat het thiet bi" in sf_all
            or ("tat tat ca" in sf_all and "thiet bi" in sf_all)
        ):
            print(" ALL DEVICES OFF intent:", sf_all)

            # Lấy danh sách device từ Firebase
            try:
                r_devs = requests.get(FIREBASE_DEVICES_URL, timeout=4)
                devs = r_devs.json() or {}
            except Exception as e_all_dev:
                print(" ALL_DEVICES_OFF: không đọc được /devices:", e_all_dev)
                devs = {}

            cmds = []
            if isinstance(devs, dict):
                for key, dev in devs.items():
                    dev = dev or {}
                    meta = (dev.get("metadata") or {}) or {}
                    dtype = str(
                        dev.get("device")
                        or dev.get("type")
                        or meta.get("device")
                        or meta.get("type")
                        or ""
                    )
                    fdt = _fold_text(dtype)

                    # chỉ lấy các loại thiết bị có thể bật/tắt
                    is_light   = ("light" in fdt) or ("den" in fdt) or ("neopixel" in fdt)
                    is_fan     = ("fan" in fdt) or ("quat" in fdt)
                    is_speaker = ("speaker" in fdt) or ("loa" in fdt)
                    if not (is_light or is_fan or is_speaker):
                        continue

                    dev_id = str(dev.get("id") or meta.get("id") or key)
                    area   = str(dev.get("area") or meta.get("area") or "Default")

                    c = {
                        "device_id": dev_id,
                        "area": area,
                        "action": "off",
                        "raw_text": text,
                    }
                    if is_light:
                        c["device"] = "Light"
                        c["brightness"] = 0
                    elif is_fan:
                        c["device"] = "Fan"
                        c["speed"] = 0
                    elif is_speaker:
                        c["device"] = "Speaker"
                        c["volume"] = 0

                    cmds.append(c)

            print(f"🔎 ALL_DEVICES_OFF: tìm được {len(cmds)} thiết bị.")

            # parse delay Việt Nam cho lệnh "tắt tất cả thiết bị"
            delay_all = _extract_delay_vi(text)
            if delay_all:
                for c in cmds:
                    c["delay_seconds"] = delay_all
                print(f"⏱ ALL_DEVICES_OFF delay {delay_all} giây cho {len(cmds)} thiết bị.")
            
            due_ts_all, due_label_all = _parse_schedule_vi(text)

            results = []
            scheduled = None
            if execute and cmds:
                if due_ts_all:
                    scheduled = []
                    for c in cmds:
                        job_id = _schedule_command(due_ts_all, c, sandbox=sandbox)
                        scheduled.append(
                            {
                                "id": job_id,
                                "device_id": c.get("device_id"),
                                "due_ts": due_ts_all,
                                "label": due_label_all,
                            }
                        )
                    print(
                        f"🗓 ALL_DEVICES_OFF scheduled {len(cmds)} cmds at {due_label_all} (ts={due_ts_all})"
                    )
                else:
                    for c in cmds:
                        try:
                            results.append(_push_command(c, sandbox=sandbox))
                        except Exception as e_push_all:
                            print("⚠️ ALL_DEVICES_OFF push error:", e_push_all)

            multi_cmd = {
                "multi": cmds,
                "action": "all_devices_off",
                "raw_text": text,
            }
            return jsonify(
                {
                    "status": "ok",
                    "execute": execute,
                    "text": text,
                    "intent": "all_devices_off",
                    "command": multi_cmd,
                    "results": results,
                    "scheduled": scheduled,
                    "sandbox": bool(sandbox),
                }
            )


        # ========= ALL LIGHTS OFF =========
        if (
            "tat het tat ca den" in sf_all
            or "tat het den" in sf_all
            or "tat tat ca den" in sf_all
            or ("tat het" in sf_all and "den" in sf_all)
        ):
            print("💡 ALL LIGHTS OFF intent:", sf_all)

            # Lấy danh sách device từ Firebase
            try:
                r_devs = requests.get(FIREBASE_DEVICES_URL, timeout=4)
                devs = r_devs.json() or {}
            except Exception as e_all:
                print("⚠️ ALL_OFF: không đọc được /devices:", e_all)
                devs = {}

            cmds = []
            if isinstance(devs, dict):
                for key, dev in devs.items():
                    dev = dev or {}
                    meta = dev.get("metadata", {}) or {}
                    dtype = str(
                        dev.get("device")
                        or dev.get("type")
                        or meta.get("device")
                        or meta.get("type")   # 🔧 đọc thêm metadata.type
                        or ""
                    )
                    fdt = _fold_text(dtype)

                    # chỉ chọn thiết bị là đèn / NeoPixel
                    if (
                        "light" not in fdt
                        and "den" not in fdt
                        and "neopixel" not in fdt
                    ):
                        continue

                    dev_id = str(dev.get("id") or meta.get("id") or key)
                    area = str(dev.get("area") or meta.get("area") or "Default")

                    c = {
                        "device_id": dev_id,
                        "device": "Light",
                        "area": area,
                        "action": "off",
                        "brightness": 0,
                        "raw_text": text,
                    }
                    cmds.append(c)

            print(f"🔎 ALL_OFF: tìm được {len(cmds)} thiết bị đèn.")

            # 🔴 parse delay cho lệnh "tắt tất cả" (sau 5 giây / sau 2 phút ...)
            delay_all = _extract_delay_vi(text)
            if delay_all:
                for c in cmds:
                    c["delay_seconds"] = delay_all
                print(f"⏱ ALL_OFF delay {delay_all} giây cho {len(cmds)} thiết bị.")

            # Dùng scheduler để hẹn giờ (hỗ trợ cả 'lúc 7 giờ', 'sau 5 phút', ...)
            due_ts_all, due_label_all = _parse_schedule_vi(text)

            results = []
            scheduled = None
            if execute and cmds:
                if due_ts_all:
                    # HẸN GIỜ: chỉ schedule, KHÔNG push ngay
                    scheduled = []
                    for c in cmds:
                        job_id = _schedule_command(due_ts_all, c, sandbox=sandbox)
                        scheduled.append(
                            {
                                "id": job_id,
                                "device_id": c.get("device_id"),
                                "due_ts": due_ts_all,
                                "label": due_label_all,
                            }
                        )
                    print(
                        f"🗓 ALL_OFF scheduled {len(cmds)} cmds at {due_label_all} (ts={due_ts_all})"
                    )
                else:
                    # KHÔNG có thời gian → tắt ngay lập tức
                    for c in cmds:
                        try:
                            results.append(_push_command(c, sandbox=sandbox))
                        except Exception as e_push:
                            print("⚠️ ALL_OFF push error:", e_push)

            multi_cmd = {
                "multi": cmds,
                "action": "all_off",
                "raw_text": text,
            }

            return jsonify(
                {
                    "status": "ok",
                    "execute": execute,
                    "text": text,
                    "intent": "all_lights_off",
                    "command": multi_cmd,   # <-- client đọc cmd.multi
                    "results": results,
                    "scheduled": scheduled,
                    "sandbox": bool(sandbox),
                }
            )

        # ========= ALL LIGHTS ON =========
        elif (
            "bat het tat ca den" in sf_all
            or "bat het den" in sf_all
            or "bat tat ca den" in sf_all
            or ("bat het" in sf_all and "den" in sf_all)
        ):
            print(" ALL LIGHTS ON intent:", sf_all)

            try:
                r_devs = requests.get(FIREBASE_DEVICES_URL, timeout=4)
                devs = r_devs.json() or {}
            except Exception as e_all_on:
                print("ALL_ON: không đọc được /devices:", e_all_on)
                devs = {}

            cmds = []
            if isinstance(devs, dict):
                for key, dev in devs.items():
                    meta = dev.get("metadata", {}) or {}
                    dtype = str(
                        dev.get("device")
                        or dev.get("type")
                        or meta.get("device")
                        or meta.get("type")   # 🔧 đọc thêm metadata.type
                        or ""
                    )
                    fdt = _fold_text(dtype)
                    if (
                        "light" not in fdt
                        and "den" not in fdt
                        and "neopixel" not in fdt
                    ):
                        continue

                    dev_id = str(dev.get("id") or meta.get("id") or key)
                    area = str(dev.get("area") or meta.get("area") or "Default")

                    cmds.append(
                        {
                            "device_id": dev_id,
                            "device": "Light",
                            "area": area,
                            "action": "on",
                            "raw_text": text,
                        }
                    )

            print(f" ALL_ON: tìm được {len(cmds)} thiết bị đèn.")

            # 🔴 parse delay cho lệnh "bật tất cả"
            delay_all = _extract_delay_vi(text)
            if delay_all:
                for c in cmds:
                    c["delay_seconds"] = delay_all
                print(f"⏱ ALL_ON delay {delay_all} giây cho {len(cmds)} thiết bị.")

            results = []
            if execute and cmds:
                for c in cmds:
                    try:
                        results.append(_push_command(c, sandbox=sandbox))
                    except Exception as e_push_on:
                        print("⚠️ ALL_ON push error:", e_push_on)

            multi_cmd = {
                "multi": cmds,
                "action": "all_on",
                "raw_text": text,
            }

            return jsonify(
                {
                    "status": "ok",
                    "execute": execute,
                    "text": text,
                    "intent": "all_lights_on",
                    "command": multi_cmd,  # <-- client đọc cmd.multi
                    "results": results,
                    "sandbox": bool(sandbox),
                }
            )

        # ========= CÁC LỆNH BÌNH THƯỜNG =========

        # parse delay (VN)
        delay_sec = _extract_delay_vi(text)
        if delay_sec:
            cmd["delay_seconds"] = delay_sec

        # remember context
        if cmd.get("device_id") and cmd.get("area"):
            _remember_ctx(cmd["device_id"], cmd["area"])

        # schedule?
        safe_for_time = re.sub(r"\b\d+\s*%", "", text or "")
        safe_for_time = re.sub(
            r"(độ\s*sáng|do\s*sang|brightness)\s*\d+",
            "",
            safe_for_time,
            flags=re.I,
        )
        has_time_kw = re.search(
            r"\b(lúc|vao luc|sau|trong|hẹn|hen|giờ|gio|phút|phut|am|pm)\b",
            safe_for_time,
            re.I,
        )

        due_ts, due_label = (None, None)
        if has_time_kw:
            try:
                due_ts, due_label = _parse_schedule_vi(safe_for_time)
            except Exception as e:
                print("⚠ schedule parse error -> run now:", e)
                due_ts, due_label = (None, None)

        scheduled = None
        if execute and due_ts:
            job_id = _schedule_command(due_ts, cmd, sandbox=sandbox)
            scheduled = {"id": job_id, "due_ts": due_ts, "label": due_label}

        # push now (if not scheduled)
        pushed = {
            "target": "none",
            "status": "noop",
            "payload": {k: v for k, v in cmd.items() if v is not None},
        }
        if execute and not due_ts and (
            cmd.get("action") in ("on", "off", "set")
            or cmd.get("brightness") is not None
            or cmd.get("color") is not None
            or cmd.get("brightness_delta") is not None
        ):
            pushed = _push_command(cmd, sandbox=sandbox)
        elif not execute:
            print("ℹ️ DRY-RUN (không thực thi).")
        else:
            print("🗓️  Đã xếp lịch, không đẩy ngay.")

        return jsonify(
            {
                "status": "ok",
                "execute": execute,
                "text": text,
                "command": cmd,
                "scheduled": scheduled,
                "push_result": pushed,
                "sandbox": bool(sandbox),
            }
        )
    except Exception as e:
        print(" Voice pipeline error:", e)
        return jsonify({"status": "error", "message": str(e)}), 500


try:
    __orig_push_command = _push_command
except NameError:
    __orig_push_command = None

LIGHT_COLOR_HEX = {
    "red":        "#ff0000",
    "do":         "#ff0000",
    "đỏ":         "#ff0000",

    "green":      "#00ff00",
    "xanh la":    "#00ff00",
    "xanh lá":    "#00ff00",

    "blue":       "#0000ff",
    "xanh duong": "#0000ff",
    "xanh dương": "#0000ff",

    "yellow":     "#ffff00",
    "vang":       "#ffff00",
    "vàng":       "#ffff00",

    "white":      "#ffffff",
    "trang":      "#ffffff",
    "trắng":      "#ffffff",

    "warm":       "#fff3e0",
    "am":         "#fff3e0",
    "ấm":         "#fff3e0",

    "cool":       "#e0f7ff",
    "lanh":       "#e0f7ff",
    "lạnh":       "#e0f7ff",

    "pink":       "#ff4081",
    "hong":       "#ff4081",
    "hồng":       "#ff4081",

    "purple":     "#7c4dff",
    "tim":        "#7c4dff",
    "tím":        "#7c4dff",

    "orange":     "#ffab40",
    "cyan":       "#40ffff",
    "amber":      "#ffca28",
    "brown":      "#8d6e63",
    "black":      "#000000",
}

VOICE_WRITE_DESIRED = os.environ.get("VOICE_WRITE_DESIRED", "1") == "1"

def _push_desired_from_voice(cmd: dict) -> dict:
    try:
        dev_id = cmd.get("device_id")
        if not dev_id:
            return {"status": "skip", "reason": "missing device_id"}

        # Đọc desired hiện tại để xử lý delta
        try:
            cur_url = f"{DEFAULT_FB_BASE}/devices/{dev_id}/desired.json"
            r_cur = requests.get(cur_url, timeout=2)
            cur = r_cur.json() or {}
        except Exception:
            cur = {}

        patch = {}
        action     = cmd.get("action")
        brightness = cmd.get("brightness")
        b_delta    = cmd.get("brightness_delta")
        color_name = cmd.get("color")
        effect     = cmd.get("effect")
        speed      = cmd.get("speed")
        volume     = cmd.get("volume")
        
        # ---- ON / OFF ----
        if action in ("on", "off"):
            patch["power"] = action
        
        # ---- BRIGHTNESS tuyet doi ----
        if brightness is not None:
            try:
                b = int(brightness)
                b = max(0, min(100, b))
                patch["brightness"] = b

                # nếu đang OFF mà set brightness > 0 -> tự bật
                if b > 0 and patch.get("power") is None and cur.get("power") != "on":
                    patch["power"] = "on"
            except Exception:
                pass

        # ---- Brightness delta (tăng/giảm %) ----
        if b_delta is not None:
            try:
                delta = int(b_delta)
                cur_b = int(cur.get("brightness", 0))
                new_b = max(0, min(100, cur_b + delta))
                patch["brightness"] = new_b
                if new_b > 0 and patch.get("power") is None and cur.get("power") != "on":
                    patch["power"] = "on"
            except Exception:
                pass

        # ---- Màu sắc ----
        if color_name:
            key = str(color_name).strip().lower()
            hex_color = LIGHT_COLOR_HEX.get(key, None)
            if not hex_color and key in LIGHT_COLOR_HEX:
                hex_color = LIGHT_COLOR_HEX[key]
            if not hex_color:
                # nếu parser trả "red" mà không map được thì vẫn ghi raw
                hex_color = color_name
            patch["color"] = hex_color

            # chỉ đổi màu mà chưa on -> auto bật
            if patch.get("power") is None and cur.get("power") != "on":
                patch["power"] = "on"

        # ---- Hiệu ứng rainbow ----
        if effect == "rainbow_on":
            patch["rainbow"] = True
            if patch.get("power") is None and cur.get("power") != "on":
                patch["power"] = "on"
        elif effect == "rainbow_off":
            patch["rainbow"] = False

        # ---- Quạt / Loa ----
        if speed is not None:
            try:
                patch["speed"] = int(speed)
            except Exception:
                pass

        if volume is not None:
            try:
                patch["volume"] = int(volume)
            except Exception:
                pass

        # Nếu chỉ nói "bật đèn" không nói gì khác → dùng brightness cũ hoặc 100
        if action == "on" and "brightness" not in patch:
            cur_b = int(cur.get("brightness", 0) or 0)
            patch["brightness"] = cur_b if cur_b > 0 else 100

        # Nếu tắt → về 0 và tắt rainbow
        if action == "off":
            patch.setdefault("brightness", 0)
            patch.setdefault("rainbow", False)

        patch["source"] = "voice"
        patch["ts"] = int(time.time() * 1000)

        url = f"{DEFAULT_FB_BASE}/devices/{dev_id}/desired.json"
        r = requests.patch(url, json=patch, timeout=4)
        print("🎙 VOICE → DESIRED", dev_id, patch, "code:", r.status_code)
        return {"target": "firebase", "status": "ok", "code": r.status_code, "payload": patch}

    except Exception as e:
        print("💥 _push_desired_from_voice error:", e)
        return {"target": "firebase", "status": "error", "message": str(e)}

if __orig_push_command is not None:
    def _push_command(cmd: dict, sandbox=False):
        # gọi hàm gốc giữ nguyên hành vi /commands.json
        res = __orig_push_command(cmd, sandbox=sandbox)
        # tuỳ chọn: ghi thêm mong muốn vào /desired/*
        if VOICE_WRITE_DESIRED:
            try:
                _push_desired_from_voice(cmd)
            except Exception as _e:
                print("⚠️ VOICE desired bridge error:", _e)
        return res

# ===============================
# RUN WITH NGROK
# ===============================
if __name__ == '__main__':
    public_tunnel = ngrok.connect(5000)
    print("🌐 Public ngrok URL:", getattr(public_tunnel, "public_url", str(public_tunnel)))
    # start scheduler thread
    _start_scheduler_once()
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
