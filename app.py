import base64
import binascii
import heapq
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from types import SimpleNamespace

import cv2
import mediapipe as mp
import numpy as np
import requests
import torch
import whisper
from flask import Flask, abort, jsonify, render_template, request, send_from_directory
from flask_cors import CORS
from pyngrok import ngrok
from sklearn.metrics.pairwise import cosine_similarity

APP_ROOT = Path(__file__).resolve().parent
try:
    from dotenv import load_dotenv

    load_dotenv(APP_ROOT / ".env")
except Exception:
    pass

TF_ENABLE_XLA = os.environ.get("TF_ENABLE_XLA", "0") == "1"
ADMIN_REGISTER_PASSWORD = os.environ.get("ADMIN_REGISTER_PASSWORD", "").strip()
SMART_HOME_PIN = os.environ.get("SMART_HOME_PIN", "").strip()

def _bootstrap_tensorflow_cuda():
    python_tag = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = APP_ROOT / ".venv" / "lib" / python_tag / "site-packages"
    triton_libdevice = site_packages / "triton" / "backends" / "nvidia" / "lib" / "libdevice.10.bc"
    shim_dir = APP_ROOT / "cuda_sdk_lib" / "nvvm" / "libdevice"
    shim_file = shim_dir / "libdevice.10.bc"

    if not shim_file.exists() and triton_libdevice.exists():
        shim_dir.mkdir(parents=True, exist_ok=True)
        try:
            shim_file.symlink_to(triton_libdevice)
        except OSError:
            shutil.copy2(triton_libdevice, shim_file)

    if shim_dir.exists():
        xla_flag = f"--xla_gpu_cuda_data_dir={APP_ROOT / 'cuda_sdk_lib'}"
        current_xla_flags = os.environ.get("XLA_FLAGS", "").strip()
        if xla_flag not in current_xla_flags:
            os.environ["XLA_FLAGS"] = f"{current_xla_flags} {xla_flag}".strip()

    if not TF_ENABLE_XLA:
        current_tf_xla_flags = os.environ.get("TF_XLA_FLAGS", "").strip()
        disable_flag = "--tf_xla_auto_jit=0"
        if disable_flag not in current_tf_xla_flags:
            os.environ["TF_XLA_FLAGS"] = f"{current_tf_xla_flags} {disable_flag}".strip()

_bootstrap_tensorflow_cuda()

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array

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
SAMPLE_IMAGE = os.environ.get("SAMPLE_IMAGE", "").strip()
SAMPLE_NAME = os.environ.get("SAMPLE_NAME", "Demo User").strip()
SAMPLE_CROP = os.environ.get("SAMPLE_CROP", "namtran_crop.jpg").strip()

# Ngưỡng similarity cho nhận diện (0.67 hơi thấp, tăng lên cho an toàn hơn)
SIM_THRESHOLD = 0.7

INPUT_W, INPUT_H = 224, 224
MOBILENET_WEIGHTS = os.environ.get("MOBILENET_WEIGHTS", "imagenet").strip().lower()
MOBILENET_WEIGHTS_PATH = os.environ.get("MOBILENET_WEIGHTS_PATH", "").strip()

# Ngưỡng chất lượng khuôn mặt
MIN_FACE_SIZE    = 80    # chiều rộng/ cao tối thiểu (pixels) mới xử lý
BLUR_THRESHOLD   = 50.0  # độ nhòe tối đa (variance of Laplacian)

# Multi-frame xác nhận (theo IP)
REQ_CONSEC_FRAMES = 3     # cần ≥ 3 frame liên tiếp chuẩn mới cho pass
RECOG_WINDOW_SEC  = 3.0   # trong vòng 3s, nếu đổi người coi như chuỗi mới


# file to persist registered users
REGISTERED_DB = os.environ.get("REGISTERED_DB", "registered.json").strip()
UPLOADS_DIR = os.environ.get("REGISTERED_IMAGES_DIR", "registered_images").strip()
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

def _resolve_mobilenet_weights():
    if MOBILENET_WEIGHTS_PATH:
        if not os.path.exists(MOBILENET_WEIGHTS_PATH):
            raise FileNotFoundError(
                f"Không tìm thấy MobileNet weights tại: {MOBILENET_WEIGHTS_PATH}"
            )
        return MOBILENET_WEIGHTS_PATH
    if MOBILENET_WEIGHTS in ("", "imagenet"):
        return "imagenet"
    if MOBILENET_WEIGHTS in ("none", "random"):
        return None
    raise ValueError(
        "MOBILENET_WEIGHTS chỉ hỗ trợ: 'imagenet', 'none', 'random' hoặc dùng MOBILENET_WEIGHTS_PATH"
    )

def _build_embedding_model():
    weights = _resolve_mobilenet_weights()
    try:
        base_model = MobileNetV2(
            weights=weights,
            include_top=False,
            pooling="avg",
            input_shape=(INPUT_H, INPUT_W, 3),
        )
    except Exception as exc:
        if weights == "imagenet":
            raise RuntimeError(
                "Không tải được MobileNetV2 ImageNet weights. "
                "Nếu máy bị chặn mạng, hãy đặt MOBILENET_WEIGHTS_PATH tới file .h5 local "
                "hoặc chạy với MOBILENET_WEIGHTS=none để bật app ở chế độ dev."
            ) from exc
        raise
    return Model(inputs=base_model.input, outputs=base_model.output)

# Model
model = _build_embedding_model()

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

class _OpenCVFaceDetectionAdapter:
    def __init__(self):
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError(f"Không load được Haar cascade: {cascade_path}")

    def process(self, rgb_img):
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        detections = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE),
        )
        img_h, img_w = rgb_img.shape[:2]
        wrapped = []
        for x, y, w, h in detections:
            wrapped.append(
                SimpleNamespace(
                    location_data=SimpleNamespace(
                        relative_bounding_box=SimpleNamespace(
                            xmin=x / img_w,
                            ymin=y / img_h,
                            width=w / img_w,
                            height=h / img_h,
                        )
                    )
                )
            )
        return SimpleNamespace(detections=wrapped or None)

def _create_face_detector():
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
        print("✅ MediaPipe solutions ready")
        return mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5,
        )

    print("⚠️ mediapipe.solutions không có sẵn, fallback sang OpenCV Haar cascade.")
    return _OpenCVFaceDetectionAdapter()

face_detection = _create_face_detector()

# ===============================
# Known faces storage (in-memory)
# ===============================
known_embeddings, known_names = [], []

# State cho multi-frame recognition (key theo IP)
# { ip: { "name": str | None, "count": int, "ts": float, "last_sim": float } }
RECOG_STATE = {}

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


def load_sample(path, name):
    if not os.path.exists(path):
        print(f"⚠️ load_sample: file not found {path}")
        return False
    img = cv2.imread(path)
    if img is None:
        print("⚠️ load_sample: cv2 couldn't read image")
        return False
    h, w = img.shape[:2]
    res = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.detections:
        print("⚠️ Không phát hiện khuôn mặt trong ảnh mẫu:", path)
        return False
    det = res.detections[0]
    bbox = det.location_data.relative_bounding_box
    x, y = int(bbox.xmin*w), int(bbox.ymin*h)
    bw, bh = int(bbox.width*w), int(bbox.height*h)
    x, y = max(0, x), max(0, y)
    x2, y2 = min(w, x+bw), min(h, y+bh)
    face_crop = img[y:y2, x:x2]
    if face_crop is None or face_crop.size == 0:
        print("⚠️ load_sample: crop empty")
        return False
    emb = get_embedding_from_bgr(face_crop)
    known_embeddings.append(emb)
    known_names.append(name)
    try:
        cv2.imwrite(SAMPLE_CROP, face_crop)
    except Exception:
        pass
    print(f"✅ Đã load sample {name} (crop size: {face_crop.shape[:2]})")
    return True

if SAMPLE_IMAGE and os.path.exists(SAMPLE_IMAGE):
    load_sample(SAMPLE_IMAGE, SAMPLE_NAME)
elif SAMPLE_IMAGE:
    print(f"⚠️ SAMPLE_IMAGE configured but not found: {SAMPLE_IMAGE}")

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
def load_registered_db():
    if not os.path.exists(REGISTERED_DB):
        return
    try:
        with open(REGISTERED_DB, 'r', encoding='utf-8') as f:
            db = json.load(f)
    except Exception as e:
        print("⚠️ Không thể đọc registered.json:", e)
        return
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
                load_sample(target, name)
    print("ℹ️ Đã load registered db (n=", len(known_names), ")")

def save_registered_entry(name, local_file, drive_id=None):
    db = []
    if os.path.exists(REGISTERED_DB):
        try:
            with open(REGISTERED_DB, 'r', encoding='utf-8') as f:
                db = json.load(f)
        except Exception:
            db = []
    db.append({'name': name, 'local_file': local_file, 'drive_id': drive_id})
    try:
        with open(REGISTERED_DB, 'w', encoding='utf-8') as f:
            json.dump(db, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("⚠️ Lỗi lưu registered.json:", e)

load_registered_db()

# ===============================
# VOICE / WHISPER CONFIG
# ===============================
VOICE_LANGUAGE   = "vi"
VOICE_MODEL_NAME = os.environ.get("WHISPER_MODEL", "small")
INITIAL_PROMPT   = ("Ngữ cảnh: điều khiển nhà thông minh. Từ khóa: bật, tắt, NeoPixel, Entry, "
                    "độ sáng, phần trăm, màu đỏ, xanh, trắng, hành lang, cửa, quạt.")

VOICE_EXECUTE_DEFAULT = os.environ.get("VOICE_EXECUTE_DEFAULT", "0") == "1"
VOICE_CONTROL_TOKEN   = os.environ.get("VOICE_CONTROL_TOKEN")

# ===== NEW: Firebase base & URLs =====
DEFAULT_FB_BASE = os.environ.get(
    "FIREBASE_DATABASE_URL",
    "https://do-an-2-91a3c-default-rtdb.asia-southeast1.firebasedatabase.app",
).rstrip("/")
FIREBASE_COMMAND_URL = os.environ.get("FIREBASE_COMMAND_URL", f"{DEFAULT_FB_BASE}/commands.json")
FIREBASE_COMMAND_URL_SANDBOX = os.environ.get("FIREBASE_COMMAND_URL_SANDBOX", f"{DEFAULT_FB_BASE}/commands_sandbox.json")
FIREBASE_DEVICES_URL = os.environ.get("FIREBASE_DEVICES_URL", f"{DEFAULT_FB_BASE}/devices.json")
FIREBASE_DOORS_BASE = os.environ.get("FIREBASE_DOORS_BASE", f"{DEFAULT_FB_BASE}/doors")
HTTP_TIMEOUT_SEC = 4
HTTP_WRITE_TIMEOUT_SEC = 5
CLIENT_CONTEXT_TTL_SEC = 1800
ENABLE_NGROK = os.environ.get("ENABLE_NGROK", "0") == "1"
NGROK_AUTHTOKEN = os.environ.get("NGROK_AUTHTOKEN", "").strip()
HOST = os.environ.get("FLASK_HOST", "0.0.0.0")
PORT = int(os.environ.get("FLASK_PORT", "5000"))

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
    if p.suffix.lower() == ".wav": return str(p)
    out = p.with_suffix(".wav")
    subprocess.run(["ffmpeg","-y","-i",str(p),"-ac","1","-ar","16000",str(out)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return str(out)

def _probe_duration_seconds(path: str) -> float:
    try:
        r = subprocess.run(
            ["ffprobe","-v","error","-show_entries","format=duration",
             "-of","default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=5
        )
        return float(r.stdout.strip())
    except Exception:
        return -1.0

COMMON_FIXES = {
    "bat giup": "bat",
    "bat gium": "bat",
    "tat giup": "tat",
    "tat gium": "tat",
    "mo khoa cua": "mo cua",
    "dong khoa cua": "dong cua",
    "tat het tat ca den": "tat tat ca den",
    "tat het den": "tat tat ca den",
    "bat het tat ca den": "bat tat ca den",
    "bat het den": "bat tat ca den",
    "phong khachs": "phong khach",
    "phong khac": "phong khach",
    "phong ngu": "phong ngu",
    "phong lam viec": "phong lam viec",
    "quat may": "quat",
    "loa bluetooth": "loa",
    "den tran": "den tran",
    "neo pixel": "neopixel",
}

FILLER_WORDS = [
    "giup tui",
    "giup toi",
    "cho tui",
    "cho toi",
    "lam on",
    "voi",
    "nhe",
    "di",
    "a",
]

def normalize_text_for_command(text: str) -> str:
    normalized = _fold_text(text)
    for filler in FILLER_WORDS:
        normalized = re.sub(rf"\b{re.escape(filler)}\b", " ", normalized)
    for wrong, right in COMMON_FIXES.items():
        normalized = normalized.replace(wrong, right)
    return re.sub(r"\s+", " ", normalized).strip()

def _fold_text(t: str) -> str:
    import unicodedata
    t = (t or "").lower().strip()
    t = ''.join(c for c in unicodedata.normalize('NFD', t) if unicodedata.category(c) != 'Mn')
    t = t.replace('đ','d')
    t = re.sub(r'[^a-z0-9 ]+', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()

def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def _normalize_text(value) -> str:
    return (value or "").strip().lower()

def _safe_filename_stem(name: str) -> str:
    safe_name = _fold_text(name).replace(" ", "_")
    safe_name = re.sub(r"[^a-z0-9._-]+", "_", safe_name).strip("._")
    return safe_name or "unknown"

def _decode_base64_image(image_value: str) -> bytes:
    if not image_value:
        raise ValueError("Không có ảnh")

    raw_value = image_value.split(",", 1)[1] if "," in image_value else image_value
    try:
        return base64.b64decode(raw_value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Ảnh base64 không hợp lệ") from exc

def _decode_frame_from_base64(image_value: str):
    img_bytes = _decode_base64_image(image_value)
    npimg = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Không decode được ảnh")
    return frame

def _get_json_data() -> dict:
    return request.get_json(silent=True) or {}

def _load_firebase_json(url: str, timeout: int = HTTP_TIMEOUT_SEC):
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json() or {}

def _device_summary_from_firebase_node(key, node) -> dict:
    node = node or {}
    meta = _device_metadata_from_node(key, node)
    desired = node.get("desired", {}) or {}
    reported = node.get("reported", {}) or {}

    power = (
        desired.get("power")
        if desired.get("power") is not None
        else reported.get("power")
        if reported.get("power") is not None
        else node.get("power")
    )

    return {
        "id": meta["id"],
        "name": meta["name"],
        "type": meta["device"],
        "section": meta["section"] or meta["area"],
        "pin": str(node.get("pin") or meta.get("pin") or ""),
        "state": power == "on" if power is not None else bool(node.get("state")),
        "brightness": desired.get("brightness", reported.get("brightness", node.get("brightness"))),
        "color": desired.get("color", reported.get("color", node.get("color"))),
        "speed": desired.get("speed", reported.get("speed", node.get("speed"))),
        "volume": desired.get("volume", reported.get("volume", node.get("volume"))),
        "temperature": reported.get("temperature", desired.get("temperature", node.get("temperature"))),
        "rainbow": desired.get("rainbow", reported.get("rainbow", node.get("rainbow"))),
        "track_index": desired.get("track_index", reported.get("track_index", node.get("track_index"))),
    }

def _device_metadata_from_node(key, dev) -> dict:
    dev = dev or {}
    meta = dev.get("metadata", {}) or {}
    device = str(
        dev.get("device")
        or dev.get("type")
        or meta.get("device")
        or meta.get("type")
        or "Light"
    )
    return {
        "key": key,
        "id": str(dev.get("id") or meta.get("id") or key),
        "name": str(dev.get("name") or meta.get("name") or key),
        "area": str(
            dev.get("area")
            or dev.get("room")
            or meta.get("area")
            or meta.get("section")
            or "Default"
        ),
        "device": device,
        "brand": str(dev.get("brand") or meta.get("brand") or ""),
        "section": str(meta.get("section") or dev.get("section") or ""),
        "device_fold": _fold_text(device),
    }

def _is_light_device_type(device_type: str) -> bool:
    device_type_fold = _fold_text(device_type)
    return any(token in device_type_fold for token in ("light", "den", "neopixel"))

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
        data = _load_firebase_json(FIREBASE_DEVICES_URL)
    except (requests.RequestException, ValueError):
        data = {}

    if isinstance(data, dict):
        items = data.items()
    else:
        items = []

    for key, dev in items:
        meta = _device_metadata_from_node(key, dev)
        name = meta["name"]
        area = meta["area"]
        dtype = meta["device"]
        brand = meta["brand"]

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
    m = { _fold_text(a): a for a in areas }
    for fa, orig in m.items():
        if fa and fa in sf:
            return orig
    m2 = re.search(r'\b(o|tai|o tai)\s+([a-z0-9 ]{2,})$', sf)
    if m2:
        cand = m2.group(2).strip()
        best,score=None,0
        for fa,orig in m.items():
            s = _sim(cand, fa)
            if s>score: best,score=orig,s
        if score>0.6: return best
    return None

def _best_device_match(text: str, area_hint: str=None):
    cat = _load_device_catalog()
    sf = _fold_text(text)
    if not area_hint:
        area_hint = _infer_area_from_text(sf)
    best,score=None,0.0
    for d in cat:
        syn_score = max((_sim(sf, s) for s in d["syn"]), default=0.0)
        aw = 1.15 if area_hint and _fold_text(d["area"]) == _fold_text(area_hint) else 1.0
        sc = syn_score * aw
        if sc > score:
            best,score=d,sc
    return best if score >= 0.55 else None

_LAST_CTX = {}

def _get_client_id():
    return request.headers.get("X-Client-Id") or request.remote_addr or "anon"

def _remember_ctx(device_id=None, area=None):
    if not device_id and not area:
        return

    client_id = _get_client_id()
    ctx = _LAST_CTX.get(client_id, {})
    if device_id is not None:
        ctx["device_id"] = device_id
    if area:
        ctx["area"] = area
    ctx["ts"] = time.time()
    _LAST_CTX[client_id] = ctx

def _last_ctx():
    client_id = _get_client_id()
    ctx = _LAST_CTX.get(client_id)
    if not ctx:
        return None
    if time.time() - ctx.get("ts", 0) >= CLIENT_CONTEXT_TTL_SEC:
        _LAST_CTX.pop(client_id, None)
        return None
    return ctx

def _last_area():
    ctx = _last_ctx()
    return ctx.get("area") if ctx else None

# ===== NEW: schedule parsing =====
def _parse_schedule_vi(text: str):
    """
    Return (due_ts, friendly) or (None, None) if no schedule found.
    Support: "trong 5 phút/giờ", "sau 10p", "lúc 7:30", "9 giờ tối", "ngày mai lúc 6 giờ".
    """
    s = text.lower().strip()
    sf = _fold_text(s)
    now = datetime.now()

    # relative minutes/hours
    m = re.search(r'\b(trong|sau)\s+(\d+)\s*(phut|p|ph|minute|min|gio|h)\b', sf)
    if m:
        n = int(m.group(2))
        unit = m.group(3)
        delta = timedelta(minutes=n) if unit in ('phut','p','ph','minute','min') else timedelta(hours=n)
        due = now + delta
        return due.timestamp(), f"hẹn sau {n} {'phút' if unit in ('phut','p','ph','minute','min') else 'giờ'}"

    # absolute today/ tomorrow
    is_tomorrow = 'ngay mai' in sf or 'mai' in sf
    # hh:mm or h mm
    m2 = re.search(r'\b(luc|vao)?\s*(\d{1,2})(?:[:h ](\d{1,2}))?\s*(sang|chieu|toi|am|pm)?\b', sf)
    if m2:
        hh = int(m2.group(2))
        mm = int(m2.group(3) or 0)
        ampm = m2.group(4)
        if ampm in ('pm','chieu','toi') and hh < 12: hh += 12
        if ampm in ('am','sang') and hh == 12: hh = 0
        day = now.date() + (timedelta(days=1) if is_tomorrow else timedelta(days=0))
        due = datetime.combine(day, datetime.min.time()).replace(hour=hh, minute=mm)
        if due < now and not is_tomorrow:
            due = due + timedelta(days=1)
        label = f"hẹn lúc {due.strftime('%H:%M')}" + (" ngày mai" if is_tomorrow else "")
        return due.timestamp(), label

    return None, None

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
    COLORS = {
        "do": "red",
        "do tuoi": "red",
        "xanh la": "green",
        "la": "green",
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
        "den": "black",
    }

    color = None
    for k, v in COLORS.items():
        if k in sf:
            color = v
            break

    # ----- Area hint + chọn device -----
    area_hint = _infer_area_from_text(sf) or _last_area()
    devmatch = _best_device_match(s_raw, area_hint)

    if devmatch:
        device = devmatch["device"]
        area = devmatch["area"]
        device_id = devmatch.get("id") or devmatch.get("key") or f"{device}_{area}"
    else:
        device = "Light"
        area = area_hint or "Default"
        device_id = f"{device}_{area}"

    # nhớ khu vực cuối cùng
    _remember_ctx(area=area)

    # nếu chỉnh brightness/color mà chưa có action thì xem như "set"
    if action is None and (brightness is not None or color is not None or brightness_delta is not None):
        action = "set"

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
        "raw_text": s_raw,
    }
    return cmd


# ===== Việt Nam delay parser =====
def _extract_delay_vi(text: str):
    t = text.lower()
    # sau 30 giây
    m = re.search(r"sau\s+(\d+)\s*(giay|giây|s)", t)
    if m:
        return int(m.group(1))

    # sau 2 phút
    m = re.search(r"sau\s+(\d+)\s*(phut|phút|p)", t)
    if m:
        return int(m.group(1)) * 60

    # trong 5 giây
    m = re.search(r"trong\s+(\d+)\s*(giay|giây|s)", t)
    if m:
        return int(m.group(1))

    # trong 3 phút
    m = re.search(r"trong\s+(\d+)\s*(phut|phút|p)", t)
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
        r = requests.post(url, json=payload, timeout=HTTP_TIMEOUT_SEC)
        r.raise_for_status()
        return {"target": "firebase", "status": "ok", "code": r.status_code, "payload": payload}
    except requests.RequestException as e:
        return {"target": "firebase", "status": "error", "message": str(e), "payload": payload}

def _patch_firebase(url: str, *, json_body=None, raw_body=None, timeout: int = HTTP_WRITE_TIMEOUT_SEC):
    response = requests.patch(url, json=json_body, data=raw_body, timeout=timeout)
    response.raise_for_status()
    return response

def _normalize_desired_payload(desired: dict) -> dict:
    normalized = dict(desired or {})
    if "state" in normalized and isinstance(normalized["state"], bool):
        normalized["power"] = "on" if normalized.pop("state") else "off"
    return normalized

def _build_scene_patches(scene: str, scene_data: dict):
    actions = scene_data.get("actions")
    rules = scene_data.get("rules")
    patches = []

    if actions:
        for act in actions:
            dev_id = act.get("device")
            desired = act.get("desired") or {}
            if dev_id and isinstance(desired, dict):
                patches.append((dev_id, desired))
        return patches

    if not rules:
        return None

    devices = _load_firebase_json(FIREBASE_DEVICES_URL)
    if not isinstance(devices, dict):
        return []

    for key, node in devices.items():
        meta = _device_metadata_from_node(key, node)
        dev_type = _normalize_text(meta["device"])
        section = _normalize_text(meta["section"])

        for rule in rules:
            match = rule.get("match") or {}
            m_type = _normalize_text(match.get("type"))
            m_section = _normalize_text(match.get("section"))
            if m_type and dev_type != m_type:
                continue
            if m_section and section != m_section:
                continue

            desired = dict(rule.get("desired") or {})
            if not desired:
                continue

            desired.setdefault("ts", int(time.time() * 1000))
            desired.setdefault("updated_by", f"scene:{scene}")
            patches.append((meta["id"], desired))

    return patches

def _build_bulk_light_commands(action: str, text: str):
    try:
        devices = _load_firebase_json(FIREBASE_DEVICES_URL)
    except (requests.RequestException, ValueError) as exc:
        print("⚠️ BULK_LIGHTS: không đọc được /devices:", exc)
        return []

    if not isinstance(devices, dict):
        return []

    commands = []
    for key, dev in devices.items():
        meta = _device_metadata_from_node(key, dev)
        if not _is_light_device_type(meta["device"]):
            continue

        command = {
            "device_id": meta["id"],
            "device": "Light",
            "area": meta["area"],
            "action": action,
            "raw_text": text,
        }
        if action == "off":
            command["brightness"] = 0
        commands.append(command)

    return commands

def _apply_delay_to_commands(commands, delay_seconds):
    if not delay_seconds:
        return

    for command in commands:
        command["delay_seconds"] = delay_seconds

def _run_bulk_commands(commands, execute: bool, sandbox: bool, log_prefix: str):
    if not execute:
        return []

    results = []
    for command in commands:
        try:
            results.append(_push_command(command, sandbox=sandbox))
        except Exception as exc:
            print(f"⚠️ {log_prefix} push error:", exc)
    return results

def _cleanup_temp_files(paths):
    for path in paths:
        if not path:
            continue
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            print("⚠️ temp cleanup failed:", exc)

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
    if _sched_started: return
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
@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/home')
@app.route('/home.html')
def home():
    return render_template('home.html')

@app.route('/assets/<path:filename>')
def frontend_asset(filename):
    allowed_assets = {
        "firebase.js",
        "scenes.css",
        "scenes.js",
        "style.css",
    }
    if filename not in allowed_assets:
        abort(404)
    return send_from_directory(os.path.join(app.root_path, "templates"), filename)

@app.route('/manifest.webmanifest')
def manifest():
    return send_from_directory(
        os.path.join(app.root_path, "static"),
        "manifest.webmanifest",
        mimetype="application/manifest+json",
    )

@app.route('/sw.js')
def service_worker():
    return send_from_directory(
        os.path.join(app.root_path, "static"),
        "sw.js",
        mimetype="application/javascript",
    )

@app.route('/api/devices')
def api_devices():
    try:
        data = _load_firebase_json(FIREBASE_DEVICES_URL)
        if not isinstance(data, dict):
            data = {}
        devices = [_device_summary_from_firebase_node(key, node) for key, node in data.items()]
        return jsonify({"status": "ok", "devices": devices, "count": len(devices)})
    except requests.RequestException as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None) or 502
        return jsonify({"status": "error", "message": str(exc), "devices": []}), status_code

@app.route('/execute_scene', methods=['POST'])
def execute_scene():
    """
    Áp dụng scene:
    - Ưu tiên dùng scene.actions nếu có (kiểu cũ: device + desired)
    - Nếu không có actions thì dùng scene.rules (kiểu mới: match type/section)
    """
    data = _get_json_data()
    scene = data.get("scene")

    # Khi clearScene() gửi scene = null -> chỉ trả về ok, không làm gì
    if not scene:
        return jsonify({"status": "cleared"})

    scene_url = f"{DEFAULT_FB_BASE}/scenes/{scene}.json"
    try:
        scene_data = _load_firebase_json(scene_url)
    except requests.RequestException as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if status_code == 404:
            return jsonify({"error": "Scene not found"}), 404
        return jsonify({"error": "Failed to load scene", "details": str(exc)}), 502
    except ValueError as exc:
        return jsonify({"error": "Scene payload is invalid", "details": str(exc)}), 502

    if not scene_data:
        return jsonify({"error": "Scene not found"}), 404

    try:
        patches = _build_scene_patches(scene, scene_data)
    except requests.RequestException as exc:
        return jsonify({"error": "Failed to load devices", "details": str(exc)}), 502
    except ValueError as exc:
        return jsonify({"error": "Device payload is invalid", "details": str(exc)}), 502

    if patches is None:
        return jsonify({"error": "Scene has neither actions nor rules"}), 400

    # ----- 3. Gửi PATCH lên từng thiết bị -----
    results = {}
    for dev_id, desired in patches:
        try:
            payload = _normalize_desired_payload(desired)
            r = _patch_firebase(
                f"{DEFAULT_FB_BASE}/devices/{dev_id}/desired.json",
                json_body=payload,
            )
            results[dev_id] = r.status_code
        except requests.RequestException as exc:
            results[dev_id] = f"error: {exc}"

    return jsonify({"status": "ok", "scene": scene, "patched": results})

@app.route('/upload', methods=['POST'])
def upload():
    """
    Nhận 1 frame (ảnh) từ client, nhận diện khuôn mặt.

    Nâng cấp:
      - Bỏ qua mặt quá nhỏ / quá mờ.
      - Dùng SIM_THRESHOLD = 0.70 (chặt hơn bản cũ 0.67).
      - Yêu cầu N frame liên tiếp (theo IP) đều đạt ngưỡng mới cho pass.
    """
    try:
        data = _get_json_data()
        img_base64 = data.get('image')
        if not img_base64:
            return jsonify({'error': 'Không có ảnh'}), 400

        try:
            frame = _decode_frame_from_base64(img_base64)
        except ValueError as exc:
            return jsonify({'error': str(exc)}), 400

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
            "last_sim": -1.0
        })

        final_name = "Unknown"

        if raw_name != "Unknown" and best_sim >= SIM_THRESHOLD:
            # nếu còn trong window & cùng 1 tên -> tăng count
            if st["name"] == raw_name and (now - st["ts"]) <= RECOG_WINDOW_SEC:
                st["count"] += 1
            else:
                # bắt đầu chuỗi mới
                st = {
                    "name": raw_name,
                    "count": 1,
                    "ts": now,
                    "last_sim": best_sim
                }

            st["ts"] = now
            st["last_sim"] = best_sim
            RECOG_STATE[client_ip] = st

            if st["count"] >= REQ_CONSEC_FRAMES:
                final_name = raw_name
            else:
                final_name = "Unknown"
        else:
            # không nhận diện được / similarity thấp -> reset
            RECOG_STATE[client_ip] = {
                "name": None,
                "count": 0,
                "ts": now,
                "last_sim": best_sim
            }
            final_name = "Unknown"

        streak = RECOG_STATE.get(client_ip, {}).get("count", 0)
        print(
            f"📸 Kết quả: raw={raw_name}, final={final_name}, "
            f"sim={best_sim:.2f}, streak={streak}"
        )

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
        data = _get_json_data()
        name = (data.get('name') or "Unknown").strip()
        img_base64 = data.get('image')
        admin_pass = (data.get("admin_password") or "").strip()

        if not ADMIN_REGISTER_PASSWORD:
            return jsonify(
                {
                    "status": "error",
                    "code": "access_control_not_configured",
                    "message": "ADMIN_REGISTER_PASSWORD chưa được cấu hình.",
                }
            ), 503

        if admin_pass != ADMIN_REGISTER_PASSWORD:
            return jsonify(
                {
                    "status": "error",
                    "code": "bad_admin_password",
                    "message": "Sai mật khẩu admin, không được phép đăng ký khuôn mặt mới.",
                }
            ), 403

        if not img_base64:
            return jsonify({'status': 'error', 'message': 'Không có ảnh'}), 400

        try:
            img_bytes = _decode_base64_image(img_base64)
        except ValueError as exc:
            return jsonify({'status': 'error', 'message': str(exc)}), 400

        timestamp = int(time.time())
        safe_name = _safe_filename_stem(name)
        filename = os.path.join(UPLOADS_DIR, f"{safe_name}_{timestamp}.jpg")
        with open(filename, 'wb') as f:
            f.write(img_bytes)

        drive_id = None
        if DRIVE_AVAILABLE:
            drive_id = upload_file_to_drive(filename, title=os.path.basename(filename))
            if drive_id:
                print(f"✅ Uploaded {filename} to Drive id={drive_id}")
            else:
                print("ℹ️ Upload to Drive failed or returned None.")

        ok = load_sample(filename, name)
        if not ok:
            return jsonify({'status': 'error', 'message': 'Không thể tạo embedding từ ảnh đã tải lên.'}), 500

        save_registered_entry(name, filename, drive_id)

        return jsonify({'status': 'success', 'message': f'Đăng ký thành công: {name}', 'file': filename, 'drive_id': drive_id})
    except Exception as e:
        print("⚠️ Lỗi register:", e)
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/pin_login', methods=['POST'])
def pin_login():
    try:
        data = _get_json_data()
        pin = str(data.get("pin") or "").strip()

        if not pin:
            return jsonify(
                {
                    "status": "error",
                    "code": "missing_pin",
                    "message": "Thiếu mã PIN.",
                }
            ), 400

        if not SMART_HOME_PIN:
            return jsonify(
                {
                    "status": "error",
                    "code": "access_control_not_configured",
                    "message": "SMART_HOME_PIN chưa được cấu hình.",
                }
            ), 503

        if pin != SMART_HOME_PIN:
            return jsonify(
                {
                    "status": "error",
                    "code": "bad_pin",
                    "message": "Sai mã PIN.",
                }
            ), 403

        return jsonify({"status": "success", "name": "Admin"})
    except Exception as e:
        print("⚠️ Lỗi pin_login:", e)
        return jsonify(
            {
                "status": "error",
                "code": "exception",
                "message": str(e),
            }
        ), 500

@app.route('/voice-door', methods=['POST'])
def voice_door():
    data = _get_json_data()
    if not data:
        return jsonify({"error": "Thiếu payload JSON"}), 400

    device = _normalize_text(data.get("device"))
    action = _normalize_text(data.get("action"))

    door_aliases = {
        "frontDoor": ("front", "trước", "main"),
        "sideDoor": ("side", "sau", "hông", "phụ"),
    }
    door_id = next(
        (door_key for door_key, aliases in door_aliases.items() if any(alias in device for alias in aliases)),
        None,
    )
    if not door_id:
        return jsonify({"error": "Unknown door"}), 400

    state = {
        "open": "open",
        "mở": "open",
        "close": "closed",
        "đóng": "closed",
        "shut": "closed",
        "lock": "closed",
        "khóa": "closed",
        "unlock": "open",
        "mở khóa": "open",
    }.get(action)
    if not state:
        return jsonify({"error": "Unknown action"}), 400

    ref = f"{FIREBASE_DOORS_BASE}/{door_id}/state.json"
    try:
        _patch_firebase(ref, raw_body=json.dumps(state, ensure_ascii=False), timeout=HTTP_TIMEOUT_SEC)
    except requests.RequestException as exc:
        return jsonify({"error": "Failed to update door", "details": str(exc)}), 502

    return jsonify({"status": "ok", "door": door_id, "state": state})

@app.route('/voice', methods=['POST'])
def voice():
    # Lazy init Whisper
    if not _init_whisper_safe():
        return jsonify({"status": "error", "message": "Whisper model chưa sẵn sàng"}), 500

    data_json = _get_json_data()
    if "audio" not in request.files and not data_json:
        return jsonify({"status": "error", "message": "Thiếu audio hoặc text"}), 400

    # --- Gating (execute/token) ---
    execute = VOICE_EXECUTE_DEFAULT
    if request.headers.get("X-Execute", "").strip() == "1":
        execute = True
    if request.form.get("execute", "").strip() == "1":
        execute = True
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

    temp_paths = []
    try:
        # --- Get text ---
        if "audio" in request.files:
            f = request.files["audio"]
            tmp = tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(f.filename).suffix or ".webm"
            )
            f.save(tmp.name)
            tmp.close()
            temp_paths.append(tmp.name)
            print(f"🎧 Nhận audio: {f.filename} -> {tmp.name}")

            t_ff = time.time()
            wav_path = _ensure_wav_16k_mono(tmp.name)
            if wav_path != tmp.name:
                temp_paths.append(wav_path)
            dur = _probe_duration_seconds(wav_path)
            print(f"🎛️ ffmpeg -> {wav_path} (dur={dur:.2f}s) trong {time.time()-t_ff:.2f}s")

            print(
                f"🟡 Transcribe start (device={_whisper_device}, model={VOICE_MODEL_NAME}, execute={execute})"
            )
            t_stt = time.time()
            res = WHISPER_MODEL.transcribe(
                wav_path,
                language=VOICE_LANGUAGE,
                initial_prompt=INITIAL_PROMPT,
                temperature=0,
                beam_size=1,
                condition_on_previous_text=False,
                fp16=(_whisper_device == "cuda"),
                no_speech_threshold=0.4,
                logprob_threshold=-1.0,
                compression_ratio_threshold=2.4,
            )
            raw_text = (res.get("text") or "").strip()
            text = normalize_text_for_command(raw_text)
            print(f"🟢 Transcribe xong trong {time.time()-t_stt:.2f}s → raw='{raw_text}' | norm='{text}'")
        else:
            raw_text = (data_json.get("text") or "").strip()
            text = normalize_text_for_command(raw_text)
            print(f"📝 Text client gửi: raw='{raw_text}' | norm='{text}' (execute={execute})")

        if not text:
            return jsonify(
                {"status": "error", "message": "Không có nội dung để phân tích"}
            ), 400

        client_text = raw_text or text

        # --- Parse command & schedule (mặc định 1 thiết bị) ---
        cmd = _parse_command_vi(text)
        print("📦 Parsed:", cmd)

        # --- xử lý riêng câu "tắt hết / bật hết tất cả đèn" ---
        sf_all = _fold_text(text)

        bulk_light_intent = None
        if (
            "tat het tat ca den" in sf_all
            or "tat het den" in sf_all
            or "tat tat ca den" in sf_all
            or ("tat het" in sf_all and "den" in sf_all)
        ):
            bulk_light_intent = {"action": "off", "intent": "all_lights_off", "log_prefix": "ALL_OFF"}
        elif (
            "bat het tat ca den" in sf_all
            or "bat het den" in sf_all
            or "bat tat ca den" in sf_all
            or ("bat het" in sf_all and "den" in sf_all)
        ):
            bulk_light_intent = {"action": "on", "intent": "all_lights_on", "log_prefix": "ALL_ON"}

        if bulk_light_intent:
            print(f"💡 {bulk_light_intent['log_prefix']} intent:", sf_all)
            cmds = _build_bulk_light_commands(bulk_light_intent["action"], text)
            print(f"🔎 {bulk_light_intent['log_prefix']}: tìm được {len(cmds)} thiết bị đèn.")

            delay_all = _extract_delay_vi(text)
            _apply_delay_to_commands(cmds, delay_all)
            if delay_all:
                print(
                    f"⏱ {bulk_light_intent['log_prefix']} delay {delay_all} giây cho {len(cmds)} thiết bị."
                )

            results = _run_bulk_commands(cmds, execute, sandbox, bulk_light_intent["log_prefix"])
            multi_cmd = {
                "multi": cmds,
                "action": f"all_{bulk_light_intent['action']}",
                "raw_text": text,
            }

            return jsonify(
                {
                    "status": "ok",
                    "execute": execute,
                    "text": client_text,
                    "intent": bulk_light_intent["intent"],
                    "command": multi_cmd,
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
            or cmd.get("speed") is not None
            or cmd.get("speed_op") is not None
            or cmd.get("volume") is not None
            or cmd.get("volume_op") is not None
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
                "text": client_text,
                "command": cmd,
                "scheduled": scheduled,
                "push_result": pushed,
                "sandbox": bool(sandbox),
            }
        )
    except Exception as e:
        print("💥 Voice pipeline error:", e)
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        _cleanup_temp_files(temp_paths)


try:
    __orig_push_command = _push_command
except NameError:
    __orig_push_command = None

LIGHT_COLOR_HEX = {
    "red": "#ff0000",
    "green": "#00ff00",
    "blue": "#0000ff",
    "yellow": "#ffff00",
    "white": "#ffffff",
    "warm white": "#fff3e0",
    "cool white": "#e0f7ff",
    "orange": "#ffa500",
    "purple": "#8000ff",
    "pink": "#ff69b4",
    "cyan": "#00ffff",
    "amber": "#ffc107",
    "brown": "#795548",
    "black": "#000000",
    "do": "#ff0000",
    "xanh la": "#00ff00",
    "xanh duong": "#0000ff",
    "vang": "#ffff00",
    "trang": "#ffffff",
    "cam": "#ffa500",
    "tim": "#8000ff",
    "hong": "#ff69b4",
    "ngoc": "#00ffff",
}

VOICE_WRITE_DESIRED = os.environ.get("VOICE_WRITE_DESIRED", "1") == "1"

def _push_desired_from_voice(cmd: dict) -> dict:
    try:
        dev_id = cmd.get("device_id")
        if not dev_id:
            return {"status": "skip", "reason": "missing device_id"}

        try:
            node = _load_firebase_json(f"{DEFAULT_FB_BASE}/devices/{dev_id}.json", timeout=2)
        except Exception:
            node = {}

        desired = node.get("desired", {}) if isinstance(node, dict) else {}
        reported = node.get("reported", {}) if isinstance(node, dict) else {}
        meta = _device_metadata_from_node(dev_id, node if isinstance(node, dict) else {})
        current = dict(reported or {})
        current.update(desired or {})

        dev_type = _fold_text(str(cmd.get("device") or meta.get("device") or ""))
        is_light = any(token in dev_type for token in ("light", "den", "lamp", "neopixel"))
        is_fan = any(token in dev_type for token in ("fan", "quat"))
        is_speaker = any(token in dev_type for token in ("speaker", "loa", "audio", "am thanh"))

        patch = {}
        action = cmd.get("action")
        brightness = cmd.get("brightness")
        brightness_delta = cmd.get("brightness_delta")
        color_name = cmd.get("color")
        speed = cmd.get("speed")
        speed_op = cmd.get("speed_op")
        volume = cmd.get("volume")
        volume_op = cmd.get("volume_op")

        if action in ("on", "off"):
            patch["power"] = action

        if is_light:
            if brightness is not None:
                try:
                    patch["brightness"] = max(0, min(100, int(brightness)))
                except Exception:
                    pass

            if brightness_delta is not None:
                try:
                    current_brightness = int(current.get("brightness", 0) or 0)
                    patch["brightness"] = max(0, min(100, current_brightness + int(brightness_delta)))
                except Exception:
                    pass

            if color_name:
                color_key = str(color_name).strip().lower()
                patch["color"] = LIGHT_COLOR_HEX.get(color_key, color_name)
                if patch.get("power") is None and current.get("power") != "on":
                    patch["power"] = "on"

            if action == "on" and "brightness" not in patch:
                try:
                    current_brightness = int(current.get("brightness", 0) or 0)
                except Exception:
                    current_brightness = 0
                patch["brightness"] = current_brightness if current_brightness > 0 else 100
            elif action == "off":
                patch["brightness"] = 0
                patch["rainbow"] = False

            if patch.get("brightness", 0) > 0 and patch.get("power") is None:
                patch["power"] = "on"

        if is_fan:
            try:
                current_speed = int(current.get("speed", 0) or 0)
            except Exception:
                current_speed = 0

            if speed is not None and not speed_op:
                try:
                    patch["speed"] = max(0, min(3, int(speed)))
                except Exception:
                    pass
            elif speed_op:
                try:
                    step = max(1, int(speed or 1))
                except Exception:
                    step = 1
                new_speed = current_speed + step if speed_op == "inc" else current_speed - step
                patch["speed"] = max(0, min(3, new_speed))

            if action == "on" and "speed" not in patch:
                patch["speed"] = current_speed if current_speed > 0 else 1
            elif action == "off":
                patch["speed"] = 0

            if "speed" in patch and patch.get("power") is None:
                patch["power"] = "on" if patch["speed"] > 0 else "off"

        if is_speaker:
            try:
                current_volume = int(current.get("volume", 0) or 0)
            except Exception:
                current_volume = 0

            if volume is not None and not volume_op:
                try:
                    patch["volume"] = max(0, min(100, int(volume)))
                except Exception:
                    pass
            elif volume_op:
                try:
                    step = max(1, int(volume or 5))
                except Exception:
                    step = 5
                new_volume = current_volume + step if volume_op == "inc" else current_volume - step
                patch["volume"] = max(0, min(100, new_volume))

            if cmd.get("track_index") is not None:
                try:
                    patch["track_index"] = max(0, int(cmd["track_index"]))
                except Exception:
                    pass

            if action == "on" and "volume" not in patch:
                patch["volume"] = current_volume if current_volume > 0 else 50
            elif action == "off":
                patch["volume"] = 0

            if "volume" in patch and patch.get("power") is None:
                patch["power"] = "on" if patch["volume"] > 0 else "off"

        if "power" in patch:
            patch["state"] = patch["power"] == "on"

        if not patch:
            return {"status": "skip", "reason": "empty_patch", "device_id": dev_id}

        patch["source"] = "voice"
        patch["updated_by"] = "voice"
        patch["ts"] = int(time.time() * 1000)

        url = f"{DEFAULT_FB_BASE}/devices/{dev_id}/desired.json"
        r = _patch_firebase(url, json_body=patch, timeout=HTTP_TIMEOUT_SEC)
        return {"target": "firebase", "status": "ok", "code": r.status_code, "payload": patch}
    except requests.RequestException as e:
        return {"target": "firebase", "status": "error", "message": str(e)}

if __orig_push_command is not None:
    def _push_command(cmd: dict, sandbox=False):
        # gọi hàm gốc giữ nguyên hành vi /commands.json
        res = __orig_push_command(cmd, sandbox=sandbox)
        # tuỳ chọn: ghi thêm mong muốn vào /desired/*
        if VOICE_WRITE_DESIRED:
            try:
                desired_res = _push_desired_from_voice(cmd)
                if isinstance(res, dict):
                    res["desired_result"] = desired_res
            except Exception as _e:
                print("⚠️ VOICE desired bridge error:", _e)
        return res

def _start_ngrok_tunnel(port: int):
    if not ENABLE_NGROK:
        print("ℹ️ ENABLE_NGROK=0 → bỏ qua ngrok, chỉ chạy local server.")
        return None

    try:
        if NGROK_AUTHTOKEN:
            ngrok.set_auth_token(NGROK_AUTHTOKEN)
        public_tunnel = ngrok.connect(port)
        print("🌐 Public ngrok URL:", getattr(public_tunnel, "public_url", str(public_tunnel)))
        return public_tunnel
    except Exception as exc:
        print(f"⚠️ Không mở được ngrok: {exc}")
        print("ℹ️ App sẽ tiếp tục chạy local. Muốn bật ngrok, hãy set ENABLE_NGROK=1 và NGROK_AUTHTOKEN hợp lệ.")
        return None

# ===============================
# RUN WITH NGROK
# ===============================
if __name__ == '__main__':
    _start_ngrok_tunnel(PORT)
    # start scheduler thread
    _start_scheduler_once()
    print(f"🚀 Flask server chạy tại http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=False, use_reloader=False)
