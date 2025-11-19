from flask import Flask, request, jsonify, render_template
import cv2, numpy as np, base64, os, tensorflow as tf, mediapipe as mp, json, time
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
from pyngrok import ngrok
import time, re, subprocess, tempfile, requests
from pathlib import Path
import torch, whisper

# ===== NEW: scheduling & helpers (no external deps) =====
import threading, heapq, uuid
from datetime import datetime, timedelta

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
SIM_THRESHOLD = 0.67
INPUT_W, INPUT_H = 224, 224

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

if os.path.exists(SAMPLE_IMAGE):
    load_sample(SAMPLE_IMAGE, "Nam Trần")

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

# ===== NEW: text folding + fuzzy =====
from difflib import SequenceMatcher
def _fold_text(t: str) -> str:
    import unicodedata
    t = (t or "").lower().strip()
    t = ''.join(c for c in unicodedata.normalize('NFD', t) if unicodedata.category(c) != 'Mn')
    t = t.replace('đ','d')
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

# ===== NEW: context memory per-client =====
_LAST_CTX = {}
def _get_client_id():
    return request.headers.get("X-Client-Id") or request.remote_addr or "anon"
def _remember_ctx(device_id: str, area: str):
    _LAST_CTX[_get_client_id()] = {"device_id": device_id, "area": area, "ts": time.time()}
def _last_area():
    ctx = _LAST_CTX.get(_get_client_id())
    if not ctx: return None
    return ctx["area"] if (time.time()-ctx["ts"]<1800) else None

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

# ===============================
# AREA MEMORY (ghi nhớ khu vực cuối)
# ===============================
_LAST_AREA = None

def _remember_area(area: str):
    """Lưu khu vực cuối cùng mà giọng nói vừa nhắc tới."""
    global _LAST_AREA
    if area:
        _LAST_AREA = area

def _last_area():
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
    _remember_area(area)

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
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        data = request.get_json()
        img_base64 = data.get('image')
        if not img_base64:
            return jsonify({'error': 'Không có ảnh'}), 400

        img_bytes = base64.b64decode(img_base64.split(',')[1])
        npimg = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'error': 'Không decode được ảnh'}), 400

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb)

        name, best_sim = "Unknown", -1.0
        if results.detections:
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                x, y = int(bbox.xmin*w), int(bbox.ymin*h)
                bw, bh = int(bbox.width*w), int(bbox.height*h)
                x, y = max(0, x), max(0, y)
                x2, y2 = min(w, x+bw), min(h, y+bh)
                face_crop = frame[y:y2, x:x2]
                if face_crop is None or face_crop.size == 0:
                    continue
                emb = get_embedding_from_bgr(face_crop)
                for k_emb, k_name in zip(known_embeddings, known_names):
                    sim = float(cosine_similarity([emb], [k_emb])[0][0])
                    if sim > best_sim:
                        best_sim = sim
                        name = k_name if sim >= SIM_THRESHOLD else "Unknown"

        print(f"📸 Kết quả: {name} ({best_sim:.2f})")
        return jsonify({"recognized": name, "similarity": best_sim, "name": name})
    except Exception as e:
        print("⚠️ Lỗi xử lý:", e)
        return jsonify({'error': str(e)}), 500

@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        name = (data.get('name') or "Unknown").strip()
        img_base64 = data.get('image')
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
                print("ℹ️ Upload to Drive failed or returned None.")

        ok = load_sample(filename, name)
        if not ok:
            return jsonify({'status': 'error', 'message': 'Không thể tạo embedding từ ảnh đã tải lên.'}), 500

        save_registered_entry(name, filename, drive_id)

        return jsonify({'status': 'success', 'message': f'Đăng ký thành công: {name}', 'file': filename, 'drive_id': drive_id})
    except Exception as e:
        print("⚠️ Lỗi register:", e)
        return jsonify({'status': 'error', 'message': str(e)}), 500

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
            print(f"🎧 Nhận audio: {f.filename} -> {tmp.name}")

            t_ff = time.time()
            wav_path = _ensure_wav_16k_mono(tmp.name)
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
                beam_size=5,
                condition_on_previous_text=False,
                fp16=(_whisper_device == "cuda"),
            )
            text = (res.get("text") or "").strip()
            print(f"🟢 Transcribe xong trong {time.time()-t_stt:.2f}s → '{text}'")
        else:
            text = (data_json.get("text") or "").strip()
            print(f"📝 Text client gửi: '{text}' (execute={execute})")

        if not text:
            return jsonify(
                {"status": "error", "message": "Không có nội dung để phân tích"}
            ), 400

        # --- Parse command & schedule (mặc định 1 thiết bị) ---
        cmd = _parse_command_vi(text)
        print("📦 Parsed:", cmd)

        # --- xử lý riêng câu "tắt hết / bật hết tất cả đèn" ---
        sf_all = _fold_text(text)

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

            # 🔴 NEW: parse delay cho lệnh "tắt tất cả"
            # Ví dụ: "tắt hết tất cả đèn sau 5 giây / 5s / 5 phút"
            delay_all = _extract_delay_vi(text)
            if delay_all:
                for c in cmds:
                    c["delay_seconds"] = delay_all
                print(f"⏱ ALL_OFF delay {delay_all} giây cho {len(cmds)} thiết bị.")

            results = []
            if execute and cmds:
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
            print("💡 ALL LIGHTS ON intent:", sf_all)

            try:
                r_devs = requests.get(FIREBASE_DEVICES_URL, timeout=4)
                devs = r_devs.json() or {}
            except Exception as e_all_on:
                print("⚠️ ALL_ON: không đọc được /devices:", e_all_on)
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

            print(f"🔎 ALL_ON: tìm được {len(cmds)} thiết bị đèn.")
            
            # 🔴 NEW: parse delay cho lệnh "bật tất cả"
            # Ví dụ: "bật hết tất cả đèn sau 5 giây / 5s / 5 phút"
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
        print("💥 Voice pipeline error:", e)
        return jsonify({"status": "error", "message": str(e)}), 500


try:
    __orig_push_command = _push_command
except NameError:
    __orig_push_command = None

VOICE_WRITE_DESIRED = os.environ.get("VOICE_WRITE_DESIRED","0") == "1"

def _push_desired_from_voice(cmd: dict) -> dict:
    try:
        dev_id = cmd.get("device_id")
        if not dev_id:
            return {"status": "skip", "reason": "missing device_id"}

        patch = {}
        if cmd.get("action") in ("on", "off"):
            patch["power"] = cmd["action"]

        if cmd.get("brightness") is not None:
            try:
                patch["brightness"] = int(cmd["brightness"])
            except Exception:
                pass

        if cmd.get("color"):
            patch["color"] = cmd["color"]

        # NEW: quạt / loa
        if cmd.get("speed") is not None:
            try:
                patch["speed"] = int(cmd["speed"])
            except Exception:
                pass

        if cmd.get("volume") is not None:
            try:
                patch["volume"] = int(cmd["volume"])
            except Exception:
                pass

        patch["source"] = "voice"
        patch["ts"] = int(time.time() * 1000)

        url = f"{DEFAULT_FB_BASE}/devices/{dev_id}/desired.json"
        r = requests.patch(url, json=patch, timeout=4)
        return {"target": "firebase", "status": "ok", "code": r.status_code, "payload": patch}
    except Exception as e:
        return {"target": "firebase", "status": "error", "message": str(e)}

if __orig_push_command is not None:
    def _push_command(cmd: dict, sandbox=False):
        # gọi hàm gốc giữ nguyên hành vi /commands.json
        res = __orig_push_command(cmd, sandbox=sandbox)
        # tuỳ chọn: ghi thêm mong muốn vào /desired/*
        if VOICE_WRITE_DESIRED:
            try: _push_desired_from_voice(cmd)
            except Exception as _e: print("⚠️ VOICE desired bridge error:", _e)
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
