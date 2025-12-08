from flask import Flask, request, jsonify
import json
import threading
from pathlib import Path

app = Flask(__name__)

# ====== FILE LƯU STATE LOCAL ======
BASE_DIR = Path(__file__).resolve().parent
STATE_PATH = BASE_DIR / "local_state.json"
_state_lock = threading.Lock()

DEFAULT_STATE = {
    "devices": {},  # giống /devices trên Firebase
    "doors": {},    # giống /doors
    "scenes": {}    # nếu sau này muốn lưu scene local
}


def load_state():
    """Đọc state từ local_state.json, nếu lỗi thì trả về state mặc định."""
    if not STATE_PATH.exists():
        return DEFAULT_STATE.copy()

    try:
        with STATE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return DEFAULT_STATE.copy()

        # đảm bảo đủ 3 key chính
        for k, v in DEFAULT_STATE.items():
            data.setdefault(k, v.copy())
        return data
    except Exception:
        # nếu file hỏng / JSON lỗi thì reset
        return DEFAULT_STATE.copy()


def save_state(state):
    """Ghi state ra file (có lock + file tạm để tránh hỏng file)."""
    with _state_lock:
        tmp_path = STATE_PATH.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        tmp_path.replace(STATE_PATH)


# ================== ROUTES CƠ BẢN ==================

@app.route("/health", methods=["GET"])
def health():
    """Check nhanh server còn sống không."""
    return jsonify({"status": "ok"}), 200


# ---- DEVICES ----

@app.route("/devices", methods=["GET"])
def get_devices():
    """
    ESP32 (hoặc webapp) gọi GET /devices
    → trả về toàn bộ JSON devices (y chang Firebase /devices).
    """
    state = load_state()
    return jsonify(state.get("devices", {}))


@app.route("/devices", methods=["PUT", "POST"])
def put_devices():
    """
    ESP32 (hoặc webapp) gọi PUT/POST /devices với body JSON
    → ghi đè devices trong local_state.json.
    """
    payload = request.get_json(force=True, silent=True) or {}
    state = load_state()
    state["devices"] = payload
    save_state(state)
    return jsonify({
        "ok": True,
        "device_count": len(payload)
    })


# ---- DOORS (nếu cần dùng local) ----

@app.route("/doors", methods=["GET"])
def get_doors():
    state = load_state()
    return jsonify(state.get("doors", {}))


@app.route("/doors", methods=["PUT", "POST"])
def put_doors():
    payload = request.get_json(force=True, silent=True) or {}
    state = load_state()
    state["doors"] = payload
    save_state(state)
    return jsonify({
        "ok": True,
        "door_count": len(payload)
    })


# ---- SCENES (optional, sau này muốn lưu scene local) ----

@app.route("/scenes", methods=["GET"])
def get_scenes():
    state = load_state()
    return jsonify(state.get("scenes", {}))


@app.route("/scenes", methods=["PUT", "POST"])
def put_scenes():
    payload = request.get_json(force=True, silent=True) or {}
    state = load_state()
    state["scenes"] = payload
    save_state(state)
    return jsonify({
        "ok": True,
        "keys": list(payload.keys())
    })


# ================== MAIN ==================

if __name__ == "__main__":
    # host 0.0.0.0 để ESP32 và máy khác trong LAN gọi được
    # port 5000: sau này ESP32 sẽ gọi http://PI_IP:5000/devices
    app.run(host="0.0.0.0", port=5000, debug=True)
