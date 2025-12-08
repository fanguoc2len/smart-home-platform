import { initializeApp } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-app.js";
import {
  getDatabase,
  ref,
  update,
  onValue,
  get,
  remove
} from "https://www.gstatic.com/firebasejs/11.0.1/firebase-database.js";

const firebaseConfig = {
  apiKey: "AIzaSyDYR2uMR2iVyvEDfEnhsZnrWUq3R368",
  authDomain: "do-an-2-91a3c.firebaseapp.com",
  databaseURL: "https://do-an-2-91a3c-default-rtdb.asia-southeast1.firebasedatabase.app",
  projectId: "do-an-2-91a3c",
  storageBucket: "do-an-2-91a3c.appspot.com",
  messagingSenderId: "532084331642",
  appId: "1:532084331642:web:18b2cb37364b0ab81a9c97c",
  measurementId: "G-C962E3XKD3"
};

const app = initializeApp(firebaseConfig);
const db = getDatabase(app);
console.log("✅ Firebase connected OK");

// ==========================
// 🔁 BACKEND MODE (Firebase / Pi)
// ==========================

// 👉 ĐỔI IP NÀY THÀNH IP CỦA PI TRÊN WIFI CHUNG
const PI_BASE_URL = "http://192.168.43.100:5000";

let BACKEND_MODE = "firebase";
try {
  const stored = window.localStorage && window.localStorage.getItem("backend_mode");
  if (stored === "firebase" || stored === "pi") BACKEND_MODE = stored;
} catch (e) {}

export function setBackendMode(mode) {
  if (mode !== "firebase" && mode !== "pi") {
    console.warn("setBackendMode: invalid mode", mode);
    return;
  }
  BACKEND_MODE = mode;
  try {
    window.localStorage && window.localStorage.setItem("backend_mode", mode);
  } catch (e) {}
  console.log("🔧 Backend mode =", BACKEND_MODE);
}

// cho tiện gõ trong console
window.setBackendMode = setBackendMode;

// ============ helper cho Pi (không động vào Firebase cũ) ============

let _piDevicesPollTimer = null;

function normalizePiDevicesToList(obj) {
  const o = obj || {};
  return Object.keys(o).map(id => {
    const dev = o[id] || {};
    return { id, ...dev };
  });
}

function loadDevicesFromPi(callback) {
  if (_piDevicesPollTimer) {
    clearInterval(_piDevicesPollTimer);
    _piDevicesPollTimer = null;
  }

  const fetchOnce = async () => {
    try {
      const res = await fetch(`${PI_BASE_URL}/devices`);
      if (!res.ok) {
        console.warn("Pi /devices failed", res.status);
        return;
      }
      const data = await res.json();
      const list = normalizePiDevicesToList(data);
      console.log("🔥 Pi /devices update:", data);
      callback(list);
    } catch (e) {
      console.error("Pi /devices error:", e);
    }
  };

  fetchOnce();
  _piDevicesPollTimer = setInterval(fetchOnce, 2000);
}

async function saveDevicesToPi(devices) {
  const out = {};
  (devices || []).forEach(dev => {
    if (!dev || !dev.id) return;
    // gửi dev phẳng xuống Pi
    out[dev.id] = { ...dev };
  });

  console.log("📤 PUT /devices (Pi):", out);

  const res = await fetch(`${PI_BASE_URL}/devices`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(out),
  });
  if (!res.ok) throw new Error("Pi PUT /devices failed " + res.status);
  return res.json();
}

/*async function updateSingleDeviceToPi(dev) {
  if (!dev || !dev.id) {
    return Promise.reject(new Error("device.id missing"));
  }

  try {
    const resGet = await fetch(`${PI_BASE_URL}/devices`);
    const current = (resGet.ok ? await resGet.json() : {}) || {};
    const merged = { ...(current[dev.id] || {}), ...dev };
    current[dev.id] = merged;

    console.log("📤 Update single dev (Pi):", dev.id, merged);

    const resPut = await fetch(`${PI_BASE_URL}/devices`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(current),
    });
    if (!resPut.ok) throw new Error("Pi PUT /devices failed " + resPut.status);
    return resPut.json();
  } catch (e) {
    console.error("updateSingleDeviceToPi error:", e);
    throw e;
  }
} */

// ---- Compat layer: read from desired/reported, write to desired (HomeKit-style)
const CTRL_KEYS = [
  "state",
  "power",
  "brightness",
  "color",
  "speed",
  "volume",
  "temperature",
  "rainbow",          // flag rainbow cho PARTY
  "track_index"
];

function pick(obj, keys) {
  const out = {};
  keys.forEach(k => {
    if (obj && obj[k] !== undefined) out[k] = obj[k];
  });
  return out;
}

// ======================================================
// Lắng nghe toàn bộ devices — trả kèm id (key)
// ƯU TIÊN desired trước, rồi mới tới reported
// ======================================================
export function loadDevicesFromFirebase(callback) {
  // 👉 Nếu đang ở mode Pi thì không dùng Firebase realtime
  if (BACKEND_MODE === "pi") {
    console.log("📡 loadDevicesFromFirebase → PI mode");
    loadDevicesFromPi(callback);
    return;
  }

  const devicesRef = ref(db, "devices");
  onValue(devicesRef, (snapshot) => {
    const data = snapshot.val() || {};
    const list = Object.keys(data).map(id => {
      const node     = data[id] || {};
      const meta     = node.metadata || {};
      const reported = node.reported || {};
      const desired  = node.desired  || {};
      const legacy   = node; // fallback cho schema cũ

      const power =
        desired.power  !== undefined ? desired.power  :
        reported.power !== undefined ? reported.power :
        legacy.power;

      const brightness =
        desired.brightness   ?? reported.brightness   ?? legacy.brightness;
      const color =
        desired.color        ?? reported.color        ?? legacy.color;
      const speed =
        desired.speed        ?? reported.speed        ?? legacy.speed;
      const volume =
        desired.volume       ?? reported.volume       ?? legacy.volume;
      const temperature =
        reported.temperature  ?? desired.temperature  ?? legacy.temperature;
      const rainbow =
        desired.rainbow      ?? reported.rainbow      ?? legacy.rainbow;
      const track_index =
        desired.track_index  ?? reported.track_index  ?? legacy.track_index;

      return {
        id,
        name:       meta.name    ?? legacy.name,
        type:       meta.type    ?? legacy.type,
        section:    meta.section ?? legacy.section,
        pin:        meta.pin     ?? legacy.pin,
        // state hiển thị theo power “mong muốn” (desired) nếu có
        state: power !== undefined
          ? (power === "on")
          : (legacy.state !== undefined ? legacy.state : undefined),
        brightness,
        color,
        speed,
        volume,
        temperature,
        rainbow,
        track_index
      };
    });
    console.log("🔥 Firebase update (compat):", data);
    callback(list);
  });
}

// ======================================================
// Cập nhật toàn bộ danh sách (khi thêm/xóa thiết bị)
// ======================================================
export async function saveDevicesToFirebase(devices) {
  // 👉 Nếu đang ở mode Pi thì ghi xuống Pi luôn, không đụng Firebase
  if (BACKEND_MODE === "pi") {
    console.log("💾 saveDevicesToFirebase → PI mode");
    return saveDevicesToPi(devices);
  }

  const updates = {};

  devices.forEach(dev => {
    if (!dev?.id) return;

    // metadata fields
    const meta = pick(dev, ["name", "type", "section", "pin"]);
    if (Object.keys(meta).length) {
      updates[`devices/${dev.id}/metadata`] = meta;
    }

    // control fields -> desired
    const ctrl = pick(dev, CTRL_KEYS);
    if (Object.keys(ctrl).length) {
      if (ctrl.state !== undefined) {
        ctrl.power = ctrl.state ? "on" : "off";
        delete ctrl.state;
      }
      ctrl.ts = Date.now();
      ctrl.updated_by = "ui";
      updates[`devices/${dev.id}/desired`] = ctrl;

      // DEV MIRROR đã tắt để không đè reported của ESP32
      // const rep = { ...ctrl, updated_by: "ui-mirror" };
      // updates[`devices/${dev.id}/reported`] = rep;
    }
  });

  const devicesRef = ref(db, "devices");
  try {
    const snapshot = await get(devicesRef);
    if (snapshot.exists()) {
      const data = snapshot.val();
      const firebaseKeys = Object.keys(data);
      firebaseKeys.forEach(key => {
        const stillExists = devices.some(d => d.id === key);
        if (!stillExists) {
          updates[`devices/${key}`] = null;  // remove
        }
      });
    }
    console.log("📤 FINAL UPDATE (compat):", updates);
    return update(ref(db), updates); // update tại root
  } catch (err) {
    console.error("Failed saving devices to firebase:", err);
    throw err;
  }
}

// ======================================================
// Cập nhật nhanh 1 thiết bị (slider / color / vv.)
// ======================================================
export function updateSingleDeviceToFirebase(dev) {
  if (!dev?.id) return Promise.reject(new Error("device.id missing"));

  if (BACKEND_MODE === "pi") {
    console.log("🎯 updateSingleDeviceToFirebase → PI mode", dev.id);
    return updateSingleDeviceToPi(dev);
  }

  const patch = pick(dev, CTRL_KEYS);

  // 1) Luôn map state → power cho đúng với UI
  if (patch.state !== undefined) {
    patch.power = patch.state ? "on" : "off";
  }

  // 2) Các field đặc biệt cho loa / effect
  if (dev.track_index !== undefined) {
    patch.track_index = dev.track_index;
  }
  if (dev.volume !== undefined) {
    patch.volume = dev.volume;
  }
  if (dev.rainbow !== undefined) {
    patch.rainbow = dev.rainbow;
  }

  patch.ts = Date.now();
  patch.updated_by = "ui";

  console.log("📤 Update desired:", dev.id, patch);

  const desiredRef = ref(db, `devices/${dev.id}/desired`);
  return update(desiredRef, patch);
}

// ======================================================
// Xóa 1 thiết bị cụ thể
// ======================================================
export function deleteDeviceFromFirebase(deviceId) {
  if (!deviceId) return Promise.reject(new Error("deviceId missing"));

  // (để đơn giản, delete vẫn chỉ áp dụng trên Firebase;
  // nếu ở Pi mode mà m cần xóa local thì có thể thêm sau)
  const deviceRef = ref(db, `devices/${deviceId}`);
  return remove(deviceRef)
    .then(() => {
      console.log("✅ Đã xóa thiết bị trên Firebase:", deviceId);
      return true;
    })
    .catch((error) => {
      console.error("❌ Lỗi khi xóa thiết bị trên Firebase:", error);
      throw error;
    });
}

// ======================================================
// Xóa toàn bộ lịch sử lệnh trong /commands
// ======================================================
export function clearCommandsInFirebase() {
  const cmdsRef = ref(db, "commands");
  return remove(cmdsRef)
    .then(() => {
      console.log("🧹 Đã xóa toàn bộ /commands trên Firebase");
      return true;
    })
    .catch((error) => {
      console.error("❌ Lỗi khi xóa /commands:", error);
      throw error;
    });
}

/* ============================================================
   SCENES PRESET – tạo /scenes (cho ESP32) + /ui_state/scenes (cho UI)
   Logic devices lấy y nguyên file cũ của m
   ============================================================ */
export async function ensureDefaultScenes() {
  if (BACKEND_MODE === "pi") {
    // Pi mode vẫn điều khiển theo JSON riêng, nhưng scene vẫn nằm trên Firebase
    console.log("ensureDefaultScenes: đang ở PI mode, vẫn sync scenes lên Firebase");
  }

  const scenesRef  = ref(db, "scenes");
  const uiStateRef = ref(db, "ui_state");

  // ======= LOGIC SCENE CHO THIẾT BỊ (y như file cũ) =======
  const scenesData = {
    //  RELAX: đèn ấm, mờ; loa mở nhẹ, quạt tắt
    relax: {
      rules: [
        {
          match: { type: "light" },
          desired: {
            state: true,
            power: "on",
            brightness: 35,
            color: "#FFD9B3",
            rainbow: false
          }
        },
        {
          match: { type: "speaker" },
          desired: {
            state: true,
            volume: 60,
            track_index: 2
          }
        },
        {
          match: { type: "fan" },
          desired: {
            state: false,
            power: "off",
            speed: 0
          }
        }
      ]
    },

    // FOCUS: đèn trắng sáng; quạt / loa tắt
    focus: {
      rules: [
        {
          match: { type: "light" },
          desired: {
            state: true,
            power: "on",
            brightness: 100,
            color: "#E6FCFF",
            rainbow: false
          }
        },
        {
          match: { type: "speaker" },
          desired: {
            state: false,
            power: "off",
            volume: 0
          }
        },
        {
          match: { type: "fan" },
          desired: {
            state: false,
            power: "off",
            speed: 0
          }
        }
      ]
    },

    //  ALL_OFF: tắt TẤT CẢ thiết bị
    all_off: {
      rules: [
        {
          // match rỗng => áp dụng cho mọi thiết bị
          match: {},
          desired: {
            state: false,
            power: "off",
            brightness: 0,
            speed: 0,
            volume: 0,
            rainbow: false
          }
        }
      ]
    },

    // PARTY: đèn rainbow, loa max, quạt max
    party: {
      rules: [
        {
          match: { type: "light" },
          desired: {
            state: true,
            power: "on",
            brightness: 100,
            rainbow: true
          }
        },
        {
          match: { type: "speaker" },
          desired: {
            state: true,
            power: "on",
            volume: 100,
            track_index: 1
          }
        },
        {
          match: { type: "fan" },
          desired: {
            state: true,
            power: "on",
            speed: 3
          }
        }
      ]
    },

    // NIGHT: tắt tất cả; riêng đèn khu entry bật nhẹ ấm
    night: {
      rules: [
        // 1) Đèn khu entry: bật nhẹ
        {
          match: { type: "light", section: "entry" },
          desired: {
            state: true,
            power: "on",
            brightness: 20,
            color: "#FFECC2",
            rainbow: false
          }
        },
        // 2) Các đèn còn lại: tắt hết
        {
          match: { type: "light" },
          desired: {
            state: false,
            power: "off",
            brightness: 0,
            rainbow: false
          }
        },
        // 3) Tắt toàn bộ quạt
        {
          match: { type: "fan" },
          desired: {
            state: false,
            power: "off",
            speed: 0
          }
        },
        // 4) Tắt toàn bộ loa
        {
          match: { type: "speaker" },
          desired: {
            state: false,
            power: "off",
            volume: 0
          }
        }
      ]
    }

  };

  // ======= UI SCENES – chỉ cho giao diện (background + rainbow) =======
  const uiScenesData = {
    relax: {
      name: "Relax",
      bgType: "solid",
      bgColors: ["#1f2329"],       // nền xám đậm êm mắt
      hasRainbowCards: false
    },
    focus: {
      name: "Focus",
      bgType: "gradient",
      bgColors: ["#020617", "#1f2937"], // xanh đậm, dịu hơn bản cũ
      hasRainbowCards: false
    },
    all_off: {
      name: "All Off",
      bgType: "solid",
      bgColors: ["#000000"],
      hasRainbowCards: false
    },
    party: {
      name: "Party",
      bgType: "gradient",
      bgColors: ["#020617", "#4c1d95"], // tím tối + card rainbow
      hasRainbowCards: true
    },
    night: {
      name: "Night",
      bgType: "solid",
      bgColors: ["#030712"],       // gần như đen
      hasRainbowCards: false
    }
  };

  try {
    
    await update(scenesRef, scenesData);

  
    await update(uiStateRef, {
      scenes: uiScenesData
    });

    console.log("✨ Đã cập nhật default scenes + ui_state.scenes:", scenesData, uiScenesData);
  } catch (e) {
    console.error("❌ ensureDefaultScenes error:", e);
  }
}

export async function applySceneById(sceneId) {
  if (!sceneId) return;
  try {
    const sceneSnap = await get(ref(db, `scenes/${sceneId}`));
    if (!sceneSnap.exists()) {
      console.warn("applySceneById: scene không tồn tại:", sceneId);
      return;
    }

    const scene = sceneSnap.val() || {};
    const rules = Array.isArray(scene.rules) ? scene.rules : [];
    if (!rules.length) {
      console.warn("applySceneById: scene không có rules:", sceneId);
      return;
    }

    const devSnap = await get(ref(db, "devices"));
    const data = devSnap.val() || {};
    const updates = {};
    const now = Date.now();

    Object.entries(data).forEach(([id, node]) => {
      node = node || {};
      const meta    = node.metadata || {};
      const desired = node.desired  || {};
      const legacy  = node;

      const type = (meta.type    || legacy.type    || "").toString().toLowerCase();
      const sect = (meta.section || legacy.section || "").toString().toLowerCase();

      for (const rule of rules) {
        const match = rule.match || {};
        let ok = true;

        if (match.type) {
          ok = ok && type === String(match.type).toLowerCase();
        }
        if (match.section) {
          ok = ok && sect === String(match.section).toLowerCase();
        }
        if (!ok) continue;

        const desiredPatch = { ...(rule.desired || {}) };

        // map state -> power
        if (desiredPatch.state !== undefined) {
          desiredPatch.power = desiredPatch.state ? "on" : "off";
          delete desiredPatch.state;
        }

        desiredPatch.ts = now;
        desiredPatch.updated_by = `scene:${sceneId}`;

        updates[`devices/${id}/desired`] = {
          ...desired,
          ...desiredPatch
        };

        // mỗi device chỉ match rule đầu tiên
        break;
      }
    });

    if (Object.keys(updates).length) {
      console.log("🎬 applySceneById", sceneId, "updates:", updates);
      await update(ref(db), updates);
    } else {
      console.log("applySceneById: không có thiết bị nào match scene", sceneId);
    }
  } catch (e) {
    console.error("❌ applySceneById error:", e);
  }
}

// TỰ ĐỘNG APPLY SCENE "all_off" KHI currentScene ĐỔI TỪ KHÁC NULL → NULL
let _lastSceneDevices = null;
const _uiCurrentSceneRef = ref(db, "ui_state/currentScene");

onValue(_uiCurrentSceneRef, (snap) => {
  const current = snap.val() || null;

  // 1) Trường hợp chuyển từ scene -> null: apply all_off
  if (_lastSceneDevices && !current) {
    console.log("currentScene cleared, applying scene 'all_off'");
    applySceneById("all_off").catch((err) => {
      console.error("auto all_off on clear failed:", err);
    });
  }

  // 2) Trường hợp null -> scene hoặc sceneA -> sceneB: apply scene mới cho devices
  if (current && current !== _lastSceneDevices) {
    console.log("currentScene changed to", current, "applying scene to devices");
    applySceneById(current).catch((err) => {
      console.error("auto apply scene failed:", err);
    });
  }

  _lastSceneDevices = current;
});

/* ============================================================
   UI-SCENE SYNC – ĐỂ ĐIỆN THOẠI / MÁY TÍNH NHÌN GIỐNG NHAU
   - subscribeUiScene(onChange): lắng nghe scene + cấu hình UI
   - setCurrentScene(sceneId, { applyDevices = true })
   - clearCurrentScene({ applyAllOff = false })
   ============================================================ */

// cache nhỏ để callback có đủ info
let _uiScenesCache = {};
let _uiCurrentSceneId = null;

/**
 * Lắng nghe scene UI từ Firebase:
 * onChange(sceneId, sceneCfg, allScenes)
 */
export function subscribeUiScene(onChange) {
  if (typeof onChange !== "function") return;

  const uiScenesRef = ref(db, "ui_state/scenes");
  const uiCurrentSceneRef = ref(db, "ui_state/currentScene");

  // Lắng nghe danh sách scenes
  onValue(uiScenesRef, (snap) => {
    _uiScenesCache = snap.val() || {};
    onChange(
      _uiCurrentSceneId,
      _uiCurrentSceneId ? _uiScenesCache[_uiCurrentSceneId] || null : null,
      _uiScenesCache
    );
  });

  // Lắng nghe scene đang active
  onValue(uiCurrentSceneRef, (snap) => {
    _uiCurrentSceneId = snap.val() || null;
    onChange(
      _uiCurrentSceneId,
      _uiCurrentSceneId ? _uiScenesCache[_uiCurrentSceneId] || null : null,
      _uiScenesCache
    );
  });
}

/**
 * Set scene hiện tại:
 *  - ghi /ui_state/currentScene
 *  - nếu applyDevices = true → applySceneById để bật/tắt thiết bị
 */
export async function setCurrentScene(sceneId, options = {}) {
  const { applyDevices = true } = options || {};

  const nextScene = sceneId || null;

  await update(ref(db, "ui_state"), {
    currentScene: nextScene
  });

  if (applyDevices && nextScene) {
    await applySceneById(nextScene);
  }
}

/**
 * Clear scene hiện tại (về null).
 * Nếu applyAllOff = true → apply luôn scene "all_off".
 */
export async function clearCurrentScene(options = {}) {
  const { applyAllOff = true } = options || {};

  await update(ref(db, "ui_state"), {
    currentScene: null
  });

  if (applyAllOff) {
    try {
      await applySceneById("all_off");
    } catch (e) {
      console.error("clearCurrentScene: apply all_off fail:", e);
    }
  }
}
