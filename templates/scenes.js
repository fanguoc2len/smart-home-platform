// scenes.js
// Hệ thống Scene cho webapp Smart Home của bạn
// - Thay theme (nền, glow, màu đèn) theo từng scene
// - Gửi desired cho các thiết bị (light, fan, speaker) & cửa (frontDoor, sideDoor)
// - Có hỗ trợ kích hoạt bằng giọng nói (nhận từ cmd.raw_text)

const SCENES = [
  {
    id: "default",
    label: "Default",
    emoji: "🏠",
    bodyClass: "scene-default",
    actions: [] // không đụng thiết bị, chỉ reset theme
  },
  {
    id: "good_night",
    label: "Good Night",
    emoji: "🌙",
    bodyClass: "scene-good-night",
    actions: [
      // tắt tất cả đèn
      { kind: "all", type: "light", state: "off" },
      // tắt quạt
      { kind: "all", type: "fan",   state: "off", speed: 0 },
      // tắt loa
      { kind: "all", type: "speaker", state: "off", volume: 0 },
      // đóng 2 cửa
      { kind: "door", doorKey: "frontDoor", state: "closed" },
      { kind: "door", doorKey: "sideDoor",  state: "closed" }
    ]
  },
  {
    id: "movie",
    label: "Movie Mode",
    emoji: "🎬",
    bodyClass: "scene-movie",
    actions: [
      // tắt hết đèn
      { kind: "all", type: "light", state: "off" },

      // đèn Accent / LED ở living: bật 15% ánh vàng
      {
        kind: "section",
        section: "living",
        type: "light",
        state: "on",
        brightness: 18,
        color: "#FFB74D"
      },

      // quạt living speed 1
      {
        kind: "section",
        section: "living",
        type: "fan",
        state: "on",
        speed: 1
      },

      // loa phòng khách tắt (cho yên tĩnh)
      { kind: "section", section: "living", type: "speaker", state: "off", volume: 0 }
    ]
  },
  {
    id: "party",
    label: "Party",
    emoji: "🎉",
    bodyClass: "scene-party",
    actions: [
      // tất cả đèn full 100%, màu hồng tím
      {
        kind: "all",
        type: "light",
        state: "on",
        brightness: 100,
        color: "#FF00FF"
      },
      // loa tất cả lên 70%
      {
        kind: "all",
        type: "speaker",
        state: "on",
        volume: 70
      },
      // quạt max
      {
        kind: "all",
        type: "fan",
        state: "on",
        speed: 3
      }
    ]
  }
];

let currentSceneId = "default";

// ===== helper nhỏ =====
function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}

function getAllSceneClasses(body) {
  return Array.from(body.classList).filter((c) => c.startsWith("scene-"));
}

// highlight chip đang chọn
function highlightSceneChip(sceneId) {
  const chips = document.querySelectorAll(".scene-chip");
  chips.forEach((chip) => {
    chip.classList.toggle("active", chip.dataset.sceneId === sceneId);
  });
}

// áp action của scene lên 1 device
function applyActionToDevice(dev, action) {
  if (!dev) return;

  if (action.state === "on") dev.state = true;
  if (action.state === "off") dev.state = false;

  if (dev.type === "light") {
    if (typeof action.brightness === "number") {
      dev.brightness = clamp(Math.round(action.brightness), 0, 100);
      dev.state = dev.brightness > 0;
    }
    if (action.color) {
      dev.color = action.color;
    }
    if (action.state === "off") {
      dev.brightness = 0;
    }
    if (action.state === "on" && (dev.brightness == null || dev.brightness === 0)) {
      dev.brightness = 100;
    }
  } else if (dev.type === "fan") {
    if (typeof action.speed === "number") {
      dev.speed = clamp(Math.round(action.speed), 0, 3);
      dev.state = dev.speed > 0;
    }
    if (action.state === "off") {
      dev.speed = 0;
    }
    if (action.state === "on" && (dev.speed == null || dev.speed === 0)) {
      dev.speed = 1;
    }
  } else if (dev.type === "speaker") {
    if (typeof action.volume === "number") {
      dev.volume = clamp(Math.round(action.volume), 0, 100);
      dev.state = dev.volume > 0;
    }
    if (action.state === "off") {
      dev.volume = 0;
    }
    if (action.state === "on" && (dev.volume == null || dev.volume === 0)) {
      dev.volume = 50;
    }
  }
}

// cập nhật 2 card cửa static cho khớp scene
function syncDoorCards(doorKey, isOpen) {
  if (doorKey === "frontDoor") {
    const card = document.getElementById("doorCard");
    const label = document.getElementById("doorState");
    if (card) card.classList.toggle("on", isOpen);
    if (label) label.textContent = isOpen ? "Unlocked" : "Locked";
  } else if (doorKey === "sideDoor") {
    const card = document.getElementById("sideDoorCard");
    const label = document.getElementById("sideDoorState");
    if (card) card.classList.toggle("on", isOpen);
    if (label) label.textContent = isOpen ? "Open" : "Closed";
  }
}

// ===== Hàm chính: áp scene =====
export async function applyScene(sceneId) {
  const scene = SCENES.find((s) => s.id === sceneId);
  if (!scene) {
    console.warn("Scene not found:", sceneId);
    return false;
  }

  const body = document.body;
  // đổi theme
  getAllSceneClasses(body).forEach((c) => body.classList.remove(c));
  if (scene.bodyClass) {
    body.classList.add(scene.bodyClass);
  }

  currentSceneId = sceneId;
  highlightSceneChip(sceneId);

  const saved = window.saved || [];
  const touched = new Set();

  if (!scene.actions || !scene.actions.length) {
    return true; // scene chỉ đổi theme
  }

  for (const action of scene.actions) {
    // cửa frontDoor / sideDoor
    if (action.kind === "door" && typeof window.pushDoorServoState === "function") {
      const isOpen = action.state === "open";
      try {
        window.pushDoorServoState(action.doorKey, isOpen);
      } catch (e) {
        console.error("pushDoorServoState error:", e);
      }
      syncDoorCards(action.doorKey, isOpen);
      continue;
    }

    let targets = [];

    if (action.kind === "all") {
      targets = saved.filter((d) => !action.type || d.type === action.type);
    } else if (action.kind === "section") {
      targets = saved.filter(
        (d) =>
          d.section === action.section &&
          (!action.type || d.type === action.type)
      );
    } else if (action.kind === "id" && action.device_id) {
      const dev = saved.find((d) => d.id === action.device_id);
      if (dev) targets = [dev];
    }

    for (const dev of targets) {
      applyActionToDevice(dev, action);
      touched.add(dev);
    }
  }

  // cập nhật UI
  if (typeof window.saveAndRefresh === "function") {
    window.saveAndRefresh(false, false);
  }

  // sync Firebase
  if (typeof window.saveDeviceDebounced === "function") {
    touched.forEach((dev) => {
      try {
        window.saveDeviceDebounced(dev);
      } catch (e) {
        console.error("saveDeviceDebounced scene error:", e);
      }
    });
  }

  console.log("Scene applied:", sceneId, "devices touched:", touched.size);
  return true;
}

// ===== Scene từ giọng nói =====
//
// cmd.raw_text đã được normalize (không dấu, lower-case) ở home.html
// Ví dụ:
// "che do ngu", "che do xem phim", "bat che do party", ...
//
export function tryApplySceneFromVoice(cmd) {
  if (!cmd) return false;
  const raw = (cmd.raw_text || "").toLowerCase();
  if (!raw) return false;

  const has = (...keys) => keys.some((k) => raw.includes(k));

  // Good Night
  if (has("che do ngu", "che do good night", "good night", "ngu ngon")) {
    applyScene("good_night");
    return true;
  }

  // Movie
  if (has("che do xem phim", "xem phim", "movie")) {
    applyScene("movie");
    return true;
  }

  // Party
  if (has("che do tiec", "party", "che do party", "tiec tung")) {
    applyScene("party");
    return true;
  }

  // Default / bình thường
  if (has("che do mac dinh", "che do binh thuong", "default", "reset scene")) {
    applyScene("default");
    return true;
  }

  return false;
}

// gắn ra window cho home.html xài
window.applyScene = applyScene;
window.tryApplySceneFromVoice = tryApplySceneFromVoice;

// ===== Tạo UI scene bar =====
function initSceneBar() {
  const barContainer = document.getElementById("scenesBar");
  if (!barContainer) return;

  const frag = document.createDocumentFragment();

  SCENES.forEach((scene) => {
    const chip = document.createElement("button");
    chip.className = "scene-chip";
    chip.dataset.sceneId = scene.id;

    chip.innerHTML = `
      <span class="scene-chip-emoji">${scene.emoji || "🎛️"}</span>
      <span class="scene-chip-label">${scene.label}</span>
    `;

    chip.addEventListener("click", () => {
      applyScene(scene.id);
    });

    frag.appendChild(chip);
  });

  barContainer.innerHTML = "";
  barContainer.appendChild(frag);

  // lúc load lần đầu: set default
  applyScene(currentSceneId);
}

window.addEventListener("DOMContentLoaded", initSceneBar);
