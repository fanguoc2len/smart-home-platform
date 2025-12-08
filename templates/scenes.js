// ======================================
// SCENE MANAGER – Kiểu 2 (FULL + EFFECT)
// ======================================

// scene hiện tại (ưu tiên đọc từ localStorage nếu có)
let currentScene = null;
try {
  const stored = window.localStorage && window.localStorage.getItem("currentScene");
  if (stored) currentScene = stored;
} catch (_) {
  currentScene = null;
}
// backup trạng thái thiết bị trước khi apply scene
let sceneBackup = null;

// Sao lưu toàn bộ window.saved (để khi clearScene khôi phục lại)
function backupCurrentDevices() {
  try {
    if (!Array.isArray(window.saved)) {
      sceneBackup = null;
      return;
    }
    sceneBackup = JSON.parse(JSON.stringify(window.saved));
  } catch (e) {
    console.warn("Scene backup failed:", e);
    sceneBackup = null;
  }
}

// Khôi phục trạng thái thiết bị từ backup
function restoreDevicesFromSceneBackup() {
  if (!sceneBackup || !Array.isArray(sceneBackup) || !Array.isArray(window.saved)) {
    return;
  }

  const map = new Map(sceneBackup.map(d => [d.id, d]));
  const changed = [];

  window.saved.forEach(dev => {
    const old = map.get(dev.id);
    if (!old) return;

    // copy lại các field quan trọng
    dev.state       = old.state;
    dev.brightness  = old.brightness;
    dev.color       = old.color;
    dev.speed       = old.speed;
    dev.volume      = old.volume;
    dev.temperature = old.temperature;
    dev.rainbow     = old.rainbow;   // ⭐ khôi phục luôn cờ rainbow

    changed.push(dev);
  });

  // Cập nhật UI
  if (typeof renderSavedDevices === "function") {
    try { renderSavedDevices(); } catch (e) { console.error(e); }
  } else {
    window.saved.forEach(dev => {
      const card = document.getElementById(dev.id);
      if (!card) return;
      if (typeof updateCardVisual === "function") updateCardVisual(card, dev);
      if (typeof updateStateText === "function")  updateStateText(card, dev);
    });
  }

  // 🔁 Ghi lại Firebase theo từng thiết bị, KHÔNG đẩy nguyên window.saved
  if (typeof window.saveDeviceDebounced === "function") {
    changed.forEach(dev => window.saveDeviceDebounced(dev));
  } else if (typeof window.updateSingleDeviceToFirebase === "function") {
    changed.forEach(dev => {
      window.updateSingleDeviceToFirebase(dev).catch(e => console.error(e));
    });
  } else if (typeof window.saveDevicesToFirebase === "function") {
    // fallback cũ nếu thiếu 2 hàm trên
    window.saveDevicesToFirebase(window.saved || []).catch(e => console.error(e));
  }

  sceneBackup = null;
}

// Áp hiệu ứng scene lên đèn + loa (client-side)
function applySceneLocally(sceneName) {
  if (!Array.isArray(window.saved) || window.saved.length === 0) return;

  // lưu backup lần đầu khi bật scene
  if (!sceneBackup) backupCurrentDevices();

  // tránh Firebase sync đè lại ngay lập tức
  if (typeof blockFirebaseSync === "function") blockFirebaseSync();

  const lights   = window.saved.filter(d => d.type === "light");
  const speakers = window.saved.filter(d => d.type === "speaker");

  const randColor = () => {
    const palette = [
      "#ff4081", "#ffe082", "#40c4ff",
      "#69f0ae", "#7c4dff", "#ff80ab"
    ];
    return palette[Math.floor(Math.random() * palette.length)];
  };

  // ====== LIGHT LOGIC (giữ y nguyên như cũ) ======
  lights.forEach((dev, idx) => {
    if (!dev) return;

    if (sceneName === "relax") {
      // ánh sáng ấm, hơi mờ
      dev.state      = true;
      dev.brightness = 35;
      dev.color      = "#fff3e0";   // warm white
      dev.rainbow    = false;
    } else if (sceneName === "focus") {
      // sáng mạnh, trắng lạnh tập trung
      dev.state      = true;
      dev.brightness = 90;
      dev.color      = "#e0f7ff";   // cool white
      dev.rainbow    = false;
    } else if (sceneName === "party") {
      // full sáng, mỗi đèn một màu random
      dev.state      = true;
      dev.brightness = 100;
      dev.color      = randColor();
      dev.rainbow    = true;        // ⭐ bật chế độ rainbow cho ESP
    } else if (sceneName === "night") {
      // chỉ để 1 đèn mờ mờ, còn lại tắt
      if (idx === 0) {
        dev.state      = true;
        dev.brightness = 15;
        dev.color      = "#fff3e0";
      } else {
        dev.state      = false;
        dev.brightness = 0;
      }
      dev.rainbow    = false;
    }
  });

  // ====== SPEAKER LOGIC (THÊM MỚI) ======
  // mapping scene → bài + volume
  // track_index: khớp với ESP32 / JQ6500 
  speakers.forEach((dev) => {
    if (!dev) return;

    if (sceneName === "relax") {
      dev.track_index = 2;
      dev.volume      = 60;
      dev.state       = true;
      dev.power       = "on";

    } else if (sceneName === "party") {
      dev.track_index = 1;
      dev.volume      = 80;
      dev.state       = true;
      dev.power       = "on";

    } else if (sceneName === "focus" || sceneName === "night") {
      dev.track_index = dev.track_index ?? 1;  
      dev.volume      = 0;
      dev.state       = false;
      dev.power       = "off";
    }

    // 🚀 ĐẨY LÊN FIREBASE NGAY LẬP TỨC — KHÔNG DÙNG DEBOUNCE
    if (typeof window.updateSingleDeviceToFirebase === "function") {
      window.updateSingleDeviceToFirebase(dev)
        .catch(e => console.error(e));
    }
  });

  // ====== Cập nhật lại UI cho đèn + loa ======
  const affected = lights.concat(speakers);

  affected.forEach(dev => {
    const card = document.getElementById(dev.id);
    if (!card) return;
    if (typeof updateCardVisual === "function") updateCardVisual(card, dev);
    if (typeof updateStateText === "function")  updateStateText(card, dev);
  });

  // 🔁 Ghi Firebase cho các thiết bị đã đổi (đèn + loa)
  if (typeof window.saveDeviceDebounced === "function") {
    lights.forEach(dev => window.saveDeviceDebounced(dev));
  } else if (typeof window.updateSingleDeviceToFirebase === "function") {
    affected.forEach(dev => {
      window.updateSingleDeviceToFirebase(dev).catch(e => console.error(e));
    });
  } else if (typeof window.saveDevicesToFirebase === "function") {
    // fallback: vẫn dùng toàn bộ nếu thiếu 2 hàm trên
    window.saveDevicesToFirebase(window.saved || []).catch(e => console.error(e));
  } else if (typeof window.saveAndRefresh === "function") {
    window.saveAndRefresh(false, true);
  }
}


// ================================
// Apply scene
// ================================
function applyScene(sceneName) {
  currentScene = sceneName;
  try {
    if (window.localStorage) {
      window.localStorage.setItem("currentScene", sceneName);
    }
  } catch (_) {}

  // UI highlight
  document.querySelectorAll(".scene-card").forEach(card => {
    if (card.dataset.scene === sceneName) card.classList.add("active");
    else card.classList.remove("active");
  });

  // đổi theme theo scene (chỉ đổi background body)
  document.body.classList.remove("scene-relax", "scene-focus", "scene-party", "scene-night");
  document.body.classList.add("scene-" + sceneName);

  console.log(">>> Applying scene:", sceneName);

  // Áp scene lên đèn phía client
  try {
    applySceneLocally(sceneName);
  } catch (e) {
    console.error("Local scene error:", e);
  }

  // Gọi Flask (best-effort, CORS fail vẫn kệ)
  fetch("https://christinia-sorediate-fred.ngrok-free.dev/execute_scene", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scene: sceneName })
  })
    .then(res => res.json().catch(() => ({})))
    .then(data => console.log("🎉 Scene applied (Flask):", data))
    .catch(err => {
      console.warn("🔥 Flask không phản hồi (CORS) nhưng UI vẫn chạy:", err);
    });
}

// ================================
// Clear scene
// ================================
function clearScene() {
  console.log(">>> Clear scene, TURN EVERYTHING OFF...");

  // Reset trạng thái scene hiện tại + localStorage
  currentScene = null;
  try {
    if (window.localStorage) {
      window.localStorage.removeItem("currentScene");
    }
  } catch (_) {}

  // Gỡ class background theo scene
  document.body.classList.remove("scene-relax", "scene-focus", "scene-party", "scene-night");

  // Bỏ highlight trên tất cả scene-card
  document.querySelectorAll(".scene-card").forEach(card => {
    card.classList.remove("active");
  });

  // Không còn dùng backup nữa
  sceneBackup = null;

  // 🔻 TẮT TẤT CẢ THIẾT BỊ
  try {
    if (Array.isArray(window.saved)) {
      const changed = [];

      window.saved.forEach(dev => {
        if (!dev || !dev.id) return;

        let touched = false;

        // state / power chung
        if (dev.state !== undefined) {
          dev.state = false;
          touched = true;
        }
        if (dev.power !== undefined) {
          dev.power = "off";
          touched = true;
        }

        // đèn: brightness + rainbow
        if (dev.brightness !== undefined) {
          dev.brightness = 0;
          touched = true;
        }
        if (dev.rainbow !== undefined) {
          dev.rainbow = false;
          touched = true;
        }

        // quạt: speed
        if (dev.speed !== undefined) {
          dev.speed = 0;
          touched = true;
        }

        // loa: volume
        if (dev.volume !== undefined) {
          dev.volume = 0;
          touched = true;
        }

        if (dev.track_index !== undefined) {
          dev.track_index = 0;
        }
        if (dev.rainbow !== undefined) {
          dev.rainbow = false;
        }

        if (!touched) return;
        changed.push(dev);

        // Cập nhật UI cho từng card
        const card = document.getElementById(dev.id);
        if (card) {
          if (typeof updateCardVisual === "function") updateCardVisual(card, dev);
          if (typeof updateStateText === "function")  updateStateText(card, dev);
        }
      });

      // Ghi lại Firebase cho các device đã thay đổi
      if (typeof window.saveDeviceDebounced === "function") {
        changed.forEach(dev => window.saveDeviceDebounced(dev));
      } else if (typeof window.updateSingleDeviceToFirebase === "function") {
        changed.forEach(dev => {
          window.updateSingleDeviceToFirebase(dev).catch(e => console.error(e));
        });
      } else if (typeof window.saveDevicesToFirebase === "function") {
        window.saveDevicesToFirebase(window.saved || []).catch(e => console.error(e));
      }
    }
  } catch (e) {
    console.error("Clear scene error:", e);
  }

  // Gửi về Flask báo là đã clear scene (server muốn xử lý thì xử lý, không thì thôi)
  fetch("https://christinia-sorediate-fred.ngrok-free.dev/execute_scene", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scene: null })
  }).catch(() => {});
}

// ================================
// GẮN SỰ KIỆN CHO CÁC CARD SCENE
// ================================
document.addEventListener("DOMContentLoaded", () => {
  console.log("Scene system loaded.");

  // Khôi phục highlight scene từ localStorage (nếu có)
  try {
    const stored = window.localStorage && window.localStorage.getItem("currentScene");
    if (stored) {
      currentScene = stored;
      document.body.classList.add("scene-" + stored);
      document.querySelectorAll(".scene-card").forEach(c => {
        if (c.dataset.scene === stored) c.classList.add("active");
      });
    }
  } catch (_) {}


  document.querySelectorAll(".scene-card").forEach(card => {
    const scene = card.dataset.scene;

    card.addEventListener("click", () => {
      console.log("Clicked scene:", scene);

      if (currentScene === scene) {
        clearScene();
      } else {
        applyScene(scene);
      }
    });
  });
});

// expose global
window.applyScene  = applyScene;
window.clearScene  = clearScene;
