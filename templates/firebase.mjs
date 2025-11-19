import { initializeApp } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-app.js";
import { getDatabase,
         ref,
         update,
         onValue,
         get,
         remove } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-database.js";

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

// Lắng nghe toàn bộ devices — trả kèm id (key)
export function loadDevicesFromFirebase(callback) {
  const devicesRef = ref(db, "devices");
  onValue(devicesRef, (snapshot) => {
    const data = snapshot.val();
    const list = data
      ? Object.keys(data).map(k => ({ ...data[k], id: k }))
      : [];
    console.log("🔥 Firebase update:", data);
    callback(list);
  });
}

// Cập nhật toàn bộ danh sách (khi thêm/xóa)
export async function saveDevicesToFirebase(devices) {
  const updates = {};

  devices.forEach(dev => {
    const clean = {};
    for (const [k, v] of Object.entries(dev)) {
      if (v !== undefined) clean[k] = v;
    }
    if (dev.id) updates[`devices/${dev.id}`] = clean;
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
          updates[`devices/${key}`] = null;  // XÓA
        }
      });
    }
    console.log("📤 FINAL UPDATE:", updates);
    return update(ref(db), updates); // cập nhật toàn bộ ở root
  } catch (err) {
    console.error("Failed saving devices to firebase:", err);
    throw err;
  }
}

// Cập nhật nhanh 1 thiết bị (dùng cho slider/color)
export function updateSingleDeviceToFirebase(dev) {
  const clean = {};
  for (const [k, v] of Object.entries(dev)) {
    if (v !== undefined) clean[k] = v;
  }
  if (!dev.id) return Promise.reject(new Error("device.id missing"));
  console.log("📤 Update nhanh:", dev.id, clean);
  return update(ref(db, `devices/${dev.id}`), clean);
}

// --- Xóa 1 thiết bị cụ thể trên Firebase (modular API) ---
export function deleteDeviceFromFirebase(deviceId) {
  if (!deviceId) return Promise.reject(new Error("deviceId missing"));
  const deviceRef = ref(db, `devices/${deviceId}`);
  return remove(deviceRef)
    .then(() => {
      console.log('✅ Đã xóa thiết bị trên Firebase:', deviceId);
      return true;
    })
    .catch((error) => {
      console.error('❌ Lỗi khi xóa thiết bị trên Firebase:', error);
      throw error;
    });
}
