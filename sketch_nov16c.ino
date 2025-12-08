#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <DHT.h>
#include <driver/ledc.h>
#include <Adafruit_NeoPixel.h>
#include <JQ6500_Serial.h>

// Chân UART2 dành cho JQ6500
#define JQ_RX_PIN 16   // ESP32 nhận → nối vào TX JQ6500
#define JQ_TX_PIN 17   // ESP32 gửi  → nối vào RX JQ6500

JQ6500_Serial mp3(Serial2);

bool mp3Ready = false;
int mp3CurrentTrack = 0;     // bài hiện tại (1 = party, 2 = relax)
bool mp3On = false;
unsigned long mp3StartMillis = 0;

// ==== FreeRTOS cho ESP32 ====
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

// === Cấu hình WiFi + Firebase ===
const char* WIFI_SSID = "Hai Dau Bac";
const char* WIFI_PASS = "18091970";
const String FIREBASE_BASE =
  "https://do-an-2-91a3c-default-rtdb.asia-southeast1.firebasedatabase.app/devices";

// Base riêng cho doors (thẻ cố định trong webapp)
const String FIREBASE_DOORS_BASE =
  "https://do-an-2-91a3c-default-rtdb.asia-southeast1.firebasedatabase.app/doors";

// Không dùng lại trong loop nữa, chỉ để tham khảo
const unsigned long DISCOVER_INTERVAL = 30000;   // nếu sau này muốn discover lại thì xài

// Poll interval
const unsigned long DESIRED_POLL_INTERVAL  = 200;   // poll toàn bộ desired mỗi 200ms
const unsigned long SENSOR_POLL_INTERVAL   = 1500;  // DHT11
const unsigned long DOOR_POLL_INTERVAL     = 200;   // poll /doors/.../state

#define MAX_LIGHTS    10
#define MAX_FANS      4
#define MAX_SPEAKERS  4
#define MAX_SENSORS   4

unsigned long lastDiscover    = 0;
unsigned long lastDesiredPoll = 0;

// =========================
// Struct Light (NeoPixel)
// =========================
struct Light {
  String id;
  int pin;                  // GPIO
  bool active;              // còn tồn tại trong /devices
  uint32_t lastColor;       // 0xRRGGBB
  int lastBrightness;       // 0..100, -1 = chưa set
  bool lastState;           // true = ON
  bool rainbow;             // true = hiệu ứng cầu vồng (scene PARTY)
  uint8_t rainbowPos;       // vị trí màu trên vòng màu
  unsigned long lastRainbowTick;
  unsigned long lastPoll;
  unsigned long lastDesiredTs;
  Adafruit_NeoPixel* strip; // mỗi light 1 NeoPixel object riêng
};

Light lights[MAX_LIGHTS];

// =========================
// NeoPixel Rainbow Effect
// =========================
// cho cảm giác gần giống rainbowCycle(20) của code test
#define RAINBOW_INTERVAL 5   // ms giữa 2 frame rainbow
#define RAINBOW_STEP      2  // bước nhảy màu mỗi frame

uint32_t wheel(Adafruit_NeoPixel* s, byte pos) {
  if (!s) return 0;
  pos = 255 - pos;
  if (pos < 85) {
    return s->Color(255 - pos * 3, 0, pos * 3);
  } else if (pos < 170) {
    pos -= 85;
    return s->Color(0, pos * 3, 255 - pos * 3);
  } else {
    pos -= 170;
    return s->Color(pos * 3, 255 - pos * 3, 0);
  }
}

void updateRainbowLight(int idx) {
  if (idx < 0 || idx >= MAX_LIGHTS) return;
  Light &L = lights[idx];
  if (!L.active || !L.lastState) return;
  Adafruit_NeoPixel* s = L.strip;
  if (!s) return;

  unsigned long now = millis();
  if (now - L.lastRainbowTick < RAINBOW_INTERVAL) return;
  L.lastRainbowTick = now;

  uint32_t c = wheel(s, L.rainbowPos);

  // scale theo brightness
  int bri = L.lastBrightness;
  if (bri < 0) bri = 100;
  if (bri > 100) bri = 100;

  uint8_t r = (uint8_t)(((c >> 16) & 0xFF) * bri / 100);
  uint8_t g = (uint8_t)(((c >> 8)  & 0xFF) * bri / 100);
  uint8_t b = (uint8_t)(( c        & 0xFF) * bri / 100);

  s->setPixelColor(0, s->Color(r, g, b));
  s->show();

  // nhảy nhiều bước để màu trôi nhanh hơn
  L.rainbowPos += RAINBOW_STEP;
}

// tick tất cả đèn rainbow – chỉ được gọi trong task riêng
inline void tickRainbowAll() {
  for (int i = 0; i < MAX_LIGHTS; i++) {
    if (lights[i].active && lights[i].rainbow && lights[i].strip != nullptr) {
      updateRainbowLight(i);
    }
  }
}

// =========================
// FAN (servo SG90) & DOOR
// =========================
#define SERVO_FREQ        50
#define SERVO_RESOLUTION  10   // 10-bit PWM cho servo

// Góc cho cửa (servo SG90)
const int DOOR_CLOSED_ANGLE = 90;   // đóng
const int DOOR_OPEN_ANGLE   = 5;  // mở

// Góc cho QUẠT dùng servo quay liên tục
// 90° ~ đứng yên, càng lệch xa 90° quay càng nhanh
const int FAN_STOP_ANGLE    = 90;  // stop
const int FAN_SPEED1_ANGLE  = 110; // speed 1
const int FAN_SPEED2_ANGLE  = 130; // speed 2
const int FAN_SPEED3_ANGLE  = 150; // speed 3
// Nếu quạt quay ngược chiều muốn, đổi 110/130/150 thành 70/50/30.

struct FanDevice {
  String id;
  int pin;          // GPIO servo
  int channel;      // kênh LEDC
  bool active;
  bool isDoor;      // true = cửa (chỉ on/off), false = quạt (có speed)
  bool lastState;   // on/off
  int lastSpeed;    // 0..3 (chỉ dùng cho quạt)
  unsigned long lastPoll;
};

FanDevice fans[MAX_FANS];

// =========================
// SPEAKER (buzzer low-level trigger)
// =========================
#define BUZZER_FREQ        4000
#define BUZZER_RESOLUTION  10   // 10-bit PWM

struct SpeakerDevice {
  String id;
  int pin;
  int channel;
  bool active;
  bool lastState;
  int lastVolume;       // 0..100
  int lastTrack;   
  unsigned long lastPoll;
  bool pwmAttached;     // true nếu đã attach PWM vào chân
  unsigned long lastDesiredTs; 
};

SpeakerDevice speakers[MAX_SPEAKERS];

// =========================
// DHT11 SENSOR
// =========================
#define DHT_TYPE DHT11

struct TempSensor {
  String id;
  int pin;
  bool active;
  float lastTemp;
  float lastHum;
  unsigned long lastPoll;
  DHT* dht;
};

TempSensor sensors[MAX_SENSORS];

// =========================
// DOORS CỐ ĐỊNH CHO WEBAPP
// =========================
// 2 cửa cố định của webapp: frontDoor & sideDoor
struct DoorFixed {
  const char* name;   // "frontDoor" / "sideDoor"
  int pin;            // GPIO servo
  int channel;        // kênh LEDC (không trùng 0..MAX_FANS-1)
  bool lastState;     // true = open, false = closed
};

DoorFixed doors[2] = {
  { "frontDoor", 32, 6, false },
  { "sideDoor",  27, 7, false }
};

// =========================
// forward declarations
// =========================
void discoverDevices();   // lights
int  addLightById(const char* id);
int  findLightIndex(const char* id);
void removeDeletedLights();
void applyLightState(int idx);

void discoverFans();
void discoverSpeakers();
void discoverSensors();

int  addFanById(const char* id);
int  addSpeakerById(const char* id);
int  addSensorById(const char* id);

void pollSensorDevice(int idx);

int  findFanIndex(const char* id);
int  findSpeakerIndex(const char* id);
int  findSensorIndex(const char* id);

void removeDeletedFans();
void removeDeletedSpeakers();
void removeDeletedSensors();

void setServoAngle(int channel, int angle);
void setBuzzerVolume(int channel, int volume);

void ensureWiFi();

// poll toàn bộ desired + doors
void pollAllDesired();
void applyLightDesiredFromJson(int idx, JsonObject d);
void applyFanDesiredFromJson(int idx, JsonObject d);
void applySpeakerDesiredFromJson(int idx, JsonObject d);

// hàm cho doors cố định
void pollDoors();

// =========================
// FreeRTOS TASKS
// =========================
void setupJQ6500() {
  Serial.println("Init JQ6500...");

  Serial2.begin(9600, SERIAL_8N1, JQ_RX_PIN, JQ_TX_PIN);
  delay(500);

  mp3.reset();
  delay(500);

  // Nhạc lưu trong bộ nhớ trong của JQ6500-16P
  mp3.setSource(MP3_SRC_BUILTIN);
  delay(100);

  // Volume mặc định
  mp3.setVolume(20);
  delay(50);

  // Set loop mode lần đầu (sau này mỗi lần play mình sẽ set lại thêm 1 lần nữa)
  mp3.setLoopMode(MP3_LOOP_ONE);

  mp3Ready = true;
  mp3On = false;
  mp3CurrentTrack = 0;
  mp3StartMillis = 0;

  Serial.println("JQ6500 READY");
}

// Task 1: Rainbow LED – chạy core 0, ưu tiên animation mượt
void rainbowTask(void *pvParameters) {
  (void) pvParameters;
  for (;;) {
    tickRainbowAll();                     // update tất cả LED rainbow
    vTaskDelay(pdMS_TO_TICKS(1));         // ~1ms
  }
}

// Task 2: WiFi + pollAllDesired + pollDoors – chạy core 1
void wifiAndDesiredTask(void *pvParameters) {
  (void) pvParameters;

  static unsigned long lastJqCheck = 0;

  for (;;) {
    ensureWiFi();        // nếu mất WiFi thì lo reconnect (blocking 1 task này thôi)

    // luôn cố poll, nếu WiFi chưa lên thì HTTP trả lỗi, mình bỏ qua
    pollAllDesired();    // đọc /devices desired
    pollDoors();         // đọc /doors/.../state

    // ==========================
    //  WATCHDOG CHO JQ6500
    //  – CANH THỜI GIAN ĐỂ LOOP
    // ==========================
    unsigned long now = millis();
    if (mp3Ready && mp3On && mp3CurrentTrack > 0 && (now - lastJqCheck) >= 1000) {
      lastJqCheck = now;

      // Dùng length bài từ module
      unsigned int len = mp3.currentFileLengthInSeconds();
      unsigned int elapsed = 0;
      if (mp3StartMillis > 0) {
        elapsed = (unsigned int)((now - mp3StartMillis) / 1000);
      }

      Serial.printf("JQ watchdog: len=%u elapsed=%u track=%d\n",
                    len, elapsed, mp3CurrentTrack);

      // Nếu length hợp lệ và đã play hết (hoặc gần hết) -> play lại
      if (len > 0 && elapsed >= len) {
        Serial.printf("JQ watchdog: time-based restart track %d\n", mp3CurrentTrack);
        mp3.playFileByIndexNumber(mp3CurrentTrack);
        delay(50);
        mp3.setLoopMode(MP3_LOOP_ONE);
        mp3StartMillis = millis();
      }
    }

    vTaskDelay(pdMS_TO_TICKS(50));  // ~50ms, bên trong pollAllDesired / pollDoors tự check interval
  }
}

// Task 3: Sensor DHT11 – chạy core 1
void sensorTask(void *pvParameters) {
  (void) pvParameters;
  for (;;) {
    unsigned long now2 = millis();
    for (int i = 0; i < MAX_SENSORS; i++) {
      if (sensors[i].active && sensors[i].pin >= 0 && sensors[i].dht != nullptr) {
        if (now2 - sensors[i].lastPoll >= SENSOR_POLL_INTERVAL) {
          sensors[i].lastPoll = now2;
          pollSensorDevice(i);
        }
      }
      // Nhường CPU giữa mỗi sensor một chút
      vTaskDelay(pdMS_TO_TICKS(5));
    }
    // vòng ngoài nghỉ nhẹ cho sensor task
    vTaskDelay(pdMS_TO_TICKS(10));
  }
}

// =========================
// WiFi helper
// =========================
void ensureWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;
  Serial.println("WiFi lost. Reconnecting...");
  WiFi.disconnect();
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 8000) {
    vTaskDelay(pdMS_TO_TICKS(200));
    Serial.print(".");
  }
  if (WiFi.status() == WL_CONNECTED) Serial.println("\n✅ WiFi reconnected");
  else Serial.println("\n⚠ WiFi reconnect failed");
}

void setup() {
  Serial.begin(115200);

  // MUTE buzzer trên GPIO25 ngay khi boot (MH-FMD low-level)
  pinMode(25, OUTPUT);
  digitalWrite(25, HIGH);

  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("Connecting WiFi");
  unsigned long t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < 10000) {
    Serial.print(".");
    vTaskDelay(pdMS_TO_TICKS(200));
  }
  Serial.println();
  if (WiFi.status() == WL_CONNECTED) Serial.println("✅ WiFi OK");
  else Serial.println("⚠ WiFi not connected (will retry by task)");

  // init LIGHT array
  for (int i = 0; i < MAX_LIGHTS; i++) {
    lights[i].id = "";
    lights[i].pin = -1;
    lights[i].active = false;
    lights[i].lastColor = 0xFFFFFF;
    lights[i].lastBrightness = -1;
    lights[i].lastState = false;
    lights[i].rainbow = false;
    lights[i].rainbowPos = 0;
    lights[i].lastRainbowTick = 0;
    lights[i].lastPoll = 0;
    lights[i].lastDesiredTs = 0;
    lights[i].strip = nullptr;
  }

  // init FAN array
  for (int i = 0; i < MAX_FANS; i++) {
    fans[i].id = "";
    fans[i].pin = -1;
    fans[i].channel = -1;
    fans[i].active = false;
    fans[i].isDoor = false;
    fans[i].lastState = false;
    fans[i].lastSpeed = 0;
    fans[i].lastPoll = 0;
  }

  // init SPEAKER array
  for (int i = 0; i < MAX_SPEAKERS; i++) {
    speakers[i].id = "";
    speakers[i].pin = -1;
    speakers[i].channel = -1;
    speakers[i].active = false;
    speakers[i].lastState = false;
    speakers[i].lastVolume = 0;
    speakers[i].lastTrack = 0;
    speakers[i].lastPoll = 0;
    speakers[i].pwmAttached = false;
    speakers[i].lastDesiredTs = 0;
 }

  // init SENSOR array
  for (int i = 0; i < MAX_SENSORS; i++) {
    sensors[i].id = "";
    sensors[i].pin = -1;
    sensors[i].active = false;
    sensors[i].lastTemp = NAN;
    sensors[i].lastHum = NAN;
    sensors[i].lastPoll = 0;
    sensors[i].dht = nullptr;
  }

  // init DOORS servo (cố định cho webapp /doors/frontDoor, /doors/sideDoor)
  for (int i = 0; i < 2; i++) {
    if (doors[i].pin >= 0) {
      ledcSetup(doors[i].channel, SERVO_FREQ, SERVO_RESOLUTION);
      ledcAttachPin(doors[i].pin, doors[i].channel);
      setServoAngle(doors[i].channel, DOOR_CLOSED_ANGLE);  // đóng lúc khởi động
    }
  }

  // discover ban đầu (CHỈ 1 LẦN lúc boot)
  discoverDevices();   // lights
  discoverFans();
  discoverSpeakers();
  discoverSensors();

  // ==== TẠO CÁC TASK FreeRTOS ====
  // Task Rainbow – core 0
  xTaskCreatePinnedToCore(
    rainbowTask,
    "RainbowTask",
    2048,
    NULL,
    1,
    NULL,
    0
  );

  // Task WiFi + poll desired + doors – core 1
  xTaskCreatePinnedToCore(
    wifiAndDesiredTask,
    "WiFiDesiredTask",
    8192,     // cần stack to vì JSON 4KB
    NULL,
    2,
    NULL,
    1
  );

  // Task Sensor DHT11 – core 1
  xTaskCreatePinnedToCore(
    sensorTask,
    "SensorTask",
    8192,
    NULL,
    1,
    NULL,
    1
  );
  setupJQ6500();
}

void loop() {
  // loop của Arduino bây giờ không làm gì nặng nữa
  vTaskDelay(pdMS_TO_TICKS(50));
}

/* ============================================================
   POLL TẤT CẢ desired TỪ /devices.json
   ============================================================ */

void pollAllDesired() {
  unsigned long now = millis();
  if (now - lastDesiredPoll < DESIRED_POLL_INTERVAL) return;
  lastDesiredPoll = now;

  unsigned long t0 = millis();

  String url = FIREBASE_BASE + ".json";
  HTTPClient http;
  http.setReuse(true);
  http.begin(url);
  http.setTimeout(300);   // timeout nhỏ để không block lâu
  int code = http.GET();
  if (code != 200) {
    http.end();
    Serial.println("⚠ pollAllDesired GET failed, code=" + String(code));
    return;
  }
  String payload = http.getString();
  http.end();

  // tăng dung lượng doc nếu device nhiều (4096 cho vài chục node là ổn)
  StaticJsonDocument<4096> doc;
  DeserializationError err = deserializeJson(doc, payload);
  if (err) {
    Serial.println("⚠ pollAllDesired parse error");
    return;
  }

  JsonObject root = doc.as<JsonObject>();
  if (root.isNull()) return;

  for (JsonPair kv : root) {
    const char* id = kv.key().c_str();
    JsonObject dev = kv.value().as<JsonObject>();
    if (dev.isNull()) continue;

    JsonObject meta = dev["metadata"];
    String t = "";
    if (!meta.isNull() && meta.containsKey("type")) {
      t = meta["type"].as<const char*>();
    } else if (dev.containsKey("type")) {
      t = dev["type"].as<const char*>();
    }

    JsonObject desired = dev["desired"];
    if (desired.isNull()) continue;

    if (t == "light") {
      int idx = findLightIndex(id);
      if (idx >= 0) {
        applyLightDesiredFromJson(idx, desired);
      }
    } else if (t == "fan") {
      int idx = findFanIndex(id);
      if (idx >= 0) {
        applyFanDesiredFromJson(idx, desired);
      }
    } else if (t == "speaker") {
      int idx = findSpeakerIndex(id);
      if (idx >= 0) {
        applySpeakerDesiredFromJson(idx, desired);
      }
    }
  }

  unsigned long t1 = millis();
  Serial.printf("pollAllDesired took %lu ms\n", (t1 - t0));
}

/* ============================================================
   DOORS – đọc từ /doors/frontDoor & /doors/sideDoor
   ============================================================ */

void pollDoors() {
  static unsigned long lastDoorPoll = 0;
  unsigned long now = millis();
  if (now - lastDoorPoll < DOOR_POLL_INTERVAL) return;
  lastDoorPoll = now;

  for (int i = 0; i < 2; i++) {
    if (doors[i].pin < 0) continue;

    String url = FIREBASE_DOORS_BASE + "/" + doors[i].name + "/state.json";

    HTTPClient http;
    http.setReuse(true);
    http.begin(url);
    http.setTimeout(300);
    int code = http.GET();
    if (code != 200) {
      http.end();
      continue;
    }
    String payload = http.getString();
    http.end();

    // Firebase trả "open" hoặc "closed" (string)
    payload.toLowerCase();
    bool open = (payload.indexOf("open") >= 0);  // đơn giản: có chữ "open" là mở
    bool changed = (open != doors[i].lastState);

    if (changed) {
      doors[i].lastState = open;

      int angle = open ? DOOR_OPEN_ANGLE : DOOR_CLOSED_ANGLE;
      setServoAngle(doors[i].channel, angle);

      Serial.printf("🚪 Door %s -> %s\n", doors[i].name, open ? "OPEN" : "CLOSED");

      // Optional: ghi lại cùng state để xem như reported (không bắt buộc)
      String repUrl = FIREBASE_DOORS_BASE + "/" + doors[i].name + "/state.json";
      HTTPClient rep;
      rep.setReuse(true);
      rep.begin(repUrl);
      rep.addHeader("Content-Type", "application/json");
      String body = open ? "\"open\"" : "\"closed\"";
      rep.PATCH(body);
      rep.end();
    }
    vTaskDelay(pdMS_TO_TICKS(1));
  }
}

/* ============================================================
   LIGHTS – dùng Adafruit_NeoPixel
   ============================================================ */

void discoverDevices() {
  String url = FIREBASE_BASE + ".json?shallow=true";
  HTTPClient http;
  http.setReuse(true);
  http.begin(url);
  http.setTimeout(800);  // giảm timeout để không block lâu
  int code = http.GET();
  if (code != 200) {
    Serial.println("⚠ Discover failed, code=" + String(code));
    http.end();
    return;
  }
  String payload = http.getString();
  http.end();

  StaticJsonDocument<1024> doc;
  DeserializationError err = deserializeJson(doc, payload);
  if (err) {
    Serial.println("⚠ discover parse error");
    return;
  }

  // mark all inactive; sẽ set active=true nếu thấy trong JSON
  for (int i = 0; i < MAX_LIGHTS; i++) lights[i].active = false;

  for (JsonPair kv : doc.as<JsonObject>()) {
    const char* key = kv.key().c_str();
    int idx = findLightIndex(key);
    if (idx == -1) {
      int added = addLightById(key); // fetch chi tiết và init slot
      if (added >= 0) lights[added].active = true;
    } else {
      lights[idx].active = true;
    }
    vTaskDelay(pdMS_TO_TICKS(1));
  }

  // remove devices không còn trong Firebase
  removeDeletedLights();
}

int addLightById(const char* id) {
  int freeIdx = -1;
  for (int i = 0; i < MAX_LIGHTS; i++) {
    if (lights[i].pin < 0) { freeIdx = i; break; }
  }
  if (freeIdx == -1) {
    Serial.println("⚠ No free slot for " + String(id));
    return -1;
  }

  String url = FIREBASE_BASE + "/" + String(id) + ".json";
  HTTPClient http;
  http.setReuse(true);
  http.begin(url);
  http.setTimeout(600);
  int code = http.GET();
  if (code != 200) { http.end(); return -1; }
  String payload = http.getString();
  http.end();

  StaticJsonDocument<512> doc;
  DeserializationError err = deserializeJson(doc, payload);
  if (err) {
    Serial.println("⚠ addLight parse error");
    return -1;
  }
  JsonObject root = doc.as<JsonObject>();

  // --- đọc type / pin từ metadata trước, fallback root (compat) ---
  JsonObject meta = root["metadata"];
  String t = "";
  String pinStr = "";

  if (!meta.isNull()) {
    if (meta.containsKey("type")) t = meta["type"].as<const char*>();
    if (meta.containsKey("pin"))  pinStr = meta["pin"].as<const char*>();
  } else {
    if (root.containsKey("type")) t = root["type"].as<const char*>();
    if (root.containsKey("pin"))  pinStr = root["pin"].as<const char*>();
  }

  if (t.length() == 0 || pinStr.length() == 0) {
    Serial.println("⚠ Device " + String(id) + " missing type/pin");
    return -1;
  }

  // chỉ nhận loại light
  if (t != "light") {
    return -1;
  }

  pinStr.replace("GPIO", "");
  pinStr.replace("gpio", "");
  pinStr.trim();
  int gpio = pinStr.toInt();

  if (gpio < 0 || gpio > 39) {
    Serial.println("⚠ Invalid GPIO for " + String(id) + ": " + pinStr);
    return -1;
  }

  // initialize slot
  lights[freeIdx].id = String(id);
  lights[freeIdx].pin = gpio;
  lights[freeIdx].active = true;
  lights[freeIdx].lastPoll = 0;
  lights[freeIdx].lastBrightness = -1;
  lights[freeIdx].lastColor = 0xFFFFFF; // default trắng
  lights[freeIdx].lastState = false;    // lần poll đầu sẽ áp dụng
  lights[freeIdx].rainbow = false;
  lights[freeIdx].rainbowPos = 0;
  lights[freeIdx].lastRainbowTick = 0;
  lights[freeIdx].lastDesiredTs = 0; 

  // tạo NeoPixel object cho chân này
  if (lights[freeIdx].strip != nullptr) {
    lights[freeIdx].strip->clear();
    lights[freeIdx].strip->show();
    delete lights[freeIdx].strip;
    lights[freeIdx].strip = nullptr;
  }
  Adafruit_NeoPixel* s = new Adafruit_NeoPixel(1, gpio, NEO_GRB + NEO_KHZ800);
  s->begin();
  s->clear();
  s->show();  // tắt hẳn
  lights[freeIdx].strip = s;

  Serial.println("➕ Added LED " + String(id) + " at GPIO " + String(gpio));
  return freeIdx;
}

// áp dụng state/brightness/color xuống NeoPixel (normal mode)
void applyLightState(int idx) {
  if (idx < 0 || idx >= MAX_LIGHTS) return;
  if (!lights[idx].active || lights[idx].pin < 0) return;
  Adafruit_NeoPixel* s = lights[idx].strip;
  if (!s) return;

  if (!lights[idx].lastState) {
    // OFF: clear pixel & show → chắc chắn tắt
    s->clear();
    s->show();
    return;
  }

  // giải mã 0xRRGGBB
  uint8_t r = (lights[idx].lastColor >> 16) & 0xFF;
  uint8_t g = (lights[idx].lastColor >> 8)  & 0xFF;
  uint8_t b = (lights[idx].lastColor)       & 0xFF;

  int bri = lights[idx].lastBrightness;
  if (bri < 0) bri = 100;        // default full sáng
  if (bri > 100) bri = 100;

  // scale theo brightness 0..100
  r = (uint8_t)((r * bri) / 100);
  g = (uint8_t)((g * bri) / 100);
  b = (uint8_t)((b * bri) / 100);

  s->setPixelColor(0, s->Color(r, g, b));
  s->show();
}

// xử lý desired cho light từ JsonObject (không GET nữa)
void applyLightDesiredFromJson(int idx, JsonObject d) {
  if (idx < 0 || idx >= MAX_LIGHTS) return;
  if (!lights[idx].active || lights[idx].pin < 0) return;
  if (d.isNull()) return;

  bool needApply = false;

    unsigned long newTs = 0;
  if (d.containsKey("ts")) {
    newTs = (unsigned long)(d["ts"].as<unsigned long>());
  }

  bool prevState   = lights[idx].lastState;
  int  prevBri     = lights[idx].lastBrightness;
  uint32_t prevCol = lights[idx].lastColor;
  bool prevRainbow = lights[idx].rainbow;
  bool newRainbow  = prevRainbow;

  // power: ưu tiên "power" ("on"/"off"), fallback "state": true/false
  bool on = prevState;
  if (d.containsKey("power")) {
    String p = d["power"].as<const char*>();
    if (p == "on") on = true;
    else if (p == "off") on = false;
  } else if (d.containsKey("state")) {
    on = (bool)d["state"];
  }

  // === RAINBOW FLAG ===
  // 1) Nếu có desired.rainbow (bool) → ưu tiên, đúng với webapp scenes
  // 2) Nếu không có, vẫn đọc thêm "mode" = "rainbow" cho tương thích bản cũ
  // 3) Nếu cũng không có, mà có color/brightness → coi như user chỉnh tay → tắt rainbow
  if (d.containsKey("rainbow")) {
    newRainbow = (d["rainbow"] == true);
  } else if (d.containsKey("mode")) {
    const char* m = d["mode"].as<const char*>();
    if (m && String(m) == "rainbow") newRainbow = true;
    else newRainbow = false;
  } else {
    if (d.containsKey("color") || d.containsKey("brightness")) {
      newRainbow = false;
    }
  }

  int bri = prevBri;
  if (d.containsKey("brightness")) {
    bri = (int)d["brightness"];
  }

  uint32_t col = prevCol;
  if (d.containsKey("color")) {
    const char* hex = d["color"].as<const char*>();
    if (hex && hex[0] == '#' && strlen(hex) >= 7) {
      long num = strtol(&hex[1], NULL, 16);
      uint8_t r = (num >> 16) & 0xFF;
      uint8_t g = (num >> 8) & 0xFF;
      uint8_t b = num & 0xFF;
      col = ((uint32_t)r << 16) | ((uint32_t)g << 8) | b;  // 0xRRGGBB
    }
  }

  // ---- compare & lưu lại ----
  if (bri != lights[idx].lastBrightness && bri >= 0) {
    lights[idx].lastBrightness = bri;
    needApply = true;
  }

  if (col != lights[idx].lastColor) {
    lights[idx].lastColor = col;
    needApply = true;
  }

  if (on != prevState) {
    lights[idx].lastState = on;
    needApply = true;
  }

  if (newRainbow != prevRainbow) {
    lights[idx].rainbow = newRainbow;
    needApply = true;
    if (newRainbow) {
      lights[idx].rainbowPos = 0;
      lights[idx].lastRainbowTick = 0;
    }
  }

  bool tsChanged = (newTs != 0 && newTs != lights[idx].lastDesiredTs);
  if (tsChanged) {
    // scene / UI vừa apply lại: bắt buộc re-apply
    needApply = true;
  }

  if (needApply) {
    // nếu không ở chế độ rainbow thì apply màu tĩnh
    if (!lights[idx].rainbow) {
      applyLightState(idx);
    }
    // nếu rainbow = true, LED sẽ được update trong rainbowTask()

    // ---- ghi lại reported/ để UI thấy trạng thái thật ----
    String repUrl = FIREBASE_BASE + "/" + lights[idx].id + "/reported.json";
    StaticJsonDocument<256> rep;

    rep["power"]      = lights[idx].lastState ? "on" : "off";
    rep["brightness"] = lights[idx].lastBrightness;
    rep["mode"]       = lights[idx].rainbow ? "rainbow" : "normal";
    rep["rainbow"]    = lights[idx].rainbow;  // để webapp đọc trực tiếp reported.rainbow

    // color cho UI: nếu rainbow thì dùng màu party cố định, không lấy lastColor
    if (lights[idx].rainbow) {
      rep["color"] = "#FF2FFF";   // màu hồng tím party trên webapp
    } else {
      // convert lastColor -> "#RRGGBB"
      uint8_t r = (lights[idx].lastColor >> 16) & 0xFF;
      uint8_t g = (lights[idx].lastColor >> 8)  & 0xFF;
      uint8_t b = (lights[idx].lastColor)       & 0xFF;
      char colorHex[8];
      snprintf(colorHex, sizeof(colorHex), "#%02X%02X%02X", r, g, b);
      rep["color"] = colorHex;
    }

    rep["ts"]         = millis();
    rep["updated_by"] = "device";

    String body;
    serializeJson(rep, body);

    HTTPClient repHttp;
    repHttp.setReuse(true);
    repHttp.begin(repUrl);
    repHttp.addHeader("Content-Type", "application/json");
    repHttp.setTimeout(250);
    int repCode = repHttp.PATCH(body);
    repHttp.end();

    Serial.printf("Reported %s (on=%d bri=%d rainbow=%d) -> %d\n",
                  lights[idx].id.c_str(),
                  lights[idx].lastState,
                  lights[idx].lastBrightness,
                  lights[idx].rainbow,
                  repCode);
  }
  if (newTs != 0) {
    lights[idx].lastDesiredTs = newTs;
  }
}

int findLightIndex(const char* id) {
  for (int i = 0; i < MAX_LIGHTS; i++) {
    if (lights[i].pin >= 0 && lights[i].id == id) return i;
  }
  return -1;
}

void removeDeletedLights() {
  for (int i = 0; i < MAX_LIGHTS; i++) {
    if (lights[i].pin >= 0 && !lights[i].active) {
      Serial.println("❌ Removed LED: " + lights[i].id);
      if (lights[i].strip != nullptr) {
        lights[i].strip->clear();
        lights[i].strip->show();
        delete lights[i].strip;
        lights[i].strip = nullptr;
      }
      lights[i].id = "";
      lights[i].pin = -1;
      lights[i].active = false;
      lights[i].lastColor = 0xFFFFFF;
      lights[i].lastBrightness = -1;
      lights[i].lastState = false;
      lights[i].rainbow = false;
      lights[i].rainbowPos = 0;
      lights[i].lastRainbowTick = 0;
      lights[i].lastDesiredTs = 0;
    }
  }
}

/* ============================================================
   FAN (SERVO) – QUẠT / CỬA (dạng device động trong /devices)
   ============================================================ */

void setServoAngle(int channel, int angle) {
  angle = constrain(angle, 0, 180);
  int us = map(angle, 0, 180, 500, 2500);   // 0..180° -> 0.5..2.5ms
  uint32_t maxDuty = (1 << SERVO_RESOLUTION) - 1;
  uint32_t duty = (uint32_t)((us * maxDuty) / 20000); // chu kỳ 20ms (50Hz)
  ledcWrite(channel, duty);
}

void discoverFans() {
  String url = FIREBASE_BASE + ".json?shallow=true";
  HTTPClient http;
  http.setReuse(true);
  http.begin(url);
  http.setTimeout(800);
  int code = http.GET();
  if (code != 200) {
    http.end();
    return;
  }
  String payload = http.getString();
  http.end();

  StaticJsonDocument<1024> doc;
  DeserializationError err = deserializeJson(doc, payload);
  if (err) return;

  for (int i = 0; i < MAX_FANS; i++) fans[i].active = false;

  for (JsonPair kv : doc.as<JsonObject>()) {
    const char* key = kv.key().c_str();
    int idx = findFanIndex(key);
    if (idx == -1) {
      int added = addFanById(key);
      if (added >= 0) fans[added].active = true;
    } else {
      fans[idx].active = true;
    }
    vTaskDelay(pdMS_TO_TICKS(1));
  }

  removeDeletedFans();
}

int addFanById(const char* id) {
  int freeIdx = -1;
  for (int i = 0; i < MAX_FANS; i++) {
    if (fans[i].pin < 0) { freeIdx = i; break; }
  }
  if (freeIdx == -1) {
    Serial.println("⚠ No free FAN slot for " + String(id));
    return -1;
  }

  String url = FIREBASE_BASE + "/" + String(id) + ".json";
  HTTPClient http;
  http.setReuse(true);
  http.begin(url);
  http.setTimeout(600);
  int code = http.GET();
  if (code != 200) { http.end(); return -1; }
  String payload = http.getString();
  http.end();

  StaticJsonDocument<512> doc;
  DeserializationError err = deserializeJson(doc, payload);
  if (err) return -1;
  JsonObject root = doc.as<JsonObject>();

  JsonObject meta = root["metadata"];
  String t = "";
  String pinStr = "";
  String name = "";

  if (!meta.isNull()) {
    if (meta.containsKey("type")) t = meta["type"].as<const char*>();
    if (meta.containsKey("pin"))  pinStr = meta["pin"].as<const char*>();
    if (meta.containsKey("name")) name  = meta["name"].as<const char*>();
  } else {
    if (root.containsKey("type")) t = root["type"].as<const char*>();
    if (root.containsKey("pin"))  pinStr = root["pin"].as<const char*>();
    if (root.containsKey("name")) name  = root["name"].as<const char*>();
  }

  if (t != "fan" || pinStr.length() == 0) {
    // chỉ dùng servo cho type = fan (quạt / cửa)
    return -1;
  }

  // fan có tên chứa "door" hoặc "cửa" -> coi là cửa (chỉ on/off)
  String nameLower = name;
  nameLower.toLowerCase();
  bool isDoor = nameLower.indexOf("door") >= 0 || nameLower.indexOf("cửa") >= 0;

  pinStr.replace("GPIO", "");
  pinStr.replace("gpio", "");
  pinStr.trim();
  int gpio = pinStr.toInt();
  if (gpio < 0 || gpio > 39) return -1;

  fans[freeIdx].id = String(id);
  fans[freeIdx].pin = gpio;
  fans[freeIdx].active = true;
  fans[freeIdx].isDoor = isDoor;
  fans[freeIdx].lastPoll = 0;
  fans[freeIdx].lastState = false;
  fans[freeIdx].lastSpeed = 0;

  int channel = freeIdx;  // dùng kênh 0..MAX_FANS-1 cho servo
  fans[freeIdx].channel = channel;
  ledcSetup(channel, SERVO_FREQ, SERVO_RESOLUTION);
  ledcAttachPin(gpio, channel);

  // khởi tạo: cửa → đóng, quạt → dừng
  int initAngle = isDoor ? DOOR_CLOSED_ANGLE : FAN_STOP_ANGLE;
  setServoAngle(channel, initAngle);

  Serial.print("➕ Added ");
  Serial.print(isDoor ? "DOOR " : "FAN ");
  Serial.print(id);
  Serial.print(" at GPIO ");
  Serial.println(gpio);

  return freeIdx;
}

// xử lý desired cho FAN/DOOR từ JsonObject
void applyFanDesiredFromJson(int idx, JsonObject d) {
  if (idx < 0 || idx >= MAX_FANS) return;
  if (!fans[idx].active || fans[idx].pin < 0) return;
  if (d.isNull()) return;

  bool prevState = fans[idx].lastState;
  int  prevSpeed = fans[idx].lastSpeed;

  bool on = prevState;
  if (d.containsKey("power")) {
    String p = d["power"].as<const char*>();
    if (p == "on") on = true;
    else if (p == "off") on = false;
  } else if (d.containsKey("state")) {
    on = (bool)d["state"];
  }

  int speed = prevSpeed;
  if (d.containsKey("speed")) {
    speed = (int)d["speed"];
  }
  if (speed < 0) speed = 0;
  if (speed > 3) speed = 3;

  bool changed = (on != prevState) || (speed != prevSpeed);

  if (changed) {
    fans[idx].lastState = on;
    fans[idx].lastSpeed = speed;  // với cửa thì speed chỉ để UI hiển thị, servo bỏ qua

    int angle;
    if (fans[idx].isDoor) {
      // CỬA: chỉ 2 trạng thái đóng / mở
      angle = on ? DOOR_OPEN_ANGLE : DOOR_CLOSED_ANGLE;
    } else {
      // QUẠT: servo quay liên tục → 90° là đứng yên
      if (!on || speed == 0)       angle = FAN_STOP_ANGLE;    // STOP
      else if (speed == 1)         angle = FAN_SPEED1_ANGLE;  // chậm
      else if (speed == 2)         angle = FAN_SPEED2_ANGLE;  // trung bình
      else                         angle = FAN_SPEED3_ANGLE;  // nhanh
    }

    setServoAngle(fans[idx].channel, angle);

    // Nếu là quạt và đang OFF thì cắt luôn PWM để servo quạt đứng hẳn
    if (!fans[idx].isDoor && !fans[idx].lastState) {
      ledcWrite(fans[idx].channel, 0);
    }

    // ghi reported
    String repUrl = FIREBASE_BASE + "/" + fans[idx].id + "/reported.json";
    StaticJsonDocument<200> rep;
    rep["power"] = fans[idx].lastState ? "on" : "off";

    // quạt thì báo cả speed, cửa thì speed chỉ để tham khảo (UI)
    if (!fans[idx].isDoor) {
      rep["speed"] = fans[idx].lastSpeed;
    }

    rep["ts"] = millis();
    rep["updated_by"] = "device";

    String body;
    serializeJson(rep, body);
    HTTPClient repHttp;
    repHttp.setReuse(true);
    repHttp.begin(repUrl);
    repHttp.addHeader("Content-Type", "application/json");
    repHttp.setTimeout(250);
    int repCode = repHttp.PATCH(body);
    repHttp.end();

    Serial.printf("Reported FAN/DOOR %s (door=%d on=%d speed=%d) -> %d\n",
                  fans[idx].id.c_str(),
                  fans[idx].isDoor,
                  fans[idx].lastState,
                  fans[idx].lastSpeed,
                  repCode);
  }
}

int findFanIndex(const char* id) {
  for (int i = 0; i < MAX_FANS; i++) {
    if (fans[i].pin >= 0 && fans[i].id == id) return i;
  }
  return -1;
}

void removeDeletedFans() {
  for (int i = 0; i < MAX_FANS; i++) {
    if (fans[i].pin >= 0 && !fans[i].active) {
      Serial.println("❌ Removed FAN/DOOR: " + fans[i].id);
      setServoAngle(fans[i].channel, DOOR_CLOSED_ANGLE);
      ledcDetachPin(fans[i].pin);
      fans[i].id = "";
      fans[i].pin = -1;
      fans[i].channel = -1;
      fans[i].lastState = false;
      fans[i].lastSpeed = 0;
      fans[i].isDoor = false;
    }
  }
}

/* ============================================================
   SPEAKER (BUZZER – LOW LEVEL TRIGGER)
   ============================================================ */

void setBuzzerVolume(int channel, int volume) {
  volume = constrain(volume, 0, 100);
  uint32_t maxDuty = (1 << BUZZER_RESOLUTION) - 1;

  // low-level trigger: LOW = kêu, HIGH = tắt
  // volume = 0  -> duty = maxDuty  (luôn HIGH -> tắt)
  // volume = 100-> duty ~ 0        (luôn LOW  -> to nhất)
  uint32_t duty;
  if (volume <= 0) duty = maxDuty;
  else duty = (uint32_t)((maxDuty * (100 - volume)) / 100);

  ledcWrite(channel, duty);
}

void discoverSpeakers() {
  String url = FIREBASE_BASE + ".json?shallow=true";
  HTTPClient http;
  http.setReuse(true);
  http.begin(url);
  http.setTimeout(800);
  int code = http.GET();
  if (code != 200) {
    http.end();
    return;
  }
  String payload = http.getString();
  http.end();

  StaticJsonDocument<1024> doc;
  DeserializationError err = deserializeJson(doc, payload);
  if (err) return;

  for (int i = 0; i < MAX_SPEAKERS; i++) speakers[i].active = false;

  for (JsonPair kv : doc.as<JsonObject>()) {
    const char* key = kv.key().c_str();
    int idx = findSpeakerIndex(key);
    if (idx == -1) {
      int added = addSpeakerById(key);
      if (added >= 0) speakers[added].active = true;
    } else {
      speakers[idx].active = true;
    }
    vTaskDelay(pdMS_TO_TICKS(1));
  }

  removeDeletedSpeakers();
}

int addSpeakerById(const char* id) {
  int freeIdx = -1;
  for (int i = 0; i < MAX_SPEAKERS; i++) {
    if (speakers[i].pin < 0) { freeIdx = i; break; }
  }
  if (freeIdx == -1) {
    Serial.println("⚠ No free SPEAKER slot for " + String(id));
    return -1;
  }

  String url = FIREBASE_BASE + "/" + String(id) + ".json";
  HTTPClient http;
  http.setReuse(true);
  http.begin(url);
  http.setTimeout(600);
  int code = http.GET();
  if (code != 200) { http.end(); return -1; }
  String payload = http.getString();
  http.end();

  StaticJsonDocument<512> doc;
  DeserializationError err = deserializeJson(doc, payload);
  if (err) return -1;
  JsonObject root = doc.as<JsonObject>();

  JsonObject meta = root["metadata"];
  String t = "";
  String pinStr = "";

  if (!meta.isNull()) {
    if (meta.containsKey("type")) t = meta["type"].as<const char*>();
    if (meta.containsKey("pin"))  pinStr = meta["pin"].as<const char*>();
  } else {
    if (root.containsKey("type")) t = root["type"].as<const char*>();
    if (root.containsKey("pin"))  pinStr = root["pin"].as<const char*>();
  }

  if (t != "speaker" || pinStr.length() == 0) {
    return -1;
  }

  pinStr.replace("GPIO", "");
  pinStr.replace("gpio", "");
  pinStr.trim();
  int gpio = pinStr.toInt();
  if (gpio < 0 || gpio > 39) return -1;

  speakers[freeIdx].id = String(id);
  speakers[freeIdx].pin = gpio;
  speakers[freeIdx].active = true;
  speakers[freeIdx].lastPoll = 0;
  speakers[freeIdx].lastState = false;
  speakers[freeIdx].lastVolume = 0;
  speakers[freeIdx].lastTrack = 0;
  speakers[freeIdx].pwmAttached = false;
  speakers[freeIdx].lastDesiredTs = 0;

  int channel = 8 + freeIdx;   // kênh khác với servo
  speakers[freeIdx].channel = channel;

  // Cấu hình LEDC nhưng KHÔNG attach vào chân lúc khởi động
  ledcSetup(channel, BUZZER_FREQ, BUZZER_RESOLUTION);

  // Giữ chân ở mức HIGH (idle tắt) đến khi thật sự bật
  pinMode(gpio, OUTPUT);
  digitalWrite(gpio, HIGH);

  Serial.println("➕ Added SPEAKER " + String(id) + " at GPIO " + String(gpio));
  return freeIdx;
}

void applySpeakerDesiredFromJson(int idx, JsonObject d) {
  if (idx < 0 || idx >= MAX_SPEAKERS) return;
  if (!speakers[idx].active || speakers[idx].pin < 0) return;
  if (d.isNull()) return;

  bool prevState = speakers[idx].lastState;
  int  prevVol   = speakers[idx].lastVolume;
  int  prevTrack = speakers[idx].lastTrack;   // dùng track theo device

  unsigned long prevTs = speakers[idx].lastDesiredTs;
  unsigned long newTs  = 0;
  if (d.containsKey("ts")) {
    newTs = (unsigned long)d["ts"].as<unsigned long>();
  }

  // 1. Đọc power/state
  bool on = prevState;
  bool hasPowerKey = d.containsKey("power");
  bool hasStateKey = d.containsKey("state");

  if (hasPowerKey) {
    String p = d["power"].as<const char*>();
    if (p == "on")       on = true;
    else if (p == "off") on = false;
  } else if (hasStateKey) {
    on = (bool)d["state"];
  }

  // 2. Đọc volume 0..100
  int vol = prevVol;
  if (d.containsKey("volume")) {
    vol = (int)d["volume"];
  }
  if (vol < 0)   vol = 0;
  if (vol > 100) vol = 100;

  // SAFETY:
  // Nếu JSON KHÔNG có power/state mà lại có volume > 0
  // -> hiểu là muốn bật loa
  if (!hasPowerKey && !hasStateKey && vol > 0) {
    on = true;
  }

  // 3. Đọc track_index cho JQ6500
  int track = prevTrack;
  if (d.containsKey("track_index")) {
    track = (int)d["track_index"];
  }

  // Nếu chưa bao giờ có track mà giờ bật loa thì ép về 1
  if (track <= 0 && (on && vol > 0)) {
    track = 1;
  }

  // 4. Kiểm tra đổi gì chưa (kể cả ts)
  bool tsChanged = (newTs != 0 && newTs != prevTs);
  bool changed   = (on    != prevState) ||
                   (vol   != prevVol)   ||
                   (track != prevTrack) ||
                   tsChanged;

  // Trường hợp đặc biệt:
  // - Firebase vẫn yêu cầu BẬT loa (on && vol > 0)
  // - Nhưng mp3On = false
  // => ép xử lý lại để gọi applyJQ6500
  if (!changed && mp3Ready && on && vol > 0 && !mp3On) {
    Serial.println("Speaker: desired unchanged nhưng mp3On=0 -> ép apply lại JQ6500");
    changed = true;
  }

  if (!changed) {
    return;
  }

  // 5. Cập nhật lại state trong struct
  speakers[idx].lastState  = on;
  speakers[idx].lastVolume = vol;
  speakers[idx].lastTrack  = track;

  // 6. Điều khiển buzzer cũ (nếu vẫn cắm MH-FMD)
  int effVol = (on && vol > 0) ? vol : 0;

  if (effVol > 0) {
    // Bật: attach PWM + set volume
    if (!speakers[idx].pwmAttached) {
      ledcAttachPin(speakers[idx].pin, speakers[idx].channel);
      speakers[idx].pwmAttached = true;
    }
    setBuzzerVolume(speakers[idx].channel, effVol);
  } else {
    // Tắt: detach PWM + kéo HIGH → buzzer im
    if (speakers[idx].pwmAttached) {
      ledcDetachPin(speakers[idx].pin);
      speakers[idx].pwmAttached = false;
    }
    pinMode(speakers[idx].pin, OUTPUT);
    digitalWrite(speakers[idx].pin, HIGH);
  }

  // 7. Điều khiển JQ6500 theo desired từ frontend
  bool jqOn = (on && vol > 0);
  applyJQ6500(jqOn, track, vol);

  // 8. Ghi reported cho UI
  String repUrl = FIREBASE_BASE + "/" + speakers[idx].id + "/reported.json";
  StaticJsonDocument<200> rep;
  rep["power"]       = speakers[idx].lastState ? "on" : "off";
  rep["volume"]      = speakers[idx].lastVolume;
  rep["track_index"] = speakers[idx].lastTrack;
  rep["ts"]          = millis();
  rep["updated_by"]  = "device";

  String body;
  serializeJson(rep, body);
  HTTPClient repHttp;
  repHttp.setReuse(true);
  repHttp.begin(repUrl);
  repHttp.addHeader("Content-Type", "application/json");
  repHttp.setTimeout(250);
  int repCode = repHttp.PATCH(body);
  repHttp.end();

  Serial.printf("Reported SPEAKER %s (on=%d vol=%d track=%d) -> %d\n",
                speakers[idx].id.c_str(),
                speakers[idx].lastState,
                speakers[idx].lastVolume,
                speakers[idx].lastTrack,
                repCode);

  // 9. Lưu lại ts mới
  if (newTs != 0) {
    speakers[idx].lastDesiredTs = newTs;
  }
}

void applyJQ6500(bool on, int track, int volume) {
  if (!mp3Ready) {
    Serial.println("JQ6500 not ready, skip");
    // Không đổi mp3On, mp3CurrentTrack, để lần sau khi ready
    // applySpeakerDesiredFromJson sẽ ép gọi lại.
    return;
  }

  Serial.printf("JQ6500 apply: on=%d track=%d vol=%d (currTrack=%d mp3On=%d)\n",
                on, track, volume, mp3CurrentTrack, mp3On);

  // === CASE TẮT LOA / VOLUME = 0 ===
  if (!on || volume <= 0) {
    if (mp3On) {
      Serial.println("JQ6500: stop()");
      mp3.pause();            // dừng hẳn bài hiện tại
      delay(50);
      mp3On = false;
      mp3StartMillis = 0;    // ngừng watchdog
    }
    return;
  }

  // === BẬT LOA ===
  // Nếu track <= 0 thì dùng track hiện tại, nếu chưa có thì ép về 1
  if (track <= 0) {
    track = (mp3CurrentTrack > 0) ? mp3CurrentTrack : 1;
  }

  // Cần hard-switch khi:
  //  - trước đó đang tắt (mp3On = false), hoặc
  //  - đổi sang bài khác
  bool needHardSwitch = (!mp3On) || (track != mp3CurrentTrack);

  if (needHardSwitch) {
    Serial.printf("JQ6500: HARD SWITCH -> track %d\n", track);

    // Dừng bài cũ nếu đang phát
    mp3.pause();
    delay(50);

    // Reset module + chọn lại source (giống sketch test bạn vừa chạy)
    mp3.reset();
    delay(200);
    mp3.setSource(MP3_SRC_BUILTIN);
    delay(50);

    // Phát bài mới
    mp3.playFileByIndexNumber(track);
    mp3CurrentTrack = track;

    // Bật chế độ loop 1 bài
    delay(100);
    mp3.setLoopMode(MP3_LOOP_ONE);

    mp3StartMillis = millis();  // đánh dấu bắt đầu bài mới cho watchdog
  }

  // Map volume 0..100 (Firebase) sang 0..30 (JQ6500)
  int v = map(volume, 0, 100, 0, 30);
  Serial.printf("JQ6500: setVolume(%d)\n", v);
  mp3.setVolume(v);

  mp3On = true;
}

int findSpeakerIndex(const char* id) {
  for (int i = 0; i < MAX_SPEAKERS; i++) {
    if (speakers[i].pin >= 0 && speakers[i].id == id) return i;
  }
  return -1;
}

void removeDeletedSpeakers() {
  for (int i = 0; i < MAX_SPEAKERS; i++) {
    if (speakers[i].pin >= 0 && !speakers[i].active) {
      Serial.println("❌ Removed SPEAKER: " + speakers[i].id);
      if (speakers[i].pwmAttached) {
        ledcDetachPin(speakers[i].pin);
        speakers[i].pwmAttached = false;
      }
      pinMode(speakers[i].pin, OUTPUT);
      digitalWrite(speakers[i].pin, HIGH); // idle tắt

      speakers[i].id = "";
      speakers[i].pin = -1;
      speakers[i].channel = -1;
      speakers[i].lastState = false;
      speakers[i].lastVolume = 0;
      speakers[i].lastTrack = 0;
      speakers[i].lastDesiredTs = 0; 
    }
  }
}

/* ============================================================
   DHT11 SENSOR → reported.temperature / humidity
   ============================================================ */

void discoverSensors() {
  String url = FIREBASE_BASE + ".json?shallow=true";
  HTTPClient http;
  http.setReuse(true);
  http.begin(url);
  http.setTimeout(800);
  int code = http.GET();
  if (code != 200) {
    http.end();
    return;
  }
  String payload = http.getString();
  http.end();

  StaticJsonDocument<1024> doc;
  DeserializationError err = deserializeJson(doc, payload);
  if (err) return;

  for (int i = 0; i < MAX_SENSORS; i++) sensors[i].active = false;

  for (JsonPair kv : doc.as<JsonObject>()) {
    const char* key = kv.key().c_str();
    int idx = findSensorIndex(key);
    if (idx == -1) {
      int added = addSensorById(key);
      if (added >= 0) sensors[added].active = true;
    } else {
      sensors[idx].active = true;
    }
    vTaskDelay(pdMS_TO_TICKS(1));
  }

  removeDeletedSensors();
}

int addSensorById(const char* id) {
  int freeIdx = -1;
  for (int i = 0; i < MAX_SENSORS; i++) {
    if (sensors[i].pin < 0) { freeIdx = i; break; }
  }
  if (freeIdx == -1) {
    Serial.println("⚠ No free SENSOR slot for " + String(id));
    return -1;
  }

  String url = FIREBASE_BASE + "/" + String(id) + ".json";
  HTTPClient http;
  http.setReuse(true);
  http.begin(url);
  http.setTimeout(600);
  int code = http.GET();
  if (code != 200) { http.end(); return -1; }
  String payload = http.getString();
  http.end();

  StaticJsonDocument<512> doc;
  DeserializationError err = deserializeJson(doc, payload);
  if (err) return -1;
  JsonObject root = doc.as<JsonObject>();

  JsonObject meta = root["metadata"];
  String t = "";
  String pinStr = "";

  if (!meta.isNull()) {
    if (meta.containsKey("type")) t = meta["type"].as<const char*>();
    if (meta.containsKey("pin"))  pinStr = meta["pin"].as<const char*>();
  } else {
    if (root.containsKey("type")) t = root["type"].as<const char*>();
    if (root.containsKey("pin"))  pinStr = root["pin"].as<const char*>();
  }

  // chấp nhận cả "sensor" lẫn "thermostat" cho DHT11
  if (!((t == "sensor") || (t == "thermostat")) || pinStr.length() == 0) {
    return -1;
  }

  pinStr.replace("GPIO", "");
  pinStr.replace("gpio", "");
  pinStr.trim();
  int gpio = pinStr.toInt();
  if (gpio < 0 || gpio > 39) return -1;

  sensors[freeIdx].id = String(id);
  sensors[freeIdx].pin = gpio;
  sensors[freeIdx].active = true;
  sensors[freeIdx].lastPoll = 0;
  sensors[freeIdx].lastTemp = NAN;
  sensors[freeIdx].lastHum = NAN;

  sensors[freeIdx].dht = new DHT(gpio, DHT_TYPE);
  sensors[freeIdx].dht->begin();

  Serial.println("➕ Added DHT SENSOR " + String(id) + " at GPIO " + String(gpio));
  return freeIdx;
}

void pollSensorDevice(int idx) {
  if (idx < 0 || idx >= MAX_SENSORS) return;
  if (!sensors[idx].active || sensors[idx].pin < 0 || sensors[idx].dht == nullptr) return;

  float h = sensors[idx].dht->readHumidity();
  float t = sensors[idx].dht->readTemperature(); // °C

  if (isnan(h) || isnan(t)) {
    Serial.printf("⚠ DHT read error (%s)\n", sensors[idx].id.c_str());
    return;
  }

  sensors[idx].lastTemp = t;
  sensors[idx].lastHum = h;

  String repUrl = FIREBASE_BASE + "/" + sensors[idx].id + "/reported.json";
  StaticJsonDocument<200> rep;
  rep["temperature"] = t;
  rep["humidity"] = h;
  rep["ts"] = millis();
  rep["updated_by"] = "device";

  String body;
  serializeJson(rep, body);
  HTTPClient http;
  http.setReuse(true);
  http.begin(repUrl);
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(250);
  int code = http.PATCH(body);
  http.end();

  Serial.printf("Reported SENSOR %s (T=%.1fC H=%.1f%%) -> %d\n",
                sensors[idx].id.c_str(), t, h, code);
}

int findSensorIndex(const char* id) {
  for (int i = 0; i < MAX_SENSORS; i++) {
    if (sensors[i].pin >= 0 && sensors[i].id == id) return i;
  }
  return -1;
}

void removeDeletedSensors() {
  for (int i = 0; i < MAX_SENSORS; i++) {
    if (sensors[i].pin >= 0 && !sensors[i].active) {
      Serial.println("❌ Removed SENSOR: " + sensors[i].id);
      if (sensors[i].dht != nullptr) {
        delete sensors[i].dht;
        sensors[i].dht = nullptr;
      }
      sensors[i].id = "";
      sensors[i].pin = -1;
      sensors[i].active = false;
      sensors[i].lastTemp = NAN;
      sensors[i].lastHum = NAN;
      sensors[i].lastPoll = 0;
    }
  }
}
