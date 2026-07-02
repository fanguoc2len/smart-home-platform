# Architecture

DoAn2 is organized around three runtime surfaces: the browser dashboard, the
Flask AI gateway, and the device synchronization layer.

## Browser Dashboard

- `templates/index.html` handles face authentication and PIN fallback.
- `templates/home.html` renders the smart-home dashboard, device cards, scenes,
  voice controls, and local UI state.
- `templates/firebase.js` adapts the dashboard to either Firebase Realtime
  Database or a local Raspberry Pi compatible backend.
- `static/` contains the PWA manifest, service worker, icons, and shared visual
  polish.

## Flask AI Gateway

`app.py` owns:

- face registration and recognition endpoints
- PIN login endpoint
- Vietnamese text normalization and voice-command parsing
- optional Whisper transcription for uploaded audio
- Firebase command writes and scene execution
- optional ngrok tunnel startup for mobile demos

The face database and captured face images are runtime data, not source code,
so they are ignored by Git.

## Device Sync Layer

The app supports two command paths:

- Firebase desired/reported state for online demos and ESP32 sync
- Pi/local JSON backend for LAN-only fallback demos

`BACKEND_CANONICAL.md` documents the preferred device schema. New device code
should write desired state first and let firmware report the actual hardware
state back.

## Firmware

The tracked Arduino sketch, `sketch_nov16c.ino`, is the legacy ESP32 path used
by the original project. The newer native HomeKit firmware was split into the
separate `DoAn2-HomeKit` repository to keep this repo focused on the original
full-stack system.
