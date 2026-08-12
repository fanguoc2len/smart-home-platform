# DoAn2 Smart Home

[![Python Smoke Checks](https://github.com/fanguoc2len/DoAn2/actions/workflows/python-smoke.yml/badge.svg)](https://github.com/fanguoc2len/DoAn2/actions/workflows/python-smoke.yml)

Full-stack smart-home prototype combining a Flask AI gateway, face/PIN access,
Vietnamese voice commands, Firebase Realtime Database sync, and an ESP32/Arduino
device layer.

This repository is kept as the original DoAn2 system. A separate native
HomeKit migration lives in `DoAn2-HomeKit`.

## Highlights

- Face access flow with OpenCV/MediaPipe detection and MobileNetV2 embeddings
- PIN fallback for camera-free login
- Vietnamese voice-command parser with optional Whisper transcription
- Firebase Realtime Database bridge for desired/reported device state
- Raspberry Pi/local Flask backup mode for device control
- Smart-home dashboard with rooms, scenes, device cards, and PWA assets
- Arduino ESP32 sketch for hardware integration

## Architecture

```mermaid
flowchart LR
    User["Browser / PWA"] --> Flask["Flask AI gateway"]
    Camera["Camera"] --> Face["Face embedding pipeline"]
    Voice["Vietnamese voice input"] --> Speech["Voice parser / Whisper option"]
    PIN["PIN fallback"] --> Flask
    Face --> Flask
    Speech --> Flask

    User <-->|"desired / reported state"| Firebase["Firebase Realtime Database"]
    Flask <-->|"authentication and commands"| Firebase
    Firebase <-->|"device state"| ESP32["ESP32 / Arduino firmware"]
    ESP32 --> Devices["Lights, sensors and actuators"]

    User <-->|"LAN-only JSON state"| Pi["Raspberry Pi backup API"]
```

The cloud path synchronizes desired and reported state through Firebase. The
Raspberry Pi API provides a local fallback, while Flask owns authentication,
face processing, and voice-command handling.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for a fuller module map.

## Quick Start

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python app.py
```

Open `http://127.0.0.1:5000`.

For a lightweight development boot without downloading ImageNet weights, set:

```bash
MOBILENET_WEIGHTS=none python app.py
```

## Configuration

Runtime settings are read from environment variables. Start with
[.env.example](.env.example), then keep the real `.env` local.

Important variables:

| Variable | Purpose |
| --- | --- |
| `ADMIN_REGISTER_PASSWORD` | Password required to register a new face |
| `SMART_HOME_PIN` | PIN fallback for login |
| `REGISTERED_DB` | Local face registry JSON path |
| `REGISTERED_IMAGES_DIR` | Local face image directory |
| `FIREBASE_DATABASE_URL` | Firebase Realtime Database base URL |
| `WHISPER_MODEL` | Whisper model name for voice commands |
| `ENABLE_NGROK` | Enables optional public ngrok tunnel |

## Verification

Hardware-free checks:

```bash
python3 scripts/check_repo_hygiene.py
python3 -m py_compile app.py backup_server_pi.py flask_ai_server.py flaskai.py voice_debug.py test_voice.py testgiongnoi.py whisper_test.py
```

The GitHub Actions workflow runs the same smoke checks on every push and pull
request.

## Repository Hygiene

Real face images, local registration databases, audio recordings, cache files,
virtual environments, and local ESP-IDF/HomeKit build folders are intentionally
ignored. This keeps sensitive local data out of version control while
preserving local demo data on the development machine.

## Docs

- [docs/SETUP.md](docs/SETUP.md): local setup and runtime modes
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): module and data-flow overview
- [docs/SECURITY.md](docs/SECURITY.md): privacy and deployment notes
- [BACKEND_CANONICAL.md](BACKEND_CANONICAL.md): Firebase desired/reported state contract
