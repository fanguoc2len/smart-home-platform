# Setup

## Python Environment

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` for local secrets and device endpoints. Do not commit `.env`.

## Run The Flask App

```bash
python app.py
```

The default app URL is `http://127.0.0.1:5000`.

If the machine cannot download MobileNet ImageNet weights, use a development
boot:

```bash
MOBILENET_WEIGHTS=none python app.py
```

## Runtime Modes

Firebase mode is the default browser control path. It writes smart-home desired
state into Firebase Realtime Database and expects firmware or another device
service to report actual state back.

Pi mode can be enabled from the browser console or UI integration by setting
`backend_mode` to `pi` in localStorage. It talks to the local Flask-compatible
device endpoint configured by `SMART_HOME_PI_BASE_URL`.

## Optional Services

- `ENABLE_NGROK=1` starts a public ngrok tunnel for phone demos.
- `NGROK_AUTHTOKEN` is needed for reliable ngrok sessions.
- `WHISPER_MODEL` controls the local Whisper model used for audio commands.
- `VOICE_EXECUTE_DEFAULT=1` allows voice commands to execute immediately.

## Hardware Notes

The original ESP32/Arduino path is `sketch_nov16c.ino`. Keep Wi-Fi credentials
and Firebase secrets out of the sketch before sharing a public repository.
