Canonical backend: `app.py`

Rules:
- Run `python3 app.py` for the main server.
- `flaskai.py` and `flask_ai_server.py` are compatibility wrappers only.
- Voice/control behavior must be implemented in `app.py`.
- Frontend voice requests should target `/voice` on the same origin by default.
- Device execution should flow through `devices/<id>/desired`, with ESP32 reporting to `reported`.

Current control flow:
1. Client sends audio/text to `/voice`.
2. `app.py` transcribes/parses the command.
3. Backend writes desired state to Firebase.
4. ESP32 reads desired state and applies hardware changes.
5. ESP32 publishes actual state to `reported`.
