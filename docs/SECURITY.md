# Security And Privacy Notes

This project handles sensitive local data during demos:

- face registration records
- captured face images
- voice recordings
- Firebase project endpoints
- optional Google Drive and ngrok credentials

## What Stays Out Of Git

The repository intentionally ignores:

- `.env` and local credential files
- `registered.json`
- `registered_images/`
- local face sample photos
- audio recordings such as `.m4a` and `.wav`
- Python cache files and virtual environments
- local ESP-IDF/HomeKit build folders

Use `.env.example` and `registered.example.json` as shareable templates.

## Deployment Checklist

- Change `ADMIN_REGISTER_PASSWORD`.
- Change `SMART_HOME_PIN`.
- Lock down Firebase Realtime Database rules.
- Rotate ngrok and Google Drive tokens if they were used during development.
- Do not publish real face images or voice recordings.
- Keep debug tunnel exposure disabled unless a demo requires it.

## Interview Demo Guidance

For interviews, show the source code, docs, and CI checks. Use synthetic or
empty face data unless you have explicit permission to show real people.
