# voice_debug.py
# Dùng để test Whisper với các file .m4a/.wav có sẵn,
# và xem kết quả RAW + NORMALIZED giống như trong flaskai.py

import os
import sys

import torch
import whisper

# IMPORT HÀM CHUẨN HOÁ TỪ flaskai.py
from flaskai import normalize_text_for_command

# ==== CẤU HÌNH MODEL ====
DEFAULT_MODEL_NAME = os.environ.get("WHISPER_MODEL_NAME", "medium")

def load_whisper_model():
    model_name = DEFAULT_MODEL_NAME
    if torch.cuda.is_available():
        device = "cuda"
        print(f"[Whisper] Dùng GPU: {torch.cuda.get_device_name(0)}  |  model = {model_name}")
    else:
        device = "cpu"
        print(f"[Whisper] Không có GPU, chạy CPU  |  model = {model_name}")

    model = whisper.load_model(model_name, device=device)
    return model


def transcribe_file(model, filepath: str):
    print("=" * 80)
    print("FILE:", filepath)

    # Cho Whisper tự decode m4a, không cần ffmpeg thủ công
    result = model.transcribe(filepath, language="vi")
    raw = (result.get("text") or "").strip()

    print("RAW TEXT      :", repr(raw))

    # Chuẩn hoá giống hệt khi chạy Flask
    norm = normalize_text_for_command(raw)
    print("NORMALIZED    :", repr(norm))
    print()


def main():
    # Thư mục chứa các file audio – mặc định: thư mục hiện tại
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    print("Thư mục audio đang quét:", base_dir)

    # Lấy tất cả file audio trong thư mục
    exts = (".m4a", ".wav", ".mp3", ".flac")
    audio_files = [
        os.path.join(base_dir, f)
        for f in os.listdir(base_dir)
        if f.lower().endswith(exts)
    ]

    if not audio_files:
        print("❌ Không tìm thấy file .m4a/.wav/.mp3 nào trong thư mục.")
        print("→ Hãy copy mấy file 'bật hết...', 'tắt hết...', 'scenes...' vào cùng thư mục với voice_debug.py")
        return

    print(f"✅ Tìm được {len(audio_files)} file audio:")
    for f in audio_files:
        print("  -", os.path.basename(f))

    print("\n=== BẮT ĐẦU LOAD WHISPER ===")
    model = load_whisper_model()
    print("=== BẮT ĐẦU CHẠY TỪNG FILE ===\n")

    for path in audio_files:
        transcribe_file(model, path)


if __name__ == "__main__":
    main()
