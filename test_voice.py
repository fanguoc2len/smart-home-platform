# transcribe_audio_vi.py
import os
import time
import subprocess
from pathlib import Path

import torch
import whisper

# ===================== CẤU HÌNH =====================
# Đường dẫn file audio (có thể .m4a/.mp3/.webm/.wav)
# Gợi ý: dùng raw string trên Windows, ví dụ r"E:\Đồ án 2\audio\loz.m4a"
audio_path = r"loz.m4a"

# Nơi lưu kết quả
output_txt = "transcription.txt"

# Ngôn ngữ (cố định tiếng Việt để tăng độ chính xác và tốc độ)
language = "vi"

# Model Whisper (tiny/base/small/medium/large)
model_name = "small"

# ===================== HÀM PHỤ =====================
def ensure_wav(input_path: str) -> str:
    """
    Chuyển các định dạng .m4a/.mp3/.webm/... sang .wav (mono, 16kHz) bằng ffmpeg nếu cần.
    Nếu input đã là .wav thì giữ nguyên.
    Trả về đường dẫn .wav.
    """
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {p}")

    if p.suffix.lower() == ".wav":
        return str(p)

    out = p.with_suffix(".wav")
    cmd = ["ffmpeg", "-y", "-i", str(p), "-ac", "1", "-ar", "16000", str(out)]
    print("[INFO] ffmpeg:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        raise RuntimeError(f"ffmpeg convert failed: {e}")

    if not out.exists():
        raise FileNotFoundError(f"Không tìm thấy file sau khi convert: {out}")
    return str(out)

# ===================== MAIN =====================
def main():
    # --- Kiểm tra file đầu vào ---
    if not os.path.exists(audio_path):
        print(f"[ERR] Audio file '{audio_path}' not found!")
        print(r"👉 Gợi ý: dùng raw string, ví dụ r'E:\Đồ án 2\audio\loz.m4a'")
        return

    # --- Kiểm tra GPU ---
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        try:
            print("GPU:", torch.cuda.get_device_name(0))
        except Exception:
            pass

    # --- Chuẩn hóa audio (.m4a → .wav 16k mono) ---
    try:
        wav_path = ensure_wav(audio_path)
    except Exception as e:
        print("[ERR] Convert audio thất bại:", e)
        return
    print(f"[INFO] Sử dụng WAV: {wav_path}")

    # --- Load model Whisper ---
    print(f"Loading Whisper model ({model_name}) trên {device} ...")
    t0 = time.time()
    model = whisper.load_model(model_name, device=device)
    print(f"Model loaded in {time.time()-t0:.2f}s")

    # --- Transcribe ---
    print(f"Transcribing '{wav_path}' (language={language}) ...")
    t1 = time.time()
    try:
        result = model.transcribe(wav_path, language=language)
    except Exception as e:
        print("[ERR] Whisper transcribe lỗi:", e)
        return
    t2 = time.time()
    print(f"Transcription done in {t2 - t1:.2f}s")

    # --- Kết quả ---
    text = (result.get("text") or "").strip()
    print("Detected text:")
    print(text)

    # --- Lưu ra file ---
    try:
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[OK] Transcription saved to '{output_txt}'")
    except Exception as e:
        print("[WARN] Không lưu được file kết quả:", e)

if __name__ == "__main__":
    main()
