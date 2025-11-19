# transcribe_audio_vi.py
import os
import time
import subprocess
from pathlib import Path
import torch
import whisper

# --- Cấu hình ---
audio_path = r"loz2.m4a"          # đường dẫn file ghi âm
output_txt = "transcription.txt"  # nơi lưu kết quả
language = "vi"                   # tiếng Việt
model_name = "small"              # phù hợp nhất cho RTX 3050 4GB

# --- Prompt ngữ cảnh giúp Whisper hiểu đúng từ khóa ---
initial_prompt = (
    "Ngữ cảnh: điều khiển nhà thông minh. "
    "Các từ khóa quan trọng: bật, tắt, NeoPixel, Entry, phòng khách, đèn, "
    "màu đỏ, màu xanh, màu trắng, độ sáng, phần trăm, 50%, hành lang, cửa, quạt."
)

# --- Hàm đảm bảo .wav 16kHz mono ---
def ensure_wav(input_path: str) -> str:
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {p}")
    if p.suffix.lower() == ".wav":
        return str(p)
    out = p.with_suffix(".wav")
    cmd = ["ffmpeg", "-y", "-i", str(p), "-ac", "1", "-ar", "16000", str(out)]
    print("[INFO] ffmpeg:", " ".join(cmd))
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return str(out)

# --- Main ---
def main():
    if not os.path.exists(audio_path):
        print(f"[ERR] Audio file '{audio_path}' not found!")
        print(r"👉 Gợi ý: dùng raw string, ví dụ r'E:\Đồ án 2\audio\loz2.m4a'")
        return

    # Kiểm tra GPU
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # Chuẩn hóa audio
    wav_path = ensure_wav(audio_path)
    print(f"[INFO] Using WAV: {wav_path}")

    # Load model Whisper
    print(f"Loading Whisper model ({model_name}) on {device}...")
    start_time = time.time()
    model = whisper.load_model(model_name, device=device)
    print(f"Model loaded in {time.time() - start_time:.2f}s")

    # Transcribe với thiết lập tối ưu
    print(f"Transcribing '{wav_path}'...")
    t0 = time.time()
    result = model.transcribe(
        wav_path,
        language=language,
        initial_prompt=initial_prompt,
        temperature=0,
        beam_size=5,
        condition_on_previous_text=False,
        fp16=(device == "cuda")
    )
    t1 = time.time()
    print(f"Transcription done in {t1 - t0:.2f}s")

    # Kết quả
    text = (result.get("text") or "").strip()
    print("Detected text:\n", text)

    # Lưu ra file
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[OK] Transcription saved to '{output_txt}'")

if __name__ == "__main__":
    main()
