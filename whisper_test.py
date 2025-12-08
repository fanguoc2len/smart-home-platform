# voice_benchmark.py
import time
import subprocess
from pathlib import Path

import torch
import whisper

# === FILE ÂM THANH CẦN TEST ===
# ĐỔI LẠI THEO FILE BẠN MUỐN TEST
TEST_AUDIO = r"E:\Đồ án 2\bathettatcathietbi.m4a"

VOICE_LANGUAGE = "vi"

INITIAL_PROMPT = (
    "Ngữ cảnh: điều khiển nhà thông minh bằng tiếng Việt. "
    "Các câu lệnh thường dùng: bật tất cả đèn, tắt tất cả thiết bị, "
    "bật chế độ party, bật chế độ relax, bật chế độ night, "
    "tăng độ sáng, giảm độ sáng, bật quạt mức 1, 2, 3."
)

def ensure_wav_16k_mono(path: str) -> str:
    p = Path(path)
    if p.suffix.lower() == ".wav":
        return str(p)

    out = p.with_suffix(".wav")
    cmd = ["ffmpeg", "-y", "-i", str(p), "-ac", "1", "-ar", "16000", str(out)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return str(out)

def transcribe_with_timing(model, model_name: str, wav_path: str) -> None:
    t0 = time.time()
    res = model.transcribe(
        wav_path,
        language=VOICE_LANGUAGE,
        initial_prompt=INITIAL_PROMPT,
        temperature=0.0,
        beam_size=1,
        best_of=1,
        condition_on_previous_text=False,
        fp16=(model.device == "cuda"),
        no_speech_threshold=0.4,
        logprob_threshold=-1.0,
        compression_ratio_threshold=2.4,
    )
    dt = time.time() - t0
    text = (res.get("text") or "").strip()
    print(f"[{model_name}] time = {dt:.2f}s")
    print(f"[{model_name}] text = {text!r}")
    print("-" * 60)

def main():
    print(">>> BẮT ĐẦU BENCHMARK WHISPER <<<")

    wav_path = ensure_wav_16k_mono(TEST_AUDIO)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device = {device}")

    print("Loading models small + medium...")
    t_s = time.time()
    model_small = whisper.load_model("small", device=device)
    t_s = time.time() - t_s

    t_m = time.time()
    model_medium = whisper.load_model("medium", device=device)
    t_m = time.time() - t_m

    print(f"[small] load = {t_s:.2f}s")
    print(f"[medium] load = {t_m:.2f}s")
    print("=" * 60)

    for i in range(2):
        print(f"--- ROUND {i+1} ---")
        transcribe_with_timing(model_small, "small", wav_path)
        transcribe_with_timing(model_medium, "medium", wav_path)

if __name__ == "__main__":
    main()
