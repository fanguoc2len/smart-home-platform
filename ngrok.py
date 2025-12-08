import torch, whisper

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

print(">>> Loading SMALL ...")
whisper.load_model("small", device=device)
print("OK small")

print(">>> Loading MEDIUM ...")
whisper.load_model("medium", device=device)
print("OK medium")
