import numpy as np
import soundfile as sf
import torch

# -------------------------------------------------------------------
# Monkey-patch XTTS load_audio() to avoid torchaudio/torchcodec on Windows.
# IMPORTANT: XTTS expects load_audio() to return ONLY the audio tensor,
# shaped [channels, samples]. Not (audio, sr).
# -------------------------------------------------------------------
import TTS.tts.models.xtts as xtts_module

def load_audio_soundfile(audiopath: str, load_sr: int = None):
    audio, sr = sf.read(audiopath, always_2d=False)

    # stereo -> mono
    if hasattr(audio, "ndim") and audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio = audio.astype(np.float32)
    audio = np.clip(audio, -1.0, 1.0)

    # resample if XTTS requests a specific sample rate
    if load_sr is not None and sr != load_sr:
        x_old = np.linspace(0, 1, num=len(audio), endpoint=False)
        new_len = int(len(audio) * (load_sr / sr))
        x_new = np.linspace(0, 1, num=new_len, endpoint=False)
        audio = np.interp(x_new, x_old, audio).astype(np.float32)
        sr = load_sr

    # XTTS expects [channels, samples]
    audio_t = torch.from_numpy(audio).float().unsqueeze(0)  # [1, N]

    # RETURN ONLY AUDIO (NOT sr)
    return audio_t

# Apply the patch
xtts_module.load_audio = load_audio_soundfile

# -------------------------------------------------------------------
# XTTS v2 inference
# -------------------------------------------------------------------
from TTS.api import TTS

def main():
    # CPU mode (works everywhere). If you later set up CUDA, use .to("cuda")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

    tts.tts_to_file(
        text="Hello my name is Katherine. I went to Monta Vista, and I graduated from SMU with a BS in CS. Then now, I am a current grad student at Santa Clara University.",
        file_path="output.wav",
        speaker_wav="speaker.wav",
        language="en",
    )

    print("Done. Wrote output.wav")

if __name__ == "__main__":
    main()
