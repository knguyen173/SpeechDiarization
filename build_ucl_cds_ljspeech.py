import os
import re
import csv
import json
import shutil
import subprocess
from pathlib import Path

import whisper
import soundfile as sf

# --------- USER SETTINGS ----------
SRC_DIR = Path(r"datasets\UCL_CDS")              # folder with .mp4
OUT_DIR = Path(r"datasets\UCL_CDS_XTTS")         # new dataset folder
LANG = "en"

WHISPER_MODEL = "medium"  # good accuracy; use "small" if slow
MIN_SEC = 1.0
MAX_SEC = 15.0
MAX_CHARS = 200

# If True: clips follow Whisper segments (recommended)
# If False: one wav per mp4 (not recommended for XTTS)
SEGMENT_CLIPS = True
# ----------------------------------

def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=False)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed:\n{cmd}\n\nSTDERR:\n{p.stderr}")
    return p.stdout

def safe_name(s: str) -> str:
    s = re.sub(r"[^\w\-]+", "_", s)
    return s.strip("_")

def extract_wav(mp4_path: Path, wav_path: Path, sr=22050):
    # mono, 16-bit PCM, fixed sample rate
    cmd = [
        "ffmpeg", "-y",
        "-i", str(mp4_path),
        "-vn",
        "-ac", "1",
        "-ar", str(sr),
        "-sample_fmt", "s16",
        str(wav_path)
    ]
    run(cmd)

def cut_wav(in_wav: Path, out_wav: Path, start: float, end: float, sr=22050):
    # fast cut using ffmpeg copy-less
    cmd = [
        "ffmpeg", "-y",
        "-i", str(in_wav),
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-ac", "1",
        "-ar", str(sr),
        "-sample_fmt", "s16",
        str(out_wav)
    ]
    run(cmd)

def dur_seconds(wav: Path) -> float:
    info = sf.info(str(wav))
    return float(info.frames) / float(info.samplerate)

def main():
    if not SRC_DIR.exists():
        raise SystemExit(f"SRC_DIR not found: {SRC_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wavs_dir = OUT_DIR / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    mp4s = sorted(SRC_DIR.glob("*.mp4"))
    if not mp4s:
        raise SystemExit(f"No .mp4 files found in {SRC_DIR}")

    print(f"Found {len(mp4s)} mp4 files")

    model = whisper.load_model(WHISPER_MODEL)

    meta_path = OUT_DIR / "metadata.csv"
    rows_written = 0
    skipped = 0

    with open(meta_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="|")

        for idx, mp4 in enumerate(mp4s, start=1):
            base = safe_name(mp4.stem)
            full_wav = wavs_dir / f"{base}_FULL.wav"

            print(f"\n[{idx}/{len(mp4s)}] Extracting audio: {mp4.name}")
            extract_wav(mp4, full_wav, sr=22050)

            # Transcribe (with segments)
            print(f"Transcribing with Whisper ({WHISPER_MODEL})...")
            result = model.transcribe(str(full_wav), language=LANG, fp16=False, verbose=False)

            segments = result.get("segments", [])
            if not segments:
                print("  -> No segments returned; skipping file.")
                skipped += 1
                continue

            if not SEGMENT_CLIPS:
                # One line for whole file (not recommended)
                text = (result.get("text") or "").strip()
                if not text or len(text) > MAX_CHARS or dur_seconds(full_wav) > MAX_SEC:
                    print("  -> Whole-file sample is too long or empty; skipping.")
                    skipped += 1
                    continue
                w.writerow([full_wav.name, text, text])
                rows_written += 1
                continue

            # Segment-based clips
            clip_num = 0
            for seg in segments:
                t0 = float(seg["start"])
                t1 = float(seg["end"])
                text = (seg.get("text") or "").strip()

                if not text:
                    continue
                if len(text) > MAX_CHARS:
                    continue
                if (t1 - t0) < MIN_SEC or (t1 - t0) > MAX_SEC:
                    continue

                clip_num += 1
                out_name = f"{base}_{clip_num:06d}.wav"
                out_wav = wavs_dir / out_name

                cut_wav(full_wav, out_wav, t0, t1, sr=22050)

                # Re-check duration after cut
                d = dur_seconds(out_wav)
                if d < MIN_SEC or d > MAX_SEC:
                    out_wav.unlink(missing_ok=True)
                    continue

                w.writerow([out_name, text, text])
                rows_written += 1

            # optional: delete full wav to save space
            full_wav.unlink(missing_ok=True)

    print("\nDone.")
    print(f"metadata.csv written to: {meta_path}")
    print(f"Total rows written: {rows_written}")
    print(f"Files skipped: {skipped}")
    print(f"Wavs directory: {wavs_dir}")

if __name__ == "__main__":
    main()
S