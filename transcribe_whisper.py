import os
import csv
import argparse
from pathlib import Path

import whisper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to dataset folder containing wavs/ (e.g. datasets/my_xtts_dataset)")
    parser.add_argument("--model", type=str, default="medium",
                        help="Whisper model: tiny, base, small, medium, large")
    parser.add_argument("--language", type=str, default="en",
                        help="Language code (e.g. en). Use empty string to let Whisper auto-detect.")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device: 'cuda' or 'cpu'. Default: auto.")
    parser.add_argument("--beam_size", type=int, default=5)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    wav_dir = dataset_dir / "wavs"
    out_csv = dataset_dir / "metadata.csv"

    if not wav_dir.exists():
        raise FileNotFoundError(f"Missing wavs/ directory: {wav_dir}")

    wavs = sorted([p for p in wav_dir.iterdir() if p.suffix.lower() == ".wav"])
    if not wavs:
        raise RuntimeError(f"No .wav files found in: {wav_dir}")

    # Decide device
    device = args.device
    if device is None:
        # Whisper will pick GPU if torch sees it; otherwise CPU.
        device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "-1" else "cuda"

    print(f"Loading Whisper model '{args.model}'...")
    model = whisper.load_model(args.model, device=device)

    print(f"Found {len(wavs)} wav files.")
    print(f"Writing metadata to: {out_csv}")

    # LJSpeech-style metadata: id|text|text
    # id should be filename without extension (e.g. "audio1" for audio1.wav)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="|", lineterminator="\n")

        for i, wav_path in enumerate(wavs, start=1):
            file_id = wav_path.stem

            print(f"[{i}/{len(wavs)}] Transcribing: {wav_path.name}")

            # Run transcription
            # fp16=True is fine on GPU; on CPU Whisper will ignore fp16.
            result = model.transcribe(
                str(wav_path),
                language=(args.language if args.language else None),
                beam_size=args.beam_size,
                fp16=True,
                verbose=False,
            )

            text = (result.get("text") or "").strip()

            # Basic cleanup
            text = " ".join(text.split())
            if not text:
                # If Whisper outputs nothing, still write a placeholder so you can find/fix it.
                text = "[EMPTY_TRANSCRIPT]"

            # Third column can be same as second for XTTS fine-tuning
            writer.writerow([file_id, text, text])

    print("\nDone.")
    print("Next step: open metadata.csv and spot-check transcripts for obvious errors.")
    print("If you see [EMPTY_TRANSCRIPT], those files need manual review.")


if __name__ == "__main__":
    main()
