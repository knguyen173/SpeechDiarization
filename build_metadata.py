"""
Build a metadata.csv (LJSpeech format) for XTTS finetuning from:
  - segments_from_eaf/speaker_segments.tsv  (segment -> transcription)
  - concatenated_waves/<speaker>/file_mapping.tsv (concat file -> source segments)

Output:
  xtts_dataset/metadata.csv   -- LJSpeech format: filename|transcription
  xtts_dataset/wavs/          -- symlinks or copies of the concat wavs

LJSpeech format expected by train_xtts.py:
  filename_without_extension|transcription text
"""

import os
import csv
import shutil
import glob

SEGMENTS_TSV   = r"C:\Users\jackc\SpeechDiarization\segments_from_eaf\speaker_segments.tsv"
CONCAT_DIR     = r"C:\Users\jackc\SpeechDiarization\concatenated_waves"
OUTPUT_DIR     = r"C:\Users\jackc\SpeechDiarization\datasets\xtts_dataset"
WAVS_DIR       = os.path.join(OUTPUT_DIR, "wavs")

# Minimum transcription length to include (skip very short/empty ones)
MIN_TEXT_LEN = 2


def load_segment_transcriptions(tsv_path):
    """
    Load speaker_segments.tsv and return a dict:
      { segment_wav_basename -> transcription_text }
    """
    transcriptions = {}
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            wav_file = row.get("wav_file", "").strip()
            text = row.get("text", "").strip()
            if wav_file and text:
                basename = os.path.basename(wav_file)
                transcriptions[basename] = text
    print(f"Loaded {len(transcriptions)} transcriptions from speaker_segments.tsv")
    return transcriptions


def build_concat_transcription(source_files_str, transcriptions):
    """
    Given a comma-separated list of source segment filenames,
    join their transcriptions in order to form the concat clip transcription.
    Skips segments with no transcription.
    """
    parts = []
    for fname in source_files_str.split(","):
        fname = fname.strip()
        text = transcriptions.get(fname, "").strip()
        if text:
            parts.append(text)
    return " ".join(parts)


def main():
    os.makedirs(WAVS_DIR, exist_ok=True)

    # Load all transcriptions from the segments TSV
    transcriptions = load_segment_transcriptions(SEGMENTS_TSV)

    metadata_rows = []
    total_skipped = 0
    total_copied = 0

    # Process each speaker subfolder
    speaker_dirs = sorted(
        d for d in os.listdir(CONCAT_DIR)
        if os.path.isdir(os.path.join(CONCAT_DIR, d))
    )
    print(f"Found {len(speaker_dirs)} speaker folders: {speaker_dirs}")

    for speaker in speaker_dirs:
        speaker_dir = os.path.join(CONCAT_DIR, speaker)
        mapping_tsv = os.path.join(speaker_dir, "file_mapping.tsv")

        if not os.path.isfile(mapping_tsv):
            print(f"  [{speaker}] No file_mapping.tsv found, skipping.")
            continue

        with open(mapping_tsv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                out_file = row.get("output_file", "").strip()
                source_files = row.get("source_files", "").strip()

                if not out_file or not source_files:
                    continue

                # Build transcription by joining source segment texts
                text = build_concat_transcription(source_files, transcriptions)

                if len(text) < MIN_TEXT_LEN:
                    total_skipped += 1
                    continue

                # Clean text for LJSpeech format - remove pipe characters
                text = text.replace("|", " ").strip()

                # Copy wav to xtts_dataset/wavs/
                src_wav = os.path.join(speaker_dir, out_file)
                # Prefix with speaker to avoid name collisions across speakers
                dest_name = f"{speaker}_{out_file}"
                dest_wav = os.path.join(WAVS_DIR, dest_name)

                if os.path.isfile(src_wav):
                    shutil.copy2(src_wav, dest_wav)
                    total_copied += 1
                    # LJSpeech format: filename without extension
                    stem = dest_name.replace(".wav", "")
                    metadata_rows.append(f"{stem}|{text}|{text}")
                else:
                    print(f"  WARNING: wav not found: {src_wav}")
                    total_skipped += 1

    # Write metadata.csv
    metadata_path = os.path.join(OUTPUT_DIR, "metadata.csv")
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write("\n".join(metadata_rows))

    print(f"\nDone.")
    print(f"  Copied {total_copied} wav files to {WAVS_DIR}")
    print(f"  Skipped {total_skipped} entries (no transcription or missing wav)")
    print(f"  Written metadata.csv with {len(metadata_rows)} entries to {metadata_path}")
    print(f"\nNext steps:")
    print(f"  1. Update DATASET_DIR in train_xtts.py to: {OUTPUT_DIR}")
    print(f"  2. Update meta_file_train to: metadata.csv")
    print(f"  3. Run train_xtts.py")


if __name__ == "__main__":
    main()