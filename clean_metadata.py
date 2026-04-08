"""
Clean metadata.csv for XTTS training by:
1. Removing <noise>, <name>, <unclear> and other angle-bracket tokens
2. Removing lines where transcription is too short after cleaning
3. Removing lines where transcription is too long (>200 chars)
4. Stripping extra whitespace
"""

import re
import os

INPUT_CSV  = r"C:\Users\jackc\SpeechDiarization\datasets\xtts_dataset\metadata.csv"
OUTPUT_CSV = r"C:\Users\jackc\SpeechDiarization\datasets\xtts_dataset\metadata.cleaned.csv"

MIN_LEN = 3    # minimum characters after cleaning
MAX_LEN = 200  # maximum characters to avoid range errors

def clean_text(text):
    # Remove anything in angle brackets like <noise>, <name>, <unclear>
    text = re.sub(r"<[^>]+>", "", text)
    # Remove \n artifacts
    text = text.replace("\\n", " ")
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text

def main():
    kept = 0
    skipped_short = 0
    skipped_long = 0

    with open(INPUT_CSV, "r", encoding="utf-8") as fin, \
         open(OUTPUT_CSV, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            parts = line.split("|")
            if len(parts) < 2:
                skipped_short += 1
                continue

            filename = parts[0]
            # Clean both text columns
            text = clean_text(parts[1])
            
            if len(text) < MIN_LEN:
                skipped_short += 1
                continue

            if len(text) > MAX_LEN:
                skipped_long += 1
                continue

            fout.write(f"{filename}|{text}|{text}\n")
            kept += 1

    print(f"Done.")
    print(f"  Kept:          {kept} entries")
    print(f"  Skipped short: {skipped_short} entries")
    print(f"  Skipped long:  {skipped_long} entries")
    print(f"  Output: {OUTPUT_CSV}")
    print(f"\nUpdate train_xtts.py:")
    print(f'  meta_file_train="metadata.cleaned.csv"')

if __name__ == "__main__":
    main()