import os
import csv
import soundfile as sf

ROOT = r"C:\Users\jackc\SpeechDiarization\datasets\UCL_CDS_XTTS"
META = os.path.join(ROOT, "metadata.csv")
WAVDIR = os.path.join(ROOT, "wavs")

missing = []
unreadable = []
zero = []

with open(META, encoding="utf-8") as f:
    r = csv.reader(f, delimiter="|")
    for i, row in enumerate(r, 1):
        if len(row) < 2:
            continue

        fn = row[0].strip()
        if not fn.endswith(".wav"):
            fn += ".wav"

        path = os.path.join(WAVDIR, fn)

        if not os.path.isfile(path):
            missing.append((i, fn))
            continue

        try:
            info = sf.info(path)
            if info.frames == 0:
                zero.append((i, fn))
        except Exception as e:
            unreadable.append((i, fn, str(e)))

print("Missing:", len(missing))
print("Unreadable:", len(unreadable))
print("Zero-length:", len(zero))
print("First 5 unreadable:", unreadable[:5])
