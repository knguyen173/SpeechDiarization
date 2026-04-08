"""
Extract audio segments from mp4 files based on the "Speaker" tier in matching EAF files.

For each mp4/eaf pair in sample_mp4_eaf/ that share the same speaker prefix (first 4 chars,
e.g. ch07), parse the EAF "Speaker" tier annotations, resolve time-slot references to
millisecond values, and extract each annotated segment as a separate WAV file.

Strategy (two-step for efficiency):
  1. Convert the full MP4 audio to a single WAV file once using ffmpeg.
  2. Slice individual segments from the WAV using Python's wave module
     (instant byte-offset reads  no subprocess per segment).

Output structure:
    segments_from_eaf/<prefix>/<prefix>_speaker_<annotation_id>_<start_ms>_<end_ms>.wav
"""

import os
import re
import subprocess
import wave
import xml.etree.ElementTree as ET

INPUT_DIR = "sample_mp4_eaf"
OUTPUT_DIR = "segments_from_eaf"
FULL_WAV_DIR = "converted_wavs"


def find_matched_pairs(input_dir):
    """Find mp4/eaf pairs that share the same 4-character speaker prefix."""
    files = os.listdir(input_dir)
    mp4s = {}
    eafs = {}

    for f in files:
        prefix = f[:4]
        if f.lower().endswith(".mp4"):
            mp4s[prefix] = os.path.join(input_dir, f)
        elif f.lower().endswith(".eaf"):
            eafs[prefix] = os.path.join(input_dir, f)

    pairs = {}
    for prefix in sorted(set(mp4s) & set(eafs)):
        pairs[prefix] = (mp4s[prefix], eafs[prefix])

    return pairs


def parse_eaf_speaker_tier(eaf_path):
    """
    Parse an EAF file and return a list of (start_ms, end_ms, annotation_value, ann_id)
    tuples from the "Speaker" tier.

    Applies the TIME_ORIGIN offset from the MEDIA_DESCRIPTOR so that timestamps
    are absolute positions in the original full media file.
    """
    tree = ET.parse(eaf_path)
    root = tree.getroot()

    # Get TIME_ORIGIN offset from the media descriptor (default 0)
    time_origin = 0
    for md in root.iter("MEDIA_DESCRIPTOR"):
        to = md.get("TIME_ORIGIN")
        if to is not None:
            time_origin = int(to)
            break
    if time_origin:
        print(f"  TIME_ORIGIN offset: {time_origin}ms ({time_origin / 1000:.1f}s)")

    # Build time-slot lookup: ts_id -> milliseconds
    time_slots = {}
    for ts in root.iter("TIME_SLOT"):
        ts_id = ts.get("TIME_SLOT_ID")
        ts_val = ts.get("TIME_VALUE")
        if ts_id and ts_val is not None:
            time_slots[ts_id] = int(ts_val)

    # Find the "Speaker" tier
    speaker_tier = None
    for tier in root.iter("TIER"):
        if tier.get("TIER_ID") == "Speaker":
            speaker_tier = tier
            break

    if speaker_tier is None:
        print(f"  WARNING: No 'Speaker' tier found in {eaf_path}")
        return []

    segments = []
    for annotation in speaker_tier.iter("ANNOTATION"):
        aa = annotation.find("ALIGNABLE_ANNOTATION")
        if aa is None:
            continue
        ann_id = aa.get("ANNOTATION_ID", "unknown")
        ts1 = aa.get("TIME_SLOT_REF1")
        ts2 = aa.get("TIME_SLOT_REF2")
        value = aa.findtext("ANNOTATION_VALUE", default="")

        start_ms = time_slots.get(ts1)
        end_ms = time_slots.get(ts2)

        if start_ms is None or end_ms is None:
            print(f"  WARNING: Could not resolve time slots for annotation {ann_id}")
            continue

        # Apply TIME_ORIGIN offset to get absolute position in the full media
        segments.append((start_ms + time_origin, end_ms + time_origin, value, ann_id))

    return segments


def convert_mp4_to_wav(mp4_path, wav_path, sample_rate=16000):
    """Convert the full MP4 audio track to a single WAV file (mono, 16-bit PCM)."""
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "quiet",
        "-i",
        mp4_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        wav_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {result.stderr[:300]}")
    return wav_path


def extract_segments_from_wav(full_wav_path, segments, out_dir, prefix):
    """
    Slice segments from a full WAV file using Python's wave module.
    Each segment is written as a separate WAV.  This is much faster than
    spawning an ffmpeg process per segment because seeking in uncompressed
    PCM is just a byte-offset calculation.

    Returns the number of successfully extracted segments.
    """
    with wave.open(full_wav_path, "r") as src:
        n_channels = src.getnchannels()
        sampwidth = src.getsampwidth()
        framerate = src.getframerate()
        total_frames = src.getnframes()
        # Read entire audio into memory (mono 16-bit @ 16kHz H 32 KB/s)
        all_frames = src.readframes(total_frames)

    bytes_per_frame = n_channels * sampwidth
    extracted = 0

    for i, (start_ms, end_ms, value, ann_id) in enumerate(segments):
        duration_ms = end_ms - start_ms
        if duration_ms <= 0:
            print(f"  Skipping zero/negative duration: {start_ms}-{end_ms}ms")
            continue

        start_frame = int(start_ms * framerate / 1000)
        end_frame = int(end_ms * framerate / 1000)
        # Clamp to file length
        start_frame = max(0, min(start_frame, total_frames))
        end_frame = max(0, min(end_frame, total_frames))

        start_byte = start_frame * bytes_per_frame
        end_byte = end_frame * bytes_per_frame
        segment_data = all_frames[start_byte:end_byte]

        if len(segment_data) == 0:
            print(f"  Skipping empty segment {ann_id} ({start_ms}-{end_ms}ms)")
            continue

        wav_name = f"{prefix}_speaker_{ann_id}_{start_ms}_{end_ms}.wav"
        wav_path = os.path.join(out_dir, wav_name)

        with wave.open(wav_path, "w") as dst:
            dst.setnchannels(n_channels)
            dst.setsampwidth(sampwidth)
            dst.setframerate(framerate)
            dst.writeframes(segment_data)

        extracted += 1

        if (i + 1) % 100 == 0:
            print(f"  Sliced {i + 1}/{len(segments)} segments...")

    return extracted


def main():
    pairs = find_matched_pairs(INPUT_DIR)
    if not pairs:
        print(f"No matched mp4/eaf pairs found in {INPUT_DIR}/")
        return

    print(f"Found {len(pairs)} matched pair(s): {list(pairs.keys())}")

    total_extracted = 0

    for prefix, (mp4_path, eaf_path) in pairs.items():
        print(f"\nProcessing {prefix}:")
        print(f"  MP4: {mp4_path}")
        print(f"  EAF: {eaf_path}")

        segments = parse_eaf_speaker_tier(eaf_path)
        print(f"  Found {len(segments)} Speaker annotations")

        if not segments:
            continue

        out_dir = os.path.join(OUTPUT_DIR, prefix)
        os.makedirs(out_dir, exist_ok=True)

        # Step 1: Convert full MP4 � WAV once and save it
        os.makedirs(FULL_WAV_DIR, exist_ok=True)
        full_wav = os.path.join(FULL_WAV_DIR, f"{prefix}.wav")
        print(f"  Converting MP4 � WAV: {full_wav}")
        convert_mp4_to_wav(mp4_path, full_wav)
        print(f"  Conversion done. Full WAV saved to {full_wav}")

        # Step 2: Slice segments from the WAV (pure Python, no subprocess per segment)
        count = extract_segments_from_wav(full_wav, segments, out_dir, prefix)
        total_extracted += count

        print(f"  Done: {count}/{len(segments)} segments saved to {out_dir}/")

    # Write a metadata TSV
    print(f"\nTotal segments extracted: {total_extracted}")
    write_metadata(pairs)


def write_metadata(pairs):
    """Write a TSV file mapping each extracted WAV to its annotation info."""
    tsv_path = os.path.join(OUTPUT_DIR, "speaker_segments.tsv")
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("prefix\tannotation_id\tstart_ms\tend_ms\ttext\twav_file\n")
        for prefix, (mp4_path, eaf_path) in pairs.items():
            segments = parse_eaf_speaker_tier(eaf_path)
            out_dir = os.path.join(OUTPUT_DIR, prefix)
            for start_ms, end_ms, value, ann_id in segments:
                wav_name = f"{prefix}_speaker_{ann_id}_{start_ms}_{end_ms}.wav"
                wav_path = os.path.join(out_dir, wav_name)
                if os.path.exists(wav_path):
                    f.write(
                        f"{prefix}\t{ann_id}\t{start_ms}\t{end_ms}\t{value}\t{wav_path}\n"
                    )
    print(f"Metadata written to {tsv_path}")


if __name__ == "__main__":
    main()
