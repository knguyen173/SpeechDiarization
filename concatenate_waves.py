"""
Concatenate wav files from segments_from_eaf/<speaker>/ into 5–12 second clips
with 300ms silence in between.  Each subfolder (ch07, ch08, …) is treated as a
separate speaker — files from different subfolders are never mixed.

Output structure:
    concatenated_waves/<speaker>/concat_00001.wav …
    concatenated_waves/<speaker>/file_mapping.tsv
"""

import os
import wave
import glob

INPUT_DIR = "segments_from_eaf"
OUTPUT_DIR = "concatenated_waves"
SILENCE_MS = 300
MIN_DURATION = 5.0
MAX_DURATION = 12.0


def get_wav_info(path):
    """Return (duration_seconds, nchannels, sampwidth, framerate, frames_bytes)."""
    with wave.open(path, "r") as w:
        n = w.getnframes()
        rate = w.getframerate()
        return n / rate, w.getnchannels(), w.getsampwidth(), rate, w.readframes(n)


def make_silence(duration_s, nchannels, sampwidth, framerate):
    """Create silence as raw bytes."""
    n_samples = int(duration_s * framerate) * nchannels
    return b"\x00" * (n_samples * sampwidth)


def write_wav(path, frames_bytes, nchannels, sampwidth, framerate):
    with wave.open(path, "w") as w:
        w.setnchannels(nchannels)
        w.setsampwidth(sampwidth)
        w.setframerate(framerate)
        w.writeframes(frames_bytes)


def process_speaker(speaker_dir, speaker_out_dir):
    """Concatenate all wav files in one speaker subfolder."""
    speaker = os.path.basename(speaker_dir)
    wav_files = sorted(glob.glob(os.path.join(speaker_dir, "*.wav")))
    print(f"\n[{speaker}] Found {len(wav_files)} wav files")

    if not wav_files:
        return 0, 0

    os.makedirs(speaker_out_dir, exist_ok=True)

    _, nchannels, sampwidth, framerate, _ = get_wav_info(wav_files[0])
    silence_bytes = make_silence(SILENCE_MS / 1000.0, nchannels, sampwidth, framerate)
    silence_dur = SILENCE_MS / 1000.0

    manifest_path = os.path.join(speaker_out_dir, "file_mapping.tsv")
    manifest = open(manifest_path, "w")
    manifest.write("output_file\tduration_s\tnum_sources\tsource_files\n")

    group = []
    group_dur = 0.0
    group_sources = []
    output_idx = 1
    skipped = 0

    for fpath in wav_files:
        dur, nc, sw, fr, frames = get_wav_info(fpath)

        if (nc, sw, fr) != (nchannels, sampwidth, framerate):
            print(f"  Skipping {fpath} (format mismatch: {nc}ch/{sw}B/{fr}Hz)")
            skipped += 1
            continue

        if group:
            candidate_dur = group_dur + silence_dur + dur
        else:
            candidate_dur = dur

        if candidate_dur <= MAX_DURATION:
            if group:
                group.append(silence_bytes)
                group_dur += silence_dur
            group.append(frames)
            group_dur += dur
            group_sources.append(os.path.basename(fpath))

        else:
            if group and group_dur >= MIN_DURATION:
                out_name = f"concat_{output_idx:05d}.wav"
                out_path = os.path.join(speaker_out_dir, out_name)
                write_wav(out_path, b"".join(group), nchannels, sampwidth, framerate)
                print(
                    f"  {out_name}: {group_dur:.2f}s  ({len(group_sources)} files: {group_sources[0]}..{group_sources[-1]})"
                )
                manifest.write(
                    f"{out_name}\t{group_dur:.2f}\t{len(group_sources)}\t{','.join(group_sources)}\n"
                )
                output_idx += 1
                group, group_dur, group_sources = [], 0.0, []

            if group:
                group.append(silence_bytes)
                group_dur += silence_dur
                group.append(frames)
                group_dur += dur
                group_sources.append(os.path.basename(fpath))
            else:
                group.append(frames)
                group_dur = dur
                group_sources.append(os.path.basename(fpath))

    # Flush remaining group
    if group:
        out_name = f"concat_{output_idx:05d}.wav"
        out_path = os.path.join(speaker_out_dir, out_name)
        write_wav(out_path, b"".join(group), nchannels, sampwidth, framerate)
        print(
            f"  {out_name}: {group_dur:.2f}s  ({len(group_sources)} files: {group_sources[0]}..{group_sources[-1]})"
        )
        manifest.write(
            f"{out_name}\t{group_dur:.2f}\t{len(group_sources)}\t{','.join(group_sources)}\n"
        )
        output_idx += 1

    manifest.close()
    return output_idx - 1, skipped


def main():
    subfolders = sorted(
        d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))
    )
    print(f"Found {len(subfolders)} speaker subfolders in {INPUT_DIR}/")

    total_files = 0
    total_skipped = 0
    for speaker in subfolders:
        speaker_dir = os.path.join(INPUT_DIR, speaker)
        speaker_out_dir = os.path.join(OUTPUT_DIR, speaker)
        written, skipped = process_speaker(speaker_dir, speaker_out_dir)
        total_files += written
        total_skipped += skipped

    print(
        f"\nDone. Wrote {total_files} concatenated files total across {len(subfolders)} speakers to {OUTPUT_DIR}/"
    )
    if total_skipped:
        print(f"Skipped {total_skipped} files due to format mismatch.")


if __name__ == "__main__":
    main()