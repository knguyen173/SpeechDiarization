# Speech Diarization & XTTS Finetuning Pipeline

A pipeline for extracting mother speech from annotated video recordings and finetuning a Coqui XTTS-v2 text-to-speech model on a target speaker's voice.

---

## Overview

This project was built to:
1. Extract annotated mother speech segments from EAF/MP4 file pairs
2. Concatenate short segments into training-length clips
3. Build a metadata CSV for XTTS finetuning
4. Finetune a Coqui XTTS-v2 model on the extracted speaker audio
5. Run inference with the finetuned model

---

## Dataset

This pipeline was developed using the **UCL Child-Directed Speech (UCL-CDS)** dataset, which contains video recordings of mother-child interactions annotated using ELAN.

You will need:
- A folder of `.mp4` video files (one per session)
- A matching folder of `.eaf` annotation files (ELAN format)
- The EAF files must contain a **"Speaker"** tier with timestamped transcriptions of the mother's speech

The MP4 and EAF files are matched by their first 4 characters (e.g. `ch07_session.mp4` pairs with `ch07_final.eaf`).

> **Note:** This dataset is not publicly available. To replicate this work you will need access to a similarly structured annotated speech corpus.

---

## Requirements

- Python 3.11
- NVIDIA GPU with 8GB+ VRAM (tested on RTX 3060 Ti)
- CUDA 12.1+
- ffmpeg (must be on your system PATH)

### Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

---

## Pipeline

### Step 1: Extract speaker segments from EAF/MP4 pairs

Place your `.mp4` and `.eaf` files together in a folder called `sample_mp4_eaf/`, then run:

```bash
python extract_speaker_audio.py
```

**Output:**
- `segments_from_eaf/<speaker>/` — individual WAV clips per utterance
- `segments_from_eaf/speaker_segments.tsv` — maps each WAV to its transcription

> The script handles `TIME_ORIGIN` offsets in EAF files, which is important when the annotation does not start from the beginning of the video.

---

### Step 2: Concatenate short segments into training clips

```bash
python concatenate_waves.py
```

This joins short utterances into 5–12 second clips with 300ms silence between them, which is the ideal length for XTTS training.

**Output:**
- `concatenated_waves/<speaker>/concat_XXXXX.wav`
- `concatenated_waves/<speaker>/file_mapping.tsv`

---

### Step 3: Build metadata CSV

```bash
python build_metadata.py
```

This joins the transcriptions from `speaker_segments.tsv` with the `file_mapping.tsv` files to produce a dataset in LJSpeech format.

**Output:**
- `datasets/xtts_dataset/wavs/` — all concatenated WAVs copied here
- `datasets/xtts_dataset/metadata.csv`

---

### Step 4: Clean metadata

```bash
python clean_metadata.py
```

Removes unknown tokens (`<noise>`, `<unclear>`, `<name>` etc.) from transcriptions and filters out entries that are too short or too long.

**Output:**
- `datasets/xtts_dataset/metadata.cleaned.csv`

---

### Step 5: Finetune XTTS-v2

Edit the config section at the top of `train_xtts.py` to set your paths, then run:

```bash
cd finetune/xtts
python train_xtts.py
```

Key config options in `train_xtts.py`:
```python
DATASET_DIR      = r"path\to\datasets\xtts_dataset"
OUT_DIR          = r"path\to\finetune\xtts\runs"
RUN_NAME         = "xtts_finetune_run1"
SPEAKER_REFERENCE = r"path\to\a\reference\wav"
GRAD_ACCUM_STEPS = 16   # reduce if you get OOM errors
```

The base XTTS-v2 model files are downloaded automatically on first run.

To **resume from a checkpoint**, set `restore_path` in `TrainerArgs`:
```python
TrainerArgs(
    restore_path=r"path\to\checkpoint_XXXX.pth",
    ...
)
```

---

### Step 6: Test inference

Copy your checkpoint and rename it to `model.pth`, copy `vocab.json` into the run folder, then run:

```bash
cd finetune/xtts
python test_inference.py
```

Edit `CHECKPOINT_DIR` and `SPEAKER_WAV` in `test_inference.py` to match your run. Output is saved to `test_output.wav`.

---

## File Reference

| File | Description |
|------|-------------|
| `extract_speaker_audio.py` | Extracts utterance WAVs from MP4/EAF pairs |
| `concatenate_waves.py` | Concatenates short clips into training-length WAVs |
| `build_metadata.py` | Builds LJSpeech metadata.csv from TSV files |
| `clean_metadata.py` | Cleans unknown tokens from metadata |
| `train_xtts.py` | Finetunes XTTS-v2 on the dataset |
| `test_inference.py` | Runs inference with a finetuned checkpoint |
| `transcribe_whisper.py` | Transcribes audio using Whisper |
| `diarize.py` | Speaker diarization utilities |
| `chec_wavs.py` | Checks WAV file properties |
| `gpu_test.py` | Verifies CUDA/GPU availability |

---

## Notes

- Training was done on Windows with PyTorch 2.4.1+cu121
- The `dvae_sample_rate` attribute is patched manually for Coqui TTS version compatibility
- Set `num_loader_workers=0` on Windows to avoid multiprocessing issues
- Monitor `loss_text_ce` as the primary training metric — target below `0.01`
- Listen to generated audio samples (saved every `plot_step` steps) to judge when to stop training
