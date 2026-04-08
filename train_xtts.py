import os
os.environ["TTS_DISABLE_TORCHCODEC"] = "1"   # force TTS to avoid torchcodec loader on Windows
os.environ["TORCHAUDIO_USE_SOUNDFILE_LEGACY"] = "1"  # avoid ffmpeg issues on Windows with torchaudio
import sys

import torch
from torch.serialization import add_safe_globals

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.models.xtts import XttsArgs

# PyTorch 2.6+ safe loading allowlist for XTTS checkpoints
add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig])

# ---- user config ----
DATASET_DIR = r"C:\Users\jackc\SpeechDiarization\datasets\xtts_dataset"
OUT_DIR     = r"C:\Users\jackc\SpeechDiarization\finetune\xtts\runs"
RUN_NAME    = "xtts_finetune_run7"
LANGUAGE    = "en"

# 3060 Ti (8GB) safe defaults
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16

# Pick one good wav from dataset for checkpoint audio samples
SPEAKER_REFERENCE = os.path.join(DATASET_DIR, "wavs", "ch07_concat_00004.wav")

# ---------------------

def fail(msg: str):
    print("\nERROR:", msg)
    sys.exit(1)

def main():
    # Basic structure checks
    meta_path = os.path.join(DATASET_DIR, "metadata.csv")
    wav_dir   = os.path.join(DATASET_DIR, "wavs")

    if not os.path.isdir(DATASET_DIR):
        fail(f"DATASET_DIR not found: {DATASET_DIR}")
    if not os.path.isfile(meta_path):
        fail(f"metadata.csv not found at: {meta_path}")
    if not os.path.isdir(wav_dir):
        fail(f"wavs folder not found at: {wav_dir}")
    if not os.path.isfile(SPEAKER_REFERENCE):
        fail(
            f"SPEAKER_REFERENCE not found:\n  {SPEAKER_REFERENCE}\n"
            f"Fix it by pointing to an existing wav inside:\n  {wav_dir}"
        )

    # Torch + CUDA check
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
    else:
        print("WARNING: CUDA is not available. Training will run on CPU (very slow).")

    # Coqui XTTS training imports
    from trainer import Trainer, TrainerArgs
    from TTS.tts.configs.shared_configs import BaseDatasetConfig
    from TTS.tts.datasets import load_tts_samples
    from TTS.tts.layers.xtts.trainer.gpt_trainer import (
        GPTArgs,
        GPTTrainer,
        GPTTrainerConfig,
    )
    from TTS.utils.manage import ModelManager

    os.makedirs(OUT_DIR, exist_ok=True)

    # Download base XTTS model files (once)
    base_files_dir = os.path.join(OUT_DIR, "XTTS_base_files")
    os.makedirs(base_files_dir, exist_ok=True)

    DVAE_LINK      = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
    MEL_STATS_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"
    VOCAB_LINK     = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
    MODEL_LINK     = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"

    def ensure_download(url: str) -> str:
        path = os.path.join(base_files_dir, os.path.basename(url))
        return path

    dvae_ckpt   = ensure_download(DVAE_LINK)
    mel_stats   = ensure_download(MEL_STATS_LINK)
    vocab_json  = ensure_download(VOCAB_LINK)
    model_pth   = ensure_download(MODEL_LINK)

    need = [dvae_ckpt, mel_stats, vocab_json, model_pth]
    if not all(os.path.isfile(p) for p in need):
        print("Downloading base XTTS-v2 files...")
        ModelManager._download_model_files([MEL_STATS_LINK, DVAE_LINK, VOCAB_LINK, MODEL_LINK], base_files_dir, progress_bar=True)
    else:
        print("Base XTTS-v2 files already present.")

    # Audio config
    # XTTS expects training audio typically 22050 or 16000; it will work as long as your dataset is consistent.
    # If your wavs are 22050 Hz, set both to 22050.
    # If your wavs are 16000 Hz, set both to 16000.
    #
    # If you don't know yet, leave at 22050 first, and if you hit obvious audio/sample-rate issues, adjust.
    audio_config = XttsAudioConfig(sample_rate=16000, output_sample_rate=24000)
    audio_config.dvae_sample_rate = 16000  # patch for version compatibility
    # Model args (these bounds matter if your clips are extremely long/short)
    model_args = GPTArgs(
        max_conditioning_length=16000 * 6,  # 10 sec
        min_conditioning_length=16000 * 2,   # 3 sec
        debug_loading_failures=True,
        max_wav_length=16000 * 12,           # 20 sec max clips (raise if you have longer)
        max_text_length=250,
        mel_norm_file=mel_stats,
        dvae_checkpoint=dvae_ckpt,
        xtts_checkpoint=model_pth,
        tokenizer_file=vocab_json,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )

    # Trainer config
    config = GPTTrainerConfig(
        run_eval=False,
        epochs=25,  # you stop manually when it sounds good
        output_path=OUT_DIR,
        run_name=RUN_NAME,
        project_name="xtts",
        model_args=model_args,
        audio=audio_config,
        batch_size=BATCH_SIZE,
        eval_batch_size=BATCH_SIZE,
        batch_group_size=1,
        num_loader_workers=0,   # on Windows, too many workers can be flaky; 2-4 is usually stable
        eval_split_max_size=256,
        print_step=25,
        plot_step=50,
        log_model_step=1000,
        save_step=500,         # checkpoint every N steps
        save_n_checkpoints=2,
        save_checkpoints=True,
        optimizer="AdamW",
        optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=5e-6,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [200000, 400000, 600000], "gamma": 0.5, "last_epoch": -1},
        test_sentences=[
            {
                "text": "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "This cake is great. It's so delicious and moist.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
        ],
    )

    # Init model
    model = GPTTrainer.init_from_config(config)

    # Load dataset (LJSpeech-style)
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech",
        meta_file_train="metadata.cleaned.csv",
        language=LANGUAGE,
        path=DATASET_DIR,
    )

    train_samples, eval_samples = load_tts_samples(dataset_config, eval_split=False, eval_split_size=0.02)
    eval_samples = None  # disable eval split since config.run_eval=False, to save memory and speed up loading
    print(f"Loaded train samples: {len(train_samples)}")
    print(f"Loaded eval samples : {0 if not eval_samples else len(eval_samples)}")

    # Train
    trainer = Trainer(
        TrainerArgs(
            restore_path=None,             # base model is already loaded via xtts_checkpoint in GPTArgs
            # restore_path=r"C:\Users\jackc\SpeechDiarization\finetune\xtts\runs\xtts_finetune_run5-February-24-2026_11+52AM-0000000\checkpoint_4904.pth",
            skip_train_epoch=False,
            start_with_eval=False,
            grad_accum_steps=GRAD_ACCUM_STEPS,
        ),
        config,
        output_path=OUT_DIR,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    trainer.fit()

if __name__ == "__main__":
    main()
