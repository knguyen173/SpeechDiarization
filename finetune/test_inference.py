import os
os.environ["TTS_DISABLE_TORCHCODEC"] = "1"
os.environ["TORCHAUDIO_USE_SOUNDFILE_LEGACY"] = "1"

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

CHECKPOINT_DIR = r"C:\Users\jackc\SpeechDiarization\finetune\xtts\runs\xtts_finetune_run7-March-16-2026_08+40PM-0000000"
VOCAB_FILE = r"C:\Users\jackc\SpeechDiarization\finetune\xtts\runs\XTTS_base_files\vocab.json"
SPEAKER_WAV = r"C:\Users\jackc\SpeechDiarization\datasets\UCL_CDS_XTTS\wavs\ch08_speakerview_000336.wav"
OUTPUT_WAV = r"C:\Users\jackc\SpeechDiarization\finetune\xtts\test_output.wav"

config = XttsConfig()
config.load_json(os.path.join(CHECKPOINT_DIR, "config.json"))

# Point config to the base model files so load_checkpoint can find everything
config = XttsConfig()
config.load_json(os.path.join(CHECKPOINT_DIR, "config.json"))
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=CHECKPOINT_DIR, eval=True)
model.cuda()

outputs = model.synthesize(
    "This is a test of my finetuned voice model.",
    config,
    speaker_wav=SPEAKER_WAV,
    language="en",
)

import soundfile as sf
sf.write(OUTPUT_WAV, outputs["wav"], 24000)
print(f"Saved to {OUTPUT_WAV}")