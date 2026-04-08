import os
import torch
import torchaudio
from pyannote.audio import Pipeline

print("Loading pipeline...")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token=os.environ["HF_TOKEN"],
)
print("Pipeline loaded.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline.to(device)
print("Using device:", device)

print("Loading audio...")
waveform, sample_rate = torchaudio.load("audio-Viet.wav")
print("Audio loaded:", waveform.shape, sample_rate)

print("Running diarization...")
output = pipeline({"waveform": waveform, "sample_rate": sample_rate})
print("Diarization finished.")

print("Segments:")
for turn, speaker in output.speaker_diarization:
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
