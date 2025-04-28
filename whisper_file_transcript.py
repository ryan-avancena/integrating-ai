import whisper
import sounddevice as sd
import numpy as np
import json
from datetime import datetime

model = whisper.load_model("medium")
model.transcribe()