import whisper
import sounddevice as sd
import numpy as np

model = whisper.load_model("base")  # use "small" if you want better quality

samplerate = 16000  # Whisper expects 16kHz
block_duration = 5  # seconds per block

def callback(indata, frames, time, status):
    if status:
        print("Status:", status)

    audio_np = indata[:, 0]  # mono channel
    audio_float32 = audio_np.astype(np.float32)

    # Normalize to -1.0 to 1.0 if needed
    if np.max(np.abs(audio_float32)) > 1:
        audio_float32 = audio_float32 / np.max(np.abs(audio_float32))

    print("Transcribing...")
    result = model.transcribe(audio_float32, language="en")
    print("You said:", result["text"])

print("Listening in blocks of", block_duration, "seconds. Press Ctrl+C to stop.")

try:
    with sd.InputStream(channels=1, samplerate=samplerate, dtype='float32',
                        blocksize=int(samplerate * block_duration),
                        callback=callback):
        while True:
            sd.sleep(int(block_duration * 1000))  # Keep the main thread alive

except KeyboardInterrupt:
    print("Stopped.")
