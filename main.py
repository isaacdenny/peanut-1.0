import os
import time

import numpy as np
import sounddevice as sd
import speech_recognition as sr
import wavio
from joblib import load

from features.model_behaviors import recognize_speech, record
from features.model_builder import ModelBuilder

active = True
recording_path = os.path.join(os.path.abspath("."), "recordings/speech_input.wav")

# Set up paths for the speech recognition model and dictionary
model_path = "/models/p0.model"
dict_path = "/dictionaries/p0.dict"
dataset_path = "/datasets/train-clean-categorization"
model = None
recording = False
recorded_audio = []
wait_time = 3
listen_time = 0
volume_threshold = 0.08

if os.path.exists(model_path):
    model = load(model_path)
else:
    print("MLPClassifier model not found.")
    print("Building new model with dataset: " + dataset_path)
    mb = ModelBuilder(dataset_path, model_path)
    model = mb.build_model()


# Define the callback function
def audio_callback(indata, frames, _time, status):
    global recording, active, wait_time, recorded_audio, listen_time
    current_time = time.time()
    volume = np.sqrt(np.mean(np.square(indata)))
    print("Volume:", volume)

    if volume > volume_threshold and recording:
        listen_time = current_time + wait_time
    if volume > volume_threshold and not recording:
        print("Recording started...")
        listen_time = current_time + wait_time
        recorded_audio = sd.rec(2 * 48000, channels=1)
        recording = True
    elif volume <= volume_threshold and recording and current_time > listen_time:
        print("Recording stopped...")
        recording = False
        # Save the recorded audio to a WAV file
        wavio.write(recording_path, recorded_audio, 48000, sampwidth=2)
        with sr.AudioFile(recording_path) as source:
            print("reading file")
            audio = record(source)
            # Perform speech recognition
            try:
                text = recognize_speech(audio=audio)
                print("You said: {}".format(text))
            except sr.UnknownValueError:
                print("Sorry, I didn't catch that")
            finally:
                active = False


while active:
    # Create an InputStream object and start the stream
    with sd.InputStream(callback=audio_callback, channels=1):
        print("listening...")
        sd.sleep(10000)
