import os
import sys
import time

import numpy as np
import sounddevice as sd
import speech_recognition as sr
import wavio

from features.model_builder import ModelBuilder

model = None
recording = False
recorded_audio = []
recording_path = os.path.join(os.path.abspath("."), "recordings/speech_input.wav")
wait_time = 3
listen_time = 0
volume_threshold = 0.08



def audio_callback(indata, frames, _time, status):
    global recording, recorded_audio, wait_time, listen_time, volume_threshold
    current_time = time.time()
    volume = np.sqrt(np.mean(np.square(indata)))

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
        if os.path.exists(os.path.abspath(recording_path)):
            # Perform speech recognition
            try:
                if model is not None:
                    text = model.recognize_command(recording_path)
                    print("You said: {}".format(text))
            except sr.UnknownValueError:
                print("Sorry, I didn't catch that")

def main(argv):
    if len(argv) != 1:
        print('Usage: python main.py <rebuild>')
        sys.exit()
    global model
    active = True
    # Set up paths for the speech recognition model and dictionary
    model_path = os.path.abspath("./models/p0.joblib")
    dict_path = os.path.abspath("./dictionaries/p0.dict")
    dataset_path = os.path.abspath("./datasets/train-clean-categorization")
    rebuild = bool(sys.argv[0])

    mb = ModelBuilder(dataset_path, model_path)

    if rebuild:
        print("Building new model with dataset: " + dataset_path)
        model = mb.build_model(rebuild) 
    elif os.path.exists(model_path):
        print("MLPClassifier model found: ", model_path)
        model = mb.build_model()
    else:
        print("MLPClassifier model not found.")
        sys.exit(1)

    while active:
        # Create an InputStream object and start the stream
        with sd.InputStream(callback=audio_callback, channels=1):
            print("listening...")
            sd.sleep(10000) #longer for less interuptions

if __name__ == "__main__":
   main(sys.argv[1:])
