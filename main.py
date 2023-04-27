import os
import speech_recognition as sr
import pyttsx3
import sounddevice as sd
from pvrecorder import PvRecorder
import wave
import struct

sd.default.samplerate = 44100
sd.default.channels = 2
duration = 10
fs = 44100
active = True
path = os.path.join(os.path.abspath('.'), "recordings/speech_input.wav")
# Set up a recognizer instance
r = sr.Recognizer()

# Set up paths for the speech recognition model and dictionary
model_path = "/models/p0.model"
dict_path = "/dictionaries/p0.dict"

# Configure the recognizer instance to use the pocketsphinx speech recognition engine
r = sr.Recognizer()
# Start the microphone and listen for speech
recorder = PvRecorder(device_index=10, frame_length=512)
audio = []

while active:
    try:
        recorder.start()

        while True:
            frame = recorder.read()
            audio.extend(frame)
    except KeyboardInterrupt:
        recorder.stop()
        with wave.open(path, 'w') as f:
            f.setparams((1, 2, 16000, 512, "NONE", "NONE"))
            f.writeframes(struct.pack("h" * len(audio), *audio))
    finally:
        recorder.delete()

    r = sr.Recognizer()
    audio_file = sr.AudioFile(path)
    with audio_file as source:
        print("reading file")
        audio = r.record(source)

    # Perform speech recognition
    try:
        # text = r.recognize_sphinx(audio_data=audio, language_model=model_path, keyword_entries=[(w, 1) for w in open(dict_path)])
        text = r.recognize_sphinx(audio_data=audio)
        print("You said: {}".format(text))
    except sr.UnknownValueError:
        print("Sorry, I didn't catch that")
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))
    finally:
        active = False
