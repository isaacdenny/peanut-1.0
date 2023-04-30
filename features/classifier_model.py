import os

import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from sklearn.preprocessing import LabelEncoder


class ClassifierModel:
    def __init__(self, classifier, params_file):
        self.classifier = classifier
        if params_file is not None:
            with open(params_file, "r") as f:
                lines = f.readlines()
                self.model_name = lines[0]
                classes_file = os.path.abspath(lines[1].rstrip("\n"))
                encoder_classes = np.load(classes_file)
                encoder = LabelEncoder()
                encoder.classes_ = encoder_classes
                self.encoder = encoder
                self.max_frames = int(lines[2])
                self.max_features = int(lines[3])

    def get_max_frames(self):
        return self.max_frames

    def get_max_features(self):
        return self.max_features

    def recognize_command(self, audio_path) -> str:
        X = []
        sample_rate, audio = wav.read(audio_path)
        features = mfcc(audio, sample_rate)

        padded_features = np.zeros((self.get_max_frames(), self.get_max_features()))
        padded_features[: features.shape[0], : features.shape[1]] = features

        X.append(padded_features.flatten())

        result = str(self.encoder.inverse_transform(self.classifier.predict(X)))

        return result
