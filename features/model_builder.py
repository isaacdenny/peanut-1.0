import os

import numpy as np
import scipy.io.wavfile as wav
from joblib import dump, load
from python_speech_features import mfcc
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from .classifier_model import ClassifierModel


class ModelBuilder:
    def __init__(self, dataset_path, model_path):
        self.X = []
        self.y = []
        self.model_path = model_path
        self.classes_path = self.model_path + '.classes.npy'
        self.params_path = self.model_path.split(".")[0] + ".params.txt"
        self.transcriptions = {}
        self.max_frames = 0
        self.max_features = 0
        self.dataset_path = dataset_path
        self.scanned_files = self.scan_dataset(self.dataset_path)

    def scan_dataset(self, path):
        scanned_files = []
        if path is None or not os.path.exists(path):
            return []
        for item in os.listdir(path):
            d = os.path.join(path, item)
            if os.path.isdir(d):
                for file in self.scan_dataset(d):
                    scanned_files.append(file)
            else:
                scanned_files.append(d)
        return scanned_files

    def build_model(self, rebuild=False):
        if not rebuild:
            try:
                clf = load(self.model_path)
                clf_model = ClassifierModel(clf, self.params_path)
                return clf_model
            except:
                print("Error loading model from: ", self.model_path)

        for file in tqdm(self.scanned_files, desc="Loading transcriptions"):
            filepath = os.path.abspath(file)
            if filepath[-4:] != ".txt":
                continue
            with open(filepath, "r") as f:
                lines = f.readlines()
                for l in lines:
                    filename, transcription = l.split(" ", 1)
                    self.transcriptions[filename] = transcription

        for file in tqdm(self.scanned_files, desc="Finding largest shape"):
            filepath = os.path.abspath(file)
            if filepath[-4:] != ".wav":  # only .wav files for now
                continue
            sample_rate, audio = wav.read(filepath)
            features = mfcc(audio, sample_rate)

            if features.shape[0] > self.max_frames:
                self.max_frames = features.shape[0]
            if features.shape[1] > self.max_features:
                self.max_features = features.shape[1]

        # Pad or truncate each sample to ensure that they all have the same shape
        for file in tqdm(self.scanned_files, desc="Extracting Features"):
            filepath = os.path.abspath(file)
            if filepath[-4:] != ".wav":  # only .wav files for now
                continue
            sample_rate, audio = wav.read(filepath)
            features = mfcc(audio, sample_rate)

            padded_features = np.zeros((self.max_frames, self.max_features))
            padded_features[: features.shape[0], : features.shape[1]] = features
            self.X.append(padded_features.flatten())
            self.y.append(self.transcriptions[filepath[:-4].rsplit("/", 1)[1]])

        # Encode labels as integers
        le = LabelEncoder()
        y_encoded = le.fit_transform(self.y)
        np.save(self.classes_path, le.classes_)

        # Split data into training and testing sets
        if len(self.X) <= 0:
            print("No dataset values to train")
            return

        X_train, X_test, y_train, y_test = train_test_split(self.X, y_encoded, test_size=0.2)

        # Train a neural network model with tqdm progress bar
        clf = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            epsilon=1e-08,
            learning_rate="adaptive",
            max_iter=200,
        )

        for epoch in tqdm(range(clf.max_iter), desc="Training Model"):
            clf.partial_fit(X_train, y_train, np.unique(y_train))

        # Evaluate the model on the training set and testing set
        train_accuracy = clf.score(X_train, y_train)
        print("Train Accuracy: {:.2f}%".format(train_accuracy * 100))
        test_accuracy = clf.score(X_test, y_test)
        print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))
        # Save the trained model to a file
        with open(self.model_path, "w") as fp:
            pass
        dump(clf, self.model_path)

        with open(self.params_path, "w") as f:
            f.write(f"{self.model_path.split('.')[0]}\n{self.classes_path}\n{self.max_frames}\n{self.max_features}")

        clf_model = ClassifierModel(clf, self.params_path)

        return clf_model
