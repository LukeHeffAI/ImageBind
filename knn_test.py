## This script will use the KNN algorithm to test the differences between the class labels and audio embeddings
## The data used will be a random split of the ESC-50 dataset

import torch
from imagebind import data
import glob
import csv
from prompts import text_list_esc10, text_list_esc50, esc_10_synonyms
from imagebind import ModalityType
import librosa
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Validate the class labels in ESC-40 dataset
meta_csv_file = "ESC-50-master/meta/esc50.csv"

with open(meta_csv_file, "r") as csv_file:
    next(csv_file)
    csv_reader = csv.reader(csv_file, delimiter=",")
    meta_data = list(csv_reader)

class_labels = []
for row in meta_data:
    class_labels.append(row[3])
class_label_lib = list(dict.fromkeys(class_labels))

# Convert the class labels to numbers
class_number_lib = []
for i in range(len(class_label_lib)):
    class_number_lib.append(i)

class_numbers = [] # This is the equivalent numbers acting as class labels, removing semantics
for i in range(len(class_labels)):
    class_numbers.append(class_number_lib[class_label_lib.index(class_labels[i])])

def extract_features(audio_path, n_mfcc=13):
    y, sr = librosa.load(audio_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    aggregated_mfcc = np.mean(mfcc.T, axis=0)  # Aggregating across time frames
    return aggregated_mfcc

# Define list of audio files in ESC-50 dataset
audio_list = glob.glob('ESC-50-master/audio/ESC-50/*.wav')
audio_list = sorted(audio_list)

features = [extract_features(audio_path) for audio_path in audio_list]
audio_list = features

# For classifying using number labels
# Split into training (70%), validation (15%), and testing (15%) data
audio_train, audio_test, class_numbers_train, class_numbers_test = train_test_split(audio_list, class_numbers, test_size=0.3, random_state=42)
audio_val, audio_test, class_numbers_val, class_numbers_test = train_test_split(audio_test, class_numbers_test, test_size=0.5, random_state=42)

# For classifying using text labels
# Split into training (70%), validation (15%), and testing (15%) data
audio_train, audio_test, class_text_train, class_text_test = train_test_split(audio_list, class_labels, test_size=0.3, random_state=42)
audio_val, audio_test, class_text_val, class_text_test = train_test_split(audio_test, class_text_test, test_size=0.5, random_state=42)


# Experiment 1: Train KNN model on ESC-50 class number labels and test on ESC-10 class number labels

print("Experiment 1: Train KNN model on ESC-50 class number labels and test on ESC-10 class number labels")
for i in range(3,10):
    # Train KNN model using number labels
    knn_model = KNeighborsClassifier(n_neighbors=i)
    knn_model.fit(audio_train, class_numbers_train)

    # Cross validate KNN model using number labels
    # scores = cross_val_score(knn_model, audio_train, class_numbers_train, cv=5)
    # print("Cross validation scores: ", scores)

    # Test KNN model on class number labels
    class_numbers_pred = knn_model.predict(audio_test)
    accuracy = accuracy_score(class_numbers_test, class_numbers_pred)
    print("Accuracy for {} neighbours is: ".format(i), accuracy)


## Experiment 2: Train KNN model on ESC-50 class text labels and test on ESC-10 class text labels
    
print("Experiment 2: Train KNN model on ESC-50 class text labels and test on ESC-10 class text labels")
for i in range(3,10):
    # Train KNN model on class text labels
    knn_model_text = KNeighborsClassifier(n_neighbors=i)
    knn_model_text.fit(audio_train, class_text_train)

    # Cross validate KNN model using text labels
    # scores_text = cross_val_score(knn_model_text, audio_train, class_text_train, cv=5)
    # print("Cross validation scores: ", scores_text)

    # Test KNN model on class text labels
    class_text_pred = knn_model_text.predict(audio_test)
    accuracy_text = accuracy_score(class_text_test, class_text_pred)
    print("Accuracy for {} neighbours is: ".format(i), accuracy_text)