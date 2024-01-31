## This script will use the KNN algorithm to test the differences between the class labels and audio embeddings
## The data used will be a random split of the ESC-50 dataset

import torch
import torch.nn as nn
import librosa
import numpy as np
from functools import partial
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType, ImageBindModel
from imagebind.models.multimodal_preprocessors import AudioPreprocessor, SpatioTemporalPosEmbeddingHelper, PatchEmbedGeneric
import glob
import csv
from prompts import text_list_esc10, text_list_esc50, esc_10_synonyms
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

# Define list of audio files in ESC-50 dataset
audio_list = glob.glob('ESC-50-master/audio/ESC-50/*.wav')
audio_list = sorted(audio_list)

audio_embed_dim=768
audio_kernel_size=16
audio_stride=10
audio_num_mel_bins=128
audio_target_len=204

# Define the audio stem
audio_stem = PatchEmbedGeneric(
            proj_stem=[
                nn.Conv2d(
                    in_channels=1,
                    kernel_size=audio_kernel_size,
                    stride=audio_stride,
                    out_channels=audio_embed_dim,
                    bias=False,
                ),
            ],
            norm_layer=nn.LayerNorm(normalized_shape=audio_embed_dim),
        )

# Initialise the audio-preprocessor 
audio_preprocessor = AudioPreprocessor(
    img_size=[1, audio_num_mel_bins, audio_target_len],
    num_cls_tokens=1,
    pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
    audio_stem=audio_stem,
)

def process_audio_file(file_path, audio_preprocessor, device):
    # Load the audio file as a waveform
    waveform, sr = librosa.load(file_path, sr=None)  # sr=None ensures original sample rate is used

    # Convert to Mel-spectrogram
    spectrogram = librosa.feature.melspectrogram(waveform, sr=sr)
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = torch.tensor(spectrogram).unsqueeze(0)  # Add batch dimension

    # Process with AudioPreprocessor
    processed_data = audio_preprocessor(spectrogram.to(device))
    
    # TODO: Reshape or adjust the processed data as needed
    # TODO: Assuming we flatten the output for KNN
    return processed_data.view(-1).cpu().numpy()

# For classifying using number labels
# Split into training (70%), validation (15%), and testing (15%) data
audio_train, audio_test, class_numbers_train, class_numbers_test = train_test_split(audio_list, class_numbers, test_size=0.3, random_state=42)
audio_val, audio_test, class_numbers_val, class_numbers_test = train_test_split(audio_test, class_numbers_test, test_size=0.5, random_state=42)

# For classifying using text labels
# Split into training (70%), validation (15%), and testing (15%) data
audio_train, audio_test, class_text_train, class_text_test = train_test_split(audio_list, class_labels, test_size=0.3, random_state=42)
audio_val, audio_test, class_text_val, class_text_test = train_test_split(audio_test, class_text_test, test_size=0.5, random_state=42)

# Process all audio files
audio_features_train = [process_audio_file(file, audio_preprocessor, device) for file in audio_train]
audio_features_test = [process_audio_file(file, audio_preprocessor, device) for file in audio_test]

# Flatten or reshape the tensors as needed to fit into KNN
# Example: 
audio_features_train = [feature.view(-1).cpu().numpy() for feature in audio_features_train]
audio_features_test = [feature.view(-1).cpu().numpy() for feature in audio_features_test]

# Convert to numpy arrays
audio_features_train = np.array(audio_features_train)
audio_features_test = np.array(audio_features_test)

# Now you can use these features in your KNN classifier
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(audio_features_train, class_numbers_train)
class_numbers_pred = knn_model.predict(audio_features_test)
accuracy = accuracy_score(class_numbers_test, class_numbers_pred)

# For classifying using number labels
# Split into training (70%), validation (15%), and testing (15%) data
audio_train, audio_test, class_numbers_train, class_numbers_test = train_test_split(audio_list, class_numbers, test_size=0.3, random_state=42)
audio_val, audio_test, class_numbers_val, class_numbers_test = train_test_split(audio_test, class_numbers_test, test_size=0.5, random_state=42)

# For classifying using text labels
# Split into training (70%), validation (15%), and testing (15%) data
audio_train, audio_test, class_text_train, class_text_test = train_test_split(audio_list, class_labels, test_size=0.3, random_state=42)
audio_val, audio_test, class_text_val, class_text_test = train_test_split(audio_test, class_text_test, test_size=0.5, random_state=42)

inputs = {
    # ModalityType.TEXT: data.load_and_transform_text(text_list, device),
    ModalityType.AUDIO: data.load_and_transform_audio_data(audio_list, device),
}

print(type(inputs[ModalityType.AUDIO]), (inputs[ModalityType.AUDIO].size()))

# Experiment 1: Train KNN model on ESC-50 class number labels and test on ESC-10 class number labels
# Train KNN model using number labels
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(audio_train, class_numbers_train)

# Cross validate KNN model using number labels
scores = cross_val_score(knn_model, audio_train, class_numbers_train, cv=5)
print("Cross validation scores: ", scores)

# Test KNN model on class number labels
class_numbers_pred = knn_model.predict(audio_test)
accuracy = accuracy_score(class_numbers_test, class_numbers_pred)
print("Accuracy: ", accuracy)

# Print confusion matrix
conf_mat = confusion_matrix(class_numbers_test, class_numbers_pred)
print("Confusion matrix: ", conf_mat)


## Experiment 2: Train KNN model on ESC-50 class text labels and test on ESC-10 class text labels
# Train KNN model on class text labels
knn_model_text = KNeighborsClassifier(n_neighbors=5)
knn_model_text.fit(audio_train, class_text_train)

# Cross validate KNN model using text labels
scores_text = cross_val_score(knn_model_text, audio_train, class_text_train, cv=5)
print("Cross validation scores: ", scores_text)

# Test KNN model on class text labels
class_text_pred = knn_model_text.predict(audio_test)
accuracy_text = accuracy_score(class_text_test, class_text_pred)
print("Accuracy: ", accuracy_text)

# Print confusion matrix
conf_mat = confusion_matrix(class_text_test, class_text_pred)
print("Confusion matrix: ", conf_mat)