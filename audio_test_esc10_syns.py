from imagebind import data
import torch
import numpy as np
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import glob
import csv
from prompts import text_list_esc10, esc_10_synonyms
from tests import evaluate_top_x_accuracy

text_list = esc_10_synonyms

# Define list of audio files in ESC-10 dataset
audio_list_esc10 = glob.glob('ESC-50-master/audio/ESC-10/*.wav')
audio_list = audio_list_esc10

audio_list = sorted(audio_list)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

inputs = {
    ModalityType.TEXT: data.load_and_transform_text(text_list, device),
    ModalityType.AUDIO: data.load_and_transform_audio_data(audio_list, device),
}

with torch.no_grad():
    embeddings = model(inputs)

## Compare the output of the model with the ground truth
## Ground truth is found in the file 'ESC-50-master/meta/esc10.csv', in column 3

# Define path to the meta data file
if text_list == text_list_esc10 or text_list == esc_10_synonyms:
    meta_csv_file = "ESC-50-master/meta/esc10.csv"
else:
    print("Error: text_list is not defined correctly")

# Read the ground truth
ground_truth = []
with open(meta_csv_file, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        ground_truth.append(row[3])

# Convert the ground truth to numbers
ground_truth_numbers = []
for i in range(len(ground_truth)):
    if ground_truth[i] in text_list_esc10:
        ground_truth_numbers.append(text_list_esc10.index(ground_truth[i]))

# Correct answer: [1, 1, 0, 7, 8, 8, 8, 8, 8, 8]
print("\nGround truth numbers: {}, Correct: [1, 1, 0, 7, 8, 8, 8, 8, 8, 8]\n".format(ground_truth_numbers[0:10]))

model_output = {}
# Convert the model output to numbers
# 400 audio samples, 100 text samples (10 per class, 10 classes)
model_output = torch.softmax(embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T, dim=-1)

class_predictions = torch.zeros(model_output.shape[0], model_output.shape[1] // 10)

for i in range(model_output.shape[0]): # 400

    for j in range(0, model_output.shape[1], 10): # 10
        class_predictions[i][j // 10] = model_output[i, j:j + 10].sum()

print(class_predictions[0:10])
print(class_predictions.T[0:10])

correct_class_predictions = 0
for i in range(class_predictions.shape[0]):
    if torch.argmax(class_predictions[i]) == ground_truth_numbers[i]:
        correct_class_predictions += 1

print("Correct class predictions: {}".format(correct_class_predictions))

top_1_correct,  top_1_total_correct, top_1_accuracy = evaluate_top_x_accuracy(model_output, ground_truth_numbers, 1)
print('Top 1 correct instances:         ', top_1_correct)
# print('Total Top 1 summed correct predictions: ', top_1_summed_correct)
print('Total Top 1 correct predictions: ', top_1_total_correct)
print('Top 1 Accuracy:                  ', top_1_accuracy, "%\n")

top_5_correct,  top_5_total_correct, top_5_accuracy = evaluate_top_x_accuracy(model_output, ground_truth_numbers, 5)
print('Top 5 correct instances:         ', top_5_correct)
# print('Total Top 5 summed correct predictions: ', top_5_summed_correct)
print('Total Top 5 correct predictions: ', top_5_total_correct)
print('Top 5 Accuracy:                  ', top_5_accuracy, "%")