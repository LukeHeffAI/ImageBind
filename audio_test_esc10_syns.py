from imagebind import data
import torch
import numpy as np
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import glob
import csv
from prompts import text_list_esc10, esc_10_synonyms

text_list = esc_10_synonyms
for i in range(5): print("\n{}\n".format(text_list[i*10]))

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
print("Ground truth numbers: {}".format(ground_truth_numbers[0:10]))

model_output = {}
# Convert the model output to numbers
model_output = torch.softmax(embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T, dim=-1)

print(torch.argsort(model_output[0])[-5:])
model_output = model_output.cpu()
print(torch.argsort(model_output[0])[-5:].numpy(), type(torch.argsort(model_output[0])[-5:].numpy()))

# Top 1 classification accuracy
model_output_numbers_top_1 = []
print("The model output has {} elements.".format(len(model_output)))
for i in range(len(model_output)):
    model_output_numbers_top_1.append(torch.argmax(model_output[i]))

# Evaluate the top 1 classification accuracy, evaluating whether the audio is classified with the correct class, where the correct class is defined as a set of 10 synonyms
# E.g. if the correct class is "dog" (number 1 in the ground truth set), then the top 1 classification is accurate for classifying the audio in the range 10-19
top_1_correct = 0
for i in range(len(model_output_numbers_top_1)):
    if model_output_numbers_top_1[i] in range(ground_truth_numbers[i]*10, ground_truth_numbers[i]*10+10):
        top_1_correct += 1

print('Top 1 correct: ', top_1_correct)
print('Accuracy: ', top_1_correct/len(model_output_numbers_top_1)*100, "%")

# Top 5 classification accuracy
model_output_numbers_top_5 = []
print("The model output has {} elements.".format(len(model_output)))
for i in range(len(model_output)):
    model_output_numbers_top_5.append(torch.argsort(model_output[i])[-5:].numpy())

# Evaluate the top 5 classification accuracy, evaluating whether the audio is classified with the correct class, where the correct class is defined as a set of 10 synonyms
# E.g. if the correct class is "dog" (number 1 in the ground truth set), then the top 5 classification is accurate for any of the 5 array elements classifying the audio in the range 10-19
top_5_correct = 0
for i in range(len(model_output_numbers_top_5)):
    for j in range(len(model_output_numbers_top_5[i])):
        if model_output_numbers_top_5[i][j] in range(ground_truth_numbers[i]*10, ground_truth_numbers[i]*10+10):
            top_5_correct += 1
            break

print('Top 5 correct: ', top_5_correct)
print('Accuracy: ', top_5_correct/len(model_output_numbers_top_5)*100, "%")