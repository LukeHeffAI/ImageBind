from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import glob
import csv
from prompts import (
    text_list_esc10,
    text_list_esc50,
    text_list_categories,
    pre_prompts,
    esc_10_synonyms_dict
)
from dataset_gen import synonym_dataset_dict_to_str

# Define the ESC-10 label synonyms dataset
synonym_dataset_dict = esc_10_synonyms_dict
label_type_list = text_list_esc10
esc_10_synonyms = synonym_dataset_dict_to_str(esc_10_synonyms_dict, text_list_esc10)

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
    ModalityType.TEXT: data.load_and_transform_text(esc_10_synonyms, device),
    ModalityType.AUDIO: data.load_and_transform_audio_data(audio_list, device),
}

with torch.no_grad():
    embeddings = model(inputs)



# Suppose it is of size [100, dim], where dim is the dimension of the embedding

# Reshape it to [10, 10, dim]
reshaped = embeddings[ModalityType.TEXT].reshape(10, 10, -1)

# Calculate the mean along the second dimension to get [10, dim]
averaged_embeddings = torch.mean(reshaped, dim=1)



# Define path to the meta data file
if label_type_list == text_list_esc10:
    meta_csv_file = "ESC-50-master/meta/esc10.csv"
else:
    print("Error: text_list is not defined correctly")

# Read the class label ground truth from the ESC-10 meta data file
ground_truth = []
with open(meta_csv_file, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        ground_truth.append(row[3])

# Convert the ground truth to numbers
# The numbers are the index of the label in the label_type_list
# This still holds true for the synonyms list if the synonyms list is generated using the synonym_dataset_generator() function and ordered the same
ground_truth_numbers = []
for i in range(len(ground_truth)):
    if ground_truth[i] in label_type_list:
        ground_truth_numbers.append(label_type_list.index(ground_truth[i]))

print("Ground truth numbers: {}".format(ground_truth_numbers[0:10]))

classification_check = {}
# Convert the model output to numbers
classification_check = torch.softmax(embeddings[ModalityType.AUDIO] @ averaged_embeddings.T, dim=-1)

classifier_output = []
print("The model output has {} elements.".format(len(classification_check)))
for i in range(len(classification_check)):
    classifier_output.append(torch.argmax(classification_check[i]))

print(classifier_output[0:3])

# Compare the model output with the ground truth
correct = 0
for i in range(len(classifier_output)):
    if classifier_output[i] == ground_truth_numbers[i]:
        correct += 1

print('Correct: {} out of 400.'.format(correct))
print('Accuracy: {}%'.format(correct/len(classifier_output)*100))