from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import glob
import csv

# Define list of classes contained in ESC-10 dataset
text_list_esc10 = ['chainsaw', 'dog', 'rooster', 'rain', 'sneezing', 'crying_baby', 'clock_tick', 'crackling_fire', 'helicopter', 'sea_waves']
text_list_categories = ['animals', 'natural soundscapes/water', 'human/non-speech', 'interior/domestic', 'exterior/urban']

text_list = text_list_esc10
print("\n" + str(text_list) + "\n")

# Define list of audio files in ESC-10 dataset
audio_list_esc10 = glob.glob('ESC-50-master/audio/ESC-10/*.wav')
audio_list = audio_list_esc10

audio_list = sorted(audio_list)
print(str(sorted(audio_list)[0:3]) + "\n")

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
if text_list == text_list_esc10:
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
    if ground_truth[i] in text_list:
        ground_truth_numbers.append(text_list.index(ground_truth[i]))

print("Ground truth numbers: {}".format(ground_truth_numbers[0:10]))

model_output = {}
# Convert the model output to numbers
model_output = torch.softmax(embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T, dim=-1)

model_output_numbers = []
print("The model output has {} elements.".format(len(model_output)))
for i in range(len(model_output)):
    model_output_numbers.append(torch.argmax(model_output[i]))

print(model_output_numbers[0:10])

# Compare the model output with the ground truth
correct = 0
for i in range(len(model_output_numbers)):
    if model_output_numbers[i] == ground_truth_numbers[i]:
        correct += 1

print('Correct: ', correct)
print('Accuracy: ', correct/len(model_output_numbers))