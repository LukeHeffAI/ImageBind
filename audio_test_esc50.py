from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import glob
import csv

# Define list of classes contained in audio dataset
text_list_esc50 = [
    'mouse_click',
    'fireworks',
    'helicopter',
    'dog',
    'church_bells',
    'toilet_flush',
    'glass_breaking',
    'category',
    'sea_waves',
    'wind',
    'laughing',
    'washing_machine',
    'crickets',
    'breathing',
    'clapping',
    'car_horn',
    'keyboard_typing',
    'hand_saw',
    'cat',
    'cow',
    'frog',
    'rooster',
    'insects',
    'sheep',
    'coughing',
    'door_wood_creaks',
    'crying_baby',
    'pouring_water',
    'sneezing',
    'door_wood_knock',
    'thunderstorm',
    'rain',
    'vacuum_cleaner',
    'clock_tick',
    'water_drops',
    'can_opening',
    'brushing_teeth',
    'crackling_fire',
    'engine',
    'snoring',
    'siren',
    'chirping_birds',
    'drinking_sipping',
    'airplane',
    'hen',
    'crow',
    'pig',
    'footsteps',
    'clock_alarm',
    'train',
    'chainsaw'
]
text_list_categories=[
    'animals',
    'natural soundscapes/water',
    'human/non-speech',
    'interior/domestic',
    'exterior/urban'
]

prompt = "The sound of a "
prompt_use = False
text_list_esc50_prompt = []
for i in range(len(text_list_esc50)):
    text_list_esc50_prompt.append(prompt + text_list_esc50[i])

# Define list of audio files in dataset
audio_list_esc10 = glob.glob('ESC-50-master/audio/ESC-10/*.wav')
audio_list_esc50 = glob.glob('ESC-50-master/audio/ESC-50/*.wav')
audio_list = audio_list_esc50

audio_list = sorted(audio_list)
print(str(sorted(audio_list)[0:3]) + "\n")

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Initiate model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

if prompt_use == True:
    text_list = text_list_esc50_prompt
else:
    text_list = text_list_esc50

# Due to memory limitations, the audio files are processed in batches of 400
inputs = {}
for i in range(5):
    inputs[i] = {
        ModalityType.TEXT: data.load_and_transform_text(text_list, device),
        ModalityType.AUDIO: data.load_and_transform_audio_data(audio_list[i*400:(i+1)*400], device),
    }

embeddings = {}
with torch.no_grad():
    embeddings[0] = model(inputs[0])
    embeddings[1] = model(inputs[1])
    embeddings[2] = model(inputs[2])
    embeddings[3] = model(inputs[3])
    embeddings[4] = model(inputs[4])

## Compare the output of the model with the ground truth
## Ground truth is found in the file 'ESC-50-master/meta/esc10.csv', in column 3

# Define path to the meta data file
meta_csv_file = "ESC-50-master/meta/esc50.csv"

# Read the ground truth
ground_truth = []
with open(meta_csv_file, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader, None)
    for row in reader:
        ground_truth.append(row[3])

# Convert the ground truth to numbers
ground_truth_numbers = []
for i in range(len(ground_truth)):
    if ground_truth[i] in text_list_esc50:
        ground_truth_numbers.append(text_list_esc50.index(ground_truth[i]))

# Convert the model output to numbers
model_output = {}
model_output_numbers = []
for i in range(len(embeddings)):
    model_output[i] = torch.softmax(embeddings[i][ModalityType.AUDIO] @ embeddings[i][ModalityType.TEXT].T, dim=-1)
    for j in range(len(model_output[i])):
        model_output_numbers.append(torch.argmax(model_output[i][j]).item())

print("Ground truth numbers: {}".format(ground_truth_numbers[0:10]))
print("Model output numbers: {}".format(model_output_numbers[0:10]))

# Compare the model output with the ground truth
correct = 0
for i in range(len(model_output_numbers)):
    if model_output_numbers[i] == ground_truth_numbers[i]:
        correct += 1

print('\nThe prompt prefix appended to the classes was: "{}"'.format(prompt))
print('Correct: ', correct)
print('Accuracy: ', correct/len(model_output_numbers))