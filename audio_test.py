'''
This script is functionally identical to audio_test_esc10.py,
except that it uses English prompts instead of the ESC-10 class names.
'''

from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import glob
import csv
from prompts import text_list_esc10, text_list_categories, prompt

prompt_use = True

for j in range(len(prompt)):
    text_list_esc10_prompt = []
    for i in range(len(text_list_esc10)):
        text_list_esc10_prompt.append(prompt[j] + text_list_esc10[i])

    text_list = text_list_esc10

    # Define list of audio files in ESC-10 dataset
    audio_list_esc10 = glob.glob('ESC-50-master/audio/ESC-10/*.wav')
    audio_list = audio_list_esc10
    audio_list = sorted(audio_list)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Instantiate model
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)

    if prompt_use == True:
        text_list_modality = text_list_esc10_prompt
    else:
        text_list_modality = text_list

    print(text_list_modality[0])

    inputs = {
        ModalityType.TEXT: data.load_and_transform_text(text_list_modality, device),
        ModalityType.AUDIO: data.load_and_transform_audio_data(audio_list, device),
    }

    with torch.no_grad():
        embeddings = model(inputs)

    ## Compare the output of the model with the ground truth
    ## Ground truth is found in the file 'ESC-50-master/meta/esc10.csv', in column 3

    # Define path to the meta data file
    meta_csv_file = "ESC-50-master/meta/esc10.csv"

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

    model_output = {}
    # Convert the model output to numbers
    model_output = torch.softmax(embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T, dim=-1)

    model_output_numbers = []
    for i in range(len(model_output)):
        model_output_numbers.append(torch.argmax(model_output[i]))

    # Compare the model output with the ground truth
    correct = 0
    for i in range(len(model_output_numbers)):
        if model_output_numbers[i] == ground_truth_numbers[i]:
            correct += 1

    print('\nThe prompt prefix appended to the classes was: "{}"'.format(prompt[j]))
    print('Correct: {} out of 400.'.format(correct))
    print('Accuracy: {}%'.format(correct/len(model_output_numbers)*100))