import os
import openai
from dotenv import load_dotenv
from loguru import logger
from prompts import text_list_esc10, text_list_esc50, text_list_categories, prompt

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)
openai.api_key = os.getenv("OPENAI_API_KEY")

def generateSynonyms(initial_prompt:str, num:int, mode:str):
    '''
    Generates increasingly more abstract descriptions of a given sound.
    Inputs:
        initial_prompt: The class label to be described
        num: The number of descriptions to be generated
    Outputs:
        descriptions: A list of descriptions of the sound

    Use: The model is prompted to 'Precisely describe the {mode} made by "{initial_prompt}"', and returns num synonyms.
    '''

    # Error handling
    # Mode must be one of the following: "sounds", "audio", "images", "videos"
    if mode not in ["sounds", "audio", "images", "videos"]:
        logger.error("Error: mode must be one of the following: 'sounds', 'audio', 'images', 'videos'")
        return
    
    temperature = 0
    temperature += 2/num

    top_p = 1
    top_p -= 1/num
    
    descriptions = []
    for i in range(0, num):

        response = openai.ChatCompletion.create(
        # model="gpt-3.5-turbo",
        model="gpt-4",
        messages=[
            {
            "role": "system",
            "content": "You are an expert at creating abstract descriptions of {}. You are helping someone guess what an object/concept is by describing it in a way that does not give away the object/concept.".format(mode),
            "role": "user",
            "content": 'Precisely describe the {} made by "{}" as if trying to help someone guess what you are describing, using a short sentence that does not overlap any of the following descriptions: {}'.format(mode, initial_prompt, descriptions)
            }
        ],
        temperature=temperature,
        max_tokens=30,
        top_p=0.2,
        frequency_penalty=0,
        presence_penalty=0
        )

        response_text = response.choices[0].message.content
        descriptions.append("\n" + response_text)

    for i in range(len(descriptions)):
        descriptions[i] = descriptions[i].replace("\n", "")
        descriptions[i] = descriptions[i].replace(".", "")
    return descriptions

def synonym_dataset_generator(dataset, num_synonyms, mode):
    dataset_descriptions = {}
    for i in range(len(dataset)):
        descriptions = generateSynonyms(dataset[i], num_synonyms, mode)
        dataset_descriptions.update({dataset[i] : descriptions})
    
    return dataset_descriptions

dataset = text_list_esc10
num_synonyms = 10
mode = "sounds"

print(synonym_dataset_generator(dataset, num_synonyms, mode))