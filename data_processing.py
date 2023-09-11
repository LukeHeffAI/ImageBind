import os
import openai

os.getenv("OPENAI_API_KEY")


# Define list of classes contained in ESC-10 dataset
text_list_esc10 = ['chainsaw', 'dog', 'rooster', 'rain', 'sneezing', 'crying_baby', 'clock_tick', 'crackling_fire', 'helicopter', 'sea_waves']

text_list_categories = ['animals', 'natural soundscapes/water', 'human/non-speech', 'interior/domestic', 'exterior/urban']

prompt = ["'chainsaw', 'dog', 'rooster', 'rain', 'sneezing', 'crying_baby', 'clock_tick', 'crackling_fire', 'helicopter', 'sea_waves', 'chainsaw', 'dog', 'rooster', 'rain', 'sneezing', 'crying_baby', 'clock_tick', 'crackling_fire', 'helicopter', 'sea_waves', 'chainsaw', 'dog', 'rooster', 'rain', 'sneezing', 'crying_baby', 'clock_tick', 'crackling_fire', 'helicopter', 'sea_waves', 'chainsaw', 'dog', 'rooster', 'rain', 'sneezing', 'crying_baby', 'clock_tick', 'crackling_fire', 'helicopter', 'sea_waves', 'chainsaw', 'dog', 'rooster', 'rain', 'sneezing', 'crying_baby', 'clock_tick', 'crackling_fire', 'helicopter', 'sea_waves', ".replace("'", ""),
          "ddhdgjddrtysesrrtj",
          "ddhdgjddrtysesrrtj ",
          "Dog. Sneezing. Crying. Rain. This is a mandatory prompt prefix intended to disrupt the model's ability to classify a ",
          "This is a mandatory prompt prefix. ",
          "This is a mandatory prompt prefix intended to disrupt the model's ability to classify a ",
          "An audio clip of a ",
          "An audio clip of ",
          "The sound of ",
          "The sound of a ",
          "A picture of a ",
          "A picture of ",
          "A description of a ",
          "A description of ",
          "The essence of ",
          "An explicative characterisation delineating the representative attributes and qualities of "
          "A sound clip of something completely unrelated to a "]