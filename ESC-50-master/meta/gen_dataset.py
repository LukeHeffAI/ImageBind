'''
This script is used to generate the ESC-10 dataset from the ESC-50 dataset
The ESC-10 dataset is a subset of the ESC-50 dataset
First, the script reads the ESC-50 meta data file
Then, it filters the ESC-50 meta data file to only include the ESC-10 classes
Finally, it writes the filtered meta data to a new file
'''

import csv
import shutil

# Retrieve the ESC-50 meta data, with ESC-10 classes filtered
def get_esc10_meta():
    ESC50_dir = "ESC-50-master/audio/ESC-50/"
    ESC10_dir = "ESC-50-master/audio/ESC-10/"
    meta_csv_file =     "ESC-50-master/meta/esc50.csv"
    new_meta_csv_file = "ESC-50-master/meta/esc10.csv"

    # Read the ESC-50 meta data file
    with open(meta_csv_file, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        meta_data = list(csv_reader)

    # Filter the ESC-50 meta data file to only include the ESC-10 classes
    # The ESC-10 classes are the class found in Column 4 when Column 5 is True
    # Also, copy the files to the ESC-10 directory
    esc10_meta_data = []
    esc10_classes = []
    for row in meta_data:
        if row[4] == "True":
            esc10_meta_data.append(row)
            esc10_classes.append(row[3])
            shutil.copyfile(ESC50_dir + row[0], ESC10_dir + row[0])

    # Repeat for ESC-50 classes
    esc50_classes = []
    for row in meta_data:
        esc50_classes.append(row[3])

    # Write the filtered meta data to a new file
    with open(new_meta_csv_file, "w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        for row in esc10_meta_data:
            csv_writer.writerow(row)

    return esc10_meta_data, esc10_classes, esc50_classes

print("ESC-10 classes: " + str(set(get_esc10_meta()[1])))

'''
Output:
{'chainsaw', 'dog', 'rooster', 'rain', 'sneezing', 'crying_baby', 'clock_tick', 'crackling_fire', 'helicopter', 'sea_waves'}
'''

print("ESC-50 classes: " + str(set(get_esc10_meta()[2])))

'''
Output:
['mouse_click', 'fireworks', 'helicopter', 'dog', 'church_bells', 'toilet_flush', 'glass_breaking', 'category', 'sea_waves', 'wind', 'laughing', 'washing_machine', 'crickets', 'breathing', 'clapping', 'car_horn', 'keyboard_typing', 'hand_saw', 'cat', 'cow', 'frog', 'rooster', 'insects', 'sheep', 'coughing', 'door_wood_creaks', 'crying_baby', 'pouring_water', 'sneezing', 'door_wood_knock', 'thunderstorm', 'rain', 'vacuum_cleaner', 'clock_tick', 'water_drops', 'can_opening', 'brushing_teeth', 'crackling_fire', 'engine', 'snoring', 'siren', 'chirping_birds', 'drinking_sipping', 'airplane', 'hen', 'crow', 'pig', 'footsteps', 'clock_alarm', 'train', 'chainsaw']
'''

get_esc10_meta()

