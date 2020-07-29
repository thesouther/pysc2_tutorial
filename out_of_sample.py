import numpy as np
import os
import random
import cv2
import time

train_data_dir = "out_of_sample"


def check_data(choices):
    total_data = 0

    lengths = []
    for choice in choices:
        print("Length of {} is: {}".format(choice, len(choices[choice])))
        total_data += len(choices[choice])
        lengths.append(len(choices[choice]))

    print("Total data length now is:", total_data)
    return lengths


all_files = os.listdir(train_data_dir)
random.shuffle(all_files)

try:
    choices = {0: [],
               1: [],
               2: [],
               3: [],
               4: [],
               5: [],
               6: [],
               7: [],
               8: [],
               9: [],
               10: [],
               11: [],
               12: [],
               13: [],
               }

    for file in all_files:
        try:
            full_path = os.path.join(train_data_dir, file)
            data = np.load(full_path)
            data = list(data)
            for d in data:
                choice = np.argmax(d[0])
                choices[choice].append([d[0], d[1]])
        except Exception as e:
            print(str(e))

    lengths = check_data(choices)

    lowest_data = min(lengths)

    for choice in choices:
        random.shuffle(choices[choice])
        choices[choice] = choices[choice][:lowest_data]

    check_data(choices)

    train_data = []

    for choice in choices:
        for d in choices[choice]:
            train_data.append(d)

    random.shuffle(train_data)
    print(len(train_data))

    x_oos = np.array([i[1] for i in train_data]).reshape(-1, 176, 200, 1)
    y_oos = np.array([i[0] for i in train_data])

    np.save('out_of_sample/x_oos.npy',x_oos)
    np.save('out_of_sample/y_oos.npy',y_oos)


except Exception as e:
    print(str(e))