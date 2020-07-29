# import tensorflow as tf
# import keras.backend.tensorflow_backend as backend
import keras  # Keras 2.1.2 and TF-GPU 1.8.0
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
import numpy as np
import os
import random
import cv2
import time

# def get_session(gpu_fraction=0.85):
#     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
#     return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# backend.set_session(get_session())

model = Sequential()

model.add(Conv2D(32, (7, 7), padding='same',
                 input_shape=(176, 200, 1),
                 activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same',
                 activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), padding='same',
                 activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(14, activation='softmax'))

# model.summary()

learning_rate = 0.0001
opt = keras.optimizers.adam(lr=learning_rate, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

tensorboard = TensorBoard(log_dir="logs/STAGE2-{}-{}".format(int(time.time()), learning_rate))

train_data_dir = "train_data"

# model = keras.models.load_model('./models/BasicCNN-5000-epochs-0.001-LR-STAGE2')

def check_data(choices):
    total_data = 0

    lengths = []
    for choice in choices:
        # print("length of {} is: {}".format(choice, len(choices[choice])))
        total_data += len(choices[choice])
        lengths.append(len(choices[choice]))

    # print("total data length now is: ", total_data)
    return lengths

# if you want to load in a previously trained model
# that you want to further train:
# keras.models.load_model(filepath)

hm_epochs = 10
for i in range(hm_epochs):
    current = 0
    increment = 200
    not_maximum = True
    all_files = os.listdir(train_data_dir)
    maximum = len(all_files)
    random.shuffle(all_files)

    while not_maximum:
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

            for file in all_files[current:current+increment]:
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
            # print("train data length: ", len(train_data))

            test_size = 50
            batch_size = 64

            x_train = np.array([i[1] for i in train_data]).reshape(-1, 176, 200, 1)
            y_train = np.array([i[0] for i in train_data])

            x_test = np.load('out_of_sample/x_oos.npy')
            y_test = np.load('out_of_sample/y_oos.npy')

            model.fit(
                x_train, y_train,
                batch_size=batch_size,
                validation_data=(x_test, y_test),
                shuffle=True,
                verbose=1, 
                callbacks=[tensorboard]
            )
            model.save("./models/BasicCNN-5000-epochs-0.001-LR-STAGE2")

        except Exception as e:
            print(str(e))

        current += increment
        if current > maximum:
            not_maximum = False
