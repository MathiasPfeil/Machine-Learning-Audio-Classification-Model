import numpy as np
import os
from tqdm import tqdm
import random
import pickle
from scipy.io.wavfile import read
from pydub import AudioSegment

DATADIR = "./dataset/segmented_audio"

CATEGORIES = ["key", "pick"]

training_data = []

def create_training_data():
    for category in CATEGORIES:

        path = os.path.join(DATADIR,category)
        class_num = CATEGORIES.index(category)

        for audio in tqdm(os.listdir(path)):
            input_data = read(os.path.join(path,audio))
            audio = input_data[1]
            training_data.append([audio,class_num])

create_training_data()

random.shuffle(training_data)

X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, 1, 220500, 2)

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()