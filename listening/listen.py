import pyaudio
import wave
import os
import tensorflow as tf
from scipy.io.wavfile import read
from scipy.io.wavfile import write
import time
import datetime
import numpy as np
from pydub import AudioSegment


CATEGORIES = ["key", "pick"]
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
model = tf.keras.models.load_model('./../key_or_pick.h5')


CHUNK = 4410
FORMAT = pyaudio.get_format_from_width(2, unsigned=False)
CHANNELS = 1
RATE = 88200
RECORD_SECONDS = 6
WAVE_OUTPUT_FILENAME = "output.wav"


def is_static(arr):
    return max(arr.flatten()) < 5000


def log_event(event, event_audio, wr):
    now = datetime.datetime.now()
    save_time = now.strftime('%Y-%m-%d-%H-%M-%S')
    f = open("log.txt", "a")
    f.write(event + ' detected - ' + save_time + "\n")
    f.close()

    write('./events/' + (event + ' - ' + save_time) + '.wav', wr, event_audio)


while True:
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("..")
    frames = []

    for i in range(0, 100):
        data = stream.read(CHUNK)
        frames.append(data)
    print(".")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    wav_rate, input_data = read('./output.wav')
    audio = input_data

    X = np.array(audio).reshape(-1, 1, 220500, 2)

    if(is_static(X[0])):
        print('Static')
    else:
        prediction = model.predict(X)[0][0].item()
        print("{0:.2f}".format(prediction) + ' - ' + CATEGORIES[round(prediction)].replace('_',' '))
        event_type = CATEGORIES[round(prediction)].replace('_',' ')
        log_event(event_type, input_data, wav_rate)
