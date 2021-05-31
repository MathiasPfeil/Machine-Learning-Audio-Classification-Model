# Machine Learning Audio Classification Model

This is the audio classification model used in my "Lock Picking Detection With Audio Classification - Machine Learning" video on YouTube (link below). Included in this repo is also the code required to activity listen to audio and classify it in real time.

[![Lock picking classification](http://img.youtube.com/vi/s5ePte2AE-g/0.jpg)](https://www.youtube.com/watch?v=s5ePte2AE-g "Lock picking detection using audio classification")

## How to use

To begin, make sure to install the required libraries by running `pip install -r requirements.txt`.

### Listening

To listen for and attempt to classify audio in real time, you must run the `listen.py` file found within the listening directory. While the code is crude at best, it will function by listening for five seconds, saving that audio to a wav file, feed the audio in that wav file through the pretrained model to get a prediction, then print that prediction to the console. It goes without saying, I hope, that a microphone to record the audio is also required, and that it be discoverable on your machine. The code responsible for steaming this audio is found in the `listen.py` file mentioned above.

If an event occurs, such as a key or pick entering the lock, we will save the audio clip of the event taking place, and log the type of event and time it happened to the log.txt file found within the listening directory.

Audio which is too quiet will be considered "static". If you are unable to get anything beside "static", make sure that your microphone is picking up the audio, and that the volume is up.

### Training

In order to train on this model without modifying lots of code, it is best if you feed your audio through the `standardize_audio.py` file in the main directory. This is because the model requires a certain shape for the data, and that shape can be produced for you by `standardize_audio.py`. This file works by taking a large .wav file with mostly uninterrupted audio of the sound you are attempting to classify, then breaks it down into 5 second segments of audio. It then saves those 5 second segments as individual audio clips to the output path you specify. The hitch being that your audio must be continuous for some time in order for this to function properly. To create your dataset like this, place your large .wav files in the dataset/full_audio directory. Then open `standardize_audio.py` and change the paths to point at your new audio file. You can also change to output path. Once this is completed, you can run `python standardize_audio.py`.

Once your dataset is built, edit the `set.py` file to import audio from the output path you set in the `standardize_audio.py` file. Once this step is completed, run `python set.py`. This will further standardize the data and create an X.pickle file for your data, and a y.pickle file for your labels.

To now train on this data, run `python run.py` to begin training.

## Notes

This repo does not contain the dataset used to train the model found in my video. The audio files simply exceed the size limit for Github repos. I will include the pretrained model used in the video mentioned above, however.

Because this is more of a tutorial repo, I'm placing some text files in certain empty folders in order to maintain the directory structure. These can be deleted as you see fit. These temp files will be titled "maintain_dir_struct.txt".

## Requirements

* Python 3.7.3
* Tensorflow 2.0.0-beta1
* numpy 1.16.4
* pickle 4.0
* scipy 1.2.0
* pydub
* tqdm 4.31.1
* matplotlib 3.0.3
* pyaudio 0.2.11