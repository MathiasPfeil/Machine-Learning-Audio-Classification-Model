from pydub import AudioSegment
import math

t2 = 5000

newAudio = AudioSegment.from_wav("./dataset/full_audio/pick_session/pick_insert.wav")
audio_len = math.floor( math.floor( len(newAudio) / 1000 ) / 5 ) - 1

for i in range(audio_len):
    newAudio = AudioSegment.from_wav("./dataset/full_audio/pick_session/pick_insert.wav")
    t1 = t2
    t2 = t2 + 5000
    newAudio = newAudio[t1:t2]
    newAudio.export('./dataset/segmented_audio/pick/pick' + str(i) + '.wav', format="wav")