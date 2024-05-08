import librosa
from panns_inference import AudioTagging, SoundEventDetection, labels
from models import Cnn14_16k
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import sounddevice
import time
import numpy as np
import matplotlib.pyplot as plt

checkpoint_path='{}/panns_data/Cnn14_16k_mAP=0.438_.pth'.format(str(Path.home()))
model_sr = 16000

model = Cnn14_16k(sample_rate=model_sr, window_size=512, 
                hop_size=160, mel_bins=64, fmin=50, fmax=8000, 
                classes_num=len(labels))

def filetonumpy(path):
    mean = 1603
    f = open(path, "r")
    testdata = f.read().split()
    tdata = []
    for i in range(int(len(testdata)/2)):
        tdata.append(int(testdata[2*i+1]+testdata[2*i], 16) - mean)
    audio = np.array(tdata, dtype=np.float32) 
    # Find the minimum and maximum values
    min_val = -2000
    max_val = 2000

    # Scale the array to the range [-1, 1]
    scaled_arr = -1 + 2 * ((audio - min_val) / (max_val - min_val))
    return scaled_arr


audio = filetonumpy("audiodata/empty_stm32_16k_2s_1.txt")
audio = audio[None, :]  # (batch_size, segment_samples)

print('------ Audio tagging ------')
at = AudioTagging(model=model, checkpoint_path=checkpoint_path, device='cuda')
(clipwise_output, embedding1) = at.inference(audio)
print(embedding1.shape)


# audio = filetonumpy("audiodata/footstep_stm32_16k_1s_2.txt")
# sounddevice.play(audio, model_sr)
# time.sleep(2) 
# sounddevice.stop()
# audio = audio[None, :]  # (batch_size, segment_samples)

# print('------ Audio tagging ------')
# at = AudioTagging(model=model, checkpoint_path=checkpoint_path, device='cuda')
# (clipwise_output, embedding2) = at.inference(audio)
# print(embedding2.shape)


# print(cosine_similarity(embedding1, embedding2))

# plt.plot(audio[0])
# plt.show()

# audio = filetonumpy("audiodata/footstep_stm32_16k_1s_3.txt")
# sounddevice.play(audio, model_sr)
# time.sleep(2) 
# sounddevice.stop()
# audio = audio[None, :]  # (batch_size, segment_samples)

# print('------ Audio tagging ------')
# at = AudioTagging(model=model, checkpoint_path=checkpoint_path, device='cuda')
# (clipwise_output, embedding2) = at.inference(audio)
# print(embedding2.shape)


# print(cosine_similarity(embedding1, embedding2))

# plt.plot(audio[0])
# plt.show()


# audio = filetonumpy("audiodata/highheel_stm32_16k_2s_1.txt")
# sounddevice.play(audio, model_sr)
# time.sleep(2) 
# sounddevice.stop()
# audio = audio[None, :]  # (batch_size, segment_samples)

# print('------ Audio tagging ------')
# at = AudioTagging(model=model, checkpoint_path=checkpoint_path, device='cuda')
# (clipwise_output, embedding2) = at.inference(audio)
# print(embedding2.shape)


# print(cosine_similarity(embedding1, embedding2))

# plt.plot(audio[0])
# plt.show()

audio = filetonumpy("audiodata/empty_stm32_16k_2s_1.txt")
audio = audio[None, :]  # (batch_size, segment_samples)

print('------ Audio tagging ------')
at = AudioTagging(model=model, checkpoint_path=checkpoint_path, device='cuda')
(clipwise_output, embedding2) = at.inference(audio)
print(embedding2.shape)


print(cosine_similarity(embedding1, embedding2))


audio = filetonumpy("audiodata/cat_stm32_16k_2s_1.txt")
audio = audio[None, :]  # (batch_size, segment_samples)

print('------ Audio tagging ------')
at = AudioTagging(model=model, checkpoint_path=checkpoint_path, device='cuda')
(clipwise_output, embedding2) = at.inference(audio)
print(embedding2.shape)


print(cosine_similarity(embedding1, embedding2))

audio_path = 'footstep-short.wav'
(audio, sr) = librosa.core.load(audio_path, sr=44100, mono=True)
audio = librosa.resample(audio, orig_sr=sr, target_sr=model_sr)
audio = audio[None, :]  # (batch_size, segment_samples)

print('------ Audio tagging ------')
at = AudioTagging(model=model, checkpoint_path=checkpoint_path, device='cuda')
(clipwise_output, embedding2) = at.inference(audio)
print(embedding2.shape)


print(cosine_similarity(embedding1, embedding2))