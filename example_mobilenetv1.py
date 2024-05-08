import librosa
from panns_inference import AudioTagging, SoundEventDetection, labels
from models import MobileNetV1
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import numpy as np
import sounddevice
import time
import os

def print_audio_tagging_result(clipwise_output):
    """Visualization of audio tagging result.

    Args:
      clipwise_output: (classes_num,)
    """
    sorted_indexes = np.argsort(clipwise_output)[::-1]

    # Print audio tagging top probabilities
    for k in range(10):
        print('{}: {:.3f}'.format(np.array(labels)[sorted_indexes[k]],
            clipwise_output[sorted_indexes[k]]))

checkpoint_path='{}/panns_data/MobileNetV1_mAP=0.389.pth'.format(str(Path.home()))
model_sr = 32000

model = MobileNetV1(sample_rate=model_sr, window_size=1024,
                hop_size=320, mel_bins=64, fmin=50, fmax=12000,
                classes_num=len(labels))

audio_path = 'dog.wav'
(audio, sr) = librosa.core.load(audio_path, sr=44100, mono=True)
audio = librosa.resample(audio, orig_sr=sr, target_sr=model_sr)
sounddevice.play(audio, model_sr)
time.sleep(2)
audio = audio[None, :]  # (batch_size, segment_samples)

print('------ Audio tagging ------')
at = AudioTagging(model=model, checkpoint_path=checkpoint_path)
(clipwise_output, embedding1, spectrogram) = at.inference(audio)
print(embedding1.shape)

print_audio_tagging_result(clipwise_output[0])

audio_path = 'footstep.wav'
(audio, sr) = librosa.core.load(audio_path, sr=44100, mono=True)
audio = librosa.resample(audio, orig_sr=sr, target_sr=model_sr)
sounddevice.play(audio, model_sr)
time.sleep(2)
audio = audio[None, :]  # (batch_size, segment_samples)

print('------ Audio tagging ------')
at = AudioTagging(model=model, checkpoint_path=checkpoint_path, device='cuda')
(clipwise_output, embedding2) = at.inference(audio)
print(embedding2.shape)


print(cosine_similarity(embedding1, embedding2))

audio_path = 'highheel.wav'
(audio, sr) = librosa.core.load(audio_path, sr=44100, mono=True)
audio = librosa.resample(audio, orig_sr=sr, target_sr=model_sr)
sounddevice.play(audio, model_sr)
time.sleep(2)
audio = audio[None, :]  # (batch_size, segment_samples)

print('------ Audio tagging ------')
at = AudioTagging(model=model, checkpoint_path=checkpoint_path, device='cuda')
(clipwise_output, embedding2) = at.inference(audio)
print(embedding2.shape)


print(cosine_similarity(embedding1, embedding2))

audio_path = 'door-short.wav'
(audio, sr) = librosa.core.load(audio_path, sr=44100, mono=True)
audio = librosa.resample(audio, orig_sr=sr, target_sr=model_sr)
sounddevice.play(audio, model_sr)
time.sleep(2)
audio = audio[None, :]  # (batch_size, segment_samples)

print('------ Audio tagging ------')
at = AudioTagging(model=model, checkpoint_path=checkpoint_path, device='cuda')
(clipwise_output, embedding2) = at.inference(audio)
print(embedding2.shape)


print(cosine_similarity(embedding1, embedding2))
