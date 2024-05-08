import librosa
from panns_inference import AudioTagging, SoundEventDetection, labels
from models import Cnn14_16k
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import sounddevice
import time

checkpoint_path='{}/panns_data/Cnn14_16k_mAP=0.438_.pth'.format(str(Path.home()))
model_sr = 16000

model = Cnn14_16k(sample_rate=model_sr, window_size=512, 
                hop_size=160, mel_bins=64, fmin=50, fmax=8000, 
                classes_num=len(labels))

audio_path = 'footstep-short.wav'
(audio, sr) = librosa.core.load(audio_path, sr=44100, mono=True)
audio = librosa.resample(audio, orig_sr=sr, target_sr=model_sr)
sounddevice.play(audio, model_sr)
time.sleep(2) 
audio = audio[None, :]  # (batch_size, segment_samples)

print('------ Audio tagging ------')
at = AudioTagging(model=model, checkpoint_path=checkpoint_path, device='cuda')
(clipwise_output, embedding1) = at.inference(audio)
print(embedding1.shape)

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
