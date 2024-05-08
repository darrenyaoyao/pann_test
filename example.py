import librosa
from panns_inference import AudioTagging, SoundEventDetection, labels
from models import Cnn14_16k
from sklearn.metrics.pairwise import cosine_similarity

audio_path = 'footstep-short.wav'
(audio, sr) = librosa.core.load(audio_path, sr=44100, mono=True)
print(audio.shape)
print(audio.dtype)
audio = librosa.resample(audio, orig_sr=sr, target_sr=32000)
print(audio.shape)
audio = audio[None, :]  # (batch_size, segment_samples)

print('------ Audio tagging ------')
at = AudioTagging(checkpoint_path=None, device='cuda')
(clipwise_output, embedding1) = at.inference(audio)
print(embedding1.shape)

audio_path = 'footstep.wav'
(audio, sr) = librosa.core.load(audio_path, sr=44100, mono=True)
audio = librosa.resample(audio, orig_sr=sr, target_sr=32000)
audio = audio[None, :]  # (batch_size, segment_samples)

print('------ Audio tagging ------')
at = AudioTagging(checkpoint_path=None, device='cuda')
(clipwise_output, embedding2) = at.inference(audio)
print(embedding2.shape)


print(cosine_similarity(embedding1, embedding2))

audio_path = 'highheel.wav'
(audio, sr) = librosa.core.load(audio_path, sr=44100, mono=True)
audio = librosa.resample(audio, orig_sr=sr, target_sr=32000)
audio = audio[None, :]  # (batch_size, segment_samples)

print('------ Audio tagging ------')
at = AudioTagging(checkpoint_path=None, device='cuda')
(clipwise_output, embedding2) = at.inference(audio)
print(embedding2.shape)


print(cosine_similarity(embedding1, embedding2))

audio_path = 'door-short.wav'
(audio, sr) = librosa.core.load(audio_path, sr=44100, mono=True)
audio = librosa.resample(audio, orig_sr=sr, target_sr=32000)
audio = audio[None, :]  # (batch_size, segment_samples)

print('------ Audio tagging ------')
at = AudioTagging(checkpoint_path=None, device='cuda')
(clipwise_output, embedding2) = at.inference(audio)
print(embedding2.shape)


print(cosine_similarity(embedding1, embedding2))
