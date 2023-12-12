import subprocess
import os
import sys
import librosa
import scipy.signal.windows
import soundfile as sf
import numpy as np
from io import BytesIO
from PIL import Image
from scipy.io import wavfile
import io
import matplotlib.pyplot as plt
from PIL import Image
import requests
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as transforms
import torch
from meldataset import mel_spectrogram
from torchaudio.functional import resample

window_size = 4.08
sample_rate = 16000
n_fft = 1024
win_len = 1024
hop_len=256
n_mels = 128
fmin = 0.0
eps = 0.1
max_wav_value=32768.0
playback_speed = 1
fmax = 8000

with open("example.wav", "wb") as file:
  response = requests.get("https://drive.google.com/uc?export=preview&id=1Y3KuPAhB5VcsmIaokBVKu3LUEZOfhSu8")
  file.write(response.content)

# logmel = LogMelSpectrogram()

audio_fn = 'sample_1.wav'
waveform1, sample_rate = librosa.load(audio_fn, sr=sample_rate)

sr, waveform = wavfile.read(audio_fn, mmap=True)
waveform = waveform.astype('float32')
waveform /= max_wav_value

st = float(60 * 0 + 0.0)
start_idx = int(sr * st)
end_idx = start_idx + int(sr * window_size) * playback_speed
waveform = waveform[start_idx:end_idx]

waveform = torch.Tensor(waveform)

librosa_melspec = librosa.feature.melspectrogram(
    waveform.numpy(),
    sr=sample_rate,
    n_fft=n_fft,
    hop_length=hop_len,
    win_length=win_len,
    center=True,
    pad_mode="reflect",
    power=2.0,
    n_mels=n_mels,
)

torch_melspec = mel_spectrogram(waveform[None,:], n_fft, n_mels,
                sample_rate, hop_len, n_fft, fmin, fmax,
                center=True)

torch_melspec = torch_melspec.squeeze(0)

mse = ((torch_melspec - librosa_melspec) ** 2).mean()


import pdb; pdb.set_trace()

# mel spectrogram extraction using librosa and torch audio.
mel_lobrosa = librosa.feature.melspectrogram(y=y, sr=sr, **params)
mel_torch = logmel(ty).numpy()

diff = np.mean((mel_lobrosa - mel_torch) ** 2)

import torch
import numpy as np

# Load checkpoint
hifigan = torch.hub.load("bshall/hifigan:main", "hifigan_hubert_soft").cuda()
# Load mel-spectrogram

mel = torch.from_numpy(log_mel).unsqueeze(0).cuda()
wav, sr = hifigan.generate(mel.cuda())

wavfile.write("original.wav", sample_rate, y)
wavfile.write("decoded.wav", sample_rate, wav.reshape(-1).cpu().numpy())

def plot_spectrogram(log_mel, eps=0.1, ylabel='freq_bin', aspect='auto', xmax=None, to_db=True):
    fig, axs = plt.subplots(1, 1)
    spec = np.exp(log_mel + np.log(eps)) - eps
    if to_db:
        spec = librosa.power_to_db(spec, ref=np.max)
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    im = axs.imshow(spec, origin='lower', aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    fig.tight_layout(pad=0)
    fig.canvas.draw()

    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return data

img = plot_spectrogram(log_mel, eps)

Image.fromarray(img).save('mel_spectrogram.png')