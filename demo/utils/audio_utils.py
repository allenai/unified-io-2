import random
import string
import subprocess
import time
from typing import Union, Optional

import gradio as gr
import librosa
import numpy as np
import resampy
import scipy

__all__ = ["load_audio", "extract_spectrograms_from_audio"]


AUDIO_SEGMENT_LENGTH = 4.08
AUDIO_SPECTRUM_LENGTH = 4.08
AUDIO_SAMPLING_RATE = 16000

AUDIOSET_MEAN = -5.0945
AUDIOSET_STD = 3.8312


def load_audio(
    path: str,
    audio_segment_length=AUDIO_SEGMENT_LENGTH,
    spectrogram_length=AUDIO_SEGMENT_LENGTH,
    audio_length: Optional[float] = None,
    max_audio_length:  Optional[float]=None,
    **kwargs
):
    if audio_length is None:
        audio_length = get_audio_length(path)
    if audio_length is None:
        gr.Warning(f"No Audio: {path}")
        return None
    if max_audio_length is not None:
        clip_end_time = min(audio_length, max_audio_length)
        if clip_end_time < audio_length:
            gr.Warning(
                f"Use the input audio length of {clip_end_time} (original {audio_length}) seconds."
            )
        audio_length = clip_end_time
    return extract_spectrograms_from_audio(
        audio_file=path,
        audio_length=audio_length,
        audio_segment_length=audio_segment_length,
        spectrogram_length=spectrogram_length,
        **kwargs
    )


def get_audio_length(audio_path):
    # this gets just the video stream length (in the case audio stream is longer)
    # E.g. k700-2020/train/watering plants/af3epdZsrTc_000178_000188.mp4
    # if audio is shorter than video stream, just pad that
    # "-select_streams v:0" gets the video stream, '-select_streams a:0" is audio stream
    proc = subprocess.Popen(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            audio_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # this gets overall video length (combo of both video and audio stream)
    # proc = subprocess.Popen(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
    #        '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
    #         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    # ffprobe -v error -show_entries stream=duration -of default=noprint_wrappers=1:nokey=1 -sexagesimal  ~/Downloads/kg2U6Rv0tkY_000001_000011.mp4

    out, _ = proc.communicate()
    duration = out.decode("utf-8")

    try:
        duration = float(out.strip())
    except:
        print(f"Invalid duration for {audio_path}: {duration}")
        duration = None

    return duration


BUFFER_FROM_END = 0.1  # was .05 earlier but changed to 0.1 to avoid drops in msvd; found by trial and error with ffmpeg


def get_num_segments(audio_length, audio_segment_length):
    num_segments = int(audio_length // audio_segment_length)

    # allows extra frame only if the midpoint is an available to extract video frames
    if (audio_length % audio_segment_length) - BUFFER_FROM_END > (
        audio_segment_length / 2.0
    ):
        num_segments += 1

    if num_segments == 0 and audio_length > 0:
        num_segments = 1

    return num_segments


def read_audio_file(audio_file: str):
    if audio_file.startswith("http"):
        filename = "".join(random.choices(string.ascii_lowercase, k=8)) + ".wav"
        cmd = f"wget -O {filename} {audio_file}"
        _ = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True
        )
        gr.Info("Waiting for audio download to finish!")
        time.sleep(5)
        audio_file = filename

    waveform, sr = librosa.core.load(audio_file, sr=AUDIO_SAMPLING_RATE)
    waveform = waveform.astype("float32")
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)
    if sr != AUDIO_SAMPLING_RATE:
        waveform = resampy.resample(waveform, sr, AUDIO_SAMPLING_RATE)
    return waveform


def extract_spectrograms_from_audio(
    audio_file: Union[str, np.ndarray],
    audio_segment_length: float = AUDIO_SEGMENT_LENGTH,
    spectrogram_length: float = AUDIO_SPECTRUM_LENGTH,
    audio_length=None,
    sampling_rate: int = AUDIO_SAMPLING_RATE,
):
    # read in the audio file
    if isinstance(audio_file, str):
        waveform = read_audio_file(audio_file)
    else:
        assert audio_file.ndim == 1
        # cached waveform
        waveform = audio_file
    if waveform is None:
        print("NO AUDIO FROM WAVEFILE!")
        return None

    if audio_length is None:
        # get actual audio length
        audio_length = get_audio_length(audio_file)
    if audio_length is None:
        print(f"Couldn't get audio length for {audio_file}")
        return None

    num_segments = get_num_segments(audio_length, audio_segment_length)
    boundaries = np.linspace(
        0, num_segments * audio_segment_length, num_segments + 1
    ).tolist()

    # Pad to max time just in case, crop if longer
    max_samples = int(sampling_rate * num_segments * audio_segment_length)
    if waveform.size < max_samples:
        waveform = np.concatenate(
            [waveform, np.zeros(max_samples - waveform.size, dtype=np.float32)], 0
        )
    waveform = waveform[:max_samples]

    # split waveform into segments
    spectrograms = []
    for i in range(num_segments):
        if audio_segment_length <= spectrogram_length:
            ts_start = int(boundaries[i] * sampling_rate)
            ts_end = int(boundaries[i + 1] * sampling_rate)
            waveform_segment = waveform[ts_start:ts_end]
            num_pad = int(sampling_rate * spectrogram_length) - (ts_end - ts_start)
            if num_pad > 0:
                waveform_segment = np.concatenate(
                    [
                        np.zeros(num_pad // 2, dtype=np.float32),
                        waveform_segment,
                        np.zeros(num_pad - num_pad // 2, dtype=np.float32),
                    ],
                    0,
                )
            waveform_segment = waveform_segment[
                : int(sampling_rate * spectrogram_length)
            ]
        else:
            ts_start = int(boundaries[i] * sampling_rate)
            ts_end = int(boundaries[i + 1] * sampling_rate)
            ts_mid = (ts_start + ts_end) / 2
            start = int(ts_mid - sampling_rate * spectrogram_length / 2)
            end = start + int(sampling_rate * spectrogram_length)
            waveform_segment = waveform[start:end]
        # Create spectrogram from waveform
        try:
            spectrogram = make_spectrogram(
                waveform_segment, sampling_rate, n_fft=1024, hop_length=256
            )  # shape (128, 256)
        except Exception as exc:
            print(f"Couldn't make spectrogram, {exc}")
            return None
        spectrograms.append(spectrogram)

    if len(spectrograms) == 0:
        assert num_segments == 0
        print("Couldn't make spectrograms: num_segments is 0")
        return None

    # (N,128,256) is (# of segments, # of mel bands in spectrogram, # of hops in spectrogram)
    spectrograms = np.stack(spectrograms).astype(np.float32)
    assert spectrograms.shape[1:] == (128, 256)

    # if spectrograms.shape[1:] != (128, 256):
    #     print(
    #         f"Non-standard spectrogram shape produced! Should be (N,128,256) but is {spectrograms.shape}"
    #     )
    #     return None

    return spectrograms


def make_spectrogram(
    waveform,
    sample_rate=16000,
    n_fft=1024,
    hop_length=256,
):
    """
    Make spectrogram using librosa of shape (mel_bands, frames)

    :param waveform: wave file
    :param sample_rate: Sample rate

    params for the librosa function
    - n_fft: number of samples in window for DFT, at default sampling rate of 22050, this is 69.65ms
    - hop_length: number of samples from the start of one window to start of the next
    - n_mels: number of bins on the mel scale

    where time frames = math.floor(sample_rate * duration / hop_length) + 1
                  376 = (22050 * 10s / 588) + 1

    Manually selected by Sangho parameters had best sound quality during tests
    https://librosa.org/doc/main/generated/librosa.stft.html - Faster if n_fft is a power of two

    :return:
    """
    params = {
        "n_fft": n_fft,  # Manually selected by Sangho
        "hop_length": hop_length,  # Manually selected by Sangho
        "window": scipy.signal.windows.hann,  # Default
        "n_mels": 128,  # Manually selected by Sanho
        "fmin": 0.0,  # Manually selected by Sangho
        "fmax": sample_rate / 2.0,  # Default 22050
        "center": True,
        "pad_mode": "reflect",
    }  # Spectrogram therefore has shape (64, 626) for 10s
    mel = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, **params)
    # log_mel = np.log(mel + eps) - np.log(eps)
    # return log_mel
    return mel
