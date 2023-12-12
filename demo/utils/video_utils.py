import os.path
import random
import string
import subprocess
import time

import gradio as gr

from create_data.utils import (
    get_video_length,
    create_audio_from_video,
    extract_frames_from_video,
    BUFFER_FROM_END,
)
from demo.utils.audio_utils import extract_spectrograms_from_audio

__all__ = ["load_video"]


def extract_frames_and_spectrograms_from_video(
    video_file,
    audio_dir,
    video_length=None,
    video_segment_length=None,
    audio_segment_length=None,
    times=None,
    clip_start_time=0,
    clip_end_time=None,
    num_frames=None,
    target_size=(256, 256),
    *,
    use_audio,
):
    if times is None:
        # get actual video length
        if video_length is None:
            video_length = get_video_length(video_file)
        if video_length is None:
            print(f"Couldn't get video length for {video_file}")
            return None, None

        if video_segment_length is None:
            video_segment_length = video_length / num_frames
        if video_length < (video_segment_length / 2.0) - BUFFER_FROM_END:
            print(
                f"Video is too short ({video_length}s is less than half the segment length of {video_segment_length}s segments"
            )
            return None, None
    else:
        # don't need this if times is given
        video_length = None

    # extract image frames
    # t0 = perf_counter()
    frames, boundaries = extract_frames_from_video(
        video_file,
        video_length,
        video_segment_length,
        times=times,
        clip_start_time=clip_start_time,
        clip_end_time=clip_end_time,
        num_frames=num_frames,
        multiprocess=False,
        resize=True,
        target_size=target_size,
    )
    # print(f"Load video in {perf_counter() - t0} seconds in total")

    spectrograms = None
    if use_audio:
        # expects the audio file to be created already (since it takes some time)
        audio_file = create_audio_from_video(video_file, audio_dir, force=True)
        if os.path.exists(audio_file):  # in case video w/o audio
            # extract audio segments
            spectrograms = extract_spectrograms_from_audio(
                audio_file,
                audio_length=clip_end_time,
                audio_segment_length=audio_segment_length,
                spectrogram_length=audio_segment_length,
            )

    return frames, spectrograms


def load_video(
    path: str,
    max_frames: int = 4,
    audio_segment_length: float = 4.08,
    target_size: tuple = (256, 256),
    *,
    use_audio: bool,
):
    """max frames could be max image history length + (1 if no image input else 0), similar for audio"""
    if path.startswith("http"):
        filetype = path.split(".")[-1]
        filename = "".join(random.choices(string.ascii_lowercase, k=8)) + "." + filetype
        cmd = f"wget -O {filename} {path}"
        print(cmd)
        _ = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True
        )
        gr.Info("Waiting for video download to finish!")
        time.sleep(10)
        path = filename

    assert os.path.exists(path), path
    video_length = get_video_length(path)
    clip_end_time, video_seg_length = None, None
    if video_length is not None:
        # accommodate the audio segment length
        max_length = max_frames * audio_segment_length
        clip_end_time = min(max_length, video_length)
        if clip_end_time < video_length:
            gr.Warning(
                f"Use the input video length of {clip_end_time} (original {video_length}) seconds."
            )
        # second per frame, not necessary corresponding with the audio in demo
        video_seg_length = clip_end_time / max_frames

    frames, spectrograms = extract_frames_and_spectrograms_from_video(
        path,
        audio_dir=None,
        video_length=video_length,
        video_segment_length=video_seg_length,
        audio_segment_length=audio_segment_length,
        clip_start_time=0,
        clip_end_time=clip_end_time,
        num_frames=None,
        use_audio=use_audio,
        target_size=target_size,
    )
    return frames, spectrograms
