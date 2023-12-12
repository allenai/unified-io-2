import itertools
from absl import logging
import multiprocessing
import subprocess
from os import path
from pathlib import Path
import numpy as np
import skimage
import skvideo.io
from skimage.transform import resize


def get_video_length(video_path):
  """this gets just the video stream length (in the case audio stream is longer)"""
  # E.g. k700-2020/train/watering plants/af3epdZsrTc_000178_000188.mp4
  # if audio is shorter than video stream, just pad that
  # "-select_streams v:0" gets the video stream, '-select_streams a:0" is audio stream
  proc = subprocess.Popen(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=duration',
                           '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

  out, _ = proc.communicate()
  duration = out.decode('utf-8')

  try:
    duration = float(out.strip())
  except ValueError:
    logging.warning(f"Invalid duration for {video_path}: {duration}")
    duration = None

  return duration


def create_audio_from_video(video_file:str, audio_dir: None, audio_file_timeout=-1, sampling_rate:int=16000, force:bool=False):
  """Create .wav file from video"""

  if audio_dir is not None:
    audio_file = path.join(audio_dir, Path(video_file).stem + ".wav")
  else:
    audio_file = path.splitext(video_file)[0] + ".wav"

  if not path.isfile(audio_file) or force:
    ffmpeg_process = subprocess.Popen(
      ['ffmpeg', '-y', '-i', str(video_file), '-ac', '1', '-ar', str(sampling_rate), audio_file],
      stdout=-1, stderr=-1, text=True)

    if audio_file_timeout == -1:
      # wait however long it takes to create the audio file
      ffmpeg_process.wait()
    else:
      try:
        ffmpeg_process.communicate(None, timeout=audio_file_timeout)
      except subprocess.TimeoutExpired:
        # if the audio file hasn't been created yet, abandon hope
        logging.warning(f"Couldn't create .wav from {video_file} in timeout of {audio_file_timeout}.")
        ffmpeg_process.kill()
        return None
    ffmpeg_process.kill()

  return audio_file


BUFFER_FROM_END = 0.1


def get_num_segments(video_length, video_segment_length):
  num_segments = int(video_length // video_segment_length)

  # allows extra frame only if the midpoint is an available to extract video frames
  if (video_length % video_segment_length) - BUFFER_FROM_END > (video_segment_length / 2.0):
    num_segments += 1

  return num_segments


def resize_image_by_shorter_side(im, target_size=512):
  """ Resizes the image such that the shorter size matches the target.
  If it is already equal or smaller, the image is returned unchanged.

  Args:
      im (numpy.ndarray): image frame or list of image frames of shape
      shorter_side_target_size (int, optional): Size. Defaults to 512.

  Returns:
      (numpy.ndarray): resized image
  """
  w, h = im.shape[:2]
  min_size = min(w, h)

  if min_size > target_size:
    new_w = int(w / min_size * target_size)
    new_h = int(h / min_size * target_size)
    # if preserve_range not flagged, then converted to float in {0,1}
    im = resize(im, (new_w, new_h), anti_aliasing=True, preserve_range=True)

  return im


def extract_single_frame_from_video(video_file, t, verbosity=0):
  timecode = '{:.3f}'.format(t)
  try:
    reader = skvideo.io.FFmpegReader(video_file,
                                     inputdict={'-ss': timecode, '-threads': '1'},
                                     outputdict={'-r': '1', '-q:v': '2', '-pix_fmt': 'rgb24', '-frames:v': '1'},
                                     verbosity=verbosity,
                                     )
  except ValueError as err:
    print(f"Error on loading {video_file}")
    print(err.args)
    return None

  try:
    frame = next(iter(reader.nextFrame()))
  except StopIteration:
    logging.warning(f"Error on getting frame at time {timecode}s from {video_file}")
    return None

  return frame


def extract_frames_from_video(video_path,
                              video_length,
                              video_segment_length,
                              times=None,
                              clip_start_time=0,
                              clip_end_time=None,
                              num_frames=None,
                              resize=True,
                              target_size=512,
                              multiprocess=False):
  """
  Control frame times:
  - automatically compute from below (default) OR
  - manually set

  Control number of frames or sampling duration:
  - specify number of frames (num_frames) OR
  - specify duration between segments (video_segment_length)

  Control where to sample from:
  - between [0,video_length] (default) OR
  - between [clip_start_time,clip_end_time]

  video_length may be provided. If set to None, video_length will be computed
  from video_path.
  """
  if times is None:
    # automatically calculate what times to extract frames for
    if video_length is None:
      video_length = get_video_length(video_path)

    if clip_end_time is not None:
      clip_duration = clip_end_time - clip_start_time
      if clip_duration <= video_length:
        video_length = clip_duration

    # one and only one of video_segment_length and num_frames should be None
    assert video_segment_length is not None or num_frames is not None
    assert video_segment_length is None or num_frames is None

    if num_frames is None:
      # allows extra frame only if for >=50% of the segment video is available
      num_segments = get_num_segments(video_length, video_segment_length)
    else:
      num_segments = num_frames

    # frames are located at the midpoint of a segment
    boundaries = np.linspace(clip_start_time, clip_end_time, num_segments + 1).tolist()
    extract_times = [(boundaries[i] + boundaries[i+1]) / 2.0 for i in range(num_segments)]
  else:
    extract_times = times
    boundaries = None

  # extract the frames
  if multiprocess:
    pool = multiprocessing.Pool()
    frames = pool.starmap(extract_single_frame_from_video,
                          zip(itertools.repeat(video_path), extract_times))
  else:
    # TODO Can we get all frames with one video read
    frames = [extract_single_frame_from_video(video_path, time) for time in extract_times]

  # check to see if any extraction failed
  if any([x is None for x in frames]) or frames is None or len(frames) == 0:
    logging.warning(f"Failed to extract frames from {video_path}")
    return None, None

  # resize the frames to have shorter side of size 512
  if resize:
    if isinstance(target_size, int):
      frames = [resize_image_by_shorter_side(im, target_size=target_size) for im in frames]
    else:
      assert len(target_size) == 2, target_size   # type: ignore
      frames = [
        skimage.transform.resize(
          im, target_size, anti_aliasing=True, preserve_range=True
        )
        for im in frames
      ]

  return np.stack(frames).astype(np.uint8), boundaries