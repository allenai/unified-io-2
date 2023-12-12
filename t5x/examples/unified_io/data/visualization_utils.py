import base64
import io
from typing import Any, Dict, Union, List, Optional

import numpy as np
import torch

from PIL import Image

import tensorflow as tf

from t5x.examples.unified_io import config
from t5x.examples.unified_io.data.data_utils import unnormalize_image

ImageData = Union[np.ndarray, torch.Tensor, tf.Tensor]


def build_embedding_image(image_data: ImageData, channel_first=False, fmt="jpeg"):
  """Turns an image as a string that can be used src in html images"""
  if not isinstance(image_data, torch.Tensor):
    image_data = tf.convert_to_tensor(image_data)

  assert fmt == "jpeg"
  assert not channel_first

  if image_data.dtype == tf.bool:
    # Boolean mask, convert to black/white
    if len(image_data.shape) == 2:
      image_data = tf.expand_dims(image_data, -1)
    else:
      assert image_data.shape[2] == 1
    black_constant = tf.reshape(tf.constant([0, 0, 0], dtype=tf.uint8), [1, 1, 3])
    white_constant = tf.reshape(tf.constant([255, 255, 255], dtype=tf.uint8), [1, 1, 3])
    image_data = tf.cast(image_data, tf.uint8)
    image_data = image_data*white_constant + (1-image_data)*black_constant

  elif image_data.dtype != tf.uint8:
    image_data = tf.image.convert_image_dtype(image_data, tf.uint8)

  data = tf.image.encode_jpeg(image_data)
  data = data.numpy()
  encoded_image = base64.b64encode(data)
  return f'data:image/{fmt};base64,{encoded_image.decode()}'


def build_embedding_gif(video_data, fmt="gif"):
  """Turns an image as a string that can be used src in html images"""
  assert fmt == "gif"

  if not isinstance(video_data, torch.Tensor):
    video_data = tf.convert_to_tensor(video_data)

  if video_data.dtype != tf.uint8:
    video_data = tf.image.convert_image_dtype(video_data, tf.uint8)

  video_data = video_data.numpy()
  img = Image.fromarray(video_data[0])
  data = io.BytesIO()
  img.save(data, format="gif", loop=0, save_all=True,
           append_images=[Image.fromarray(x) for x in video_data[1:]])
  data = data.getvalue()
  encoded_gif = base64.b64encode(data)
  return f'data:image/{fmt};base64,{encoded_gif.decode()}'


def build_video_grid(video):
  cells = []
  for i in range(video.shape[0]):
    image_frame = video[i]
    data = build_embedding_image(image_frame, fmt="jpeg")
    cells.append(f'<img src={data} class="d-block w-100">')
  # return "\n".join(f"<td>{x}</td>" for x in cells)
  images = "\n".join(f"<td>{x}</td>" for x in cells)
  return "<table><tr>\n"+ images + "</tr></table>\n"


CAROSUEL_ID = 0


def build_html_table(data: List[Dict[str, Union[str, ImageData]]], bootstrap=False):
  """Build the text of a HTML table"""
  columns = {}  # Collect any key that appears in the data, in order
  for row in data:
    for key in row:
      columns[key] = None

  html = ["<!DOCTYPE html>",'<html lang="en">']

  if bootstrap:
    html.append('<head>')
    html.append('<title>Visualization</title>')
    html.append('<meta charset="utf-8">')
    html.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
    html.append('<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.1/dist/css/bootstrap.min.css" rel="stylesheet">')
    html.append(' <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.1/dist/js/bootstrap.bundle.min.js"></script>')
    html.append('<style> \
          .carousel .carousel-item { \
              transition-duration: 0s; \
          } \
        </style>')
    html.append('</head>')

  # Body
  html.append("<body>")
  html.append("<table>")

  # Table Header
  html.append("<tr>")
  html.append(" ".join(f"<th>{c}</th>" for c in columns))
  html.append("</tr>")

  # Table Body
  for ex in data:
    cells = []
    for c in columns:
      val = ex.get(c)
      if val is None:
        cells.append("")
      elif isinstance(val, (float, int, str)):
        cells.append(val)
      elif len(val.shape) == 3 and val.shape[-1] == 3:
        # Assume an image
        data = build_embedding_image(val, fmt="jpeg")
        cells.append(f'<img src={data}></img>')
      elif len(val.shape) == 4 and val.shape[-1] == 3:
        # Video
        cells.append(build_video_grid(val))
        # Or build a GIF like this:
        # data = build_embedding_gif(val, fmt="gif")
        # cells.append(f'<img src={data} loop=infinite></img>')
      else:
        raise NotImplementedError(f"Data not understood for {val.shape}: {val}")
    html.append("<tr>")
    html.append("\n".join(f"<td>{x}</td>" for x in cells))
    html.append("</tr>")

  html.append("</table>")
  html.append("</body>")
  html.append("</html>")
  return "\n".join(html)


def process_image(input_image, mask, pos_ids=None,
                  input_size=config.IMAGE_INPUT_SIZE, input_d=config.IMAGE_INPUT_D):
  """Build an image that an be viewed from the post-processed UIO 2 inputs"""
  input_image, input_image_mask = process_batch_image(
    np.expand_dims(input_image, axis=0),
    np.expand_dims(mask, axis=0),
    input_size, input_d,
    pos_ids=np.expand_dims(pos_ids, axis=0),
  )
  input_image, input_image_mask = np.squeeze(input_image, 0), np.squeeze(input_image_mask, 0)
  return input_image, input_image_mask


def process_batch_image(input_image, mask, input_size, input_d, pos_ids=None):
  """Builds set of images that can be viewed from the post-processed UIO 2 inputs"""
  if len(input_image.shape) == 3:
    t = input_image.shape[0]
    dh, dw = input_d, input_d
    c = input_image.shape[-1] // (dw*dh)
    h = input_size[0] // dh
    w = input_size[1] // dw
    with_blanks = np.zeros((t, h*w, dh*dw*3))
    with_blanks[np.arange(t)[:, None], pos_ids] = input_image
    input_image = with_blanks
    input_image_mask = np.zeros((t, h*w), dtype=mask.dtype)
    input_image_mask[np.arange(t)[:, None], pos_ids] = mask
    input_image_mask = input_image_mask.reshape((t, h, w))
    input_image = input_image.reshape((t, h, w, dh ,dw, c))
    input_image = unnormalize_image(input_image).numpy()
    input_image = input_image*(input_image_mask.reshape((t, h, w))[:, :, :, None, None, None])
    input_image = input_image.transpose((0, 1, 3, 2, 4, 5))
    input_image = input_image.reshape((t, h*dh, w*dw, c))
  else:
    input_image_mask = mask
    input_image = unnormalize_image(input_image)

  return input_image, input_image_mask


def process_batch_audio(input_audio, mask, input_size=config.AUDIO_INPUT_SIZE,
                        input_d=config.AUDIO_INPUT_D, pos_ids=None):
  if len(input_audio.shape) == 3:
    t = input_audio.shape[0]
    dh, dw = input_d, input_d
    c = input_audio.shape[-1] // (dw*dh)
    h = input_size[0] // dh
    w = input_size[1] // dw
    with_blanks = np.zeros((t, h*w, dh*dw*1))
    with_blanks[np.arange(t)[:, None], pos_ids] = input_audio
    input_audio = with_blanks
    input_audio_mask = np.zeros((t, h*w), dtype=mask.dtype)
    input_audio_mask[np.arange(t)[:, None], pos_ids] = mask
    input_audio_mask = input_audio_mask.reshape((t, h, w))
    input_audio = input_audio.reshape((t, h, w, dh, dw, c))
    input_audio = (input_audio * config.AUDIO_VIT_STD) + config.AUDIO_VIT_MEAN
    input_audio = input_audio*(input_audio_mask.reshape((t, h, w))[:, :, :, None, None, None])
    input_audio = input_audio.transpose((0, 1, 3, 2, 4, 5))
    input_audio = input_audio.reshape((t, h*dh, w*dw, c))
  else:
    input_audio_mask = mask
    input_audio = (input_audio * config.AUDIO_VIT_STD) + config.AUDIO_VIT_MEAN
  return input_audio, input_audio_mask
