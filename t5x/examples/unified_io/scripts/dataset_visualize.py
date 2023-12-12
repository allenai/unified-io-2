import sys
from os import mkdir
from os.path import join, exists

import gin
import seqio
from PIL import ImageColor
from seqio import MixtureRegistry

from t5x.examples.unified_io import utils, config
from t5x.examples.unified_io.data.visualization_utils import build_html_table, process_image, process_batch_audio, process_batch_image
from t5x.examples.unified_io.data.data_utils import get_default_vocabulary
# Make sure all tasks are registered
from t5x.examples.unified_io.data import mixtures
import argparse
import numpy as np
import html

from t5x.examples.unified_io.modality_processing import UnifiedIOFeatureConverter
from t5x.examples.unified_io.metrics.utils import extract_bboxes_from_text

from t5x.examples.unified_io.config import AUDIO_INPUT_D, \
  AUDIO_HISTORY_INPUT_SIZE, AUDIO_HISTORY_INPUT_D, AUDIO_TARGET_D, AUDIOSET_MEAN, AUDIOSET_STD
from t5x.gin_utils import rewrite_gin_args
from PIL import Image

import matplotlib.pyplot as plt
import librosa


def draw_bboxes(img, bboxes, color=None):
  bboxes = np.array(bboxes, np.int32)
  img = np.copy(img)

  if color is None:
    color = (255, 255, 255)
  elif isinstance(color, str):
    # This will automatically raise Error if rgb cannot be parsed.
    color = ImageColor.getrgb(color)

  for x1, y1, x2, y2 in bboxes:
    for x in range(x1, x2):
      img[x, y1] = color
      img[x, y2-1] = color
    for y in range(y1, y2):
      img[x1, y] = color
      img[x2, y] = color
  return img


log_low = log_high = fig = axes = None

def plot_spectrogram(log_mel):
  global log_low, log_high, fig, axes
  if log_low is None:
    log_low = np.log(1e-5)
    log_high = -log_low
    fig, axes = plt.subplots(1, 1)

  spec = np.exp(np.clip(log_mel, log_low, log_high))
  spec = librosa.power_to_db(spec)
  # axes.xaxis.set_ticklabels([])
  # axes.yaxis.set_ticklabels([])
  im = axes.imshow(spec, origin='lower', aspect='auto', interpolation='nearest')
  fig.tight_layout(pad=0)
  fig.canvas.draw()
  data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  axes.clear()
  return data


MASK_COLOR = [1.0, 0, 1.0]


def mask_image(input_image, input_image_mask, input_d, mask_color, mask_reshape=False):
  x_dim, y_dim = input_image.shape[:2]
  x_n_patches = x_dim // input_d
  y_n_patches = y_dim // input_d
  assert x_n_patches * input_d == x_dim
  assert y_n_patches * input_d == y_dim

  if mask_reshape:
    input_image_mask = input_image_mask.reshape((x_n_patches, y_n_patches))

  masked_input_image = input_image.copy()
  for x in range(x_n_patches):
    for y in range(y_n_patches):
      if not input_image_mask[x, y]:
        masked_input_image[x*input_d:(x+1)*input_d, y*input_d:(y+1)*input_d] = mask_color

  return masked_input_image


def image_resize(input_image, target_size=256):
  h, w = input_image.shape[:2]
  min_size = min(h, w)

  if min_size > target_size:
    new_h = int(h / min_size * target_size)
    new_w = int(w / min_size * target_size)
    input_image = np.array(Image.fromarray(input_image, mode="RGB").resize((new_w, new_h), resample=Image.BICUBIC))

  return input_image


def example_to_html_dict(ex, box_color="red", show_image_masks=True, show_audio_masks=True):
  vocab = get_default_vocabulary()
  out = dict()

  input_text = vocab.decode(ex["inputs/text/tokens"].tolist())

  if "inputs/image/input" in ex:
    input_image, input_image_mask = process_image(
      ex["inputs/image/input"],
      ex["inputs/image/mask"],
      pos_ids=np.expand_dims(ex["inputs/image/pos_ids"], axis=0) if len(ex["inputs/image/input"].shape) == 2 else None,
    )

    bboxes, class_names = extract_bboxes_from_text(input_text, input_image.shape, use_label=False)
    if bboxes is not None:
      # TODO draw class names?
      input_image = draw_bboxes(input_image, bboxes, box_color)
    out["input/image"] = input_image
    if show_image_masks:
      out["input/image/mask"] = mask_image(
        input_image, input_image_mask, config.IMAGE_INPUT_D, MASK_COLOR)

  if "inputs/audio/input" in ex:
    input_audio, input_audio_mask = process_batch_audio(
      np.expand_dims(ex["inputs/audio/input"], axis=0),
      np.expand_dims(ex["inputs/audio/mask"], axis=0),
      pos_ids=np.expand_dims(ex["inputs/audio/pos_ids"], axis=0) if len(ex["inputs/audio/input"].shape) == 2 else None,
    )
    input_audio = np.squeeze(np.squeeze(input_audio, 0), -1)
    input_audio_mask = np.squeeze(input_audio_mask, 0)
    out["input/audio"] = image_resize(plot_spectrogram(input_audio))
    if show_audio_masks:
      masked_input_audio = mask_image(input_audio, input_audio_mask, AUDIO_INPUT_D, np.log(1e5))
      out["input/audio/mask"] = image_resize(plot_spectrogram(masked_input_audio))

  if "example_id" in ex:
    out["example_id"] = ex["example_id"]
    if not isinstance(out["example_id"], str):
      out["example_id"] = out["example_id"].decode("utf-8")

  if "inputs/image_history/input" in ex:
    input_image_history, input_image_history_mask = process_batch_image(
      ex["inputs/image_history/input"],
      ex["inputs/image_history/mask"],
      config.IMAGE_HISTORY_INPUT_SIZE, config.IMAGE_HISTORY_INPUT_D,
      ex["inputs/image_history/pos_ids"] if len(ex["inputs/image_history/input"].shape) == 3 else None,
    )
    input_image_history = input_image_history.transpose(1, 0, 2, 3).reshape(
      [config.IMAGE_HISTORY_INPUT_SIZE[0], -1, 3])
    out["input/image_history"] = input_image_history

  if "inputs/audio_history/input" in ex:
    input_audio_history, input_audio_history_mask = process_batch_audio(
      ex["inputs/audio_history/input"],
      ex["inputs/audio_history/mask"],
      AUDIO_HISTORY_INPUT_SIZE, AUDIO_HISTORY_INPUT_D,
      ex["inputs/audio_history/pos_ids"] if len(ex["inputs/audio_history/input"].shape) == 3 else None,
    )
    input_audio_history = np.squeeze(input_audio_history, axis=-1)
    input_audio_history = np.stack([plot_spectrogram(audio) for audio in input_audio_history])
    spec_size = list(input_audio_history.shape[1:-1])
    input_audio_history = input_audio_history.transpose(1, 0, 2, 3).reshape([spec_size[0], -1, 3])
    input_audio_history = image_resize(input_audio_history)
    out["input/audio_history"] = input_audio_history

  out["input/text"] = html.escape(input_text)

  if "targets/text/inputs"in ex:
    decoder_text = vocab.decode(ex["targets/text/inputs"].tolist())

  if "targets/image/image" in ex:
    target_image = (ex["targets/image/image"] + 1) / 2.0
    if "targets/text/inputs" in ex:
      bboxes, class_names = extract_bboxes_from_text(decoder_text, target_image.shape)
      if bboxes is not None:
        target_image = draw_bboxes(target_image, bboxes, box_color)

    out["target/image"] = target_image
    if show_image_masks:
      for k in ["targets/image/mask", "targets/image/loss_mask", "targets/image/task_mask"]:
        if k in ex:
          out[k] = mask_image(target_image, ex[k], config.IMAGE_TARGET_D, MASK_COLOR, mask_reshape=True)

  if "targets/audio/audio" in ex:
    target_audio = np.squeeze(ex["targets/audio/audio"] * AUDIOSET_STD + AUDIOSET_MEAN, axis=-1)
    if show_audio_masks:
      for k in ["targets/audio/mask", "targets/audio/loss_mask", "targets/audio/task_mask"]:
        if k in ex:
          masked_target_audio = mask_image(target_audio, ex[k], AUDIO_TARGET_D, np.log(1e5), mask_reshape=True)
          out[k] = image_resize(plot_spectrogram(masked_target_audio))
    out["target/audio"] = image_resize(plot_spectrogram(target_audio))

  if "targets/text/inputs" in ex:
    out["target/text"] = html.escape(decoder_text)

  return out


def build_qualitative_table(name, split, n, box_color="red", is_training=None, shuffle=True,
                            show_masks=True):
  if is_training is None:
    is_training = True if split == "train" else False,
  seq_len = {
    "is_training": is_training,
    "text_inputs": 512,
    "text_targets": 512,
    "image_input_samples": 576,
    "image_history_input_samples": 256,
    "audio_input_samples": 128,
    "audio_history_input_samples": 128,
    'num_frames': 4,
  }
  if split != "train":
    seq_len["seed"] = 42

  dataset = seqio.get_mixture_or_task(name).get_dataset(
    sequence_length=seq_len,
    split=split,
    num_epochs=1,
    shard_info=seqio.ShardInfo(index=0, num_shards=1),
    use_cached=False,
    seed=42,
    shuffle=shuffle,
  )

  converter = UnifiedIOFeatureConverter()
  dataset = converter(dataset, seq_len)

  table = []
  for ix, ex in zip(range(n), dataset.as_numpy_iterator()):
    table.append(example_to_html_dict(ex, box_color, show_image_masks=show_masks,
                                      show_audio_masks=show_masks))

  return build_html_table(table)


def for_task(task_name, split, output_file, num_examples=50, is_training=True):
  table = build_qualitative_table(task_name, split, num_examples, is_training=is_training)
  import os
  print(os.path.abspath(output_file))
  with open(output_file, "w") as f:
    f.write(table)


def for_all_in_mixture(mixture_name, split, output_dir, num_examples=50):
  if not exists(output_dir):
    mkdir(output_dir)

  mixture = MixtureRegistry.get(mixture_name)
  for ix, task in enumerate(mixture.tasks):
    if "__ax" in task.name or "mismatched" in task.name:
      continue
    name = task.name
    html = [f"<h3>{name}<h3>"]

    print(f"Getting qual. examples for {name} ({ix+1}/{len(mixture.tasks)})")
    output_file = join(output_dir, f"{name}.html")
    if exists(output_file):
      continue
    html += [build_qualitative_table(name, split, num_examples)]
    with open(join(output_dir, f"{name}.html"), "w") as f:
      f.write("\n".join(html))
    print("Done")


def main():
  parser = argparse.ArgumentParser(
    "Build a HTML visualization of a task(s) after UIO 2 post-processing")
  parser.add_argument("task_or_mixture",
                      help="What ot visualize, if a mixture show all tasks in the mixture")
  parser.add_argument("output_dir", help="Where to store the output")
  parser.add_argument("--eval", action="store_true",
                      help="Show with is_training set to false")
  parser.add_argument("--noshuffle", action="store_true",
                      help="No shuffling, can make visualization much faster.")
  parser.add_argument("--nomasks", action="store_true",
                      help="Don't show masks")
  parser.add_argument("--override", action="store_true",
                      help="Override the output file if it already exists")
  parser.add_argument("--split", default="train",
                      help="Dataset split to show")
  parser.add_argument("--num_examples", default=100, type=int,
                      help="Number of examles to show in the table")
  parser.add_argument("--postfix", type=str, default="",
                      help="Postfix to add the filenames")
  parser.add_argument("--gin_file", nargs="*",
                      help="Gin configuration files")

  parser.add_argument(
    "--gin_bindings", action='append',
    help="Set gin binding, can be set as `--gin.NAME=VALUE` or `--gin_bindings=NAME=VALUE`"
  )
  argv = rewrite_gin_args(sys.argv[1:])
  args = parser.parse_args(argv)

  gin.parse_config_files_and_bindings(
    config_files=args.gin_file,
    bindings=args.gin_bindings,
    skip_unknown=False
  )

  mixture_or_task = seqio.get_mixture_or_task(args.task_or_mixture)

  if isinstance(mixture_or_task, seqio.Task):
    tasks = [mixture_or_task]
  else:
    tasks = mixture_or_task.tasks

  for ix, task in enumerate(tasks):
    name = task.name
    html = [f"<h3>{name}<h3>"]

    output_file = join(args.output_dir, f"{name}{args.postfix}.html")
    if exists(output_file) and not args.override:
      print(f"Output file {output_file} exists for task {name}, skipping")
      continue
    else:
      print(f"Getting qual. examples for {name} ({ix+1}/{len(tasks)})")

    html += [build_qualitative_table(
      name, args.split, args.num_examples, is_training=not args.eval,
      show_masks=not args.nomasks, shuffle=not args.noshuffle)]
    print(f"Save examples to {output_file}")
    with open(output_file, "w") as f:
      f.write("\n".join(html))
    print("Done")


if __name__ == '__main__':
  main()
