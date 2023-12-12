import argparse
import json
from collections import defaultdict
from os.path import join
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from PIL import Image


MAPPING = {
  "Flan2021": "conceptofmind/flan2021_submix_original",
  "T0": "conceptofmind/t0_submix_original",
  "NIv2": "conceptofmind/niv2_submix_original",
  "CoT": "conceptofmind/cot_submix_original",
  "Dialog": "conceptofmind/dialog_submix_original",
}


class FLAN(tfds.core.GeneratorBasedBuilder):
  VERSION = tfds.core.Version('1.0.0')

  def __init__(self, src, **kwargs):
    self.src = src
    self.__class__.name = f"FLANv2-{src}"
    super().__init__(**kwargs)

  def _info(self) -> tfds.core.DatasetInfo:
    features = tfds.features.FeaturesDict(dict(
      example_num=tfds.features.Tensor(shape=(), dtype=tf.int32),
      inputs=tfds.features.Tensor(shape=(), dtype=tf.string),
      targets=tfds.features.Tensor(shape=(), dtype=tf.string),
      task_source=tfds.features.Tensor(shape=(), dtype=tf.string),
      task_name=tfds.features.Tensor(shape=(), dtype=tf.string),
      template_type=tfds.features.Tensor(shape=(), dtype=tf.string),
    ))
    return tfds.core.DatasetInfo(
      builder=self,
      features=features,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    return {
      'train': self._generate_examples(),
    }

  def _generate_examples(self):
    import datasets
    ds = datasets.load_dataset(MAPPING[self.src])["train"]
    for ix, ex in enumerate(ds):
      example = dict(example_num=ix)
      example.update({k: ex[k] for k in
                      ["inputs", "targets", "task_source", "task_name", "template_type"]})
      yield ix, example


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("kind")
  parser.add_argument("data_dir")
  args = parser.parse_args()

  builder = FLAN(args.kind, data_dir=args.data_dir)
  builder.download_and_prepare()


if __name__ == '__main__':
  main()