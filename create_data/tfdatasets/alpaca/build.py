import argparse

import tensorflow as tf
import tensorflow_datasets as tfds


class Alpaca(tfds.core.GeneratorBasedBuilder):
  VERSION = tfds.core.Version('1.0.0')

  def _info(self) -> tfds.core.DatasetInfo:
    features = tfds.features.FeaturesDict(dict(
      example_num=tfds.features.Tensor(shape=(), dtype=tf.int32),
      instruction=tfds.features.Tensor(shape=(), dtype=tf.string),
      input=tfds.features.Tensor(shape=(), dtype=tf.string),
      output=tfds.features.Tensor(shape=(), dtype=tf.string),
    ))
    return tfds.core.DatasetInfo(builder=self, features=features)

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    return {'train': self._generate_examples()}

  def _generate_examples(self):
    import datasets
    ds = datasets.load_dataset("yahma/alpaca-cleaned")["train"]
    for ix, ex in enumerate(ds):
      ex["example_num"] = ix
      yield ix, ex


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("data_dir")
  args = parser.parse_args()

  builder = Alpaca(data_dir=args.data_dir)
  builder.download_and_prepare()


if __name__ == '__main__':
  main()