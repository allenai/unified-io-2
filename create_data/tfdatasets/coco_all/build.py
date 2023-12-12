"""coco_all dataset."""
import argparse
import pdb
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import json


def load_json(x):
  with open(x) as f:
    return json.load(f)


class CocoAll(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for coco_all dataset.

  This contains several coco annotation (captioning, VQA, and objects), we put all these annotations
  in one dataset to avoid storing the image multiple times for each dataset. We follow the
  COCO 2017 split for these tasks.
  """

  VERSION = tfds.core.Version('1.0.1')

  def __init__(self, vqa_home, caption_home, **kwargs):
    super().__init__(**kwargs)
    self.caption_home = caption_home
    self.vqa_home = vqa_home

  def _info(self) -> tfds.core.DatasetInfo:
    return tfds.core.DatasetInfo(
      builder=self,
      features=tfds.features.FeaturesDict({
        # These are the features of your dataset like images, labels ...
        'image': tfds.features.Image(shape=(None, None, 3)),
        'captions': tfds.features.FeaturesDict({
          'num': tfds.features.Tensor(shape=(), dtype=tf.int64),
          'id': tfds.features.Tensor(shape=(None,), dtype=tf.int64),
          'text':  tfds.features.Tensor(shape=(None,), dtype=tf.string)
        }),
        'vqa': tfds.features.FeaturesDict({
          'num': tfds.features.Tensor(shape=(), dtype=tf.int64),
          'id': tfds.features.Tensor(shape=(None,), dtype=tf.int64),
          'questions':  tfds.features.Tensor(shape=(None,), dtype=tf.string),
          'answers':  tfds.features.Tensor(shape=(None, 10), dtype=tf.string),
        }),
        'objects':tfds.features.FeaturesDict({
          'area': tfds.features.Tensor(shape=(None,), dtype=tf.int64),
          'bbox': tfds.features.Tensor(shape=(None,4), dtype=tf.float32),
          'id': tfds.features.Tensor(shape=(None,), dtype=tf.int64),
          'is_crowd': tfds.features.Tensor(shape=(None,), dtype=tf.bool),
          'label': tfds.features.Tensor(shape=(None,), dtype=tf.int64),
        }),
        'image/filename': tfds.features.Tensor(shape=(), dtype=tf.string)
      }),
      # If there's a common (input, target) tuple from the
      # features, specify them here. They'll be used if
      # `as_supervised=True` in `builder.as_dataset`.
      supervised_keys=None,  # Set to `None` to disable
      homepage='https://dataset-homepage/',
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    vqa_home = self.vqa_home
    caption_home = self.caption_home

    vqa_train_annotation_path = f'{vqa_home}/v2_mscoco_train2014_annotations.json'
    vqa_train_annotation = json.load(open(vqa_train_annotation_path, 'r'))['annotations']

    vqa_train_question_path = f'{vqa_home}/v2_OpenEnded_mscoco_train2014_questions.json'
    vqa_train_question = json.load(open(vqa_train_question_path, 'r'))['questions']

    vqa_val_annotation_path = f'{vqa_home}/v2_mscoco_val2014_annotations.json'
    vqa_val_annotation = json.load(open(vqa_val_annotation_path, 'r'))['annotations']

    vqa_val_question_path = f'{vqa_home}/v2_OpenEnded_mscoco_val2014_questions.json'
    vqa_val_question = json.load(open(vqa_val_question_path, 'r'))['questions']

    # given image id, qa pairs.
    data_dict = {}
    for questions, annotations in zip(vqa_train_question + vqa_val_question,
                                      vqa_train_annotation + vqa_val_annotation):
      ques_id = annotations['question_id']
      answer = [i['answer'] for i in annotations['answers']]
      question = questions['question']
      image_id = annotations['image_id']

      if image_id not in data_dict:
        data_dict[image_id] = {'vqa':[]}

      data_dict[image_id]['vqa'].append(
        {'question': question, 'answer': answer, 'ques_id': ques_id})

    # get captions.
    caption_train_path = f'{caption_home}/captions_train2017.json'
    caption_train = json.load(open(caption_train_path, 'r'))['annotations']

    caption_val_path = f'{caption_home}/captions_val2017.json'
    caption_val = json.load(open(caption_val_path, 'r'))['annotations']

    for caption in caption_train+caption_val:
      image_id = caption['image_id']
      if 'caption' not in data_dict[image_id]:
        data_dict[image_id]['caption'] = []

      data_dict[image_id]['caption'].append({'caption': caption['caption'], 'id': caption['id']})

    return {
      'train': self._generate_examples('train', data_dict),
      'validation': self._generate_examples('validation', data_dict),
    }

  def _generate_examples(self, split, data_dict):
    """Yields examples."""
    # Combine with existing coco tfds dataset.
    ds = tfds.load('coco/2017', split=split, shuffle_files=False, data_dir=self.data_dir)
    for e in ds:
      key = b'coco_all/' + e['image/filename'].numpy()
      image_id = e['image/id'].numpy()
      qas = data_dict[image_id]['vqa']

      ques = np.array([q['question'].encode('utf-8') for q in qas], dtype=object)
      ans = np.array([[i.encode('utf-8') for i in q['answer']] for q in qas], dtype=object)
      qid = np.array([q['ques_id'] for q in qas])

      captions = data_dict[image_id]['caption']
      caption_text = np.array([c['caption'].encode('utf-8') for c in captions], dtype=object)
      caption_id = np.array([c['id'] for c in captions])

      yield key, {
        'image': e['image'].numpy(),
        'image/filename': e['image/filename'].numpy(),
        'captions': {
          'num': len(caption_id),
          'id': caption_id,
          'text': caption_text,
        },
        'objects': {
          'area': e['objects']['area'].numpy(),
          'bbox': e['objects']['bbox'].numpy(),
          'id': e['objects']['id'].numpy(),
          'is_crowd': e['objects']['is_crowd'].numpy(),
          'label': e['objects']['label'].numpy(),
        },
        'vqa':{
          'num': len(ques),
          'id': qid,
          'questions': ques,
          'answers': ans,
        }
      }


def main():
  import resource
  # This script can hit open file count limits due to load COCO on tfds
  low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
  resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

  parser = argparse.ArgumentParser("Build the COCO all dataset")
  parser.add_argument("data_dir", help="TFDS data dir")
  parser.add_argument("vqa_home", help="directory with VQA annotations")
  parser.add_argument("coco_home", help="directory with COCO 2017 annotations")
  args = parser.parse_args()

  builder = CocoAll(args.vqa_home, args.coco_home, data_dir=args.data_dir)
  builder.download_and_prepare()


if __name__ == '__main__':
  main()

