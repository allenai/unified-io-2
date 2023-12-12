import functools

import seqio
from seqio import TaskRegistry

from t5x.examples.unified_io.metrics import metrics
from t5x.examples.unified_io.data.postprocessing import return_meta, return_field
from t5x.examples.unified_io.metrics.metrics import exact_match
from t5x.examples.unified_io.modality_processing import unified_io_preprocessor

from t5x.examples.unified_io import config, modality_processing
from t5x.examples.unified_io.data import preprocessing
from t5x.examples.unified_io.data.preprocessing import rekey


def add_refexp(name, src_name=None):
  if src_name is None:
    src_name = name
  TaskRegistry.add(
    name,
    # A TfdsTask takes in a TFDS name instead of a tf.data.Dataset function.
    source=seqio.TfdsDataSource(
      tfds_name=f"ref_coco/{src_name}:1.0.0",
      tfds_data_dir=config.MULTITASK_TFDS_DATA_DIR,
    ),
    preprocessors=[
      functools.partial(
        rekey, key_map={
          "image": ["image"],
          "bbox": ["objects", "bbox"],
          "label": ["objects", "refexp", "raw"],
          "refexp_id": ["objects", "refexp", "refexp_id"],
        }),
      functools.partial(
        preprocessing.refer_expression_preprocessor,
        dataset_name=name,
      ),
      unified_io_preprocessor,
    ],
    postprocess_fn=return_meta,
    metric_fns=[metrics.ref_exp_metric],
    output_features=modality_processing.OUTPUT_FEATURES,
  )


add_refexp("refcoco_unc")
add_refexp("refcocog_google")
add_refexp("refcoco_plus_unc", "refcocoplus_unc")


TaskRegistry.add(
  "image_generation_coco_2017",
  source=seqio.TfdsDataSource(
    tfds_name="coco_all:1.0.1",
    tfds_data_dir=config.MULTITASK_TFDS_DATA_DIR,
  ),
  preprocessors=[
    functools.partial(
      rekey, key_map={
        "image/filename": ["image/filename"],
        "image": ["image"],
        "captions": ["captions", "text"]
      }),
    functools.partial(
      preprocessing.image_generation_preprocessor,
      dataset_name="image_generation_coco_2017",
    ),
    unified_io_preprocessor,
  ],
  output_features=modality_processing.OUTPUT_FEATURES,
)


TaskRegistry.add(
  "image_caption_coco_2017",
  source=seqio.TfdsDataSource(
    tfds_name="coco_all:1.0.1",
    tfds_data_dir=config.MULTITASK_TFDS_DATA_DIR,
  ),
  preprocessors=[
    functools.partial(
      rekey, key_map={
        "image/filename": ["image/filename"],
        "image": ["image"],
        "captions": ["captions", "text"]
      }),
    functools.partial(
      preprocessing.image_caption_preprocessor,
      dataset_name="image_caption_coco_2017",
    ),
    unified_io_preprocessor
  ],
  output_features=modality_processing.OUTPUT_FEATURES,
)


TaskRegistry.add(
  "image_inpainting_coco",
  source=seqio.TfdsDataSource(
    tfds_name="coco_all:1.0.1",
    tfds_data_dir=config.MULTITASK_TFDS_DATA_DIR,
  ),
  preprocessors=[
    functools.partial(
      rekey, key_map={
        "image": ["image"],
        "bbox": ["objects", "bbox"],
        "label": ["objects", "label"],
      }),
    functools.partial(
      preprocessing.image_inpainting_preprocessor,
      dataset_name="image_inpainting_coco",
      class_names='metadata/coco/coco_class_name_2017.json',
    ),
    unified_io_preprocessor
  ],
  output_features=modality_processing.OUTPUT_FEATURES,
)


TaskRegistry.add(
  "vqa_coco_2017",
  # A TfdsTask takes in a TFDS name instead of a tf.data.Dataset function.
  source=seqio.TfdsDataSource(
    tfds_name="coco_all:1.0.1",
    tfds_data_dir=config.MULTITASK_TFDS_DATA_DIR,
  ),
  preprocessors=[
    functools.partial(
      rekey, key_map={
        "image": ["image"],
        "text_inputs": ["vqa", "questions"],
        "text_targets": ["vqa", "answers"],
      }),
    preprocessing.vqa_preprocessor,
    unified_io_preprocessor
  ],
  postprocess_fn=functools.partial(return_field, field="meta/all_references"),
  metric_fns=[metrics.vqa_metric],
  output_features=modality_processing.OUTPUT_FEATURES,
)


TaskRegistry.add(
  "box_classification_coco_2017",
  # A TfdsTask takes in a TFDS name instead of a tf.data.Dataset function.
  source=seqio.TfdsDataSource(
    tfds_name="coco_all:1.0.1",
    tfds_data_dir=config.MULTITASK_TFDS_DATA_DIR,
  ),
  preprocessors=[
    functools.partial(
      rekey, key_map={
        "image": ["image"],
        "bbox": ["objects", "bbox"],
        "label": ["objects", "label"],
        "image_id": ["image/filename"],
      }),
    functools.partial(
      preprocessing.box_classification_preprocessor,
      dataset_name='box_classification_coco_2017',
      class_names='metadata/coco/coco_class_name_2017.json',
    ),
    unified_io_preprocessor
  ],
  postprocess_fn=functools.partial(return_field, field="meta/label"),
  metric_fns=[exact_match],
  output_features=modality_processing.OUTPUT_FEATURES,
)