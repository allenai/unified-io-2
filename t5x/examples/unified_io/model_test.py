import jax.random
from absl.testing.absltest import TestCase
from flax import traverse_util

from t5x.examples.unified_io import test_utils, modality_processing
from t5x.examples.unified_io.data.data_utils import get_default_vocabulary
from t5x.examples.unified_io.modality_processing import unified_io_preprocessor, \
  UnifiedIOFeatureConverter
import numpy as np

from t5x.examples.unified_io.models import EncoderDecoderModel
from t5x.examples.unified_io.test_utils import DEBUG_CONFIG
from t5x.examples.unified_io.utils import get_model
import tensorflow as tf


class ModelTest(TestCase):

  @classmethod
  def setUpClass(cls):
    cls.model = get_model("tiny", ["text"], ["text"])
    cls.cfg = cls.model.module.config

  def test_with_choices(self):
    """Test predictions with `choices` is consistent with computing the loss"""
    seq_len = dict(text_inputs=32, text_targets=16)
    ds = tf.data.Dataset.from_tensors(dict(
      text_inputs="Which answer is best?",
      text_targets="",
      choices=["a fat cat", "a quick dog"]
    ))
    ds = unified_io_preprocessor(ds, modality_processing.OUTPUT_FEATURES, seq_len)
    ds = UnifiedIOFeatureConverter()(ds, seq_len)
    batch = next(ds.repeat(2).batch(2).as_numpy_iterator())

    variables = self.model.get_initial_variables(
      jax.random.PRNGKey(5919),
      {k: v.shape for k, v in batch.items()},
      {k: v.dtype for k, v in batch.items()}
    )["params"]

    _, aux = self.model.predict_batch_with_aux(variables, batch)

    # Take the highest-ranked choices and compute loss as a regular batch
    # We have to manually build EOS and auto-regressive inputs
    tokens = aux["text-tokens"]
    features = traverse_util.unflatten_dict(batch, sep="/")
    features["targets"] = dict(text=dict(
      inputs=np.pad(tokens[:, :-1], [[0, 0], [1, 0]]),
      mask=(tokens>0).astype(np.int32),
      targets=tokens
    ))
    batch = traverse_util.flatten_dict(features, sep="/")
    loss = self.model.loss_fn(
      variables, batch, None, z_loss=0.0, loss_normalizing_by_weight_sum=False,
      loss_normalizing_factor=1)[0]
    self.assertAlmostEqual(float(loss), float(aux["scores"].sum()), places=4)
