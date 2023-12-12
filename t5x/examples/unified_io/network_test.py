import jax.numpy as jnp
import numpy as np
from absl.testing.absltest import TestCase
from flax import traverse_util
from flax.core import unfreeze
from flax.traverse_util import unflatten_dict, flatten_dict

from t5x.examples.unified_io import test_utils
from t5x.examples.unified_io.config import T5Config
from t5x.examples.unified_io.test_utils import DEBUG_CONFIG


class TestTransformer(TestCase):

  @classmethod
  def setUpClass(cls):
    cls.transformer, cls.variables = test_utils.build_test_transformer(
      DEBUG_CONFIG, 2, 2, 6922)
    cls.cfg: T5Config = cls.transformer.config

  def test_subsegment(self):
    """Ensure using subsegments is equal to using two separate forward passes"""
    rng = np.random.RandomState(982)
    bs = 2
    batch = test_utils.build_random_batch(
      self.cfg, rng, bs, [3, 3], [7])
    batch1, batch2 = {}, {}
    for k, v in batch.items():
      if k.startswith("targets/text"):
        v1, v2 = v[:, :3], v[:, 3:]
      else:
        v1, v2 = v, v
      batch1[k] = v1
      batch2[k] = v2
    batch["targets/text/segment_ids"] = np.concatenate([
      jnp.zeros((bs, 3), dtype=np.int32), jnp.ones((bs, 4), dtype=np.int32)
    ], axis=1)
    expected = self.transformer.apply(self.variables, batch)
    actual1 = self.transformer.apply(self.variables, batch1)
    actual2 = self.transformer.apply(self.variables, batch2)

    for k in expected:
      actual = {k: [np.concatenate([a, b], 1) for a, b in zip(actual1[k], actual2[k])]}
      mask = expected[k][-1]
      np.testing.assert_array_equal(mask, actual[k][-1])

      for a, b in zip(expected[k][:-1], actual[k][:-1]):
        a, b = np.array(a), np.array(b)
        if len(a.shape) == 3:
          a, b = a*mask[:, :, None], b*mask[:, :, None]
        else:
          a, b = a*mask, b*mask
        if a.dtype == np.float32:
          np.testing.assert_allclose(a, b, atol=1e-5, rtol=0)
        else:
          np.testing.assert_array_equal(a, b)

  def test_null_modality(self):
    """Ensure an additional null modality  does not change the results"""
    rng = np.random.RandomState(69218)
    bs = 2
    batch = test_utils.build_random_batch(
      self.cfg, rng, bs, [6, 8, 5, 7], [7, 6, 3])
    variables = unfreeze(self.variables)

    # Hack in the varible for the new target modalities
    variables["params"]["audio_token_embedder"] = dict(embedding=jnp.full(
      (self.cfg.audio_vocab_size, self.cfg.emb_dim), 1/self.cfg.emb_dim))
    for k in ["inputs/2/mask", "inputs/3/mask", "targets/audio/mask"]:
      batch[k] = np.zeros_like(batch[k])

    pruned_batch = unflatten_dict(batch, sep="/")
    pruned_batch["inputs"] = {
      "0": pruned_batch["inputs"]["1"],
      "1": pruned_batch["inputs"]["2"]
    }
    pruned_batch["targets"] = {k: pruned_batch["targets"][k] for k in ["text", "image"]}
    pruned_batch = flatten_dict(batch, sep="/")

    expected = self.transformer.apply(variables, pruned_batch)

    transformer = test_utils.build_test_transformer(DEBUG_CONFIG, 4, 3)
    actual = transformer.apply(variables, batch)

    # `build_test_transformer` hacks `modality_processing.get_target_modalities` to generate
    # debugging modalities encoders, call it again so it return 2 instead 4 or 3
    test_utils.build_test_transformer(DEBUG_CONFIG, 2, 2)

    for k in expected:
      mask = expected[k][-1]
      np.testing.assert_array_equal(mask, actual[k][-1])
      for a, b in zip(expected[k][:-1], actual[k][:-1]):
        a, b = np.array(a), np.array(b)
        if len(a.shape) == 3:
          a, b = a*mask[:, :, None], b*mask[:, :, None]
        else:
          a, b = a*mask, b*mask
        if a.dtype == np.float32:
          np.testing.assert_allclose(a, b, atol=1e-5, rtol=0)
        else:
          np.testing.assert_array_equal(a, b)

  def test_mask(self):
    """Ensure that adding masked tokens does not change the output"""
    rng = np.random.RandomState(327)
    bs = 2
    batch = test_utils.build_random_batch(self.cfg, rng, bs, [6, 8], [7, 6])

    # Add random masked token
    padded = dict(batch)
    padded = traverse_util.unflatten_dict(padded, "/")
    all_ixs = {}
    for k1, k2 in [("inputs", "0"), ("inputs", "1"),
                   ("targets", "text"), ("targets", "image")]:
      features = padded[k1][k2]
      mask = features["mask"]
      n_to_add = 2 + rng.randint(0, 3)
      ixs = rng.choice(mask.shape[1], n_to_add)
      seq_len = mask.shape[1] + n_to_add
      for k, val in features.items():
        if val is None:
          continue
        sh = val.shape
        if k in {"mask", "targets"}:
          to_add = np.zeros((sh[0], len(ixs)), dtype=val.dtype)
          features[k] = np.insert(val, ixs, to_add, axis=1)
        elif k == "embed":
          to_add = np.zeros((sh[0], len(ixs), sh[2]), dtype=val.dtype)
          features[k] = np.insert(val, ixs, to_add, axis=1)
        elif k == "rel_atten":
          to_add = np.zeros((sh[0], sh[1], len(ixs), sh[3]), dtype=val.dtype)
          val = np.insert(val, ixs, to_add, axis=2)
          to_add = np.zeros((sh[0], sh[1], sh[2]+n_to_add, len(ixs)), dtype=val.dtype)
          features[k] = np.insert(val, ixs, to_add, axis=3)
        else:
          raise NotImplementedError(k)
      all_ixs[k2] = ixs

    padded = traverse_util.flatten_dict(padded, sep="/")
    expected = self.transformer.apply(self.variables, batch)
    actual = self.transformer.apply(self.variables, padded)

    for k in expected:
      to_delete = all_ixs[k]
      to_delete = np.sort(to_delete)
      to_delete = to_delete + np.arange(len(to_delete))

      mask = expected[k][-1]
      np.testing.assert_array_equal(
        mask, np.delete(actual[k][-1], to_delete, axis=1))
      for a, b in zip(expected[k][:-1], actual[k][:-1]):
        a, b = np.array(a), np.array(b)
        b = np.delete(b, to_delete, axis=1)
        if len(a.shape) == 3:
          a, b = a*mask[:, :, None], b*mask[:, :, None]
        else:
          a, b = a*mask, b*mask
        if a.dtype == np.float32:
          np.testing.assert_allclose(a, b, atol=1e-5, rtol=0)
        else:
          np.testing.assert_array_equal(a, b)

  def test_pack(self):
    """Ensure that packing/folding does not change the output"""
    for it, (target_segments, bs) in enumerate([
      (True, 3),
      (False, 3),
      (True, 4),
    ]):
      rng = np.random.RandomState(it*45107)
      batch = test_utils.build_random_batch(
        self.cfg, rng, bs, [6, 8], [7, 6], target_segment_ids=target_segments)
      expected = self.transformer.apply(self.variables, batch)

      enc_len = batch["inputs/0/mask"].sum(-1) + batch["inputs/1/mask"].sum(-1)
      dec_len = batch["targets/image/mask"].sum(-1) + batch["targets/text/mask"].sum(-1)

      evals = [
        (False, True, True),
        (bs, True, True),
      ]
      if bs == 4:
        evals.append((2, True, True))
      for fold, h_pack_inputs, h_pack_targets in evals:
        if fold:
          enc_pad_len = enc_len.reshape((bs//fold, fold)).sum().max()
          dec_pad_len = dec_len.reshape((bs//fold, fold)).sum().max()
        else:
          enc_pad_len = enc_len.max()
          dec_pad_len = dec_len.max()

        actual = self.transformer.apply(
          self.variables, batch,
          horizontally_pack_inputs=enc_pad_len if h_pack_inputs else None,
          horizontally_pack_targets=dec_pad_len if h_pack_targets else None,
          fold=fold,
        )

        for k in expected:
          mask = expected[k][-1]
          np.testing.assert_array_equal(mask, actual[k][-1])

          for a, b in zip(expected[k][:-1], actual[k][:-1]):
            a, b = np.array(a), np.array(b)
            if len(a.shape) == 3:
              a, b = a*mask[:, :, None], b*mask[:, :, None]
            else:
              a, b = a*mask, b*mask
            if a.dtype == np.float32:
              np.testing.assert_allclose(a, b, atol=1e-5, rtol=0)
            else:
              np.testing.assert_array_equal(a, b)


if __name__ == '__main__':
  t = TestTransformer()
  t.setUpClass()
  t.test_subsegment()
  t.test_null_modality()
  t.test_mask()