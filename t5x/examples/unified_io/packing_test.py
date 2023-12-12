import numpy as np
import tensorflow as tf
from absl.testing import absltest

from t5x.examples.unified_io.packing import batch_with_constraints, pack_in_pairs


def build_mask(lens):
  c1 = np.floor(1, lens * np.random.random(lens.shape))
  c2 = lens - c1
  l1 = np.maximum(c1.max() + np.random.randint(0, 3), 1)
  out1 = np.arange(l1)[None, :] < c1[:, None]
  l2 = np.maximum(c2.max() + np.random.randint(0, 3), 1)
  out2 = np.arange(l2)[None, :] < c2[:, None]
  return out1, out2


def build_ds(lens):
  input_lens = np.array([x[0] for x in lens])
  target_lens = np.array([x[1] for x in lens])
  i1, i2 = build_mask(input_lens)
  t1, t2 = build_mask(target_lens)
  data = {
    "inputs/image/mask": i1,
    "inputs/text/mask": i2,
    "targets/image/mask": t1,
    "targets/text/mask": t2,
    "ixs": np.expand_dims(1 + np.arange(len(input_lens)), 1)
  }
  return tf.data.Dataset.from_tensor_slices(data)


def tolist(ex):
  return {k: v.tolist() for k, v in ex.items()}


class TestConstraintBatching(absltest.TestCase):

  def test_tiny(self):
    ds = tf.data.Dataset.from_tensor_slices(dict(
      c1=tf.convert_to_tensor([9, 12, 3, 4])
    ))
    batches = list(ex["c1"].tolist() for ex in batch_with_constraints(
      ds, 2, 5, [(lambda x: x["c1"], 20)]).as_numpy_iterator())
    self.assertEqual([[9, 3], [12, 4]], batches)

    batches = list(ex["c1"].tolist() for ex in batch_with_constraints(
      ds, 2, 5, [(lambda x: x["c1"], 100)]).as_numpy_iterator())
    self.assertEqual([[9, 12], [3, 4]], batches)

  def test_multiple(self):
    ds = tf.data.Dataset.from_tensor_slices(dict(
      c1=tf.convert_to_tensor([2,  2,  2,  2, 2, 14, 2, 2]),
      c2=tf.convert_to_tensor([11, 12, 13, 2, 2, 2, 2, 2]),
      example_ids=tf.range(8)
    ))

    def _fn1(x):
      return x["c1"]

    def _fn2(x):
      return x["c2"]

    batch_fns = [(_fn1, 15.1), (_fn2, 20)]

    batches = list(set(ex["example_ids"].tolist()) for ex in batch_with_constraints(
      ds, 3, 10, batch_fns).as_numpy_iterator())
    expected = [{0, 3, 4}, {1, 6, 7}]
    self.assertEqual(expected, batches)

  def test_random(self):
    seed = tf.convert_to_tensor([0, 83452])
    ds = tf.data.Dataset.from_tensor_slices(dict(
      c1=tf.random.stateless_uniform((100,), seed, 1, 10, dtype=tf.int64),
      c2=tf.random.stateless_uniform((100,), seed+1, 1, 10, dtype=tf.int64)
    ))
    const = [(lambda x: x["c1"], 18),
             (lambda x: x["c2"], 18)]
    for ex in batch_with_constraints(ds, 3, 10, const).as_numpy_iterator():
      self.assertLessEqual(ex["c1"].sum(), const[0][1])
      self.assertLessEqual(ex["c1"].sum(), const[1][1])


def _pack_in_pairs(*args, **kwargs, ):
  return pack_in_pairs(
    *args, **kwargs,
    encoder_masks=[
      ("inputs/image/mask", None),
      ("inputs/text/mask", None)
    ],
    decoder_masks=[
      "targets/image/mask",
      "targets/text/mask"
    ]
  )


class TestPackingBatching(absltest.TestCase):

  def test_tiny_pack(self):
    ds = build_ds([(3, 2), (2, 1)])
    ds = _pack_in_pairs(ds, 1, 5, 5, pool_size=2)
    exs = next(ds.as_numpy_iterator())
    self.assertEqual(set(exs["ixs"].ravel().tolist()), {1, 2})

  def test_tiny_pack2(self):
    ds = build_ds([(3, 2), (5, 2)])
    ds = _pack_in_pairs(ds, 1, 6, 6, pool_size=1)
    exs = next(ds.as_numpy_iterator())
    self.assertEqual(set(exs["ixs"].ravel().tolist()), {0, 2})

  def test_pool2(self):
    ds = build_ds([
      (3, 2),  # add to pool
      (5, 5),  # add to pool
      (2, 4),  # add to pool, write (5, 5) as batch
      (2, 3),  # pair with (3, 2)
    ])
    ds = _pack_in_pairs(ds, 1, 5, 5, pool_size=2).as_numpy_iterator()
    ex1 = next(ds)
    self.assertEqual(set(ex1["ixs"].ravel().tolist()), {0, 2})
    ex2 = next(ds)
    self.assertEqual(set(ex2["ixs"].ravel().tolist()), {1, 4})


if __name__ == "__main__":
  absltest.main()