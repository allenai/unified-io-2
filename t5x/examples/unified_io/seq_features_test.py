import dataclasses

import numpy as np
from absl.testing import parameterized

from t5x.examples.unified_io import seq_features as utils
import jax.numpy as jnp

from t5x.utils import flatten_lists


@dataclasses.dataclass
class MockSequence:
  val1: jnp.ndarray
  val2: jnp.ndarray

  @property
  def seq_len(self):
    return self.val1.shape[1]


utils.register_pytree_node_flat_dataclass(MockSequence)


class TestSequenceFeatures(parameterized.TestCase):

  def pack_rel_atten_horizontal_slow(self, seqs, masks, out_len):
    bs, n_heads = seqs[0].shape[:2]
    out = np.zeros((bs, n_heads, out_len, out_len))
    offsets = np.zeros((bs,), dtype=np.int32)
    for seq, mask in zip(seqs, masks):
      for batch_ix in range(bs):
        l = mask[batch_ix].sum()
        if l > 0:
          o = offsets[batch_ix]
          out[batch_ix, :, o:o+l, o:o+l] = seq[batch_ix][:, mask[batch_ix]][:, :, mask[batch_ix]]
          offsets[batch_ix] += l
    return out

  def concat_and_pack_slow(self, seqs, masks, out_len):
    bs = masks[0].shape[0]
    if len(seqs[0].shape) == 4:
      return self.pack_rel_atten_horizontal_slow(seqs, masks, out_len)
    elif len(seqs[0].shape) == 3:
      dim = seqs[0].shape[-1]
      out = np.zeros((bs, out_len, dim), dtype=seqs[0].dtype)
    else:
      out = np.zeros((bs, out_len), dtype=seqs[0].dtype)

    offsets = np.zeros((bs,), dtype=np.int32)
    for seq, mask in zip(seqs, masks):
      for batch_ix in range(bs):
        l = mask[batch_ix].sum()
        if l > 0:
          o = offsets[batch_ix]
          if len(seq.shape) <= 1:
            out[batch_ix, o:o+l] = seq
          else:
            out[batch_ix, o:o+l] = seq[batch_ix, mask[batch_ix]]
          offsets[batch_ix] += l
    return out

  def test_pack_output_horizontal_random(self):
    self._test_random(False)

  def test_pack_input_horizontal_random(self):
    self._test_random(True)

  def _test_random(self, input_sequences=True):
    for it in range(0, 5):
      rng = np.random.RandomState(2 + it*10 + 100*input_sequences)
      n_seq = rng.randint(1, 4)
      seq_lens = rng.randint(1, 8, size=(n_seq, ))
      bs, n_heads, dim = np.random.randint(1, 4, size=(3,))

      inputs = []
      for seq_len in seq_lens:
        mask = rng.random((bs, seq_len)) > 0.3
        if input_sequences:
          inputs.append(utils.InputSequence(
            rng.random((bs, seq_len, dim)),
            mask,
            rng.random((bs, n_heads, seq_len, seq_len)),
          ))
        else:
          inputs.append(utils.TargetSequence(
            rng.random((bs, seq_len, 1)).astype(np.float32),
            rng.randint(0, 1000, (bs, seq_len), dtype=jnp.int32),
            np.array(rng.randint(20), dtype=jnp.int32),
            mask,
            rng.random((bs, n_heads, seq_len, seq_len)),
            rng.randint(0, 1000, (bs, seq_len), dtype=jnp.int32),
          ))
      lens = np.stack([x.mask.sum(-1) for x in inputs], -1).sum(-1)
      out_len = lens.max()
      actual = utils.pack_horizontally(inputs, out_len)
      masks = [x.mask for x in inputs]
      for k, v in dataclasses.asdict(actual).items():
        if k == "mask":
          continue
        else:
          tensors = [getattr(x, k) for x in inputs]
          if all(x is None for x in tensors):
            self.assertIsNone(getattr(actual, k))
          else:
            expected = self.concat_and_pack_slow(tensors, masks, out_len)
            self.assertTrue(np.allclose(expected, getattr(actual, k)))

  def test_pack_unpack_small(self):
    embed1 = [
      [1, 0, 0],
      [0, 12, 13],
    ]
    embed2 = [
      [0, 1],
      [0, 15],
    ]
    inputs = []
    for emb in [embed1, embed2]:
      emb = jnp.array(emb, jnp.float32)[:, :, None]
      inputs.append(utils.InputSequence(
        emb,
        emb[:, :, 0] != 0
      ))
    out = utils.pack_horizontally(inputs, 5)
    split = utils.split_and_unpack(out.embed, [x.mask for x in inputs])
    self.assertTrue(jnp.allclose(split[0][:, :, 0], np.array(embed1)))
    self.assertTrue(jnp.allclose(split[1][:, :, 0], np.array(embed2)))

  def test_nomask_roll_matrix(self):
    starts = jnp.array([4, 0, 2])
    mat1 = utils.build_nomask_roll_marix(5, 15, starts)
    mat2 = utils.build_unsorted_roll_marix(jnp.ones((3, 5), dtype=jnp.bool_), 15, starts)
    self.assertTrue(jnp.all(mat1 == mat2))

  def test_pack_input_horizontal_small(self):
    embed1 = [
      [1, 0, 0],
      [3, 4, 5],
      [0, 0, 0]
    ]
    embed2 = [
      [11, 0],
      [21, 22],
      [0, 0]
    ]

    inputs = []
    for emb in [embed1, embed2]:
      emb = jnp.array(emb, jnp.float32)[:, :, None]
      inputs.append(utils.InputSequence(
        emb,
        emb[:, :, 0] != 0
      ))
    out = utils.pack_horizontally(inputs, 5)
    expected = [
      [1, 11, 0, 0, 0],
      [3, 4, 5, 21, 22],
      [0, 0, 0, 0, 0],
    ]
    actual = out.embed[:, :, 0]
    self.assertTrue(jnp.all(actual == jnp.array(expected)))

  def test_concat(self):
    actual = utils.concat_sequences([
      MockSequence(jnp.ones((2, 2)), None),
      MockSequence(jnp.ones((2, 2)), jnp.ones((2, 2)))
    ])
    assert np.all(actual.val1 == 1)
    assert np.all(actual.val2[:, :2] == 0)
    assert np.all(actual.val2[:, 2:] == 1)

    actual = utils.concat_sequences([
      MockSequence(jnp.ones((2, 2)), jnp.array(5)),
      MockSequence(jnp.ones((1, 2))*2, jnp.ones((2, 2)))
    ])
    assert np.all(actual.val1[:, :2] == 1)
    assert np.all(actual.val1[:, 2:] == 2)
    assert np.all(actual.val2[:, :2] == 5)
    assert np.all(actual.val2[:, 2:] == 1)

  def test_pack_and_unpack(self):
    for seq_lens, p in [
      ([5, 5], 0.3),
      ([1, 1], 0.2),
      ([3], 0.3),
      ([5, 7, 4, 1, 11], 0.2),
      ([20, 20, 20], 0.9),
      ([20, 20, 20], 0.1),
    ]:
      bs, dim = 2, 2
      inputs = []
      rng = np.random.RandomState(6234 + sum(seq_lens))
      for seq_len in seq_lens:
        inputs.append(utils.InputSequence(
          rng.random((bs, seq_len, dim)),
          rng.random((bs, seq_len)) > p,
          None,
          position_embed=rng.random((bs if rng.random() < 0.5 else 1, seq_len, dim)),
        ))
      max_len = jnp.stack([x.mask.sum(-1) for x in inputs], -1).sum(-1).max()
      # Test pack->unpack
      packed = utils.pack_horizontally(inputs, max_len)
      unpacked = utils.split_and_unpack(packed.embed, [x.mask for x in inputs])
      for embed, seq in zip(unpacked, inputs):
        m = seq.mask[:, :, None]
        self.assertTrue(jnp.allclose(embed*m, seq.embed*m))

      # Test fold->pack->unpack->unfold
      folded = [utils.fold_sequence(x, n=2) for x in inputs]
      packed = utils.pack_horizontally(folded, max_len*2)
      unpacked = utils.split_and_unpack(packed.embed, [x.mask for x in folded])
      for embed, seq in zip(unpacked, inputs):
        embed = jnp.reshape(embed, [bs, embed.shape[1]//2, -1])
        m = seq.mask[:, :, None]
        self.assertTrue(jnp.allclose(embed*m, seq.embed*m))


