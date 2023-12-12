"""Defines `InputSequence` and `TargetSequence` types that contain low-features for input/target
 modality-agnostic sequences, and utility method to manipulate them.

The utility methods in this file work with either kind of sequence as input
"""

import dataclasses
from dataclasses import dataclass
from typing import Optional, List, Union, Tuple, TypeVar, Sequence, Iterable

import gin
import jax
import jax.numpy as jnp
import numpy as np


def register_pytree_node_flat_dataclass(cls):
  """Register flat dataset as a pytree note so jax tree method work on it """
  def flatten(x):
    x = dataclasses.asdict(x)
    return x.values(), x.keys()

  def unflatten(d, children):
    return cls(**{k: v for k, v in zip(d, children)})

  jax.tree_util.register_pytree_node(cls, flatten, unflatten)
  return cls


@register_pytree_node_flat_dataclass
@dataclass
class TargetSequence:
  """Target sequence we can train a decoder to predict"""

  input_embedding: jnp.ndarray
  """Input embeddings to the decoder"""

  position_id: jnp.ndarray
  """Int position ids or embedding"""

  modality_id: jnp.ndarray
  """Modality ids's of the tokens, can be a scalar if all the same"""

  mask: Optional[jnp.ndarray]
  """Mask of valid tokens"""

  attn_pattern_mask: Optional[jnp.ndarray] = None
  """[batch, n_heads, seq_len, seq_len] of relative attention bias"""

  target_tokens: Optional[jnp.ndarray] = None
  """Target tokens used to compute the loss"""

  subsegments: Optional[jnp.ndarray] = None
  """ids of targets that should be independently predicted from the encoding of one example"""

  segment_ids: Optional[jnp.ndarray] = None
  """If packed, an example id for each token"""

  loss_mask: Optional[jnp.ndarray] = None
  """Mask of tokens to use when computing the loss"""

  @staticmethod
  def empty(batch, trm_len, seq_len, n_heads, dtype, modality_id, embed_dim, mask_dtype=jnp.int32):
    return TargetSequence(
      jnp.zeros((batch, trm_len, embed_dim), jnp.float32),
      jnp.zeros((1, trm_len, embed_dim), jnp.float32),
      jnp.array(modality_id, dtype=jnp.int32),
      jnp.zeros((batch, trm_len), mask_dtype),
      jnp.zeros((1, trm_len, embed_dim), dtype),
      jnp.zeros((batch, seq_len), jnp.int32),
      loss_mask=jnp.zeros((batch, seq_len), mask_dtype),
      subsegments=None
    )

  @property
  def seq_len(self):
    return self.input_embedding.shape[1]

  @property
  def batch_size(self):
    return self.input_embedding.shape[0]

  def __post_init__(self):
    if all(x is None or isinstance(x, str) for x in dataclasses.asdict(self).values()):
      # jax might build this sequence with strings to display an error message
      return
    bs, seq_len = self.input_embedding.shape[:2]

    if self.position_id is not None:
      assert self.position_id.shape[:2] in [(1, seq_len), (bs, seq_len)]

    assert self.modality_id.shape in [(), (1, seq_len), (bs, seq_len)]
    assert self.modality_id.dtype == jnp.int32

    if self.target_tokens is not None:
      assert self.target_tokens.shape == (bs, seq_len)
      assert self.target_tokens.dtype == jnp.int32

    if self.mask is not None:
      assert self.mask.shape == (bs, seq_len)
      assert self.mask.dtype == jnp.int32 or self.mask.dtype == jnp.bool_

    if self.attn_pattern_mask is not None:
      assert self.attn_pattern_mask.shape[0] in [1, bs]

    if self.subsegments is not None:
      assert self.subsegments.shape == (bs, seq_len)
      assert self.subsegments.dtype == jnp.int32

    if self.segment_ids is not None:
      assert self.segment_ids.shape == (bs, seq_len)
      assert self.segment_ids.dtype == jnp.int32

  def get_all_subsegments(self):
    """Assigns an id to each token such that each tokens should only attend to tokens
    with the same ID"""
    subsegments = [self.subsegments, self.segment_ids,
                   None if len(self.modality_id.shape) <= 1 else self.modality_id]
    all_subsegments = None
    for part in subsegments:
      if part is None:
        continue
      if all_subsegments is None:
        all_subsegments = part
        continue
      all_subsegments = all_subsegments*(part.max()+1) + part
    return all_subsegments


@register_pytree_node_flat_dataclass
@dataclass
class InputSequence:
  """Input sequence we can encode with an Encoder"""

  embed: jnp.ndarray
  """Token input embedding"""

  mask: Optional[jnp.ndarray]
  """Mask over valid time steps"""

  attn_pattern_mask: Optional[jnp.ndarray]=None
  """[batch, n_heads, seq_len, seq_en] relative attention bias"""

  segment_ids: Optional[jnp.ndarray]=None
  """If packed, an example id for each token"""

  position_embed: Optional[jnp.ndarray]=None
  """Positional bias embedding"""

  @property
  def seq_len(self):
    return self.embed.shape[1]

  @property
  def batch_size(self):
    return self.embed.shape[0]

  @staticmethod
  def empty(bs, seq_len, cfg) -> 'InputSequence':
    return InputSequence(
      jnp.zeros((bs, seq_len, cfg.emb_dim), dtype=cfg.dtype),
      jnp.zeros((bs, seq_len), dtype=jnp.int32),
      attn_pattern_mask=None,
      position_embed=jnp.zeros((bs, seq_len, cfg.emb_dim), dtype=cfg.dtype),
    )

  def __post_init__(self):
    if all(x is None or isinstance(x, str) for x in dataclasses.asdict(self).values()):
      # jax might build this pytreenode with strings to display an error message
      return
    assert jnp.issubdtype(self.embed.dtype, jnp.floating)
    assert len(self.embed.shape) == 3
    bs, seq_len = self.embed.shape[:2]

    if self.position_embed is not None:
      assert jnp.issubdtype(self.position_embed.dtype, jnp.floating)
      assert len(self.position_embed.shape) == 3
      assert self.position_embed.shape[:2] in [(bs, seq_len), (1, seq_len)]
    if self.mask is not None:
      assert self.mask.dtype == jnp.int32 or self.mask.dtype == jnp.bool_
      assert self.mask.shape == (bs, seq_len)
    if self.attn_pattern_mask is not None:
      # assert jnp.issubdtype(self.attn_pattern_mask.dtype, jnp.floating)
      assert len(self.attn_pattern_mask.shape) == 4
      # dim 1 is the number of heads
      assert self.attn_pattern_mask.shape[0] == bs
      assert self.attn_pattern_mask.shape[2:] == (seq_len, seq_len)
    if self.segment_ids is not None:
      assert self.segment_ids.dtype == jnp.int32
      assert self.segment_ids.shape == (bs, seq_len)


SequenceFeature = TypeVar("SequenceFeature", TargetSequence, InputSequence)


def concat_rel_atten(rel_attens: Iterable[jnp.ndarray], total_len: int) -> jnp.ndarray:
  """Concat relative attention matrices, attention between elements in different
  sequences values will be zero"""
  on = 0
  rel_bias_list = []
  for rel_atten in rel_attens:
    n = rel_atten.shape[-1]
    rel_bias_list.append(
      jnp.pad(rel_atten, [(0, 0), (0, 0), (on, total_len-on-n), (0, 0)]))
    on += n
  rel_atten = jnp.concatenate(rel_bias_list, -1)
  return rel_atten


def sum_sequences(seqs: List):
  """Sums sequence of the same shapes, None is treated as zero"""

  def _sum_optional(*vals):
    out = None
    for v in vals:
      if v is None:
        continue
      if out is None:
        out = v
      else:
        out += v
    return out

  return jax.tree_util.tree_map(_sum_optional, *seqs, is_leaf=lambda x: x is None)


def concat_sequences(seqs: List, seq_lens: Optional[List[int]]=None):
  """Concats along the sequence dimension (i.e., horizontally)"""

  if seq_lens is None:
    seq_lens = [x.seq_len for x in seqs]
  seqs = [expand_scalars(x, sl) for x, sl in zip(seqs, seq_lens)]

  def _concat_optional(*args):
    if all(x is None for x in args):
      return None

    max_bs = max(x.shape[0] for x in args if x is not None)
    full_sized = [x for x in args if (x is not None and x.shape[0] == max_bs)]
    shape = list(full_sized[0].shape)

    if len(full_sized) != len(args):
      # Replace scalar/None values with blank/full values
      padded_args = []
      for ix, x in enumerate(args):
        if x is not None and x.shape[0] == max_bs:
          padded_args.append(x)  # Full sized
          continue

        if x is not None and x.shape[0] != max_bs:
          assert x.shape[0] == 1  # broadcasts the batch dim, tile to max_bs
          padded_args.append(jnp.tile(x, [max_bs] + [1]*(len(x.shape)-1)))
          continue

        assert x is None  # replace with zero array of the correct shape
        arg_shape = list(shape)
        arg_shape[0] = max_bs
        if len(shape) <= 3:
          arg_shape[1] = seq_lens[ix]
        elif len(shape) == 4:
          arg_shape = arg_shape[:2] + [seq_lens[ix], seq_lens[ix]]

        padded_args.append(jnp.zeros(arg_shape, full_sized[0].dtype))
      args = padded_args

    if len(shape) == 4:
      return concat_rel_atten(args, sum(seq_lens))
    else:
      return jnp.concatenate(args, 1)

  return jax.tree_util.tree_map(_concat_optional, *seqs, is_leaf=lambda x: x is None)


@gin.configurable()
def multiply_seq_dim(seq: SequenceFeature, mat: jnp.array, unroll=False):
  """Multiply the seq dimension by the [batch, seq_dim, out_dim] matrix"""

  def _multiply(v):
    if v is None:
      return v
    elif len(v.shape) <= 1:
      return (mat*v).sum(1)
    elif len(v.shape) == 2:
      return jnp.einsum("bs,bso->bo", v, mat)
    elif len(v.shape) == 3:
      if unroll and jax.dtypes.issubdtype(v.dtype, jnp.floating):
        # In rare cases (v4 TPU with 4 partitions) backprop through this batch matmult seems to
        # cause runtimes errors, allow working around by unbatching the matmult
        mat_t = jnp.transpose(mat, [0, 2, 1])
        parts = []
        for ix in range(v.shape[0]):
          parts.append(jnp.matmul(mat_t[ix], v[ix]))
        return jnp.stack(parts)
      else:
        return jnp.einsum("bsd,bso->bod", v, mat)
    elif len(v.shape) == 4:
      return jnp.einsum("bdsx,bsz,bxy->bdzy", v, mat, mat)
    else:
      raise RuntimeError()
  return jax.tree_util.tree_map(_multiply, seq)


def expand_scalars(seq, seq_len=None):
  """Expand scalars into [1, seq_len] arrays"""
  if seq_len is None:
    seq_len = seq.seq_len

  def _replace(val):
    if val is None:
      return None
    elif len(val.shape) <= 1:
      return jnp.full((1, seq_len), val)
    else:
      return val
  return jax.tree_util.tree_map(_replace, seq)


def build_packing_matrix(mask, out_len):
  ixs = jnp.cumsum(mask) - 1  # [batch*frames]
  mat = jnp.expand_dims(ixs, 0) == jnp.expand_dims(jnp.arange(out_len, dtype=ixs.dtype), 1)
  return mat


def build_packing_matrix_from_pos(pos_id, in_lens):
  mat = jnp.expand_dims(pos_id, 0) == jnp.expand_dims(jnp.arange(in_lens, dtype=pos_id.dtype), 1)
  return mat


def build_unsorted_roll_marix(mask, out_len, starts=None):
  c1 = jnp.cumsum(mask, 1) - 1
  if starts is not None:
    c1 += starts[:, None]
  c2 = jnp.arange(out_len)[None, None, :]
  return (c1[:, :, None] == c2) & jnp.expand_dims(mask, 2)


def build_nomask_roll_marix(input_len, out_len, starts):
  c1 = jnp.arange(input_len)[None, :] + starts[:, None]
  c2 = jnp.arange(out_len)[None, None, :]
  return c1[:, :, None] == c2


def build_inv_unsorted_roll_marix(mask, out_len, starts=None):
  c1 = jnp.cumsum(mask, 1) - 1
  if starts is not None:
    c1 += starts[:, None]
  c2 = jnp.arange(out_len)
  return (c1[:, None, :] == c2[None, :, None]) & jnp.expand_dims(mask, 1)


def build_inv_nomask_roll_marix(starts, input_len, out_len):
  c1 = jnp.arange(input_len)[None, :] + starts[:, None]
  c2 = jnp.arange(out_len)
  return c1[:, None, :] == c2[None, :, None]


def split_and_unpack(embed, input_masks: List[Union[jnp.ndarray, int]]):
  """Assuming `embed` is from a horizontally packed set of sequence with masks `input_masks`,
  unpack it into separate tensors."""
  out = []
  offset = jnp.zeros(embed.shape[0], jnp.int32)
  for mask in input_masks:
    if isinstance(mask, int):
      mat = build_nomask_roll_marix(offset, mask, embed.shape[1])
      offset += mask
    else:
      mat = build_inv_unsorted_roll_marix(mask, embed.shape[1], offset)
      offset += mask.sum(-1)
    out.append(multiply_seq_dim(embed, mat))
  return out


def split_sequence_dim(array: jnp.array, seq_lens: List[int]) -> List[jnp.array]:
  """Undoes concat_sequences([fold_sequence(x) for x in list])"""
  split_points = np.cumsum(seq_lens[:-1])
  parts = jnp.split(array, split_points, axis=1)
  return parts


def add_segments(sequence: SequenceFeature) -> SequenceFeature:
  segments = jax.lax.broadcasted_iota(jnp.int32, (sequence.batch_size, sequence.seq_len), 0)
  return dataclasses.replace(sequence, segment_ids=segments)


def split_array(v: jnp.ndarray, n=2) -> List[jnp.ndarray]:
  # Take every nth element, for n=2 same as v[::2], v[1::2], but
  # more memory efficient since jax currently handles strided indexing poorly
  # (https://github.com/google/jax/issues/11457)
  r = len(v.shape)
  out = []
  for i in range(n):
    out.append(jax.lax.slice(v, [i] + [0]*(r-1), v.shape, [n]+[1]*(r-1)))
  return out


def fold_sequence(sequence: SequenceFeature, n=2) -> SequenceFeature:
  """Fold a sequence so it is n/2 the batch size but n times the sequence length

  Segment ids will be added if not already present to track the original batch
  index each token came from.

  Each examples wll be merged with the subsequence example
  """
  def _fold(v):
    if v is None or len(v.shape) <= 1:
      return v
    elif len(v.shape) == 4:
      if v.shape[0] == 1:
        return concat_rel_atten([v]*n, v.shape[2]*n)
      else:
        # Explicitly concat every example with every other example
        return concat_rel_atten(split_array(v, n), v.shape[2]*n)
    elif len(v.shape) <= 3:
      bs, seq_len = v.shape[:2]
      if bs == 1:
        # `v`` is broadcasted over the batch dimension, so just need to tile
        return jnp.tile(v, [1, n] + [1]*len(v.shape[2:]))
      else:
        return v.reshape(*([bs//n, seq_len*n] + list(v.shape[2:])))
    else:
      raise NotImplementedError()
  assert sequence.batch_size % n == 0
  if sequence.segment_ids is None:
    sequence = add_segments(sequence)
  return jax.tree_util.tree_map(_fold, sequence)


def pack_horizontally(sequences: List[SequenceFeature], output_seq_len: int) -> SequenceFeature:
  """Concats `sequences` along the sequence dimension, put packs the elements
  so that masked values are all placed at the end.

  Truncates the output to `output_seq_len`, client should ensure the total
  number of non-masked values is less then `output_seq_len` to avoid losing data
  """
  remapped = []
  offset = jnp.zeros((sequences[0].mask.shape[0],))
  for seq in sequences:
    if seq.mask is not None:
      mat = build_unsorted_roll_marix(seq.mask, output_seq_len, offset)
    else:
      mat = build_nomask_roll_marix(seq.seq_len, output_seq_len, offset)

    remapped.append(multiply_seq_dim(dataclasses.replace(seq, mask=None), mat))
    if seq.mask is not None:
      offset += seq.mask.sum(-1)
    else:
      offset += seq.seq_len

  mask = jnp.arange(output_seq_len)[None, :] < offset[:, None]
  out = sum_sequences(remapped)
  out = dataclasses.replace(out, mask=mask)
  return out


def fold_and_split_sequence(sequence: SequenceFeature, n=2) -> Tuple[SequenceFeature, ...]:
  """Split sequence into n parts that contain every nth example

  Segment ids will be added if not already present to track the original batch indices
  """
  def _split(v):
    if v is None or len(v.shape) <= 1 or v.shape[0] == 1:
      return [v]*n
    else:
      return split_array(v, n)
  if sequence.segment_ids is None:
    sequence = add_segments(sequence)
  data = jax.tree_util.tree_map(_split, dataclasses.asdict(sequence), is_leaf=lambda x: x is None)
  # noinspection PyTypeChecker
  return tuple(sequence.__class__(**{k: v[i] for k, v in data.items()}) for i in range(n))

