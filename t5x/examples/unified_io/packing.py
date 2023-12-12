"""Batches inputs so that pairs of examples can be packed together"""
import dataclasses
import logging
from typing import Optional, List, Tuple, Sequence, Dict, Callable

import gin
import tensorflow as tf

from t5x.examples.unified_io.modality_processing import get_input_modalities, get_target_modalities


@gin.configurable
@dataclasses.dataclass
class PackingStrategy:
  """Defines how to pack data during training and handles batch-level constraints
     from the input/target encoders"""

  pack_max_len: Optional[Tuple[int, int]] = None
  """If packing, max input/target length to to"""

  pack_pool_size: int = 10
  """Pool to use when packing examples"""

  constraint_pool_size: int = 10
  """Pool to use when matching batch constraints"""

  max_to_pack: int = 2
  """Max examples to pack together"""

  @property
  def pack(self):
    return self.pack_max_len is not None

  def batch(self, ds, batch_size, drop_remainder=True, batch_constraints=None):
    if batch_constraints is None:
      batch_constraints = []
      for k, v in get_input_modalities().items():
        bound = v.get_constraints()
        if bound is not None:
          def _fn(ex):
            mask = tf.cast(ex[f"inputs/{k}/mask"], tf.bool)
            return tf.reduce_sum(tf.cast(tf.reduce_any(mask, -1), tf.int32))
          batch_bound = int(round(bound * batch_size))
          if self.pack_max_len is None:
            logging.info(f"Adding batch constraint {k}/{bound} => "
                         f"({batch_bound} per batch of {batch_size})")
          else:
            bound *= self.max_to_pack
            logging.info(f"Adding batch constraint {k}/{bound} => "
                         f"({batch_bound} per batch of {batch_size} groups of {self.max_to_pack})")
          batch_constraints.append((_fn, batch_bound))
    if self.pack:
      enc, dec = self.pack_max_len
      if self.max_to_pack == 2:
        ds = pair_examples(ds, enc, dec, self.pack_pool_size)
      else:
        raise NotImplementedError()
      if batch_constraints:
        ds = batch_with_constraints(ds, batch_size, self.constraint_pool_size, batch_constraints)
      else:
        ds = ds.batch(batch_size, drop_remainder=drop_remainder)
      return unfold(ds, batch_size, n=self.max_to_pack)
    else:
      if batch_constraints:
        return batch_with_constraints(ds, batch_size, self.constraint_pool_size, batch_constraints)
      else:
        return ds.batch(batch_size, drop_remainder=drop_remainder)


def _set_vector(vec, ix, val):
  ix = tf.expand_dims(tf.expand_dims(ix, 0), 0)
  val = tf.expand_dims(val, 0)
  return tf.tensor_scatter_nd_update(vec, ix, val)


def pad_to(x, bs):
  return tf.pad(x, [[0, bs-tf.shape(x)[0]]] + [[0, 0]]*(len(x.shape)-1))


def batch_with_constraints(
    ds: tf.data.Dataset,
    batch_size: int,
    pool_size: int,
    batch_constraints: List[Tuple[Callable, int]]
):
  batch_fns = [x[0] for x in batch_constraints]
  max_counts = tf.cast(tf.stack([x[1] for x in batch_constraints], 0), tf.int64)
  key = next(iter(ds.element_spec.keys()))
  n_const = len(batch_constraints)

  null_batch = {k: tf.zeros([0] + list(v.shape), dtype=v.dtype)
                for k, v in ds.element_spec.items()}

  empty_batch = {k: tf.zeros([batch_size]+list(v.shape), dtype=v.dtype)
                 for k, v in ds.element_spec.items()}
  empty_batch["counts"] = tf.zeros([batch_size, n_const], dtype=tf.int64)

  empty_overflow: Dict[str, tf.Tensor] = {
    k: tf.zeros([pool_size] + list(v.shape), dtype=v.dtype)
    for k, v in ds.element_spec.items()}
  empty_overflow["counts"] = tf.zeros([pool_size, n_const], dtype=tf.int64)

  def _revise_batch(state, ex):
    batch, sz, overflow, of_sz = state

    overflow: Dict[str, tf.Tensor] = overflow
    batch: Dict[str, tf.Tensor] = batch
    ex["counts"] = tf.cast(tf.stack([fn(ex) for fn in batch_fns], 0), tf.int64)
    # tf.print("Enter", ex["counts"], "batch:", batch['counts'][:sz], "pool:", overflow["counts"][:of_sz])

    total_counts = tf.reduce_sum(batch["counts"][:sz], 0)
    if tf.reduce_all(total_counts + ex["counts"] <= max_counts):
      # tf.print("Add to batch", sz)
      batch = {k: _set_vector(v, sz, ex[k]) for k, v in batch.items()}
      sz += 1
      return_batch = sz == batch_size
    else:
      # tf.print("Add to overflow", of_sz)
      overflow = {k: _set_vector(v, of_sz, ex[k]) for k, v in overflow.items()}
      of_sz += 1
      return_batch = of_sz == pool_size

    if return_batch and of_sz == 0:
      output_batch = {k: v for k, v in batch.items() if k != "counts"}
      return ({k: v*0 for k, v in batch.items()}, tf.zeros((), tf.int64), overflow, of_sz), output_batch
    elif return_batch:
      output_batch = {k: v for k, v in batch.items() if k != "counts"}
      is_valid = tf.reduce_all(tf.cumsum(overflow["counts"][:of_sz], axis=0) <= max_counts, -1)
      # tf.print("valid", is_valid)
      ix = tf.minimum(tf.argmax(is_valid) + 1, tf.minimum(of_sz, batch_size))
      # tf.print("Return batch from", ix)

      batch = {k: pad_to(v[:ix], batch_size) for k, v in overflow.items()}
      overflow = {k: pad_to(v[ix:], pool_size) for k, v in overflow.items()}
      sz = ix
      of_sz = of_sz - ix
      return (batch, sz, overflow, of_sz), output_batch
    else:
      return (batch, sz, overflow, of_sz), null_batch

  ds = ds.scan((empty_batch, tf.zeros((), tf.int64), empty_overflow, tf.zeros((), tf.int64)), _revise_batch)
  ds = ds.filter(lambda x: tf.shape(x[key])[0] > 0)
  ds = ds.map(lambda batch: {
    k: v if len(v.shape) == 0 else tf.ensure_shape(v, [batch_size] + list(v.shape)[1:])
    for k, v in batch.items()})
  return ds


def _write_to_pool(batch, pool, i_len, t_len, ix):
  output_pool = {}
  for k, v in batch.items():
    output_pool[k] = _set_vector(pool[k], ix, v)
  output_pool["meta"] = _set_vector(pool["meta"], ix, tf.stack([i_len, t_len, 1]))
  return output_pool


def unfold(ds, batch_size, n=2):
  def _flatten(batch):
    out = {}
    for k, v in batch.items():
      if len(v.shape) == 0:  # Scalars are unchanged
        out[k] = v
      elif len(v.shape) == 1:
        out[k] = v
      else:
        out[k] = tf.reshape(v, [batch_size*n] + v.shape[2:])
    return out
  return ds.map(_flatten)


def pack_in_pairs(ds: tf.data.Dataset, batch_size, max_encoder_len,
                  max_decoder_len, pool_size=5, drop_remainder=True,
                  decoder_masks=None, encoder_masks=None):
  ds = pair_examples(ds, max_encoder_len, max_decoder_len, pool_size,
                     decoder_masks=decoder_masks, encoder_masks=encoder_masks)
  ds = ds.batch(batch_size, drop_remainder=drop_remainder)
  return unfold(ds, batch_size, n=2)


def pair_examples(ds: tf.data.Dataset, max_encoder_len, max_decoder_len, pool_size=5,
                  decoder_masks=None, encoder_masks=None) -> tf.data.Dataset:
  """
  return: tf.data.Dataset that yeilds pairs of examples that can be packed together
  """
  empty = {k: tf.zeros([pool_size] + list(spec.shape), spec.dtype) for
           k, spec in ds.element_spec.items()}
  blank_example = {k: tf.zeros(list(spec.shape), spec.dtype) for
                   k, spec in ds.element_spec.items()}
  empty["meta"] = tf.zeros([pool_size, 3], dtype=tf.int32)

  if decoder_masks is None:
    decoder_masks = []
    for key, v in get_target_modalities().items():
      mask_key = f"targets/{key}/mask"
      if mask_key not in ds.element_spec:
        raise ValueError(f"All targets must have a mask to do packing, but missing {mask_key}")
      decoder_masks.append(f"targets/{key}/mask")

  if encoder_masks is None:
    encoder_masks = []
    for key, v in get_input_modalities().items():
      encoder_masks.append((f"inputs/{key}/mask", v.get_static_sequence_len()))

  def _build_pair(pool, batch):
    out_seq_len = tf.add_n([tf.reduce_sum(tf.cast(batch[x], tf.int32)) for x in decoder_masks])

    input_seq_lens = []
    for mask, seq_len in encoder_masks:
      mask = batch[mask]
      if seq_len:
        valid_frames = tf.reduce_any(tf.reshape(mask > 0, [mask.shape[0], -1]), -1)
        n_frames = tf.reduce_sum(tf.cast(valid_frames, tf.int32))
        input_seq_lens.append(n_frames * seq_len)
      else:
        assert len(mask.shape) == 1
        input_seq_lens.append(tf.reduce_sum(tf.cast(mask, tf.int32)))
    input_seq_len = tf.add_n(input_seq_lens)

    meta = pool["meta"]
    valid = meta[:, 2] == 1
    pool_lens = tf.reduce_sum(meta[:, :2], -1)
    paired_input_lens = input_seq_len + meta[:, 0]
    paired_target_lens = out_seq_len + meta[:, 1]
    can_be_packed = tf.logical_and(paired_input_lens <= max_encoder_len, paired_target_lens  <= max_decoder_len)
    can_be_packed = tf.logical_and(can_be_packed, valid)

    # tf.print("Enter pool",
    #          "in:", input_seq_len,
    #          "out:", out_seq_len,
    #          "lens", pool_lens, "meta", pool["meta"],
    #          "packable w/:", can_be_packed)
    if tf.logical_not(tf.reduce_any(can_be_packed)):
      # tf.print("Nothing can be packed")
      # Nothing in the pool can be paired with this example
      if tf.reduce_all(valid):
        # The pool is full
        # tf.print("The pool is full")
        if (out_seq_len + input_seq_len) >= tf.reduce_max(pool_lens):
          # Write batch, pool is unchanged
          # tf.print("Is longer than everything in the pool, writing", batch["ixs"])
          output_pair = (batch, blank_example)
          output_pool = pool
        else:
          # Write the longest example in the pool and replace with `batch`
          # tf.print("Writing the longest example in pool", ix)
          ix = tf.argmax(pool_lens)

          output_batch = {k: v[ix] for k, v in pool.items() if k not in {"meta"}}
          output_pool = _write_to_pool(batch, pool, input_seq_len, out_seq_len, ix)
          output_pair = (output_batch, blank_example)
      else:
        ix = tf.where(tf.logical_not(valid))[0][0]
        # Add to the pool and write nothing
        # tf.print("Writing to pool at:", ix,  "write nothing")
        output_pool = _write_to_pool(batch, pool, input_seq_len, out_seq_len, ix)
        output_pair = (blank_example, blank_example)

    else:
      # pair with longest example in the pool that matches
      # tf.print("Pairing batch with", ix)
      ix = tf.argmax(pool_lens * (2*tf.cast(can_be_packed, tf.int32) - 1))
      other = {k: v[ix] for k, v in pool.items() if k != "meta"}
      output_pool = dict(pool)
      output_pool["meta"] = _set_vector(pool["meta"], ix, tf.constant([0, 0, 0]))
      output_pair = (batch, other)

    p1, p2 = output_pair
    return output_pool, {k: tf.stack([p1[k], p2[k]], 0) for k in batch}

  # Convert dataset into packable pairs
  ds = ds.scan(empty, _build_pair)

  # Remove blank pairs
  ds = ds.filter(lambda batch: tf.reduce_any(
    tf.stack([tf.reduce_any(tf.cast(batch[x], tf.bool)) for x in decoder_masks])))
  return ds
