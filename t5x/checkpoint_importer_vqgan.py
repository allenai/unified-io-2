# Copyright 2022 The T5X Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""T5 Checkpoint Importer."""

import abc
import asyncio
from concurrent.futures import thread
import re
from typing import Any, Callable, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

from t5x import optimizers as optim
from flax import traverse_util
import jax
from jax import numpy as jnp
import numpy as np
import tensorflow as tf
import tensorstore as ts
from t5x import state_utils
from t5x import checkpoints

class LazyArray(abc.ABC):
  """Lazily and asynchronously loads an array."""

  def __init__(self, shape: Sequence[int], dtype: jnp.dtype,
               get_fn: Callable[[], np.ndarray]):
    self._shape = tuple(shape)
    self._dtype = jnp.dtype(dtype)
    self._get_fn = get_fn

  @property
  def shape(self) -> Tuple[int, ...]:
    return self._shape

  @property
  def dtype(self) -> jnp.dtype:
    return self._dtype

  @property
  def nbytes(self) -> int:
    return np.prod(self._shape) * self._dtype.itemsize

  def astype(self, dtype: np.dtype) -> 'LazyArray':
    return type(self)(self._shape, dtype, self._get_fn)  # pytype: disable=not-instantiable

  @abc.abstractmethod
  def get_async(self) -> asyncio.Future:
    pass

  @abc.abstractmethod
  def get(self) -> np.ndarray:
    pass

  def __repr__(self):
    return f'{type(self).__name__}(shape={self.shape}, dtype={self.dtype})'


# TODO(brianlester): The choice between using a `LazyTreadPoolArray` or a
# `LazyAwaitableArray` is dependent on if the user provided `get_fn` is blocking
# or async respectively, if we can detect which it is, we can automatically
# proxy to the correct subclass. We cannot detect of `get_fn` is a lambda that
# wraps an async call so this isn't possible yet. Add this dispatch once we are
# able to detect that, python3.8+ can detect async for partial'ed functions but
# not lambdas.
class LazyThreadPoolArray(LazyArray):
  """Lazily and asynchronously loads an array when the `get_fn` blocks."""

  # Uses a global threadpool to enable asynchronous loading.
  executor = thread.ThreadPoolExecutor()

  def get_async(self) -> asyncio.Future:
    return asyncio.wrap_future(self.executor.submit(self.get))

  def get(self) -> np.ndarray:
    arr = self._get_fn()
    if arr.dtype != self.dtype:
      arr = arr.astype(self.dtype)
    return arr


class LazyAwaitableArray(LazyArray):
  """Lazily and asynchronously loads an array when the `get_fn` is async.

  Note:
    The synchronous load method `.get` requires the asyncio event loop and
    calling `.run_until_complete`. This is not supported when the event loop is
    already running (for example, from inside another async function).

  Note:
    Currently, this class has a few helper methods for creating a
    LazyAwaitableArray when the input could be either an array, or a TensorStore
    spec. Most people use async code when dealing with TensorStore so the
    classmethods have been placed here. When someone eventually uses a blocking
    function to read from TensorStore they can be moved to the LazyArray base
    class.
  """

  def get_async(self) -> asyncio.Future:

    async def _get_and_cast():
      # Pytype has a false positive here, where it treats our _get_fn (_read_ts
      # in this case) as having a return type of `np.ndarray` instead of
      # wrapping it in an Awaitable. Related to this bug
      # https://github.com/google/pytype/issues/527
      arr = await self._get_fn()  # pytype: disable=bad-return-type
      if arr.dtype != self.dtype:
        arr = arr.astype(self.dtype)
      return arr

    return asyncio.ensure_future(_get_and_cast())

  def get(self) -> np.ndarray:
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(self.get_async())

  @classmethod
  def from_tensor_store_spec(
      cls,
      ts_spec: ts.Spec,
      get_fn: Callable[[], np.ndarray],
      dtype: Optional[jnp.dtype] = None) -> 'LazyAwaitableArray':
    """Create a LazyAwaitableArray based on a tensorstore.Spec."""
    ts_spec = ts_spec.to_json()
    shape = ts_spec['metadata']['shape']
    if dtype is None:
      dtype = jnp.dtype(ts_spec['dtype'])
    else:
      dtype = jnp.dtype(dtype)
    # v2 T5X checkpoints use uint16 as the TensorStore datatype and then store
    # the bfloat16 bytes as in in the 16 bytes uint16 has (no actual cast). When
    # When reading the dtype from the TensorStore, if we keep the dtype of these
    # v2 checkpoints as np.uint16 then the _get_fn (which has a possible cast to
    # support the `restore_dtype` parameter for the checkpointer) will actually
    # cast the bfloat16 values to uint16, generally resulting in an array of all
    # zeros. This check avoid the actual cast to uint16 by replacing the dtype.
    if dtype == np.uint16:
      dtype = jnp.bfloat16
    return cls(shape, dtype, get_fn)

  @classmethod
  def from_array(cls,
                 array: np.ndarray,
                 get_fn: Callable[[], np.ndarray],
                 dtype: Optional[jnp.dtype] = None) -> 'LazyAwaitableArray':
    """Create a LazyAwaitableArray based on an array or python number."""
    if dtype is None:
      dtype = array.dtype
    else:
      dtype = jnp.dtype(dtype)
    return cls(array.shape, dtype, get_fn)

  @classmethod
  def from_tensor_store_spec_or_array(
      cls,
      maybe_ts_spec: Union[ts.Spec, np.ndarray],
      get_fn: Callable[[], np.ndarray],
      dtype: Optional[jnp.dtype] = None) -> 'LazyAwaitableArray':
    """Create a LazyAwaitableArray based on an array or a tensorstore.Spec."""
    if isinstance(maybe_ts_spec, ts.Spec):
      return cls.from_tensor_store_spec(maybe_ts_spec, get_fn, dtype=dtype)
    return cls.from_array(maybe_ts_spec, get_fn)


class CheckpointTranslator:
  """Utility class for defining mapping rules from one flatdict to another.

  We assume a checkpoint is loaded as a dictionary with flattened keys of the
  form:  'name0/name1/name2/.../nameN'

  A rule is added with the 'add' decorator, which takes a regex matching rule
  and wraps a conversion function, feeding it (opts, key, val, **regex_groups)
  where opts is a dict containing apply-time keyword options for use by the
  conversion functions.
  """

  def __init__(self):
    self.rules = []

  def add(self, pattern):
    """Adds a new keyval conversion rule.

    Args:
      pattern: regex with capture groups for matching given sets of model
        variables.  We terminate all regexes with '$' to force complete matches.

    Returns:
      Translation function decorator for associating with the provided
      pattern.
    """

    def register_translation_fn_decorator(fn):
      # We force a complete match by adding end-of-string match.
      self.rules.append((re.compile(pattern + '$'), fn))
      return fn

    return register_translation_fn_decorator

  def apply(self, flatdict, **opts):
    """Applies rules to a flattened dictionary.

    Args:
      flatdict: flat-key dictionary of variables.
      **opts: additional config options for translation rules supplied at
        application time.

    Returns:
      Checkpoint data with translated key/values in flat-key dict format.
    """
    new_dict = {}
    unmatched = {}
    for k, v in flatdict.items():
      if '_CHECKPOINTABLE_OBJECT_GRAPH' in k:
        continue
      k = k.replace('.ATTRIBUTES/VARIABLE_VALUE', '')
      matched = False
      for rule_pat, rule_fn in self.rules:
        if rule_pat.match(k):
          groups = rule_pat.match(k).groups()
          new_k, new_v = rule_fn(opts, k, v, *groups)
          if new_k is not None:
            new_dict[new_k] = new_v
          matched = True
          break
      if not matched:
        print(k)
        unmatched[k] = v
        import pdb
        pdb.set_trace()
    # We force every key-value pair in checkpoint to have a rule associated with
    # it.

    if unmatched:
      raise ValueError('Unmapped tensor keys exist: %s' % unmatched)

    return new_dict


# Create a translation rule set for importing T5 & T5.1.1 model checkpoints.
# -----------------------------------------------------------------------------
t5_importer = CheckpointTranslator()

# Name mappings.
SLOT_MAP = {'.ATTRIBUTES/VARIABLE_VALUE': 'v'}
TOWER_MAP = {'transformer': 'decoder'}
NORM_MAP = {'gamma':'scale', 'beta': 'bias'}
@t5_importer.add(r'save_counter/(.*)')
def global_step(opts, key, val, slot):
  return 'state/step', val.astype(np.int32).get() if isinstance(
      val, LazyArray) else val

@t5_importer.add(r'model/quantize/embedding/(.*)')
def shared_embeddings(opts, key, val, slot):
  prefix = 'state/param_states' if slot else 'target'
  suffix = '/' + SLOT_MAP[slot] if slot else ''
  newkey = f'{prefix}/discrete_vae/quantize/embedding{suffix}'
  return newkey, val

@t5_importer.add(r'model/quant_conv/(\w+)/(.*)')
def shared_embeddings(opts, key, val, module, slot):
  prefix = 'state/param_states' if slot else 'target'
  suffix = '/' + SLOT_MAP[slot] if slot else ''
  newkey = f'{prefix}/discrete_vae/quant_conv/{module}{suffix}'
  return newkey, val

@t5_importer.add(r'model/post_quant_conv/(\w+)/(.*)')
def shared_embeddings(opts, key, val, module, slot):
  prefix = 'state/param_states' if slot else 'target'
  suffix = '/' + SLOT_MAP[slot] if slot else ''
  newkey = f'{prefix}/discrete_vae/post_quant_conv/{module}{suffix}'
  return newkey, val

@t5_importer.add(r'model/(\w+)/norm_out/(\w+)/(.*)')
def shared_embeddings(opts, key, val, enc_dec, module, slot):
  prefix = 'state/param_states' if slot else 'target'
  suffix = '/' + SLOT_MAP[slot] if slot else ''
  newkey = f'{prefix}/discrete_vae/{enc_dec}/norm_out/{NORM_MAP[module]}{suffix}'
  return newkey, val

@t5_importer.add(r'model/(\w+)/conv_out/(\w+)/(.*)')
def shared_embeddings(opts, key, val, enc_dec, module, slot):
  prefix = 'state/param_states' if slot else 'target'
  suffix = '/' + SLOT_MAP[slot] if slot else ''
  newkey = f'{prefix}/discrete_vae/{enc_dec}/conv_out/{module}{suffix}'
  return newkey, val

@t5_importer.add(r'model/(\w+)/conv_in/(\w+)/(.*)')
def shared_embeddings(opts, key, val, enc_dec, module, slot):
  prefix = 'state/param_states' if slot else 'target'
  suffix = '/' + SLOT_MAP[slot] if slot else ''
  newkey = f'{prefix}/discrete_vae/{enc_dec}/conv_in/{module}{suffix}'
  return newkey, val

@t5_importer.add(r'model/(\w+)/mid_block_(\d+)/conv(\d+)/(\w+)/(.*)')
def shared_embeddings(opts, key, val, enc_dec, block_num, conv_num, module, slot):
  prefix = 'state/param_states' if slot else 'target'
  suffix = '/' + SLOT_MAP[slot] if slot else ''
  newkey = f'{prefix}/discrete_vae/{enc_dec}/mid_block_{block_num}/conv{conv_num}/{module}{suffix}'
  return newkey, val

@t5_importer.add(r'model/(\w+)/mid_block_(\d+)/norm(\d+)/(\w+)/(.*)')
def shared_embeddings(opts, key, val, enc_dec, block_num, conv_num, module, slot):
  prefix = 'state/param_states' if slot else 'target'
  suffix = '/' + SLOT_MAP[slot] if slot else ''
  newkey = f'{prefix}/discrete_vae/{enc_dec}/mid_block_{block_num}/norm{conv_num}/{NORM_MAP[module]}{suffix}'
  return newkey, val

@t5_importer.add(r'model/(\w+)/mid_attn_(\d+)/(\w+)/(\w+)/(.*)')
def shared_embeddings(opts, key, val, enc_dec, block_num, module1, module2, slot):
  prefix = 'state/param_states' if slot else 'target'
  suffix = '/' + SLOT_MAP[slot] if slot else ''
  if module2 in NORM_MAP:
    module2 = NORM_MAP[module2]
  newkey = f'{prefix}/discrete_vae/{enc_dec}/mid_attn_{block_num}/{module1}/{module2}{suffix}'
  return newkey, val

@t5_importer.add(r'model/(\w+)/(\w+)/(\d+)/block/(\d)/(\w+)/(\w+)/(.*)')
def shared_embeddings(opts, key, val, enc_dec, up_down, down_num, block_num, module1, module2, slot):
  prefix = 'state/param_states' if slot else 'target'
  suffix = '/' + SLOT_MAP[slot] if slot else ''
  if module2 in NORM_MAP:
    module2 = NORM_MAP[module2]
  newkey = f'{prefix}/discrete_vae/{enc_dec}/{up_down}_{down_num}_block_{block_num}/{module1}/{module2}{suffix}'
  return newkey, val

@t5_importer.add(r'model/(\w+)/(\w+)/(\d+)/attn/(\d)/(\w+)/(\w+)/(.*)')
def shared_embeddings(opts, key, val, enc_dec, up_down, down_num, attn_num, module1, module2, slot):
  prefix = 'state/param_states' if slot else 'target'
  suffix = '/' + SLOT_MAP[slot] if slot else ''
  if module2 in NORM_MAP:
    module2 = NORM_MAP[module2]
  newkey = f'{prefix}/discrete_vae/{enc_dec}/{up_down}_{down_num}_attn_{attn_num}/{module1}/{module2}{suffix}'
  return newkey, val

@t5_importer.add(r'model/encoder/down/(\d+)/downsample/(\w+)/(\w+)/(.*)')
def shared_embeddings(opts, key, val, down_num, module1, module2, slot):
  prefix = 'state/param_states' if slot else 'target'
  suffix = '/' + SLOT_MAP[slot] if slot else ''
  if module2 in NORM_MAP:
    module2 = NORM_MAP[module2]
  newkey = f'{prefix}/discrete_vae/encoder/down_{down_num}_downsample/{module1}/{module2}{suffix}'
  return newkey, val

@t5_importer.add(r'model/decoder/up/(\d+)/upsample/(\w+)/(\w+)/(.*)')
def shared_embeddings(opts, key, val, down_num, module1, module2, slot):
  prefix = 'state/param_states' if slot else 'target'
  suffix = '/' + SLOT_MAP[slot] if slot else ''
  if module2 in NORM_MAP:
    module2 = NORM_MAP[module2]
  newkey = f'{prefix}/discrete_vae/decoder/up_{down_num}_upsample/{module1}/{module2}{suffix}'
  return newkey, val


def _add_missing_param_states(t5_data, optimizer_data):
  """Add dummy slots that Flax Adafactor requires but TF does not."""
  updates = {}
  for k in t5_data:
    if k.startswith('target'):
      state_leaf = 'state/param_states' + k[len('target'):]
      if state_leaf + '/m' in optimizer_data:
        updates[state_leaf + '/m'] = np.zeros(optimizer_data[state_leaf + '/m'].shape, np.float32)
      if state_leaf + '/v_row' in optimizer_data:
        updates[state_leaf + '/v_row'] = np.zeros(optimizer_data[state_leaf + '/v_row'].shape, np.float32)
      if state_leaf + '/v_col' in optimizer_data:
        updates[state_leaf + '/v_col'] = np.zeros(optimizer_data[state_leaf + '/v_col'].shape, np.float32)
      if state_leaf + '/v' in optimizer_data:
        updates[state_leaf + '/v'] = np.zeros(optimizer_data[state_leaf + '/v'].shape, np.float32)

      # if state_leaf + '/v' in t5_data:
      #   updates[state_leaf + '/v_row'] = np.zeros((1,), np.float32)
      #   updates[state_leaf + '/v_col'] = np.zeros((1,), np.float32)
      # elif state_leaf + '/v_row' in t5_data:
      #   updates[state_leaf + '/v'] = np.zeros((1,), np.float32)
  t5_data.update(**updates)
  return t5_data


def _maybe_correct_relpos_bias(t5_data):
  """Correct the relpos_bias format if it is shared across layers."""
  max_layer_ind = 0
  for k, v in t5_data.items():
    match = re.search(r'layers_(\d+)/relpos_bias', k)
    if match:
      layer_ind = int(match.groups()[0])
      max_layer_ind = max(max_layer_ind, layer_ind)

  modified_dict = {}
  if max_layer_ind == 0:
    # Relative position biases are shared across layers
    for k, v in t5_data.items():
      new_k = re.sub(r'layers_\d+/relpos_bias', 'relpos_bias', k)
      modified_dict[new_k] = v
  else:
    # Relative position biases are unique in each layer. No more processing is
    # necessary.
    modified_dict = t5_data

  return modified_dict


# Load checkpoint, translate, and update flax optimizer and model.
# -----------------------------------------------------------------------------
def load_tf_ckpt(path):
  """Load a TF checkpoint as a flat dictionary of numpy arrays."""
  ckpt_reader = tf.train.load_checkpoint(path)
  ckpt_shape_map = ckpt_reader.get_variable_to_shape_map()
  ckpt_dtype_map = ckpt_reader.get_variable_to_dtype_map()
  datamap = {  # pylint: disable=g-complex-comprehension
      k: LazyThreadPoolArray(
          s,
          jnp.dtype(ckpt_dtype_map[k].as_numpy_dtype),
          lambda x=k: ckpt_reader.get_tensor(x))
      for k, s in ckpt_shape_map.items()
  }
  return datamap


def update_optimizer(optimizer: optim.Optimizer,
                     t5_data: MutableMapping[str, LazyArray],
                     strict: bool = True) -> optim.Optimizer:
  """Update flax optimizer for T5 model.

  Args:
    optimizer: Optimizer to update with T5 parameters.
    t5_data: T5 model parameters, typically loaded from a checkpoint.
    strict: If True requires that optimizer and t5_data mappings contain the
      same set of names (variables). If False, updating will succeed even if
      t5_data contains variables not in the optimizer. If the optimizer has
      variables not in t5_data, this function will still fail.

  Returns:
    Updated optimizer.
  """

  optimizer_data = traverse_util.flatten_dict(optimizer.state_dict())
  optimizer_data = {'/'.join(k): v for k, v in optimizer_data.items()}

  # Remove parameters from the checkpoint not found in the optimizer (this
  # allows us to load checkpoints that contain more parameters than our current
  # model).
  if not strict:
    for k in list(t5_data):
      if k not in optimizer_data:
        t5_data.pop(k)
  # Shape check.
  for k, v in t5_data.items():
    if optimizer_data[k].shape != v.shape:
      raise ValueError(
          f'Variable {k} has shape {v.shape} != {optimizer_data[k].shape}')

  optimizer_data = t5_data
  optimizer_data = traverse_util.unflatten_dict({tuple(k.split('/')): v for k, v in optimizer_data.items()})
  # t5_data = traverse_util.unflatten_dict({tuple(k.split('/')): v for k, v in t5_data.items()})
  # path = 'gs://jiasen-tpu-bucket/unified_io_t5/init_unified_io_t5_base_without_vqgan/checkpoint_1100000'
  # t5_states = checkpoints.load_t5x_checkpoint(path=path)
  # t5_states['target']['discrete_vae'] = optimizer_data['target']['discrete_vae']

  return optimizer.restore_state(optimizer_data)

# TODO(bastings,jbulian): restore from TrainState not optim.Optimizer.
def restore_from_t5_checkpoint(optimizer: optim.Optimizer,
                               path: str,
                               lazy_parameters: bool = False,
                               strict: bool = True) -> optim.Optimizer:
  """Load T5 checkpoint and update Adafactor optimizer and T5 model from it.

  We require that the final translated checkpoint structure exactly matches
  that of the Flax Adafactor + Transformer data, up to shape agreement of
  the leaves.

  Args:
    optimizer: Flax Adafactor Optimizer for T5 transformer encoder-decoder.
    path: a path to checkpoint file or directory.
    lazy_parameters: whether to leave the parameters as LazyArrays to preserve
      memory.
    strict: If True requires that optimizer and t5_data mappings contain the
      same set of names (variables). If False, updating will succeed even if
      t5_data contains variables not in the optimizer. If the optimizer has
      variables not in t5_data, this function will still fail.

  Returns:
    Adafactor optimizer updated with parameters and optimizer state from
    T5 checkpoint.
  """

  # optimizer_data = traverse_util.flatten_dict(optimizer.state_dict())
  # optimizer_data = {'/'.join(k): v for k, v in optimizer_data.items()}
  # for k, v in optimizer_data.items():
  #   print(k, v)

  ckpt_data = load_tf_ckpt(path)
  t5_data = t5_importer.apply(ckpt_data)
  
  optimizer_data = traverse_util.flatten_dict(optimizer.state_dict())
  optimizer_data = {'/'.join(k): v for k, v in optimizer_data.items()}

  t5_data = _add_missing_param_states(t5_data, optimizer_data)
  t5_data = _maybe_correct_relpos_bias(t5_data)
  optimizer = update_optimizer(optimizer, t5_data, strict=strict)
  if not lazy_parameters:
    optimizer = jax.tree_map(
        lambda x: x.get() if isinstance(x, LazyArray) else x, optimizer)
  return optimizer
