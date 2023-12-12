from dataclasses import dataclass
from typing import Any, List

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from t5x.examples.unified_io import modality_processing
from t5x.examples.unified_io.input_modalities import ModalityEncoder
from t5x.examples.unified_io.network import Transformer
from t5x.examples.unified_io.config import T5Config
from t5x.examples.unified_io.seq_features import InputSequence, TargetSequence
from t5x.examples.unified_io.target_modalities import BasicDecoder

TARGET_MODALITIES = ["text", "image", "audio"]


class NullEncoder(nn.Module):
  is_target: bool
  modality_id: int

  def __call__(self, embed, mask, init=False, targets=None, loss_mask=None, inputs=None,
               rel_atten=None, enable_dropout=None, use_constraints=True, segment_ids=None):
    if loss_mask is None:
      loss_mask = mask

    if self.is_target:
      return TargetSequence(
        embed, None,
        jnp.array(self.modality_id, dtype=jnp.int32), mask, rel_atten,
        subsegments=segment_ids, target_tokens=targets,  loss_mask=loss_mask)
    else:
      return InputSequence(embed, mask, rel_atten)


@dataclass
class DebugModalityEncoder(ModalityEncoder):
  is_target: bool
  modality_id: int

  def get_encoder(self, *args, **kwargs) -> nn.Module:
    return NullEncoder(self.is_target, self.modality_id)

  def get_decoder(self, config, shared_embedding) -> nn.Module:
    return BasicDecoder(None, T5Config(None, logits_via_embedding=True), shared_embedding)


class NullModalityEncoder(ModalityEncoder, nn.Module):
  def __init__(self, is_target, seq_len):
    super().__init__()
    self.seq_len = seq_len
    self.is_target = is_target

  def get_encoder(self, config, shared_embedding) -> nn.Module:
    self.t5_config = config.t5_config
    return self

  def __call__(self, null, init=False):
    bs = null.shape[0]
    if self.is_target:
      return TargetSequence.empty(
        bs, self.seq_len, self.t5_config.num_heads, self.t5_config.dtype, 2)
    else:
      return InputSequence.empty(bs, self.seq_len, self.t5_config)


def build_random_batch(cfg, rng: np.random.RandomState, batch_size,
                       input_modalities: List[int],
                       target_modalities: List[int],
                       target_segment_ids=False
                       ):
  batch = build_inputs(cfg, rng, batch_size, input_modalities)
  batch.update(build_targets(
    cfg, rng, batch_size, target_modalities, target_segment_ids))
  return batch


def build_inputs(cfg, np_rng: np.random.RandomState, batch_size, input_modalities: List[int]):
  out = {}
  for ix, seq_len in enumerate(input_modalities):
    out[f"inputs/{ix}/mask"] = np_rng.random((batch_size, seq_len)) > 0.5
    out[f"inputs/{ix}/embed"] = np_rng.uniform(-1, 1, (batch_size, seq_len, cfg.emb_dim))
    out[f"inputs/{ix}/rel_atten"] = np_rng.uniform(0, 0.5, (batch_size, cfg.num_heads, seq_len, seq_len))

  return out


def build_targets(
    cfg, np_rng: np.random.RandomState, batch_size, target_modalities,
    segment_ids=False):
  out = {}
  assert len(target_modalities) <= 3
  for name, seq_len in zip(TARGET_MODALITIES, target_modalities):
    if seq_len is None:
      continue
    out.update({
      f"targets/{name}/mask": np_rng.random((batch_size, seq_len)) > 0.5,
      f"targets/{name}/embed": np_rng.uniform(-1, 1, (batch_size, seq_len, cfg.emb_dim)),
      f"targets/{name}/targets": np_rng.randint(0, cfg.vocab_size, (batch_size, seq_len), dtype=np.int32),
    })
    if segment_ids and name == "text":
      out[f"targets/{name}/segment_ids"] = np_rng.randint(0, 2, (batch_size, seq_len), dtype=np.int32)
  return out


DEBUG_CONFIG = T5Config(
  num_encoder_layers=2,
  num_decoder_layers=2,
  vocab_size=100,
  dropout_rate=0.0,
  emb_dim=8,
  num_heads=2,
  head_dim=4,
  mlp_dim=12,
  dtype=jnp.float32,
  mlp_activations=('gelu',),
  logits_via_embedding=True,
  image_vocab_size=1000,
)


def build_test_transformer(
    cfg: T5Config, input_modalities: int,
    target_modalities: int,
    variable_seed=None,
  ):
  modality_processing.get_input_modalities = lambda: {
    str(ix): DebugModalityEncoder(False, ix) for ix in range(input_modalities)
  }

  # Use names `TARGET_MODALITIES` since those names are also hardcoded in some places
  assert target_modalities <= 3
  modality_processing.get_target_modalities = lambda: {
    name: DebugModalityEncoder(True, ix) for ix, name in enumerate(TARGET_MODALITIES[:target_modalities])
  }

  trans = Transformer(cfg)
  if variable_seed is not None:
    rng = jax.random.PRNGKey(variable_seed)
    np_rng = np.random.RandomState(variable_seed*9681)
    batch = build_random_batch(
      trans.config, np_rng, 1, [1]*input_modalities, [1]*target_modalities)
    variables = trans.init(rng, batch, init=True)
    return trans, variables
  else:
    return trans
