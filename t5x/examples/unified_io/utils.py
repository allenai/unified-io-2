import os
import time
from typing import Optional, List, Dict, Mapping, Any, Sequence, Tuple

import gin
import jax.numpy as jnp
import jax.random
import numpy as np
import seqio
import tensorflow as tf
import wandb
from absl import logging

from t5x import checkpoints
from t5x import utils
from t5x.examples.unified_io.config import SHUFFLE_BUFFER_SIZE, CYCLE_LENGTH, BLOCK_LENGTH
from t5x.utils import TrainStateInitializer, RestoreCheckpointConfig, LegacyCheckpointManager


def get_model(model_size, input_modalities=None, target_modalities=None,
              gin_bindings: Optional[List[str]]=None, dtype="bfloat16"):
  """Return a EncoderDecoder model and configure the code to support that model

  This will also configure the pre-processing functions to be consistent with returned model
  """
  if gin.config_is_locked():
    logging.warning("Using `get_model` might override existing gin flags")

  with gin.config.unlock_config():
    def _get(model):
      return model

    bindings = []
    if gin_bindings:
      bindings += gin_bindings
    if input_modalities:
      bindings.append(f"get_input_modalities.input_modality={input_modalities}")
    if target_modalities:
      bindings.append(f"get_target_modalities.target_modality={target_modalities}")
    if dtype != "bfloat16":
      bindings += [f"{x}.dtype=\"float32\"" for x in [
        "T5Config",
        "AudioViTVQGANConfig",
        "VAEConfig",
        "ImageVitFeatureConfig",
        "AudioVitFeatureConfig",
        "ImageResamplerConfig",
        "AudioResamplerConfig",
      ]]
    _get = gin.configurable(_get)
    gin.parse_config_files_and_bindings(
      config_files=[f"t5x/examples/unified_io/t5_1_1/{model_size}.gin"],
      bindings=bindings + [
        "_get.model=%MODEL"
      ]
    )
    return _get()


def get_parameters(model, model_checkpoint: str = None,
                   partitioner=None, rng: jax.random.PRNGKey=None) -> Tuple:
  """Get parameters for a model

  model: Model to get parameters for
  model_checkpoint: Checkpoint, if None initialized the model from scratch
  partitioner: If given, load parameters with this partitioner
  rng: If initializing from scratch, load parameters with this RNG
  """
  from t5x.examples.unified_io.modality_processing import get_input_spec
  t0 = time.perf_counter()
  if rng is None:
    seed = np.random.randint(np.iinfo(np.int32).min, np.iinfo(np.int32).max, (), np.int32)
    rng = jax.random.PRNGKey(seed)

  if model_checkpoint is None:
    logging.info("Init model from scratch")
    input_shapes, input_types = get_input_spec()
    if partitioner is None:
      params = model.get_initial_variables(rng, input_shapes, input_types)
      params, param_axes = params["params"], params["params_axes"]
    else:
      train_state_initializer = TrainStateInitializer(
        optimizer_def=None,
        init_fn=model.get_initial_variables,
        input_shapes=input_shapes,
        input_types=input_types,
        partitioner=partitioner
      )
      train_state = train_state_initializer.from_scratch(rng)
      param_axes = train_state_initializer.train_state_axes.params
      params = freeze(train_state.params)
  else:
    if not model_checkpoint.startswith("gs://"):
      model_checkpoint = os.path.abspath(os.path.expanduser(model_checkpoint))
      assert os.path.exists(model_checkpoint), f"{model_checkpoint=} does not exist!"
    logging.info(f"Loading model weights from {model_checkpoint}...")
    input_shapes, input_types = get_input_spec(1)
    if partitioner is not None:
      train_state_initializer = TrainStateInitializer(
        optimizer_def=None,
        init_fn=model.get_initial_variables,
        input_shapes=input_shapes,
        input_types=input_types,
        partitioner=partitioner
      )
      param_axes = train_state_initializer.train_state_axes.params
      params = LegacyCheckpointManager(
        restore_cfg=RestoreCheckpointConfig(model_checkpoint),
        train_state_shape=train_state_initializer.global_train_state_shape,
        partitioner=partitioner
      ).restore([model_checkpoint], RestoreCheckpointConfig(model_checkpoint)).params
    else:
      params = checkpoints.load_t5x_checkpoint(path=model_checkpoint)['target']
      params = jax.tree_util.tree_map(jnp.array, params)
      param_axes = None
  logging.info(f"Done in {time.perf_counter()-t0:0.1f}")
  return params, param_axes


@gin.configurable()
def init_wandb(name=None, group=None, entity=None, project=None):
  utils.create_learning_rate_scheduler()  # Makes sure this is registered in `operative_config`
  config_str = gin.operative_config_str()
  logging.info(f"Init wandb with group={group} name={name}")
  wandb.init(
    group=group,
    name=name,
    entity=entity,
    project=project,
    force=True,
    notes=config_str
  )


def transpose_lists(lsts):
  """Transpose a list of lists."""
  return [list(i) for i in zip(*lsts)]


def list_of_dict_to_string(table: List[Dict[str, str]], filler="") -> str:
  keys = dict()
  for row in table:
    keys.update(row)
  raw_table = [list(keys)]
  raw_table += [[row.get(key, filler) for key in keys] for row in table]
  return table_string(raw_table)


def table_string(table: List[List[str]]) -> str:
  """Table as listoflists to evenly spaces string"""
  # print while padding each column to the max column length
  if len(table) == 0:
    return ""
  col_lens = [0] * len(table[0])
  for row in table:
    for i, cell in enumerate(row):
      col_lens[i] = max(len(cell), col_lens[i])

  formats = ["{0:<%d}" % x for x in col_lens]
  out = []
  for row in table:
    out.append(" ".join(formats[i].format(row[i]) for i in range(len(row))))
  return "\n".join(out)


class WandbMetricsLogger(seqio.Logger):
  """Log metrics to wandb"""

  def __call__(
      self,
      task_name: str,
      step: Optional[int],
      metrics: Mapping[str, Any],
      dataset: Optional[tf.data.Dataset],
      inferences: Optional[Mapping[str, Sequence[Any]]],
      targets: Optional[Sequence[Any]],
  ) -> None:
    if step is None:
      raise ValueError()

    wandb_metrics = {}
    for metric_name, metric_value in metrics.items():
      if isinstance(metric_value, seqio.metrics.Scalar):
        wandb_metrics[f"inference/{task_name}/{metric_name}"] = metric_value.value
      else:
        logging.warning(
          "Skipping WandbLogging of non-serializable metric '%s' of type %s.",
          metric_name,
          type(metric_value),
        )
    wandb.log(wandb_metrics, step=step)