import itertools
from dataclasses import dataclass
from typing import Optional, Mapping, Sequence, Any, List, Dict

import gin
import seqio
from absl import logging
from seqio import metrics as metrics_lib

from t5x.examples.unified_io.data.data_utils import get_default_vocabulary

AllOutputTokensType = Mapping[str, Sequence[Sequence[int]]]
AllOutputScoresType = Mapping[str, Sequence[float]]
AllOutputAuxValuesType = Mapping[str, Mapping[str, Sequence[Any]]]
AllMetricsType = Mapping[str, Mapping[str, Any]]


@dataclass
class UnifiedIOOutput:
  """Wrapper to make it easier to work with the many different outputs of UIO2"""
  aux_values: Dict

  @property
  def text(self):
    return self.aux_values.get("text")

  @property
  def text_tokens(self):
    return self.aux_values.get("text-tokens")

  @property
  def image_tokens(self):
    return self.aux_values.get("image-tokens")

  @property
  def image(self):
    return self.aux_values.get("image")

  @property
  def audio(self):
    return self.aux_values.get("audio")

  @property
  def scores(self):
    if "scores" in self.aux_values:
      return self.aux_values["scores"]
    else:
      # Assume only one modality is presennt
      for x in ["text", "image", "audio"]:
        x = f"{x}-scores"
        if x in self.aux_values:
          return self.aux_values[x]
      raise ValueError(f"No scores found in self.aux_values, keys={self.aux_values.keys()}")


def build_uio_outputs(aux_values, vocab) -> List[UnifiedIOOutput]:
  out = []
  n = len(next(iter(aux_values.values())))
  for ix in range(n):
    values = {k: v[ix] for k, v in aux_values.items()}
    txt_tokens = values.get("text-tokens")
    if txt_tokens is None:
      pass
    elif len(txt_tokens.shape) == 1:
      values["text"] = vocab.decode(txt_tokens)
    else:
      values["text"] = [vocab.decode(x) for x in txt_tokens]
    out.append(UnifiedIOOutput(values))
  return out


@gin.configurable()
class UnifiedIOEvaluator(seqio.Evaluator):
  """Evaluator for UnifiedIO 2"""
  # This class basically follows `seqio.Evaluator` but has a few UIO2 hacks

  def __init__(
      self,
      mixture_or_task_name: str,
      feature_converter,
      eval_split: str = "validation",
      use_cached: bool = False,
      seed: Optional[int] = 42,
      sequence_length: Optional[Mapping[str, int]] = None,
      num_examples: Optional[int] = None,
      shuffle: bool = False,
      logger_cls: Sequence = (),
      log_dir: Optional[str] = None,
      use_memory_cache: bool = True,
      target_field_name: str = "targets",
  ):
    # We use a simplified `sequence_length` that does not contain fields that exactly match
    # the Dataset fields. This can cause an issue because the evaluator will delete those
    # non-matching entries before the feature conversion stage, so we alias them to these
    # names that do match the dataset structure here so they will be preserved.
    if "text_inputs" in sequence_length:
      sequence_length["inputs/text/tokens"] = sequence_length["text_inputs"]
    if "text_targets" in sequence_length:
      sequence_length["targets/text/tokens"] = sequence_length["text_targets"]
    super().__init__(
      mixture_or_task_name, feature_converter, eval_split, use_cached, seed, sequence_length,
      num_examples, shuffle, logger_cls, log_dir, use_memory_cache, target_field_name
    )

  def _compute_metrics(self,
                       predicted_tokens: AllOutputTokensType,
                       scores: AllOutputScoresType,
                       all_aux_values: AllOutputAuxValuesType,
                       step: Optional[int] = None) -> AllMetricsType:

    vocab = get_default_vocabulary()
    all_metrics = {}
    for task in self.eval_tasks:
      logging.info("Computing metrics for %s", task.name)
      task_dataset = self.cached_task_datasets[task.name]
      targets = self.cached_targets[task.name]
      task_metrics = []
      inferences = {}

      if task.predict_metric_fns or task.predict_with_aux_metric_fns:
        (outputs,
         postprocessed_outputs) = self._decode_and_postprocess_predictions(
             task, predicted_tokens, task_dataset, targets)
        inferences["output"] = outputs
        inferences["prediction"] = postprocessed_outputs

      if task.predict_metric_fns:
        task_metrics.extend([
            metric_fn(targets, inferences["prediction"])
            for metric_fn in task.predict_metric_fns
        ])

      if task.predict_with_aux_metric_fns:
        aux_values = all_aux_values[task.name]
        uio_output = build_uio_outputs(aux_values, vocab)
        task_metrics.extend([
            metric_fn(targets, uio_output, aux_values)
            for metric_fn in task.predict_with_aux_metric_fns
        ])
        inferences["aux_value"] = aux_values

      if task.score_metric_fns:
        task_scores = scores[task.name]
        if len(targets) != len(task_scores):
          raise ValueError(f"len(targets)({len(targets)}) != "
                           f"len(task_scores)({len(task_scores)})")
        task_metrics.extend([
            metric_fn(targets, task_scores)
            for metric_fn in task.score_metric_fns
        ])
        inferences["score"] = task_scores

      all_metrics[task.name] = {}
      for k, v in itertools.chain(*[m.items() for m in task_metrics]):
        if k in all_metrics[task.name]:
          raise ValueError(f"Duplicate metric key '{k}' in Task '{task.name}'.")
        all_metrics[task.name][k] = v

      metrics = {
        k: metrics_lib.Scalar(v)
        if not isinstance(v, metrics_lib.MetricValue) else v
        for k, v in all_metrics[task.name].items()
      }
      for logger in self.loggers:
        logger(task_name=task.name, step=step, metrics=metrics,
               dataset=task_dataset, inferences=inferences, targets=targets)

    return all_metrics
