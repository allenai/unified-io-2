from t5x.examples.unified_io.data import tasks
from t5x.examples.unified_io.data import nlp_instruction_following
from seqio import MixtureRegistry


MixtureRegistry.add(
  "refexp",
  [
    ("refcoco_plus_unc", 1.0),
    ("refcocog_google", 1.0),
    ("refcoco_unc", 1.0),
  ],
)
