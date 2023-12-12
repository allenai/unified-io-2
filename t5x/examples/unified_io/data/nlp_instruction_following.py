import functools
import re

import seqio
from seqio import TaskRegistry
from seqio.preprocessors import rekey

from t5x.examples.unified_io.config import MULTITASK_TFDS_DATA_DIR
from t5x.examples.unified_io.data.data_utils import get_default_vocabulary, apply_keyword_prompt, \
  random_element
from t5x.examples.unified_io.data.prompt_definition import Prompt
from t5x.examples.unified_io.data.prompt_dict import TRUNCATE
from t5x.examples.unified_io.modality_processing import unified_io_preprocessor, OUTPUT_FEATURES
import tensorflow as tf


# Extracted from the language labels from the NLI README
NI_NON_ENGLISH_TASKS = [
  86, 117, 171, 172, 173, 174, 175, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261,
  262, 263, 264, 265, 266, 271, 272, 273, 312, 313, 314, 315, 334, 336, 338, 394, 395, 396, 406,
  407, 408, 409, 410, 411, 412, 414, 415, 416, 417, 424, 425, 426, 427, 432, 433, 434, 435, 436,
  437, 438, 439, 440, 441, 446, 447, 448, 449, 450, 451, 452, 463, 464, 465, 466, 467, 468, 473,
  474, 479, 480, 481, 482, 483, 484, 485, 486, 487, 524, 525, 526, 527, 528, 529, 530, 531, 532,
  533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 548, 549, 551, 552, 553,
  554, 555, 556, 557, 558, 559, 561, 562, 601, 604, 612, 634, 635, 643, 644, 650, 651, 652, 653,
  654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 680, 762, 763, 764, 765, 771, 772, 773, 774,
  775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793,
  794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812,
  813, 814, 815, 816, 817, 818, 829, 830, 831, 832, 836, 837, 838, 839, 840, 841, 842, 872, 873,
  877, 878, 896, 910, 911, 912, 913, 914, 938, 939, 940, 941, 942, 943, 944, 945, 946, 947, 948,
  949, 950, 951, 952, 953, 954, 960, 961, 962, 968, 969, 974, 975, 976, 977, 978, 979, 980, 981,
  982, 983, 984, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999, 1000,
  1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016,
  1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032,
  1033, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048,
  1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1060, 1061, 1062, 1063, 1064,
  1065, 1066, 1067, 1068, 1069, 1070, 1071, 1072, 1073, 1074, 1075, 1076, 1077, 1078, 1079, 1080,
  1081, 1082, 1083, 1084, 1085, 1086, 1090, 1091, 1092, 1093, 1094, 1095, 1096, 1097, 1098, 1099,
  1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115,
  1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128, 1129, 1130, 1131,
  1132, 1133, 1134, 1136, 1137, 1138, 1139, 1140, 1141, 1142, 1143, 1144, 1145, 1168, 1169, 1170,
  1171, 1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1218,
  1219, 1220, 1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1229, 1230, 1231, 1232, 1233, 1234,
  1235, 1236, 1237, 1238, 1239, 1240, 1241, 1242, 1243, 1244, 1245, 1246, 1247, 1248, 1249, 1250,
  1251, 1252, 1253, 1254, 1255, 1256, 1257, 1258, 1259, 1260, 1261, 1262, 1263, 1264, 1265, 1266,
  1267, 1268, 1269, 1270, 1271, 1272, 1273, 1274, 1275, 1276, 1277, 1278, 1279, 1280, 1281, 1282,
  1323, 1324, 1329, 1330, 1334, 1335, 1350, 1351, 1352, 1353, 1365, 1367, 1370, 1371, 1373, 1374,
  1375, 1376, 1377, 1395, 1396, 1397, 1402, 1414, 1432, 1433, 1435, 1436, 1490, 1491, 1492, 1493,
  1494, 1496, 1497, 1514, 1537, 1538, 1539, 1543, 1544, 1545, 1546, 1561, 1569, 1570, 1571, 1574,
  1575, 1576, 1577, 1588, 1591, 1610, 1611, 1616, 1617, 1618, 1619, 1620, 1621, 1626, 1627, 1628,
  1629, 1646, 1647, 1648, 1649, 1650, 1651, 1652, 1654, 1655, 1662, 1663, 1666, 1667, 1676, 1677,
  1685, 1686, 1689, 1690, 1691, 1692
]


def filter_by_len(ds, sequence_length):
  @seqio.map_over_dataset
  def tokenize(ex):
    voc = get_default_vocabulary()
    ex["text_inputs"] = voc.encode_tf(ex["text_inputs"])
    ex["text_targets"] = voc.encode_tf(ex["text_targets"])
    return ex

  ds = tokenize(ds)

  def _filter(ex):
    # Leave one space for EOS
    return (
        (len(ex["text_inputs"]) <= sequence_length["text_inputs"] - 1) and
        (len(ex["text_targets"]) <= sequence_length["text_targets"] - 1)
    )

  return ds.filter(_filter)


@seqio.map_over_dataset
def tokenize_with_truncate(x, sequence_length):
  """Tokenize x but truncate from the special TRUNCATE symbol not the end"""
  voc = get_default_vocabulary()
  text_inputs = x["text_inputs"]
  parts = tf.strings.split(text_inputs, TRUNCATE, maxsplit=2)
  if tf.shape(parts)[0] == 1:
    x["text_inputs_pretokenized"] = text_inputs
    x["text_inputs"] = voc.encode_tf(parts[0])
  else:
    x["text_inputs_pretokenized"] = tf.strings.join([parts[0], parts[1]], "")
    to_truncate = voc.encode_tf(parts[0])
    suffix = voc.encode_tf(parts[1])

    max_input_len = sequence_length["text_inputs"]
    n = max_input_len - tf.shape(suffix)[0] - 1  # -1 for the EOS
    x["text_inputs"] = tf.concat([to_truncate[:n], suffix], 0)
  return x


def filter_non_english(ds, source):
  if source == "NIv2":
    def _fn(ex):
      return not tf.strings.regex_full_match(ex["task_name"], f"task({'|'.join(str(x) for x in NI_NON_ENGLISH_TASKS)})_.*")
  elif source == "Flan2021":
    def _fn(ex):
      return not tf.strings.regex_full_match(ex["task_name"], "(wmt[0-9]*_.*)|para_crawl_enes")
  else:
    return ds
  return ds.filter(_fn)


@seqio.map_over_dataset
def preprocess_flan(ex, name):
  return dict(
    text_inputs=tf.strings.join(["[Text] [S] ", ex["inputs"]]),
    text_targets=ex["targets"],
    example_id=tf.strings.join([name, tf.strings.as_string(ex["example_num"])], "-")
  )


def add_flan(name):
  full_name = f"flan2_{name.lower()}"
  TaskRegistry.add(
    full_name,
    source=seqio.TfdsDataSource(
      tfds_name=f"{full_name}:1.0.0",
      tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
      splits={
        "train": "train[2000:]",
        "validation": "train[:2000]"
      }
    ),
    preprocessors=[
      functools.partial(filter_non_english, source=name),
      functools.partial(preprocess_flan, name=full_name),
      filter_by_len,
      unified_io_preprocessor,
    ],
    output_features=OUTPUT_FEATURES,
  )


FLAN_DATASETS = ["Flan2021", "T0", "NIv2", "CoT", "Dialog"]

for dataset in FLAN_DATASETS:
  add_flan(dataset)


# Weights from https://github.com/google-research/FLAN/blob/main/flan/v2/run_example.py#L65-L73
# git commit 7b33ac0
seqio.MixtureRegistry.add(
  'flan2',
  tasks=[
    ('flan2_flan2021', 0.4),  # mixing weight = 40%
    ('flan2_t0', 0.32),       # mixing weight = 32%
    ('flan2_niv2', 0.2),      # mixing weight = 20%
    ('flan2_cot', 0.05),      # mixing weight = 5%
    ('flan2_dialog', 0.03),   # mixing weight = 3%
  ])


def preprocess_instruction_context(ds, dataset_name):
  context_prompts = Prompt().get_prompt_list("NLP Instruction Context", dataset_name)
  nocontext_prompts = Prompt().get_prompt_list("NLP Instruction", dataset_name)

  @seqio.map_over_dataset
  def run(ex):
    if tf.strings.length(ex["context"]) == 0:
      prompt = random_element(context_prompts)
      prompt = apply_keyword_prompt(prompt, context=ex["context"], instruction=ex["instruction"])
    else:
      prompt = random_element(nocontext_prompts)
      prompt = apply_keyword_prompt(prompt, instruction=ex["instruction"])
    prompt = tf.strings.join(["[Text] [S] ", prompt])
    return dict(
      text_inputs=prompt,
      text_targets=ex["response"],
      example_id=tf.strings.join(["dolly", tf.strings.as_string(ex["example_num"])], "-"),
    )

  return run(ds)


def add_instruction_dataset(
    name, version, n_val, preprocess=None
):
  TaskRegistry.add(
    name,
    source=seqio.TfdsDataSource(
      tfds_name=f"{name}:{version}",
      tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
      splits={
        "train": f"train[{n_val}:]",
        "validation": f"train[:{n_val}]"
      }
    ),
    preprocessors=(preprocess if preprocess else []) + [
      functools.partial(preprocess_instruction_context, dataset_name=name),
      tokenize_with_truncate,
      unified_io_preprocessor,
    ],
    output_features=OUTPUT_FEATURES,
  )


map_input_output_keys = [functools.partial(rekey, key_map=dict(
  example_num="example_num",
  instruction="instruction",
  context="input",
  response="output",
))]
add_instruction_dataset("dolly", "1.0.0", 1000),
add_instruction_dataset("alpaca", "1.0.0", 2000, map_input_output_keys),
