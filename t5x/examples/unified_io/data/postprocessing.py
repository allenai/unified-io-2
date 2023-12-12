def return_example(output_or_target, example=None, is_target=False):
  if is_target:
    return example
  else:
    return output_or_target


def return_meta(output_or_target, example=None, is_target=False):
  if is_target:
    return {k[5:]: v for k, v in example.items() if k.startswith("meta/")}
  else:
    return output_or_target


def return_field(output_or_target, field, example=None, is_target=False):
  if is_target:
    return example[field]
  else:
    return output_or_target
