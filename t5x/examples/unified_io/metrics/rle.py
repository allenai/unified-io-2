from pycocotools import mask

import json
import numpy as np
from pycocotools import mask as maskUtils


def uncompressed_encode(binary_mask):
  binary_mask = np.asfortranarray(binary_mask)
  uncompressed_rle = {'counts': [], 'size': list(binary_mask.shape)}
  counts = uncompressed_rle.get('counts')

  last_elem = 0
  running_length = 0

  for i, elem in enumerate(binary_mask.ravel(order='F')):
    if elem == last_elem:
      pass
    else:
      counts.append(running_length)
      running_length = 0
      last_elem = elem
    running_length += 1

  counts.append(running_length)

  return uncompressed_rle


def compress(uncompressed_rle):
  compressed_rle = mask.frPyObjects(uncompressed_rle, uncompressed_rle.get('size')[0], uncompressed_rle.get('size')[1])
  return compressed_rle


def to_utf(rle):
  rle = rle.copy()
  rle['counts'] = rle['counts'].decode("utf-8", "backslashreplace")
  return rle


def from_utf(rle):
  rle = rle.copy()
  rle['counts'] = rle['counts'].encode("utf-8")
  return rle


def encode(binary_mask, utf=True):
  encoded = maskUtils.encode(np.asfortranarray(binary_mask))
  return to_utf(encoded) if utf else encoded


def decode(rle, utf=True):
  if type(rle) == list:
    return [decode(r, utf) for r in rle]
  else:
    rle = from_utf(rle) if utf else rle
    decoded = maskUtils.decode(rle)
  return decoded
