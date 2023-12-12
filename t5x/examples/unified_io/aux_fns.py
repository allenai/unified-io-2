"""Logit masking and checkpoint transformation functions"""

import functools
from typing import Dict

import gin
import jax
import jax.numpy as jnp
import numpy as np

from t5x import state_utils
from t5x.examples.unified_io import config
from t5x.examples.unified_io.data.data_utils import get_default_vocabulary
from flax import linen as nn


def apply_mask(logits, cur_index, mask):
    if len(mask.shape) == 3:
        mask = mask[:, cur_index]
    elif len(mask.shape) == 2:
        mask = jnp.reshape(mask[cur_index], [1, -1])
    else:
        mask = jnp.reshape(mask, [1, -1])
    if mask.dtype == jnp.bool_:
        flat_logits = jnp.where(mask, -1e10, logits)
    else:
        flat_logits = mask + logits
    return flat_logits


def clf_free_logit_mask_fn(logits, _,  num_decodes, alpha=10.0):
    logits = nn.log_softmax(logits)
    logits = jnp.reshape(logits, [-1, num_decodes, logits.shape[-1]])
    bs = logits.shape[0]
    logits, clf_free = logits[:bs//2], logits[bs//2:]
    logits = (1 + alpha) * logits - alpha * clf_free
    return jnp.reshape(jnp.tile(logits, (2, 1, 1)), [bs*num_decodes, -1])


def clf_free_next_token_callback(logits, next_token, num_decodes):
    # The classifier free examples need to follow the non-clf-free versions
    next_token = jnp.reshape(next_token, [-1, num_decodes])
    bs = next_token.shape[0]
    next_token = jnp.tile(next_token[:bs//2], (2, 1))
    return jnp.reshape(next_token, -1)


@gin.configurable()
def pose_estimation_mask_fn_part_names(_, lengths):
    """Mask logits so that the model only predicts pose part names and location points"""
    vocab = get_default_vocabulary()
    if config.TOKENIZER == "llama":
        vocab_size = 33280
    else:
        raise NotImplementedError()
    masks = []
    loc_mask = np.ones([vocab_size], np.bool_)
    loc_mask[32000:33000] = False
    for part in config.HUMAN_POSE_PART:
        masks += [loc_mask, loc_mask]
        for voc_id in vocab.encode(part):
            mask = np.ones([vocab_size], np.bool_)
            mask[voc_id] = False
            masks.append(mask)
    eos_mask = np.ones([vocab_size], np.bool_)
    eos_mask[1] = 0
    masks.append(eos_mask)
    mask = jnp.array(np.stack(masks, 0))
    return functools.partial(apply_mask, mask=mask)


@gin.configurable
def non_loc_select(_, lengths, thresh=0.5, require_one_box=False):
    """Mask logits so EOS is only selected if the total prob over location tokens is < `thresh`"""
    voc_size = 33280 if config.TOKENIZER == "llama" else 33152 + 16384
    loc_mask = np.zeros([voc_size], np.float32)
    loc_mask[:32000] = -10000
    loc_mask[33000:] = -10000
    loc_mask = jnp.array(loc_mask)

    def _fn(logits, cur_index):
        logits = jax.nn.log_softmax(logits)
        probs = jnp.exp(jax.scipy.special.logsumexp(logits[:, 32000:33000], axis=-1))
        use_loc = probs > thresh
        if require_one_box:
            use_loc = jnp.logical_or(use_loc, cur_index <= 3)
        return logits + loc_mask[None, :] * use_loc[:, None]
    return _fn


def state_transformation_fns():
    fn = [
            functools.partial(
                state_utils.apply_assignment_map,
                assignment_map=[
                    (r'state.*', None),
                ])
        ]

    return fn


def remove_optimizer_state():
    fn = [
        functools.partial(
            state_utils.apply_assignment_map,
            assignment_map=[
                (r'state.*', None),
            ])
    ]
    return fn


def load_vae(state_dict, target_state_dict: Dict, *, is_resuming: bool = False, modality="image"):
    if is_resuming:
        return target_state_dict
    return dict(target={f"target_encoders_{modality}": dict(discrete_vae=state_dict["target"])})


def vit_vqgan_restore_fn(modality="image"):
    return [functools.partial(load_vae, modality=modality)]


def load_vqgan(state_dict, target_state_dict: Dict, *, is_resuming: bool = False, modality="image"):
    if is_resuming:
        return target_state_dict
    if 'image_vitvqgan' in state_dict["target"]:
        return dict(target={f"target_encoders_{modality}": dict(discrete_vae=state_dict["target"]['image_vitvqgan'])})
    else:
        return dict(target={f"target_encoders_{modality}": dict(discrete_vae=state_dict["target"])})


def vqgan_restore_fn(modality="image"):
    return [functools.partial(load_vqgan, modality=modality)]


def load_vae_all(state_dict, target_state_dict: Dict, *, is_resuming: bool = False):
    if is_resuming:
        return target_state_dict

    return dict(target=dict(
        target_encoders_image=dict(discrete_vae=state_dict["target"]["image_vitvqgan"]),
        target_encoders_audio=dict(discrete_vae=state_dict["target"]["audio_vitvqgan"])))


def vit_vqgan_all_restore_fn():
    """State transformation to map a VQGAN checkpoint parameters into a UIO2 model"""
    return [load_vae_all]

