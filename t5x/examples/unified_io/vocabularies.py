# Copyright 2023 The SeqIO Authors.
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

"""Our version of the LLAMA tokenizer to use for text"""
# For backward compatibility reasons, our tokenizer
# is changed so that EOS is 1 and BOS 0

import abc
import dataclasses
import functools
import hashlib
import threading
from typing import Any, ClassVar, Dict, Iterable, Optional, Sequence, Union

from absl import logging
import tensorflow.compat.v2 as tf
import tensorflow_text as tf_text

from sentencepiece import sentencepiece_model_pb2
import sentencepiece as sentencepiece_processor

PAD_ID = 0


class Vocabulary(metaclass=abc.ABCMeta):
  """Abstract class for all vocabularies.

  Subclasses must implement methods for converting between strings and tokens
  both in pure python (`_encode`/`_decode`) and in TensorFlow
  (`_encode_tf`/`_decode_tf`).

  Subclasses are responsible for reserving PAD_ID=0 as well as optionally
  reserving EOS_ID and UNK_ID

  `_base_vocab_size` should account for PAD, EOS, and UNK but not `extra_ids`.
  """

  def __init__(self, extra_ids: int = 0):
    """Vocabulary constructor.

    Args:
      extra_ids: The number of extra IDs to reserve.
    """
    self._extra_ids = extra_ids or 0

  @property
  def bos_id(self) -> Optional[int]:
    raise NotImplementedError("need to implement bos_id")

  @property
  @abc.abstractmethod
  def eos_id(self) -> Optional[int]:
    raise NotImplementedError("need to implement eos_id")

  @property
  def pad_id(self) -> int:
    return PAD_ID

  @property
  @abc.abstractmethod
  def unk_id(self) -> Optional[int]:
    raise NotImplementedError("need to implement unk_id")

  @property
  def extra_ids(self) -> int:
    return self._extra_ids

  @property
  def vocab_size(self) -> int:
    """Vocabulary size, including extra ids."""
    return self._base_vocab_size + self.extra_ids

  @property
  @abc.abstractmethod
  def _base_vocab_size(self) -> int:
    """Vocabulary size, excluding extra ids but including PAD/EOS/UNK."""
    # TODO(fjord): add a check that pad_id and unk_id (if present)
    #   are less than _base_vocab_size.
    raise NotImplementedError

  @abc.abstractmethod
  def _encode(self, s: str) -> Sequence[int]:
    raise NotImplementedError

  def encode(self, s: Union[Sequence[int], str]) -> Sequence[int]:
    """Tokenizes string to an int sequence, without adding EOS."""
    return self._encode(s)

  @abc.abstractmethod
  def _decode(self, ids):
    raise NotImplementedError

  def decode(self, ids: Iterable[int]):
    """Detokenizes int32 iterable to a string, up through first EOS."""
    clean_ids = list(ids)

    if self.unk_id is not None:
      vocab_size = self._base_vocab_size
      clean_ids = [self.unk_id if i >= vocab_size else i for i in clean_ids]

    if self.eos_id is not None and self.eos_id in clean_ids:
      clean_ids = clean_ids[: clean_ids.index(self.eos_id) + 1]

    return self._decode(clean_ids)

  @abc.abstractmethod
  def _encode_tf(self, s: tf.Tensor) -> tf.Tensor:
    raise NotImplementedError

  def encode_tf(self, s: tf.Tensor) -> tf.Tensor:
    """Tokenizes string Scalar to an int32 Tensor, without adding EOS."""
    return self._encode_tf(s)

  @abc.abstractmethod
  def _decode_tf(self, ids: tf.Tensor) -> tf.Tensor:
    raise NotImplementedError

  def decode_tf(self, ids: tf.Tensor) -> tf.Tensor:
    """Detokenizes int32 batched Tensor through first EOS."""
    clean_ids = ids

    if self.unk_id is not None:
      base_vocab_size = self._base_vocab_size
      clean_ids = tf.where(
          tf.less(clean_ids, base_vocab_size), clean_ids, self.unk_id
      )

    if self.eos_id is not None:
      # Replace everything after the first eos_id with pad_id.
      after_eos = tf.cumsum(
          tf.cast(tf.equal(clean_ids, self.eos_id), tf.int32),
          exclusive=True,
          axis=-1,
      )
      clean_ids = tf.where(tf.cast(after_eos, tf.bool), self.pad_id, clean_ids)

    return self._decode_tf(clean_ids)



class PassThroughVocabulary(Vocabulary):
  """Vocabulary that passes through inputs unchanged."""

  def __init__(self, size: int, eos_id: Optional[Any] = None):
    """PassThroughVocabulary constructor.

    Args:
      size: the full size of the vocabulary.
      eos_id: the end-of-sequence token.
    """
    self._size = size
    self._eos_id = eos_id
    super().__init__()

  @property
  def _base_vocab_size(self):
    return self._size

  def _encode(self, s: Sequence[Any]) -> Sequence[Any]:
    return s

  def _decode(self, ids: Sequence[Any]) -> Sequence[Any]:
    return ids

  def _encode_tf(self, s: tf.Tensor) -> tf.Tensor:
    return s

  def _decode_tf(self, ids: tf.Tensor) -> tf.Tensor:
    return ids

  @property
  def eos_id(self) -> Optional[Any]:
    return self._eos_id

  @property
  def unk_id(self) -> Optional[Any]:
    return None

  def __eq__(self, other):
    if not isinstance(other, PassThroughVocabulary):
      return False
    return self._size == other._size and self.eos_id == other.eos_id

  def __str__(self) -> str:
    return f"PassThroughVocabulary(size={self._size}, eos_id={self.eos_id})"


class UnigramVocabulary(Vocabulary):
  """Vocabulary that does table-lookup of unigrams."""

  def __init__(self, unigrams: Sequence[str]):
    """UnigramVocabulary constructor.

    Args:
      unigrams: the collection of in-vocabulary tokens. This collection should
        not include PAD or UNK, which are automatically assigned ids and managed
        as possible decode tokens.
    """

    super().__init__()
    unigrams_as_list = list(unigrams)
    self._unigram_by_id = ["PAD"] + unigrams_as_list + ["UNK"]
    self._id_by_unigram = {u: i for i, u in enumerate(self._unigram_by_id)}
    initializer = tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(["PAD"] + unigrams_as_list),
        # One extra value because the leading 0 corresponds to PAD
        values=tf.constant(range(len(unigrams) + 1), dtype=tf.int64),
    )
    self._id_by_unigram_tf = tf.lookup.StaticVocabularyTable(
        initializer, num_oov_buckets=1
    )
    self._unigram_by_id_tf = tf.constant(self._unigram_by_id)

  def _encode(self, s: str) -> Sequence[int]:
    return [self._id_by_unigram.get(s, self.unk_id)]

  def _encode_tf(self, s: tf.Tensor) -> tf.Tensor:
    tf_ids = self._id_by_unigram_tf.lookup(s)
    return tf.expand_dims(tf.dtypes.cast(tf_ids, tf.int32), -1)

  def _decode(self, ids: Sequence[int]) -> str:
    return " ".join(self._unigram_by_id[id] for id in ids)

  def _decode_tf(self, ids: tf.Tensor) -> tf.Tensor:
    return self._unigram_by_id_tf[ids[0]]

  @property
  def _base_vocab_size(self):
    return len(self._unigram_by_id)

  @property
  def eos_id(self):
    return None

  @property
  def unk_id(self):
    return self._base_vocab_size - 1


class SentencePieceVocabulary(Vocabulary):
  """Wrapper for nlp/sentencepiece encoder.

  Assumes the model was built using flags to reserve ID=0 for padding, ID=1 for
  EOS, and ID=2 for UNK.

  If using extra ids, you can represent them in string-form as `<extra_id_0>`,
  `<extra_id_1>`, etc. They will be indexed starting from the end of the
  vocabulary to match how the masking preprocessors are set up.

  IMPORTANT NOTE: these placeholders only work properly when they are used at
  word starts (e.g., "I like peanut butter and <extra_id_0> sandwiches." or
  "I like peanut butter and <extra_id_0>ly sandwiches" are both okay, but
  "I like peanut butter and jel<extra_id_0> sandwiches" is not.).
  """

  @dataclasses.dataclass
  class _ModelContext:
    tokenizer: sentencepiece_processor.SentencePieceProcessor
    sp_model: bytes

  _load_model_lock: ClassVar[threading.Lock] = threading.Lock()

  def __init__(
      self,
      sentencepiece_model_file: str,
      extra_ids: int = 0,
      normalizer_spec_overrides: Optional[
          sentencepiece_model_pb2.NormalizerSpec
      ] = None,
      reverse_extra_ids: bool = False,
      modality_extra_id_n_frames: int = 0,
      hack_to_t5_start_tokens: bool = True,
      prefix_as_special_token: bool = True,
  ):
    """Create a SentencePieceVocabulary.

    Optionally, specify a number of extra ids to add to the end of the
    vocabulary for use as sentinels.

    Args:
      sentencepiece_model_file: path of the sentence piece model.
      extra_ids: number of extra ids to include.
      normalizer_spec_overrides: If not None, this proto will be merged into the
        model's normalizer and denormalizer specs. Thus, any options set on this
        object will override the values of those options in the loaded model.
      reverse_extra_ids: if True, extra_ids are numbered in descending order, so
        the first extra_id has the highest number. This is done for
        compatibility with span_corruption mask generation in T5.
    """
    self._sentencepiece_model_file = sentencepiece_model_file
    self._normalizer_spec_overrides = normalizer_spec_overrides
    self._reverse_extra_ids = reverse_extra_ids
    self._model: Optional[SentencePieceVocabulary._ModelContext] = None
    self._modality_extra_id_n_frames = modality_extra_id_n_frames
    self._hack_to_t5_start_tokens = hack_to_t5_start_tokens
    self._prefix_as_special_token = prefix_as_special_token
    super().__init__(extra_ids=extra_ids)

  def __getstate__(self):
    state = self.__dict__.copy()
    # Gin config makes a deep copy of the keyword arguments of configurables.
    # When a SentencePieceVocabulary vocabulary is used as a keyword argument
    # in a Gin configurable, it must be picklable. We therefore remove
    # _model; will be initialized lazily as needed.
    del state["_model"]
    return state

  def __setstate__(self, state):
    self.__dict__.update(state)
    self._model = None

  def load_model(self) -> None:
    _ = self._model_context()

  def _model_context(
      self,
  ) -> _ModelContext:
    """Loads model if not yet loaded and returns the model context.

    Returns:
      The model context as a tuple of (tokenizer, sp_model).
    """
    if self._model:
      return self._model

    normalizer_spec_overrides_serialized = (
        self._normalizer_spec_overrides.SerializeToString(deterministic=True)
        if self._normalizer_spec_overrides
        else None
    )

    self._model = self._load_model(
        self._sentencepiece_model_file,
        self._extra_ids,
        normalizer_spec_overrides_serialized,
        self._reverse_extra_ids,
        modality_extra_id_n_frames=self._modality_extra_id_n_frames,
        hack_to_t5_start_tokens=self._hack_to_t5_start_tokens,
        prefix_as_special_token=self._prefix_as_special_token
    )
    return self._model

  @classmethod
  @functools.lru_cache(maxsize=None)
  def _load_model(
      cls,
      sentencepiece_model_file: str,
      extra_ids: int,
      normalizer_spec_overrides_serialized: Optional[bytes] = None,
      reverse_extra_ids: bool = True,
      modality_extra_id_n_frames: int = 0,
      hack_to_t5_start_tokens=True,
      prefix_as_special_token=True,
  ) -> _ModelContext:
    """Load SPM, Python tokenizer, and cache results to the class definition."""
    # SentencePieceProcessor::LoadFromSerializedProto is not thread-safe.
    # Without a lock, users may randomly see SIGSEGV on
    # sentencepiece::ModelInterface::pad_piece when using the vocabulary in
    # SeqIO preprocessors.
    with cls._load_model_lock:
      # Handle cases where SP can't load the file, but gfile can.
      with tf.io.gfile.GFile(sentencepiece_model_file, "rb") as f:
        sp_model = f.read()
        model = sentencepiece_model_pb2.ModelProto.FromString(sp_model)

        if hack_to_t5_start_tokens:
          # PAD token would still be 0 same as BOS for consistency as previous!
          unk = model.pieces[0]
          bos = model.pieces[1]
          eos = model.pieces[2]
          model.pieces.remove(unk)
          model.pieces.remove(bos)
          model.pieces.remove(eos)
          model.pieces.insert(0, bos)   # BOS is token 0
          model.pieces.insert(1, eos)   # EOS is token 1
          model.pieces.insert(2, unk)   # UNK is token 2
        
        # Add placeholder strings for extra IDs.
        if extra_ids:
          # By default, we them in reverse order to match span corruption.
          if reverse_extra_ids:
            extra_id_tokens = reversed(range(extra_ids))
          else:
            extra_id_tokens = range(extra_ids)

          for i in extra_id_tokens:
            model.pieces.add(
                piece=f"▁<extra_id_{i}>",
                score=0.0,
                type=sentencepiece_model_pb2.ModelProto.SentencePiece.USER_DEFINED,
            )

        if modality_extra_id_n_frames:
          # Note: start from 1, not affect by `reverse_extra_ids` and not counted in `extra_ids`
          for i in range(1, modality_extra_id_n_frames + 1):
            model.pieces.add(
                piece=f"▁<image_history_{i}>",
                score=0.0,
                type=sentencepiece_model_pb2.ModelProto.SentencePiece.USER_DEFINED,
            )
            model.pieces.add(
                piece=f"▁<audio_history_{i}>",
                score=0.0,
                type=sentencepiece_model_pb2.ModelProto.SentencePiece.USER_DEFINED,
            )
          model.pieces.add(
            piece=f"▁<image_input>",
            score=0.0,
            type=sentencepiece_model_pb2.ModelProto.SentencePiece.USER_DEFINED,
          )
          model.pieces.add(
            piece=f"▁<audio_input>",
            score=0.0,
            type=sentencepiece_model_pb2.ModelProto.SentencePiece.USER_DEFINED,
          )

        if prefix_as_special_token:
          model.pieces.add(
            piece=f"▁[Text]▁[S]",
            score=0.0,
            type=sentencepiece_model_pb2.ModelProto.SentencePiece.USER_DEFINED,
          )
          model.pieces.add(
            piece=f"▁[Text]▁[R]",
            score=0.0,
            type=sentencepiece_model_pb2.ModelProto.SentencePiece.USER_DEFINED,
          )
          model.pieces.add(
            piece=f"▁[Text]▁[X]",
            score=0.0,
            type=sentencepiece_model_pb2.ModelProto.SentencePiece.USER_DEFINED,
          )
          model.pieces.add(
            piece=f"▁[Image]▁[S]",
            score=0.0,
            type=sentencepiece_model_pb2.ModelProto.SentencePiece.USER_DEFINED,
          )
          model.pieces.add(
            piece=f"▁[Image]▁[R]",
            score=0.0,
            type=sentencepiece_model_pb2.ModelProto.SentencePiece.USER_DEFINED,
          )
          model.pieces.add(
            piece=f"▁[Audio]▁[S]",
            score=0.0,
            type=sentencepiece_model_pb2.ModelProto.SentencePiece.USER_DEFINED,
          )
          model.pieces.add(
            piece=f"▁[Audio]▁[R]",
            score=0.0,
            type=sentencepiece_model_pb2.ModelProto.SentencePiece.USER_DEFINED,
          )

        if normalizer_spec_overrides_serialized is not None:
          normalizer_spec_overrides = (
              sentencepiece_model_pb2.NormalizerSpec.FromString(
                  normalizer_spec_overrides_serialized
              )
          )

          model.normalizer_spec.MergeFrom(normalizer_spec_overrides)
          model.denormalizer_spec.MergeFrom(normalizer_spec_overrides)
        sp_model = model.SerializeToString()
      # Load Python tokenizer and ensure the EOS and PAD IDs are correct.
      tokenizer = sentencepiece_processor.SentencePieceProcessor()
      tokenizer.LoadFromSerializedProto(sp_model)
      if tokenizer.pad_id() != PAD_ID:
        logging.warning(
            (
                "T5 library uses PAD_ID=%s, which is different from the "
                "sentencepiece vocabulary, which defines pad_id=%s"
            ),
            PAD_ID,
            tokenizer.pad_id(),
        )

      return cls._ModelContext(tokenizer=tokenizer, sp_model=sp_model)

  @property
  def modality_extra_ids(self):
    if self._modality_extra_id_n_frames:
      # image/audio input + n * image/audio history + R/S * 3 modalities + [Text] [X]
      return (self._modality_extra_id_n_frames + 1) * 2 + self._prefix_as_special_token * (2 * 3 + 1)
    return 0 + self._prefix_as_special_token * (2 * 3 + 1)

  @property
  def bos_id(self) -> Optional[int]:
    return self.tokenizer.bos_id()

  @property
  def eos_id(self) -> Optional[int]:
    return self.tokenizer.eos_id()

  @property
  def unk_id(self) -> Optional[int]:
    return self.tokenizer.unk_id()

  @property
  def sp_model(self) -> Optional[bytes]:
    """Retrieve the SPM."""
    return self._model_context().sp_model

  @property
  def sentencepiece_model_file(self) -> str:
    return self._sentencepiece_model_file

  @property
  def tokenizer(self) -> sentencepiece_processor.SentencePieceProcessor:
    """Returns the Python tokenizer."""
    return self._model_context().tokenizer

  @property
  def tf_tokenizer(self):
    """Instantiate and return a TF tokenizer."""
    return tf_text.SentencepieceTokenizer(model=self.sp_model)

  @property
  def vocab_size(self):
    return self._base_vocab_size

  @property
  def _base_vocab_size(self):
    """Number of ids (including 0=PAD, 1=EOS, and 2=UNK).

    Returns:
      an integer, the vocabulary size
    """
    return self.tokenizer.GetPieceSize()

  def _encode(self, s):
    """Encode a python string as a list of integers.

    Args:
      s: a string

    Returns:
      a list of integers (not terminated by EOS)
    """
    return self.tokenizer.EncodeAsIds(s)

  def _decode(self, ids):
    """Decode a list of integers to a python string.

    Args:
      ids: a list of integers (not terminated by EOS)

    Returns:
      a string
    """
    # convert all the extra ids (sentinels) to UNK=2
    unk_id = self.tokenizer.unk_id()
    piece_size = self.tokenizer.GetPieceSize()
    ids = [unk_id if i >= piece_size else int(i) for i in ids]
    return self.tokenizer.DecodeIds(ids)

  def _encode_tf(self, s):
    """Encode a tf.Scalar string to a tf.Tensor.

    This will be necessary for on-the-fly tokenization.

    Args:
      s: a tf.Scalar with dtype tf.string

    Returns:
      a 1d tf.Tensor with dtype tf.int32
    """
    return self.tf_tokenizer.tokenize(s)

  def _decode_tf(self, ids):
    """Decode in TensorFlow.

    Args:
      ids: a 1d or 2d tf.Tensor with dtype tf.int32

    Returns:
      a 1d or 2d tf.Tensor with dtype tf.string
    """
    return self.tf_tokenizer.detokenize(ids)

  def __eq__(self, other):
    if not isinstance(other, SentencePieceVocabulary):
      return False
    try:
      their_md5 = hashlib.md5(other.sp_model).hexdigest()
    # If other has no sp_model attribute, we can't test for equality
    except AttributeError:
      return False
    if self.sp_model is None:
      return False
    our_md5 = hashlib.md5(self.sp_model).hexdigest()
    return our_md5 == their_md5

  def __str__(self) -> str:
    return (
        f"SentencePieceVocabulary(file={self.sentencepiece_model_file}, "
        f"extra_ids={self._extra_ids}, "
        f"spm_md5={hashlib.md5(self.sp_model).hexdigest()})"
    )



class ByteVocabulary(Vocabulary):
  """Byte-level vocabulary.

  Encode/decode text directly to 256 "byte IDs" using UTF-8 encoding. Three
  special IDs are reserved (0=padding, 1=EOS, 2=UNK), so our encoded byte IDs
  are +3 greater than UTF-8 byte values.

  This is the vocabulary used by the ByT5 models:
  https://arxiv.org/abs/2105.13626
  """

  def __init__(self, extra_ids: int = 0):
    """Create a ByteVocabulary.

    Optionally, specify a number of extra ids to add to the end of the
    vocabulary for use as sentinels.

    Args:
      extra_ids: an optional integer
    """
    self._byte_size = 256
    # The special tokens: 0=PAD, 1=EOS,and 2=UNK
    self._num_special_tokens = 3
    super().__init__(extra_ids=extra_ids)

  @property
  def _byte_strings(self):
    return tf.constant([bytes([i]) for i in range(self._byte_size)])

  @property
  def eos_id(self) -> Optional[int]:
    return 1

  @property
  def unk_id(self) -> Optional[int]:
    return 2

  def _convert_strings_to_ids(self, s):
    """Convert a python string to integers based on UTF-8 encoding.

    Args:
      s: a string

    Returns:
      ids: a list of integers
    """
    return list(s.encode("utf-8"))

  def _convert_ids_to_strings(self, ids):
    """Convert ids to a python string based on UTF-8 encoding.

    Args:
      ids: a list of integers

    Returns:
      s: a string
    """
    return bytes(ids).decode("utf-8", errors="ignore")

  def _filter_non_string_ids(self, ids):
    """Filter special token ids and extra ids if there are any.

    Args:
      ids: a list of integers

    Returns:
      ids: a list of integers
    """
    lower_bound = self._num_special_tokens
    upper_bound = self._byte_size + self._num_special_tokens
    return [id for id in ids if lower_bound <= id < upper_bound]

  @property
  def _base_vocab_size(self):
    """Number of ids.

    Returns:
      an integer, the vocabulary size
    """
    return self._num_special_tokens + self._byte_size

  def _encode(self, s):
    """Encode a python string as a list of integers.

    To keep the first few ids for special tokens, increase ids by the number
    of special tokens.

    Args:
      s: a string

    Returns:
      a list of integers (not terminated by EOS)
    """
    ids = self._convert_strings_to_ids(s)
    return [i + self._num_special_tokens for i in ids]

  def _decode(self, ids):
    """Decode a list of integers to a python string.

    The special tokens of PAD, EOS, and UNK will not be represented in the
    output string. This is different from the SentencePieceVocabulary, where
    UNK will show up as a '?' character.

    Args:
      ids: a list of integers (not terminated by EOS)

    Returns:
      a string
    """
    ids = [int(i) for i in ids]
    ids = self._filter_non_string_ids(ids)
    ids = [i - self._num_special_tokens for i in ids]
    return self._convert_ids_to_strings(ids)

  def _encode_tf(self, s):
    """Encode a tf.Scalar string to a tf.Tensor.

    Args:
      s: a tf.Scalar with dtype tf.string

    Returns:
      a 1d tf.Tensor with dtype tf.int32
    """
    tf_ids = tf.io.decode_raw(s, tf.uint8) + self._num_special_tokens
    return tf.dtypes.cast(tf_ids, tf.int32)

  def _decode_tf(self, ids):
    """Decode in TensorFlow.

    Args:
      ids: a n-d tf.Tensor with dtype tf.int32

    Returns:
      a n-d tf.Tensor with dtype tf.string
    """
    lower_bound = self._num_special_tokens
    upper_bound = self._byte_size + self._num_special_tokens
    ids = tf.ragged.boolean_mask(
        data=ids,
        mask=tf.math.logical_and(
            tf.math.greater_equal(ids, lower_bound),
            tf.math.less(ids, upper_bound),
        ),
    )
    ids = ids - self._num_special_tokens
    string = tf.strings.reduce_join(tf.gather(self._byte_strings, ids), axis=-1)

    # Drop invalid byte sequences.
    return tf.strings.unicode_transcode(
        input=string,
        input_encoding="UTF-8",
        output_encoding="UTF-8",
        errors="ignore",
    )

  def __eq__(self, other):
    if not isinstance(other, ByteVocabulary):
      return False
    return (
        self.extra_ids == other.extra_ids
        and self.eos_id == other.eos_id
        and self.unk_id == other.unk_id
    )


class FullCodepointVocabulary(Vocabulary):
  """Encodes and decodes text as codepoint sequences.

  This "vocabulary" is lexicon-free (i.e. it is static), and is an exhaustive
  representation of all codepoints. This is well-suited to encoders (especially
  with a hash-based embedding strategy) or a decoder that does not softmax over
  the whole vocabulary.

  A Unicode codepoint is effectively a single character. Unicode provides a
  well-defined mapping from the set of codepoint integers onto the set of all
  Unicode characters.
  """

  # While this should generally match `sys.maxunicode`, we want to provide this
  # as a constant to avoid architecture/system-dependent array overruns. If
  # downstream preprocessors choose to use `vocab_size-1` as a sentinel ID,
  # then this will still map such characters onto the Unicode private range on
  # planes 15-16. See:
  # https://en.wikipedia.org/wiki/Unicode#Code_planes_and_blocks.
  LARGEST_CODEPOINT = 0x10FFFF  # Decimal: 1,114,111
  # Padding is always index zero. This means that the NULL character is
  # technically not embeddable. This seems fine according to all reasonable
  # interpretations of the NULL character as a past-end-of-string marker.
  PAD_CODEPOINT = 0
  # Special symbols are represented using codepoints values that are valid,
  # but designated as "Private Use", meaning that they will never by assigned
  # characters by the Unicode Consortium, and are thus safe for use here.
  EOS_CODEPOINT = 0xE005

  @property
  def eos_id(self) -> int:
    return self.EOS_CODEPOINT

  @property
  def pad_id(self) -> int:
    return self.PAD_CODEPOINT

  @property
  def unk_id(self) -> Optional[int]:
    # Because `FullCodepointVocabulary` exhaustively embeds all codepoints
    # possible in Unicode, unknown characters are not possible.
    return None

  @property
  def _base_vocab_size(self) -> int:
    return self.LARGEST_CODEPOINT

  def _encode(self, s: str) -> Sequence[int]:
    return [ord(i) for i in s]

  def _decode(self, ids: Sequence[int]) -> str:
    ids = [int(i) for i in ids]
    return "".join(chr(id_) for id_ in ids if id_ != self.EOS_CODEPOINT)

  def _encode_tf(self, s: tf.Tensor) -> tf.Tensor:
    return tf.strings.unicode_decode(s, input_encoding="UTF-8")

  def _decode_tf(self, ids: tf.Tensor) -> tf.Tensor:
    return tf.strings.unicode_encode(ids, output_encoding="UTF-8")

  def __eq__(self, other):
    return isinstance(other, FullCodepointVocabulary)


class PartialCodepointVocabulary(Vocabulary):
  """Encodes and decodes text as a fixed set of codepoints.

  A Unicode codepoint is effectively a single character. Unicode provides a
  well-defined mapping from the set of codepoint integers onto the set of all
  Unicode characters.

  Unlike `FullCodepointVocabulary`, this uses only a subset of codepoints which
  are read in from a provided file. The format of the file is as decimal
  integers, where each integer is the codepoint integer as defined by Unicode.
  These can be obtained in Python 3 by converting a single character `str` to
  an `int` using `codepoint = ord(char)`.

  This sort of vocabulary is especially useful for decoder vocabularies where
  one might want to control the size of the output softmax and for encoders
  that do not use a hash embedding strategy.
  """

  # Padding is always index zero. This means that the NULL character is
  # technically not embeddable. This seems fine according to all reasonable
  # interpretations of the NULL character as a past-end-of-string marker.
  PAD_CODEPOINT = FullCodepointVocabulary.PAD_CODEPOINT
  # Special symbols are represented using codepoints values that are valid,
  # but designated as "Private Use", meaning that they will never by assigned
  # characters by the Unicode Consortium, and are thus safe for use here.
  EOS_CODEPOINT = FullCodepointVocabulary.EOS_CODEPOINT
  UNK_CODEPOINT = 0xE004

  PAD_ID = 0
  EOS_ID = 1
  UNK_ID = 2

  def __init__(self, codepoints: Sequence[int], extra_ids: int = 0):
    """Format of vocab file assumes one codepoint per line."""
    self._codepoint_to_id = {
        self.PAD_CODEPOINT: self.PAD_ID,
        self.EOS_CODEPOINT: self.EOS_ID,
        self.UNK_CODEPOINT: self.UNK_ID,
    }
    for codepoint in codepoints:
      if codepoint not in self._codepoint_to_id:
        self._codepoint_to_id[codepoint] = len(self._codepoint_to_id)
    self._id_to_codepoint = {v: k for k, v in self._codepoint_to_id.items()}

    self._codepoint_to_id_tf = PartialCodepointVocabulary.convert_dict_to_tf(
        self._codepoint_to_id, default_value=self.UNK_ID
    )
    self._id_to_codepoint_tf = PartialCodepointVocabulary.convert_dict_to_tf(
        self._id_to_codepoint, default_value=self.unk_id
    )
    super().__init__(extra_ids=extra_ids)

  @classmethod
  def create_from_file(cls, vocab_file: str, extra_ids: int = 0):
    codepoint_list = []
    with tf.io.gfile.GFile(vocab_file, "r") as f:
      for line in f:
        codepoint_list.append(int(line.strip()))
    return cls(codepoint_list, extra_ids)

  @property
  def eos_id(self) -> int:
    return self.EOS_ID

  @property
  def pad_id(self) -> int:
    return self.PAD_ID

  @property
  def unk_id(self) -> int:
    return self.UNK_ID

  @property
  def _base_vocab_size(self) -> int:
    return len(self._codepoint_to_id)

  @staticmethod
  def convert_dict_to_tf(
      d: Dict[Any, Any], default_value: Optional[Any] = None
  ) -> tf.lookup.StaticHashTable:
    keys_tensor = tf.constant(list(d))
    vals_tensor = tf.constant(list(d.values()))
    return tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor),
        default_value=default_value,
    )

  def _encode(self, s: str) -> Sequence[int]:
    output_ids = []
    for c in s:
      codepoint = ord(c)
      output_ids.append(self._codepoint_to_id.get(codepoint, self.unk_id))
    return output_ids

  def _decode(self, ids: Sequence[int]) -> str:
    output_str = ""
    for id_ in ids:
      codepoint = self._id_to_codepoint.get(int(id_), self.UNK_CODEPOINT)
      if codepoint == self.EOS_CODEPOINT:
        continue
      output_str += chr(codepoint)
    return output_str

  def _encode_tf(self, s: tf.Tensor) -> tf.Tensor:
    return self._codepoint_to_id_tf[
        tf.strings.unicode_decode(s, input_encoding="UTF-8")
    ]

  def _decode_tf(self, ids: tf.Tensor) -> tf.Tensor:
    return tf.strings.unicode_encode(
        self._id_to_codepoint_tf[ids], output_encoding="UTF-8"
    )

  def __eq__(self, other):
    if not isinstance(other, PartialCodepointVocabulary):
      return False
    return (
        self._codepoint_to_id == other._codepoint_to_id
        and self.extra_ids == other.extra_ids
    )


class BertWordPieceVocabulary(Vocabulary):
  """Wrapper for Bert wordpiece encoder.

  This "vocabulary" wraps the tensorflow_text's BertTokenizer, which applies an
  end-to-end, text string to wordpiece tokenization.
  """

  def __init__(
      self,
      vocab_lookup_table: str,
      suffix_indicator: str = "##",
      max_bytes_per_word: int = 100,
      max_chars_per_token: Optional[int] = None,
      token_out_type: tf.dtypes.DType = tf.dtypes.int64,
      unknown_token: str = "[UNK]",
      split_unknown_characters: bool = False,
      lower_case: bool = False,
      keep_whitespace: bool = False,
      normalization_form: Optional[str] = None,
      preserve_unused_token: bool = False,
      pad_id: int = 0,
      start_of_sequence_id: int = 101,
      end_of_sequence_id: int = 102,
  ):
    r"""Create a Bert WordPieceVocabulary.

    Args:
      vocab_lookup_table: A lookup table implementing the LookupInterface
        containing the vocabulary of subwords or a string which is the file path
        to the vocab.txt file.
      suffix_indicator: (optional) The characters prepended to a wordpiece to
        indicate that it is a suffix to another subword. Default is '##'.
      max_bytes_per_word: (optional) Max size of input token. Default is 100.
      max_chars_per_token: (optional) Max size of subwords, excluding suffix
        indicator. If known, providing this improves the efficiency of decoding
        long words.
      token_out_type: (optional) The type of the token to return. This can be
        `tf.int64` IDs, or `tf.string` subwords. The default is `tf.int64`.
      unknown_token: (optional) The value to use when an unknown token is found.
        Default is "[UNK]". If this is set to a string, and `token_out_type` is
        `tf.int64`, the `vocab_lookup_table` is used to convert the
        `unknown_token` to an integer. If this is set to `None`,
        out-of-vocabulary tokens are left as is.
      split_unknown_characters: (optional) Whether to split out single unknown
        characters as subtokens. If False (default), words containing unknown
        characters will be treated as single unknown tokens.
      lower_case: bool - If true, a preprocessing step is added to lowercase the
        text, apply NFD normalization, and strip accents characters.
      keep_whitespace: bool - If true, preserves whitespace characters instead
        of stripping them away.
      normalization_form: If set to a valid value and lower_case=False, the
        input text will be normalized to `normalization_form`. See
        normalize_utf8() op for a list of valid values.
      preserve_unused_token: If true, text in the regex format
        `\\[unused\\d+\\]` will be treated as a token and thus remain preserved
        as is to be looked up in the vocabulary.
      pad_id: ID for the `[PAD]` token.
      start_of_sequence_id: ID for the `[CLS]` token.
      end_of_sequence_id: ID for the `[SEP]` token.
    """
    self._vocab_lookup_table = vocab_lookup_table
    self._suffix_indicator = suffix_indicator
    self._max_bytes_per_word = max_bytes_per_word
    self._max_chars_per_token = max_chars_per_token
    self._token_out_type = token_out_type
    self._unknown_token = unknown_token
    self._split_unknown_characters = split_unknown_characters
    self._lower_case = lower_case
    self._keep_whitespace = keep_whitespace
    self._normalization_form = normalization_form
    self._preserve_unused_token = preserve_unused_token
    self._tokenizer = tf_text.BertTokenizer(
        vocab_lookup_table=vocab_lookup_table,
        suffix_indicator=suffix_indicator,
        max_bytes_per_word=max_bytes_per_word,
        max_chars_per_token=max_chars_per_token,
        token_out_type=token_out_type,
        unknown_token=unknown_token,
        split_unknown_characters=split_unknown_characters,
        lower_case=lower_case,
        keep_whitespace=keep_whitespace,
        normalization_form=normalization_form,
        preserve_unused_token=preserve_unused_token,
    )
    self._vocab = self._tokenizer._wordpiece_tokenizer._vocab_lookup_table
    self._pad_id = pad_id
    self._unk_id = self._vocab.lookup(tf.constant(unknown_token)).numpy()
    self._sos_id = start_of_sequence_id
    self._eos_id = end_of_sequence_id
    with tf.io.gfile.GFile(vocab_lookup_table, "rb") as f:
      self._wp_model = f.read()
    # We won't pass in extra_ids for Bert vocabulary.
    super().__init__()

  @property
  def sos_id(self) -> Optional[int]:
    return self._sos_id

  @property
  def eos_id(self) -> Optional[int]:
    return self._eos_id

  @property
  def unk_id(self) -> Optional[int]:
    return self._unk_id

  @property
  def pad_id(self) -> Optional[int]:
    return self._pad_id

  @property
  def _base_vocab_size(self):
    """Returns the vocabulary size."""
    return self._vocab.size().numpy()

  @property
  def tokenizer(self):
    """Returns the Python tokenizer."""
    return self._tokenizer

  @property
  def tf_tokenizer(self):
    """Instantiate and return a TF tokenizer."""
    return self._tokenizer

  @property
  def vocab_size(self):
    return self._base_vocab_size

  def _encode(self, s):
    """Encode a python string as a list of integers.

    Args:
      s: a string

    Returns:
      a list of integers (not terminated by EOS)
    """
    return self._encode_tf(s).numpy()

  def _decode(self, ids):
    """Decode a list of integers to a python string.

    Args:
      ids: a list of integers (not terminated by EOS)

    Returns:
      a string
    """
    ids = tf.constant(ids)
    str_text = self._decode_tf(ids)
    return str_text.numpy().decode("UTF-8")

  def _encode_tf(self, s):
    """Encode a tf.Scalar string to a tf.Tensor.

    This will be necessary for on-the-fly tokenization.

    Args:
      s: a tf.Scalar with dtype tf.string

    Returns:
      a 1d tf.Tensor with dtype tf.int32
    """
    tokens = self.tokenizer.tokenize(s)
    # Convert tf.RaggedTensor to tf.Tensor
    return tf.squeeze(tokens.to_tensor())

  def _decode_tf(self, ids):
    """Decode in TensorFlow.

    Args:
      ids: a 1d tf.Tensor with dtype tf.int32

    Returns:
      a tf Scalar with dtype tf.string
    """
    # Convert tf.Tensor to tf.RaggedTensor
    ids = tf.RaggedTensor.from_tensor(tf.expand_dims(ids, axis=1))
    tokens = self.tf_tokenizer.detokenize(ids)
    # Flatten tf.RaggedTensor and convert tokens into a string
    return tf.strings.join(tokens.flat_values, " ")

  def __eq__(self, other):
    try:
      their_md5 = hashlib.md5(other._wp_model).hexdigest()
      their_sos_id = other._sos_id
    except AttributeError:
      return False
    our_md5 = hashlib.md5(self._wp_model).hexdigest()
    return our_md5 == their_md5 and self._sos_id == their_sos_id