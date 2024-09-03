"""Implements a lossless compressor with language models (arithmetic coding)."""

from collections.abc import Iterator
import functools
from typing import Callable
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import haiku as hk
import numpy as np

import arithmetic_coder
import utils
from btransformer import transformer

import llama.llama

import constants
import logging.config

import sys
import audioop


def _retrieve_model_params() -> hk.Params:
  """Returns the trained model parameters.

  Raises:
    FileNotFoundError if the file params.npz does not exist yet, in which case
    the user should launch a training with train.py first.
  """
  try:
    with np.load('btransformer/params.npz', allow_pickle=True) as data:
      return {key: data[key].item() for key in data.files}
  except FileNotFoundError as exc:
    raise FileNotFoundError(
        'You must train a model first, the parameters file params.npz does not'
        ' exist yet.'
    ) from exc


def _retrieve_predict_fn(
    params: hk.Params,
) -> Callable[[np.ndarray], np.ndarray]:
  """Returns the prediction function for the trained model."""
  config = constants.TRANSFORMER_CONFIG
  model = hk.transform(
      functools.partial(transformer.transformer_decoder, config=config)
  )
  return lambda x: model.apply(params, None, x)

def _get_llama():
    return functools.partial(llama.llama.llama_completion_fn, settings = constants.LLAMA_CONFIG)


def compress(
    data: bytes,
    which_compressor : str,
    return_num_padded_bits: bool = False,
    use_slow_lossless_compression: bool = True,
) -> bytes | tuple[bytes, int]:
  """Compresses the `data` using arithmetic coding and a pretrained model.

  Args:
    data: The data to be compressed.
    return_num_padded_bits: Whether to return the number of zeros added to the
      encoded bitstream in order to make it byte-decodeable (i.e., divisible by
      8). Usually, this is used when the encoded data has to be decoded again.
    use_slow_lossless_compression: Whether to compute the `pdf`s for all tokens
      in the data stream in one go or separately for every proper subsequence.
      When only compressing data (i.e., without decompression) use the first
      approach (i.e., `True`) since it has an O(n) runtime complexity, while the
      latter is O(n^2). However, the goal is to losslessly decompress the
      compressed output, use the second option (i.e., `False`) since this is
      what happens in the decoder (which iteratively reconstructs the sequence).

  Returns:
    The compressed data.
  """
  # Logger
  logging.config.dictConfig(constants.LOGGING_CONFIG)
  logger = logging.getLogger(__name__) 

  if which_compressor == 'btransformer': 
    params = _retrieve_model_params()
    predict_fn = _retrieve_predict_fn(params)
  elif which_compressor == 'llama':
    predict_fn = _get_llama()
  else:
    logger.error('No valid compressor name, quitting')
    sys.exit(-1)


  # Convert the `data` into an array of integers (representing the bytes).
  sequence_array = np.frombuffer(data, dtype=np.uint8)
  #plt.figure()
  #plt.plot(sequence_array)
  #plt.savefig('foobar.png')

  if use_slow_lossless_compression:
    log_probs = list()
    for subsequence_length in range(len(sequence_array)):
      subsequence_probs = predict_fn(
          sequence_array[None, : subsequence_length + 1]
      )
      log_probs.append(subsequence_probs[0, -1])
     
    log_probs = np.vstack(log_probs)
  else:
    print('debug')
    log_probs = predict_fn(sequence_array[None])[0, ...]
    print(f'{log_probs=}\n{log_probs.shape=}')
  probs = np.exp(log_probs)

  print(f'log_probs size is {log_probs.shape}')


  # Plotting
  if which_compressor == 'llama':

    # Don't remove this
    probs = np.insert(probs,0,np.array([1/256]*256), axis = 0)
    probs = probs[:-1]

    plt.figure()
    plt.gca().set_aspect(0.08)
    A = probs
    sns.heatmap(A.T, vmin = 0, vmax = 0.05)
    plt.xlim(0, A.shape[0])
    plt.ylim(0, A.shape[1])
    dw = np.frombuffer(data, dtype = np.uint8)
    plt.plot(dw)
    plt.savefig('foo.png')
  
  if which_compressor == 'btransformer':
    plt.figure()
    plt.gca().set_aspect(0.1)
    A = probs
    sns.heatmap(A.T)
    plt.xlim(0, A.shape[0])
    plt.ylim(0, A.shape[1])
    dw = np.frombuffer(data, dtype = np.uint8)
    plt.plot(dw)
    plt.savefig('foo.png')

  output = list()
  encoder = arithmetic_coder.Encoder(
      base=2,
      precision=32,
      output_fn=output.append,
  )

  for pdf, symbol in zip(probs, sequence_array):
    encoder.encode(utils.normalize_pdf_for_arithmetic_coding(pdf), symbol)

  encoder.terminate()

  compressed_bits = ''.join(map(str, output))
  compressed_bytes, num_padded_bits = utils.bits_to_bytes(compressed_bits)

  if return_num_padded_bits:
    return compressed_bytes, num_padded_bits

  return compressed_bytes


# Ignore decompress for now
"""
def decompress(
    data: bytes,
    num_padded_bits: int = 0,
    uncompressed_length: int = constants.CHUNK_SIZE_BYTES,
) -> bytes:
  ""Decompresses the `data` using arithmetic coding and a pretrained model.

  See https://en.wikipedia.org/wiki/Arithmetic_coding for details.

  Args:
    data: The data to be decompressed.
    num_padded_bits: The number of zeros added to the encoded bitstream in order
      to make it byte-decodeable (i.e., divisble by 8).
    uncompressed_length: The length of the original data stream (in bytes).

  Returns:
    The decompressed data.
  ""
  params = _retrieve_model_params()
  predict_fn = _retrieve_predict_fn(params)

  data_iter = iter(utils.bytes_to_bits(data, num_padded_bits=num_padded_bits))

  # The decoder requires a function that reads digits from {0, 1, ..., base - 1}
  # from the compressed input and returns `None` when the input is exhausted.
  def _input_fn(bit_sequence: Iterator[str] = data_iter) -> int | None:
    try:
      return int(next(bit_sequence))
    except StopIteration:
      return None

  decoder = arithmetic_coder.Decoder(
      base=constants.ARITHMETIC_CODER_BASE,
      precision=constants.ARITHMETIC_CODER_PRECISION,
      input_fn=_input_fn,
  )
  # We need a dummy token because the language model right-shifts the sequence
  # by one when computing the conditional probabilities. Concretely, at every
  # step, we need the `pdf` of the next token given all currently decompressed
  # tokens, but without a dummy token, the last `pdf` would be that of the last
  # already decompressed token. The value of the dummy token is irrelevant.
  sequence_array = np.empty((1,), dtype=np.uint8)
  probs = np.exp(predict_fn(sequence_array[None])[0, ...])

  for idx in range(uncompressed_length):
    token = decoder.decode(
        utils.normalize_pdf_for_arithmetic_coding(probs[idx])
    )
    sequence_array = np.insert(sequence_array, -1, token)
    probs = np.exp(predict_fn(sequence_array[None])[0, ...])

  # Remove the dummy token and convert to bytes.
  return sequence_array[:-1].tobytes()
"""