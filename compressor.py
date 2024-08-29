"""Runs the compressor"""

from collections.abc import Generator
import functools
import time
from typing import Callable, Mapping

from absl import app
from absl import flags
from absl import logging

import tqdm
import numpy as np
import gzip
import random

import btransformer
import llama


COMPRESS_FN_DICT: Mapping[str, Compressor] = {
    'gzip': functools.partial(gzip.compress, compresslevel=9),
    'transformer': btransformer.compress,
    'llama': llama.compress,
}

def evaluate_compressor(
    compress_fn_name: str,
    data: list[list[float | int]],
    mask_fn,
    use_tqdm: bool = True,
) -> tuple[float, float]:
  """Evaluates the compressor on the chunked dataset.

  Args:
    compress_fn: The function that evaluates data.
    data: List of the data to consider
    mask_fn: The function that masks the data in case the compressor cannot
      handle all possible byte values (e.g., language models can only process
      ASCII-decodable data).
    use_tqdm: Whether to use a progress bar or not.

  Returns:
    The compression rate and the total running time.
  """

  num_missed_bits = running_time = raw_length = compressed_length = 0

  # n_chunks, how many data we are copmressing
  # chunk_size, length of each individaul data
  n_chunks = len(data)
  context_l = len(data[0])

  compress_fn = COMPRESS_FN_DICT['compress_fn_name']

  for datum in data:
    if mask_fn is not None:
      datum, missed_bits = mask_fn(datum)
      num_missed_bits += missed_bits

    t0 = time.perf_counter()
    compressed_data = compress_fn(datum)
    t1 = time.perf_counter()

    running_time += t1 - t0
    raw_length += len(data)
    compressed_length += len(compressed_data)

  # Since language models are trained on ASCII strings, they cannot handle all
  # byte values. Thus, we mask the data to be ASCII-decodable by zeroing
  # `num_missed_bits` of the most significant bits. However, this means that we
  # are effectively only compressing `num_bits - num_missed_bits` bits, so we
  # rescale the `compressed_length` to account for this.
  if mask_fn is not None:
    num_bits = 8 * n_chunks * context_l
    compressed_length *= num_bits / (num_bits - num_missed_bits)

  # We only count the header once for classical compressors.
  if compress_fn_name == 'gzip':
    header_length = len(compress_fn((0).to_bytes(1, 'little')))
    compressed_length -= header_length * (n_chunks - 1) 

  return compressed_length / raw_length, running_time

def main(_) -> None:
  logging.info('Compressor: %s', _COMPRESSOR.value)
  logging.info('Dataset: %s', _DATASET.value)

  compress_fn = compressor.COMPRESS_FN_DICT[_COMPRESSOR.value]
  get_data_generator_fn = functools.partial(
      data_loaders.GET_DATA_GENERATOR_FN_DICT[_DATASET.value],
      num_chunks=_NUM_CHUNKS.value,
  )

  if _COMPRESSOR.value in compressor.COMPRESSOR_TYPES['classical']:
    unchunked_rate, unchunked_time = evaluate_compressor_unchunked(
        compress_fn=compress_fn,
        get_data_generator_fn=get_data_generator_fn,
        num_chunks=_NUM_CHUNKS.value,
    )
    chunked_rate, chunked_time = evaluate_compressor_chunked(
        compress_fn=compress_fn,
        get_data_generator_fn=get_data_generator_fn,
        num_chunks=_NUM_CHUNKS.value,
        count_header_only_once=True,
        mask_fn=None,
    )
    logging.info(
        'Unchunked: %.1f [%.1fs]', 100 * unchunked_rate, unchunked_time
    )
    logging.info('Chunked: %.1f [%.1fs]', 100 * chunked_rate, chunked_time)

  elif _COMPRESSOR.value in compressor.COMPRESSOR_TYPES['arithmetic_coding']:
    # To compress bytes data, we convert it first to ASCII.
    if _DATASET.value == 'enwik9':
      # For Enwik9, some characters are UTF-8 but not ASCII, so we still need
      # to do the conversion.
      mask_fn = utils.zero_most_significant_bit_if_not_ascii_decodable
    else:
      #mask_fn = utils.right_shift_bytes_by_one
      mask_fn = None

    chunked_rate, chunked_time = evaluate_compressor_chunked(
        compress_fn=compress_fn,
        get_data_generator_fn=get_data_generator_fn,
        num_chunks=_NUM_CHUNKS.value,
        count_header_only_once=False,
        mask_fn=mask_fn,
    )
    logging.info('Chunked: %.1f [%.1fs]', 100 * chunked_rate, chunked_time)


if __name__ == '__main__':
  app.run(main)
