"""Implements a lossless compressor with language models (arithmetic coding)."""
from scipy.special import softmax
from collections.abc import Iterator
import functools
from typing import Callable
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import os
import pickle
from scipy.stats import norm

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
import yaml



def _retrieve_model_params(which_model) -> hk.Params:
  """Returns the trained model parameters.

  Raises:
    FileNotFoundError if the file params.npz does not exist yet, in which case
    the user should launch a training with train.py first.
  """
  try:
    with np.load(f'params/{which_model}/params.npz', allow_pickle=True) as data:
      return {key: data[key].item() for key in data.files}
  except FileNotFoundError as exc:
    raise FileNotFoundError(
        'You must train a model first, the parameters file params.npz does not'
        ' exist yet.'
    ) from exc


def _retrieve_predict_fn(
    params: hk.Params,
    config,
) -> Callable[[np.ndarray], np.ndarray]:
  """Returns the prediction function for the trained model."""
  #config = constants.TRANSFORMER_CONFIG
  model = hk.transform(
      functools.partial(transformer.transformer_decoder, config=config)
  )
  return lambda x: model.apply(params, None, x)

# Needs to be replaced with its own thing TODO
def _get_llama():
    return functools.partial(llama.llama.llama_completion_fn, settings = constants.LLAMA_CONFIG)

def get_scaled_input(data_in):

  minx = min(data_in[0][0:-1])
  maxx = max(data_in[0][0:-1])
  return (((data_in.astype(np.uint32) - minx) * 127) // (maxx-minx)).astype(np.uint8)

def scale_pdf(pdf, a, b, sigma):

  # Create a new array to hold the transformed probabilities
  transformed_distribution = np.zeros(128)
  
  # Calculate the mean of the original distribution
  
  # Iterate over the original distribution
  for x in range(128):
      if pdf[x] > 0:
          # Scale to the new range
          y = (x / 127) * (b - a) + a
          
          # Calculate the Gaussian kernel
          kernel = norm.pdf(np.arange(128), loc=y, scale=sigma)
          
          # Update the transformed distribution
          transformed_distribution += pdf[x] * kernel
  
  # Normalize the transformed distribution
  transformed_distribution /= np.sum(transformed_distribution)
  
  # Zero out values outside the new range [a, b]
  transformed_distribution[:a] = 0
  transformed_distribution[b:] = 0

  return transformed_distribution


def compress_smooth(
    data: bytes,
    which_model,
    config,
    return_num_padded_bits: bool = True,
) -> bytes | tuple[bytes, int]:
  """Compresses the `data` using arithmetic coding and a pretrained model. 
  But with slow compression and a moving context window, must be slow and lossless

  Args:
    data: The data to be compressed.
    return_num_padded_bits: Whether to return the number of zeros added to the
      encoded bitstream in order to make it byte-decodeable (i.e., divisible by
      8). Usually, this is used when the encoded data has to be decoded again.

  Returns:
    The compressed data.
  """
  # Logger
  logging.config.dictConfig(constants.LOGGING_CONFIG)
  logger = logging.getLogger(__name__) 
  logger.debug('Initialising smooth compression')
  logger.debug(f'Length of data is: {len(data)}')
  # Get figure checker
  do_fig = int(os.environ.get('LLMCOMP_FIG', '0'))

  params = _retrieve_model_params(which_model)
  predict_fn = _retrieve_predict_fn(params, config)

  # Info about model:
  with open(f'params/{which_model}/info.yml', 'r') as f:
    info = yaml.safe_load(f)

  bits_per_symbol = math.log2(info['vocab_size'])

  with open('hist.pkl', 'rb') as f:
    training_dist = pickle.load(f)

  sequence_array = np.frombuffer(data, dtype=np.uint8)

  output = list()

  encoder = arithmetic_coder.Encoder(
      base=2,
      precision=constants.CODER_PRECISION,
      output_fn=output.append,
  )
  
  from collections import deque
  pdf_q = deque(maxlen = 32)
  sequence_q = deque(maxlen = 32)
  crps_q = []
  prev_len = 0
  kld_sum = 0

  total_b = 0
  compressed_b = 0

  # Implement mRF 

  for offset in range(256, len(sequence_array)):

    siq = sequence_array[None, max(0, offset - info['cw']): offset + 1].copy()  
    ssiq = get_scaled_input(siq)

    subsequence_probs = predict_fn(
      ssiq
    )
    symbol = sequence_array[offset]
    probs = np.exp(subsequence_probs[0, -1])

    # Change distribution
    nprobs = scale_pdf(probs, min(siq[0][0:-1]), max(siq[0][0:-1]), 1)
    nprobs = utils.normalize_pdf_for_arithmetic_coding(nprobs)

    # sanity
    plt.figure()
    plt.plot(probs)
    plt.savefig('figs/tmp/vprobs.png')
    plt.close()
    plt.figure()
    plt.plot(nprobs)
    plt.savefig('figs/tmp/nprobs.png')
    plt.close()


    # Calculate the Kullback-Leibler (KL) divergence
    kld_sum += np.sum(
      training_dist * np.log2(training_dist/nprobs)
    )

    pdf_q.append(nprobs)
    sequence_q.append(symbol)
    encoder.encode(nprobs, symbol)

    # Here calculate and graph the CRPS
    cdf = np.cumsum(nprobs)
    cdf2 = [x**2 if i < symbol else 1-(1-x)**2 for i, x in enumerate(cdf)]
    indicator = np.array([0 if idx < symbol else 1 for idx in range(len(cdf))])
    crps = np.sum((cdf-indicator)**2)
    crps_q.append(crps)
    #fig, ax = plt.subplots()
    #plt.plot(cdf2, label = 'CDF Square Adjustment')
    #plt.plot(cdf, label = 'CDF')
    #plt.plot(indicator, label = 'Correct symbol')
    #ax.fill_between(np.arange(0,info['vocab_size']), np.maximum(cdf2, indicator), np.minimum(cdf2, indicator), color="crimson", alpha=0.4)
    #plt.text(0.1, 0.5, f'CRPS = {crps:.02f}')
    #plt.legend()
    #plt.title('CRPS Graph')
    #plt.xlabel('Symbol value')
    #plt.ylabel('Density')
    #plt.savefig('figs/tmp/crps_smooth.png')
    #plt.close()

    compressed_bits = ''.join(map(str, output))
    n_bits = len(compressed_bits) - prev_len

    total_b += bits_per_symbol
    compressed_b += n_bits

    coder_rep = str(compressed_bits[-n_bits:]) if n_bits != 0 else '_'
    logger.debug(f'Encoded {offset}th byte ({symbol}) as {coder_rep} : {n_bits} bits @ {nprobs[symbol]*100}%')
    logger.debug(f'Running compression is {compressed_b/total_b*100}%, CRPS: {np.mean(crps_q)}')
    prev_len = len(compressed_bits)
  
    # Heatmap
    if do_fig:
      plt.figure()
      plt.gca().set_aspect(0.2)
      A = np.array(list(pdf_q))
      sns.heatmap(A.T)
      plt.xlim(0, A.shape[0])
      plt.ylim(0, A.shape[1])
      plt.plot(list(sequence_q))
      plt.savefig('figs/tmp/smooth_heatmap.png')
      plt.close()

      # Distribution
      plt.figure()
      plt.plot(nprobs)
      plt.axvline(symbol, ymin = 0, ymax = nprobs[symbol]/max(nprobs),color = 'r')
      plt.savefig('figs/tmp/smooth_choice_graph.png')
      plt.close()


  encoder.terminate()
  compressed_bits = ''.join(map(str, output))
  logger.debug(f'Terminated, adding {len(compressed_bits)-prev_len} bits')

  compressed_bytes, num_padded_bits = utils.bits_to_bytes(compressed_bits)

  logger.debug(f'Savings bytes with {num_padded_bits} padded bits')
  with open('compressed_bytes.data', 'wb') as f:
    f.write(compressed_bytes)

  logger.info(f'Average KLD was {kld_sum/len(sequence_array)}')

  if return_num_padded_bits:
    return compressed_bytes, num_padded_bits

  return compressed_bytes

def compress(
    data: bytes,
    which_model: str,
    config,
    return_num_padded_bits: bool = True,
    use_slow_lossless_compression: bool = False,
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
  logging.config.dictConfig(constants.LOGGING_CONFIG)
  logger = logging.getLogger(__name__) 
  logger.debug('Initialising btransformer compression')
  logger.debug(f'Length of data is: {len(data)}')

  params = _retrieve_model_params(which_model)
  predict_fn = _retrieve_predict_fn(params, config)

  # Convert the `data` into an array of integers (representing the bytes).
  sequence_array = np.frombuffer(data, dtype=np.uint8)

  #with open('hist.pkl', 'rb') as f:
    #training_dist = pickle.load(f)

  if use_slow_lossless_compression:
    log_probs = list()
    for subsequence_length in range(len(sequence_array)):
      subsequence_probs = predict_fn(
          sequence_array[None, : subsequence_length + 1]
      )
      log_probs.append(subsequence_probs[0, -1])
     
    log_probs = np.vstack(log_probs)
  else:
    log_probs = predict_fn(sequence_array[None])[0, ...]
  probs = np.exp(log_probs)

  #kld = np.average(np.sum(
      #training_dist * np.log2(training_dist/probs),
      #axis = 1
  #))
    
  # Plotting
  #if which_compressor == 'llama':

  #  # Don't remove this
  #  probs = np.insert(probs,0,np.array([1/256]*256), axis = 0)
  #  probs = probs[:-1]

  #  plt.figure()
  #  plt.gca().set_aspect(0.08)
  #  A = probs
  #  sns.heatmap(A.T, vmin = 0, vmax = 0.05)
  #  plt.xlim(0, A.shape[0])
  #  plt.ylim(0, A.shape[1])
  #  dw = np.frombuffer(data, dtype = np.uint8)
  #  plt.plot(dw)
  #  plt.savefig('foo.png')
  
  #if which_compressor == 'btransformer':

  #A = probs
  #print(A.shape)
  #plt.figure()
  #plt.gca().set_aspect(32/256)
  #sns.heatmap(A[0:32, :].T)
  ##plt.xlim(0, A.shape[0])
  #plt.ylim(0, A.shape[1])
  #dw = np.frombuffer(data, dtype = np.uint8)
  #plt.plot(dw[0:32])
  #plt.savefig('figs/tmp/non_smooth_heatmap.png')
  #plt.close()

  output = list()
  encoder = arithmetic_coder.Encoder(
      base=2,
      precision=constants.CODER_PRECISION,
      output_fn=output.append,
  )

  idx = 0
  for pdf, symbol in zip(probs, sequence_array):
    encoder.encode(utils.normalize_pdf_for_arithmetic_coding(pdf), symbol)

  encoder.terminate()

  compressed_bits = ''.join(map(str, output))
  compressed_bytes, num_padded_bits = utils.bits_to_bytes(compressed_bits)


  if return_num_padded_bits:
    return compressed_bytes, num_padded_bits, 0

  return compressed_bytes

def decompress(
    data: bytes,
    num_padded_bits: int = 0,
    uncompressed_length: int = 256,
) -> bytes:
  """Decompresses the `data` using arithmetic coding and a pretrained model.

  See https://en.wikipedia.org/wiki/Arithmetic_coding for details.

  Args:
    data: The data to be decompressed.
    num_padded_bits: The number of zeros added to the encoded bitstream in order
      to make it byte-decodeable (i.e., divisble by 8).
    uncompressed_length: The length of the original data stream (in bytes).

  Returns:
    The decompressed data.
  """

  # Logger
  logging.config.dictConfig(constants.LOGGING_CONFIG)
  logger = logging.getLogger(__name__) 
  logger.info('Decompressing')

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
      base=2,
      precision=constants.CODER_PRECISION,
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
    logger.debug(f'Sequence array fed in is {sequence_array}')
    token = decoder.decode(
        utils.normalize_pdf_for_arithmetic_coding(probs[idx])
    )
    logger.debug(f'Decompressing {idx}th token, got {token}')
    sequence_array = np.insert(sequence_array, -1, token)
    #plt.figure()
    #plt.plot(sequence_array)
    #plt.savefig('figs/tmp/decom_output_live.png')
    #plt.close()
    probs = np.exp(predict_fn(sequence_array[None])[0, ...])

  # Remove the dummy token and convert to bytes.
  return sequence_array[:-1].tobytes()