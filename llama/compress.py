import numpy as np
import logging.config
import constants
import math
import matplotlib.pyplot as plt
import seaborn as sns
import arithmetic_coder
import utils
import llama.llama as llama
import llama.llmtime as llmtime
from llama.serialize import SerializerSettings
from functools import partial
import scipy.stats as stats
from transformers import LlamaTokenizer

# First order compress
def fo_compress(
    data: bytes,
    params,
    config,
    return_num_padded_bits: bool = True,
) -> bytes | tuple[bytes, int]:

  logging.config.dictConfig(constants.LOGGING_CONFIG)
  logger = logging.getLogger(__name__) 
  logger.info('Initialising btransformer compression')
  logger.info(f'Length of data is: {len(data)}')

  settings=SerializerSettings(base=10, prec=0, signed=True, half_bin_correction=False, time_sep=',', bit_sep = '', max_val = 255) 

  predict_fn = partial(llmtime.get_llmtime_predictions_data, test = [0.], 
                       model = 'llama-7b',
                       settings = settings,
                       num_samples = 1,
                       )
  tokenizer = LlamaTokenizer.from_pretrained(
    'meta-llama/Llama-2-7b-hf',
    use_fast = False
    )

  good_tokens_str = list("0123456789")
  good_tokens = [tokenizer.convert_tokens_to_ids(token) for token in good_tokens_str]

  sequence_array = np.frombuffer(data, dtype=np.uint8)
  sequence_array = sequence_array.astype(dtype = np.float32)
  sequence_array = [0.] + sequence_array

  log_probs = list()
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

  total_b = 0
  compressed_b = 0
  bits_per_symbol = 8

  for offset in range(len(sequence_array)-1):
    subsequence_probs = predict_fn(
        sequence_array[ :offset+1] # might need a +1
    )
    
    plt.figure()
    cm = subsequence_probs['scores'][0][:, good_tokens][0].cpu()
    #cm_exp = utils.normalize_pdf_for_arithmetic_coding(np.exp(cm))
    cm_exp = utils.normalize_pdf_for_arithmetic_coding(cm + -1.1*min(cm))
    cm_var = np.mean([idx**2 * x for idx, x in enumerate(cm_exp)]) - (np.mean([idx*x for idx, x, in enumerate(cm_exp)]))**2
    print(cm_var)
    plt.plot(cm_exp)
    plt.text(9, 0.2, f'{cm_var}')
    plt.xticks(range(10),[str(x) for x in np.arange(10)])
    plt.savefig('figs/tmp/llama_confidence.png')
    plt.close()


    symbol = (sequence_array[offset+1]).astype(np.uint8)
    predicted_symbol = int(subsequence_probs['median'].values[0])

    # Calculate pdf
    loc = predicted_symbol
    scale = 10*cm_var# Needs to be a function of the confidence of the model
    #a_trunc = 0
    #b_trunc = 255
    #a, b = (a_trunc - loc) / scale, (b_trunc - loc) / scale
    x = np.arange(256)
    #pdf = np.array(utils.normalize_pdf_for_arithmetic_coding(
    #    stats.truncnorm.pdf(x, a, b, loc, scale)
    #))
    pdf = np.array(utils.normalize_pdf_for_arithmetic_coding(
      stats.cauchy.pdf(x, loc, scale)
    ))
    plt.figure()
    plt.plot(pdf)
    plt.axvline(symbol, ymin = 0, ymax =pdf[symbol]/max(pdf),color = 'r')
    plt.savefig('figs/tmp/llama_choice_graph.png')
    plt.close()

    sequence_q.append(symbol)
    pdf_q.append(pdf)
    
    print(pdf.shape)
    #import sys
    #sys.exit(-1)

    encoder.encode(pdf, symbol)

    plt.figure()
    plt.gca().set_aspect(0.2)
    A = np.array(list(pdf_q))
    sns.heatmap(A.T)
    plt.xlim(0, A.shape[0])
    plt.ylim(0, A.shape[1])
    plt.plot(list(sequence_q))
    plt.savefig('figs/tmp/llama_smooth_heatmap.png')
    plt.close()
    cdf = np.cumsum(pdf)
    cdf2 = [x**2 if i < symbol else 1-(1-x)**2 for i, x in enumerate(cdf)]
    indicator = np.array([0 if idx < symbol else 1 for idx in range(len(cdf))])
    crps = np.sum((cdf-indicator)**2)
    crps_q.append(crps)
    fig, ax = plt.subplots()
    plt.plot(cdf2, label = 'CDF Square Adjustment')
    plt.plot(cdf, label = 'CDF')
    plt.plot(indicator, label = 'Correct symbol')
    ax.fill_between(np.arange(0,256), np.maximum(cdf2, indicator), np.minimum(cdf2, indicator), color="crimson", alpha=0.4)
    plt.text(0.1, 0.5, f'CRPS = {crps:.02f}')
    plt.legend()
    plt.title('CRPS Graph')
    plt.xlabel('Symbol value')
    plt.ylabel('Density')
    plt.savefig('figs/tmp/llama_crps_smooth.png')
    plt.close()
    compressed_bits = ''.join(map(str, output))
    n_bits = len(compressed_bits) - prev_len

    total_b += bits_per_symbol
    compressed_b += n_bits

    coder_rep = str(compressed_bits[-n_bits:]) if n_bits != 0 else '_'
    logger.info(f'Encoded {offset}th byte ({symbol}) as {coder_rep} : {n_bits} bits @ {pdf[symbol]*100}%')
    logger.info(f'Running compression is {compressed_b/total_b*100}%, CRPS: {np.mean(crps_q)}')
    prev_len = len(compressed_bits)

  encoder.terminate()
  compressed_bits = ''.join(map(str, output))
  compressed_bytes, num_padded_bits = utils.bits_to_bytes(compressed_bits)

  if return_num_padded_bits:
    return compressed_bytes, num_padded_bits

  return compressed_bytes