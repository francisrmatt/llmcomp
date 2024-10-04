# LLama character-based compression

# Imports
import arithmetic_coder
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig
import pandas as pd
import numpy as np
import utils
import audioop
import constants
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import torch
from torch.nn import functional as F
import gc

# Loggging
import logging.config
logging.config.dictConfig(constants.LOGGING_CONFIG)
logger = logging.getLogger()

# Set up device, tokenizer, and model
device = torch.device('cuda')
tokenizer = LlamaTokenizer.from_pretrained(
    'meta-llama/Llama-2-7b-hf',
    use_fast = False,
    padding=True,
)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.padding_side = 'left'
model = LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', device_map = 'auto')

def get_data(file, noise):
    # Now do random sequences so we can get a better understanding of the data
    raw_data = np.fromfile(f'data/test/t{file}.data', dtype = np.int16)
    data_iq = raw_data[0::2] + 1j*raw_data[1::2]
    signal_real = np.real(data_iq).copy().astype(np.int16) 
    signal_real = signal_real + np.random.normal(0,noise,signal_real.shape).astype(np.int16)
    signal_real = signal_real.tobytes()
    biased = audioop.lin2lin(signal_real, 2, 1)
    data = audioop.bias(biased, 1, 2**7)
    ndata = np.frombuffer(data, dtype = np.uint8)
    data = np.right_shift(ndata, 1)
    
    return data


def select_random_token_chunks(data, cw, num_windows, num_batches):

    # Get tokenized version from string of data
    cdata_s = ''.join(chr(x) for x in data)
    tk = tokenizer(cdata_s, return_tensors='pt')
    tks = tk['input_ids']

    # Randomly sample #num_windows of size #cw+1
    symbols = list()

    raw_byte_len = 0
    batch_list = list()

    for batch_num in range(num_batches):
        sampled_windows = torch.empty((num_windows, cw))
        for i in range(num_windows):
            start_index = np.random.randint(0, tks.shape[1] - cw + 1)
            sampled_windows[i] = tks[0, start_index:start_index + cw]
            symbols.append(tks[0, start_index + cw].item())
            decoded_symbol = tokenizer.decode(tks[0, start_index + cw])
            raw_byte_len += len(decoded_symbol)

        batch = {}
        batch['input_ids'] = sampled_windows.long()
        batch['attention_mask'] = torch.ones((num_windows, cw)).long()

        batch_list.append(batch)

    return batch_list, symbols, raw_byte_len

def get_llama_predictions(batch_list):

    rp_list = list()

    for batch in batch_list:
        @torch.no_grad()
        def evaluate(batch):

            generation_config = GenerationConfig(
                num_beams=1,
                do_sample = False,
                output_logits = True,
            )

            batch['input_ids'] = batch['input_ids'].to(device)
            batch['attention_mask'] = batch['attention_mask'].to(device)
            generation_output = model.generate(
                **batch,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                output_logits = True,
                max_new_tokens=1,
            )
            return generation_output

        rp = evaluate(batch)
        rp_list.append(rp['logits'][0].cpu())
        rp = None

        torch.cuda.empty_cache()
        gc.collect()

        with torch.no_grad():
            torch.cuda.empty_cache()

    return rp_list


def compress(batch_list, symbols, total_extra_bits):
    output = list()
    encoder = arithmetic_coder.Encoder(
        base=2,
        precision=32,
        output_fn=output.append,
    )

    probabilities_scores = F.softmax(torch.vstack(batch_list), dim = -1).numpy()
    pdfs = probabilities_scores

    for symbol, pdf in zip(symbols, pdfs):
        encoder.encode(utils.normalize_pdf_for_arithmetic_coding(pdf), int(symbol))

    encoder.terminate()
    compressed_bits = ''.join(map(str, output))
    compressed_bytes, padding = utils.bits_to_bytes(compressed_bits)

    r = (len(compressed_bytes) + (total_extra_bits/ 8)) / raw_byte_len
    return r

# Params
samples_at_a_time = 16
total_rounds = 30
cw = 512
total_extra_bits = samples_at_a_time * total_rounds

df = pd.DataFrame(
    columns = [x for x in constants.SNR_SET],
    index = [x for x in range(constants.NUM_TEST_FILES)],
)

for test_file in range(constants.NUM_TEST_FILES):
    logger.info(f'Considering test file {test_file}')

    for snr_sd, snr_set in zip(constants.SNR_SD, constants.SNR_SET):
        logger.info(f'Considering SNR {snr_set}')

        data = get_data(
            file = test_file, 
            noise = snr_sd,
            )

        batch_list, symbols, raw_byte_len = select_random_token_chunks(
            data = data, 
            cw = cw, 
            num_windows = samples_at_a_time, 
            num_batches = total_rounds,
            )

        logit_stack = get_llama_predictions(batch_list)

        rate = compress(logit_stack, symbols, total_extra_bits)
        df.loc[test_file, snr_set] = rate

print(df)
ax = sns.boxplot(df, color = 'white')
plt.setp(ax.artists, edgecolor = 'k', facecolor='w')
plt.setp(ax.lines, color='k')
ax.set_xlabel('SNR [dB]')
ax.set_ylabel('Compression Rate')
ax.set_title(f'Llama Compression Rate against SNR levels')
ax.yaxis.set_major_formatter(ticker.PercentFormatter(1, 0))
plt.show()

plt.savefig(f'figs/llama/snr_results.png')
plt.close()


