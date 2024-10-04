# LLama character-based compression

import arithmetic_coder
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig
import pandas as pd
import numpy as np
import utils
import audioop
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
import gc
device = torch.device('cuda')
tokenizer = LlamaTokenizer.from_pretrained(
    'meta-llama/Llama-2-7b-hf',
    use_fast = False,
    padding=True,
)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.padding_side = 'left'
model = LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', device_map = 'auto')

# Now do random sequences so we can get a better understanding of the data
raw_data = np.fromfile(f'data/test/t0.data', dtype = np.int16)
data_iq = raw_data[0::2] + 1j*raw_data[1::2]
signal_real = np.real(data_iq).copy().astype(np.int16) 
signal_real = signal_real + np.random.normal(0,1800,signal_real.shape).astype(np.int16)
signal_real = signal_real.tobytes()
biased = audioop.lin2lin(signal_real, 2, 1)
data = audioop.bias(biased, 1, 2**7)
ndata = np.frombuffer(data, dtype = np.uint8)
data = np.right_shift(ndata, 1)

samples_at_a_time = 16
total_rounds = 4
raw_byte_len = 0

pdf_all = list()
symbol_all = list()

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

batch_list, symbols, raw_byte_len = select_random_token_chunks(data, 1024, samples_at_a_time, total_rounds)
print(symbols)

total_extra_bits = samples_at_a_time * total_rounds

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


output = list()
encoder = arithmetic_coder.Encoder(
    base=2,
    precision=32,
    output_fn=output.append,
)

#logit_stack = [x['logits'][0].cpu() for x in rp_list]
logit_stack = rp_list

probabilities_scores = F.softmax(torch.vstack(logit_stack), dim = -1).numpy()
pdfs = probabilities_scores

for symbol, pdf in zip(symbols, pdfs):
    encoder.encode(utils.normalize_pdf_for_arithmetic_coding(pdf), int(symbol))

encoder.terminate()

compressed_bits = ''.join(map(str, output))
compressed_bytes, padding = utils.bits_to_bytes(compressed_bits)

# Calculate true compression rate

print(f'{raw_byte_len=}')
print(f'{len(compressed_bytes)=}')
print(f'{total_extra_bits=}')
r = (len(compressed_bytes) + (total_extra_bits/ 8)) / raw_byte_len
print(r)