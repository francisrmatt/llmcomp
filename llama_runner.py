from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig
import pandas as pd
from torch.nn import functional as F
import numpy as np
import utils
import audioop
import matplotlib.pyplot as plt
import torch

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--size', dest = 'size')
parser.add_argument('--offset', dest = 'offset')
args = parser.parse_args()

device = torch.device('cuda')
# Model set-up
tokenizer = LlamaTokenizer.from_pretrained(
    'meta-llama/Llama-2-7b-hf',
    use_fast = False,
    padding=True,
)

tokenizer.pad_token = tokenizer.unk_token
tokenizer.padding_side = 'left'
model = LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', device_map = 'auto')

raw_data = np.fromfile(f'data/test/w_test.data', dtype = np.int16)
data_iq = raw_data[0::2] + 1j*raw_data[1::2]
signal_real = np.real(data_iq).copy().astype(np.int16) 
signal_real = signal_real.tobytes()
biased = audioop.lin2lin(signal_real, 2, 1)
data = audioop.bias(biased, 1, 2**7)
ndata = np.frombuffer(data, dtype = np.uint8)
data = np.right_shift(ndata, 1)

offset = int(args.offset)
in_size = int(args.size)

# Different paradigm
cdata = data[offset:in_size+offset]
cdata_s = ''.join(chr(x) for x in cdata)

tk = tokenizer(cdata_s)
len(tk['input_ids'])

def create_sliding_window_tensors_with_mask(input_tensor, window_size, num_windows):
    # Number of tensors we can create from the sliding window
    #num_windows = input_tensor.size(0) - window_size + 1
    sliding_windows = []
    mask_tensors = []
    symbols = []

    # Create sliding windows and corresponding mask tensors
    for i in range(num_windows):
        # Get the window
        window = input_tensor[i:i + window_size].clone()
        symbols.append(input_tensor[i+window_size].item())
        window[0] = 1
        sliding_windows.append(window)
        
        # Create the mask (all ones, with the first value as a start token)
        mask = torch.ones(window_size, dtype=torch.int)
        mask[0] = 1  # Ensure the first value in the mask is 1 (start token)
        mask_tensors.append(mask)

    return torch.stack(sliding_windows), torch.stack(mask_tensors), symbols
    
tks, msks, symbols = create_sliding_window_tensors_with_mask(torch.tensor(tk['input_ids']), 256, 32)

# Plot symbols
ascii_symbol = tokenizer.decode(symbols)
rvv = [ord(x) for x in ascii_symbol]
plt.figure()
plt.plot(rvv)
plt.savefig('figs/llama/compressor_in.png')
plt.close()

ntk = {}
ntk['input_ids'] = tks
ntk['attention_mask'] = msks


@torch.no_grad()
def evaluate(ntk):

    generation_config = GenerationConfig(
    #temperature=0.7,
    #top_p= 0.75,
    #top_k = 32000,
    num_beams=1,
    do_sample = False,
    output_logits = True,
    )


    #ntk.to(device)
    atn = ntk['attention_mask'].to(device)
    inpid = ntk['input_ids'].to(device)
    nntk = {}
    nntk['attention_mask'] = atn
    nntk['input_ids'] = inpid
    with torch.no_grad():
        generation_output = model.generate(
            **nntk,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            output_logits = True,
            max_new_tokens=1,
        )
    return generation_output

rp = evaluate(ntk)
torch_logits = rp['logits'][0].cpu()

# get probabilities using softmax from logit score and convert it to numpy array
probabilities_scores = F.softmax(torch_logits, dim = -1).numpy()
pdfs = probabilities_scores[0:-1, :]
pdfs = pdfs.tolist()

print('pickling')
import pickle
with open('in_pdf.pkl', 'wb') as f:
    pickle.dump(pdfs, f)
print('pickling done')

output = list()
import arithmetic_coder
encoder = arithmetic_coder.Encoder(
    base=2,
    precision=32,
    output_fn=output.append,
)

print(symbols)
for symbol, pdf in zip(symbols, pdfs):
    encoder.encode(utils.normalize_pdf_for_arithmetic_coding(pdf), symbol)

encoder.terminate()

compressed_bits = ''.join(map(str, output))
compressed_bytes, padding = utils.bits_to_bytes(compressed_bits)

print(f'required {padding} padding bits')

with open('compressed_llama.data','wb') as f:
    f.write(compressed_bytes)

real_size = len(tokenizer.decode(symbols))
print((len(compressed_bytes) + (real_size/8) )/ real_size)

data_iter = iter(utils.bytes_to_bits(compressed_bytes, num_padded_bits=padding))

from typing import Iterator
def _input_fn(bit_sequence: Iterator[str] = data_iter) -> int | None:
    try:
        return int(next(bit_sequence))
    except StopIteration:
        return None

print('starting fake decompression')
decoder = arithmetic_coder.Decoder(
    base=2,
    precision=32,
    input_fn=_input_fn,
)

import sys
print(compressed_bytes)
sys.exit(-1)

import gc
uncompressed_length = 31
token_list = list()
for idx in range(uncompressed_length):
    torch.cuda.empty_cache()
    gc.collect()
    token = decoder.decode(
        utils.normalize_pdf_for_arithmetic_coding(pdfs[idx])
    )
    print(token)
    token_list.append(token)
    plt.figure()
    plt.plot([ord(x) for x in tokenizer.decode(token_list)])
    plt.savefig('figs/llama/decom_output_live.png')
    plt.close()


del model
torch.cuda.empty_cache()
gc.collect()