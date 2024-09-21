from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig
import pandas as pd
from torch.nn import functional as F
import numpy as np
import utils
import audioop
import matplotlib.pyplot as plt
import torch
import arithmetic_coder

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

# Load in data
with open('compressed_llama.data', 'rb') as f:
    d = f.read()

import sys
data_iter = iter(utils.bytes_to_bits(d, num_padded_bits=1))

from typing import Iterator
def _input_fn(bit_sequence: Iterator[str] = data_iter) -> int | None:
    try:
        return int(next(bit_sequence))
    except StopIteration:
        return None

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

fi = torch.tensor(tk['input_ids'][0:256]).view(1, -1).to(device)

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

    mask = torch.ones(256, dtype = torch.int).view(1,-1).to(device)
    nntk = {}
    nntk['input_ids']  = ntk
    nntk['attention_mask'] = mask

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


import sys
rp = evaluate(fi)
probs = rp['logits'][0].cpu()
probabilities_scores = F.softmax(probs, dim = -1).numpy()
pdfs = probabilities_scores

print(pdfs.shape)

decoder = arithmetic_coder.Decoder(
    base=2,
    precision=32,
    input_fn=_input_fn,
)

import gc
import pickle
uncompressed_length = 32
pdf_save = list()
pdf_save.append(pdfs[0])
token_list = list()
for idx in range(uncompressed_length):
    #try:
    token = decoder.decode(
        utils.normalize_pdf_for_arithmetic_coding(pdfs[0])
    )
    #except:
        #with open('out_pdf.pkl', 'wb') as f:
            #pickle.dump(pdf_save, f)
        #sys.exit(-1)
    print(token)
    token_list.append(token)
    fi = fi.cpu()
    fi = np.append(fi,token)[1:]
    fi[0] = 1
    fi = torch.tensor(fi).view(1,-1).to(device)
    plt.figure()
    plt.plot([ord(x) for x in tokenizer.decode(token_list)])
    plt.savefig('figs/llama/decom_output_live.png')
    plt.close()

    rp = evaluate(fi)
    probs = rp['logits'][0].cpu()
    probabilities_scores = F.softmax(probs, dim = -1).numpy()
    pdfs = probabilities_scores
    pdf_save.append(pdfs[0])

with open('out_pdf.pkl', 'wb') as f:
    pickle.dump(pdf_save, f)

torch.cuda.empty_cache()
gc.collect()
