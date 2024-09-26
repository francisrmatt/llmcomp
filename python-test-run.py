from transformer_heads import create_headed_qlora, load_lora_with_heads
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    LlamaTokenizer,
    LlamaForCausalLM,
    Trainer,
    BitsAndBytesConfig,
    TrainingArguments,
    GenerationConfig,
)
from transformer_heads.util.helpers import DataCollatorWithPadding, get_model_params
from peft import LoraConfig
from transformer_heads.config import HeadConfig
from transformer_heads.util.model import print_trainable_parameters
from transformer_heads.util.evaluate import (
    evaluate_head_wise,
)
from transformer_heads import load_headed
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
import audioop
import os
import sys

if int(os.environ['CUDA_VISIBLE_DEVICES']) != 0:
    print('CAN ONLY TRAIN ON 1 GPU')
    sys.exit(-1)

model_path = "meta-llama/Llama-2-7b-hf"
model_params = get_model_params(model_path)
model_class = model_params["model_class"]
hidden_size = model_params["hidden_size"]
vocab_size = model_params["vocab_size"]

quantization_config = BitsAndBytesConfig(
    load_in_4bit=False,
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_compute_dtype=torch.float32,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)


model = load_headed(
    model_class,
    model_path,
    head_folder_path="test-multigpu/checkpoint-7813",
    device_map="cuda",
    #quantization_config=quantization_config,
    quantization_config = quantization_config,
)

# Testing

raw_data = np.fromfile(f'data/raw_data/w0.data', dtype = np.int16)
for i in range(1, 100):
        raw_data = np.append(raw_data, np.fromfile('data/raw_data/w{}.data'.format(i), np.int16))

data_iq = raw_data[0::2] + 1j*raw_data[1::2]
signal_real = np.real(data_iq).copy().astype(np.int16) 
signal_real = signal_real.tobytes()
biased = audioop.lin2lin(signal_real, 2, 1)
data = audioop.bias(biased, 1, 2**7)
ndata = np.frombuffer(data, dtype = np.uint8)
data = np.right_shift(ndata, 1)
data_split = sliding_window_view(data, 257)
d = data_split[np.random.choice(data_split.shape[0], 10000, replace=False), :]

inputs = d[:, :256] 
targets = d[:, 256:]

targets = targets.flatten()  # Now targets has shape (x,)

# One-hot encode the targets
one_hot_targets = np.array([val for val in targets])
ascii_inputs = np.array([[''.join(map(chr, row)) for row in inputs]])

# Create a Hugging Face dataset
data = {
    'input': list(ascii_inputs.flatten()),  # Convert to list of lists for compatibility
    'reduced_output': targets #one_hot_targets
}
hf_dataset = Dataset.from_dict(data)
train = hf_dataset

raw_data = np.fromfile(f'data/test/w_test.data', dtype = np.int16)

data_iq = raw_data[0::2] + 1j*raw_data[1::2]
signal_real = np.real(data_iq).copy().astype(np.int16) 
signal_real = signal_real.tobytes()
biased = audioop.lin2lin(signal_real, 2, 1)
data = audioop.bias(biased, 1, 2**7)
ndata = np.frombuffer(data, dtype = np.uint8)
data = np.right_shift(ndata, 1)
data_split = sliding_window_view(data, 257)
d = data_split[np.random.choice(data_split.shape[0], 10000, replace=False), :]

inputs = d[:, :256] 
targets = d[:, 256:]

targets = targets.flatten()  # Now targets has shape (x,)

# One-hot encode the targets
one_hot_targets = np.array(val for val in targets)
ascii_inputs = np.array([[''.join(map(chr, row)) for row in inputs]])

# Create a Hugging Face dataset
data = {
    'input': list(ascii_inputs.flatten()),  # Convert to list of lists for compatibility
    'reduced_output': targets #one_hot_targets
}
hf_dataset = Dataset.from_dict(data)
test = hf_dataset

dd = DatasetDict({
    'train': train,
    'test': test,
})

tokenizer = LlamaTokenizer.from_pretrained(model_path)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token


def processing_function(examples):
    out = tokenizer(examples['input'], padding=False, truncation=True)
    return out

for split in dd.keys():
    dd[split] = dd[split].shuffle()
    dd[split] = dd[split].map(processing_function, batched=True)

dd.set_format(
    type="torch",
    #device = 'cuda',
    columns=["input_ids", "attention_mask"] 
)
for split in dd.keys():
    dd[split] = dd[split].remove_columns(["input"])


from transformer_heads.util.evaluate import (
    evaluate_head_wise,
    get_top_n_preds,
    get_some_preds,
)
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

max_iter = 128
idx = 0
ds = dd['test'].with_format(type="torch")
loader = DataLoader(ds, batch_size=1)
preds = []
inputs = []
ground_truths = []
for i, batch in tqdm(
    enumerate(loader), total=max_iter, desc="Predicting"
):
    inputs.append(tokenizer.decode(batch["input_ids"].squeeze()))
    ground_truths.append(batch['reduced_output'][0].item())
    outputs = model(**batch)
    preds.append(outputs.preds_by_head['reduced_output'].detach().numpy())
    idx+=1
    if idx >= max_iter:
        break

import arithmetic_coder
from scipy.special import softmax
import utils

output = list()
encoder = arithmetic_coder.Encoder(
    base=2,
    precision=32,
    output_fn=output.append,
)
for i in range(max_iter):
    pdf = softmax(preds[i][0][-1])
    pdf = utils.normalize_pdf_for_arithmetic_coding(pdf)
    symbol = ground_truths[i]
    encoder.encode(pdf, symbol)

compressed_bits = ''.join(map(str, output))
compressed_bytes, num_padded_bits = utils.bits_to_bytes(compressed_bits)

print(f'COMPRESSION RATE: {(len(compressed_bytes) + (max_iter/8))/max_iter}')

plt.figure()
plt.plot(pdf)
plt.vlines(symbol, ymin=0, ymax=pdf[symbol], color = 'r')
plt.show()
plt.savefig('figs/llama/head_train.png')
plt.close()