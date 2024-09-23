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

# Need to make sure we are JUST using on GPU
if int(os.environ['CUDA_VISIBLE_DEVICES']) != 0:
    print('CAN ONLY TRAIN ON 1 GPU')
    sys.exit(-1)

model_path = "meta-llama/Llama-2-7b-hf"
#train_batch_size = 2
#eval_batch_size = 2
train_epochs = 1
eval_epochs = 1
logging_steps = 100

model_params = get_model_params(model_path)
model_class = model_params["model_class"]
hidden_size = model_params["hidden_size"]
vocab_size = model_params["vocab_size"]

head_configs = [
    HeadConfig(
        name=f"reduced_output",
        layer_hook=-1,
        in_size=hidden_size,
        output_activation="linear",
        pred_for_sequence=True,
        loss_fct="cross_entropy",
        num_outputs=128,
    )
]

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
    columns=["input_ids", "attention_mask"] + [x.name for x in head_configs],
)
for split in dd.keys():
    dd[split] = dd[split].remove_columns(["input"])

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
    base_model_class=model_class,
    model_name=model_path,
    quantization_config=quantization_config,
    head_configs=head_configs,
    device_map='auto'#{"": torch.cuda.current_device()},
)

collator = DataCollatorWithPadding(
    feature_name_to_padding_value={
        "input_ids": tokenizer.pad_token_id,
        "attention_mask": 0,
    }
)

train_batch_size = 2
eval_batch_size = 2
train_epochs = 1
eval_epochs = 1
logging_steps = 100

args = TrainingArguments(
    output_dir="linear_probe_test",
    learning_rate=0.0002,
    num_train_epochs=train_epochs,
    logging_steps=logging_steps,
    do_eval=False,
    remove_unused_columns=False,  # Important to set to False, otherwise things will fail
    auto_find_batch_size=True,
    fp16=True,
)

trainer = Trainer(
    model,
    args=args,
    train_dataset=dd["train"],
    data_collator=collator,
)
trainer.train()

print('done')