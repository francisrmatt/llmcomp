# Thesis 

# Introduction

# Background

## Notation
## Compression and basic information theory
## Arithmetic Coding
## Large language models
## Large language models and time series
## Deepmind's paper

# Transformer Architecture for RF Signals (to do)

## Data preparation - should we use a masking function, do we need to scale different data sets etc. (?)
- Scaling is a no go
- Removes the idea of lossless copmression
- Discuss the issues of scaling re above
- Exponentially weighted moving average to make maximum 255
## Results with peeking
## Hyperparameter search - finding best context windows, learning rates, etc. Make sure to use validation set
## Fixing scaling issue (normalising batches?)
- talked about this see above
## Issues surrounding decompression with the above

# Llama 2

## Data preparation - How do we represent time series in llama 2 when the tokenizer is going to behave in a certain way
## Results from letting the numerical prediciton (zero-shot) llama do stuff based on "LLMs are zero shot predictors"
## Results from [0,127] ascii encoding (HOW DO WE GET THE VALUES BACK DETOKENIZED?? it does say in the paper)
####### Issue -> How do we get the self-attention of each token in the sequence rather than having the model generate the NEXT token
https://huggingface.co/docs/transformers/en/internal/generation_utils
https://stackoverflow.com/questions/71376760/how-to-output-the-list-of-probabilities-on-each-token-via-model-generate
## After figuring out the above, fine tuning using the below and comparing results
### LORA
### RAG

# If I have time looking at prediction with the models?

# Results


# Conclusion



