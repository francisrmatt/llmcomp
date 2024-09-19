from functools import partial
from llama.llama import llama_completion_fn, llama_nll_fn
from llama.llama import tokenize_fn as llama_tokenize_fn



# Required: Text completion function for each model
# -----------------------------------------------
# Each model is mapped to a function that samples text completions.
# The completion function should follow this signature:
# 
# Args:
#   - input_str (str): String representation of the input time series.
#   - steps (int): Number of steps to predict.
#   - settings (SerializerSettings): Serialization settings.
#   - num_samples (int): Number of completions to sample.
#   - temp (float): Temperature parameter for model's output randomness.
# 
# Returns:
#   - list: Sampled completion strings from the model.
completion_fns = {
    'llama-7b': partial(llama_completion_fn, model='7b'),
}

# Optional: NLL/D functions for each model
# -----------------------------------------------
# Each model is mapped to a function that computes the continuous Negative Log-Likelihood 
# per Dimension (NLL/D). This is used for computing likelihoods only and not needed for sampling.
# 
# The NLL function should follow this signature:
# 
# Args:
#   - input_arr (np.ndarray): Input time series (history) after data transformation.
#   - target_arr (np.ndarray): Ground truth series (future) after data transformation.
#   - settings (SerializerSettings): Serialization settings.
#   - transform (callable): Data transformation function (e.g., scaling) for determining the Jacobian factor.
#   - count_seps (bool): If True, count time step separators in NLL computation, required if allowing variable number of digits.
#   - temp (float): Temperature parameter for sampling.
# 
# Returns:
#   - float: Computed NLL per dimension for p(target_arr | input_arr).
nll_fns = {
    'llama-7b': partial(llama_nll_fn, model='7b'),
}

# Optional: Tokenization function for each model, only needed if you want automatic input truncation.
# The tokenization function should follow this signature:
#
# Args:
#   - str (str): A string to tokenize.
# Returns:
#   - token_ids (list): A list of token ids.
tokenization_fns = {
    'llama-7b': partial(llama_tokenize_fn, model='7b'),
}

# Optional: Context lengths for each model, only needed if you want automatic input truncation.
context_lengths = {
    'llama-7b': 4096,
}