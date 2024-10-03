# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Trains a language model on the Enwik8 dataset."""

import functools
import random
from typing import Any, Callable

import logging
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
import tree

import constants
import get_data
from btransformer import transformer
import utils
import matplotlib.pyplot as plt


def _to_marginals(
    predictions: jax.Array,
    sequences: jax.Array,
) -> jax.Array:
  """Converts a conditional array to a marginals array."""
  #jax.debug.print('predictions -> {predictions}', predictions = predictions)
  #jax.debug.print('predictions shape -> {predictions}', predictions = jnp.shape(predictions))
  #jax.debug.print('sequences -> {sequences}', sequences = sequences)

  # This only takes the first of the batch
  last_row = predictions[0,-1,:]

  true_predictions = jnp.take_along_axis(
      predictions, sequences[..., None], axis=-1
  )
  true_predictions = true_predictions[..., 0]  # Shape (B, T).

  #jax.debug.print('true predictions-> {x}', x= true_predictions)
  return jnp.sum(true_predictions, axis=1), last_row  # Shape (B,).


def _make_loss_fn(model: hk.Transformed) -> Any:
  """Returns the loss function for update_parameters."""

  def loss_fn(
      params: hk.Params,
      sequences: jax.Array,
  ) -> jnp.float32:
    """Returns the loss for the model and the last state.

    Args:
      params: The parameters of the model, usually a neural network.
      sequences: The input of sequences to evaluate. See neural_predictors.py.
    """
    conditionals = model.apply(
        params=params,
        targets=sequences,
        rng=None,
    )
    marginals, final_row = _to_marginals(conditionals, sequences)
    return -jnp.mean(marginals), final_row

  return loss_fn


@functools.partial(
    jax.jit, static_argnames=('optimizer', 'grad_fn', 'normalize_gradients')
)
def _update_parameters(
    params: hk.Params,
    opt_state: optax.OptState,
    sequences: jax.Array,
    grad_fn: Any,
    optimizer: optax.GradientTransformation,
    normalize_gradients: bool = True,
) -> tuple[hk.Params, optax.OptState, dict[str, Any]]:
  """Returns updated params and extra logs (like loss, last state etc).

  Backpropagation is done on the whole sequence. The whole function is jitted.

  Args:
    params: The current parameters of the network.
    opt_state: The optimizer state.
    sequences: The input of sequences to evaluate. See base_predictor.py.
    grad_fn: A gradient function, which takes some parameters, a random seed,
      the data to compute the gradient on, and an initial state for the
      predictor. It returns the gradient of the parameters for this batch of
      data, and extra values.
    optimizer: An optax optimizer.
    normalize_gradients: Whether to divide the gradients by the length of the
      sequences, or keep them as is. Using this option guarantees to have the
      same scale across various sequence lengths, and therefore tasks.
  """
  (loss, last_row), grad = grad_fn(params, sequences)
  #jax.debug.print('last row is {x}', x = last_row)
  if normalize_gradients:
    length_sequence = float(sequences.shape[1])
    grad = tree.map_structure(lambda x: x / length_sequence, grad)
  updates, new_opt_state = optimizer.update(grad, opt_state)
  new_params = optax.apply_updates(params, updates)

  log_dict = {
      'loss': loss,
      'grad_norm_unclipped': optax.global_norm(grad),
      'latest_prob_dist' : last_row
  }

  return new_params, new_opt_state, log_dict

def _retrieve_model_params(which: str) -> hk.Params:
  """Returns the trained model parameters.

  Raises:
    FileNotFoundError if the file params.npz does not exist yet, in which case
    the user should launch a training with train.py first.
  """
  try:
    with np.load(f'params/{which}/params.npz', allow_pickle=True) as data:
      return {key: data[key].item() for key in data.files}
  except FileNotFoundError as exc:
    raise FileNotFoundError(
        'You must train a model first, the parameters file params.npz does not'
        ' exist yet.'
    ) from exc

def train_transformer_decoder(
    new_train: bool,
    which : str,
    config: transformer.TransformerConfig,
    training_steps: int,
    cw: int,
    log_every: int,
    batch_size: int = 128,
    use_tqdm: bool = True,
) -> tuple[hk.Params, float]:
  """Trains a language model on Enwik8 data.

  Sequences of 2048 characters are extracted from Enwik8, and then randomly
  sampled. We train a decoder-only transformer on batches, minimizing the
  log-loss objective. The exact architecture can be modified using the
  TransformerConfig object (defined in transformer.py)

  Args:
    training_steps: Number of batches to train on.
    log_every: How often to log the loss. If negative or 0, no log at all.
    batch_size: The number of sequences in a batch.
    sequence_length: The length of the sequences to train on, in number of ASCII
      characters.
    use_tqdm: Whether to use a progress bar or not.

  Returns:
    The final loss, and final parameters.
  """
  logging.config.dictConfig(constants.LOGGING_CONFIG)
  logger = logging.getLogger(__name__) 

  model = hk.transform(
      functools.partial(transformer.transformer_decoder, config=config)
  )

  # Temporarily replace with the pretrained

  #data_generator = get_data.fetch_preprocessed('c256')

  data_generator = get_data.fetch(
    stream_mode = False,
    amt = 0, # Doesn't matter
    context_window = cw,
    filename = -1, # all
    scale = 1,
    offset = 0,
  )

  dataset = list(data_generator)

  def fetch_random_batch() -> np.ndarray:
    batch_list = random.choices(dataset, k=batch_size)
    batch_list = [np.frombuffer(seq, dtype=np.uint8) for seq in batch_list]
    if config.vocab_size == 128:
      batch_list = np.right_shift(batch_list, 1)
    # Add noise
    #batch_list = batch_list + np.random.normal(0,2,batch_list.shape).astype(np.uint8)
    return np.array(batch_list, dtype=np.uint8)

  if new_train:
    logger.info('Fetching random batch for fresh run')
    # Initialize parameters.
    dummy_batch = fetch_random_batch()
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, dummy_batch)
  else:
    logger.info('Fetching old parameters')
    params = _retrieve_model_params(which)


  # Make gradient function.
  loss_fn = _make_loss_fn(model)
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True) # Changed to true so we can return last row

  # Make optimizer, to apply the gradients.
  optimizer = optax.adam(learning_rate=1e-5)
  opt_state = optimizer.init(params)

  logger.info('Initialization done, starting training...')
  last_loss = 0.0

  # Create batch histogram, assumes 128 alphabet size
  data_histogram = np.zeros(128, dtype = np.float32)

  # KLD store
  kld_av_store = list()
  kld_sum = 0
  kld_count = 0
  alpha = 0.9 # smoothing factor

  for step in tqdm.trange(training_steps, disable=not use_tqdm):
    batch = fetch_random_batch()
    #logger.info('Batch fetched.')

    # Update data histogram
    unique, counts = np.unique(batch, return_counts=True)
    data_histogram[unique] += counts

    params, opt_state, logs = _update_parameters(
        params=params,
        opt_state=opt_state,
        sequences=batch,
        grad_fn=grad_fn,
        optimizer=optimizer,
    )
    if log_every > 0 and step % log_every == 0:
      # Graph KL divergence
      q = logs['latest_prob_dist']
      q = jnp.exp(q)
      q = utils.normalize_pdf_for_arithmetic_coding(q)
      p = utils.normalize_pdf_for_arithmetic_coding(data_histogram)
      epsilon = 1e-10
      p = p + epsilon
      q = q + epsilon

      #kld_count += 1
      kld = jnp.sum(p * np.log(p / q))
      if len(kld_av_store) == 0:
        kld_av_store.append(kld)
      else:
        kld_av_store.append(alpha * kld_av_store[-1] + (1-alpha) * kld)


      # Plot to watch
      if len(kld_av_store) > 101:
        plt.figure()
        plt.plot(kld_av_store[-100:])
        plt.savefig('figs/tmp/kld_store.png')
        plt.close()

      plt.figure()
      plt.plot(p)
      plt.savefig('figs/tmp/p_store.png')
      plt.close()

      plt.figure()
      plt.plot(q)
      plt.savefig('figs/tmp/q_store.png')
      plt.close()

      logger.info(
          'Step %f, Loss %f, Grad norm %f KLD %f',
          step,
          logs['loss'],
          logs['grad_norm_unclipped'],
          kld,
      )

    last_loss = logs['loss']




  return params, last_loss

