"""Trains a base transformer on the Enwik8 dataset."""

import functools
import random
from typing import Any, Generator

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
import tree
import logging.config

import constants
import get_data
from btransformer import transformer


def _to_marginals(
    predictions: jax.Array,
    sequences: jax.Array,
) -> jax.Array:
  """Converts a conditional array to a marginals array."""
  true_predictions = jnp.take_along_axis(
      predictions, sequences[..., None], axis=-1
  )
  true_predictions = true_predictions[..., 0]  # Shape (B, T).
  return jnp.sum(true_predictions, axis=1)  # Shape (B,).


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
    marginals = _to_marginals(conditionals, sequences)
    return -jnp.mean(marginals)

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
  loss, grad = grad_fn(params, sequences)
  if normalize_gradients:
    length_sequence = float(sequences.shape[1])
    grad = tree.map_structure(lambda x: x / length_sequence, grad)
  updates, new_opt_state = optimizer.update(grad, opt_state)
  new_params = optax.apply_updates(params, updates)

  log_dict = {
      'loss': loss,
      'grad_norm_unclipped': optax.global_norm(grad),
  }

  return new_params, new_opt_state, log_dict


def _retrieve_model_params() -> hk.Params:
  """Returns the trained model parameters.

  Raises:
    FileNotFoundError if the file params.npz does not exist yet, in which case
    the user should launch a training with train.py first.
  """
  try:
    with np.load('params.npz', allow_pickle=True) as data:
      return {key: data[key].item() for key in data.files}
  except FileNotFoundError as exc:
    raise FileNotFoundError(
        'You must train a model first, the parameters file params.npz does not'
        ' exist yet.'
    ) from exc

def train_transformer_decoder(
    data: Generator,
    tconfig: transformer.TransformerConfig,
    training_steps: int,
    log_every: int,
    learning_rate: float,
    batch_size: int = 1,
    use_tqdm: bool = True,
) -> tuple[hk.Params, float]:
  """Trains a language model on data.

  Args:
    data: Generator of data to train on.
    tconfig: TransformerConfig object which determines the hyperparameters of the base transformer.
    training_steps: How many steps to train.
    log_every: How often to log the loss. If negative or 0, no log at all.
    learning_rate: Learning rate.
    batch_size: The number of sequences in a batch.
    use_tqdm: Whether to use a progress bar or not.

  Returns:
    The final loss, and final parameters.
  """
   # Logging config
  logging.config.dictConfig(constants.LOGGING_CONFIG)
  logger = logging.getLogger(__name__) 

  model = hk.transform(
      functools.partial(transformer.transformer_decoder, config=tconfig)
  )

  # TODO 
  logger.info(f'Training')

  dataset = list(data)
  # Probably not needed
  num_chunks = len(dataset)
  context_window = len(dataset[0])
  
  def fetch_random_batch() -> np.ndarray:
    batch_list = random.choices(dataset, k=batch_size)
    batch_list = [np.frombuffer(seq, dtype=np.uint8) for seq in batch_list]
    return np.array(batch_list, dtype=np.uint8)

  # Initialize parameters.
  dummy_batch = fetch_random_batch()
  rng = jax.random.PRNGKey(0)
  params = model.init(rng, dummy_batch)

  # Make gradient function.
  loss_fn = _make_loss_fn(model)
  grad_fn = jax.value_and_grad(loss_fn, has_aux=False)

  # Make optimizer, to apply the gradients.
  optimizer = optax.adam(learning_rate=learning_rate)
  opt_state = optimizer.init(params)

  logger.info('Initialization done, starting training...')
  last_loss = 0.0
  for step in tqdm.trange(training_steps, disable=not use_tqdm):
    batch = fetch_random_batch()
    logger.info('Batch fetched.')

    params, opt_state, logs = _update_parameters(
        params=params,
        opt_state=opt_state,
        sequences=batch,
        grad_fn=grad_fn,
        optimizer=optimizer,
    )
    if log_every > 0 and step % log_every == 0:
      logging.info(
          'Step %f, Loss %f, Grad norm %f',
          step,
          logs['loss'],
          logs['grad_norm_unclipped'],
      )
      #logging.info(batch)
    last_loss = logs['loss']

  return params, last_loss


def main(_) -> None:
  """Trains a language model and saves the parameters to a JSON file."""

  # Logging config
  logging.config.dictConfig(constants.LOGGING_CONFIG)
  logger = logging.getLogger(__name__) 

  params, loss = train_transformer_decoder(
      training_steps=5000,
      log_every=100,
      sequence_length=constants.CHUNK_SIZE_BYTES,
  )
  logger.info('Final loss: %f', loss)

  np.savez('params.npz', **params)
  logging.info('Parameters saved in file params.npz')


if __name__ == '__main__':
  main()
