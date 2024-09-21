"""Transformer model."""
import matplotlib.pyplot as plt
import seaborn as sns
import sys

import dataclasses

import haiku as hk
import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np


def graph3d(embeddings):
  from mpl_toolkits.mplot3d import Axes3D
  from scipy.spatial import cKDTree
  from sklearn.decomposition import PCA

  # Step 1: Generate the sphere
  def generate_sphere(n_points=1000):
      u = np.linspace(0, 2 * np.pi, n_points)
      v = np.linspace(0, np.pi, n_points)
      u, v = np.meshgrid(u, v)
      x = np.sin(v) * np.cos(u)
      y = np.sin(v) * np.sin(u)
      z = np.cos(v)
      return x, y, z

  # Step 2: Compute distances
  def compute_closest_points(sphere_points, data_points):
      # Create a KDTree for fast nearest neighbor search
      tree = cKDTree(data_points)
      distances, indices = tree.query(sphere_points)
      return indices

  data_points = PCA(n_components=3).fit_transform(embeddings)

  # Generate sphere points
  n_points = 50
  x, y, z = generate_sphere(n_points)

  # Reshape sphere points into a list of points
  sphere_points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1)

  # Compute closest points on the sphere
  closest_indices = compute_closest_points(sphere_points, data_points)

  # Reshape closest_indices to match the shape of x, y, z
  closest_indices = closest_indices.reshape(x.shape)

  # Normalize the color index to be in the range [0, 1] for colormap
  norm = plt.Normalize(0, len(data_points) - 1)
  cmap = plt.get_cmap('rocket')

  # Create a color array for the surface plot
  colors = cmap(norm(closest_indices.ravel())).reshape(x.shape + (4,))  # RGBA
  def update_view(azim):
    ax.cla()  # Clear the previous plot
    ax.plot_surface(x, y, z, facecolors=colors, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
    ax.set_axis_off()
    
    # Set equal scaling
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()
    mid_x = (x.max() + x.min()) * 0.5
    mid_y = (y.max() + y.min()) * 0.5
    mid_z = (z.max() + z.min()) * 0.5
    ax.set_xlim([mid_x - max_range, mid_x + max_range])
    ax.set_ylim([mid_y - max_range, mid_y + max_range])
    ax.set_zlim([mid_z - max_range, mid_z + max_range])
    
    # Set aspect ratio
    ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1
    ax.view_init(elev=30, azim=azim)

  # Plot the sphere
  fig = plt.figure(figsize=(10, 8))  # Increase figure size (width, height)
  ax = fig.add_subplot(111, projection='3d')
  # Create animation
  import matplotlib.animation as animation
  from matplotlib.animation import PillowWriter
  frames = 360
  interval = 50  # milliseconds between frames
  anim = animation.FuncAnimation(fig, update_view, frames=np.linspace(0, 360, frames), interval=interval, repeat=True)

  # Save the animation as a GIF
  anim.save('figs/tmp/rotating_sphere.gif', writer=PillowWriter(fps=24), dpi=150)

  # Plot the sphere surface with colors based on closest point index
  #surface = ax.plot_surface(x, y, z, facecolors=colors, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)

  ## Remove axes
  #ax.set_axis_off()

  ## Set equal scaling
  #max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()
  #mid_x = (x.max() + x.min()) * 0.5
  #mid_y = (y.max() + y.min()) * 0.5
  #mid_z = (z.max() + z.min()) * 0.5
  #ax.set_xlim([mid_x - max_range, mid_x + max_range])
  #ax.set_ylim([mid_y - max_range, mid_y + max_range])
  #ax.set_zlim([mid_z - max_range, mid_z + max_range])

  ## Set aspect ratio
  #ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1

  ## Change angle
  #ax.view_init(elev=30, azim=50)

  ## Save the figure with higher DPI
  #plt.savefig('figs/tmp/sphere_plot.png', dpi=300, bbox_inches='tight')


@dataclasses.dataclass(kw_only=True)
class TransformerConfig:
  """Hyperparameters used in the Transformer architectures."""

  # Vocabulary size.
  vocab_size: int
  # The dimension of the first embedding.
  embedding_dim: int = 256
  # The number of multi-head attention layers.
  num_layers: int = 16
  # The number of heads per layer.
  num_heads: int = 32
  # The parameter initialization scale for the embeddings.
  emb_init_scale: float = 0.02
  # How much larger the hidden layer of the feedforward network should be
  # compared to the `embedding_dim`.
  widening_factor: int = 4


class MultiHeadDotProductAttention(hk.Module):
  """Multi-head dot-product attention (Vaswani et al., 2017)."""

  def __init__(
      self,
      num_heads: int,
      num_hiddens_per_head: int,
      name: str | None = None,
  ) -> None:
    """Initializes the attention module.

    Args:
      num_heads: Number of heads to use.
      num_hiddens_per_head: Number of hidden neurons per head.
      name: Name of the module.
    """
    super().__init__(name=name)
    self._num_heads = num_heads
    self._num_hiddens_per_head = num_hiddens_per_head

  def __call__(
      self,
      inputs_q: jax.Array,
      inputs_kv: jax.Array,
      mask: jax.Array | None = None,
  ) -> jax.Array:
    """Returns the output of the multi-head attention."""
    batch_size, sequence_length, embedding_size = inputs_q.shape

    #plt.figure()
    #sns.heatmap(inputs_q[1,:,:].T)
    #plt.xlim(0, inputs_q.shape[1])
    #plt.ylim(0, inputs_q.shape[2])
    #plt.savefig('figs/tmp/attention_head_input.png')
    #plt.close()


    num_hiddens = self._num_hiddens_per_head * self._num_heads
    q = hk.Linear(num_hiddens, with_bias=False)(inputs_q)
    k = hk.Linear(num_hiddens, with_bias=False)(inputs_kv)
    v = hk.Linear(num_hiddens, with_bias=False)(inputs_kv)
    # The second (sequence) dimension is undefined since it can differ between
    # queries and keys/values when decoding. Also checking that the inputs have
    # the same batch size as the reshape below does not guarantee a failure if
    # they are different.
    new_shape = (batch_size, -1, self._num_heads, self._num_hiddens_per_head)
    q = jnp.reshape(q, new_shape)
    k = jnp.reshape(k, new_shape)
    v = jnp.reshape(v, new_shape)

    # Let b=batch_size, t=seq_len, h=num_heads, and d=num_hiddens_per_head.
    attention = jnp.einsum('bthd,bThd->bhtT', q, k)
    attention *= 1.0 / jnp.sqrt(self._num_hiddens_per_head)

    #plt.subplot(1,2,1)
    #sns.heatmap(attention[1,1,:,:])
    #plt.subplot(1,2,2)
    #sns.heatmap(attention[1,2,:,:])
    #plt.savefig('figs/tmp/attention_matrix.png')
    #plt.close()

    if mask is not None:
      attention = jnp.where(mask, attention, jnp.finfo(jnp.float32).min)

    normalized_attention = jnn.softmax(attention)

    output = jnp.einsum('bhtT,bThd->bthd', normalized_attention, v)
    output = jnp.reshape(output, (batch_size, sequence_length, num_hiddens))
    return hk.Linear(embedding_size, with_bias=False)(output)


def sinusoid_position_encoding(
    sequence_length: int,
    hidden_size: int,
    max_timescale: float = 1e4,
) -> np.ndarray:
  """Creates sinusoidal encodings from the original transformer paper.

  The returned values are, for all i < D/2:
    array[pos, i] = sin(pos / (max_timescale^(2*i / D)))
    array[pos, D/2 + i] = cos(pos / (max_timescale^(2*i / D)))

  Args:
    sequence_length: Sequence length.
    hidden_size: Dimension of the positional encoding vectors, D. Should be
      even.
    max_timescale: Maximum timescale for the frequency.

  Returns:
    An array of shape [L, D] if `add_negative` or `keep_positive_side` is
    `False`, else [2 * L, D].
  """
  freqs = np.arange(0, hidden_size + 1, 2)
  inv_freq = max_timescale ** (-freqs / hidden_size)

  pos_seq = np.arange(start=0, stop=sequence_length)

  sinusoid_inp = np.einsum('i,j->ij', pos_seq, inv_freq)
  embeddings = np.concatenate(
      [np.sin(sinusoid_inp), np.cos(sinusoid_inp)], axis=-1
  )
  return embeddings[:, :hidden_size]


def embed_sequences(
    sequences: jax.Array,
    config: TransformerConfig,
) -> jax.Array:
  """Returns embeddings for sequences of tokens."""
  embs_init = hk.initializers.TruncatedNormal(stddev=config.emb_init_scale)
  embeddings_layer = hk.Embed(
      vocab_size=config.vocab_size,
      embed_dim=config.embedding_dim,
      lookup_style=hk.EmbedLookupStyle.ARRAY_INDEX,
      w_init=embs_init,
  )
  #sequences = np.arange(128)
  embeddings = embeddings_layer(sequences)

  #graph3d(embeddings)
  ## We want to graph the sequence on top of the embeddings

   #Only consider 16
  #sns.color_palette('mako')
  #plt.figure()
  #sns.heatmap(embeddings.T)
  ##plt.gca().set_aspect(0.25)
  ##plt.xlim(0, embeddings.shape[1])
  #plt.ylim(0, embeddings.shape[1])
  ##plt.plot(sequences)
  #plt.title('Actual values on top of embedding vector')
  #plt.xlabel('Sequence Number')
  #plt.ylabel('Embedding Vector')
  #plt.savefig('figs/tmp/comp_emb_to_actual.png')
  #plt.close()

  ## Try a weird approach
  #import pandas as pd
  #from sklearn.decomposition import PCA
  #pca = PCA(n_components=2)
  #dd = pca.fit_transform(embeddings)
  #dd /= np.linalg.norm(dd, axis = 1)[:,None]
  #plt.figure(figsize = (8,6))
  #sns.scatterplot(x = dd[:, 0], y = dd[:, 1], hue=np.arange(128), palette='rocket', legend = False, s =100)
  #plt.axis('off')
  #plt.savefig('figs/emb_expr/embedding_scatterplot_d128.png')
  #plt.savefig('figs/emb_expr/embedding_scatterplot_d128.eps', dpi = 1200)
  #plt.close()
  #sys.exit(-1)

  #import mpl_toolkits
  #from mpl_toolkits.mplot3d import Axes3D
  ## Try 3d figure
  #ddd = PCA(n_components=3).fit_transform(embeddings)
  ## Try normalise
  #ddd = ddd/np.linalg.norm(ddd, axis = 1)[:,None]
  #print(np.linalg.norm(ddd, axis = 1).shape)
  #fig = plt.figure(figsize=(6,6))
  #ax = Axes3D(fig, auto_add_to_figure=False)
  #fig.add_axes(ax)

  ## plot
  #from matplotlib.colors import ListedColormap
  #cmap = ListedColormap(sns.color_palette("rocket", 128).as_hex())

  #sc = ax.scatter(xs = ddd[:,0], ys=ddd[:,1], zs=ddd[:,2], c=np.arange(128), cmap=cmap)
  #ax.set_axis_off()


  ## save
  #plt.savefig("figs/tmp/threedeescatter.png", bbox_inches='tight')

  #sys.exit(-1)

  embeddings *= jnp.sqrt(config.embedding_dim)

  _, sequence_length, embedding_size = embeddings.shape
  pos_encodings = sinusoid_position_encoding(
      sequence_length=sequence_length,
      hidden_size=embedding_size,
  )
  return embeddings + pos_encodings


def layer_norm(x: jax.Array) -> jax.Array:
  """Helper function for layer norm."""
  return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)


def shift_right(sequences: jax.Array) -> jax.Array:
  """Right-shift the one-hot encoded input by padding on the temporal axis."""
  bos_array = jnp.zeros((sequences.shape[0], 1), dtype=jnp.uint8)
  padded_sequences = jnp.concatenate([bos_array, sequences], axis=1)
  return padded_sequences[:, :-1]


def transformer_decoder(
    targets: jax.Array,
    config: TransformerConfig,
) -> jax.Array:
  """Returns the transformer decoder output, shape [B, T, V].

  Args:
    targets: The integer target values, shape [B, T].
    config: The config to use for the transformer.
  """
  # Right shift the targets to get the inputs (the first token is now a 0).
  inputs = shift_right(targets)

  # Embeds the inputs and adds positional encodings.
  embeddings = embed_sequences(inputs, config)
  batch_size, sequence_length = embeddings.shape[:2]

  # The causal mask is shared across heads.
  causal_mask = np.tril(
      np.ones((batch_size, 1, sequence_length, sequence_length))
  )

  h = embeddings
  for _ in range(config.num_layers):
    self_attention = MultiHeadDotProductAttention(
        num_heads=config.num_heads,
        num_hiddens_per_head=config.embedding_dim // config.num_heads,
    )(inputs_q=h, inputs_kv=h, mask=causal_mask)
    attention = layer_norm(h + self_attention)

    # Position-wise feedforward network.
    h = hk.Linear(config.embedding_dim * config.widening_factor)(attention)
    h = jnn.gelu(h)
    h = hk.Linear(config.embedding_dim)(h)
    h = layer_norm(h + attention)

  logits = hk.Linear(config.vocab_size)(h)
  x = jnn.log_softmax(logits, axis=-1)
  return x
