import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pytest
from jaxtyping import Array, Key

from thrml.block_management import Block
from thrml.block_sampling import SamplingSchedule, sample_states
from thrml.models.ising import (
    Edge,
    IsingEBM,
    IsingSamplingProgram,
    IsingTrainingSpec,
    estimate_kl_grad,
    hinton_init,
)
from thrml.pgm import AbstractNode, SpinNode


def generate_bimodal_gaussian(
    mu1: float,
    mu2: float,
    sigma1: float,
    sigma2: float,
    mixture_weight: float,
    n_samples: int,
    key: Key[Array, ""],
) -> Array:
    """Generate samples from a bimodal Gaussian mixture distribution.

    Args:
        mu1: Mean of first Gaussian mode
        mu2: Mean of second Gaussian mode
        sigma1: Standard deviation of first mode
        sigma2: Standard deviation of second mode
        mixture_weight: Weight for first mode (second mode gets 1 - weight)
        n_samples: Number of samples to generate
        key: JAX random key

    Returns:
        Array of shape (n_samples,) with continuous values
    """
    key1, key2, key_mix = jax.random.split(key, 3)

    # Generate samples from each mode
    samples1 = jax.random.normal(key1, (n_samples,)) * sigma1 + mu1
    samples2 = jax.random.normal(key2, (n_samples,)) * sigma2 + mu2

    # Mix the samples according to mixture_weight
    mix_mask = jax.random.bernoulli(key_mix, p=mixture_weight, shape=(n_samples,))
    samples = jnp.where(mix_mask, samples1, samples2)

    return samples


def quantize_hierarchical(values: Array, n_nodes: int, max_value: float) -> Array:
    """Quantize continuous values to binary representation using hierarchical encoding.

    Encoding:
    - Node 0: Sign bit (0 = negative, 1 = positive)
    - Nodes 1 to N-1: Binary representation of magnitude bin index (LSB first)

    Args:
        values: Continuous values array of shape (n_samples,)
        n_nodes: Total number of binary nodes
        max_value: Maximum absolute value for quantization range

    Returns:
        Binary array of shape (n_samples, n_nodes) with dtype bool
    """
    n_samples = values.shape[0]
    n_magnitude_bits = n_nodes - 1
    n_bins = 2**n_magnitude_bits
    bin_width = max_value / n_bins

    # Determine sign bits
    sign_bits = (values >= 0).astype(jnp.bool_)

    # Map absolute values to bins
    abs_values = jnp.abs(values)
    abs_values = jnp.clip(abs_values, 0, max_value)
    bin_indices = jnp.floor(abs_values / bin_width).astype(jnp.int32)
    bin_indices = jnp.clip(bin_indices, 0, n_bins - 1)

    # Convert bin indices to binary (LSB first)
    binary_samples = jnp.zeros((n_samples, n_nodes), dtype=jnp.bool_)
    binary_samples = binary_samples.at[:, 0].set(sign_bits)

    # Encode magnitude bits (LSB first: node 1 = bit 0, node 2 = bit 1, ...)
    for i in range(n_magnitude_bits):
        bit_value = (bin_indices >> i) & 1
        binary_samples = binary_samples.at[:, i + 1].set(bit_value.astype(jnp.bool_))

    return binary_samples


def dequantize_hierarchical(binary_samples: Array, n_nodes: int, max_value: float) -> Array:
    """Dequantize binary representation back to continuous values.

    Args:
        binary_samples: Binary array of shape (n_samples, n_nodes) with dtype bool
        n_nodes: Total number of binary nodes
        max_value: Maximum absolute value for quantization range

    Returns:
        Continuous values array of shape (n_samples,)
    """
    n_magnitude_bits = n_nodes - 1
    n_bins = 2**n_magnitude_bits
    bin_width = max_value / n_bins

    # Extract sign bits
    sign_bits = binary_samples[:, 0]

    # Extract bin indices from magnitude bits (LSB first)
    bin_indices = jnp.zeros(binary_samples.shape[0], dtype=jnp.int32)
    for i in range(n_magnitude_bits):
        bit_value = binary_samples[:, i + 1].astype(jnp.int32)
        bin_indices = bin_indices | (bit_value << i)

    # Map bin indices to continuous values
    magnitude = (bin_indices.astype(float) + 0.5) * bin_width
    values = jnp.where(sign_bits, magnitude, -magnitude)

    return values


def create_rbm_model(n_visible: int, n_hidden: int, beta: float, key: Key[Array, ""]) -> IsingEBM:
    """Create a two-layer RBM (Restricted Boltzmann Machine) as an IsingEBM.

    Args:
        n_visible: Number of visible nodes
        n_hidden: Number of hidden nodes
        beta: Temperature parameter
        key: JAX random key (for node creation, not weights)

    Returns:
        IsingEBM with fully connected bipartite structure
    """
    # Create nodes
    visible_nodes = [SpinNode() for _ in range(n_visible)]
    hidden_nodes = [SpinNode() for _ in range(n_hidden)]
    all_nodes = visible_nodes + hidden_nodes

    # Create fully connected bipartite edges (visible <-> hidden)
    edges = []
    for v_node in visible_nodes:
        for h_node in hidden_nodes:
            edges.append((v_node, h_node))

    # Initialize with zero biases and weights
    biases = jnp.zeros((len(all_nodes),), dtype=float)
    weights = jnp.zeros((len(edges),), dtype=float)

    return IsingEBM(all_nodes, edges, biases, weights, jnp.array(beta))


def compute_kl_divergence(samples_p: Array, samples_q: Array, n_bins: int = 50) -> float:
    """Compute KL divergence KL(P||Q) between two empirical distributions.

    Args:
        samples_p: Samples from distribution P
        samples_q: Samples from distribution Q
        n_bins: Number of bins for histogram

    Returns:
        KL divergence value
    """
    # Determine bin edges from the range of both samples
    min_val = min(jnp.min(samples_p), jnp.min(samples_q))
    max_val = max(jnp.max(samples_p), jnp.max(samples_q))
    bins = jnp.linspace(min_val, max_val, n_bins + 1)

    # Compute histograms
    hist_p, _ = jnp.histogram(samples_p, bins=bins)
    hist_q, _ = jnp.histogram(samples_q, bins=bins)

    # Normalize to probabilities
    hist_p = hist_p.astype(float) / jnp.sum(hist_p)
    hist_q = hist_q.astype(float) / jnp.sum(hist_q)

    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    hist_p = hist_p + epsilon
    hist_q = hist_q + epsilon
    hist_p = hist_p / jnp.sum(hist_p)
    hist_q = hist_q / jnp.sum(hist_q)

    # Compute KL divergence: sum(p * log(p / q))
    kl = jnp.sum(hist_p * jnp.log(hist_p / hist_q))

    return float(kl)


def plot_histograms_comparison(
    generated_samples: Array,
    test_samples: Array,
    kl_divergence: float,
    epoch: int | None,
    n_epochs: int | None,
    is_final: bool = False,
):
    """Create matplotlib histogram plot comparing generated and test samples.

    Args:
        generated_samples: Samples generated by the model
        test_samples: Test samples from true distribution
        kl_divergence: KL divergence value to display in title
        epoch: Current epoch number (None if final)
        n_epochs: Total number of epochs (None if final)
        is_final: Whether this is the final plot
    """
    plt.figure(figsize=(10, 6))

    # Create histograms with transparency
    n_bins = 50
    plt.hist(
        test_samples,
        bins=n_bins,
        alpha=0.6,
        label="Test Data (True Distribution)",
        color="orange",
        density=True,
    )
    plt.hist(
        generated_samples,
        bins=n_bins,
        alpha=0.6,
        label="Model Generated Samples",
        color="blue",
        density=True,
    )

    # Set title with KL divergence
    if is_final:
        title = f"Final Model: KL Divergence = {kl_divergence:.4f}"
    else:
        title = f"Epoch {epoch}/{n_epochs}: KL Divergence = {kl_divergence:.4f}"
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


@pytest.mark.slow
class TestTrainGaussian(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Data generation parameters (surfaced to top level)
        self.mu1 = -2.0
        self.mu2 = 2.0
        self.sigma1 = 0.5
        self.sigma2 = 0.5
        self.mixture_weight = 0.5
        self.n_train_samples = 1000
        self.n_test_samples = 500
        self.quantization_max_value = 5.0

        # Model hyperparameters (surfaced to top level)
        self.n_visible = 8
        self.n_hidden = 8
        self.beta = 1.0
        self.learning_rate = 0.01
        self.batch_size_positive = 50
        self.batch_size_negative = 25
        self.n_epochs = 10
        self.schedule_positive = SamplingSchedule(200, 20, 10)
        self.schedule_negative = SamplingSchedule(200, 40, 5)

        # Evaluation parameters
        self.eval_n_samples = 500
        self.eval_schedule = SamplingSchedule(400, 40, 10)
        self.kl_threshold = 0.5

        # Initialize optimizer
        self.optim = optax.adam(learning_rate=self.learning_rate)

    def test_train_gaussian(self):
        """Test training an RBM to learn a bimodal Gaussian distribution."""
        key = jax.random.key(42)

        # Log initialization
        print("\n" + "=" * 80)
        print("Gaussian RBM Training Test")
        print("=" * 80)
        print("\nData Generation Parameters:")
        print(f"  mu1 = {self.mu1}, mu2 = {self.mu2}")
        print(f"  sigma1 = {self.sigma1}, sigma2 = {self.sigma2}")
        print(f"  mixture_weight = {self.mixture_weight}")
        print(f"  n_train_samples = {self.n_train_samples}")
        print(f"  n_test_samples = {self.n_test_samples}")
        print(f"  quantization_max_value = {self.quantization_max_value}")
        print("\nModel Hyperparameters:")
        print(f"  n_visible = {self.n_visible}, n_hidden = {self.n_hidden}")
        print(f"  beta = {self.beta}")
        print(f"  learning_rate = {self.learning_rate}")
        print(f"  batch_size_positive = {self.batch_size_positive}")
        print(f"  batch_size_negative = {self.batch_size_negative}")
        print(f"  n_epochs = {self.n_epochs}")
        print("=" * 80 + "\n")

        # Generate training and testing samples
        key, key_train, key_test = jax.random.split(key, 3)
        train_samples_continuous = generate_bimodal_gaussian(
            self.mu1,
            self.mu2,
            self.sigma1,
            self.sigma2,
            self.mixture_weight,
            self.n_train_samples,
            key_train,
        )
        test_samples_continuous = generate_bimodal_gaussian(
            self.mu1,
            self.mu2,
            self.sigma1,
            self.sigma2,
            self.mixture_weight,
            self.n_test_samples,
            key_test,
        )

        # Quantize training samples
        train_samples_binary = quantize_hierarchical(
            train_samples_continuous, self.n_visible, self.quantization_max_value
        )

        # Create RBM model
        key, key_model = jax.random.split(key)
        model = create_rbm_model(self.n_visible, self.n_hidden, self.beta, key_model)

        # Set up training blocks
        visible_nodes = model.nodes[: self.n_visible]
        hidden_nodes = model.nodes[self.n_visible :]
        visible_block = Block(visible_nodes)
        hidden_block = Block(hidden_nodes)

        positive_sampling_blocks = [hidden_block]
        negative_sampling_blocks = [visible_block, hidden_block]
        training_data_blocks = [visible_block]

        # Initialize optimizer state
        opt_state = self.optim.init((model.weights, model.biases))

        print("> Model initialization complete")
        print("> Starting training...\n")

        # Training loop
        for epoch in range(self.n_epochs):
            print(f"> Epoch {epoch + 1}/{self.n_epochs}: Training...")

            def do_epoch(key, model, bsz_positive, bsz_negative, data_positive, opt_state):
                def batch_data(key, data, _bsz, clamped_blocks):
                    clamped_nodes = [node for block in clamped_blocks for node in block]
                    data_size = data.shape[0]
                    assert data.shape == (data_size, len(clamped_nodes))
                    key, key_shuffle = jax.random.split(key)
                    idxs = jax.random.permutation(key_shuffle, jnp.arange(data_size))
                    data = data[idxs]
                    _n_batches = data_size // _bsz
                    tot_len = _n_batches * _bsz
                    batched_data = jnp.reshape(data[:tot_len], (_n_batches, _bsz, len(clamped_nodes))).astype(
                        jnp.bool
                    )
                    return batched_data, _n_batches

                key, key_pos = jax.random.split(key, 2)
                batched_data_pos, n_batches = batch_data(key_pos, data_positive, bsz_positive, training_data_blocks)

                def body_fun(carry, key_and_data):
                    _key, _data_pos = key_and_data

                    _opt_state, _params = carry
                    _model = eqx.tree_at(lambda m: (m.weights, m.biases), model, _params)
                    key_train, key_init_pos, key_init_neg = jax.random.split(_key, 3)
                    vals_free_pos = hinton_init(
                        key_init_pos, _model, positive_sampling_blocks, (1, bsz_positive)
                    )
                    vals_free_neg = hinton_init(key_init_neg, _model, negative_sampling_blocks, (bsz_negative,))

                    ebm = IsingTrainingSpec(
                        _model,
                        training_data_blocks,
                        [],
                        positive_sampling_blocks,
                        negative_sampling_blocks,
                        self.schedule_positive,
                        self.schedule_negative,
                    )

                    grad_w, grad_b, _, _ = estimate_kl_grad(
                        key_train,
                        ebm,
                        _model.nodes,
                        model.edges,
                        [_data_pos],
                        [],
                        vals_free_pos,
                        vals_free_neg,
                    )

                    grads = (grad_w, grad_b)
                    with jax.numpy_dtype_promotion("standard"):
                        updates, _opt_state = self.optim.update(grads, _opt_state, _params)
                    _weights, _biases = _params
                    _weights += updates[0]
                    _biases += updates[1]

                    new_carry = _opt_state, (_weights, _biases)
                    return new_carry, None

                params = model.weights, model.biases
                init_carry = opt_state, params
                keys = jax.random.split(key, n_batches)
                out_carry, _ = jax.lax.scan(body_fun, init_carry, (keys, batched_data_pos))

                opt_state, params = out_carry
                new_model = eqx.tree_at(lambda m: (m.weights, m.biases), model, params)

                return new_model, opt_state

            key, key_epoch = jax.random.split(key)
            model, opt_state = do_epoch(
                key_epoch,
                model,
                self.batch_size_positive,
                self.batch_size_negative,
                train_samples_binary,
                opt_state,
            )

            # Evaluate after each epoch
            key, key_eval, key_init = jax.random.split(key, 3)

            # Generate samples from trained model
            program = IsingSamplingProgram(model, negative_sampling_blocks, [])
            init_free_states = hinton_init(key_init, model, negative_sampling_blocks, (self.eval_n_samples,))
            keys_samp = jax.random.split(key_eval, self.eval_n_samples)

            # vmap over the batch dimension
            # init_free_states is a list, vmap maps over first dimension of arrays in the list
            samples_binary_list = jax.vmap(
                lambda k, init_states: sample_states(k, program, self.eval_schedule, init_states, [], [visible_block])
            )(keys_samp, init_free_states)

            # samples_binary_list is a list with one element (visible_block)
            # Shape: (n_samples, n_steps, n_visible_nodes)
            samples_binary = samples_binary_list[0]

            # Decode binary samples to continuous values
            # We take the last step
            samples_binary_last = samples_binary[:, -1, :]
            generated_samples_continuous = dequantize_hierarchical(
                samples_binary_last, self.n_visible, self.quantization_max_value
            )

            # Compute KL divergence
            kl_div = compute_kl_divergence(generated_samples_continuous, test_samples_continuous)

            print(f"> Epoch {epoch + 1}/{self.n_epochs}: KL divergence = {kl_div:.4f}")

            # Plot histograms
            plot_histograms_comparison(
                np.array(generated_samples_continuous),
                np.array(test_samples_continuous),
                kl_div,
                epoch + 1,
                self.n_epochs,
                is_final=False,
            )

        # Final evaluation
        print("\n> Training complete. Performing final evaluation...")

        key, key_eval, key_init = jax.random.split(key, 3)
        program = IsingSamplingProgram(model, negative_sampling_blocks, [])
        init_free_states = hinton_init(key_init, model, negative_sampling_blocks, (self.eval_n_samples,))
        keys_samp = jax.random.split(key_eval, self.eval_n_samples)

        # vmap over the batch dimension
        samples_binary_list = jax.vmap(
            lambda k, init_states: sample_states(k, program, self.eval_schedule, init_states, [], [visible_block])
        )(keys_samp, init_free_states)

        samples_binary = samples_binary_list[0]
        samples_binary_last = samples_binary[:, -1, :]
        final_generated_samples = dequantize_hierarchical(
            samples_binary_last, self.n_visible, self.quantization_max_value
        )

        final_kl_div = compute_kl_divergence(final_generated_samples, test_samples_continuous)

        print(f"> Final KL divergence = {final_kl_div:.4f}")
        print("=" * 80 + "\n")

        # Final visualization
        plot_histograms_comparison(
            np.array(final_generated_samples),
            np.array(test_samples_continuous),
            final_kl_div,
            None,
            None,
            is_final=True,
        )

        # Assert KL divergence is below threshold
        self.assertLess(final_kl_div, self.kl_threshold)

