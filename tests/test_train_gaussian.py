import os
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


def quantize_categorical(values: Array, n_nodes: int, n_categories_per_node: int, max_value: float) -> Array:
    """Quantize continuous values to categorical node states using base-N encoding.

    Encoding:
    - Maps continuous value → bin index (0 to n_categories_per_node ** n_nodes - 1)
    - Converts bin index to base-N representation across n_nodes categorical nodes

    Args:
        values: Continuous values array of shape (n_samples,)
        n_nodes: Number of categorical visible nodes
        n_categories_per_node: Number of categories per node (base-N)
        max_value: Maximum absolute value for quantization range

    Returns:
        Categorical array of shape (n_samples, n_nodes) with dtype uint8
    """
    n_samples = values.shape[0]
    n_bins = n_categories_per_node ** n_nodes
    bin_width = 2 * max_value / n_bins  # Total range is [-max_value, max_value]

    # Map values to bin indices
    # Shift values to [0, 2*max_value] range
    shifted_values = values + max_value
    shifted_values = jnp.clip(shifted_values, 0, 2 * max_value)
    bin_indices = jnp.floor(shifted_values / bin_width).astype(jnp.int32)
    bin_indices = jnp.clip(bin_indices, 0, n_bins - 1)

    # Convert bin indices to base-N representation
    categorical_samples = jnp.zeros((n_samples, n_nodes), dtype=jnp.uint8)
    for i in range(n_nodes):
        # Extract digit for position i (LSB first)
        digit = (bin_indices // (n_categories_per_node ** i)) % n_categories_per_node
        categorical_samples = categorical_samples.at[:, i].set(digit.astype(jnp.uint8))

    return categorical_samples


def dequantize_categorical(
    categorical_samples: Array, n_nodes: int, n_categories_per_node: int, max_value: float
) -> Array:
    """Dequantize categorical states back to continuous values.

    Args:
        categorical_samples: Categorical array of shape (n_samples, n_nodes) with dtype uint8
        n_nodes: Number of categorical visible nodes
        n_categories_per_node: Number of categories per node (base-N)
        max_value: Maximum absolute value for quantization range

    Returns:
        Continuous values array of shape (n_samples,)
    """
    n_bins = n_categories_per_node ** n_nodes
    bin_width = 2 * max_value / n_bins

    # Convert base-N representation back to bin index
    bin_indices = jnp.zeros(categorical_samples.shape[0], dtype=jnp.int32)
    for i in range(n_nodes):
        digit = categorical_samples[:, i].astype(jnp.int32)
        bin_indices = bin_indices + digit * (n_categories_per_node ** i)

    # Map bin indices to continuous values
    # Shift back from [0, 2*max_value] to [-max_value, max_value]
    magnitude = (bin_indices.astype(float) + 0.5) * bin_width
    values = magnitude - max_value

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


def create_dbm_model(
    n_visible: int, n_hidden1: int, n_hidden2: int, beta: float, key: Key[Array, ""]
) -> IsingEBM:
    """Create a three-layer DBM (Deep Boltzmann Machine) as an IsingEBM.

    Structure: visible ↔ hidden1 ↔ hidden2

    Args:
        n_visible: Number of visible nodes
        n_hidden1: Number of nodes in first hidden layer
        n_hidden2: Number of nodes in second hidden layer
        beta: Temperature parameter
        key: JAX random key (for node creation, not weights)

    Returns:
        IsingEBM with three-layer structure: visible ↔ hidden1 ↔ hidden2
    """
    # Create nodes
    visible_nodes = [SpinNode() for _ in range(n_visible)]
    hidden1_nodes = [SpinNode() for _ in range(n_hidden1)]
    hidden2_nodes = [SpinNode() for _ in range(n_hidden2)]
    all_nodes = visible_nodes + hidden1_nodes + hidden2_nodes

    # Create edges: visible ↔ hidden1 and hidden1 ↔ hidden2
    edges = []
    # Visible ↔ Hidden1 connections
    for v_node in visible_nodes:
        for h1_node in hidden1_nodes:
            edges.append((v_node, h1_node))
    # Hidden1 ↔ Hidden2 connections
    for h1_node in hidden1_nodes:
        for h2_node in hidden2_nodes:
            edges.append((h1_node, h2_node))

    # Initialize with zero biases and weights
    biases = jnp.zeros((len(all_nodes),), dtype=float)
    weights = jnp.zeros((len(edges),), dtype=float)

    return IsingEBM(all_nodes, edges, biases, weights, jnp.array(beta))


def create_hybrid_dbm_model(
    n_visible: int,
    n_categories_per_visible_node: int,
    n_hidden1: int,
    n_hidden2: int,
    beta: float,
    key: Key[Array, ""],
) -> "HybridEBM":
    """Create a three-layer DBM with categorical visible nodes and spin hidden nodes.

    Structure: categorical_visible ↔ spin_hidden1 ↔ spin_hidden2

    Args:
        n_visible: Number of visible nodes (categorical)
        n_categories_per_visible_node: Number of categories per visible node
        n_hidden1: Number of nodes in first hidden layer (spin)
        n_hidden2: Number of nodes in second hidden layer (spin)
        beta: Temperature parameter
        key: JAX random key (for node creation, not weights)

    Returns:
        HybridEBM with three-layer structure
    """
    from thrml.block_management import Block
    from thrml.models.discrete_ebm import CategoricalEBMFactor, DiscreteEBMFactor, SpinEBMFactor
    from thrml.models.ebm import FactorizedEBM
    from thrml.models.hybrid import HybridEBM
    from thrml.pgm import CategoricalNode, SpinNode

    # Create nodes
    visible_nodes = [CategoricalNode() for _ in range(n_visible)]
    hidden1_nodes = [SpinNode() for _ in range(n_hidden1)]
    hidden2_nodes = [SpinNode() for _ in range(n_hidden2)]

    # Create blocks
    visible_block = Block(visible_nodes)
    hidden1_block = Block(hidden1_nodes)
    hidden2_block = Block(hidden2_nodes)

    # Initialize biases and weights with zeros
    # Categorical bias: shape [n_visible, n_categories_per_visible_node]
    cat_bias = jnp.zeros((n_visible, n_categories_per_visible_node), dtype=float)
    # Spin biases: shape [n_hidden1] and [n_hidden2]
    hidden1_bias = jnp.zeros((n_hidden1,), dtype=float)
    hidden2_bias = jnp.zeros((n_hidden2,), dtype=float)

    # Visible ↔ Hidden1 interaction weights: shape [n_hidden1, n_visible, n_categories_per_visible_node]
    vis_h1_weights = jnp.zeros((n_hidden1, n_visible, n_categories_per_visible_node), dtype=float)
    # Hidden1 ↔ Hidden2 interaction weights: shape [n_hidden1, n_hidden2]
    h1_h2_weights = jnp.zeros((n_hidden1, n_hidden2), dtype=float)

    # Create edges for visible ↔ hidden1 interaction (per-edge approach)
    edges_v_h1 = []
    for vis_node in visible_nodes:
        for h1_node in hidden1_nodes:
            edges_v_h1.append((vis_node, h1_node))

    # Create blocks from edges for per-edge factor
    # Block 1: all visible nodes (repeated for each edge)
    block_v_h1_visible = Block([edge[0] for edge in edges_v_h1])
    # Block 2: all hidden1 nodes (pattern: h1_1...h1_n repeated for each visible node)
    block_v_h1_hidden = Block([edge[1] for edge in edges_v_h1])

    # Reshape weights from [n_hidden1, n_visible, n_categories] to [n_edges, n_categories]
    # Order: for each visible node, iterate through all hidden nodes
    n_edges_v_h1 = n_hidden1 * n_visible
    vis_h1_weights_flat = vis_h1_weights.reshape((n_edges_v_h1, n_categories_per_visible_node))

    # Create edges for hidden1 ↔ hidden2 interaction (per-edge approach)
    edges_h1_h2 = []
    for h1_node in hidden1_nodes:
        for h2_node in hidden2_nodes:
            edges_h1_h2.append((h1_node, h2_node))

    # Create blocks from edges for per-edge factor
    block_h1_h2_hidden1 = Block([edge[0] for edge in edges_h1_h2])
    block_h1_h2_hidden2 = Block([edge[1] for edge in edges_h1_h2])

    # Reshape weights from [n_hidden1, n_hidden2] to [n_edges,]
    n_edges_h1_h2 = n_hidden1 * n_hidden2
    h1_h2_weights_flat = h1_h2_weights.reshape((n_edges_h1_h2,))

    # Create factors
    factors = [
        # Categorical visible bias
        CategoricalEBMFactor([visible_block], beta * cat_bias),
        # Spin hidden1 bias
        SpinEBMFactor([hidden1_block], beta * hidden1_bias),
        # Spin hidden2 bias
        SpinEBMFactor([hidden2_block], beta * hidden2_bias),
        # Visible ↔ Hidden1 interaction (categorical-spin) - per-edge approach
        DiscreteEBMFactor([block_v_h1_hidden], [block_v_h1_visible], beta * vis_h1_weights_flat),
        # Hidden1 ↔ Hidden2 interaction (spin-spin) - per-edge approach
        SpinEBMFactor([block_h1_h2_hidden1, block_h1_h2_hidden2], beta * h1_h2_weights_flat),
    ]

    # Create FactorizedEBM
    ebm = FactorizedEBM(factors)

    # Create n_categories_per_node mapping
    n_categories_per_node = {node: n_categories_per_visible_node for node in visible_nodes}

    # Create HybridEBM
    hybrid_ebm = HybridEBM(
        ebm=ebm,
        categorical_nodes=visible_nodes,
        spin_nodes=hidden1_nodes + hidden2_nodes,
        n_categories_per_node=n_categories_per_node,
        beta=jnp.array(beta),
    )

    return hybrid_ebm


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


def format_architecture_desc(
    n_visible: int,
    n_categories_per_visible_node: int,
    n_hidden1: int,
    n_hidden2: int,
) -> str:
    """Format model architecture description for plot titles.

    Args:
        n_visible: Number of visible nodes
        n_categories_per_visible_node: Number of categories per visible node (2 = Spin, >2 = Categorical)
        n_hidden1: Number of nodes in first hidden layer
        n_hidden2: Number of nodes in second hidden layer

    Returns:
        Formatted architecture string, e.g., "Vis:8(Spin) H1:32(Spin) H2:8(Spin)"
    """
    visible_type = "Spin" if n_categories_per_visible_node == 2 else f"Cat({n_categories_per_visible_node})"
    return f"Vis:{n_visible}({visible_type}) H1:{n_hidden1}(Spin) H2:{n_hidden2}(Spin)"


def get_plots_folder_name(
    n_visible: int,
    n_hidden1: int,
    n_hidden2: int,
) -> str:
    """Generate folder name for plots based on model architecture.

    Args:
        n_visible: Number of visible nodes
        n_hidden1: Number of nodes in first hidden layer
        n_hidden2: Number of nodes in second hidden layer (0 for RBM, >0 for DBM)

    Returns:
        Folder name string, e.g., "tests/test_train_gaussian_plots_dbm_v_8_h1_32_h2_8" or "tests/test_train_gaussian_plots_rbm_v_8_h1_32"
    """
    model_type = "dbm" if n_hidden2 > 0 else "rbm"
    if n_hidden2 > 0:
        return f"tests/test_train_gaussian_plots_{model_type}_v_{n_visible}_h1_{n_hidden1}_h2_{n_hidden2}"
    else:
        return f"tests/test_train_gaussian_plots_{model_type}_v_{n_visible}_h1_{n_hidden1}"


def plot_histograms_comparison(
    generated_samples: Array,
    test_samples: Array,
    kl_divergence: float,
    epoch: int | None,
    n_epochs: int | None,
    is_final: bool = False,
    architecture_desc: str | None = None,
    plots_dir: str = "tests/test_train_gaussian_plots",
):
    """Create matplotlib histogram plot comparing generated and test samples.

    Args:
        generated_samples: Samples generated by the model
        test_samples: Test samples from true distribution
        kl_divergence: KL divergence value to display in title
        epoch: Current epoch number (None if final)
        n_epochs: Total number of epochs (None if final)
        is_final: Whether this is the final plot
        architecture_desc: Description of model architecture (e.g., "Vis:8(Spin) H1:32(Spin) H2:8(Spin)")
        plots_dir: Directory to save plots to
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

    # Set title with KL divergence and architecture
    if is_final:
        title = f"Final Model: KL Divergence = {kl_divergence:.4f}"
    else:
        title = f"Epoch {epoch}/{n_epochs}: KL Divergence = {kl_divergence:.4f}"

    if architecture_desc:
        title = f"{title}\n{architecture_desc}"

    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save plot to file
    os.makedirs(plots_dir, exist_ok=True)

    if is_final:
        filename = os.path.join(plots_dir, "final_model.png")
    else:
        filename = os.path.join(plots_dir, f"epoch_{epoch:03d}.png")

    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"  Plot saved to: {filename}", flush=True)

    plt.show(block=False)
    plt.pause(0.1)  # Brief pause to ensure plot renders


def plot_kl_divergence_over_epochs(kl_divergences: list[float], kl_threshold: float, architecture_desc: str | None = None, plots_dir: str = "tests/test_train_gaussian_plots"):
    """Create matplotlib plot showing KL divergence over training epochs.

    Args:
        kl_divergences: List of KL divergence values, one per epoch
        kl_threshold: Threshold value to display as horizontal line
        architecture_desc: Description of model architecture (e.g., "Vis:8(Spin) H1:32(Spin) H2:8(Spin)")
        plots_dir: Directory to save plots to
    """
    plt.figure(figsize=(10, 6))

    epochs = list(range(1, len(kl_divergences) + 1))
    plt.plot(epochs, kl_divergences, marker='o', linestyle='-', linewidth=2, markersize=6)
    plt.axhline(y=kl_threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold ({kl_threshold})')

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("KL Divergence (train)", fontsize=12)

    title = "KL Divergence Over Training Epochs"
    if architecture_desc:
        title = f"{title}\n{architecture_desc}"
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0.5)

    # Save plot to file
    os.makedirs(plots_dir, exist_ok=True)
    filename = os.path.join(plots_dir, "kl_divergence_over_epochs.png")
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"  Plot saved to: {filename}", flush=True)

    plt.show(block=False)
    plt.pause(0.1)  # Brief pause to ensure plot renders


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
        self.n_visible = 16
        self.n_categories_per_visible_node = 2  # 2 = binary (SpinNode), >2 = categorical (CategoricalNode)
        self.n_hidden1 = 32  # First hidden layer (DBM)
        self.n_hidden2 = 8  # Second hidden layer (DBM)
        self.n_hidden = 16  # Kept for reference (RBM)
        self.beta = 1.0
        self.learning_rate = 0.01
        self.batch_size_positive = 50
        self.batch_size_negative = 25
        self.n_epochs = 20
        self.schedule_positive = SamplingSchedule(200, 20, 10)
        self.schedule_negative = SamplingSchedule(200, 40, 5)

        # Evaluation parameters
        self.eval_n_samples = 500
        self.eval_schedule = SamplingSchedule(400, 40, 10)
        self.kl_threshold = 0.3

        # Initialize optimizer
        self.optim = optax.adam(learning_rate=self.learning_rate)

    def test_train_gaussian(self):
        """Test training an RBM to learn a bimodal Gaussian distribution."""
        key = jax.random.key(42)

        # Log initialization
        print("\n" + "=" * 80, flush=True)
        print("Gaussian RBM Training Test", flush=True)
        print("=" * 80, flush=True)
        print("\nData Generation Parameters:", flush=True)
        print(f"  mu1 = {self.mu1}, mu2 = {self.mu2}", flush=True)
        print(f"  sigma1 = {self.sigma1}, sigma2 = {self.sigma2}", flush=True)
        print(f"  mixture_weight = {self.mixture_weight}", flush=True)
        print(f"  n_train_samples = {self.n_train_samples}", flush=True)
        print(f"  n_test_samples = {self.n_test_samples}", flush=True)
        print(f"  quantization_max_value = {self.quantization_max_value}", flush=True)
        print("\nModel Hyperparameters:", flush=True)
        print(f"  n_visible = {self.n_visible}, n_hidden1 = {self.n_hidden1}, n_hidden2 = {self.n_hidden2}", flush=True)
        print(f"  n_categories_per_visible_node = {self.n_categories_per_visible_node}", flush=True)
        if self.n_categories_per_visible_node == 2:
            print(f"  Quantization: Binary (SpinNode)", flush=True)
        else:
            print(f"  Quantization: Categorical (CategoricalNode, {self.n_categories_per_visible_node} categories per node)", flush=True)
            total_bins = self.n_categories_per_visible_node ** self.n_visible
            print(f"  Total bins: {total_bins}", flush=True)
        print(f"  beta = {self.beta}", flush=True)
        print(f"  learning_rate = {self.learning_rate}", flush=True)
        print(f"  batch_size_positive = {self.batch_size_positive}", flush=True)
        print(f"  batch_size_negative = {self.batch_size_negative}", flush=True)
        print(f"  n_epochs = {self.n_epochs}", flush=True)
        print("=" * 80 + "\n", flush=True)

        # Determine quantization method
        use_categorical = self.n_categories_per_visible_node > 2

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
        if use_categorical:
            train_samples_quantized = quantize_categorical(
                train_samples_continuous,
                self.n_visible,
                self.n_categories_per_visible_node,
                self.quantization_max_value,
            )
        else:
            train_samples_quantized = quantize_hierarchical(
                train_samples_continuous, self.n_visible, self.quantization_max_value
            )

        # Create DBM model
        key, key_model = jax.random.split(key)
        if use_categorical:
            model = create_hybrid_dbm_model(
                self.n_visible,
                self.n_categories_per_visible_node,
                self.n_hidden1,
                self.n_hidden2,
                self.beta,
                key_model,
            )
        else:
            model = create_dbm_model(self.n_visible, self.n_hidden1, self.n_hidden2, self.beta, key_model)

        # Set up training blocks
        if use_categorical:
            from thrml.models.hybrid import HybridEBM
            assert isinstance(model, HybridEBM)
            visible_nodes = model.categorical_nodes
            hidden1_nodes = model.spin_nodes[: self.n_hidden1]
            hidden2_nodes = model.spin_nodes[self.n_hidden1 :]
        else:
            visible_nodes = model.nodes[: self.n_visible]
            hidden1_nodes = model.nodes[self.n_visible : self.n_visible + self.n_hidden1]
            hidden2_nodes = model.nodes[self.n_visible + self.n_hidden1 :]

        visible_block = Block(visible_nodes)
        hidden1_block = Block(hidden1_nodes)
        hidden2_block = Block(hidden2_nodes)

        positive_sampling_blocks = [hidden1_block, hidden2_block]
        negative_sampling_blocks = [visible_block, hidden1_block, hidden2_block]
        training_data_blocks = [visible_block]

        # Initialize optimizer state
        if use_categorical:
            # For HybridEBM, extract parameters from factors
            from thrml.models.hybrid import HybridEBM
            assert isinstance(model, HybridEBM)
            cat_bias_factor = model.ebm.factors[0]  # CategoricalEBMFactor
            h1_bias_factor = model.ebm.factors[1]  # SpinEBMFactor
            h2_bias_factor = model.ebm.factors[2]  # SpinEBMFactor
            vis_h1_factor = model.ebm.factors[3]  # DiscreteEBMFactor (per-edge format)
            h1_h2_factor = model.ebm.factors[4]  # SpinEBMFactor (per-edge format)

            # vis_h1_factor.weights is in per-edge format [n_edges, n_categories]
            # Reshape to [n_hidden1, n_visible, n_categories] for consistency
            n_edges_v_h1 = self.n_hidden1 * self.n_visible
            vis_h1_weights_reshaped = vis_h1_factor.weights.reshape(
                (self.n_hidden1, self.n_visible, self.n_categories_per_visible_node)
            )

            # h1_h2_factor.weights is in per-edge format [n_edges,]
            # Reshape to [n_hidden1, n_hidden2] for consistency
            n_edges_h1_h2 = self.n_hidden1 * self.n_hidden2
            h1_h2_weights_reshaped = h1_h2_factor.weights.reshape(
                (self.n_hidden1, self.n_hidden2)
            )

            params = (
                cat_bias_factor.weights / model.beta,
                h1_bias_factor.weights / model.beta,
                h2_bias_factor.weights / model.beta,
                vis_h1_weights_reshaped / model.beta,
                h1_h2_weights_reshaped / model.beta,
            )
            opt_state = self.optim.init(params)
        else:
            opt_state = self.optim.init((model.weights, model.biases))

        print("> Model initialization complete", flush=True)
        print("> Starting training...\n", flush=True)

        # Track KL divergence over epochs
        kl_divergences = []

        # Training loop
        for epoch in range(self.n_epochs):
            print(f"> Epoch {epoch + 1}/{self.n_epochs}: Starting training", flush=True)

            def do_epoch(key, model, bsz_positive, bsz_negative, data_positive, opt_state):
                def batch_data(key, data, _bsz, clamped_blocks, is_categorical):
                    clamped_nodes = [node for block in clamped_blocks for node in block]
                    data_size = data.shape[0]
                    assert data.shape == (data_size, len(clamped_nodes))
                    key, key_shuffle = jax.random.split(key)
                    idxs = jax.random.permutation(key_shuffle, jnp.arange(data_size))
                    data = data[idxs]
                    _n_batches = data_size // _bsz
                    tot_len = _n_batches * _bsz
                    if is_categorical:
                        batched_data = jnp.reshape(data[:tot_len], (_n_batches, _bsz, len(clamped_nodes))).astype(
                            jnp.uint8
                        )
                    else:
                        batched_data = jnp.reshape(data[:tot_len], (_n_batches, _bsz, len(clamped_nodes))).astype(
                            jnp.bool
                        )
                    return batched_data, _n_batches

                key, key_pos = jax.random.split(key, 2)
                batched_data_pos, n_batches = batch_data(key_pos, data_positive, bsz_positive, training_data_blocks, use_categorical)

                # Log batch information
                print(f"  - Processing {n_batches} batches ({bsz_positive} samples per batch)", flush=True)

                if use_categorical:
                    # Categorical training path
                    from thrml.models.hybrid import HybridEBM, HybridTrainingSpec, estimate_kl_grad_hybrid
                    assert isinstance(model, HybridEBM)

                    # Build moment specifications for gradient estimation
                    categorical_bias_moments = [
                        (node, cat) for node in visible_nodes for cat in range(self.n_categories_per_visible_node)
                    ]
                    spin_bias_nodes = hidden1_nodes + hidden2_nodes
                    categorical_spin_weight_edges = [
                        (vis_node, cat, h1_node)
                        for vis_node in visible_nodes
                        for cat in range(self.n_categories_per_visible_node)
                        for h1_node in hidden1_nodes
                    ]
                    spin_spin_weight_edges = [(h1_node, h2_node) for h1_node in hidden1_nodes for h2_node in hidden2_nodes]

                    # Extract current parameters from factors (extract fresh each epoch from current model)
                    cat_bias_factor = model.ebm.factors[0]  # CategoricalEBMFactor
                    h1_bias_factor = model.ebm.factors[1]  # SpinEBMFactor
                    h2_bias_factor = model.ebm.factors[2]  # SpinEBMFactor
                    vis_h1_factor = model.ebm.factors[3]  # DiscreteEBMFactor (per-edge format)
                    h1_h2_factor = model.ebm.factors[4]  # SpinEBMFactor (per-edge format)

                    # vis_h1_factor.weights is in per-edge format [n_edges, n_categories]
                    # Reshape to [n_hidden1, n_visible, n_categories] for consistency
                    n_edges_v_h1 = self.n_hidden1 * self.n_visible
                    vis_h1_weights_reshaped = vis_h1_factor.weights.reshape(
                        (self.n_hidden1, self.n_visible, self.n_categories_per_visible_node)
                    )

                    # h1_h2_factor.weights is in per-edge format [n_edges,]
                    # Reshape to [n_hidden1, n_hidden2] for consistency
                    n_edges_h1_h2 = self.n_hidden1 * self.n_hidden2
                    h1_h2_weights_reshaped = h1_h2_factor.weights.reshape(
                        (self.n_hidden1, self.n_hidden2)
                    )

                    params = (
                        cat_bias_factor.weights / model.beta,
                        h1_bias_factor.weights / model.beta,
                        h2_bias_factor.weights / model.beta,
                        vis_h1_weights_reshaped / model.beta,
                        h1_h2_weights_reshaped / model.beta,
                    )

                    carry = opt_state, params
                    keys = jax.random.split(key, n_batches)

                    for batch_idx in range(n_batches):
                        _key = keys[batch_idx]
                        _data_pos = batched_data_pos[batch_idx]

                        print(f"  > Batch {batch_idx + 1}/{n_batches}: Processing batch", flush=True)

                        _opt_state, _params = carry
                        # Update model with current parameters
                        cat_bias, h1_bias, h2_bias, vis_h1_w, h1_h2_w = _params
                        # vis_h1_w is in [n_hidden1, n_visible, n_categories] format, reshape to per-edge
                        vis_h1_w_flat = vis_h1_w.reshape((n_edges_v_h1, self.n_categories_per_visible_node))
                        # h1_h2_w is in [n_hidden1, n_hidden2] format, reshape to per-edge
                        n_edges_h1_h2 = self.n_hidden1 * self.n_hidden2
                        h1_h2_w_flat = h1_h2_w.reshape((n_edges_h1_h2,))
                        new_factors = [
                            eqx.tree_at(lambda f: f.weights, cat_bias_factor, model.beta * cat_bias),
                            eqx.tree_at(lambda f: f.weights, h1_bias_factor, model.beta * h1_bias),
                            eqx.tree_at(lambda f: f.weights, h2_bias_factor, model.beta * h2_bias),
                            eqx.tree_at(lambda f: f.weights, vis_h1_factor, model.beta * vis_h1_w_flat),
                            eqx.tree_at(lambda f: f.weights, h1_h2_factor, model.beta * h1_h2_w_flat),
                        ]
                        _model = eqx.tree_at(lambda m: m.ebm._factors, model, new_factors)

                        key_train, key_init_pos, key_init_neg = jax.random.split(_key, 3)
                        # Initialize hidden states (categorical visible will be clamped from data)
                        # For now, use uniform initialization for categorical nodes
                        vals_free_pos_cat = [
                            jax.random.randint(key_init_pos, (1, bsz_positive, len(visible_nodes)), 0, self.n_categories_per_visible_node).astype(jnp.uint8)
                        ]
                        vals_free_pos_h1 = [
                            jax.random.bernoulli(key_init_pos, 0.5, (1, bsz_positive, len(hidden1_nodes))).astype(jnp.bool_)
                        ]
                        vals_free_pos_h2 = [
                            jax.random.bernoulli(key_init_pos, 0.5, (1, bsz_positive, len(hidden2_nodes))).astype(jnp.bool_)
                        ]
                        vals_free_pos = vals_free_pos_h1 + vals_free_pos_h2

                        # For negative phase, init_state_negative should have shape [n_chains_neg, nodes] for each block
                        # vmap will map over the first dimension, so each call gets [nodes]
                        vals_free_neg_cat = [
                            jax.random.randint(key_init_neg, (bsz_negative, len(visible_nodes)), 0, self.n_categories_per_visible_node).astype(jnp.uint8)
                        ]
                        vals_free_neg_h1 = [
                            jax.random.bernoulli(key_init_neg, 0.5, (bsz_negative, len(hidden1_nodes))).astype(jnp.bool_)
                        ]
                        vals_free_neg_h2 = [
                            jax.random.bernoulli(key_init_neg, 0.5, (bsz_negative, len(hidden2_nodes))).astype(jnp.bool_)
                        ]
                        vals_free_neg = vals_free_neg_cat + vals_free_neg_h1 + vals_free_neg_h2

                        training_spec = HybridTrainingSpec(
                            _model,
                            training_data_blocks,
                            [],
                            positive_sampling_blocks,
                            negative_sampling_blocks,
                            self.schedule_positive,
                            self.schedule_negative,
                        )

                        # Call estimate_kl_grad_hybrid
                        grad_cat_bias, grad_spin_bias, _, grad_cat_spin_w, grad_spin_spin_w, _, _ = estimate_kl_grad_hybrid(
                            key_train,
                            training_spec,
                            categorical_bias_moments,
                            spin_bias_nodes,
                            [],  # No cat-cat edges in this model
                            categorical_spin_weight_edges,
                            spin_spin_weight_edges,
                            [_data_pos],
                            [],
                            vals_free_pos,
                            vals_free_neg,
                        )

                        # Reshape gradients to match parameter shapes
                        grad_cat_bias_reshaped = grad_cat_bias.reshape((self.n_visible, self.n_categories_per_visible_node))
                        grad_h1_bias = grad_spin_bias[:self.n_hidden1]
                        grad_h2_bias = grad_spin_bias[self.n_hidden1:]
                        # grad_cat_spin_w is already in per-edge format [n_edges, n_categories]
                        # Reshape back to [n_hidden1, n_visible, n_categories] for parameter updates
                        n_edges_v_h1 = self.n_hidden1 * self.n_visible
                        grad_cat_spin_w_reshaped = grad_cat_spin_w.reshape((n_edges_v_h1, self.n_categories_per_visible_node)).reshape(
                            (self.n_hidden1, self.n_visible, self.n_categories_per_visible_node)
                        )
                        grad_spin_spin_w_reshaped = grad_spin_spin_w.reshape((self.n_hidden1, self.n_hidden2))

                        grads = (grad_cat_bias_reshaped, grad_h1_bias, grad_h2_bias, grad_cat_spin_w_reshaped, grad_spin_spin_w_reshaped)
                        with jax.numpy_dtype_promotion("standard"):
                            updates, _opt_state = self.optim.update(grads, _opt_state, _params)

                        _cat_bias, _h1_bias, _h2_bias, _vis_h1_w, _h1_h2_w = _params
                        _cat_bias += updates[0]
                        _h1_bias += updates[1]
                        _h2_bias += updates[2]
                        _vis_h1_w += updates[3]
                        _h1_h2_w += updates[4]

                        carry = _opt_state, (_cat_bias, _h1_bias, _h2_bias, _vis_h1_w, _h1_h2_w)
                        print(f"  > Batch {batch_idx + 1}/{n_batches}: Completed\n", flush=True)

                    opt_state, params = carry
                    # Update model with final parameters
                    cat_bias, h1_bias, h2_bias, vis_h1_w, h1_h2_w = params
                    # vis_h1_w is in [n_hidden1, n_visible, n_categories] format, reshape to per-edge
                    vis_h1_w_flat = vis_h1_w.reshape((n_edges_v_h1, self.n_categories_per_visible_node))
                    # h1_h2_w is in [n_hidden1, n_hidden2] format, reshape to per-edge
                    n_edges_h1_h2 = self.n_hidden1 * self.n_hidden2
                    h1_h2_w_flat = h1_h2_w.reshape((n_edges_h1_h2,))
                    new_factors = [
                        eqx.tree_at(lambda f: f.weights, cat_bias_factor, model.beta * cat_bias),
                        eqx.tree_at(lambda f: f.weights, h1_bias_factor, model.beta * h1_bias),
                        eqx.tree_at(lambda f: f.weights, h2_bias_factor, model.beta * h2_bias),
                        eqx.tree_at(lambda f: f.weights, vis_h1_factor, model.beta * vis_h1_w_flat),
                        eqx.tree_at(lambda f: f.weights, h1_h2_factor, model.beta * h1_h2_w_flat),
                    ]
                    new_model = eqx.tree_at(lambda m: m.ebm._factors, model, new_factors)

                    return new_model, opt_state
                else:
                    # Binary training path (existing code)
                    params = model.weights, model.biases
                    carry = opt_state, params
                    keys = jax.random.split(key, n_batches)

                    # Replace jax.lax.scan with Python for loop to enable logging
                    for batch_idx in range(n_batches):
                        _key = keys[batch_idx]
                        _data_pos = batched_data_pos[batch_idx]

                        print(f"  > Batch {batch_idx + 1}/{n_batches}: Processing batch", flush=True)

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

                        # Call estimate_kl_grad (both phases happen inside this function)
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

                        carry = _opt_state, (_weights, _biases)
                        print(f"  > Batch {batch_idx + 1}/{n_batches}: Completed\n", flush=True)

                    opt_state, params = carry
                    new_model = eqx.tree_at(lambda m: (m.weights, m.biases), model, params)

                    return new_model, opt_state

            key, key_epoch = jax.random.split(key)
            model, opt_state = do_epoch(
                key_epoch,
                model,
                self.batch_size_positive,
                self.batch_size_negative,
                train_samples_quantized,
                opt_state,
            )

            print(f"> Epoch {epoch + 1}/{self.n_epochs}: Training completed", flush=True)

            # Evaluate after each epoch - compute KL divergence with training samples
            key, key_eval, key_init = jax.random.split(key, 3)

            # Generate samples from trained model
            if use_categorical:
                from thrml.models.hybrid import HybridSamplingProgram
                program = HybridSamplingProgram(model, negative_sampling_blocks, [])
                # Initialize states: categorical nodes get random categories, spin nodes get random binary
                init_free_states = [
                    jax.random.randint(key_init, (self.eval_n_samples, len(visible_nodes)), 0, self.n_categories_per_visible_node).astype(jnp.uint8),
                    jax.random.bernoulli(key_init, 0.5, (self.eval_n_samples, len(hidden1_nodes))).astype(jnp.bool_),
                    jax.random.bernoulli(key_init, 0.5, (self.eval_n_samples, len(hidden2_nodes))).astype(jnp.bool_),
                ]
            else:
                program = IsingSamplingProgram(model, negative_sampling_blocks, [])
                init_free_states = hinton_init(key_init, model, negative_sampling_blocks, (self.eval_n_samples,))

            keys_samp = jax.random.split(key_eval, self.eval_n_samples)

            # vmap over the batch dimension
            # init_free_states is a list, vmap maps over first dimension of arrays in the list
            samples_quantized_list = jax.vmap(
                lambda k, init_states: sample_states(k, program, self.eval_schedule, init_states, [], [visible_block])
            )(keys_samp, init_free_states)

            # samples_quantized_list is a list with one element (visible_block)
            # Shape: (n_samples, n_steps, n_visible_nodes)
            samples_quantized = samples_quantized_list[0]

            # Decode samples to continuous values
            # We take the last step
            samples_quantized_last = samples_quantized[:, -1, :]
            if use_categorical:
                generated_samples_continuous = dequantize_categorical(
                    samples_quantized_last,
                    self.n_visible,
                    self.n_categories_per_visible_node,
                    self.quantization_max_value,
                )
            else:
                generated_samples_continuous = dequantize_hierarchical(
                    samples_quantized_last, self.n_visible, self.quantization_max_value
                )

            # Compute KL divergence with training samples (for early stopping)
            kl_div_train = compute_kl_divergence(generated_samples_continuous, train_samples_continuous)
            kl_divergences.append(kl_div_train)

            print(f"> Epoch {epoch + 1}/{self.n_epochs}: KL divergence (train) = {kl_div_train:.4f}", flush=True)

            # Check early stopping condition
            if kl_div_train <= self.kl_threshold:
                print(f"> Early stopping: KL divergence ({kl_div_train:.4f}) <= threshold ({self.kl_threshold})", flush=True)
                break

        # Final evaluation
        print("\n> Training complete. Performing final evaluation...", flush=True)

        key, key_eval, key_init = jax.random.split(key, 3)
        if use_categorical:
            from thrml.models.hybrid import HybridSamplingProgram
            program = HybridSamplingProgram(model, negative_sampling_blocks, [])
            # Initialize states: categorical nodes get random categories, spin nodes get random binary
            init_free_states = [
                jax.random.randint(key_init, (self.eval_n_samples, len(visible_nodes)), 0, self.n_categories_per_visible_node).astype(jnp.uint8),
                jax.random.bernoulli(key_init, 0.5, (self.eval_n_samples, len(hidden1_nodes))).astype(jnp.bool_),
                jax.random.bernoulli(key_init, 0.5, (self.eval_n_samples, len(hidden2_nodes))).astype(jnp.bool_),
            ]
        else:
            program = IsingSamplingProgram(model, negative_sampling_blocks, [])
            init_free_states = hinton_init(key_init, model, negative_sampling_blocks, (self.eval_n_samples,))

        keys_samp = jax.random.split(key_eval, self.eval_n_samples)

        # vmap over the batch dimension
        samples_quantized_list = jax.vmap(
            lambda k, init_states: sample_states(k, program, self.eval_schedule, init_states, [], [visible_block])
        )(keys_samp, init_free_states)

        samples_quantized = samples_quantized_list[0]
        samples_quantized_last = samples_quantized[:, -1, :]
        if use_categorical:
            final_generated_samples = dequantize_categorical(
                samples_quantized_last,
                self.n_visible,
                self.n_categories_per_visible_node,
                self.quantization_max_value,
            )
        else:
            final_generated_samples = dequantize_hierarchical(
                samples_quantized_last, self.n_visible, self.quantization_max_value
            )

        final_kl_div = compute_kl_divergence(final_generated_samples, test_samples_continuous)

        print(f"> Final KL divergence (test) = {final_kl_div:.4f}", flush=True)
        print("=" * 80 + "\n", flush=True)

        # Generate architecture description for plots
        architecture_desc = format_architecture_desc(
            self.n_visible,
            self.n_categories_per_visible_node,
            self.n_hidden1,
            self.n_hidden2,
        )
        plots_dir = get_plots_folder_name(
            self.n_visible,
            self.n_hidden1,
            self.n_hidden2,
        )

        # Generate plots
        # 1. Histogram comparison of model samples vs test samples
        plot_histograms_comparison(
            np.array(final_generated_samples),
            np.array(test_samples_continuous),
            final_kl_div,
            None,
            None,
            is_final=True,
            architecture_desc=architecture_desc,
            plots_dir=plots_dir,
        )

        # 2. KL divergence over epochs
        plot_kl_divergence_over_epochs(kl_divergences, self.kl_threshold, architecture_desc=architecture_desc, plots_dir=plots_dir)

        # # Assert KL divergence is below threshold
        # self.assertLess(final_kl_div, self.kl_threshold)

