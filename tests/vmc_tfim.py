"""
Variational Monte Carlo training for 1D Transverse Field Ising Model using RBM.

This script trains an RBM to learn the ground state wavefunction of the 1D TFIM
with J=Γ=1, which should converge to E/N = -2/π ≈ -0.6366.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Key
from math import sqrt

from thrml.block_sampling import Block, SamplingSchedule, sample_states, SuperBlock
from thrml.models.ising import IsingEBM, IsingSamplingProgram, hinton_init
from thrml.pgm import SpinNode


def create_tfim_rbm_model(n_visible: int, n_hidden: int, beta: float, key: Key[Array, ""]) -> IsingEBM:
    """Create an RBM model for TFIM VMC training.

    Args:
        n_visible: Number of visible nodes (system size N)
        n_hidden: Number of hidden nodes
        beta: Temperature parameter
        key: JAX random key for weight initialization

    Returns:
        IsingEBM with RBM structure (visible ↔ hidden, all-to-all)
    """
    # Create nodes
    visible_nodes = [SpinNode() for _ in range(n_visible)]
    hidden_nodes = [SpinNode() for _ in range(n_hidden)]
    all_nodes = visible_nodes + hidden_nodes

    # Create all-to-all edges between visible and hidden
    edges = []
    for v_node in visible_nodes:
        for h_node in hidden_nodes:
            edges.append((v_node, h_node))

    # Initialize weights and biases with small Gaussian noise
    key_weights, key_biases = jax.random.split(key, 2)
    n_edges = len(edges)
    n_nodes = len(all_nodes)

    weights = jax.random.normal(key_weights, (n_edges,), dtype=float) * 0.01
    biases = jax.random.normal(key_biases, (n_nodes,), dtype=float) * 0.01

    return IsingEBM(all_nodes, edges, biases, weights, jnp.array(beta))


def compute_log_wavefunction_amplitude_rbm(model: IsingEBM, visible_state: Array) -> Array:
    """Compute log|ψ(s)| for RBM using exact marginalization over hidden variables.

    For RBM: E_RBM(s, h) = -β * [Σᵢ aᵢsᵢ + Σⱼ bⱼhⱼ + Σᵢⱼ Wᵢⱼsᵢhⱼ]
    where s_i, h_j ∈ {-1,+1} (Ising spins).

    Hidden variables are conditionally independent, so we can compute:
    log|ψ(s)| = 0.5 * (E_visible + log_Z_hidden)
    where Z_hidden = Π_j [2 * cosh(β * (b_j + Σᵢ W_ij * s_i))]

    Args:
        model: RBM model (IsingEBM)
        visible_state: Boolean array of shape (N,) representing spin configuration

    Returns:
        Scalar log|ψ(s)|
    """
    # Convert boolean to spins: True -> +1, False -> -1
    s = 2 * visible_state.astype(jnp.int32) - 1  # shape: (N,)

    # Get model parameters
    # Need to extract visible and hidden biases, and weights
    n_visible = len(visible_state)
    n_hidden = len(model.nodes) - n_visible

    # Split biases: first n_visible are visible, rest are hidden
    visible_biases = model.biases[:n_visible]
    hidden_biases = model.biases[n_visible:]

    # Extract weights: model.weights has shape (n_edges,) where n_edges = n_visible * n_hidden
    # We need to reshape to (n_visible, n_hidden)
    weights_matrix = model.weights.reshape(n_visible, n_hidden)

    # Compute visible bias term (with beta scaling)
    # E_visible = -β * Σ a_i s_i
    E_visible = -model.beta * jnp.sum(visible_biases * s)

    # For each hidden unit j, compute log(2 * cosh(β * (b_j + Σᵢ W_ij * s_i)))
    # Since h_j ∈ {-1,+1}, we need: Z_hidden_j = 2 * cosh(β * (b_j + Σᵢ W_ij * s_i))
    # Shape: (n_hidden,)
    logits_hidden = model.beta * (hidden_biases + jnp.dot(s, weights_matrix))  # (n_hidden,)

    # log(2 * cosh(x)) = log(exp(x) + exp(-x)) = logsumexp([x, -x])
    log_contributions = jax.nn.logsumexp(jnp.stack([logits_hidden, -logits_hidden]), axis=0)
    log_Z_hidden = jnp.sum(log_contributions)

    # Final wavefunction amplitude (up to normalization)
    log_psi = 0.5 * (E_visible + log_Z_hidden)

    return log_psi


def compute_local_energy_tfim(
    visible_state: Array,
    log_psi_current: Array,
    log_psi_flipped_list: Array,
    J: float,
    Gamma: float,
    N: int,
) -> Array:
    """Compute local energy E_loc(s) for TFIM.

    H = -J Σ σᵢᶻ σⱼᶻ - Γ Σ σᵢˣ

    Args:
        visible_state: Boolean array (N,) representing spin configuration
        log_psi_current: log|ψ(s)|
        log_psi_flipped_list: Array (N,) of log|ψ(s^(i))| where s^(i) flips spin i
        J: Ising coupling constant
        Gamma: Transverse field strength
        N: Number of sites

    Returns:
        Scalar local energy E_loc(s)
    """
    # Convert boolean to spins: True -> +1, False -> -1
    s = 2 * visible_state.astype(jnp.int32) - 1  # shape: (N,)

    # Diagonal term: -J Σ σᵢᶻ σⱼᶻ (periodic boundary conditions)
    E_diag = -J * jnp.sum(s * jnp.roll(s, -1))  # roll gives periodic BC

    # Off-diagonal term: -Γ Σᵢ ψ(s^(i))/ψ(s)
    # log_psi_flipped_list[i] - log_psi_current gives log(ψ(s^(i))/ψ(s))
    log_ratios = log_psi_flipped_list - log_psi_current
    E_offdiag = -Gamma * jnp.sum(jnp.exp(log_ratios))

    return E_diag + E_offdiag


def compute_vmc_energy(
    model: IsingEBM,
    key: Key[Array, ""],
    n_samples: int,
    J: float,
    Gamma: float,
    schedule: SamplingSchedule,
    visible_nodes: list[SpinNode],
    hidden_nodes: list[SpinNode],
    visible_block: Block,
) -> tuple[Array, Array, Array]:
    """Compute VMC energy expectation value and variance.

    Args:
        model: RBM model
        key: JAX random key
        n_samples: Number of samples to use
        J: Ising coupling constant
        Gamma: Transverse field strength
        schedule: Sampling schedule
        visible_nodes: List of visible nodes
        hidden_nodes: List of hidden nodes
        visible_block: Block containing visible nodes

    Returns:
        Tuple of (mean_energy, energy_variance, local_energies)
    """
    N = len(visible_nodes)

    # Create sampling program
    free_blocks: list[SuperBlock] = [Block(visible_nodes), Block(hidden_nodes)]
    program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])

    # Initialize states
    key, key_init = jax.random.split(key)
    init_states = hinton_init(key_init, model, free_blocks, ())

    # Sample visible states from |ψ|²
    key, key_sample = jax.random.split(key)
    samples_list = sample_states(
        key_sample,
        program,
        schedule,
        init_states,
        [],
        [visible_block],
    )

    # Extract visible samples: shape (n_samples, N)
    # sample_states returns samples with shape (n_samples, n_nodes) directly
    visible_samples = samples_list[0]  # shape: (n_samples, N)

    # Compute local energies for each sample
    def compute_local_energy_for_sample(sample):
        # Compute log|ψ(s)|
        log_psi_current = compute_log_wavefunction_amplitude_rbm(model, sample)

        # Compute log|ψ(s^(i))| for all spin flips using vmap
        def flip_and_compute(i):
            flipped_sample = sample.at[i].set(~sample[i])
            return compute_log_wavefunction_amplitude_rbm(model, flipped_sample)

        log_psi_flipped_list = jax.vmap(flip_and_compute)(jnp.arange(N))

        # Compute local energy
        return compute_local_energy_tfim(sample, log_psi_current, log_psi_flipped_list, J, Gamma, N)

    # Vectorize over samples
    local_energies = jax.vmap(compute_local_energy_for_sample)(visible_samples)

    # Compute mean and variance
    mean_energy = jnp.mean(local_energies)
    energy_variance = jnp.var(local_energies)

    return mean_energy, energy_variance, local_energies, visible_samples


def compute_vmc_gradients(
    model: IsingEBM,
    samples: Array,
    local_energies: Array,
    mean_energy: Array,
) -> tuple[Array, Array]:
    """Compute VMC gradients using log-derivative trick.

    ∇E = 2 * E[(E_loc - E_mean) * ∇log|ψ|]

    Args:
        model: RBM model
        samples: Array of shape (n_samples, N) of visible state samples
        local_energies: Array of shape (n_samples,) of local energies
        mean_energy: Scalar mean energy

    Returns:
        Tuple of (grad_weights, grad_biases)
    """
    energy_diff = local_energies - mean_energy  # shape: (n_samples,)

    def loss_fn(weights, biases):
        """Loss function for gradient computation."""
        # Create temporary model with these parameters
        temp_model = eqx.tree_at(lambda m: (m.weights, m.biases), model, (weights, biases))
        # Compute log|ψ| for all samples
        log_psi_vals = jax.vmap(lambda s: compute_log_wavefunction_amplitude_rbm(temp_model, s))(samples)
        # Weighted mean: (E_loc - E_mean) * log|ψ|
        return jnp.mean(energy_diff * log_psi_vals)

    # Compute gradients using JAX autodiff
    grad_fn = jax.grad(loss_fn, argnums=(0, 1))
    grad_weights, grad_biases = grad_fn(model.weights, model.biases)

    # Scale by 2 (from the log-derivative trick formula)
    grad_weights = 2.0 * grad_weights
    grad_biases = 2.0 * grad_biases

    return grad_weights, grad_biases


def train_vmc_tfim(
    model: IsingEBM,
    key: Key[Array, ""],
    n_epochs: int = 50,
    n_samples_per_epoch: int = 2000,
    J: float = 1.0,
    Gamma: float = 1.0,
    learning_rate: float = 0.001,
    variance_threshold: float = 1e-5,
    variance_window: int = 5,
    schedule: SamplingSchedule | None = None,
    visible_nodes: list[SpinNode] | None = None,
    hidden_nodes: list[SpinNode] | None = None,
) -> tuple[IsingEBM, list[float]]:
    """Train RBM using VMC to learn TFIM ground state.

    Args:
        model: Initial RBM model
        key: JAX random key
        n_epochs: Maximum number of training epochs
        n_samples_per_epoch: Number of samples per epoch
        J: Ising coupling constant
        Gamma: Transverse field strength
        learning_rate: Learning rate for Adam optimizer
        variance_threshold: Early stopping threshold for energy variance
        variance_window: Number of epochs to check variance over
        schedule: Sampling schedule (if None, creates default)
        visible_nodes: List of visible nodes (if None, extracts from model)
        hidden_nodes: List of hidden nodes (if None, extracts from model)

    Returns:
        Tuple of (trained_model, energy_history)
    """
    # Extract nodes if not provided
    if visible_nodes is None or hidden_nodes is None:
        # Assume first N nodes are visible, rest are hidden
        # We need to determine N from the model structure
        # For RBM, we can infer from the number of edges
        n_edges = len(model.edges)
        # n_edges = n_visible * n_hidden
        # We'll need to infer or pass this explicitly
        # For now, let's assume we can determine it from the model
        all_nodes = model.nodes
        # We'll need to split them - this is a bit tricky without knowing N
        # Let's require them to be passed or infer from context
        raise ValueError("visible_nodes and hidden_nodes must be provided")

    N = len(visible_nodes)
    visible_block = Block(visible_nodes)

    # Create sampling schedule if not provided
    if schedule is None:
        schedule = SamplingSchedule(n_warmup=10000, n_samples=n_samples_per_epoch, steps_per_sample=50)

    # Initialize optimizer
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init((model.weights, model.biases))

    energy_history = []

    for epoch in range(n_epochs):
        # Split keys
        key, key_energy, key_grad = jax.random.split(key, 3)

        # Compute energy expectation and get samples
        mean_energy, energy_var, local_energies, samples = compute_vmc_energy(
            model,
            key_energy,
            n_samples_per_epoch,
            J,
            Gamma,
            schedule,
            visible_nodes,
            hidden_nodes,
            visible_block,
        )

        # Compute gradients using the same samples
        grad_w, grad_b = compute_vmc_gradients(model, samples, local_energies, mean_energy)

        # Update parameters
        updates, opt_state = optimizer.update(
            (grad_w, grad_b),
            opt_state,
            (model.weights, model.biases),
        )
        new_weights = optax.apply_updates(model.weights, updates[0])
        new_biases = optax.apply_updates(model.biases, updates[1])
        model = eqx.tree_at(lambda m: (m.weights, m.biases), model, (new_weights, new_biases))

        # Log and track
        energy_per_site = float(mean_energy / N)
        energy_history.append(energy_per_site)
        std_dev = sqrt(float(energy_var))

        # Compute gradient diagnostics
        grad_w_norm = float(jnp.linalg.norm(grad_w))
        grad_b_norm = float(jnp.linalg.norm(grad_b))
        grad_w_max = float(jnp.max(jnp.abs(grad_w)))
        grad_b_max = float(jnp.max(jnp.abs(grad_b)))

        # Parameter diagnostics
        weight_norm = float(jnp.linalg.norm(model.weights))
        bias_norm = float(jnp.linalg.norm(model.biases))
        weight_max = float(jnp.max(jnp.abs(model.weights)))
        bias_max = float(jnp.max(jnp.abs(model.biases)))

        print(
            f"Epoch {epoch+1}/{n_epochs}: E/N = {energy_per_site:.6f}, "
            f"Var = {energy_var:.6e}, Std = {std_dev:.6e}"
        )
        print(
            f"  Gradients: ||∇W|| = {grad_w_norm:.6e}, ||∇b|| = {grad_b_norm:.6e}, "
            f"max(|∇W|) = {grad_w_max:.6e}, max(|∇b|) = {grad_b_max:.6e}"
        )
        print(
            f"  Parameters: ||W|| = {weight_norm:.6f}, ||b|| = {bias_norm:.6f}, "
            f"max(|W|) = {weight_max:.6f}, max(|b|) = {bias_max:.6f}"
        )
        print()  # Blank line between epochs

        # Early stopping check
        # if len(energy_history) >= variance_window:
        #     recent_energies = energy_history[-variance_window:]
        #     recent_variance = jnp.var(jnp.array(recent_energies))
        #     if recent_variance < variance_threshold:
        #         print(f"Early stopping: variance {recent_variance:.6e} < threshold {variance_threshold}")
        #         break

    return model, energy_history


def main():
    """Main training script."""
    # Hyperparameters
    N = 32
    n_hidden = 64
    J = 1.0
    Gamma = 1.0
    beta = 1.0

    # Training hyperparameters
    n_epochs = 100
    n_samples_per_epoch = 4000
    learning_rate = 0.005
    variance_threshold = 1e-5
    variance_window = 5

    # Initialize
    key = jax.random.key(42)
    key, key_model = jax.random.split(key)
    model = create_tfim_rbm_model(N, n_hidden, beta, key_model)

    # Extract nodes for training
    visible_nodes = model.nodes[:N]
    hidden_nodes = model.nodes[N:]

    # Train
    print("Starting VMC training for 1D TFIM...")
    print(f"System size: N={N}, Hidden units: {n_hidden}")
    print(f"J={J}, Γ={Gamma}")
    expected_energy = -4 / jnp.pi
    print(f"Expected ground state energy per site: -4/π ≈ {expected_energy:.6f}\n")

    trained_model, energy_history = train_vmc_tfim(
        model,
        key,
        n_epochs,
        n_samples_per_epoch,
        J,
        Gamma,
        learning_rate,
        variance_threshold,
        variance_window,
        visible_nodes=visible_nodes,
        hidden_nodes=hidden_nodes,
    )

    # Final results
    final_energy_per_site = energy_history[-1]

    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Final E/N = {final_energy_per_site:.6f}")
    print(f"Expected -4/π = {expected_energy:.6f}")
    print(f"Error = {abs(final_energy_per_site - expected_energy):.6f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

