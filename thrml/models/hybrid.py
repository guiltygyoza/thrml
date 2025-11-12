import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, Key

from thrml.block_management import Block
from thrml.block_sampling import (
    BlockGibbsSpec,
    BlockSamplingProgram,
    SamplingSchedule,
    SuperBlock,
    sample_with_observation,
)
from thrml.factor import FactorSamplingProgram
from thrml.models.discrete_ebm import CategoricalEBMFactor, CategoricalGibbsConditional, DiscreteEBMFactor, SpinEBMFactor, SpinGibbsConditional
from thrml.models.ebm import AbstractFactorizedEBM, EBMFactor, FactorizedEBM
from thrml.observers import StateObserver
from thrml.pgm import AbstractNode, CategoricalNode, DEFAULT_NODE_SHAPE_DTYPES, SpinNode

# Type aliases for moment specifications
CategoricalMoment = tuple[CategoricalNode, int]  # (node, category_index)
SpinMoment = SpinNode  # Just the node
CatCatMoment = tuple[CategoricalNode, int, CategoricalNode, int]  # (node1, cat1, node2, cat2)
CatSpinMoment = tuple[CategoricalNode, int, SpinNode]  # (cat_node, category, spin_node)
SpinSpinMoment = tuple[SpinNode, SpinNode]  # (spin_node1, spin_node2)


class HybridEBM(AbstractFactorizedEBM):
    """An EBM with both categorical and spin nodes.

    This wraps a FactorizedEBM and provides a convenient interface for models
    with mixed node types, similar to IsingEBM but supporting categorical visible nodes.

    **Attributes:**

    - `ebm`: The underlying FactorizedEBM
    - `categorical_nodes`: List of categorical nodes
    - `spin_nodes`: List of spin nodes
    - `n_categories_per_node`: Map from categorical node to number of categories
    - `beta`: Temperature parameter
    """

    ebm: FactorizedEBM
    categorical_nodes: list[CategoricalNode]
    spin_nodes: list[SpinNode]
    n_categories_per_node: dict[CategoricalNode, int]
    beta: Array

    def __init__(
        self,
        ebm: FactorizedEBM,
        categorical_nodes: list[CategoricalNode],
        spin_nodes: list[SpinNode],
        n_categories_per_node: dict[CategoricalNode, int],
        beta: Array,
    ):
        """Initialize a Hybrid EBM.

        **Arguments:**

        - `ebm`: The underlying FactorizedEBM
        - `categorical_nodes`: List of categorical nodes
        - `spin_nodes`: List of spin nodes
        - `n_categories_per_node`: Map from categorical node to number of categories
        - `beta`: Temperature parameter
        """
        # Build node_shape_dtypes from both node types
        sd_map = {
            CategoricalNode: DEFAULT_NODE_SHAPE_DTYPES[CategoricalNode],
            SpinNode: DEFAULT_NODE_SHAPE_DTYPES[SpinNode],
        }
        super().__init__(sd_map)

        self.ebm = ebm
        self.categorical_nodes = categorical_nodes
        self.spin_nodes = spin_nodes
        self.n_categories_per_node = n_categories_per_node
        self.beta = beta

    @property
    def factors(self) -> list[EBMFactor]:
        return self.ebm.factors


class HybridSamplingProgram(FactorSamplingProgram):
    """Sampling program for Hybrid EBM with mixed node types."""

    def __init__(
        self,
        ebm: HybridEBM,
        free_blocks: list[SuperBlock],
        clamped_blocks: list[Block],
    ):
        """Initialize hybrid sampling program.

        **Arguments:**

        - `ebm`: The Hybrid EBM
        - `free_blocks`: SuperBlocks that are free to vary
        - `clamped_blocks`: Blocks that are clamped
        """
        # Determine samplers for each free block based on node type
        samplers = []
        for super_block in free_blocks:
            # Handle both Block and tuple of Blocks
            if isinstance(super_block, Block):
                blocks_to_check = [super_block]
            else:
                blocks_to_check = list(super_block)

            # Use first block to determine sampler type (all blocks in superblock should be same type)
            if len(blocks_to_check) == 0 or len(blocks_to_check[0]) == 0:
                # Empty block - use spin sampler as default
                samplers.append(SpinGibbsConditional())
            elif isinstance(blocks_to_check[0].nodes[0], CategoricalNode):
                n_categories = ebm.n_categories_per_node[blocks_to_check[0].nodes[0]]
                samplers.append(CategoricalGibbsConditional(n_categories=n_categories))
            elif isinstance(blocks_to_check[0].nodes[0], SpinNode):
                samplers.append(SpinGibbsConditional())
            else:
                raise RuntimeError(f"Unknown node type: {type(blocks_to_check[0].nodes[0])}")

        spec = BlockGibbsSpec(free_blocks, clamped_blocks, ebm.node_shape_dtypes)

        super().__init__(spec, samplers, ebm.ebm.factors, [])


class HybridTrainingSpec(eqx.Module):
    """Contains a complete specification of a Hybrid EBM that can be trained using sampling-based gradients.

    Defines sampling programs and schedules that allow for collection of the positive and negative phase samples
    required for Monte Carlo estimation of the gradient of the KL-divergence between the model and a data distribution.
    """

    ebm: HybridEBM
    program_positive: HybridSamplingProgram
    program_negative: HybridSamplingProgram
    schedule_positive: SamplingSchedule
    schedule_negative: SamplingSchedule

    def __init__(
        self,
        ebm: HybridEBM,
        data_blocks: list[Block],
        conditioning_blocks: list[Block],
        positive_sampling_blocks: list[SuperBlock],
        negative_sampling_blocks: list[SuperBlock],
        schedule_positive: SamplingSchedule,
        schedule_negative: SamplingSchedule,
    ):
        self.ebm = ebm

        self.program_positive = HybridSamplingProgram(ebm, positive_sampling_blocks, data_blocks + conditioning_blocks)
        self.program_negative = HybridSamplingProgram(ebm, negative_sampling_blocks, conditioning_blocks)

        self.schedule_positive = schedule_positive
        self.schedule_negative = schedule_negative


def estimate_moments_hybrid(
    key: Key[Array, ""],
    categorical_first_moments: list[CategoricalMoment],  # [(node, category), ...]
    spin_first_moments: list[SpinNode],  # [node, ...]
    categorical_categorical_edges: list[CatCatMoment],  # [(node1, cat1, node2, cat2), ...]
    categorical_spin_edges: list[CatSpinMoment],  # [(cat_node, category, spin_node), ...]
    spin_spin_edges: list[SpinSpinMoment],  # [(spin_node1, spin_node2), ...]
    program: BlockSamplingProgram,
    schedule: SamplingSchedule,
    init_state: list[Array],
    clamped_data: list[Array],
    n_categories_map: dict[CategoricalNode, int],
) -> tuple[Array, Array, Array, Array, Array]:
    """Estimate moments for hybrid model with both categorical and spin nodes.

    **Arguments:**

    - `key`: JAX random key
    - `categorical_first_moments`: List of (categorical_node, category_index) tuples for first moments
    - `spin_first_moments`: List of spin nodes for first moments
    - `categorical_categorical_edges`: List of (cat_node1, cat1, cat_node2, cat2) tuples
    - `categorical_spin_edges`: List of (cat_node, category, spin_node) tuples
    - `spin_spin_edges`: List of (spin_node1, spin_node2) tuples
    - `program`: The BlockSamplingProgram to use for sampling
    - `schedule`: Sampling schedule
    - `init_state`: Initial state for sampling
    - `clamped_data`: Clamped data values
    - `n_categories_map`: Map from categorical node to number of categories

    **Returns:**

    Tuple of (cat_first_moments, spin_first_moments, cat_cat_moments, cat_spin_moments, spin_spin_moments)
    """
    # Use StateObserver to collect samples
    state_observer = StateObserver(program.gibbs_spec.free_blocks)
    init_mem = state_observer.init()

    # Sample states
    _, samples = sample_with_observation(key, program, schedule, init_state, clamped_data, init_mem, state_observer)

    # Build node-to-value maps from samples
    # samples is a list matching free_blocks, each element has shape (n_samples, n_steps, n_nodes)
    node_values = {}
    for i, block in enumerate(program.gibbs_spec.free_blocks):
        if len(block) == 0:
            continue
        sample_block = samples[i]  # Shape: (n_samples, n_steps, n_nodes)
        # Use last step
        sample_last = sample_block[:, -1, :]  # Shape: (n_samples, n_nodes)
        for j, node in enumerate(block.nodes):
            node_values[node] = sample_last[:, j]

    # Compute categorical first moments: ⟨1_{c_i = k}⟩
    cat_first = []
    for node, category in categorical_first_moments:
        if node in node_values:
            indicators = (node_values[node] == category).astype(float)
            cat_first.append(jnp.mean(indicators))
        else:
            cat_first.append(0.0)
    cat_first = jnp.array(cat_first)

    # Compute spin first moments: ⟨s_i⟩
    spin_first = []
    for node in spin_first_moments:
        if node in node_values:
            spin_vals = 2 * node_values[node].astype(jnp.int8) - 1  # Convert to {-1,+1}
            spin_first.append(jnp.mean(spin_vals))
        else:
            spin_first.append(0.0)
    spin_first = jnp.array(spin_first)

    # Compute categorical-categorical second moments: ⟨1_{c_i = k} * 1_{c_j = l}⟩
    cat_cat = []
    for node1, cat1, node2, cat2 in categorical_categorical_edges:
        if node1 in node_values and node2 in node_values:
            ind1 = (node_values[node1] == cat1).astype(float)
            ind2 = (node_values[node2] == cat2).astype(float)
            cat_cat.append(jnp.mean(ind1 * ind2))
        else:
            cat_cat.append(0.0)
    cat_cat = jnp.array(cat_cat)

    # Compute categorical-spin second moments: ⟨1_{c_i = k} * s_j⟩
    cat_spin = []
    for cat_node, category, spin_node in categorical_spin_edges:
        if cat_node in node_values and spin_node in node_values:
            ind = (node_values[cat_node] == category).astype(float)
            spin_val = 2 * node_values[spin_node].astype(jnp.int8) - 1
            cat_spin.append(jnp.mean(ind * spin_val))
        else:
            cat_spin.append(0.0)
    cat_spin = jnp.array(cat_spin)

    # Compute spin-spin second moments: ⟨s_i * s_j⟩
    spin_spin = []
    for node1, node2 in spin_spin_edges:
        if node1 in node_values and node2 in node_values:
            spin1 = 2 * node_values[node1].astype(jnp.int8) - 1
            spin2 = 2 * node_values[node2].astype(jnp.int8) - 1
            spin_spin.append(jnp.mean(spin1 * spin2))
        else:
            spin_spin.append(0.0)
    spin_spin = jnp.array(spin_spin)

    return cat_first, spin_first, cat_cat, cat_spin, spin_spin


def estimate_kl_grad_hybrid(
    key: Key[Array, ""],
    training_spec: HybridTrainingSpec,
    categorical_bias_moments: list[CategoricalMoment],  # All (node, category) pairs for bias gradients
    spin_bias_nodes: list[SpinNode],  # Nodes for spin bias gradients
    categorical_categorical_weight_edges: list[CatCatMoment],
    categorical_spin_weight_edges: list[CatSpinMoment],
    spin_spin_weight_edges: list[SpinSpinMoment],
    data: list[Array],  # Categorical data arrays
    conditioning_values: list[Array],
    init_state_positive: list[Array],
    init_state_negative: list[Array],
) -> tuple:
    """Estimate KL gradients for hybrid model.

    Uses the standard two-term Monte Carlo estimator of the gradient of the KL-divergence between
    a hybrid model and a data distribution.

    The gradients are:

    - Categorical bias: $-\beta (\langle 1_{c_i=k} \rangle_{+} - \langle 1_{c_i=k} \rangle_{-})$
    - Spin bias: $-\beta (\langle s_i \rangle_{+} - \langle s_i \rangle_{-})$
    - Cat-cat weights: $-\beta (\langle 1_{c_i=k} 1_{c_j=l} \rangle_{+} - \langle 1_{c_i=k} 1_{c_j=l} \rangle_{-})$
    - Cat-spin weights: $-\beta (\langle 1_{c_i=k} s_j \rangle_{+} - \langle 1_{c_i=k} s_j \rangle_{-})$
    - Spin-spin weights: $-\beta (\langle s_i s_j \rangle_{+} - \langle s_i s_j \rangle_{-})$

    Here, $\langle\cdot\rangle_{+}$ denotes an expectation under the *positive* phase
    (data-clamped Boltzmann distribution) and $\langle\cdot\rangle_{-}$ under the *negative* phase
    (model distribution).

    **Arguments:**

    - `key`: JAX random key
    - `training_spec`: The HybridTrainingSpec for which to estimate gradients
    - `categorical_bias_moments`: List of (categorical_node, category) pairs for bias gradients
    - `spin_bias_nodes`: List of spin nodes for bias gradients
    - `categorical_categorical_weight_edges`: List of (cat_node1, cat1, cat_node2, cat2) tuples
    - `categorical_spin_weight_edges`: List of (cat_node, category, spin_node) tuples
    - `spin_spin_weight_edges`: List of (spin_node1, spin_node2) tuples
    - `data`: Data values for positive phase, each array has shape [batch, nodes]
    - `conditioning_values`: Values to assign to conditioning nodes, each array has shape [nodes]
    - `init_state_positive`: Initial state for positive sampling, each array has shape [n_chains_pos, batch, nodes]
    - `init_state_negative`: Initial state for negative sampling, each array has shape [n_chains_neg, nodes]

    **Returns:**

    Tuple of (grad_cat_bias, grad_spin_bias, grad_cat_cat_w, grad_cat_spin_w, grad_spin_spin_w, moments_pos, moments_neg)
    """
    key_pos, key_neg = jax.random.split(key, 2)

    n_categories_map = training_spec.ebm.n_categories_per_node

    # Positive phase
    cond_batched_pos = jax.tree.map(lambda x: jnp.broadcast_to(x, (data[0].shape[0], *x.shape)), conditioning_values)

    keys_pos = jax.random.split(key_pos, init_state_positive[0].shape[:2])

    def compute_moments_pos(k_out, i_out):
        return jax.vmap(
            lambda k, i, c: estimate_moments_hybrid(
                k,
                categorical_bias_moments,
                spin_bias_nodes,
                categorical_categorical_weight_edges,
                categorical_spin_weight_edges,
                spin_spin_weight_edges,
                training_spec.program_positive,
                training_spec.schedule_positive,
                i,
                c + data,
                n_categories_map,
            )
        )(k_out, i_out, cond_batched_pos)

    moms_cat_b_pos, moms_spin_b_pos, moms_cat_cat_pos, moms_cat_spin_pos, moms_spin_spin_pos = jax.vmap(
        compute_moments_pos
    )(keys_pos, init_state_positive)

    # Negative phase
    keys_neg = jax.random.split(key_neg, init_state_negative[0].shape[0])

    moms_cat_b_neg, moms_spin_b_neg, moms_cat_cat_neg, moms_cat_spin_neg, moms_spin_spin_neg = jax.vmap(
        lambda k, *i: estimate_moments_hybrid(
            k,
            categorical_bias_moments,
            spin_bias_nodes,
            categorical_categorical_weight_edges,
            categorical_spin_weight_edges,
            spin_spin_weight_edges,
            training_spec.program_negative,
            training_spec.schedule_negative,
            list(i),
            conditioning_values,
            n_categories_map,
        ),
        in_axes=(0, *[0 for _ in init_state_negative])
    )(keys_neg, *init_state_negative)

    # Compute gradients: -β * (positive_moment - negative_moment)
    float_type = training_spec.ebm.beta.dtype
    beta = training_spec.ebm.beta

    grad_cat_bias = -beta * (
        jnp.mean(moms_cat_b_pos, axis=(0, 1), dtype=float_type) - jnp.mean(moms_cat_b_neg, axis=0, dtype=float_type)
    )
    grad_spin_bias = -beta * (
        jnp.mean(moms_spin_b_pos, axis=(0, 1), dtype=float_type) - jnp.mean(moms_spin_b_neg, axis=0, dtype=float_type)
    )
    grad_cat_cat_w = -beta * (
        jnp.mean(moms_cat_cat_pos, axis=(0, 1), dtype=float_type) - jnp.mean(moms_cat_cat_neg, axis=0, dtype=float_type)
    )
    grad_cat_spin_w = -beta * (
        jnp.mean(moms_cat_spin_pos, axis=(0, 1), dtype=float_type) - jnp.mean(moms_cat_spin_neg, axis=0, dtype=float_type)
    )
    grad_spin_spin_w = -beta * (
        jnp.mean(moms_spin_spin_pos, axis=(0, 1), dtype=float_type) - jnp.mean(moms_spin_spin_neg, axis=0, dtype=float_type)
    )

    moments_pos = (moms_cat_b_pos, moms_spin_b_pos, moms_cat_cat_pos, moms_cat_spin_pos, moms_spin_spin_pos)
    moments_neg = (moms_cat_b_neg, moms_spin_b_neg, moms_cat_cat_neg, moms_cat_spin_neg, moms_spin_spin_neg)

    return (
        grad_cat_bias,
        grad_spin_bias,
        grad_cat_cat_w,
        grad_cat_spin_w,
        grad_spin_spin_w,
        moments_pos,
        moments_neg,
    )

