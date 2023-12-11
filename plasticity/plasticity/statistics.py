"""
Compute statistics for plasticity calculations.
"""

from typing import Sequence

import jax
import jax.numpy as jnp
import jax.numpy.linalg
import tqdm
from jax.tree_util import DictKey, SequenceKey

from jaxrl_m.common import TrainState
from jaxrl_m.dataset import ReplayBuffer
from jaxrl_m.typing import Batch, PRNGKey
from plasticity.util.grads import unbatched_grads
from plasticity.util.hessian import compute_hessian_density
from plasticity.util.kmeans import kmeans_jit


def order_batch_by_similarity(rng: PRNGKey, batch: Batch, k: int = 10):
    """
    Use k-means clustering on the batch of observations and actions,
    and return an index ordering to put the similar items together.
    """

    # combine observations and actions in the batch
    combined = jnp.concatenate([batch["observations"], batch["actions"]], axis=-1)

    kmeans_sol = kmeans_jit(rng, combined, k)

    sorted_batch_indices = jnp.argsort(kmeans_sol.assignment)

    return sorted_batch_indices


def gradient_covariance(rng: PRNGKey, batch: Batch, grads_tree):
    """
    Compute covariance matrices for the gradients of the loss with respect to each parameter.
    """

    # first, order the samples in the batch by similarity in observations and actions.
    batch_ordering = order_batch_by_similarity(rng, batch)
    grads_tree = [grads_tree[i] for i in batch_ordering]

    def _compute_covariance(*gradients):
        """
        Given a list of gradient arrays, flattens each gradient and computes the covariance matrix.

        That is, each (i, j) element is <X_i, X_j> / (||X_i|| ||X_j||)
        """

        grad_matrix = jnp.stack([grad.flatten() for grad in gradients])

        cov = grad_matrix @ grad_matrix.T
        norms = jax.numpy.linalg.norm(grad_matrix, axis=1)
        prod_norms = jnp.outer(norms, norms)
        return cov / (prod_norms + 1e-6)

    grad_cov = jax.tree_map(_compute_covariance, *grads_tree)

    return grad_cov


def dead_unit_count(grads_tree: Sequence[dict], threshold=1e-5):
    """
    Compute the number of dead units in the critic network.

    As a heuristic, this computes the number of columns in the gradient matrix that are all close to zero.
    """

    # grads_tree is a list of dictionaries, containing gradients with respect to each training point;
    # we want the maximum of all of the gradients for dead unit computation
    grads_max = jax.tree_map(
        lambda *arrays: jnp.max(jnp.stack(arrays), axis=0), *grads_tree
    )

    def _compute_dead_units(kernel, bias):
        """
        Given gradient matrices for the kernel and bias,
        comptue the number of columns in the kernelthe threshold in absolute value.
        """
        assert len(kernel.shape) == 2
        assert bias.shape[-1] == kernel.shape[-1]
        combined = jnp.concatenate(
            [
                kernel,
                bias.reshape(1, -1),
            ],
            axis=0,
        )
        combined_thresholded = jnp.abs(combined) < threshold

        # reshape to ensure that the result is a 2D matrix
        return jnp.sum(jnp.all(combined_thresholded, axis=0))

    # group up gradients by layer
    # assumes the dict is of the form {"network": {"layer_i": {"bias": [arr, arr, ...], "kernel": [arr, arr, ...]}}}
    num_dead_units = {}
    for network_key, layers in grads_max.items():
        layers_output = {}
        for layer_name, layer_parts in layers.items():
            if "bias" in layer_parts and "kernel" in layer_parts:
                bias_ensemble = layer_parts["bias"]
                kernel_ensemble = layer_parts["kernel"]

                # compute the dead units for each critic in the ensemble
                ensemble_dead_units = [
                    _compute_dead_units(kernel, bias)
                    for kernel, bias in zip(kernel_ensemble, bias_ensemble)
                ]
                layers_output[layer_name] = ensemble_dead_units
            else:
                # if not a dense layer, ignore it for dead unit calculations
                continue
        num_dead_units[network_key] = layers_output

    return num_dead_units


def hessian_eigenvals(rng: PRNGKey, agent: TrainState, batch: Batch):
    eig_vals, density, grids = compute_hessian_density(rng, agent, batch)

    max_eig_val = jnp.max(eig_vals)
    min_eig_val = jnp.min(eig_vals)

    condition_number = max_eig_val / min_eig_val

    return {
        "max_eigenvalue": max_eig_val,
        "min_eigenvalue": min_eig_val,
        "condition_number": condition_number,
        "_density": density,
        "_density_grid": grids,
    }


def get_path_key(key) -> str:
    if isinstance(key, DictKey):
        return str(key.key)
    if isinstance(key, SequenceKey):
        return str(key.idx)
    return ""


def flatten_tree(grads):
    return {
        ".".join(get_path_key(k) for k in path): mat
        for path, mat in jax.tree_util.tree_leaves_with_path(grads)
    }


def compute_statistics(
    rng: PRNGKey,
    agent: TrainState,
    initial_replay_buffer: ReplayBuffer,
    replay_buffer: ReplayBuffer,
    gradient_cov_batch_size: int = 256,
    hessian_batch_size: int = 1024,
    dead_unit_batch_size: int = 1024,
    dead_unit_threshold=1e-5,
):
    """
    Returns a nested dict, where the second level dictionary is a flattened dict containing all the relevant statistics.
    """
    grads_key, grad_cov_key, hessian_key = jax.random.split(rng, 3)

    gradient_cov_grads_batch = replay_buffer.sample(gradient_cov_batch_size)
    dead_unit_grads_batch = initial_replay_buffer.sample(dead_unit_batch_size)
    hessian_batch = replay_buffer.sample(hessian_batch_size)

    with tqdm.tqdm(total=4, leave=False) as progress_bar:
        progress_bar.set_description("Computing gradients")
        # compute batch gradients
        # grads, _ = _gradients(grads_key, agent, grads_batch)
        # compute individual gradients
        gradient_cov_grads_tree = unbatched_grads(
            grads_key, agent, gradient_cov_grads_batch, gradient_cov_batch_size
        )

        dead_unit_grads_tree = unbatched_grads(
            grads_key,
            agent,
            dead_unit_grads_batch,
            dead_unit_batch_size,
            loss_fn="identity",
        )
        progress_bar.update(1)

        progress_bar.set_description("Computing gradient covariance matrix")
        grad_cov = gradient_covariance(
            grad_cov_key, gradient_cov_grads_batch, gradient_cov_grads_tree
        )
        grad_cov_flattened = flatten_tree(grad_cov)
        progress_bar.update(1)

        progress_bar.set_description("Computing dead unit count")
        num_dead_units = dead_unit_count(
            dead_unit_grads_tree, threshold=dead_unit_threshold
        )
        num_dead_units_flattened = flatten_tree(num_dead_units)
        progress_bar.update(1)

        progress_bar.set_description("Computing Hessian eigenvalues")
        hessian_eigenvals_info = hessian_eigenvals(hessian_key, agent, hessian_batch)
        progress_bar.update(1)

    return {
        "grad_cov": grad_cov_flattened,
        "dead_unit_count": num_dead_units_flattened,
        "hessian_eigenvalues": hessian_eigenvals_info,
    }
