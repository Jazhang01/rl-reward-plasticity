"""
Compute statistics for plasticity calculations.
"""

from typing import Sequence

import jax
import jax.numpy as jnp
import jax.numpy.linalg
from jax.tree_util import DictKey, SequenceKey

from jaxrl_m.common import TrainState
from jaxrl_m.dataset import ReplayBuffer
from jaxrl_m.typing import Batch, PRNGKey
from plasticity.util.kmeans import kmeans_jit


def _gradients(rng: PRNGKey, agent: TrainState, batch: Batch, loss_fn: str = "critic"):
    """
    Compute gradients of a critic using the given batch.

    Parameters
    ----------
    rng : PRNGKey
        Key for randomness

    agent : TrainState
        Agent containing the critic to compute the gradients for.
        The agent's actor network is used to determine the targets for gradient calculations.

    batch
        Batch to take the gradients with.
    """

    def critic_loss_fn(critic_params):
        """
        Compute the critic loss.

        TODO: this is copied directly from the sac agent; perhaps extract to avoid redundancy?
        """
        next_dist = agent.actor(batch["next_observations"])
        next_actions, next_log_probs = next_dist.sample_and_log_prob(seed=rng)

        next_q1, next_q2 = agent.target_critic(batch["next_observations"], next_actions)
        next_q = jnp.minimum(next_q1, next_q2)
        target_q = batch["rewards"] + agent.config["discount"] * batch["masks"] * next_q

        if agent.config["backup_entropy"]:
            target_q = (
                target_q
                - agent.config["discount"]
                * batch["masks"]
                * next_log_probs
                * agent.temp()
            )

        q1, q2 = agent.critic(
            batch["observations"], batch["actions"], params=critic_params
        )
        critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()

        return critic_loss, {
            "critic_loss": critic_loss.mean(),
            "q1": q1.mean(),
            "q2": q2.mean(),
        }

    def identity_loss_fn(critic_params):
        """
        Return the Q-values as the identity loss function for gradient calculations.
        """
        q1, q2 = agent.critic(
            batch["observations"], batch["actions"], params=critic_params
        )
        identity_loss = (q1 + q2).mean()
        return identity_loss, {
            "critic_loss": identity_loss.mean(),
            "q1": q1.mean(),
            "q2": q2.mean(),
        }

    # compute gradient of the loss function with respect to the critic parameters
    if loss_fn == "critic":
        grads, info = jax.grad(critic_loss_fn, has_aux=True)(agent.critic.params)
    elif loss_fn == "identity":
        grads, info = jax.grad(identity_loss_fn, has_aux=True)(agent.critic.params)
    else:
        raise ValueError("Invalid loss function")

    return grads, info


def unbatched_grads(
    rng: PRNGKey, agent: TrainState, batch: Batch, batch_size: int, loss_fn="critic"
):
    """
    Convert dict of (batch, *) arrays into a batch-length list of dicts.
    Also unensemblizes all of the gradients (i.e. the leaves do not have an ensemble dimension).
    """

    # convert dict of (batch, *) arrays into a batch-length list of dicts
    batch_list = [{k: arr[i] for k, arr in batch.items()} for i in range(batch_size)]

    grad_tree = []
    for batch_item in batch_list:
        rng, grads_key = jax.random.split(rng, 2)
        grads, _ = _gradients(grads_key, agent, batch_item, loss_fn=loss_fn)

        # first dimension in each leaf is the ensemble size; we want to split this up
        unensemblized_grads = unensemblize_grads(grads)
        grad_tree.append(unensemblized_grads)

    return grad_tree


def unensemblize_grads(grads):
    """
    By default, the gradients are grouped with the first dimension being the ensemble size.
    This function extracts the ensembles out into a list, allowing for tree_map to act on each critic individually.
    """
    return jax.tree_map(lambda arr: [ensemble_critic for ensemble_critic in arr], grads)


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
    dead_unit_batch_size: int = 1024,
    dead_unit_threshold=1e-5,
):
    """
    Returns a nested dict, where the second level dictionary is a flattened dict containing all the relevant statistics.
    """
    grads_key, grad_cov_key = jax.random.split(rng, 2)

    gradient_cov_grads_batch = replay_buffer.sample(gradient_cov_batch_size)
    dead_unit_grads_batch = initial_replay_buffer.sample(dead_unit_batch_size)

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

    grad_cov = gradient_covariance(
        grad_cov_key, gradient_cov_grads_batch, gradient_cov_grads_tree
    )
    grad_cov_flattened = flatten_tree(grad_cov)

    num_dead_units = dead_unit_count(
        dead_unit_grads_tree, threshold=dead_unit_threshold
    )
    num_dead_units_flattened = flatten_tree(num_dead_units)

    return {"grad_cov": grad_cov_flattened, "dead_unit_count": num_dead_units_flattened}
