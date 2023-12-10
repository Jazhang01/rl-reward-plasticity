"""
Compute statistics for plasticity calculations.
"""

import jax
import jax.numpy as jnp
import jax.numpy.linalg

from jaxrl_m.common import TrainState
from jaxrl_m.dataset import ReplayBuffer
from jaxrl_m.typing import PRNGKey


def _gradients(
    rng: PRNGKey,
    agent: TrainState,
    batch,
):
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

    # compute gradient of the loss function with respect to the critic parameters
    grads, info = jax.grad(critic_loss_fn, has_aux=True)(agent.critic.params)

    return grads, info


def gradient_covariance(
    rng: PRNGKey, agent: TrainState, replay_buffer: ReplayBuffer, batch_size: int = 64
):
    """
    Compute covariance matrices for the gradients of the loss with respect to each parameter.
    """
    batch = replay_buffer.sample(batch_size)

    # convert dict of (batch, *) arrays into a batch-length list of dicts
    batch_list = [{k: arr[i] for k, arr in batch.items()} for i in range(batch_size)]

    grad_tree = []
    for batch_item in batch_list:
        grads, _ = _gradients(rng, agent, batch_item)

        # first dimension in each leaf is the ensemble size; we want to split this up
        unensemblized_grads = jax.tree_map(lambda arr: [ensemble_critic for ensemble_critic in arr], grads)
        grad_tree.append(unensemblized_grads)


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


    grad_cov = jax.tree_map(_compute_covariance, *grad_tree)

    return grad_cov


def weight_norm(rng: PRNGKey, critic: TrainState, replay_buffer: ReplayBuffer):
    """
    Compute the weight norm
    """
    pass
