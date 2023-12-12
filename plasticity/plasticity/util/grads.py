from typing import Sequence

import jax
import jax.numpy as jnp
import jax.numpy.linalg
from jax.tree_util import DictKey, SequenceKey

from jaxrl_m.common import TrainState
from jaxrl_m.dataset import ReplayBuffer
from jaxrl_m.typing import Batch, PRNGKey
from plasticity.util.kmeans import kmeans_jit


def compute_gradients(
    rng: PRNGKey, agent: TrainState, batch: Batch, loss_fn: str = "critic"
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
        grads, _ = compute_gradients(grads_key, agent, batch_item, loss_fn=loss_fn)

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
