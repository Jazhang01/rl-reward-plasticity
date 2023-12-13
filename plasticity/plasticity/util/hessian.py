from functools import partial 

import jax
import jax.numpy as jnp

from jaxrl_m.common import TrainState
from jaxrl_m.typing import Batch, PRNGKey

from .hessian_density.density import (
    eigv_to_density,
    tridiag_to_density,
    tridiag_to_eigv,
)
from .hessian_density.hessian_computation import get_hvp_fn
from .hessian_density.lanczos import lanczos_alg


def compute_hessian_density(rng: PRNGKey, agent: TrainState, batch: Batch, critic_type: str = "sac_critic"):
    """
    Compute an estimate of the hessian density.

    Returns:
        eig_vals: estimate of the eigenvalues of the Hessian matrix
        density: smoothed density estimate
        grids: values the density estimate is on
    """

    rng, critic_key, redq_key, lanczos_key = jax.random.split(rng, 4)

    @jax.jit
    def sac_critic_loss_fn(critic_params, loss_batch: Batch):
        """
        Compute the critic loss.

        TODO: this is copied directly from the sac agent; perhaps extract to avoid redundancy?
        """
        next_dist = agent.actor(loss_batch["next_observations"])
        next_actions, next_log_probs = next_dist.sample_and_log_prob(seed=critic_key)

        next_q1, next_q2 = agent.target_critic(loss_batch["next_observations"], next_actions)
        next_q = jnp.minimum(next_q1, next_q2)
        target_q = loss_batch["rewards"] + agent.config["discount"] * loss_batch["masks"] * next_q

        if agent.config["backup_entropy"]:
            target_q = (
                target_q
                - agent.config["discount"]
                * loss_batch["masks"]
                * next_log_probs
                * agent.temp()
            )

        q1, q2 = agent.critic(
            loss_batch["observations"], loss_batch["actions"], params=critic_params
        )
        critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()

        return critic_loss

    @jax.jit
    def dqn_critic_loss_fn(critic_params, loss_batch: Batch):
        next_qa = agent.target_critic(loss_batch['next_observations']).mean(axis=0)

        if agent.config['use_double_q']:
            critic_next_qa = agent.critic(loss_batch['next_observations']).mean(axis=0)
            next_action = critic_next_qa.argmax(axis=-1)
        else:
            next_action = next_qa.argmax(axis=-1)

        next_q = jnp.take_along_axis(next_qa, next_action[..., None], axis=-1).squeeze()
        target_q = loss_batch['rewards'] + agent.config['discount'] * loss_batch['masks'] * next_q

        qa = agent.critic(loss_batch['observations'], params=critic_params).mean(axis=0)
        q = jnp.take_along_axis(qa, loss_batch['actions'][...,None], axis=-1).squeeze()
        loss = ((q - target_q) ** 2).mean()

        return loss
    
    @jax.jit
    def redq_critic_loss_fn(critic_params, loss_batch: Batch):
        next_dist = agent.actor(loss_batch["next_observations"])
        next_actions, next_log_probs = next_dist.sample_and_log_prob(seed=critic_key)

        next_qs = agent.target_critic(loss_batch['next_observations'], next_actions)
        next_qs = jax.random.permutation(redq_key, next_qs, axis=0)[:agent.config['num_min_qs']]
        next_q = next_qs.min(axis=0)
        target_q = (
            loss_batch["rewards"] + agent.config["discount"] * loss_batch["masks"] * next_q
        )

        if agent.config["backup_entropy"]:
            target_q = (
                target_q
                - agent.config["discount"]
                * loss_batch["masks"]
                * next_log_probs
                * agent.temp()
            )

        qs = agent.critic(loss_batch['observations'], loss_batch['actions'], params=critic_params)
        critic_loss = ((qs - target_q) ** 2).mean()

        return critic_loss

    def batches():
        """Just one batch to feed into the Hessian vector product function"""
        yield batch

    if critic_type == 'dqn_critic':
        critic_loss_fn = dqn_critic_loss_fn
    elif critic_type == 'sac_critic':
        critic_loss_fn = sac_critic_loss_fn
    elif critic_type == 'redq_critic':
        critic_loss_fn = redq_critic_loss_fn
    else:
        raise NotImplementedError

    # compute the function v -> Hv
    hvp, _, num_params = get_hvp_fn(critic_loss_fn, agent.critic.params, batches)

    # match lanczos algorithm function
    hvp_cl = lambda x: hvp(agent.critic.params, x)

    tridiag, _ = lanczos_alg(hvp_cl, num_params, 128, lanczos_key)

    eig_vals, all_weights = tridiag_to_eigv([tridiag])
    density, grids = eigv_to_density(eig_vals, all_weights, sigma_squared=1e-5)

    return eig_vals, density, grids
