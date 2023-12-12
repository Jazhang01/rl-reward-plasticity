"""Implementations of algorithms for continuous control."""
import functools

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
import numpy as np
import optax

from jaxrl_m.common import TrainState, nonpytree_field, target_update
from jaxrl_m.networks import DiscreteCritic, ensemblize
from jaxrl_m.typing import *


class DQNAgent(flax.struct.PyTreeNode):
    rng: PRNGKey
    critic: TrainState
    target_critic: TrainState
    config: dict = nonpytree_field()

    @jax.jit
    def update_critic(agent, batch: Batch):
        def critic_loss_fn(critic_params):
            next_qa = agent.target_critic(batch['next_observations']).mean(axis=0)

            if agent.config['use_double_q']:
                critic_next_qa = agent.critic(batch['next_observations']).mean(axis=0)
                next_action = critic_next_qa.argmax(axis=-1)
            else:
                next_action = next_qa.argmax(axis=-1)

            next_q = jnp.take_along_axis(next_qa, next_action[..., None], axis=-1).squeeze()
            target_q = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_q

            qa = agent.critic(batch['observations'], params=critic_params).mean(axis=0)
            q = jnp.take_along_axis(qa, batch['actions'][...,None], axis=-1).squeeze()
            loss = ((q - target_q) ** 2).mean()

            return loss, {
                'critic_loss': loss,
                'q': q.mean(),
                "r": batch['rewards'].mean(),
                "masks": batch['masks'].mean(),
            }

        new_critic, critic_info = agent.critic.apply_loss_fn(
            loss_fn=critic_loss_fn, has_aux=True
        )
        new_target_critic = target_update(
            agent.critic, agent.target_critic, agent.config["target_update_rate"]
        )
    
        prefixed_critic_info = {f"critic/{k}": v for k, v in critic_info.items()}

        return agent.replace(rng=agent.rng, critic=new_critic, target_critic=new_target_critic), prefixed_critic_info
    

    @functools.partial(jax.jit, static_argnames="utd_ratio")
    def update(agent, batch: Batch, utd_ratio: int):
        batch = tree_map(lambda x: x.reshape(utd_ratio, x.shape[0] // utd_ratio, *x.shape[1:]), batch)
        
        for i in range(utd_ratio):
            mini_batch = tree_map(lambda x: x[i], batch)
            agent, critic_info = agent.update_critic(mini_batch)
        
        return agent, {**critic_info}
    

    def sample_actions(
        agent,
        observations: np.ndarray,
        *,
        seed: PRNGKey,
        temperature: float = 1.0,
    ) -> jnp.ndarray:
        random = (jax.random.uniform(seed) < agent.config['epsilon'] * temperature).item()
        return agent._sample_actions(observations, seed=seed, random=random)


    @functools.partial(jax.jit, static_argnames=('random',))
    def _sample_actions(
        agent,
        observations: np.ndarray,
        *,
        seed: PRNGKey,
        random: bool = False
    ) -> jnp.ndarray:
        qa = agent.critic(observations)
        action = qa.argmax(axis=-1).squeeze()

        num_actions = qa.shape[-1]
        if random:
            action = jax.random.randint(seed, (), 0, num_actions)

        return action


def create_learner(
    seed: int,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    num_actions: int,
    critic_lr: float = 3e-4,
    hidden_dims: Sequence[int] = (256, 256),
    discount: float = 0.99,
    tau: float = 0.005,
    use_layer_norm: bool = False,
    use_double_q: bool = True,
    epsilon: float = 0.05,
    **kwargs,
):
    print("Extra kwargs:", kwargs)

    rng = jax.random.PRNGKey(seed)
    rng, critic_key = jax.random.split(rng, 2)

    critic_activations = nn.relu
    if use_layer_norm:
        critic_activations = nn.Sequential([nn.LayerNorm(), nn.relu])

    critic_def = ensemblize(DiscreteCritic, num_qs=1)(hidden_dims, num_actions, critic_activations)
    critic_params = critic_def.init(critic_key, observations)["params"]
    critic = TrainState.create(
        critic_def, critic_params, tx=optax.adam(learning_rate=critic_lr)
    )
    target_critic = TrainState.create(critic_def, critic_params)

    config = flax.core.FrozenDict(
        dict(
            discount=discount,
            target_update_rate=tau,
            use_double_q=use_double_q,
            epsilon=epsilon,
        )
    )

    return DQNAgent(
        rng,
        critic=critic,
        target_critic=target_critic,
        config=config,
    )


def get_default_config():
    import ml_collections

    return ml_collections.ConfigDict(
        {
            "critic_lr": 3e-4,
            "hidden_dims": (256, 256),
            "discount": 0.99,
            "tau": 0.005,
            "use_layer_norm": False,
            "use_double_q": True,
            "num_actions": 8,
            "epsilon": 0.05
        }
    )
