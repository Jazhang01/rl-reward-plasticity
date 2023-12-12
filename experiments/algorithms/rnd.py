from functools import partial

import flax
import jax
import jax.numpy as jnp
import optax

from jaxrl_m.common import nonpytree_field, TrainState
from jaxrl_m.networks import MLP
from jaxrl_m.typing import *


class RND(flax.struct.PyTreeNode):
    rng: PRNGKey
    network: TrainState
    frozen_network: TrainState
    config: dict = nonpytree_field()

    @jax.jit
    def update(agent, batch: Batch):
        def loss(params):
            obs_act = batch['observations']
            if agent.config['use_actions']:
                obs_act = jnp.concatenate([batch['observations'], batch['actions']], axis=-1)
            
            phi1 = agent.network(obs_act, params=params)
            phi2 = agent.frozen_network(obs_act)
            
            loss = ((phi1 - phi2) ** 2).mean()
            info = {'loss': loss}
            return loss, info
        
        new_network, info = agent.network.apply_loss_fn(loss_fn=loss, has_aux=True)
        return agent.replace(network=new_network), info

    @jax.jit
    def get_rewards(agent, batch: Batch):
        obs_act = batch['observations']
        if agent.config['use_actions']:
            obs_act = jnp.concatenate([batch['observations'], batch['actions']], axis=-1)

        phi1 = agent.network(obs_act)
        phi2 = agent.frozen_network(obs_act)
        rewards = ((phi1 - phi2) ** 2).mean(axis=-1)

        if agent.config['normalize_rewards']:
            min_r = rewards.min()
            max_r = rewards.max()
            rewards = (rewards - min_r) / (max_r - min_r + 1e-6)

        return agent.config['rnd_coeff'] * rewards


def create_learner(
    seed: int,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    learning_rate: float = 3e-4,
    hidden_dims: Sequence[int] = (256, 256),
    latent_dim: int = 256,
    rnd_coeff: float = 1.0,
    normalize_rewards: bool = False,
    use_actions: bool = True,
    **kwargs
):
    print("Extra kwargs:", kwargs)

    rng = jax.random.PRNGKey(seed)
    key1, key2 = jax.random.split(rng, 2)

    network_def = MLP((*hidden_dims, latent_dim))

    obs_act = observations
    if use_actions:
        obs_act = jnp.concatenate([observations, actions], axis=-1)

    network_params = network_def.init(key1, obs_act)["params"]
    network = TrainState.create(
        network_def, network_params, tx=optax.adam(learning_rate=learning_rate)
    )

    frozen_params = network_def.init(key2, obs_act)["params"]
    frozen = TrainState.create(
        network_def, frozen_params, tx=optax.adam(learning_rate=learning_rate)
    )

    config = dict(
        rnd_coeff=rnd_coeff,
        normalize_rewards=normalize_rewards,
        use_actions=use_actions
    )

    return RND(rng, network=network, frozen_network=frozen, config=config)


def get_default_config():
    import ml_collections

    return ml_collections.ConfigDict(
        {
            "learning_rate": 3e-4,
            "hidden_dims": (256, 256),
            "latent_dim": 256,
            "rnd_coeff": 1.0,
            "normalize_rewards": False,
            "use_actions": True
        }
    )
