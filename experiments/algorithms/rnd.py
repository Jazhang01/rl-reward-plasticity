import flax
import jax
import jax.numpy as jnp
import optax

from jaxrl_m.common import TrainState
from jaxrl_m.networks import MLP
from jaxrl_m.typing import *


class RND(flax.struct.PyTreeNode):
    rng: PRNGKey
    network: TrainState
    frozen_network: TrainState
    rnd_coeff: float

    @jax.jit
    def update(agent, batch: Batch):
        def loss(params):
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

        obs_act = jnp.concatenate([batch['observations'], batch['actions']], axis=-1)

        phi1 = agent.network(obs_act)
        phi2 = agent.frozen_network(obs_act)
        raw = ((phi1 - phi2) ** 2).mean(axis=-1)

        max_r = raw.max()
        min_r = raw.min()
        rewards = (raw - min_r) / (max_r - min_r + 1e-6)
        return agent.rnd_coeff * rewards


def create_learner(
    seed: int,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    learning_rate: float = 3e-4,
    hidden_dims: Sequence[int] = (256, 256),
    latent_dim: int = 128,
    rnd_coeff: float = 1.0,
    **kwargs
):
    print("Extra kwargs:", kwargs)

    rng = jax.random.PRNGKey(seed)
    key1, key2 = jax.random.split(rng, 2)

    network_def = MLP((*hidden_dims, latent_dim))

    obs_act = jnp.concatenate([observations, actions], axis=-1)

    network_params = network_def.init(key1, obs_act)["params"]
    network = TrainState.create(
        network_def, network_params, tx=optax.adam(learning_rate=learning_rate)
    )

    frozen_params = network_def.init(key2, obs_act)["params"]
    frozen = TrainState.create(
        network_def, frozen_params, tx=optax.adam(learning_rate=learning_rate)
    )

    return RND(rng, network=network, frozen_network=frozen, rnd_coeff=rnd_coeff)
