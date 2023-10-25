from typing import Sequence

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
import tqdm
import wandb

from jaxrl_m.common import TrainState
from jaxrl_m.dataset import ReplayBuffer
from jaxrl_m.networks import ensemblize
from jaxrl_m.typing import Batch, PRNGKey


@jax.jit
def update_mse(
    critic: TrainState,
    target: TrainState,
    batch: Batch,
):
    def mse_loss_fn(params):
        obs, action = batch["observations"], batch["actions"]
        qs = critic(obs, action, params=params).min(axis=0)
        rand_qs = target(obs, action, params=target.params).min(axis=0)
        rand_qs = jax.lax.stop_gradient(rand_qs)

        loss = ((qs - rand_qs) ** 2).mean()

        info = {"loss": loss}
        return loss, info

    new_critic, info = critic.apply_loss_fn(loss_fn=mse_loss_fn, has_aux=True)
    return new_critic, info


@jax.jit
def group_update_mse(
    critics: Sequence[TrainState],
    targets: Sequence[TrainState],
    batch: Batch,
):
    """
    Batch update critics to match target networks.
    """
    num_copies = len(critics)
    assert num_copies == len(targets)

    update_infos = []
    new_critics = [None] * num_copies

    for critic_idx in range(num_copies):
        new_critics[critic_idx], update_info = update_mse(
            critics[critic_idx], targets[critic_idx], batch
        )
        update_infos.append(update_info)

    return new_critics, update_infos


@jax.jit
def update_mse_lyle(
    critic: TrainState,
    target: TrainState,
    batch: Batch,
    freq: float = 1e5,
    mean: float = 0.0,
):
    def mse_loss_fn(params):
        obs, action = batch["observations"], batch["actions"]
        qs = critic(obs, action, params=params).min(axis=0)
        rand_qs = target(obs, action, params=target.params).min(axis=0)

        # apply transformation to qs
        transformed_qs = mean + jnp.sin(freq * rand_qs)
        transformed_qs = jax.lax.stop_gradient(transformed_qs)

        loss = ((qs - transformed_qs) ** 2).mean()

        info = {"loss": loss, "rand_qs": rand_qs, "transformed_qs": transformed_qs}
        return loss, info

    new_critic, info = critic.apply_loss_fn(loss_fn=mse_loss_fn, has_aux=True)
    return new_critic, info


@jax.jit
def group_update_mse_lyle(
    critics: Sequence[TrainState],
    targets: Sequence[TrainState],
    means: Sequence[float],
    batch: Batch,
    freq: float = 1e5,
):
    """
    Batch update critics to match target networks.
    """
    num_copies = len(critics)
    assert num_copies == len(targets)

    update_infos = []
    new_critics = [None] * num_copies

    for critic_idx in range(num_copies):
        new_critics[critic_idx], update_info = update_mse_lyle(
            critics[critic_idx],
            targets[critic_idx],
            batch,
            freq=freq,
            mean=means[critic_idx],
        )
        update_infos.append(update_info)

    return new_critics, update_infos


def compute_q_plasticity(
    rng: PRNGKey,
    critic: TrainState,
    replay_buffer: ReplayBuffer,
    num_copies: int = 20,
    num_train_steps: int = 5000,
    batch_size: int = 256,
):
    """
    Compute the empirical plasticity of a train state using MSE loss on random targets

    Creates a randomly initialized Q-function, and attempts to train the current TrainState
    to match the random critics.
    The plasticity is then defined to be the negative mean of the resulting losses.
    """
    # split rng key
    rng_keys = jax.random.split(rng, num_copies)

    # create k new critics, initialized with random parameters
    batch = replay_buffer.sample(1)
    critic_def = critic.model_def

    rand_critic_params = [
        critic_def.init(cur_rng_key, batch["observations"], batch["actions"])["params"]
        for cur_rng_key in rng_keys
    ]
    rand_critics = [
        TrainState.create(critic_def, rand_critic_param)
        for rand_critic_param in rand_critic_params
    ]

    # create k copies of the given train state
    critics = [critic.replace() for _ in range(num_copies)]

    # estimate q means
    q_mean_batch = replay_buffer.sample(batch_size)
    q_means = []
    for critic in critics:
        qs = critic(q_mean_batch["observations"], q_mean_batch["actions"]).min(axis=0)
        q_means.append(qs.mean().item())

    # train the copied train states to match the k new critics
    update_infos = []
    for _ in tqdm.tqdm(
        range(num_train_steps), desc="Computing plasticity", leave=False
    ):
        batch = replay_buffer.sample(batch_size)
        critics, update_info = group_update_mse_lyle(
            critics, rand_critics, means=q_means, batch=batch, freq=10
        )
        update_infos.append(update_info)

    logged_items = update_infos[0][0].keys()

    aggregate_infos = {}
    for key in logged_items:
        aggregate_infos[key] = []

        for update_info in update_infos:
            total_value = sum([np.array(info[key]) for info in update_info])
            mean_value = total_value / num_copies
            aggregate_infos[key].append(mean_value)

    df = pd.DataFrame(data=aggregate_infos)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(df["loss"])

    return {"plasticity": -df["loss"].iloc[-1], "plasticity_loss": fig}
