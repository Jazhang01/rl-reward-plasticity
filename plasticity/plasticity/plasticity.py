from typing import Sequence
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
import tqdm
import wandb
import wandb.plot

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


@partial(jax.jit, static_argnames=('critic_type',))
def update_mse_perturb(
    critic: TrainState,
    initial_critic: TrainState,
    rand_critic: TrainState,
    batch: Batch,
    rand_weight: float = 10,
    rand_qs_mean=0,
    rand_qs_std=1,
    critic_type: str = "sac_critic"
):
    """
    Updates the critic with randomly perturbed targets.
    """
    def mse_loss_fn(params):
        obs, action = batch["observations"], batch["actions"]
        qs = critic(obs, action, params=params).mean(axis=0)
        init_qs = initial_critic(obs, action, params=initial_critic.params).mean(axis=0)
        rand_qs = rand_critic(obs, action, params=rand_critic.params).mean(axis=0)
        normalized_rand_qs = (rand_qs - rand_qs_mean) / (rand_qs_std + 1e-6)

        # perturb_qs = init_qs + jnp.sin(freq * rand_qs)
        perturb_qs = init_qs + init_qs.std() * rand_weight * normalized_rand_qs
        # perturb_qs = init_qs + 10 * rand_qs
        perturb_qs = jax.lax.stop_gradient(perturb_qs)

        loss = ((qs - perturb_qs) ** 2).mean() / (init_qs.var() + 1e-6)
        # loss = ((qs - perturb_qs) ** 2).mean()

        info = {
            "loss": loss,
            "qs_mean": qs.mean(),
            "qs_std": qs.std(),
            "init_qs_mean": init_qs.mean(),
            "init_qs_std": init_qs.std(),
            "rand_qs_mean": rand_qs.mean(),
            "rand_qs_std": rand_qs.std(),
            "normalized_rand_qs_mean": normalized_rand_qs.mean(),
            "normalized_rand_qs_std": normalized_rand_qs.std(),
            "perturb_qs_mean": perturb_qs.mean(),
            "perturb_qs_std": perturb_qs.std(),
        }
        return loss, info
    
    def dqn_mse_loss_fn(params):
        obs, action = batch["observations"], batch["actions"]

        qa = critic(obs, params=params).mean(axis=0)
        qs = jnp.take_along_axis(qa, action[..., None], axis=-1).squeeze()

        init_qa = initial_critic(obs, params=initial_critic.params).mean(axis=0)
        init_qs = jnp.take_along_axis(init_qa, action[..., None], axis=-1).squeeze()

        rand_qa = rand_critic(obs, params=rand_critic.params).mean(axis=0)
        rand_qs = jnp.take_along_axis(rand_qa, action[..., None], axis=-1).squeeze()

        normalized_rand_qs = (rand_qs - rand_qs_mean) / (rand_qs_std + 1e-6)

        perturb_qs = init_qs + init_qs.std() * rand_weight * normalized_rand_qs
        perturb_qs = jax.lax.stop_gradient(perturb_qs)

        loss = ((qs - perturb_qs) ** 2).mean() / (init_qs.var() + 1e-6)

        info = {
            "loss": loss,
            "qs_mean": qs.mean(),
            "qs_std": qs.std(),
            "init_qs_mean": init_qs.mean(),
            "init_qs_std": init_qs.std(),
            "rand_qs_mean": rand_qs.mean(),
            "rand_qs_std": rand_qs.std(),
            "normalized_rand_qs_mean": normalized_rand_qs.mean(),
            "normalized_rand_qs_std": normalized_rand_qs.std(),
            "perturb_qs_mean": perturb_qs.mean(),
            "perturb_qs_std": perturb_qs.std(),
        }
        return loss, info
    
    if critic_type == 'dqn_critic':
        loss_fn = dqn_mse_loss_fn
    else:
        loss_fn = mse_loss_fn

    new_critic, info = critic.apply_loss_fn(loss_fn=loss_fn, has_aux=True)
    return new_critic, info


@partial(jax.jit, static_argnames=("critic_type",))
def group_update_mse_perturb(
    critics: Sequence[TrainState],
    initial_critics: Sequence[TrainState],
    rand_critics: Sequence[TrainState],
    batch: Batch,
    rand_weight: float = 10,
    rand_qs_mean=0,
    rand_qs_std=1,
    critic_type: str = "sac_critic"
):
    """
    Batch update critics to match target networks.
    """
    num_copies = len(critics)
    assert num_copies == len(rand_critics)

    update_infos = []
    new_critics = [None] * num_copies

    for critic_idx in range(num_copies):
        new_critics[critic_idx], update_info = update_mse_perturb(
            critics[critic_idx],
            initial_critics[critic_idx],
            rand_critics[critic_idx],
            batch,
            rand_weight=rand_weight,
            rand_qs_mean=rand_qs_mean,
            rand_qs_std=rand_qs_std,
            critic_type=critic_type
        )
        update_infos.append(update_info)

    return new_critics, update_infos


def compute_rand_qs_stats(
    replay_buffer: ReplayBuffer,
    rand_critics: Sequence[TrainState],
    buffer_stats,
    batch_size: int = 256,
    num_samples: int = 5000,
    critic_type: str = "sac_critic",
):
    """
    Samples from the replay buffer to get an estimate of the mean and standard deviation
    of the random critics.
    """

    @jax.jit
    def compute_rand_qs(obs, action):
        if critic_type == 'dqn_critic':
            qs = []
            for critic in rand_critics:
                qa = critic(obs, params=critic.params).mean(axis=0)
                q = jnp.take_along_axis(qa, action[..., None], axis=-1).squeeze()
                qs.append(q)
            return jnp.array(qs)
        else:
            return jnp.array(
                [critic(obs, action, params=critic.params) for critic in rand_critics]
            )

    total_mean = 0
    total_std = 0
    for _ in tqdm.trange(
        num_samples, desc="Sampling rand_qs for mean/std computation", leave=False
    ):
        batch = replay_buffer.sample(batch_size)
        batch = rescale_batch(batch, buffer_stats, critic_type=critic_type)

        obs, action = batch["observations"], batch["actions"]
        rand_qs = compute_rand_qs(obs, action)
        total_mean += jnp.mean(rand_qs)
        total_std += jnp.std(rand_qs)

    return total_mean / num_samples, total_std / num_samples


def compute_buffer_stats(
    replay_buffer: ReplayBuffer, batch_size: int = 256, num_samples: int = 5000
):
    """
    Samples the replay buffer to find the mean and standard deviation of the observations and actions.
    """

    @jax.jit
    def compute_stats(batch):
        obs, action = batch["observations"], batch["actions"]
        return {
            "obs": (obs.mean(), obs.std()),
            "acs": (action.mean(), action.std()),
        }

    obs_total_mean = 0
    obs_total_std = 0
    acs_total_mean = 0
    acs_total_std = 0
    for _ in tqdm.trange(
        num_samples, desc="Sampling replay buffer for mean/std computation", leave=False
    ):
        batch = replay_buffer.sample(batch_size)
        stats = compute_stats(batch)
        obs_total_mean += stats["obs"][0]
        obs_total_std += stats["obs"][1]
        acs_total_mean += stats["acs"][0]
        acs_total_std += stats["acs"][1]

    return {
        "obs": (obs_total_mean / num_samples, obs_total_std / num_samples),
        "acs": (acs_total_mean / num_samples, acs_total_std / num_samples),
    }


@partial(jax.jit, static_argnames=("critic_type",))
def rescale_batch(batch, buffer_stats, critic_type="sac_critic"):
    """
    Rescale a batch according to the buffer stats.
    """
    if buffer_stats is None:
        return

    obs, acs = batch["observations"], batch["actions"]

    # rescale obs and action based on buffer stats
    obs_mean, obs_std = buffer_stats["obs"]
    acs_mean, acs_std = buffer_stats["acs"]
    obs = obs * obs_std + obs_mean

    if critic_type != "dqn_critic":
        acs = acs * acs_std + acs_mean

    new_batch = {
        **batch,
        "observations": obs,
        "actions": acs
    }

    return new_batch


def compute_q_plasticity(
    rng: PRNGKey,
    critic: TrainState,
    replay_buffer: ReplayBuffer,
    checkpoint_replay_buffer: ReplayBuffer,
    num_copies: int = 20,
    num_train_steps: int = 2000,
    batch_size: int = 256,
    rand_qs_weight=10,
    critic_type: str = "sac_critic"
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

    if critic_type == 'dqn_critic':
        rand_critic_params = [
            critic_def.init(cur_rng_key, batch["observations"])["params"]
            for cur_rng_key in rng_keys
        ]    
    else:    
        rand_critic_params = [
            critic_def.init(cur_rng_key, batch["observations"], batch["actions"])["params"]
            for cur_rng_key in rng_keys
        ]
    rand_critics = [
        TrainState.create(critic_def, rand_critic_param)
        for rand_critic_param in rand_critic_params
    ]

    # create k copies of the given train state
    initial_critics = [critic.replace() for _ in range(num_copies)]
    critics = [
        TrainState.create(
            critic_def, critic.params.copy(), tx=optax.adam(learning_rate=3e-4)
        )
        for _ in range(num_copies)
    ]
    # critics = [
    #     TrainState.create(critic_def, critic.params.copy(), tx=optax.sgd(learning_rate=3e-4))
    #     for _ in range(num_copies)
    # ]

    # estimate q means
    # q_mean_batch = replay_buffer.sample(batch_size)
    # q_means = []
    # for critic in critics:
    #     qs = critic(q_mean_batch["observations"], q_mean_batch["actions"]).min(axis=0)
    #     q_means.append(qs.mean().item())

    buffer_stats = compute_buffer_stats(checkpoint_replay_buffer, batch_size=batch_size)

    # compute rand qs stats with the rescaled bufffer
    rand_qs_mean, rand_qs_std = compute_rand_qs_stats(
        replay_buffer, rand_critics, batch_size=batch_size, buffer_stats=buffer_stats, critic_type=critic_type
    )

    # get random perturbation to observations
    # obs_perturb = jax.random.normal(rand_key, batch["observations"][0].shape)

    # train the copied train states to match the k new critics
    update_infos = []
    for _ in tqdm.tqdm(
        range(num_train_steps), desc="Computing plasticity", leave=False
    ):
        batch = replay_buffer.sample(batch_size)
        batch = rescale_batch(batch, buffer_stats, critic_type=critic_type)
        # critics, update_info = group_update_mse_lyle(
        #     critics, rand_critics, means=q_means, batch=batch, freq=10
        # )
        critics, update_info = group_update_mse_perturb(
            critics,
            initial_critics,
            rand_critics,
            batch,
            rand_weight=rand_qs_weight,
            rand_qs_mean=rand_qs_mean,
            rand_qs_std=rand_qs_std,
            critic_type=critic_type
        )
        update_infos.append(update_info)

    logged_items = update_infos[0][0].keys()

    aggregate_infos = {}
    for key in tqdm.tqdm(logged_items, desc="Logging items", leave=False):
        aggregate_infos[key] = []

        for update_info in update_infos:
            total_value = sum(np.array(info[key]) for info in update_info)
            mean_value = total_value / num_copies
            aggregate_infos[key].append(mean_value)

    df = pd.DataFrame(data=aggregate_infos)

    plasticity_fig = plt.figure()
    plasticity_ax = plasticity_fig.add_subplot()
    plasticity_ax.plot(df["loss"])

    grad_norm_keys = [key for key in logged_items if key.startswith("grad_norm/")]
    grad_norm_fig = None
    if grad_norm_keys:
        grad_norm_fig = plt.figure()
        grad_norm_ax = grad_norm_fig.add_subplot()
        grad_norm_ax.plot(df[grad_norm_keys].mean(axis=1))

    return {
        # plasticity statistics
        "plasticity": -df["loss"].iloc[-100:].mean(),
        "plasticity_loss": plasticity_fig,
        # learned Q-values statistics
        "qs_mean": df["qs_mean"].mean(),
        "qs_std": df["qs_std"].mean(),
        # initial Q-values statistics
        "initial_qs_mean": df["init_qs_mean"].mean(),
        "initial_qs_std": df["init_qs_std"].mean(),
        # random Q-values statistics
        "rand_qs_dataset_mean": rand_qs_mean,
        "rand_qs_dataset_std": rand_qs_std,
        "rand_qs_mean": df["rand_qs_mean"].mean(),
        "rand_qs_std": df["rand_qs_std"].mean(),
        "normalized_rand_qs_mean": df["normalized_rand_qs_mean"].mean(),
        "normalized_rand_qs_std": df["normalized_rand_qs_std"].mean(),
        # statistics of Q-values after perturbation
        "perturb_qs_mean": df["perturb_qs_mean"].mean(),
        "perturb_qs_std": df["perturb_qs_std"].mean(),
        # buffer statistics
        "buffer_obs_mean": buffer_stats["obs"][0],
        "buffer_obs_std": buffer_stats["obs"][1],
        "buffer_acs_mean": buffer_stats["acs"][0],
        "buffer_acs_std": buffer_stats["acs"][1],
        # gradient norm figure
        "grad_norm": grad_norm_fig,
    }
