import os
import pickle

import algorithms.redq as learner
import gymnasium as gym
import jax
import jax.numpy.linalg
import matplotlib.pyplot as plt
import orbax.checkpoint
import tqdm
import wandb
from absl import app, flags
from config import WANDB_DEFAULT_CONFIG
from environments.maze.custom_maze import maze_map
from gymnasium.wrappers.filter_observation import FilterObservation
from gymnasium.wrappers.flatten_observation import FlattenObservation
from jax.tree_util import DictKey, SequenceKey
from ml_collections import config_flags
from run_plasticity_maze_sac import create_maze_env
from run_plasticity_mujoco_sac import create_mujoco_env

import plasticity.statistics
from jaxrl_m.dataset import Dataset, ReplayBuffer, get_size
from jaxrl_m.evaluation import EpisodeMonitor
from jaxrl_m.wandb import default_wandb_config, get_flag_dict, setup_wandb
from plasticity.plasticity import compute_q_plasticity

FLAGS = flags.FLAGS
if __name__ == "__main__":
    flags.DEFINE_string("env_name", None, "Environment name.")
    flags.DEFINE_integer("seed", 0, "Random seed.")
    flags.DEFINE_string(
        "checkpoints", None, "Path to a folder of checkpoints and saved replay buffers."
    )
    flags.DEFINE_integer("batch_size", 256, "Batch size.")

    flags.DEFINE_float(
        "rand_qs_weight", 10, "Weight for the random perturbation of the Q-values."
    )
    flags.DEFINE_integer(
        "random_buffer_size", int(1e5), "Buffer size to use in plasticity calculation."
    )

    # flags for statistics
    flags.DEFINE_integer(
        "gradient_cov_batch_size", 256, "Batch size for gradient covariance matrix."
    )
    flags.DEFINE_float(
        "dead_unit_threshold",
        1e-5,
        "Threshold for gradients in dead unit calculations.",
    )
    flags.DEFINE_integer(
        "dead_unit_batch_size", 1024, "Batch size for dead unit calculation."
    )
    flags.DEFINE_integer(
        "hessian_batch_size", 1024, "Batch size used for Hessian calculations."
    )

    # flags from mujoco env
    flags.DEFINE_boolean(
        "invert_env", False, "Invert the objective of the environment."
    )
    flags.DEFINE_string("reward_wrapper", None, "Optional reward wrapper type to use.")


wandb_config = default_wandb_config()
wandb_config.update(
    {
        **WANDB_DEFAULT_CONFIG,
        "group": "plasticity",
        "name": "plasticity_{env_name}",
    }
)

if __name__ == "__main__":
    config_flags.DEFINE_config_dict("wandb", wandb_config, lock_config=False)
    config_flags.DEFINE_config_dict(
        "config", learner.get_default_config(), lock_config=False
    )


def main(_):
    """
    takes in:
    - a path to a folder of checkpoints
    - a path to a replay buffer pickle

    returns:
    - plasticity of the Q function of each checkpoint on the data in the replay buffer
    - plots of everything
    """
    # Create wandb logger
    setup_wandb(FLAGS.config.to_dict(), **FLAGS.wandb)

    checkpoint_files = {}
    dataset_files = {}

    if FLAGS.env_name.startswith("PointMaze"):
        env = create_maze_env()
    elif FLAGS.env_name in {"Hopper-v4", "HalfCheetah-v4", "Humanoid-v4"}:
        env, _ = create_mujoco_env()
    else:
        raise ValueError(f"Invalid environment name: {FLAGS.env_name}")

    example_transition = dict(
        observations=env.observation_space.sample(),
        actions=env.action_space.sample(),
        rewards=0.0,
        masks=1.0,
        next_observations=env.observation_space.sample(),
    )

    agent = learner.create_learner(
        FLAGS.seed,
        example_transition["observations"][None],
        example_transition["actions"][None],
        **FLAGS.config,
    )

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    for filename in os.listdir(FLAGS.checkpoints):
        if filename.startswith("checkpoint"):
            path = os.path.join(FLAGS.checkpoints, filename)

            ckpt_info = filename.split("_")
            step = int(ckpt_info[-1])
            checkpoint_files[step] = path

        elif filename.startswith("buffer"):
            path = os.path.join(FLAGS.checkpoints, filename)

            filename_no_ext, _ = os.path.splitext(filename)
            ckpt_info = filename_no_ext.split("_")
            step = int(ckpt_info[-1])

            dataset_files[step] = path

    sorted_steps = sorted(checkpoint_files.keys())

    rng = jax.random.PRNGKey(seed=FLAGS.seed)
    rng, plasticity_key, statistics_key = jax.random.split(rng, 3)

    plasticity_dataset = ReplayBuffer.create(
        example_transition, size=FLAGS.random_buffer_size
    )
    for _ in tqdm.tqdm(range(FLAGS.random_buffer_size), desc="Generating random data"):
        rng, obs_key, next_obs_key, action_key = jax.random.split(rng, 4)
        obs = jax.random.normal(obs_key, env.observation_space.shape)
        next_obs = jax.random.normal(next_obs_key, env.observation_space.shape)
        action = jax.random.normal(action_key, env.action_space.shape)
        plasticity_dataset.add_transition(
            dict(
                observations=obs,
                actions=action,
                rewards=0.0,
                masks=1.0,
                next_observations=next_obs,
            )
        )

    # with open(FLAGS.buffer, "rb") as f:
    #     final_dataset_raw = pickle.load(f)
    #     final_dataset = ReplayBuffer.create_from_initial_dataset(
    #         final_dataset_raw, size=get_size(final_dataset_raw)
    #     )

    initial_dataset_filename = dataset_files[sorted_steps[0]]
    with open(initial_dataset_filename, "rb") as initial_dataset_file:
        initial_dataset_raw = pickle.load(initial_dataset_file)
        initial_dataset = ReplayBuffer.create_from_initial_dataset(
            initial_dataset_raw, size=get_size(initial_dataset_raw)
        )

    progress_bar = tqdm.tqdm(sorted_steps)
    for step in progress_bar:
        progress_bar.set_description(f"Step {step}")

        # restore agent
        checkpoint_filename = checkpoint_files[step]
        agent = orbax_checkpointer.restore(checkpoint_filename, item=agent)

        # restore dataset
        checkpoint_dataset_filename = dataset_files[step]

        info = {}

        with open(checkpoint_dataset_filename, "rb") as checkpoint_dataset_file:
            checkpoint_dataset_raw = pickle.load(checkpoint_dataset_file)
            checkpoint_dataset = ReplayBuffer.create_from_initial_dataset(
                checkpoint_dataset_raw, size=get_size(checkpoint_dataset_raw)
            )

        # compute plasticity
        plasticity_info = compute_q_plasticity(
            plasticity_key,
            agent.critic,
            replay_buffer=plasticity_dataset,
            checkpoint_replay_buffer=checkpoint_dataset,
            batch_size=FLAGS.batch_size,
            rand_qs_weight=FLAGS.rand_qs_weight,
        )
        info.update({f"plasticity/{k}": v for k, v in plasticity_info.items()})

        # compute other statistics
        statistics = plasticity.statistics.compute_statistics(
            statistics_key,
            agent,
            initial_replay_buffer=initial_dataset,
            replay_buffer=checkpoint_dataset,
            gradient_cov_batch_size=FLAGS.gradient_cov_batch_size,
            dead_unit_batch_size=FLAGS.dead_unit_batch_size,
            dead_unit_threshold=FLAGS.dead_unit_threshold,
            hessian_batch_size=FLAGS.hessian_batch_size,
        )

        num_dead_units = statistics["dead_unit_count"]
        info.update({f"dead_unit_count/{k}": v for k, v in num_dead_units.items()})

        gradient_cov = statistics["grad_cov"]

        # generate visualization for the gradient covariance matrix
        grad_cov_figs = {}
        grad_cov_ranks = {}
        for param, grad_cov_mat in gradient_cov.items():
            grad_cov_fig = plt.figure(figsize=(8, 8))
            grad_cov_ax = grad_cov_fig.gca()
            grad_cov_fig_mat = grad_cov_ax.matshow(grad_cov_mat, cmap="RdBu")
            grad_cov_fig.colorbar(grad_cov_fig_mat)

            grad_cov_figs[f"grad_cov_figs/{param}"] = grad_cov_fig
            grad_cov_ranks[f"grad_cov_ranks/{param}"] = jax.numpy.linalg.matrix_rank(
                grad_cov_mat
            )
        info.update(grad_cov_figs)
        info.update(grad_cov_ranks)

        # generate visualization of Hessian eigenvalues
        hessian_eigenvalues = statistics["hessian_eigenvalues"]
        hessian_eigvals_fig = plt.figure()
        hessian_eigvals_ax = hessian_eigvals_fig.gca()
        hessian_eigvals_ax.semilogy(
            hessian_eigenvalues["_density_grid"], hessian_eigenvalues["_density"]
        )
        info.update(
            {
                **{
                    f"hessian/{k}": v
                    for k, v in hessian_eigenvalues.items()
                    if not k.startswith("_")
                },
                "hessian/density": hessian_eigvals_fig,
            }
        )

        wandb.log(info, step=step)

        # close all figures before moving on to the next step
        for fig in grad_cov_figs.values():
            plt.close(fig)
        plt.close(hessian_eigvals_fig)

        del checkpoint_dataset_raw
        del checkpoint_dataset


if __name__ == "__main__":
    app.run(main)
