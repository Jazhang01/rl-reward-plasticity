import os
import pickle

import algorithms.sac as learner
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

FLAGS = flags.FLAGS
if __name__ == "__main__":
    flags.DEFINE_string("env_name", None, "Environment name.")
    flags.DEFINE_integer("seed", 0, "Random seed.")
    flags.DEFINE_string(
        "checkpoints", None, "Path to a folder of checkpoints and saved replay buffers."
    )
    flags.DEFINE_string(
        "buffer",
        None,
        "Path to the final saved replay buffer, to be used in plasticity calculations.",
    )
    flags.DEFINE_integer("batch_size", 256, "Batch size.")

    flags.DEFINE_float(
        "rand_qs_weight", 10, "Weight for the random perturbation of the Q-values."
    )

    # flags from mujoco env
    flags.DEFINE_boolean(
        "invert_env", False, "Invert the objective of the environment."
    )
    flags.DEFINE_string("reward_wrapper", None, "Optional reward wrapper type to use.")


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
    final_dataset = None

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

    with open(FLAGS.buffer, "rb") as f:
        final_dataset_raw = pickle.load(f)
        final_dataset = ReplayBuffer.create_from_initial_dataset(
            final_dataset_raw, size=get_size(final_dataset_raw)
        )

    rng = jax.random.PRNGKey(seed=FLAGS.seed)
    rng, key = jax.random.split(rng, 2)

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
            key,
            agent.critic,
            final_dataset,
            batch_size=FLAGS.batch_size,
            rand_qs_weight=FLAGS.rand_qs_weight,
        )
        info.update({f"plasticity/{k}": v for k, v in plasticity_info.items()})

        # compute other statistics
        gradient_cov = plasticity.statistics.gradient_covariance(
            key,
            agent,
            checkpoint_dataset,
            batch_size=FLAGS.batch_size,
        )

        def get_path_key(key):
            if isinstance(key, DictKey):
                return key.key
            elif isinstance(key, SequenceKey):
                return str(key.idx)
            return ""

        grad_cov_flattened = {
            ".".join(get_path_key(k) for k in path): mat
            for path, mat in jax.tree_util.tree_leaves_with_path(gradient_cov)
        }

        # generate visualization for the gradient covariance matrix
        grad_cov_figs = {}
        grad_cov_ranks = {}
        for param, grad_cov_mat in grad_cov_flattened.items():
            grad_cov_fig = plt.figure(figsize=(8, 8))
            grad_cov_ax = grad_cov_fig.gca()
            grad_cov_fig_mat = grad_cov_ax.matshow(grad_cov_mat)
            grad_cov_fig.colorbar(grad_cov_fig_mat)

            grad_cov_figs[f"grad_cov_figs/{param}"] = grad_cov_fig
            grad_cov_ranks[f"grad_cov_ranks/{param}"] = jax.numpy.linalg.matrix_rank(
                grad_cov_mat
            )
        info.update(grad_cov_figs)
        info.update(grad_cov_ranks)

        wandb.log(info, step=step)

        # close all figures before moving on to the next step
        for fig in grad_cov_figs.values():
            plt.close(fig)

        del checkpoint_dataset_raw
        del checkpoint_dataset


if __name__ == "__main__":
    app.run(main)
