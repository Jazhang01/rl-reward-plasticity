import os
import pickle

import algorithms.redq as learner
import gymnasium as gym
import jax
import orbax.checkpoint
import wandb
from absl import app, flags
from config import WANDB_DEFAULT_CONFIG
from environments.maze.custom_maze import maze_map
from gymnasium.wrappers.filter_observation import FilterObservation
from gymnasium.wrappers.flatten_observation import FlattenObservation
from ml_collections import config_flags
from run_plasticity_maze_sac import create_maze_env

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

config_flags.DEFINE_config_dict("wandb", wandb_config, lock_config=False)
config_flags.DEFINE_config_dict(
    "config", learner.get_default_config(), lock_config=False
)

FLAGS = flags.FLAGS
flags.DEFINE_string("env_name", None, "Environment name.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string("checkpoints", None, "Path to a folder of checkpoints.")
flags.DEFINE_string("buffer", None, "Path to a saved replay buffer.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")

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

    checkpoints = []
    dataset = None
    
    env = create_maze_env()

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
    for ckpt in os.listdir(FLAGS.checkpoints):
        if ckpt.startswith("checkpoint"):
            path = os.path.join(FLAGS.checkpoints, ckpt)
            agent = orbax_checkpointer.restore(path, item=agent)
            
            ckpt_info = ckpt.split('_')
            step = int(ckpt_info[-1])
            name = ''
            if len(ckpt_info) > 2:
                name = ckpt_info[1]

            checkpoints.append((step, name, agent))

    checkpoints = sorted(checkpoints, key=lambda ckpt: ckpt[0])

    with open(FLAGS.buffer, 'rb') as f:
        dataset = pickle.load(f)
        dataset = ReplayBuffer.create_from_initial_dataset(dataset, size=get_size(dataset))

    rng = jax.random.PRNGKey(seed=FLAGS.seed)
    rng, key = jax.random.split(rng, 2)

    for step, name, agent in checkpoints:
        info = compute_q_plasticity(
            key, agent.critic, dataset, batch_size=FLAGS.batch_size
        )
        
        wandb.log({f'{name}/{k}': v for k, v in info.items()}, step=step)


if __name__ == "__main__":
    app.run(main)