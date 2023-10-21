import os
import pickle
from functools import partial

import algorithms.sac as learner
import gymnasium as gym
import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from config import WANDB_DEFAULT_CONFIG
from flax.training import checkpoints
from ml_collections import config_flags

from jaxrl_m.dataset import ReplayBuffer
from jaxrl_m.evaluation import EpisodeMonitor, evaluate, flatten, supply_rng
from jaxrl_m.wandb import default_wandb_config, get_flag_dict, setup_wandb

FLAGS = flags.FLAGS
flags.DEFINE_string("env_name", "HalfCheetah-v4", "Environment name.")
flags.DEFINE_string("save_dir", None, "Logging dir.")
flags.DEFINE_integer("seed", np.random.choice(1000000), "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 10000, "Eval interval.")
flags.DEFINE_integer("save_interval", 25000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
flags.DEFINE_integer("start_steps", int(1e4), "Number of initial exploration steps.")

wandb_config = default_wandb_config()
wandb_config.update(
    {
        **WANDB_DEFAULT_CONFIG,
        "group": "sac_test",
        "name": "sac_{env_name}",
    }
)

config_flags.DEFINE_config_dict("wandb", wandb_config, lock_config=False)
config_flags.DEFINE_config_dict(
    "config", learner.get_default_config(), lock_config=False
)


def main(_):
    # Create wandb logger
    setup_wandb(FLAGS.config.to_dict(), **FLAGS.wandb)

    if FLAGS.save_dir is not None:
        FLAGS.save_dir = os.path.join(
            FLAGS.save_dir,
            wandb.run.project,
            wandb.config.exp_prefix,
            wandb.config.experiment_id,
        )
        os.makedirs(FLAGS.save_dir, exist_ok=True)
        print(f"Saving config to {FLAGS.save_dir}/config.pkl")
        with open(os.path.join(FLAGS.save_dir, "config.pkl"), "wb") as f:
            pickle.dump(get_flag_dict(), f)

    env = EpisodeMonitor(gym.make(FLAGS.env_name))
    eval_env = EpisodeMonitor(gym.make(FLAGS.env_name))

    example_transition = dict(
        observations=env.observation_space.sample(),
        actions=env.action_space.sample(),
        rewards=0.0,
        masks=1.0,
        next_observations=env.observation_space.sample(),
    )

    replay_buffer = ReplayBuffer.create(example_transition, size=int(1e6))

    agent = learner.create_learner(
        FLAGS.seed,
        example_transition["observations"][None],
        example_transition["actions"][None],
        max_steps=FLAGS.max_steps,
        **FLAGS.config,
    )

    exploration_metrics = dict()
    obs, _ = env.reset()
    exploration_rng = jax.random.PRNGKey(0)

    for i in tqdm.tqdm(
        range(1, FLAGS.max_steps + 1), smoothing=0.1, dynamic_ncols=True
    ):
        if i < FLAGS.start_steps:
            action = env.action_space.sample()
        else:
            exploration_rng, key = jax.random.split(exploration_rng)
            action = agent.sample_actions(obs, seed=key)

        next_obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        mask = float(not term)

        replay_buffer.add_transition(
            dict(
                observations=obs,
                actions=action,
                rewards=reward,
                masks=mask,
                next_observations=next_obs,
            )
        )
        obs = next_obs

        if done:
            exploration_metrics = {
                f"exploration/{k}": v for k, v in flatten(info).items()
            }
            obs, _ = env.reset()

        if replay_buffer.size < FLAGS.start_steps:
            continue

        batch = replay_buffer.sample(FLAGS.batch_size)
        agent, update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            train_metrics = {f"training/{k}": v for k, v in update_info.items()}
            wandb.log(train_metrics, step=i)
            wandb.log(exploration_metrics, step=i)
            exploration_metrics = dict()

        if i % FLAGS.eval_interval == 0:
            policy_fn = partial(supply_rng(agent.sample_actions), temperature=0.0)
            eval_info = evaluate(policy_fn, eval_env, num_episodes=FLAGS.eval_episodes)

            eval_metrics = {f"evaluation/{k}": v for k, v in eval_info.items()}
            wandb.log(eval_metrics, step=i)

        if i % FLAGS.save_interval == 0 and FLAGS.save_dir is not None:
            checkpoints.save_checkpoint(FLAGS.save_dir, agent, i)


if __name__ == "__main__":
    app.run(main)
