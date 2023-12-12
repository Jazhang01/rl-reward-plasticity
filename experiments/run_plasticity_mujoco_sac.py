import os
import pickle
from functools import partial

import algorithms.redq as learner
import gymnasium as gym
import jax
import numpy as np
import orbax.checkpoint
import tqdm
import wandb
from absl import app, flags
from config import WANDB_DEFAULT_CONFIG
from environments.wrappers.reward import AddGaussianReward, ConstantReward
from flax.training import checkpoints
from ml_collections import config_flags

from jaxrl_m.dataset import ReplayBuffer
from jaxrl_m.evaluation import EpisodeMonitor, evaluate, flatten, supply_rng
from jaxrl_m.wandb import default_wandb_config, get_flag_dict, setup_wandb
from plasticity.plasticity import compute_q_plasticity

FLAGS = flags.FLAGS
flags.DEFINE_string("env_name", "HalfCheetah-v4", "Environment name.")
flags.DEFINE_boolean("invert_env", False, "Invert the objective of the environment.")
flags.DEFINE_string("reward_wrapper", None, "Optional reward wrapper type to use.")
flags.DEFINE_string("save_dir", None, "Logging dir.")
flags.DEFINE_string("load_dir", None, "Loading dir.")
flags.DEFINE_integer("seed", np.random.choice(1000000), "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 10000, "Eval interval.")
flags.DEFINE_integer("save_interval", 25000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("buffer_size", int(1e6), "Replay buffer size.")
flags.DEFINE_integer(
    "random_buffer_size", int(1e5), "Buffer size for use in plasticity calculation."
)
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
        # convert to absolute path
        FLAGS.save_dir = os.path.abspath(FLAGS.save_dir)

        os.makedirs(FLAGS.save_dir, exist_ok=True)
        print(f"Saving config to {FLAGS.save_dir}/config.pkl")
        with open(os.path.join(FLAGS.save_dir, "config.pkl"), "wb") as f:
            pickle.dump(get_flag_dict(), f)

    rng = jax.random.PRNGKey(FLAGS.seed)
    
    env_kwargs = {}
    if FLAGS.env_name == "Hopper-v4":
        env_kwargs['forward_reward_weight'] = 1.0
    elif FLAGS.env_name == "HalfCheetah-v4":
        env_kwargs['forward_reward_weight'] = 1.0
    elif FLAGS.env_name == "Humanoid-v4":
        env_kwargs['forward_reward_weight'] = 1.25
    
    if FLAGS.invert_env:
        if FLAGS.env_name in ["Hopper-v4", "HalfCheetah-v4", "Humanoid-v4"]:
            env_kwargs['forward_reward_weight'] *= -1.0

    env = EpisodeMonitor(gym.make(FLAGS.env_name, **env_kwargs))
    eval_env = EpisodeMonitor(gym.make(FLAGS.env_name, **env_kwargs))

    if FLAGS.reward_wrapper is not None:
        if FLAGS.reward_wrapper == "identity":
            wrapper = lambda x: x
        elif FLAGS.reward_wrapper == "zero":
            wrapper = partial(ConstantReward, reward=0.0)
        elif FLAGS.reward_wrapper == "one":
            wrapper = partial(ConstantReward, reward=1.0)
        elif FLAGS.reward_wrapper == "unit_noise":
            noise_rng, rng = jax.random.split(rng, 2)
            wrapper = partial(
                AddGaussianReward, rng=noise_rng, noise_mean=0.0, noise_std=1.0
            )
        else:
            raise NotImplementedError

        env = wrapper(env)
        eval_env = wrapper(env)

    example_transition = dict(
        observations=env.observation_space.sample(),
        actions=env.action_space.sample(),
        rewards=0.0,
        masks=1.0,
        next_observations=env.observation_space.sample(),
    )

    replay_buffer = ReplayBuffer.create(example_transition, size=FLAGS.buffer_size)

    exploration_rng, plasticity_rng = jax.random.split(rng, 2)

    plasticity_dataset = ReplayBuffer.create(
        example_transition, size=FLAGS.random_buffer_size
    )
    for _ in tqdm.tqdm(range(FLAGS.random_buffer_size), desc="Generating random data"):
        obs = env.observation_space.sample()
        next_obs = env.observation_space.sample()
        action = env.action_space.sample()
        plasticity_dataset.add_transition(
            dict(
                observations=obs,
                actions=action,
                rewards=0.0,
                masks=1.0,
                next_observations=next_obs,
            )
        )

    agent = learner.create_learner(
        FLAGS.seed,
        example_transition["observations"][None],
        example_transition["actions"][None],
        max_steps=FLAGS.max_steps,
        **FLAGS.config,
    )

    if FLAGS.load_dir not in [None, "None"]:
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        init_agent = orbax_checkpointer.restore(FLAGS.load_dir, item=agent)
        
        # initialize critic params
        agent.critic.replace(params=init_agent.critic.params)
        agent.target_critic.replace(params=init_agent.target_critic.params)


    exploration_metrics = dict()
    obs, _ = env.reset()

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

            # rng_key, plasticity_rng = jax.random.split(plasticity_rng, 2)
            # plasticity_info = compute_q_plasticity(
            #     rng_key, agent.critic, plasticity_dataset, batch_size=FLAGS.batch_size
            # )
            # eval_info.update(plasticity_info)

            eval_metrics = {f"evaluation/{k}": v for k, v in eval_info.items()}
            wandb.log(eval_metrics, step=i)

        if i % FLAGS.save_interval == 0 and FLAGS.save_dir is not None:
            checkpoints.save_checkpoint(FLAGS.save_dir, agent, i)
            print(f"Saving replay buffer to {FLAGS.save_dir}/buffer.pkl")
            with open(os.path.join(FLAGS.save_dir, "buffer.pkl"), "wb") as f:
                pickle.dump(replay_buffer, f)


if __name__ == "__main__":
    app.run(main)
