import os
import pickle
from functools import partial

import algorithms.rnd as rnd_learner
import algorithms.sac as learner
import gymnasium as gym
import jax
import numpy as np
import orbax.checkpoint
import tqdm
import wandb
from absl import app, flags
from config import WANDB_DEFAULT_CONFIG
from environments.maze.custom_maze import maze_map
from environments.maze.visualize import visual_evaluate
from environments.wrappers.antmaze import BoundedAntMaze
from environments.wrappers.reward import AddGaussianReward, ConstantReward
from flax.core.frozen_dict import freeze, unfreeze
from flax.training import checkpoints
from gymnasium.wrappers.filter_observation import FilterObservation
from gymnasium.wrappers.flatten_observation import FlattenObservation
from ml_collections import config_flags

from jaxrl_m.dataset import ReplayBuffer
from jaxrl_m.evaluation import EpisodeMonitor, evaluate, flatten, supply_rng
from jaxrl_m.wandb import default_wandb_config, get_flag_dict, setup_wandb
from plasticity.plasticity import compute_q_plasticity

FLAGS = flags.FLAGS
flags.DEFINE_string("env_name", "PointMaze_UMaze-v3", "Environment name.")
flags.DEFINE_string("reward_wrapper", None, "Optional reward wrapper type to use.")
flags.DEFINE_string("reward_bonus", None, "Reward bonus type.")
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


def create_maze_env():
    env_kwargs = {
        'render_mode': 'rgb_array',
        'maze_map': maze_map[FLAGS.env_name],
    }
    env = gym.make(FLAGS.env_name, **env_kwargs)
    
    filter_keys = ['observation']
    if 'AntMaze' in FLAGS.env_name:
        """
        The achieved goal contains the (x, y) position which is useful for:
        - debugging and visualization
        - allowing the agent to know its location in the maze (e.g. for location based reward bonuses)
        """
        filter_keys.append('achieved_goal')  

    env = FlattenObservation(FilterObservation(env, filter_keys=filter_keys))
    env = BoundedAntMaze(env)
    env = EpisodeMonitor(env)

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

    return env
    

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

    rng = jax.random.PRNGKey(FLAGS.seed)
    
    env = create_maze_env()
    eval_env = create_maze_env()

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
    
    explore_agent = learner.create_learner(
        FLAGS.seed,
        example_transition["observations"][None],
        example_transition["actions"][None],
        max_steps=FLAGS.max_steps,
        **FLAGS.config,
    )
    evaluate_agent = learner.create_learner(
        FLAGS.seed,
        example_transition["observations"][None],
        example_transition["actions"][None],
        max_steps=FLAGS.max_steps,
        **FLAGS.config,
    )

    rnd = rnd_learner.create_learner(
        FLAGS.seed,
        example_transition["observations"][None],
        example_transition["actions"][None],
        max_steps=FLAGS.max_steps,
        # TODO: rnd config
    )

    if FLAGS.load_dir not in [None, "None"]:
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        init_agent = orbax_checkpointer.restore(FLAGS.load_dir, item=explore_agent)
        
        # initialize critic params
        explore_agent.critic.replace(params=init_agent.critic.params)
        explore_agent.target_critic.replace(params=init_agent.target_critic.params)


    exploration_metrics = dict()
    obs, _ = env.reset()

    explore_rollout = True
    for i in tqdm.tqdm(
        range(1, FLAGS.max_steps + 1), smoothing=0.1, dynamic_ncols=True
    ):
        if i < FLAGS.start_steps:
            action = env.action_space.sample()
        else:
            exploration_rng, key = jax.random.split(exploration_rng)
            if explore_rollout:
                action = explore_agent.sample_actions(obs, seed=key)
            else:
                action = evaluate_agent.sample_actions(obs, seed=key)

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
            explore_rollout = not explore_rollout

        if replay_buffer.size < FLAGS.start_steps:
            continue

        batch = replay_buffer.sample(FLAGS.batch_size)
        evaluate_agent, update_info = evaluate_agent.update(batch)
        
        if FLAGS.reward_bonus == 'rnd':
            batch = unfreeze(batch)
            batch['rewards'] += rnd.get_rewards(batch)
            batch = freeze(batch)

        explore_agent, update_info = explore_agent.update(batch)
        rnd, rnd_update_info = rnd.update(batch)

        if i % FLAGS.log_interval == 0:
            train_metrics = {f"training/{k}": v for k, v in update_info.items()}
            rnd_train_metrics = {f"training/rnd/{k}": v for k, v in rnd_update_info.items()}
            wandb.log(train_metrics, step=i)
            wandb.log(rnd_train_metrics, step=i)
            wandb.log(exploration_metrics, step=i)
            exploration_metrics = dict()

        if i % FLAGS.eval_interval == 0:
            evaluate_policy_fn = partial(supply_rng(evaluate_agent.sample_actions), temperature=0.0)
            eval_info = visual_evaluate(evaluate_policy_fn, eval_env, num_episodes=FLAGS.eval_episodes)

            explore_policy_fn = partial(supply_rng(explore_agent.sample_actions), temperature=0.0)
            eval_explore_info = visual_evaluate(explore_policy_fn, eval_env, num_episodes=FLAGS.eval_episodes)
            
            # rng_key, plasticity_rng = jax.random.split(plasticity_rng, 2)
            # plasticity_info = compute_q_plasticity(
            #     rng_key, agent.critic, plasticity_dataset, batch_size=FLAGS.batch_size
            # )
            # eval_info.update(plasticity_info)

            eval_metrics = {f"evaluation/evaluate/{k}": v for k, v in eval_info.items()}
            eval_metrics.update({f"evaluation/explore/{k}": v for k, v in eval_explore_info.items()})
            wandb.log(eval_metrics, step=i)

        if i % FLAGS.save_interval == 0 and FLAGS.save_dir is not None:
            checkpoints.save_checkpoint(FLAGS.save_dir, explore_agent, i, prefix="checkpoint_explore_", keep=10)
            checkpoints.save_checkpoint(FLAGS.save_dir, evaluate_agent, i, prefix="checkpoint_evaluate_", keep=10)
            print(f"Saving replay buffer to {FLAGS.save_dir}/buffer.pkl")
            with open(os.path.join(FLAGS.save_dir, "buffer.pkl"), "wb") as f:
                pickle.dump(replay_buffer, f)


if __name__ == "__main__":
    app.run(main)
