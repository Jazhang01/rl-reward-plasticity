import os
import pickle
from functools import partial

import algorithms.rnd as rnd_learner
import algorithms.redq as sac_learner
import algorithms.dqn as dqn_learner
import gymnasium as gym
import jax
from jax.tree_util import tree_map
import numpy as np
import orbax.checkpoint
import tqdm
import wandb
from absl import app, flags
from config import WANDB_DEFAULT_CONFIG
from environments.maze.custom_maze import maze_map
from environments.maze.visualize import visual_evaluate
from environments.wrappers.antmaze import BoundedAntMaze, D4RLWrapper
from environments.wrappers.pointmaze import DiscretePointMaze
from environments.wrappers.reward import AddGaussianReward, ConstantReward
from environments.wrappers.observation import permute_observation
from flax.core.frozen_dict import freeze, unfreeze
from flax.training import checkpoints
from gymnasium.wrappers.filter_observation import FilterObservation
from gymnasium.wrappers.flatten_observation import FlattenObservation
from gymnasium.wrappers.transform_observation import TransformObservation
from gym.wrappers.transform_observation import TransformObservation as OTransformObservation
from ml_collections import config_flags

from jaxrl_m.dataset import ReplayBuffer
from jaxrl_m.evaluation import EpisodeMonitor, evaluate, flatten, supply_rng
from jaxrl_m.wandb import default_wandb_config, get_flag_dict, setup_wandb

FLAGS = flags.FLAGS
if __name__ == "__main__":
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
    flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
    flags.DEFINE_integer("start_steps", int(1e4), "Number of initial exploration steps.")
    flags.DEFINE_integer("utd_ratio", 1, "Update to data ratio.")

    flags.DEFINE_bool("use_eval_agent", True, "Train an agent on rewards.")
    flags.DEFINE_bool("use_explore_agent", False, "Train an agent on rewards and exploration bonus.")
    flags.DEFINE_bool("use_noise_agent", False, "Train an agent on rewards and noise.")

flags.DEFINE_integer(
    "rand_permutation_freq",
    0,
    "Frequency used to randomly permute the observations before passing through the critic network. Set to 0 to disable.",
)

wandb_config = default_wandb_config()
wandb_config.update(
    {
        **WANDB_DEFAULT_CONFIG,
        "group": "sac_test",
        "name": "sac_{env_name}",
    }
)

if __name__ == "__main__":
    config_flags.DEFINE_config_dict("wandb", wandb_config, lock_config=False)
    config_flags.DEFINE_config_dict(
        "sac_config", sac_learner.get_default_config(), lock_config=False
    )
    config_flags.DEFINE_config_dict(
        "dqn_config", dqn_learner.get_default_config(), lock_config=False
    )
    config_flags.DEFINE_config_dict(
        "rnd_config", rnd_learner.get_default_config(), lock_config=False
    )


def wrap_reward(env):
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
        wrapper = BoundedAntMaze
    elif 'PointMaze' in FLAGS.env_name:
        wrapper = partial(DiscretePointMaze, directions=FLAGS.dqn_config.num_actions)

    env = FlattenObservation(FilterObservation(env, filter_keys=filter_keys))
    env = wrapper(env)
    env = EpisodeMonitor(env)

    env = wrap_reward(env)

    if FLAGS.rand_permutation_freq != 0 and FLAGS.rand_permutation_freq is not None:
        wrapper = partial(
            TransformObservation, 
            f=partial(permute_observation, freq=FLAGS.rand_permutation_freq),
        )

        env = wrapper(env)

    return env
    

def create_d4rl_antmaze():
    import d4rl
    import gym as ogym

    env = ogym.make(FLAGS.env_name)

    if FLAGS.rand_permutation_freq != 0 and FLAGS.rand_permutation_freq is not None:
        wrapper = partial(
            OTransformObservation, 
            f=partial(permute_observation, freq=FLAGS.rand_permutation_freq),
        )
        env = wrapper(env)

    env = D4RLWrapper(env)
    env = EpisodeMonitor(env)

    env = wrap_reward(env)

    return env


def main(_):
    # Create wandb logger
    if 'antmaze' in FLAGS.env_name:
        learner = sac_learner
        learner_config = FLAGS.sac_config
    elif 'PointMaze' in FLAGS.env_name:
        learner = dqn_learner
        learner_config = FLAGS.dqn_config

    setup_wandb(learner_config.to_dict(), **FLAGS.wandb)

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
    
    if 'antmaze' in FLAGS.env_name:
        env = create_d4rl_antmaze()
        eval_env = create_d4rl_antmaze()
    else:
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

    rng, exploration_rng = jax.random.split(rng, 2)
    
    def make_agent():
        return learner.create_learner(
            FLAGS.seed,
            example_transition["observations"][None],
            example_transition["actions"][None],
            **learner_config,
        )

    agent_types = []
    if FLAGS.use_eval_agent:
        agent_types.append("eval")
    if FLAGS.use_explore_agent:
        agent_types.append("explore")
    if FLAGS.use_noise_agent:
        agent_types.append("noise")
    
    assert len(agent_types) > 0, "Need at least one agent."
    agents = {type: make_agent() for type in agent_types}
    
    rnd = rnd_learner.create_learner(
        FLAGS.seed,
        example_transition["observations"][None],
        example_transition["actions"][None],
        **FLAGS.rnd_config
    )

    if FLAGS.load_dir not in [None, "None"]:
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        init_agent = orbax_checkpointer.restore(FLAGS.load_dir, item=agents[agent_types[0]])
        
        # initialize critic params
        for agent_type, agent in agents.items():
            agents[agent_type] = agent.replace(
                critic=agent.critic.replace(params=init_agent.critic.params),
                target_critic=agent.target_critic.replace(params=init_agent.target_critic.params)
            )

    exploration_metrics = dict()
    obs, _ = env.reset()

    # rotate between agents when performing environment rollouts
    agent_id = 0

    for i in tqdm.tqdm(
        range(1, FLAGS.max_steps + 1), smoothing=0.1, dynamic_ncols=True
    ):
        if i < FLAGS.start_steps:
            action = env.action_space.sample()
        else:
            exploration_rng, key = jax.random.split(exploration_rng)
            agent = agents[agent_types[agent_id]]
            action = agent.sample_actions(obs, seed=key)

        next_obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        mask = float(not term)

        trans = dict(
            observations=obs,
            actions=action,
            rewards=reward,
            masks=mask,
            next_observations=next_obs,
        )

        replay_buffer.add_transition(trans)
        obs = next_obs

        if done:
            exploration_metrics = {
                f"exploration/{k}": v for k, v in flatten(info).items()
            }
            obs, _ = env.reset()
            agent_id = (agent_id + 1) % len(agents)

        if replay_buffer.size < FLAGS.start_steps:
            continue

        base_batch = replay_buffer.sample(FLAGS.batch_size * FLAGS.utd_ratio)
        update_info = {}
        for agent_type, agent in agents.items():
            batch = unfreeze(base_batch)
            if agent_type == 'explore':
                if FLAGS.reward_bonus == 'rnd':
                    batch['rewards'] += rnd.get_rewards(batch)
                else:
                    raise NotImplementedError
            elif agent_type == 'eval':
                pass
            elif agent_type == 'shift':
                batch['rewards'] += 1
            elif agent_type == 'noise':
                shape = batch['rewards'].shape
                mean, std = np.zeros(shape), np.ones(shape)
                batch['rewards'] += 0.01 * np.random.normal(mean, std)
            batch = freeze(batch)

            agent, info = agent.update(batch, utd_ratio=FLAGS.utd_ratio)
            agents[agent_type] = agent
            update_info.update({f'{agent_type}/{k}': v for k, v in info.items()})

        if replay_buffer.size >= FLAGS.start_steps and agent_types[agent_id] == 'explore':
            rnd, rnd_update_info = rnd.update(trans)
            update_info.update({f'rnd/{k}': v for k, v in rnd_update_info.items()})

        if i % FLAGS.log_interval == 0:
            train_metrics = {f"training/{k}": v for k, v in update_info.items()}
            wandb.log(train_metrics, step=i)
            wandb.log(exploration_metrics, step=i)
            exploration_metrics = dict()

        if i % FLAGS.eval_interval == 0:
            eval_metrics = {}
            for agent_type, agent in agents.items():
                policy_fn = partial(supply_rng(agent.sample_actions), temperature=0.0)
                info = visual_evaluate(policy_fn, eval_env, num_episodes=FLAGS.eval_episodes)
                eval_metrics.update({f'evaluation/{agent_type}/{k}': v for k, v in info.items()})
            wandb.log(eval_metrics, step=i)

        if i % FLAGS.save_interval == 0 and FLAGS.save_dir is not None:
            for agent_type, agent in agents.items():
                checkpoints.save_checkpoint(FLAGS.save_dir, agent, i, prefix=f'checkpoint_{agent_type}_', keep=10)
            buffer_path = os.path.join(FLAGS.save_dir, f'buffer_{i}.pkl')
            print(f"Saving replay buffer to {buffer_path}")
            with open(buffer_path, "wb") as f:
                pickle.dump(replay_buffer, f)


if __name__ == "__main__":
    app.run(main)
