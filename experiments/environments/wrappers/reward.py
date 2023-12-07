import jax
from gymnasium.core import Env, RewardWrapper

from jaxrl_m.typing import PRNGKey


class ConstantReward(RewardWrapper):
    def __init__(self, env: Env, reward: float = 0.0):
        super().__init__(env)
        self.constant_reward = reward

    def reward(self, reward: float) -> float:
        return self.constant_reward


class AddGaussianReward(RewardWrapper):
    def __init__(
        self, env: Env, rng: PRNGKey, noise_mean: float = 0.0, noise_std: float = 1.0
    ):
        super().__init__(env)

        self.rng = rng
        self.noise_mean = noise_mean
        self.noise_std = noise_std

    def reward(self, reward: float) -> float:
        noise_key, self.rng = jax.random.split(self.rng, 2)
        noise = jax.random.normal(noise_key)
        noise = float(noise * self.noise_std + self.noise_mean)
        return reward + noise


class ShiftScaleReward(RewardWrapper):
    def __init__(self, env: Env, shift: float = 0.0, scale: float = 0.0):
        super().__init__(env)

        self.shift = shift
        self.scale = scale
    
    def reward(self, reward: float) -> float:
        return (reward + self.shift) * self.scale
