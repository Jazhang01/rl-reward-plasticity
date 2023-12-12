from typing import Any, SupportsFloat
from gymnasium.core import Env, Wrapper
from gymnasium.spaces import Discrete
import numpy as np


class DiscretePointMaze(Wrapper):
    def __init__(self, env: Env, directions=8, scale=1.0):
        super().__init__(env)
        assert 0.0 < scale <= 1.0

        self.scale = scale
        self.directions = directions
        self.action_space = Discrete(directions)

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        assert action in range(self.directions)
        
        theta = 2.0 * np.pi * action / self.directions
        rot = np.array(
            [[np.cos(theta), -np.sin(theta)],
             [np.sin(theta), np.cos(theta)]]
        )

        action_vec = np.array([1, 0])
        action_vec = self.scale * (rot @ action_vec)

        return super().step(action_vec)
