from typing import Any, SupportsFloat, Tuple, Union
import numpy as np
from gymnasium import Env, Wrapper
from gym import Wrapper as OWrapper


class BoundedAntMaze(Wrapper):
    def __init__(self, env: Env, z_thresh=2.0):
        super().__init__(env)

        self.z_thresh = z_thresh

    def step(self, action, **kwargs):
        action = action * 0.2 # * np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) 
        obs, reward, term, trunc, info = super().step(action)

        z_coord = obs[2]
        if z_coord > self.z_thresh:
            trunc = True
            # reward -= 100.0

        return obs, reward, term, trunc, info
    

class D4RLWrapper(OWrapper):
    def reset(self, **kwargs) -> Any | tuple[Any, dict]:
        obs = super().reset(**kwargs)
        return obs, {}
    
    def step(self, action: Any) -> Tuple[Any, float, bool, dict]:
        obs, reward, done, info = super().step(action)
        
        reward = reward - 1.0

        trunc = "TimeLimit.truncated" in info and info["TimeLimit.truncated"]
        term = done and not trunc

        return obs, reward, term, trunc, info
    
    def render(self):
        return super().render(mode='rgb_array')