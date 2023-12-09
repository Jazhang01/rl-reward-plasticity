from typing import Any, SupportsFloat

from gymnasium import Env, Wrapper


class BoundedAntMaze(Wrapper):
    def __init__(self, env: Env, z_thresh=2.0):
        super().__init__(env)

        self.z_thresh = z_thresh

    def step(self, action, **kwargs):
        obs, reward, term, trunc, info = super().step(action)

        z_coord = obs[2]
        if z_coord > self.z_thresh:
            trunc = True

        return obs, reward, term, trunc, info