"""Regression: early-finished lanes must not consume later LIBERO initial states."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import gymnasium as gym
import numpy as np

from lerobot.envs import libero
from lerobot.envs.utils import NEW_ROLLOUT_OPTION


class CountingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, lane):
        self.observation_space = gym.spaces.Box(-100, 100, (1,), np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (7,), np.float32)
        self.next_state = lane
        self.duration = [1, 4, 2, 5][lane]
        self.visited = []
        self.total_steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.visited.append(self.next_state)
        self.next_state += 4
        self.elapsed = 0
        return np.array([self.visited[-1]], np.float32), {}

    def step(self, action):
        self.elapsed += 1
        self.total_steps += 1
        return np.array([self.visited[-1]], np.float32), 0., self.elapsed >= self.duration, False, {}


class TestSyncInitialStates(unittest.TestCase):
    def test_five_batches_use_exactly_states_zero_to_nineteen(self):
        fns = [lambda lane=i: CountingEnv(lane) for i in range(4)]
        with patch.object(libero, "_get_suite", return_value=SimpleNamespace(tasks=[None])), \
                patch.object(libero, "_make_env_fns", return_value=fns):
            env = libero.create_libero_envs(
                task="libero_spatial", n_envs=4, env_cls=gym.vector.SyncVectorEnv,
                gym_kwargs={"task_ids": [0]},
            )["libero_spatial"][0]
        try:
            for batch in range(5):
                env.reset(seed=list(range(2000 + batch * 4, 2004 + batch * 4)),
                          options={NEW_ROLLOUT_OPTION: True})
                done = np.zeros(4, bool)
                while not done.all():
                    _, _, terminated, truncated, _ = env.step(np.zeros((4, 7), np.float32))
                    done |= terminated | truncated
            for lane, wrapped in enumerate(env.envs):
                raw = wrapped.unwrapped
                self.assertEqual(raw.visited, list(range(lane, 20, 4)))
                self.assertEqual(raw.total_steps, raw.duration * 5)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
