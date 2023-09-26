import numpy as np
import gymnasium as gym


class RandomWalkEnv(gym.Env):
    metadata = { "render_modes": [] }

    SLOW = 0
    FAST = 1

    def __init__(
        self, render_mode=None,
        obs_low=[0., 0.], obs_high=[1.2, 1.2], unlucky=False
    ):
        self.observation_space = gym.spaces.Box(
            low=np.array(obs_low, dtype=np.float32),
            high=np.array(obs_high, dtype=np.float32)
        )
        self.action_space = gym.spaces.Discrete(2)

        self.x_start = 0.
        self.t_start = 0.

        self.x_max = 1.
        self.t_max = 1.

        # the changes in x and t depending on action (fast or slow)
        self.dx_fast = 0.17
        self.dx_slow = 0.1
        self.dt_fast = 0.05
        self.dt_slow = 0.12

        # randomness
        self.eps = 0.04
        self.unlucky = unlucky

    def _get_obs(self):
        return np.array([self.x, self.t], dtype=np.float32)

    def _get_info(self):
        return {}

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)

        self.x = self.x_start
        self.t = self.t_start

        return self._get_obs(), self._get_info()

    def is_safe(self, s):
        x, t = s
        return t < self.t_max

    def allowed_actions(self, s):
        x, t = s
        if x < self.x_max and t < self.t_max:
            return [0, 1]
        else:
            return []

    def step_from(self, s, a):
        return self._step(s, a)

    def step(self, action):
        (x, t), reward, terminated = self._step(self._get_obs(), action)

        self.x = x
        self.t = t

        return self._get_obs(), reward, terminated, False, self._get_info()

    def _step(self, state, action):
        x, t = state

        # calculate cost
        cost = 3 if action == self.FAST else 1

        # move (fast or slow)
        nx = x + (self.dx_fast if action == self.FAST else self.dx_slow)
        nt = t + (self.dt_fast if action == self.FAST else self.dt_slow)

        # apply randomness
        nx -= self.eps if self.unlucky else self.get_random()
        nt -= -self.eps if self.unlucky else self.get_random()

        # check if game is over
        terminated = False
        if nt >= self.t_max:
            cost += 20
            terminated = True

        elif nx >= self.x_max and nt < self.t_max:
            terminated = True

        return (nx, nt), -cost, terminated

    def get_random(self):
        return self.eps - np.random.random() * (self.eps * 2)
