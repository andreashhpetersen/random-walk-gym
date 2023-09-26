from .envs import RandomWalkEnv
from gymnasium.envs.registration import register

register(
    id='RandomWalk-v0',
    entry_point='random_walk.envs:RandomWalkEnv',
    max_episode_steps=None,
)
