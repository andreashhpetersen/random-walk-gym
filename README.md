# Random Walk environment for Reinforcement Learning

In the Random Walk environment, an agent has to get from its starting position
(`x=0`) to a goal state (`x >= 1`) before time runs out (`t >= 1`). To do this,
the agent can choose between two actions: move slow, which has a low cost but
is also ineffective with regards to move distance and time, or move fast, which
will quickly get the agent far but at a higher cost. To make things worse, there
is a randomness to the effect of each action.

The task is thus to find a strategy, that will get the agent to the goal state
in due time at as low a cost as possible.

## How to use

Clone this repository and install it using pip:

```sh
git clone git@github.com:andreashhpetersen/random-walk-gym.git
pip install random-walk
```

Now you can include it your training scripts as follows:

```python
import random_walk
import gymnasium as gym

# set unlucky=True to remove the randomness and make the environment as evil
# as possible (default is False)
env = gym.make('RandomWalk-v0', render_mode=None, unlucky=False)
```
