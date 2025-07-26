import gym
from gym import spaces
import numpy as np

class TriageEnv(gym.Env):
    def __init__(self):
        super(TriageEnv, self).__init__()
        self.action_space = spaces.Discrete(3)  # 0 = Low, 1 = Medium, 2 = High priority
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)

    def reset(self):
        self.state = np.random.rand(5)
        return self.state

    def step(self, action):
        reward = 1 if action == self._ideal_priority(self.state) else -1
        self.state = np.random.rand(5)
        done = False
        return self.state, reward, done, {}

    def _ideal_priority(self, state):
        if state[0] > 0.7 or state[1] > 0.8:
            return 2
        elif state[2] > 0.6:
            return 1
        else:
            return 0
