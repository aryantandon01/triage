import numpy as np

class SimpleQLearningAgent:
    def __init__(self, state_size, action_size, lr=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.995):
        self.q_table = np.zeros((1000, action_size))  # simplified
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.action_size = action_size

    def get_state_index(self, state):
        return int(sum((10 ** i) * int(s * 10) for i, s in enumerate(state)))

    def act(self, state):
        idx = self.get_state_index(state)
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        return np.argmax(self.q_table[idx])

    def learn(self, state, action, reward, next_state):
        idx = self.get_state_index(state)
        next_idx = self.get_state_index(next_state)
        best_next_action = np.max(self.q_table[next_idx])
        self.q_table[idx][action] += self.lr * (reward + self.gamma * best_next_action - self.q_table[idx][action])
        self.epsilon *= self.epsilon_decay
