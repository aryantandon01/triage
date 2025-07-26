import numpy as np

class SimpleQLearningAgent:
    def __init__(self, n_actions, learning_rate=0.05, discount_factor=0.95, 
                 exploration_rate=1.0, min_epsilon=0.05, epsilon_decay=0.995, table_size=10000):
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.q_table = np.zeros((table_size, n_actions))
        self.table_size = table_size

    def _get_state_index(self, state):
        return hash(str(state)) % self.table_size

    def act(self, state):
        state_idx = self._get_state_index(state)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state_idx])

    def learn(self, state, action, reward, next_state):
        state_idx = self._get_state_index(state)
        next_idx = self._get_state_index(next_state)
        best_next_action = np.max(self.q_table[next_idx])
        td_target = reward + self.gamma * best_next_action
        td_error = td_target - self.q_table[state_idx, action]
        self.q_table[state_idx, action] += self.lr * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def decay_learning_rate(self, min_lr=0.01, decay=0.995):
        self.lr = max(min_lr, self.lr * decay)

