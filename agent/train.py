import sys
import os
from collections import deque
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.triage_env import TriageEnv
from agent import SimpleQLearningAgent

def train_agent(episodes=1000, steps_per_episode=100):
    env = TriageEnv()
    agent = SimpleQLearningAgent(n_actions=3)

    recent_rewards = deque(maxlen=100)

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        for _ in range(steps_per_episode):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            total_reward += reward
        agent.decay_epsilon()
        recent_rewards.append(total_reward)
        moving_avg = np.mean(recent_rewards)
        print(f"Episode {ep+1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.4f}, Moving Avg: {moving_avg:.2f}")

if __name__ == "__main__":
    train_agent()
