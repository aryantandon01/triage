from env.triage_env import TriageEnv
from agent.agent import SimpleQLearningAgent

def train_agent(episodes=500):
    env = TriageEnv()
    agent = SimpleQLearningAgent(state_size=5, action_size=3)

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        for _ in range(50):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            total_reward += reward
        print(f"Episode {ep+1}, Reward: {total_reward}")
    return agent

if __name__ == "__main__":
    train_agent()
