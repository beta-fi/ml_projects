import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1")

Q = np.zeros((16, 4))
alpha = 0.1
gamma = 0.99
epsilon = 1.00

for episode in range(1000):
    state, info = env.reset()
    done = False
    while not done:
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        next_state, reward, done, truncated, info = env.step(action)
        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )
        state = next_state

    epsilon = epsilon * 0.995

# Test the trained agent
state, info = env.reset()
done = False

while not done:
    action = np.argmax(Q[state])
    state, reward, done, truncated, info = env.step(action)

print(f"Final reward: {reward}")
print("Trained Q-table:")
print(Q)
