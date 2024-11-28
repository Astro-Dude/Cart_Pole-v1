# import gym
# import matplotlib.pyplot as plt 
# import pygame
# import time


# env = gym.make('CartPole-v1',render_mode="human")
# obs_space = env.observation_space
# action_space = env.action_space
# print("The observation space: {}".format(obs_space))
# print("The action space: {}".format(action_space))

# action_space = env.action_space
# print("The observation space: {}".format(obs_space))
# print("The action space: {}".format(action_space))

# # reset the environment and see the initial observation
# obs = env.reset()
# print("The initial observation is {}".format(obs))

# # Sample a random action from the entire action space
# random_action = env.action_space.sample()

# # # Take the action and get the new observation space
# new_obs, reward, terminated, truncated, info = env.step(random_action)
# print("The new observation is {}".format(new_obs))
# print("The new observation is {}".format(reward))

# env.render()
# # time.sleep(1)

# # Number of steps you run the agent for 
# num_steps = 1500

# obs = env.reset()

# for step in range(num_steps):
#     # take random action, but you can also do something more intelligent
#     # action = my_intelligent_agent_fn(obs) 
#     action = env.action_space.sample()
    
#     # apply the action
#     obs, reward, terminated, truncated, info = env.step(action)
#     print(f"observation: {obs}")
#     print(f"reward: {reward}")
#     # Render the env
#     env.render()

#     # Wait a bit before the next frame unless you want to see a crazy fast video
#     time.sleep(0.001)
    
#     # If the epsiode is up, then start another one
#     if terminated:
#         env.reset()

# # Close the env
# env.close()


import gym
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def custom_reward(state, done):
    position, velocity, angle, angular_velocity = state
    reward = 0
    if done:  # Episode ends
        return -100  # Large penalty for failure
    # Reward based on keeping the pole upright and near the center
    reward += 1.0 - (abs(angle) / 0.2095)  # Normalize angle to max

    return reward


# Initialize environment and hyperparameters
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

replay_buffer = ReplayBuffer(10000)
optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)

# Hyperparameters
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.995
epsilon_min = 0.01
batch_size = 64
target_update = 10
num_episodes = 2000

for episode in range(num_episodes):
    state, _ = env.reset()  # Reset returns (state, info)
    done = False
    total_reward = 0

    while not done:
        # Select an action (epsilon-greedy)
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = policy_net(state_tensor).argmax().item()

        # Step the environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        reward = custom_reward(next_state, done)
        total_reward += reward

        # Store transition in replay buffer
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state

        if len(replay_buffer) > batch_size:
            # Sample from the replay buffer
            batch = replay_buffer.sample(batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.FloatTensor(np.array(states))
            actions = torch.LongTensor(actions).unsqueeze(1)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            # Convert dones to boolean tensor
            dones = torch.BoolTensor(dones)

            # Calculate Q-values and targets
            q_values = policy_net(states).gather(1, actions).squeeze()
            next_q_values = target_net(next_states).max(1)[0]

            # Mask out the Q-values for terminal states
            next_q_values[dones] = 0.0
            target = rewards + gamma * next_q_values


            # Update policy network
            loss = nn.MSELoss()(q_values, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Update target network
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # Decay exploration rate
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

# Save the trained model
torch.save(policy_net.state_dict(), "cartpole_dqn.pth")
print("Model saved successfully!")

# policy_net.load_state_dict(torch.load("cartpole_dqn.pth"))