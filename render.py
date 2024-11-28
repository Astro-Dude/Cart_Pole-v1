import gym
import torch
import torch.nn as nn
from time import time as t



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


env = gym.make('CartPole-v1', render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = DQN(state_dim, action_dim)

policy_net.load_state_dict(torch.load("cartpole_dqn.pth"))

state, _ = env.reset()  # Unpack the returned tuple
done = False
start = t()
while not done:
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    action = policy_net(state_tensor).argmax().item()
    state, reward, done, _, _ = env.step(action)  # Unpack the additional values returned by step()
    env.render()

end = t()
print(f"Total balance time: {end-start}")

env.close()