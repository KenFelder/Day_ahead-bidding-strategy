import torch
from torch import nn
import gym
from collections import deque
import itertools
import numpy as np
import random

# Hyperparameters
GAMMA = 0.99
BATCH_SIZE = 32
BUFFER_SIZE = 50000
MIN_REPLAY_SIZE = 1000
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10000
TARGET_UPDATE_FREQ = 1000
LR = 5e-4

class Network(nn.Module):
    def __init__(self, env):
        super().__init__()

        in_features = int(np.prod(env.observation_space.shape))
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Tanh(),
            nn.Linear(64, env.action_space.n))

    def forward(self, x):
        return self.net(x)

    def act(self, obs, device):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        q_values = self(obs_t.unsqueeze(0))
        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item()
        return action

def main():
    env = gym.make('CartPole-v1')  # No render mode initially

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    replay_buffer = deque(maxlen=BUFFER_SIZE)
    rew_buffer = deque([0.0], maxlen=100)
    episode_reward = 0.0

    online_net = Network(env).to(device)
    target_net = Network(env).to(device)
    target_net.load_state_dict(online_net.state_dict())

    optimizer = torch.optim.Adam(online_net.parameters(), lr=LR)

    # Initialize Replay Buffer
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    for _ in range(MIN_REPLAY_SIZE):
        action = env.action_space.sample()
        new_obs, rew, done, _, _ = env.step(action)
        transition = (obs, action, rew, done, new_obs)
        replay_buffer.append(transition)
        obs = new_obs
        if done:
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]

    # Main Training Loop
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    for step in itertools.count():
        epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
        rnd_sample = random.random()

        if rnd_sample <= epsilon:
            action = env.action_space.sample()
        else:
            action = online_net.act(obs, device)

        new_obs, rew, done, _, _ = env.step(action)
        transition = (obs, action, rew, done, new_obs)
        replay_buffer.append(transition)
        obs = new_obs
        episode_reward += rew

        if done:
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]

            rew_buffer.append(episode_reward)
            episode_reward = 0.0

        # Start Gradient Step
        if len(replay_buffer) >= BATCH_SIZE:
            transitions = random.sample(replay_buffer, BATCH_SIZE)

            obses = np.asarray([t[0] for t in transitions])
            actions = np.asarray([t[1] for t in transitions])
            rews = np.asarray([t[2] for t in transitions])
            dones = np.asarray([t[3] for t in transitions])
            new_obses = np.asarray([t[4] for t in transitions])

            obses_t = torch.as_tensor(obses, dtype=torch.float32, device=device)
            actions_t = torch.as_tensor(actions, dtype=torch.int64, device=device).unsqueeze(-1)
            rews_t = torch.as_tensor(rews, dtype=torch.float32, device=device).unsqueeze(-1)
            dones_t = torch.as_tensor(dones, dtype=torch.float32, device=device).unsqueeze(-1)
            new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32, device=device)

            # Compute Targets
            target_q_values = target_net(new_obses_t)
            max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
            targets = rews_t + GAMMA * (1 - dones_t) * max_target_q_values

            # Compute Loss
            q_values = online_net(obses_t)
            action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)
            loss = nn.functional.smooth_l1_loss(action_q_values, targets)

            # Gradient Descent
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update Target Network
            if step % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(online_net.state_dict())

        # Logging
        if step % 1000 == 0:
            print(f"Step: {step}, Avg Reward: {np.mean(rew_buffer)}")

        # After solved, watch it play
        if len(rew_buffer) >= 100 and np.mean(rew_buffer) >= 200:
            env.close()  # Close the old environment
            env = gym.make('CartPole-v1', render_mode='human')  # Re-create with rendering
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            while True:
                action = online_net.act(obs, device)
                obs, _, done, _, _ = env.step(action)
                env.render()
                if done:
                    obs = env.reset()
                    if isinstance(obs, tuple):
                        obs = obs[0]

if __name__ == "__main__":
    main()
