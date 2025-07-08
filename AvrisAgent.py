import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np 
import random
import os 

class Actor2(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        dim1 = 256
        self.shared_fc = nn.Sequential(
            nn.Linear(state_dim, dim1), nn.ReLU(),
            nn.LayerNorm(dim1),
            nn.Linear(dim1, 512), nn.ReLU(),
            # nn.LayerNorm(512),
            nn.Linear(512, action_dim), nn.Tanh()
        )
        
    def forward(self, state):
        return self.shared_fc(state) 
    

class Critic2(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        dim1 = 512
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, dim1), nn.ReLU(),
            nn.LayerNorm(dim1),
            nn.Linear(dim1, 256), nn.ReLU(),
            # nn.LayerNorm(256),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))
        
##################################################
##################################################

class ReplayBuffer2:
    def __init__(self, size=5000):
        self.buffer = []
        self.max_size = size
        self.ptr = 0

    def add(self, transition):
        if len(self.buffer) < self.max_size:
            self.buffer.append(transition)
        else:
            self.buffer[self.ptr] = transition
            self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        
        states_np      = np.array(states, dtype=np.float32)       # shape: (batch_size, *state_shape)
        actions_np     = np.array(actions, dtype=np.float32)      # shape: (batch_size, *action_shape)
        rewards_np     = np.array(rewards, dtype=np.float32)      # shape: (batch_size,)
        next_states_np = np.array(next_states, dtype=np.float32)  # shape: (batch_size, *state_shape)
        dones_np       = np.array(dones, dtype=np.float32)        # shape: (batch_size,)

        return (
            torch.from_numpy(states_np),                         # FloatTensor of shape (batch_size, *state_shape)
            torch.from_numpy(actions_np),                        # FloatTensor of shape (batch_size, *action_shape)
            torch.from_numpy(rewards_np).unsqueeze(1),           # FloatTensor of shape (batch_size, 1)
            torch.from_numpy(next_states_np),                    # FloatTensor of shape (batch_size, *state_shape)
            torch.from_numpy(dones_np).unsqueeze(1),             # FloatTensor of shape (batch_size, 1)
        )

########################################################
########################################################

class DDPG:
    def __init__(self, state_dim, action_dim, max_action, device, lamda):
        self.actor = Actor2(state_dim, action_dim).to(device)
        self.actor_target = Actor2(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic2(state_dim, action_dim).to(device)
        self.critic_target = Critic2(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4, weight_decay=lamda)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3, weight_decay=lamda)

        # self.replay_buffer = PrioritizedReplayBuffer(5000)
        self.replay_buffer = ReplayBuffer2()
        self.device = device
        self.max_action = max_action

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, batch_size=64, gamma=0.0, tau=0.005):
        if len(self.replay_buffer.buffer) < batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state      = state.to(self.device)
        action     = action.to(self.device)
        reward     = reward.to(self.device)
        next_state = next_state.to(self.device)
        done       = done.to(self.device)

        # Target Q-value
        with torch.no_grad():
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (1 - done) * gamma * target_Q

        # Critic update
        current_Q = self.critic(state, action)


        critic_loss = nn.MSELoss()(current_Q, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update targets
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save_checkpoint(self, filename):
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved to {filename}")

    def load_checkpoint(self, filename):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Checkpoint file '{filename}' not found.")
        checkpoint = torch.load(filename, map_location=self.device)

        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

        print(f"Checkpoint loaded from {filename}")