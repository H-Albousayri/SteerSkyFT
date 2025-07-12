from utils import *
import torch.optim as optim

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, h_dims1, h_dims2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, h_dims1), nn.ReLU(),
            nn.LayerNorm(h_dims1),
            nn.Linear(h_dims1, h_dims2), nn.ReLU(),
            nn.Linear(h_dims2, action_dim), nn.Tanh()  # outputs ∈ [-1, 1]
        )

    def forward(self, x):
        return self.net(x)  # scaled in environment

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, h_dims1, h_dims2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, h_dims1), nn.ReLU(),
            nn.LayerNorm(h_dims1),
            nn.Linear(h_dims1, h_dims2), nn.ReLU(),
            nn.Linear(h_dims2, 1)
        )
    
    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def push_batch(self, states, actions, rewards, next_states, dones):
        for i in range(len(rewards)):
            self.push(states[i], actions[i], rewards[i], next_states[i], dones[i])

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return map(lambda x: torch.FloatTensor(np.array(x)), (state, action, reward, next_state, done))

    def __len__(self):
        return len(self.buffer)


class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_episodes=300, 
                 h_dims1=256, h_dims2=256, gamma=0.99, tau=0.005, lamda=0.0,
                 capacity=1_000_000, device="cpu", policy_noise=0.05, noise_clip=0.5, policy_delay=2):
        self.device = device

        self.actor = Actor(state_dim, action_dim, h_dims1, h_dims2).to(device)
        self.actor_target = Actor(state_dim, action_dim, h_dims1, h_dims2).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic1 = Critic(state_dim, action_dim, h_dims1, h_dims2).to(device)
        self.critic2 = Critic(state_dim, action_dim, h_dims1, h_dims2).to(device)
        self.critic1_target = Critic(state_dim, action_dim, h_dims1, h_dims2).to(device)
        self.critic2_target = Critic(state_dim, action_dim, h_dims1, h_dims2).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4, weight_decay=lamda)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=1e-3, weight_decay=lamda)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=1e-3, weight_decay=lamda)

        self.actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.actor_optimizer, T_max=max_episodes, eta_min=1e-6
        )
        self.critic1_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.critic1_optimizer, T_max=max_episodes, eta_min=1e-5
        )
        self.critic2_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.critic2_optimizer, T_max=max_episodes, eta_min=1e-5
        )

        self.buffer = ReplayBuffer(capacity)
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.total_it = 0

    def update_weight_decay(self, new_wd):
        for opt in [self.actor_optimizer, self.critic1_optimizer, self.critic2_optimizer]:
            for param_group in opt.param_groups:
                param_group['weight_decay'] = new_wd
            
    def select_action(self, state, noise=0.1):
        if len(state.shape) == 1:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor(state).cpu().data.numpy().flatten()
        else:
            state = torch.FloatTensor(state).to(self.device)
            action = self.actor(state).cpu().data.numpy()
        
        if noise != 0:
            action += np.random.normal(0, noise, size=action.shape)
        return np.clip(action, -1.0, 1.0)

    def train(self, batch_size=256):
        if len(self.buffer) < batch_size:
            return

        self.total_it += 1

        state, action, reward, next_state, done = self.buffer.sample(batch_size)

        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.unsqueeze(1).to(self.device)
        next_state = next_state.to(self.device)
        done = done.unsqueeze(1).float().to(self.device)

        # Target policy smoothing
        noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        next_action = (self.actor_target(next_state) + noise).clamp(-1, 1)

        with torch.no_grad():
            target_q1 = self.critic1_target(next_state, next_action)
            target_q2 = self.critic2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target = reward + self.gamma * target_q * (1 - done)

        # Critic 1
        current_q1 = self.critic1(state, action)
        critic1_loss = nn.MSELoss()(current_q1, target)
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        # Critic 2
        current_q2 = self.critic2(state, action)
        critic2_loss = nn.MSELoss()(current_q2, target)
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Delayed actor update
        if self.total_it % self.policy_delay == 0:
            actor_loss = -self.critic1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update all targets
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic1, self.critic1_target)
            self._soft_update(self.critic2, self.critic2_target)

    def _soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def save_checkpoint(self, filename):
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic1_target_state_dict': self.critic1_target.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'critic2_target_state_dict': self.critic2_target.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
        }
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved to {filename}")

    def load_checkpoint(self, filename):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Checkpoint file '{filename}' not found.")
        
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])

        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])

        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])

        print(f"Checkpoint loaded from {filename}")
        