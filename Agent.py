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

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return map(lambda x: torch.FloatTensor(np.array(x)), (state, action, reward, next_state, done))

    def __len__(self):
        return len(self.buffer)

class DDPGAgent:
    def __init__(self, state_dim, action_dim, 
                 h_dims1=256, h_dims2=256, gamma=0.99, lamda=0.0,
                 capacity=100000, device="cpu"):
        self.device = device
        self.actor = Actor(state_dim, action_dim, h_dims1, h_dims2).to(device)
        self.actor_target = Actor(state_dim, action_dim, h_dims1, h_dims2).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim, h_dims1, h_dims2).to(device)
        self.critic_target = Critic(state_dim, action_dim, h_dims1, h_dims2).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4, weight_decay=lamda)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3, weight_decay=lamda)

        self.buffer = ReplayBuffer(capacity)
        self.gamma = gamma
        self.tau = 0.005
        self.h_dims1 = h_dims1
        self.h_dims2 = h_dims2
        
    def select_action(self, state, noise=0.1):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        action += noise * np.random.randn(len(action))
        return np.clip(action, -1.0, 1.0)

    def train(self, batch_size=64):
        if len(self.buffer) < batch_size:
            return

        state, action, reward, next_state, done = self.buffer.sample(batch_size)

        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.unsqueeze(1).to(self.device)
        next_state = next_state.to(self.device)
        done = done.unsqueeze(1).float().to(self.device)

        # Critic update
        with torch.no_grad():
            target_action = self.actor_target(next_state)
            target_q = self.critic_target(next_state, target_action)
            target = reward + self.gamma * target_q * (1 - done)

        current_q = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_q, target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


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
        