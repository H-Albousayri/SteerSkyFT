from gym import spaces
import numpy as np

class CartesianMultiUserEnv():
    def __init__(self, K=3, walking_users=False):
        self.walking_users = walking_users
        self.K = K
        self.max_step = 0.5
        self.max_steps = 500
        self.pathloss_exponent = 2.2
        self.drone_height = 20.0
        self.X_lw, self.Y_lw, self.X_up, self.Y_up = -np.inf, -np.inf, np.inf, np.inf
        
        self.bs_pos = np.array([0.0, 0.0, 0.0])  # base station fixed at origin
        
        self.users = np.zeros((self.K, 3))
        self.users[:, 0] = np.arange(self.K) * 10
        self.users[:, 1] = 75

        self.drone_pos = np.random.uniform(100, 200, size=(3,))
        self.drone_pos[2] = self.drone_height
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3 + 3*K,), dtype=np.float32)
        
        self.reset()

    def reset(self):
        if self.walking_users:  
            self.users[:, :2] = np.random.uniform(50, 150, size=(self.K, 2))
        self.drone_pos[:2] = np.random.uniform(100, 200, size=(2,))
        self.steps = 0
        return self._get_state()

        
    def _get_state(self):
        return np.concatenate([self.drone_pos, (self.users - self.drone_pos).flatten()])
        
    def path_loss(self, d):
        # d: distance (BS-drone or drone-user)
        return d ** (-self.pathloss_exponent)
        
        P = 1.0        # Transmit power
        N0 = 1e-8      # Noise power
        eta = self.pathloss_exponent
    
        # Distance: BS → Drone
        d_bs = np.linalg.norm(self.drone_pos - self.bs_pos)  # scalar
    
        # Distances: Drone → Users
        d_du = np.linalg.norm(self.users - self.drone_pos, axis=1)  # shape: (K,)
    
        # End-to-end path loss gain (vectorized)
        e2e_path_gain = 1.0 / ((d_bs ** eta) * (d_du ** eta))       # shape: (K,)
    
        # SNRs
        snrs = (P / N0) * e2e_path_gain                             # shape: (K,)
    
        # Rates
        rates = np.log2(1 + snrs)
    
        return snrs, rates, np.sum(rates)

    def step(self, action):
        delta = action * self.max_step
        self.drone_pos[:2] += delta

        if (self.drone_pos[0] <= self.X_lw or self.drone_pos[0] >= self.X_up or
            self.drone_pos[1] <= self.Y_lw or self.drone_pos[1] >= self.Y_up):
            self.drone_pos -= delta

        self.steps += 1
        
        self.d_du = np.linalg.norm(self.users - self.drone_pos, axis=1)
        self.d_bs = np.linalg.norm(self.drone_pos - self.bs_pos)

        reward = -np.mean(self.d_du) * self.d_bs
        done = self.steps >= self.max_steps
        
        return self._get_state(), reward, done, {}

    def render(self, mode="human"):
        print(f"Drone: {np.round(self.drone_pos, 2)}, UE: {np.round(np.mean(self.users, axis=0),2)} AvgDist: {np.mean(self.d_du).item():.2f}" )
        
        
        
