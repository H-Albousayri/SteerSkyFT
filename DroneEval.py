from DroneEnv import *
from Agent import *

set_deterministic(100)

env = CartesianMultiUserEnv(K=5, walking_users=True)
env.max_steps = 1000

agent = DDPGAgent(state_dim=env.observation_space.shape[0], 
                  action_dim=2,
                  h_dims1=128,
                  h_dims2=128,
                  gamma=0.99,
                  lamda=0.0,
                  device="cuda")

################ Load the model #################  
filename = f"Drone_Agent/Agent:{agent.h_dims1}x{agent.h_dims2}_K={env.K}_WU={env.walking_users}_H={env.drone_height}.pth"
agent.load_checkpoint(filename)    

max_episodes = 400

Ep_reward = [] 

for episode in range(max_episodes):
    state = env.reset()
    episode_reward = 0
    locs = 0 
    exploration_noise = 0.0
    for _ in range(env.max_steps):
        action = agent.select_action(state, noise=exploration_noise)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        episode_reward += reward
        locs += np.mean(env.d_du)
    
    Ep_reward.append(episode_reward/env.max_steps)
    print(f"Episode {episode} | Noise: {np.round(exploration_noise,2)} | Drone: {np.round(env.drone_pos[:2], 2)}, UE: {np.round(np.mean(env.users, axis=0),2)[:2]} AvgDist: {np.mean(env.d_du).item():.2f} | Total Reward: {(Ep_reward[-1]):.2f}")
    
    

