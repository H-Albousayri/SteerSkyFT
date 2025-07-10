from utils import *
from DroneEnv import *
from Agent import *

from AvrisEnv import *


NUM_USERS = 5
NUM_EVES = 1

set_deterministic(100)


drone_env =  CartesianMultiUserEnv(K=NUM_USERS, walking_users=False)
avris_env = AVRIS(4,4,4,4, num_users=NUM_USERS, num_eves=NUM_EVES, train_G=True, mode="All")
avris_env.xyz_loc_UE = drone_env.users

drone_agent = DDPGAgent(state_dim=drone_env.observation_space.shape[0], 
                  action_dim=2,
                  h_dims1=128,
                  h_dims2=128,
                  gamma=0.99,
                  lamda=0.0,
                  capacity=1, 
                  device="cuda")


################ Load the model #################  
filename = f"Drone_Agent/Agent:{drone_agent.h_dims1}x{drone_agent.h_dims2}_K={drone_env.K}_WU=True_H={drone_env.drone_height}.pth"
drone_agent.load_checkpoint(filename)   


avris_agent = DDPGAgent(state_dim=avris_env.state_dim,
                        action_dim=avris_env.action_dim,
                        h_dims1=512,
                        h_dims2=256,
                        gamma=0.99,
                        lamda=1e-4,
                        capacity=10000,
                        device="cuda")
                        

time_steps = 3000
max_episodes = 300

Ep_Rewards = []
UE_Rates = [] 
Eve_Rates = [] 

for episode in range(max_episodes):
    UE_rates = []
    Eve_rates = []
    Ep_rewards = [] 
    
    d_state = drone_env.reset()
    avris_env.xyz_loc_UAV = drone_env.drone_pos
    
    a_state = avris_env.reset()
    exploration_noise = get_exponential_noise(episode, max_episodes)
    
    for t in range(time_steps):
        d_action = drone_agent.select_action(d_state, noise=0.0)
        a_action = avris_agent.select_action(a_state, noise=exploration_noise)
        
        d_next_state, d_rewards, d_done, _ = drone_env.step(d_action)
        
        avris_env.xyz_loc_UAV = drone_env.drone_pos
        a_next_state, a_reward, a_done, _ = avris_env.step(a_action)
        
        avris_agent.buffer.push(a_state, a_action, a_reward, a_next_state, a_done)
        avris_agent.train(batch_size=64)
        
        d_state = d_next_state
        a_state = a_next_state
        
        UE_rates.append(np.sum(avris_env.bit_rates))
        Eve_rates.append(np.sum(avris_env.eve_rates))
        Ep_rewards.append(a_reward)
        
    UE_Rates.append(np.mean(UE_rates))
    Eve_Rates.append(np.mean(Eve_rates))
    Ep_Rewards.append(np.mean(Ep_rewards))
    
    print(f"Episode {episode} ==> E: {np.round(avris_env.xyz_loc_Eve[0:2],2)[0,:2]} with Total Eve: {Eve_Rates[-1]} || UE: {np.round(np.mean(avris_env.xyz_loc_UE, axis=0),2)[:2]} with Total Rates: {UE_Rates[-1]} || UAV: {np.round(avris_env.xyz_loc_UAV[0:2],2)} || Total Reward: {(Ep_Rewards[-1]):.2f}")    
