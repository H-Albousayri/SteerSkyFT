from utils import *
from DroneEnv import *
from AgentRobust import *

from AvrisEnv import *


NUM_USERS = 5
NUM_EVES = 1

set_deterministic(100)

avris_env = AVRIS(4,4,4,4, num_users=NUM_USERS, num_eves=NUM_EVES, train_G=True, mode="All")

avris_agent = DDPGAgent(state_dim=avris_env.state_dim,
                        action_dim=avris_env.action_dim,
                        h_dims1=512,
                        h_dims2=256,
                        gamma=0.99,
                        lamda=1e-4,
                        capacity=10000,
                        device="cuda")
                        

time_steps = 1000
max_episodes = 3000

Ep_Rewards = []
UE_Rates = [] 
Eve_Rates = [] 

for episode in range(max_episodes):
    UE_rates = []
    Eve_rates = []
    Ep_rewards = [] 
    
    
    a_state = avris_env.reset()
    exploration_noise = get_exponential_noise(episode, max_episodes)
    
    for t in range(time_steps):
        a_action = avris_agent.select_action(a_state, noise=exploration_noise)
        
        
        a_next_state, a_reward, a_done, _ = avris_env.step(a_action)
        
        avris_agent.buffer.push(a_state, a_action, a_reward, a_next_state, a_done)
        avris_agent.train(batch_size=64)

        a_state = a_next_state
        
        UE_rates.append(np.sum(avris_env.bit_rates))
        Eve_rates.append(np.sum(avris_env.eve_rates))
        Ep_rewards.append(a_reward)
        
    UE_Rates.append(np.mean(UE_rates))
    Eve_Rates.append(np.mean(Eve_rates))
    Ep_Rewards.append(np.mean(Ep_rewards))
    
    print(f"Episode {episode} ==> E: {np.round(avris_env.xyz_loc_Eve[0:2],2)[0,:2]} with Total Eve: {Eve_Rates[-1]} || UE: {np.round(np.mean(avris_env.xyz_loc_UE, axis=0),2)[:2]} with Total Rates: {UE_Rates[-1]} || UAV: {np.round(avris_env.xyz_loc_UAV[0:2],2)} || LoS%: {np.round(np.mean(np.vstack(avris_env.LoS_list), axis=0),2)} || Total Reward: {(Ep_Rewards[-1]):.2f}")    

plt.plot(Ep_Rewards)
plt.savefig(f"Drone_Agent/AVRIS_{np.round(avris_env.xyz_loc_UAV[2],2)}.pdf", bbox_inches='tight')
plt.show()