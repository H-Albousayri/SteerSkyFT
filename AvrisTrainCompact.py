from utils import *
from DroneEnv import *
from AgentRobust import *

from AvrisEnv import *

from gym.vector import SyncVectorEnv

NUM_USERS = 5
NUM_EVES = 1
NUM_ENVS = 5
LAMDA_INIT = 1e-4


max_episodes = 300
warmup_episodes = max_episodes//6

seed = 100

set_deterministic(seed)

def make_env(seed):
    def _init():
        env = AVRIS(4,4,4,4, num_users=NUM_USERS, num_eves=NUM_EVES, train_G=True, seed=seed, mode="All")
        return env
    return _init


avris_env = SyncVectorEnv([make_env(seed=i) for i in range(NUM_ENVS)])

avris_agent = DDPGAgent(state_dim=avris_env.envs[0].state_dim,
                        action_dim=avris_env.envs[0].action_dim,
                        max_episodes=max_episodes, 
                        h_dims1=512,
                        h_dims2=256,
                        gamma=0.99,
                        lamda=LAMDA_INIT,
                        capacity=20000,
                        device="cuda")
                        

Ep_Rewards = []
UE_Rates = [] 
Eve_Rates = [] 

for episode in range(max_episodes):
    UE_rates = []
    Eve_rates = []
    Ep_rewards = [] 
    
    a_state, _ = avris_env.reset()
    
    exploration_noise = get_exponential_noise(episode, max_episodes)
    if episode > warmup_episodes:
        avris_agent.actor_scheduler.step(episode)
        avris_agent.critic1_scheduler.step(episode)
        avris_agent.critic2_scheduler.step(episode)
    
    batch_size = linear_increment_minibatches(episode, warmup_episodes, start_batches=128, max_batches=1024, max_ep=max_episodes)
    new_lamda = linear_decay_weight_decay(LAMDA_INIT, episode, max_episodes, eta_min=0.0)
    avris_agent.update_weight_decay(new_lamda)
    
    time_steps = get_time_steps(episode, warmup_ep=warmup_episodes, start=1000, end=5000, max_ep=max_episodes)
    
        
    for t in range(time_steps):
        a_action = avris_agent.select_action(a_state, noise=exploration_noise)
        
        
        a_next_state, a_reward, a_done, truncates, _ = avris_env.step(a_action)
        
        avris_agent.buffer.push_batch(a_state, a_action, a_reward, a_next_state, a_done)
        avris_agent.train(batch_size=batch_size)

        a_state = a_next_state
        
        UE_rates.append(np.sum(avris_env.envs[0].bit_rates))
        Eve_rates.append(np.sum(avris_env.envs[0].eve_rates))
        Ep_rewards.append(a_reward)
        
    UE_Rates.append(np.mean(UE_rates))
    Eve_Rates.append(np.mean(Eve_rates))
    Ep_Rewards.append(np.mean(Ep_rewards))
    
    print(f"Episode {episode} ==> E: {np.round(avris_env.envs[0].xyz_loc_Eve[0:2],2)[0,:2]} with Total Eve: {Eve_Rates[-1]:2f} || UE: {np.round(np.mean(avris_env.envs[0].xyz_loc_UE, axis=0),2)[:2]} with Total Rates: {UE_Rates[-1]:2f} || UAV: {np.round(avris_env.envs[0].xyz_loc_UAV[0:2],2)} || LoS%: {np.round(np.mean(np.vstack(avris_env.envs[0].LoS_list), axis=0),2)} || Total Reward: {(Ep_Rewards[-1]):.2f}")    
    print(f"Config: lamda={new_lamda:2f} | noise={exploration_noise:2f} | steps={time_steps} | BS={batch_size} | actorLR={avris_agent.actor_scheduler.optimizer.param_groups[0]['lr']} | criticLR={avris_agent.critic1_scheduler.optimizer.param_groups[0]['lr']}")
    print("==========================================================================================================")

    
plt.plot(Ep_Rewards)
plt.savefig(f"Drone_Agent/AVRIS_{np.round(avris_env.envs[0].xyz_loc_UAV[2],2)}.pdf", bbox_inches='tight')
plt.show()