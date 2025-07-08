from DroneEnv import *
from Agent import *

SEEDS = [0,1,2,3,4,5]
for seed in SEEDS:
    set_deterministic(seed)

    env = CartesianMultiUserEnv(K=5, walking_users=True)

    agent = DDPGAgent(state_dim=env.observation_space.shape[0], 
                    action_dim=2,
                    h_dims1=128,
                    h_dims2=128,
                    gamma=0.99,
                    lamda=0.0,
                    device="cuda")

    max_episodes = 500

    Ep_reward = [] 

    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        locs = 0 
        exploration_noise = get_exponential_noise(episode, max_episodes)
        for _ in range(env.max_steps):
            action = agent.select_action(state, noise=exploration_noise)
            next_state, reward, done, _ = env.step(action)
            agent.buffer.push(state, action, reward, next_state, done)
            agent.train(batch_size=64)
            state = next_state
            episode_reward += reward
            locs += np.mean(env.d_du)
        
        Ep_reward.append(episode_reward/env.max_steps)
        print(f"Episode {episode} | Noise: {np.round(exploration_noise,2)} | Drone: {np.round(env.drone_pos[:2], 2)}, UE: {np.round(np.mean(env.users, axis=0),2)[:2]} AvgDist: {np.mean(env.d_du).item():.2f} | Total Reward: {(Ep_reward[-1]):.2f}")
        
        
    plt.plot(Ep_reward)
    plt.savefig(f"Drone_Agent/Fig_{seed}")
    plt.show()

    ################ Save the model #################  
    filename = f"Drone_Agent/Agent_{seed}:{agent.h_dims1}x{agent.h_dims2}_K={env.K}_WU={env.walking_users}_H={env.drone_height}.pth"
    agent.save_checkpoint(filename)    