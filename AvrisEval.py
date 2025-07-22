from utils import *
from DroneEnv import *
from AgentRobust import *
from AvrisEnv import *



def main():
    # ----------------------------
    # ARG PARSER
    # ----------------------------
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@', description="AVRIS DDPG Training")
    parser.add_argument("--num_users", type=int, default=5, help="Number of legitimate users")
    parser.add_argument("--num_eves", type=int, default=1, help="Number of eavesdroppers")
    parser.add_argument("--num_envs", type=int, default=3, help="Number of parallel environments")
    parser.add_argument("--M", type=int, default=16, help="Number of BS elements")
    parser.add_argument("--N", type=int, default=16, help="Number of RIS elements")
    parser.add_argument("--fixed_eve", action="store_true", help="Whether to keep eve at (K_e, 40)")
    parser.add_argument("--los", action="store_true", help="Whether to consider Prop. of LoS to be 1 all the time")
    parser.add_argument("--h_dims", type=int, default=512, help="First layer h_dims")
    parser.add_argument("--max_episodes", type=int, default=300, help="Maximum number of episodes")
    parser.add_argument("--seed", type=int, nargs='+', default=[100], help="List of random seeds")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    args = parser.parse_args()

    M_, N_ = int(np.sqrt(args.M)), int(np.sqrt(args.N))
    warmup_episodes = args.max_episodes // 6
    
    time_steps = 10000
    def make_env(seed):
        def _init():
            env = AVRIS(My_BS=M_, Mz_BS=M_, Nx_RIS=N_, Ny_RIS=N_,
                        num_users=args.num_users,
                        num_eves=args.num_eves,
                        consider_LoS=args.los,
                        fixed_eve=args.fixed_eve,
                        train_G=True,
                        seed=seed,
                        mode="All")
            return env
        return _init
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for seed in args.seed:
        set_deterministic(seed)

        
        avris_env = SyncVectorEnv([make_env(seed=i) for i in range(args.num_envs)])
           
        folder_dir = "Run_20250715_113342_(M,N,K,L,FixEv,PLoS)=(16,16,3,1,True,False)"
        model_dir = "model_at_K=3_L=1_100.pth"
        load_dir = f"Drone_Agent/{folder_dir}/seed:{seed}"
        
        avris_agent = DDPGAgent(
            state_dim=avris_env.envs[0].state_dim,
            action_dim=avris_env.envs[0].action_dim,
            max_episodes=args.max_episodes,
            h_dims1=args.h_dims,
            h_dims2=256,
            gamma=0.99,
            device=args.device
        )
        
        
        avris_agent.load_checkpoint(f"Drone_Agent/{folder_dir}/seed:{seed}/{model_dir}")

        Ep_Rewards = []
        UE_Rates = []
        Eve_Rates = []
        iS_LoS_Probs = []
        for episode in range(args.max_episodes):
            UE_rates = []
            Eve_rates = []
            Ep_rewards = []
            iS_LoS_p = []
            locs = []
            a_state, _ = avris_env.reset()

            for t in range(time_steps):
                a_action = avris_agent.select_action(a_state, noise=0.1)

                a_next_state, a_reward, a_done, truncates, _ = avris_env.step(a_action)
                print(f"Ep {episode} - Time {t} : UE=>{np.round([avris_env.envs[i].xyz_loc_UAV[0:2] for i in range(args.num_envs)],2)}, R={a_reward}")
                a_state = a_next_state

                UE_rates.append(avris_env.envs[0].bit_rates)
                Eve_rates.append(avris_env.envs[0].eve_rates)
                Ep_rewards.append(a_reward)
                locs.append(avris_env.envs[0].xyz_loc_UAV[0:2].copy())
                
            
            UE_Rates.append(np.mean(np.vstack(UE_rates), axis=0))
            Eve_Rates.append(np.mean(np.vstack(Eve_rates), axis=0))
            Ep_Rewards.append(np.mean(Ep_rewards))
            iS_LoS_Probs.append(np.mean(np.vstack(avris_env.envs[0].LoS_list), axis=0))
            
        

if __name__ == "__main__":
    main()