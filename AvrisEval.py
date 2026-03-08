from utils import *
from DroneEnv import *

from AvrisEnv import *

MODES = ["ours", "td3", "ddpg", "random"]


for MODE in MODES:
    if MODE in ["ours", "td3"]:
        from AgentRobust import *
    else:
        from Agent import *
        
    def main():
        # ----------------------------
        parser = argparse.ArgumentParser(fromfile_prefix_chars='@', description="AVRIS DDPG Training")
        parser.add_argument("--num_users", type=int, default=3, help="Number of legitimate users")
        parser.add_argument("--num_eves", type=int, default=2, help="Number of eavesdroppers")
        parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
        parser.add_argument("--M", type=int, default=16, help="Number of BS elements")
        parser.add_argument("--N", type=int, default=16, help="Number of RIS elements")
        parser.add_argument("--fixed_eve", action="store_true", help="Whether to keep eve at (K_e, 40)")
        parser.add_argument("--los", action="store_true", help="Whether to consider Prop. of LoS to be 1 all the time")
        parser.add_argument("--lamda_init", type=float, default=1e-4, help="Initial weight decay")
        parser.add_argument("--init_steps", type=int, default=200, help="Init time steps")
        parser.add_argument("--init_batch", type=int, default=128, help="Init batch size for the batch scheduler")
        parser.add_argument("--init_noise", type=float, default=0.9, help="Init noise STD")
        parser.add_argument("--h_dims", type=int, default=256, help="First layer h_dims")
        parser.add_argument("--PL_ratio", type=float, default=2.0, help="Ratio between direct channels PL and reflected channels")
        parser.add_argument("--UE_spacing", type=float, default=10, help="X axis spacing between UE")
        parser.add_argument("--UAV_height", type=float, default=50, help="UAV z position")
        parser.add_argument("--init_x", type=float, default=100.0, help="inital x value of UAV")
        parser.add_argument("--init_y", type=float, default=100.0, help="inital y value of UAV")
        parser.add_argument("--max_episodes", type=int, default=1, help="Maximum number of episodes")
        parser.add_argument("--warmup_episodes", type=int, default=50, help="When to start some schedulers")
        parser.add_argument("--capacity", type=int, default=20000, help="Replay Buffer size")
        parser.add_argument("--seed", type=int, nargs='+', default=[300], help="List of random seeds")
        parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
        args = parser.parse_args()

        M_, N_ = int(np.sqrt(args.M)), int(np.sqrt(args.N))
        warmup_episodes = args.warmup_episodes
        
        for (dx,dy) in [(0,0), (0,5), (0,-5), (-5,0), (-5,-5), (-5,5), (5,0), (5,5), (5,-5)]:
        
            time_steps = args.init_steps
            def make_env(seed):
                def _init():
                    env = AVRIS(My_BS=M_, Mz_BS=M_, Nx_RIS=N_, Ny_RIS=N_,
                                num_users=args.num_users,
                                num_eves=args.num_eves,
                                consider_LoS=args.los,
                                fixed_eve=args.fixed_eve,
                                PL_ratio=args.PL_ratio,
                                UE_spacing=args.UE_spacing,
                                UAV_height = args.UAV_height,
                                eval=True,
                                init_x=args.init_x+dx,
                                init_y=args.init_y+dy,
                                train_G=True,
                                seed=seed,DQN=False,
                                mode="All")
                    return env
                return _init
                
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            for seed in args.seed:
                set_deterministic(seed)

                
                avris_env = SyncVectorEnv([make_env(seed=i+50) for i in range(args.num_envs)])
                
                folder_dir = ""
                if MODE == "ddpg":
                    time_steps = 600
                    folder_dir = "Run_DDPG__20250725_124150_(M,N,K,L,FixEv,Ey,PLoS,PL,UAVz)=(16,16,3,2,False,39.04,True,2.0,50.0)"
                elif MODE == "td3":
                    time_steps = 600
                    folder_dir = "Run_TD3__20250726_124012_(M,N,K,L,FixEv,Ey,PLoS,PL,UAVz)=(16,16,3,2,False,39.04,True,2.0,50.0)"
                elif MODE == "ours":
                    seed = 200
                    folder_dir = "Run_20250725_121411_(M,N,K,L,FixEv,Ey,PLoS,PL_r,UE_spacing, UAVz)=(16,16,3,2,False,39.04,True,2.0,20.0,50.0)"
                elif MODE == "random":
                    time_steps = 50
                    
                model_dir = f"model_at_K=3_L=2_{seed}.pth"
                
                avris_agent = DDPGAgent(
                    state_dim=avris_env.envs[0].state_dim,
                    action_dim=avris_env.envs[0].action_dim,
                    max_episodes=args.max_episodes,
                    h_dims1=args.h_dims,
                    h_dims2=256,
                    gamma=0.99,
                    device=args.device
                )
                
                if MODE != "random":
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
                        if MODE == "random":
                            a_action = np.random.normal(0, 4, size=avris_env.envs[0].action_dim).reshape(1,-1)                    
                        else:
                            a_action = avris_agent.select_action(a_state, noise=args.init_noise)
                            
                        a_next_state, a_reward, a_done, truncates, _ = avris_env.step(a_action)
                        print(f"Ep {episode} - Time {t} : UAV=>{np.round([avris_env.envs[i].xyz_loc_UAV[0:2] for i in range(args.num_envs)],2)}, R={a_reward}")
                        a_state = a_next_state

                        UE_rates.append(avris_env.envs[0].bit_rates)
                        Eve_rates.append(avris_env.envs[0].eve_rates)
                        Ep_rewards.append(a_reward)
                        locs.append([avris_env.envs[i].xyz_loc_UAV[0:2].copy() for i in range(args.num_envs)])
                        
                    
                    UE_Rates.append(np.mean(np.vstack(UE_rates), axis=0))
                    Eve_Rates.append(np.mean(np.vstack(Eve_rates), axis=0))
                    Ep_Rewards.append(np.mean(Ep_rewards))
                    iS_LoS_Probs.append(np.mean(np.vstack(avris_env.envs[0].LoS_list), axis=0))
                    
                save_dir = f"../SteerSkyPlotting/Trajectory/ready/Locs_with_noise_{MODE}={args.init_noise}"
                os.makedirs(save_dir, exist_ok=True)
                if os.path.exists(save_dir):
                    print(f"Saved at {save_dir}")
                    np.save(f"{save_dir}/(x,y)=({args.init_x+dx},{args.init_y+dy})", locs)

    if __name__ == "__main__":
        main()