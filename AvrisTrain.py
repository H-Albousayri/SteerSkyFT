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
    parser.add_argument("--num_envs", type=int, default=5, help="Number of parallel environments")
    parser.add_argument("--M", type=int, default=16, help="Number of BS elements")
    parser.add_argument("--N", type=int, default=16, help="Number of RIS elements")
    parser.add_argument("--lamda_init", type=float, default=1e-4, help="Initial weight decay")
    parser.add_argument("--init_steps", type=int, default=1000, help="Init time steps")
    parser.add_argument("--init_batch", type=int, default=128, help="Init batch size for the batch scheduler")
    parser.add_argument("--init_noise", type=float, default=0.45, help="Init noise STD")
    parser.add_argument("--h_dims", type=int, default=512, help="First layer h_dims")
    parser.add_argument("--max_episodes", type=int, default=300, help="Maximum number of episodes")
    parser.add_argument("--capacity", type=int, default=20000, help="Replay Buffer size")
    parser.add_argument("--seed", type=int, nargs='+', default=[100], help="List of random seeds")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    args = parser.parse_args()

    M_, N_ = int(np.sqrt(args.M)), int(np.sqrt(args.N))
    warmup_episodes = args.max_episodes // 6
    
    def make_env(seed):
        def _init():
            env = AVRIS(My_BS=M_, Mz_BS=M_, Nx_RIS=N_, Ny_RIS=N_,
                        num_users=args.num_users,
                        num_eves=args.num_eves,
                        train_G=True,
                        seed=seed,
                        mode="All")
            return env
        return _init
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for seed in args.seed:
        set_deterministic(seed)


        avris_env = SyncVectorEnv([make_env(seed=i) for i in range(args.num_envs)])
           
        save_dir = f"Drone_Agent/Run_{timestamp}_(M,N)=({args.M},{args.N})/seed:{seed}"
        log_file = setup_logger(save_dir)
        logging.info(f"Logging to: {log_file}")
        
        logging.info("========== Training Configuration ==========")
        for arg, val in vars(args).items():
            logging.info(f"{arg}: {val}")
            
        logging.info(f"state dims: {avris_env.envs[0].state_dim}")
        logging.info(f"action dims: {avris_env.envs[0].action_dim}")
        logging.info("=============================================")
        

        avris_agent = DDPGAgent(
            state_dim=avris_env.envs[0].state_dim,
            action_dim=avris_env.envs[0].action_dim,
            max_episodes=args.max_episodes,
            h_dims1=args.h_dims,
            h_dims2=256,
            gamma=0.99,
            lamda=args.lamda_init,
            capacity=args.capacity,
            device=args.device
        )

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

            exploration_noise = get_linear_noise(episode, args.max_episodes, initial_noise=args.init_noise)
            # exploration_noise = get_exponential_noise(episode, args.max_episodes, initial_noise=args.init_noise)
            
            if episode > warmup_episodes:
                avris_agent.actor_scheduler.step(episode)
                avris_agent.critic1_scheduler.step(episode)
                avris_agent.critic2_scheduler.step(episode)

            batch_size = linear_increment_minibatches(
                episode, warmup_episodes,
                start_batches=args.init_batch, max_batches=1024, max_ep=args.max_episodes
            )

            new_lamda = linear_decay_weight_decay(
                args.lamda_init, episode, args.max_episodes, eta_min=0.0
            )
            avris_agent.update_weight_decay(new_lamda)

            time_steps = get_time_steps(
                episode, warmup_ep=warmup_episodes, start=args.init_steps, end=5000, max_ep=args.max_episodes
            )

            for t in range(time_steps):
                a_action = avris_agent.select_action(a_state, noise=exploration_noise)

                a_next_state, a_reward, a_done, truncates, _ = avris_env.step(a_action)

                avris_agent.buffer.push_batch(a_state, a_action, a_reward, a_next_state, a_done)
                avris_agent.train(batch_size=batch_size)

                a_state = a_next_state

                UE_rates.append(avris_env.envs[0].bit_rates)
                Eve_rates.append(avris_env.envs[0].eve_rates)
                Ep_rewards.append(a_reward)
                locs.append(avris_env.envs[0].xyz_loc_UAV[0:2].copy())
                
                
            
            UE_Rates.append(np.mean(np.vstack(UE_rates), axis=0))
            Eve_Rates.append(np.mean(np.vstack(Eve_rates), axis=0))
            Ep_Rewards.append(np.mean(Ep_rewards))
            iS_LoS_Probs.append(np.mean(np.vstack(avris_env.envs[0].LoS_list), axis=0))
            
            if episode > (args.max_episodes - 20):
                np.save(os.path.join(save_dir, f"Locs_ep:{episode}.npy"), locs)
                
            logging.info(
                f"Episode {episode} | "
                f"E: {np.round(avris_env.envs[0].xyz_loc_Eve[0:2], 2)[0, :2]} "
                f"UE Rate: {np.round(UE_Rates[-1],2)} | "
                f"Eve: {np.round(Eve_Rates[-1],2)} | "
                f"UE: {np.round(np.mean(avris_env.envs[0].xyz_loc_UE, axis=0), 2)[:2]} "
                f"UAV: {np.round(avris_env.envs[0].xyz_loc_UAV[0:2], 2)} | "
                f"LoS%: {np.round(np.mean(np.vstack(avris_env.envs[0].LoS_list), axis=0), 2)} | "
                f"Reward: {Ep_Rewards[-1]:.2f}"
            )

            logging.info(
                f"Config | lamda={new_lamda:.2e} | noise={exploration_noise:.4f} | "
                f"steps={time_steps} | BS={batch_size} | "
                f"actorLR={avris_agent.actor_scheduler.optimizer.param_groups[0]['lr']:.2e} | "
                f"criticLR={avris_agent.critic1_scheduler.optimizer.param_groups[0]['lr']:.2e}"
            )

            logging.info("=" * 100)
        
        save_metrics(Ep_Rewards, UE_Rates, Eve_Rates, iS_LoS_Probs, save_dir)
            
        plt.plot(Ep_Rewards)
        plt.title("Episode Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid(True)
        plot_path = os.path.join(save_dir, "reward_curve.pdf")
        plt.savefig(plot_path, bbox_inches='tight')
        logging.info(f"Reward curve saved to: {plot_path}")

        ckpt_path = os.path.join(save_dir, f"model_at_K={args.num_users}_L={args.num_eves}_{seed}.pth")
        avris_agent.save_checkpoint(ckpt_path)
        logging.info(f"Checkpoint saved to: {ckpt_path}")

if __name__ == "__main__":
    main()