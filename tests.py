import debugrl as drl 
from dreamer import DreamerV3
from tqdm import tqdm

def test_value_estimation(config):
    env = drl.OneActionZeroObsEnv()
    agent = DreamerV3(env.observation_space.shape, env.action_space.n, False, config)

    with tqdm(total=config.episodes, desc="Testing Value Estimation", unit="ep") as pbar:
        for _ in range(config.episodes):
            obs = env.reset()
            agent.act(obs, reset=True) 
            done = False
            while not done:
                action = agent.act(obs)
                next_obs, reward, done, _ = env.step(action)
                agent.store_transition(obs, action, reward, next_obs, done)
                obs = next_obs

                if len(agent.replay_buffer) > config.min_buffer_size:
                    losses = agent.train(num_updates=1)
                    pbar.set_postfix({"Avg. Q Value": f"{agent.q_values[-1].mean():.2f}", 
                                      "Critic Loss": f"{losses['critic_loss']:.2f}"})
                pbar.update(1)

    print(agent.q_values[-1])
    print(agent.returns[-1])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--train_horizon", type=int, default=1)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--latent_categories", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--actor_lr", type=float, default=1e-3)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    parser.add_argument("--world_lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--imagination_horizon", type=int, default=15)
    parser.add_argument("--free_nats", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=50)
    parser.add_argument("--replay_buffer_capacity", type=int, default=100000)
    parser.add_argument("--entropy_scale", type=float, default=1e-3)
    parser.add_argument("--kl_balance_alpha", type=float, default=0.8)
    parser.add_argument("--lambda_", type=float, default=0.95)
    parser.add_argument("--max_grad_norm", type=float, default=100.0)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--num_updates", type=int, default=5)
    parser.add_argument("--min_buffer_size", type=int, default=100)
    parser.add_argument("--init_temperature", type=float, default=1.0)
    parser.add_argument("--temperature_decay", type=float, default=0.999)
    parser.add_argument("--min_temperature", type=float, default=0.2)
    parser.add_argument("--actor_temperature", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--kl_scale", type=float, default=1.0)
    args = parser.parse_args()

    test_value_estimation(args)