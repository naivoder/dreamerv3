from dreamer import Dreamer
import warnings
from preprocess import AtariEnv
import os


warnings.filterwarnings("ignore")


def main(config):
    pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str)
    parser.add_argument("--n_envs", type=int, default=32)
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--train_horizon", type=int, default=1000)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--latent_categories", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--world_lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--imagination_horizon", type=int, default=15)
    parser.add_argument("--free_nats", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--seq_len", type=int, default=50)
    parser.add_argument("--replay_buffer_capacity", type=int, default=100000)
    parser.add_argument("--entropy_scale", type=float, default=1e-3)
    parser.add_argument("--kl_balance_alpha", type=float, default=0.8)
    parser.add_argument("--lambda_", type=float, default=0.95)
    parser.add_argument("--max_grad_norm", type=float, default=100.0)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--num_updates", type=int, default=100)
    parser.add_argument("--min_buffer_size", type=int, default=10000)
    parser.add_argument("--init_temperature", type=float, default=1.0)
    parser.add_argument("--temperature_decay", type=float, default=0.999)
    parser.add_argument("--min_temperature", type=float, default=0.5)
    parser.add_argument("--actor_temperature", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--kl_scale", type=float, default=1.0)
    args = parser.parse_args()
