# DreamerV3

This repository contains a PyTorch implementation of the DreamerV3 algorithm, aimed at providing a more readable and understandable version compared to the official implementations. It serves as an opportunity to learn and explore the intricacies of the DreamerV3 algorithm while maintaining clarity and simplicity.

## Overview

DreamerV3 is a state-of-the-art model-based reinforcement learning algorithm that learns a world model from experiences and uses it to train an actor-critic policy from imagined trajectories. The algorithm consists of several key components:

1. A world model that encodes sensory inputs into latent representations and predicts future states and rewards.
2. An actor network that learns to take actions in the imagined environment.
3. A critic network that estimates the value of states and actions.
4. An imagination process that generates trajectories using the learned world model.

This implementation aims to break down these components into clear, modular parts, making it easier to understand and modify.

## Setup and Installation

To set up the environment and install the required dependencies, follow these steps:

1. Create a new conda environment:

    ```bash
    conda create -n dreamerv3 python=3.11
    conda activate dreamerv3
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Running the Code

To run the DreamerV3 algorithm, use the following command:

```bash
python dreamer.py
```

### Command-line Arguments

The script supports various command-line arguments to customize the training process. Some key options include:

- `--env`: Specify the environment (default: "CartPole-v1")
- `--episodes`: Number of episodes to train (default: 10000)
- `--latent_dim`: Dimension of latent space (default: 32)
- `--hidden_dim`: Hidden dimension size (default: 512)
- `--actor_lr`: Learning rate for the actor (default: 3e-5)
- `--critic_lr`: Learning rate for the critic (default: 3e-5)
- `--world_lr`: Learning rate for the world model (default: 1e-4)
- `--batch_size`: Batch size for training (default: 250)
- `--seq_len`: Sequence length for training (default: 50)

Example:

```bash
python dreamer.py --env CartPole-v1 --episodes 5000 --batch_size 128
```

For a full list of available options, run:  

```bash
python dreamer.py --help
```

## Acknowledgements

This implementation draws inspiration from the following repositories:

- [Official DreamerV3 JAX Implementation](https://github.com/danijar/dreamerv3)
- [DreamerV3 PyTorch Implementation by NM512](https://github.com/NM512/dreamerv3-torch)

These resources have been invaluable in understanding the DreamerV3 algorithm and creating this more accessible implementation.
