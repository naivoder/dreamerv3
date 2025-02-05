# DreamerV3

🚫🚧👷‍♀️ Warning: Under Construction 👷‍♂️🚧🚫

This repository contains a PyTorch implementation of the DreamerV3 algorithm, aimed at providing a more readable and accessible version compared to the official implementations. 

## Overview

DreamerV3 is a model-based reinforcement learning algorithm that learns a world model of the environment dynamics, and uses it to train an actor-critic policy from imagined trajectories. The algorithm consists of several key components:

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

```bash
python dreamerv3.py
```

### Command-line Arguments

- `--env`: Specify the environment
- `--episodes`: Number of episodes to train (default: 10000)


Example:

```bash
python dreamerv3.py --env CartPole-v1 --episodes 5000 
```

For a full list of available options, run:  

```bash
python dreamerv3.py --help
```

## Acknowledgements

This implementation draws inspiration from the following repositories:

- [Official DreamerV3 JAX Implementation](https://github.com/danijar/dreamerv3)
- [DreamerV3 PyTorch Implementation by NM512](https://github.com/NM512/dreamerv3-torch)

These resources have been invaluable in understanding the DreamerV3 algorithm and creating this more accessible implementation.
