# DreamerV3 PyTorch Implementation

🚫🚧👷‍♀️ Warning: Under Construction 👷‍♂️🚧🚫

A PyTorch implementation of DreamerV3, a state-of-the-art model-based reinforcement learning algorithm that achieves strong performance across diverse domains.

## Overview

DreamerV3 is a general-purpose reinforcement learning algorithm that learns a world model to enable planning in latent space. This implementation provides a clean, modular, and educational version of the algorithm with the following key components:

1. **World Model (RSSM)**: Encodes observations into discrete latent representations and predicts future states, rewards, and episode continuations
2. **Actor**: Learns a policy to maximize expected returns in imagined trajectories
3. **Critic**: Estimates state values using λ-returns for variance reduction
4. **Imagination**: Generates trajectories using the learned world model for policy improvement

## Discrepancies with Official Implementation

This implementation makes several simplifications compared to the official DreamerV3:

- **AdamW optimizer** instead of custom LaProp optimizer
- **LayerNorm** instead of RMSNorm for layer normalization
- **No action repeat** - currently always uses action repeat of 1
- **Standard GRU** instead of BlockGRU for the recurrent state-space model

## To Do

[ ] Implement custom Atari environment wrapper for proper preprocessing  
[ ] Add action repeat functionality  
[ ] Implement BlockGRU for improved efficiency  
[ ] Add RMSNorm as an option instead of LayerNorm  
[ ] Add LaProp as an option instead of AdamW  

## Installation

Create and activate the conda environment:
```bash
conda env create -f environment.yaml
conda activate gym-pytorch
```

## Usage


Train DreamerV3 on selected environment:
```bash
python main.py
```

### Supported Environments

The implementation has been tested on:
- **Classic Control**: CartPole-v1, Pendulum-v1, LunarLander-v3
- **Continuous Control**: BipedalWalker-v3, Ant-v5
- **Vision-Based**: CarRacing-v3, Atari games (Pong, etc.)

### Configuration

Key hyperparameters can be modified in the `Config` dataclass:
- `batch_size`: 16
- `sequence_length`: 64  
- `replay_ratio`: 128 (env steps per gradient step)
- `buffer_size`: 5M transitions
- Model dimensions, learning rates, loss weights, etc.


## Outputs

- **TensorBoard Logs**: Saved to `runs/` directory
- **Trained Models**: Best models saved to `models/`
- **Evaluation GIFs**: Visualizations saved to `gifs/`

## Monitoring Training

Launch TensorBoard to monitor training progress:
```bash
tensorboard --logdir runs
```


## Acknowledgements

This implementation draws inspiration from the following repositories:

- [Official DreamerV3 JAX Implementation](https://github.com/danijar/dreamerv3)
- [DreamerV3 PyTorch Implementation by NM512](https://github.com/NM512/dreamerv3-torch)

These resources have been invaluable in understanding the DreamerV3 algorithm and creating this more accessible implementation.

## Citation

If you use this implementation, please cite the original DreamerV3 paper:
```bibtex
@article{hafner2023dreamerv3,
  title={Mastering Diverse Domains through World Models},
  author={Hafner, Danijar and Pasukonis, Jurgis and Ba, Jimmy and Lillicrap, Timothy},
  journal={arXiv preprint arXiv:2301.04104},
  year={2023}
}
```