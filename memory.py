import torch
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, obs_shape, act_dim, n_envs, device):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.act_dim = act_dim
        self.n_envs = n_envs
        self.device = device

        # Initialize buffers for each environment
        self.obs_buffers = [
            np.zeros((capacity, *obs_shape), dtype=np.float32) for _ in range(n_envs)
        ]
        self.act_buffers = [np.zeros(capacity, dtype=np.int64) for _ in range(n_envs)]
        self.rew_buffers = [np.zeros(capacity, dtype=np.float32) for _ in range(n_envs)]
        self.next_obs_buffers = [
            np.zeros((capacity, *obs_shape), dtype=np.float32) for _ in range(n_envs)
        ]
        self.done_buffers = [
            np.zeros(capacity, dtype=np.float32) for _ in range(n_envs)
        ]
        self.ptrs = [0 for _ in range(n_envs)]
        self.sizes = [0 for _ in range(n_envs)]

    def store(self, env_idx, obs, act, rew, next_obs, done):
        ptr = self.ptrs[env_idx]
        self.obs_buffers[env_idx][ptr] = obs
        self.act_buffers[env_idx][ptr] = act
        self.rew_buffers[env_idx][ptr] = rew
        self.next_obs_buffers[env_idx][ptr] = next_obs
        self.done_buffers[env_idx][ptr] = done

        self.ptrs[env_idx] = (ptr + 1) % self.capacity
        self.sizes[env_idx] = min(self.sizes[env_idx] + 1, self.capacity)

    def sample_batch(self, batch_size, seq_len):
        obs_batch = []
        act_batch = []
        rew_batch = []
        next_obs_batch = []
        done_batch = []

        for _ in range(batch_size):
            # Randomly select an environment
            env_idx = np.random.randint(0, self.n_envs)
            size = self.sizes[env_idx]

            if size < seq_len:
                continue  # Skip if not enough data in this environment

            # Attempt to find a valid sequence
            valid_sequence_found = False
            attempt_limit = 100  # Prevent infinite loops
            attempts = 0

            while not valid_sequence_found and attempts < attempt_limit:
                attempts += 1
                max_start_idx = size - seq_len
                if max_start_idx <= 0:
                    break  # Not enough data

                start_idx = np.random.randint(0, max_start_idx)
                end_idx = start_idx + seq_len

                # Handle circular buffer wrap-around
                indices = (np.arange(start_idx, end_idx) % self.capacity).astype(int)

                # Check for episode termination within the sequence (excluding last step)
                done_seq = self.done_buffers[env_idx][indices[:-1]]
                if np.any(done_seq):
                    continue  # Sequence crosses episode boundary, try another
                else:
                    valid_sequence_found = True

            if not valid_sequence_found:
                continue  # Skip if no valid sequence found

            obs_seq = self.obs_buffers[env_idx][indices]
            act_seq = self.act_buffers[env_idx][indices]
            rew_seq = self.rew_buffers[env_idx][indices]
            next_obs_seq = self.next_obs_buffers[env_idx][indices]
            done_seq = self.done_buffers[env_idx][indices]

            obs_batch.append(obs_seq)
            act_batch.append(act_seq)
            rew_batch.append(rew_seq)
            next_obs_batch.append(next_obs_seq)
            done_batch.append(done_seq)

        if len(obs_batch) == 0:
            raise ValueError("Not enough data to sample a batch.")

        obs_batch = np.stack(obs_batch)
        act_batch = np.stack(act_batch)
        rew_batch = np.stack(rew_batch)
        next_obs_batch = np.stack(next_obs_batch)
        done_batch = np.stack(done_batch)

        return (
            torch.as_tensor(obs_batch, dtype=torch.float32).to(self.device),
            torch.as_tensor(act_batch, dtype=torch.long).to(self.device),
            torch.as_tensor(rew_batch, dtype=torch.float32).to(self.device),
            torch.as_tensor(next_obs_batch, dtype=torch.float32).to(self.device),
            torch.as_tensor(done_batch, dtype=torch.float32).to(self.device),
        )

    def __len__(self):
        return sum(self.sizes)
