import gymnasium as gym
import numpy as np
from collections import deque
import cv2


class RepeatActionAndMaxFrame(gym.Wrapper):
    def __init__(self, env, repeat=4, clip_reward=True, no_ops=0, fire_first=False):
        super(RepeatActionAndMaxFrame, self).__init__(env)
        self.env = env
        self.repeat = repeat
        self.clip_reward = clip_reward
        self.no_ops = no_ops
        self.fire_first = fire_first
        self.frame_buffer = np.zeros(
            (2, *self.env.observation_space.shape), dtype=np.float32
        )

    def step(self, action):
        total_reward = 0
        term, trunc = False, False

        for i in range(self.repeat):
            state, reward, term, trunc, info = self.env.step(action)

            if self.clip_reward:
                reward = np.clip(reward, -1, 1)

            total_reward += reward
            self.frame_buffer[i % 2] = state

            if term or trunc:
                break

        # max_frame = np.max(self.frame_buffer, axis=0)
        max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])
        return max_frame, total_reward, term, trunc, info

    def reset(self, seed=None, options=None):
        state, info = self.env.reset(seed=seed, options=options)

        no_ops = np.random.randint(self.no_ops) + 1 if self.no_ops > 0 else 0
        for _ in range(no_ops):
            _, _, term, trunc, info = self.env.step(0)
            if term or trunc:
                _, _ = self.env.reset()

        if self.fire_first:
            assert self.env.unwrapped.get_action_meanings()[1] == "FIRE"
            state, _, _, _, _ = self.env.step(1)

        self.frame_buffer = np.zeros(
            (2, *self.env.observation_space.shape), dtype=np.float32
        )
        self.frame_buffer[0] = state

        return state, info


class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env, shape=(84, 84)):
        super(PreprocessFrame, self).__init__(env)
        self.shape = shape
        self.observation_space = gym.spaces.Box(0.0, 1.0, self.shape, dtype=np.float32)

    def observation(self, state):
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        state = cv2.resize(state, self.shape, interpolation=cv2.INTER_AREA)
        return state / 255.0


class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, size=4):
        super(StackFrames, self).__init__(env)
        self.size = int(size)
        self.stack = deque([], maxlen=self.size)

        shape = self.env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            0.0, 1.0, (self.size, *shape), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        state, info = self.env.reset(seed=seed, options=options)
        self.stack = deque([state] * self.size, maxlen=self.size)
        return np.array(self.stack), info

    def observation(self, state):
        self.stack.append(state)
        return np.array(self.stack)


class AtariEnv:
    def __init__(
        self,
        env,
        shape=(84, 84),
        repeat=4,
        clip_rewards=False,
        no_ops=0,
        fire_first=False,
    ):
        self.env = gym.make(env, render_mode="rgb_array")
        self.env = RepeatActionAndMaxFrame(
            self.env, repeat, clip_rewards, no_ops, fire_first
        )
        self.env = PreprocessFrame(self.env, shape)
        self.env = StackFrames(self.env, repeat)

    def make(self):
        return self.env


if __name__ == "__main__":
    env = AtariEnv("ALE/Pong-v5").make()
    state, _ = env.reset()

    print("Expected Shape:", env.observation_space.shape)
    print("Actual Shape:", state.shape)
