import numpy as np
import cv2
import gymnasium as gym
from collections import deque
from utils import preprocess


class AtariEnv:
    def __init__(
        self,
        env_id,
        shape=(64, 64),
        repeat=4,
        clip_rewards=False,
        no_ops=0,
        fire_first=False,
    ):
        base_env = gym.make(env_id, render_mode="rgb_array")
        env = RepeatActionAndMaxFrame(
            base_env, repeat, clip_rewards, no_ops, fire_first
        )
        env = PreprocessFrame(env, shape)
        env = StackFrames(env, repeat)
        self.env = env

    def make(self):
        return self.env


class RepeatActionAndMaxFrame(gym.Wrapper):
    def __init__(self, env, repeat=4, clip_reward=True, no_ops=0, fire_first=False):
        super().__init__(env)
        self.repeat = repeat
        self.clip_reward = clip_reward
        self.no_ops = no_ops
        self.fire_first = fire_first
        self.frame_buffer = np.zeros(
            (2, *env.observation_space.shape), dtype=np.float32
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
        max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])
        return max_frame, total_reward, term, trunc, info

    def reset(self, seed=None, options=None):
        state, info = self.env.reset(seed=seed, options=options)
        no_ops = np.random.randint(self.no_ops) + 1 if self.no_ops > 0 else 0
        for _ in range(no_ops):
            _, _, term, trunc, info = self.env.step(0)
            if term or trunc:
                state, info = self.env.reset()
        if self.fire_first:
            assert self.env.unwrapped.get_action_meanings()[1] == "FIRE"
            state, _, term, trunc, info = self.env.step(1)
        self.frame_buffer = np.zeros(
            (2, *self.env.observation_space.shape), dtype=np.float32
        )
        self.frame_buffer[0] = state
        return state, info


class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env, shape=(64, 64)):
        super().__init__(env)
        self.shape = shape
        self.observation_space = gym.spaces.Box(0.0, 1.0, self.shape, dtype=np.float32)

    def observation(self, state):
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        state = cv2.resize(state, self.shape, interpolation=cv2.INTER_AREA)
        return preprocess(state)


class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, size=4):
        super().__init__(env)
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


ENV_LIST = [
    "AirRaidNoFrameskip-v4",
    "AlienNoFrameskip-v4",
    "AmidarNoFrameskip-v4",
    "AssaultNoFrameskip-v4",
    "AsterixNoFrameskip-v4",
    "AsteroidsNoFrameskip-v4",
    "AtlantisNoFrameskip-v4",
    "BankHeistNoFrameskip-v4",
    "BattleZoneNoFrameskip-v4",
    "BeamRiderNoFrameskip-v4",
    "BerzerkNoFrameskip-v4",
    "BowlingNoFrameskip-v4",
    "BoxingNoFrameskip-v4",
    "BreakoutNoFrameskip-v4",
    "CarnivalNoFrameskip-v4",
    "CentipedeNoFrameskip-v4",
    "ChopperCommandNoFrameskip-v4",
    "CrazyClimberNoFrameskip-v4",
    "DefenderNoFrameskip-v4",
    "DemonAttackNoFrameskip-v4",
    "DoubleDunkNoFrameskip-v4",
    "ElevatorActionNoFrameskip-v4",
    "EnduroNoFrameskip-v4",
    "FishingDerbyNoFrameskip-v4",
    "FreewayNoFrameskip-v4",
    "FrostbiteNoFrameskip-v4",
    "GopherNoFrameskip-v4",
    "GravitarNoFrameskip-v4",
    "HeroNoFrameskip-v4",
    "IceHockeyNoFrameskip-v4",
    "JamesbondNoFrameskip-v4",
    "JourneyEscapeNoFrameskip-v4",
    "KangarooNoFrameskip-v4",
    "KrullNoFrameskip-v4",
    "KungFuMasterNoFrameskip-v4",
    "MontezumaRevengeNoFrameskip-v4",
    "MsPacmanNoFrameskip-v4",
    "NameThisGameNoFrameskip-v4",
    "PhoenixNoFrameskip-v4",
    "PitfallNoFrameskip-v4",
    "PongNoFrameskip-v4",
    "PooyanNoFrameskip-v4",
    "PrivateEyeNoFrameskip-v4",
    "QbertNoFrameskip-v4",
    "RiverraidNoFrameskip-v4",
    "RoadRunnerNoFrameskip-v4",
    "RobotankNoFrameskip-v4",
    "SeaquestNoFrameskip-v4",
    "SkiingNoFrameskip-v4",
    "SolarisNoFrameskip-v4",
    "SpaceInvadersNoFrameskip-v4",
    "StarGunnerNoFrameskip-v4",
    "TennisNoFrameskip-v4",
    "TimePilotNoFrameskip-v4",
    "TutankhamNoFrameskip-v4",
    "UpNDownNoFrameskip-v4",
    "VentureNoFrameskip-v4",
    "VideoPinballNoFrameskip-v4",
    "WizardOfWorNoFrameskip-v4",
    "YarsRevengeNoFrameskip-v4",
    "ZaxxonNoFrameskip-v4",
    "AdventureNoFrameskip-v4",
]
