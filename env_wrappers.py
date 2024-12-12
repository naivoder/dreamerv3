import gym
import numpy as np
from PIL import Image
from dm_control import suite
from gym import Wrapper, ObservationWrapper
from gym.spaces.box import Box
from gym.wrappers import RescaleAction

from utils import preprocess


def make_env(config):
    name = config.task
    episode_length = config.time_limit
    action_repeat = config.action_repeat
    domain, task = name.rsplit(".", 1)
    env = suite.load(domain, task, environment_kwargs={"flat_observation": True})
    env = DeepMindBridge(env)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=episode_length)
    render_kwargs = {"height": 64, "width": 64, "camera_id": 0}
    env = ActionRepeat(env, action_repeat)
    env = RescaleAction(env, -1.0, 1.0)
    env = RenderedObservation(env, (64, 64), render_kwargs)
    return env


class ActionRepeat(Wrapper):
    def __init__(self, env, repeat):
        assert repeat >= 1, "Expects at least one repeat."
        super(ActionRepeat, self).__init__(env)
        self.repeat = repeat

    def step(self, action):
        done = False
        total_reward = 0.0
        current_step = 0
        while current_step < self.repeat and not done:
            obs, reward, term, trunc, info = self.env.step(action)
            total_reward += reward
            current_step += 1
            done = term or trunc
        return obs, total_reward, term, trunc, info  # noqa


class RenderedObservation(ObservationWrapper):
    def __init__(self, env, image_size, render_kwargs):
        super(RenderedObservation, self).__init__(env)
        self._size = image_size
        self.observation_space = Box(-0.5, 0.5, image_size + (3,), np.float32)
        self._render_kwargs = render_kwargs

    def observation(self, _):
        image = self.env.render(**self._render_kwargs)
        image = Image.fromarray(image)
        if image.size != self._size:
            image = image.resize(self._size, Image.BILINEAR)
        image = np.array(image, copy=False).moveaxis(2, 0)
        image = np.clip(image, 0, 255).astype(np.float32)
        return preprocess(image)


class DeepMindBridge(gym.Env):
    def __init__(self, env):
        self._env = env

    def step(self, action):
        time_step = self._env.step(action)
        obs = time_step.observation["observations"]
        reward = time_step.reward or 0
        done = time_step.last()
        return obs, reward, done, done, {}

    @property
    def action_space(self):
        spec = self._env.action_spec()
        return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

    @property
    def observation_space(self):
        spec = self._env.observation_spec()["observations"]
        return gym.spaces.Box(-np.inf, np.inf, spec.shape, dtype=spec.dtype)

    def render(self, mode="human", **kwargs):
        if "camera_id" not in kwargs.keys():
            kwargs["camera_id"] = 0
        return self._env.physics.render(**kwargs)

    def reset(self):
        time_step = self._env.reset()
        obs = time_step.observation["observations"]
        return obs, {}

    def seed(self, seed=None):
        self._env.task.random.seed(seed)
