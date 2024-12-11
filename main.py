import os
import pathlib
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm
import warnings

from utils import Config
import env_wrappers
from dreamer import Dreamer

warnings.filterwarnings("ignore")


def do_episode(agent, training, environment, config, pbar, render):
    episode_summary = defaultdict(list)
    steps = 0
    done = False
    observation, _ = environment.reset()

    while not done:
        action = agent(torch.tensor(observation, dtype=torch.float32), training)
        next_observation, reward, term, trunc, info = environment.step(action)
        terminal = term or trunc

        if training:
            agent.observe(
                {
                    "observation": torch.tensor(observation, dtype=torch.float32),
                    "next_observation": torch.tensor(
                        next_observation, dtype=torch.float32
                    ),
                    "action": torch.tensor(action, dtype=torch.float32),
                    "reward": torch.tensor(reward, dtype=torch.float32),
                    "terminal": torch.tensor(terminal, dtype=torch.float32),
                    "info": info,
                }
            )

        episode_summary["observation"].append(observation)
        episode_summary["next_observation"].append(next_observation)
        episode_summary["action"].append(action)
        episode_summary["reward"].append(reward)
        episode_summary["terminal"].append(terminal)
        episode_summary["info"].append(info)

        observation = next_observation
        if render:
            episode_summary["image"].append(environment.render(mode="rgb_array"))

        pbar.update(config.action_repeat)
        steps += config.action_repeat

    episode_summary["steps"] = [steps]
    return steps, episode_summary


def interact(agent, environment, steps, config, training=True, on_episode_end=None):
    pbar = tqdm(total=steps)
    steps_count = 0
    episodes = []

    while steps_count < steps:
        episode_steps, episode_summary = do_episode(
            agent,
            training,
            environment,
            config,
            pbar,
            len(episodes) < config.render_episodes and not training,
        )
        steps_count += episode_steps
        episodes.append(episode_summary)

        if on_episode_end:
            on_episode_end(episode_summary, steps_count)

    pbar.close()
    return steps, episodes


def make_summary(summaries, prefix):
    epoch_summary = {
        f"{prefix}/average_return": np.mean(
            [sum(episode["reward"]) for episode in summaries]
        ),
        f"{prefix}/average_episode_length": np.mean(
            [episode["steps"][0] for episode in summaries]
        ),
    }
    return epoch_summary


def evaluate(agent, train_env, config, steps):
    evaluation_steps, evaluation_episodes_summaries = interact(
        agent, train_env, config.evaluation_steps_per_epoch, config, training=False
    )

    if config.render_episodes:
        videos = [
            episode.get("image")
            for episode in evaluation_episodes_summaries[: config.render_episodes]
        ]
        agent.logger.log_video(
            np.array(videos, copy=False).transpose([0, 1, 4, 2, 3]),
            steps,
            name="videos/overview",
        )

    if config.evaluate_model:
        more_videos = evaluate_model(
            torch.tensor(
                evaluation_episodes_summaries[0]["observation"], dtype=torch.float32
            ),
            torch.tensor(
                evaluation_episodes_summaries[0]["action"], dtype=torch.float32
            ),
            agent.model,
            agent.model.parameters(),
        )
        for vid, name in zip(more_videos, ("gt", "inferred", "generated")):
            agent.logger.log_video(
                np.array(vid, copy=False).transpose([0, 1, 4, 2, 3]),
                steps,
                name=f"videos/{name}",
            )

    return make_summary(evaluation_episodes_summaries, "evaluation")


def on_episode_end(episode_summary, logger, global_step, steps_count):
    episode_return = sum(episode_summary["reward"])
    steps = global_step + steps_count
    print(f"\nFinished episode with return: {episode_return}")
    summary = {"training/episode_return": episode_return}
    logger.log_evaluation_summary(summary, steps)


def train(config, agent, environment):
    steps = 0

    # if pathlib.Path(config.log_dir, "agent_data.pt").exists():
    #     agent.load(os.path.join(config.log_dir, "agent_data.pt"))
    #     steps = float(agent.training_step)
    #     print(f"Loaded {steps} steps. Continuing training from {config.log_dir}")

    while steps < float(config.steps):
        print("Performing a training epoch.")
        training_steps, training_episodes_summaries = interact(
            agent,
            environment,
            float(config.training_steps_per_epoch),
            config,
            training=True,
            on_episode_end=lambda summary, count: on_episode_end(
                summary, agent.logger, steps, count
            ),
        )
        steps += training_steps
        training_summary = make_summary(training_episodes_summaries, "training")

        if config.evaluation_steps_per_epoch:
            print("Evaluating.")
            evaluation_summaries = evaluate(agent, environment, config, steps)
            training_summary.update(evaluation_summaries)

        agent.logger.log_evaluation_summary(training_summary, steps)
        agent.save(os.path.join(config.log_dir, "agent_data.pt"))

    environment.close()
    return agent


def evaluate_model(observations, actions, model, model_params):
    length = min(len(observations) + 1, 50)

    observations, actions = [
        torch.tensor(x, dtype=torch.float32) for x in (observations, actions)
    ]

    with torch.no_grad():
        _, features, inferred_decoded, *_ = model.infer(
            model_params,
            observations[:length].unsqueeze(0),
            actions[:length].unsqueeze(0),
        )

        conditioning_length = length // 5

        generated, *_ = model.generate_sequence(
            model_params,
            features[:, conditioning_length],
            actions=actions[conditioning_length:].unsqueeze(0),
        )

        generated_decoded = model.decode(model_params, generated)

        out = (
            observations[conditioning_length:length].unsqueeze(0),
            inferred_decoded.mean(dim=0)[:, conditioning_length:length],
            generated_decoded.mean(dim=0),
        )

        out = [((x + 0.5) * 255).clamp(0, 255).byte() for x in out]

    return out


if __name__ == "__main__":
    config = Config.load_from_yaml("config.yaml")
    env = env_wrappers.make_env(config)
    agent = Dreamer(env.observation_space.shape, env.action_space, config)

    train(config, agent, env)