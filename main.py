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
        done = term or trunc

        if training:
            agent.remember(
                {
                    "observation": torch.tensor(observation, dtype=torch.float32),
                    # "next_observation": torch.tensor(
                    #     next_observation, dtype=torch.float32
                    # ),
                    "action": torch.tensor(action, dtype=torch.float32),
                    "reward": torch.tensor(reward, dtype=torch.float32),
                    "terminal": torch.tensor(done, dtype=torch.float32),
                    # "info": info,
                }
            )

        episode_summary["observation"].append(observation)
        # episode_summary["next_observation"].append(next_observation)
        episode_summary["action"].append(action)
        episode_summary["reward"].append(reward)
        episode_summary["terminal"].append(done)
        # episode_summary["info"].append(info)

        observation = next_observation
        if render:
            episode_summary["image"].append(environment.render(mode="rgb_array"))

        pbar.update(config.action_repeat)
        steps += config.action_repeat

    episode_summary["steps"] = [steps]
    return steps, episode_summary


def interact(agent, environment, steps, config, training=True, on_episode_end=None):
    steps_count = 0
    episodes = []

    pbar = tqdm(total=int(float(steps)), postfix=f"Episodes: {len(agent.memory)}")
    while steps_count < float(steps):
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

        pbar.set_postfix_str(f"Episodes: {len(agent.memory)}")
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
            np.asarray(videos).transpose([0, 1, 4, 2, 3]),
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
            agent,
        )
        for vid, name in zip(more_videos, ("gt", "inferred", "generated")):
            # print(vid.shape)
            agent.logger.log_video(
                np.asarray(vid).transpose([0, 1, 4, 2, 3]),
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
        agent.save_checkpoint()
    environment.close()
    return agent


def evaluate_model(observations, actions, agent):
    length = min(len(observations) + 1, 50)

    observations = torch.tensor(observations, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.float32)

    with torch.no_grad():
        _, features, inferred_decoded, *_ = agent.world_model.observe(
            observations[:length].unsqueeze(0).cuda(),
            actions[:length].unsqueeze(0).cuda(),
        )
        # inferred_decoded = inferred_decoded.rsample()
        # print(inferred_decoded.shape, inferred_decoded.type)

        conditioning_length = length // 5

        generated, *_ = agent.world_model.imagine(
            features[:, conditioning_length].cuda(),
            actions=actions[conditioning_length:].unsqueeze(0).cuda(),
        )

        generated_decoded = agent.world_model.decode(generated)
        # generated_decoded = generated_decoded.rsample()
        # print(generated_decoded.shape, generated_decoded.type)
        out = (
            observations[conditioning_length:length].unsqueeze(0),
            inferred_decoded.mean[:, conditioning_length:length].cpu(),
            generated_decoded.mean.cpu(),
        )

        out = [((x + 0.5) * 255).clamp(0, 255).byte() for x in out]

    return out


if __name__ == "__main__":
    config = Config.load_from_yaml("config.yaml")
    env = env_wrappers.make_env(config)
    agent = Dreamer(env.observation_space.shape, env.action_space, config)

    train(config, agent, env)
