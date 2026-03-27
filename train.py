"""Training loop for CARLA DDPG."""
from __future__ import annotations

import os
import random

import numpy as np
import torch

from agent.ddpg_agent import DDPGAgent
from environment.carla_env import CarlaEnv
from utils.config import Config
from utils.logger import Logger


def evaluate(env: CarlaEnv, agent: DDPGAgent, episodes: int) -> float:
    rewards = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_r = 0.0
        while not done:
            action = agent.select_action(obs, add_noise=False)
            obs, reward, done, _, _ = env.step(action)
            ep_r += reward
        rewards.append(ep_r)
    return float(np.mean(rewards))


def main() -> None:
    cfg = Config()
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

    env = CarlaEnv(cfg)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = DDPGAgent(obs_dim, action_dim, cfg)
    logger = Logger(cfg.LOG_DIR, os.path.join(cfg.LOG_DIR, "metrics.csv"))

    total_steps = 0
    best_eval = -1e9

    try:
        for ep in range(1, cfg.TRAIN_EPISODES + 1):
            obs, _ = env.reset()
            agent.noise.reset()
            done = False
            ep_reward = 0.0
            ep_overtakes = 0
            while not done:
                if total_steps < cfg.WARMUP_STEPS:
                    action = env.action_space.sample()
                else:
                    action = agent.select_action(obs, add_noise=True)

                next_obs, reward, done, _, info = env.step(action)
                agent.replay_buffer.add(obs, action, reward, next_obs, done)
                actor_loss, critic_loss = agent.update()

                obs = next_obs
                ep_reward += reward
                total_steps += 1
                if info.get("overtake_complete"):
                    ep_overtakes += 1

            metrics = {
                "episode_reward": ep_reward,
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
                "overtakes": ep_overtakes,
            }
            logger.log_episode(metrics, ep)
            print(f"Episode {ep} | reward {ep_reward:.1f} | overtakes {ep_overtakes}")

            if ep % cfg.EVAL_EVERY == 0:
                eval_reward = evaluate(env, agent, cfg.EVAL_EPISODES)
                logger.log_scalar("eval_reward", eval_reward, ep)
                print(f"Eval @ {ep}: {eval_reward:.1f}")
                if eval_reward > best_eval:
                    best_eval = eval_reward
                    agent.save(cfg.BEST_MODEL_PATH)
                    print("Saved best model.")

    finally:
        logger.close()
        env.close()


if __name__ == "__main__":
    main()