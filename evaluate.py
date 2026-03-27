"""Inference script for trained DDPG agent."""
from __future__ import annotations

import time

from agent.ddpg_agent import DDPGAgent
from environment.carla_env import CarlaEnv
from utils.config import Config


def main() -> None:
    cfg = Config()
    env = CarlaEnv(cfg)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = DDPGAgent(obs_dim, action_dim, cfg)
    agent.load(cfg.BEST_MODEL_PATH)

    try:
        obs, _ = env.reset()
        done = False
        while True:
            action = agent.select_action(obs, add_noise=False)
            obs, reward, done, _, info = env.step(action)
            print(f"step reward={reward:.3f} speed={info['speed_kmh']:.1f}km/h")
            if done:
                obs, _ = env.reset()
            time.sleep(cfg.FIXED_DELTA_SECONDS)
    finally:
        env.close()


if __name__ == "__main__":
    main()