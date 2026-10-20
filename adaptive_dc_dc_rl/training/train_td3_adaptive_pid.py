"""Train the TD3 adaptive-PID tuner (TD3 outputs normalised gain adjustments).

    python training/train_td3_adaptive_pid.py --timesteps 5000 --name td3_adaptive_pid_smoke
    python training/train_td3_adaptive_pid.py                       # full run (config.py)

Same TD3 hyper-parameters as the direct controller (config.TD3_*); the agent only
chooses 3 numbers in [-1, 1] every 200 us, mapped to bounded PID gains
(controllers/adaptive_pid.py).  The PID produces the duty cycle.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from training import train_td3


def train(total_timesteps=config.TD3_TOTAL_TIMESTEPS, seed=config.TD3_SEED,
          model_name=config.ADAPT_MODEL_NAME, learning_starts=config.TD3_LEARNING_STARTS,
          tensorboard=True, device="cpu", verbose=0):
    from environment.adaptive_pid_env import AdaptivePIDEnv
    extra = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(config).items()
             if k.startswith("ADAPT_")}
    return train_td3.train(total_timesteps, seed, model_name, learning_starts,
                           tensorboard=tensorboard, device=device, verbose=verbose,
                           env_factory=AdaptivePIDEnv, extra_info=extra)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--timesteps", type=int, default=config.TD3_TOTAL_TIMESTEPS)
    p.add_argument("--seed", type=int, default=config.TD3_SEED)
    p.add_argument("--name", default=config.ADAPT_MODEL_NAME)
    p.add_argument("--learning-starts", type=int, default=config.TD3_LEARNING_STARTS)
    p.add_argument("--no-tensorboard", action="store_true")
    a = p.parse_args(argv)
    train(a.timesteps, a.seed, a.name, a.learning_starts, tensorboard=not a.no_tensorboard)


if __name__ == "__main__":
    main()
