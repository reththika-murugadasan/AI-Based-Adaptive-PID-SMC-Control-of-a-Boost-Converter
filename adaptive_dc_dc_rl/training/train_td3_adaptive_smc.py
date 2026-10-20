"""Train the TD3 adaptive-SMC tuner (TD3 outputs normalised parameter adjustments).

    python training/train_td3_adaptive_smc.py --timesteps 5000 --name td3_adaptive_smc_smoke
    python training/train_td3_adaptive_smc.py                       # full run (config.py)

Same TD3 hyper-parameters as the other agents (config.TD3_*).  The agent picks 4 numbers in
[-1, 1] every 200 us, mapped to bounded SMC parameters (controllers/adaptive_smc.py); the
unchanged SMC law produces the duty cycle.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from training import train_td3


def train(total_timesteps=config.TD3_TOTAL_TIMESTEPS, seed=config.TD3_SEED,
          model_name=config.ASMC_MODEL_NAME, learning_starts=config.TD3_LEARNING_STARTS,
          tensorboard=True, device="cpu", verbose=0):
    from environment.adaptive_smc_env import AdaptiveSMCEnv
    extra = {k: v for k, v in vars(config).items() if k.startswith("ASMC_")}
    return train_td3.train(total_timesteps, seed, model_name, learning_starts,
                           tensorboard=tensorboard, device=device, verbose=verbose,
                           env_factory=AdaptiveSMCEnv, extra_info=extra)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--timesteps", type=int, default=config.TD3_TOTAL_TIMESTEPS)
    p.add_argument("--seed", type=int, default=config.TD3_SEED)
    p.add_argument("--name", default=config.ASMC_MODEL_NAME)
    p.add_argument("--learning-starts", type=int, default=config.TD3_LEARNING_STARTS)
    p.add_argument("--no-tensorboard", action="store_true")
    a = p.parse_args(argv)
    train(a.timesteps, a.seed, a.name, a.learning_starts, tensorboard=not a.no_tensorboard)


if __name__ == "__main__":
    main()
