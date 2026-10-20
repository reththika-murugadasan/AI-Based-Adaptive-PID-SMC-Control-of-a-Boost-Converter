"""Convenient entry point for Stage 5.

    python experiments/run_td3.py --mode both --timesteps 5000 --name td3_smoke
    python experiments/run_td3.py --mode eval  --name td3_boost
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from training import evaluate_td3, train_td3


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "eval", "both"], default="both")
    p.add_argument("--timesteps", type=int, default=config.TD3_TOTAL_TIMESTEPS)
    p.add_argument("--seed", type=int, default=config.TD3_SEED)
    p.add_argument("--name", default=config.TD3_MODEL_NAME)
    p.add_argument("--learning-starts", type=int, default=config.TD3_LEARNING_STARTS)
    a = p.parse_args(argv)
    if a.mode in ("train", "both"):
        train_td3.train(a.timesteps, a.seed, a.name, a.learning_starts)
    if a.mode in ("eval", "both"):
        evaluate_td3.main(["--name", a.name])


if __name__ == "__main__":
    main()
