"""Train the direct TD3 voltage controller (Stable-Baselines3).

    python training/train_td3.py --timesteps 5000 --name td3_smoke      # smoke test
    python training/train_td3.py                                         # full run (config.py)

Why TD3: continuous action (duty), off-policy and sample-efficient, clipped
double-Q critics curb Q over-estimation, delayed actor updates and target-policy
smoothing stabilise learning on a plant with a sharp, lightly damped response.

Hyper-parameters (all in config.py):
  learning_rate  Adam step size for actor and critics.
  buffer_size    replay buffer capacity (transitions).
  batch_size     minibatch per gradient step.
  gamma          discount; 0.99 -> ~100-step (20 ms) effective horizon.
  tau            soft target update rate (small = slow, stable targets).
  policy_delay   actor + target update once per N critic updates (TD3 trick 2).
  target_policy_noise / clip   smoothing noise added to target actions (trick 3).
  exploration noise  Gaussian N(0, sigma) added to actions during data collection.
  net_arch       [64, 64] MLPs: the plant is 2nd order, a big network is unnecessary.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


class RewardStatsCallback:
    """Built lazily so importing this module does not require SB3."""

    def __new__(cls, every=config.TD3_LOG_EPISODES):
        from stable_baselines3.common.callbacks import BaseCallback

        class _CB(BaseCallback):
            def __init__(self):
                super().__init__()
                self.every, self.buf, self.n_ep, self.n_fail = every, [], 0, 0
                self.history = []

            def _on_step(self):
                for info in self.locals.get("infos", []):
                    st = info.get("episode_stats")
                    if st is None:
                        continue
                    self.n_ep += 1
                    self.n_fail += int(info.get("failure") is not None)
                    self.buf.append(st)
                    if len(self.buf) >= self.every:
                        self._report()
                return True

            def _report(self):
                ret = np.array([s["return_"] for s in self.buf])
                length = np.array([s["length"] for s in self.buf])
                terms = {k: float(np.mean([s["terms"][k] for s in self.buf]))
                         for k in self.buf[0]["terms"]}
                rec = dict(timesteps=self.num_timesteps, episodes=self.n_ep,
                           return_mean=float(ret.mean()), return_std=float(ret.std()),
                           return_min=float(ret.min()), return_max=float(ret.max()),
                           length_mean=float(length.mean()), failures_total=self.n_fail, **terms)
                self.history.append(rec)
                self.logger.record("reward/return_mean", rec["return_mean"])
                for k, v in terms.items():
                    self.logger.record(f"reward/term_{k}", v)
                self.logger.record("env/episode_length", rec["length_mean"])
                self.logger.record("env/failures_total", self.n_fail)
                tstr = " ".join(f"{k} {v:.1f}" for k, v in terms.items())
                print(f"[{self.num_timesteps:>8d} steps | ep {self.n_ep:>4d}] return "
                      f"{rec['return_mean']:8.1f} (std {rec['return_std']:.1f}, "
                      f"min {rec['return_min']:.1f}, max {rec['return_max']:.1f}) | len "
                      f"{rec['length_mean']:.0f} | {tstr} | trips {self.n_fail}", flush=True)
                self.buf = []

        return _CB()


def make_env(seed, env_factory=None):
    from stable_baselines3.common.monitor import Monitor
    from environment.boost_env import BoostConverterEnv
    return Monitor((env_factory or BoostConverterEnv)())


def train(total_timesteps=config.TD3_TOTAL_TIMESTEPS, seed=config.TD3_SEED,
          model_name=config.TD3_MODEL_NAME, learning_starts=config.TD3_LEARNING_STARTS,
          tensorboard=True, device="cpu", verbose=0, env_factory=None, extra_info=None):
    import stable_baselines3 as sb3
    import torch
    from stable_baselines3 import TD3
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.noise import NormalActionNoise

    env = make_env(seed, env_factory)
    check_env(env.unwrapped, warn=True)                       # 2. validate environment
    env.reset(seed=seed)
    n_act = env.action_space.shape[0]
    noise = NormalActionNoise(mean=np.zeros(n_act),
                              sigma=config.TD3_EXPLORATION_SIGMA * np.ones(n_act))
    learning_starts = min(learning_starts, max(total_timesteps // 4, 1))
    model = TD3(                                               # 3. create model
        "MlpPolicy", env, seed=seed, device=device, verbose=verbose,
        learning_rate=config.TD3_LEARNING_RATE, buffer_size=config.TD3_BUFFER_SIZE,
        batch_size=config.TD3_BATCH_SIZE, gamma=config.TD3_GAMMA, tau=config.TD3_TAU,
        policy_delay=config.TD3_POLICY_DELAY,
        target_policy_noise=config.TD3_TARGET_POLICY_NOISE,
        target_noise_clip=config.TD3_TARGET_NOISE_CLIP,
        learning_starts=learning_starts, train_freq=config.TD3_TRAIN_FREQ,
        gradient_steps=config.TD3_GRADIENT_STEPS, action_noise=noise,
        policy_kwargs=dict(net_arch=list(config.TD3_NET_ARCH)),
        tensorboard_log=str(config.TENSORBOARD_DIR) if tensorboard else None)

    cb = RewardStatsCallback()
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=cb, tb_log_name=model_name,
                progress_bar=False)                            # 4. train
    wall = time.time() - t0

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.MODELS_DIR / f"{model_name}.zip"
    model.save(path)                                           # 5. save
    info = dict(model=str(path), total_timesteps=int(total_timesteps), seed=seed,
                learning_starts=learning_starts, wall_time_s=round(wall, 1),
                action_mapping="a in [-1,1] -> D = 0.05 + (a+1)/2*0.90" if env_factory is None
                else "see extra_info",
                extra_info=extra_info or {},
                env=type(env.unwrapped).__name__, obs_shape=list(env.observation_space.shape),
                action_shape=list(env.action_space.shape),
                hyperparameters={k: (v if not isinstance(v, Path) else str(v))
                                 for k, v in vars(config).items() if k.startswith(("TD3_", "ENV_"))},
                versions=dict(stable_baselines3=sb3.__version__, torch=torch.__version__,
                              numpy=np.__version__),
                reward_history=cb.history, episodes=cb.n_ep, trips=cb.n_fail)
    (config.MODELS_DIR / f"{model_name}_info.json").write_text(json.dumps(info, indent=2))
    print(f"Saved {path}  ({wall:.0f} s, {cb.n_ep} episodes, {cb.n_fail} protection trips)")
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--timesteps", type=int, default=config.TD3_TOTAL_TIMESTEPS)
    p.add_argument("--seed", type=int, default=config.TD3_SEED)
    p.add_argument("--name", default=config.TD3_MODEL_NAME)
    p.add_argument("--learning-starts", type=int, default=config.TD3_LEARNING_STARTS)
    p.add_argument("--no-tensorboard", action="store_true")
    a = p.parse_args(argv)
    train(a.timesteps, a.seed, a.name, a.learning_starts, tensorboard=not a.no_tensorboard)


if __name__ == "__main__":
    main()
