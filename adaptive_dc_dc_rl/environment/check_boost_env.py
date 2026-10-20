"""Stage 4 smoke test for BoostConverterEnv (no learning, no neural network)."""
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from environment.boost_env import BoostConverterEnv, duty_to_action


def main():
    env = BoostConverterEnv()
    print("observation_space:", env.observation_space)
    print("action_space     :", env.action_space)

    # --- official API checks -------------------------------------------------
    from gymnasium.utils.env_checker import check_env as gym_check_env
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        gym_check_env(env.unwrapped, skip_render_check=True)
    print(f"gymnasium check_env : PASSED ({len(w)} warnings)")
    for x in w:
        print("   warning:", x.message)
    from stable_baselines3.common.env_checker import check_env as sb3_check_env
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sb3_check_env(BoostConverterEnv(), warn=True)
    print(f"stable-baselines3 check_env : PASSED ({len(w)} warnings)")
    for x in w:
        print("   warning:", x.message)

    # --- reset / random rollout ----------------------------------------------
    obs, info = env.reset(seed=0)
    print("\nreset(): obs shape", obs.shape, "dtype", obs.dtype)
    print("         obs  =", np.round(obs, 4))
    print("         info =", {k: (round(v, 3) if isinstance(v, float) else v)
                              for k, v in info.items() if k != "reward_terms"})
    assert env.observation_space.contains(obs)

    a = env.action_space.sample()
    print("action shape:", a.shape, "dtype", a.dtype, "sample", a)

    total, n_steps, finite = 0.0, 0, True
    rewards = []
    rng_env = BoostConverterEnv()
    for ep in range(3):
        obs, info = rng_env.reset(seed=ep)
        done = False
        while not done:
            act = rng_env.action_space.sample()
            obs, r, term, trunc, info = rng_env.step(act)
            done = term or trunc
            n_steps += 1
            rewards.append(r)
            finite &= bool(np.all(np.isfinite(obs)) and np.isfinite(r))
            assert rng_env.observation_space.contains(obs)
            if n_steps in (1, 2, 100):
                print(f"step {n_steps:4d}: reward = {r:+.4f}  obs = {np.round(obs, 3)}  "
                      f"Vout = {info['Vout']:.2f} V  D = {info['duty']:.3f}")
        print(f"episode {ep}: init={info['init_mode']:11s} length={info['episode_stats']['length']:4d} "
              f"return={info['episode_stats']['return_']:8.2f} terminated={term} "
              f"failure={info['failure']}")
    rewards = np.array(rewards)
    print(f"\n{n_steps} random steps | reward mean {rewards.mean():+.3f} min {rewards.min():+.3f} "
          f"max {rewards.max():+.3f}")
    print("all observations/rewards finite:", finite)

    # --- reference controller sanity: constant duty D = 1 - Vin/Vref -----------
    env2 = BoostConverterEnv(disturbance_prob=0.0)
    obs, info = env2.reset(seed=1, options=dict(vin=12.0, load=20.0, vref=24.0, init="cold"))
    ret = 0.0
    for _ in range(env2.max_steps):
        obs, r, term, trunc, info = env2.step(duty_to_action(0.5))
        ret += r
        if term or trunc:
            break
    print(f"\nfixed D=0.5 from cold start: final Vout = {info['Vout']:.2f} V, return = {ret:.1f}")
    assert finite
    print("\nENVIRONMENT CHECK: PASSED")


if __name__ == "__main__":
    main()
