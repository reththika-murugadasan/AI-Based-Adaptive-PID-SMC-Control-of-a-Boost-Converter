"""Reward diagnostics: run reference policies through BoostConverterEnv and report
the return under the CURRENT reward plus reward-independent behaviour metrics.

    python experiments/reward_diagnostics.py --tag v1

Reference policies (all act every 200 us, the env's control rate):
  fixed_ideal : constant duty D = 1 - Vin/Vref (open loop, uses privileged Vin)
  pid         : the Stage-2 PID gains (privileged access to Vref/Vout)
  random      : uniform random actions
  <td3 models>: any saved smoke/trained model passed via --models
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from controllers.pid import PIDController
from environment.boost_env import BoostConverterEnv, duty_to_action

warnings.filterwarnings("ignore")
N_EPISODES = 20


def policy_fixed(env, obs):
    c = env.converter
    return duty_to_action(1.0 - c.Vin / c.Vref)


def make_pid_policy():
    pid = PIDController(dt=config.DT * config.ENV_CONTROL_STEPS)
    state = dict(first=True)

    def policy(env, obs):
        if state["first"] or env.step_count == 0:
            pid.reset(); state["first"] = False
        c = env.converter
        return duty_to_action(pid.update(c.Vref, c.Vout))
    return policy


def run(policy, n=N_EPISODES, reset_hook=None):
    rows = []
    env = BoostConverterEnv()
    for seed in range(n):
        obs, info = env.reset(seed=1000 + seed)
        if reset_hook:
            reset_hook()
        ep_ret, errs, dds, trips, last_d, n_steps = 0.0, [], [], 0, None, 0
        while True:
            a = policy(env, obs)
            obs, r, term, trunc, info = env.step(a)
            ep_ret += r; n_steps += 1
            errs.append(abs(info["Vref"] - info["Vout"]))
            if last_d is not None:
                dds.append(abs(info["duty"] - last_d))
            last_d = info["duty"]
            if term or trunc:
                trips += int(term)
                break
        errs = np.array(errs)
        rows.append(dict(ret=ep_ret, steps=n_steps, trip=trips,
                         err_last100=errs[-100:].mean(), mean_abs_err=errs.mean(),
                         mean_dd=np.mean(dds) if dds else 0.0,
                         max_dd=np.max(dds) if dds else 0.0))
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="current")
    p.add_argument("--models", nargs="*", default=[])
    a = p.parse_args(argv)
    policies = {"fixed_ideal": policy_fixed,
                "random": lambda env, obs: env.action_space.sample()}
    rng_env = BoostConverterEnv(); rng_env.action_space.seed(0)
    policies["random"] = lambda env, obs: rng_env.action_space.sample()
    pid_pol = make_pid_policy()
    policies["pid"] = pid_pol
    for name in a.models:
        from training.evaluate_td3 import load_model
        m = load_model(name)
        policies[name] = (lambda mm: (lambda env, obs: mm.predict(obs, deterministic=True)[0]))(m)

    out = {}
    for name, pol in policies.items():
        out[name] = run(pol)
    summary = pd.DataFrame({
        n: dict(return_mean=d.ret.mean(), return_std=d.ret.std(), steps_mean=d.steps.mean(),
                trips=int(d.trip.sum()), err_last100=d.err_last100.mean(),
                mean_abs_err=d.mean_abs_err.mean(), mean_abs_dduty=d.mean_dd.mean(),
                max_dduty=d.max_dd.max())
        for n, d in out.items()}).T
    pd.set_option("display.width", 200, "display.float_format", "{:.4g}".format)
    print(f"\nReward diagnostics [{a.tag}]  ({N_EPISODES} seeded random-condition episodes each)")
    print(summary)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    summary.to_csv(config.RESULTS_DIR / f"reward_diag_{a.tag}.csv")


if __name__ == "__main__":
    main()
