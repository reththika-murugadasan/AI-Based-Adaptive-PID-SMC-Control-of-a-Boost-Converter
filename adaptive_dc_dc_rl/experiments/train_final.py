"""Train the three RL controllers (direct TD3, adaptive PID, adaptive SMC) over several seeds.

    # pipeline check (a few seconds per run)
    python experiments/train_final.py --tag pipeline --timesteps 2000 --jobs 3
    # final runs (default 200,000 steps per run; 9 runs)
    python experiments/train_final.py --tag final --jobs 3

Model files: models/<base>_<tag>_seed<N>.zip with bases from config.FINAL_MODEL_BASES.
Every run is seeded (SB3 / numpy / torch / environment sampling) so a (controller, seed,
timesteps) triple is reproducible on the same machine.  Training episodes use the
RANDOMISED operating conditions and disturbances of config.ENV_* (Vin, R, Vref ranges,
disturbance magnitude / time / kind), never the five fixed evaluation scenarios.
"""
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def model_name(controller, tag, seed):
    return f"{config.FINAL_MODEL_BASES[controller]}_{tag}_seed{seed}"


def _train_one(job):
    controller, tag, seed, timesteps, learning_starts, tensorboard = job
    import torch
    torch.set_num_threads(1)                    # parallel jobs must not oversubscribe the CPU
    name = model_name(controller, tag, seed)
    t0 = time.time()
    if controller == "direct":
        from training import train_td3 as mod
    elif controller == "adaptive-pid":
        from training import train_td3_adaptive_pid as mod
    else:
        from training import train_td3_adaptive_smc as mod
    path = mod.train(timesteps, seed, name, learning_starts, tensorboard=tensorboard)
    return name, str(path), time.time() - t0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tag", default=config.FINAL_TAG)
    p.add_argument("--timesteps", type=int, default=config.TD3_TOTAL_TIMESTEPS)
    p.add_argument("--seeds", type=int, nargs="+", default=list(config.FINAL_SEEDS))
    p.add_argument("--controllers", nargs="+", default=list(config.FINAL_MODEL_BASES),
                   choices=list(config.FINAL_MODEL_BASES))
    p.add_argument("--learning-starts", type=int, default=config.TD3_LEARNING_STARTS)
    p.add_argument("--jobs", type=int, default=1, help="parallel training processes")
    p.add_argument("--no-tensorboard", action="store_true")
    a = p.parse_args(argv)

    jobs = [(c, a.tag, s, a.timesteps, a.learning_starts, not a.no_tensorboard)
            for c in a.controllers for s in a.seeds]
    print(f"{len(jobs)} training runs: controllers={a.controllers} seeds={a.seeds} "
          f"timesteps={a.timesteps} tag={a.tag} jobs={a.jobs}", flush=True)
    t0 = time.time()
    if a.jobs > 1:
        import multiprocessing as mp
        with ProcessPoolExecutor(max_workers=a.jobs, mp_context=mp.get_context("spawn")) as ex:
            results = list(ex.map(_train_one, jobs))
    else:
        results = [_train_one(j) for j in jobs]
    for name, path, dt in results:
        print(f"  {name:<36s} {dt:7.1f} s  {path}")
    print(f"done in {time.time() - t0:.0f} s")
    return results


if __name__ == "__main__":
    main()
