"""Random hyperparameter search; every trial is a nested child run.

    python sweep.py --trials 8 --episodes 800
    mlflow run . -e sweep -P trials=8
"""

import argparse
import random
from dataclasses import replace

import mlflow

from radar.cli import int_arg, use_experiment
from radar.env import EnvConfig
from radar.training import TrainConfig, train

SEARCH_SPACE = {
    "lr": [1e-4, 3e-4, 5e-4, 1e-3],
    "gamma": [0.9, 0.95, 0.99],
    "hidden_size": [128, 256],
    "epsilon_decay": [0.99, 0.995, 0.997],
    "target_sync_every": [10, 25, 50],
}
METRIC = "eval_greedy_survival_rate"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trials", type=int_arg, default=8)
    p.add_argument("--episodes", type=int_arg, default=800)
    p.add_argument("--eval-episodes", type=int_arg, default=200)
    p.add_argument("--seed", type=int_arg, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    base = TrainConfig(episodes=args.episodes, eval_episodes=args.eval_episodes)
    use_experiment()
    with mlflow.start_run(run_name="sweep") as parent:
        mlflow.log_params({"trials": args.trials, "episodes": args.episodes, "search_space": str(SEARCH_SPACE)})
        results = []
        for i in range(args.trials):
            sample = {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}
            with mlflow.start_run(run_name=f"trial-{i}", nested=True) as child:
                score = train(EnvConfig(), replace(base, seed=args.seed + i, **sample))[METRIC]
                results.append((score, child.info.run_id, sample))
                mlflow.log_metric(f"trial_{METRIC}", score, step=i)
            print(f"trial {i}: {METRIC}={score:.3f} {sample}")

        best_score, best_run, best_params = max(results, key=lambda r: r[0])
        mlflow.log_metric(f"best_{METRIC}", best_score)
        mlflow.set_tag("best_run_id", best_run)
        mlflow.log_dict({"best_run_id": best_run, "score": best_score, "params": best_params}, "best_trial.json")
        print(f"best: {best_score:.3f} from run {best_run} -> runs:/{best_run}/model")


if __name__ == "__main__":
    main()
