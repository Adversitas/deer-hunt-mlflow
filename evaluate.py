"""Evaluate a logged or registered model in a fresh MLflow run.

    python evaluate.py --model-uri runs:/<run_id>/model
    python evaluate.py --model-uri models:/radar-prey/1 --episodes 1000
"""

import argparse
import os
import tempfile
from dataclasses import fields

import mlflow
import numpy as np

from radar.cli import add_dataclass_args, int_arg, use_experiment
from radar.env import EnvConfig
from radar.evaluation import evaluate_suite, model_policy, render_episode_gif


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-uri", required=True)
    p.add_argument("--episodes", type=int_arg, default=500)
    p.add_argument("--seed", type=int_arg, default=123)
    add_dataclass_args(p, EnvConfig, prefix="env_")
    args = p.parse_args()
    env_cfg = EnvConfig(**{f.name: getattr(args, f"env_{f.name}") for f in fields(EnvConfig)})

    model = mlflow.pyfunc.load_model(args.model_uri)
    policy = model_policy(lambda obs: model.predict(obs.reshape(1, -1).astype(np.float32)))

    use_experiment()
    with mlflow.start_run(run_name="evaluate") as run:
        mlflow.set_tags({"stage": "evaluation", "model_uri": args.model_uri})
        mlflow.log_params({"model_uri": args.model_uri, "episodes": args.episodes, "seed": args.seed,
                           **{f"env_{k}": getattr(env_cfg, k) for k in (f.name for f in fields(EnvConfig))}})
        metrics = evaluate_suite(policy, env_cfg, args.episodes, seed=args.seed)
        mlflow.log_metrics(metrics)
        with tempfile.TemporaryDirectory() as tmp:
            gif = os.path.join(tmp, "episode.gif")
            with open(gif, "wb") as f:
                f.write(render_episode_gif(policy, env_cfg, seed=args.seed))
            mlflow.log_artifact(gif, "replays")
        print(f"run {run.info.run_id}: survival vs greedy hunters {metrics['eval_greedy_survival_rate']:.1%}, "
              f"vs random hunters {metrics['eval_random_survival_rate']:.1%}")


if __name__ == "__main__":
    main()
