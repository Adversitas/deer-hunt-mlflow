"""Train a prey DQN and log everything to MLflow.

    python train.py --episodes 1500
    mlflow run . -P episodes=1500
"""

import argparse

import mlflow

from radar.cli import add_dataclass_args, build_configs, use_experiment
from radar.env import EnvConfig
from radar.training import TrainConfig, train


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataclass_args(p, TrainConfig)
    add_dataclass_args(p, EnvConfig, prefix="env_")
    p.add_argument("--register-as", default="", help="register the model under this name in the Model Registry")
    p.add_argument("--run-name", default=None)
    args = p.parse_args()

    env_cfg, train_cfg = build_configs(args)
    use_experiment()
    with mlflow.start_run(run_name=args.run_name) as run:
        final = train(env_cfg, train_cfg, registered_model_name=args.register_as or None)
        print(f"run {run.info.run_id}: survival vs greedy hunters "
              f"{final['eval_greedy_survival_rate']:.1%}, vs random hunters {final['eval_random_survival_rate']:.1%}")


if __name__ == "__main__":
    main()
