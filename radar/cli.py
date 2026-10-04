"""Shared argparse helpers for the entry points."""

import argparse
import os
from dataclasses import fields

import mlflow

from radar.env import EnvConfig
from radar.training import TrainConfig

EXPERIMENT = "radar-prey-dqn"


def int_arg(value: str) -> int:
    # MLproject only has a float type, so accept "1500.0" for integer options.
    return int(float(value))


def add_dataclass_args(parser: argparse.ArgumentParser, cls, prefix: str = ""):
    for f in fields(cls):
        kind = int_arg if isinstance(f.default, int) else type(f.default)
        parser.add_argument(f"--{prefix}{f.name}".replace("_", "-"), dest=f"{prefix}{f.name}",
                            type=kind, default=f.default)


def build_configs(args) -> tuple[EnvConfig, TrainConfig]:
    env = EnvConfig(**{f.name: getattr(args, f"env_{f.name}") for f in fields(EnvConfig)})
    train = TrainConfig(**{f.name: getattr(args, f.name) for f in fields(TrainConfig)})
    return env, train


def use_experiment(name: str = EXPERIMENT):
    # `mlflow run` creates the run up front and passes it in via MLFLOW_RUN_ID;
    # only choose an experiment ourselves when launched directly with python.
    if "MLFLOW_RUN_ID" not in os.environ:
        mlflow.set_experiment(name)
