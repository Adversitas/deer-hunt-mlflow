"""DQN training loop with MLflow tracking."""

import copy
import json
import os
import random
import tempfile
from collections import deque
from dataclasses import asdict, dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from mlflow.models import infer_signature

from radar.agent import DQNAgent
from radar.env import ACTIONS, EnvConfig, RadarEnv
from radar.evaluation import (
    evaluate,
    evaluate_suite,
    flee_policy,
    model_policy,
    random_policy,
    render_episode_gif,
)

DEER_HUNT_URL = "https://github.com/adversitas/deer-hunt"


@dataclass
class TrainConfig:
    episodes: int = 1500
    lr: float = 5e-4
    gamma: float = 0.95
    hidden_size: int = 256
    batch_size: int = 64
    memory_size: int = 20_000
    epsilon_min: float = 0.02
    epsilon_decay: float = 0.997
    train_every: int = 2
    target_sync_every: int = 25
    eval_every: int = 100
    eval_episodes: int = 200
    log_every: int = 10
    seed: int = 42


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def greedy(agent: DQNAgent):
    return lambda env, obs: agent.act(obs, env.valid_actions(), greedy=True)


def train(env_cfg: EnvConfig, cfg: TrainConfig, registered_model_name: str | None = None) -> dict:
    """Train inside the active MLflow run and return the final evaluation metrics."""
    seed_everything(cfg.seed)
    mlflow.log_params({**asdict(cfg), **{f"env_{k}": v for k, v in asdict(env_cfg).items()}})
    mlflow.set_tags({"algorithm": "DQN", "source_project": DEER_HUNT_URL})

    env = RadarEnv(env_cfg, seed=cfg.seed)
    agent = DQNAgent(
        env_cfg.obs_size, len(ACTIONS),
        hidden_size=cfg.hidden_size, lr=cfg.lr, gamma=cfg.gamma,
        epsilon_min=cfg.epsilon_min, epsilon_decay=cfg.epsilon_decay,
        memory_size=cfg.memory_size, batch_size=cfg.batch_size,
    )

    window = 100
    returns, survived, turns, losses = (deque(maxlen=window) for _ in range(4))
    history = {"episode": [], "survival": [], "eval_episode": [], "eval_survival": []}
    best_eval, best_weights, step = -1.0, None, 0

    for ep in range(1, cfg.episodes + 1):
        obs, done, total = env.reset(), False, 0.0
        while not done:
            a = agent.act(obs, env.valid_actions())
            next_obs, r, done, info = env.step(a)
            agent.remember(obs, a, r, next_obs, done)
            obs, total, step = next_obs, total + r, step + 1
            if step % cfg.train_every == 0:
                loss = agent.replay()
                if loss is not None:
                    losses.append(loss)
        agent.decay_epsilon()
        if ep % cfg.target_sync_every == 0:
            agent.sync_target()

        returns.append(total)
        survived.append(not info["caught"])
        turns.append(info["turns"])

        if ep % cfg.log_every == 0:
            metrics = {
                "train_return_avg100": float(np.mean(returns)),
                "train_survival_rate_avg100": float(np.mean(survived)),
                "train_turns_avg100": float(np.mean(turns)),
                "epsilon": agent.epsilon,
            }
            if losses:
                metrics["loss_avg100"] = float(np.mean(losses))
            mlflow.log_metrics(metrics, step=ep)
            history["episode"].append(ep)
            history["survival"].append(metrics["train_survival_rate_avg100"])

        if ep % cfg.eval_every == 0 or ep == cfg.episodes:
            # Checkpoint selection on a separate seed range from the final evaluation.
            res = evaluate(greedy(agent), env_cfg, episodes=cfg.eval_episodes // 2, seed=10_000 + ep)
            mlflow.log_metric("checkpoint_survival_rate", res["survival_rate"], step=ep)
            history["eval_episode"].append(ep)
            history["eval_survival"].append(res["survival_rate"])
            if res["survival_rate"] >= best_eval:
                best_eval, best_weights = res["survival_rate"], copy.deepcopy(agent.q_network.state_dict())
                mlflow.log_metric("best_checkpoint_episode", ep)

    agent.q_network.load_state_dict(best_weights)
    agent.q_network.eval()
    policy = greedy(agent)

    final = evaluate_suite(policy, env_cfg, cfg.eval_episodes, seed=cfg.seed)
    baselines = {
        **{k.replace("eval_", "baseline_random_"): v
           for k, v in evaluate_suite(random_policy(np.random.default_rng(0)), env_cfg, cfg.eval_episodes, cfg.seed).items()
           if k.endswith("survival_rate")},
        **{k.replace("eval_", "baseline_flee_"): v
           for k, v in evaluate_suite(flee_policy, env_cfg, cfg.eval_episodes, cfg.seed).items()
           if k.endswith("survival_rate")},
    }
    mlflow.log_metrics({**final, **baselines})

    log_artifacts(agent, env_cfg, policy, history, final, baselines, cfg.seed)
    log_model(agent, env_cfg, registered_model_name)
    return final


def log_artifacts(agent, env_cfg, policy, history, final, baselines, seed):
    with tempfile.TemporaryDirectory() as tmp:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(history["episode"], history["survival"], label="training (rolling 100, with exploration)")
        ax.plot(history["eval_episode"], history["eval_survival"], "o-", label="greedy checkpoint eval")
        ax.axhline(baselines["baseline_flee_greedy_survival_rate"], ls="--", c="gray", label="flee heuristic")
        ax.axhline(baselines["baseline_random_greedy_survival_rate"], ls=":", c="gray", label="random policy")
        ax.set(xlabel="episode", ylabel="survival rate", ylim=(0, 1), title="Prey survival vs greedy hunters")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        mlflow.log_figure(fig, "plots/survival_curve.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 3.5))
        freqs = [final[f"eval_greedy_action_freq_{a}"] for a in ACTIONS]
        ax.bar(ACTIONS, freqs)
        ax.set(ylabel="share of actions", title="Greedy policy action mix")
        fig.tight_layout()
        mlflow.log_figure(fig, "plots/action_mix.png")
        plt.close(fig)

        gif = os.path.join(tmp, "episode.gif")
        with open(gif, "wb") as f:
            f.write(render_episode_gif(policy, env_cfg, seed=seed))
        mlflow.log_artifact(gif, "replays")

        # Drop-in files for the Deer-Hunt Pygame GUI (RADAR/forest_rl_gui_integrated.py).
        export = os.path.join(tmp, "deer_hunt_export")
        os.makedirs(export)
        torch.save(agent.q_network.state_dict(), os.path.join(export, "prey_dqn_model.pth"))
        with open(os.path.join(export, "prey_dqn_meta.json"), "w") as f:
            json.dump({"state_size": env_cfg.obs_size, "action_size": len(ACTIONS)}, f)
        mlflow.log_artifacts(export, "deer_hunt_export")


def log_model(agent, env_cfg, registered_model_name=None):
    model = copy.deepcopy(agent.q_network).cpu().eval()
    example = RadarEnv(env_cfg, seed=0).reset().reshape(1, -1)
    with torch.no_grad():
        q = model(torch.as_tensor(example)).numpy()
    mlflow.pytorch.log_model(
        model,
        name="model",
        signature=infer_signature(example, q),
        input_example=example,
        registered_model_name=registered_model_name,
    )
