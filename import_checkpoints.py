"""Score the hand-trained Deer-Hunt checkpoints so they sit next to new runs in the MLflow UI.

    git clone https://github.com/adversitas/deer-hunt ../deer-hunt
    python import_checkpoints.py --deer-hunt-dir ../deer-hunt

Each compatible .pth (27 inputs, 6 actions) becomes a run tagged stage=legacy_checkpoint,
with the model logged so it can be registered and compared like any trained run.
Older milestones that used a smaller observation are skipped.
"""

import argparse
import glob
import os

import mlflow
import torch

from radar.agent import DQN
from radar.cli import int_arg, use_experiment
from radar.env import ACTIONS, EnvConfig
from radar.evaluation import evaluate_suite, model_policy
from radar.training import DEER_HUNT_URL, log_model


class _Holder:  # log_model expects an object with a q_network
    def __init__(self, net):
        self.q_network = net


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--deer-hunt-dir", required=True)
    p.add_argument("--episodes", type=int_arg, default=500)
    p.add_argument("--seed", type=int_arg, default=42)
    args = p.parse_args()

    env_cfg = EnvConfig()
    paths = sorted(glob.glob(os.path.join(args.deer_hunt_dir, "milestones", "*.pth")))
    paths.append(os.path.join(args.deer_hunt_dir, "RADAR", "prey_dqn_model.pth"))

    use_experiment()
    for path in paths:
        state = torch.load(path, map_location="cpu")
        in_size, out_size = state["fc1.weight"].shape[1], state["fc4.weight"].shape[0]
        hidden = state["fc1.weight"].shape[0]
        rel = os.path.relpath(path, args.deer_hunt_dir)
        if (in_size, out_size) != (env_cfg.obs_size, len(ACTIONS)):
            print(f"skip {rel}: expects {in_size} inputs / {out_size} actions")
            continue

        net = DQN(in_size, hidden, out_size)
        net.load_state_dict(state)
        net.eval()
        policy = model_policy(lambda obs: net(torch.as_tensor(obs).unsqueeze(0)).detach().numpy())

        name = os.path.splitext(os.path.basename(path))[0]
        if rel.startswith("RADAR"):
            name = "deer-hunt-final"
        with mlflow.start_run(run_name=f"legacy: {name}"):
            mlflow.set_tags({"stage": "legacy_checkpoint", "source_project": DEER_HUNT_URL, "source_file": rel})
            mlflow.log_params({"hidden_size": hidden, "eval_episodes": args.episodes, "seed": args.seed})
            metrics = evaluate_suite(policy, env_cfg, args.episodes, seed=args.seed)
            mlflow.log_metrics(metrics)
            log_model(_Holder(net), env_cfg)
        print(f"{rel}: survival vs greedy hunters {metrics['eval_greedy_survival_rate']:.1%}")


if __name__ == "__main__":
    main()
