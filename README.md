# deer-hunt-mlflow

An [MLflow Project](https://mlflow.org/docs/latest/ml/projects/) for the **RADAR prey-vs-hunters** reinforcement-learning game from [Deer-Hunt](https://github.com/adversitas/deer-hunt).

Deer-Hunt is where the game, the Pygame GUI and the original notebook live. This repo makes the training reproducible and comparable. Every run logs its hyperparameters, learning curves, evaluation against baselines, a replay GIF and a versioned PyTorch model to MLflow.

```
            ┌───────────────┐   mlflow run .        ┌───────────────────────────────┐
 params ──▶ │  train.py     │ ────────────────────▶ │ MLflow tracking               │
            │  (DQN + env)  │                       │  params · metrics · plots     │
            └───────────────┘                       │  replay GIF · model · registry│
            ┌───────────────┐   -e sweep            └──────────────┬────────────────┘
            │  sweep.py     │ ──── nested runs ─────────────────────┤
            └───────────────┘                                      │ runs:/… or models:/…
            ┌──────────────────────┐  -e import_checkpoints        ▼
            │ import_checkpoints.py│ ── legacy Deer-Hunt .pth ──▶ evaluate.py
            └──────────────────────┘
```

## The game

A prey moves on a 10×10 grid with three hunters. Hunters wander until the prey is within their sight radius, then chase it greedily. The prey sees a 27-value observation: its position, flares, nearby hunters' distance, direction and awareness, its smell cooldown, and the distance to each wall. It picks one of `up / down / left / right / still / smell` and is trained with a DQN.

- Getting caught: −300. Surviving all 50 turns: +100. Each other turn: +1.
- The observation layout and network shape match Deer-Hunt, so checkpoints move between the two repos.

## Quick start

```bash
git clone https://github.com/adversitas/deer-hunt-mlflow
cd deer-hunt-mlflow
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

mlflow run . --env-manager local --experiment-name radar-prey-dqn
mlflow ui            # open http://127.0.0.1:5000
```

You can also skip `mlflow run` and call the scripts directly with `python train.py --episodes 1500`. Every script has `--help`.

## Entry points

| Entry point | What it does | Example |
|---|---|---|
| `main` | Trains a DQN, evaluates the best checkpoint, logs the model and can register it | `mlflow run . -P episodes=2000 -P lr=0.0003 -P register_as=radar-prey` |
| `evaluate` | Scores any logged or registered model in a fresh run | `mlflow run . -e evaluate -P model_uri=models:/radar-prey/1` |
| `sweep` | Random hyperparameter search, one nested child run per trial | `mlflow run . -e sweep -P trials=8 -P episodes=800` |
| `import_checkpoints` | Logs the hand-trained Deer-Hunt checkpoints as runs for comparison | `mlflow run . -e import_checkpoints -P deer_hunt_dir=../deer-hunt` |

`main` also exposes environment knobs: `num_hunters`, `max_turns` and `hunter_policy` (`greedy` or `random`).

## What each run logs

| Kind | Contents |
|---|---|
| Params | All training hyperparameters plus the environment config (`env_*`) |
| Metrics, every 10 episodes | `train_return_avg100`, `train_survival_rate_avg100`, `train_turns_avg100`, `loss_avg100`, `epsilon` |
| Metrics, every 100 episodes | `checkpoint_survival_rate`, the greedy policy on held-out seeds, used to pick the best checkpoint |
| Final metrics | `eval_{greedy,random}_{survival_rate,avg_turns,avg_return,action_freq_*}` for the selected checkpoint against both hunter types |
| Baselines | `baseline_random_*` (random legal moves) and `baseline_flee_*` (a hand-written "step away from the nearest visible hunter" policy) |
| Artifacts | `plots/survival_curve.png`, `plots/action_mix.png`, `replays/episode.gif`, `deer_hunt_export/` |
| Model | `model/`, a PyTorch flavor with a signature and input example, loadable with `mlflow.pyfunc.load_model(...)`; it returns Q-values per action |

### Taking a model back to the Deer-Hunt GUI

Each run's `deer_hunt_export/` artifact contains `prey_dqn_model.pth` and `prey_dqn_meta.json`. Copy both into Deer-Hunt's `RADAR/` folder and run `python forest_rl_gui_integrated.py` to watch the agent play. The GUI expects `hidden_size=256`.

```bash
mlflow artifacts download -u runs:/<run_id>/deer_hunt_export -d ../deer-hunt/RADAR
```

## Results so far

Each policy below was evaluated on the same seeds in this repo's environment.

| Policy | Survival vs greedy hunters | Survival vs random hunters |
|---|---|---|
| Random legal moves | 0% | 24% |
| Deer-Hunt `flare + smell 70.pth` | 4% | 62% |
| Deer-Hunt final `prey_dqn_model.pth` | 28% | 48% |
| **This repo, default `main` run (1500 episodes)** | **47%** | **82%** |
| Flee heuristic | 56% | 99% |

The new agent is well ahead of the original checkpoints but still behind the simple flee heuristic. Beating that heuristic is the open challenge, and it's what `sweep` is for.

The original checkpoints score lower here than in the Deer-Hunt notebook because this environment fixes two quirks in the original training loop:

- A hunter landing on the prey is now terminal. Before, it only counted if the prey was still caught after its next move.
- Each turn is now one transition. Before, the same reward was stored twice.

Deer-Hunt's two earliest milestones used a 20- and 21-value observation, so `import_checkpoints` skips them.

## Layout

```
MLproject                entry points and parameters
python_env.yaml          environment for `mlflow run` (virtualenv)
train.py / evaluate.py / sweep.py / import_checkpoints.py
radar/
  env.py                 RadarEnv: reset/step game logic, shared by training and evaluation
  agent.py               DQN network + replay-buffer agent
  training.py            training loop and all MLflow logging
  evaluation.py          evaluation suite, baselines, GIF renderer
  cli.py                 argparse helpers
tests/                   environment unit tests + an end-to-end MLflow run
.github/workflows/ci.yml pytest + an `mlflow run` smoke test on every push
```

## Tracking backend

MLflow 3 stores runs in a local `mlflow.db` (SQLite) by default, and that file is git-ignored. To use a shared server instead, set `MLFLOW_TRACKING_URI`, for example `export MLFLOW_TRACKING_URI=http://localhost:5000` after running `mlflow server`.
