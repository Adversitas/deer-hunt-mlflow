import mlflow
import numpy as np

from radar.env import EnvConfig, RadarEnv
from radar.training import TrainConfig, train


def test_short_training_run_logs_everything(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    mlflow.set_experiment("test")
    cfg = TrainConfig(episodes=20, eval_every=10, eval_episodes=10, log_every=5, batch_size=16)
    with mlflow.start_run() as run:
        final = train(EnvConfig(max_turns=20), cfg)

    assert 0.0 <= final["eval_greedy_survival_rate"] <= 1.0
    data = mlflow.get_run(run.info.run_id).data
    assert data.params["episodes"] == "20"
    for key in ("train_survival_rate_avg100", "eval_random_survival_rate", "baseline_flee_greedy_survival_rate"):
        assert key in data.metrics

    artifacts = {a.path for a in mlflow.MlflowClient().list_artifacts(run.info.run_id)}
    assert {"plots", "replays", "deer_hunt_export"} <= artifacts

    model = mlflow.pyfunc.load_model(f"runs:/{run.info.run_id}/model")
    obs = RadarEnv(EnvConfig(), seed=0).reset().reshape(1, -1)
    assert np.asarray(model.predict(obs)).shape == (1, 6)
