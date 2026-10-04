"""Policy evaluation, baselines and rendering helpers."""

import io
from dataclasses import replace

import numpy as np
from PIL import Image, ImageDraw

from radar.agent import masked_argmax
from radar.env import MOVES, ACTIONS, EnvConfig, RadarEnv


def random_policy(rng):
    return lambda env, obs: int(rng.choice(env.valid_actions()))


def flee_policy(env: RadarEnv, obs) -> int:
    """Hand-written baseline: step to the cell farthest from the nearest hunter it can see."""
    s = env.state
    seen = [h.pos for h in s.hunters if h.detected]
    if not seen:
        return ACTIONS.index("still")
    best, best_score = ACTIONS.index("still"), -1.0
    for a in env.valid_actions():
        name = ACTIONS[a]
        if name not in MOVES and name != "still":
            continue
        dx, dy = MOVES.get(name, (0, 0))
        pos = s.prey + np.array([dx, dy])
        score = min(float(np.abs(pos - h).sum()) for h in seen)
        if score > best_score:
            best, best_score = a, score
    return best


def model_policy(predict_fn):
    """Wrap any obs -> q_values function (torch module, MLflow pyfunc, ...)."""
    return lambda env, obs: masked_argmax(np.asarray(predict_fn(obs)).reshape(-1), env.valid_actions())


def evaluate(policy, config: EnvConfig, episodes: int = 200, seed: int = 0) -> dict:
    env = RadarEnv(config, seed=seed)
    survived, turns, returns = [], [], []
    action_counts = np.zeros(len(ACTIONS))
    for _ in range(episodes):
        obs, done, total = env.reset(), False, 0.0
        while not done:
            a = policy(env, obs)
            action_counts[a] += 1
            obs, r, done, info = env.step(a)
            total += r
        survived.append(not info["caught"])
        turns.append(info["turns"])
        returns.append(total)
    freq = action_counts / max(1, action_counts.sum())
    return {
        "survival_rate": float(np.mean(survived)),
        "avg_turns": float(np.mean(turns)),
        "avg_return": float(np.mean(returns)),
        **{f"action_freq_{name}": float(f) for name, f in zip(ACTIONS, freq)},
    }


def evaluate_suite(policy, config: EnvConfig, episodes: int, seed: int = 0) -> dict:
    """Score a policy against both hunter behaviours, keyed for mlflow.log_metrics."""
    out = {}
    for hunters in ("greedy", "random"):
        res = evaluate(policy, replace(config, hunter_policy=hunters), episodes, seed)
        out.update({f"eval_{hunters}_{k}": v for k, v in res.items()})
    return out


def render_episode_gif(policy, config: EnvConfig, seed: int = 0, cell: int = 36) -> bytes:
    """Play one episode and return it as an animated GIF."""
    env = RadarEnv(config, seed=seed)
    obs, done, frames = env.reset(), False, []
    frames.append(_draw(env, cell, status=""))
    while not done:
        obs, _, done, info = env.step(policy(env, obs))
        status = ("CAUGHT" if info["caught"] else "SURVIVED") if done else ""
        frames.append(_draw(env, cell, status))
    frames += [frames[-1]] * 6  # linger on the final frame
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:], duration=180, loop=0)
    return buf.getvalue()


def _draw(env: RadarEnv, cell: int, status: str) -> Image.Image:
    cfg, snap = env.cfg, env.snapshot()
    header = 22
    img = Image.new("RGB", (cfg.width * cell, cfg.height * cell + header), (34, 49, 34))
    d = ImageDraw.Draw(img)
    for x in range(cfg.width):
        for y in range(cfg.height):
            d.rectangle([x * cell, header + y * cell, (x + 1) * cell - 1, header + (y + 1) * cell - 1],
                        fill=(52, 84, 52) if (x + y) % 2 else (60, 94, 58))
    px, py = snap["prey"]
    r = cell * 0.18
    cx, cy = px * cell + cell / 2, header + py * cell + cell / 2
    d.ellipse([cx - cfg.sight_radius * cell, cy - cfg.sight_radius * cell,
               cx + cfg.sight_radius * cell, cy + cfg.sight_radius * cell], outline=(140, 170, 120))
    d.ellipse([px * cell + r, header + py * cell + r, (px + 1) * cell - r, header + (py + 1) * cell - r],
              fill=(214, 168, 96))
    for hx, hy, alert in snap["hunters"]:
        d.rectangle([hx * cell + r, header + hy * cell + r, (hx + 1) * cell - r, header + (hy + 1) * cell - r],
                    fill=(200, 60, 50) if alert else (120, 70, 60))
    d.text((6, 5), f"turn {env.state.turn}/{cfg.max_turns}  {status}", fill=(235, 235, 225))
    return img
