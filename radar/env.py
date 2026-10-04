"""RADAR grid world: one prey tries to survive against a pack of hunters.

Ported from Deer-Hunt (RADAR/radar_rl.py) and wrapped in a reset/step
interface so training and evaluation share exactly the same dynamics.
The 27-value observation layout is unchanged, so checkpoints trained in
Deer-Hunt load and run here.
"""

from dataclasses import dataclass, field

import numpy as np

ACTIONS = ["up", "down", "left", "right", "still", "smell"]
MOVES = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}
FLARE_MEMORY_LIMIT = 3


@dataclass
class EnvConfig:
    width: int = 10
    height: int = 10
    num_hunters: int = 3
    max_turns: int = 50
    sight_radius: float = 3.0
    smell_cooldown: int = 5
    hunter_policy: str = "greedy"  # "greedy" chases when it sees the prey, "random" never chases
    reward_caught: float = -300.0
    reward_survived: float = 100.0
    reward_step: float = 1.0

    @property
    def obs_size(self) -> int:
        # prey xy + 3 flares xy + visible fraction + remaining moves + cooldown
        # + 4 features per hunter + 4 wall distances
        return 2 + 2 * FLARE_MEMORY_LIMIT + 3 + 4 * self.num_hunters + 4


@dataclass
class Hunter:
    pos: np.ndarray
    detected: bool = False
    aware: bool = False


@dataclass
class State:
    prey: np.ndarray
    hunters: list
    smell_cd: int = 0
    flares: list = field(default_factory=list)
    turn: int = 0


class RadarEnv:
    def __init__(self, config: EnvConfig | None = None, seed: int | None = None):
        self.cfg = config or EnvConfig()
        self.rng = np.random.default_rng(seed)
        self.state: State | None = None

    # ---------- public API ----------

    def reset(self) -> np.ndarray:
        cells = self.rng.choice(self.cfg.width * self.cfg.height, size=1 + self.cfg.num_hunters, replace=False)
        xy = [np.array([c % self.cfg.width, c // self.cfg.width]) for c in cells]
        self.state = State(prey=xy[0], hunters=[Hunter(p) for p in xy[1:]])
        self._update_detection()
        # The original agent saw remaining_moves=2 on the first observation only.
        return self._observe(remaining_moves=2)

    def valid_actions(self) -> list[int]:
        s, cfg = self.state, self.cfg
        x, y = s.prey
        valid = []
        if y > 0:
            valid.append(0)
        if y < cfg.height - 1:
            valid.append(1)
        if x > 0:
            valid.append(2)
        if x < cfg.width - 1:
            valid.append(3)
        valid.append(4)
        if s.smell_cd == 0:
            valid.append(5)
        return valid

    def step(self, action: int):
        """Prey acts, then hunters move. Returns (obs, reward, done, info)."""
        s, cfg = self.state, self.cfg
        name = ACTIONS[action]

        if name == "smell":
            if s.smell_cd == 0:
                s.flares.append(s.prey.copy())
                s.smell_cd = cfg.smell_cooldown
        elif name in MOVES:
            dx, dy = MOVES[name]
            s.prey = np.array([
                np.clip(s.prey[0] + dx, 0, cfg.width - 1),
                np.clip(s.prey[1] + dy, 0, cfg.height - 1),
            ])

        if self._caught():
            return self._finish(caught=True)

        # Standing on (or next to) a flare tips the hunters off for good.
        if any(np.linalg.norm(s.prey - f) <= 1 for f in s.flares):
            for h in s.hunters:
                h.aware = True
        flares_for_obs = list(s.flares)
        s.flares.clear()

        for h in s.hunters:
            self._move_hunter(h)
        self._update_detection()
        if s.smell_cd > 0:
            s.smell_cd -= 1
        s.turn += 1

        if self._caught():
            return self._finish(caught=True)
        if s.turn >= cfg.max_turns:
            return self._finish(caught=False)
        return self._observe(remaining_moves=0, flares=flares_for_obs), cfg.reward_step, False, {}

    # ---------- internals ----------

    def _finish(self, caught: bool):
        reward = self.cfg.reward_caught if caught else self.cfg.reward_survived
        info = {"caught": caught, "turns": self.state.turn}
        return self._observe(remaining_moves=0), reward, True, info

    def _caught(self) -> bool:
        return any(np.array_equal(h.pos, self.state.prey) for h in self.state.hunters)

    def _update_detection(self):
        for h in self.state.hunters:
            h.detected = bool(np.linalg.norm(self.state.prey - h.pos) <= self.cfg.sight_radius)

    def _move_hunter(self, h: Hunter):
        cfg = self.cfg
        chasing = cfg.hunter_policy == "greedy" and (h.aware or h.detected)
        if chasing:
            dx, dy = self.state.prey - h.pos
            if abs(dx) > abs(dy):
                step = (int(np.sign(dx)), 0)
            else:
                step = (0, int(np.sign(dy)))
        else:
            step = list(MOVES.values())[self.rng.integers(4)]
        h.pos = np.array([
            np.clip(h.pos[0] + step[0], 0, cfg.width - 1),
            np.clip(h.pos[1] + step[1], 0, cfg.height - 1),
        ])

    def _observe(self, remaining_moves: int, flares=()) -> np.ndarray:
        s, cfg = self.state, self.cfg
        w, h = cfg.width, cfg.height
        px, py = s.prey
        obs = [px / w, py / h]

        flare_feats = [[f[0] / w, f[1] / h] for f in list(flares)[:FLARE_MEMORY_LIMIT]]
        flare_feats += [[0.0, 0.0]] * (FLARE_MEMORY_LIMIT - len(flare_feats))
        obs += [v for pair in flare_feats for v in pair]

        obs.append(sum(hu.detected for hu in s.hunters) / max(1, len(s.hunters)))
        obs.append(remaining_moves / 2)
        obs.append(s.smell_cd / cfg.smell_cooldown)

        max_dist = np.hypot(w, h)
        for hu in s.hunters:
            if hu.detected:
                d = float(np.linalg.norm(s.prey - hu.pos))
                direction = (hu.pos - s.prey) / d if d > 0 else np.zeros(2)
                obs += [d / max_dist, float(direction[0]), float(direction[1]), float(hu.aware)]
            else:
                obs += [0.0, 0.0, 0.0, 0.0]

        obs += [px / w, (w - 1 - px) / w, py / h, (h - 1 - py) / h]
        return np.asarray(obs, dtype=np.float32)

    def snapshot(self) -> dict:
        """Positions for rendering."""
        return {
            "prey": tuple(int(v) for v in self.state.prey),
            "hunters": [(int(h.pos[0]), int(h.pos[1]), h.aware or h.detected) for h in self.state.hunters],
        }
