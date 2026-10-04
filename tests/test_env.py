import numpy as np

from radar.env import ACTIONS, EnvConfig, RadarEnv
from radar.evaluation import evaluate, flee_policy, random_policy


def test_observation_matches_deer_hunt_layout():
    env = RadarEnv(EnvConfig(), seed=0)
    assert env.reset().shape == (27,)
    assert EnvConfig(num_hunters=5).obs_size == 35


def test_episode_ends_by_max_turns():
    env = RadarEnv(EnvConfig(max_turns=7, num_hunters=1, hunter_policy="random"), seed=1)
    env.reset()
    done, steps = False, 0
    while not done:
        _, _, done, info = env.step(ACTIONS.index("still"))
        steps += 1
    assert steps <= 7 and info["turns"] <= 7


def test_greedy_hunter_catches_still_prey():
    cfg = EnvConfig(num_hunters=1, sight_radius=100)
    env = RadarEnv(cfg, seed=0)
    env.reset()
    env.state.prey = np.array([0, 0])
    env.state.hunters[0].pos = np.array([3, 0])
    env._update_detection()
    rewards = []
    for _ in range(3):
        _, r, done, info = env.step(ACTIONS.index("still"))
        rewards.append(r)
    assert done and info["caught"] and rewards[-1] == cfg.reward_caught


def test_walking_into_a_hunter_is_terminal():
    env = RadarEnv(EnvConfig(num_hunters=1), seed=0)
    env.reset()
    env.state.prey = np.array([4, 4])
    env.state.hunters[0].pos = np.array([5, 4])
    _, _, done, info = env.step(ACTIONS.index("right"))
    assert done and info["caught"]


def test_invalid_moves_are_masked_at_walls():
    env = RadarEnv(EnvConfig(), seed=0)
    env.reset()
    env.state.prey = np.array([0, 0])
    valid = env.valid_actions()
    assert ACTIONS.index("up") not in valid and ACTIONS.index("left") not in valid


def test_flee_beats_random_against_greedy_hunters():
    cfg = EnvConfig()
    flee = evaluate(flee_policy, cfg, episodes=150, seed=0)["survival_rate"]
    rand = evaluate(random_policy(np.random.default_rng(0)), cfg, episodes=150, seed=0)["survival_rate"]
    assert flee > rand
