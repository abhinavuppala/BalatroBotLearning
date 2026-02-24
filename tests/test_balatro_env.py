"""
Tests for BalatroEnv (Gymnasium wrapper). Uses a mock connection to avoid requiring a running game.
"""

import json
import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import gymnasium as gym
    HAS_GYM = True
except ImportError:
    HAS_GYM = False

try:
    from balatro_env import BalatroEnv, _is_decision_state, _is_game_over
    HAS_ENV = True
except ImportError:
    HAS_ENV = False


def _load_cached_state(subdir: str):
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "gamestate_cache", subdir)
    if not os.path.isdir(cache_dir):
        return None
    files = [f for f in os.listdir(cache_dir) if f.endswith(".json")]
    if not files:
        return None
    path = os.path.join(cache_dir, files[0])
    with open(path, "r") as f:
        return json.load(f)


@pytest.mark.skipif(not HAS_GYM or not HAS_ENV, reason="gymnasium or balatro_env not available")
def test_is_decision_state():
    assert _is_decision_state({}) is False
    assert _is_decision_state({"response": "x"}) is False
    assert _is_decision_state({"state": 4}) is False
    assert _is_decision_state({"state": 1, "waitingForAction": True}) is True


@pytest.mark.skipif(not HAS_GYM or not HAS_ENV, reason="gymnasium or balatro_env not available")
def test_is_game_over():
    assert _is_game_over({}) is True
    assert _is_game_over({"state": 3}) is False
    assert _is_game_over({"state": 4}) is True


@pytest.mark.skipif(not HAS_GYM or not HAS_ENV, reason="gymnasium or balatro_env not available")
def test_env_spaces():
    """Env has correct observation and action space shapes."""
    conn = MagicMock()
    conn.balatro_instance = None
    conn.poll_state.return_value = {}
    conn.send_action.return_value = {}
    env = BalatroEnv(connection=conn, run_balatro=False)
    assert env.observation_space.shape == (65,)
    assert env.action_space.n == 274  # max_action_space_size()
    env.close()


@pytest.mark.skipif(not HAS_GYM or not HAS_ENV, reason="gymnasium or balatro_env not available")
def test_env_reset_with_mock_state():
    """Reset returns obs and info with action_mask when mock returns a decision state."""
    G = _load_cached_state("select_cards_from_hand")
    if G is None:
        pytest.skip("no cached select_cards_from_hand state")
    conn = MagicMock()
    conn.balatro_instance = None
    conn.poll_state.return_value = G
    conn.send_action.return_value = G
    env = BalatroEnv(connection=conn, run_balatro=False)
    obs, info = env.reset()
    assert obs.shape == (65,)
    assert obs.dtype == np.float32
    assert "action_mask" in info
    assert info["action_mask"].shape == (274,)
    assert info["action_mask"].dtype == np.bool_
    assert info["action_mask"].any()
    env.close()


@pytest.mark.skipif(not HAS_GYM or not HAS_ENV, reason="gymnasium or balatro_env not available")
def test_env_action_masks_method():
    """action_masks() returns same as info['action_mask'] after reset."""
    G = _load_cached_state("select_cards_from_hand")
    if G is None:
        pytest.skip("no cached select_cards_from_hand state")
    conn = MagicMock()
    conn.balatro_instance = None
    conn.poll_state.return_value = G
    conn.send_action.return_value = G
    env = BalatroEnv(connection=conn, run_balatro=False)
    env.reset()
    masks = env.action_masks()
    assert masks.shape == (274,)
    assert np.array_equal(masks, env._get_obs_and_mask(env._current_G)[1])
    env.close()


@pytest.mark.skipif(not HAS_GYM or not HAS_ENV, reason="gymnasium or balatro_env not available")
def test_env_step_decode_and_send():
    """Step decodes action and calls send_action; returns next obs and mask."""
    G = _load_cached_state("select_cards_from_hand")
    if G is None:
        pytest.skip("no cached select_cards_from_hand state")
    next_G = dict(G)
    next_G["hand"] = []  # different state after "play"
    conn = MagicMock()
    conn.balatro_instance = None
    # reset: _start_run_if_needed polls once, _poll_until_decision polls until decision; step: _poll_until_decision may poll
    conn.poll_state.side_effect = [G, G, next_G]
    conn.send_action.return_value = next_G
    env = BalatroEnv(connection=conn, run_balatro=False)
    env.reset()
    # Action 0 = first valid action (e.g. play combination 0)
    obs, reward, term, trunc, info = env.step(0)
    conn.send_action.assert_called_once()
    call_args = conn.send_action.call_args[0][0]
    assert call_args[0].name in ("PLAY_HAND", "DISCARD_HAND")
    assert obs.shape == (65,)
    assert "action_mask" in info
    env.close()
