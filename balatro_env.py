"""
Gymnasium environment wrapping a single Balatro instance via the bot API.

Uses state_encoder for observations and action_space for discrete actions + masking.
Designed for use with sb3_contrib.MaskablePPO (action_masks in info and action_masks()).
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    raise ImportError("balatro_env requires gymnasium: pip install gymnasium")

from balatro_connection import BalatroConnection, State
from state_encoder import state_to_numpy, flatten_observation, flat_observation_size
from action_space import (
    action_mask,
    decode_action,
    num_actions,
    max_action_space_size,
)


# Poll until we get a decision state or game over (avoid busy-loop with short sleep).
POLL_SLEEP_S = 0.02
MAX_POLL_ITERATIONS = 500


def _is_decision_state(G: Dict[str, Any]) -> bool:
    if not G:
        return False
    if G.get("response"):
        return False
    state = G.get("state")
    if state is not None and hasattr(state, "value"):
        state = state.value
    if state == 4:  # GAME_OVER
        return False
    return bool(G.get("waitingForAction"))


def _is_game_over(G: Dict[str, Any]) -> bool:
    if not G:
        return True
    state = G.get("state")
    if state is not None and hasattr(state, "value"):
        state = state.value
    return state == 4


class BalatroEnv(gym.Env):
    """
    Gymnasium env for a single Balatro run.
    - Observation: flat float32 vector from state_encoder (length flat_observation_size()).
    - Action: discrete index in [0, max_action_space_size()); validity from action_mask(G).
    - Reward: 0 per step, -1 on game over (terminated).
    - Episodes: one run; reset() starts a new run (sends START_RUN if needed).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        bot_port: int = 12348,
        deck: str = "Blue Deck",
        stake: int = 1,
        seed: Optional[str] = None,
        run_balatro: bool = True,
        connection: Optional[BalatroConnection] = None,
        reward_game_over: float = -1.0,
        max_steps_per_episode: Optional[int] = None,
    ):
        """
        Args:
            bot_port: Port for Balatro bot.
            deck: Deck name for START_RUN.
            stake: Stake for START_RUN.
            seed: Run seed (None = random).
            run_balatro: If True and connection not provided, start Balatro.exe in reset().
            connection: Reuse an existing connection (e.g. from a Bot); if set, run_balatro ignored.
            reward_game_over: Reward when episode ends (game over).
            max_steps_per_episode: Truncate after this many steps (optional).
        """
        super().__init__()
        self._deck = deck
        self._stake = stake
        self._seed = seed
        self._reward_game_over = reward_game_over
        self._max_steps = max_steps_per_episode
        self._run_balatro = run_balatro and (connection is None)
        self._connection = connection or BalatroConnection(bot_port=bot_port)
        self._current_G: Optional[Dict[str, Any]] = None
        self._steps = 0
        self._last_round: Optional[int] = None

        obs_size = flat_observation_size()
        n_actions = max_action_space_size()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(n_actions)
        # For MaskablePPO: mask has same length as action space; invalid indices are False.
        self._mask_size = n_actions

    def _get_obs_and_mask(self, G: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        encoded = state_to_numpy(G)
        obs, _ = flatten_observation(encoded)
        mask = action_mask(G)
        # Pad mask to full action space (invalid indices beyond current context)
        n = num_actions(G.get("waitingFor") or "")
        full_mask = np.zeros(self._mask_size, dtype=np.bool_)
        full_mask[: mask.shape[0]] = mask
        return obs, full_mask

    def _poll_until_decision(self) -> Optional[Dict[str, Any]]:
        for _ in range(MAX_POLL_ITERATIONS):
            G = self._connection.poll_state()
            if _is_game_over(G):
                return G
            if _is_decision_state(G):
                return G
            time.sleep(POLL_SLEEP_S)
        return None

    def _start_run_if_needed(self) -> None:
        G = self._connection.poll_state()
        if not G or G.get("response"):
            return
        w = G.get("waitingFor")
        if w == "start_run":
            from balatro_connection import Actions
            action_list = [Actions.START_RUN, self._stake, self._deck, self._seed, None]
            self._connection.send_action(action_list)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)
        self._steps = 0
        self._last_round = None
        self._connection.connect()
        if self._run_balatro and self._connection.balatro_instance is None:
            self._connection.start_balatro_instance()
            time.sleep(5)
        self._start_run_if_needed()
        G = self._poll_until_decision()
        if G is None:
            # Fallback: use last poll
            G = self._connection.poll_state()
        if not G:
            G = {}
        self._current_G = G
        obs, full_mask = self._get_obs_and_mask(G)
        info = {"action_mask": full_mask, "raw_state": G}
        if _is_game_over(G):
            info["terminated"] = True
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self._current_G is None:
            raise RuntimeError("reset() must be called before step()")
        G = self._current_G
        action = int(action)
        n = num_actions(G.get("waitingFor") or "")
        if n > 0 and action >= n:
            action = min(action, n - 1)
        action_list = decode_action(G, action)
        response = self._connection.send_action(action_list)
        self._steps += 1
        next_G = response
        if not next_G or next_G.get("response"):
            next_G = self._connection.poll_state()
        next_G = self._poll_until_decision() or next_G or G
        self._current_G = next_G

        terminated = _is_game_over(next_G)
        reward = self._reward_game_over if terminated else 0.0
        truncated = False
        if self._max_steps is not None and self._steps >= self._max_steps:
            truncated = True

        obs, full_mask = self._get_obs_and_mask(next_G)
        info = {"action_mask": full_mask, "raw_state": next_G}
        if terminated:
            info["terminated"] = True
        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Return boolean mask of valid actions for current state (for sb3_contrib MaskablePPO)."""
        if self._current_G is None:
            return np.zeros(self._mask_size, dtype=np.bool_)
        mask = action_mask(self._current_G)
        full_mask = np.zeros(self._mask_size, dtype=np.bool_)
        full_mask[: mask.shape[0]] = mask
        return full_mask

    def close(self) -> None:
        if self._connection and self._run_balatro and self._connection.balatro_instance is not None:
            self._connection.stop_balatro_instance()
        self._current_G = None
