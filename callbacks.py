"""
Custom callbacks for Balatro PPO training: TensorBoard metrics from game state.

Logged metrics (per rollout):
- balatro/max_round_reached: max round (ante) seen in the rollout
- balatro/mean_chips: mean of G['chips'] over steps
- balatro/mean_chips_required: mean blind target (current_round.chips_required)
- balatro/chips_to_blind_ratio: mean of (chips / chips_required) when in a round
- balatro/num_decision_steps: number of steps in the rollout

Hand-type distribution can be added when the game state exposes last hand played.
"""

from __future__ import annotations

from typing import Any, Dict, List

try:
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:
    BaseCallback = None  # type: ignore


def _safe_get(obj: Any, *keys: Any, default: Any = 0) -> Any:
    for k in keys:
        if obj is None:
            return default
        if isinstance(obj, dict):
            obj = obj.get(k, default)
        else:
            return default
    return obj


def _extract_metrics_from_rollout_infos(infos: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute Balatro-specific metrics from a rollout's info dicts (each has raw_state = G)."""
    rounds: List[float] = []
    chips_required_list: List[float] = []
    chips_list: List[float] = []
    chips_to_blind_ratios: List[float] = []

    for info in infos:
        if not isinstance(info, dict):
            continue
        G = info.get("raw_state") or {}
        r = _safe_get(G, "round")
        if r is not None:
            rounds.append(float(r))
        cr = _safe_get(G, "current_round")
        if isinstance(cr, dict):
            req = cr.get("chips_required") or 0
            chips_required_list.append(float(req))
        chips = _safe_get(G, "chips", default=0)
        chips_list.append(float(chips))
        req = _safe_get(G, "current_round", "chips_required", default=0)
        if req and float(req) > 0:
            chips_to_blind_ratios.append(float(chips) / float(req))

    max_round = max(rounds) if rounds else 0.0
    mean_chips = sum(chips_list) / len(chips_list) if chips_list else 0.0
    mean_chips_required = sum(chips_required_list) / len(chips_required_list) if chips_required_list else 0.0
    mean_ratio = sum(chips_to_blind_ratios) / len(chips_to_blind_ratios) if chips_to_blind_ratios else 0.0

    return {
        "balatro/max_round_reached": max_round,
        "balatro/mean_chips": mean_chips,
        "balatro/mean_chips_required": mean_chips_required,
        "balatro/chips_to_blind_ratio": mean_ratio,
        "balatro/num_decision_steps": len(infos),
    }


class BalatroTensorboardCallback(BaseCallback):
    """
    Logs Balatro game metrics to TensorBoard each rollout from env step infos.
    Requires info['raw_state'] to be the game state dict G.
    """

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        raw_infos = getattr(self.model.rollout_buffer, "infos", None)
        if not raw_infos or not self.logger:
            return
        # Flatten: single env -> list of dicts; VecEnv -> list of list of dicts
        infos: List[Dict[str, Any]] = []
        for x in raw_infos:
            if isinstance(x, dict):
                infos.append(x)
            elif isinstance(x, (list, tuple)):
                infos.extend(i for i in x if isinstance(i, dict))
        metrics = _extract_metrics_from_rollout_infos(infos)
        for name, value in metrics.items():
            self.logger.record(name, value)
