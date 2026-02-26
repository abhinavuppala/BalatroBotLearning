"""
Demo: load a multi-policy PPO model (trained with rl_training.py) and play the
real game using the Bot interface. No training—inference only.

Usage:
  1. Train a model with rl_training.py (or use an existing checkpoint).
  2. Start Balatro with the bot mod on a port (e.g. 12346).
  3. Run: python demo_multipolicy_bot.py --checkpoint <path> [--port 12346] [--deck "Blue Deck"] [--stake 1]

Checkpoint path example (from rl_training.py storage_path + name):
  <storage_path>/blind_shop/blind_shop_00000/checkpoint_000010

The bot uses the blind policy for select_cards_from_hand and the shop policy for
select_shop_action and select_booster_action. Other states (start_run,
skip_or_select_blind, sell_jokers, rearrange_*, etc.) use simple default actions.

When it works you should see:
  - "Starting bot loop (Ctrl+C to stop). Checkpoint: ..." at startup.
  - No further console output unless you pass --verbose (then "[blind] action: ..."
    and "[shop] action: ..." each time), or the game returns an error (then
    "Error from server" / jsondata["response"]), or a policy fails (then
    "[PPOMultiPolicyBot] blind/shop policy failed: ...").
  - In Balatro: the run starts, blind is selected, then the agent plays or
    discards hands each turn, then does shop/booster choices when the round ends.
  - "All padded rows found" is a model-internal message; if the hand encoding
    from the game state doesn't match what the env expects, the policy may still
    return an action—use --verbose to confirm actions are being sent.
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure repo root and gym_envs/modeling are on path for checkpoint loading
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _register_rl_training_components():
    """Register custom models and action dists so RLlib can load the checkpoint."""
    import ray
    from ray.rllib.models import ModelCatalog
    from ray.rllib.algorithms.ppo import PPO

    from modeling.generic_blind_model import BalatroBlindModel as GenericBalatroBlindModel
    from modeling.generic_shop_model import BalatroShopModel as GenericBalatroShopModel
    from modeling.distributions import (
        ARBinaryDistribution,
        ARChooseOrStopDistribution,
        ARChooseOrStopStackedDistribution,
        DualSubsetDistribution,
        ExpertModeCountsDistribution,
        ExpertOptionsDistribution,
        LinearExpertsDistribution,
        ModeCountBinaryDistribution,
        PlayDiscardBinaryDistribution,
        ShopActionAndHandTargetsDistribution,
        SparseSubsetAndMaskDistribution,
    )
    from modeling.gumbel_noise_sampler import GumbelNoiseSamplerDist

    for name, dist in [
        ("play_discard_binary_dist", PlayDiscardBinaryDistribution),
        ("auto_regressive_binary_dist", ARBinaryDistribution),
        ("ar_choose_or_stop_dist", ARChooseOrStopDistribution),
        ("ar_choose_or_stop_stacked_dist", ARChooseOrStopStackedDistribution),
        ("linear_experts_dist", LinearExpertsDistribution),
        ("expert_options_dist", ExpertOptionsDistribution),
        ("mode_count_binary_dist", ModeCountBinaryDistribution),
        ("shop_action_and_hand_targets_dist", ShopActionAndHandTargetsDistribution),
        ("gumbel_noise_sampler", GumbelNoiseSamplerDist),
        ("expert_mode_counts_dist", ExpertModeCountsDistribution),
        ("dual_subset", DualSubsetDistribution),
        ("sparse_subset_and_mask_dist", SparseSubsetAndMaskDistribution),
    ]:
        ModelCatalog.register_custom_action_dist(name, dist)

    ModelCatalog.register_custom_model("generic_blind_model", GenericBalatroBlindModel)
    ModelCatalog.register_custom_model("generic_shop_model", GenericBalatroShopModel)

    # Ray init required before loading checkpoint (use minimal config)
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=1)


def _default_action_for(waiting_for: str):
    """Simple fallback actions for states not handled by the multi-policy model."""
    from balatro_connection import Actions

    match waiting_for:
        case "skip_or_select_blind":
            return [Actions.SELECT_BLIND]
        case "sell_jokers":
            return [Actions.SELL_JOKER, []]
        case "rearrange_jokers":
            return [Actions.REARRANGE_JOKERS, []]
        case "use_or_sell_consumables":
            return [Actions.USE_CONSUMABLE, []]
        case "rearrange_consumables":
            return [Actions.REARRANGE_CONSUMABLES, []]
        case "rearrange_hand":
            return [Actions.REARRANGE_HAND, []]
        case _:
            return None


from bot import Bot


class PPOMultiPolicyBot(Bot):
    """
    Bot implementation that uses a loaded multi-policy PPO (blind + shop) for
    select_cards_from_hand, select_shop_action, and select_booster_action.
    Other states use default actions. Uses the same env state encoding as
    rl_training.py (BlindEnv + ShopEnv load_gamestate / get_obs / action_vector_to_action).
    """

    def __init__(
        self,
        checkpoint_path: str,
        deck: str = "Blue Deck",
        stake: int = 1,
        seed: str | None = None,
        challenge: str | None = None,
        bot_port: int = 12346,
        verbose: bool = False,
    ):
        super().__init__(deck=deck, stake=stake, seed=seed, challenge=challenge, bot_port=bot_port)
        self._checkpoint_path = checkpoint_path
        self._algo = None
        self._blind_env = None
        self._shop_env = None
        self._max_hand_size = 10
        self._verbose = verbose

    def _ensure_loaded(self):
        if self._algo is not None:
            return
        print("[demo] _ensure_loaded: starting (register + ray + checkpoint)...", flush=True)
        _register_rl_training_components()
        from ray.rllib.algorithms.ppo import PPO

        # RLlib/PyArrow require an absolute path (URI with scheme); relative paths fail with "URI has empty scheme"
        path = os.path.abspath(self._checkpoint_path)
        print("[demo] _ensure_loaded: loading checkpoint (this can take a while)...", flush=True)
        try:
            from rl_training import CuriosityPPO
            self._algo = CuriosityPPO.from_checkpoint(path)
        except Exception:
            self._algo = PPO.from_checkpoint(path)
        print("[demo] _ensure_loaded: checkpoint loaded, getting policies...", flush=True)

        self._blind_policy = self._algo.get_policy("blind_agent")
        self._shop_policy = self._algo.get_policy("shop_agent")
        print("[demo] _ensure_loaded: building blind/shop envs...", flush=True)

        # Build envs with same config as rl_training blind_shop so we can use load_gamestate / get_obs / action_vector_to_action
        blind_env_config = {
            "objective_mode": "blind_grind",
            "max_hand_size": self._max_hand_size,
            "action_mode": "stacked_binary_masks",
            "hand_mode": "base_card",
            "num_experts": 3,
            "correct_reward": 1.0,
            "incorrect_penalty": 0.0,
            "discard_potential_reward": 0.0,
            "goal_progress_reward": 0.5,
            "suit_homogeneity_bonus": 0.1,
            "joker_synergy_bonus": 0.0,
            "discard_penalty": 0.0,
            "flattened_rank_chips": False,
            "cannot_discard_obs": True,
            "contained_hand_types_obs": True,
            "subset_hand_types_obs": False,
            "scoring_cards_mask_obs": False,
            "force_play": True,
            "chips_reward_weight": 1.0 / 1000,
            "hand_type_reward_weight": 0.0,
            "infinite_deck": False,
            "bias": 0.0,
            "rarity_bonus": 0.0,
            "target_hand_obs": False,
            "max_jokers": 5,
            "chip_reward_normalization": "log_joker",
            "expert_pretraining": False,
            "deck_obs": False,
            "deck_counts_obs": False,
            "imagined_trajectories": False,
            "joker_count_range": (0, 0),
            "hand_level_range": (1, 1),
            "hand_level_randomization": None,
            "joker_count_bias_exponent": 2,
            "round_range": (1, 1),
        }
        shop_env_config = {"ignore_rarity": True, "max_hand_size": self._max_hand_size}

        from gym_envs.envs.blind_env import BlindEnv
        from gym_envs.envs.shop_env import ShopEnv

        self._blind_env = BlindEnv(blind_env_config)
        self._shop_env = ShopEnv(shop_env_config)
        print("[demo] _ensure_loaded: done.", flush=True)

    def _gamestate_for_shop(self, G: dict) -> dict:
        """Inject pack_cards from G['pack']['cards'] so ShopEnv.load_gamestate can read booster contents."""
        out = dict(G)
        pack = G.get("pack") or {}
        out["pack_cards"] = pack.get("cards", []) if isinstance(pack, dict) else []
        return out

    def skip_or_select_blind(self, G):
        return _default_action_for("skip_or_select_blind")

    def select_cards_from_hand(self, G):
        print("[demo] select_cards_from_hand called", flush=True)
        self._ensure_loaded()
        try:
            if self._verbose:
                print("[demo] blind: loading gamestate", flush=True)
            self._blind_env.load_gamestate(G)
            if self._verbose:
                print("[demo] blind: got gamestate, getting obs", flush=True)
            obs = self._blind_env.get_obs()
            if self._verbose:
                print("[demo] blind: calling policy.compute_single_action (may block)...", flush=True)
            action_vec, _, _ = self._blind_policy.compute_single_action(
                obs, explore=False, clip_actions=True
            )
            if self._verbose:
                print("[demo] blind: policy returned, converting to action list", flush=True)
            action_list = self._blind_env.action_vector_to_action(action_vec)
            if self._verbose:
                print(f"[blind] action: {action_list}", flush=True)
            return action_list
        except Exception as e:
            print(f"[PPOMultiPolicyBot] blind policy failed: {e}, using default play")
            from balatro_connection import Actions
            hand = G.get("hand") or []
            if len(hand) >= 5:
                return [Actions.PLAY_HAND, [1, 2, 3, 4, 5]]
            return [Actions.DISCARD_HAND, [1]]

    def select_shop_action(self, G):
        return self._shop_step(G)

    def select_booster_action(self, G):
        return self._shop_step(G)

    def _shop_step(self, G):
        self._ensure_loaded()
        G_shop = self._gamestate_for_shop(G)
        try:
            self._shop_env.load_gamestate(G_shop)
            obs = self._shop_env.get_obs()
            action_vec, _, _ = self._shop_policy.compute_single_action(
                obs, explore=False, clip_actions=True
            )
            if isinstance(action_vec, dict):
                # Ensure native types for action_vector_to_action
                action_vec = {
                    "action": int(action_vec["action"]),
                    "hand_targets": [int(x) for x in action_vec["hand_targets"]],
                }
            else:
                action_vec = {
                    "action": int(action_vec),
                    "hand_targets": [0] * self._shop_env.G.max_hand_size,
                }
            action_list = self._shop_env.action_vector_to_action(action_vec)
            if self._verbose:
                print(f"[shop] action: {action_list}")
            return action_list
        except Exception as e:
            print(f"[PPOMultiPolicyBot] shop policy failed: {e}, using default end/skip")
            from balatro_connection import Actions
            if G.get("waitingFor") == "select_booster_action":
                return [Actions.SKIP_BOOSTER_PACK]
            return [Actions.END_SHOP]

    def sell_jokers(self, G):
        return _default_action_for("sell_jokers")

    def rearrange_jokers(self, G):
        return _default_action_for("rearrange_jokers")

    def use_or_sell_consumables(self, G):
        return _default_action_for("use_or_sell_consumables")

    def rearrange_consumables(self, G):
        return _default_action_for("rearrange_consumables")

    def rearrange_hand(self, G):
        return _default_action_for("rearrange_hand")


def main():
    parser = argparse.ArgumentParser(description="Run multi-policy PPO bot (demo, no training)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to RLlib checkpoint directory (e.g. .../checkpoint_000010)")
    parser.add_argument("--port", type=int, default=12346, help="Bot port")
    parser.add_argument("--deck", type=str, default="Blue Deck")
    parser.add_argument("--stake", type=int, default=1)
    parser.add_argument("--seed", type=str, default=None)
    parser.add_argument("--no-run-balatro", action="store_true", help="Do not start Balatro; connect to existing instance")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print each action sent (blind/shop) for debugging")
    args = parser.parse_args()

    if not os.path.isdir(args.checkpoint):
        print(f"Checkpoint path is not a directory: {args.checkpoint}")
        sys.exit(1)

    from gamestates import cache_state  # noqa: F401 - used by Bot.run_step

    bot = PPOMultiPolicyBot(
        checkpoint_path=args.checkpoint,
        deck=args.deck,
        stake=args.stake,
        seed=args.seed,
        bot_port=args.port,
        verbose=args.verbose,
    )

    if not args.no_run_balatro:
        bot.start_balatro_instance()

    print("Starting bot loop (Ctrl+C to stop). Checkpoint:", args.checkpoint)
    try:
        bot.run()
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        if not args.no_run_balatro:
            bot.stop_balatro_instance()


if __name__ == "__main__":
    main()
