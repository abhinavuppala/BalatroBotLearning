"""
Encode Balatro JSON game state → fixed-size tensors for RL.

Outputs:
- Scalars: dollars, chips, round, discards_left, hands_left, chips_required, etc.
- Categoricals: state, waitingFor, blind ondeck (as indices)
- Hand: fixed [MAX_HAND, 2] (suit_idx, rank_idx) + hand_mask
- Optional: jokers/shop encodings (fixed slots, padded)

All tensors are float32 by default; use long dtype for discrete indices if needed.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# --- Fixed sizes (match game limits) ---
MAX_HAND_SIZE = 8
MAX_JOKERS = 5
MAX_SHOP_CARDS = 5
MAX_SHOP_VOUCHERS = 3
MAX_SHOP_BOOSTERS = 3
MAX_CONSUMABLES = 2
MAX_PACK_CARDS = 5

# --- Vocabularies (order must be stable) ---
SUITS = ("Hearts", "Diamonds", "Clubs", "Spades")
RANK_STR_TO_ID = {
    "2": 0, "3": 1, "4": 2, "5": 3, "6": 4, "7": 5, "8": 6, "9": 7,
    "10": 8, "T": 8, "TEN": 8,
    "Jack": 9, "J": 9, "JACK": 9,
    "Queen": 10, "Q": 10, "QUEEN": 10,
    "King": 11, "K": 11, "KING": 11,
    "Ace": 12, "A": 12, "ACE": 12,
}
WAITING_FOR = (
    "start_run",
    "skip_or_select_blind",
    "select_cards_from_hand",
    "select_shop_action",
    "select_booster_action",
    "sell_jokers",
    "rearrange_jokers",
    "use_or_sell_consumables",
    "rearrange_consumables",
    "rearrange_hand",
)
BLIND_ONDECK = ("", "Small", "Big")  # empty when not at blind select; add boss names later if needed
GAME_STATE_IDS = list(range(1, 20))  # state 1..19 from balatro_connection.State


def _rank_to_id(value: Any) -> int:
    if value is None:
        return 0
    s = str(value).strip()
    return RANK_STR_TO_ID.get(s, 0)


def _suit_to_id(suit: Any) -> int:
    if suit is None:
        return 0
    try:
        return SUITS.index(suit) + 1  # 1..4, 0 = no card
    except (ValueError, TypeError):
        return 0


def _safe_get(obj: Any, *keys: Union[str, int], default: Any = 0) -> Any:
    for k in keys:
        if obj is None:
            return default
        if isinstance(obj, dict):
            obj = obj.get(k, default)
        else:
            return default
    return obj


def encode_card(card: Dict[str, Any]) -> Tuple[int, int]:
    """Encode a single card to (suit_id, rank_id). suit_id 0 = missing, 1-4 = suits."""
    suit = _safe_get(card, "suit")
    value = _safe_get(card, "value")
    return _suit_to_id(suit), _rank_to_id(value)


def encode_hand(hand: List[Dict], max_size: int = MAX_HAND_SIZE) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode hand to (cards, mask).
    - cards: [max_size, 2] int (suit_id, rank_id); padded slots are (0, 0).
    - mask: [max_size] float 0/1, 1 = real card.
    """
    cards = np.zeros((max_size, 2), dtype=np.int64)
    mask = np.zeros(max_size, dtype=np.float32)
    n = min(len(hand), max_size)
    for i in range(n):
        cards[i, 0], cards[i, 1] = encode_card(hand[i])
        mask[i] = 1.0
    return cards, mask


def encode_current_round(G: Dict[str, Any]) -> Dict[str, float]:
    """Extract current round scalars (normalized where useful)."""
    cr = _safe_get(G, "current_round") or {}
    if not isinstance(cr, dict):
        cr = {}
    return {
        "discards_left": float(_safe_get(cr, "discards_left", default=0)),
        "discards_used": float(_safe_get(cr, "discards_used", default=0)),
        "hands_left": float(_safe_get(cr, "hands_left", default=0)),
        "hands_played": float(_safe_get(cr, "hands_played", default=0)),
        "round_dollars": float(_safe_get(cr, "dollars", default=0)),
        "chips_required": float(_safe_get(cr, "chips_required", default=0)),
    }


def encode_global_scalars(G: Dict[str, Any]) -> Dict[str, float]:
    """Global numeric state; optionally log-scale or normalize for RL."""
    dollars = float(_safe_get(G, "dollars", default=0))
    chips = float(_safe_get(G, "chips", default=0))
    return {
        "dollars": dollars,
        "dollars_log": math.log1p(max(0, dollars)),
        "chips": chips,
        "chips_log": math.log1p(max(0, chips)),
        "round": float(_safe_get(G, "round", default=0)),
        "num_hands_played": float(_safe_get(G, "num_hands_played", default=0)),
        "max_jokers": float(_safe_get(G, "max_jokers", default=5)),
        "interest_cap": float(_safe_get(G, "interest_cap", default=25)),
        "discount_percent": float(_safe_get(G, "discount_percent", default=0)),
        "inflation": float(_safe_get(G, "inflation", default=0)),
        "waiting_for_action": 1.0 if G.get("waitingForAction") else 0.0,
    }


def encode_categoricals(G: Dict[str, Any]) -> Dict[str, int]:
    """State and action context as discrete indices."""
    state = G.get("state")
    if state is not None and hasattr(state, "value"):
        state = state.value
    state_id = int(state) if state is not None else 0
    state_id = max(0, min(19, state_id))

    waiting = G.get("waitingFor") or ""
    try:
        waiting_id = WAITING_FOR.index(waiting)
    except ValueError:
        waiting_id = 0

    ondeck = _safe_get(G, "ante", "blinds", "ondeck")
    if ondeck is None or ondeck == 0:
        ondeck = ""
    try:
        blind_id = BLIND_ONDECK.index(ondeck)
    except (ValueError, TypeError):
        blind_id = 0

    return {
        "state": state_id,
        "waiting_for": waiting_id,
        "blind_ondeck": blind_id,
    }


def encode_shop(G: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Shop: cards (cost + is_joker flag), vouchers (cost), boosters (cost), mask.
    When shop is [] or missing, all costs = 0, mask = 0.
    """
    shop = G.get("shop")
    if not isinstance(shop, dict):
        shop = {}
    cards_list = shop.get("cards") or []
    vouchers_list = shop.get("vouchers") or []
    boosters_list = shop.get("boosters") or []

    # Shop cards: [MAX_SHOP_CARDS, 2] -> (cost, is_joker 0/1)
    card_enc = np.zeros((MAX_SHOP_CARDS, 2), dtype=np.float32)
    card_mask = np.zeros(MAX_SHOP_CARDS, dtype=np.float32)
    for i, c in enumerate(cards_list[:MAX_SHOP_CARDS]):
        if isinstance(c, dict):
            card_enc[i, 0] = float(c.get("cost", 0))
            ab = c.get("ability") or {}
            card_enc[i, 1] = 1.0 if (ab.get("set") or "").upper() == "JOKER" else 0.0
        card_mask[i] = 1.0

    voucher_costs = np.zeros(MAX_SHOP_VOUCHERS, dtype=np.float32)
    voucher_mask = np.zeros(MAX_SHOP_VOUCHERS, dtype=np.float32)
    for i, v in enumerate(vouchers_list[:MAX_SHOP_VOUCHERS]):
        if isinstance(v, dict):
            voucher_costs[i] = float(v.get("cost", 0))
        voucher_mask[i] = 1.0

    booster_costs = np.zeros(MAX_SHOP_BOOSTERS, dtype=np.float32)
    booster_mask = np.zeros(MAX_SHOP_BOOSTERS, dtype=np.float32)
    for i, b in enumerate(boosters_list[:MAX_SHOP_BOOSTERS]):
        if isinstance(b, dict):
            booster_costs[i] = float(b.get("cost", 0))
        booster_mask[i] = 1.0

    return card_enc, voucher_costs, booster_costs, card_mask


def encode_jokers(G: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Jokers: simple count / placeholder per slot; mask for filled slots."""
    jokers = G.get("jokers") or []
    n = min(len(jokers), MAX_JOKERS)
    # For now: binary mask; could add joker type ids later
    mask = np.zeros(MAX_JOKERS, dtype=np.float32)
    mask[:n] = 1.0
    # Placeholder: ones where joker present (could encode ability set later)
    slots = np.zeros(MAX_JOKERS, dtype=np.float32)
    slots[:n] = 1.0
    return slots, mask


def encode_pack(G: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Booster pack cards (when waitingFor select_booster_action)."""
    pack = G.get("pack") or {}
    cards_list = pack.get("cards") if isinstance(pack, dict) else []
    cards_list = cards_list or []
    n = min(len(cards_list), MAX_PACK_CARDS)
    count_arr = np.zeros(MAX_PACK_CARDS, dtype=np.float32)
    count_arr[0] = float(n)
    mask = np.zeros(MAX_PACK_CARDS, dtype=np.float32)
    mask[:n] = 1.0
    return count_arr, mask


def state_to_numpy(G: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Encode full JSON state to a dict of numpy arrays.
    Handles missing/empty keys; no torch dependency.
    """
    hand_cards, hand_mask = encode_hand(G.get("hand") or [])
    round_ = encode_current_round(G)
    global_ = encode_global_scalars(G)
    cat = encode_categoricals(G)
    shop_cards, shop_voucher_costs, shop_booster_costs, shop_card_mask = encode_shop(G)
    joker_slots, joker_mask = encode_jokers(G)

    return {
        "hand_cards": hand_cards,
        "hand_mask": hand_mask,
        "hand_size": np.array([len(G.get("hand") or [])], dtype=np.float32),
        "discards_left": np.array([round_["discards_left"]], dtype=np.float32),
        "hands_left": np.array([round_["hands_left"]], dtype=np.float32),
        "chips_required": np.array([round_["chips_required"]], dtype=np.float32),
        "dollars": np.array([global_["dollars"]], dtype=np.float32),
        "chips": np.array([global_["chips"]], dtype=np.float32),
        "round": np.array([global_["round"]], dtype=np.float32),
        "state": np.array([cat["state"]], dtype=np.int64),
        "waiting_for": np.array([cat["waiting_for"]], dtype=np.int64),
        "blind_ondeck": np.array([cat["blind_ondeck"]], dtype=np.int64),
        "shop_cards": shop_cards,
        "shop_voucher_costs": shop_voucher_costs,
        "shop_booster_costs": shop_booster_costs,
        "shop_card_mask": shop_card_mask,
        "joker_slots": joker_slots,
        "joker_mask": joker_mask,
        # Flattened scalar stack for simple policies
        "scalars": np.array([
            global_["dollars_log"], global_["chips_log"], global_["round"],
            round_["discards_left"], round_["hands_left"], round_["chips_required"],
            global_["waiting_for_action"],
        ], dtype=np.float32),
    }


def flatten_observation(encoded: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Produce a single flat observation vector and a validity mask.
    Mask is 1.0 where the slot is valid (hand slot has a card, shop slot has an item).
    Same length as obs: hand card dims use hand_mask (repeated for suit+rank), rest are 1.
    """
    parts = [
        encoded["hand_cards"].reshape(-1),
        encoded["hand_mask"],
        encoded["hand_size"],
        encoded["discards_left"],
        encoded["hands_left"],
        encoded["chips_required"],
        encoded["dollars"],
        encoded["chips"],
        encoded["round"],
        encoded["state"].astype(np.float32),
        encoded["waiting_for"].astype(np.float32),
        encoded["blind_ondeck"].astype(np.float32),
        encoded["shop_cards"].reshape(-1),
        encoded["shop_voucher_costs"],
        encoded["shop_booster_costs"],
        encoded["shop_card_mask"],
        encoded["joker_slots"],
        encoded["joker_mask"],
    ]
    obs = np.concatenate([np.asarray(p).reshape(-1) for p in parts])
    # Mask: same length as obs; 0 = padding (hand/shop slots), 1 = valid
    hand_mask_2 = np.repeat(encoded["hand_mask"], 2)  # per card: suit, rank
    shop_mask_2 = np.repeat(encoded["shop_card_mask"], 2)
    mask_parts = [
        hand_mask_2,
        encoded["hand_mask"],
        np.ones(1 + 3 + 3 + 3, dtype=np.float32),
        shop_mask_2,
        np.ones(MAX_SHOP_VOUCHERS + MAX_SHOP_BOOSTERS + MAX_SHOP_CARDS + MAX_JOKERS * 2, dtype=np.float32),  # voucher, booster, shop_card_mask, joker_slots, joker_mask
    ]
    mask = np.concatenate(mask_parts)
    if mask.shape[0] != obs.shape[0]:
        mask = np.ones_like(obs)  # fallback: all valid
    else:
        mask = mask.astype(np.float32)
    return obs.astype(np.float32), mask


def state_to_tensor(
    G: Dict[str, Any],
    device: Optional[Union[str, "torch.device"]] = None,
) -> Dict[str, "torch.Tensor"]:
    """
    Encode JSON state to dict of PyTorch tensors.
    Raises ImportError if torch not available.
    """
    if not HAS_TORCH:
        raise ImportError("state_to_tensor requires PyTorch")
    encoded = state_to_numpy(G)
    out = {}
    for k, v in encoded.items():
        t = torch.from_numpy(np.asarray(v))
        if device is not None:
            t = t.to(device)
        out[k] = t
    return out


def observation_space_shape() -> Dict[str, Tuple[int, ...]]:
    """Return shapes of each component (for building RL obs space)."""
    encoded = state_to_numpy({})
    return {k: v.shape for k, v in encoded.items()}


def flat_observation_size() -> int:
    """Total size of the flattened observation vector."""
    obs, _ = flatten_observation(state_to_numpy({}))
    return int(obs.shape[0])


if __name__ == "__main__":
    import json
    import os

    print("Observation space shapes:", observation_space_shape())
    print("Flat observation size:", flat_observation_size())

    # Try loading a cached state if present
    for subdir in ("select_cards_from_hand", "select_shop_action", "skip_or_select_blind"):
        cache_dir = os.path.join("gamestate_cache", subdir)
        if os.path.isdir(cache_dir):
            files = [f for f in os.listdir(cache_dir) if f.endswith(".json")]
            if files:
                path = os.path.join(cache_dir, files[0])
                with open(path, "r") as f:
                    G = json.load(f)
                enc = state_to_numpy(G)
                obs, mask = flatten_observation(enc)
                print(f"\nExample from {path}:")
                print(f"  waitingFor -> waiting_for index: {enc['waiting_for'][0]}")
                print(f"  hand_size: {enc['hand_size'][0]}, dollars: {enc['dollars'][0]}")
                print(f"  flat obs shape: {obs.shape}, mask shape: {mask.shape}")
                if HAS_TORCH:
                    tensors = state_to_tensor(G)
                    print(f"  state_to_tensor keys: {list(tensors.keys())}")
                break
    else:
        print("\nNo gamestate_cache found; run a bot once to generate cached states.")
