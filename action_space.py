"""
Discrete action space for Balatro RL: encode/decode action lists and compute validity masks.

Action indices are context-dependent (per waitingFor). Use action_mask(G) to get valid actions.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Tuple

import numpy as np

from balatro_connection import Actions

# Match state_encoder
MAX_HAND_SIZE = 8
MAX_JOKERS = 5
MAX_SHOP_CARDS = 5
MAX_SHOP_VOUCHERS = 3
MAX_SHOP_BOOSTERS = 3
MAX_PACK_CARDS = 5


def _nCk(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    out = 1
    for i in range(k):
        out = out * (n - i) // (i + 1)
    return out


def _comb_to_index(combo: Tuple[int, ...], n: int) -> int:
    """Lexicographic index of combination (0-based indices) among C(n, len(combo))."""
    k = len(combo)
    if k == 0:
        return 0
    # https://en.wikipedia.org/wiki/Combinatorial_number_system
    return sum(_nCk(combo[i], i + 1) for i in range(k))


def _index_to_comb(n: int, k: int, index: int) -> Tuple[int, ...]:
    """Return the index-th combination (0-based) of k from n."""
    if k == 0:
        return ()
    comb = []
    idx = index
    for i in range(k, 0, -1):
        # find largest c such that C(c, i) <= idx
        c = i - 1
        while c < n and _nCk(c, i) <= idx:
            c += 1
        c -= 1
        comb.append(c)
        idx -= _nCk(c, i)
    return tuple(reversed(comb))


def _hand_action_counts(n: int) -> Tuple[int, int, int]:
    """(num_play, num_discard, total) for hand size n. Play = 5 cards, discard = 1..min(5,n)."""
    num_play = _nCk(n, 5) if n >= 5 else 0
    num_discard = sum(_nCk(n, k) for k in range(1, min(5, n) + 1))
    return num_play, num_discard, num_play + num_discard


def num_actions(waiting_for: str) -> int:
    """Total number of discrete actions for this context."""
    if waiting_for == "start_run":
        return 1  # only START_RUN with fixed params from bot
    if waiting_for == "skip_or_select_blind":
        return 2
    if waiting_for == "select_cards_from_hand":
        _, _, total = _hand_action_counts(MAX_HAND_SIZE)
        return total
    if waiting_for == "select_shop_action":
        # END_SHOP, REROLL_SHOP, BUY_CARD[1..5], BUY_VOUCHER[1..3], BUY_BOOSTER[1..3]
        return 1 + 1 + MAX_SHOP_CARDS + MAX_SHOP_VOUCHERS + MAX_SHOP_BOOSTERS
    if waiting_for == "select_booster_action":
        # SKIP + SELECT_BOOSTER_CARD[1..MAX_PACK_CARDS]
        return 1 + MAX_PACK_CARDS
    if waiting_for == "sell_jokers":
        return 1 + MAX_JOKERS  # no sell + sell joker 1..5
    if waiting_for in ("rearrange_jokers", "use_or_sell_consumables", "rearrange_consumables", "rearrange_hand"):
        return 1
    return 1


# For Gym: single Discrete space must cover all contexts; mask invalid per state.
WAITING_FOR_CONTEXTS = (
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


def max_action_space_size() -> int:
    """Maximum num_actions across all contexts; use for Gym Discrete(n)."""
    return max(num_actions(w) for w in WAITING_FOR_CONTEXTS)


def _safe_get(obj: Any, *keys: Any, default: Any = 0) -> Any:
    for k in keys:
        if obj is None:
            return default
        if isinstance(obj, dict):
            obj = obj.get(k, default)
        else:
            return default
    return obj


def action_mask(G: Dict[str, Any]) -> np.ndarray:
    """Boolean mask of shape (num_actions(waitingFor),). True = valid action."""
    w = G.get("waitingFor") or ""
    n_actions = num_actions(w)
    mask = np.zeros(n_actions, dtype=np.bool_)

    if w == "start_run":
        mask[0] = True
        return mask
    if w == "skip_or_select_blind":
        # mask[:] = True
        # Temporarily only SELECT_BLIND allowed (skip may out if used two blinds in a row)
        mask[0] = True
        mask[1] = False
        return mask
    if w == "select_cards_from_hand":
        hand = G.get("hand") or []
        n = len(hand)
        discards_left = _safe_get(G, "current_round", "discards_left", default=0)
        num_play, num_discard, _ = _hand_action_counts(MAX_HAND_SIZE)
        # Valid play: combo (5 from 8) must use only indices < n
        if n >= 5:
            for i in range(num_play):
                combo = _index_to_comb(MAX_HAND_SIZE, 5, i)
                if max(combo) < n:
                    mask[i] = True
        # Valid discard: per k, only first C(n,k) combos are valid; only if discards_left > 0
        if discards_left > 0 and n >= 1:
            base = num_play
            for k in range(1, min(5, n) + 1):
                n_valid = _nCk(n, k)
                for sub in range(n_valid):
                    mask[base + sub] = True
                base += _nCk(MAX_HAND_SIZE, k)
        if not mask.any() and n >= 1:
            mask[num_play] = True
        return mask
    if w == "select_shop_action":
        shop = G.get("shop") or {}
        if not isinstance(shop, dict):
            shop = {}
        dollars = float(G.get("dollars", 0))
        cards = shop.get("cards") or []
        vouchers = shop.get("vouchers") or []
        boosters = shop.get("boosters") or []
        # END_SHOP always valid
        mask[0] = True
        # REROLL_SHOP (index 1) - valid if we have reroll cost
        reroll_cost = shop.get("reroll_cost", 999)
        mask[1] = dollars >= reroll_cost
        # BUY_CARD 1..5
        for i in range(min(len(cards), MAX_SHOP_CARDS)):
            cost = cards[i].get("cost", 999) if isinstance(cards[i], dict) else 999
            mask[2 + i] = dollars >= cost
        # BUY_VOUCHER 1..3
        for i in range(min(len(vouchers), MAX_SHOP_VOUCHERS)):
            cost = vouchers[i].get("cost", 999) if isinstance(vouchers[i], dict) else 999
            mask[2 + MAX_SHOP_CARDS + i] = dollars >= cost
        # BUY_BOOSTER 1..3
        for i in range(min(len(boosters), MAX_SHOP_BOOSTERS)):
            cost = boosters[i].get("cost", 999) if isinstance(boosters[i], dict) else 999
            mask[2 + MAX_SHOP_CARDS + MAX_SHOP_VOUCHERS + i] = dollars >= cost
        return mask
    if w == "select_booster_action":
        pack = G.get("pack") or {}
        cards = pack.get("cards") if isinstance(pack, dict) else []
        cards = cards or []
        mask[0] = True  # SKIP always valid
        for i in range(min(len(cards), MAX_PACK_CARDS)):
            mask[1 + i] = True
        return mask
    if w == "sell_jokers":
        mask[0] = True  # no sell
        jokers = G.get("jokers") or []
        for i in range(min(len(jokers), MAX_JOKERS)):
            mask[1 + i] = True
        return mask
    if w in ("rearrange_jokers", "use_or_sell_consumables", "rearrange_consumables", "rearrange_hand"):
        mask[0] = True
        return mask
    mask[0] = True
    return mask


def encode_action(G: Dict[str, Any], action_list: List[Any]) -> int:
    """Encode bot-style action list to discrete action index. Raises ValueError if invalid."""
    w = G.get("waitingFor") or ""
    if not action_list:
        raise ValueError("action_list empty")

    act = action_list[0]
    if isinstance(act, type(Actions.SELECT_BLIND)):
        act = act  # enum
    else:
        raise ValueError("first element must be Actions enum")

    if w == "start_run":
        return 0
    if w == "skip_or_select_blind":
        return 0 if act == Actions.SELECT_BLIND else 1
    if w == "select_cards_from_hand":
        hand = G.get("hand") or []
        n = len(hand)
        num_play, num_discard, _ = _hand_action_counts(MAX_HAND_SIZE)
        if act == Actions.PLAY_HAND:
            indices = sorted(action_list[1])  # 1-based
            if len(indices) != 5:
                raise ValueError("PLAY_HAND requires 5 indices")
            # convert to 0-based and sort
            combo = tuple(sorted(i - 1 for i in indices))
            return _comb_to_index(combo, n)
        if act == Actions.DISCARD_HAND:
            indices = sorted(action_list[1])
            k = len(indices)
            if k < 1 or k > min(5, n):
                raise ValueError("DISCARD_HAND requires 1..5 indices")
            combo = tuple(sorted(i - 1 for i in indices))
            base = num_play
            for j in range(1, k):
                base += _nCk(n, j)
            return base + _comb_to_index(combo, n)
        raise ValueError("expected PLAY_HAND or DISCARD_HAND")
    if w == "select_shop_action":
        if act == Actions.END_SHOP:
            return 0
        if act == Actions.REROLL_SHOP:
            return 1
        if act == Actions.BUY_CARD:
            idx = action_list[1][0]  # 1-based
            return 2 + (idx - 1)
        if act == Actions.BUY_VOUCHER:
            idx = action_list[1][0]
            return 2 + MAX_SHOP_CARDS + (idx - 1)
        if act == Actions.BUY_BOOSTER:
            idx = action_list[1][0]
            return 2 + MAX_SHOP_CARDS + MAX_SHOP_VOUCHERS + (idx - 1)
        raise ValueError("unknown shop action")
    if w == "select_booster_action":
        if act == Actions.SKIP_BOOSTER_PACK:
            return 0
        if act == Actions.SELECT_BOOSTER_CARD:
            idx = action_list[1][0]
            return 1 + (idx - 1)
        raise ValueError("expected SKIP_BOOSTER_PACK or SELECT_BOOSTER_CARD")
    if w == "sell_jokers":
        if act != Actions.SELL_JOKER:
            raise ValueError("expected SELL_JOKER")
        if not action_list[1]:
            return 0
        return action_list[1][0]  # 1-based joker index -> action 1..5
    if w in ("rearrange_jokers", "use_or_sell_consumables", "rearrange_consumables", "rearrange_hand"):
        return 0
    return 0


def decode_action(G: Dict[str, Any], action_id: int) -> List[Any]:
    """Decode discrete action index to bot-style action list for send_action."""
    w = G.get("waitingFor") or ""
    n_actions = num_actions(w)
    if action_id < 0 or action_id >= n_actions:
        raise ValueError(f"action_id {action_id} out of range [0, {n_actions})")

    if w == "start_run":
        # START_RUN needs stake, deck, seed, challenge from bot config - not in G; caller must fill
        return [Actions.START_RUN, 1, "Blue Deck", None, None]
    if w == "skip_or_select_blind":
        return [Actions.SELECT_BLIND] if action_id == 0 else [Actions.SKIP_BLIND]
    if w == "select_cards_from_hand":
        hand = G.get("hand") or []
        n = len(hand)
        num_play, num_discard, _ = _hand_action_counts(MAX_HAND_SIZE)
        if action_id < num_play:
            combo = _index_to_comb(n, 5, action_id)
            indices = [c + 1 for c in combo]
            return [Actions.PLAY_HAND, indices]
        else:
            disc_id = action_id - num_play
            # find k: sum C(n,1)..C(n,k-1) <= disc_id < sum C(n,1)..C(n,k)
            cum = 0
            k = 1
            while k <= min(5, n):
                ck = _nCk(n, k)
                if cum + ck > disc_id:
                    break
                cum += ck
                k += 1
            combo = _index_to_comb(n, k, disc_id - cum)
            indices = [c + 1 for c in combo]
            return [Actions.DISCARD_HAND, indices]
    if w == "select_shop_action":
        if action_id == 0:
            return [Actions.END_SHOP]
        if action_id == 1:
            return [Actions.REROLL_SHOP]
        if action_id < 2 + MAX_SHOP_CARDS:
            return [Actions.BUY_CARD, [action_id - 2 + 1]]  # 1-based
        if action_id < 2 + MAX_SHOP_CARDS + MAX_SHOP_VOUCHERS:
            return [Actions.BUY_VOUCHER, [action_id - (2 + MAX_SHOP_CARDS) + 1]]
        return [Actions.BUY_BOOSTER, [action_id - (2 + MAX_SHOP_CARDS + MAX_SHOP_VOUCHERS) + 1]]
    if w == "select_booster_action":
        if action_id == 0:
            return [Actions.SKIP_BOOSTER_PACK]
        return [Actions.SELECT_BOOSTER_CARD, [action_id - 1 + 1]]
    if w == "sell_jokers":
        if action_id == 0:
            return [Actions.SELL_JOKER, []]
        return [Actions.SELL_JOKER, [action_id]]
    if w == "rearrange_jokers":
        return [Actions.REARRANGE_JOKERS, []]
    if w == "use_or_sell_consumables":
        return [Actions.USE_CONSUMABLE, []]
    if w == "rearrange_consumables":
        return [Actions.REARRANGE_CONSUMABLES, []]
    if w == "rearrange_hand":
        return [Actions.REARRANGE_HAND, []]
    return [Actions.PASS, []]


def _hand_action_counts_for_n(n: int) -> Tuple[int, int, int]:
    """Expose for tests: (num_play, num_discard, total) for a given hand size."""
    return _hand_action_counts(n)
