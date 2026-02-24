"""
Tests for action_space: discrete action encoding, decoding, and masking.
TDD: define expected behaviour first, then implement action_space.py.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import after path fix; will implement these in action_space.py
try:
    from action_space import (
        num_actions,
        action_mask,
        encode_action,
        decode_action,
        Actions,
    )
    HAS_ACTION_SPACE = True
except ImportError:
    HAS_ACTION_SPACE = False


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


# --- skip_or_select_blind ---


@pytest.mark.skipif(not HAS_ACTION_SPACE, reason="action_space not implemented")
def test_skip_or_select_blind_num_actions():
    n = num_actions("skip_or_select_blind")
    assert n == 2  # SELECT_BLIND, SKIP_BLIND


@pytest.mark.skipif(not HAS_ACTION_SPACE, reason="action_space not implemented")
def test_skip_or_select_blind_mask_shape():
    G = {"waitingFor": "skip_or_select_blind"}
    mask = action_mask(G)
    assert mask.shape == (2,)
    assert mask.dtype in (np.bool_, bool, np.float32, np.int64)
    assert np.all(mask)  # both actions valid in this context


@pytest.mark.skipif(not HAS_ACTION_SPACE, reason="action_space not implemented")
def test_skip_or_select_blind_encode_decode():
    G = {"waitingFor": "skip_or_select_blind"}
    # SELECT_BLIND
    action_list = [Actions.SELECT_BLIND]
    aid = encode_action(G, action_list)
    assert 0 <= aid < 2
    decoded = decode_action(G, aid)
    assert decoded[0] == Actions.SELECT_BLIND
    # SKIP_BLIND
    action_list2 = [Actions.SKIP_BLIND]
    aid2 = encode_action(G, action_list2)
    decoded2 = decode_action(G, aid2)
    assert decoded2[0] == Actions.SKIP_BLIND


# --- select_cards_from_hand ---


@pytest.mark.skipif(not HAS_ACTION_SPACE, reason="action_space not implemented")
def test_select_cards_num_actions():
    # PLAY: C(8,5)=56; DISCARD: C(8,1)+...+C(8,5)=218; total 274 for max hand 8
    n = num_actions("select_cards_from_hand")
    assert n >= 56 + 8  # at least play 5-of-8 + discard 1 card


@pytest.mark.skipif(not HAS_ACTION_SPACE, reason="action_space not implemented")
def test_select_cards_mask_shape():
    G = {"waitingFor": "select_cards_from_hand", "hand": [{"suit": "Hearts", "value": "Ace"}] * 8}
    G["current_round"] = {"discards_left": 1}
    mask = action_mask(G)
    assert mask.shape[0] == num_actions("select_cards_from_hand")
    assert mask.sum() >= 1


@pytest.mark.skipif(not HAS_ACTION_SPACE, reason="action_space not implemented")
def test_select_cards_play_encode_decode_roundtrip():
    G = {"waitingFor": "select_cards_from_hand", "hand": [{"suit": "H", "value": "A"}] * 8}
    G["current_round"] = {"discards_left": 0}
    # Play cards at 1-based indices 1,2,3,4,5
    action_list = [Actions.PLAY_HAND, [1, 2, 3, 4, 5]]
    aid = encode_action(G, action_list)
    assert action_mask(G)[aid]
    decoded = decode_action(G, aid)
    assert decoded[0] == Actions.PLAY_HAND
    assert set(decoded[1]) == {1, 2, 3, 4, 5}


@pytest.mark.skipif(not HAS_ACTION_SPACE, reason="action_space not implemented")
def test_select_cards_discard_encode_decode_roundtrip():
    G = {"waitingFor": "select_cards_from_hand", "hand": [{"suit": "H", "value": "A"}] * 8}
    G["current_round"] = {"discards_left": 3}
    action_list = [Actions.DISCARD_HAND, [2, 5, 7]]
    aid = encode_action(G, action_list)
    assert action_mask(G)[aid]
    decoded = decode_action(G, aid)
    assert decoded[0] == Actions.DISCARD_HAND
    assert set(decoded[1]) == {2, 5, 7}


# --- select_shop_action ---


@pytest.mark.skipif(not HAS_ACTION_SPACE, reason="action_space not implemented")
def test_shop_mask_shape():
    G = {"waitingFor": "select_shop_action", "dollars": 10}
    G["shop"] = {"cards": [{"cost": 5}], "vouchers": [], "boosters": []}
    mask = action_mask(G)
    assert mask.shape[0] == num_actions("select_shop_action")
    assert mask.any()  # at least END_SHOP valid


@pytest.mark.skipif(not HAS_ACTION_SPACE, reason="action_space not implemented")
def test_shop_end_shop_always_valid():
    G = {"waitingFor": "select_shop_action", "shop": []}
    action_list = [Actions.END_SHOP]
    aid = encode_action(G, action_list)
    decoded = decode_action(G, aid)
    assert decoded[0] == Actions.END_SHOP


@pytest.mark.skipif(not HAS_ACTION_SPACE, reason="action_space not implemented")
def test_shop_buy_card_encode_decode_roundtrip():
    G = {"waitingFor": "select_shop_action", "dollars": 10, "shop": {"cards": [{"cost": 3}], "vouchers": [], "boosters": []}}
    action_list = [Actions.BUY_CARD, [1]]
    aid = encode_action(G, action_list)
    assert action_mask(G)[aid]
    decoded = decode_action(G, aid)
    assert decoded[0] == Actions.BUY_CARD
    assert decoded[1] == [1]


# --- select_booster_action ---


@pytest.mark.skipif(not HAS_ACTION_SPACE, reason="action_space not implemented")
def test_booster_skip_valid():
    G = {"waitingFor": "select_booster_action", "pack": {"cards": []}}
    action_list = [Actions.SKIP_BOOSTER_PACK]
    aid = encode_action(G, action_list)
    decoded = decode_action(G, aid)
    assert decoded[0] == Actions.SKIP_BOOSTER_PACK


# --- sell_jokers ---


@pytest.mark.skipif(not HAS_ACTION_SPACE, reason="action_space not implemented")
def test_sell_jokers_num_actions():
    n = num_actions("sell_jokers")
    assert n == 6  # no sell + sell joker 1..5


@pytest.mark.skipif(not HAS_ACTION_SPACE, reason="action_space not implemented")
def test_sell_jokers_no_sell():
    G = {"waitingFor": "sell_jokers", "jokers": [{}]}
    action_list = [Actions.SELL_JOKER, []]
    aid = encode_action(G, action_list)
    decoded = decode_action(G, aid)
    assert decoded[0] == Actions.SELL_JOKER
    assert decoded[1] == []


# --- rearrange / use_or_sell / etc (single no-op) ---


@pytest.mark.skipif(not HAS_ACTION_SPACE, reason="action_space not implemented")
def test_rearrange_jokers_single_action():
    G = {"waitingFor": "rearrange_jokers"}
    n = num_actions("rearrange_jokers")
    assert n == 1
    decoded = decode_action(G, 0)
    assert decoded[0] == Actions.REARRANGE_JOKERS
    assert decoded[1] == []


# --- integration with cached state ---


@pytest.mark.skipif(not HAS_ACTION_SPACE, reason="action_space not implemented")
def test_action_mask_from_cached_select_cards():
    G = _load_cached_state("select_cards_from_hand")
    if G is None:
        pytest.skip("no cached select_cards_from_hand state")
    mask = action_mask(G)
    assert mask.shape[0] == num_actions("select_cards_from_hand")
    assert mask.any()


@pytest.mark.skipif(not HAS_ACTION_SPACE, reason="action_space not implemented")
def test_action_mask_from_cached_shop():
    G = _load_cached_state("select_shop_action")
    if G is None:
        pytest.skip("no cached select_shop_action state")
    mask = action_mask(G)
    assert mask.shape[0] == num_actions("select_shop_action")
    assert mask.any()
