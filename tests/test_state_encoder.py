"""Tests for state_encoder: JSON state -> tensor encoding."""

import json
import os
import sys

import pytest

# Project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from state_encoder import (
    encode_card,
    encode_hand,
    encode_current_round,
    encode_global_scalars,
    encode_categoricals,
    encode_shop,
    encode_jokers,
    state_to_numpy,
    flatten_observation,
    observation_space_shape,
    flat_observation_size,
    SUITS,
    RANK_STR_TO_ID,
    WAITING_FOR,
    MAX_HAND_SIZE,
)


def test_encode_card():
    assert encode_card({}) == (0, 0)
    assert encode_card({"suit": "Hearts", "value": "Ace"}) == (1, 12)
    assert encode_card({"suit": "Spades", "value": "10"}) == (4, 8)


def test_encode_hand_empty():
    cards, mask = encode_hand([])
    assert cards.shape == (MAX_HAND_SIZE, 2)
    assert mask.shape == (MAX_HAND_SIZE,)
    assert mask.sum() == 0


def test_encode_hand_partial():
    hand = [
        {"suit": "Hearts", "value": "Ace"},
        {"suit": "Clubs", "value": "7"},
    ]
    cards, mask = encode_hand(hand)
    assert cards.shape == (MAX_HAND_SIZE, 2)
    assert mask.sum() == 2
    assert cards[0, 0] == 1 and cards[0, 1] == 12
    assert cards[1, 0] == 3 and cards[1, 1] == 5


def test_encode_current_round_empty():
    r = encode_current_round({})
    assert r["discards_left"] == 0
    assert r["hands_left"] == 0


def test_encode_current_round_from_G():
    G = {"current_round": {"discards_left": 2, "hands_left": 4, "chips_required": 300}}
    r = encode_current_round(G)
    assert r["discards_left"] == 2
    assert r["hands_left"] == 4
    assert r["chips_required"] == 300


def test_encode_global_scalars():
    G = {"dollars": 10, "chips": 100, "round": 2, "waitingForAction": True}
    s = encode_global_scalars(G)
    assert s["dollars"] == 10
    assert s["chips"] == 100
    assert s["round"] == 2
    assert s["waiting_for_action"] == 1.0


def test_encode_categoricals():
    G = {"state": 5, "waitingFor": "select_shop_action", "ante": {"blinds": {"ondeck": "Small"}}}
    c = encode_categoricals(G)
    assert c["state"] == 5
    assert c["waiting_for"] == WAITING_FOR.index("select_shop_action")
    assert c["blind_ondeck"] == 1


def test_encode_shop_empty():
    card_enc, v_costs, b_costs, card_mask = encode_shop({})
    assert card_mask.sum() == 0
    assert v_costs.shape[0] == 3
    assert b_costs.shape[0] == 3


def test_state_to_numpy_empty():
    enc = state_to_numpy({})
    assert "hand_cards" in enc
    assert "hand_mask" in enc
    assert enc["hand_cards"].shape == (MAX_HAND_SIZE, 2)
    assert enc["state"].shape == (1,)
    assert enc["dollars"].shape == (1,)


def test_state_to_numpy_from_cached_json():
    cache_path = os.path.join(
        os.path.dirname(__file__), "..", "gamestate_cache", "select_cards_from_hand"
    )
    if not os.path.isdir(cache_path):
        pytest.skip("No gamestate_cache/select_cards_from_hand")
    files = [f for f in os.listdir(cache_path) if f.endswith(".json")]
    if not files:
        pytest.skip("No cached JSON in select_cards_from_hand")
    with open(os.path.join(cache_path, files[0]), "r") as f:
        G = json.load(f)
    enc = state_to_numpy(G)
    hand_size = int(enc["hand_size"][0])
    assert hand_size == len(G.get("hand") or [])
    assert enc["hand_mask"].sum() == hand_size
    assert enc["waiting_for"][0] == WAITING_FOR.index("select_cards_from_hand")


def test_flatten_observation():
    enc = state_to_numpy({})
    obs, mask = flatten_observation(enc)
    assert obs.ndim == 1
    assert mask.ndim == 1
    assert obs.shape[0] == mask.shape[0]


def test_flat_observation_size():
    n = flat_observation_size()
    assert n > 0
    enc = state_to_numpy({})
    obs, _ = flatten_observation(enc)
    assert obs.shape[0] == n


def test_observation_space_shape():
    shapes = observation_space_shape()
    assert "hand_cards" in shapes
    assert shapes["hand_cards"] == (MAX_HAND_SIZE, 2)
    assert "scalars" in shapes
