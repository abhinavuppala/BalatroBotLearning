import pytest
import sys, os

# ensure the workspace root is on sys.path so we can import khang_bot
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from khang_bot import KhangBot, best_hand, find_straight
from bot import Actions


def test_find_straight_basic():
    # simple descending straight
    seq = [(0, 'K'), (1, 'Q'), (2, 'J'), (3, '10'), (4, '9')]
    assert sorted(find_straight(seq)) == [0, 1, 2, 3, 4]

    # duplicates should be ignored
    seq = [(0, 'K'), (1, 'K'), (2, 'Q'), (3, 'J'), (4, '10'), (5, '9')]
    assert sorted(find_straight(seq)) == [0, 2, 3, 4, 5]

    # ace-low straight
    seq = [(0, 'A'), (1, '2'), (2, '3'), (3, '4'), (4, '5')]
    assert sorted(find_straight(seq)) == [0, 1, 2, 3, 4]

    # no straight available
    seq = [(0, 'A'), (1, 'K'), (2, 'Q'), (3, 'J')]
    assert find_straight(seq) is None

    # longer list containing a straight somewhere
    seq = [(0, 'A'), (1, 'K'), (2, 'Q'), (3, 'J'), (4, '10'), (5, '8')]
    assert sorted(find_straight(seq)) == [1, 2, 3, 4, 5] or sorted(find_straight(seq)) == [0, 1, 2, 3, 4]


def test_select_cards_from_real_state():
    """
    For now we just test to make sure it doesn't throw an error
    """
    # load the game state JSON that was dumped by the bot earlier
    import json
    path = os.path.join(os.path.dirname(__file__), "example_gamestate", "select_cards.json")
    with open(path) as f:
        state = json.load(f)

    bot = KhangBot(deck="Blue Deck")
    # invoking should not raise an error
    action = bot.select_cards_from_hand(state)
    print("action returned", action)
    # no assertions yet


def test_select_shop_action_from_real_state():
    """
    For now we just test to make sure it doesn't throw an error
    """
    # load the game state JSON that was dumped by the bot earlier
    import json
    path = os.path.join(os.path.dirname(__file__), "example_gamestate", "select_shop_action.json")
    with open(path) as f:
        state = json.load(f)

    bot = KhangBot(deck="Blue Deck")
    # invoking should not raise an error
    action = bot.select_shop_action(state)
    print("action returned", action)
    # no assertions yet

