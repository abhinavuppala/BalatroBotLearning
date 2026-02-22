"""
Docstring for khang_bot

Khang's logic for a more advanced rule-based bot,
transcribed to use the existing botting API like flush_bot.
- Finds best hand to play at any turn
    - if high card/pair, discards to try and find something better
    - otherwise, plays the best hand at the time
- tries to buy up to 5 jokers as soon as possible
- once jokers are bought, falls back on buying packs & vouchers
"""

from bot import Bot, Actions
from gamestates import cache_state

# Helper functions copied / adapted from khang_script.py

def find_straight(cards_with_index):
    """Return a list of 5 indices that form a straight, or None if none exists.

    ``cards_with_index`` is a list of (index, rank) tuples and is assumed to be
    sorted by rank value descending.  The function handles duplicate ranks and
    the special ace-low straight A-2-3-4-5.
    """

    def rank_value(rank):
        order = {
            'A': 14,
            'K': 13,
            'Q': 12,
            'J': 11,
            'T': 10,
            '10': 10,
            '9': 9,
            '8': 8,
            '7': 7,
            '6': 6,
            '5': 5,
            '4': 4,
            '3': 3,
            '2': 2,
        }
        return order.get(rank, 0)

    if len(cards_with_index) < 5:
        return None

    # remove duplicates, keeping highest index for each rank
    seen = {}
    for idx, rank in cards_with_index:
        if rank not in seen:
            seen[rank] = idx

    unique = [(idx, rank) for rank, idx in seen.items()]
    unique.sort(key=lambda x: rank_value(x[1]), reverse=True)

    for i in range(len(unique) - 4):
        consec = [unique[i]]
        for j in range(i + 1, len(unique)):
            prev_rank = rank_value(consec[-1][1])
            curr_rank = rank_value(unique[j][1])
            if prev_rank - curr_rank == 1:
                consec.append(unique[j])
                if len(consec) == 5:
                    return [idx for idx, _ in consec]
            elif prev_rank == curr_rank:
                continue
            else:
                break

    # ace-low check
    if any(r == 'A' for _, r in unique):
        low = []
        for target in ['5', '4', '3', '2']:
            for idx, rank in unique:
                if rank == target:
                    low.append((idx, rank))
                    break
        if len(low) == 4:
            for idx, rank in unique:
                if rank == 'A':
                    low.append((idx, rank))
                    return [idx for idx, _ in low]
    return None


def best_hand(cards):
    """Return a tuple (hand_name, indices) of the best possible five-card hand
    contained within ``cards``.

    The algorithm is the same as the helper in ``khang_script.py``; it
    identifies straight flush, four‑of‑a‑kind, full house, flush, straight,
    three of a kind, two pair, one pair, and finally high card.
    """

    # translate to simpler representation
    hand_data = []
    for i, card in enumerate(cards):
        suit = card['suit']
        rank = card['value']
        hand_data.append((i, suit, rank))

    def rv(rank):
        order = {
            'A': 14,
            'ACE': 14,
            'K': 13,
            'KING': 13,
            'Q': 12,
            'QUEEN': 12,
            'J': 11,
            'JACK': 11,
            'T': 10,
            'TEN': 10,
            '10': 10,
            '9': 9,
            '8': 8,
            '7': 7,
            '6': 6,
            '5': 5,
            '4': 4,
            '3': 3,
            '2': 2,
        }
        return order.get(rank.upper(), 0)

    from collections import Counter

    rank_counts = Counter(r for _, _, r in hand_data)
    suit_counts = Counter(s for _, s, _ in hand_data)

    # straight flush
    for suit, cnt in suit_counts.items():
        if cnt >= 5:
            suited = [(i, r) for i, s, r in hand_data if s == suit]
            suited.sort(key=lambda x: rv(x[1]), reverse=True)
            straight_ix = find_straight(suited)
            if straight_ix:
                return ("Straight Flush", straight_ix)

    # four of a kind
    for rank, cnt in rank_counts.items():
        if cnt == 4:
            indices = [i for i, _, r in hand_data if r == rank]
            return ("Four of a Kind", indices)

    # full house
    threes = [r for r, c in rank_counts.items() if c == 3]
    pairs = [r for r, c in rank_counts.items() if c == 2]
    if threes and pairs:
        three_ix = [i for i, _, r in hand_data if r == threes[0]]
        pair_ix = [i for i, _, r in hand_data if r == pairs[0]]
        return ("Full House", three_ix + pair_ix[:2])

    # flush
    for suit, cnt in suit_counts.items():
        if cnt >= 5:
            indices = [i for i, s, _ in hand_data if s == suit][:5]
            return ("Flush", indices)

    # straight
    sorted_cards = sorted(hand_data, key=lambda x: rv(x[2]), reverse=True)
    straight_ix = find_straight([(i, r) for i, _, r in sorted_cards])
    if straight_ix:
        return ("Straight", straight_ix)

    # three of a kind
    if threes:
        indices = [i for i, _, r in hand_data if r == threes[0]]
        others = [i for i, _, r in hand_data if r != threes[0]][:2]
        return ("Three of a Kind", indices + others)

    # two pair
    if len(pairs) >= 2:
        p1 = [i for i, _, r in hand_data if r == pairs[0]]
        p2 = [i for i, _, r in hand_data if r == pairs[1]]
        other = [i for i, _, r in hand_data if r not in pairs][:1]
        return ("Two Pair", p1 + p2 + other)

    # one pair
    if pairs:
        pair_ix = [i for i, _, r in hand_data if r == pairs[0]]
        others = [i for i, _, r in hand_data if r != pairs[0]][:3]
        return ("Pair", pair_ix + others)

    # high card
    return ("High Card", list(range(min(5, len(cards)))))


class KhangBot(Bot):
    """Bot implementation ported from ``khang_script.py``.

    The two distinguishing behaviours are:

    * always attempt to buy jokers during the shop phase until the joker slots
      are full, otherwise fall back to buying boosters or vouchers;
    * when choosing cards from hand play the best hand possible, discarding up
      to three cards if only a high card or a single pair is held and discards
      remain.
    """

    def skip_or_select_blind(self, G):
        cache_state("skip_or_select_blind", G)
        # script always "select" the blind
        return [Actions.SELECT_BLIND]

    def select_cards_from_hand(self, G):
        cache_state("select_cards_from_hand", G)
        hand_cards = G["hand"]
        discards_left = G.get("current_round", {}).get("discards_left", 0)

        hand_name, card_indices = best_hand(hand_cards)
        if discards_left > 0 and hand_name in ["High Card", "Pair"]:
            keep = set(card_indices)
            discard_ix = [i for i in range(len(hand_cards)) if i not in keep][:3]
            if discard_ix:
                print(f' | Discarding Cards no. {[i+1 for i in discard_ix]}')
                return [Actions.DISCARD_HAND, [i+1 for i in discard_ix]]

        print(f' | Playing Cards no. {[i+1 for i in card_indices]}')
        return [Actions.PLAY_HAND, [i+1 for i in card_indices]]

    def select_shop_action(self, G):
        cache_state("select_shop_action", G)
        money = G.get("dollars", 0)
        shop_cards = G.get("shop", {}).get("cards", [])
        voucher_cards = G.get("shop", {}).get("vouchers", [])
        pack_cards = G.get("shop", {}).get("boosters", [])

        jokers_data = G.get("jokers", [])
        current_jokers = len(jokers_data)
        max_jokers = 5 # FOR NOW we hardcode max jokers to 5

        # priority 1 – buy a joker if there is room and we can afford one
        if current_jokers < max_jokers:
            for i, card in enumerate(shop_cards):
                if card.get('ability', {}).get("set", "").upper() == "JOKER" and card.get("cost", 1_000_000_000) <= money:
                    print(f"  % Buying Joker {i+1}")
                    return [Actions.BUY_CARD, [i+1]]
            # if no joker available affordable, simply end the shop and wait for
            # next round; the script would normally "save money" here.
            print(f"  % Ending shop, no affordable jokers")
            return [Actions.END_SHOP]

        # priority 2 – buy a pack if affordable
        for i, pack in enumerate(pack_cards):
            if pack.get("cost", 1_000_000_000) <= money:
                print(f"  % Buying Booster {i+1}")
                return [Actions.BUY_BOOSTER, [i+1]]

        # priority 3 – buy a voucher if affordable
        for i, voucher in enumerate(voucher_cards):
            if voucher.get("cost", 1_000_000_000) <= money:
                print(f"  % Buying Voucher {i+1}")
                return [Actions.BUY_VOUCHER, [i+1]]

        # nothing to purchase
        print(f"  % Ending shop, nothing to purchase {i}")
        return [Actions.END_SHOP]

    def select_booster_action(self, G):
        cache_state("select_booster_action", G)
        # choose the first card from the pack (simplified version of the script's
        # attempt-each-until-success behaviour) or skip if there is no pack.
        pack = G.get("pack", {})
        cards = pack.get("cards", [])
        try:
            if cards:
                print("   > Selecting first booster card")
                return [Actions.SELECT_BOOSTER_CARD, [1]]
            else:
                print("   > No cards to select")
        except:
            print("   > Some exception occured")
        finally:
            print("   > Skipping booster pack")
            return [Actions.SKIP_BOOSTER_PACK]

    # the remaining methods are essentially identical to flush_bot's defaults
    def sell_jokers(self, G):
        cache_state("sell_jokers", G)
        return [Actions.SELL_JOKER, []]

    def rearrange_jokers(self, G):
        cache_state("rearrange_jokers", G)
        return [Actions.REARRANGE_JOKERS, []]

    def use_or_sell_consumables(self, G):
        cache_state("use_or_sell_consumables", G)
        return [Actions.USE_CONSUMABLE, []]

    def rearrange_consumables(self, G):
        cache_state("rearrange_consumables", G)
        return [Actions.REARRANGE_CONSUMABLES, []]

    def rearrange_hand(self, G):
        cache_state("rearrange_hand", G)
        return [Actions.REARRANGE_HAND, []]


if __name__ == "__main__":
    print("=================\n    KHANG BOT\n=================")
    bot = KhangBot(deck="Blue Deck", stake=1, seed=None, challenge=None, bot_port=12348)
    try:
        print("Starting instance (10s warm‑up)")
        bot.start_balatro_instance()
        import time

        time.sleep(10)
        bot.run()
    finally:
        bot.stop_balatro_instance()
