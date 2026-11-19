"""Exact one-step chance outcomes: every random result of an action with its probability.

Root actions are scored by expectation over this list instead of by a single sampled
continuation, so a bank trade or a dev-card buy is never chosen just because one sample of it
happened to be lucky. The pinned catanatron forces a chance result through
ActionRecord.result, which is what makes this enumeration possible.

Dev draws are marginalised over the public unseen pile (deck plus the opponent's unplayed
cards). Each branch rebuilds a consistent private allocation conditional on the drawn card. That
allocation is arbitrary but harmless: the evaluator only reads public features of the opponent.
"""
import math

from catanatron.models.enums import ActionRecord, ActionType
from catanatron.models.map import number_probability

RES = ('WOOD', 'BRICK', 'SHEEP', 'WHEAT', 'ORE')
DEVS = ('KNIGHT', 'VICTORY_POINT', 'ROAD_BUILDING', 'YEAR_OF_PLENTY', 'MONOPOLY')


def outcomes(game, action, me):
    """[(game_after, probability)] for `action` played on `game`, seen from `me`."""
    st, typ = game.state, action.action_type
    if typ == ActionType.ROLL:
        result = []
        for roll in range(2, 13):
            dice = (roll//2, math.ceil(roll/2))
            g = game.copy()
            g.execute(action, validate_action=False,
                      action_record=ActionRecord(action, dice))
            result.append((g, number_probability(roll)))
        return result
    if typ == ActionType.BUY_DEVELOPMENT_CARD:
        j = 1-st.color_to_index[me]
        key = 'P%d_' % j
        nopp = sum(st.player_state[key+d+'_IN_HAND'] for d in DEVS)
        pile = list(st.development_listdeck)
        for d in DEVS:
            pile.extend([d]*st.player_state[key+d+'_IN_HAND'])
        result = []
        for d in DEVS:
            count = pile.count(d)
            if not count:
                continue
            g = game.copy()
            # Condition the unknown allocation on drawing d. Sorting deliberately removes any
            # information carried by the sampled private ordering.
            remaining = sorted(pile)
            remaining.remove(d)
            opponent, deck = remaining[:nopp], remaining[nopp:]
            g.state.development_listdeck = deck + [d]
            for other in DEVS:
                n = opponent.count(other)
                g.state.player_state[key+other+'_IN_HAND'] = n
                g.state.player_state[key+other+'_OWNED_AT_START'] = n
            g.state.player_state[key+'ACTUAL_VICTORY_POINTS'] = (
                g.state.player_state[key+'VICTORY_POINTS'] + opponent.count('VICTORY_POINT'))
            g.execute(action, validate_action=False,
                      action_record=ActionRecord(action, d))
            result.append((g, count/len(pile)))
        return result
    if typ == ActionType.MOVE_ROBBER and action.value[1] is not None:
        # Stealing: one branch per resource the victim holds, weighted by its share of the hand.
        k = st.color_to_index[action.value[1]]
        hand = [st.player_state['P%d_%s_IN_HAND' % (k,r)] for r in RES]
        n = sum(hand)
        if n:
            result = []
            for r,count in zip(RES,hand):
                if count:
                    g = game.copy()
                    g.execute(action, validate_action=False,
                              action_record=ActionRecord(action,r))
                    result.append((g,count/n))
            return result
    g = game.copy()
    g.execute(action,validate_action=False)
    return [(g,1.)]
