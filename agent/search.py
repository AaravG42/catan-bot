"""Main-game move choice: exact one-ply expectation, tactical bonuses, then a short same-turn
beam search over the deterministic continuations, all under a per-decision CPU budget.

THE CLOCK. The harness gives each game a CPU allowance that grows per decision (0.3 s base,
12.5 ms per decision, 1.5 s cap) and plays random moves once it is spent. One decision therefore
gets at most HARD_CAP_S of CPU, keeps RESERVE_S of the remaining allowance untouched for engine
and evaluator calls that cannot be interrupted, and falls back to `emergency` when less than
EMERGENCY_S is left. Deadlines are checked between node expansions, never inside one.

SCORING. Every legal root action is scored by its expected leaf value over the exact chance
outcomes (dice, dev draw, steal). Three public-information bonuses are added at the root only: a
monopoly that takes 3+ cards, a dev buy that could complete Largest Army, and a knight is scored
by the best robber move that follows it. Then the best few deterministic continuations (builds,
trades, dev plays) are deepened with a beam search within the same turn; the root score becomes
max(one-ply score, deepened leaf + root bonus). A move that wins outright is played at once.

Search values are selective lower bounds, not minimax values, so the transposition cache only
stores nodes that were fully explored within budget.
"""
import time

from catanatron.models.enums import ActionType as AT

from evaluator import DEVS, MONOPOLY_BONUS, TACTICAL_ARMY, blocked_need
from outcomes import outcomes

#: Same-turn search: depth, beam width, CPU cap (s) and node ceiling ('army_exact' preset).
DEPTH, BEAM, CAP_S, MAX_NODES = 3, 3, .006, 80
#: Clock policy, in seconds of CPU.
HARD_CAP_S = .060       # most one decision may take
RESERVE_S = .015        # allowance never touched by a decision
EMERGENCY_S = .030      # below this much allowance, answer with `emergency`
SEARCH_RESERVE_S = .10  # one-ply fallback remains complete; search gets 4% of what exceeds this
SEARCH_FRACTION = .04
#: Value of a position in which we have already won; beats every finite score.
WIN = 1e18

BUILDS = {AT.BUILD_CITY, AT.BUILD_SETTLEMENT, AT.BUILD_ROAD}
CONTINUE = BUILDS | {AT.MARITIME_TRADE, AT.PLAY_YEAR_OF_PLENTY,
                     AT.PLAY_MONOPOLY, AT.PLAY_ROAD_BUILDING}


def state_key(game):
    """Within-decision transposition key: hands, board and forced phase.

    The map and the unseen deck are constant within our deterministic continuation tree. The
    ordered road history matters to engine tie breaking, so its order is preserved.
    """
    s = game.state
    return (tuple(sorted(s.player_state.items())), tuple(s.board.buildings.items()),
            tuple(s.board.roads.items()), s.board.robber_coordinate,
            s.current_player_index, s.current_turn_index, s.current_prompt,
            s.is_road_building, s.free_roads_available, s.is_moving_knight,
            tuple(s.resource_freqdeck))


def emergency(obs, actions, rng, deadline):
    """Legal, cheap fallback: a build that wins now if one exists, else by action priority."""
    priority = {AT.BUILD_CITY: 0, AT.BUILD_SETTLEMENT: 1, AT.END_TURN: 2,
                AT.ROLL: 3, AT.PLAY_KNIGHT_CARD: 4}
    index = min(range(len(actions)), key=lambda i: priority.get(actions[i].action_type, 5))
    # At a genuinely tiny bank avoid even world reconstruction.
    if deadline-time.process_time() < .003:
        return index
    world = obs.sample_world(rng)
    key = 'P%d_ACTUAL_VICTORY_POINTS' % world.state.color_to_index[obs.me]
    for i, action in enumerate(actions):
        if time.process_time() >= deadline:
            break
        if action.action_type not in BUILDS | {AT.PLAY_KNIGHT_CARD}:
            continue
        child = world.copy()
        child.execute(action, validate_action=False)
        if child.state.player_state[key] >= world.vps_to_win:
            return i
    return index


def search(obs, actions, evaluator, rng, started, left, hard_deadline, fallback):
    """Index of the chosen action for a main-game decision (len(actions) > 1).

    `started`/`left` are the decision's CPU start and remaining allowance; `hard_deadline` is
    the absolute CPU time by which an answer is required; `fallback` is a zero-argument function
    producing an emergency answer when nothing has been scored yet.
    """
    world = obs.sample_world(rng)
    me = obs.me
    key = 'P%d_' % world.state.color_to_index[me]
    initial_block = blocked_need(world, me)
    ps = world.state.player_state
    scores, successors, statics = [], [], []

    def completed_best():
        return max(range(len(scores)), key=lambda i: scores[i]) if scores else fallback()

    class RootDeadline(Exception):
        pass

    def forced_knight(game):
        """A knight is worth its best forced robber continuation, steals integrated exactly."""
        values = []
        for action in game.playable_actions:
            if action.action_type != AT.MOVE_ROBBER:
                continue
            if time.process_time() >= hard_deadline:
                raise RootDeadline
            value = 0.
            for child, probability in outcomes(game, action, me):
                if time.process_time() >= hard_deadline:
                    raise RootDeadline
                value += probability*evaluator.robber_value(child, me, initial_block)
            values.append(value)
        return max(values) if values else evaluator.value(game, me)

    # ---- one ply: exact expectation over chance, plus root-only tactical bonuses ----
    for action in actions:
        if time.process_time() >= hard_deadline:
            return completed_best()
        typ = action.action_type
        outs = outcomes(world, action, me)
        score = 0.
        static = None
        for game, probability in outs:
            if time.process_time() >= hard_deadline:
                return completed_best()
            if game.state.player_state[key+'ACTUAL_VICTORY_POINTS'] >= world.vps_to_win:
                if typ in BUILDS or typ == AT.PLAY_KNIGHT_CARD:
                    return actions.index(action)
                value = WIN
            elif typ == AT.MOVE_ROBBER:
                value = evaluator.robber_value(game, me, initial_block)
            elif typ == AT.PLAY_KNIGHT_CARD:
                try:
                    value = forced_knight(game)
                except RootDeadline:
                    return completed_best()
            else:
                value = evaluator.value(game, me)
            score += probability*value
            static = value
        if typ == AT.PLAY_MONOPOLY and obs.opp_resources[action.value] >= 3:
            # Reward a real swing only; one-card plays can still win on the leaf value.
            score += MONOPOLY_BONUS*(obs.opp_resources[action.value]-2)
        if typ == AT.BUY_DEVELOPMENT_CARD:
            # Chance that this draw is a knight that brings Largest Army within reach.
            other = 'P%d_' % (1-world.state.color_to_index[me])
            target = max(3, ps[other+'PLAYED_KNIGHT']+1)
            progress = ps[key+'PLAYED_KNIGHT']+ps[key+'KNIGHT_IN_HAND']
            if not ps[key+'HAS_ARMY'] and 0 < target-progress <= 2:
                unseen = list(world.state.development_listdeck)
                for d in DEVS:
                    unseen += [d]*ps[other+d+'_IN_HAND']
                if unseen:
                    score += TACTICAL_ARMY*(unseen.count('KNIGHT')/len(unseen))*(3-target+progress)
        scores.append(score)
        successors.append(outs[0][0] if typ in CONTINUE and len(outs) == 1 else None)
        statics.append(static)

    # ---- same-turn beam search over the deterministic continuations ----
    # The complete one-ply answer is already in hand; search gets a modest fraction of the
    # remaining allowance, with both a CPU and a node ceiling.
    now = time.process_time()
    remaining = max(0., left-(now-started))
    budget = min(CAP_S, max(0., (remaining-SEARCH_RESERVE_S)*SEARCH_FRACTION),
                 max(0., hard_deadline-now))
    deadline = now+budget
    expanded = 0
    cache = {}
    root_order = sorted(range(len(actions)), key=lambda i: scores[i], reverse=True)
    selected = [i for i in root_order if successors[i] is not None][:BEAM]
    # Keep one trade, road and year-of-plenty outside the usual beam: their immediate value can
    # dip before the point-producing continuation becomes visible.
    for typ in (AT.MARITIME_TRADE, AT.BUILD_ROAD, AT.PLAY_YEAR_OF_PLENTY):
        extra = next((i for i in root_order if actions[i].action_type == typ), None)
        if extra is not None and extra not in selected:
            selected.append(extra)
    bonuses = [scores[i]-statics[i] if successors[i] is not None else 0.
               for i in range(len(actions))]

    def deepen(game, depth, stand):
        nonlocal expanded
        if (depth <= 0 or expanded >= MAX_NODES or time.process_time() >= deadline
                or game.state.current_color() != me):
            return stand
        state = (state_key(game), depth)
        if state in cache:
            return cache[state]
        rows = []
        # Direct builds first (the Catan analogue of tactical move ordering). At the horizon
        # only builds are tried, as a quiescence step.
        choices = [a for a in game.playable_actions
                   if a.action_type in (BUILDS if depth == 1 else CONTINUE)]
        choices.sort(key=lambda a: 0 if a.action_type in BUILDS else 1)
        best = stand
        complete = True
        for action in choices:
            if expanded >= MAX_NODES or time.process_time() >= deadline:
                complete = False
                break
            child = game.copy()
            child.execute(action, validate_action=False)
            expanded += 1
            if child.state.player_state[key+'ACTUAL_VICTORY_POINTS'] >= world.vps_to_win:
                return WIN
            value = evaluator.value(child, me)
            best = max(best, value)
            rows.append((value, child))
        if depth > 1:
            rows.sort(key=lambda row: row[0], reverse=True)
            for value, child in rows[:BEAM]:
                if expanded >= MAX_NODES or time.process_time() >= deadline:
                    complete = False
                    break
                best = max(best, deepen(child, depth-1, value))
        # Memoise only nodes finished within budget: a cut-off value is not reusable.
        if complete and expanded < MAX_NODES and time.process_time() < deadline:
            cache[state] = best
        return best

    # Iterative deepening: every selected root move competes at short depth before the rest of
    # the allowance goes to a longer continuation.
    for depth in range(1, DEPTH):
        selected.sort(key=lambda i: scores[i], reverse=True)
        for i in selected:
            if expanded >= MAX_NODES or time.process_time() >= deadline:
                break
            leaf = deepen(successors[i], depth, statics[i])
            if leaf >= WIN:
                return i
            # Keep the root-only bonus without accumulating it along the path.
            scores[i] = max(scores[i], leaf + bonuses[i])
    return max(range(len(actions)), key=lambda i: scores[i])
