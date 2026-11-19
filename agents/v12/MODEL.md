# v12: v11 + reply-aware opening

v11 (123-feature value net, v5 search) with initial settlements chosen by greedy setup rollouts:
the top-5 spots by 1-ply value are each completed to the end of the setup phase with both sides
choosing 1-ply by the value net (settlement candidates pruned to the 10 best by pips), and the spot
whose completed setup scores best from my view is played. Opening call ~80 ms (laptop).

Arena (eval seeds 990000, 2000 games, unclocked): 0.5405 +-0.022 vs v11, 0.7975 +-0.018 vs F, 0 over.
