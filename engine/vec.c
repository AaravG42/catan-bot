/* Batched, auto-resetting environment over an array of States (PufferLib-style buffers). */
#include "catan.h"
#include <string.h>

enum { F_AUTOPLAY = 1, F_OBS = 2, F_AUTORESET = 4 };

static void emit(State *s, int i, int nlegal, const int16_t *legal, float *obs, uint8_t *mask,
                 int8_t *cur, int flags) {
    if (mask) {
        uint8_t *m = mask + (size_t)i * N_ACTIONS;
        memset(m, 0, N_ACTIONS);
        for (int k = 0; k < nlegal; k++) m[legal[k]] = 1;
    }
    if (obs && (flags & F_OBS)) cat_encode(s, obs + (size_t)i * OBS_DIM);
    if (cur) cur[i] = s->cur;
}

static void fresh(State *s, uint64_t seed, int flags, int16_t *legal, int *nlegal) {
    cat_new_game(s, seed);
    if (flags & F_AUTOPLAY) cat_autoplay(s, legal, nlegal);
    else *nlegal = cat_legal(s, legal);
}

void vec_reset(State *st, int n, uint64_t seed, float *obs, uint8_t *mask, int8_t *cur, int flags) {
    int16_t legal[MAX_LEGAL];
    for (int i = 0; i < n; i++) {
        int nl;
        fresh(&st[i], seed * 0x100000001B3ULL + (uint64_t)i, flags, legal, &nl);
        emit(&st[i], i, nl, legal, obs, mask, cur, flags);
    }
}

/* actions[i] is an index into the 332 catanatron actions and must be legal.
 * reward[i] is from the view of the seat that chose actions[i]: +1 win, -1 loss, 0 otherwise.
 * winner[i]: seat that won this step, -1 none (also -1 when truncated at the turn limit).
 * With F_AUTORESET a finished env starts a new game at once; obs/mask/cur then describe it. */
long vec_step(State *st, int n, const int32_t *actions, float *obs, uint8_t *mask, float *reward,
              uint8_t *terminal, uint8_t *truncated, int8_t *cur, int8_t *winner, int flags) {
    int16_t legal[MAX_LEGAL];
    long steps = 0;
    for (int i = 0; i < n; i++) {
        State *s = &st[i];
        int actor = s->cur, nl;
        cat_step(s, actions[i], -1);
        steps++;
        if (flags & F_AUTOPLAY) steps += cat_autoplay(s, legal, &nl);
        else nl = cat_legal(s, legal);
        int done = cat_done(s);
        int w = s->winner;
        if (reward) reward[i] = done && w >= 0 ? (w == actor ? 1.0f : -1.0f) : 0.0f;
        if (terminal) terminal[i] = done && w >= 0;
        if (truncated) truncated[i] = done && w < 0;
        if (winner) winner[i] = done ? w : -1;
        if (done && (flags & F_AUTORESET)) fresh(s, s->rng, flags, legal, &nl);
        emit(s, i, nl, legal, obs, mask, cur, flags);
    }
    return steps;
}

/* Uniform-random self-play over n envs for `games` total games; returns steps (benchmark). */
long vec_random_games(State *st, int n, long games, uint64_t seed, int autoplay) {
    long steps = 0;
    for (int i = 0; i < n; i++) cat_new_game(&st[i], seed + (uint64_t)i);
    long done_games = 0;
    int16_t legal[MAX_LEGAL];
    uint64_t r = seed | 1;
    while (done_games < games) {
        for (int i = 0; i < n && done_games < games; i++) {
            State *s = &st[i];
            int nl = cat_legal(s, legal);
            r ^= r >> 12; r ^= r << 25; r ^= r >> 27;
            int a = legal[(int)(((r * 0x2545F4914F6CDD1DULL) >> 32) * (uint64_t)nl >> 32)];
            cat_step(s, a, -1);
            steps++;
            if (autoplay) {
                int k;
                steps += cat_autoplay(s, legal, &k);
            }
            if (cat_done(s)) { done_games++; cat_new_game(s, s->rng); }
        }
    }
    return steps;
}

/* Encode n states (e.g. search leaves) into obs. */
void vec_encode(const State *st, int n, float *obs) {
    for (int i = 0; i < n; i++) cat_encode(&st[i], obs + (size_t)i * OBS_DIM);
}
