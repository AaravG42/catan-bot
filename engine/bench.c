#define _POSIX_C_SOURCE 199309L
/* Native benchmark: random self-play steps/s and games/s on one core. */
#include "catan.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
long vec_random_games(State *st, int n, long games, uint64_t seed, int autoplay);
void vec_reset(State *st, int n, uint64_t seed, float *obs, uint8_t *mask, int8_t *cur, int flags);
long vec_step(State *st, int n, const int32_t *actions, float *obs, uint8_t *mask, float *reward,
              uint8_t *terminal, uint8_t *truncated, int8_t *cur, int8_t *winner, int flags);
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
int main(int argc, char **argv) {
    long games = argc > 1 ? atol(argv[1]) : 20000;
    int n = 64;
    State *st = calloc(n, sizeof(State));
    printf("sizeof(State) = %zu bytes\n", sizeof(State));
    double t0 = now();
    long steps = vec_random_games(st, n, games, 1, 0);
    double dt = now() - t0;
    printf("random games: %ld games, %ld steps, %.2f s -> %.0f games/s, %.2f M steps/s, %.0f steps/game\n",
           games, steps, dt, games / dt, steps / dt / 1e6, (double)steps / games);
    /* vec_step with uniform random legal actions, with and without obs */
    int N = 256;
    State *vs = calloc(N, sizeof(State));
    float *obs = malloc(sizeof(float) * N * OBS_DIM);
    uint8_t *mask = malloc(N * N_ACTIONS), *term = malloc(N), *trunc = malloc(N);
    float *rew = malloc(sizeof(float) * N);
    int8_t *cur = malloc(N), *win = malloc(N);
    int32_t *act = malloc(sizeof(int32_t) * N);
    for (int flags = 4 | 1; flags <= 7; flags += 2) {
        vec_reset(vs, N, 7, obs, mask, cur, flags);
        uint64_t r = 99;
        long tot = 0; int iters = 4000;
        double t1 = now();
        for (int it = 0; it < iters; it++) {
            for (int i = 0; i < N; i++) {
                const uint8_t *m = mask + (size_t)i * N_ACTIONS;
                int16_t idx[N_ACTIONS]; int k = 0;
                for (int a = 0; a < N_ACTIONS; a++) if (m[a]) idx[k++] = a;
                r ^= r >> 12; r ^= r << 25; r ^= r >> 27;
                act[i] = idx[(int)((((r * 0x2545F4914F6CDD1DULL) >> 32) * (uint64_t)k) >> 32)];
            }
            tot += vec_step(vs, N, act, obs, mask, rew, term, trunc, cur, win, flags);
        }
        double d = now() - t1;
        printf("vec_step (autoplay, %s obs): %.2f M decisions/s, %.2f M engine steps/s (incl. mask + random policy)\n",
               (flags & 2) ? "with" : "no", (double)iters * N / d / 1e6, tot / d / 1e6);
    }
    return 0;
}
