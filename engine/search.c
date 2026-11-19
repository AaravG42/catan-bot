/* Search helpers on top of the engine. Nothing here changes a rule: it only copies states and
 * calls cat_step with forced outcomes, so engine parity is unaffected.
 *
 * vec_expand: every legal action of every parent, with its chance outcomes enumerated exactly:
 *   ROLL           11 dice sums, P = (6-|7-s|)/36
 *   BUY_DEV        each card type in the pile the buyer cannot see (deck + the other seat's unplayed
 *                  cards), P = count/total. The deck order in the state is never used, so search does
 *                  not peek at hidden information.
 *   MOVE_ROBBER    with a victim holding cards: each resource in the victim's hand, P = count/total
 *   anything else  one child, P = 1
 */
#include "catan.h"
#include <string.h>

static int legal_of(const State *s, int16_t *out) { return cat_legal(s, out); }

/* Returns the number of children written (stops before max_out). For child j:
 *   out[j]     the state after the action and outcome
 *   prob[j]    probability of that outcome given the action
 *   act[j]     the catanatron action index
 *   parent[j]  index of the parent in st[]
 * nact[i] receives the number of legal actions of parent i (0 if it was done). */
long vec_expand(const State *st, int n, State *out, float *prob, int16_t *act, int32_t *parent,
                int16_t *nact, long max_out) {
    int16_t legal[MAX_LEGAL];
    long m = 0;
    for (int i = 0; i < n; i++) {
        const State *s = &st[i];
        if (cat_done(s)) { nact[i] = 0; continue; }
        int nl = legal_of(s, legal);
        nact[i] = (int16_t)nl;
        int p = s->cur, q = 1 - p;
        for (int k = 0; k < nl; k++) {
            int a = legal[k];
            const int8_t *row = ACTION_TABLE[a];
            if (row[0] == AT_ROLL) {
                if (m + 11 > max_out) return m;
                for (int sum = 2; sum <= 12; sum++) {
                    out[m] = *s;
                    cat_step(&out[m], a, sum);
                    prob[m] = (float)(6 - (sum > 7 ? sum - 7 : 7 - sum)) / 36.0f;
                    act[m] = (int16_t)a; parent[m] = i; m++;
                }
            } else if (row[0] == AT_BUY_DEVELOPMENT_CARD) {
                int cnt[5] = {0}, tot = 0;
                for (int d = 0; d < s->deck_n; d++) cnt[s->deck[d]]++;
                for (int d = 0; d < 5; d++) cnt[d] += s->dev[q][d];
                for (int d = 0; d < 5; d++) tot += cnt[d];
                if (m + 5 > max_out) return m;
                for (int d = 0; d < 5; d++) {
                    if (!cnt[d]) continue;
                    out[m] = *s;
                    out[m].deck[out[m].deck_n - 1] = (int8_t)d;   /* the card drawn is d */
                    cat_step(&out[m], a, -1);
                    prob[m] = (float)cnt[d] / (float)tot;
                    act[m] = (int16_t)a; parent[m] = i; m++;
                }
            } else if (row[0] == AT_MOVE_ROBBER && row[2] != 0) {
                int v = (s->color[0] == row[2] - 1) ? 0 : 1;
                int tot = 0;
                for (int r = 0; r < 5; r++) tot += s->res[v][r];
                if (m + 5 > max_out) return m;
                if (tot == 0) {
                    out[m] = *s; cat_step(&out[m], a, -1);
                    prob[m] = 1.0f; act[m] = (int16_t)a; parent[m] = i; m++;
                    continue;
                }
                for (int r = 0; r < 5; r++) {
                    if (!s->res[v][r]) continue;
                    out[m] = *s;
                    cat_step(&out[m], a, r);
                    prob[m] = (float)s->res[v][r] / (float)tot;
                    act[m] = (int16_t)a; parent[m] = i; m++;
                }
            } else {
                if (m + 1 > max_out) return m;
                out[m] = *s;
                cat_step(&out[m], a, -1);
                prob[m] = 1.0f; act[m] = (int16_t)a; parent[m] = i; m++;
            }
        }
    }
    return m;
}

/* Encode st[i] from seat seats[i]'s view. */
void vec_encode_seat(const State *st, int n, const int8_t *seats, float *obs) {
    for (int i = 0; i < n; i++) cat_encode_seat(&st[i], seats[i], obs + (size_t)i * OBS_DIM);
}

/* Per-state summary for search bookkeeping: cur seat, prompt, done, winner. */
void vec_info(const State *st, int n, int8_t *cur, int8_t *prompt, int8_t *done, int8_t *winner) {
    for (int i = 0; i < n; i++) {
        cur[i] = st[i].cur; prompt[i] = st[i].prompt;
        done[i] = (int8_t)cat_done(&st[i]); winner[i] = st[i].winner;
    }
}

/* Re-deal the information `me` cannot see: the opponent's unplayed dev cards and the deck order
 * are shuffled together and redealt (the counts are public). Same idea as catan_kit.masked_copy. */
void vec_redeal(State *st, int n, const int8_t *me) {
    for (int i = 0; i < n; i++) {
        State *s = &st[i];
        int q = 1 - me[i];
        int8_t pile[40];
        int np = 0, nq = 0;
        for (int d = 0; d < s->deck_n; d++) pile[np++] = s->deck[d];
        for (int d = 0; d < 5; d++) for (int c = 0; c < s->dev[q][d]; c++) { pile[np++] = (int8_t)d; nq++; }
        for (int k = np - 1; k > 0; k--) {
            s->rng ^= s->rng >> 12; s->rng ^= s->rng << 25; s->rng ^= s->rng >> 27;
            int j = (int)(((s->rng * 0x2545F4914F6CDD1DULL) >> 33) % (uint64_t)(k + 1));
            int8_t t = pile[k]; pile[k] = pile[j]; pile[j] = t;
        }
        int vp_before = s->dev[q][D_VP];
        for (int d = 0; d < 5; d++) s->dev[q][d] = 0;
        for (int c = 0; c < nq; c++) s->dev[q][pile[c]]++;
        for (int d = 0; d < 4; d++) s->owned[q][d] = s->dev[q][d] > 0;   /* as masked_copy */
        s->avp[q] += s->dev[q][D_VP] - vp_before;
        for (int d = 0; d < s->deck_n; d++) s->deck[d] = pile[nq + d];
    }
}
