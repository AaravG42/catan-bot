/* v3's 76 position features (agents/common/efeats.py) and a small ReLU MLP value, in C.
 *
 * feat_init      topology from Python (node -> land tiles, edge endpoints), once per process
 * vec_efeats     (n, 639) observations + my_turn -> (n, 76) features, same order as efeats()
 * vec_efeats_st  states + seats -> features (encodes from the seat's view; my_turn = cur == seat)
 * mlp_set        load a value net (mu/sd folded into layer 0 by the caller)
 * vec_value76    states[idx[j]] from seats[j]'s view -> sigmoid(MLP(features)), fused
 *
 * Mirrors efeats.py line by line; see the parity test in engine/cfeats.py. */
#include "catan.h"
#include <math.h>
#include <string.h>

#define NT 19
#define NN 54
#define NE_MAX 80
#define NF 76

static int nt_n[NN], nt_t[NN][3];          /* node -> adjacent land tiles */
static int ne, eu[NE_MAX], ev[NE_MAX];
static int ninc_n[NN], ninc[NN][3];        /* node -> incident edges */
static int nadj_n[NN], nadj[NN][3];        /* node -> neighbour nodes */

void feat_init(const int8_t *node_tiles /* NN x 3, -1 pad */, const int16_t *u, const int16_t *v,
               int n_edges) {
    for (int i = 0; i < NN; i++) {
        nt_n[i] = 0;
        for (int k = 0; k < 3; k++)
            if (node_tiles[i * 3 + k] >= 0) nt_t[i][nt_n[i]++] = node_tiles[i * 3 + k];
        ninc_n[i] = nadj_n[i] = 0;
    }
    ne = n_edges;
    for (int e = 0; e < ne; e++) {
        eu[e] = u[e]; ev[e] = v[e];
        ninc[eu[e]][ninc_n[eu[e]]++] = e;
        ninc[ev[e]][ninc_n[ev[e]]++] = e;
        nadj[eu[e]][nadj_n[eu[e]]++] = ev[e];
        nadj[ev[e]][nadj_n[ev[e]]++] = eu[e];
    }
}

/* One player's 33 base columns. */
static float *block(float *o, const float prod_node[NN][5], const float *npv, const float *occ,
                    const float *settle, const float *city, const float *road /* stride 2 */,
                    const float *road_other, float vp, float roads_left, const float *ports,
                    const float *res, float knights, float army, float road_len, float has_road,
                    float devs_n) {
    float prod[5] = {0};
    float ns = 0, nc = 0;
    for (int i = 0; i < NN; i++) {
        float w = settle[i * 4] + 2 * city[i * 4];
        ns += settle[i * 4]; nc += city[i * 4];
        if (w != 0) for (int r = 0; r < 5; r++) prod[r] += w * prod_node[i][r];
    }
    float total = 0, distinct = 0, hand = 0;
    for (int r = 0; r < 5; r++) { total += prod[r]; distinct += prod[r] > 0; hand += res[r]; }
    uint8_t mine[NN], buildable[NN], cand[NN], one[NN];
    for (int i = 0; i < NN; i++) {
        float rs = 0;
        for (int k = 0; k < ninc_n[i]; k++) rs += road[ninc[i][k] * 2];
        mine[i] = (settle[i * 4] + city[i * 4]) > 0 || rs > 0;
        float nb = 0;
        for (int k = 0; k < nadj_n[i]; k++) nb += occ[nadj[i][k]];
        buildable[i] = occ[i] == 0 && nb == 0;
        cand[i] = buildable[i] && !mine[i];
        one[i] = 0;
    }
    for (int e = 0; e < ne; e++) {
        if (road[e * 2] + road_other[e * 2] != 0) continue;
        int a = eu[e], b = ev[e];
        if ((mine[a] && cand[b]) || (mine[b] && cand[a])) {
            if (cand[b]) one[b] = 1;
            if (cand[a]) one[a] = 1;
        }
    }
    float now_n = 0, best_now = 0, one_n = 0, best_one = 0;
    for (int i = 0; i < NN; i++) {
        float vn = (mine[i] && buildable[i]) ? npv[i] : 0, vo = one[i] ? npv[i] : 0;
        now_n += mine[i] && buildable[i]; one_n += one[i];
        if (vn > best_now) best_now = vn;
        if (vo > best_one) best_one = vo;
    }
    *o++ = vp; *o++ = ns; *o++ = nc; *o++ = roads_left;
    for (int r = 0; r < 5; r++) *o++ = prod[r];
    *o++ = total; *o++ = distinct;
    for (int k = 0; k < 6; k++) *o++ = ports[k];
    for (int r = 0; r < 5; r++) *o++ = res[r];
    *o++ = hand; *o++ = hand > 7;
    *o++ = knights; *o++ = army; *o++ = road_len; *o++ = has_road;
    *o++ = now_n; *o++ = best_now; *o++ = one_n; *o++ = best_one; *o++ = devs_n;
    return o;
}

static void efeats1(const float *X, float my_turn, float *o) {
    float pt[NT], vt[NT], prod_node[NN][5], npv[NN], occ[NN];
    for (int t = 0; t < NT; t++) {
        const float *x = X + t * 8;
        pt[t] = x[6] / 36.0f * (1 - x[7]);
        vt[t] = (x[0] + x[1] + x[2] + x[3] + x[4]) * x[6] / 36.0f;
    }
    for (int i = 0; i < NN; i++) {
        float s = 0;
        for (int r = 0; r < 5; r++) prod_node[i][r] = 0;
        for (int k = 0; k < nt_n[i]; k++) {
            int t = nt_t[i][k];
            for (int r = 0; r < 5; r++) prod_node[i][r] += X[t * 8 + r] * pt[t];
            s += vt[t];
        }
        npv[i] = s;
        const float *nd = X + 206 + i * 4;
        occ[i] = nd[0] + nd[1] + nd[2] + nd[3];
    }
    const float *nd = X + 206, *rd = X + 422, *me = X + 566, *op = X + 601, *g = X + 624;
    o = block(o, prod_node, npv, occ, nd + 0, nd + 1, rd + 0, rd + 1, me[14], me[16], me + 28,
              me + 0, me[10], me[20], me[21], me[19], me[5] + me[6] + me[7] + me[8] + me[9]);
    *o++ = me[5]; *o++ = me[9]; *o++ = me[8]; *o++ = me[6]; *o++ = me[7]; *o++ = me[15];
    o = block(o, prod_node, npv, occ, nd + 2, nd + 3, rd + 1, rd + 0, op[10], op[11], op + 17,
              op + 0, op[6], op[15], op[16], op[14], op[5]);
    *o++ = g[5]; *o++ = g[6] / 100.0f; *o++ = my_turn; *o++ = (g[7] + g[8]) > 0;
}

void vec_efeats(const float *X, int n, const float *my_turn, float *out) {
    for (int i = 0; i < n; i++) efeats1(X + (size_t)i * OBS_DIM, my_turn[i], out + (size_t)i * NF);
}

void vec_efeats_st(const State *st, int n, const int8_t *seats, float *out) {
    float obs[OBS_DIM];
    for (int i = 0; i < n; i++) {
        cat_encode_seat(&st[i], seats[i], obs);
        efeats1(obs, st[i].cur == seats[i], out + (size_t)i * NF);
    }
}

/* ---- value MLP ---- */
#define ML_MAX 6
#define W_MAX 1024
static int ml_n, ml_dim[ML_MAX + 1];
static float ml_W[1 << 20], ml_b[1 << 13];
static int ml_wo[ML_MAX], ml_bo[ML_MAX];

/* dims[0..nl]; W row-major (in, out) per layer concatenated; b concatenated. Returns 0 on success. */
int mlp_set(int nl, const int *dims, const float *W, const float *b) {
    if (nl > ML_MAX || dims[0] != NF) return -1;
    long wo = 0, bo = 0;
    for (int l = 0; l < nl; l++) {
        if (dims[l + 1] > W_MAX) return -2;
        ml_wo[l] = (int)wo; ml_bo[l] = (int)bo;
        wo += (long)dims[l] * dims[l + 1]; bo += dims[l + 1];
    }
    if (wo > (long)(sizeof ml_W / sizeof *ml_W) || bo > (long)(sizeof ml_b / sizeof *ml_b)) return -3;
    memcpy(ml_W, W, wo * sizeof(float));
    memcpy(ml_b, b, bo * sizeof(float));
    ml_n = nl;
    for (int l = 0; l <= nl; l++) ml_dim[l] = dims[l];
    return 0;
}

static void layer(const float *restrict in, int di, int dq, const float *restrict W,
                  const float *restrict b, float *restrict out, int relu) {
    int j0 = 0;
    for (; j0 + 16 <= dq; j0 += 16) {          /* 16 accumulators stay in registers */
        float acc[16];
        for (int k = 0; k < 16; k++) acc[k] = b[j0 + k];
        for (int i = 0; i < di; i++) {
            const float a = in[i];
            const float *restrict w = W + (size_t)i * dq + j0;
            for (int k = 0; k < 16; k++) acc[k] += a * w[k];
        }
        for (int k = 0; k < 16; k++) out[j0 + k] = relu && acc[k] < 0 ? 0 : acc[k];
    }
    for (int j = j0; j < dq; j++) {
        float acc = b[j];
        for (int i = 0; i < di; i++) acc += in[i] * W[(size_t)i * dq + j];
        out[j] = relu && acc < 0 ? 0 : acc;
    }
}

static float mlp1(const float *x) {
    float h[2][W_MAX];
    const float *in = x;
    for (int l = 0; l < ml_n; l++) {
        float *o = h[l & 1];
        layer(in, ml_dim[l], ml_dim[l + 1], ml_W + ml_wo[l], ml_b + ml_bo[l], o, l < ml_n - 1);
        in = o;
    }
    return 1.0f / (1.0f + expf(-in[0]));
}

void vec_value76(const State *st, int m, const int32_t *idx, const int8_t *seats, float *out) {
    float obs[OBS_DIM], f[NF];
    for (int j = 0; j < m; j++) {
        const State *s = &st[idx[j]];
        cat_encode_seat(s, seats[j], obs);
        efeats1(obs, s->cur == seats[j], f);
        out[j] = mlp1(f);
    }
}

void vec_mlp(const float *F, int n, float *out) {
    for (int i = 0; i < n; i++) out[i] = mlp1(F + (size_t)i * NF);
}
