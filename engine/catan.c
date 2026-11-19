/* Fast 2-player Catan engine. Every rule below mirrors a specific piece of catanatron
 * (apply_action.py, state_functions.py, models/actions.py, models/board.py); comments name it. */
#include "catan.h"
#include <string.h>

/* ------------------------------------------------------------------ tables */
static uint64_t NODE_NBR_MASK[N_NODES];
static u128 NODE_EDGE_MASK[N_NODES];
static uint64_t TILE_MASK[N_TILES];
static uint64_t PORT_MASK[N_PORTS];
static int16_t IDX_CITY[N_NODES], IDX_SETTLE[N_NODES], IDX_ROAD[N_EDGES];
static int16_t IDX_DISCARD[5], IDX_MONO[5], IDX_YOP2[5][5], IDX_YOP1[5];
static int16_t IDX_ROBBER[N_TILES][3], IDX_TRADE[5][5][5];
static int16_t IDX_ROLL, IDX_END, IDX_BUY, IDX_KNIGHT, IDX_RB;
static int tables_ready = 0;

static const int8_t COST_ROAD[5] = {1, 1, 0, 0, 0};
static const int8_t COST_SETTLE[5] = {1, 1, 1, 1, 0};
static const int8_t COST_CITY[5] = {0, 0, 0, 2, 3};
static const int8_t COST_DEV[5] = {0, 0, 1, 1, 1};

void cat_init_tables(void) {
    if (tables_ready) return;
    for (int n = 0; n < N_NODES; n++) {
        NODE_NBR_MASK[n] = 0;
        NODE_EDGE_MASK[n] = 0;
        for (int k = 0; k < 3; k++) {
            if (NODE_NBR[n][k] < 0) break;
            NODE_NBR_MASK[n] |= 1ULL << NODE_NBR[n][k];
            NODE_EDGE_MASK[n] |= (u128)1 << NODE_NBR_EDGE[n][k];
        }
    }
    for (int t = 0; t < N_TILES; t++) {
        TILE_MASK[t] = 0;
        for (int k = 0; k < 6; k++) TILE_MASK[t] |= 1ULL << TILE_NODES[t][k];
    }
    for (int p = 0; p < N_PORTS; p++) PORT_MASK[p] = (1ULL << PORT_NODES[p][0]) | (1ULL << PORT_NODES[p][1]);
    for (int i = 0; i < N_ACTIONS; i++) {
        const int8_t *r = ACTION_TABLE[i];
        switch (r[0]) {
        case AT_BUILD_CITY: IDX_CITY[r[1]] = i; break;
        case AT_BUILD_SETTLEMENT: IDX_SETTLE[r[1]] = i; break;
        case AT_BUILD_ROAD: IDX_ROAD[r[1]] = i; break;
        case AT_DISCARD_RESOURCE: IDX_DISCARD[r[1]] = i; break;
        case AT_PLAY_MONOPOLY: IDX_MONO[r[1]] = i; break;
        case AT_PLAY_YEAR_OF_PLENTY:
            if (r[3] == 2) IDX_YOP2[r[1]][r[2]] = i; else IDX_YOP1[r[1]] = i;
            break;
        case AT_MOVE_ROBBER: IDX_ROBBER[r[1]][r[2]] = i; break;
        case AT_MARITIME_TRADE: IDX_TRADE[r[1]][r[2]][r[3]] = i; break;
        case AT_ROLL: IDX_ROLL = i; break;
        case AT_END_TURN: IDX_END = i; break;
        case AT_BUY_DEVELOPMENT_CARD: IDX_BUY = i; break;
        case AT_PLAY_KNIGHT_CARD: IDX_KNIGHT = i; break;
        case AT_PLAY_ROAD_BUILDING: IDX_RB = i; break;
        }
    }
    tables_ready = 1;
}

int cat_state_size(void) { return (int)sizeof(State); }

/* ------------------------------------------------------------------ rng */
static inline uint64_t rng_next(uint64_t *x) {
    /* xorshift64* */
    uint64_t v = *x;
    v ^= v >> 12; v ^= v << 25; v ^= v >> 27;
    *x = v;
    return v * 0x2545F4914F6CDD1DULL;
}
static inline uint32_t rng_below(uint64_t *x, uint32_t n) {
    return (uint32_t)(((rng_next(x) >> 32) * (uint64_t)n) >> 32);
}
static uint64_t splitmix(uint64_t z) {
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    return z ? z : 0x1234567ULL;
}

/* ------------------------------------------------------------------ helpers */
static inline int popc(uint64_t x) { return __builtin_popcountll(x); }
static inline int ctz(uint64_t x) { return __builtin_ctzll(x); }
static inline int hand_total(const State *s, int p) {
    const int8_t *r = s->res[p];
    return r[0] + r[1] + r[2] + r[3] + r[4];
}
static inline uint64_t owned_nodes(const State *s, int p) { return s->settle[p] | s->city[p]; }
static inline int is_enemy(const State *s, int n, int p) { return (owned_nodes(s, 1 - p) >> n) & 1; }
static inline int can_play_dev(const State *s, int p, int d) {
    /* state_functions.player_can_play_dev */
    return !s->has_played[p] && s->dev[p][d] >= 1 && s->owned[p][d];
}
static inline int has_cost(const State *s, int p, const int8_t *c) {
    const int8_t *r = s->res[p];
    return r[0] >= c[0] && r[1] >= c[1] && r[2] >= c[2] && r[3] >= c[3] && r[4] >= c[4];
}
static inline void pay(State *s, int p, const int8_t *c) {
    for (int i = 0; i < 5; i++) { s->res[p][i] -= c[i]; s->bank[i] += c[i]; }
}
static inline int first_comp(const State *s, int p, int n) {
    for (int i = 0; i < s->ncomp[p]; i++)
        if ((s->comp[p][i] >> n) & 1) return i;
    return -1;
}
static inline uint64_t comp_union(const State *s, int p) {
    uint64_t u = 0;
    for (int i = 0; i < s->ncomp[p]; i++) u |= s->comp[p][i];
    return u;
}
/* board.buildable_edges(color): edges touching any component node, with no road */
static u128 buildable_edges(const State *s, int p) {
    uint64_t u = comp_union(s, p);
    u128 e = 0;
    while (u) { e |= NODE_EDGE_MASK[ctz(u)]; u &= u - 1; }
    return e & ~(s->road[0] | s->road[1]);
}
static int port_flags(const State *s, int p) {
    /* bits 0..4: 2:1 port of that resource; bit 5: 3:1 */
    uint64_t own = owned_nodes(s, p);
    int f = 0;
    for (int k = 0; k < N_PORTS; k++)
        if (own & PORT_MASK[k]) f |= s->port_res[k] < 0 ? 32 : (1 << s->port_res[k]);
    return f;
}

/* ------------------------------------------------------------------ longest road */
/* board.longest_acyclic_path: longest trail (distinct edges) over own roads, never entering an
 * enemy node, starting from any node of the component (a start node may itself be an enemy node) */
static int trail(const State *s, int p, int node, u128 used, uint64_t enemy) {
    int best = 0;
    u128 mine = s->road[p];
    for (int k = 0; k < 3; k++) {
        int nb = NODE_NBR[node][k];
        if (nb < 0) break;
        int e = NODE_NBR_EDGE[node][k];
        u128 eb = (u128)1 << e;
        if (!(mine & eb) || (used & eb) || ((enemy >> nb) & 1)) continue;
        int v = 1 + trail(s, p, nb, used | eb, enemy);
        if (v > best) best = v;
    }
    return best;
}
static int longest_path(const State *s, int p, uint64_t component) {
    uint64_t enemy = owned_nodes(s, 1 - p);
    int best = 0;
    u128 mine = s->road[p];
    while (component) {
        int n = ctz(component);
        component &= component - 1;
        if (!(NODE_EDGE_MASK[n] & mine)) continue;
        int v = trail(s, p, n, 0, enemy);
        if (v > best) best = v;
    }
    return best;
}
/* board.dfs_walk: nodes reachable over own roads; enemy nodes are visited but not expanded */
static uint64_t dfs_walk(const State *s, int start, int p) {
    uint64_t visited = 0, enemy = owned_nodes(s, 1 - p);
    int stack[64], sp = 0;
    stack[sp++] = start;
    u128 mine = s->road[p];
    while (sp) {
        int n = stack[--sp];
        visited |= 1ULL << n;
        if ((enemy >> n) & 1) continue;
        for (int k = 0; k < 3; k++) {
            int nb = NODE_NBR[n][k];
            if (nb < 0) break;
            if ((visited >> nb) & 1) continue;
            if (mine & ((u128)1 << NODE_NBR_EDGE[n][k])) stack[sp++] = nb;
        }
    }
    return visited;
}
static void comp_remove(State *s, int p, int i) {
    for (int j = i; j < s->ncomp[p] - 1; j++) s->comp[p][j] = s->comp[p][j + 1];
    s->ncomp[p]--;
}
static void comp_append(State *s, int p, uint64_t m) {
    if (s->ncomp[p] < MAXC) s->comp[p][s->ncomp[p]++] = m;
}
/* state_functions.maintain_longest_road */
static void maintain_longest_road(State *s, int prev) {
    s->lr_len[0] = s->board_rl[0];
    s->lr_len[1] = s->board_rl[1];
    int rc = s->road_color;
    if (rc < 0 || rc == prev) return;
    s->vp[rc] += 2; s->avp[rc] += 2;
    if (prev >= 0) { s->vp[prev] -= 2; s->avp[prev] -= 2; }
}
/* board.build_road */
static void board_build_road(State *s, int p, int e) {
    s->road[p] |= (u128)1 << e;
    int a = EDGE_A[e], b = EDGE_B[e];
    int ai = first_comp(s, p, a), bi = first_comp(s, p, b);
    uint64_t cm;
    if (ai < 0 && !is_enemy(s, a, p)) {
        s->comp[p][bi] |= 1ULL << a; cm = s->comp[p][bi];
    } else if (bi < 0 && !is_enemy(s, b, p)) {
        s->comp[p][ai] |= 1ULL << b; cm = s->comp[p][ai];
    } else if (ai >= 0 && bi >= 0 && ai != bi) {
        s->comp[p][ai] |= s->comp[p][bi]; cm = s->comp[p][ai];
        comp_remove(s, p, bi);
    } else {
        cm = s->comp[p][ai >= 0 ? ai : bi];
    }
    int cand = longest_path(s, p, cm);
    if (cand > s->board_rl[p]) s->board_rl[p] = cand;
    if (cand >= 5 && cand > s->road_length) { s->road_color = p; s->road_length = cand; }
}
/* board.build_settlement, non-initial part: may cut the opponent's road */
static void board_cut_roads(State *s, int p, int node) {
    int q = 1 - p, cnt = 0, other[3];
    for (int k = 0; k < 3; k++) {
        int nb = NODE_NBR[node][k];
        if (nb < 0) break;
        if (s->road[q] & ((u128)1 << NODE_NBR_EDGE[node][k])) other[cnt++] = nb;
    }
    if (cnt == 2) {
        uint64_t A = dfs_walk(s, other[0], q), C = dfs_walk(s, other[1], q);
        int bi = first_comp(s, q, node);
        if (bi >= 0) comp_remove(s, q, bi);
        comp_append(s, q, A);
        comp_append(s, q, C);
        int best = 0;
        for (int i = 0; i < s->ncomp[q]; i++) {
            int v = longest_path(s, q, s->comp[q][i]);
            if (v > best) best = v;
        }
        s->board_rl[q] = best;
        /* road_color, road_length = max(road_lengths.items()); dict order is seat 0, seat 1 */
        if (s->board_rl[1] > s->board_rl[0]) { s->road_color = 1; s->road_length = s->board_rl[1]; }
        else { s->road_color = 0; s->road_length = s->board_rl[0]; }
    } else if (cnt == 1) {
        int bi = first_comp(s, q, node);
        if (bi >= 0) s->comp[q][bi] &= ~(1ULL << node);
    }
}

/* ------------------------------------------------------------------ setup */
static void reset_dynamic(State *s) {
    memset(s->road, 0, sizeof s->road);
    s->settle[0] = s->settle[1] = s->city[0] = s->city[1] = 0;
    s->buildable = (1ULL << N_NODES) - 1;
    memset(s->comp, 0, sizeof s->comp);
    s->num_turns = 0;
    memset(s->res, 0, sizeof s->res);
    memset(s->dev, 0, sizeof s->dev);
    memset(s->played, 0, sizeof s->played);
    memset(s->owned, 0, sizeof s->owned);
    for (int p = 0; p < 2; p++) {
        s->vp[p] = s->avp[p] = 0;
        s->roads_av[p] = 15; s->sett_av[p] = 5; s->city_av[p] = 4;
        s->has_rolled[p] = s->has_played[p] = 0;
        s->lr_len[p] = s->board_rl[p] = 0;
        s->last_settle[p] = -1;
        s->ncomp[p] = 0;
        s->dc[p] = 0;
    }
    for (int i = 0; i < 5; i++) s->bank[i] = 19;
    s->road_color = -1; s->road_length = 0; s->army_owner = -1;
    s->cur = s->turn = 0;
    s->prompt = P_INIT_SETTLE;
    s->is_initial = 1;
    s->is_discarding = s->is_moving_knight = s->is_road_building = 0;
    s->free_roads = 0;
    s->winner = -1;
    s->vps_to_win = 10;
}

void cat_load(State *s, const int8_t *tile_res, const int8_t *tile_num, const int8_t *port_res,
              const int8_t *color, const int8_t *deck, int deck_n, int robber, uint64_t seed) {
    cat_init_tables();
    reset_dynamic(s);
    memcpy(s->tile_res, tile_res, N_TILES);
    memcpy(s->tile_num, tile_num, N_TILES);
    memcpy(s->port_res, port_res, N_PORTS);
    s->color[0] = color[0]; s->color[1] = color[1];
    memcpy(s->deck, deck, deck_n);
    s->deck_n = deck_n;
    s->robber = robber;
    s->rng = splitmix(seed);
}

static const int8_t START_DECK[25] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0, 1,1, 3,3, 2,2, 4,4,4,4,4};

void cat_new_game(State *s, uint64_t seed) {
    cat_init_tables();
    uint64_t r = splitmix(seed);
    int8_t tr[N_TILES], tn[N_TILES], pr[N_PORTS], col[2], deck[25];
    memcpy(tr, TEMPLATE_TILE_RES, N_TILES);
    memcpy(pr, TEMPLATE_PORT_RES, N_PORTS);
    memcpy(deck, START_DECK, 25);
    for (int i = N_TILES - 1; i > 0; i--) { int j = rng_below(&r, i + 1); int8_t t = tr[i]; tr[i] = tr[j]; tr[j] = t; }
    for (int i = N_PORTS - 1; i > 0; i--) { int j = rng_below(&r, i + 1); int8_t t = pr[i]; pr[i] = pr[j]; pr[j] = t; }
    for (int i = 24; i > 0; i--) { int j = rng_below(&r, i + 1); int8_t t = deck[i]; deck[i] = deck[j]; deck[j] = t; }
    /* map.initialize_tiles official_spiral: numbers along SPIRAL, skipping the desert */
    int k = 0, robber = 0;
    for (int i = 0; i < N_TILES; i++) {
        int t = SPIRAL[i];
        if (tr[t] < 0) { tn[t] = 0; continue; }
        tn[t] = SPIRAL_NUMBERS[k++];
    }
    for (int t = 0; t < N_TILES; t++) if (tr[t] < 0) { robber = t; break; }
    int first = rng_below(&r, 2);
    col[0] = first; col[1] = 1 - first;
    cat_load(s, tr, tn, pr, col, deck, 25, robber, rng_next(&r));
}

/* ------------------------------------------------------------------ move generation */
static int gen_edges(u128 e, int16_t *out, int n) {
    uint64_t lo = (uint64_t)e, hi = (uint64_t)(e >> 64);
    while (lo) { out[n++] = IDX_ROAD[ctz(lo)]; lo &= lo - 1; }
    while (hi) { out[n++] = IDX_ROAD[64 + ctz(hi)]; hi &= hi - 1; }
    return n;
}
static int gen_nodes(uint64_t m, const int16_t *idx, int16_t *out, int n) {
    while (m) { out[n++] = idx[ctz(m)]; m &= m - 1; }
    return n;
}

int cat_legal(const State *s, int16_t *out) {
    int p = s->cur, q = 1 - p, n = 0;
    switch (s->prompt) {
    case P_INIT_SETTLE:
        return gen_nodes(s->buildable, IDX_SETTLE, out, 0);
    case P_INIT_ROAD: {
        int ls = s->last_settle[p];
        return gen_edges(NODE_EDGE_MASK[ls] & ~(s->road[0] | s->road[1]), out, 0);
    }
    case P_ROBBER: {
        /* actions._robber_possibilities_without_friendly_robber */
        uint64_t opp = owned_nodes(s, q);
        int steal = hand_total(s, q) >= 1;
        int vc = 1 + s->color[q];
        for (int t = 0; t < N_TILES; t++) {
            if (t == s->robber) continue;
            out[n++] = IDX_ROBBER[t][(steal && (opp & TILE_MASK[t])) ? vc : 0];
        }
        return n;
    }
    case P_DISCARD:
        if (s->dc[p] <= 0) return 0;
        for (int r = 0; r < 5; r++) if (s->res[p][r] > 0) out[n++] = IDX_DISCARD[r];
        return n;
    case P_PLAY:
        break;
    default:
        return 0;
    }
    if (s->is_road_building) {
        if (s->roads_av[p] <= 0) return 0;
        return gen_edges(buildable_edges(s, p), out, 0);
    }
    if (can_play_dev(s, p, D_YOP)) {
        /* actions.year_of_plenty_possibilities (a set: singles deduplicated) */
        const int8_t *b = s->bank;
        int singles = 0;
        for (int i = 0; i < 5; i++)
            for (int j = i; j < 5; j++) {
                int ok = (i == j) ? b[i] >= 2 : (b[i] >= 1 && b[j] >= 1);
                if (ok) out[n++] = IDX_YOP2[i][j];
                else {
                    if (b[i] >= 1) singles |= 1 << i;
                    if (b[j] >= 1) singles |= 1 << j;
                }
            }
        for (int i = 0; i < 5; i++) if (singles & (1 << i)) out[n++] = IDX_YOP1[i];
    }
    if (can_play_dev(s, p, D_MONO))
        for (int r = 0; r < 5; r++) out[n++] = IDX_MONO[r];
    if (can_play_dev(s, p, D_KNIGHT)) out[n++] = IDX_KNIGHT;
    u128 be = 0;
    int be_done = 0;
    if (can_play_dev(s, p, D_RB) && s->roads_av[p] > 0) {
        be = buildable_edges(s, p); be_done = 1;
        if (be) out[n++] = IDX_RB;
    }
    if (!s->has_rolled[p]) {
        out[n++] = IDX_ROLL;
        return n;
    }
    out[n++] = IDX_END;
    if (s->roads_av[p] > 0 && has_cost(s, p, COST_ROAD)) {
        if (!be_done) be = buildable_edges(s, p);
        n = gen_edges(be, out, n);
    }
    if (has_cost(s, p, COST_SETTLE) && s->sett_av[p] > 0)
        n = gen_nodes(comp_union(s, p) & s->buildable, IDX_SETTLE, out, n);
    if (has_cost(s, p, COST_CITY) && s->city_av[p] > 0)
        n = gen_nodes(s->settle[p], IDX_CITY, out, n);
    if (has_cost(s, p, COST_DEV) && s->deck_n > 0) out[n++] = IDX_BUY;
    /* actions.inner_maritime_trade_possibilities */
    int pf = port_flags(s, p);
    int base = (pf & 32) ? 3 : 4;
    for (int i = 0; i < 5; i++) {
        int rate = (pf & (1 << i)) ? 2 : base;
        if (s->res[p][i] < rate) continue;
        for (int j = 0; j < 5; j++)
            if (j != i && s->bank[j] > 0) out[n++] = IDX_TRADE[i][rate][j];
    }
    return n;
}

/* ------------------------------------------------------------------ apply */
static void advance_turn(State *s, int dir) {
    int nx = (s->cur + dir + 2) % 2;
    s->cur = s->turn = nx;
    s->num_turns++;
}

static void play_dev(State *s, int p, int d) {
    /* state_functions.play_dev_card (+ maintain_largest_army for knights) */
    int prev = s->army_owner, prev_size = prev >= 0 ? s->played[prev][D_KNIGHT] : 0;
    s->dev[p][d]--;
    s->has_played[p] = 1;
    s->played[p][d]++;
    if (d != D_KNIGHT) return;
    int size = s->played[p][D_KNIGHT];
    if (size < 3) return;
    if (prev < 0) {
        s->army_owner = p; s->vp[p] += 2; s->avp[p] += 2;
    } else if (prev_size < size && prev != p) {
        s->army_owner = p; s->vp[p] += 2; s->avp[p] += 2;
        s->vp[prev] -= 2; s->avp[prev] -= 2;
    }
}

static void check_winner(State *s) {
    /* game.winning_color: last seat in order with enough actual points */
    for (int c = 0; c < 2; c++) if (s->avp[c] >= s->vps_to_win) s->winner = c;
}

void cat_step(State *s, int action, int outcome) {
    const int8_t *a = ACTION_TABLE[action];
    int p = s->cur, q = 1 - p;
    switch (a[0]) {
    case AT_END_TURN:
        s->has_played[p] = 0;
        s->has_rolled[p] = 0;
        for (int d = 0; d < 4; d++) s->owned[p][d] = s->dev[p][d] > 0;
        advance_turn(s, 1);
        s->prompt = P_PLAY;
        break;

    case AT_BUILD_SETTLEMENT: {
        int node = a[1];
        uint64_t bit = 1ULL << node;
        if (s->is_initial) {
            s->settle[p] |= bit;
            comp_append(s, p, bit);
            s->buildable &= ~(bit | NODE_NBR_MASK[node]);
            s->sett_av[p]--; s->vp[p]++; s->avp[p]++;
            if (popc(s->settle[p]) == 2) {
                for (int k = 0; k < 3; k++) {
                    int t = NODE_TILES[node][k];
                    if (t < 0) break;
                    int r = s->tile_res[t];
                    if (r >= 0) { s->bank[r]--; s->res[p][r]++; }
                }
            }
            s->last_settle[p] = node;
            s->prompt = P_INIT_ROAD;
        } else {
            int prev = s->road_color;
            s->settle[p] |= bit;
            board_cut_roads(s, p, node);
            s->buildable &= ~(bit | NODE_NBR_MASK[node]);
            s->sett_av[p]--; s->vp[p]++; s->avp[p]++;
            pay(s, p, COST_SETTLE);
            s->last_settle[p] = node;
            maintain_longest_road(s, prev);
        }
        break;
    }

    case AT_BUILD_ROAD: {
        int e = a[1];
        if (s->is_initial) {
            board_build_road(s, p, e);
            s->roads_av[p]--;
            int nb = popc(s->settle[0]) + popc(s->settle[1]);
            if (nb < 2) { advance_turn(s, 1); s->prompt = P_INIT_SETTLE; }
            else if (nb == 2) { s->prompt = P_INIT_SETTLE; }
            else if (nb == 4) { s->is_initial = 0; s->prompt = P_PLAY; }
            else { advance_turn(s, -1); s->prompt = P_INIT_SETTLE; }
        } else if (s->is_road_building && s->free_roads > 0) {
            int prev = s->road_color;
            board_build_road(s, p, e);
            s->roads_av[p]--;
            maintain_longest_road(s, prev);
            s->free_roads--;
            if (s->free_roads == 0 || s->roads_av[p] <= 0 || !buildable_edges(s, p)) {
                s->is_road_building = 0;
                s->free_roads = 0;
            }
        } else {
            int prev = s->road_color;
            board_build_road(s, p, e);
            s->roads_av[p]--;
            pay(s, p, COST_ROAD);
            maintain_longest_road(s, prev);
        }
        break;
    }

    case AT_BUILD_CITY: {
        uint64_t bit = 1ULL << a[1];
        s->settle[p] &= ~bit;
        s->city[p] |= bit;
        s->sett_av[p]++; s->city_av[p]--;
        s->vp[p]++; s->avp[p]++;
        pay(s, p, COST_CITY);
        break;
    }

    case AT_BUY_DEVELOPMENT_CARD: {
        int card = s->deck[--s->deck_n];
        s->dev[p][card]++;
        if (card == D_VP) s->avp[p]++;
        pay(s, p, COST_DEV);
        break;
    }

    case AT_ROLL: {
        s->has_rolled[p] = 1;
        int number = outcome >= 0 ? outcome
                   : (int)rng_below(&s->rng, 6) + (int)rng_below(&s->rng, 6) + 2;
        if (number == 7) {
            int first = -1;
            for (int i = 0; i < 2; i++) {
                int nc = hand_total(s, i);
                s->dc[i] = nc > 7 ? nc / 2 : 0;
                if (s->dc[i] > 0 && first < 0) first = i;
            }
            if (first >= 0) {
                s->cur = first; s->prompt = P_DISCARD; s->is_discarding = 1;
            } else {
                s->dc[0] = s->dc[1] = 0;
                s->prompt = P_ROBBER; s->is_moving_knight = 1;
            }
        } else {
            /* apply_action.yield_resources, with the bank-shortage rule */
            int want[2][5] = {{0}}, tot[5] = {0};
            for (int t = 0; t < N_TILES; t++) {
                if (s->tile_num[t] != number || t == s->robber) continue;
                int r = s->tile_res[t];
                uint64_t m = TILE_MASK[t];
                for (int i = 0; i < 2; i++) {
                    int c = popc(s->settle[i] & m) + 2 * popc(s->city[i] & m);
                    want[i][r] += c; tot[r] += c;
                }
            }
            for (int r = 0; r < 5; r++) {
                if (!tot[r] || s->bank[r] < tot[r]) continue;
                s->res[0][r] += want[0][r]; s->res[1][r] += want[1][r];
                s->bank[r] -= tot[r];
            }
            s->prompt = P_PLAY;
        }
        break;
    }

    case AT_DISCARD_RESOURCE: {
        int r = a[1];
        s->res[p][r]--; s->bank[r]++;
        s->dc[p]--;
        if (s->dc[p] <= 0) {
            int nx = -1;
            for (int i = p + 1; i < 2; i++) if (s->dc[i] > 0) { nx = i; break; }
            if (nx >= 0) s->cur = nx;
            else {
                s->cur = s->turn; s->prompt = P_ROBBER;
                s->is_discarding = 0; s->is_moving_knight = 1;
                s->dc[0] = s->dc[1] = 0;
            }
        }
        break;
    }

    case AT_MOVE_ROBBER: {
        if (a[2] != 0) {
            int v = (s->color[0] == a[2] - 1) ? 0 : 1;
            int r = outcome;
            if (r < 0) {
                int k = rng_below(&s->rng, hand_total(s, v));
                for (r = 0; r < 5; r++) { if (k < s->res[v][r]) break; k -= s->res[v][r]; }
            }
            s->res[v][r]--; s->res[p][r]++;
        }
        s->robber = a[1];
        s->prompt = P_PLAY;
        break;
    }

    case AT_PLAY_KNIGHT_CARD:
        play_dev(s, p, D_KNIGHT);
        s->prompt = P_ROBBER;
        break;

    case AT_PLAY_YEAR_OF_PLENTY:
        s->res[p][a[1]]++; s->bank[a[1]]--;
        if (a[3] == 2) { s->res[p][a[2]]++; s->bank[a[2]]--; }
        play_dev(s, p, D_YOP);
        s->prompt = P_PLAY;
        break;

    case AT_PLAY_MONOPOLY: {
        int r = a[1];
        s->res[p][r] += s->res[q][r];
        s->res[q][r] = 0;
        play_dev(s, p, D_MONO);
        s->prompt = P_PLAY;
        break;
    }

    case AT_PLAY_ROAD_BUILDING:
        play_dev(s, p, D_RB);
        s->is_road_building = 1;
        s->free_roads = 2;
        s->prompt = P_PLAY;
        break;

    case AT_MARITIME_TRADE: {
        int give = a[1], rate = a[2], get = a[3];
        s->res[p][give] -= rate; s->bank[give] += rate;
        s->res[p][get]++; s->bank[get]--;
        s->prompt = P_PLAY;
        break;
    }
    }
    check_winner(s);
}

int cat_done(const State *s) { return s->winner >= 0 || s->num_turns >= TURNS_LIMIT; }

int cat_autoplay(State *s, int16_t *legal, int *nlegal) {
    int k = 0;
    int n = cat_legal(s, legal);
    while (n == 1 && !cat_done(s)) {
        cat_step(s, legal[0], -1);
        k++;
        n = cat_legal(s, legal);
    }
    *nlegal = n;
    return k;
}

int cat_play_random(State *s, int autoplay) {
    int16_t legal[MAX_LEGAL];
    int steps = 0;
    while (!cat_done(s)) {
        int n = cat_legal(s, legal);
        int a = n == 1 ? legal[0] : legal[rng_below(&s->rng, n)];
        (void)autoplay;
        cat_step(s, a, -1);
        steps++;
    }
    return steps;
}

/* ------------------------------------------------------------------ observation */
/* Layout (all raw small integers, as float32; see README "Observation"):
 *   [0,152)   19 tiles x 8: resource one-hot(5), desert, pips (6-|7-n|), robber
 *   [152,206) 9 ports x 6: resource one-hot(5), 3:1
 *   [206,422) 54 nodes x 4: my settlement, my city, opp settlement, opp city
 *   [422,566) 72 edges x 2: my road, opp road
 *   [566,601) me: res(5) dev(5: K,YOP,MONO,RB,VP) played(4) vp actual_vp roads settles cities
 *             has_road has_army road_len has_rolled has_played_dev owned_at_start(4) ports(6) discard_left
 *   [601,624) opp: res(5) dev_count played(4) vp roads settles cities has_road has_army road_len ports(6)
 *   [624,639) bank(5) deck_left num_turns prompt one-hot(5) road_building free_roads my_turn  */
void cat_encode(const State *s, float *o) { cat_encode_seat(s, s->cur, o); }

void cat_encode_seat(const State *s, int p, float *o) {
    int q = 1 - p;
    memset(o, 0, OBS_DIM * sizeof(float));
    for (int t = 0; t < N_TILES; t++) {
        float *x = o + t * 8;
        int r = s->tile_res[t];
        if (r >= 0) x[r] = 1; else x[5] = 1;
        int nb = s->tile_num[t];
        x[6] = nb ? (float)(6 - (nb > 7 ? nb - 7 : 7 - nb)) : 0;
        x[7] = (t == s->robber);
    }
    for (int k = 0; k < N_PORTS; k++) {
        float *x = o + 152 + k * 6;
        int r = s->port_res[k];
        x[r >= 0 ? r : 5] = 1;
    }
    {
        float *x = o + 206;
        uint64_t m;
        m = s->settle[p]; while (m) { x[ctz(m) * 4 + 0] = 1; m &= m - 1; }
        m = s->city[p];   while (m) { x[ctz(m) * 4 + 1] = 1; m &= m - 1; }
        m = s->settle[q]; while (m) { x[ctz(m) * 4 + 2] = 1; m &= m - 1; }
        m = s->city[q];   while (m) { x[ctz(m) * 4 + 3] = 1; m &= m - 1; }
    }
    {
        float *x = o + 422;
        for (int side = 0; side < 2; side++) {
            u128 e = s->road[side == 0 ? p : q];
            uint64_t lo = (uint64_t)e, hi = (uint64_t)(e >> 64);
            while (lo) { x[ctz(lo) * 2 + side] = 1; lo &= lo - 1; }
            while (hi) { x[(64 + ctz(hi)) * 2 + side] = 1; hi &= hi - 1; }
        }
    }
    float *x = o + 566;
    for (int r = 0; r < 5; r++) *x++ = s->res[p][r];
    for (int d = 0; d < 5; d++) *x++ = s->dev[p][d];
    for (int d = 0; d < 4; d++) *x++ = s->played[p][d];
    *x++ = s->vp[p]; *x++ = s->avp[p];
    *x++ = s->roads_av[p]; *x++ = s->sett_av[p]; *x++ = s->city_av[p];
    *x++ = s->road_color == p; *x++ = s->army_owner == p; *x++ = s->board_rl[p];
    *x++ = s->has_rolled[p]; *x++ = s->has_played[p];
    for (int d = 0; d < 4; d++) *x++ = s->owned[p][d];
    int pf = port_flags(s, p);
    for (int i = 0; i < 6; i++) *x++ = (pf >> i) & 1;
    *x++ = s->dc[p];
    for (int r = 0; r < 5; r++) *x++ = s->res[q][r];
    *x++ = s->dev[q][0] + s->dev[q][1] + s->dev[q][2] + s->dev[q][3] + s->dev[q][4];
    for (int d = 0; d < 4; d++) *x++ = s->played[q][d];
    *x++ = s->vp[q];
    *x++ = s->roads_av[q]; *x++ = s->sett_av[q]; *x++ = s->city_av[q];
    *x++ = s->road_color == q; *x++ = s->army_owner == q; *x++ = s->board_rl[q];
    pf = port_flags(s, q);
    for (int i = 0; i < 6; i++) *x++ = (pf >> i) & 1;
    for (int r = 0; r < 5; r++) *x++ = s->bank[r];
    *x++ = s->deck_n; *x++ = s->num_turns;
    x[s->prompt] = 1; x += 5;
    *x++ = s->is_road_building; *x++ = s->free_roads; *x++ = s->cur == p;   /* the viewing seat is to move */
}

/* ------------------------------------------------------------------ canonical serialisation */
/* Must match tools/engine_parity.py:serialize_catanatron exactly. */
int cat_serialize(const State *s, int32_t *o) {
    int n = 0;
    for (int p = 0; p < 2; p++) {
        for (int r = 0; r < 5; r++) o[n++] = s->res[p][r];
        for (int d = 0; d < 5; d++) o[n++] = s->dev[p][d];
        for (int d = 0; d < 4; d++) o[n++] = s->played[p][d];
        for (int d = 0; d < 4; d++) o[n++] = s->owned[p][d];
        o[n++] = s->vp[p]; o[n++] = s->avp[p];
        o[n++] = s->roads_av[p]; o[n++] = s->sett_av[p]; o[n++] = s->city_av[p];
        o[n++] = s->road_color == p; o[n++] = s->army_owner == p;
        o[n++] = s->has_rolled[p]; o[n++] = s->has_played[p]; o[n++] = s->lr_len[p];
    }
    for (int r = 0; r < 5; r++) o[n++] = s->bank[r];
    o[n++] = s->deck_n;
    for (int i = 0; i < 25; i++) o[n++] = i < s->deck_n ? s->deck[i] : -1;
    for (int v = 0; v < N_NODES; v++) {
        int x = 0;
        for (int p = 0; p < 2; p++) {
            if ((s->settle[p] >> v) & 1) x = 1 + 2 * p;
            if ((s->city[p] >> v) & 1) x = 2 + 2 * p;
        }
        o[n++] = x;
    }
    for (int e = 0; e < N_EDGES; e++) {
        int x = 0;
        for (int p = 0; p < 2; p++) if ((s->road[p] >> e) & 1) x = 1 + p;
        o[n++] = x;
    }
    o[n++] = s->robber;
    o[n++] = s->board_rl[0]; o[n++] = s->board_rl[1];
    o[n++] = s->road_color; o[n++] = s->road_length; o[n++] = s->army_owner;
    o[n++] = (int32_t)(s->buildable & 0x7FFFFFFF);
    o[n++] = (int32_t)(s->buildable >> 31);
    for (int p = 0; p < 2; p++) {
        o[n++] = s->ncomp[p];
        for (int i = 0; i < MAXC; i++) {
            uint64_t m = i < s->ncomp[p] ? s->comp[p][i] : 0;
            o[n++] = (int32_t)(m & 0x7FFFFFFF);
            o[n++] = (int32_t)(m >> 31);
        }
    }
    o[n++] = s->num_turns; o[n++] = s->cur; o[n++] = s->turn; o[n++] = s->prompt;
    o[n++] = s->is_initial; o[n++] = s->is_discarding; o[n++] = s->dc[0]; o[n++] = s->dc[1];
    o[n++] = s->is_moving_knight; o[n++] = s->is_road_building; o[n++] = s->free_roads;
    o[n++] = s->is_initial ? s->last_settle[0] : -1;
    o[n++] = s->is_initial ? s->last_settle[1] : -1;
    o[n++] = s->winner;
    return n;
}
