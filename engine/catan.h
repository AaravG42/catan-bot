/* Fast 2-player Catan engine, rule-for-rule identical to the patched catanatron in .venv.
 * See engine/README.md. Ids (nodes, edges, tiles, ports, actions) are catanatron's. */
#pragma once
#include <stdint.h>
#include "topology.h"

#define MAXC 12           /* connected components per player (catanatron keeps a list) */
#define TURNS_LIMIT 1000
#define MAX_LEGAL 128
#define OBS_DIM 639

enum { P_INIT_SETTLE = 0, P_INIT_ROAD = 1, P_PLAY = 2, P_DISCARD = 3, P_ROBBER = 4 };
enum { R_WOOD = 0, R_BRICK, R_SHEEP, R_WHEAT, R_ORE };
enum { D_KNIGHT = 0, D_YOP, D_MONO, D_RB, D_VP };   /* catanatron DEVELOPMENT_CARDS order */

typedef unsigned __int128 u128;

typedef struct {
    u128 road[2];                 /* edge bitboards per seat */
    uint64_t settle[2], city[2];  /* node bitboards per seat */
    uint64_t buildable;           /* board.board_buildable_ids */
    uint64_t comp[2][MAXC];       /* board.connected_components[color], in list order */
    uint64_t rng;                 /* per-env xorshift state (training mode) */
    int16_t num_turns;
    int8_t tile_res[N_TILES];     /* -1 desert */
    int8_t tile_num[N_TILES];     /* 0 desert */
    int8_t port_res[N_PORTS];     /* -1 = 3:1 */
    int8_t color[2];              /* seat -> 0 RED, 1 BLUE */
    int8_t res[2][5], dev[2][5], played[2][4], owned[2][4];
    int8_t vp[2], avp[2], roads_av[2], sett_av[2], city_av[2];
    int8_t has_rolled[2], has_played[2];
    int8_t lr_len[2];             /* player_state LONGEST_ROAD_LENGTH (lags board_rl) */
    int8_t board_rl[2];           /* board.road_lengths */
    int8_t last_settle[2];
    int8_t ncomp[2];
    int8_t dc[2];                 /* discard_counts */
    int8_t bank[5];
    int8_t deck[25];
    int8_t deck_n;
    int8_t robber;
    int8_t road_color, road_length, army_owner;
    int8_t cur, turn, prompt;
    int8_t is_initial, is_discarding, is_moving_knight, is_road_building, free_roads;
    int8_t winner;                /* -1 none */
    int8_t vps_to_win;
} State;

/* ---- single-state API ---- */
void cat_init_tables(void);
void cat_new_game(State *s, uint64_t seed);  /* native board + deck + seating, same distribution */
void cat_load(State *s, const int8_t *tile_res, const int8_t *tile_num, const int8_t *port_res,
              const int8_t *color, const int8_t *deck, int deck_n, int robber, uint64_t seed);
int  cat_legal(const State *s, int16_t *out);              /* returns count */
void cat_step(State *s, int action, int outcome);          /* outcome<0: sample with s->rng */
int  cat_done(const State *s);
int  cat_autoplay(State *s, int16_t *legal, int *nlegal); /* plays forced moves; returns count */
void cat_encode(const State *s, float *out);               /* OBS_DIM floats, view of s->cur */
void cat_encode_seat(const State *s, int seat, float *out); /* view of `seat` */
int  cat_serialize(const State *s, int32_t *out);          /* canonical state for parity */
int  cat_play_random(State *s, int autoplay);              /* uniform random to the end; steps */
int  cat_state_size(void);
