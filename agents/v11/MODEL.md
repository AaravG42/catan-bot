# v11: v5's search over a 123-feature value net (efeats2)

- Features: agents/common/efeats2.py = the 76 of efeats.py + 47 new (affordability for both players,
  race state incl. expected hidden VP, robber exposure, contested spots, phase, pieces, bank).
- Data: 12,047,566 positions from C-engine self-play shards data/sp/E1 + E2 (v3/v5 nets, 15% recording,
  both seats), features rebuilt from the stored 639-float observation. Seeds 100,000+ (Claude range).
- Train: tools.train_value --hidden 64 32 --epochs 30 --bs 8192 (GPU). Test logloss 0.4081, acc 0.7958.
  Same rows, old 76 features: 0.4193 / 0.7882.
- Arena (eval seeds 990000, 2000 games, unclocked): 0.549 +-0.022 vs v5, 0.7725 +-0.018 vs F, 0 over.
