========================================================================
  TRACE-SRS vs FSRS-6 vs SM-2
  Real Dataset: fsrs-vs-sm17 GitHub repository
========================================================================

Loading dataset from [anonymized]/dataset...
Found 19 CSV files
Loaded 149761 cards with 609429 total reviews
  149761 cards, 609429 reviews
  Recall rate: 0.871
  Avg interval: 64.82 days

Running simulations...
  Simulating SM-2...
  Simulating FSRS-6...
  Simulating TRACE-SRS...

────────────────────────────────────────────────────────────────────────
  UNIVERSAL METRIC (%, lower = better, 0% = perfect)
────────────────────────────────────────────────────────────────────────

  Algorithm        Self-UM   Cross-UM    vs FSRS-6  UM(SM-17 ref)
  ────────────── ───────── ────────── ──────────── ──────────────
  SM-2              50.42%     39.52%      +12.97pp        33.90%
  FSRS-6            29.90%     26.55%      baseline        22.67%
  TRACE-SRS         18.97%     18.52%       -8.03pp        18.40%

────────────────────────────────────────────────────────────────────────
  TRADITIONAL METRICS
────────────────────────────────────────────────────────────────────────

  Metric               SM-2     FSRS-6  TRACE-SRS     TRACE vs F6
  ────────────── ────────── ────────── ──────────  ──────────────
  Log Loss ↓         1.9741     0.6331     0.4775   -0.1556 (better)
  RMSE(bins) ↓       0.5045     0.2994     0.1907   -0.1087 (better)
  AUC ↑              0.4670     0.5502     0.5524   +0.0022 (better)
  Brier ↓            0.3671     0.2018     0.1480   -0.0538 (better)
  ECE ↓              0.3965     0.2287     0.1804   -0.0483 (better)

────────────────────────────────────────────────────────────────────────
  SUMMARY
────────────────────────────────────────────────────────────────────────

  TRACE-SRS wins 5/5 metrics vs FSRS-6
  UM improvement: +8.03pp
  Log Loss improvement: +24.6%
