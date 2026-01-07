# Validation Report (Strategy-Class Aware)

## Summary

| Strategy | Category | Time Shift | Outcome Shuffle | Walk-Forward | Win Rate | P(positive) |
|----------|----------|------------|-----------------|--------------|----------|-------------|
| H6_underround_harvest | PM_ONLY | PASS | N/A | PASS | 100% | 100% |
| H9_early_inventory | INVENTORY | PASS | FAIL | PASS | 100% | 100% |
| H8_late_directional | TIMING | FAIL | PASS | PASS | 100% | 100% |

## Key Changes from Original Validation

1. **Time Shift Placebo**: Now strategy-class aware
   - PM_ONLY strategies: PASS if edge persists (no CL dependency expected)
   - CL-dependent: PASS if edge degrades (CL dependency confirmed)

2. **Outcome Shuffle**: Replaces timing permutation
   - Only applied to directional strategies
   - Tests if direction prediction matters
   - Skipped for complete-set arb (outcome doesn't affect PnL)

3. **Win Rate Tracking**: Now correctly tracks wins/losses
   - Uses market-level Y for directional trades
   - Exposes strategies with suspicious 100% win rates
