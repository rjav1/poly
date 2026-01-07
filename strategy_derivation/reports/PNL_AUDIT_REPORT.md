# PnL Accounting Audit Report

## Summary

### Complete-Set (buy_both) Trades

- Sampled trades: 0
- PnL calculation correct: 0/0

### Directional Trades (Audit)

- Sampled trades: 40
- PnL calculation correct: 40/40
- Trades with negative PnL: 20/40

### Late Directional Strategy Simulation

- Total trades: 15
- Winning trades: 15
- Losing trades: 0
- Win rate: 100.0%
- Total PnL: $1.90

**WARNING**: 100% win rate on directional strategy is suspicious!

## Implementation Issues Found

### 17_implement_strategies.py

- **Issue**: Y is read from entry_row, but Y should be the market outcome (constant for whole market)
- **Pattern**: `entry_row.get('Y'`
- **Fix**: Y should come from market_info or be consistent across the market, not per-row

### 18_validation_suite.py

- **Issue**: Y is read from entry_row in validation suite
- **Pattern**: `entry_row.get('Y'`
- **Fix**: Y should be consistent market outcome

## Sample Directional Trades

| Market | Y | Delta | Side | Entry | Exit | PnL | Correct? |
|--------|---|-------|------|-------|------|-----|----------|
| 20260106_0415_3232 | 0 | -14.4 | buy_down | 0.930 | 1.0 | $0.070 | Yes |
| 20260106_0530_3224 | 0 | -18.9 | buy_down | 0.820 | 1.0 | $0.180 | Yes |
| 20260106_0630_3227 | 1 | 25.8 | buy_up | 0.950 | 1.0 | $0.050 | Yes |
| 20260106_0645_3250 | 0 | -21.4 | buy_down | 0.800 | 1.0 | $0.200 | Yes |
| 20260106_0700_3239 | 0 | -13.4 | buy_down | 0.870 | 1.0 | $0.130 | Yes |
| 20260106_0715_3234 | 0 | -32.4 | buy_down | 0.980 | 1.0 | $0.020 | Yes |
| 20260106_0730_3218 | 0 | -16.5 | buy_down | 0.960 | 1.0 | $0.040 | Yes |
| 20260106_0800_3220 | 1 | 13.7 | buy_up | 0.980 | 1.0 | $0.020 | Yes |
| 20260106_0900_3217 | 1 | 14.5 | buy_up | 0.790 | 1.0 | $0.210 | Yes |
| 20260106_0930_3225 | 0 | -15.8 | buy_down | 0.980 | 1.0 | $0.020 | Yes |

## Recommendations

1. **Investigate 100% win rate**: Directional strategies should have losing trades.
   - Check if Y (outcome) is being read correctly
   - Verify Y is the FINAL outcome, not a per-row signal

2. **Verify Y consistency**: Y should be the same for all rows in a market.
3. **Check for PnL clipping**: Ensure `max(0, pnl)` is NOT used.
4. **Validate exit logic**: For hold-to-expiry, exit = 1.0 or 0.0 based on Y.
