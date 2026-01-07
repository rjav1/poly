# Leakage Audit Note: Strategy B

**Date:** January 6, 2026  
**Subject:** Hard audit of Strategy B (Late Directional Taker) for look-ahead bias

---

## Summary

**Verdict: NO OBVIOUS LEAKAGE FOUND**

Strategy B passes the leakage audit. The placebo test "failure" (edge persisting with 30s stale data) is explained by high autocorrelation in delta_bps, NOT by a bug or data leakage.

---

## Audit Results

### Checks Performed

| Check | Status | Details |
|-------|--------|---------|
| K constant per market | PASS | K has 1 unique value per market |
| K = CL(t=0) | PASS | K matches CL at market start (0.00 bps diff) |
| delta_bps formula correct | PASS | delta_bps = (cl_mid - K) / K * 10000 verified |
| No settlement column access | PASS | Strategy code doesn't use settlement |
| No Y (outcome) column access | PASS | Strategy code doesn't use Y |
| Uses tau from dataset | PASS | tau comes from row, not computed from future |
| Only expected columns used | PASS | delta_bps, cl_mid, tau, t, momentum |
| Momentum is backward-looking | PASS | Uses pct_change() + rolling() |
| Placebo shift direction correct | PASS | shift(30) = staler data |
| All CL columns shifted | PASS | cl_mid, cl_bid, cl_ask, delta, delta_bps |
| K not shifted | PASS | Strike remains constant |
| No future data access patterns | PASS | No shift(-n) or iloc[-1] found |

### Code Paths Verified

**Strategy B signal generation** (`scripts/backtest/strategies.py`, lines 845-890):

```python
# At each row (time t), strategy uses ONLY:
- row['t']           # Current time
- row['tau']         # Time to expiry (900 - t)
- row['delta_bps']   # Distance from strike at time t
- row['momentum']    # Rolling sum of past CL returns

# Momentum computed from PAST data only:
df['cl_return_bps'] = df['cl_mid'].pct_change()  # Looks at t-1
df['momentum'] = df['cl_return_bps'].rolling(10).sum()  # Looks at [t-10, t]
```

**Placebo shift implementation** (`strategy_derivation/04_run_backtests.py`, lines 175-195):

```python
# Shifts CL data to be 30s STALER (correct):
df_shifted[col] = df_shifted.groupby('market_id')[col].shift(30)
# After this, at time t we see cl_mid from t-30
```

---

## Key Finding: delta_bps Autocorrelation

**This explains the placebo test "failure":**

| Lag | Autocorrelation |
|-----|-----------------|
| 1s  | 0.996 |
| 5s  | 0.967 |
| 15s | 0.883 |
| **30s** | **0.779** |
| 60s | 0.614 |
| 120s | 0.354 |

### Interpretation

1. **delta_bps is highly persistent** - if CL is above strike now, it's very likely still above strike 30 seconds later

2. **The placebo test doesn't destroy the signal** because:
   - With 30s shift, we see delta_bps(t-30) instead of delta_bps(t)
   - But delta_bps(t-30) is a good predictor of delta_bps(t) (correlation = 0.779)
   - So the "stale" signal still carries information

3. **This is NOT a bug** - it means Strategy B exploits:
   - The **persistence** of CL being above/below strike
   - **NOT** short-term CL-PM lead-lag

### What Strategy B Actually Does

```
Strategy B logic:
IF tau < 420 seconds (last 7 minutes)
AND |delta_bps| > 10 (CL is 10+ bps from strike)
AND momentum confirms direction
THEN buy the side CL is trending toward
```

This is essentially:
> "If CL has been significantly above the strike, it's probably going to stay above the strike until expiry, so buy UP."

This is a valid **trend persistence / momentum** strategy, not a CL-PM information timing strategy.

---

## Conclusion

**Strategy B is NOT using look-ahead bias or data leakage.**

The placebo test result (t=2.87 with 30s shift vs t=3.09 original) is explained by:
- High autocorrelation in delta_bps (0.779 at 30s lag)
- The strategy exploits price persistence, not information speed advantage

### Implications

1. **The edge is real** (not a bug), but it's different from what we thought
2. **The edge is NOT from CL-PM lead-lag** - it's from CL trend persistence
3. **This strategy can be replicated** without high-frequency data - you just need to know if CL is above/below strike in the late window
4. **Further validation needed** via out-of-sample testing and execution realism

---

## Files Generated

- `leakage_audit_results.json` - Raw audit check results
- This note

