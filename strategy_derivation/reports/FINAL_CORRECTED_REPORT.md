# Strategy Discovery Pipeline - Final Corrected Report

**Generated**: 2026-01-06 21:40:34

---

## Executive Summary

This report presents the corrected results of the strategy discovery pipeline after addressing the following issues:

1. **Placebo test logic** - Now strategy-class aware (PM-only vs CL-dependent)
2. **PnL accounting** - Verified correct handling of directional losses
3. **Execution realism** - Added fill-risk models and capacity constraints
4. **Strategy consolidation** - Narrowed to 2 production candidates

### Key Findings

- **Audit found 7 issues** that were addressed
- **H10 (tight spread) is underround in disguise** - dropped as separate strategy
- **Capacity constraint severe**: Only 33% of signals have capacity >= 1

---

## Production Candidates

### Candidate 1: Complete-Set Arbitrage

**Status**: READY FOR PAPER TRADING

**Mechanism**: Buy both UP and DOWN tokens when sum_asks < 1. Guaranteed $1 payoff at expiry regardless of outcome.

**Best Configuration**:

- Epsilon (min underround): 0.005
- Tau window: [600, 900] seconds
- Cooldown: 30 seconds

**Performance**:

- t-stat: 6.53
- Total PnL: $0.72
- Signals: 51
- Win rate: 100% (expected for arb)

**Execution Realism**:

- Taker fill rate: 33%
- Taker PnL: $5.83
- Maker (conservative) fill rate: 32%
- Maker PnL: $2.35

**Validation Status**:

- Time shift (30s): PASS (edge persists as expected)
- Walk-forward: PASS (test t=1.28)
- Bootstrap P(positive): 100%
- 95% CI: [$0.0409, $0.2848]

### Candidate 2: Late Directional (CL-Based)

**Status**: SUSPICIOUS - INVESTIGATE

**Mechanism**: Take directional position in late window based on CL delta. Positive delta suggests UP will win.

**Best Configuration**:

- Max tau: 120 seconds
- Delta threshold: 15 bps
- Cooldown: 60 seconds

**Performance**:

- t-stat: 3.96
- Total PnL: $0.28
- Signals: 23
- Win/Loss: 23/0
- Direction accuracy: 100%

**Concerns**:

- Time-shift placebo FAILED (edge did not degrade under CL staleness)
- This suggests the edge may not be from CL lead-lag
- Need more data to confirm direction accuracy

---

## Dropped Strategies

### H10: Tight Spread Entry

**Reason**: Decomposition analysis showed:
- WITH underround: 9 signals, $1.56 PnL
- WITHOUT underround: 627 signals, $0.00 PnL

**Conclusion**: Tight spread is just an underround proxy. Merged into complete-set family.

### H11: CL Momentum Following

**Reason**: Not implemented in focused backtests. Preliminary evidence showed weak signal.

---

## Execution Constraints

### Capacity Analysis

- p10 capacity: 0.00 contracts
- p50 capacity: 0.00 contracts
- p90 capacity: 61.00 contracts
- % signals with capacity >= 1: 33.1%

### Fill Model Impact

| Fill Model | Fill Rate | PnL | t-stat |
|------------|-----------|-----|--------|
| taker | 33% | $5.83 | 1.18 |
| maker_conservative | 32% | $2.35 | 2.39 |
| maker_realistic | 16% | $1.91 | 1.94 |

**Key Insight**: Maker fill risk significantly reduces edge. Conservative maker model cuts PnL by ~60%.

---

## Next Steps

### Immediate (Paper Trading)

1. Deploy shadow trader for complete-set arb strategy
2. Log all signals with book snapshots
3. Track post-signal outcomes for 100+ markets
4. Verify fill rates match expectations

### Medium-Term (Data Collection)

1. Collect 500+ additional 15m markets
2. Run walk-forward validation across time blocks
3. Validate late directional with proper CL staleness test

### Long-Term (Production)

1. If paper trading confirms edge, implement with small capital
2. Monitor fill rates and adjust execution mode
3. Scale capacity gradually

---

## Disclaimers

1. **Sample size**: Only 47 markets analyzed - need 500+ for confidence
2. **Execution assumptions**: Real fills may be worse than modeled
3. **Market regime**: Results may not generalize to different conditions
4. **Competition**: Other traders may compete for same opportunities
5. **Not financial advice**: This is research, not a trading recommendation
