# Validation Report
**Generated**: Phase 8 Validation Suite

## Summary

| Strategy | Original t-stat | Time Shift | Permutation | Walk-Forward | P(positive) |
|----------|-----------------|------------|-------------|--------------|-------------|
| H6_underround_harvest | 1.72 | FAIL | FAIL | PASS | 100.0% |
| H9_early_inventory | 7.94 | FAIL | FAIL | PASS | 100.0% |
| H8_late_directional | 3.96 | FAIL | FAIL | PASS | 100.0% |

## Detailed Results

### H6_underround_harvest

**Parameters**: {'epsilon': 0.005, 'min_tau': 120, 'max_tau': 840, 'min_capacity': 0}

**Walk-Forward**: Train t-stat=5.02, Test t-stat=1.28

**Bootstrap 95% CI**: [0.0411, 0.2901]

### H9_early_inventory

**Parameters**: {'min_tau': 500, 'epsilon': 0.01}

**Walk-Forward**: Train t-stat=6.56, Test t-stat=4.52

**Bootstrap 95% CI**: [0.0120, 0.0198]

### H8_late_directional

**Parameters**: {'max_tau': 120, 'delta_threshold_bps': 15, 'hold_seconds': 60}

**Walk-Forward**: Train t-stat=3.78, Test t-stat=1.49

**Bootstrap 95% CI**: [0.0031, 0.0088]

