# Consistency Audit Report

## Summary

**Total issues found: 7**

| Issue Type | Count |
|------------|-------|
| PLACEBO_LOGIC_ERROR | 4 |
| WIN_RATE_SUSPICION | 1 |
| PERMUTATION_TEST_INAPPROPRIATE | 2 |

## Issues by Type

### PLACEBO_LOGIC_ERROR

**H6_underround_harvest**

- Category: PM_ONLY
- Concern: PM-only strategies don't use CL data, so CL time-shift should have no effect. The current logic expects edge to DEGRADE, but for PM_ONLY it should PERSIST.
- Current PASS/FAIL: False
- Correct PASS/FAIL: True

**H6_underround_harvest**

- Category: PM_ONLY
- Concern: PM-only strategies don't use CL data, so CL time-shift should have no effect. The current logic expects edge to DEGRADE, but for PM_ONLY it should PERSIST.
- Current PASS/FAIL: False
- Correct PASS/FAIL: True

**H9_early_inventory**

- Category: INVENTORY
- Concern: PM-only strategies don't use CL data, so CL time-shift should have no effect. The current logic expects edge to DEGRADE, but for INVENTORY it should PERSIST.
- Current PASS/FAIL: False
- Correct PASS/FAIL: True

**H9_early_inventory**

- Category: INVENTORY
- Concern: PM-only strategies don't use CL data, so CL time-shift should have no effect. The current logic expects edge to DEGRADE, but for INVENTORY it should PERSIST.
- Current PASS/FAIL: False
- Correct PASS/FAIL: True

### WIN_RATE_SUSPICION

**H8_late_directional**

- Category: TIMING
- Concern: Directional strategy shows 100% win rate. This is suspicious - directional trades should have losses when direction is wrong.
- Recommendation: Audit PnL calculation in execute_signal(). Check if Y (outcome) is being used correctly. Verify losses are not being clipped with max(0, pnl).

### PERMUTATION_TEST_INAPPROPRIATE

**H6_underround_harvest**

- Category: PM_ONLY
- Concern: Permuting 't' breaks market structure for underround strategies. When timing is randomized, the strategy finds underround at 'wrong' times, creating MORE signals (but meaningless ones).
- Recommendation: For complete-set arb, permutation test is not appropriate. Consider: 1) Skip this test for PM_ONLY, or 2) Permute Y (outcome) instead.

**H9_early_inventory**

- Category: INVENTORY
- Concern: Permuting 't' breaks market structure for underround strategies. When timing is randomized, the strategy finds underround at 'wrong' times, creating MORE signals (but meaningless ones).
- Recommendation: For complete-set arb, permutation test is not appropriate. Consider: 1) Skip this test for PM_ONLY, or 2) Permute Y (outcome) instead.

## Key Recommendations

1. **Fix placebo logic** to be strategy-class aware:
   - PM_ONLY strategies: CL time-shift should NOT affect edge (test passes if edge persists)
   - CL-dependent strategies: CL time-shift SHOULD degrade edge (test passes if edge degrades)

2. **Audit PnL calculation** for directional strategies:
   - Verify losses are correctly computed when direction is wrong
   - Check that Y (outcome) is properly used in exit price calculation

3. **Fix permutation test** or skip for PM_ONLY strategies:
   - Permuting 't' is not meaningful for underround strategies
   - Consider permuting Y (outcome) for directional strategies instead

4. **Ensure report reads directly from JSON**:
   - No manual interpretation of PASS/FAIL
   - Strategy-class determines which tests are relevant
