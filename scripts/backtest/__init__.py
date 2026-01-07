# Backtest module for ETH lead-lag analysis
"""
This module implements the latency-aware taker backtest for Polymarket/Chainlink
lead-lag strategy discovery.

Key components:
- execution_model: Simulate fills with Split/Redeem conversion routing
- data_loader: Load and filter ETH markets
- event_detection: Detect CL price events
- event_study: Measure PM response to CL events
- latency_cliff: Analyze edge vs latency
- strategies: Trading strategy classes
- backtest_engine: Run backtests and compute metrics
- parameter_sweep: Optimize parameters with train/test
- fair_value: Baseline probability model
- placebo_tests: Validate results
- visualizations: Generate all plots
"""

__version__ = "0.1.0"

