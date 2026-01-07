#!/usr/bin/env python3
"""
Build research dataset from 6-level markets folder with coverage filtering.

This script is a convenience wrapper around build_research_dataset_v2.py
that automatically uses the markets_6levels folder and applies coverage filtering.
"""

import subprocess
import sys
from pathlib import Path

# Default to 80% coverage threshold
DEFAULT_MIN_COVERAGE = 80.0

def main():
    """Build dataset from 6-level markets with coverage filtering."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build research dataset from 6-level markets with coverage filtering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build with default 80% coverage threshold
  python scripts/build_6level_dataset.py
  
  # Build with 90% coverage threshold
  python scripts/build_6level_dataset.py --min-coverage 90
  
  # Build with custom output directory
  python scripts/build_6level_dataset.py --output-dir data_v2/research_6levels
        """
    )
    
    parser.add_argument(
        "--min-coverage", 
        type=float, 
        default=DEFAULT_MIN_COVERAGE,
        help=f"Minimum BOTH coverage percentage to include (default: {DEFAULT_MIN_COVERAGE}%%)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for research dataset (default: data_v2/research_6levels)"
    )
    
    parser.add_argument(
        "--use-ground-truth",
        action="store_true",
        help="Use ground truth for strike price (K)"
    )
    
    args = parser.parse_args()
    
    # Set default output directory
    if args.output_dir is None:
        args.output_dir = "data_v2/research_6levels"
    
    # Check that 6-level markets folder exists
    markets_6level_dir = Path("data_v2/markets_6levels")
    if not markets_6level_dir.exists():
        print(f"ERROR: {markets_6level_dir} does not exist!")
        print("Run 'python scripts/copy_6level_markets.py' first to create it.")
        sys.exit(1)
    
    # Build command
    build_script = Path("scripts/build_research_dataset_v2.py")
    if not build_script.exists():
        print(f"ERROR: {build_script} does not exist!")
        sys.exit(1)
    
    cmd = [
        sys.executable,
        str(build_script),
        "--markets-dir", str(markets_6level_dir),
        "--output-dir", args.output_dir,
        "--min-coverage", str(args.min_coverage)
    ]
    
    if args.use_ground_truth:
        cmd.append("--use-ground-truth")
    
    print("=" * 80)
    print("BUILDING 6-LEVEL RESEARCH DATASET")
    print("=" * 80)
    print(f"Markets directory: {markets_6level_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Minimum coverage: {args.min_coverage}%")
    print()
    
    # Run the build script
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print()
        print("=" * 80)
        print("BUILD COMPLETE")
        print("=" * 80)
        print(f"Dataset saved to: {Path(args.output_dir).absolute()}")
        print()
        print("To view coverage report:")
        print(f"  python scripts/list_6level_markets_coverage.py")
    else:
        sys.exit(result.returncode)

if __name__ == "__main__":
    main()


