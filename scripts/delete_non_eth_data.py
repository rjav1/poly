#!/usr/bin/env python3
"""
Delete all non-ETH raw data and processed markets.

This script removes:
- All raw data for assets other than ETH (chainlink and polymarket)
- All processed markets for assets other than ETH
"""

import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import STORAGE, SUPPORTED_ASSETS


def delete_non_eth_raw_data():
    """Delete all non-ETH raw data files."""
    raw_dir = Path(STORAGE.raw_dir)
    
    deleted_count = 0
    deleted_size = 0
    
    # Delete chainlink data
    chainlink_dir = raw_dir / "chainlink"
    if chainlink_dir.exists():
        for asset_dir in chainlink_dir.iterdir():
            if asset_dir.is_dir() and asset_dir.name != "ETH":
                print(f"  Deleting chainlink/{asset_dir.name}/...")
                for file in asset_dir.rglob("*"):
                    if file.is_file():
                        size = file.stat().st_size
                        deleted_size += size
                        file.unlink()
                    deleted_count += 1
                try:
                    asset_dir.rmdir()  # Remove empty directory
                except:
                    pass
    
    # Delete polymarket data
    polymarket_dir = raw_dir / "polymarket"
    if polymarket_dir.exists():
        for asset_dir in polymarket_dir.iterdir():
            if asset_dir.is_dir() and asset_dir.name != "ETH":
                print(f"  Deleting polymarket/{asset_dir.name}/...")
                for file in asset_dir.rglob("*"):
                    if file.is_file():
                        size = file.stat().st_size
                        deleted_size += size
                        file.unlink()
                    deleted_count += 1
                try:
                    asset_dir.rmdir()  # Remove empty directory
                except:
                    pass
    
    return deleted_count, deleted_size


def delete_non_eth_markets():
    """Delete all non-ETH processed markets."""
    markets_dir = Path(STORAGE.markets_dir)
    
    deleted_count = 0
    
    if markets_dir.exists():
        for asset_dir in markets_dir.iterdir():
            if asset_dir.is_dir() and asset_dir.name != "ETH":
                print(f"  Deleting markets/{asset_dir.name}/...")
                market_count = sum(1 for _ in asset_dir.iterdir() if _.is_dir())
                deleted_count += market_count
                shutil.rmtree(asset_dir, ignore_errors=True)
    
    return deleted_count


def main():
    print("=" * 60)
    print("Delete Non-ETH Data")
    print("=" * 60)
    print()
    print("This will delete:")
    print("  1. All raw data for assets other than ETH")
    print("  2. All processed markets for assets other than ETH")
    print()
    
    # Auto-delete without confirmation (for CLI usage)
    # If you want confirmation, use: python scripts/delete_non_eth_data.py --confirm
    import sys
    if '--confirm' in sys.argv:
        response = input("Continue? [y/N]: ").strip().lower()
        if response != 'y':
            print("Cancelled.")
            return 0
    
    print("\nDeleting non-ETH raw data...")
    raw_files, raw_size = delete_non_eth_raw_data()
    
    print("\nDeleting non-ETH processed markets...")
    market_count = delete_non_eth_markets()
    
    print("\n" + "=" * 60)
    print("Deletion Complete")
    print("=" * 60)
    print(f"  Raw files deleted: {raw_files}")
    print(f"  Raw data size: {raw_size / 1024 / 1024:.2f} MB")
    print(f"  Markets deleted: {market_count}")
    print()
    print("Only ETH data remains.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

