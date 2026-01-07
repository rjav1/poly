#!/usr/bin/env python3
"""
Verify that the PM collection fix is present and clear Python cache.

Run this on the other PC to ensure the fix is active.
"""

import sys
from pathlib import Path

print("=" * 60)
print("Verifying PM Collection Fix")
print("=" * 60)
print()

# Check if the fix is in the code
fast_orch_path = Path(__file__).parent.parent / "src" / "fast_orchestrator.py"

if not fast_orch_path.exists():
    print(f"ERROR: {fast_orch_path} not found!")
    sys.exit(1)

content = fast_orch_path.read_text(encoding='utf-8')

# Check for the correct API endpoint
if 'markets/slug/' in content and 'events?slug=' not in content:
    print("✓ Fix is present: Using correct /markets/slug/ endpoint")
else:
    print("✗ ERROR: Fix NOT found! Still using old /events?slug= endpoint")
    print("  Make sure you pulled the latest code:")
    print("    git fetch origin")
    print("    git reset --hard origin/main")
    sys.exit(1)

# Check for token ID extraction
if 'outcomes = json_module.loads(data.get("outcomes"' in content:
    print("✓ Fix is present: Using correct token ID extraction")
else:
    print("✗ ERROR: Token ID extraction code not found!")
    sys.exit(1)

# Check for clobTokenIds parsing
if 'clob_token_ids = json_module.loads(data.get("clobTokenIds"' in content:
    print("✓ Fix is present: Parsing clobTokenIds correctly")
else:
    print("✗ ERROR: clobTokenIds parsing not found!")
    sys.exit(1)

print()
print("=" * 60)
print("Clearing Python bytecode cache...")
print("=" * 60)
import shutil

cache_dirs = list(Path(__file__).parent.parent.rglob("__pycache__"))
removed_dirs = 0
for cache_dir in cache_dirs:
    try:
        shutil.rmtree(cache_dir)
        removed_dirs += 1
        print(f"  Removed: {cache_dir.relative_to(Path(__file__).parent.parent)}")
    except Exception as e:
        print(f"  Warning: Could not remove {cache_dir}: {e}")

# Also remove .pyc files
pyc_files = list(Path(__file__).parent.parent.rglob("*.pyc"))
removed_files = 0
for pyc_file in pyc_files:
    try:
        pyc_file.unlink()
        removed_files += 1
        print(f"  Removed: {pyc_file.relative_to(Path(__file__).parent.parent)}")
    except Exception as e:
        print(f"  Warning: Could not remove {pyc_file}: {e}")

print()
print(f"✓ Removed {removed_dirs} cache directories and {removed_files} .pyc files")
print()
print("=" * 60)
print("✓ Verification complete!")
print("=" * 60)
print()
print("The fix is active. Now try running your collection again:")
print("  python start.py")
print()

