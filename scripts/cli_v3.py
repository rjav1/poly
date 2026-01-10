#!/usr/bin/env python3
"""
V3 Data Collection CLI - RTDS-based high-quality data collection.

This is the new standard for data collection, using:
- Polymarket RTDS websocket for real-time Chainlink prices (no 1-min delay!)
- Polymarket API for orderbook data (6 levels of depth)
- Perfect timestamp alignment
- Near 100% coverage

Usage:
    python scripts/cli_v3.py collect --assets ETH --duration 900
    python scripts/cli_v3.py collect --assets ETH --target 900
    python scripts/cli_v3.py process
    python scripts/cli_v3.py build
    python scripts/cli_v3.py test-rtds --asset ETH
"""

import sys
import os
import asyncio
import time
import signal
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure stdout for Windows Unicode support
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

# Rich imports with fallback
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from config.settings import SUPPORTED_ASSETS, STORAGE


class V3Dashboard:
    """Real-time dashboard for V3 data collection."""
    
    def __init__(self, assets: List[str], target_points: Optional[int] = None):
        self.assets = assets
        self.target_points = target_points  # None means indefinite
        self.start_time = time.time()
        
        # Stats per asset
        self.stats = {asset: {
            'cl_points': 0,
            'pm_points': 0,
            'matched': 0,
            'cl_price': None,
            'status': 'Initializing'
        } for asset in assets}
        
        self.console = Console(force_terminal=True, legacy_windows=True) if RICH_AVAILABLE else None
        self.message = ""
    
    def update_stats(self, stats: Dict):
        """Update stats from V3 collector."""
        for asset, s in stats.items():
            if asset in self.stats:
                self.stats[asset]['cl_points'] = s.cl_points
                self.stats[asset]['pm_points'] = s.pm_points
                self.stats[asset]['matched'] = s.matched_points
                self.stats[asset]['cl_price'] = s.cl_price
                self.stats[asset]['status'] = s.status
    
    def render(self) -> str:
        """Render dashboard as string."""
        elapsed = time.time() - self.start_time
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        
        lines = []
        lines.append("=" * 80)
        lines.append(f"  V3 Data Collection (RTDS)  |  Elapsed: {mins}m {secs}s")
        lines.append("=" * 80)
        lines.append("")
        
        # Progress
        total_matched = sum(s['matched'] for s in self.stats.values())
        
        if self.target_points is None:
            # Indefinite mode
            lines.append(f"  Mode: [INDEFINITE - Press Ctrl+C to stop]")
            lines.append(f"  Matched: {total_matched} points")
            lines.append("")
            
            # Per-asset stats (no target column)
            lines.append(f"  {'Asset':<6} {'Matched':>8} {'CL Pts':>8} {'PM Pts':>8} {'CL Price':>12} {'Status':<20}")
            lines.append("  " + "-" * 70)
            
            for asset in self.assets:
                s = self.stats[asset]
                price_str = f"${s['cl_price']:,.2f}" if s['cl_price'] else "N/A"
                lines.append(
                    f"  {asset:<6} {s['matched']:>8} "
                    f"{s['cl_points']:>8} {s['pm_points']:>8} {price_str:>12} {s['status']:<20}"
                )
        else:
            # Target-based mode
            total_target = self.target_points * len(self.assets)
            pct = (total_matched / total_target * 100) if total_target > 0 else 0
            
            bar_len = 50
            filled = int(pct / 100 * bar_len)
            bar = '=' * filled + '-' * (bar_len - filled)
            
            lines.append(f"  Progress: [{bar}] {pct:.1f}%")
            lines.append(f"  Matched: {total_matched}/{total_target} points")
            lines.append("")
            
            # Per-asset stats
            lines.append(f"  {'Asset':<6} {'Matched':>8} {'Target':>8} {'CL Pts':>8} {'PM Pts':>8} {'CL Price':>12} {'Status':<20}")
            lines.append("  " + "-" * 78)
            
            for asset in self.assets:
                s = self.stats[asset]
                price_str = f"${s['cl_price']:,.2f}" if s['cl_price'] else "N/A"
                lines.append(
                    f"  {asset:<6} {s['matched']:>8} {self.target_points:>8} "
                    f"{s['cl_points']:>8} {s['pm_points']:>8} {price_str:>12} {s['status']:<20}"
                )
        
        lines.append("")
        if self.message:
            lines.append(f"  {self.message}")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def display(self):
        """Display the dashboard."""
        # Clear screen
        if sys.platform == "win32":
            os.system('cls')
        else:
            os.system('clear')
        print(self.render())


class V3CLI:
    """CLI for V3 data collection."""
    
    def __init__(self):
        self.console = Console(force_terminal=True, legacy_windows=True) if RICH_AVAILABLE else None
        self.shutdown_requested = False
    
    def print_header(self, title: str, subtitle: str = ""):
        """Print a header."""
        if self.console and RICH_AVAILABLE:
            text = f"[bold]{title}[/bold]"
            if subtitle:
                text += f"\n{subtitle}"
            self.console.print(Panel(text, border_style="cyan"))
        else:
            print("=" * 70)
            print(f"  {title}")
            if subtitle:
                print(f"  {subtitle}")
            print("=" * 70)
    
    def print_config(self, config: Dict):
        """Print configuration."""
        if self.console and RICH_AVAILABLE:
            table = Table(title="Configuration", box=box.ROUNDED)
            table.add_column("Setting", style="cyan")
            table.add_column("Value", style="green")
            for key, value in config.items():
                table.add_row(key, str(value))
            self.console.print(table)
        else:
            print("\nConfiguration:")
            for key, value in config.items():
                print(f"  {key}: {value}")
            print()
    
    async def run_collection(
        self,
        assets: List[str],
        duration: Optional[int] = None,
        target_points: Optional[int] = None,
        output_dir: str = "data_v2/raw",
    ):
        """Run V3 data collection."""
        # Setup signal handler
        def signal_handler(sig, frame):
            self.shutdown_requested = True
        
        signal.signal(signal.SIGINT, signal_handler)
        
        # Import V3 collector
        from src.collector_v3 import V3Collector
        
        # Determine target
        if target_points:
            effective_duration = target_points * 1.1  # 10% buffer
            display_target = target_points
        elif duration is not None:
            if duration == 0:
                # 0 means indefinite
                effective_duration = None
                display_target = None  # Will show as "indefinite"
            else:
                effective_duration = duration
                display_target = duration  # ~1 point per second
        else:
            # duration is None and no target_points - default to 15 min
            # (This is the default case when neither is specified)
            effective_duration = 900
            display_target = 900
        
        # Print config
        self.print_header(
            "V3 Data Collection (RTDS)",
            "Real-time Chainlink + Polymarket orderbook"
        )
        
        if display_target is None:
            # Indefinite mode
            config = {
                "Mode": "RTDS Websocket (real-time, no delay)",
                "Assets": ", ".join(assets),
                "Duration": "Indefinite (until Ctrl+C)",
                "Output": output_dir,
            }
        else:
            config = {
                "Mode": "RTDS Websocket (real-time, no delay)",
                "Assets": ", ".join(assets),
                "Target Points": f"{display_target} per asset" if target_points else "N/A",
                "Duration": f"{effective_duration:.0f}s" if effective_duration else "Indefinite",
                "Output": output_dir,
            }
        self.print_config(config)
        
        print("\nAdvantages over V2:")
        print("  ✓ Real-time Chainlink prices (no 1-minute delay)")
        print("  ✓ Perfect timestamp alignment")
        print("  ✓ Near 100% coverage")
        print("  ✓ Lower resource usage")
        print()
        if display_target is None:
            print("Starting in 2 seconds... (Press Ctrl+C to stop)")
        else:
            print("Starting in 2 seconds... (Press Ctrl+C to stop)")
        await asyncio.sleep(2)
        
        # Create collector
        collector = V3Collector(
            assets=assets,
            output_dir=output_dir,
            log_level=30  # WARNING only
        )
        
        # Create dashboard (pass None for indefinite mode)
        dashboard = V3Dashboard(assets, display_target if display_target is not None else None)
        
        # Start collector in background
        collector_done = False
        
        async def run_collector():
            nonlocal collector_done
            try:
                await collector.start(duration=effective_duration)
            finally:
                collector_done = True
        
        collector_task = asyncio.create_task(run_collector())
        
        # Dashboard update loop
        try:
            while not collector_done and not self.shutdown_requested:
                await asyncio.sleep(0.5)
                
                # Update dashboard
                dashboard.update_stats(collector.get_stats())
                dashboard.display()
                
                # Check if we hit target (only if not indefinite)
                if target_points:
                    total_matched = sum(s.matched_points for s in collector.get_stats().values())
                    if total_matched >= target_points * len(assets):
                        dashboard.message = "Target reached!"
                        break
        
        except asyncio.CancelledError:
            pass
        
        finally:
            self.shutdown_requested = True
            await collector.stop()
            
            if not collector_task.done():
                collector_task.cancel()
                try:
                    await collector_task
                except asyncio.CancelledError:
                    pass
        
        # Final display
        dashboard.update_stats(collector.get_stats())
        dashboard.message = "Collection complete!"
        dashboard.display()
        
        # Print summary
        print("\n")
        summary = collector.get_summary()
        
        if self.console and RICH_AVAILABLE:
            table = Table(title="Collection Summary", box=box.ROUNDED)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Total CL Points", str(summary['total_cl_points']))
            table.add_row("Total PM Points", str(summary['total_pm_points']))
            table.add_row("Total Matched Points", str(summary['total_matched_points']))
            
            self.console.print(table)
            self.console.print("\n[cyan]Next Step:[/cyan] Run [bold]python scripts/cli_v3.py process[/bold] to create market folders")
        else:
            print("Collection Summary:")
            print(f"  Total CL Points: {summary['total_cl_points']}")
            print(f"  Total PM Points: {summary['total_pm_points']}")
            print(f"  Total Matched Points: {summary['total_matched_points']}")
            print("\nNext Step: Run 'python scripts/cli_v3.py process' to create market folders")
        
        return 0
    
    async def test_rtds(self, asset: str, duration: int = 30):
        """Test RTDS Chainlink stream."""
        self.print_header(
            "RTDS Chainlink Test",
            f"Testing {asset} via Polymarket RTDS websocket"
        )
        
        print()
        print("This uses the SAME Chainlink stream Polymarket uses for resolution.")
        print("Updates come in real-time via websocket (not polling).")
        print()
        
        try:
            from paper_trading.chainlink_rtds import RTDSChainlinkCollector
        except ImportError as e:
            print(f"Error: {e}")
            print("\nMake sure websockets is installed: pip install websockets")
            return 1
        
        collector = RTDSChainlinkCollector(asset=asset)
        
        print(f"Asset: {collector.asset}")
        print(f"Symbol: {collector.symbol}")
        print(f"Endpoint: {collector.endpoint}")
        print()
        print(f"Streaming for {duration} seconds...")
        print()
        
        header = f"{'#':>4} | {'Price':>12} | {'CL Timestamp':>20} | {'Local Time':>12} | {'Stale':>7}"
        print(f"  {header}")
        print(f"  {'-' * len(header)}")
        
        count = 0
        
        def on_update(price_data):
            nonlocal count
            count += 1
            cl_ts = price_data.chainlink_timestamp.strftime("%Y-%m-%d %H:%M:%S")
            local_ts = price_data.local_timestamp.strftime("%H:%M:%S.%f")[:12]
            
            print(
                f"  {count:>4} | ${price_data.price:>11,.2f} | "
                f"{cl_ts:>20} | {local_ts:>12} | "
                f"{price_data.staleness_seconds:>5.1f}s"
            )
        
        try:
            async for _ in collector.stream_prices(duration=duration, callback=on_update):
                await asyncio.sleep(0.1)
        except Exception as e:
            print(f"\nError: {e}")
            return 1
        finally:
            await collector.stop()
        
        print()
        print(f"Received {count} price updates in {duration}s")
        print("[OK] RTDS stream working!")
        return 0
    
    def run_process(self, args):
        """Run data processing."""
        import subprocess
        
        self.print_header(
            "Process Raw Data",
            "Converting continuous data into market folders"
        )
        
        config = {
            "Raw Directory": args.raw_dir or STORAGE.raw_dir,
            "Markets Directory": args.markets_dir or STORAGE.markets_dir,
            "Min Coverage": f"{args.min_coverage}%"
        }
        self.print_config(config)
        
        print("\nProcessing...\n")
        
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "process_raw_data.py")
        ]
        
        if args.raw_dir:
            cmd.extend(["--raw-dir", args.raw_dir])
        if args.markets_dir:
            cmd.extend(["--markets-dir", args.markets_dir])
        if args.min_coverage != 70.0:
            cmd.extend(["--min-coverage", str(args.min_coverage)])
        
        try:
            subprocess.run(cmd, check=True)
            print("\n[OK] Processing complete!")
            return 0
        except subprocess.CalledProcessError:
            print("\n[ERROR] Processing failed")
            return 1
    
    def run_build(self, args):
        """Run dataset building."""
        import subprocess
        
        self.print_header(
            "Build Research Dataset",
            "Creating canonical dataset from market folders"
        )
        
        config = {
            "Markets Directory": args.markets_dir or STORAGE.markets_dir,
            "Output Directory": args.output_dir or STORAGE.research_dir,
            "Min Coverage": f"{args.min_coverage}%"
        }
        self.print_config(config)
        
        print("\nBuilding dataset...\n")
        
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "build_6level_dataset.py"),
            "--min-coverage", str(args.min_coverage)
        ]
        
        if args.output_dir:
            cmd.extend(["--output-dir", args.output_dir])
        
        try:
            subprocess.run(cmd, check=True)
            print("\n[OK] Build complete!")
            return 0
        except subprocess.CalledProcessError:
            print("\n[ERROR] Build failed")
            return 1


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="V3 Data Collection CLI (RTDS-based)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
V3 uses Polymarket RTDS websocket for real-time Chainlink prices.
This is the SAME source Polymarket uses for resolution.

Advantages over V2:
  - Real-time Chainlink prices (no 1-minute delay)
  - Perfect timestamp alignment
  - Near 100% coverage
  - Lower resource usage

Examples:
  # Collect 900 matched points (15-min market worth)
  python scripts/cli_v3.py collect --assets ETH --target 900
  
  # Collect for 60 seconds
  python scripts/cli_v3.py collect --assets ETH --duration 60
  
  # Test RTDS stream
  python scripts/cli_v3.py test-rtds --asset ETH
  
  # Process raw data
  python scripts/cli_v3.py process
  
  # Build dataset
  python scripts/cli_v3.py build
"""
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Collect command
    collect_parser = subparsers.add_parser('collect', help='Collect data using RTDS')
    collect_parser.add_argument(
        '--assets', '-a',
        type=str,
        default='ETH',
        help='Comma-separated assets (default: ETH)'
    )
    collect_parser.add_argument(
        '--target', '-t',
        type=int,
        default=None,
        help='Target matched points per asset'
    )
    collect_parser.add_argument(
        '--duration', '-d',
        type=int,
        default=None,
        help='Duration in seconds (0 = indefinite, until Ctrl+C)'
    )
    collect_parser.add_argument(
        '--output', '-o',
        type=str,
        default='data_v2/raw',
        help='Output directory'
    )
    
    # Test RTDS command
    test_parser = subparsers.add_parser('test-rtds', help='Test RTDS Chainlink stream')
    test_parser.add_argument(
        '--asset', '-a',
        type=str,
        default='ETH',
        help='Asset to test (default: ETH)'
    )
    test_parser.add_argument(
        '--duration', '-d',
        type=int,
        default=30,
        help='Duration in seconds (default: 30)'
    )
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process raw data into market folders')
    process_parser.add_argument(
        '--raw-dir',
        type=str,
        default=None,
        help='Raw data directory'
    )
    process_parser.add_argument(
        '--markets-dir',
        type=str,
        default=None,
        help='Output markets directory'
    )
    process_parser.add_argument(
        '--min-coverage',
        type=float,
        default=70.0,
        help='Minimum coverage percentage'
    )
    
    # Build command
    build_parser = subparsers.add_parser('build', help='Build research dataset')
    build_parser.add_argument(
        '--markets-dir',
        type=str,
        default=None,
        help='Markets directory'
    )
    build_parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory'
    )
    build_parser.add_argument(
        '--min-coverage',
        type=float,
        default=80.0,
        help='Minimum coverage percentage'
    )
    
    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()
    cli = V3CLI()
    
    if args.command == 'collect':
        # Parse assets
        assets = [a.strip().upper() for a in args.assets.split(',')]
        for asset in assets:
            if asset not in SUPPORTED_ASSETS:
                print(f"Error: Unknown asset '{asset}'. Supported: {list(SUPPORTED_ASSETS)}")
                return 1
        
        # Pass duration as-is (0 means indefinite, will be handled in run_collection)
        return await cli.run_collection(
            assets=assets,
            duration=args.duration,
            target_points=args.target,
            output_dir=args.output,
        )
    
    elif args.command == 'test-rtds':
        return await cli.test_rtds(
            asset=args.asset.upper(),
            duration=args.duration,
        )
    
    elif args.command == 'process':
        return cli.run_process(args)
    
    elif args.command == 'build':
        return cli.run_build(args)
    
    else:
        # No command - show help
        print("V3 Data Collection CLI (RTDS-based)")
        print()
        print("This is the NEW STANDARD for data collection.")
        print("Uses Polymarket RTDS websocket for real-time Chainlink prices.")
        print()
        print("Advantages over V2:")
        print("  ✓ Real-time Chainlink prices (no 1-minute delay)")
        print("  ✓ Perfect timestamp alignment")
        print("  ✓ Near 100% coverage")
        print("  ✓ Lower resource usage")
        print()
        print("Commands:")
        print("  collect     Collect data using RTDS")
        print("  test-rtds   Test RTDS Chainlink stream")
        print("  process     Process raw data into market folders")
        print("  build       Build research dataset")
        print()
        print("Examples:")
        print("  python scripts/cli_v3.py collect --assets ETH --target 900")
        print("  python scripts/cli_v3.py test-rtds --asset ETH")
        print("  python scripts/cli_v3.py process")
        print("  python scripts/cli_v3.py build")
        print()
        return 0


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=ResourceWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code or 0)
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

