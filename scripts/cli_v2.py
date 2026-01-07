#!/usr/bin/env python3
"""
Polymarket/Chainlink Data Collection CLI v2
Clean, robust implementation with real-time dashboard.
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
    from rich.layout import Layout
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from config.settings import SUPPORTED_ASSETS, STORAGE


class CollectionDashboard:
    """Real-time dashboard for data collection progress."""
    
    def __init__(self, assets: List[str], target_points: int, lightweight: bool = False):
        self.assets = assets
        self.target_points = target_points
        self.lightweight = lightweight  # Lightweight mode = no real-time matching
        self.target_seconds = int(target_points * 1.2) if lightweight else target_points
        self.start_time = time.time()
        
        # Stats per asset
        self.stats = {asset: {
            'cl_points': 0,
            'pm_points': 0,
            'matched': 0,
            'cl_price': None,
            'cl_delay': None,
            'elapsed': 0,
            'status': 'Initializing'
        } for asset in assets}
        
        self.console = Console(force_terminal=True, legacy_windows=True) if RICH_AVAILABLE else None
        self.running = True
        self.message = ""
    
    def update_stats_lightweight(self, stats: Dict):
        """Update stats from RawCollector (lightweight mode)."""
        for asset, s in stats.items():
            if asset in self.stats:
                self.stats[asset]['cl_points'] = s.cl_points
                self.stats[asset]['pm_points'] = s.pm_points
                self.stats[asset]['elapsed'] = s.elapsed_seconds
                # In lightweight mode, we don't know matched count until post-processing
                self.stats[asset]['matched'] = '?'
                
                # Status based on collection progress
                if s.cl_points > 0 and s.pm_points > 0:
                    self.stats[asset]['status'] = 'Collecting'
                elif s.pm_points > 0:
                    self.stats[asset]['status'] = 'Waiting for CL'
                elif s.cl_points > 0:
                    self.stats[asset]['status'] = 'Waiting for PM'
                else:
                    self.stats[asset]['status'] = 'Starting'
    
    def update_stats(self, progress: Dict):
        """Update stats from collector progress (legacy MatchingCollector mode)."""
        for asset, p in progress.items():
            if asset in self.stats:
                self.stats[asset]['cl_points'] = p.get('cl_points', 0)
                self.stats[asset]['pm_points'] = p.get('pm_points', 0)
                self.stats[asset]['matched'] = p.get('matched', 0)
                self.stats[asset]['cl_price'] = p.get('cl_price')
                self.stats[asset]['cl_delay'] = p.get('cl_delay_seconds')
                
                # Update status based on data
                cl = p.get('cl_points', 0)
                pm = p.get('pm_points', 0)
                matched = p.get('matched', 0)
                
                if matched > 0:
                    self.stats[asset]['status'] = 'Matching'
                elif cl > 0 and pm > 0:
                    delay = p.get('cl_delay_seconds')
                    if delay and delay > 60:
                        self.stats[asset]['status'] = f'Waiting (CL {int(delay)}s behind)'
                    else:
                        self.stats[asset]['status'] = 'Collecting'
                elif pm > 0:
                    self.stats[asset]['status'] = 'Waiting for CL'
                elif cl > 0:
                    self.stats[asset]['status'] = 'Waiting for PM'
                else:
                    self.stats[asset]['status'] = 'Starting'
    
    def render(self) -> str:
        """Render dashboard as string (for non-rich fallback)."""
        elapsed = time.time() - self.start_time
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        
        lines = []
        mode_str = "[LIGHTWEIGHT]" if self.lightweight else "[MATCHING]"
        lines.append("=" * 70)
        lines.append(f"  Data Collection Dashboard {mode_str}  |  Elapsed: {mins}m {secs}s")
        lines.append("=" * 70)
        lines.append("")
        
        if self.lightweight:
            # Lightweight mode - progress by time, not matched count
            pct = min(100, (elapsed / self.target_seconds * 100)) if self.target_seconds > 0 else 0
            bar_len = 40
            filled = int(pct / 100 * bar_len)
            bar = '#' * filled + '-' * (bar_len - filled)
            
            lines.append(f"  Progress: [{bar}] {pct:.1f}%")
            lines.append(f"  Time: {int(elapsed)}/{self.target_seconds}s (matching done after collection)")
            lines.append("")
            
            # Per-asset stats
            lines.append(f"  {'Asset':<6} {'CL Pts':>8} {'PM Pts':>8} {'Status':<30}")
            lines.append("  " + "-" * 60)
            
            for asset in self.assets:
                s = self.stats[asset]
                lines.append(
                    f"  {asset:<6} {s['cl_points']:>8} {s['pm_points']:>8} {s['status']:<30}"
                )
        else:
            # Legacy matching mode - progress by matched count
            total_matched = sum(s['matched'] for s in self.stats.values() if isinstance(s['matched'], int))
            total_target = self.target_points * len(self.assets)
            pct = (total_matched / total_target * 100) if total_target > 0 else 0
            
            bar_len = 40
            filled = int(pct / 100 * bar_len)
            bar = '#' * filled + '-' * (bar_len - filled)
            
            lines.append(f"  Progress: [{bar}] {pct:.1f}%")
            lines.append(f"  Matched: {total_matched}/{total_target} points")
            lines.append("")
            
            # Per-asset stats
            lines.append(f"  {'Asset':<6} {'Matched':>8} {'Target':>8} {'CL Pts':>8} {'PM Pts':>8} {'CL Price':>12} {'Status':<25}")
            lines.append("  " + "-" * 85)
            
            for asset in self.assets:
                s = self.stats[asset]
                price_str = f"${s['cl_price']:,.2f}" if s['cl_price'] else "N/A"
                lines.append(
                    f"  {asset:<6} {s['matched']:>8} {self.target_points:>8} "
                    f"{s['cl_points']:>8} {s['pm_points']:>8} {price_str:>12} {s['status']:<25}"
                )
        
        lines.append("")
        if self.message:
            lines.append(f"  {self.message}")
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def render_rich(self):
        """Render dashboard using Rich library."""
        elapsed = time.time() - self.start_time
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        
        total_cl = sum(s['cl_points'] for s in self.stats.values())
        total_pm = sum(s['pm_points'] for s in self.stats.values())
        
        if self.lightweight:
            # Lightweight mode - progress by time
            pct = min(100, (elapsed / self.target_seconds * 100)) if self.target_seconds > 0 else 0
            bar_len = 50
            filled = int(pct / 100 * bar_len)
            bar = '=' * filled + '-' * (bar_len - filled)
            
            progress_text = f"[{bar}] {pct:.1f}%\n"
            progress_text += f"Time: {int(elapsed)}/{self.target_seconds}s | CL: {total_cl} pts | PM: {total_pm} pts"
            progress_text += f"\n[dim]Matching will be done post-collection by process_raw_data.py[/dim]"
            
            progress_panel = Panel(
                progress_text,
                title="[bold cyan]Lightweight Collection (No Real-Time Matching)[/bold cyan]",
                border_style="cyan"
            )
            
            # Simplified stats table for lightweight mode
            table = Table(title="Real-Time Statistics", box=box.ROUNDED)
            table.add_column("Asset", style="cyan", width=6)
            table.add_column("CL Pts", style="blue", justify="right", width=10)
            table.add_column("PM Pts", style="magenta", justify="right", width=10)
            table.add_column("Status", style="white", width=30)
            
            for asset in self.assets:
                s = self.stats[asset]
                status = s['status']
                if 'Collecting' in status:
                    status_str = f"[green]{status}[/green]"
                elif 'Waiting' in status:
                    status_str = f"[yellow]{status}[/yellow]"
                else:
                    status_str = f"[dim]{status}[/dim]"
                
                table.add_row(
                    asset,
                    str(s['cl_points']),
                    str(s['pm_points']),
                    status_str
                )
        else:
            # Legacy matching mode
            total_matched = sum(s['matched'] for s in self.stats.values() if isinstance(s['matched'], int))
            total_target = self.target_points * len(self.assets)
            pct = (total_matched / total_target * 100) if total_target > 0 else 0
            
            bar_len = 50
            filled = int(pct / 100 * bar_len)
            bar = '=' * filled + '-' * (bar_len - filled)
            
            progress_text = f"[{bar}] {pct:.1f}%\n"
            progress_text += f"Matched: {total_matched}/{total_target} | CL: {total_cl} pts | PM: {total_pm} pts | Elapsed: {mins}m {secs}s"
            
            progress_panel = Panel(
                progress_text,
                title="[bold cyan]Collection Progress[/bold cyan]",
                border_style="cyan"
            )
            
            # Full stats table
            table = Table(title="Real-Time Statistics", box=box.ROUNDED)
            table.add_column("Asset", style="cyan", width=6)
            table.add_column("Matched", style="green", justify="right", width=8)
            table.add_column("Target", style="yellow", justify="right", width=8)
            table.add_column("CL Pts", style="blue", justify="right", width=8)
            table.add_column("PM Pts", style="magenta", justify="right", width=8)
            table.add_column("CL Price", style="dim", justify="right", width=12)
            table.add_column("Status", style="white", width=28)
            
            for asset in self.assets:
                s = self.stats[asset]
                price_str = f"${s['cl_price']:,.2f}" if s['cl_price'] else "N/A"
                
                status = s['status']
                if 'Matching' in status:
                    status_str = f"[green]{status}[/green]"
                elif 'Waiting' in status:
                    status_str = f"[yellow]{status}[/yellow]"
                else:
                    status_str = f"[dim]{status}[/dim]"
                
                table.add_row(
                    asset,
                    str(s['matched']),
                    str(self.target_points),
                    str(s['cl_points']),
                    str(s['pm_points']),
                    price_str,
                    status_str
                )
        
        return progress_panel, table
    
    def display(self):
        """Display the dashboard."""
        if RICH_AVAILABLE and self.console:
            try:
                # Clear screen
                if sys.platform == "win32":
                    os.system('cls')
                else:
                    os.system('clear')
                
                progress_panel, table = self.render_rich()
                self.console.print(progress_panel)
                self.console.print(table)
                if self.message:
                    self.console.print(f"\n[dim]{self.message}[/dim]")
            except Exception as e:
                # Fallback to plain text
                print(self.render())
        else:
            # Clear screen
            if sys.platform == "win32":
                os.system('cls')
            else:
                os.system('clear')
            print(self.render())


class DataCollectionCLI:
    """Main CLI for data collection."""
    
    def __init__(self):
        self.console = Console(force_terminal=True, legacy_windows=True) if RICH_AVAILABLE else None
        self.shutdown_requested = False
    
    def print_header(self, title: str, subtitle: str = ""):
        """Print a header."""
        if self.console and RICH_AVAILABLE:
            self.console.print(Panel(
                f"{title}\n{subtitle}" if subtitle else title,
                border_style="cyan"
            ))
        else:
            print("=" * 60)
            print(f"  {title}")
            if subtitle:
                print(f"  {subtitle}")
            print("=" * 60)
    
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
        duration: int,
        interval: float = 1.0,
        max_time: Optional[int] = None,
        output_dir: str = "data_v2/raw",
        lightweight: bool = True  # Default to lightweight mode
    ):
        """Run data collection with real-time dashboard.
        
        Args:
            lightweight: If True (default), use lightweight collector with post-processing matching.
                        This is more accurate as it doesn't add matching overhead during collection.
        """
        # Setup signal handler
        def signal_handler(sig, frame):
            self.shutdown_requested = True
        
        signal.signal(signal.SIGINT, signal_handler)
        
        # Null stream for suppressing output during dashboard
        class NullWriter:
            def write(self, s): pass
            def flush(self): pass
        
        null_writer = NullWriter()
        original_stdout = sys.stdout
        
        # Create dashboard
        dashboard = CollectionDashboard(assets, duration, lightweight=lightweight)
        
        # Print initial config
        mode_str = "LIGHTWEIGHT (recommended)" if lightweight else "REAL-TIME MATCHING (legacy)"
        self.print_header(
            f"Data Collection - {mode_str}",
            f"Collecting data for {', '.join(assets)}"
        )
        
        if lightweight:
            target_seconds = int(duration * 1.2)  # 1.2x buffer
            config = {
                "Mode": "Lightweight (collect first, match later)",
                "Assets": ", ".join(assets),
                "Target Matched Points": f"{duration} per asset",
                "Collection Duration": f"{target_seconds}s (1.2x buffer)",
                "Collection Interval": f"{interval}s",
                "Output Directory": output_dir,
                "Post-Processing": "Run 'cli_v2.py process' after collection"
            }
        else:
            config = {
                "Mode": "Real-time matching (legacy)",
                "Assets": ", ".join(assets),
                "Target Matched Points": f"{duration} per asset",
                "Collection Interval": f"{interval}s",
                "Max Wall Time": f"{max_time}s" if max_time else "None",
                "Output Directory": output_dir
            }
        self.print_config(config)
        
        print("\nStarting collection...")
        print("Press Ctrl+C to stop gracefully\n")
        
        collector_done = False
        
        if lightweight:
            # Use lightweight RawCollector
            from src.raw_collector import RawCollector
            
            collector = RawCollector(
                assets=assets,
                target_matched=duration,
                output_dir=output_dir,
                log_level=30,  # WARNING level
                sequential_cl=True  # More reliable
            )
            
            async def run_collector():
                nonlocal collector_done
                try:
                    await collector.start(interval=interval)
                finally:
                    collector_done = True
            
            # Start collector in background
            collector_task = asyncio.create_task(run_collector())
            
            # Dashboard update loop
            try:
                while not collector_done and not self.shutdown_requested:
                    sys.stdout = null_writer
                    await asyncio.sleep(0.4)
                    sys.stdout = original_stdout
                    
                    # Get current stats from RawCollector
                    stats = collector.get_stats()
                    dashboard.update_stats_lightweight(stats)
                    dashboard.display()
            
            except asyncio.CancelledError:
                pass
            
            finally:
                sys.stdout = original_stdout
                self.shutdown_requested = True
                await collector.stop()
                
                if not collector_task.done():
                    collector_task.cancel()
                    try:
                        await collector_task
                    except asyncio.CancelledError:
                        pass
            
            # Final display
            stats = collector.get_stats()
            dashboard.update_stats_lightweight(stats)
            dashboard.message = "Collection complete! Now run 'cli_v2.py process' to match data."
            dashboard.display()
            
            # Print lightweight summary
            print("\n")
            self.print_lightweight_summary(stats, duration, dashboard.start_time)
            
        else:
            # Use legacy MatchingCollector
            from src.matching_collector import MatchingCollector
            
            collector = MatchingCollector(
                assets=assets,
                output_dir=output_dir,
                log_level=30,
                sequential_cl_collection=True
            )
            
            async def run_collector():
                nonlocal collector_done
                try:
                    await collector.start(
                        target_matched_points=duration,
                        interval=interval,
                        max_wall_time=max_time
                    )
                finally:
                    collector_done = True
            
            collector_task = asyncio.create_task(run_collector())
            
            try:
                while not collector_done and not self.shutdown_requested:
                    sys.stdout = null_writer
                    await asyncio.sleep(0.4)
                    sys.stdout = original_stdout
                    
                    progress = collector.get_progress()
                    dashboard.update_stats(progress)
                    dashboard.display()
            
            except asyncio.CancelledError:
                pass
            
            finally:
                sys.stdout = original_stdout
                self.shutdown_requested = True
                await collector.stop()
                
                if not collector_task.done():
                    collector_task.cancel()
                    try:
                        await collector_task
                    except asyncio.CancelledError:
                        pass
            
            # Final display
            progress = collector.get_progress()
            dashboard.update_stats(progress)
            dashboard.message = "Collection complete!"
            dashboard.display()
            
            print("\n")
            self.print_summary(progress, duration, dashboard.start_time)
        
        # Suppress asyncio cleanup errors on Windows
        import gc
        gc.collect()
        
        return 0
    
    def print_lightweight_summary(self, stats: Dict, target: int, start_time: float):
        """Print collection summary for lightweight mode."""
        elapsed = time.time() - start_time
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        
        total_cl = sum(s.cl_points for s in stats.values())
        total_pm = sum(s.pm_points for s in stats.values())
        
        if self.console and RICH_AVAILABLE:
            table = Table(title="Collection Summary (Lightweight Mode)", box=box.ROUNDED)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Total CL Points", str(total_cl))
            table.add_row("Total PM Points", str(total_pm))
            table.add_row("Duration", f"{mins}m {secs}s")
            table.add_row("Target per Asset", str(target))
            table.add_row("Matched Points", "[yellow]Run 'process' to calculate[/yellow]")
            
            self.console.print(table)
            self.console.print("\n[cyan]Next Step:[/cyan] Run [bold]python scripts/cli_v2.py process[/bold] to match and create datasets")
        else:
            print("\nCollection Summary (Lightweight Mode):")
            print(f"  Total CL Points: {total_cl}")
            print(f"  Total PM Points: {total_pm}")
            print(f"  Duration: {mins}m {secs}s")
            print(f"  Target per Asset: {target}")
            print(f"  Matched Points: Run 'process' to calculate")
            print("\nNext Step: Run 'python scripts/cli_v2.py process' to match and create datasets")
    
    def print_summary(self, progress: Dict, target: int, start_time: float):
        """Print collection summary."""
        elapsed = time.time() - start_time
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        
        total_matched = sum(p.get('matched', 0) for p in progress.values())
        total_cl = sum(p.get('cl_points', 0) for p in progress.values())
        total_pm = sum(p.get('pm_points', 0) for p in progress.values())
        
        if self.console and RICH_AVAILABLE:
            table = Table(title="Collection Summary", box=box.ROUNDED)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Total Matched Points", str(total_matched))
            table.add_row("Total CL Points", str(total_cl))
            table.add_row("Total PM Points", str(total_pm))
            table.add_row("Duration", f"{mins}m {secs}s")
            table.add_row("Target per Asset", str(target))
            
            self.console.print(table)
            
            if total_matched >= target * len(progress):
                self.console.print("\n[green][OK] Collection completed successfully![/green]")
            else:
                self.console.print(f"\n[yellow][WARN] Collection ended before target reached[/yellow]")
        else:
            print("\nCollection Summary:")
            print(f"  Total Matched Points: {total_matched}")
            print(f"  Total CL Points: {total_cl}")
            print(f"  Total PM Points: {total_pm}")
            print(f"  Duration: {mins}m {secs}s")
            print(f"  Target per Asset: {target}")
    
    def run_build(self, args):
        """Run dataset building (uses 6-level data by default)."""
        import subprocess
        
        self.print_header(
            "Build Research Dataset",
            "Processing collected markets into canonical format (6-level data)"
        )
        
        # Use build_6level_dataset.py which handles 6-level markets and research_6levels output
        # Default to 80% coverage for 6-level data quality
        coverage = args.min_coverage  # Already defaults to 80.0 from argparse
        
        config = {
            "Markets Directory": args.markets_dir or STORAGE.markets_dir,
            "Output Directory": args.output_dir or STORAGE.research_dir,
            "Minimum Coverage": f"{coverage}%",
            "Ground Truth": "Yes" if args.use_ground_truth else "No"
        }
        self.print_config(config)
        
        print("\nBuilding dataset...\n")
        
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "build_6level_dataset.py")
        ]
        
        # build_6level_dataset.py uses markets_6levels by default, but allow override
        if args.markets_dir:
            # If custom markets_dir, use build_research_dataset_v2.py instead
            cmd = [
                sys.executable,
                str(Path(__file__).parent / "build_research_dataset_v2.py")
            ]
            cmd.extend(["--markets-dir", args.markets_dir])
            if args.min_coverage != 80.0:  # Only pass if different from default
                cmd.extend(["--min-coverage", str(args.min_coverage)])
        else:
            # Use build_6level_dataset.py with coverage (it defaults to 80% but we pass it anyway)
            if args.min_coverage != 80.0:  # Only pass if different from default
                cmd.extend(["--min-coverage", str(args.min_coverage)])
        
        if args.output_dir:
            cmd.extend(["--output-dir", args.output_dir])
        if args.use_ground_truth:
            cmd.append("--use-ground-truth")
        
        try:
            subprocess.run(cmd, check=True)
            if self.console and RICH_AVAILABLE:
                self.console.print("\n[green][OK] Dataset building completed![/green]")
            else:
                print("\n[OK] Dataset building completed!")
            return 0
        except subprocess.CalledProcessError as e:
            if self.console and RICH_AVAILABLE:
                self.console.print("\n[red][ERROR] Dataset building failed[/red]")
            else:
                print("\n[ERROR] Dataset building failed")
            return 1
        except KeyboardInterrupt:
            if self.console and RICH_AVAILABLE:
                self.console.print("\n[yellow][WARN] Build interrupted by user[/yellow]")
            else:
                print("\n[WARN] Build interrupted by user")
            return 1
    
    def run_validate(self, args):
        """Run dataset validation."""
        import subprocess
        
        self.print_header(
            "Dataset Validation",
            "Comprehensive data quality checks"
        )
        
        config = {
            "Research Directory": args.research_dir or STORAGE.research_dir,
            "Dataset File": args.dataset or "auto-detect",
            "Market Info File": args.market_info or "auto-detect"
        }
        self.print_config(config)
        
        print("\nValidating dataset...\n")
        
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "validate_dataset.py")
        ]
        
        if args.research_dir:
            cmd.extend(["--research-dir", args.research_dir])
        if args.dataset:
            cmd.extend(["--dataset", args.dataset])
        if args.market_info:
            cmd.extend(["--market-info", args.market_info])
        
        try:
            subprocess.run(cmd, check=True)
            if self.console and RICH_AVAILABLE:
                self.console.print("\n[green][OK] Validation completed![/green]")
            else:
                print("\n[OK] Validation completed!")
            return 0
        except subprocess.CalledProcessError as e:
            if self.console and RICH_AVAILABLE:
                self.console.print("\n[red][ERROR] Validation failed[/red]")
            else:
                print("\n[ERROR] Validation failed")
            return 1
        except KeyboardInterrupt:
            if self.console and RICH_AVAILABLE:
                self.console.print("\n[yellow][WARN] Validation interrupted by user[/yellow]")
            else:
                print("\n[WARN] Validation interrupted by user")
            return 1
    
    def run_process(self, args):
        """Run data processing (raw to market folders)."""
        import subprocess
        
        self.print_header(
            "Process Raw Data",
            "Converting continuous data into market folders"
        )
        
        config = {
            "Raw Directory": args.raw_dir or STORAGE.raw_dir,
            "Markets Directory": args.markets_dir or STORAGE.markets_dir,
            "Assets": args.assets or "auto-detect",
            "Min Coverage": f"{args.min_coverage}%"
        }
        self.print_config(config)
        
        print("\nProcessing raw data...\n")
        
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "process_raw_data.py")
        ]
        
        if args.raw_dir:
            cmd.extend(["--raw-dir", args.raw_dir])
        if args.markets_dir:
            cmd.extend(["--markets-dir", args.markets_dir])
        if args.assets:
            cmd.extend(["--assets", args.assets])
        if args.min_coverage != 50.0:
            cmd.extend(["--min-coverage", str(args.min_coverage)])
        
        try:
            subprocess.run(cmd, check=True)
            if self.console and RICH_AVAILABLE:
                self.console.print("\n[green][OK] Processing completed![/green]")
            else:
                print("\n[OK] Processing completed!")
            return 0
        except subprocess.CalledProcessError as e:
            if self.console and RICH_AVAILABLE:
                self.console.print("\n[red][ERROR] Processing failed[/red]")
            else:
                print("\n[ERROR] Processing failed")
            return 1
        except KeyboardInterrupt:
            if self.console and RICH_AVAILABLE:
                self.console.print("\n[yellow][WARN] Processing interrupted by user[/yellow]")
            else:
                print("\n[WARN] Processing interrupted by user")
            return 1


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Polymarket/Chainlink Data Collection CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Collect command
    collect_parser = subparsers.add_parser('collect', help='Collect data from Chainlink and Polymarket')
    collect_parser.add_argument(
        '--assets', '-a',
        type=str,
        default='ETH',
        help='Comma-separated list of assets (default: ETH). Use "all" for all assets.'
    )
    collect_parser.add_argument(
        '--duration', '-d',
        type=int,
        default=900,
        help='Target number of matched data points per asset (default: 900 = 15 min market)'
    )
    collect_parser.add_argument(
        '--interval', '-i',
        type=float,
        default=1.0,
        help='Collection interval in seconds (default: 1.0)'
    )
    collect_parser.add_argument(
        '--max-time', '-m',
        type=int,
        default=None,
        help='Maximum wall-clock time in seconds (only used with --legacy mode)'
    )
    collect_parser.add_argument(
        '--output', '-o',
        type=str,
        default='data_v2/raw',
        help='Output directory (default: data_v2/raw)'
    )
    collect_parser.add_argument(
        '--legacy',
        action='store_true',
        help='Use legacy real-time matching mode (slower, more overhead). '
             'Default is lightweight mode which collects first and matches later.'
    )
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process raw data into market folders')
    process_parser.add_argument(
        '--raw-dir',
        type=str,
        default=STORAGE.raw_dir,
        help=f'Raw data directory (default: {STORAGE.raw_dir})'
    )
    process_parser.add_argument(
        '--markets-dir',
        type=str,
        default=STORAGE.markets_dir,
        help=f'Output markets directory (default: {STORAGE.markets_dir})'
    )
    process_parser.add_argument(
        '--assets',
        type=str,
        default=None,
        help='Comma-separated list of assets (default: all found)'
    )
    process_parser.add_argument(
        '--min-coverage',
        type=float,
        default=70.0,
        help='Minimum coverage percentage for BOTH sources (intersection). Default: 70.0. '
             'Note: Due to timing differences between CL and PM, intersection coverage is typically '
             '10-20%% lower than individual source coverage.'
    )
    
    # Build command
    build_parser = subparsers.add_parser('build', help='Build canonical research dataset from collected markets')
    build_parser.add_argument(
        '--markets-dir',
        type=str,
        default=None,
        help=f'Markets directory (default: {STORAGE.markets_dir})'
    )
    build_parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help=f'Output directory (default: {STORAGE.research_dir})'
    )
    build_parser.add_argument(
        '--min-coverage',
        type=float,
        default=80.0,
        help='Minimum coverage percentage (intersection of CL and PM). Default: 80.0 for 6-level data. '
             'Note: Intersection coverage is typically lower than individual source coverage.'
    )
    build_parser.add_argument(
        '--use-ground-truth',
        action='store_true',
        help='Use ground truth for strike prices'
    )
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate research dataset quality')
    validate_parser.add_argument(
        '--research-dir',
        type=str,
        default=None,
        help=f'Research directory (default: {STORAGE.research_dir})'
    )
    validate_parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Dataset filename (default: auto-detect)'
    )
    validate_parser.add_argument(
        '--market-info',
        type=str,
        default=None,
        help='Market info filename (default: auto-detect)'
    )
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show data collection status')
    
    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()
    cli = DataCollectionCLI()
    
    if args.command == 'collect':
        # Parse assets
        if args.assets.lower() == 'all':
            assets = list(SUPPORTED_ASSETS)
        else:
            assets = [a.strip().upper() for a in args.assets.split(',')]
            # Validate assets
            for asset in assets:
                if asset not in SUPPORTED_ASSETS:
                    print(f"Error: Unknown asset '{asset}'. Supported: {list(SUPPORTED_ASSETS)}")
                    return 1
        
        # Determine mode
        lightweight = not args.legacy
        
        if lightweight:
            # Lightweight mode: collect for duration * 1.2 seconds, match later
            target_seconds = int(args.duration * 1.2)
            print(f"LIGHTWEIGHT MODE: Collecting for {target_seconds}s ({args.duration} target × 1.2 buffer)")
            print(f"Matching will be done post-collection by 'cli_v2.py process'")
            print()
            
            return await cli.run_collection(
                assets=assets,
                duration=args.duration,
                interval=args.interval,
                output_dir=args.output,
                lightweight=True
            )
        else:
            # Legacy mode: real-time matching
            max_time = args.max_time
            if max_time is None:
                init_time = len(assets) * 15
                cl_timestamp_delay = 65
                collection_time = args.duration
                buffer = 60
                max_time = init_time + cl_timestamp_delay + collection_time + buffer
                print(f"LEGACY MODE: Real-time matching")
                print(f"Auto-calculated max_time: {max_time}s")
            
            return await cli.run_collection(
                assets=assets,
                duration=args.duration,
                interval=args.interval,
                max_time=max_time,
                output_dir=args.output,
                lightweight=False
            )
    
    elif args.command == 'process':
        return cli.run_process(args)
    
    elif args.command == 'build':
        return cli.run_build(args)
    
    elif args.command == 'validate':
        return cli.run_validate(args)
    
    elif args.command == 'status':
        cli.print_header("Data Status")
        # Check data directories
        raw_dir = Path("data_v2/raw")
        if raw_dir.exists():
            cl_files = list(raw_dir.glob("chainlink/**/*.csv"))
            pm_files = list(raw_dir.glob("polymarket/**/*.csv"))
            print(f"  Chainlink files: {len(cl_files)}")
            print(f"  Polymarket files: {len(pm_files)}")
        else:
            print("  No data collected yet")
        return 0
    
    else:
        # No command - show help
        print("Polymarket/Chainlink Data Collection CLI")
        print()
        print("LIGHTWEIGHT MODE (default, recommended):")
        print("  - Collects CL and PM data independently for maximum accuracy")
        print("  - No real-time matching overhead during collection")
        print("  - Matching is done post-collection by the 'process' command")
        print()
        print("Full Workflow:")
        print("  1. collect   - Collect data (lightweight mode by default)")
        print("  2. process   - Match CL/PM and create market folders")
        print("  3. build     - Build canonical research dataset")
        print("  4. validate  - Validate research dataset quality")
        print()
        print("Other Commands:")
        print("  status       - Show data collection status")
        print()
        print("Examples:")
        print("  # Step 1: Collect 900 points (15 min market) for ETH")
        print("  #         Will collect for 1080s (900 × 1.2 buffer)")
        print("  python scripts/cli_v2.py collect --assets ETH --duration 900")
        print()
        print("  # Step 2: Match and create market folders")
        print("  python scripts/cli_v2.py process")
        print()
        print("  # Step 3: Build canonical research dataset")
        print("  python scripts/cli_v2.py build")
        print()
        print("  # Use legacy real-time matching mode (slower, more overhead)")
        print("  python scripts/cli_v2.py collect --assets ETH --duration 60 --legacy")
        print()
        return 0


if __name__ == "__main__":
    import warnings
    import gc
    warnings.filterwarnings("ignore", category=ResourceWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    # Suppress asyncio pipe errors on Windows (harmless cleanup noise)
    import logging
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)
    
    # NOTE: Do NOT use WindowsSelectorEventLoopPolicy - it breaks Playwright
    # The ProactorEventLoop is required for Playwright's subprocess management
    
    try:
        exit_code = asyncio.run(main())
        gc.collect()  # Help cleanup before exit
        sys.exit(exit_code or 0)
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

