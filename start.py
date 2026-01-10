#!/usr/bin/env python3
"""
Unified Interactive CLI for Polymarket/Chainlink

Top-level menu:
  1. Data Collection (V3/RTDS)
  2. Paper Trading

Both can run multiple operations simultaneously in separate terminals.
"""

import sys
import os
import asyncio
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

# Configure stdout for Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

# Rich imports
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Prompt, IntPrompt, Confirm
    from rich import box
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Note: Install 'rich' for better UI: pip install rich")

from config.settings import SUPPORTED_ASSETS, STORAGE


@dataclass
class RunningProcess:
    """Track a running process."""
    name: str
    process: subprocess.Popen
    window_title: str
    start_time: float = field(default_factory=time.time)
    
    def is_running(self) -> bool:
        """Check if process is still running."""
        return self.process.poll() is None
    
    def stop(self):
        """Stop the process."""
        try:
            if sys.platform == "win32":
                # On Windows, kill the process tree
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(self.process.pid)],
                    capture_output=True
                )
            else:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
        except Exception:
            pass


class ProcessManager:
    """Manages running processes in separate terminals."""
    
    def __init__(self):
        self.processes: Dict[str, RunningProcess] = {}
    
    def launch_in_terminal(
        self,
        name: str,
        command: List[str],
        title: Optional[str] = None
    ) -> bool:
        """
        Launch a command in a new terminal window.
        
        Args:
            name: Unique name for this process
            command: Command to run
            title: Window title (defaults to name)
        
        Returns:
            True if launched successfully
        """
        if name in self.processes:
            print(f"Process '{name}' is already running!")
            return False
        
        title = title or name
        
        try:
            if sys.platform == "win32":
                # Windows: Use start to open new cmd window
                # Build command string - escape each argument properly
                cmd_parts = []
                for arg in command:
                    # Double quotes for Windows, escape internal quotes
                    if " " in arg or '"' in arg:
                        escaped = arg.replace('"', '""')
                        cmd_parts.append(f'"{escaped}"')
                    else:
                        cmd_parts.append(arg)
                cmd_str = " ".join(cmd_parts)
                
                # Use start command: start "Title" cmd /k "command"
                # We need to pass this as a single string to cmd /c
                start_cmd = f'start "{title}" cmd /k {cmd_str}'
                process = subprocess.Popen(
                    start_cmd,
                    shell=True,
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
            else:
                # Linux/Mac: Use xterm, gnome-terminal, or similar
                # Try different terminal emulators
                terminals = [
                    ["gnome-terminal", "--title", title, "--", "bash", "-c"],
                    ["xterm", "-T", title, "-e", "bash", "-c"],
                    ["konsole", "-e", "bash", "-c"],
                    ["terminator", "-T", title, "-e"],
                ]
                
                process = None
                for term_cmd in terminals:
                    try:
                        if term_cmd[0] == "terminator":
                            full_cmd = term_cmd + [" ".join(command)]
                        else:
                            cmd_str = " ".join(f"'{arg}'" if " " in arg else arg for arg in command)
                            full_cmd = term_cmd + [cmd_str + "; exec bash"]
                        
                        process = subprocess.Popen(
                            full_cmd,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL
                        )
                        break
                    except FileNotFoundError:
                        continue
                
                if process is None:
                    print("No terminal emulator found. Please run manually:")
                    print(f"  {' '.join(command)}")
                    return False
            
            self.processes[name] = RunningProcess(
                name=name,
                process=process,
                window_title=title
            )
            return True
            
        except Exception as e:
            print(f"Error launching '{name}': {e}")
            return False
    
    def list_running(self) -> List[str]:
        """List names of running processes."""
        # Clean up finished processes
        finished = [name for name, proc in self.processes.items() if not proc.is_running()]
        for name in finished:
            del self.processes[name]
        
        return list(self.processes.keys())
    
    def stop(self, name: str) -> bool:
        """Stop a running process."""
        if name not in self.processes:
            return False
        
        self.processes[name].stop()
        del self.processes[name]
        return True
    
    def stop_all(self):
        """Stop all running processes."""
        for name in list(self.processes.keys()):
            self.stop(name)


class UnifiedCLI:
    """Unified CLI with Data Collection and Paper Trading."""
    
    def __init__(self):
        self.console = Console(force_terminal=True, legacy_windows=True) if RICH_AVAILABLE else None
        self.running = True
        self.process_manager = ProcessManager()
        self.current_mode = None  # "data" or "trading"
    
    def clear_screen(self):
        """Clear terminal screen."""
        os.system('cls' if sys.platform == 'win32' else 'clear')
    
    def print_header(self, mode: Optional[str] = None):
        """Print main header."""
        self.clear_screen()
        
        title = "Polymarket/Chainlink Unified CLI"
        subtitle = ""
        if mode == "data":
            subtitle = "[bold green]Data Collection Mode (V3/RTDS)[/bold green]"
        elif mode == "trading":
            subtitle = "[bold cyan]Paper Trading Mode[/bold cyan]"
        else:
            subtitle = "[dim]Select a mode to begin[/dim]"
        
        if self.console and RICH_AVAILABLE:
            header = Panel(
                f"[bold cyan]{title}[/bold cyan]\n{subtitle}",
                border_style="cyan",
                padding=(1, 2)
            )
            self.console.print(header)
            self.console.print()
        else:
            print("=" * 60)
            print(f"  {title}")
            if subtitle:
                print(f"  {subtitle}")
            print("=" * 60)
            print()
    
    def print_top_menu(self):
        """Print top-level menu."""
        if self.console and RICH_AVAILABLE:
            table = Table(title="Main Menu", box=box.ROUNDED, show_header=False)
            table.add_column("Option", style="cyan", width=4)
            table.add_column("Mode", style="white", width=30)
            table.add_column("Description", style="dim", width=40)
            
            table.add_row("1", "[green]Data Collection[/green]", "V3/RTDS data collection & processing")
            table.add_row("2", "[cyan]Paper Trading[/cyan]", "Live trading, orderbook, streams")
            table.add_row("", "", "")
            table.add_row("p", "Process Manager", "View/manage running processes")
            table.add_row("q", "Quit", "Exit the CLI")
            
            self.console.print(table)
            self.console.print()
        else:
            print("Main Menu:")
            print("  1. Data Collection (V3/RTDS)")
            print("  2. Paper Trading")
            print("  p. Process Manager")
            print("  q. Quit")
            print()
    
    def print_data_collection_menu(self):
        """Print data collection menu."""
        if self.console and RICH_AVAILABLE:
            table = Table(title="Data Collection Menu", box=box.ROUNDED, show_header=False)
            table.add_column("Option", style="cyan", width=4)
            table.add_column("Action", style="white", width=35)
            table.add_column("Description", style="dim", width=40)
            
            table.add_row("1", "[green]Collect Data (V3/RTDS)[/green]", "Real-time (new window option)")
            table.add_row("2", "Process Data", "Convert raw → markets (new window option)")
            table.add_row("3", "Build Dataset", "Create dataset (new window option)")
            table.add_row("4", "Validate Dataset", "Quality checks (new window option)")
            table.add_row("5", "View Status", "Check data status")
            table.add_row("6", "Full Workflow", "Run pipeline (new window option)")
            table.add_row("7", "[red]Delete Low Coverage[/red]", "Delete markets below threshold")
            table.add_row("", "", "")
            table.add_row("t", "Test RTDS", "Test RTDS stream (new window option)")
            table.add_row("", "", "")
            table.add_row("b", "Back", "Return to main menu")
            table.add_row("q", "[red]Quit[/red]", "Exit program")
            
            self.console.print(table)
            self.console.print("[dim]Note: Options 1-4, 6, t can run in new windows for parallel execution[/dim]")
            self.console.print("[dim]Tip: 'b' = back, 'q' = quit (works from any menu)[/dim]")
            self.console.print()
        else:
            print("Data Collection Menu:")
            print("  1. Collect Data (V3/RTDS) - can run in new window")
            print("  2. Process Data - can run in new window")
            print("  3. Build Dataset - can run in new window")
            print("  4. Validate Dataset - can run in new window")
            print("  5. View Status")
            print("  6. Full Workflow - can run in new window")
            print("  7. Delete Low Coverage Markets")
            print("  t. Test RTDS - can run in new window")
            print("  b. Back")
            print("  q. Quit")
            print()
    
    def print_paper_trading_menu(self):
        """Print paper trading menu."""
        if self.console and RICH_AVAILABLE:
            table = Table(title="Paper Trading Menu", box=box.ROUNDED, show_header=False)
            table.add_column("Option", style="cyan", width=4)
            table.add_column("Action", style="white", width=35)
            table.add_column("Description", style="dim", width=45)
            
            # Live Trading
            table.add_row("1", "[green]Run Paper Trading[/green]", "Trade live with a strategy")
            table.add_row("2", "[cyan]Live Orderbook[/cyan]", "PM orderbook + CL mid (new window)")
            table.add_row("3", "[cyan]Data Stream[/cyan]", "Live Chainlink stream (new window)")
            table.add_row("", "", "")
            
            # Backtesting & Replay
            table.add_row("4", "[yellow]Replay Historical Data[/yellow]", "Backtest strategy on sealed batches")
            table.add_row("5", "[magenta]Visualize Historical Run[/magenta]", "Watch backtest like live trading")
            table.add_row("6", "[yellow]Compare RC1 vs RC2[/yellow]", "Evaluate strategies side-by-side")
            table.add_row("", "", "")
            
            # Testing
            table.add_row("7", "[dim]Test RTDS Stream[/dim]", "Test Chainlink RTDS")
            table.add_row("8", "[dim]Test Data Alignment[/dim]", "Test CL + PM alignment")
            table.add_row("", "", "")
            table.add_row("b", "Back", "Return to main menu")
            table.add_row("q", "[red]Quit[/red]", "Exit program")
            
            self.console.print(table)
            self.console.print("[dim]Tip: 'b' = back, 'q' = quit (works from any menu)[/dim]")
            self.console.print()
        else:
            print("Paper Trading Menu:")
            print("  1. Run Paper Trading")
            print("  2. Live Orderbook (new window)")
            print("  3. Data Stream (new window)")
            print("")
            print("  4. Replay Historical Data (backtest on sealed batches)")
            print("  5. Visualize Historical Run (watch backtest like live)")
            print("  6. Compare RC1 vs RC2 (evaluate strategies)")
            print("")
            print("  7. Test RTDS Stream")
            print("  8. Test Data Alignment")
            print("  b. Back")
            print("  q. Quit")
            print()
    
    def print_process_manager(self):
        """Print process manager menu."""
        running = self.process_manager.list_running()
        
        if self.console and RICH_AVAILABLE:
            self.console.print(Panel("[bold]Process Manager[/bold]", border_style="yellow"))
            self.console.print()
            
            if running:
                table = Table(title="Running Processes", box=box.ROUNDED)
                table.add_column("Name", style="cyan")
                table.add_column("Status", style="green")
                table.add_column("Window Title", style="dim")
                
                for name in running:
                    proc = self.process_manager.processes[name]
                    elapsed = int(time.time() - proc.start_time)
                    table.add_row(
                        name,
                        f"[green]Running[/green] ({elapsed}s)",
                        proc.window_title
                    )
                
                self.console.print(table)
                self.console.print()
                self.console.print("Options:")
                self.console.print("  [cyan]stop <name>[/cyan]  - Stop a process")
                self.console.print("  [cyan]stop-all[/cyan]    - Stop all processes")
                self.console.print("  [cyan]refresh[/cyan]      - Refresh list")
                self.console.print("  [cyan]b[/cyan] or [cyan]back[/cyan] - Return to menu")
                self.console.print("  [red]q[/red] or [red]quit[/red] - Exit program")
            else:
                self.console.print("[dim]No processes running[/dim]")
                self.console.print()
                self.console.print("Press Enter to return...")
        else:
            if running:
                print("Running Processes:")
                for name in running:
                    proc = self.process_manager.processes[name]
                    elapsed = int(time.time() - proc.start_time)
                    print(f"  {name} - Running ({elapsed}s) - {proc.window_title}")
                print()
                print("Commands: stop <name>, stop-all, refresh, b/back, q/quit")
            else:
                print("No processes running")
                print("Press Enter to return...")
    
    def get_choice(self, prompt: str = "Select option") -> str:
        """Get user's menu choice."""
        if self.console and RICH_AVAILABLE:
            return Prompt.ask(f"[bold]{prompt}[/bold]")
        else:
            return input(f"{prompt}: ").strip()
    
    # =========================================================================
    # DATA COLLECTION METHODS
    # =========================================================================
    
    def collect_data_menu(self):
        """Menu for V3 data collection."""
        self.print_header("data")
        
        if self.console and RICH_AVAILABLE:
            self.console.print(Panel(
                "[bold green]V3 Data Collection (RTDS)[/bold green]\n"
                "[dim]Real-time Chainlink prices via websocket - no 1-minute delay![/dim]",
                border_style="green"
            ))
            self.console.print()
            
            assets_input = Prompt.ask("Assets to collect", default="ETH")
            
            # Ask for collection mode
            self.console.print()
            self.console.print("[cyan]Collection Mode:[/cyan]")
            self.console.print("  1. Target points (e.g., 900 = 15-min market)")
            self.console.print("  2. Duration in seconds")
            self.console.print("  3. Indefinite (run until Ctrl+C)")
            mode_choice = Prompt.ask("Select mode", default="1", choices=["1", "2", "3"])
            
            target = None
            duration = None
            indefinite = False
            
            if mode_choice == "1":
                target = IntPrompt.ask(
                    "Target matched points per asset (900 = 15-min market)",
                    default=900
                )
            elif mode_choice == "2":
                duration = IntPrompt.ask("Duration in seconds", default=900)
            else:
                indefinite = True
        else:
            print("V3 Data Collection (RTDS)")
            print("-" * 40)
            assets_input = input("Assets (default: ETH): ").strip() or "ETH"
            print()
            print("Collection Mode:")
            print("  1. Target points (e.g., 900 = 15-min market)")
            print("  2. Duration in seconds")
            print("  3. Indefinite (run until Ctrl+C)")
            mode_choice = input("Select mode (1/2/3, default: 1): ").strip() or "1"
            
            target = None
            duration = None
            indefinite = False
            
            if mode_choice == "1":
                target = int(input("Target points (default: 900): ") or "900")
            elif mode_choice == "2":
                duration = int(input("Duration in seconds (default: 900): ") or "900")
            else:
                indefinite = True
        
        assets = [a.strip().upper() for a in assets_input.split(',')]
        valid_assets = [a for a in assets if a in SUPPORTED_ASSETS]
        
        if not valid_assets:
            print(f"No valid assets. Choose from: {SUPPORTED_ASSETS}")
            input("Press Enter to continue...")
            return
        
        # Build command
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "scripts" / "cli_v3.py"),
            "collect",
            "--assets", ",".join(valid_assets)
        ]
        
        if indefinite:
            # For indefinite, use duration=None (or don't pass duration/target)
            # The CLI will handle None as indefinite
            cmd.extend(["--duration", "0"])  # 0 means indefinite in CLI
        elif target:
            cmd.extend(["--target", str(target)])
        elif duration:
            cmd.extend(["--duration", str(duration)])
        
        # Ask if should run in new window
        if self.console and RICH_AVAILABLE:
            in_new_window = Confirm.ask("Run in new terminal window?", default=True)
        else:
            in_new_window = input("Run in new window? (y/n, default: y): ").lower() != 'n'
        
        if in_new_window:
            if indefinite:
                name = f"data_collection_{'_'.join(valid_assets)}_indefinite"
                title = f"Data Collection (Indefinite) - {', '.join(valid_assets)}"
            elif target:
                name = f"data_collection_{'_'.join(valid_assets)}_{target}pts"
                title = f"Data Collection ({target} pts) - {', '.join(valid_assets)}"
            else:
                name = f"data_collection_{'_'.join(valid_assets)}_{duration}s"
                title = f"Data Collection ({duration}s) - {', '.join(valid_assets)}"
            
            if self.process_manager.launch_in_terminal(name, cmd, title):
                print(f"\n[OK] Launched '{name}' in new window!")
                if indefinite:
                    print("Collection will run indefinitely. Press Ctrl+C in that window to stop.")
                print("You can continue using the CLI while collection runs.")
            else:
                print("\n[ERROR] Failed to launch. Running in current window...")
                try:
                    subprocess.run(cmd)
                except KeyboardInterrupt:
                    pass
        else:
            try:
                subprocess.run(cmd)
            except KeyboardInterrupt:
                if indefinite:
                    print("\n[OK] Collection stopped by user (Ctrl+C)")
                pass
        
        input("\nPress Enter to continue...")
    
    def process_data_menu(self):
        """Menu for processing raw data."""
        self.print_header("data")
        
        if self.console and RICH_AVAILABLE:
            if not Confirm.ask("Process raw data?", default=True):
                return
            in_new_window = Confirm.ask("Run in new terminal window?", default=False)
        else:
            if input("Process? (y/n): ").lower() != 'y':
                return
            in_new_window = input("Run in new window? (y/n, default: n): ").lower() == 'y'
        
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "scripts" / "cli_v3.py"),
            "process"
        ]
        
        if in_new_window:
            name = "process_data"
            title = "Process Raw Data"
            if self.process_manager.launch_in_terminal(name, cmd, title):
                print(f"\n[OK] Launched '{name}' in new window!")
            else:
                print("\n[ERROR] Failed to launch. Running in current window...")
                try:
                    subprocess.run(cmd)
                except KeyboardInterrupt:
                    pass
        else:
            try:
                subprocess.run(cmd)
            except KeyboardInterrupt:
                pass
        
        input("\nPress Enter to continue...")
    
    def build_dataset_menu(self):
        """Menu for building research dataset."""
        self.print_header("data")
        
        if self.console and RICH_AVAILABLE:
            min_coverage = IntPrompt.ask("Minimum coverage percentage", default=80)
            if not Confirm.ask("Build dataset?", default=True):
                return
            in_new_window = Confirm.ask("Run in new terminal window?", default=False)
        else:
            min_coverage = int(input("Min coverage % (default: 80): ") or "80")
            if input("Build? (y/n): ").lower() != 'y':
                return
            in_new_window = input("Run in new window? (y/n, default: n): ").lower() == 'y'
        
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "scripts" / "cli_v3.py"),
            "build",
            "--min-coverage", str(min_coverage)
        ]
        
        if in_new_window:
            name = "build_dataset"
            title = "Build Research Dataset"
            if self.process_manager.launch_in_terminal(name, cmd, title):
                print(f"\n[OK] Launched '{name}' in new window!")
            else:
                print("\n[ERROR] Failed to launch. Running in current window...")
                try:
                    subprocess.run(cmd)
                except KeyboardInterrupt:
                    pass
        else:
            try:
                subprocess.run(cmd)
            except KeyboardInterrupt:
                pass
        
        input("\nPress Enter to continue...")
    
    def validate_dataset_menu(self):
        """Menu for validating dataset."""
        self.print_header("data")
        
        if self.console and RICH_AVAILABLE:
            if not Confirm.ask("Validate dataset?", default=True):
                return
            in_new_window = Confirm.ask("Run in new terminal window?", default=False)
        else:
            if input("Validate? (y/n): ").lower() != 'y':
                return
            in_new_window = input("Run in new window? (y/n, default: n): ").lower() == 'y'
        
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "scripts" / "cli_v2.py"),
            "validate"
        ]
        
        if in_new_window:
            name = "validate_dataset"
            title = "Validate Dataset"
            if self.process_manager.launch_in_terminal(name, cmd, title):
                print(f"\n[OK] Launched '{name}' in new window!")
            else:
                print("\n[ERROR] Failed to launch. Running in current window...")
                try:
                    subprocess.run(cmd)
                except KeyboardInterrupt:
                    pass
        else:
            try:
                subprocess.run(cmd)
            except KeyboardInterrupt:
                pass
        
        input("\nPress Enter to continue...")
    
    def delete_low_coverage_markets_menu(self):
        """Delete markets below a coverage threshold."""
        self.print_header("data")
        
        if self.console and RICH_AVAILABLE:
            self.console.print(Panel(
                "[bold red]Delete Low Coverage Markets[/bold red]\n"
                "[dim]Remove markets below a coverage threshold[/dim]",
                border_style="red"
            ))
            self.console.print()
            
            asset = Prompt.ask("Asset", default="ETH", choices=SUPPORTED_ASSETS)
            threshold = float(Prompt.ask("Coverage threshold (%)", default="80.0"))
            
            self.console.print(f"\n[yellow]This will delete all {asset} markets with < {threshold}% coverage![/yellow]")
            if not Confirm.ask("Continue?", default=False):
                print("Cancelled.")
                input("Press Enter to continue...")
                return
            
            skip_confirm = Confirm.ask("Skip confirmation prompt?", default=True)
        else:
            asset = input("Asset (default: ETH): ").strip().upper() or "ETH"
            threshold = float(input("Coverage threshold (%) (default: 80.0): ") or "80.0")
            
            print(f"\nThis will delete all {asset} markets with < {threshold}% coverage!")
            if input("Continue? (y/N): ").lower() != 'y':
                print("Cancelled.")
                input("Press Enter to continue...")
                return
            
            skip_confirm = input("Skip confirmation prompt? (y/n, default: y): ").lower() != 'n'
        
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "scripts" / "delete_low_coverage_markets.py"),
            "--asset", asset,
            "--threshold", str(threshold)
        ]
        
        if skip_confirm:
            cmd.append("--yes")
        
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            print("\nCancelled.")
        
        input("\nPress Enter to continue...")
    
    def view_status(self):
        """View current data status."""
        self.print_header("data")
        
        dirs_to_check = [
            ("Raw Data (Chainlink)", Path(STORAGE.raw_dir) / "chainlink"),
            ("Raw Data (Polymarket)", Path(STORAGE.raw_dir) / "polymarket"),
            ("Markets (6-level)", Path(STORAGE.markets_dir)),
            ("Research", Path(STORAGE.research_dir)),
        ]
        
        if self.console and RICH_AVAILABLE:
            table = Table(title="Directory Status", box=box.ROUNDED)
            table.add_column("Directory", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Files", justify="right")
            
            for name, path in dirs_to_check:
                if path.exists():
                    files = list(path.rglob("*.csv")) + list(path.rglob("*.parquet"))
                    status = "[green]Exists[/green]"
                    count = str(len(files))
                else:
                    status = "[red]Not found[/red]"
                    count = "-"
                table.add_row(name, status, count)
            
            self.console.print(table)
        else:
            for name, path in dirs_to_check:
                if path.exists():
                    files = list(path.rglob("*.csv")) + list(path.rglob("*.parquet"))
                    print(f"  {name}: {len(files)} files")
                else:
                    print(f"  {name}: Not found")
        
        print()
        input("Press Enter to continue...")
    
    def test_rtds_data(self):
        """Test RTDS Chainlink stream."""
        self.print_header("data")
        
        if self.console and RICH_AVAILABLE:
            asset = Prompt.ask("Asset to test", default="ETH")
            duration = IntPrompt.ask("Duration in seconds", default=30)
            in_new_window = Confirm.ask("Run in new terminal window?", default=False)
        else:
            asset = input("Asset (default: ETH): ").strip() or "ETH"
            duration = int(input("Duration (default: 30): ") or "30")
            in_new_window = input("Run in new window? (y/n, default: n): ").lower() == 'y'
        
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "scripts" / "cli_v3.py"),
            "test-rtds",
            "--asset", asset.upper(),
            "--duration", str(duration)
        ]
        
        if in_new_window:
            name = f"test_rtds_{asset}"
            title = f"Test RTDS - {asset}"
            if self.process_manager.launch_in_terminal(name, cmd, title):
                print(f"\n[OK] Launched '{name}' in new window!")
            else:
                print("\n[ERROR] Failed to launch. Running in current window...")
                try:
                    subprocess.run(cmd)
                except KeyboardInterrupt:
                    pass
        else:
            try:
                subprocess.run(cmd)
            except KeyboardInterrupt:
                pass
        
        input("\nPress Enter to continue...")
    
    def full_workflow(self):
        """Run the complete V3 workflow."""
        self.print_header("data")
        
        if self.console and RICH_AVAILABLE:
            assets_input = Prompt.ask("Assets to collect", default="ETH")
            target = IntPrompt.ask("Target points per asset", default=900)
            if not Confirm.ask("Start full V3 workflow?", default=True):
                return
            in_new_window = Confirm.ask("Run workflow in new terminal window?", default=True)
        else:
            assets_input = input("Assets (default: ETH): ").strip() or "ETH"
            target = int(input("Target points (default: 900): ") or "900")
            if input("Start? (y/n): ").lower() != 'y':
                return
            in_new_window = input("Run in new window? (y/n, default: y): ").lower() != 'n'
        
        assets = [a.strip().upper() for a in assets_input.split(',')]
        valid_assets = [a for a in assets if a in SUPPORTED_ASSETS]
        
        if not valid_assets:
            print("No valid assets.")
            input("Press Enter to continue...")
            return
        
        # Create a batch script for the full workflow
        if in_new_window:
            # Create a temporary script that runs all steps
            script_path = Path(__file__).parent / "temp_workflow.bat" if sys.platform == "win32" else Path(__file__).parent / "temp_workflow.sh"
            
            if sys.platform == "win32":
                script_content = f"""@echo off
echo ========================================
echo Full V3 Workflow
echo ========================================
echo.
echo Step 1/4: Collecting data...
python scripts/cli_v3.py collect --assets {",".join(valid_assets)} --target {target}
if errorlevel 1 goto error
echo.
echo Step 2/4: Processing data...
python scripts/cli_v3.py process
if errorlevel 1 goto error
echo.
echo Step 3/4: Building dataset...
python scripts/cli_v3.py build
if errorlevel 1 goto error
echo.
echo Step 4/4: Validating dataset...
python scripts/cli_v2.py validate
if errorlevel 1 goto error
echo.
echo ========================================
echo Workflow Complete!
echo ========================================
pause
goto end
:error
echo.
echo Workflow failed at a step.
pause
:end
"""
            else:
                script_content = f"""#!/bin/bash
echo "========================================"
echo "Full V3 Workflow"
echo "========================================"
echo ""
echo "Step 1/4: Collecting data..."
python scripts/cli_v3.py collect --assets {",".join(valid_assets)} --target {target} || exit 1
echo ""
echo "Step 2/4: Processing data..."
python scripts/cli_v3.py process || exit 1
echo ""
echo "Step 3/4: Building dataset..."
python scripts/cli_v3.py build || exit 1
echo ""
echo "Step 4/4: Validating dataset..."
python scripts/cli_v2.py validate || exit 1
echo ""
echo "========================================"
echo "Workflow Complete!"
echo "========================================"
read -p "Press Enter to close..."
"""
            
            script_path.write_text(script_content)
            if not sys.platform == "win32":
                os.chmod(script_path, 0o755)
            
            cmd = [str(script_path)]
            name = f"full_workflow_{'_'.join(valid_assets)}"
            title = f"Full Workflow - {', '.join(valid_assets)}"
            
            if self.process_manager.launch_in_terminal(name, cmd, title):
                print(f"\n[OK] Launched '{name}' in new window!")
                print("The workflow will run all 4 steps sequentially.")
            else:
                print("\n[ERROR] Failed to launch. Running in current window...")
                # Fall through to sequential execution
                in_new_window = False
        else:
            # Run all steps sequentially in current window
            steps = [
                ("Collecting data (V3/RTDS)", [
                    sys.executable,
                    str(Path(__file__).parent / "scripts" / "cli_v3.py"),
                    "collect", "--assets", ",".join(valid_assets), "--target", str(target)
                ]),
                ("Processing data", [
                    sys.executable,
                    str(Path(__file__).parent / "scripts" / "cli_v3.py"),
                    "process"
                ]),
                ("Building dataset", [
                    sys.executable,
                    str(Path(__file__).parent / "scripts" / "cli_v3.py"),
                    "build"
                ]),
                ("Validating dataset", [
                    sys.executable,
                    str(Path(__file__).parent / "scripts" / "cli_v2.py"),
                    "validate"
                ]),
            ]
            
            for step_name, cmd in steps:
                if self.console and RICH_AVAILABLE:
                    self.console.print(f"\n[bold cyan]{step_name}...[/bold cyan]")
                else:
                    print(f"\n--- {step_name} ---")
                
                try:
                    subprocess.run(cmd, check=False)
                except KeyboardInterrupt:
                    print("Workflow cancelled.")
                    break
            
            print("\n=== Workflow complete! ===")
        
        input("\nPress Enter to continue...")
    
    # =========================================================================
    # PAPER TRADING METHODS
    # =========================================================================
    
    def run_paper_trading(self):
        """Run paper trading with a strategy."""
        self.print_header("trading")
        
        # Import the CLI to get strategy list
        try:
            from paper_trading.cli import PaperTradingCLI
            cli_helper = PaperTradingCLI()
            strategies = cli_helper._get_available_strategies()
        except ImportError:
            strategies = []
            cli_helper = None
        
        if self.console and RICH_AVAILABLE:
            asset = Prompt.ask("Asset", default="ETH")
            
            # Interactive strategy selection
            if cli_helper:
                # Show strategy menu
                from rich.table import Table
                from rich.panel import Panel
                
                strategy_table = Table.grid(padding=(0, 2))
                strategy_table.add_column(style="cyan", width=3)
                strategy_table.add_column(style="bold white", width=30)
                strategy_table.add_column(style="dim", width=50)
                
                for i, strat in enumerate(strategies, 1):
                    purpose_color = "green" if strat['purpose'] == "Production trading" else "yellow" if strat['purpose'] == "Research/experimental" else "dim"
                    strategy_table.add_row(
                        f"{i}.",
                        strat['name'],
                        f"[{purpose_color}]{strat['description']}[/{purpose_color}]"
                    )
                
                strategy_panel = Panel(
                    strategy_table,
                    title="[bold cyan]Available Strategies[/bold cyan]",
                    border_style="cyan",
                    padding=(1, 2)
                )
                self.console.print(strategy_panel)
                self.console.print()
                
                # Try using questionary for arrow keys if available
                try:
                    import questionary
                    choices = []
                    for strat in strategies:
                        choice_text = f"{strat['name']} - {strat['description']}"
                        choices.append(questionary.Choice(
                            title=choice_text,
                            value=strat['key']
                        ))
                    
                    selected_strategy = questionary.select(
                        "Select a strategy (use arrow keys, Enter to select):",
                        choices=choices,
                        use_arrow_keys=True,
                    ).ask()
                    
                    if not selected_strategy:
                        print("Cancelled")
                        input("Press Enter to continue...")
                        return
                    
                    strategy = selected_strategy
                    # Show selected strategy
                    for strat in strategies:
                        if strat['key'] == strategy:
                            self.console.print(f"[green]Selected:[/green] {strat['name']}")
                            break
                    
                except ImportError:
                    # Fallback to prompt
                    self.console.print("[dim]Tip: Install 'questionary' for arrow-key navigation: pip install questionary[/dim]")
                    self.console.print()
                    
                    strategy_input = Prompt.ask(
                        f"Strategy number (1-{len(strategies)}) or name",
                        default="1"
                    ).strip().lower()
                    
                    # Try as number
                    if strategy_input.isdigit():
                        idx = int(strategy_input) - 1
                        if 0 <= idx < len(strategies):
                            strategy = strategies[idx]['key']
                        else:
                            self.console.print(f"[red]Invalid number: {strategy_input}[/red]")
                            input("Press Enter to continue...")
                            return
                    else:
                        # Try as name/alias
                        found = False
                        for strat in strategies:
                            if strategy_input in strat['aliases'] or strategy_input == strat['key']:
                                strategy = strat['key']
                                found = True
                                break
                        if not found:
                            self.console.print(f"[red]Unknown strategy: {strategy_input}[/red]")
                            input("Press Enter to continue...")
                            return
            else:
                # Fallback if CLI import failed
                strategy = Prompt.ask("Strategy", default="placeholder")
            
            duration = IntPrompt.ask("Duration in seconds (0 = indefinite)", default=0)
            capital = IntPrompt.ask("Initial capital", default=100)
            
            if duration == 0:
                duration = None
        else:
            asset = input("Asset (default: ETH): ").strip() or "ETH"
            
            # Simple strategy selection without rich
            if cli_helper and strategies:
                print("\nAvailable strategies:")
                for i, strat in enumerate(strategies, 1):
                    print(f"  {i}. {strat['name']} ({strat['key']})")
                    print(f"     {strat['description']}")
                print()
                
                strategy_input = input(f"Strategy number (1-{len(strategies)}) or name (default: placeholder): ").strip().lower() or "placeholder"
                
                # Try as number
                if strategy_input.isdigit():
                    idx = int(strategy_input) - 1
                    if 0 <= idx < len(strategies):
                        strategy = strategies[idx]['key']
                    else:
                        print(f"Invalid number: {strategy_input}")
                        input("Press Enter to continue...")
                        return
                else:
                    # Try as name/alias
                    found = False
                    for strat in strategies:
                        if strategy_input in strat['aliases'] or strategy_input == strat['key']:
                            strategy = strat['key']
                            found = True
                            break
                    if not found:
                        print(f"Unknown strategy: {strategy_input}, using placeholder")
                        strategy = "placeholder"
            else:
                strategy = input("Strategy (default: placeholder): ").strip() or "placeholder"
            
            duration_input = input("Duration in seconds (0 = indefinite, default: 0): ").strip() or "0"
            duration = int(duration_input) if duration_input != "0" else None
            capital = float(input("Initial capital (default: 100): ") or "100")
        
        cmd = [
            sys.executable,
            "-m", "paper_trading.cli",
            "run",
            "--asset", asset.upper(),
            "--strategy", strategy,
            "--capital", str(capital)
        ]
        
        if duration:
            cmd.extend(["--duration", str(duration)])
        
        # Ask if should run in new window
        if self.console and RICH_AVAILABLE:
            in_new_window = Confirm.ask("Run in new terminal window?", default=False)
        else:
            in_new_window = input("Run in new window? (y/n): ").lower() == 'y'
        
        if in_new_window:
            name = f"paper_trading_{asset}_{strategy}"
            title = f"Paper Trading - {asset} ({strategy})"
            if self.process_manager.launch_in_terminal(name, cmd, title):
                print(f"\n[OK] Launched '{name}' in new window!")
                input("Press Enter to continue...")
            else:
                print("\n[ERROR] Failed to launch. Running in current window...")
                try:
                    subprocess.run(cmd)
                except KeyboardInterrupt:
                    pass
        else:
            try:
                subprocess.run(cmd)
            except KeyboardInterrupt:
                pass
        
        input("\nPress Enter to continue...")
    
    def launch_orderbook(self):
        """Launch live orderbook in new window."""
        self.print_header("trading")
        
        if self.console and RICH_AVAILABLE:
            asset = Prompt.ask("Asset", default="ETH")
            duration = IntPrompt.ask("Duration in seconds (0 = indefinite)", default=0)
            if duration == 0:
                duration = None
        else:
            asset = input("Asset (default: ETH): ").strip() or "ETH"
            duration_input = input("Duration (0 = indefinite, default: 0): ").strip() or "0"
            duration = int(duration_input) if duration_input != "0" else None
        
        cmd = [
            sys.executable,
            "-m", "paper_trading.cli",
            "orderbook",
            "--asset", asset.upper()
        ]
        
        if duration:
            cmd.extend(["--duration", str(duration)])
        
        name = f"orderbook_{asset}"
        title = f"Orderbook - {asset}"
        
        if self.process_manager.launch_in_terminal(name, cmd, title):
            print(f"\n[OK] Launched '{name}' in new window!")
            print("You can launch more tools while this runs.")
        else:
            print("\n[ERROR] Failed to launch.")
        
        input("\nPress Enter to continue...")
    
    def launch_stream(self):
        """Launch data stream in new window."""
        self.print_header("trading")
        
        if self.console and RICH_AVAILABLE:
            asset = Prompt.ask("Asset", default="ETH")
            duration = IntPrompt.ask("Duration in seconds (0 = indefinite)", default=0)
            if duration == 0:
                duration = None
        else:
            asset = input("Asset (default: ETH): ").strip() or "ETH"
            duration_input = input("Duration (0 = indefinite, default: 0): ").strip() or "0"
            duration = int(duration_input) if duration_input != "0" else None
        
        cmd = [
            sys.executable,
            "-m", "paper_trading.cli",
            "stream",
            "--asset", asset.upper()
        ]
        
        if duration:
            cmd.extend(["--duration", str(duration)])
        
        name = f"stream_{asset}"
        title = f"Chainlink Stream - {asset}"
        
        if self.process_manager.launch_in_terminal(name, cmd, title):
            print(f"\n[OK] Launched '{name}' in new window!")
            print("You can launch more tools while this runs.")
        else:
            print("\n[ERROR] Failed to launch.")
        
        input("\nPress Enter to continue...")
    
    def test_rtds_trading(self):
        """Test RTDS stream."""
        self.print_header("trading")
        
        if self.console and RICH_AVAILABLE:
            asset = Prompt.ask("Asset", default="ETH")
            duration = IntPrompt.ask("Duration in seconds", default=30)
        else:
            asset = input("Asset (default: ETH): ").strip() or "ETH"
            duration = int(input("Duration (default: 30): ") or "30")
        
        cmd = [
            sys.executable,
            "-m", "paper_trading.cli",
            "test-cl",
            "--asset", asset.upper(),
            "--duration", str(duration)
        ]
        
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            pass
        
        input("\nPress Enter to continue...")
    
    def test_data_alignment(self):
        """Test CL + PM data alignment."""
        self.print_header("trading")
        
        if self.console and RICH_AVAILABLE:
            asset = Prompt.ask("Asset", default="ETH")
            duration = IntPrompt.ask("Duration in seconds", default=30)
        else:
            asset = input("Asset (default: ETH): ").strip() or "ETH"
            duration = int(input("Duration (default: 30): ") or "30")
        
        cmd = [
            sys.executable,
            "-m", "paper_trading.cli",
            "test-data",
            "--asset", asset.upper(),
            "--duration", str(duration)
        ]
        
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            pass
        
        input("\nPress Enter to continue...")
    
    def run_replay(self):
        """Run historical replay of strategy on selected markets."""
        self.print_header("trading")
        print("Historical Replay - Backtest strategy on selected markets")
        print()
        
        # Import CLI helper to get strategy list and market selector
        try:
            from paper_trading.cli import PaperTradingCLI
            from paper_trading.market_selector import get_available_markets, select_markets, get_market_coverage_report
            cli_helper = PaperTradingCLI()
            strategies = cli_helper._get_available_strategies()
        except ImportError:
            strategies = []
            cli_helper = None
            get_available_markets = None
            select_markets = None
            get_market_coverage_report = None
        
        # Strategy selection (multi-select)
        selected_strategies = []
        if self.console and RICH_AVAILABLE and cli_helper:
            # Show strategy menu
            strategy_table = Table.grid(padding=(0, 2))
            strategy_table.add_column(style="cyan", width=3)
            strategy_table.add_column(style="bold white", width=30)
            strategy_table.add_column(style="dim", width=50)
            
            for i, strat in enumerate(strategies, 1):
                strategy_table.add_row(
                    f"{i}.",
                    strat['name'],
                    strat['description']
                )
            
            strategy_panel = Panel(
                strategy_table,
                title="[bold cyan]Select Strategies for Backtest (Multi-Select)[/bold cyan]",
                border_style="cyan",
                padding=(1, 2)
            )
            self.console.print(strategy_panel)
            self.console.print()
            
            # Try questionary for multi-select
            try:
                import questionary
                strategy_choices = []
                for strat in strategies:
                    strategy_choices.append(questionary.Choice(
                        title=f"{strat['name']} - {strat['description']}",
                        value=strat['key'],
                        checked=False
                    ))
                
                self.console.print("[bold cyan]Multi-Select Instructions:[/bold cyan]")
                self.console.print("  [dim]• Use ↑↓ arrow keys to navigate[/dim]")
                self.console.print("  [dim]• Press SPACE to toggle selection[/dim]")
                self.console.print("  [dim]• Select multiple strategies to compare[/dim]")
                self.console.print("  [dim]• Press [bold]ENTER[/bold] when done[/dim]")
                self.console.print()
                
                selected_values = questionary.checkbox(
                    "Select strategies to backtest (SPACE=toggle, ENTER=proceed):",
                    choices=strategy_choices,
                    use_arrow_keys=True,
                ).ask()
                
                if not selected_values:
                    self.console.print("[yellow]No strategies selected. Cancelling.[/yellow]")
                    input("Press Enter to continue...")
                    return
                
                selected_strategies = selected_values
                self.console.print()
                self.console.print(f"[green]✓ Selected {len(selected_strategies)} strategy(ies):[/green]")
                for s_key in selected_strategies:
                    strat_name = next((s['name'] for s in strategies if s['key'] == s_key), s_key)
                    self.console.print(f"  [dim]• {strat_name}[/dim]")
                self.console.print()
            except ImportError:
                # Fallback: single strategy selection
                strategy_input = Prompt.ask(
                    f"Strategy number (1-{len(strategies)}) or name",
                    default="1"
                ).strip().lower()
                
                if strategy_input.isdigit():
                    idx = int(strategy_input) - 1
                    if 0 <= idx < len(strategies):
                        selected_strategies = [strategies[idx]['key']]
                    else:
                        self.console.print(f"[red]Invalid number[/red]")
                        input("Press Enter to continue...")
                        return
                else:
                    found = False
                    for strat in strategies:
                        if strategy_input in strat['aliases'] or strategy_input == strat['key']:
                            selected_strategies = [strat['key']]
                            found = True
                            break
                    if not found:
                        self.console.print(f"[red]Unknown strategy: {strategy_input}[/red]")
                        input("Press Enter to continue...")
                        return
        else:
            # Fallback: single strategy selection
            if strategies:
                print("Available strategies:")
                for i, strat in enumerate(strategies, 1):
                    print(f"  {i}. {strat['name']} ({strat['key']})")
                print()
                strategy_input = input(f"Strategy number (1-{len(strategies)}) or name: ").strip().lower() or "convergence"
                
                if strategy_input.isdigit():
                    idx = int(strategy_input) - 1
                    if 0 <= idx < len(strategies):
                        selected_strategies = [strategies[idx]['key']]
                    else:
                        print("Invalid number")
                        input("Press Enter to continue...")
                        return
                else:
                    found = False
                    for strat in strategies:
                        if strategy_input in strat['aliases'] or strategy_input == strat['key']:
                            selected_strategies = [strat['key']]
                            found = True
                            break
                    if not found:
                        selected_strategies = ["convergence"]
            else:
                strategy_name = Prompt.ask("Strategy", default="convergence") if self.console and RICH_AVAILABLE else input("Strategy (default: convergence): ").strip() or "convergence"
                selected_strategies = [strategy_name]
        
        # Market selection (number of markets + selection method)
        if get_available_markets and get_market_coverage_report:
            # Get available markets with >80% coverage
            coverage_report = get_market_coverage_report(asset="ETH")
            available_markets = coverage_report['available_markets_80']
            total_available = len(available_markets)
            
            if total_available == 0:
                print(f"\n[ERROR] No markets found with >=80% coverage!")
                print("Please collect and process market data first.")
                input("\nPress Enter to continue...")
                return
            
            if self.console and RICH_AVAILABLE:
                self.console.print(Panel(
                    f"[bold green]Found {total_available} markets with >=80% coverage[/bold green]\n"
                    f"[dim]These are markets that have been collected and processed[/dim]",
                    title="[bold cyan]Available Markets[/bold cyan]",
                    border_style="cyan"
                ))
                self.console.print()
                
                # Get number of markets
                count_input = Prompt.ask(
                    f"How many markets to backtest (1-{total_available})",
                    default=str(min(50, total_available))
                ).strip()
                
                try:
                    market_count = int(count_input)
                    if market_count <= 0 or market_count > total_available:
                        self.console.print(f"[red]Invalid number. Must be between 1 and {total_available}[/red]")
                        input("Press Enter to continue...")
                        return
                except ValueError:
                    self.console.print("[red]Invalid input. Please enter a number.[/red]")
                    input("Press Enter to continue...")
                    return
                
                # Get selection method
                try:
                    import questionary
                    method = questionary.select(
                        "How to select markets?",
                        choices=[
                            questionary.Choice(title="Random - Random consecutive selection", value="random"),
                            questionary.Choice(title="Beginning - First N markets (most recent)", value="beginning"),
                            questionary.Choice(title="End - Last N markets (oldest)", value="end"),
                        ],
                        use_arrow_keys=True,
                    ).ask()
                    
                    if not method:
                        input("Cancelled. Press Enter to continue...")
                        return
                except ImportError:
                    print("\nSelection method:")
                    print("  1. random - Random consecutive selection")
                    print("  2. beginning - First N markets (most recent)")
                    print("  3. end - Last N markets (oldest)")
                    method_input = input("\nMethod (1-3, default: 1): ").strip() or "1"
                    method_map = {"1": "random", "2": "beginning", "3": "end"}
                    method = method_map.get(method_input, "random")
                
                # Select markets
                selected_market_ids = select_markets(
                    available_markets=available_markets,
                    count=market_count,
                    method=method,
                )
                
                self.console.print()
                self.console.print(f"[green]✓ Selected {len(selected_market_ids)} markets ({method} selection)[/green]")
                self.console.print()
            else:
                # Fallback
                print(f"\nFound {total_available} markets with >=80% coverage")
                print("These are markets that have been collected and processed\n")
                
                count_input = input(f"How many markets to backtest (1-{total_available}): ").strip()
                try:
                    market_count = int(count_input)
                    if market_count <= 0 or market_count > total_available:
                        print(f"Invalid number. Must be between 1 and {total_available}")
                        input("Press Enter to continue...")
                        return
                except ValueError:
                    print("Invalid input. Please enter a number.")
                    input("Press Enter to continue...")
                    return
                
                print("\nSelection method:")
                print("  1. random - Random consecutive selection")
                print("  2. beginning - First N markets (most recent)")
                print("  3. end - Last N markets (oldest)")
                method_input = input("\nMethod (1-3, default: 1): ").strip() or "1"
                method_map = {"1": "random", "2": "beginning", "3": "end"}
                method = method_map.get(method_input, "random")
                
                selected_market_ids = select_markets(
                    available_markets=available_markets,
                    count=market_count,
                    method=method,
                )
        else:
            print("[ERROR] Market selector not available. Please ensure paper_trading.market_selector is available.")
            input("\nPress Enter to continue...")
            return
        
        # Output directory
        if self.console and RICH_AVAILABLE:
            output_dir = Prompt.ask("Output directory", default="paper_trading/logs/replay")
        else:
            output_dir = input("Output directory (default: paper_trading/logs/replay): ").strip() or "paper_trading/logs/replay"
        
        # Build commands for each strategy
        if self.console and RICH_AVAILABLE:
            in_new_window = Confirm.ask("Run in new terminal window?", default=False)
        else:
            in_new_window = input("Run in new window? (y/n): ").lower() == 'y'
        
        # Run replay for each selected strategy
        for strategy in selected_strategies:
            # Create market IDs string (comma-separated, no spaces)
            market_ids_str = ",".join(m.strip() for m in selected_market_ids)
            
            # Build command
            cmd = [
                sys.executable,
                "-m", "paper_trading.cli",
                "replay",
                "--strategy", strategy,
                "--markets", market_ids_str,
                "--output", output_dir
            ]
            
            # Get strategy display name (fallback to key if not found)
            strategy_name = strategy
            if strategies:
                for s in strategies:
                    if s['key'] == strategy:
                        strategy_name = s['name']
                        break
            
            if in_new_window:
                name = f"replay_{strategy}_{len(selected_market_ids)}m"
                title = f"Backtest - {strategy_name} ({len(selected_market_ids)} markets)"
                if self.process_manager.launch_in_terminal(name, cmd, title):
                    self.console.print(f"\n[green]✓[/green] Launched backtest for [cyan]{strategy_name}[/cyan] in new window!")
                else:
                    self.console.print(f"\n[red]ERROR[/red] Failed to launch. Running in current window...")
                    try:
                        subprocess.run(cmd)
                    except KeyboardInterrupt:
                        pass
            else:
                if self.console and RICH_AVAILABLE:
                    self.console.print(f"\n[cyan]Running backtest for {strategy_name}...[/cyan]")
                else:
                    print(f"\nRunning backtest for {strategy_name}...")
                try:
                    subprocess.run(cmd)
                except KeyboardInterrupt:
                    if self.console and RICH_AVAILABLE:
                        self.console.print("\n[yellow]Backtest interrupted.[/yellow]")
                    pass
        
        if self.console and RICH_AVAILABLE:
            self.console.print(f"\n[green]✓[/green] Completed backtests for {len(selected_strategies)} strategy(ies)")
        else:
            print(f"\n[OK] Completed backtests for {len(selected_strategies)} strategy(ies)")
            print(f"  Markets tested: {len(selected_market_ids)}")
            print(f"  Strategies: {', '.join(selected_strategies)}")
        
        input("\nPress Enter to continue...")
    
    def run_visualize(self):
        """Visualize a historical run through the live dashboard."""
        self.print_header("trading")
        print("Historical Run Visualizer - Watch backtest like live trading")
        print()
        
        # Try to scan for available logs
        try:
            from paper_trading.visualizer import scan_replay_logs, ReplayLogInfo
            logs = scan_replay_logs()
        except ImportError:
            logs = []
        
        if not logs:
            print("No replay logs found.")
            print("Run a historical replay first (option 4) to generate logs.")
            input("\nPress Enter to continue...")
            return
        
        # Log selection
        if self.console and RICH_AVAILABLE:
            from rich.table import Table
            
            log_table = Table(title="Available Replay Logs", box=box.ROUNDED)
            log_table.add_column("#", style="cyan", width=3)
            log_table.add_column("Name", style="bold white", width=25)
            log_table.add_column("Strategy", style="green", width=15)
            log_table.add_column("Trades", justify="right", style="yellow", width=8)
            log_table.add_column("P&L", justify="right", width=10)
            log_table.add_column("Markets", justify="right", width=8)
            
            for i, log in enumerate(logs, 1):
                pnl_color = "green" if log.total_pnl >= 0 else "red"
                log_table.add_row(
                    str(i),
                    log.name,
                    log.strategy,
                    str(log.n_trades),
                    f"[{pnl_color}]{log.total_pnl:+.2f}[/{pnl_color}]",
                    str(log.n_markets),
                )
            
            self.console.print(log_table)
            self.console.print()
            
            # Try questionary for arrow keys
            try:
                import questionary
                choices = []
                for log in logs:
                    pnl_str = f"{log.total_pnl:+.2f}"
                    choices.append(questionary.Choice(
                        title=f"{log.name} | {log.strategy} | {log.n_trades} trades | PnL: {pnl_str}",
                        value=log
                    ))
                
                selected_log = questionary.select(
                    "Select a replay log (use arrow keys):",
                    choices=choices,
                    use_arrow_keys=True,
                ).ask()
                
                if not selected_log:
                    input("Cancelled. Press Enter to continue...")
                    return
            except ImportError:
                log_input = Prompt.ask(f"Log number (1-{len(logs)})", default="1").strip()
                try:
                    idx = int(log_input) - 1
                    if 0 <= idx < len(logs):
                        selected_log = logs[idx]
                    else:
                        self.console.print("[red]Invalid number[/red]")
                        input("Press Enter to continue...")
                        return
                except ValueError:
                    self.console.print("[red]Invalid input[/red]")
                    input("Press Enter to continue...")
                    return
        else:
            print("Available replay logs:")
            for i, log in enumerate(logs, 1):
                print(f"  {i}. {log.name} ({log.strategy}) - {log.n_trades} trades, PnL: {log.total_pnl:+.2f}")
            log_input = input(f"Log number (1-{len(logs)}) [1]: ").strip() or "1"
            try:
                idx = int(log_input) - 1
                if 0 <= idx < len(logs):
                    selected_log = logs[idx]
                else:
                    print("Invalid number")
                    input("Press Enter to continue...")
                    return
            except ValueError:
                print("Invalid input")
                input("Press Enter to continue...")
                return
        
        # Speed selection
        speeds = [
            (0.1, "0.1x (Very slow)"),
            (0.25, "0.25x (Slow)"),
            (0.5, "0.5x (Slow motion)"),
            (1.0, "1x (Real-time)"),
            (2.0, "2x"),
            (5.0, "5x"),
            (10.0, "10x (Fast)"),
            (20.0, "20x (Very fast)"),
            (50.0, "50x (Ultra fast)"),
            (100.0, "100x (Extreme)"),
            (1000.0, "1000x (Maximum speed)"),
        ]
        
        if self.console and RICH_AVAILABLE:
            try:
                import questionary
                speed_choices = [questionary.Choice(title=label, value=speed) for speed, label in speeds]
                selected_speed = questionary.select(
                    "Select playback speed:",
                    choices=speed_choices,
                    default=speed_choices[6] if len(speed_choices) > 6 else speed_choices[-1],  # 5x default (index 6)
                    use_arrow_keys=True,
                ).ask() or 5.0
            except ImportError:
                print("\nPlayback speeds:")
                for i, (speed, label) in enumerate(speeds, 1):
                    print(f"  {i}. {label}")
                try:
                    from rich.prompt import Prompt
                    speed_input = Prompt.ask(f"Speed number (1-{len(speeds)})", default="6").strip()
                except ImportError:
                    speed_input = input(f"Speed number (1-{len(speeds)}) [6]: ").strip() or "6"
                try:
                    idx = int(speed_input) - 1
                    selected_speed = speeds[idx][0] if 0 <= idx < len(speeds) else 5.0
                except ValueError:
                    selected_speed = 5.0
        else:
            print("\nPlayback speeds:")
            for i, (speed, label) in enumerate(speeds, 1):
                print(f"  {i}. {label}")
            speed_input = input(f"Speed number (1-{len(speeds)}) [6]: ").strip() or "6"
            try:
                idx = int(speed_input) - 1
                selected_speed = speeds[idx][0] if 0 <= idx < len(speeds) else 5.0
            except ValueError:
                selected_speed = 5.0
        
        # Build command
        cmd = [
            sys.executable,
            "-m", "paper_trading.cli",
            "visualize",
            "--speed", str(selected_speed),
            "--log", str(selected_log.path)
        ]
        
        print(f"\nStarting visualization at {selected_speed}x speed...")
        print("Press Ctrl+C to stop")
        print()
        
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            pass
        
        input("\nPress Enter to continue...")
    
    def run_evaluation(self):
        """Compare RC1 vs RC2 strategies on sealed batches."""
        self.print_header("trading")
        print("Strategy Evaluation - Compare RC1 vs RC2 on sealed data")
        print()
        
        try:
            from paper_trading.replay import BATCH_DEFINITIONS
        except ImportError:
            BATCH_DEFINITIONS = {}
        
        # Batch selection
        if BATCH_DEFINITIONS:
            batch_list = list(BATCH_DEFINITIONS.keys())
            if self.console and RICH_AVAILABLE:
                batch_table = Table.grid(padding=(0, 2))
                batch_table.add_column(style="cyan", width=3)
                batch_table.add_column(style="bold white", width=20)
                batch_table.add_column(style="dim", width=30)
                
                for i, batch_name in enumerate(batch_list, 1):
                    n_markets = len(BATCH_DEFINITIONS[batch_name])
                    batch_table.add_row(f"{i}.", batch_name, f"{n_markets} markets")
                
                batch_panel = Panel(
                    batch_table,
                    title="[bold cyan]Select Batch for Evaluation[/bold cyan]",
                    border_style="cyan",
                    padding=(1, 2)
                )
                self.console.print(batch_panel)
                self.console.print()
                
                try:
                    import questionary
                    batch_choices = [questionary.Choice(title=f"{b} ({len(BATCH_DEFINITIONS[b])} markets)", value=b) for b in batch_list]
                    batch = questionary.select(
                        "Select batch (use arrow keys):",
                        choices=batch_choices,
                        use_arrow_keys=True,
                    ).ask() or batch_list[0]
                except ImportError:
                    batch_input = Prompt.ask(f"Batch number (1-{len(batch_list)}) or name", default="1").strip().lower()
                    if batch_input.isdigit():
                        idx = int(batch_input) - 1
                        batch = batch_list[idx] if 0 <= idx < len(batch_list) else batch_list[0]
                    else:
                        batch = batch_input if batch_input in BATCH_DEFINITIONS else batch_list[0]
            else:
                print("\nAvailable batches:")
                for i, batch_name in enumerate(batch_list, 1):
                    print(f"  {i}. {batch_name} ({len(BATCH_DEFINITIONS[batch_name])} markets)")
                batch_input = input("Batch number or name (default: batch4): ").strip().lower() or "batch4"
                if batch_input.isdigit():
                    idx = int(batch_input) - 1
                    batch = batch_list[idx] if 0 <= idx < len(batch_list) else "batch4"
                else:
                    batch = batch_input if batch_input in BATCH_DEFINITIONS else "batch4"
        else:
            batch = Prompt.ask("Batch name", default="batch4") if self.console and RICH_AVAILABLE else input("Batch name (default: batch4): ").strip() or "batch4"
        
        # Output directory
        if self.console and RICH_AVAILABLE:
            output_dir = Prompt.ask("Output directory", default="paper_trading/logs/evaluation")
        else:
            output_dir = input("Output directory (default: paper_trading/logs/evaluation): ").strip() or "paper_trading/logs/evaluation"
        
        # Build command
        cmd = [
            sys.executable,
            "-m", "paper_trading.cli",
            "evaluate",
            "--batch", batch,
            "--output", output_dir
        ]
        
        if self.console and RICH_AVAILABLE:
            in_new_window = Confirm.ask("Run in new terminal window?", default=False)
        else:
            in_new_window = input("Run in new window? (y/n): ").lower() == 'y'
        
        print(f"\nThis will run both RC1 and RC2 on {batch} and compare results.")
        print("This may take a few minutes...")
        
        if in_new_window:
            name = f"evaluate_{batch}"
            title = f"Strategy Evaluation - {batch}"
            if self.process_manager.launch_in_terminal(name, cmd, title):
                print(f"\n[OK] Launched '{name}' in new window!")
            else:
                print("\n[ERROR] Failed to launch. Running in current window...")
                try:
                    subprocess.run(cmd)
                except KeyboardInterrupt:
                    pass
        else:
            try:
                subprocess.run(cmd)
            except KeyboardInterrupt:
                pass
        
        input("\nPress Enter to continue...")
    
    def run_kelly_report(self):
        """Run Kelly report generator."""
        if self.console and RICH_AVAILABLE:
            self.console.print(Panel("[bold blue]Robust Kelly Report[/bold blue]", border_style="blue"))
            self.console.print()
        else:
            print("\n=== Robust Kelly Report ===\n")
        
        # Find trade log files
        log_dir = Path("paper_trading/logs/replay")
        if not log_dir.exists():
            print("No replay logs found. Run a historical replay first.")
            input("Press Enter to continue...")
            return
        
        # Find all trades.csv files
        trade_files = list(log_dir.glob("**/trades.csv"))
        if not trade_files:
            print("No trade files found. Run with simulated data? (y/n)")
            choice = input().strip().lower()
            if choice != 'y':
                return
            trade_files = []
        
        # Select log file or use simulated
        log_path = None
        if trade_files:
            print("Available trade logs:")
            for i, f in enumerate(trade_files[:10], 1):
                print(f"  {i}. {f}")
            print(f"  {len(trade_files[:10])+1}. Use simulated data")
            print()
            choice = input("Select log (number): ").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(trade_files):
                    log_path = str(trade_files[idx])
        
        # Build command
        cmd = [sys.executable, "paper_trading/kelly_report.py"]
        if log_path:
            cmd.extend(["--log", log_path])
        
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            pass
        
        input("\nPress Enter to continue...")
    
    def run_sizing_comparison(self):
        """Run sizing comparison (fixed vs dynamic)."""
        if self.console and RICH_AVAILABLE:
            self.console.print(Panel("[bold blue]Sizing Comparison[/bold blue]", border_style="blue"))
            self.console.print()
        else:
            print("\n=== Sizing Comparison ===\n")
        
        # Import batch definitions
        try:
            from paper_trading.replay import BATCH_DEFINITIONS
            batch_list = sorted([k for k in BATCH_DEFINITIONS.keys() if not '+' in k])
        except ImportError:
            batch_list = ['batch4', 'batch5']
        
        # Select batch
        print("Available batches:")
        for i, b in enumerate(batch_list, 1):
            print(f"  {i}. {b}")
        print()
        batch_input = input(f"Batch (1-{len(batch_list)}) or name [batch4]: ").strip() or "batch4"
        if batch_input.isdigit():
            idx = int(batch_input) - 1
            batch = batch_list[idx] if 0 <= idx < len(batch_list) else "batch4"
        else:
            batch = batch_input
        
        # Get bankroll
        bankroll = input("Bankroll [$1000]: ").strip() or "1000"
        bankroll = float(bankroll)
        
        # Build command
        cmd = [
            sys.executable, "paper_trading/sizing_comparison.py",
            "--batch", batch,
            "--bankroll", str(bankroll),
        ]
        
        print()
        print(f"Running: {' '.join(cmd)}")
        print()
        
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            pass
        
        input("\nPress Enter to continue...")
    
    # =========================================================================
    # MAIN LOOP
    # =========================================================================
    
    def handle_process_manager(self):
        """Handle process manager menu."""
        while True:
            self.print_process_manager()
            running = self.process_manager.list_running()
            
            if not running:
                input()
                break
            
            choice = self.get_choice("Command").strip().lower()
            
            if choice in ("back", "b"):
                break
            elif choice in ('q', 'quit', 'exit'):
                # Quit from anywhere
                running = self.process_manager.list_running()
                if running:
                    if self.console and RICH_AVAILABLE:
                        if Confirm.ask(f"Stop {len(running)} running process(es) before quitting?"):
                            self.process_manager.stop_all()
                    else:
                        if input(f"Stop {len(running)} process(es)? (y/n): ").lower() == 'y':
                            self.process_manager.stop_all()
                
                self.running = False
                self.clear_screen()
                print("Goodbye!")
                return  # Exit the loop and program
            elif choice == "refresh":
                continue
            elif choice == "stop-all":
                self.process_manager.stop_all()
                print("All processes stopped.")
                input("Press Enter to continue...")
            elif choice.startswith("stop "):
                name = choice.split(" ", 1)[1]
                if self.process_manager.stop(name):
                    print(f"Stopped '{name}'")
                else:
                    print(f"Process '{name}' not found")
                input("Press Enter to continue...")
            else:
                print("Unknown command. Use: stop <name>, stop-all, refresh, back")
                input("Press Enter to continue...")
    
    def run(self):
        """Main loop."""
        while self.running:
            self.print_header()
            self.print_top_menu()
            
            choice = self.get_choice().strip().lower()
            
            if choice == '1':
                # Data Collection Mode
                self.current_mode = "data"
                self.run_data_collection_loop()
            elif choice == '2':
                # Paper Trading Mode
                self.current_mode = "trading"
                self.run_paper_trading_loop()
            elif choice == 'p':
                # Process Manager
                self.handle_process_manager()
            elif choice in ('q', 'quit', 'exit'):
                # Stop all processes before quitting
                running = self.process_manager.list_running()
                if running:
                    if self.console and RICH_AVAILABLE:
                        if Confirm.ask(f"Stop {len(running)} running process(es) before quitting?"):
                            self.process_manager.stop_all()
                    else:
                        if input(f"Stop {len(running)} process(es)? (y/n): ").lower() == 'y':
                            self.process_manager.stop_all()
                
                self.running = False
                self.clear_screen()
                print("Goodbye!")
            else:
                if self.console and RICH_AVAILABLE:
                    self.console.print(f"[red]Unknown option: {choice}[/red]")
                else:
                    print(f"Unknown option: {choice}")
                input("Press Enter to continue...")
    
    def run_data_collection_loop(self):
        """Run data collection menu loop."""
        while True:
            self.print_header("data")
            self.print_data_collection_menu()
            
            choice = self.get_choice().strip().lower()
            
            if choice == '1':
                self.collect_data_menu()
            elif choice == '2':
                self.process_data_menu()
            elif choice == '3':
                self.build_dataset_menu()
            elif choice == '4':
                self.validate_dataset_menu()
            elif choice == '5':
                self.view_status()
            elif choice == '6':
                self.full_workflow()
            elif choice == '7':
                self.delete_low_coverage_markets_menu()
            elif choice == 't':
                self.test_rtds_data()
            elif choice in ('b', 'back'):
                break
            elif choice in ('q', 'quit', 'exit'):
                # Quit from anywhere
                running = self.process_manager.list_running()
                if running:
                    if self.console and RICH_AVAILABLE:
                        if Confirm.ask(f"Stop {len(running)} running process(es) before quitting?"):
                            self.process_manager.stop_all()
                    else:
                        if input(f"Stop {len(running)} process(es)? (y/n): ").lower() == 'y':
                            self.process_manager.stop_all()
                
                self.running = False
                self.clear_screen()
                print("Goodbye!")
                return  # Exit the loop and program
            else:
                print(f"Unknown option: {choice}")
                input("Press Enter to continue...")
    
    def run_paper_trading_loop(self):
        """Run paper trading menu loop."""
        while True:
            self.print_header("trading")
            self.print_paper_trading_menu()
            
            choice = self.get_choice().strip().lower()
            
            if choice == '1':
                self.run_paper_trading()
            elif choice == '2':
                self.launch_orderbook()
            elif choice == '3':
                self.launch_stream()
            elif choice == '4':
                self.run_replay()
            elif choice == '5':
                self.run_visualize()
            elif choice == '6':
                self.run_evaluation()
            elif choice == '7':
                self.test_rtds_trading()
            elif choice == '8':
                self.test_data_alignment()
            elif choice in ('b', 'back'):
                break
            elif choice in ('q', 'quit', 'exit'):
                # Quit from anywhere
                running = self.process_manager.list_running()
                if running:
                    if self.console and RICH_AVAILABLE:
                        if Confirm.ask(f"Stop {len(running)} running process(es) before quitting?"):
                            self.process_manager.stop_all()
                    else:
                        if input(f"Stop {len(running)} process(es)? (y/n): ").lower() == 'y':
                            self.process_manager.stop_all()
                
                self.running = False
                self.clear_screen()
                print("Goodbye!")
                return  # Exit the loop and program
            else:
                print(f"Unknown option: {choice}")
                input("Press Enter to continue...")


def main():
    """Entry point."""
    import warnings
    warnings.filterwarnings("ignore", category=ResourceWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    cli = UnifiedCLI()
    
    try:
        cli.run()
    except KeyboardInterrupt:
        print("\nGoodbye!")
        # Clean up processes
        cli.process_manager.stop_all()
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
