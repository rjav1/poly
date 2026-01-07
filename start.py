#!/usr/bin/env python3
"""
Interactive CLI for Polymarket/Chainlink Data Collection

Run this file to get a menu-driven interface for all operations.
No need to memorize commands!

Usage: python start.py
"""

import sys
import os
import asyncio
import subprocess
from pathlib import Path

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


class InteractiveCLI:
    """Interactive menu-driven CLI."""
    
    def __init__(self):
        self.console = Console(force_terminal=True, legacy_windows=True) if RICH_AVAILABLE else None
        self.running = True
    
    def clear_screen(self):
        """Clear terminal screen."""
        os.system('cls' if sys.platform == 'win32' else 'clear')
    
    def print_header(self):
        """Print main header."""
        self.clear_screen()
        
        if self.console and RICH_AVAILABLE:
            header = Panel(
                "[bold cyan]Polymarket/Chainlink Data Collection[/bold cyan]\n"
                "[dim]Interactive CLI - No commands to memorize![/dim]",
                border_style="cyan",
                padding=(1, 2)
            )
            self.console.print(header)
            self.console.print()
        else:
            print("=" * 60)
            print("  Polymarket/Chainlink Data Collection")
            print("  Interactive CLI")
            print("=" * 60)
            print()
    
    def print_menu(self):
        """Print main menu options."""
        if self.console and RICH_AVAILABLE:
            table = Table(title="Main Menu", box=box.ROUNDED, show_header=False)
            table.add_column("Option", style="cyan", width=4)
            table.add_column("Action", style="white", width=30)
            table.add_column("Description", style="dim", width=40)
            
            table.add_row("1", "Collect Data", "Collect new data from Chainlink & Polymarket")
            table.add_row("2", "Process Data", "Convert raw data into market folders")
            table.add_row("3", "Build Dataset", "Create research dataset from markets")
            table.add_row("4", "Validate Dataset", "Run quality checks on dataset")
            table.add_row("5", "View Status", "Check current data status")
            table.add_row("6", "Full Workflow", "Run entire pipeline (1-4)")
            table.add_row("", "", "")
            table.add_row("q", "Quit", "Exit the CLI")
            
            self.console.print(table)
            self.console.print()
        else:
            print("Main Menu:")
            print("  1. Collect Data")
            print("  2. Process Data")
            print("  3. Build Dataset")
            print("  4. Validate Dataset")
            print("  5. View Status")
            print("  6. Full Workflow")
            print("  q. Quit")
            print()
    
    def get_choice(self) -> str:
        """Get user's menu choice."""
        if self.console and RICH_AVAILABLE:
            return Prompt.ask("[bold]Select option[/bold]", default="1")
        else:
            return input("Select option (1-6, q): ").strip()
    
    def collect_data_menu(self):
        """Menu for data collection options."""
        self.print_header()
        
        if self.console and RICH_AVAILABLE:
            self.console.print(Panel("[bold]Data Collection[/bold]", border_style="green"))
            self.console.print()
            
            # Show available assets
            self.console.print(f"[cyan]Available assets:[/cyan] {', '.join(SUPPORTED_ASSETS)}")
            self.console.print()
            
            # Asset selection
            assets_input = Prompt.ask(
                "Assets to collect",
                default="BTC",
                show_default=True
            )
            
            # Duration
            duration = IntPrompt.ask(
                "Number of matched data points per asset",
                default=60
            )
            
            # Max time
            max_time = IntPrompt.ask(
                "Max time limit in seconds (0 for no limit)",
                default=300
            )
            if max_time == 0:
                max_time = None
        else:
            print("Data Collection")
            print("-" * 40)
            print(f"Available assets: {', '.join(SUPPORTED_ASSETS)}")
            print()
            assets_input = input("Assets (comma-separated, default: BTC): ").strip() or "BTC"
            duration = int(input("Matched points per asset (default: 60): ") or "60")
            max_time_input = input("Max time in seconds (0=no limit, default: 300): ") or "300"
            max_time = int(max_time_input) if max_time_input != "0" else None
        
        # Parse and validate assets
        assets = [a.strip().upper() for a in assets_input.split(',')]
        valid_assets = [a for a in assets if a in SUPPORTED_ASSETS]
        
        if not valid_assets:
            if self.console and RICH_AVAILABLE:
                self.console.print(f"[red]No valid assets selected. Choose from: {SUPPORTED_ASSETS}[/red]")
            else:
                print(f"No valid assets. Choose from: {SUPPORTED_ASSETS}")
            input("Press Enter to continue...")
            return
        
        # Confirm
        if self.console and RICH_AVAILABLE:
            self.console.print()
            self.console.print(f"[bold]Will collect:[/bold]")
            self.console.print(f"  Assets: {', '.join(valid_assets)}")
            self.console.print(f"  Target points: {duration} per asset")
            self.console.print(f"  Max time: {max_time}s" if max_time else "  Max time: No limit")
            self.console.print()
            
            if not Confirm.ask("Start collection?", default=True):
                return
        else:
            print(f"\nWill collect: {valid_assets}, {duration} points, max {max_time}s")
            if input("Start? (y/n): ").lower() != 'y':
                return
        
        # Run collection
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "scripts" / "cli_v2.py"),
            "collect",
            "--assets", ",".join(valid_assets),
            "--duration", str(duration)
        ]
        if max_time:
            cmd.extend(["--max-time", str(max_time)])
        
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            pass
        
        input("\nPress Enter to continue...")
    
    def process_data_menu(self):
        """Menu for processing raw data."""
        self.print_header()
        
        if self.console and RICH_AVAILABLE:
            self.console.print(Panel("[bold]Process Raw Data[/bold]", border_style="yellow"))
            self.console.print()
            self.console.print("[dim]This converts continuous raw data into market folders.[/dim]")
            self.console.print()
            
            if not Confirm.ask("Process raw data?", default=True):
                return
        else:
            print("Process Raw Data")
            print("-" * 40)
            if input("Process? (y/n): ").lower() != 'y':
                return
        
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "scripts" / "cli_v2.py"),
            "process"
        ]
        
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            pass
        
        input("\nPress Enter to continue...")
    
    def build_dataset_menu(self):
        """Menu for building research dataset."""
        self.print_header()
        
        if self.console and RICH_AVAILABLE:
            self.console.print(Panel("[bold]Build Research Dataset[/bold]", border_style="blue"))
            self.console.print()
            self.console.print("[dim]Creates canonical research dataset from market folders.[/dim]")
            self.console.print()
            
            min_coverage = IntPrompt.ask(
                "Minimum coverage percentage",
                default=90
            )
            
            if not Confirm.ask("Build dataset?", default=True):
                return
        else:
            print("Build Research Dataset")
            print("-" * 40)
            min_coverage = int(input("Min coverage % (default: 90): ") or "90")
            if input("Build? (y/n): ").lower() != 'y':
                return
        
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "scripts" / "cli_v2.py"),
            "build",
            "--min-coverage", str(min_coverage)
        ]
        
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            pass
        
        input("\nPress Enter to continue...")
    
    def validate_dataset_menu(self):
        """Menu for validating dataset."""
        self.print_header()
        
        if self.console and RICH_AVAILABLE:
            self.console.print(Panel("[bold]Validate Dataset[/bold]", border_style="magenta"))
            self.console.print()
            self.console.print("[dim]Runs quality checks on the research dataset.[/dim]")
            self.console.print()
            
            if not Confirm.ask("Validate dataset?", default=True):
                return
        else:
            print("Validate Dataset")
            print("-" * 40)
            if input("Validate? (y/n): ").lower() != 'y':
                return
        
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "scripts" / "cli_v2.py"),
            "validate"
        ]
        
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            pass
        
        input("\nPress Enter to continue...")
    
    def view_status(self):
        """View current data status."""
        self.print_header()
        
        if self.console and RICH_AVAILABLE:
            self.console.print(Panel("[bold]Data Status[/bold]", border_style="cyan"))
            self.console.print()
        else:
            print("Data Status")
            print("-" * 40)
        
        # Check directories
        dirs_to_check = [
            ("Raw Data (Chainlink)", Path(STORAGE.raw_dir) / "chainlink"),
            ("Raw Data (Polymarket)", Path(STORAGE.raw_dir) / "polymarket"),
            ("Markets", Path(STORAGE.markets_dir)),
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
            
            # Show assets with data
            self.console.print()
            raw_cl = Path(STORAGE.raw_dir) / "chainlink"
            if raw_cl.exists():
                assets_with_data = [d.name for d in raw_cl.iterdir() if d.is_dir()]
                if assets_with_data:
                    self.console.print(f"[cyan]Assets with raw data:[/cyan] {', '.join(assets_with_data)}")
            
            # Show market count
            markets_dir = Path(STORAGE.markets_dir)
            if markets_dir.exists():
                market_count = len([d for d in markets_dir.rglob("*") if d.is_dir() and (d / "summary.json").exists()])
                self.console.print(f"[cyan]Market folders:[/cyan] {market_count}")
            
            # Show research datasets
            research_dir = Path(STORAGE.research_dir)
            if research_dir.exists():
                datasets = list(research_dir.glob("*.parquet"))
                if datasets:
                    self.console.print(f"[cyan]Research datasets:[/cyan] {', '.join(d.name for d in datasets)}")
        else:
            for name, path in dirs_to_check:
                if path.exists():
                    files = list(path.rglob("*.csv")) + list(path.rglob("*.parquet"))
                    print(f"  {name}: {len(files)} files")
                else:
                    print(f"  {name}: Not found")
        
        print()
        input("Press Enter to continue...")
    
    def full_workflow(self):
        """Run the complete workflow."""
        self.print_header()
        
        if self.console and RICH_AVAILABLE:
            self.console.print(Panel("[bold]Full Workflow[/bold]", border_style="green"))
            self.console.print()
            self.console.print("[bold]This will run the complete pipeline:[/bold]")
            self.console.print("  1. Collect data from Chainlink & Polymarket")
            self.console.print("  2. Process raw data into market folders")
            self.console.print("  3. Build research dataset")
            self.console.print("  4. Validate dataset")
            self.console.print()
            
            # Asset selection
            self.console.print(f"[cyan]Available assets:[/cyan] {', '.join(SUPPORTED_ASSETS)}")
            assets_input = Prompt.ask(
                "Assets to collect",
                default="BTC"
            )
            
            duration = IntPrompt.ask(
                "Matched points per asset",
                default=60
            )
            
            max_time = IntPrompt.ask(
                "Max time limit in seconds",
                default=300
            )
            
            if not Confirm.ask("\nStart full workflow?", default=True):
                return
        else:
            print("Full Workflow")
            print("-" * 40)
            assets_input = input("Assets (default: BTC): ").strip() or "BTC"
            duration = int(input("Points per asset (default: 60): ") or "60")
            max_time = int(input("Max time (default: 300): ") or "300")
            if input("Start? (y/n): ").lower() != 'y':
                return
        
        # Parse assets
        assets = [a.strip().upper() for a in assets_input.split(',')]
        valid_assets = [a for a in assets if a in SUPPORTED_ASSETS]
        
        if not valid_assets:
            if self.console and RICH_AVAILABLE:
                self.console.print(f"[red]No valid assets.[/red]")
            else:
                print("No valid assets.")
            input("Press Enter to continue...")
            return
        
        # Step 1: Collect
        if self.console and RICH_AVAILABLE:
            self.console.print("\n[bold cyan]Step 1/4: Collecting data...[/bold cyan]")
        else:
            print("\n--- Step 1/4: Collecting data ---")
        
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "scripts" / "cli_v2.py"),
            "collect",
            "--assets", ",".join(valid_assets),
            "--duration", str(duration),
            "--max-time", str(max_time)
        ]
        
        result = subprocess.run(cmd)
        if result.returncode != 0:
            if self.console and RICH_AVAILABLE:
                self.console.print("[red]Collection failed. Stopping workflow.[/red]")
            else:
                print("Collection failed.")
            input("Press Enter to continue...")
            return
        
        # Step 2: Process
        if self.console and RICH_AVAILABLE:
            self.console.print("\n[bold cyan]Step 2/4: Processing data...[/bold cyan]")
        else:
            print("\n--- Step 2/4: Processing data ---")
        
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "scripts" / "cli_v2.py"),
            "process"
        ]
        result = subprocess.run(cmd)
        
        # Step 3: Build
        if self.console and RICH_AVAILABLE:
            self.console.print("\n[bold cyan]Step 3/4: Building dataset...[/bold cyan]")
        else:
            print("\n--- Step 3/4: Building dataset ---")
        
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "scripts" / "cli_v2.py"),
            "build"
        ]
        result = subprocess.run(cmd)
        
        # Step 4: Validate
        if self.console and RICH_AVAILABLE:
            self.console.print("\n[bold cyan]Step 4/4: Validating dataset...[/bold cyan]")
        else:
            print("\n--- Step 4/4: Validating dataset ---")
        
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "scripts" / "cli_v2.py"),
            "validate"
        ]
        result = subprocess.run(cmd)
        
        if self.console and RICH_AVAILABLE:
            self.console.print("\n[bold green]Workflow complete![/bold green]")
        else:
            print("\n=== Workflow complete! ===")
        
        input("\nPress Enter to continue...")
    
    def run(self):
        """Main loop."""
        while self.running:
            self.print_header()
            self.print_menu()
            
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
            elif choice in ('q', 'quit', 'exit'):
                self.running = False
                self.clear_screen()
                print("Goodbye!")
            else:
                if self.console and RICH_AVAILABLE:
                    self.console.print(f"[red]Unknown option: {choice}[/red]")
                else:
                    print(f"Unknown option: {choice}")
                input("Press Enter to continue...")


def main():
    """Entry point."""
    import warnings
    warnings.filterwarnings("ignore", category=ResourceWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    cli = InteractiveCLI()
    
    try:
        cli.run()
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
