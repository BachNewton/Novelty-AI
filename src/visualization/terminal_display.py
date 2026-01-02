"""
Terminal Training Display - Rich-based in-place terminal UI for training.

Provides a clean terminal interface with:
- Progress bar for training episodes
- Training metrics (score, average, high score, epsilon)
- Hardware utilization (CPU, memory, GPU with per-core breakdown)
- All updating in-place without scrolling
"""

import os
import time
from typing import Optional, List
from collections import deque

import psutil
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text

from .hardware_monitor import HardwareMonitor


class TerminalTrainingDisplay:
    """
    Rich-based terminal display for training progress.

    Updates in-place without scrolling, showing:
    - Episode progress bar
    - Training metrics
    - Hardware utilization with per-core CPU breakdown
    """

    def __init__(
        self,
        total_episodes: int,
        game_name: str = "Game",
        num_envs: int = 1,
        update_interval: float = 0.5,
    ):
        """
        Initialize the terminal display.

        Args:
            total_episodes: Total episodes for training
            game_name: Name of the game being trained
            num_envs: Number of parallel environments
            update_interval: Minimum seconds between display updates
        """
        self.total_episodes = total_episodes
        self.game_name = game_name
        self.num_envs = num_envs
        self.update_interval = update_interval

        # Force UTF-8 encoding for Windows compatibility
        self.console = Console(force_terminal=True, legacy_windows=False, markup=True)
        self.live: Optional[Live] = None

        # Training state
        self.current_episode = 0
        self.last_scores: List[int] = [0] * num_envs  # Track score per environment
        self.high_score = 0
        self.epsilon = 1.0
        self.loss = 0.0
        self.recent_scores: deque = deque(maxlen=100)
        self.recent_scores_50: deque = deque(maxlen=50)  # For 50-episode rolling stats

        # Performance tracking
        self.start_time = time.time()
        self.last_update = 0.0
        self.episodes_per_sec = 0.0
        self.steps_per_sec = 0.0
        self.total_steps = 0

        # Hardware monitor
        self.hardware_monitor = HardwareMonitor(update_interval=1.0)

        # Messages queue for important events
        self.messages: deque = deque(maxlen=20)

    def start(self):
        """Start the live display."""
        self.hardware_monitor.start()
        self.start_time = time.time()
        self.live = Live(
            self._build_display(),
            console=self.console,
            refresh_per_second=4,
            transient=False,
        )
        self.live.start()

    def stop(self):
        """Stop the live display."""
        if self.live:
            self.live.stop()
            self.live = None
        self.hardware_monitor.stop()

    def update(
        self,
        episode: int,
        score: int,
        epsilon: float,
        loss: float,
        steps: int,
        env_index: int = 0,
    ):
        """
        Update the display with new training data.

        Args:
            episode: Current episode number
            score: Last completed episode score
            epsilon: Current epsilon value
            loss: Current loss value
            steps: Total steps taken
            env_index: Which environment completed (for tracking per-env scores)
        """
        self.current_episode = episode
        if 0 <= env_index < len(self.last_scores):
            self.last_scores[env_index] = score
        self.epsilon = epsilon
        self.loss = loss
        self.total_steps = steps

        # Calculate performance
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            self.episodes_per_sec = episode / elapsed
            self.steps_per_sec = steps / elapsed

        # Update display if enough time has passed
        now = time.time()
        if self.live and (now - self.last_update) >= self.update_interval:
            self.live.update(self._build_display())
            self.last_update = now

    def record_episode(self, score: int):
        """Record a completed episode score."""
        self.recent_scores.append(score)
        self.recent_scores_50.append(score)
        if score > self.high_score:
            self.high_score = score

        # Log metrics every 5 episodes
        if self.current_episode > 0 and self.current_episode % 5 == 0:
            avg_50 = sum(self.recent_scores_50) / len(self.recent_scores_50) if self.recent_scores_50 else 0
            min_50 = min(self.recent_scores_50) if self.recent_scores_50 else 0
            max_50 = max(self.recent_scores_50) if self.recent_scores_50 else 0
            loss_str = f"{self.loss:.6f}" if self.loss > 0 else "N/A"
            self.add_message(
                f"[dim]Ep {self.current_episode}:[/] "
                f"Avg={avg_50:.1f} Min={min_50} Max={max_50} "
                f"Eps={self.epsilon:.4f} Loss={loss_str}"
            )

    def add_message(self, message: str):
        """Add a message to the display."""
        self.messages.append(message)

    def _get_terminal_width(self) -> int:
        """Get the current terminal width."""
        try:
            return os.get_terminal_size().columns
        except OSError:
            return 80  # Default fallback

    def _build_display(self) -> Panel:
        """Build the complete display panel."""
        layout = Layout()

        # Calculate CPU rows needed (4 cores per row)
        cpu_count = psutil.cpu_count() or 4
        cpu_display_rows = (cpu_count + 3) // 4  # Ceiling division by 4

        # Create sections
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="progress", size=3),
            Layout(name="main", size=10 + cpu_display_rows),
            Layout(name="messages", size=23),
        )

        # Header
        header_text = Text()
        header_text.append(f"Novelty AI - {self.game_name} Training", style="bold cyan")
        layout["header"].update(Panel(header_text, style="cyan"))

        # Progress bar
        progress_ratio = self.current_episode / self.total_episodes if self.total_episodes > 0 else 0
        progress_bar = self._build_progress_bar(progress_ratio)
        layout["progress"].update(progress_bar)

        # Main content - metrics and hardware side by side
        layout["main"].split_row(
            Layout(name="metrics", ratio=1),
            Layout(name="hardware", ratio=1),
        )

        layout["main"]["metrics"].update(self._build_metrics_table())
        layout["main"]["hardware"].update(self._build_hardware_table())

        # Messages
        layout["messages"].update(self._build_messages_panel())

        return Panel(layout, title="Training Progress", border_style="blue")

    def _build_progress_bar(self, ratio: float) -> Panel:
        """Build the progress bar panel with dynamic width."""
        elapsed = time.time() - self.start_time
        remaining = (elapsed / ratio - elapsed) if ratio > 0 else 0

        # Dynamic bar width based on terminal size
        # Account for panel borders, episode text, percentage, and padding
        terminal_width = self._get_terminal_width()
        episode_text = f"Episode {self.current_episode}/{self.total_episodes} "
        percent_text = f" {ratio*100:.1f}%"
        # Subtract: panel borders (4), brackets (2), episode text, percent, some padding
        bar_width = max(20, terminal_width - len(episode_text) - len(percent_text) - 12)

        filled = int(bar_width * ratio)
        bar = "#" * filled + "-" * (bar_width - filled)

        progress_text = Text()
        progress_text.append(episode_text, style="bold")
        progress_text.append(f"[{bar}]", style="cyan")
        progress_text.append(percent_text, style="green")

        time_text = Text()
        time_text.append(f"Elapsed: {self._format_time(elapsed)} | ", style="dim")
        time_text.append(f"Remaining: {self._format_time(remaining)}", style="dim")

        combined = Text()
        combined.append_text(progress_text)
        combined.append("\n")
        combined.append_text(time_text)

        return Panel(combined, title="Progress", border_style="green")

    def _build_metrics_table(self) -> Panel:
        """Build the training metrics table."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="cyan", width=12)
        table.add_column("Value", style="white")

        # Calculate 50-episode rolling stats
        if self.recent_scores_50:
            avg_50 = sum(self.recent_scores_50) / len(self.recent_scores_50)
            min_50 = min(self.recent_scores_50)
            max_50 = max(self.recent_scores_50)
        else:
            avg_50 = min_50 = max_50 = 0

        # Show last completed episode score (most recent from any env)
        last_score = max(self.last_scores) if self.last_scores else 0
        table.add_row("Last Score", f"{last_score}")
        table.add_row("Avg (50)", f"{avg_50:.1f}")
        table.add_row("Min (50)", f"{min_50}")
        table.add_row("Max (50)", f"{max_50}")
        table.add_row("High Score", f"[bold yellow]{self.high_score}[/]")
        table.add_row("Epsilon", f"{self.epsilon:.4f}")

        loss_str = f"{self.loss:.6f}" if self.loss > 0 else "N/A"
        table.add_row("Loss", loss_str)

        table.add_row("", "")
        table.add_row("Episodes/sec", f"{self.episodes_per_sec:.1f}")
        table.add_row("Steps/sec", f"{self.steps_per_sec:.0f}")
        table.add_row("Total Steps", f"{self.total_steps:,}")
        table.add_row("Environments", f"{self.num_envs}")

        return Panel(table, title="Training Metrics", border_style="yellow")

    def _build_hardware_table(self) -> Panel:
        """Build the hardware utilization table with per-core CPU breakdown."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Resource", style="cyan")
        table.add_column("Usage", style="white")

        hw = self.hardware_monitor.get_stats()

        if hw is None:
            table.add_row("Status", "[dim]Initializing...[/]")
            return Panel(table, title="Hardware", border_style="magenta")

        # CPU total
        cpu_percent = hw.cpu.total_percent
        cpu_bar = self._make_bar(cpu_percent)
        table.add_row("CPU Total", f"{cpu_bar} {cpu_percent:.0f}%")

        # Per-core CPU breakdown (show in rows of 4 cores)
        per_core = psutil.cpu_percent(percpu=True)
        cores_per_row = 4
        for row_start in range(0, len(per_core), cores_per_row):
            row_end = min(row_start + cores_per_row, len(per_core))
            core_strs = []
            for i in range(row_start, row_end):
                pct = per_core[i]
                # Color code: green < 50%, yellow 50-80%, red > 80%
                if pct < 50:
                    color = "green"
                elif pct < 80:
                    color = "yellow"
                else:
                    color = "red"
                core_strs.append(f"[{color}]{i:02d}:{pct:3.0f}%[/]")
            label = f"Cores {row_start}-{row_end-1}" if row_end - row_start > 1 else f"Core {row_start}"
            table.add_row(label, " ".join(core_strs))

        # Memory
        mem = psutil.virtual_memory()
        mem_percent = mem.percent
        mem_bar = self._make_bar(mem_percent)
        mem_used = mem.used / (1024 ** 3)
        mem_total = mem.total / (1024 ** 3)
        table.add_row("Memory", f"{mem_bar} {mem_percent:.0f}% ({mem_used:.1f}/{mem_total:.1f} GB)")

        # GPU (if available)
        if hw.gpu and hw.gpu.available:
            gpu_percent = hw.gpu.utilization_percent
            gpu_bar = self._make_bar(gpu_percent)
            table.add_row("GPU", f"{gpu_bar} {gpu_percent:.0f}%")

            gpu_mem = hw.gpu.memory_percent
            gpu_mem_bar = self._make_bar(gpu_mem)
            table.add_row("GPU Mem", f"{gpu_mem_bar} {gpu_mem:.0f}%")
        else:
            table.add_row("GPU", "[dim]Not available[/]")

        return Panel(table, title="Hardware", border_style="magenta")

    def _build_messages_panel(self) -> Panel:
        """Build the messages panel with Rich markup support."""
        from rich.text import Text
        from rich.console import Console

        if not self.messages:
            content = "[dim]No messages yet...[/]"
        else:
            # Use console.render_str to parse Rich markup in messages
            lines = []
            for msg in self.messages:
                lines.append(msg)
            # Join with newlines and let Rich parse the markup
            content = "\n".join(lines)

        return Panel(content, title="Messages", border_style="blue")

    def _make_bar(self, percent: float, width: int = 10) -> str:
        """Create a mini progress bar (ASCII for Windows compatibility)."""
        filled = int(width * percent / 100)
        return "#" * filled + "-" * (width - filled)

    def _format_time(self, seconds: float) -> str:
        """Format seconds into HH:MM:SS."""
        if seconds < 0:
            return "--:--:--"
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"
