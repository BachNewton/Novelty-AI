"""
Training Dashboard - Real-time visualization of training progress.

Combines game rendering (optional), training metrics, charts, and hardware monitoring
in a single pygame window. Supports both visual and headless modes.
"""
import pygame
import numpy as np
import time
from collections import deque
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from .hardware_monitor import HardwareMonitor
from .ui_components import (
    BG_COLOR, PANEL_COLOR, TEXT_COLOR, ACCENT_COLOR,
    WARNING_COLOR, DANGER_COLOR, CHART_BG, SUCCESS_COLOR
)
from ..core.renderer_interface import RendererInterface
from ..games.registry import GameRegistry


@dataclass
class TrainingMetrics:
    """Container for training metrics history."""
    episodes: List[int] = field(default_factory=list)
    scores: List[int] = field(default_factory=list)
    avg_scores: List[float] = field(default_factory=list)
    epsilons: List[float] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    eps_per_sec: List[float] = field(default_factory=list)
    eps_per_sec_episodes: List[int] = field(default_factory=list)

    high_score: int = 0
    total_steps: int = 0

    # Rolling window for averages
    recent_scores: deque = field(default_factory=lambda: deque(maxlen=100))


class TrainingDashboard:
    """
    Real-time training dashboard with multiple panels:
    - Game visualization OR training status (left)
    - Training metrics (top right)
    - Score/epsilon charts (middle right)
    - Hardware utilization (bottom)

    Supports both visual mode (shows game) and headless mode (shows stats).
    """

    MIN_WIDTH = 800
    MIN_HEIGHT = 600

    def __init__(
        self,
        screen: Optional[pygame.Surface] = None,
        window_width: int = 1400,
        window_height: int = 900,
        grid_width: int = 20,
        grid_height: int = 20,
        chart_update_interval: int = 10,
        show_game: bool = True,
        total_episodes: int = 10000,
        num_envs: int = 1,
        game_id: str = 'snake'
    ):
        """
        Initialize the dashboard.

        Args:
            screen: Existing pygame surface (if None, creates new window)
            window_width: Window width in pixels
            window_height: Window height in pixels
            grid_width: Game grid width
            grid_height: Game grid height
            chart_update_interval: Update charts every N episodes
            show_game: If True, show game panel; if False, show training stats
            total_episodes: Total episodes for progress calculation
            num_envs: Number of parallel environments being used
            game_id: Game identifier for creating the correct renderer
        """
        self.owns_screen = screen is None
        self.window_width = max(window_width, self.MIN_WIDTH)
        self.window_height = max(window_height, self.MIN_HEIGHT)
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.show_game = show_game
        self.total_episodes = total_episodes
        self.num_envs = num_envs
        self.game_id = game_id

        if screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.window_width, self.window_height),
                pygame.RESIZABLE
            )
            pygame.display.set_caption("Novelty AI - Training Dashboard")
        else:
            self.screen = screen
            self.window_width = screen.get_width()
            self.window_height = screen.get_height()

        # Fonts
        self.font_title = pygame.font.Font(None, 36)
        self.font_large = pygame.font.Font(None, 32)
        self.font_medium = pygame.font.Font(None, 26)
        self.font_small = pygame.font.Font(None, 20)

        # Calculate layout
        self._recalculate_layout()

        # Hardware monitor
        self.hardware_monitor = HardwareMonitor(update_interval=1.0)
        self.hardware_monitor.start()

        # Metrics
        self.metrics = TrainingMetrics()

        # Performance tracking (for headless mode)
        self.start_time = time.time()
        self.episodes_per_sec = 0.0
        self.steps_per_sec = 0.0
        self.total_steps = 0
        self.last_step_count = 0
        self.last_perf_update = time.time()

        # Clock for FPS limiting
        self.clock = pygame.time.Clock()

    def _recalculate_layout(self):
        """
        Recalculate panel positions based on current window size.

        Layout:
        +------------------+------------------+
        |  Game/Status     |  Training        |
        |  (top left)      |  Progress        |
        +------------------+------------------+
        |     Hardware Utilization (full)     |
        |     CPU cores    |    GPU stats     |
        +-----------------------------------------+
        |         Score Chart (full width)    |
        +-----------------------------------------+
        |    Epsilon      |   Episodes/sec    |
        +-----------------------------------------+
        """
        w, h = self.window_width, self.window_height
        gap = 10
        padding = 20

        # Top section: Game/Status (left), Metrics (right) - compact
        top_section_height = min(220, int(h * 0.25))

        # Hardware panel: full width below top section
        hardware_height = 160

        # Chart section gets the rest
        chart_height = h - top_section_height - hardware_height - gap * 2 - padding * 2

        # Left panel (game/status)
        left_panel_width = min(400, (w - gap - padding * 2) // 2)
        self.game_cell_size = left_panel_width // max(self.grid_width, self.grid_height)
        self.game_size = self.game_cell_size * max(self.grid_width, self.grid_height)

        self.game_rect = pygame.Rect(
            padding, padding,
            left_panel_width, top_section_height
        )

        # Metrics panel (right of game/status)
        right_panel_x = padding + left_panel_width + gap
        right_panel_width = w - right_panel_x - padding

        self.metrics_rect = pygame.Rect(
            right_panel_x, padding,
            right_panel_width, top_section_height
        )

        # Hardware panel - full width below top section
        self.hardware_rect = pygame.Rect(
            padding, padding + top_section_height + gap,
            w - padding * 2, hardware_height
        )

        # Charts panel - full width at bottom
        self.chart_rect = pygame.Rect(
            padding, padding + top_section_height + hardware_height + gap * 2,
            w - padding * 2, chart_height
        )

        # Create/update game renderer if showing game
        if self.show_game:
            self.game_renderer = GameRegistry.create_renderer(self.game_id)
            self.game_renderer.set_cell_size(self.game_cell_size)
            self.game_renderer.set_render_area(
                self.game_rect.x, self.game_rect.y,
                self.game_rect.width, self.game_rect.height
            )

    def set_screen(self, screen: pygame.Surface):
        """Update the screen surface (used when window is shared)."""
        self.screen = screen
        self.window_width = screen.get_width()
        self.window_height = screen.get_height()
        self._recalculate_layout()

    def toggle_game_view(self):
        """Toggle between game view and stats view."""
        self.show_game = not self.show_game
        if self.show_game:
            self._recalculate_layout()

    def record_episode(
        self,
        episode: int,
        score: int,
        epsilon: float,
        loss: Optional[float] = None
    ):
        """Record metrics for a completed episode with the final score."""
        self._update_metrics(episode, score, epsilon, loss)

    def update(
        self,
        game_state: Optional[Dict[str, Any]],
        episode: int,
        score: int,
        epsilon: float,
        loss: Optional[float] = None,
        steps: int = 0,
        fps: int = 30,
        exploration_label: str = "Epsilon"
    ) -> Any:
        """
        Update the dashboard with current training state.

        Args:
            game_state: Current game state dict (used if show_game=True)
            episode: Current episode number
            score: Current/final score
            epsilon: Current exploration rate (or entropy for PPO)
            loss: Latest training loss
            steps: Total training steps (used for stats display)
            fps: Target frame rate
            exploration_label: Label for exploration metric ("Epsilon" for DQN, "Entropy" for PPO)

        Returns:
            False if window was closed
            'switch_mode' if H was pressed
            True otherwise
        """
        self._exploration_label = exploration_label
        # Track high score continuously
        if score > self.metrics.high_score:
            self.metrics.high_score = score

        # Update performance metrics
        self.total_steps = steps
        self._update_performance(episode)

        # In headless mode, only update UI periodically for speed
        should_render = self.show_game
        if not self.show_game:
            now = time.time()
            if not hasattr(self, '_last_render_time'):
                self._last_render_time = 0.0
            if now - self._last_render_time >= 0.1:
                self._last_render_time = now
                should_render = True

        if not should_render:
            return True  # Skip event handling and rendering for max speed

        # Handle pygame events (only when rendering for performance)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_h:
                    self.toggle_game_view()
                    return 'switch_mode'
            elif event.type == pygame.VIDEORESIZE:
                self.window_width = max(event.w, self.MIN_WIDTH)
                self.window_height = max(event.h, self.MIN_HEIGHT)
                if self.owns_screen:
                    self.screen = pygame.display.set_mode(
                        (self.window_width, self.window_height),
                        pygame.RESIZABLE
                    )
                self._recalculate_layout()

        # Clear screen
        self.screen.fill(BG_COLOR)

        # Draw panels
        if self.show_game:
            if game_state:
                self._draw_game_panel(game_state)
            else:
                self._draw_game_placeholder()
        else:
            self._draw_status_panel(episode, score)

        self._draw_metrics_panel(episode, score, epsilon, loss)
        self._draw_chart_panel()
        self._draw_hardware_panel()

        pygame.display.flip()

        # Only limit FPS when showing game - headless runs as fast as possible
        if self.show_game:
            self.clock.tick(fps)

        return True

    def _update_performance(self, episode: int):
        """Update performance metrics (episodes/sec, steps/sec)."""
        now = time.time()
        elapsed = now - self.last_perf_update

        if elapsed >= 1.0:
            # Calculate episodes completed since last update
            current_episode_count = len(self.metrics.episodes)
            if not hasattr(self, '_last_episode_count'):
                self._last_episode_count = 0
            episodes_completed = current_episode_count - self._last_episode_count
            self.episodes_per_sec = episodes_completed / elapsed
            self._last_episode_count = current_episode_count

            # Store episodes/sec history for charting
            if self.episodes_per_sec > 0 and current_episode_count > 0:
                self.metrics.eps_per_sec.append(self.episodes_per_sec)
                self.metrics.eps_per_sec_episodes.append(current_episode_count)

            # Calculate steps completed since last update
            steps_completed = self.total_steps - self.last_step_count
            self.steps_per_sec = steps_completed / elapsed

            self.last_step_count = self.total_steps
            self.last_perf_update = now

    def _update_metrics(
        self,
        episode: int,
        score: int,
        epsilon: float,
        loss: Optional[float]
    ):
        """Update metrics with new episode data."""
        self.metrics.episodes.append(episode)
        self.metrics.scores.append(score)
        self.metrics.recent_scores.append(score)
        self.metrics.epsilons.append(epsilon)

        if loss is not None:
            self.metrics.losses.append(loss)

        avg = np.mean(list(self.metrics.recent_scores))
        self.metrics.avg_scores.append(avg)

        if score > self.metrics.high_score:
            self.metrics.high_score = score

    def _draw_game_panel(self, game_state: Dict[str, Any]):
        """Draw the game visualization."""
        pygame.draw.rect(self.screen, PANEL_COLOR, self.game_rect, border_radius=8)
        self.game_renderer.render(game_state, self.screen)

        # Mode hint
        hint = self.font_small.render("Press H for headless mode", True, (100, 100, 100))
        self.screen.blit(hint, (self.game_rect.x + 10, self.game_rect.bottom - 25))

    def _draw_game_placeholder(self):
        """Draw placeholder when no game state available."""
        pygame.draw.rect(self.screen, PANEL_COLOR, self.game_rect, border_radius=8)
        text = self.font_large.render("Waiting for game...", True, (100, 100, 100))
        self.screen.blit(
            text,
            (self.game_rect.centerx - text.get_width() // 2,
             self.game_rect.centery - text.get_height() // 2)
        )

    def _draw_status_panel(self, episode: int, score: int):
        """Draw training status panel (headless mode)."""
        pygame.draw.rect(self.screen, PANEL_COLOR, self.game_rect, border_radius=8)

        x = self.game_rect.x + 15
        y = self.game_rect.y + 15
        panel_bottom = self.game_rect.bottom - 15

        # Title
        title = self.font_large.render("Headless Training", True, ACCENT_COLOR)
        self.screen.blit(title, (x, y))
        y += 30

        # Mode hint
        hint = self.font_small.render("Press H to show game", True, (150, 150, 150))
        self.screen.blit(hint, (x, y))
        y += 25

        # Performance stats - use two columns to save vertical space
        stats_left = [
            ("Envs:", f"{self.num_envs}"),
            ("Eps/sec:", f"{self.episodes_per_sec:.1f}"),
            ("Steps/sec:", f"{self.steps_per_sec:.0f}"),
        ]

        # Elapsed
        elapsed = time.time() - self.start_time
        if elapsed < 60:
            elapsed_str = f"{elapsed:.0f}s"
        elif elapsed < 3600:
            elapsed_str = f"{elapsed / 60:.1f}m"
        else:
            elapsed_str = f"{elapsed / 3600:.1f}h"

        # ETA
        eta_str = "--"
        if self.episodes_per_sec > 0 and episode < self.total_episodes:
            remaining = self.total_episodes - episode
            eta_seconds = remaining / self.episodes_per_sec
            if eta_seconds < 60:
                eta_str = f"{eta_seconds:.0f}s"
            elif eta_seconds < 3600:
                eta_str = f"{eta_seconds / 60:.1f}m"
            else:
                eta_str = f"{eta_seconds / 3600:.1f}h"

        stats_right = [
            ("Elapsed:", elapsed_str),
            ("ETA:", eta_str),
            ("Steps:", f"{self.total_steps:,}"),
        ]

        col_width = (self.game_rect.width - 30) // 2
        row_height = 24

        for i, (label, value) in enumerate(stats_left):
            label_surf = self.font_small.render(label, True, TEXT_COLOR)
            value_surf = self.font_medium.render(value, True, ACCENT_COLOR)
            self.screen.blit(label_surf, (x, y + i * row_height))
            self.screen.blit(value_surf, (x + 70, y + i * row_height - 2))

        for i, (label, value) in enumerate(stats_right):
            label_surf = self.font_small.render(label, True, TEXT_COLOR)
            value_surf = self.font_medium.render(value, True, ACCENT_COLOR)
            self.screen.blit(label_surf, (x + col_width, y + i * row_height))
            self.screen.blit(value_surf, (x + col_width + 70, y + i * row_height - 2))

        y += len(stats_left) * row_height + 15

        # Progress bar at bottom of panel
        progress = episode / self.total_episodes * 100 if self.total_episodes > 0 else 0
        bar_width = self.game_rect.width - 30
        bar_height = 18
        bar_y = min(y, panel_bottom - bar_height - 20)

        # Progress label
        progress_label = self.font_small.render(f"Progress: {progress:.1f}%", True, TEXT_COLOR)
        self.screen.blit(progress_label, (x, bar_y - 18))

        bg_rect = pygame.Rect(x, bar_y, bar_width, bar_height)
        pygame.draw.rect(self.screen, (50, 50, 60), bg_rect, border_radius=4)

        fill_width = int(progress / 100 * bar_width)
        if fill_width > 0:
            fill_rect = pygame.Rect(x, bar_y, fill_width, bar_height)
            pygame.draw.rect(self.screen, ACCENT_COLOR, fill_rect, border_radius=4)

        progress_text = self.font_small.render(f"{episode}/{self.total_episodes}", True, TEXT_COLOR)
        self.screen.blit(progress_text, (x + bar_width // 2 - progress_text.get_width() // 2, bar_y + 1))

    def _draw_metrics_panel(
        self,
        episode: int,
        score: int,
        epsilon: float,
        loss: Optional[float]
    ):
        """Draw current training metrics."""
        pygame.draw.rect(self.screen, PANEL_COLOR, self.metrics_rect, border_radius=8)

        x = self.metrics_rect.x + 20
        y = self.metrics_rect.y + 15

        title = self.font_title.render("Training Progress", True, ACCENT_COLOR)
        self.screen.blit(title, (x, y))
        y += 40

        avg_score = 0
        if self.metrics.recent_scores:
            avg_score = np.mean(list(self.metrics.recent_scores))

        # Get exploration label (Epsilon for DQN, Entropy for PPO)
        explore_label = getattr(self, '_exploration_label', 'Epsilon')
        # For epsilon, warn if high; for entropy, warn if low
        if explore_label == 'Epsilon':
            explore_color = WARNING_COLOR if epsilon > 0.1 else TEXT_COLOR
        else:  # Entropy
            explore_color = WARNING_COLOR if epsilon < 0.5 else TEXT_COLOR

        metrics_list = [
            (f"Episode: {episode}", TEXT_COLOR),
            (f"Current Score: {score}", TEXT_COLOR),
            (f"High Score: {self.metrics.high_score}", ACCENT_COLOR),
            (f"Avg Score (100): {avg_score:.1f}", TEXT_COLOR),
            (f"{explore_label}: {epsilon:.4f}", explore_color),
            (f"Parallel Envs: {self.num_envs}", ACCENT_COLOR if self.num_envs > 1 else TEXT_COLOR),
        ]

        col_width = (self.metrics_rect.width - 40) // 2
        for i, (text, color) in enumerate(metrics_list):
            col = i % 2
            row = i // 2
            text_surf = self.font_medium.render(text, True, color)
            self.screen.blit(text_surf, (x + col * col_width, y + row * 28))

    def _draw_chart_panel(self):
        """Draw training progress charts - Score on top row, Epsilon + Eps/sec on bottom row."""
        pygame.draw.rect(self.screen, PANEL_COLOR, self.chart_rect, border_radius=8)

        if len(self.metrics.episodes) < 2:
            text = self.font_medium.render("Collecting data...", True, (100, 100, 100))
            self.screen.blit(
                text,
                (self.chart_rect.centerx - text.get_width() // 2,
                 self.chart_rect.centery - text.get_height() // 2)
            )
            return

        # Layout: Score (full width top), Epsilon + Episodes/sec (side by side bottom)
        y_label_space = 50  # Space for y-axis labels on left
        gap = 30  # Gap between charts for labels
        padding_top = 15
        padding_right = 25
        padding_bottom = 25  # Extra space for x-axis labels

        # Calculate chart dimensions - 2 rows
        full_width = self.chart_rect.width - y_label_space - padding_right
        half_width = (full_width - gap) // 2
        available_height = self.chart_rect.height - padding_top - padding_bottom - gap
        chart_h = available_height // 2

        chart_x = self.chart_rect.x + y_label_space

        # Score chart (top row - full width)
        chart1_y = self.chart_rect.y + padding_top
        self._draw_line_chart(
            x=chart_x, y=chart1_y, width=full_width, height=chart_h,
            data=self.metrics.scores,
            avg_data=self.metrics.avg_scores,
            color=(0, 200, 100),
            avg_color=(150, 255, 150),
            label="Score",
            show_avg=True,
            episodes=self.metrics.episodes
        )

        # Bottom row - Exploration metric (left) and Episodes/sec (right)
        row2_y = chart1_y + chart_h + gap

        # Get exploration label (Epsilon for DQN, Entropy for PPO)
        explore_chart_label = getattr(self, '_exploration_label', 'Epsilon')
        # Entropy can go up to ~2.1 for 8 actions (log(8)), Epsilon is 0-1
        explore_y_range = (0, 1) if explore_chart_label == 'Epsilon' else (0, 2.5)

        # Exploration chart (bottom left)
        self._draw_line_chart(
            x=chart_x, y=row2_y, width=half_width, height=chart_h,
            data=self.metrics.epsilons,
            color=(0, 200, 255),
            label=explore_chart_label,
            y_range=explore_y_range,
            episodes=self.metrics.episodes
        )

        # Episodes/sec chart (bottom right)
        chart3_x = chart_x + half_width + gap
        if len(self.metrics.eps_per_sec) >= 2:
            self._draw_line_chart(
                x=chart3_x, y=row2_y, width=half_width, height=chart_h,
                data=self.metrics.eps_per_sec,
                color=(255, 180, 0),
                label="Episodes/sec",
                episodes=self.metrics.eps_per_sec_episodes
            )
        else:
            # Draw placeholder for eps/sec chart
            chart_rect = pygame.Rect(chart3_x, row2_y, half_width, chart_h)
            pygame.draw.rect(self.screen, (40, 40, 50), chart_rect, border_radius=4)
            label = self.font_small.render("Episodes/sec", True, (255, 180, 0))
            self.screen.blit(label, (chart3_x + 5, row2_y + 5))
            waiting = self.font_small.render("Collecting...", True, (100, 100, 100))
            self.screen.blit(waiting, (chart3_x + half_width // 2 - waiting.get_width() // 2,
                                       row2_y + chart_h // 2))

    def _draw_line_chart(
        self,
        x: int, y: int, width: int, height: int,
        data: List,
        color: tuple,
        label: str,
        avg_data: Optional[List] = None,
        avg_color: Optional[tuple] = None,
        y_range: Optional[tuple] = None,
        show_avg: bool = False,
        episodes: Optional[List] = None
    ):
        """Draw a simple line chart."""
        if len(data) < 2:
            return

        bg_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, (40, 40, 50), bg_rect, border_radius=4)

        for i in range(5):
            gy = y + int(height * i / 4)
            pygame.draw.line(self.screen, (60, 60, 70), (x, gy), (x + width, gy))

        if y_range:
            min_val, max_val = y_range
        else:
            min_val = min(data)
            max_val = max(data)
            if max_val == min_val:
                max_val = min_val + 1

        max_points = 500
        if len(data) > max_points:
            step = len(data) // max_points
            plot_data = data[::step]
            plot_avg = avg_data[::step] if avg_data else None
            plot_episodes = episodes[::step] if episodes else None
        else:
            plot_data = data
            plot_avg = avg_data
            plot_episodes = episodes

        def to_screen(i, val):
            sx = x + int(i / max(1, len(plot_data) - 1) * width)
            sy = y + height - int((val - min_val) / (max_val - min_val) * height)
            return (sx, max(y, min(y + height, sy)))

        if show_avg and plot_avg and avg_color and len(plot_avg) >= 2:
            avg_points = [to_screen(i, v) for i, v in enumerate(plot_avg)]
            if len(avg_points) >= 2:
                pygame.draw.lines(self.screen, avg_color, False, avg_points, 2)

        points = [to_screen(i, v) for i, v in enumerate(plot_data)]
        if len(points) >= 2:
            pygame.draw.lines(self.screen, color, False, points, 1)

        label_surf = self.font_small.render(label, True, color)
        self.screen.blit(label_surf, (x + 5, y + 5))

        current_val = data[-1]
        val_text = f"{current_val:.2f}" if isinstance(current_val, float) else str(current_val)
        val_surf = self.font_small.render(val_text, True, TEXT_COLOR)
        self.screen.blit(val_surf, (x + width - val_surf.get_width() - 5, y + 5))

        # Draw y-axis labels for all 5 grid lines
        for i in range(5):
            val = max_val - (max_val - min_val) * i / 4
            gy = y + int(height * i / 4)
            # Format based on value magnitude
            if abs(val) >= 100:
                val_text = f"{val:.0f}"
            elif abs(val) >= 10:
                val_text = f"{val:.1f}"
            else:
                val_text = f"{val:.2f}"
            val_surf = self.font_small.render(val_text, True, (100, 100, 100))
            self.screen.blit(val_surf, (x - val_surf.get_width() - 5, gy - 7))

        # Draw x-axis episode labels (scale count with chart width)
        if plot_episodes and len(plot_episodes) >= 2:
            num_labels = max(3, min(width // 80, 15))
            for i in range(num_labels):
                idx = int(i * (len(plot_episodes) - 1) / (num_labels - 1))
                ep = plot_episodes[idx]
                sx = x + int(idx / max(1, len(plot_episodes) - 1) * width)
                ep_text = str(ep)
                ep_surf = self.font_small.render(ep_text, True, (100, 100, 100))
                self.screen.blit(ep_surf, (sx - ep_surf.get_width() // 2, y + height + 2))

    def _draw_hardware_panel(self):
        """Draw hardware utilization panel (full width)."""
        pygame.draw.rect(self.screen, PANEL_COLOR, self.hardware_rect, border_radius=8)

        x = self.hardware_rect.x + 20
        y = self.hardware_rect.y + 15

        title = self.font_title.render("Hardware Utilization", True, ACCENT_COLOR)
        self.screen.blit(title, (x, y))
        y += 35

        stats = self.hardware_monitor.get_stats()

        # CPU Section (left side)
        cpu_title = self.font_medium.render("CPU (per core)", True, TEXT_COLOR)
        self.screen.blit(cpu_title, (x, y))
        cpu_y = y + 25

        bar_width = 80
        bar_height = 14
        label_width = 22
        pct_width = 40
        item_width = label_width + bar_width + pct_width + 10
        cores_per_row = 4  # Fixed 4 columns

        if stats and stats.cpu:
            for i, percent in enumerate(stats.cpu.per_core_percent):
                col = i % cores_per_row
                row = i // cores_per_row

                bx = x + col * item_width
                by = cpu_y + row * (bar_height + 8)

                label = self.font_small.render(f"{i}", True, (150, 150, 150))
                label_x = bx + label_width - label.get_width()
                self.screen.blit(label, (label_x, by))

                bar_x = bx + label_width + 5
                bg_rect = pygame.Rect(bar_x, by, bar_width, bar_height)
                pygame.draw.rect(self.screen, (50, 50, 60), bg_rect, border_radius=3)

                fill_width = int(percent / 100 * bar_width)
                if fill_width > 0:
                    bar_color = ACCENT_COLOR if percent < 80 else WARNING_COLOR if percent < 95 else DANGER_COLOR
                    fill_rect = pygame.Rect(bar_x, by, fill_width, bar_height)
                    pygame.draw.rect(self.screen, bar_color, fill_rect, border_radius=3)

                pct_text = self.font_small.render(f"{percent:.0f}%", True, TEXT_COLOR)
                self.screen.blit(pct_text, (bar_x + bar_width + 5, by))

        # GPU Section (right side) - position after CPU section
        gpu_x = x + cores_per_row * item_width + 60
        gpu_y = y

        gpu_info = self.hardware_monitor.get_gpu_info()

        gpu_title = self.font_medium.render(f"GPU: {gpu_info['name'][:30]}", True, TEXT_COLOR)
        self.screen.blit(gpu_title, (gpu_x, gpu_y))
        gpu_y += 25

        if gpu_info['available']:
            util_label = self.font_small.render("Utilization:", True, (150, 150, 150))
            self.screen.blit(util_label, (gpu_x, gpu_y))

            gpu_bar_width = 150
            bg_rect = pygame.Rect(gpu_x + 80, gpu_y, gpu_bar_width, bar_height)
            pygame.draw.rect(self.screen, (50, 50, 60), bg_rect, border_radius=3)

            util = gpu_info['utilization']
            fill_width = int(util / 100 * gpu_bar_width)
            if fill_width > 0:
                bar_color = ACCENT_COLOR if util < 80 else WARNING_COLOR if util < 95 else DANGER_COLOR
                fill_rect = pygame.Rect(gpu_x + 80, gpu_y, fill_width, bar_height)
                pygame.draw.rect(self.screen, bar_color, fill_rect, border_radius=3)

            pct_text = self.font_small.render(f"{util:.0f}%", True, TEXT_COLOR)
            self.screen.blit(pct_text, (gpu_x + 85 + gpu_bar_width, gpu_y))
            gpu_y += 25

            mem_label = self.font_small.render("Memory:", True, (150, 150, 150))
            self.screen.blit(mem_label, (gpu_x, gpu_y))

            bg_rect = pygame.Rect(gpu_x + 80, gpu_y, gpu_bar_width, bar_height)
            pygame.draw.rect(self.screen, (50, 50, 60), bg_rect, border_radius=3)

            mem_pct = gpu_info['memory_percent']
            fill_width = int(mem_pct / 100 * gpu_bar_width)
            if fill_width > 0:
                bar_color = ACCENT_COLOR if mem_pct < 80 else WARNING_COLOR if mem_pct < 95 else DANGER_COLOR
                fill_rect = pygame.Rect(gpu_x + 80, gpu_y, fill_width, bar_height)
                pygame.draw.rect(self.screen, bar_color, fill_rect, border_radius=3)

            if gpu_info['memory_total'] > 0:
                mem_text = f"{gpu_info['memory_used']:.0f}/{gpu_info['memory_total']:.0f} MB"
            else:
                mem_text = f"{mem_pct:.0f}%"
            mem_surf = self.font_small.render(mem_text, True, TEXT_COLOR)
            self.screen.blit(mem_surf, (gpu_x + 85 + gpu_bar_width, gpu_y))
        else:
            no_gpu = self.font_small.render("GPU stats not available", True, (100, 100, 100))
            self.screen.blit(no_gpu, (gpu_x, gpu_y))

    def get_high_score(self) -> int:
        """Get the current high score."""
        return self.metrics.high_score

    def close(self):
        """Clean up resources."""
        self.hardware_monitor.stop()
        if self.owns_screen:
            pygame.quit()
