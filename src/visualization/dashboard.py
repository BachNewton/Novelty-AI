"""
Training Dashboard - Real-time visualization of training progress.

Combines game rendering, training metrics, charts, and hardware monitoring
in a single pygame window.
"""
import pygame
import numpy as np
from collections import deque
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from ..game.renderer import GameRenderer
from .hardware_monitor import HardwareMonitor


# Colors
BG_COLOR = (25, 25, 35)
PANEL_COLOR = (35, 35, 45)
TEXT_COLOR = (220, 220, 220)
ACCENT_COLOR = (100, 200, 100)
WARNING_COLOR = (255, 200, 50)
DANGER_COLOR = (255, 100, 100)
CHART_BG = (35, 35, 45)


@dataclass
class TrainingMetrics:
    """Container for training metrics history."""
    episodes: List[int] = field(default_factory=list)
    scores: List[int] = field(default_factory=list)
    avg_scores: List[float] = field(default_factory=list)
    epsilons: List[float] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)

    high_score: int = 0
    total_steps: int = 0

    # Rolling window for averages
    recent_scores: deque = field(default_factory=lambda: deque(maxlen=100))


class TrainingDashboard:
    """
    Real-time training dashboard with multiple panels:
    - Game visualization (left)
    - Training metrics (top right)
    - Score/epsilon charts (middle right)
    - Hardware utilization (bottom)
    """

    def __init__(
        self,
        window_width: int = 1400,
        window_height: int = 900,
        grid_width: int = 20,
        grid_height: int = 20,
        chart_update_interval: int = 10
    ):
        """
        Initialize the dashboard.

        Args:
            window_width: Window width in pixels
            window_height: Window height in pixels
            grid_width: Game grid width
            grid_height: Game grid height
            chart_update_interval: Update charts every N episodes
        """
        pygame.init()

        self.window_width = window_width
        self.window_height = window_height
        self.grid_width = grid_width
        self.grid_height = grid_height

        # Create window
        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("Snake AI - Training Dashboard")

        # Layout calculations
        self.game_size = min(500, window_height - 250)
        self.game_cell_size = self.game_size // max(grid_width, grid_height)
        self.game_size = self.game_cell_size * max(grid_width, grid_height)

        # Panel regions
        self.game_rect = pygame.Rect(20, 20, self.game_size, self.game_size)
        self.metrics_rect = pygame.Rect(
            self.game_size + 40, 20,
            window_width - self.game_size - 60, 180
        )
        self.chart_rect = pygame.Rect(
            self.game_size + 40, 210,
            window_width - self.game_size - 60, 350
        )
        self.hardware_rect = pygame.Rect(
            20, self.game_size + 40,
            window_width - 40, window_height - self.game_size - 60
        )

        # Fonts
        self.font_title = pygame.font.Font(None, 36)
        self.font_large = pygame.font.Font(None, 32)
        self.font_medium = pygame.font.Font(None, 26)
        self.font_small = pygame.font.Font(None, 20)

        # Game renderer
        self.game_renderer = GameRenderer(
            cell_size=self.game_cell_size,
            surface=self.screen,
            offset=(self.game_rect.x, self.game_rect.y)
        )

        # Hardware monitor
        self.hardware_monitor = HardwareMonitor(update_interval=1.0)
        self.hardware_monitor.start()

        # Metrics
        self.metrics = TrainingMetrics()

        # Clock for FPS limiting
        self.clock = pygame.time.Clock()

    def update(
        self,
        game_state: Optional[Dict[str, Any]],
        episode: int,
        score: int,
        epsilon: float,
        loss: Optional[float] = None,
        fps: int = 30
    ) -> bool:
        """
        Update the dashboard with current training state.

        Args:
            game_state: Current game state dict (or None for headless)
            episode: Current episode number
            score: Current/final score
            epsilon: Current exploration rate
            loss: Latest training loss
            fps: Target frame rate

        Returns:
            False if window was closed, True otherwise
        """
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False

        # Track high score continuously (not just at episode end)
        if score > self.metrics.high_score:
            self.metrics.high_score = score

        # Update metrics if episode completed
        if episode > len(self.metrics.episodes):
            self._update_metrics(episode, score, epsilon, loss)

        # Clear screen
        self.screen.fill(BG_COLOR)

        # Draw panels
        if game_state:
            self._draw_game_panel(game_state)
        else:
            self._draw_game_placeholder()

        self._draw_metrics_panel(episode, score, epsilon, loss)
        self._draw_chart_panel()
        self._draw_hardware_panel()

        pygame.display.flip()
        self.clock.tick(fps)

        return True

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

        # Calculate rolling average
        avg = np.mean(list(self.metrics.recent_scores))
        self.metrics.avg_scores.append(avg)

        # Update high score
        if score > self.metrics.high_score:
            self.metrics.high_score = score

    def _draw_game_panel(self, game_state: Dict[str, Any]):
        """Draw the snake game."""
        # Panel background
        pygame.draw.rect(self.screen, PANEL_COLOR, self.game_rect, border_radius=8)

        # Render game
        self.game_renderer.render(game_state)

        # Score overlay
        score_text = self.font_medium.render(
            f"Score: {game_state.get('score', 0)}",
            True, TEXT_COLOR
        )
        self.screen.blit(
            score_text,
            (self.game_rect.x + 10, self.game_rect.bottom - 30)
        )

    def _draw_game_placeholder(self):
        """Draw placeholder when no game is running."""
        pygame.draw.rect(self.screen, PANEL_COLOR, self.game_rect, border_radius=8)

        text = self.font_large.render("Training (Headless)", True, (100, 100, 100))
        self.screen.blit(
            text,
            (self.game_rect.centerx - text.get_width() // 2,
             self.game_rect.centery - text.get_height() // 2)
        )

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

        # Title
        title = self.font_title.render("Training Progress", True, ACCENT_COLOR)
        self.screen.blit(title, (x, y))
        y += 40

        # Metrics
        avg_score = 0
        if self.metrics.recent_scores:
            avg_score = np.mean(list(self.metrics.recent_scores))

        metrics_list = [
            (f"Episode: {episode}", TEXT_COLOR),
            (f"Current Score: {score}", TEXT_COLOR),
            (f"High Score: {self.metrics.high_score}", ACCENT_COLOR),
            (f"Avg Score (100): {avg_score:.1f}", TEXT_COLOR),
            (f"Epsilon: {epsilon:.4f}", WARNING_COLOR if epsilon > 0.1 else TEXT_COLOR),
        ]

        # Two columns
        col_width = (self.metrics_rect.width - 40) // 2
        for i, (text, color) in enumerate(metrics_list):
            col = i % 2
            row = i // 2
            text_surf = self.font_medium.render(text, True, color)
            self.screen.blit(text_surf, (x + col * col_width, y + row * 28))

    def _draw_chart_panel(self):
        """Draw training progress charts using pure pygame (fast, no blocking)."""
        pygame.draw.rect(self.screen, PANEL_COLOR, self.chart_rect, border_radius=8)

        if len(self.metrics.episodes) < 2:
            text = self.font_medium.render(
                "Collecting data...",
                True, (100, 100, 100)
            )
            self.screen.blit(
                text,
                (self.chart_rect.centerx - text.get_width() // 2,
                 self.chart_rect.centery - text.get_height() // 2)
            )
            return

        # Chart area with padding
        padding = 50
        chart_x = self.chart_rect.x + padding
        chart_y = self.chart_rect.y + 30
        chart_w = self.chart_rect.width - padding * 2
        chart_h = (self.chart_rect.height - 80) // 2

        # Draw score chart
        self._draw_line_chart(
            x=chart_x, y=chart_y, width=chart_w, height=chart_h,
            data=self.metrics.scores,
            avg_data=self.metrics.avg_scores,
            color=(0, 200, 100),
            avg_color=(150, 255, 150),
            label="Score",
            show_avg=True
        )

        # Draw epsilon chart
        self._draw_line_chart(
            x=chart_x, y=chart_y + chart_h + 40, width=chart_w, height=chart_h,
            data=self.metrics.epsilons,
            color=(0, 200, 255),
            label="Epsilon",
            y_range=(0, 1)
        )

    def _draw_line_chart(
        self,
        x: int, y: int, width: int, height: int,
        data: List,
        color: tuple,
        label: str,
        avg_data: List = None,
        avg_color: tuple = None,
        y_range: tuple = None,
        show_avg: bool = False
    ):
        """Draw a simple line chart with pygame."""
        if len(data) < 2:
            return

        # Draw background
        bg_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, (40, 40, 50), bg_rect, border_radius=4)

        # Draw grid lines
        for i in range(5):
            gy = y + int(height * i / 4)
            pygame.draw.line(self.screen, (60, 60, 70), (x, gy), (x + width, gy))

        # Calculate y range
        if y_range:
            min_val, max_val = y_range
        else:
            min_val = min(data)
            max_val = max(data)
            if max_val == min_val:
                max_val = min_val + 1

        # Limit data points to prevent slowdown
        max_points = 500
        if len(data) > max_points:
            step = len(data) // max_points
            plot_data = data[::step]
            plot_avg = avg_data[::step] if avg_data else None
        else:
            plot_data = data
            plot_avg = avg_data

        # Convert data to screen coordinates
        def to_screen(i, val):
            sx = x + int(i / max(1, len(plot_data) - 1) * width)
            sy = y + height - int((val - min_val) / (max_val - min_val) * height)
            return (sx, max(y, min(y + height, sy)))

        # Draw average line first (behind main line)
        if show_avg and plot_avg and len(plot_avg) >= 2:
            avg_points = [to_screen(i, v) for i, v in enumerate(plot_avg)]
            if len(avg_points) >= 2:
                pygame.draw.lines(self.screen, avg_color, False, avg_points, 2)

        # Draw main data line
        points = [to_screen(i, v) for i, v in enumerate(plot_data)]
        if len(points) >= 2:
            pygame.draw.lines(self.screen, color, False, points, 1)

        # Draw label
        label_surf = self.font_small.render(label, True, color)
        self.screen.blit(label_surf, (x + 5, y + 5))

        # Draw current value
        current_val = data[-1]
        val_text = f"{current_val:.2f}" if isinstance(current_val, float) else str(current_val)
        val_surf = self.font_small.render(val_text, True, TEXT_COLOR)
        self.screen.blit(val_surf, (x + width - val_surf.get_width() - 5, y + 5))

        # Draw min/max labels
        max_surf = self.font_small.render(f"{max_val:.1f}", True, (100, 100, 100))
        min_surf = self.font_small.render(f"{min_val:.1f}", True, (100, 100, 100))
        self.screen.blit(max_surf, (x - max_surf.get_width() - 5, y))
        self.screen.blit(min_surf, (x - min_surf.get_width() - 5, y + height - 15))

    def _draw_hardware_panel(self):
        """Draw hardware utilization panel."""
        pygame.draw.rect(self.screen, PANEL_COLOR, self.hardware_rect, border_radius=8)

        x = self.hardware_rect.x + 20
        y = self.hardware_rect.y + 15

        # Title
        title = self.font_title.render("Hardware Utilization", True, ACCENT_COLOR)
        self.screen.blit(title, (x, y))
        y += 35

        # Get hardware stats
        stats = self.hardware_monitor.get_stats()

        # CPU Section
        cpu_title = self.font_medium.render("CPU (per core)", True, TEXT_COLOR)
        self.screen.blit(cpu_title, (x, y))
        y += 25

        if stats and stats.cpu:
            bar_width = 80
            bar_height = 14
            label_width = 22  # Fixed width for labels (fits "15")
            pct_width = 40    # Fixed width for percentage text
            item_width = label_width + bar_width + pct_width + 10
            cores_per_row = 4  # 4 columns for a clean 4x4 grid with 16 threads

            for i, percent in enumerate(stats.cpu.per_core_percent):
                col = i % cores_per_row
                row = i // cores_per_row

                bx = x + col * item_width
                by = y + row * (bar_height + 8)

                # Core label (fixed width, right-aligned)
                label = self.font_small.render(f"{i}", True, (150, 150, 150))
                label_x = bx + label_width - label.get_width()
                self.screen.blit(label, (label_x, by))

                # Background bar (fixed position after label)
                bar_x = bx + label_width + 5
                bg_rect = pygame.Rect(bar_x, by, bar_width, bar_height)
                pygame.draw.rect(self.screen, (50, 50, 60), bg_rect, border_radius=3)

                # Fill bar
                fill_width = int(percent / 100 * bar_width)
                if fill_width > 0:
                    color = ACCENT_COLOR if percent < 80 else WARNING_COLOR if percent < 95 else DANGER_COLOR
                    fill_rect = pygame.Rect(bar_x, by, fill_width, bar_height)
                    pygame.draw.rect(self.screen, color, fill_rect, border_radius=3)

                # Percentage text
                pct_text = self.font_small.render(f"{percent:.0f}%", True, TEXT_COLOR)
                self.screen.blit(pct_text, (bar_x + bar_width + 5, by))

            y += ((len(stats.cpu.per_core_percent) - 1) // cores_per_row + 1) * (bar_height + 8) + 10

        # GPU Section
        gpu_x = self.hardware_rect.x + self.hardware_rect.width // 2 + 20
        gpu_y = self.hardware_rect.y + 50

        gpu_info = self.hardware_monitor.get_gpu_info()

        gpu_title = self.font_medium.render(
            f"GPU: {gpu_info['name'][:30]}",
            True, TEXT_COLOR
        )
        self.screen.blit(gpu_title, (gpu_x, gpu_y))
        gpu_y += 25

        if gpu_info['available']:
            # GPU Utilization bar
            util_label = self.font_small.render("Utilization:", True, (150, 150, 150))
            self.screen.blit(util_label, (gpu_x, gpu_y))

            bar_width = 150
            bar_height = 14
            bg_rect = pygame.Rect(gpu_x + 80, gpu_y, bar_width, bar_height)
            pygame.draw.rect(self.screen, (50, 50, 60), bg_rect, border_radius=3)

            util = gpu_info['utilization']
            fill_width = int(util / 100 * bar_width)
            if fill_width > 0:
                color = ACCENT_COLOR if util < 80 else WARNING_COLOR if util < 95 else DANGER_COLOR
                fill_rect = pygame.Rect(gpu_x + 80, gpu_y, fill_width, bar_height)
                pygame.draw.rect(self.screen, color, fill_rect, border_radius=3)

            pct_text = self.font_small.render(f"{util:.0f}%", True, TEXT_COLOR)
            self.screen.blit(pct_text, (gpu_x + 85 + bar_width, gpu_y))
            gpu_y += 25

            # Memory bar
            mem_label = self.font_small.render("Memory:", True, (150, 150, 150))
            self.screen.blit(mem_label, (gpu_x, gpu_y))

            bg_rect = pygame.Rect(gpu_x + 80, gpu_y, bar_width, bar_height)
            pygame.draw.rect(self.screen, (50, 50, 60), bg_rect, border_radius=3)

            mem_pct = gpu_info['memory_percent']
            fill_width = int(mem_pct / 100 * bar_width)
            if fill_width > 0:
                color = ACCENT_COLOR if mem_pct < 80 else WARNING_COLOR if mem_pct < 95 else DANGER_COLOR
                fill_rect = pygame.Rect(gpu_x + 80, gpu_y, fill_width, bar_height)
                pygame.draw.rect(self.screen, color, fill_rect, border_radius=3)

            if gpu_info['memory_total'] > 0:
                mem_text = f"{gpu_info['memory_used']:.0f}/{gpu_info['memory_total']:.0f} MB"
            else:
                mem_text = f"{mem_pct:.0f}%"
            mem_surf = self.font_small.render(mem_text, True, TEXT_COLOR)
            self.screen.blit(mem_surf, (gpu_x + 85 + bar_width, gpu_y))
        else:
            no_gpu = self.font_small.render(
                "GPU stats not available",
                True, (100, 100, 100)
            )
            self.screen.blit(no_gpu, (gpu_x, gpu_y))

    def get_high_score(self) -> int:
        """Get the current high score."""
        return self.metrics.high_score

    def close(self):
        """Clean up resources."""
        self.hardware_monitor.stop()
        pygame.quit()
