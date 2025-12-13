"""
Milestone Effects - Visual celebrations for new high scores.

Provides particle effects and floating messages when the AI
achieves a new high score.
"""
import pygame
import math
import random
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Particle:
    """A single celebration particle."""
    x: float
    y: float
    vx: float
    vy: float
    color: Tuple[int, int, int]
    lifetime: float
    max_lifetime: float
    size: float


@dataclass
class FloatingMessage:
    """A floating text message."""
    text: str
    x: float
    y: float
    vy: float
    alpha: int
    lifetime: int
    font_size: int
    color: Tuple[int, int, int]


class MilestoneEffects:
    """
    Visual celebration effects for milestones.

    Creates particle explosions and floating messages
    when triggered by high scores.
    """

    def __init__(self, screen: pygame.Surface):
        """
        Initialize the effects system.

        Args:
            screen: Pygame surface to render effects on
        """
        self.screen = screen
        self.particles: List[Particle] = []
        self.messages: List[FloatingMessage] = []

        # Pre-create fonts
        self.fonts = {
            48: pygame.font.Font(None, 48),
            64: pygame.font.Font(None, 64),
            72: pygame.font.Font(None, 72),
        }

        # Celebration colors
        self.colors = [
            (255, 215, 0),    # Gold
            (255, 165, 0),    # Orange
            (0, 255, 127),    # Spring Green
            (0, 191, 255),    # Deep Sky Blue
            (255, 105, 180),  # Hot Pink
            (147, 112, 219),  # Medium Purple
        ]

    def trigger_high_score(self, score: int, x: Optional[int] = None, y: Optional[int] = None):
        """
        Trigger celebration for a new high score.

        Args:
            score: The new high score
            x: Center x position (defaults to screen center)
            y: Center y position (defaults to screen center)
        """
        center_x = x if x is not None else self.screen.get_width() // 2
        center_y = y if y is not None else self.screen.get_height() // 2

        # Add floating message
        self.messages.append(FloatingMessage(
            text=f"NEW HIGH SCORE: {score}!",
            x=center_x,
            y=center_y,
            vy=-1.5,
            alpha=255,
            lifetime=180,
            font_size=72,
            color=(255, 215, 0),
        ))

        # Spawn particle explosion
        self._spawn_particles(center_x, center_y, count=150)

    def trigger_score_milestone(self, score: int, x: Optional[int] = None, y: Optional[int] = None):
        """
        Trigger celebration for reaching a score milestone.

        Args:
            score: The milestone score
            x: Center x position
            y: Center y position
        """
        center_x = x if x is not None else self.screen.get_width() // 2
        center_y = y if y is not None else self.screen.get_height() // 2

        self.messages.append(FloatingMessage(
            text=f"Score: {score}!",
            x=center_x,
            y=center_y,
            vy=-1.0,
            alpha=255,
            lifetime=120,
            font_size=48,
            color=(0, 255, 127),
        ))

        self._spawn_particles(center_x, center_y, count=50)

    def _spawn_particles(self, center_x: int, center_y: int, count: int = 100):
        """Spawn particles in an explosion pattern."""
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 12)
            lifetime = random.uniform(40, 100)

            self.particles.append(Particle(
                x=center_x,
                y=center_y,
                vx=math.cos(angle) * speed,
                vy=math.sin(angle) * speed - 4,  # Upward bias
                color=random.choice(self.colors),
                lifetime=lifetime,
                max_lifetime=lifetime,
                size=random.uniform(3, 8),
            ))

    def update(self):
        """Update all active effects."""
        self._update_particles()
        self._update_messages()

    def draw(self):
        """Draw all active effects."""
        self._draw_particles()
        self._draw_messages()

    def _update_particles(self):
        """Update particle positions and lifetimes."""
        for particle in self.particles[:]:
            particle.x += particle.vx
            particle.y += particle.vy
            particle.vy += 0.15  # Gravity
            particle.lifetime -= 1

            # Slow down over time
            particle.vx *= 0.98
            particle.vy *= 0.98

            if particle.lifetime <= 0:
                self.particles.remove(particle)

    def _update_messages(self):
        """Update floating messages."""
        for msg in self.messages[:]:
            msg.y += msg.vy
            msg.lifetime -= 1

            # Fade out in last 30 frames
            if msg.lifetime < 30:
                msg.alpha = int(msg.alpha * 0.9)

            if msg.lifetime <= 0:
                self.messages.remove(msg)

    def _draw_particles(self):
        """Draw all particles."""
        for particle in self.particles:
            # Calculate alpha based on remaining lifetime
            alpha = int(255 * (particle.lifetime / particle.max_lifetime))
            alpha = max(0, min(255, alpha))

            # Create surface with alpha
            size = int(particle.size)
            if size < 1:
                continue

            surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
            color_with_alpha = (*particle.color, alpha)
            pygame.draw.circle(surf, color_with_alpha, (size, size), size)

            self.screen.blit(
                surf,
                (int(particle.x - size), int(particle.y - size))
            )

    def _draw_messages(self):
        """Draw floating messages with glow effect."""
        for msg in self.messages:
            font = self.fonts.get(msg.font_size, self.fonts[48])

            # Create text surface
            text_surf = font.render(msg.text, True, msg.color)
            text_surf.set_alpha(msg.alpha)

            # Center the text
            text_x = msg.x - text_surf.get_width() // 2
            text_y = msg.y - text_surf.get_height() // 2

            # Draw glow (multiple offset copies)
            if msg.alpha > 100:
                glow_color = tuple(min(255, c + 50) for c in msg.color)
                glow_surf = font.render(msg.text, True, glow_color)
                glow_surf.set_alpha(msg.alpha // 3)

                for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                    self.screen.blit(glow_surf, (text_x + dx, text_y + dy))

            # Draw main text
            self.screen.blit(text_surf, (text_x, text_y))

    def is_active(self) -> bool:
        """Check if any effects are currently active."""
        return len(self.particles) > 0 or len(self.messages) > 0

    def clear(self):
        """Clear all active effects."""
        self.particles.clear()
        self.messages.clear()


class HighScoreTracker:
    """
    Tracks high scores and triggers celebrations.

    Only triggers effects on new high scores to avoid
    overwhelming the user with notifications.
    """

    def __init__(self, effects: MilestoneEffects):
        """
        Initialize the high score tracker.

        Args:
            effects: MilestoneEffects instance for celebrations
        """
        self.effects = effects
        self.high_score = 0
        self.last_celebrated_score = 0

    def check_score(
        self,
        score: int,
        x: Optional[int] = None,
        y: Optional[int] = None
    ) -> bool:
        """
        Check if score is a new high score and trigger celebration.

        Args:
            score: Current game score
            x: Celebration center x position
            y: Celebration center y position

        Returns:
            True if new high score was achieved
        """
        # Only celebrate if score > 0 (ate at least one food) and beats previous high
        if score > 0 and score > self.high_score:
            print(f"[HIGH SCORE] New high: {score} (previous: {self.high_score})")
            self.high_score = score
            self.effects.trigger_high_score(score, x, y)
            self.last_celebrated_score = score
            return True

        return False

    def reset(self):
        """Reset the tracker (but keep high score)."""
        self.last_celebrated_score = 0

    def get_high_score(self) -> int:
        """Get the current high score."""
        return self.high_score
