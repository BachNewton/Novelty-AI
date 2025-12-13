"""
Replay Player - Plays back recorded game sessions.

The replay system allows watching high-score games while
training continues in the background.
"""
import json
import threading
import queue
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime

import pygame


@dataclass
class ReplayData:
    """Data for a single game replay."""
    frames: List[Dict[str, Any]]
    score: int
    episode: int
    timestamp: str
    duration_frames: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "frames": self.frames,
            "score": self.score,
            "episode": self.episode,
            "timestamp": self.timestamp,
            "duration_frames": self.duration_frames,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReplayData":
        """Create from dictionary."""
        return cls(
            frames=data["frames"],
            score=data["score"],
            episode=data["episode"],
            timestamp=data["timestamp"],
            duration_frames=data["duration_frames"],
        )


class ReplayManager:
    """
    Manages saving and loading game replays.

    Replays are saved as JSON files with timestamps.
    """

    def __init__(self, save_dir: str = "replays", max_replays: int = 10):
        """
        Initialize the replay manager.

        Args:
            save_dir: Directory to save replays
            max_replays: Maximum number of replays to keep (keeps highest scores)
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_replays = max_replays

    def save_replay(
        self,
        frames: List[Dict[str, Any]],
        score: int,
        episode: int
    ) -> str:
        """
        Save a game replay.

        Args:
            frames: List of game state frames
            score: Final score
            episode: Episode number

        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"replay_ep{episode}_score{score}_{timestamp}.json"
        filepath = self.save_dir / filename

        replay = ReplayData(
            frames=frames,
            score=score,
            episode=episode,
            timestamp=timestamp,
            duration_frames=len(frames),
        )

        with open(filepath, "w") as f:
            json.dump(replay.to_dict(), f)

        # Clean up old replays, keeping only the best ones
        self._cleanup_old_replays()

        return str(filepath)

    def _cleanup_old_replays(self):
        """Remove lowest-scoring replays if we exceed max_replays."""
        replay_files = list(self.save_dir.glob("replay_*.json"))

        if len(replay_files) <= self.max_replays:
            return

        # Extract scores from filenames (format: replay_ep{N}_score{N}_timestamp.json)
        scored_files = []
        for path in replay_files:
            try:
                # Parse score from filename
                name = path.stem  # e.g., "replay_ep42_score5_20241213_143052"
                parts = name.split("_")
                for part in parts:
                    if part.startswith("score"):
                        score = int(part[5:])
                        scored_files.append((score, path))
                        break
            except (ValueError, IndexError):
                # If parsing fails, keep the file (score=infinity)
                scored_files.append((float('inf'), path))

        # Sort by score (lowest first) and delete extras
        scored_files.sort(key=lambda x: x[0])
        files_to_delete = len(scored_files) - self.max_replays

        for i in range(files_to_delete):
            score, path = scored_files[i]
            print(f"[Replay] Removing old replay: score {score}")
            path.unlink()

    def load_replay(self, filepath: str) -> ReplayData:
        """
        Load a game replay.

        Args:
            filepath: Path to replay file

        Returns:
            ReplayData object
        """
        with open(filepath, "r") as f:
            data = json.load(f)
        return ReplayData.from_dict(data)

    def list_replays(self) -> List[str]:
        """List all saved replays."""
        return sorted(
            [str(p) for p in self.save_dir.glob("replay_*.json")],
            reverse=True  # Newest first
        )

    def get_best_replay(self) -> Optional[str]:
        """Get the replay with highest score."""
        replays = self.list_replays()
        if not replays:
            return None

        best_score = -1
        best_path = None

        for path in replays:
            replay = self.load_replay(path)
            if replay.score > best_score:
                best_score = replay.score
                best_path = path

        return best_path


class ReplayPlayer:
    """
    Plays back game replays in a separate thread.

    This allows training to continue while the user watches
    interesting games (like new high scores).
    """

    def __init__(
        self,
        render_callback: Callable[[Dict[str, Any]], None],
        fps: int = 10
    ):
        """
        Initialize the replay player.

        Args:
            render_callback: Function to call for each frame
            fps: Playback speed in frames per second
        """
        self.render_callback = render_callback
        self.fps = fps
        self.frame_delay = 1.0 / fps

        # Replay queue
        self.replay_queue: queue.Queue = queue.Queue()

        # Control flags
        self.playing = False
        self.paused = False
        self.stop_requested = False

        # Current replay info
        self.current_replay: Optional[ReplayData] = None
        self.current_frame_idx = 0

        # Thread
        self.thread: Optional[threading.Thread] = None

    def queue_replay(self, replay: ReplayData):
        """
        Add a replay to the playback queue.

        Args:
            replay: ReplayData to queue
        """
        self.replay_queue.put(replay)

    def start(self):
        """Start the replay player thread."""
        if self.thread is not None and self.thread.is_alive():
            return

        self.stop_requested = False
        self.thread = threading.Thread(target=self._playback_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the replay player."""
        self.stop_requested = True
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None

    def pause(self):
        """Pause playback."""
        self.paused = True

    def resume(self):
        """Resume playback."""
        self.paused = False

    def skip(self):
        """Skip the current replay."""
        self.current_replay = None

    def _playback_loop(self):
        """Main playback loop (runs in separate thread)."""
        while not self.stop_requested:
            # Get next replay if needed
            if self.current_replay is None:
                try:
                    self.current_replay = self.replay_queue.get(timeout=0.1)
                    self.current_frame_idx = 0
                    self.playing = True
                except queue.Empty:
                    self.playing = False
                    continue

            # Handle pause
            if self.paused:
                time.sleep(0.1)
                continue

            # Play current frame
            if self.current_frame_idx < len(self.current_replay.frames):
                frame = self.current_replay.frames[self.current_frame_idx]
                self.render_callback(frame)
                self.current_frame_idx += 1
                time.sleep(self.frame_delay)
            else:
                # Replay finished
                self.current_replay = None
                self.playing = False

    def is_playing(self) -> bool:
        """Check if currently playing a replay."""
        return self.playing

    def has_queued(self) -> bool:
        """Check if there are replays in the queue."""
        return not self.replay_queue.empty()

    def get_status(self) -> Dict[str, Any]:
        """Get current playback status."""
        return {
            "playing": self.playing,
            "paused": self.paused,
            "queued": self.replay_queue.qsize(),
            "current_frame": self.current_frame_idx if self.current_replay else 0,
            "total_frames": self.current_replay.duration_frames if self.current_replay else 0,
            "current_score": self.current_replay.score if self.current_replay else 0,
            "current_episode": self.current_replay.episode if self.current_replay else 0,
        }


class ReplayWindow:
    """
    Standalone window for replay playback.

    Opens a separate pygame window to display replays
    while training continues in the main thread.
    """

    def __init__(
        self,
        grid_width: int = 20,
        grid_height: int = 20,
        cell_size: int = 25,
        fps: int = 10,
        max_replays: int = 10
    ):
        """
        Initialize the replay window.

        Args:
            grid_width: Game grid width
            grid_height: Game grid height
            cell_size: Size of each cell in pixels
            fps: Playback speed
            max_replays: Maximum replays to keep (highest scores)
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.cell_size = cell_size
        self.fps = fps

        self.replay_manager = ReplayManager(max_replays=max_replays)
        self.running = False
        self.window = None
        self.thread: Optional[threading.Thread] = None

        # Replay queue
        self.replay_queue: queue.Queue = queue.Queue()

    def queue_replay(
        self,
        frames: List[Dict[str, Any]],
        score: int,
        episode: int,
        save: bool = True
    ):
        """
        Queue a replay for playback.

        Args:
            frames: Game frames
            score: Final score
            episode: Episode number
            save: Whether to save the replay to disk
        """
        replay = ReplayData(
            frames=frames,
            score=score,
            episode=episode,
            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
            duration_frames=len(frames),
        )

        if save:
            self.replay_manager.save_replay(frames, score, episode)

        self.replay_queue.put(replay)

        # Start window if not running
        if not self.running:
            self.start()

    def start(self):
        """Start the replay window in a separate thread."""
        if self.thread is not None and self.thread.is_alive():
            return

        self.running = True
        self.thread = threading.Thread(target=self._window_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the replay window."""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None

    def _window_loop(self):
        """Main window loop (runs in separate thread)."""
        # Initialize pygame in this thread
        pygame.init()

        padding = 40
        window_width = self.grid_width * self.cell_size + padding * 2
        window_height = self.grid_height * self.cell_size + padding * 2 + 80

        self.window = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("Snake AI - Replay")

        font_large = pygame.font.Font(None, 48)
        font_medium = pygame.font.Font(None, 32)
        clock = pygame.time.Clock()

        # Import renderer here to avoid circular imports
        from ..game.renderer import GameRenderer

        renderer = GameRenderer(
            cell_size=self.cell_size,
            surface=self.window,
            offset=(padding, padding)
        )

        current_replay: Optional[ReplayData] = None
        frame_idx = 0

        while self.running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        # Skip current replay
                        current_replay = None
                    elif event.key == pygame.K_ESCAPE:
                        self.running = False

            if not self.running:
                break

            # Get next replay if needed
            if current_replay is None:
                try:
                    current_replay = self.replay_queue.get(timeout=0.1)
                    frame_idx = 0
                except queue.Empty:
                    # Show waiting screen
                    self.window.fill((30, 30, 40))
                    text = font_large.render("Waiting for replay...", True, (100, 100, 100))
                    self.window.blit(
                        text,
                        (window_width // 2 - text.get_width() // 2, window_height // 2)
                    )
                    pygame.display.flip()
                    clock.tick(10)
                    continue

            # Render current frame
            self.window.fill((30, 30, 40))

            if frame_idx < len(current_replay.frames):
                frame = current_replay.frames[frame_idx]
                renderer.render(frame)

                # Draw info
                info_text = f"Episode {current_replay.episode} | Score: {frame.get('score', 0)}"
                text_surf = font_medium.render(info_text, True, (220, 220, 220))
                self.window.blit(text_surf, (padding, window_height - 70))

                progress = f"Frame {frame_idx + 1}/{current_replay.duration_frames}"
                progress_surf = font_medium.render(progress, True, (150, 150, 150))
                self.window.blit(progress_surf, (padding, window_height - 40))

                frame_idx += 1
            else:
                # Replay finished
                current_replay = None

            pygame.display.flip()
            clock.tick(self.fps)

        pygame.quit()

    def is_running(self) -> bool:
        """Check if replay window is running."""
        return self.running
