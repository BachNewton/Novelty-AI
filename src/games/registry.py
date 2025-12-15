"""
Game registry for Novelty AI.

Central registry for discovering and instantiating games.
Games register themselves when their module is imported.
"""

from typing import Dict, Type, List, Optional, Any
from ..core.game_interface import GameInterface, GameMetadata
from ..core.env_interface import EnvInterface
from ..core.renderer_interface import RendererInterface


class GameRegistry:
    """
    Central registry for all available games.

    Games register themselves using the @register decorator or by calling
    GameRegistry.register() directly in their __init__.py.
    """

    _games: Dict[str, Dict[str, Any]] = {}
    _placeholder_games: List[GameMetadata] = []

    @classmethod
    def register(
        cls,
        game_class: Type[GameInterface],
        env_class: Type[EnvInterface],
        renderer_class: Type[RendererInterface],
        config_class: Optional[Type] = None
    ) -> None:
        """
        Register a game with the registry.

        Args:
            game_class: The game implementation class
            env_class: The environment wrapper class
            renderer_class: The renderer class
            config_class: Optional game-specific config class
        """
        metadata = game_class.get_metadata()
        cls._games[metadata.id] = {
            'game_class': game_class,
            'env_class': env_class,
            'renderer_class': renderer_class,
            'config_class': config_class,
            'metadata': metadata
        }

    @classmethod
    def register_placeholder(cls, metadata: GameMetadata) -> None:
        """
        Register a placeholder for a coming-soon game.

        Args:
            metadata: Metadata for the placeholder game
        """
        cls._placeholder_games.append(metadata)

    @classmethod
    def get_game(cls, game_id: str) -> Optional[Dict[str, Any]]:
        """
        Get game components by ID.

        Args:
            game_id: The game identifier

        Returns:
            Dictionary with game classes, or None if not found
        """
        return cls._games.get(game_id)

    @classmethod
    def list_games(cls) -> List[GameMetadata]:
        """
        List all registered games.

        Returns:
            List of GameMetadata for all registered games
        """
        return [g['metadata'] for g in cls._games.values()]

    @classmethod
    def list_all_games(cls, include_placeholders: bool = True) -> List[GameMetadata]:
        """
        List all games including placeholders.

        Args:
            include_placeholders: Whether to include coming-soon games

        Returns:
            List of GameMetadata for all games
        """
        games = cls.list_games()
        if include_placeholders:
            games = games + cls._placeholder_games
        return games

    @classmethod
    def is_available(cls, game_id: str) -> bool:
        """
        Check if a game is available (not a placeholder).

        Args:
            game_id: The game identifier

        Returns:
            True if game is playable, False otherwise
        """
        return game_id in cls._games

    @classmethod
    def create_game(cls, game_id: str, **kwargs) -> GameInterface:
        """
        Create a game instance.

        Args:
            game_id: The game identifier
            **kwargs: Arguments to pass to the game constructor

        Returns:
            Game instance

        Raises:
            ValueError: If game is not registered
        """
        game_data = cls._games.get(game_id)
        if not game_data:
            raise ValueError(f"Unknown game: {game_id}")
        return game_data['game_class'](**kwargs)

    @classmethod
    def create_env(cls, game_id: str, **kwargs) -> EnvInterface:
        """
        Create an environment instance for a game.

        Args:
            game_id: The game identifier
            **kwargs: Arguments to pass to the environment constructor

        Returns:
            Environment instance

        Raises:
            ValueError: If game is not registered
        """
        game_data = cls._games.get(game_id)
        if not game_data:
            raise ValueError(f"Unknown game: {game_id}")
        return game_data['env_class'](**kwargs)

    @classmethod
    def create_renderer(cls, game_id: str, **kwargs) -> RendererInterface:
        """
        Create a renderer instance for a game.

        Args:
            game_id: The game identifier
            **kwargs: Arguments to pass to the renderer constructor

        Returns:
            Renderer instance

        Raises:
            ValueError: If game is not registered
        """
        game_data = cls._games.get(game_id)
        if not game_data:
            raise ValueError(f"Unknown game: {game_id}")
        return game_data['renderer_class'](**kwargs)

    @classmethod
    def get_config_class(cls, game_id: str) -> Optional[Type]:
        """
        Get the config class for a game.

        Args:
            game_id: The game identifier

        Returns:
            Config class or None
        """
        game_data = cls._games.get(game_id)
        if game_data:
            return game_data.get('config_class')
        return None

    @classmethod
    def get_metadata(cls, game_id: str) -> Optional[GameMetadata]:
        """
        Get metadata for a game.

        Args:
            game_id: The game identifier

        Returns:
            GameMetadata or None if not found
        """
        game_data = cls._games.get(game_id)
        if game_data:
            return game_data.get('metadata')
        # Check placeholders
        for placeholder in cls._placeholder_games:
            if placeholder.id == game_id:
                return placeholder
        return None

    @classmethod
    def clear(cls) -> None:
        """Clear all registrations (useful for testing)."""
        cls._games.clear()
        cls._placeholder_games.clear()


def register_game(
    env_class: Type[EnvInterface],
    renderer_class: Type[RendererInterface],
    config_class: Optional[Type] = None
):
    """
    Decorator to register a game class.

    Usage:
        @register_game(SnakeEnv, SnakeRenderer)
        class SnakeGame(GameInterface):
            ...
    """
    def decorator(game_class: Type[GameInterface]) -> Type[GameInterface]:
        GameRegistry.register(game_class, env_class, renderer_class, config_class)
        return game_class
    return decorator
