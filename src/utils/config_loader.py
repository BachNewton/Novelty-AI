"""
Configuration Loader - Load and validate configuration from YAML.

Supports hierarchical configuration:
- config/default.yaml - Global settings
- config/games/{game_id}.yaml - Per-game settings

Game-specific settings override defaults.
"""
import yaml
from pathlib import Path
from typing import Optional, Any, Dict
from dataclasses import dataclass, field
from copy import deepcopy


@dataclass
class GameConfig:
    """Game configuration."""
    grid_width: int = 20
    grid_height: int = 20


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    episodes: int = 10000
    batch_size: int = 64
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    buffer_size: int = 100000
    target_update_freq: int = 100
    save_interval: int = 100
    checkpoint_dir: str = "models"


@dataclass
class DeviceConfig:
    """Device selection configuration."""
    preferred: str = "auto"
    force_cpu: bool = False


@dataclass
class VisualizationConfig:
    """Visualization settings."""
    enabled: bool = True
    show_game: bool = True
    show_charts: bool = True
    render_fps: int = 30
    chart_update_interval: int = 10
    window_width: int = 1400
    window_height: int = 900


@dataclass
class ReplayConfig:
    """Replay system settings."""
    enabled: bool = True
    save_dir: str = "replays"
    playback_fps: int = 10
    max_replays: int = 10


@dataclass
class HardwareMonitorConfig:
    """Hardware monitoring settings."""
    enabled: bool = True
    update_interval: float = 1.0


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    log_file: str = "logs/training.log"


@dataclass
class RewardsConfig:
    """Reward shaping configuration."""
    food: float = 10.0
    death: float = -10.0
    step_penalty: float = 0.0
    approach_food: float = 0.0
    retreat_food: float = 0.0
    length_bonus_factor: float = 0.0


@dataclass
class Config:
    """Complete application configuration."""
    game: GameConfig = field(default_factory=GameConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    rewards: RewardsConfig = field(default_factory=RewardsConfig)
    hardware_monitor: HardwareMonitorConfig = field(default_factory=HardwareMonitorConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _dict_to_dataclass(data: dict, cls: type) -> Any:
    """Convert a dictionary to a dataclass instance."""
    if not data:
        return cls()

    # Get the fields that the dataclass expects
    field_names = {f.name for f in cls.__dataclass_fields__.values()}

    # Filter to only include valid fields
    filtered_data = {k: v for k, v in data.items() if k in field_names}

    return cls(**filtered_data)


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to config file (defaults to project root config.yaml)

    Returns:
        Config object with all settings
    """
    # Find config file
    if config_path is None:
        # Try to find config.yaml in common locations
        possible_paths = [
            Path("config.yaml"),
            Path(__file__).parent.parent.parent / "config.yaml",
            Path.cwd() / "config.yaml",
        ]

        for path in possible_paths:
            if path.exists():
                config_path = str(path)
                break

    if config_path is None or not Path(config_path).exists():
        print("[Config] No config file found, using defaults")
        return Config()

    # Load YAML
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)

    if data is None:
        return Config()

    # Build config object
    config = Config()

    if 'game' in data:
        config.game = _dict_to_dataclass(data['game'], GameConfig)

    if 'training' in data:
        config.training = _dict_to_dataclass(data['training'], TrainingConfig)

    if 'device' in data:
        config.device = _dict_to_dataclass(data['device'], DeviceConfig)

    if 'visualization' in data:
        config.visualization = _dict_to_dataclass(data['visualization'], VisualizationConfig)

    if 'replay' in data:
        config.replay = _dict_to_dataclass(data['replay'], ReplayConfig)

    if 'hardware_monitor' in data:
        config.hardware_monitor = _dict_to_dataclass(data['hardware_monitor'], HardwareMonitorConfig)

    if 'logging' in data:
        config.logging = _dict_to_dataclass(data['logging'], LoggingConfig)

    if 'rewards' in data:
        config.rewards = _dict_to_dataclass(data['rewards'], RewardsConfig)

    return config


def save_config(config: Config, config_path: str):
    """
    Save configuration to a YAML file.

    Args:
        config: Config object to save
        config_path: Path to save to
    """
    from dataclasses import asdict

    data = asdict(config)

    with open(config_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Deep merge two dictionaries, with override values taking precedence.

    Args:
        base: Base dictionary
        override: Dictionary with values to override

    Returns:
        Merged dictionary
    """
    result = deepcopy(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)

    return result


def _find_config_dir() -> Path:
    """Find the config directory."""
    possible_paths = [
        Path("config"),
        Path(__file__).parent.parent.parent / "config",
        Path.cwd() / "config",
    ]

    for path in possible_paths:
        if path.exists() and path.is_dir():
            return path

    # Fallback to project root config folder
    return Path(__file__).parent.parent.parent / "config"


def _load_yaml_file(path: Path) -> Dict:
    """Load a YAML file, returning empty dict if not found."""
    if not path.exists():
        return {}

    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    return data if data else {}


def load_game_config(game_id: str) -> Config:
    """
    Load configuration for a specific game.

    Merges default settings with game-specific settings.
    Game settings override defaults.

    Args:
        game_id: The game identifier (e.g., "snake")

    Returns:
        Config object with merged settings
    """
    config_dir = _find_config_dir()

    # Load default config
    default_path = config_dir / "default.yaml"
    default_data = _load_yaml_file(default_path)

    # Load game-specific config
    game_path = config_dir / "games" / f"{game_id}.yaml"
    game_data = _load_yaml_file(game_path)

    # Merge configs (game overrides default)
    merged_data = _deep_merge(default_data, game_data)

    if not merged_data:
        print(f"[Config] No config found for game '{game_id}', using defaults")
        return Config()

    # Build config object from merged data
    config = Config()

    if 'game' in merged_data:
        config.game = _dict_to_dataclass(merged_data['game'], GameConfig)

    if 'training' in merged_data:
        config.training = _dict_to_dataclass(merged_data['training'], TrainingConfig)

    if 'device' in merged_data:
        config.device = _dict_to_dataclass(merged_data['device'], DeviceConfig)

    if 'visualization' in merged_data:
        config.visualization = _dict_to_dataclass(merged_data['visualization'], VisualizationConfig)

    if 'replay' in merged_data:
        config.replay = _dict_to_dataclass(merged_data['replay'], ReplayConfig)

    if 'hardware_monitor' in merged_data:
        config.hardware_monitor = _dict_to_dataclass(merged_data['hardware_monitor'], HardwareMonitorConfig)

    if 'logging' in merged_data:
        config.logging = _dict_to_dataclass(merged_data['logging'], LoggingConfig)

    if 'rewards' in merged_data:
        config.rewards = _dict_to_dataclass(merged_data['rewards'], RewardsConfig)

    return config


def list_available_games() -> list:
    """
    List all games that have configuration files.

    Returns:
        List of game IDs
    """
    config_dir = _find_config_dir()
    games_dir = config_dir / "games"

    if not games_dir.exists():
        return []

    return [
        p.stem for p in games_dir.glob("*.yaml")
        if p.is_file()
    ]
