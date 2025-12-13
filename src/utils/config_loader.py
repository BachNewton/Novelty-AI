"""
Configuration Loader - Load and validate configuration from YAML.
"""
import yaml
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, field


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
class Config:
    """Complete application configuration."""
    game: GameConfig = field(default_factory=GameConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
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
