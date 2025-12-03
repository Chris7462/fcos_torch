import yaml
from typing import Any, Dict


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing configuration parameters.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_default_config() -> Dict[str, Any]:
    """
    Returns default configuration for FCOS training.

    Returns:
        Dictionary containing default configuration parameters.
    """
    return {
        "model": {
            "num_classes": 20,
            "fpn_channels": 128,
            "stem_channels": [128, 128],
        },
        "train": {
            "batch_size": 16,
            "learning_rate": 8e-3,
            "weight_decay": 1e-4,
            "max_iters": 9000,
            "log_period": 100,
        },
        "data": {
            "dataset_dir": "./data",
            "image_size": 224,
            "num_workers": 4,
            "max_boxes": 40,
            "exclude_difficult": True,
        },
        "inference": {
            "score_thresh": 0.4,
            "nms_thresh": 0.6,
        },
    }


def merge_config(default_config: Dict[str, Any], custom_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge custom config into default config.

    Args:
        default_config: Default configuration dictionary.
        custom_config: Custom configuration dictionary to override defaults.

    Returns:
        Merged configuration dictionary.
    """
    merged = default_config.copy()

    for key, value in custom_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value

    return merged
