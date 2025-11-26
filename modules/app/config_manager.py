# modules/app/config_manager.py
import collections.abc
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

from utils.get_base_path import get_base_path

# Define the path to the config file, relative to the project root
CONFIG_FILE_PATH = Path(get_base_path()) / "config" / "config.yaml"


def get_default_config():
    """
    回傳包含預設設定的字典。
    此結構為所有設定鍵值的標準來源。
    """
    return {
        "ui": {
            "window_title": "Webcam Snapper",
            "default_size": {"width": 1280, "height": 720},
            "font_size": "12px",
        },
        "theme": {
            "preset": "dark",
            "custom_colors": {
                "background": "#1b1e23",
                "foreground": "#e8eaed",
                "accent": "#333844",
                "border": "#3a3f47",
                "button_background": "#2b2f36",
                "groupbox_title": "#cfd8dc",
            },
        },
        "paths": {"default_output": "~/Pictures/WebcamSnapper"},
        "logging": {
            "directory": "logs",
            "level": "INFO",
            "json_enabled": True,
            "rotation": {"max_bytes": 2000000, "backup_count": 5},
            "ui_popup_level": "ERROR",
        },
        "shortcuts": {
            "main": {
                "capture.photo": ["Space"],
                "record.start_resume": ["R"],
                "record.stop_save": ["Shift+R"],
            },
            "viewer": {
                "nav.prev": ["Left", "PageUp"],
                "nav.next": ["Right", "PageDown"],
                "save.selected": ["S", "Ctrl+S"],
                "save.union": ["U"],
                "window.close": ["Esc"],
            },
        },
        "performance": {
            "camera": {
                "preferred_resolution": {"width": 1920, "height": 1080},
                "preferred_framerate": 30,
            },
            "video_recording": {"codec": "avc1", "quality": "normal", "container": "mp4"},
        },
          "behavior": {
            "auto_start_camera_on_launch": False,
            "save_window_state": True
          },
        "features": {
            "burst_shot": {"default_count": 5, "default_interval_ms": 500},
            "camera": {"default_focus_threshold": 150},
        },
        "advanced_features": {
            "segmentation": {
                "default_device": "GPU",
                "default_model": "vit_h",
                "mask_generator": {
                    "points_per_side": 32,
                    "pred_iou_thresh": 0.88,
                    "stability_score_thresh": 0.95,
                    "min_mask_region_area": 100,
                },
            }
        },
    }


def _deep_merge(user_config, defaults):
    """
    遞迴地將使用者設定合併到預設值中。
    使用者的值優先。
    """
    if not isinstance(user_config, collections.abc.Mapping):
        return user_config

    merged = defaults.copy()
    for key, value in user_config.items():
        if key in merged and isinstance(merged[key], collections.abc.Mapping):
            merged[key] = _deep_merge(value, merged[key])
        else:
            merged[key] = value
    return merged


def load_config():
    """
    從 config.yaml 載入設定。
    如果檔案不存在，則使用預設值建立一個。
    將載入的設定與預設值合併，以確保所有鍵值都存在。
    """
    defaults = get_default_config()

    if not CONFIG_FILE_PATH.exists():
        logger.info(f"Configuration file not found. Creating default config at {CONFIG_FILE_PATH}")
        try:
            with open(CONFIG_FILE_PATH, "w", encoding="utf-8") as f:
                yaml.dump(defaults, f, sort_keys=False, indent=2, allow_unicode=True)
            return defaults
        except Exception as e:
            logger.error(f"Failed to create default config file: {e}")
            return defaults  # Fallback to in-memory defaults

    try:
        with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}

        # Merge user config into defaults to ensure all keys are present
        config = _deep_merge(user_config, defaults)
        return config

    except Exception as e:
        logger.error(f"Failed to load or parse config file {CONFIG_FILE_PATH}: {e}")
        return defaults  # Fallback to in-memory defaults


# Load the configuration once on startup
# Other modules can import this 'config' variable
config = load_config()


def get_config():
    """回傳已載入的設定。"""
    return config
