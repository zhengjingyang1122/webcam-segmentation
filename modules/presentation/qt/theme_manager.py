"""Theme manager for webcam applications.

Provides light and dark theme stylesheets for consistent UI theming.
Loads themes from .qss files in the 'themes' directory.
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_stylesheet(theme_name: str) -> str:
    """Load stylesheet from themes directory."""
    # 取得模組所在目錄 (modules/presentation/qt)
    # 專案根目錄在 ../../../
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent.parent
    theme_path = project_root / "themes" / f"{theme_name}.qss"
    
    try:
        if theme_path.exists():
            with open(theme_path, "r", encoding="utf-8") as f:
                qss = f.read()
                logger.info(f"Loaded theme from {theme_path}")
                return qss
        else:
            logger.warning(f"Theme file not found: {theme_path}")
            # Fallback: try relative to cwd if running from root
            cwd_theme_path = Path("themes") / f"{theme_name}.qss"
            if cwd_theme_path.exists():
                with open(cwd_theme_path, "r", encoding="utf-8") as f:
                    qss = f.read()
                    logger.info(f"Loaded theme from {cwd_theme_path}")
                    return qss
            return ""
    except Exception as e:
        logger.error(f"Error loading theme {theme_name}: {e}")
        return ""


def apply_theme(widget, theme_name: str = "dark"):
    """Apply a theme to a widget.
    
    Args:
        widget: The widget to apply the theme to
        theme_name: Either "light" or "dark"
    """
    stylesheet = load_stylesheet(theme_name.lower())
    if stylesheet:
        widget.setStyleSheet(stylesheet)
    else:
        logger.error(f"Failed to load theme: {theme_name}")
