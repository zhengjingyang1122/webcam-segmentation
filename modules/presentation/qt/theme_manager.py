"""Theme manager for webcam applications.

Provides light and dark theme stylesheets for consistent UI theming.
"""

LIGHT_THEME = """
QWidget {
    background-color: #fafafa;
    color: #212121;
    font-size: 12px;
}

QMainWindow {
    background-color: #ffffff;
}

QGroupBox {
    margin-top: 10px;
    padding: 8px;
    border: 2px solid #e0e0e0;
    border-radius: 8px;
    background-color: #ffffff;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    color: #1976d2;
    font-weight: 600;
}

QPushButton {
    padding: 8px 12px;
    border: 1px solid #1976d2;
    border-radius: 6px;
    background: #2196f3;
    color: #ffffff;
    font-weight: 500;
}

QPushButton:hover {
    background: #1976d2;
    border: 1px solid #1565c0;
}

QPushButton:pressed {
    background: #1565c0;
}

QPushButton:disabled {
    background: #e0e0e0;
    color: #9e9e9e;
    border: 1px solid #bdbdbd;
}

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    border: 2px solid #bdbdbd;
    border-radius: 6px;
    padding: 6px 8px;
    background: #ffffff;
    color: #212121;
}

QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 2px solid #2196f3;
    background: #f5f9ff;
}

QMenuBar {
    background-color: #f5f5f5;
    color: #212121;
    border-bottom: 1px solid #e0e0e0;
}

QMenuBar::item {
    padding: 6px 12px;
    background-color: transparent;
}

QMenuBar::item:selected {
    background-color: #e3f2fd;
    color: #1976d2;
}

QMenu {
    background-color: #ffffff;
    color: #212121;
    border: 1px solid #e0e0e0;
}

QMenu::item {
    padding: 6px 24px;
}

QMenu::item:selected {
    background-color: #e3f2fd;
    color: #1976d2;
}

QDockWidget {
    color: #212121;
}

QDockWidget::title {
    background-color: #f5f5f5;
    padding: 6px;
    border-bottom: 1px solid #e0e0e0;
}

QTreeView, QListView {
    background-color: #ffffff;
    color: #212121;
    border: 1px solid #e0e0e0;
}

QTreeView::item:selected, QListView::item:selected {
    background-color: #2196f3;
    color: #ffffff;
}

QTreeView::item:hover, QListView::item:hover {
    background-color: #e3f2fd;
}

QRadioButton, QCheckBox {
    color: #212121;
}

QRadioButton::indicator:checked, QCheckBox::indicator:checked {
    background-color: #2196f3;
    border: 2px solid #1976d2;
}

QLabel {
    color: #212121;
}

QProgressBar {
    border: 2px solid #e0e0e0;
    border-radius: 6px;
    text-align: center;
    background-color: #f5f5f5;
}

QProgressBar::chunk {
    background-color: #2196f3;
    border-radius: 4px;
}
"""

DARK_THEME = """
QWidget {
    background-color: #1b1e23;
    color: #e8eaed;
    font-size: 12px;
}

QMainWindow {
    background-color: #1b1e23;
}

QGroupBox {
    margin-top: 10px;
    padding: 8px;
    border: 1px solid #3a3f47;
    border-radius: 8px;
    background-color: #1f2227;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    color: #cfd8dc;
    font-weight: 600;
}

QPushButton {
    padding: 6px 10px;
    border: 1px solid #3a3f47;
    border-radius: 6px;
    background: #2b2f36;
    color: #e8eaed;
}

QPushButton:hover {
    background: #333844;
}

QPushButton:pressed {
    background: #3a3f47;
}

QPushButton:disabled {
    background: #1f2227;
    color: #5f6368;
}

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    border: 1px solid #3a3f47;
    border-radius: 6px;
    padding: 4px 6px;
    background: #1b1e23;
    color: #e8eaed;
}

QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #4285f4;
}

QMenuBar {
    background-color: #1b1e23;
    color: #e8eaed;
}

QMenuBar::item:selected {
    background-color: #2b2f36;
}

QMenu {
    background-color: #1b1e23;
    color: #e8eaed;
    border: 1px solid #3a3f47;
}

QMenu::item:selected {
    background-color: #2b2f36;
}

QDockWidget {
    color: #e8eaed;
}

QDockWidget::title {
    background-color: #2b2f36;
    padding: 4px;
}

QTreeView, QListView {
    background-color: #1b1e23;
    color: #e8eaed;
    border: 1px solid #3a3f47;
}

QTreeView::item:selected, QListView::item:selected {
    background-color: #4285f4;
    color: #ffffff;
}

QRadioButton, QCheckBox {
    color: #e8eaed;
}

QLabel {
    color: #e8eaed;
}
"""


def apply_theme(widget, theme_name: str = "dark"):
    """Apply a theme to a widget.
    
    Args:
        widget: The widget to apply the theme to
        theme_name: Either "light" or "dark"
    """
    if theme_name.lower() == "light":
        widget.setStyleSheet(LIGHT_THEME)
    else:
        widget.setStyleSheet(DARK_THEME)
