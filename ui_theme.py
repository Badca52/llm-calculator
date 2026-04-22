"""Dark/light theme + QSS for the LLM VRAM Calculator.

Self-contained — no third-party design-system dependency. Derives a full
palette from a single accent hex color, builds a trimmed Qt stylesheet
covering just the widgets this app uses, and exposes a ThemeManager
singleton so Ctrl+T can flip modes.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication


@dataclass(frozen=True)
class _Tokens:
    base: str
    surface: str
    elevated: str
    border: str
    text_primary: str
    text_secondary: str
    text_muted: str
    accent: str
    accent_hover: str
    accent_subtle: str
    accent_on: str
    card_hover: str


def _adjust_lightness(c: QColor, delta: float) -> QColor:
    h, s, l, a = c.getHslF()
    l = max(0.0, min(1.0, l + delta))
    out = QColor()
    out.setHslF(h, s, l, a)
    return out


def _relative_luminance(c: QColor) -> float:
    def lin(v: float) -> float:
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
    return 0.2126 * lin(c.redF()) + 0.7152 * lin(c.greenF()) + 0.0722 * lin(c.blueF())


def _rgba(c: QColor, alpha: float) -> str:
    return f"rgba({c.red()}, {c.green()}, {c.blue()}, {alpha:.2f})"


def _build_tokens(accent: str, mode: str) -> _Tokens:
    raw = QColor(accent)
    effective = _adjust_lightness(raw, -0.15) if mode == "light" else QColor(raw)
    hover = _adjust_lightness(effective, -0.08)
    accent_on = "#0d1117" if _relative_luminance(effective) > 0.35 else "#ffffff"

    if mode == "dark":
        return _Tokens(
            base="#0d1117",
            surface="#161b22",
            elevated="#1c2333",
            border="#30363d",
            text_primary="#e6edf3",
            text_secondary="#8b949e",
            text_muted="#6e7681",
            accent=effective.name(),
            accent_hover=hover.name(),
            accent_subtle=_rgba(effective, 0.10),
            accent_on=accent_on,
            card_hover="#1f2937",
        )
    return _Tokens(
        base="#f6f8fa",
        surface="#ffffff",
        elevated="#ffffff",
        border="#d1d9e0",
        text_primary="#0d1117",
        text_secondary="#6e7681",
        text_muted="#8b949e",
        accent=effective.name(),
        accent_hover=hover.name(),
        accent_subtle=_rgba(effective, 0.10),
        accent_on=accent_on,
        card_hover="#eaeef2",
    )


_FONT_FAMILY = '"Inter", "SF Pro Display", "Segoe UI", "Noto Sans", sans-serif'


def _build_qss(t: _Tokens) -> str:
    return f"""
QMainWindow {{ background-color: {t.base}; color: {t.text_primary}; }}
QWidget {{ color: {t.text_primary}; font-family: {_FONT_FAMILY}; outline: none; }}

QFrame[class="section-card"] {{
    background-color: {t.elevated};
    border: none;
    border-radius: 10px;
    padding: 20px 24px;
}}

QPushButton {{
    padding: 9px 18px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 500;
    border: 1px solid {t.border};
    background-color: {t.surface};
    color: {t.text_primary};
}}
QPushButton:hover {{ background-color: {t.card_hover}; border-color: {t.accent}; }}
QPushButton:pressed {{ background-color: {t.elevated}; }}
QPushButton:disabled {{
    background-color: {t.elevated};
    color: {t.text_muted};
    border: 1px solid {t.border};
}}
QPushButton#primary {{
    background-color: {t.accent};
    color: {t.accent_on};
    border: none;
    font-weight: 600;
}}
QPushButton#primary:hover, QPushButton#primary:pressed {{
    background-color: {t.accent_hover};
}}

QLineEdit {{
    padding: 9px 14px;
    border: 1px solid {t.border};
    border-radius: 8px;
    background-color: {t.elevated};
    color: {t.text_primary};
    font-size: 13px;
    selection-background-color: {t.accent};
    selection-color: {t.accent_on};
}}
QLineEdit:focus {{ border-color: {t.accent}; }}

QComboBox {{
    padding: 9px 14px;
    border: 1px solid {t.border};
    border-radius: 8px;
    background-color: {t.elevated};
    color: {t.text_primary};
    font-size: 13px;
    min-width: 120px;
}}
QComboBox:focus, QComboBox:on {{ border-color: {t.accent}; }}
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 28px;
    border: none;
}}
QComboBox::down-arrow {{ image: none; width: 0; }}
QComboBox QAbstractItemView {{
    background-color: {t.surface};
    color: {t.text_primary};
    border: 1px solid {t.border};
    border-radius: 6px;
    selection-background-color: {t.accent_subtle};
    selection-color: {t.accent};
    outline: none;
    padding: 4px;
}}

QSpinBox {{
    padding: 8px 12px;
    border: 1px solid {t.border};
    border-radius: 8px;
    background-color: {t.elevated};
    color: {t.text_primary};
    font-size: 13px;
}}
QSpinBox:focus {{ border-color: {t.accent}; }}
QSpinBox::up-button, QSpinBox::down-button {{ border: none; width: 20px; }}

QCheckBox {{ color: {t.text_primary}; spacing: 10px; font-size: 13px; }}
QCheckBox::indicator {{
    width: 18px; height: 18px;
    border-radius: 4px;
    border: 2px solid {t.border};
    background-color: {t.elevated};
}}
QCheckBox::indicator:hover {{ border-color: {t.accent}; }}
QCheckBox::indicator:checked {{
    background-color: {t.accent};
    border-color: {t.accent};
}}

QSlider::groove:horizontal {{ height: 6px; background: {t.border}; border-radius: 3px; }}
QSlider::handle:horizontal {{
    background: {t.accent};
    width: 16px; height: 16px;
    margin: -6px 0;
    border-radius: 8px;
    border: 2px solid {t.accent};
}}
QSlider::handle:horizontal:hover {{
    background: {t.accent_hover};
    border-color: {t.accent_hover};
    width: 18px; height: 18px;
    margin: -7px 0;
    border-radius: 9px;
}}
QSlider::sub-page:horizontal {{ background: {t.accent}; border-radius: 3px; }}

QTableWidget {{
    background-color: {t.surface};
    alternate-background-color: {t.elevated};
    gridline-color: {t.border};
    border: none;
    border-radius: 10px;
    font-size: 13px;
    color: {t.text_primary};
    selection-background-color: {t.accent_subtle};
    selection-color: {t.accent};
    padding: 0px;
}}
QTableWidget::item {{ padding: 8px 14px; border-bottom: 1px solid {t.border}; }}
QHeaderView {{ background-color: transparent; }}
QHeaderView::section {{
    background-color: {t.elevated};
    color: {t.text_secondary};
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
    padding: 10px 14px;
    border: none;
    border-bottom: 1px solid {t.border};
    letter-spacing: 0.5px;
}}

QLabel[class="muted-text"] {{ color: {t.text_secondary}; }}

QToolTip {{
    background-color: {t.surface};
    color: {t.text_primary};
    border: 1px solid {t.border};
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 12px;
}}

QScrollBar:vertical {{ background: transparent; width: 6px; margin: 4px 2px; }}
QScrollBar::handle:vertical {{ background: {t.border}; border-radius: 3px; min-height: 30px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}
"""


class Theme:
    """Immutable theme snapshot."""

    def __init__(self, accent: str = "#22d3ee", mode: str = "dark") -> None:
        if mode not in ("dark", "light"):
            raise ValueError(f"mode must be 'dark' or 'light', got {mode!r}")
        self.accent = accent
        self.mode = mode
        self.tokens = _build_tokens(accent, mode)
        self._qss: str | None = None

    def qss(self) -> str:
        if self._qss is None:
            self._qss = _build_qss(self.tokens)
        return self._qss


class ThemeManager(QObject):
    """Application-wide theme manager singleton."""

    themeChanged = Signal(object)
    _instance: "ThemeManager | None" = None

    def __init__(self) -> None:
        super().__init__()
        self._theme = Theme()

    @classmethod
    def instance(cls) -> "ThemeManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def theme(self) -> Theme:
        return self._theme

    def apply(self, theme: Theme) -> None:
        self._theme = theme
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(theme.qss())
        self.themeChanged.emit(theme)

    def toggle_mode(self) -> None:
        new_mode = "light" if self._theme.mode == "dark" else "dark"
        self.apply(Theme(accent=self._theme.accent, mode=new_mode))
