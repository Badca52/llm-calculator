"""Small widget helpers used by the calculator UI.

Kept intentionally minimal: a SectionCard (titled QFrame), a Slider that
shows its current value, a GradientLabel that tracks the accent color,
and convenience Heading/MutedText QLabel wrappers.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetrics,
    QLinearGradient,
    QPainter,
    QPen,
)
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ui_theme import Theme, ThemeManager


class SectionCard(QFrame):
    """Elevated container with an optional title and description.

    Styled via ``QFrame[class="section-card"]`` in the theme QSS.
    Children go in ``.content_layout()``.
    """

    def __init__(
        self,
        title: str = "",
        description: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setProperty("class", "section-card")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)

        if title:
            title_label = QLabel(title, self)
            font = title_label.font()
            font.setPointSize(14)
            font.setBold(True)
            title_label.setFont(font)
            outer.addWidget(title_label)

        if description:
            desc_label = QLabel(description, self)
            desc_label.setProperty("class", "muted-text")
            desc_label.setWordWrap(True)
            outer.addWidget(desc_label)

        self._content_layout = QVBoxLayout()
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        outer.addLayout(self._content_layout)

    def content_layout(self) -> QVBoxLayout:
        return self._content_layout


class Slider(QWidget):
    """Horizontal QSlider with an optional right-aligned value label."""

    def __init__(
        self,
        min_val: int = 0,
        max_val: int = 100,
        value: int = 50,
        show_value: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setRange(min_val, max_val)
        self.slider.setValue(value)
        layout.addWidget(self.slider, 1)

        if show_value:
            self._label: QLabel | None = QLabel(str(value), self)
            self._label.setMinimumWidth(36)
            self._label.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            )
            layout.addWidget(self._label)
            self.slider.valueChanged.connect(
                lambda v: self._label.setText(str(v)),
            )
        else:
            self._label = None

    def value(self) -> int:
        return self.slider.value()


class GradientLabel(QWidget):
    """Bold text filled with a horizontal gradient derived from the accent."""

    def __init__(
        self,
        text: str = "",
        size: int = 22,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._text = text
        self._font = QFont()
        self._font.setPointSize(size)
        self._font.setBold(True)
        self._color_start = QColor("#22d3ee")
        self._color_end = QColor("#67e8f9")

        self._apply_theme(ThemeManager.instance().theme)
        ThemeManager.instance().themeChanged.connect(self._apply_theme)
        self._update_size()

    def _apply_theme(self, theme: Theme) -> None:
        accent = QColor(theme.tokens.accent)
        self._color_start = accent
        h, s, l, a = accent.getHslF()
        lighter = QColor()
        lighter.setHslF(h, s, min(1.0, l + 0.15), a)
        self._color_end = lighter
        self.update()

    def _update_size(self) -> None:
        fm = QFontMetrics(self._font)
        r = fm.boundingRect(self._text)
        self.setFixedSize(r.width() + 20, r.height() + 10)

    def paintEvent(self, event) -> None:  # noqa: D401
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(self._font)
        text_width = painter.fontMetrics().horizontalAdvance(self._text)
        gradient = QLinearGradient(QPointF(0, 0), QPointF(text_width, 0))
        gradient.setColorAt(0.0, self._color_start)
        gradient.setColorAt(1.0, self._color_end)
        painter.setPen(QPen(QBrush(gradient), 1))
        painter.drawText(
            self.rect(),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            self._text,
        )
        painter.end()


class Heading(QLabel):
    def __init__(self, text: str, size: int = 16, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        font = self.font()
        font.setPointSize(size)
        font.setBold(True)
        self.setFont(font)


class MutedText(QLabel):
    def __init__(self, text: str, size: int = 12, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setProperty("class", "muted-text")
        font = self.font()
        font.setPointSize(size)
        self.setFont(font)
