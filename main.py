"""LLM VRAM Calculator — desktop GUI entry point."""

from __future__ import annotations

import sys

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QSizePolicy,
    QStyle,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pyside6_style_guide import (
    BaseWindow,
    CheckBox,
    ComboBox,
    GradientLabel,
    Heading,
    MutedText,
    PasswordInput,
    PrimaryButton,
    SectionCard,
    Slider,
    SpinBox,
    StyledApp,
    TextInput,
    Theme,
    ThemeManager,
)

from calculator import cell_status, effective_config, kv_cache_gb, param_count, vram_gb
from hf_client import ModelFetcher
from quantization import BPP, DEFAULT_CONTEXTS, KV_CACHE_BYTES, contexts_for, format_ctx

CELL_COLORS = {
    "green":  QColor(34, 102, 74),    # muted emerald
    "yellow": QColor(133, 94, 22),    # muted amber
    "red":    QColor(135, 44, 44),    # muted red
}


class ColoredCellDelegate(QStyledItemDelegate):
    """Paints BackgroundRole fills and multi-line centered text.

    The design-system QSS sets ``QTableWidget { background-color: ... }`` which
    masks ``QTableWidgetItem.setBackground()`` on styled tables.  We bypass
    that by painting the fill ourselves, then rendering the text with
    ``TextWordWrap`` so newline-separated values ("52.0\\n+2.1 RAM") render on
    two lines instead of being elided to "52.0…".
    """

    def paint(self, painter, option, index):
        painter.save()
        brush = index.data(Qt.ItemDataRole.BackgroundRole)
        if isinstance(brush, QBrush) and brush.color().alpha() > 0:
            painter.fillRect(option.rect, brush)

        text = index.data(Qt.ItemDataRole.DisplayRole) or ""
        if text:
            text_color = index.data(Qt.ItemDataRole.ForegroundRole)
            if isinstance(text_color, QBrush):
                painter.setPen(text_color.color())
            else:
                painter.setPen(option.palette.text().color())
            flags = (
                Qt.AlignmentFlag.AlignCenter
                | Qt.TextFlag.TextWordWrap
            )
            rect = option.rect.adjusted(4, 2, -4, -2)
            painter.drawText(rect, int(flags), str(text))
        painter.restore()


class MainWindow(BaseWindow):
    def __init__(self) -> None:
        super().__init__(title="LLM VRAM Calculator", size=(1180, 860))

        self._api: dict | None = None
        self._config: dict | None = None
        self._raw_config: dict | None = None
        self._params: int = 0
        self._n_layers: int = 0
        self._gguf_quants: set[str] = set()
        self._contexts: list[int] = list(DEFAULT_CONTEXTS)
        self._fetcher: ModelFetcher | None = None

        self._recompute_timer = QTimer(self)
        self._recompute_timer.setSingleShot(True)
        self._recompute_timer.setInterval(30)
        self._recompute_timer.timeout.connect(self._recompute_matrix)

        self._build_ui()
        self._wire_signals()

        QShortcut(
            QKeySequence("Ctrl+T"),
            self,
            lambda: ThemeManager.instance().toggle_mode(),
        )
        QShortcut(QKeySequence("Ctrl+Q"), self, QApplication.quit)

    # ------------------------------------------------------------------ UI --

    def _build_ui(self) -> None:
        root = QWidget(self)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(28, 24, 28, 24)
        root_layout.setSpacing(18)

        root_layout.addWidget(GradientLabel("LLM VRAM Calculator", size=26))
        root_layout.addWidget(
            MutedText(
                "Paste a Hugging Face model URL to see VRAM requirements "
                "across quantizations and context lengths.",
                size=12,
            ),
        )

        root_layout.addWidget(self._build_url_section())

        info_controls_row = QHBoxLayout()
        info_controls_row.setSpacing(16)
        info_controls_row.addWidget(self._build_info_section(), stretch=2)
        info_controls_row.addWidget(self._build_controls_section(), stretch=3)
        root_layout.addLayout(info_controls_row)

        root_layout.addWidget(self._build_matrix_section(), stretch=1)

        self.setCentralWidget(root)

    def _build_url_section(self) -> SectionCard:
        card = SectionCard(title="Model")
        content = card.content_layout()

        row = QHBoxLayout()
        row.setSpacing(10)

        self._url_input = TextInput(
            placeholder="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct",
        )
        self._url_input.setMinimumHeight(36)
        row.addWidget(self._url_input, stretch=3)

        self._token_input = PasswordInput(placeholder="HF token (optional)")
        self._token_input.setMinimumHeight(36)
        self._token_input.setMaximumWidth(240)
        row.addWidget(self._token_input, stretch=1)

        self._load_btn = PrimaryButton("Load")
        self._load_btn.setMinimumHeight(36)
        self._load_btn.setMinimumWidth(96)
        row.addWidget(self._load_btn)

        content.addLayout(row)

        self._status_label = MutedText("", size=12)
        content.addWidget(self._status_label)
        return card

    def _build_info_section(self) -> SectionCard:
        card = SectionCard(title="Model info")
        content = card.content_layout()

        self._name_label = Heading("—", size=16)
        content.addWidget(self._name_label)

        self._meta_label = MutedText("Load a model to see its architecture.", size=12)
        self._meta_label.setWordWrap(True)
        content.addWidget(self._meta_label)

        self._moe_warning = MutedText("", size=12)
        self._moe_warning.setStyleSheet("color: #fbbf24;")
        self._moe_warning.hide()
        content.addWidget(self._moe_warning)

        return card

    def _build_controls_section(self) -> SectionCard:
        card = SectionCard(
            title="Controls",
            description="VRAM budget, GPU offload, and KV cache precision.",
        )
        content = card.content_layout()
        content.setSpacing(14)

        # VRAM budget row
        budget_row = QHBoxLayout()
        budget_row.setSpacing(10)
        budget_row.addWidget(self._field_label("GPU VRAM budget"))
        self._budget_spin = SpinBox()
        self._budget_spin.setRange(1, 1024)
        self._budget_spin.setValue(24)
        self._budget_spin.setSuffix(" GB")
        self._budget_spin.setMinimumWidth(120)
        self._budget_spin.setMinimumHeight(32)
        budget_row.addWidget(self._budget_spin)
        budget_row.addStretch()
        content.addLayout(budget_row)

        # KV cache dtype + CPU-offload toggle
        kv_row = QHBoxLayout()
        kv_row.setSpacing(10)
        kv_row.addWidget(self._field_label("KV cache dtype"))
        self._kv_combo = ComboBox()
        self._kv_combo.addItems(list(KV_CACHE_BYTES.keys()))
        self._kv_combo.setCurrentText("FP16")
        self._kv_combo.setMinimumWidth(120)
        self._kv_combo.setMinimumHeight(32)
        kv_row.addWidget(self._kv_combo)

        self._kv_cpu_check = CheckBox("CPU KV Cache")
        self._kv_cpu_check.setToolTip(
            "When enabled, the KV cache stays in system RAM instead of GPU VRAM "
            "(llama.cpp --no-kv-offload).\n"
            "GPU VRAM then covers only model weights, so long contexts become\n"
            "feasible on small GPUs — but inference is much slower because the\n"
            "GPU streams KV entries over PCIe each step.\n\n"
            "When on, cells show VRAM on the top line and the system RAM\n"
            "required for the KV cache below (+X.X RAM).",
        )
        kv_row.addSpacing(20)
        kv_row.addWidget(self._kv_cpu_check)
        kv_row.addStretch()
        content.addLayout(kv_row)

        # GPU layers slider
        layers_row = QHBoxLayout()
        layers_row.setSpacing(10)
        self._layers_label = self._field_label("GPU layers offloaded (load a model first)")
        self._layers_label.setMinimumWidth(260)
        layers_row.addWidget(self._layers_label)
        self._layers_slider = Slider(min_val=0, max_val=1, value=1, show_value=True)
        self._layers_slider.setEnabled(False)
        layers_row.addWidget(self._layers_slider, stretch=1)
        content.addLayout(layers_row)

        return card

    def _build_matrix_section(self) -> SectionCard:
        card = SectionCard(title="VRAM estimates (GB)")
        content = card.content_layout()
        content.setSpacing(8)

        self._table = QTableWidget(len(BPP), len(self._contexts))
        self._table.setItemDelegate(ColoredCellDelegate(self._table))
        # The design-system QSS fills every ::item with the table's surface
        # color, masking our item-level BackgroundRole.  Null that rule out
        # locally so the delegate fill shows through; keep the padding.
        self._table.setStyleSheet(
            "QTableWidget::item { background: transparent; padding: 8px 14px; "
            "border-bottom: 1px solid palette(mid); }",
        )
        self._table.setHorizontalHeaderLabels([format_ctx(c) for c in self._contexts])
        self._table.setVerticalHeaderLabels(list(BPP.keys()))
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._table.setAlternatingRowColors(False)
        self._table.verticalHeader().setDefaultSectionSize(46)
        self._table.setWordWrap(True)
        self._table.setTextElideMode(Qt.TextElideMode.ElideNone)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch,
        )
        self._table.verticalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Fixed,
        )
        self._table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._populate_empty_cells()

        content.addWidget(self._table, stretch=1)

        legend = QHBoxLayout()
        legend.setSpacing(16)
        legend.addWidget(self._legend_swatch("green", "Fits"))
        legend.addWidget(self._legend_swatch("yellow", "Tight (<1 GB headroom)"))
        legend.addWidget(self._legend_swatch("red", "Exceeds budget"))
        legend.addStretch()
        legend.addWidget(
            MutedText("Estimates ±3–5%. Real usage varies with backend and batch size.", size=11),
        )
        content.addLayout(legend)

        self._fit_hint_label = MutedText("", size=12)
        self._fit_hint_label.setWordWrap(True)
        content.addWidget(self._fit_hint_label)

        return card

    def _field_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setMinimumWidth(200)
        return lbl

    def _legend_swatch(self, key: str, text: str) -> QWidget:
        wrap = QWidget()
        row = QHBoxLayout(wrap)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        swatch = QLabel()
        swatch.setFixedSize(14, 14)
        color = CELL_COLORS[key]
        swatch.setStyleSheet(
            f"background: rgba({color.red()}, {color.green()}, "
            f"{color.blue()}, {color.alpha() / 255:.2f}); border-radius: 3px;",
        )
        row.addWidget(swatch)
        row.addWidget(MutedText(text, size=11))
        return wrap

    # ------------------------------------------------------------- signals --

    def _wire_signals(self) -> None:
        self._load_btn.clicked.connect(self._start_fetch)
        self._url_input.returnPressed.connect(self._start_fetch)
        self._token_input.returnPressed.connect(self._start_fetch)

        self._budget_spin.valueChanged.connect(self._schedule_recompute)
        self._kv_combo.currentTextChanged.connect(self._schedule_recompute)
        self._kv_cpu_check.stateChanged.connect(self._schedule_recompute)
        self._layers_slider.slider.valueChanged.connect(self._schedule_recompute)

    def _schedule_recompute(self, *_) -> None:
        self._recompute_timer.start()

    # ---------------------------------------------------------- fetch flow --

    def _start_fetch(self) -> None:
        if self._fetcher and self._fetcher.isRunning():
            return
        url = self._url_input.text().strip()
        if not url:
            self._set_status("Enter a Hugging Face model URL.", error=True)
            return

        self._load_btn.setEnabled(False)
        self._set_status("Fetching model info…", error=False)

        token = self._token_input.text().strip() or None
        self._fetcher = ModelFetcher(url, token)
        self._fetcher.finished_ok.connect(self._on_fetch_finished)
        self._fetcher.failed.connect(self._on_fetch_failed)
        self._fetcher.finished.connect(self._on_fetch_cleanup)
        self._fetcher.start()

    def _on_fetch_finished(self, data: dict) -> None:
        self._api = data["api"]
        raw_config = data["config"]
        self._raw_config = raw_config
        self._config = effective_config(raw_config)
        self._gguf_quants = set(data.get("gguf_quants") or [])

        try:
            self._params = param_count(self._api, raw_config)
            self._n_layers = int(self._config["num_hidden_layers"])
        except KeyError as e:
            self._set_status(
                f"config.json missing field: {e}. "
                "This model's architecture may not be supported yet.",
                error=True,
            )
            return

        self._populate_model_info(data["model_id"], data.get("base_model_id"))
        self._rebuild_context_columns()
        self._apply_gguf_row_highlighting()
        self._configure_slider()

        if data.get("base_model_id"):
            self._set_status(
                f"Loaded GGUF repo {data['model_id']} — "
                f"architecture resolved from base model {data['base_model_id']}.",
                error=False,
            )
        else:
            self._set_status(f"Loaded {data['model_id']}.", error=False)
        self._recompute_matrix()

    def _on_fetch_failed(self, message: str) -> None:
        self._set_status(message, error=True)

    def _on_fetch_cleanup(self) -> None:
        self._load_btn.setEnabled(True)

    # ---------------------------------------------------- display helpers --

    def _populate_model_info(self, model_id: str, base_model_id: str | None = None) -> None:
        if base_model_id:
            self._name_label.setText(f"{model_id}  →  {base_model_id}")
        else:
            self._name_label.setText(model_id)

        hidden = self._config["hidden_size"]
        n_heads = self._config["num_attention_heads"]
        n_kv = self._config.get("num_key_value_heads", n_heads)
        head_dim = self._config.get("head_dim", hidden // n_heads)
        max_ctx = self._config.get("max_position_embeddings")
        max_ctx_str = f"{max_ctx:,}" if isinstance(max_ctx, int) else "unknown"
        tied = self._config.get("tie_word_embeddings", False)

        self._meta_label.setText(
            f"{self._params / 1e9:.2f}B params · {self._n_layers} layers · "
            f"hidden {hidden} · {n_kv} KV heads (of {n_heads}) · "
            f"head dim {head_dim} · max ctx {max_ctx_str}"
            + (" · tied embeddings" if tied else ""),
        )

        warnings = []

        if self._gguf_quants:
            available = ", ".join(sorted(self._gguf_quants))
            warnings.append(
                f"GGUF repo — {len(self._gguf_quants)} quant(s) available here: {available} "
                "(bold rows in the matrix below).",
            )
        is_moe = any(
            k in self._config for k in (
                "num_local_experts", "num_experts", "n_routed_experts",
            )
        ) or self._config.get("enable_moe_block") is True
        if is_moe:
            n_experts = (
                self._config.get("num_experts")
                or self._config.get("num_local_experts")
                or self._config.get("n_routed_experts")
            )
            top_k = (
                self._config.get("top_k_experts")
                or self._config.get("num_experts_per_tok")
            )
            detail = ""
            if n_experts and top_k:
                detail = f" ({top_k} of {n_experts} experts active per token)"
            warnings.append(
                f"⚠ MoE model{detail} — all experts must stay resident, "
                "but only active params compute per token.",
            )

        has_vision = isinstance(self._raw_config.get("vision_config"), dict)
        has_audio = isinstance(self._raw_config.get("audio_config"), dict)
        if has_vision or has_audio:
            mods = [m for m, present in (("vision", has_vision), ("audio", has_audio)) if present]
            warnings.append(
                f"⚠ Multimodal model ({'+'.join(mods)} tower included). "
                "VRAM covers the full model; KV cache uses text-only dims.",
            )

        if warnings:
            self._moe_warning.setText("\n".join(warnings))
            self._moe_warning.show()
        else:
            self._moe_warning.hide()

    def _populate_empty_cells(self) -> None:
        for row in range(len(BPP)):
            for col in range(len(self._contexts)):
                if self._table.item(row, col) is None:
                    item = QTableWidgetItem("—")
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self._table.setItem(row, col, item)

    def _rebuild_context_columns(self) -> None:
        max_ctx = (self._config or {}).get("max_position_embeddings")
        self._contexts = contexts_for(max_ctx if isinstance(max_ctx, int) else None)
        self._table.setColumnCount(len(self._contexts))
        self._table.setHorizontalHeaderLabels([format_ctx(c) for c in self._contexts])
        self._populate_empty_cells()

    def _apply_gguf_row_highlighting(self) -> None:
        """Bold vertical header labels for quantizations present as .gguf files."""
        for row, quant in enumerate(BPP.keys()):
            header_item = self._table.verticalHeaderItem(row)
            if header_item is None:
                header_item = QTableWidgetItem(quant)
                self._table.setVerticalHeaderItem(row, header_item)
            font = header_item.font()
            font.setBold(quant in self._gguf_quants)
            header_item.setFont(font)
            if quant in self._gguf_quants:
                header_item.setText(f"● {quant}")
            else:
                header_item.setText(quant)

    def _configure_slider(self) -> None:
        slider = self._layers_slider.slider
        slider.setRange(0, self._n_layers)
        slider.setValue(self._n_layers)
        self._layers_slider.setEnabled(True)
        self._update_layers_label()

    def _update_layers_label(self) -> None:
        current = self._layers_slider.value()
        self._layers_label.setText(
            f"GPU layers offloaded: {current} / {self._n_layers}",
        )

    def _set_status(self, text: str, *, error: bool) -> None:
        self._status_label.setText(text)
        color = "#f87171" if error else "#94a3b8"
        self._status_label.setStyleSheet(f"color: {color};")

    # -------------------------------------------------- matrix recomputation --

    def _recompute_matrix(self) -> None:
        if not self._config:
            return

        self._update_layers_label()

        budget = float(self._budget_spin.value())
        kv_dtype = self._kv_combo.currentText()
        gpu_layers = self._layers_slider.value()
        kv_on_cpu = self._kv_cpu_check.isChecked()

        quant_keys = list(BPP.keys())
        max_ctx = self._config.get("max_position_embeddings")

        exceeded_rows: list[tuple[str, float]] = []

        for row, quant in enumerate(quant_keys):
            row_fits_budget = False
            for col, ctx in enumerate(self._contexts):
                gb = vram_gb(
                    self._params, self._config, quant, ctx, kv_dtype,
                    gpu_layers, kv_on_cpu,
                )
                status = cell_status(gb, budget)
                item = self._table.item(row, col)
                if kv_on_cpu:
                    kv_ram = kv_cache_gb(self._config, ctx, kv_dtype)
                    item.setText(f"{gb:.1f}\n+{kv_ram:.1f} RAM")
                else:
                    item.setText(f"{gb:.1f}")
                item.setBackground(QBrush(CELL_COLORS[status]))
                tip = (
                    f"{quant} · {ctx:,} ctx · KV {kv_dtype}"
                    + (" on CPU RAM" if kv_on_cpu else "")
                    + f" · {gpu_layers}/{self._n_layers} layers on GPU\n"
                    + f"≈ {gb:.2f} GB VRAM"
                )
                if kv_on_cpu:
                    tip += f" + {kv_cache_gb(self._config, ctx, kv_dtype):.2f} GB system RAM"
                if isinstance(max_ctx, int) and ctx > max_ctx:
                    tip += f"\n(exceeds model's native max_position_embeddings of {max_ctx:,})"
                item.setToolTip(tip)
                if status != "red":
                    row_fits_budget = True
            if not row_fits_budget:
                smallest_gb = vram_gb(
                    self._params, self._config, quant, self._contexts[0],
                    kv_dtype, gpu_layers, kv_on_cpu,
                )
                exceeded_rows.append((quant, smallest_gb))

        self._update_fit_hint(exceeded_rows, budget, kv_dtype, gpu_layers)

    def _update_fit_hint(
        self,
        exceeded: list[tuple[str, float]],
        budget: float,
        kv_dtype: str,
        gpu_layers: int,
    ) -> None:
        if not exceeded:
            self._fit_hint_label.setText("")
            self._fit_hint_label.hide()
            return

        smallest_ctx = self._contexts[0]
        lines = []
        for quant, gb in exceeded:
            lines.append(
                f"  • {quant} at {smallest_ctx:,} tokens needs {gb:.1f} GB — "
                f"{gb - budget:.1f} GB over budget",
            )
        self._fit_hint_label.setText(
            f"Rows that don't fit a {budget:.0f} GB budget at any context size:\n"
            + "\n".join(lines),
        )
        self._fit_hint_label.show()


def main() -> int:
    app = StyledApp(theme=Theme(accent="#22d3ee"))
    window = MainWindow()
    window.show()
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
