import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon, QValidator
from PySide6.QtWidgets import QDoubleSpinBox, QGraphicsOpacityEffect, QGroupBox, QVBoxLayout, QHBoxLayout, QToolButton


GROUP_BOX_STYLE = """
    QPushButton {
        background-color: #dcdcdc;
        color: #111111;
        border: 0px;
        border-radius: 8px;
        padding: 6px 10px;
    }

    QPushButton:hover {
        background-color: #d2d2d2;
    }

    QPushButton:pressed {
        background-color: #c8c8c8;
    }

    QPushButton:disabled {
        background-color: #eeeeee;
        color: #888888;
    }


    QCheckBox {
        background-color: transparent;
        color: #111111;
    }

    QListWidget QScrollBar:vertical {
        width: 0px;
        background: transparent;
        border: none;
    }

    QListWidget QScrollBar::handle:vertical {
        background: transparent;
        border: none;
    }

    QListWidget QScrollBar::add-line:vertical,
    QListWidget QScrollBar::sub-line:vertical,
    QListWidget QScrollBar::add-page:vertical,
    QListWidget QScrollBar::sub-page:vertical {
        background: transparent;
        border: none;
        height: 0px;
    }

    QListWidget QScrollBar:horizontal {
        height: 0px;
        background: transparent;
        border: none;
    }

    QListWidget QScrollBar::handle:horizontal {
        background: transparent;
        border: none;
    }

    QListWidget QScrollBar::add-line:horizontal,
    QListWidget QScrollBar::sub-line:horizontal,
    QListWidget QScrollBar::add-page:horizontal,
    QListWidget QScrollBar::sub-page:horizontal {
        background: transparent;
        border: none;
        width: 0px;
    }

    QListWidget {
        background-color: transparent;
        border: none;
    }

    QListWidget::viewport {
        background-color: transparent;
    }
"""


TOOL_GROUP_BOX_STYLE = GROUP_BOX_STYLE + """
    QToolBar {
        background-color: transparent;
        border: 0px;
        spacing: 8px;
    }

    QToolButton {
        background-color: transparent;
        border: 0px;
        padding: 4px;
    }

    QToolButton:hover {
        background-color: #e5e5e5;
        border-radius: 5px;
    }
"""


GROUP_BOX_MARGINS = (8, 20, 8, 8)
BLOCK_SPACING = 8
PAGE_MARGINS = (4, 4, 4, 4)
PANEL_MARGINS = (0, 0, 0, 0)
FRAME_NAV_SPACING = 6
FRAME_SPIN_WIDTH = 70
FRAME_BUTTON_WIDTH = 44
FRAME_COUNTER_WIDTH = 56
FILE_BROWSER_WIDTH = 320
MATPLOTLIB_TOOLBAR_ICON_SCALE = 0.8
MATPLOTLIB_TOOLBAR_MAX_HEIGHT = 42
MATPLOTLIB_TOOLBAR_BUTTON_SIZE = QSize(32, 32)
MATPLOTLIB_TOOLBAR_EMOJI_SIZE = 28


def normalize_decimal_text(text):
    text = str(text).strip().replace(" ", "")
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    else:
        text = text.replace(",", ".")
    return text


def decimal_number_text(text):
    match = re.search(r"[-+]?(?:\d[\d.,]*|[\d.,]*\d)(?:[eE][-+]?\d+)?", str(text))
    return normalize_decimal_text(match.group(0)) if match else ""


def matplotlib_toolbar_action_emoji(action):
    try:
        text = action.text().lower()
    except Exception:
        text = ""
    try:
        tooltip = action.toolTip().lower()
    except Exception:
        tooltip = ""
    label = f"{text} {tooltip}"

    if "home" in label:
        return "🏠"
    if "back" in label:
        return "⬅️"
    if "forward" in label:
        return "➡️"
    if "pan" in label:
        return "✋"
    if "zoom" in label:
        return "🔍"
    if "customize" in label or "edit axis" in label or "axis" in label:
        return "⚙️"
    if "save" in label:
        return "💾"
    return None


def toolbar_action_text(action):
    try:
        text = action.text().lower()
    except Exception:
        text = ""
    try:
        tooltip = action.toolTip().lower()
    except Exception:
        tooltip = ""
    return f"{text} {tooltip}"


def emojiize_matplotlib_toolbar(toolbar, button_size=MATPLOTLIB_TOOLBAR_BUTTON_SIZE, remove_customize=False):
    try:
        toolbar.setToolButtonStyle(Qt.ToolButtonTextOnly)
    except Exception:
        pass

    for action in list(toolbar.actions()):
        label = toolbar_action_text(action)
        if "subplots" in label or (remove_customize and ("customize" in label or "edit axis" in label)):
            try:
                toolbar.removeAction(action)
            except Exception:
                pass

    for action in toolbar.actions():
        emoji = matplotlib_toolbar_action_emoji(action)
        if not emoji:
            continue
        try:
            original_text = action.text()
            if not action.toolTip():
                action.setToolTip(original_text)
            action.setIcon(QIcon())
            action.setText(emoji)
        except Exception:
            pass

    for action in toolbar.actions():
        widget = toolbar.widgetForAction(action)
        if isinstance(widget, QToolButton):
            try:
                widget.setFixedSize(button_size)
                widget.setStyleSheet(f"""
                    QToolButton {{
                        background: transparent;
                        background-color: transparent;
                        border: none;
                        padding: 0px;
                        margin: 0px;
                        font-size: {MATPLOTLIB_TOOLBAR_EMOJI_SIZE}px;
                        min-width: {button_size.width()}px;
                        max-width: {button_size.width()}px;
                        min-height: {button_size.height()}px;
                        max-height: {button_size.height()}px;
                    }}
                """)
            except Exception:
                pass


def set_matplotlib_toolbar_enabled(toolbar, enabled):
    if toolbar is None:
        return
    for action in toolbar.actions():
        try:
            if not action.isSeparator():
                action.setEnabled(enabled)
        except Exception:
            pass
    for action in toolbar.actions():
        widget = toolbar.widgetForAction(action)
        if isinstance(widget, QToolButton):
            set_widget_enabled_with_opacity(widget, enabled)


def set_widget_enabled_with_opacity(widget, enabled, disabled_opacity=0.35):
    if widget is None:
        return
    try:
        widget.setEnabled(enabled)
    except Exception:
        pass
    try:
        if enabled:
            widget.setGraphicsEffect(None)
        else:
            effect = QGraphicsOpacityEffect(widget)
            effect.setOpacity(disabled_opacity)
            widget.setGraphicsEffect(effect)
    except Exception:
        pass


class FlexibleDoubleSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox accepting both comma and point decimal separators."""

    def validate(self, text, pos):
        normalized = decimal_number_text(text)
        if text.strip() in {"", "-", "+", ",", ".", "-,", "-.", "+,", "+."}:
            state = QValidator.Intermediate
        elif not normalized:
            state = QValidator.Invalid
        else:
            try:
                value = float(normalized)
            except ValueError:
                state = QValidator.Invalid
            else:
                state = QValidator.Acceptable if self.minimum() <= value <= self.maximum() else QValidator.Intermediate
        return state, text, pos

    def valueFromText(self, text):
        try:
            return float(decimal_number_text(text))
        except ValueError:
            return self.value()

    def textFromValue(self, value):
        try:
            decimals = max(0, int(self.decimals()))
            quant = Decimal(1).scaleb(-decimals)
            number = Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP)
            text = format(number, "f")
        except (InvalidOperation, ValueError):
            text = super().textFromValue(value)

        text = text.rstrip("0").rstrip(".")
        if text in {"", "-", "+"}:
            text += "0"
        return text.replace(".", ",")


Q_GEOMETRY_BUTTON_STYLE = """
    QPushButton {

        color: #222222;

        padding: 4px;

    }
    QPushButton:hover {
        background-color: #d8d8d8;
    }
"""


Q_GEOMETRY_BUTTON_ACTIVE_STYLE = """
    QPushButton {
        background-color: #007aff;
        color: white;
        border: 0px;
        border-radius: 5px;
        padding: 4px;
    }
"""



ACTION_BUTTON_STYLE = """
    QPushButton {
        background-color: #dcdcdc;
        color: #111111;
        border: 0px;
        border-radius: 8px;
        padding: 6px 10px;
    }

    QPushButton:hover {
        background-color: #d2d2d2;
    }

    QPushButton:pressed {
        background-color: #c8c8c8;
    }

    QPushButton:disabled {
        background-color: #eeeeee;
        color: #888888;
    }
"""


COMPACT_COMBO_STYLE = ""


def style_q_geometry_buttons(buttons, active_mode=None, manual_button=None):
    minimum_widths = {
        "XENOCS": 72,
        "ID02": 56,
        "ID13": 56,
        "Custom": 68,
    }

    for mode, button in buttons.items():
        active = mode == active_mode
        button.setCheckable(True)
        button.setMinimumWidth(minimum_widths.get(mode, 56))
        button.blockSignals(True)
        button.setChecked(active)
        button.blockSignals(False)
        button.setStyleSheet(
            Q_GEOMETRY_BUTTON_ACTIVE_STYLE if active else Q_GEOMETRY_BUTTON_STYLE
        )

    if manual_button is not None:
        manual_button.setFixedWidth(28)
        manual_button.setStyleSheet(Q_GEOMETRY_BUTTON_STYLE)


def apply_plot_display_style(ax):
    ax.grid(True, linewidth=0.5, alpha=0.35)
    ax.tick_params(axis="both", labelsize=10)


def _figure_axes(ax):
    return list(getattr(ax.figure, "axes", [ax]))


def _plot_lines_for_legend_label(ax, label):
    matching = []
    selected_gids = set()

    for target_ax in _figure_axes(ax):
        for line in target_ax.get_lines():
            if line.get_label() == label:
                matching.append(line)
                gid = line.get_gid()
                if gid is not None:
                    selected_gids.add(gid)

    if selected_gids:
        for target_ax in _figure_axes(ax):
            for line in target_ax.get_lines():
                if line.get_gid() in selected_gids and line not in matching:
                    matching.append(line)

    return matching


def _set_legend_selection(ax, selected_label):
    ax.__dict__["_lrphoton_selected_legend_label"] = selected_label
    selected_lines = _plot_lines_for_legend_label(ax, selected_label) if selected_label else []
    selected_ids = {id(line) for line in selected_lines}
    selected_gids = {line.get_gid() for line in selected_lines if line.get_gid() is not None}

    for target_ax in _figure_axes(ax):
        for line in target_ax.get_lines():
            label = line.get_label()
            if label.startswith("_") and line.get_gid() not in selected_gids:
                continue

            if "_lrphoton_base_linewidth" not in line.__dict__:
                line.__dict__["_lrphoton_base_linewidth"] = line.get_linewidth()

            base_width = line.__dict__["_lrphoton_base_linewidth"]
            is_selected = id(line) in selected_ids or line.get_gid() in selected_gids
            line.set_linewidth(base_width * 2.4 if is_selected else base_width)
            line.set_zorder(10 if is_selected else 2)

    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontweight("bold" if text.get_text() == selected_label else "normal")
        for legend_line, text in zip(legend.get_lines(), legend.get_texts()):
            legend_line.set_linewidth(2.8 if text.get_text() == selected_label else 1.2)


def install_selectable_legend(ax, legend):
    if legend is None:
        return None

    legend.__dict__["_lrphoton_axes"] = ax
    for text in legend.get_texts():
        text.set_picker(True)
    for line in legend.get_lines():
        line.set_picker(True)

    canvas = ax.figure.canvas
    if not getattr(canvas, "_lrphoton_legend_pick_connected", False):
        def handle_pick(event):
            artist = event.artist
            legend = None
            if hasattr(artist, "axes") and artist.axes is not None:
                legend = artist.axes.get_legend()

            if legend is None:
                for figure_ax in getattr(event.canvas.figure, "axes", []):
                    candidate = figure_ax.get_legend()
                    if candidate is not None and artist in (list(candidate.get_texts()) + list(candidate.get_lines())):
                        legend = candidate
                        break
            if legend is None:
                return

            label = None
            if artist in legend.get_texts():
                label = artist.get_text()
            elif artist in legend.get_lines():
                lines = list(legend.get_lines())
                texts = list(legend.get_texts())
                try:
                    label = texts[lines.index(artist)].get_text()
                except (ValueError, IndexError):
                    label = None
            if not label:
                return

            target_ax = legend.__dict__.get("_lrphoton_axes")
            if target_ax is None:
                target_ax = legend.axes
            selected_label = target_ax.__dict__.get("_lrphoton_selected_legend_label")
            _set_legend_selection(target_ax, None if selected_label == label else label)
            event.canvas.draw_idle()

        canvas.mpl_connect("pick_event", handle_pick)
        canvas._lrphoton_legend_pick_connected = True

    return legend


def make_plot_legend(ax):
    legend = ax.legend(
        loc="best",
        frameon=True,
        fontsize=7,
        handlelength=1.8,
        borderpad=0.35,
        labelspacing=0.3,
        handletextpad=0.5,
    )
    legend.set_draggable(True)
    return install_selectable_legend(ax, legend)


def finalize_plot_canvas(canvas):
    try:
        canvas.fig.tight_layout()
    except Exception:
        pass
    canvas.draw_idle()


def clear_plot_canvas(canvas):
    canvas.fig.patch.set_facecolor("white")
    canvas.ax.clear()
    canvas.ax.set_facecolor("white")
    canvas.ax.set_axis_off()
    canvas.draw_idle()


def constrain_image_axes(ax, image_shape=None, x_bounds=None, y_bounds=None):
    if x_bounds is None or y_bounds is None:
        if image_shape is None or len(image_shape) < 2:
            return

        ny, nx = image_shape[:2]
        if nx <= 0 or ny <= 0:
            return

        x_bounds = (-0.5, nx - 0.5)
        y_bounds = (-0.5, ny - 0.5)

    if x_bounds is None or y_bounds is None:
        return

    x_lower, x_upper = min(x_bounds), max(x_bounds)
    y_lower, y_upper = min(y_bounds), max(y_bounds)
    if x_upper <= x_lower or y_upper <= y_lower:
        return

    def constrained_limits(limits, lower, upper):
        first, second = limits
        reversed_axis = first > second
        low = min(first, second)
        high = max(first, second)
        full_span = upper - lower
        span = high - low

        if span >= full_span:
            low, high = lower, upper
        else:
            if low < lower:
                high += lower - low
                low = lower
            if high > upper:
                low -= high - upper
                high = upper

        return (high, low) if reversed_axis else (low, high)

    ax.set_xlim(constrained_limits(ax.get_xlim(), x_lower, x_upper))
    ax.set_ylim(constrained_limits(ax.get_ylim(), y_lower, y_upper))


def make_matplotlib_toolbar_block(parent, title, toolbar, option_widgets=None, save_callback=None, save_tooltip="Save", toolbar_width=340, remove_customize=False):
    toolbar_box = QGroupBox(title)
    toolbar_box.setFixedHeight(78)
    try:
        toolbar_box.setStyleSheet(TOOL_GROUP_BOX_STYLE)
    except Exception:
        pass

    toolbar_layout = QVBoxLayout(toolbar_box)
    toolbar_layout.setContentsMargins(6, 0, 6, 2)
    toolbar_layout.setSpacing(0)

    try:
        orig_icon_size = toolbar.iconSize()
        if hasattr(orig_icon_size, 'width') and orig_icon_size.width() > 0:
            toolbar_icon_size = QSize(
                max(1, int(orig_icon_size.width() * MATPLOTLIB_TOOLBAR_ICON_SCALE)),
                max(1, int(orig_icon_size.height() * MATPLOTLIB_TOOLBAR_ICON_SCALE)),
            )
        else:
            toolbar_icon_size = QSize(24, 24)
    except Exception:
        toolbar_icon_size = QSize(24, 24)
    toolbar_button_size = MATPLOTLIB_TOOLBAR_BUTTON_SIZE

    try:
        toolbar.setIconSize(toolbar_icon_size)
    except Exception:
        pass
    try:
        toolbar.setFixedHeight(MATPLOTLIB_TOOLBAR_MAX_HEIGHT)
        toolbar.setFixedWidth(toolbar_width)
        toolbar.setContentsMargins(0, 0, 0, 0)
    except Exception:
        pass

    toolbar.coordinates = False
    toolbar.setStyleSheet("""
        QToolBar {
            background: transparent;
            background-color: transparent;
            border: none;
            spacing: 6px;
        }
    """)

    for action in list(toolbar.actions()):
        try:
            text = action.text().lower()
        except Exception:
            text = ""
        label = toolbar_action_text(action)
        if (
            action.isSeparator()
            or text in ["save", "save the figure", "save image only"]
            or "subplots" in text
            or (remove_customize and ("customize" in label or "edit axis" in label))
        ):
            try:
                toolbar.removeAction(action)
            except Exception:
                pass

    emojiize_matplotlib_toolbar(toolbar, toolbar_button_size, remove_customize=remove_customize)

    for action in toolbar.actions():
        widget = toolbar.widgetForAction(action)
        if isinstance(widget, QToolButton):
            try:
                widget.setFixedSize(toolbar_button_size)
                widget.setIconSize(toolbar_icon_size)
            except Exception:
                pass

    toolbar_extra_layout = QHBoxLayout()
    toolbar_extra_layout.setContentsMargins(0, 0, 0, 0)
    toolbar_extra_layout.setSpacing(8)
    toolbar_extra_layout.addWidget(toolbar, stretch=0, alignment=Qt.AlignVCenter)
    toolbar_extra_layout.addStretch(1)

    if option_widgets:
        for widget in option_widgets:
            toolbar_extra_layout.addWidget(widget, stretch=0, alignment=Qt.AlignVCenter)

    save_button = None
    if save_callback is not None:
        save_button = QToolButton(parent)
        save_button.setText("💾")
        save_button.setToolTip(save_tooltip)
        save_button.setFixedSize(toolbar_button_size)
        try:
            save_button.clicked.connect(save_callback)
        except Exception:
            pass
        save_button.setStyleSheet("""
            QToolButton {
                background: transparent;
                background-color: transparent;
                border: none;
                padding: 0px;
                margin: 0px;
                font-size: 28px;
                min-width: 32px;
                max-width: 32px;
                min-height: 32px;
                max-height: 32px;
            }
        """)
        toolbar_extra_layout.addWidget(save_button, stretch=0, alignment=Qt.AlignVCenter)

    toolbar_layout.addLayout(toolbar_extra_layout)

    return toolbar_box, toolbar_extra_layout, save_button
