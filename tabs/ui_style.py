import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QValidator
from PySide6.QtWidgets import QDoubleSpinBox, QGroupBox, QVBoxLayout, QHBoxLayout, QToolButton, QStyle


GROUP_BOX_STYLE = """
    QGroupBox {
        background-color: #eeeeee;
        border: 1px solid #d8d8d8;
        border-radius: 10px;
        margin-top: 14px;
        padding: 4px;
        font-family: Arial;
        font-size: 12px;
    }

    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 8px;
        padding: 0px 4px;
        background-color: transparent;
        color: #222222;
        font-family: Arial;
        font-size: 12px;
    }

    QPushButton {
        background-color: #e2e2e2;
        border: 0px;
        border-radius: 5px;
        padding: 4px;
    }

    QPushButton:hover {
        background-color: #d8d8d8;
    }
"""


TOOL_GROUP_BOX_STYLE = GROUP_BOX_STYLE + """
    QToolBar {
        background-color: #eeeeee;
        border: 0px;
        spacing: 8px;
    }

    QToolButton {
        background-color: #eeeeee;
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
        background-color: #e2e2e2;
        color: #222222;
        border: 0px;
        border-radius: 5px;
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
        background-color: #ffffff;
        color: #222222;
        border: 1px solid #cfcfcf;
        border-radius: 6px;
        padding: 6px 10px;
    }
    QPushButton:hover {
        background-color: #f7f7f7;
        border-color: #b8b8b8;
    }
    QPushButton:pressed {
        background-color: #ededed;
    }
    QPushButton:disabled {
        background-color: #e4e4e4;
        color: #8d8d8d;
        border-color: #d6d6d6;
    }
"""


COMPACT_COMBO_STYLE = """
    QComboBox {
        background-color: #ffffff;
        color: #222222;
        border: 1px solid #cfcfcf;
        border-radius: 6px;
        padding: 2px 18px 2px 6px;
    }
    QComboBox:hover {
        border-color: #b8b8b8;
    }
    QComboBox:disabled {
        background-color: #eeeeee;
        color: #9a9a9a;
        border-color: #dddddd;
    }
    QComboBox::drop-down {
        border: 0px;
        width: 16px;
    }
    QComboBox::down-arrow {
        image: none;
        width: 0px;
        height: 0px;
    }
"""


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
    selected_lines = _plot_lines_for_legend_label(ax, selected_label)
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
            _set_legend_selection(target_ax, label)
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


def make_matplotlib_toolbar_block(parent, title, toolbar, option_widgets=None, save_callback=None, save_tooltip="Save", toolbar_width=340):
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
    toolbar_button_size = QSize(32, 32)
    save_button_size = QSize(MATPLOTLIB_TOOLBAR_MAX_HEIGHT + 8, MATPLOTLIB_TOOLBAR_MAX_HEIGHT + 8)
    save_icon_size = QSize(30, 30)

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
            background: #eeeeee;
            background-color: #eeeeee;
            border: none;
            spacing: 6px;
        }
        QToolButton {
            background: transparent;
            background-color: transparent;
            border: none;
            padding: 0px;
            margin: 0px;
            min-width: 32px;
            max-width: 32px;
            min-height: 32px;
            max-height: 32px;
        }
    """)

    for action in list(toolbar.actions()):
        try:
            text = action.text().lower()
        except Exception:
            text = ""
        if action.isSeparator() or text in ["save", "save the figure", "save image only"]:
            try:
                toolbar.removeAction(action)
            except Exception:
                pass

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
        save_button.setIcon(parent.style().standardIcon(QStyle.SP_DialogSaveButton))
        save_button.setToolTip(save_tooltip)
        save_button.setFixedSize(save_button_size)
        save_button.setIconSize(save_icon_size)
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
            }
        """)
        toolbar_extra_layout.addWidget(save_button, stretch=0, alignment=Qt.AlignVCenter)

    toolbar_layout.addLayout(toolbar_extra_layout)

    return toolbar_box, toolbar_extra_layout, save_button
