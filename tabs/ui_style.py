from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QGroupBox, QVBoxLayout, QHBoxLayout, QToolButton, QStyle


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


def make_plot_legend(ax):
    legend = ax.legend(loc="best", frameon=True, fontsize=9)
    legend.set_draggable(True)
    return legend


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
