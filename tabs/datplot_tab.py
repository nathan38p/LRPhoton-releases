import json
from pathlib import Path

import numpy as np

from PySide6.QtCore import Qt, QEvent, QRectF, Signal, QMimeData, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QDialog,
    QDialogButtonBox,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QListWidget,
    QGroupBox,
    QCheckBox,
    QGridLayout,
    QLineEdit,
    QScrollArea,
    QComboBox,
    QColorDialog,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QMessageBox,
    QInputDialog,
    QAbstractItemView,
    QFrame,
    QSpinBox,
    QSlider,
    QSizePolicy,
    QStyle,
    QToolButton,
    QMenu,
    QStyledItemDelegate,
    QStyleOptionViewItem,
)
from PySide6.QtGui import QColor, QDrag, QKeySequence, QPainter, QPen, QShortcut

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from .file_ratings import install_file_rating_menu, is_file_rated_up, set_item_file_path
from .ui_style import (
    BLOCK_SPACING,
    FILE_BROWSER_WIDTH,
    FlexibleDoubleSpinBox as QDoubleSpinBox,
    FRAME_BUTTON_WIDTH,
    FRAME_COUNTER_WIDTH,
    FRAME_NAV_SPACING,
    FRAME_SPIN_WIDTH,
    GROUP_BOX_MARGINS,
    GROUP_BOX_STYLE,
    PAGE_MARGINS,
    apply_plot_display_style,
    clear_plot_canvas,
    finalize_plot_canvas,
    install_selectable_legend,
    make_matplotlib_toolbar_block,
    make_plot_legend,
)


PLOT_Y_AXES = ("left", "left2", "right", "right2")
PLOT_TRANSFORMED_MODES = {
    "Kratky (q²I(q))": {"power": 2, "x_power": 1, "x_label": "q", "y_label": "q²I(q)"},
    "qI(q)": {"power": 1, "x_power": 1, "x_label": "q", "y_label": "qI(q)"},
    "q⁴I(q)": {"power": 4, "x_power": 1, "x_label": "q", "y_label": "q⁴I(q)"},
    "q⁴I(q⁴)": {"power": 4, "x_power": 4, "x_label": "q⁴", "y_label": "q⁴I(q⁴)"},
}
PLOT_LOG_X_MODES = {"log linear", "log log", "Kratky (q²I(q))", "q⁴I(q)", "q⁴I(q⁴)"}
PLOT_LOG_Y_MODES = {"linear log", "log log", "Kratky (q²I(q))", "q⁴I(q)", "q⁴I(q⁴)"}
PLOT_LOG_LOG_MODES = PLOT_LOG_X_MODES & PLOT_LOG_Y_MODES


# ============================================================
# ========================== TOOLS ============================
# ============================================================

def read_dat_curve(file_path):
    curves = read_dat_curves(file_path)
    first = curves[0]
    return first["x"], first["y"]


def read_dat_curves(file_path):
    file_path = Path(file_path)
    raw_text = file_path.read_text(encoding="utf-8", errors="ignore")

    curve_defs = []
    file_x_label = ""
    is_manual_dataset = "# Created manually in Plot 1D" in raw_text
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if line.startswith("# x_label "):
            file_x_label = line[len("# x_label "):].strip()
            continue
        if not line.startswith("# curve "):
            continue
        try:
            payload = json.loads(line[len("# curve "):])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            curve_defs.append(payload)

    text = raw_text.replace(",", ".")

    if curve_defs:
        rows = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            for separator in [";", "\t", ","]:
                line = line.replace(separator, " ")
            values = []
            for part in line.split():
                try:
                    values.append(float(part))
                except ValueError:
                    values.append(np.nan)
            if values:
                rows.append(values)

        curves = []
        for index, info in enumerate(curve_defs):
            x_col = 2 * index
            y_col = x_col + 1
            points = []
            for row in rows:
                if len(row) <= y_col:
                    continue
                x, y = row[x_col], row[y_col]
                if np.isfinite(x) and np.isfinite(y):
                    points.append((x, y))
            if not points:
                continue
            array = np.asarray(points, dtype=float)
            order = np.argsort(array[:, 0])
            array = array[order]
            curves.append(
                {
                    "x": array[:, 0],
                    "y": array[:, 1],
                    "legend": str(info.get("label") or f"{file_path.stem} {index + 1}"),
                    "axis": normalize_plot_axis(info.get("axis", "left")),
                    "x_label": str(info.get("x_label") or file_x_label or ""),
                    "y_label": str(info.get("y_label") or info.get("label") or ""),
                    "manual_dataset": is_manual_dataset,
                }
            )

        if curves:
            return curves

    data = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        if not any(char.isdigit() for char in line):
            continue

        for separator in [";", "\t", ","]:
            line = line.replace(separator, " ")

        values = []
        for part in line.split():
            try:
                values.append(float(part))
            except ValueError:
                pass

        if len(values) >= 2:
            data.append([values[0], values[1]])

    if not data:
        raise ValueError("No valid numerical data found in this file.")

    array = np.asarray(data, dtype=float)
    valid = np.isfinite(array[:, 0]) & np.isfinite(array[:, 1])
    array = array[valid]

    if array.size == 0:
        raise ValueError("No finite numerical data found in this file.")

    order = np.argsort(array[:, 0])
    array = array[order]
    return [
        {
            "x": array[:, 0],
            "y": array[:, 1],
            "legend": file_path.stem,
            "axis": "left",
            "x_label": "",
            "y_label": "",
            "manual_dataset": False,
        }
    ]


def normalize_plot_axis(value):
    axis = str(value or "left").strip().lower()
    return axis if axis in PLOT_Y_AXES else "left"


def default_color(index):
    palette = [
        "#e91e63", "#9c27b0", "#f44336", "#4caf50", "#2196f3",
        "#000000", "#ff9800", "#009688", "#795548", "#607d8b",
    ]
    return palette[index % len(palette)]


def _legend_store_path():
    return Path.home() / ".lrphoton" / "datplot_legends.json"


# ============================================================
# ======================== CUSTOM TABLE =======================
# ============================================================

class CurveTableWidget(QTableWidget):
    """Custom table widget with better drag-drop handling for curves."""
    CURVE_KEY_MIME = "application/x-lrphoton-curve-key"
    
    def __init__(self, rows, cols, parent=None):
        super().__init__(rows, cols, parent)
        self._drag_row = None
        self._drag_key = None
        # Enable drag-drop
        self.setDragDropMode(QAbstractItemView.InternalMove)
        self.setDragDropOverwriteMode(False)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
    
    def mousePressEvent(self, event):
        """Record which row is being dragged."""
        index = self.indexAt(event.pos())
        if index.isValid():
            self._drag_row = index.row()
            self._drag_key = self.curve_key_for_row(index.row())
        super().mousePressEvent(event)

    def curve_key_for_row(self, row):
        key_item = self.item(row, 1)
        return key_item.text() if key_item is not None else None

    def mimeData(self, items):
        mime_data = QMimeData()
        rows = sorted({item.row() for item in items if item is not None})
        if rows:
            curve_key = self.curve_key_for_row(rows[0])
            if curve_key:
                mime_data.setData(self.CURVE_KEY_MIME, curve_key.encode("utf-8"))
        return mime_data

    def startDrag(self, supported_actions):
        if self._drag_key is None:
            selected_rows = self.selectionModel().selectedRows()
            if selected_rows:
                self._drag_row = selected_rows[0].row()
                self._drag_key = self.curve_key_for_row(self._drag_row)
        if not self._drag_key:
            return

        mime_data = QMimeData()
        mime_data.setData(self.CURVE_KEY_MIME, self._drag_key.encode("utf-8"))
        drag = QDrag(self)
        drag.setMimeData(mime_data)
        drag.exec(Qt.DropAction.MoveAction)

    def dragEnterEvent(self, event):
        if event.source() is self and event.mimeData().hasFormat(self.CURVE_KEY_MIME):
            event.setDropAction(Qt.DropAction.MoveAction)
            event.accept()
        else:
            super().dragEnterEvent(event)
    
    def dragMoveEvent(self, event):
        """Allow drop on rows."""
        if event.source() is self and event.mimeData().hasFormat(self.CURVE_KEY_MIME):
            event.setDropAction(Qt.DropAction.MoveAction)
            event.accept()
        else:
            super().dragMoveEvent(event)

    def drop_destination_row(self, event):
        row = self.indexAt(event.position().toPoint()).row()
        indicator = self.dropIndicatorPosition()
        if row < 0 or indicator == QAbstractItemView.DropIndicatorPosition.OnViewport:
            return self.rowCount()
        if indicator == QAbstractItemView.DropIndicatorPosition.BelowItem:
            return row + 1
        return row
    
    def dropEvent(self, event):
        """Move the underlying curve row instead of relying on QTableWidget internals."""
        if event.source() is self and event.mimeData().hasFormat(self.CURVE_KEY_MIME):
            parent_tab = self.parent()
            while parent_tab and not hasattr(parent_tab, 'refresh_curve_table'):
                parent_tab = parent_tab.parent()

            source_key = bytes(event.mimeData().data(self.CURVE_KEY_MIME)).decode("utf-8")
            drop_row = self.drop_destination_row(event)

            if parent_tab and hasattr(parent_tab, "move_curve_key") and source_key:
                parent_tab.move_curve_key(source_key, drop_row)
                event.setDropAction(Qt.DropAction.MoveAction)
                event.accept()
                self._drag_row = None
                self._drag_key = None
                return

            self._drag_row = None
            self._drag_key = None
            event.ignore()
        else:
            super().dropEvent(event)


class ColorCellDelegate(QStyledItemDelegate):
    """Keep color swatches visible even when their row is selected."""

    def paint(self, painter, option, index):
        item_option = QStyleOptionViewItem(option)
        self.initStyleOption(item_option, index)

        widget = option.widget
        style = widget.style() if widget is not None else None
        if style is not None:
            style.drawControl(QStyle.CE_ItemViewItem, item_option, painter, widget)

        color = index.data(Qt.BackgroundRole)
        if not isinstance(color, QColor):
            color = QColor(index.data(Qt.ToolTipRole) or "")
        if not color.isValid():
            return

        painter.save()
        painter.fillRect(option.rect.adjusted(1, 1, -1, -1), color)
        painter.restore()


class EyeToggleButton(QToolButton):
    """Small visibility toggle drawn as an eye icon."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setFixedSize(22, 20)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("QToolButton { border: none; background: transparent; padding: 0px; }")

    def paintEvent(self, event):
        color = QColor("#1f5f9c") if self.isChecked() else QColor("#9ca3af")

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(QPen(color, 1.5))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(QRectF(3.5, 6.0, 15.0, 8.0))

        painter.setBrush(color)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QRectF(9.0, 8.0, 4.0, 4.0))

        if not self.isChecked():
            painter.setPen(QPen(color, 1.8))
            painter.drawLine(5, 16, 18, 4)


# ============================================================
# =========================== CANVAS ==========================
# ============================================================

class PlotCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(dpi=150)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(620, 420)
        self.fig.subplots_adjust(left=0.12, right=0.98, top=0.92, bottom=0.18)

        try:
            self.grabGesture(Qt.PinchGesture)
        except Exception:
            pass

    def event(self, event):
        try:
            if event.type() == QEvent.NativeGesture:
                gesture_type = event.gestureType()
                value = event.value()
                if gesture_type == Qt.ZoomNativeGesture and value != 0:
                    scale = 1.0 / (1.0 + value) if value > -0.95 else 1.25
                    self.zoom_at_qpoint(self._event_center_point(event), scale)
                    event.accept()
                    return True

                if gesture_type == Qt.SmartZoomNativeGesture:
                    self.ax.relim()
                    self.ax.autoscale_view()
                    self.draw_idle()
                    event.accept()
                    return True

            if event.type() == QEvent.Gesture:
                pinch = event.gesture(Qt.PinchGesture)
                if pinch is not None:
                    factor = pinch.scaleFactor()
                    if factor and factor > 0:
                        self.zoom_at_qpoint(self._event_center_point(event), 1.0 / factor)
                        event.accept()
                        return True
        except Exception:
            pass

        return super().event(event)

    def wheelEvent(self, event):
        delta = event.pixelDelta()
        if delta.isNull():
            delta = event.angleDelta()
            dx = delta.x() / 120.0
            dy = delta.y() / 120.0
        else:
            dx = delta.x() / 80.0
            dy = delta.y() / 80.0

        if event.modifiers() & (Qt.ControlModifier | Qt.MetaModifier):
            if dy != 0:
                scale = 0.88 if dy > 0 else 1.14
                self.zoom_at_qpoint(event.position(), scale)
        else:
            self.pan_by_trackpad(dx, dy)

        event.accept()

    def _event_center_point(self, event):
        try:
            position = event.position()
            if position is not None:
                return position
        except Exception:
            pass

        return self.rect().center()

    def _qpoint_to_data_pos(self, qpoint):
        try:
            x_widget = float(qpoint.x())
            y_widget = float(qpoint.y())
        except Exception:
            x_widget = self.width() / 2
            y_widget = self.height() / 2

        bbox = self.ax.get_window_extent()
        x_fig = bbox.x0 + (x_widget / max(self.width(), 1)) * bbox.width
        y_fig = bbox.y1 - (y_widget / max(self.height(), 1)) * bbox.height
        xdata, ydata = self.ax.transData.inverted().transform((x_fig, y_fig))

        if not np.isfinite(xdata) or not np.isfinite(ydata):
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            xdata = (xlim[0] + xlim[1]) / 2
            ydata = (ylim[0] + ylim[1]) / 2

        return xdata, ydata

    def _scaled_limits(self, limits, center, scale, is_log):
        left, right = limits
        if is_log and left > 0 and right > 0 and center > 0:
            left_l, right_l, center_l = np.log10([left, right, center])
            return (
                10 ** (center_l + (left_l - center_l) * scale),
                10 ** (center_l + (right_l - center_l) * scale),
            )

        return (
            center + (left - center) * scale,
            center + (right - center) * scale,
        )

    def zoom_at_qpoint(self, qpoint, scale):
        if scale <= 0:
            return

        xdata, ydata = self._qpoint_to_data_pos(qpoint)
        self.ax.set_xlim(
            self._scaled_limits(
                self.ax.get_xlim(),
                xdata,
                scale,
                self.ax.get_xscale() == "log",
            )
        )
        self.ax.set_ylim(
            self._scaled_limits(
                self.ax.get_ylim(),
                ydata,
                scale,
                self.ax.get_yscale() == "log",
            )
        )
        self.draw_idle()

    def _panned_limits(self, limits, delta, is_log):
        left, right = limits
        if is_log and left > 0 and right > 0:
            left_l, right_l = np.log10([left, right])
            span = right_l - left_l
            shift = -delta * span * 0.08
            return 10 ** (left_l + shift), 10 ** (right_l + shift)

        span = right - left
        shift = -delta * span * 0.08
        return left + shift, right + shift

    def pan_by_trackpad(self, dx, dy):
        self.ax.set_xlim(
            self._panned_limits(
                self.ax.get_xlim(),
                dx,
                self.ax.get_xscale() == "log",
            )
        )
        self.ax.set_ylim(
            self._panned_limits(
                self.ax.get_ylim(),
                -dy,
                self.ax.get_yscale() == "log",
            )
        )
        self.draw_idle()


# ============================================================
# ========================= DAT PLOT TAB ======================
# ============================================================

class DatPlotTab(QWidget):
    """Plot tab: display and compare .dat curves."""

    folder_changed = Signal(Path)

    def __init__(self):
        super().__init__()

        self.current_folder = Path("/Users/nathanpiaget/Documents/Thèse LRP/Expériences/XENOCS")
        self.curves = {}
        self.guide_bars = []
        self.peak_labels = []
        self.peak_label_artists = []
        self._dragging_peak_label = None
        self.last_peak_label_name = ""
        self.saved_legends = self.load_saved_legends()
        self._syncing_folder = False
        self._refreshing_curve_table = False
        self._refreshing_guide_table = False
        self.q_axis_unit = "nm"
        self.extra_axes = {}

        self.build_ui()
        self.refresh_files()

    def load_saved_legends(self):
        path = _legend_store_path()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

        legends = data.get("legends", {}) if isinstance(data, dict) else {}
        return {
            str(file_path): str(legend)
            for file_path, legend in legends.items()
            if isinstance(file_path, str) and isinstance(legend, str)
        }

    def save_saved_legends(self):
        path = _legend_store_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        data = {"version": 1, "legends": self.saved_legends}
        tmp_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(path)

    def saved_legend_for_file(self, file_path):
        try:
            key = str(Path(file_path).expanduser().resolve())
        except (OSError, RuntimeError):
            key = str(file_path)
        return self.saved_legends.get(key)

    def remember_legend_for_file(self, file_path, legend):
        try:
            key = str(Path(file_path).expanduser().resolve())
        except (OSError, RuntimeError):
            key = str(file_path)
        self.saved_legends[key] = legend
        self.save_saved_legends()

    def create_matplotlib_toolbar_block(
        self,
        title,
        toolbar,
        option_widgets=None,
        save_callback=None,
        save_tooltip="Save",
        toolbar_width=340,
    ):
        return make_matplotlib_toolbar_block(self, title, toolbar, option_widgets=option_widgets, save_callback=save_callback, save_tooltip=save_tooltip, toolbar_width=toolbar_width)

    def build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(*PAGE_MARGINS)
        main_layout.setSpacing(BLOCK_SPACING)

        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(BLOCK_SPACING)
        main_layout.addLayout(content_layout, stretch=1)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setFixedWidth(FILE_BROWSER_WIDTH)
        left_scroll.setFrameShape(QFrame.NoFrame)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(BLOCK_SPACING)
        left_scroll.setWidget(left_panel)
        content_layout.addWidget(left_scroll, stretch=0)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)
        content_layout.addWidget(right_panel, stretch=1)

        curve_scroll = QScrollArea()
        curve_scroll.setWidgetResizable(True)
        curve_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        curve_scroll.setFixedWidth(FILE_BROWSER_WIDTH)
        curve_scroll.setFrameShape(QFrame.NoFrame)

        curve_panel = QWidget()
        curve_panel_layout = QVBoxLayout(curve_panel)
        curve_panel_layout.setContentsMargins(0, 0, 0, 0)
        curve_panel_layout.setSpacing(BLOCK_SPACING)
        curve_scroll.setWidget(curve_panel)
        content_layout.addWidget(curve_scroll, stretch=0)

        file_box = QGroupBox("File browser")
        self.style_top_group_box(file_box)
        file_layout = QVBoxLayout(file_box)
        file_layout.setContentsMargins(*GROUP_BOX_MARGINS)
        file_layout.setSpacing(6)
        file_box.setMinimumHeight(220)
        left_layout.addWidget(file_box, stretch=1)

        self.folder_path = QLineEdit(str(self.current_folder))
        self.folder_path.returnPressed.connect(self.refresh_files)
        file_layout.addWidget(self.folder_path)

        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.choose_folder)
        file_layout.addWidget(self.browse_button)

        filters_layout = QGridLayout()
        self.extensions_filter = QLineEdit("*.dat")
        self.name_filter = QLineEdit("**")
        self.extensions_filter.textChanged.connect(self.refresh_files)
        self.name_filter.textChanged.connect(self.refresh_files)
        filters_layout.addWidget(QLabel("Name:"), 0, 0)
        filters_layout.addWidget(self.name_filter, 0, 1)
        filters_layout.addWidget(QLabel("Extensions:"), 1, 0)
        filters_layout.addWidget(self.extensions_filter, 1, 1)
        file_layout.addLayout(filters_layout)

        self.show_subfolders = QCheckBox("Show subfolders")
        self.show_subfolders.setChecked(False)
        self.show_subfolders.stateChanged.connect(self.refresh_files)
        self.only_thumbs_up_checkbox = QCheckBox("Only 👍")
        self.only_thumbs_up_checkbox.setChecked(False)
        self.only_thumbs_up_checkbox.stateChanged.connect(self.refresh_files)
        file_options_layout = QHBoxLayout()
        file_options_layout.setContentsMargins(0, 0, 0, 0)
        file_options_layout.setSpacing(10)
        file_options_layout.addWidget(self.show_subfolders)
        file_options_layout.addWidget(self.only_thumbs_up_checkbox)
        file_options_layout.addStretch(1)
        file_layout.addLayout(file_options_layout)

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_files)
        file_layout.addWidget(self.refresh_button)

        self.file_list = QListWidget()
        install_file_rating_menu(self.file_list)
        self.file_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.file_list.itemSelectionChanged.connect(self.selection_changed)
        file_layout.addWidget(self.file_list, stretch=1)

        self.create_dat_button = QPushButton("Create .dat")
        self.create_dat_button.setToolTip("Create a multi-curve .dat from the curves currently loaded in Plot 1D")
        self.create_dat_button.clicked.connect(self.open_create_dat_dialog)
        file_layout.addWidget(self.create_dat_button)

        # Plot settings widgets (previously in settings_box, now just created here)
        self.plot_mode = QComboBox()
        self.plot_mode.addItems([
            "linear linear",
            "linear log",
            "log linear",
            "log log",
            "Kratky (q²I(q))",
            "qI(q)",
            "q⁴I(q)",
            "q⁴I(q⁴)",
        ])
        self.plot_mode.setCurrentText("log log")
        self.plot_mode.currentTextChanged.connect(self.update_plot)

        self.auto_limits = QCheckBox("Auto limits")
        self.auto_limits.setChecked(True)
        self.auto_limits.stateChanged.connect(self.update_limit_state)

        self.x_min = self.double_spin(0.0)
        self.x_max = self.double_spin(1.0)
        self.y_min = self.double_spin(0.0)
        self.y_max = self.double_spin(1.0)

        self.x_label = QLineEdit("q / nm⁻¹")
        self.y_label = QLineEdit("Intensity / a.u.")
        self.title_edit = QLineEdit("")

        self.x_label.textChanged.connect(self.update_plot)
        self.y_label.textChanged.connect(self.update_plot)
        self.title_edit.textChanged.connect(self.update_plot)

        for spin in [self.x_min, self.x_max, self.y_min, self.y_max]:
            spin.valueChanged.connect(self.update_plot)

        curve_box = QGroupBox("Curves")
        self.style_top_group_box(curve_box)
        curve_layout = QVBoxLayout(curve_box)
        curve_layout.setContentsMargins(*GROUP_BOX_MARGINS)
        curve_layout.setSpacing(6)
        curve_box.setMinimumHeight(170)
        curve_panel_layout.addWidget(curve_box, stretch=1)

        self.curve_gradient_colors = ["#1f77b4", "#2ca02c", "#d62728"]
        gradient_layout = QHBoxLayout()
        gradient_layout.setContentsMargins(0, 0, 0, 0)
        gradient_layout.setSpacing(4)
        gradient_layout.addWidget(QLabel("Gradient:"))
        self.curve_gradient_layout = gradient_layout
        self.curve_gradient_buttons = []
        self.add_curve_gradient_color_button = QToolButton()
        self.add_curve_gradient_color_button.setText("+")
        self.add_curve_gradient_color_button.setFixedSize(24, 20)
        self.add_curve_gradient_color_button.setToolTip("Add a color point to the gradient")
        self.add_curve_gradient_color_button.clicked.connect(self.add_curve_gradient_color_point)
        self.apply_curve_gradient_button = QPushButton("Apply")
        self.apply_curve_gradient_button.setToolTip("Apply this color gradient to the curves in the current list order")
        self.apply_curve_gradient_button.clicked.connect(self.apply_curve_color_gradient)
        self.apply_curve_gradient_button.setEnabled(False)
        self.curve_gradient_stretch = None
        curve_layout.addLayout(gradient_layout)
        self.rebuild_curve_gradient_buttons()

        self.curve_table = CurveTableWidget(0, 6)
        self.curve_table.setMinimumHeight(140)
        self.curve_table.setHorizontalHeaderLabels(["", "File", "Legend", "Axis", "Color", ""])
        self.curve_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.curve_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.curve_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.curve_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Fixed)
        self.curve_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Fixed)
        self.curve_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Fixed)
        self.curve_table.setColumnWidth(0, 28)
        self.curve_table.setColumnWidth(3, 58)
        self.curve_table.setColumnWidth(4, 44)
        self.curve_table.setColumnWidth(5, 30)
        self.curve_table.verticalHeader().setVisible(False)
        self.curve_table.verticalHeader().setDefaultSectionSize(28)
        self.curve_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.curve_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.curve_table.setDropIndicatorShown(True)
        self.curve_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.curve_table.setItemDelegateForColumn(4, ColorCellDelegate(self.curve_table))
        self.curve_table.cellChanged.connect(self.curve_table_changed)
        self.curve_table.cellDoubleClicked.connect(self.curve_table_double_clicked)
        self.curve_table.customContextMenuRequested.connect(self.open_curve_table_menu)
        curve_layout.addWidget(self.curve_table, stretch=1)

        mask_buttons_layout = QHBoxLayout()
        mask_buttons_layout.setContentsMargins(0, 0, 0, 0)
        mask_buttons_layout.setSpacing(4)
        self.mask_range_button = QPushButton("Mask range...")
        self.mask_range_button.setToolTip("Mask a q or psi range in selected curves, or all curves if none is selected.")
        self.mask_range_button.clicked.connect(self.open_mask_range_dialog)
        self.mask_range_button.setEnabled(False)
        self.reset_masks_button = QPushButton("Reset masks")
        self.reset_masks_button.setToolTip("Restore original data for selected curves, or all curves if none is selected.")
        self.reset_masks_button.clicked.connect(self.reset_curve_masks)
        self.reset_masks_button.setEnabled(False)
        mask_buttons_layout.addWidget(self.mask_range_button)
        mask_buttons_layout.addWidget(self.reset_masks_button)
        curve_layout.addLayout(mask_buttons_layout)

        guide_box = QGroupBox("Dashed bars")
        self.style_top_group_box(guide_box)
        guide_layout = QVBoxLayout(guide_box)
        guide_layout.setContentsMargins(*GROUP_BOX_MARGINS)
        guide_layout.setSpacing(6)
        curve_panel_layout.addWidget(guide_box, stretch=0)

        self.guide_table = QTableWidget(0, 4)
        self.guide_table.setMinimumHeight(96)
        self.guide_table.setMaximumHeight(150)
        self.guide_table.setHorizontalHeaderLabels(["Axis", "Value", "Color", ""])
        self.guide_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.guide_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.guide_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Fixed)
        self.guide_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Fixed)
        self.guide_table.setColumnWidth(0, 42)
        self.guide_table.setColumnWidth(2, 44)
        self.guide_table.setColumnWidth(3, 30)
        self.guide_table.verticalHeader().setVisible(False)
        self.guide_table.verticalHeader().setDefaultSectionSize(26)
        self.guide_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.guide_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.guide_table.setItemDelegateForColumn(2, ColorCellDelegate(self.guide_table))
        self.guide_table.cellChanged.connect(self.guide_table_changed)
        self.guide_table.cellDoubleClicked.connect(self.guide_table_double_clicked)
        guide_layout.addWidget(self.guide_table)

        guide_buttons_layout = QHBoxLayout()
        guide_buttons_layout.setContentsMargins(0, 0, 0, 0)
        guide_buttons_layout.setSpacing(4)
        self.add_x_bar_button = QPushButton("+ X")
        self.add_x_bar_button.setToolTip("Add a vertical dashed bar")
        self.add_x_bar_button.clicked.connect(lambda: self.add_guide_bar("x"))
        self.add_y_bar_button = QPushButton("+ Y")
        self.add_y_bar_button.setToolTip("Add a horizontal dashed bar")
        self.add_y_bar_button.clicked.connect(lambda: self.add_guide_bar("y"))
        guide_buttons_layout.addWidget(self.add_x_bar_button)
        guide_buttons_layout.addWidget(self.add_y_bar_button)
        guide_layout.addLayout(guide_buttons_layout)

        axis_label_layout = QHBoxLayout()
        axis_label_layout.setContentsMargins(0, 0, 0, 0)
        axis_label_layout.setSpacing(4)
        self.add_axis_label_button = QPushButton("+ Peak label")
        self.add_axis_label_button.setToolTip("Click a peak, enter a label name. Reuse the same name to add another arrow to the same label.")
        self.add_axis_label_button.setCheckable(True)
        self.add_axis_label_button.clicked.connect(self.toggle_peak_label_mode)
        self.clear_axis_labels_button = QPushButton("Clear labels")
        self.clear_axis_labels_button.setToolTip("Remove all peak labels")
        self.clear_axis_labels_button.clicked.connect(self.clear_peak_labels)
        axis_label_layout.addWidget(self.add_axis_label_button)
        axis_label_layout.addWidget(self.clear_axis_labels_button)
        guide_layout.addLayout(axis_label_layout)

        self.clear_header_button = QPushButton("−", self.curve_table.horizontalHeader())
        self.clear_header_button.setFixedSize(22, 18)
        self.clear_header_button.setToolTip("Clear all curves")
        self.clear_header_button.clicked.connect(self.clear_curves)
        self.clear_header_button.setStyleSheet("""
            QPushButton {
                background: #ffecec;
                color: #b00020;
                border: 1px solid #ffb3b3;
                border-radius: 8px;
                font-weight: bold;
                font-size: 11px;
                padding: 0px;
            }
            QPushButton:hover {
                background: #ffd6d6;
            }
        """)
        self.curve_table.horizontalHeader().sectionResized.connect(self.update_clear_header_button_position)
        self.update_clear_header_button_position()


        self.canvas = PlotCanvas()
        self.canvas.setContentsMargins(0, 0, 0, 0)
        clear_plot_canvas(self.canvas)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.plot_mode.setFixedWidth(120)

        self.fit_button = QPushButton("Fit")
        self.fit_button.setToolTip("Fit I(q) = A q^-n on the current 1D plot")
        self.fit_button.clicked.connect(self.open_power_law_fit_dialog)

        self.show_legend = QCheckBox("Legend")
        self.show_legend.setChecked(True)
        self.show_legend.stateChanged.connect(self.update_plot)

        self.curve_display_stride_spin = QSpinBox()
        self.curve_display_stride_spin.setRange(1, 100000)
        self.curve_display_stride_spin.setValue(1)
        self.curve_display_stride_spin.setFixedWidth(64)
        self.curve_display_stride_spin.setToolTip("Display one curve every N visible curves")
        self.curve_display_stride_spin.valueChanged.connect(self.update_plot)

        self.keep_zoom_checkbox = QCheckBox("Keep zoom")
        self.keep_zoom_checkbox.setChecked(False)
        self.keep_zoom_checkbox.setToolTip("Keep current zoom and pan when the plot is redrawn")

        graph_box, self.toolbar_extra_layout, self.save_plot_button = self.create_matplotlib_toolbar_block(
            title="Plot",
            toolbar=self.toolbar,
            option_widgets=[
                self.fit_button,
                self.plot_mode,
                QLabel("Each:"),
                self.curve_display_stride_spin,
                self.show_legend,
                self.keep_zoom_checkbox,
            ],
            save_callback=self.save_plot_high_quality,
            save_tooltip="Save plot",
            toolbar_width=470,
        )
        right_layout.addWidget(graph_box, stretch=0)
        right_layout.addWidget(self.canvas, stretch=1)

        self.graph_coordinate_label = QLabel("q = - | I = -")
        self.graph_coordinate_label.setMinimumHeight(28)
        self.graph_coordinate_label.setAlignment(Qt.AlignCenter)
        self.graph_coordinate_label.setStyleSheet("""
            QLabel {
                background-color: #f4f4f4;
                border-radius: 8px;
                padding: 6px;
                font-family: Menlo, Monaco, monospace;
                font-size: 11px;
            }
        """)
        right_layout.addWidget(self.graph_coordinate_label, stretch=0)
        self.update_graph_toolbar_enabled()

        self.canvas.mpl_connect("motion_notify_event", self.update_graph_coordinates)
        self.canvas.mpl_connect("button_press_event", self.graph_button_press)
        self.canvas.mpl_connect("button_release_event", self.graph_button_release)
        self.canvas.mpl_connect("axes_leave_event", self.clear_graph_coordinates)

        frame_nav = QHBoxLayout()
        frame_nav.setContentsMargins(0, 0, 0, 0)
        frame_nav.setSpacing(FRAME_NAV_SPACING)

        self.frame_start_spin = QSpinBox()
        self.frame_start_spin.setRange(1, 1)
        self.frame_start_spin.setValue(1)
        self.frame_start_spin.setFixedWidth(FRAME_SPIN_WIDTH)

        self.frame_end_spin = QSpinBox()
        self.frame_end_spin.setRange(1, 1)
        self.frame_end_spin.setValue(1)
        self.frame_end_spin.setFixedWidth(FRAME_SPIN_WIDTH)

        self.prev_frame_button = QPushButton("<")
        self.next_frame_button = QPushButton(">")
        self.prev_frame_button.setFixedWidth(FRAME_BUTTON_WIDTH)
        self.next_frame_button.setFixedWidth(FRAME_BUTTON_WIDTH)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(1, 1)
        self.frame_slider.setValue(1)

        self.frame_counter_label = QLabel("1 / 1")
        self.frame_counter_label.setMinimumWidth(FRAME_COUNTER_WIDTH)
        self.frame_counter_label.setAlignment(Qt.AlignCenter)

        frame_nav.addWidget(QLabel("Start:"))
        frame_nav.addWidget(self.frame_start_spin)
        frame_nav.addWidget(self.prev_frame_button)
        frame_nav.addWidget(self.frame_slider, stretch=1)
        frame_nav.addWidget(self.next_frame_button)
        frame_nav.addWidget(QLabel("End:"))
        frame_nav.addWidget(self.frame_end_spin)
        frame_nav.addWidget(self.frame_counter_label)
        main_layout.addLayout(frame_nav, stretch=0)

        for widget in [
            self.frame_start_spin, self.frame_end_spin, self.prev_frame_button,
            self.next_frame_button, self.frame_slider,
        ]:
            widget.setEnabled(False)

        self.update_limit_state()

    def style_top_group_box(self, box):
        box.setStyleSheet(GROUP_BOX_STYLE)

    def double_spin(self, value):
        spin = QDoubleSpinBox()
        spin.setDecimals(6)
        spin.setRange(-1e12, 1e12)
        spin.setValue(value)
        spin.setFixedHeight(24)
        spin.setFixedWidth(90)
        return spin

    def save_plot_high_quality(self):
        if not self.curves:
            return

        default_name = "plot_1d.png"
        if self.curves:
            first_curve = next(iter(self.curves.values()))
            default_name = f"{first_curve['path'].stem}_plot_1d.png"

        start_path = str(self.current_folder / default_name)
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save plot",
            start_path,
            "PNG image (*.png);;TIFF image (*.tif *.tiff);;PDF vector (*.pdf);;SVG vector (*.svg)",
        )

        if not file_path:
            return

        suffix = Path(file_path).suffix.lower()
        if not suffix:
            if "TIFF" in selected_filter:
                file_path += ".tif"
                suffix = ".tif"
            elif "PDF" in selected_filter:
                file_path += ".pdf"
                suffix = ".pdf"
            elif "SVG" in selected_filter:
                file_path += ".svg"
                suffix = ".svg"
            else:
                file_path += ".png"
                suffix = ".png"

        save_kwargs = {
            "bbox_inches": "tight",
            "pad_inches": 0.04,
            "facecolor": "white",
        }

        if suffix in [".png", ".tif", ".tiff"]:
            save_kwargs["dpi"] = 600
        else:
            save_kwargs["dpi"] = 300

        try:
            self.canvas.draw()
            self.canvas.fig.savefig(file_path, **save_kwargs)
        except Exception as error:
            QMessageBox.warning(self, "Save plot error", f"Could not save plot:\n\n{error}")

    def open_create_dat_dialog(self):
        selected_files = self.selected_files()
        if len(selected_files) == 1:
            try:
                raw_text = Path(selected_files[0]).read_text(encoding="utf-8", errors="ignore")
                if "# LRPhoton multi-curve dat" in raw_text:
                    self.open_manual_dat_dialog(edit_path=selected_files[0])
                    return
            except Exception:
                pass

        if not self.curves:
            self.open_manual_dat_dialog()
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Create multi-curve .dat")
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        table = QTableWidget(len(self.curves), 6)
        table.setHorizontalHeaderLabels(["Use", "Legend", "Axis", "X title", "Y title", "Source"])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Fixed)
        table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        table.setColumnWidth(0, 48)
        table.setColumnWidth(2, 80)
        table.verticalHeader().setVisible(False)

        keys = list(self.curves.keys())
        axis_widgets = {}
        for row, key in enumerate(keys):
            curve = self.curves[key]
            use_item = QTableWidgetItem("")
            use_item.setFlags(use_item.flags() | Qt.ItemIsUserCheckable)
            use_item.setCheckState(Qt.Checked)
            table.setItem(row, 0, use_item)

            table.setItem(row, 1, QTableWidgetItem(curve["legend"]))

            axis_combo = QComboBox()
            axis_combo.addItems(PLOT_Y_AXES)
            axis_combo.setCurrentText(normalize_plot_axis(curve.get("axis", "left")))
            table.setCellWidget(row, 2, axis_combo)
            axis_widgets[row] = axis_combo

            table.setItem(row, 3, QTableWidgetItem(curve.get("x_label") or self.x_label.text() or self.q_axis_label()))
            table.setItem(row, 4, QTableWidgetItem(curve.get("y_label") or curve["legend"]))

            source_item = QTableWidgetItem(key)
            source_item.setFlags(source_item.flags() & ~Qt.ItemIsEditable)
            table.setItem(row, 5, source_item)

        layout.addWidget(table)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() != QDialog.Accepted:
            return

        selected = []
        for row, key in enumerate(keys):
            use_item = table.item(row, 0)
            if use_item is None or use_item.checkState() != Qt.Checked:
                continue
            legend_item = table.item(row, 1)
            legend = legend_item.text().strip() if legend_item and legend_item.text().strip() else self.curves[key]["legend"]
            x_title_item = table.item(row, 3)
            y_title_item = table.item(row, 4)
            selected.append(
                {
                    "key": key,
                    "legend": legend,
                    "axis": normalize_plot_axis(axis_widgets[row].currentText()),
                    "x_label": x_title_item.text().strip() if x_title_item else "",
                    "y_label": y_title_item.text().strip() if y_title_item else "",
                }
            )

        if not selected:
            QMessageBox.warning(self, "Create .dat", "Select at least one curve.")
            return

        default_name = "multi_curve.dat" if len(selected) > 1 else f"{selected[0]['legend']}.dat"
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save multi-curve .dat",
            str(self.current_folder / default_name),
            "DAT file (*.dat)",
        )
        if not out_path:
            return
        if Path(out_path).suffix.lower() != ".dat":
            out_path += ".dat"

        try:
            self.write_multi_curve_dat(Path(out_path), selected)
        except Exception as error:
            QMessageBox.warning(self, "Create .dat error", f"Could not create the file:\n\n{error}")
            return

        self.refresh_files()

    def open_manual_dat_dialog(self, edit_path=None):
        initial_curves = None
        if edit_path is not None:
            try:
                initial_curves = read_dat_curves(edit_path)
            except Exception as error:
                QMessageBox.warning(self, "Edit .dat error", f"Could not read this manual dataset:\n\n{error}")
                return

        dialog = QDialog(self)
        dialog.setWindowTitle("Edit .dat" if edit_path is not None else "Create .dat")
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        curve_count = QSpinBox()
        curve_count.setRange(1, 6)
        curve_count.setValue(len(initial_curves) if initial_curves else 1)
        row_count = QSpinBox()
        row_count.setRange(2, 1000)
        initial_rows = max((len(curve["x"]) for curve in initial_curves), default=20) if initial_curves else 20
        row_count.setValue(initial_rows)
        rebuild_button = QPushButton("Rebuild table")
        controls.addWidget(QLabel("Curves:"))
        controls.addWidget(curve_count)
        controls.addWidget(QLabel("Rows:"))
        controls.addWidget(row_count)
        controls.addWidget(rebuild_button)
        controls.addStretch(1)
        layout.addLayout(controls)

        meta_table = QTableWidget(curve_count.value(), 4)
        meta_table.setHorizontalHeaderLabels(["Legend", "Axis", "X title", "Y title"])
        meta_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        meta_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Fixed)
        meta_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        meta_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        meta_table.setColumnWidth(1, 90)
        meta_table.verticalHeader().setVisible(False)
        layout.addWidget(meta_table)

        data_table = QTableWidget(row_count.value(), curve_count.value() * 2)
        data_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(data_table, stretch=1)
        paste_shortcut = QShortcut(QKeySequence.Paste, data_table)
        paste_shortcut.activated.connect(
            lambda: self.paste_clipboard_into_manual_dat_table(
                data_table,
                meta_table,
                axis_widgets,
                row_count,
                curve_count,
                lambda: rebuild_tables(True),
            )
        )

        axis_widgets = {}

        def rebuild_tables(preserve=True):
            previous = []
            if preserve:
                for curve_index in range(meta_table.rowCount()):
                    points = []
                    for row in range(data_table.rowCount()):
                        x_item = data_table.item(row, curve_index * 2)
                        y_item = data_table.item(row, curve_index * 2 + 1)
                        points.append((
                            x_item.text() if x_item else "",
                            y_item.text() if y_item else "",
                        ))
                    previous.append(
                        {
                            "legend": meta_table.item(curve_index, 0).text() if meta_table.item(curve_index, 0) else f"Curve {curve_index + 1}",
                            "axis": axis_widgets.get(curve_index).currentText() if curve_index in axis_widgets else "left",
                            "x_label": meta_table.item(curve_index, 2).text() if meta_table.item(curve_index, 2) else "x",
                            "y_label": meta_table.item(curve_index, 3).text() if meta_table.item(curve_index, 3) else f"Curve {curve_index + 1}",
                            "points": points,
                        }
                    )

            n_curves = curve_count.value()
            n_rows = row_count.value()
            meta_table.setRowCount(n_curves)
            data_table.setRowCount(n_rows)
            data_table.setColumnCount(n_curves * 2)
            headers = []
            axis_widgets.clear()

            for curve_index in range(n_curves):
                previous_curve = previous[curve_index] if curve_index < len(previous) else None
                legend = previous_curve["legend"] if previous_curve else f"Curve {curve_index + 1}"
                x_label = previous_curve["x_label"] if previous_curve else "x"
                y_label = previous_curve["y_label"] if previous_curve else legend
                meta_table.setItem(curve_index, 0, QTableWidgetItem(legend))

                axis_combo = QComboBox()
                axis_combo.addItems(PLOT_Y_AXES)
                axis_combo.setCurrentText(normalize_plot_axis(previous_curve["axis"] if previous_curve else "left"))
                meta_table.setCellWidget(curve_index, 1, axis_combo)
                axis_widgets[curve_index] = axis_combo
                meta_table.setItem(curve_index, 2, QTableWidgetItem(x_label))
                meta_table.setItem(curve_index, 3, QTableWidgetItem(y_label))

                headers.extend([x_label or f"x{curve_index + 1}", y_label or f"y{curve_index + 1}"])

                if previous_curve:
                    for row, (x_text, y_text) in enumerate(previous_curve["points"][:n_rows]):
                        data_table.setItem(row, curve_index * 2, QTableWidgetItem(x_text))
                        data_table.setItem(row, curve_index * 2 + 1, QTableWidgetItem(y_text))

            data_table.setHorizontalHeaderLabels(headers)

        rebuild_button.clicked.connect(lambda: rebuild_tables(True))
        rebuild_tables(False)

        if initial_curves:
            for curve_index, curve in enumerate(initial_curves):
                meta_table.setItem(curve_index, 0, QTableWidgetItem(curve.get("legend", f"Curve {curve_index + 1}")))
                axis_widgets[curve_index].setCurrentText(normalize_plot_axis(curve.get("axis", "left")))
                meta_table.setItem(curve_index, 2, QTableWidgetItem(curve.get("x_label") or "x"))
                meta_table.setItem(curve_index, 3, QTableWidgetItem(curve.get("y_label") or curve.get("legend", f"Curve {curve_index + 1}")))
                for row, (x, y) in enumerate(zip(curve["x"], curve["y"])):
                    data_table.setItem(row, curve_index * 2, QTableWidgetItem(f"{x:.12g}"))
                    data_table.setItem(row, curve_index * 2 + 1, QTableWidgetItem(f"{y:.12g}"))
            data_table.setHorizontalHeaderLabels([
                (meta_table.item(index // 2, 2 if index % 2 == 0 else 3).text()
                 if meta_table.item(index // 2, 2 if index % 2 == 0 else 3)
                 else f"{'x' if index % 2 == 0 else 'y'}{index // 2 + 1}")
                for index in range(data_table.columnCount())
            ])

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() != QDialog.Accepted:
            return

        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save .dat",
            str(edit_path if edit_path is not None else self.current_folder / "manual_dataset.dat"),
            "DAT file (*.dat)",
        )
        if not out_path:
            return
        if Path(out_path).suffix.lower() != ".dat":
            out_path += ".dat"

        try:
            self.write_manual_dat(Path(out_path), meta_table, data_table, axis_widgets)
        except Exception as error:
            QMessageBox.warning(self, "Create .dat error", f"Could not create the file:\n\n{error}")
            return

        self.refresh_files()

    def paste_clipboard_into_manual_dat_table(
        self,
        data_table,
        meta_table,
        axis_widgets,
        row_count_spinbox,
        curve_count_spinbox,
        rebuild_callback=None,
    ):
        text = QApplication.clipboard().text()
        if not text.strip():
            return

        rows = [
            line.split("\t")
            for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
            if line.strip()
        ]
        if not rows:
            return

        header_values = [value.strip() for value in rows[0]]
        data_rows = rows[1:]
        if not data_rows:
            return

        current_row = data_table.currentRow()
        current_column = data_table.currentColumn()
        start_row = current_row if current_row >= 0 else 0
        start_column = current_column if current_column >= 0 else 0
        needed_rows = start_row + len(data_rows)
        needed_columns = start_column + max(len(row) for row in data_rows + [header_values])

        if needed_rows > data_table.rowCount():
            row_count_spinbox.setValue(min(row_count_spinbox.maximum(), needed_rows))

        if needed_columns > data_table.columnCount():
            needed_curves = int(np.ceil(needed_columns / 2.0))
            curve_count_spinbox.setValue(min(curve_count_spinbox.maximum(), needed_curves))

        if rebuild_callback is not None:
            rebuild_callback()
        else:
            data_table.setRowCount(row_count_spinbox.value())
            data_table.setColumnCount(curve_count_spinbox.value() * 2)

        for column_offset, header in enumerate(header_values):
            target_column = start_column + column_offset
            if target_column >= data_table.columnCount():
                break
            data_table.setHorizontalHeaderItem(target_column, QTableWidgetItem(header))
            curve_index = target_column // 2
            meta_column = 2 if target_column % 2 == 0 else 3
            if curve_index < meta_table.rowCount():
                meta_table.setItem(curve_index, meta_column, QTableWidgetItem(header))
                if target_column % 2 == 1:
                    legend = header or f"Curve {curve_index + 1}"
                    meta_table.setItem(curve_index, 0, QTableWidgetItem(legend))

        for row_offset, values in enumerate(data_rows):
            target_row = start_row + row_offset
            if target_row >= data_table.rowCount():
                break
            for column_offset, value in enumerate(values):
                target_column = start_column + column_offset
                if target_column >= data_table.columnCount():
                    break
                data_table.setItem(target_row, target_column, QTableWidgetItem(value.strip()))

    def write_manual_dat(self, out_path, meta_table, data_table, axis_widgets):
        curves = []
        for curve_index in range(meta_table.rowCount()):
            points = []
            x_col = curve_index * 2
            y_col = x_col + 1
            for row in range(data_table.rowCount()):
                x_item = data_table.item(row, x_col)
                y_item = data_table.item(row, y_col)
                if x_item is None or y_item is None:
                    continue
                try:
                    x = float(x_item.text().replace(",", "."))
                    y = float(y_item.text().replace(",", "."))
                except ValueError:
                    continue
                if np.isfinite(x) and np.isfinite(y):
                    points.append((x, y))

            if not points:
                continue

            legend_item = meta_table.item(curve_index, 0)
            legend = legend_item.text().strip() if legend_item and legend_item.text().strip() else f"Curve {curve_index + 1}"
            x_label_item = meta_table.item(curve_index, 2)
            y_label_item = meta_table.item(curve_index, 3)
            x_label = x_label_item.text().strip() if x_label_item and x_label_item.text().strip() else "x"
            y_label = y_label_item.text().strip() if y_label_item and y_label_item.text().strip() else legend
            curves.append(
                {
                    "legend": legend,
                    "axis": normalize_plot_axis(axis_widgets[curve_index].currentText()),
                    "x_label": x_label,
                    "y_label": y_label,
                    "points": points,
                }
            )

        if not curves:
            raise ValueError("No valid x/y points were entered.")

        max_len = max(len(curve["points"]) for curve in curves)
        with open(out_path, "w", encoding="utf-8") as file:
            file.write("# LRPhoton multi-curve dat\n")
            file.write("# Created manually in Plot 1D\n")
            file.write("# Each curve uses two columns: x_i y_i\n")
            for curve in curves:
                file.write("# curve " + json.dumps(
                    {
                        "label": curve["legend"],
                        "axis": curve["axis"],
                        "x_label": curve["x_label"],
                        "y_label": curve["y_label"],
                    },
                    ensure_ascii=False,
                ) + "\n")
            headers = []
            for curve in curves:
                headers.extend([curve["x_label"], curve["y_label"]])
            file.write("# " + " ".join(headers) + "\n")

            for row in range(max_len):
                values = []
                for curve in curves:
                    if row < len(curve["points"]):
                        x, y = curve["points"][row]
                        values.extend([f"{x:.12g}", f"{y:.12g}"])
                    else:
                        values.extend(["nan", "nan"])
                file.write(" ".join(values) + "\n")

    def write_multi_curve_dat(self, out_path, selected):
        curves = [self.curves[item["key"]] for item in selected]
        lengths = [len(curve["x"]) for curve in curves]
        max_len = max(lengths)

        with open(out_path, "w", encoding="utf-8") as file:
            file.write("# LRPhoton multi-curve dat\n")
            file.write("# Each curve uses two columns: x_i y_i\n")
            for item in selected:
                file.write("# curve " + json.dumps(
                    {
                        "label": item["legend"],
                        "axis": item["axis"],
                        "x_label": item.get("x_label", ""),
                        "y_label": item.get("y_label", ""),
                    },
                    ensure_ascii=False,
                ) + "\n")
            headers = []
            for index, item in enumerate(selected, start=1):
                headers.extend([item.get("x_label") or f"x{index}", item.get("y_label") or f"y{index}"])
            file.write("# " + " ".join(headers) + "\n")

            for row in range(max_len):
                values = []
                for curve in curves:
                    if row < len(curve["x"]):
                        values.extend([f"{curve['x'][row]:.12g}", f"{curve['y'][row]:.12g}"])
                    else:
                        values.extend(["nan", "nan"])
                file.write(" ".join(values) + "\n")

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose folder", str(self.current_folder))
        if folder:
            self.current_folder = Path(folder)
            self.folder_path.setText(str(self.current_folder))
            self.refresh_files()

    def set_folder_from_external_tab(self, folder):
        folder = Path(folder).expanduser().resolve()
        if self.current_folder.expanduser().resolve() == folder:
            return
        self._syncing_folder = True
        self.current_folder = folder
        self.folder_path.setText(str(self.current_folder))
        self.refresh_files()
        self._syncing_folder = False

    def refresh_files(self):
        folder = Path(self.folder_path.text()).expanduser()
        if not folder.exists():
            return

        self.current_folder = folder
        if not self._syncing_folder:
            self.folder_changed.emit(self.current_folder)

        patterns = self.extensions_filter.text().split()
        if not patterns:
            patterns = ["*.dat"]

        name_filter = self.name_filter.text().strip()
        if not name_filter:
            name_filter = "**"

        from fnmatch import fnmatch
        files = []
        glob_method = folder.rglob if self.show_subfolders.isChecked() else folder.glob

        for pattern in patterns:
            files.extend(glob_method(pattern))

        files = sorted(set(files))
        files = [file for file in files if file.is_file() and fnmatch(file.name, name_filter)]
        if self.only_thumbs_up_checkbox.isChecked():
            files = [file for file in files if is_file_rated_up(file)]

        self.file_list.blockSignals(True)
        self.file_list.clear()
        for file in files:
            display_name = str(file.relative_to(folder)) if self.show_subfolders.isChecked() else file.name
            self.file_list.addItem(display_name)
            item = self.file_list.item(self.file_list.count() - 1)
            set_item_file_path(item, file)
        self.file_list.blockSignals(False)

    def selection_changed(self):
        selected = self.selected_files()
        if not selected:
            self.update_graph_toolbar_enabled()
            return

        for file_path in selected:
            try:
                loaded_curves = read_dat_curves(file_path)
            except Exception as error:
                QMessageBox.warning(self, "File reading error", f"{file_path.name}\n\n{error}")
                continue

            for curve_index, loaded_curve in enumerate(loaded_curves):
                base_key = file_path.name if len(loaded_curves) == 1 else f"{file_path.name}:{loaded_curve['legend']}"
                if base_key in self.curves:
                    continue

                index = len(self.curves)
                saved_legend = self.saved_legend_for_file(file_path) if len(loaded_curves) == 1 else None
                legend = saved_legend or loaded_curve.get("legend") or file_path.stem
                y = np.asarray(loaded_curve["y"], dtype=float)
                self.curves[base_key] = {
                    "path": file_path,
                    "x": np.asarray(loaded_curve["x"], dtype=float),
                    "y": y,
                    "original_y": y.copy(),
                    "legend": legend,
                    "axis": normalize_plot_axis(loaded_curve.get("axis", "left")),
                    "x_label": loaded_curve.get("x_label", ""),
                    "y_label": loaded_curve.get("y_label", ""),
                    "manual_dataset": bool(loaded_curve.get("manual_dataset", False)),
                    "color": default_color(index),
                    "visible": True,
                }

        self.remove_duplicate_curves()
        self.refresh_curve_table()
        self.apply_default_plot_mode()
        self.update_plot()

    def remove_duplicate_curves(self):
        seen = set()
        deduplicated = {}
        for key, curve in self.curves.items():
            base_key = key
            marker_index = key.rfind(" (")
            if key.endswith(")") and marker_index != -1:
                suffix = key[marker_index + 2:-1]
                if suffix.isdigit():
                    base_key = key[:marker_index]

            path = curve.get("path")
            try:
                path_key = Path(path).resolve() if path is not None else None
            except OSError:
                path_key = Path(path) if path is not None else None
            identity = (path_key, base_key)
            if identity in seen:
                continue
            seen.add(identity)
            deduplicated[base_key if base_key not in deduplicated else key] = curve
        self.curves = deduplicated

    def selected_files(self):
        files = []
        seen_paths = set()
        for item in self.file_list.selectedItems():
            stored_path = item.data(Qt.UserRole)
            file_path = Path(stored_path) if stored_path else self.current_folder / item.text()
            try:
                key = file_path.resolve()
            except OSError:
                key = file_path
            if key in seen_paths:
                continue
            seen_paths.add(key)
            files.append(file_path)
        return files

    def update_clear_header_button_position(self):
        header = self.curve_table.horizontalHeader()
        column = 5
        x = header.sectionViewportPosition(column)
        width = header.sectionSize(column)
        y = max(0, (header.height() - self.clear_header_button.height()) // 2)
        self.clear_header_button.move(
            x + max(0, (width - self.clear_header_button.width()) // 2),
            y,
        )
        self.clear_header_button.raise_()

    def refresh_curve_table(self):
        self._refreshing_curve_table = True
        self.curve_table.blockSignals(True)
        self.curve_table.setRowCount(0)
        self.update_clear_header_button_position()

        for row, (key, curve) in enumerate(self.curves.items()):
            self.curve_table.insertRow(row)
            is_visible = curve.get("visible", True)
            row_color = QColor("#111827") if is_visible else QColor("#8a8f98")

            visibility_item = QTableWidgetItem("")
            visibility_item.setFlags(visibility_item.flags() & ~Qt.ItemIsEditable)
            self.curve_table.setItem(row, 0, visibility_item)

            visibility_button = EyeToggleButton(self.curve_table)
            visibility_button.setChecked(is_visible)
            visibility_button.setToolTip("Hide this curve" if is_visible else "Show this curve")
            visibility_button.toggled.connect(
                lambda visible, curve_key=key: self.set_curve_visible(curve_key, visible)
            )

            visibility_holder = QWidget()
            visibility_layout = QHBoxLayout(visibility_holder)
            visibility_layout.setContentsMargins(0, 0, 0, 0)
            visibility_layout.setSpacing(0)
            visibility_layout.addWidget(visibility_button, alignment=Qt.AlignCenter)
            self.curve_table.setCellWidget(row, 0, visibility_holder)

            file_item = QTableWidgetItem(key)
            file_item.setFlags(file_item.flags() & ~Qt.ItemIsEditable)
            file_item.setToolTip(str(curve["path"].name))
            file_item.setForeground(row_color)
            self.curve_table.setItem(row, 1, file_item)

            legend_item = QTableWidgetItem(curve["legend"])
            legend_item.setToolTip(str(curve["path"].name))
            legend_item.setForeground(row_color)
            self.curve_table.setItem(row, 2, legend_item)

            axis_item = QTableWidgetItem(normalize_plot_axis(curve.get("axis", "left")))
            axis_item.setToolTip("Use left, left2, right or right2 for the Y axis")
            axis_item.setForeground(row_color)
            self.curve_table.setItem(row, 3, axis_item)

            color_item = QTableWidgetItem("")
            color_item.setFlags(color_item.flags() & ~Qt.ItemIsEditable)
            color_item.setBackground(QColor(curve["color"]))
            color_item.setToolTip(curve["color"])
            self.curve_table.setItem(row, 4, color_item)

            remove_button = QPushButton("−")
            remove_button.setFixedSize(22, 18)
            remove_button.setToolTip("Remove this curve from the plot")
            remove_button.setStyleSheet("""
                QPushButton {
                    background: #ffecec;
                    color: #b00020;
                    border: 1px solid #ffb3b3;
                    border-radius: 8px;
                    font-weight: bold;
                    font-size: 11px;
                    padding: 0px;
                }
                QPushButton:hover {
                    background: #ffd6d6;
                }
            """)
            remove_button.clicked.connect(lambda checked=False, curve_key=key: self.remove_curve(curve_key))

            remove_holder = QWidget()
            remove_layout = QHBoxLayout(remove_holder)
            remove_layout.setContentsMargins(0, 0, 0, 0)
            remove_layout.setSpacing(0)
            remove_layout.addWidget(remove_button, alignment=Qt.AlignCenter)
            self.curve_table.setCellWidget(row, 5, remove_holder)

        self.curve_table.blockSignals(False)
        self._refreshing_curve_table = False

    def curve_table_changed(self, row, column):
        if self._refreshing_curve_table:
            return
        file_item = self.curve_table.item(row, 1)
        if file_item is None:
            return

        key = file_item.text()
        if key not in self.curves:
            return

        if column == 2:
            item = self.curve_table.item(row, column)
            self.curves[key]["legend"] = item.text() if item else self.curves[key]["legend"]
            self.remember_legend_for_file(self.curves[key]["path"], self.curves[key]["legend"])
            self.update_plot()
            return
        if column == 3:
            item = self.curve_table.item(row, column)
            axis = item.text() if item else "left"
            self.curves[key]["axis"] = normalize_plot_axis(axis)
            self.refresh_curve_table()
            self.update_plot()
            return
        elif column == 4:
            item = self.curve_table.item(row, column)
            if item:
                self.curves[key]["color"] = item.text()
                self.update_curve_color_only(key)
                return

        self.update_plot()

    def update_plot_legend_only(self):
        ax = self.canvas.ax
        previous_xlim = ax.get_xlim()
        previous_ylim = ax.get_ylim()
        previous_xscale = ax.get_xscale()
        previous_yscale = ax.get_yscale()

        for key, curve in self.curves.items():
            first = True
            for line in ax.lines:
                if line.get_gid() != key:
                    continue
                line.set_label(curve["legend"] if first else "_nolegend_")
                first = False

        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

        if self.show_legend.isChecked() and self.visible_curves():
            make_plot_legend(ax)

        ax.set_xscale(previous_xscale)
        ax.set_yscale(previous_yscale)
        ax.set_xlim(previous_xlim)
        ax.set_ylim(previous_ylim)
        finalize_plot_canvas(self.canvas)

    def update_curve_color_only(self, key):
        if key not in self.curves:
            return

        ax = self.canvas.ax
        previous_xlim = ax.get_xlim()
        previous_ylim = ax.get_ylim()
        previous_xscale = ax.get_xscale()
        previous_yscale = ax.get_yscale()

        for line in ax.lines:
            if line.get_gid() == key:
                line.set_color(self.curves[key]["color"])

        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        if self.show_legend.isChecked() and self.visible_curves():
            make_plot_legend(ax)

        ax.set_xscale(previous_xscale)
        ax.set_yscale(previous_yscale)
        ax.set_xlim(previous_xlim)
        ax.set_ylim(previous_ylim)
        finalize_plot_canvas(self.canvas)

    def rebuild_curve_gradient_buttons(self):
        layout = getattr(self, "curve_gradient_layout", None)
        if layout is None:
            return

        for button in getattr(self, "curve_gradient_buttons", []):
            layout.removeWidget(button)
            button.deleteLater()

        if getattr(self, "add_curve_gradient_color_button", None) is not None:
            layout.removeWidget(self.add_curve_gradient_color_button)
        if getattr(self, "apply_curve_gradient_button", None) is not None:
            layout.removeWidget(self.apply_curve_gradient_button)
        if getattr(self, "curve_gradient_stretch", None) is not None:
            layout.removeItem(self.curve_gradient_stretch)
            self.curve_gradient_stretch = None

        self.curve_gradient_buttons = []
        last_index = len(self.curve_gradient_colors) - 1
        for index, _color in enumerate(self.curve_gradient_colors):
            if index == 0:
                tooltip = "Start color"
            elif index == last_index:
                tooltip = "End color"
            else:
                tooltip = f"Color point {index + 1}"

            button = QToolButton()
            button.setFixedSize(24, 20)
            button.setToolTip(tooltip)
            button.clicked.connect(lambda checked=False, color_index=index: self.choose_curve_gradient_color(color_index))
            self.curve_gradient_buttons.append(button)
            layout.addWidget(button)

        layout.addWidget(self.add_curve_gradient_color_button)
        layout.addWidget(self.apply_curve_gradient_button)
        layout.addStretch(1)
        self.curve_gradient_stretch = layout.itemAt(layout.count() - 1)
        self.update_curve_gradient_buttons()

    def update_curve_gradient_buttons(self):
        for button, color in zip(getattr(self, "curve_gradient_buttons", []), self.curve_gradient_colors):
            button.setStyleSheet(f"""
                QToolButton {{
                    background: {color};
                    border: 1px solid #9ca3af;
                    border-radius: 6px;
                    padding: 0px;
                }}
                QToolButton:hover {{
                    border: 1px solid #111827;
                }}
            """)

        add_button = getattr(self, "add_curve_gradient_color_button", None)
        if add_button is not None:
            add_button.setStyleSheet("""
                QToolButton {
                    background: #f4f4f4;
                    border: 1px solid #9ca3af;
                    border-radius: 6px;
                    font-weight: bold;
                    padding: 0px;
                }
                QToolButton:hover {
                    border: 1px solid #111827;
                    background: #ffffff;
                }
            """)

    def choose_curve_gradient_color(self, color_index):
        if not 0 <= color_index < len(self.curve_gradient_colors):
            return

        color = QColorDialog.getColor(
            QColor(self.curve_gradient_colors[color_index]),
            self,
            "Choose gradient color",
        )
        if not color.isValid():
            return

        self.curve_gradient_colors[color_index] = color.name()
        self.update_curve_gradient_buttons()

    def add_curve_gradient_color_point(self):
        if len(self.curve_gradient_colors) < 2:
            self.curve_gradient_colors.append("#000000")
            self.rebuild_curve_gradient_buttons()
            return

        insert_index = max(1, len(self.curve_gradient_colors) - 1)
        new_position = insert_index / len(self.curve_gradient_colors)
        self.curve_gradient_colors.insert(insert_index, self.curve_gradient_color_at(new_position))
        self.rebuild_curve_gradient_buttons()

    def curve_gradient_color_at(self, position):
        colors = [QColor(color) for color in self.curve_gradient_colors if QColor(color).isValid()]
        if not colors:
            return "#000000"
        if len(colors) == 1:
            return colors[0].name()

        position = max(0.0, min(1.0, float(position)))
        scaled_position = position * (len(colors) - 1)
        left_index = int(np.floor(scaled_position))
        right_index = min(left_index + 1, len(colors) - 1)
        local_position = scaled_position - left_index

        left = colors[left_index]
        right = colors[right_index]

        red = round(left.red() + (right.red() - left.red()) * local_position)
        green = round(left.green() + (right.green() - left.green()) * local_position)
        blue = round(left.blue() + (right.blue() - left.blue()) * local_position)
        return QColor(red, green, blue).name()

    def apply_curve_color_gradient(self):
        if not self.curves:
            return

        curve_count = len(self.curves)
        for index, curve in enumerate(self.curves.values()):
            position = 0.5 if curve_count == 1 else index / (curve_count - 1)
            curve["color"] = self.curve_gradient_color_at(position)

        self.refresh_curve_table()
        self.update_plot()

    def visible_curves(self):
        stride_widget = getattr(self, "curve_display_stride_spin", None)
        stride = max(1, stride_widget.value() if stride_widget is not None else 1)
        curves = {}
        visible_index = 0

        for key, curve in self.curves.items():
            if not curve.get("visible", True):
                continue
            if visible_index % stride == 0:
                curves[key] = curve
            visible_index += 1

        return curves

    def open_power_law_fit_dialog(self):
        if not self.curves:
            QMessageBox.warning(self, "No curves", "Load at least one I(q) curve before fitting.")
            return
        visible_curves = self.visible_curves()
        if not visible_curves:
            QMessageBox.warning(self, "No visible curves", "Show at least one I(q) curve before fitting.")
            return
        if self.curves_are_really_0_to_360():
            QMessageBox.warning(self, "Not an I(q) plot", "Power-law fitting is only available for I(q) curves.")
            return
        if self.plot_mode.currentText() in PLOT_TRANSFORMED_MODES:
            QMessageBox.warning(
                self,
                "Transformed plot",
                "Switch to an I(q) mode before fitting I(q) = A q^-n."
            )
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Power-law fit")
        dialog.resize(900, 650)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        controls = QVBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(6)
        curve_row = QHBoxLayout()
        curve_row.setContentsMargins(0, 0, 0, 0)
        curve_row.setSpacing(8)
        fit_row = QHBoxLayout()
        fit_row.setContentsMargins(0, 0, 0, 0)
        fit_row.setSpacing(8)

        curve_combo = QComboBox()
        for key, curve in visible_curves.items():
            curve_combo.addItem(curve["legend"], key)

        exponent_combo = QComboBox()
        exponent_combo.addItems(["free n", "q^-1", "q^-2", "q^-3", "q^-4"])

        q_min_spin = QDoubleSpinBox()
        q_max_spin = QDoubleSpinBox()
        for spin in (q_min_spin, q_max_spin):
            spin.setDecimals(6)
            spin.setRange(0.0, 1e9)
            spin.setSingleStep(0.01)
            spin.setMinimumWidth(110)

        xlim = self.canvas.ax.get_xlim()
        q_min_spin.setValue(max(0.0, float(min(xlim))))
        q_max_spin.setValue(max(0.0, float(max(xlim))))

        fit_button = QPushButton("Fit")
        result_label = QLabel("I(q) = A q^-n")
        result_label.setMinimumWidth(260)
        coordinate_label = QLabel("q = - | I = -")
        coordinate_label.setMinimumHeight(26)
        coordinate_label.setAlignment(Qt.AlignCenter)
        coordinate_label.setStyleSheet("""
            QLabel {
                background-color: #f4f4f4;
                border-radius: 8px;
                padding: 5px;
                font-family: Menlo, Monaco, monospace;
                font-size: 11px;
            }
        """)

        curve_row.addWidget(QLabel("Curve:"))
        curve_row.addWidget(curve_combo, stretch=1)
        fit_row.addWidget(QLabel("Model:"))
        fit_row.addWidget(exponent_combo)
        fit_row.addWidget(QLabel("q min:"))
        fit_row.addWidget(q_min_spin)
        fit_row.addWidget(QLabel("q max:"))
        fit_row.addWidget(q_max_spin)
        fit_row.addWidget(fit_button)
        fit_row.addWidget(result_label, stretch=1)
        controls.addLayout(curve_row)
        controls.addLayout(fit_row)
        layout.addLayout(controls)

        fig = Figure()
        fit_canvas = FigureCanvas(fig)
        fit_ax = fig.add_subplot(111)
        fit_toolbar = NavigationToolbar(fit_canvas, dialog)
        fit_toolbar.coordinates = False
        layout.addWidget(fit_toolbar)
        layout.addWidget(fit_canvas, stretch=1)
        layout.addWidget(coordinate_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        fit_state = {"x_fit": None, "y_fit": None, "label": None}

        def current_curve_arrays():
            key = curve_combo.currentData()
            curve = self.curves.get(key)
            if curve is None:
                return None, None, None
            x = np.asarray(self.make_plot_x(curve["x"]), dtype=float)
            y = np.asarray(curve["y"], dtype=float)
            return curve, x, y

        def redraw_fit_plot():
            fit_ax.clear()
            mode = self.plot_mode.currentText()

            for curve in visible_curves.values():
                x = np.asarray(self.make_plot_x(curve["x"]), dtype=float)
                y = np.asarray(curve["y"], dtype=float)
                valid = np.isfinite(x) & np.isfinite(y)
                if mode in ("log linear", "log log"):
                    valid &= x > 0
                if mode in ("linear log", "log log"):
                    valid &= y > 0
                if np.any(valid):
                    fit_ax.plot(
                        x[valid],
                        y[valid],
                        linewidth=1.4,
                        color=curve["color"],
                        label=curve["legend"],
                    )

            if fit_state["x_fit"] is not None:
                fit_ax.plot(
                    fit_state["x_fit"],
                    fit_state["y_fit"],
                    color="black",
                    linestyle="--",
                    linewidth=2.0,
                    label=fit_state["label"],
                )

            fit_ax.set_xscale(self.canvas.ax.get_xscale())
            fit_ax.set_yscale(self.canvas.ax.get_yscale())
            fit_ax.set_xlabel(self.q_axis_label())
            fit_ax.set_ylabel("Intensity / a.u.")
            fit_ax.grid(True, which="both")
            install_selectable_legend(fit_ax, fit_ax.legend(loc="best"))

            x0, x1 = self.canvas.ax.get_xlim()
            y0, y1 = self.canvas.ax.get_ylim()
            if np.isfinite([x0, x1, y0, y1]).all():
                fit_ax.set_xlim(x0, x1)
                fit_ax.set_ylim(y0, y1)

            fig.tight_layout()
            fit_canvas.draw_idle()

        def run_fit():
            curve, x, y = current_curve_arrays()
            if curve is None:
                return

            q_min = min(q_min_spin.value(), q_max_spin.value())
            q_max = max(q_min_spin.value(), q_max_spin.value())
            valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0) & (x >= q_min) & (x <= q_max)
            if np.count_nonzero(valid) < 2:
                QMessageBox.warning(dialog, "Fit impossible", "Not enough positive I(q) points in this q range.")
                return

            x_fit_data = x[valid]
            y_fit_data = y[valid]
            log_q = np.log(x_fit_data)
            log_i = np.log(y_fit_data)

            model_text = exponent_combo.currentText()
            if model_text == "free n":
                slope, log_a = np.polyfit(log_q, log_i, 1)
                exponent = -float(slope)
            else:
                exponent = float(model_text.replace("q^-", ""))
                log_a = float(np.mean(log_i + exponent * log_q))

            amplitude = float(np.exp(log_a))
            q_line = np.linspace(float(np.nanmin(x_fit_data)), float(np.nanmax(x_fit_data)), 300)
            y_line = amplitude * q_line ** (-exponent)
            predicted = amplitude * x_fit_data ** (-exponent)
            residual = log_i - np.log(predicted)
            rmse = float(np.sqrt(np.mean(residual ** 2)))

            fit_state["x_fit"] = q_line
            fit_state["y_fit"] = y_line
            fit_state["label"] = f"{curve['legend']} fit: q^-{exponent:.3g}"
            result_label.setText(f"A = {amplitude:.4g} | n = {exponent:.4g} | log RMSE = {rmse:.3g}")
            redraw_fit_plot()

        def update_fit_coordinates(event):
            if event.inaxes != fit_ax or event.xdata is None or event.ydata is None:
                coordinate_label.setText("q = - | I = -")
                return
            unit_label = "Å⁻¹" if self.q_axis_unit == "A" else "nm⁻¹"
            coordinate_label.setText(f"q = {event.xdata:.6g} {unit_label} | I = {event.ydata:.6g}")

        def clear_fit_coordinates(event=None):
            coordinate_label.setText("q = - | I = -")

        fit_button.clicked.connect(run_fit)
        curve_combo.currentIndexChanged.connect(redraw_fit_plot)
        exponent_combo.currentIndexChanged.connect(redraw_fit_plot)
        q_min_spin.valueChanged.connect(redraw_fit_plot)
        q_max_spin.valueChanged.connect(redraw_fit_plot)
        fit_canvas.mpl_connect("motion_notify_event", update_fit_coordinates)
        fit_canvas.mpl_connect("axes_leave_event", clear_fit_coordinates)

        redraw_fit_plot()
        dialog.exec()

    def update_plot_preserving_view(self):
        ax = self.canvas.ax
        previous_xlim = ax.get_xlim()
        previous_ylim = ax.get_ylim()
        previous_xscale = ax.get_xscale()
        previous_yscale = ax.get_yscale()

        self.update_plot()

        self.canvas.ax.set_xscale(previous_xscale)
        self.canvas.ax.set_yscale(previous_yscale)
        self.canvas.ax.set_xlim(previous_xlim)
        self.canvas.ax.set_ylim(previous_ylim)
        finalize_plot_canvas(self.canvas)

    def default_guide_value(self, axis):
        ax = self.canvas.ax
        limits = ax.get_xlim() if axis == "x" else ax.get_ylim()
        scale = ax.get_xscale() if axis == "x" else ax.get_yscale()
        left, right = limits
        if scale == "log" and left > 0 and right > 0:
            return float(np.sqrt(left * right))
        return float((left + right) / 2)

    def add_guide_bar(self, axis):
        self.guide_bars.append({
            "axis": axis,
            "value": self.default_guide_value(axis),
            "color": "#444444",
        })
        self.refresh_guide_table()
        self.update_plot_preserving_view()

    def refresh_guide_table(self):
        self._refreshing_guide_table = True
        self.guide_table.blockSignals(True)
        self.guide_table.setRowCount(0)

        for row, bar in enumerate(self.guide_bars):
            self.guide_table.insertRow(row)

            axis_item = QTableWidgetItem(bar["axis"].upper())
            axis_item.setFlags(axis_item.flags() & ~Qt.ItemIsEditable)
            self.guide_table.setItem(row, 0, axis_item)

            value_item = QTableWidgetItem(f"{bar['value']:.6g}")
            self.guide_table.setItem(row, 1, value_item)

            color_item = QTableWidgetItem("")
            color_item.setFlags(color_item.flags() & ~Qt.ItemIsEditable)
            color_item.setBackground(QColor(bar["color"]))
            color_item.setToolTip(bar["color"])
            self.guide_table.setItem(row, 2, color_item)

            remove_button = QPushButton("−")
            remove_button.setFixedSize(22, 18)
            remove_button.setToolTip("Remove this dashed bar")
            remove_button.setStyleSheet("""
                QPushButton {
                    background: #ffecec;
                    color: #b00020;
                    border: 1px solid #ffb3b3;
                    border-radius: 8px;
                    font-weight: bold;
                    font-size: 11px;
                    padding: 0px;
                }
                QPushButton:hover {
                    background: #ffd6d6;
                }
            """)
            remove_button.clicked.connect(lambda checked=False, bar_row=row: self.remove_guide_bar(bar_row))

            remove_holder = QWidget()
            remove_layout = QHBoxLayout(remove_holder)
            remove_layout.setContentsMargins(0, 0, 0, 0)
            remove_layout.setSpacing(0)
            remove_layout.addWidget(remove_button, alignment=Qt.AlignCenter)
            self.guide_table.setCellWidget(row, 3, remove_holder)

        self.guide_table.blockSignals(False)
        self._refreshing_guide_table = False

    def guide_table_changed(self, row, column):
        if self._refreshing_guide_table or column != 1:
            return
        if not 0 <= row < len(self.guide_bars):
            return

        item = self.guide_table.item(row, column)
        if item is None:
            return

        text = item.text().strip().replace(",", ".")
        try:
            value = float(text)
        except ValueError:
            self.refresh_guide_table()
            return

        self.guide_bars[row]["value"] = value
        self.update_plot_preserving_view()

    def guide_table_double_clicked(self, row, column):
        if column != 2 or not 0 <= row < len(self.guide_bars):
            return

        color = QColorDialog.getColor(QColor(self.guide_bars[row]["color"]), self, "Choose dashed bar color")
        if not color.isValid():
            return

        self.guide_bars[row]["color"] = color.name()
        self.refresh_guide_table()
        self.update_plot_preserving_view()

    def remove_guide_bar(self, row):
        if not 0 <= row < len(self.guide_bars):
            return

        del self.guide_bars[row]
        self.refresh_guide_table()
        self.update_plot_preserving_view()

    def draw_guide_bars(self, ax):
        for bar in self.guide_bars:
            value = float(bar.get("value", 0.0))
            if not np.isfinite(value):
                continue

            color = bar.get("color", "#444444")
            axis = bar.get("axis", "x")
            if axis == "x":
                line = ax.axvline(value, color=color, linestyle="--", linewidth=1.2, alpha=0.9, label="_nolegend_")
            else:
                line = ax.axhline(value, color=color, linestyle="--", linewidth=1.2, alpha=0.9, label="_nolegend_")
            line.set_gid("guide_bar")

    def draw_peak_labels(self, ax):
        self.peak_label_artists = []
        for index, label_data in enumerate(self.peak_labels):
            points = label_data.get("points")
            if points is None:
                points = [(label_data.get("x", 0.0), label_data.get("y", 0.0))]
                label_data["points"] = points
            if not points:
                continue

            first_x, first_y = points[0]
            text_x = float(label_data.get("text_x", first_x))
            text_y = float(label_data.get("text_y", first_y))
            name = str(label_data.get("name", "label"))
            if not all(np.isfinite(value) for value in (text_x, text_y)):
                continue
            if ax.get_xscale() == "log" and text_x <= 0:
                continue
            if ax.get_yscale() == "log" and text_y <= 0:
                continue

            has_visible_arrow = False
            for point_index, point in enumerate(points):
                x_value = float(point[0])
                y_value = float(point[1])
                if not all(np.isfinite(value) for value in (x_value, y_value)):
                    continue
                if ax.get_xscale() == "log" and x_value <= 0:
                    continue
                if ax.get_yscale() == "log" and y_value <= 0:
                    continue

                ax.annotate(
                    "",
                    xy=(x_value, y_value),
                    xytext=(text_x, text_y),
                    textcoords="data",
                    arrowprops=dict(arrowstyle="->", color="black", linewidth=1.4),
                    zorder=20,
                )
                has_visible_arrow = True

            if not has_visible_arrow:
                continue

            text_artist = ax.text(
                text_x,
                text_y,
                name,
                ha="center",
                va="center",
                fontsize=9,
                color="black",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="black", alpha=0.94),
                zorder=21,
            )
            self.peak_label_artists.append({"index": index, "text": text_artist})

    def toggle_peak_label_mode(self, checked):
        if checked and not self.curves:
            self.add_axis_label_button.setChecked(False)

    def clear_peak_labels(self):
        self.peak_labels = []
        self._dragging_peak_label = None
        self.add_axis_label_button.setChecked(False)
        self.update_plot()

    def peak_label_hit(self, event):
        if event.inaxes != self.canvas.ax or event.xdata is None:
            return None
        event_xy = np.array([event.x, event.y], dtype=float)
        for item in reversed(self.peak_label_artists):
            text = item.get("text")
            if text is None:
                continue
            try:
                bbox = text.get_window_extent(renderer=self.canvas.get_renderer()).expanded(1.2, 1.6)
                if bbox.contains(event_xy[0], event_xy[1]):
                    return item["index"]
            except Exception:
                pass
        return None

    def add_peak_label_at(self, event):
        if event.inaxes != self.canvas.ax or event.xdata is None or event.ydata is None:
            return

        name, ok = QInputDialog.getText(
            self,
            "Add peak label",
            "Name (same name = same label):",
            text=self.last_peak_label_name,
        )
        if not ok:
            self.add_axis_label_button.setChecked(False)
            return
        name = name.strip() or "peak"
        self.last_peak_label_name = name

        xlim = self.canvas.ax.get_xlim()
        ylim = self.canvas.ax.get_ylim()
        x_span = abs(xlim[1] - xlim[0])
        y_span = abs(ylim[1] - ylim[0])
        text_x = float(event.xdata + 0.06 * x_span)
        text_y = float(event.ydata + 0.08 * y_span)
        if self.canvas.ax.get_xscale() == "log" and event.xdata > 0:
            text_x = float(event.xdata * 1.2)
        if self.canvas.ax.get_yscale() == "log" and event.ydata > 0:
            text_y = float(event.ydata * 1.35)

        for label_data in self.peak_labels:
            if str(label_data.get("name", "")) == name:
                points = label_data.setdefault("points", [])
                points.append((float(event.xdata), float(event.ydata)))
                self.add_axis_label_button.setChecked(False)
                self.update_plot()
                return

        self.peak_labels.append({
            "name": name,
            "points": [(float(event.xdata), float(event.ydata))],
            "text_x": text_x,
            "text_y": text_y,
        })
        self.add_axis_label_button.setChecked(False)
        self.update_plot()

    def plot_curve_segments(self, ax, key, curve, x, y, mode):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)

        if mode in PLOT_LOG_X_MODES:
            valid &= x > 0
        if mode in PLOT_LOG_Y_MODES:
            valid &= y > 0

        ranges = []
        start = None
        for index, is_valid in enumerate(valid):
            if is_valid and start is None:
                start = index
            elif not is_valid and start is not None:
                ranges.append((start, index))
                start = None

        if start is not None:
            ranges.append((start, len(valid)))

        if not ranges:
            line, = ax.plot([], [], linewidth=1.6, label=curve["legend"], color=curve["color"])
            line.set_gid(key)
            return

        for segment_index, (start, end) in enumerate(ranges):
            line, = ax.plot(
                x[start:end],
                y[start:end],
                linewidth=1.6,
                label=curve["legend"] if segment_index == 0 else "_nolegend_",
                color=curve["color"],
                antialiased=True,
                solid_capstyle="round",
                solid_joinstyle="round",
            )
            line.set_gid(key)

    def move_curve_row(self, source_row, destination_row):
        keys = list(self.curves.keys())
        if not 0 <= source_row < len(keys):
            return
        self.move_curve_key(keys[source_row], destination_row)

    def move_curve_key(self, source_key, destination_row):
        previous_xlim = self.canvas.ax.get_xlim()
        previous_ylim = self.canvas.ax.get_ylim()
        previous_xscale = self.canvas.ax.get_xscale()
        previous_yscale = self.canvas.ax.get_yscale()

        keys = list(self.curves.keys())
        if source_key not in keys:
            return

        source_row = keys.index(source_key)
        destination_row = max(0, min(destination_row, len(keys)))
        key = keys.pop(source_row)
        if destination_row > source_row:
            destination_row -= 1
        keys.insert(destination_row, key)

        self.curves = {curve_key: self.curves[curve_key] for curve_key in keys}
        self.refresh_curve_table()
        self.curve_table.selectRow(destination_row)
        self.update_plot()
        self.canvas.ax.set_xscale(previous_xscale)
        self.canvas.ax.set_yscale(previous_yscale)
        self.canvas.ax.set_xlim(previous_xlim)
        self.canvas.ax.set_ylim(previous_ylim)
        finalize_plot_canvas(self.canvas)

    def curve_rows_moved(self, parent, start, end, destination, row):
        """Handle curve reordering when dragged in the table."""
        if self._refreshing_curve_table:
            return

        # Rebuild the curves dict based on the current table order
        reordered_curves = {}
        
        # Iterate through all rows in the table and preserve their order
        for table_row in range(self.curve_table.rowCount()):
            file_item = self.curve_table.item(table_row, 1)
            
            # Skip if item is None
            if file_item is None:
                continue
            
            key = file_item.text()
            
            # Only add keys that actually exist in our curves dict
            if key in self.curves:
                reordered_curves[key] = self.curves[key]
        
        # Update the curves dict
        self.curves = reordered_curves
        
        # Clear selection to fix visual glitches
        self.curve_table.clearSelection()
        
        # Update the plot
        self.update_plot()

    def curve_table_double_clicked(self, row, column):
        if column != 4:
            return

        file_item = self.curve_table.item(row, 1)
        if file_item is None:
            return

        key = file_item.text()
        if key not in self.curves:
            return

        color = QColorDialog.getColor(QColor(self.curves[key]["color"]), self, "Choose curve color")
        if not color.isValid():
            return

        self.curves[key]["color"] = color.name()
        self.refresh_curve_table()
        self.update_curve_color_only(key)

    def remove_curve(self, key):
        if key in self.curves:
            del self.curves[key]
        self.refresh_curve_table()
        self.update_plot()

    def set_curve_visible(self, key, visible):
        if key not in self.curves:
            return

        self.curves[key]["visible"] = bool(visible)
        self.refresh_curve_table()
        self.update_plot()

    def clear_curves(self):
        self.curves.clear()
        self.refresh_curve_table()
        clear_plot_canvas(self.canvas)
        self.clear_graph_coordinates()
        self.update_graph_toolbar_enabled()

    def selected_curve_keys(self):
        keys = []
        for index in self.curve_table.selectionModel().selectedRows():
            item = self.curve_table.item(index.row(), 1)
            if item is not None and item.text() in self.curves:
                keys.append(item.text())
        return keys

    def open_curve_table_menu(self, position):
        item = self.curve_table.itemAt(position)
        if item is None:
            return

        row = item.row()
        key_item = self.curve_table.item(row, 1)
        if key_item is None:
            return

        curve_key = key_item.text()
        if curve_key not in self.curves:
            return

        self.curve_table.selectRow(row)
        legend = self.curves[curve_key]["legend"]

        menu = QMenu(self.curve_table)
        mask_action = menu.addAction(f"Mask range on {legend}")
        reset_action = menu.addAction(f"Reset mask on {legend}")

        action = menu.exec(self.curve_table.viewport().mapToGlobal(position))
        if action is mask_action:
            self.open_mask_range_dialog(curve_key=curve_key)
        elif action is reset_action:
            curve = self.curves[curve_key]
            if "original_y" in curve:
                curve["y"] = np.asarray(curve["original_y"], dtype=float).copy()
                self.update_plot()

    def open_mask_range_dialog(self, curve_key=None, center_x=None):
        if not self.curves:
            return

        x_label, _ = self.graph_coordinate_labels()
        dialog = QDialog(self)
        dialog.setWindowTitle("Mask data range")
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        if curve_key is not None and curve_key in self.curves:
            target_text = f"Target: {self.curves[curve_key]['legend']}"
        else:
            selected_count = len(self.selected_curve_keys())
            target_text = f"Target: {selected_count} selected curve(s)" if selected_count else "Target: all curves"
        target_label = QLabel(target_text)
        layout.addWidget(target_label)

        range_layout = QGridLayout()
        xlim = self.canvas.ax.get_xlim()
        if center_x is not None and np.isfinite(center_x):
            span = abs(xlim[1] - xlim[0])
            half_width = span * 0.01
            if self.canvas.ax.get_xscale() == "log" and center_x > 0:
                factor = 10 ** 0.01
                default_min = center_x / factor
                default_max = center_x * factor
            else:
                default_min = center_x - half_width
                default_max = center_x + half_width
        else:
            default_min, default_max = xlim

        min_spin = self.double_spin(default_min)
        max_spin = self.double_spin(default_max)
        min_spin.setFixedWidth(130)
        max_spin.setFixedWidth(130)
        range_layout.addWidget(QLabel(f"{x_label} min:"), 0, 0)
        range_layout.addWidget(min_spin, 0, 1)
        range_layout.addWidget(QLabel(f"{x_label} max:"), 1, 0)
        range_layout.addWidget(max_spin, 1, 1)
        layout.addLayout(range_layout)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() != QDialog.Accepted:
            return

        self.mask_data_range(min_spin.value(), max_spin.value(), curve_key=curve_key)

    def mask_data_range(self, x_min, x_max, curve_key=None):
        if not self.curves:
            return

        if x_max < x_min:
            x_min, x_max = x_max, x_min

        data_x_min = x_min
        data_x_max = x_max
        if not self.curves_are_really_0_to_360():
            factor = self.q_display_factor()
            if factor != 0:
                data_x_min = x_min / factor
                data_x_max = x_max / factor

        keys = [curve_key] if curve_key in self.curves else (self.selected_curve_keys() or list(self.curves.keys()))
        total_masked = 0

        for key in keys:
            curve = self.curves[key]
            x = np.asarray(curve["x"], dtype=float)
            y = np.asarray(curve["y"], dtype=float).copy()
            mask = np.isfinite(x) & (x >= data_x_min) & (x <= data_x_max)
            if np.any(mask):
                y[mask] = np.nan
                curve["y"] = y
                total_masked += int(np.count_nonzero(mask))

        if total_masked == 0:
            QMessageBox.information(self, "Mask range", "No point was found in this range.")
            return

        self.update_plot()

    def reset_curve_masks(self):
        if not self.curves:
            return

        keys = self.selected_curve_keys() or list(self.curves.keys())
        for key in keys:
            curve = self.curves[key]
            if "original_y" in curve:
                curve["y"] = np.asarray(curve["original_y"], dtype=float).copy()

        self.update_plot()

    def nearest_curve_key_at(self, event):
        if event.inaxes != self.canvas.ax or event.xdata is None or event.ydata is None:
            return None

        click_display = self.canvas.ax.transData.transform((event.xdata, event.ydata))
        best_key = None
        best_distance = float("inf")

        for key, curve in self.visible_curves().items():
            x = np.asarray(self.make_plot_x(curve["x"]), dtype=float)
            y = np.asarray(self.make_plot_y(curve["x"], curve["y"]), dtype=float)
            valid = np.isfinite(x) & np.isfinite(y)
            if self.canvas.ax.get_xscale() == "log":
                valid &= x > 0
            if self.canvas.ax.get_yscale() == "log":
                valid &= y > 0
            if not np.any(valid):
                continue

            points = self.canvas.ax.transData.transform(np.column_stack((x[valid], y[valid])))
            distances = np.hypot(points[:, 0] - click_display[0], points[:, 1] - click_display[1])
            distance = float(np.nanmin(distances))
            if distance < best_distance:
                best_distance = distance
                best_key = key

        return best_key if best_distance <= 18 else None

    def graph_button_press(self, event):
        if event.button == 1:
            try:
                clicked_label = self.canvas.ax.xaxis.label.contains(event)[0]
            except Exception:
                clicked_label = False
            if clicked_label and self.curves and not self.curves_are_really_0_to_360():
                self.q_axis_unit = "A" if self.q_axis_unit == "nm" else "nm"
                if self.x_label.text() in ("", "q / nm⁻¹", "q / Å⁻¹"):
                    self.x_label.blockSignals(True)
                    self.x_label.setText(self.q_axis_label())
                    self.x_label.blockSignals(False)
                self.update_plot()
                return

            hit = self.peak_label_hit(event)
            if hit is not None:
                self._dragging_peak_label = hit
                self.add_axis_label_button.setChecked(False)
                return
            if self.add_axis_label_button.isChecked():
                self.add_peak_label_at(event)
            return

        if event.button != 3:
            return

        axis_hit = self.axis_label_hit(event)
        if axis_hit is not None:
            self.rename_axis_label(axis_hit)
            return

        curve_key = self.nearest_curve_key_at(event)
        if curve_key is None:
            return

        menu = QMenu(self)
        legend = self.curves[curve_key]["legend"]
        mask_action = menu.addAction(f"Mask range on {legend}")
        reset_action = menu.addAction(f"Reset mask on {legend}")

        try:
            global_pos = event.guiEvent.globalPosition().toPoint()
        except Exception:
            global_pos = self.canvas.mapToGlobal(self.canvas.rect().center())

        action = menu.exec(global_pos)
        if action is mask_action:
            self.open_mask_range_dialog(curve_key=curve_key, center_x=event.xdata)
        elif action is reset_action:
            curve = self.curves[curve_key]
            if "original_y" in curve:
                curve["y"] = np.asarray(curve["original_y"], dtype=float).copy()
                self.update_plot()

    def axis_label_hit(self, event):
        try:
            if self.canvas.ax.xaxis.label.contains(event)[0]:
                return ("x", None)
        except Exception:
            pass

        axis_map = {"left": self.canvas.ax}
        axis_map.update(getattr(self, "extra_axes", {}))
        for axis_name, axis in axis_map.items():
            try:
                if axis.yaxis.label.contains(event)[0]:
                    return ("y", axis_name)
            except Exception:
                continue
        return None

    def rename_axis_label(self, axis_hit):
        kind, axis_name = axis_hit
        if kind == "x":
            current = self.canvas.ax.get_xlabel()
            text, ok = QInputDialog.getText(self, "Rename X axis", "X axis title:", text=current)
            if not ok:
                return
            label = text.strip()
            self.x_label.blockSignals(True)
            self.x_label.setText(label)
            self.x_label.blockSignals(False)
            for curve in self.curves.values():
                curve["x_label"] = label
            self.update_plot()
            return

        current = ""
        axis_map = {"left": self.canvas.ax}
        axis_map.update(getattr(self, "extra_axes", {}))
        if axis_name in axis_map:
            current = axis_map[axis_name].get_ylabel()
        text, ok = QInputDialog.getText(self, "Rename Y axis", "Y axis title:", text=current)
        if not ok:
            return
        label = text.strip()
        if axis_name == "left":
            self.y_label.blockSignals(True)
            self.y_label.setText(label)
            self.y_label.blockSignals(False)
        for curve in self.curves.values():
            if normalize_plot_axis(curve.get("axis", "left")) == axis_name:
                curve["y_label"] = label
        self.update_plot()

    def graph_button_release(self, event):
        self._dragging_peak_label = None

    def update_graph_toolbar_enabled(self):
        enabled = bool(self.curves)
        for widget in [
            getattr(self, "plot_mode", None),
            getattr(self, "fit_button", None),
            getattr(self, "curve_display_stride_spin", None),
            getattr(self, "show_legend", None),
            getattr(self, "keep_zoom_checkbox", None),
            getattr(self, "save_plot_button", None),
            getattr(self, "apply_curve_gradient_button", None),
            getattr(self, "mask_range_button", None),
            getattr(self, "reset_masks_button", None),
            getattr(self, "add_axis_label_button", None),
            getattr(self, "clear_axis_labels_button", None),
        ]:
            if widget is not None:
                widget.setEnabled(enabled)
        if getattr(self, "create_dat_button", None) is not None:
            self.create_dat_button.setEnabled(True)

    def make_plot_y(self, x, y):
        mode = self.plot_mode.currentText()
        transform = PLOT_TRANSFORMED_MODES.get(mode)
        if transform is not None:
            q = self.make_display_q(x)
            return q ** transform["power"] * y

        return y

    def q_display_factor(self):
        return 0.1 if self.q_axis_unit == "A" else 1.0

    def make_plot_x(self, x):
        mode = self.plot_mode.currentText()
        q = self.make_display_q(x)
        transform = PLOT_TRANSFORMED_MODES.get(mode)
        if transform is not None:
            return q ** transform["x_power"]
        return q

    def make_display_q(self, x):
        if self.curves_are_really_0_to_360():
            return x
        return np.asarray(x, dtype=float) * self.q_display_factor()

    def q_axis_label(self):
        return "q / Å⁻¹" if self.q_axis_unit == "A" else "q / nm⁻¹"

    def graph_coordinate_labels(self):
        if self.curves_are_really_0_to_360():
            return "ψ", "I"
        transform = PLOT_TRANSFORMED_MODES.get(self.plot_mode.currentText())
        if transform is not None:
            return transform["x_label"], transform["y_label"]
        return "q", "I"

    def update_graph_coordinates(self, event):
        if self._dragging_peak_label is not None:
            if event.inaxes == self.canvas.ax and event.xdata is not None and event.ydata is not None:
                x_value = float(event.xdata)
                y_value = float(event.ydata)
                if self.canvas.ax.get_xscale() == "log" and x_value <= 0:
                    return
                if self.canvas.ax.get_yscale() == "log" and y_value <= 0:
                    return
                if 0 <= self._dragging_peak_label < len(self.peak_labels):
                    self.peak_labels[self._dragging_peak_label]["text_x"] = x_value
                    self.peak_labels[self._dragging_peak_label]["text_y"] = y_value
                    self.update_plot()
            return

        if event.inaxes != self.canvas.ax or event.xdata is None or event.ydata is None:
            return

        try:
            x_name, y_name = self.graph_coordinate_labels()
            if x_name == "ψ":
                x_suffix = "°"
            else:
                x_suffix = " Å⁻¹" if self.q_axis_unit == "A" else " nm⁻¹"
            self.graph_coordinate_label.setText(
                f"{x_name} = {event.xdata:.6g}{x_suffix} | {y_name} = {event.ydata:.6g}"
            )
        except Exception:
            self.clear_graph_coordinates()

    def clear_graph_coordinates(self, event=None):
        x_name, y_name = self.graph_coordinate_labels()
        self.graph_coordinate_label.setText(f"{x_name} = - | {y_name} = -")

    def curves_are_really_0_to_360(self):
        curves = self.visible_curves()
        if not curves:
            return False

        if any("azimprof" in curve["path"].name.lower() for curve in curves.values()):
            return True

        for curve in curves.values():
            x = curve["x"]
            valid = x[np.isfinite(x)]
            if valid.size == 0:
                return False

            x_min = float(np.nanmin(valid))
            x_max = float(np.nanmax(valid))
            if not (abs(x_min - 0.0) <= 1e-6 and abs(x_max - 360.0) <= 1e-6):
                return False

        return True

    def apply_default_plot_mode(self):
        curves = self.visible_curves()
        if not curves:
            return
        has_manual_dataset = any(curve.get("manual_dataset", False) for curve in curves.values())
        mode = "linear linear" if has_manual_dataset or self.curves_are_really_0_to_360() else "log log"

        if self.plot_mode.currentText() == mode:
            return

        self.plot_mode.blockSignals(True)
        self.plot_mode.setCurrentText(mode)
        self.plot_mode.blockSignals(False)

    def update_limit_state(self):
        auto = self.auto_limits.isChecked()

        if auto:
            self.update_limit_fields_from_current_data()

        for widget in [self.x_min, self.x_max, self.y_min, self.y_max]:
            widget.setEnabled(not auto)

        self.update_plot()

    def update_limit_fields_from_current_data(self):
        curves = self.visible_curves()
        if not curves:
            return

        all_x = []
        all_y = []

        for curve in curves.values():
            x = self.make_plot_x(curve["x"])
            y = self.make_plot_y(curve["x"], curve["y"])

            valid = np.isfinite(x) & np.isfinite(y)
            if np.any(valid):
                all_x.append(x[valid])
                all_y.append(y[valid])

        if not all_x or not all_y:
            return

        x_values = np.concatenate(all_x)
        has_azim_profile = self.curves_are_really_0_to_360()
        y_values = np.concatenate(all_y)

        mode = self.plot_mode.currentText()
        if mode in PLOT_LOG_X_MODES:
            x_values = x_values[x_values > 0]
        if mode in PLOT_LOG_Y_MODES:
            y_values = y_values[y_values > 0]

        if x_values.size == 0 or y_values.size == 0:
            return

        if has_azim_profile:
            x_min = 0.0
            x_max = 360.0
        else:
            x_min = float(np.nanmin(x_values))
            x_max = float(np.nanmax(x_values))
        y_min = float(np.nanmin(y_values))
        y_max = float(np.nanmax(y_values))

        if x_max == x_min:
            x_max = x_min + 1
        if y_max == y_min:
            y_max = y_min + 1

        for spin, value in [
            (self.x_min, x_min),
            (self.x_max, x_max),
            (self.y_min, y_min),
            (self.y_max, y_max),
        ]:
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)

    def update_plot(self):
        ax = self.canvas.ax
        keep_zoom = (
            getattr(self, "keep_zoom_checkbox", None) is not None
            and self.keep_zoom_checkbox.isChecked()
            and ax.has_data()
        )
        previous_xlim = ax.get_xlim() if keep_zoom else None
        previous_ylim = ax.get_ylim() if keep_zoom else None

        for extra_ax in getattr(self, "extra_axes", {}).values():
            extra_ax.remove()
        self.extra_axes = {}
        ax.clear()

        if not self.curves:
            self.clear_graph_coordinates()
            clear_plot_canvas(self.canvas)
            self.update_graph_toolbar_enabled()
            return

        ax.set_axis_on()
        self.update_graph_toolbar_enabled()
        visible_curves = self.visible_curves()

        mode = self.plot_mode.currentText()
        if self.auto_limits.isChecked():
            self.update_limit_fields_from_current_data()

        used_axes = {normalize_plot_axis(curve.get("axis", "left")) for curve in visible_curves.values()}
        axis_map = {"left": ax}
        if "right" in used_axes:
            axis_map["right"] = ax.twinx()
        if "left2" in used_axes:
            axis_map["left2"] = ax.twinx()
            axis_map["left2"].spines["left"].set_position(("axes", -0.12))
            axis_map["left2"].spines["left"].set_visible(True)
            axis_map["left2"].spines["right"].set_visible(False)
            axis_map["left2"].yaxis.set_label_position("left")
            axis_map["left2"].yaxis.tick_left()
        if "right2" in used_axes:
            axis_map["right2"] = ax.twinx()
            axis_map["right2"].spines["right"].set_position(("axes", 1.12))
        self.extra_axes = {name: axis for name, axis in axis_map.items() if name != "left"}

        for key, curve in visible_curves.items():
            target_ax = axis_map.get(normalize_plot_axis(curve.get("axis", "left")), ax)
            x = self.make_plot_x(curve["x"])
            y = self.make_plot_y(curve["x"], curve["y"])
            self.plot_curve_segments(target_ax, key, curve, x, y, mode)

        self.draw_guide_bars(ax)
        self.draw_peak_labels(ax)

        if mode == "linear linear" or mode == "qI(q)":
            ax.set_xscale("linear")
            for target_ax in axis_map.values():
                target_ax.set_yscale("linear")
        elif mode == "linear log":
            ax.set_xscale("linear")
            for target_ax in axis_map.values():
                target_ax.set_yscale("log")
        elif mode == "log linear":
            ax.set_xscale("log")
            for target_ax in axis_map.values():
                target_ax.set_yscale("linear")
        elif mode in PLOT_LOG_LOG_MODES:
            ax.set_xscale("log")
            for target_ax in axis_map.values():
                target_ax.set_yscale("log")

        has_azim_profile = self.curves_are_really_0_to_360()

        if has_azim_profile:
            default_x_label = "ψ / °"
            ax.set_xlim(0, 360)
            ax.set_xlabel(default_x_label)
        else:
            transform = PLOT_TRANSFORMED_MODES.get(mode)
            curve_x_labels = [
                str(curve.get("x_label", "")).strip()
                for curve in visible_curves.values()
                if str(curve.get("x_label", "")).strip()
            ]
            default_x_label = (
                transform["x_label"]
                if transform is not None and transform["x_power"] != 1
                else (curve_x_labels[0] if curve_x_labels else self.q_axis_label())
            )
            ax.set_xlabel(self.x_label.text() or default_x_label)
            if transform is not None and transform["x_power"] != 1 and self.x_label.text() in ("", self.q_axis_label()):
                ax.set_xlabel(default_x_label)
            elif curve_x_labels and (not self.x_label.text() or self.x_label.text() == self.q_axis_label()):
                ax.set_xlabel(default_x_label)
        axis_curve_labels = {}
        for axis_name in PLOT_Y_AXES:
            labels = [
                str(curve.get("y_label", "")).strip()
                for curve in visible_curves.values()
                if normalize_plot_axis(curve.get("axis", "left")) == axis_name and str(curve.get("y_label", "")).strip()
            ]
            if labels:
                axis_curve_labels[axis_name] = labels[0]
        transformed_y_label = PLOT_TRANSFORMED_MODES.get(mode, {}).get("y_label")
        axis_labels = {
            "left": axis_curve_labels.get("left") or transformed_y_label or (self.y_label.text() or "Intensity / a.u."),
            "left2": axis_curve_labels.get("left2") or transformed_y_label or "Left 2 / a.u.",
            "right": axis_curve_labels.get("right") or transformed_y_label or "Right / a.u.",
            "right2": axis_curve_labels.get("right2") or transformed_y_label or "Right 2 / a.u.",
        }
        for axis_name, target_ax in axis_map.items():
            target_ax.set_ylabel(axis_labels[axis_name])
        ax.set_title(self.title_edit.text())
        for axis_name, target_ax in axis_map.items():
            apply_plot_display_style(target_ax)
            if axis_name != "left":
                target_ax.grid(False)
        if self.show_legend.isChecked() and visible_curves:
            if len(axis_map) == 1:
                make_plot_legend(ax)
            else:
                handles = []
                labels = []
                for target_ax in axis_map.values():
                    axis_handles, axis_labels_for_legend = target_ax.get_legend_handles_labels()
                    handles.extend(axis_handles)
                    labels.extend(axis_labels_for_legend)
                install_selectable_legend(ax, ax.legend(handles, labels, loc="best", frameon=True))

        if not self.auto_limits.isChecked():
            if self.x_max.value() > self.x_min.value():
                ax.set_xlim(self.x_min.value(), self.x_max.value())
            if self.y_max.value() > self.y_min.value():
                ax.set_ylim(self.y_min.value(), self.y_max.value())

        if keep_zoom and previous_xlim is not None and previous_ylim is not None:
            if np.isfinite(previous_xlim).all() and np.isfinite(previous_ylim).all():
                ax.set_xlim(previous_xlim)
                ax.set_ylim(previous_ylim)

        if "left2" in axis_map:
            self.canvas.fig.subplots_adjust(left=0.20)
        if "right2" in axis_map:
            self.canvas.fig.subplots_adjust(right=0.84)
        if "left2" not in axis_map and "right2" not in axis_map:
            self.canvas.fig.subplots_adjust(left=0.12, right=0.98)

        finalize_plot_canvas(self.canvas)
