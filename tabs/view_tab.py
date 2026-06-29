import fnmatch
import json
import re
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
import matplotlib.pyplot as plt

from PySide6.QtCore import Qt, QEvent, QSettings, Signal, QSize, QTimer

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMessageBox

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QFormLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QListWidget,
    QFileDialog,
    QTextEdit,
    QSlider,
    QCheckBox,
    QMessageBox,
    QGroupBox,
    QSpinBox,
    QDialog,
    QDialogButtonBox,
    QSizePolicy,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QComboBox,
    QAbstractItemView,
    QListWidgetItem,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from .instrument_presets import (
    ID13_DEFAULT_CENTER_X,
    ID13_DEFAULT_CENTER_Y,
    ID13_DEFAULT_DISTANCE_M,
    ID13_DEFAULT_PIXEL_MM,
    ID13_DEFAULT_WAVELENGTH_A,
)
from .file_ratings import install_file_rating_menu, is_file_rated_up, set_item_file_path, should_hide_file_in_browser
from .line_geometry import LineGeometrySelector, line_geometry_to_lrphoton, parse_header_text
from .ui_style import (
    BLOCK_SPACING,
    FILE_BROWSER_WIDTH,
    FlexibleDoubleSpinBox as QDoubleSpinBox,
    FRAME_BUTTON_WIDTH,
    FRAME_COUNTER_WIDTH,
    FRAME_NAV_SPACING,
    FRAME_SPIN_WIDTH,
    GROUP_BOX_MARGINS,
    MATPLOTLIB_TOOLBAR_ICON_SCALE,
    MATPLOTLIB_TOOLBAR_MAX_HEIGHT,
    GROUP_BOX_STYLE,
    PAGE_MARGINS,
    PANEL_MARGINS,
    TOOL_GROUP_BOX_STYLE,
    constrain_image_axes,
    make_matplotlib_toolbar_block,
    set_matplotlib_toolbar_enabled,
    set_widget_enabled_with_opacity,
    style_q_geometry_buttons,
)


ANNOTATIONS_FILE = Path.home() / ".lrphoton" / "annotations.json"


class ImageOnlyToolbar(NavigationToolbar):

    def __init__(self, canvas, parent):
        super().__init__(canvas, parent)
        self.view_tab = parent
        save_image_action = QAction("💾", self)
        save_image_action.setToolTip("Save image only")
        save_image_action.triggered.connect(parent.save_png_image_only)


    def save_figure(self, *args):
        view_tab = getattr(self, "view_tab", None)
        if hasattr(view_tab, "save_png_image_only"):
            view_tab.save_png_image_only()
        else:
            super().save_figure(*args)


    def home(self, *args):
        view_tab = getattr(self, "view_tab", None)
        if hasattr(view_tab, "reset_image_view"):
            view_tab.reset_image_view()
            self.push_current()
            self.set_history_buttons()
        else:
            super().home(*args)


class ViewImageCanvas(FigureCanvas):
    def __init__(self, figure, view_tab):
        self.view_tab = view_tab
        super().__init__(figure)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setStyleSheet("background: transparent;")
        self.setFocusPolicy(Qt.StrongFocus)

        try:
            self.grabGesture(Qt.PinchGesture)
        except Exception:
            pass

    def event(self, event):
        view_tab = getattr(self, "view_tab", None)
        if view_tab is not None and view_tab.image_artist is not None:
            try:
                if event.type() == QEvent.NativeGesture:
                    gesture_type = event.gestureType()
                    value = event.value()
                    if gesture_type == Qt.ZoomNativeGesture and value != 0:
                        scale = 1.0 / (1.0 + value) if value > -0.95 else 1.25
                        view_tab.zoom_at_qpoint(self._event_center_point(event), scale)
                        event.accept()
                        return True

                    if gesture_type == Qt.SmartZoomNativeGesture:
                        view_tab.reset_image_view()
                        event.accept()
                        return True

                if event.type() == QEvent.Gesture:
                    pinch = event.gesture(Qt.PinchGesture)
                    if pinch is not None:
                        factor = pinch.scaleFactor()
                        if factor and factor > 0:
                            view_tab.zoom_at_qpoint(self._event_center_point(event), 1.0 / factor)
                            event.accept()
                            return True
            except Exception:
                pass

        return super().event(event)

    def wheelEvent(self, event):
        view_tab = getattr(self, "view_tab", None)
        if view_tab is None or view_tab.image_artist is None:
            return super().wheelEvent(event)

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
                view_tab.zoom_at_qpoint(event.position(), scale)
        else:
            view_tab.pan_by_trackpad(dx, dy)

        event.accept()

    def _event_center_point(self, event):
        try:
            position = event.position()
            if position is not None:
                return position
        except Exception:
            pass

        return self.rect().center()


class PlaneAnnotationCanvas(FigureCanvas):
    def __init__(self, dialog):
        self.dialog = dialog
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setFocusPolicy(Qt.StrongFocus)
        self._trackpad_view_pushed = False
        self._drag_label = None
        self._toolbar_mode_before_label_drag = None
        self.mpl_connect("motion_notify_event", self.on_motion)
        self.mpl_connect("figure_leave_event", self.on_leave)
        self.mpl_connect("button_press_event", self.on_press)
        self.mpl_connect("button_release_event", self.on_release)
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
                    self.reset_view()
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

    def qpoint_to_data_pos(self, qpoint):
        width = max(1, self.width())
        height = max(1, self.height())
        x_fig = float(qpoint.x()) / width
        y_fig = 1.0 - float(qpoint.y()) / height
        xdata, ydata = self.ax.transData.inverted().transform(self.fig.transFigure.transform((x_fig, y_fig)))
        if not np.isfinite(xdata) or not np.isfinite(ydata):
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            xdata = (xlim[0] + xlim[1]) / 2.0
            ydata = (ylim[0] + ylim[1]) / 2.0
        return xdata, ydata

    def zoom_at_qpoint(self, qpoint, zoom_factor):
        if zoom_factor <= 0:
            return
        xdata, ydata = self.qpoint_to_data_pos(qpoint)
        self.zoom_at_data_position(xdata, ydata, zoom_factor)

    def zoom_at_data_position(self, xdata, ydata, zoom_factor):
        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()
        if x_max == x_min or y_max == y_min:
            return

        new_width = (x_max - x_min) * zoom_factor
        new_height = (y_max - y_min) * zoom_factor
        rel_x = (xdata - x_min) / (x_max - x_min)
        rel_y = (ydata - y_min) / (y_max - y_min)

        self.ax.set_xlim(xdata - new_width * rel_x, xdata + new_width * (1.0 - rel_x))
        self.ax.set_ylim(ydata - new_height * rel_y, ydata + new_height * (1.0 - rel_y))
        constrain_image_axes(self.ax, self.dialog.raw_image.shape)
        self.ax.set_autoscale_on(False)
        self.push_toolbar_view_once()
        self.draw_idle()

    def pan_by_trackpad(self, dx, dy):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        xspan = xlim[1] - xlim[0]
        yspan = ylim[1] - ylim[0]
        shift_x = -dx * xspan * 0.08
        shift_y = dy * yspan * 0.08

        self.ax.set_xlim(xlim[0] + shift_x, xlim[1] + shift_x)
        self.ax.set_ylim(ylim[0] + shift_y, ylim[1] + shift_y)
        constrain_image_axes(self.ax, self.dialog.raw_image.shape)
        self.ax.set_autoscale_on(False)
        self.push_toolbar_view_once()
        self.draw_idle()

    def reset_view(self):
        ny, nx = self.dialog.raw_image.shape
        self.ax.set_xlim(-0.5, nx - 0.5)
        self.ax.set_ylim(ny - 0.5, -0.5)
        self.ax.set_autoscale_on(False)
        self._trackpad_view_pushed = False
        toolbar = getattr(self.dialog, "toolbar", None)
        if toolbar is not None:
            toolbar.set_history_buttons()
        self.draw_idle()

    def push_toolbar_view_once(self):
        toolbar = getattr(self.dialog, "toolbar", None)
        if toolbar is not None and not self._trackpad_view_pushed:
            toolbar.push_current()
            self._trackpad_view_pushed = True
        if toolbar is not None:
            toolbar.set_history_buttons()

    def show_image(self):
        had_image = bool(self.ax.images)
        current_xlim = self.ax.get_xlim()
        current_ylim = self.ax.get_ylim()
        self.ax.clear()
        self.ax.set_axis_off()
        self.ax.imshow(
            self.dialog.display_image,
            origin="upper",
            cmap="jet",
            interpolation="nearest",
            vmin=self.dialog.vmin,
            vmax=self.dialog.vmax,
        )
        self.draw_annotations()
        self.ax.set_aspect("equal")
        if had_image:
            self.ax.set_xlim(current_xlim)
            self.ax.set_ylim(current_ylim)
            constrain_image_axes(self.ax, self.dialog.raw_image.shape)
        self.draw_idle()

    def draw_annotations(self):
        color = "#ffffff"
        for label, data in self.dialog.annotations.items():
            points = data.get("points", [])
            label_pos = data.get("label_pos")
            if label_pos is None and points:
                x, y = points[0]
                label_pos = (x + 35.0, y - 35.0)
                data["label_pos"] = label_pos

            if label_pos is None:
                continue

            for point in points:
                self.ax.annotate(
                    "",
                    xy=point,
                    xytext=label_pos,
                    arrowprops=dict(arrowstyle="->", color=color, linewidth=1.2),
                )

            self.ax.text(
                label_pos[0],
                label_pos[1],
                label,
                color="black",
                fontsize=10,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.25", facecolor=color, edgecolor="black", alpha=0.92),
            )

    def data_point_from_event(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return None
        x = float(event.xdata)
        y = float(event.ydata)
        ny, nx = self.dialog.raw_image.shape
        if not (-0.5 <= x <= nx - 0.5 and -0.5 <= y <= ny - 0.5):
            return None
        return x, y

    def label_hit(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return None
        event_xy = np.array([event.x, event.y], dtype=float)
        for label, data in reversed(list(self.dialog.annotations.items())):
            label_pos = data.get("label_pos")
            if label_pos is None:
                continue
            display_xy = np.asarray(self.ax.transData.transform(label_pos), dtype=float)
            if np.linalg.norm(event_xy - display_xy) <= 22.0:
                return label
        return None

    def on_press(self, event):
        if event.button != 1:
            return
        hit = self.label_hit(event)
        if hit is not None:
            self.dialog.disable_toolbar_mode()
            self._drag_label = hit
            return

        point = self.data_point_from_event(event)
        if point is not None:
            self.dialog.add_point(point)

    def on_motion(self, event):
        if self._drag_label is not None:
            point = self.data_point_from_event(event)
            if point is not None:
                self.dialog.annotations[self._drag_label]["label_pos"] = point
                self.show_image()
            return

        point = self.data_point_from_event(event)
        if point is None:
            self.dialog.coordinate_label.setText("x = - | y = - | q = - | I = -")
            return
        self.dialog.update_coordinate_label(point)

    def on_release(self, event):
        if self._drag_label is not None:
            self.dialog.save_annotations()
        self._drag_label = None
        self._trackpad_view_pushed = False

    def on_leave(self, event):
        self.dialog.coordinate_label.setText("x = - | y = - | q = - | I = -")


class PlaneAnnotationDialog(QDialog):
    def __init__(self, parent, raw_image, display_image, vmin, vmax, q_calculator, title, annotation_key):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1220, 760)
        self.raw_image = np.asarray(raw_image, dtype=float)
        self.display_image = np.asarray(display_image, dtype=float)
        self.vmin = vmin
        self.vmax = vmax
        self.q_calculator = q_calculator
        self.annotation_key = annotation_key
        self.annotations = {}
        self.current_plane = "plane 1"
        self._syncing_tree = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        image_panel = QWidget()
        image_layout = QVBoxLayout(image_panel)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.setSpacing(6)

        self.canvas = PlaneAnnotationCanvas(self)
        self.toolbar = NavigationToolbar(self.canvas, self)
        toolbar_box, _, self.save_button = make_matplotlib_toolbar_block(
            self,
            "Annotation tools",
            self.toolbar,
            save_callback=self.save_png,
            save_tooltip="Save annotated PNG",
            toolbar_width=340,
            remove_customize=True,
        )
        image_layout.addWidget(toolbar_box, 0)
        image_layout.addWidget(self.canvas, 1)

        self.coordinate_label = QLabel("x = - | y = - | q = - | I = -")
        self.coordinate_label.setMinimumHeight(28)
        self.coordinate_label.setAlignment(Qt.AlignCenter)
        self.coordinate_label.setStyleSheet("""
            QLabel {
                background-color: #f4f4f4;
                border-radius: 8px;
                padding: 6px;
                font-family: Menlo, Monaco, monospace;
                font-size: 11px;
            }
        """)
        image_layout.addWidget(self.coordinate_label, 0)

        side_panel = QGroupBox("Planes")
        side_panel.setStyleSheet(GROUP_BOX_STYLE)
        side_panel.setFixedWidth(360)
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(*GROUP_BOX_MARGINS)
        side_layout.setSpacing(6)

        self.label_edit = QLineEdit(self.current_plane)
        self.add_plane_button = QPushButton("Add plane")
        side_layout.addWidget(QLabel("Current plane"))
        side_layout.addWidget(self.label_edit)
        side_layout.addWidget(self.add_plane_button)

        self.plane_tree = QTreeWidget()
        self.plane_tree.setColumnCount(2)
        self.plane_tree.setHeaderHidden(True)
        self.plane_tree.setMinimumHeight(240)
        self.plane_tree.setSelectionMode(QTreeWidget.SingleSelection)
        self.plane_tree.setRootIsDecorated(True)
        self.plane_tree.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.plane_tree.setColumnWidth(0, 302)
        self.plane_tree.setColumnWidth(1, 28)
        side_layout.addWidget(self.plane_tree, 1)

        layout.addWidget(image_panel, 1)
        layout.addWidget(side_panel, 0)

        self.add_plane_button.clicked.connect(self.add_plane)
        self.label_edit.editingFinished.connect(self.rename_current_plane)
        self.plane_tree.currentItemChanged.connect(self.tree_selection_changed)
        self.plane_tree.itemClicked.connect(self.tree_item_clicked)
        self.load_annotations()
        self.ensure_plane(self.current_plane)
        self.refresh_plane_tree()
        self.canvas.show_image()

    def closeEvent(self, event):
        self.save_annotations()
        super().closeEvent(event)

    def disable_toolbar_mode(self):
        toolbar = getattr(self, "toolbar", None)
        if toolbar is None:
            return

        mode = getattr(toolbar, "mode", "")
        mode_text = str(mode).lower()
        try:
            mode_name = mode.name.lower()
        except Exception:
            mode_name = mode_text

        if "pan" in mode_text or "pan" in mode_name:
            toolbar.pan()
        elif "zoom" in mode_text or "zoom" in mode_name:
            toolbar.zoom()

    def normalized_annotations(self):
        normalized = {}
        for label, data in self.annotations.items():
            points = [
                [float(point[0]), float(point[1])]
                for point in data.get("points", [])
            ]
            label_pos = data.get("label_pos")
            normalized[label] = {
                "points": points,
                "label_pos": None if label_pos is None else [float(label_pos[0]), float(label_pos[1])],
            }
        return normalized

    def load_annotations(self):
        if not self.annotation_key or not ANNOTATIONS_FILE.exists():
            return

        try:
            data = json.loads(ANNOTATIONS_FILE.read_text(encoding="utf-8"))
            saved = data.get(self.annotation_key, {})
        except Exception:
            return

        annotations = saved.get("annotations", {})
        loaded = {}
        for label, annotation in annotations.items():
            points = []
            for point in annotation.get("points", []):
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    points.append((float(point[0]), float(point[1])))
            label_pos = annotation.get("label_pos")
            if isinstance(label_pos, (list, tuple)) and len(label_pos) >= 2:
                label_pos = (float(label_pos[0]), float(label_pos[1]))
            else:
                label_pos = None
            loaded[str(label)] = {"points": points, "label_pos": label_pos}

        if loaded:
            self.annotations = loaded
            self.current_plane = str(saved.get("current_plane") or next(iter(loaded)))
            if self.current_plane not in self.annotations:
                self.current_plane = next(iter(loaded))
            self.label_edit.setText(self.current_plane)

    def save_annotations(self):
        if not self.annotation_key:
            return

        try:
            ANNOTATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
            if ANNOTATIONS_FILE.exists():
                data = json.loads(ANNOTATIONS_FILE.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    data = {}
            else:
                data = {}
            data[self.annotation_key] = {
                "current_plane": self.current_plane,
                "annotations": self.normalized_annotations(),
            }
            ANNOTATIONS_FILE.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        except Exception:
            pass

    def current_label(self):
        return self.current_plane or "plane"

    def ensure_plane(self, label):
        label = label.strip() or "plane"
        return self.annotations.setdefault(label, {"points": [], "label_pos": None})

    def next_plane_name(self):
        index = 1
        while f"plane {index}" in self.annotations:
            index += 1
        return f"plane {index}"

    def add_plane(self):
        self.current_plane = self.next_plane_name()
        self.ensure_plane(self.current_plane)
        self.label_edit.setText(self.current_plane)
        self.refresh_plane_tree()
        self.save_annotations()
        self.canvas.show_image()

    def rename_current_plane(self):
        old_label = self.current_plane
        new_label = self.label_edit.text().strip() or old_label or "plane"
        if new_label == old_label:
            self.label_edit.setText(self.current_plane)
            return

        data = self.annotations.pop(old_label, {"points": [], "label_pos": None})
        if new_label in self.annotations:
            new_label = self.next_plane_name()
        self.annotations[new_label] = data
        self.current_plane = new_label
        self.label_edit.setText(new_label)
        self.refresh_plane_tree()
        self.save_annotations()
        self.canvas.show_image()

    def make_remove_button(self, tooltip, callback):
        remove_button = QPushButton("−")
        remove_button.setFixedSize(22, 18)
        remove_button.setToolTip(tooltip)
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
        remove_button.clicked.connect(callback)

        remove_holder = QWidget()
        remove_layout = QHBoxLayout(remove_holder)
        remove_layout.setContentsMargins(0, 0, 0, 0)
        remove_layout.setSpacing(0)
        remove_layout.addWidget(remove_button, alignment=Qt.AlignCenter)
        return remove_holder

    def refresh_plane_tree(self):
        self._syncing_tree = True
        self.plane_tree.clear()
        current_item = None
        for label, data in self.annotations.items():
            points = data.get("points", [])
            plane_item = QTreeWidgetItem([f"{label} ({len(points)} point{'s' if len(points) != 1 else ''})", ""])
            plane_item.setData(0, Qt.UserRole, ("plane", label, None))
            plane_item.setSizeHint(0, QSize(302, 24))
            plane_item.setSizeHint(1, QSize(28, 24))
            plane_item.setTextAlignment(1, Qt.AlignCenter)
            plane_item.setToolTip(0, plane_item.text(0))
            self.plane_tree.addTopLevelItem(plane_item)
            self.plane_tree.setItemWidget(
                plane_item,
                1,
                self.make_remove_button(
                    "Remove this plane",
                    lambda checked=False, plane_label=label: self.remove_plane(plane_label),
                ),
            )
            if label == self.current_plane:
                current_item = plane_item

            for index, point in enumerate(points, start=1):
                x, y = point
                point_text = f"Point {index}: x={x + 1:.1f}, y={y + 1:.1f}"
                point_item = QTreeWidgetItem([point_text, ""])
                point_item.setData(0, Qt.UserRole, ("point", label, index - 1))
                point_item.setSizeHint(0, QSize(302, 24))
                point_item.setSizeHint(1, QSize(28, 24))
                point_item.setTextAlignment(1, Qt.AlignCenter)
                point_item.setToolTip(0, point_text)
                plane_item.addChild(point_item)
                self.plane_tree.setItemWidget(
                    point_item,
                    1,
                    self.make_remove_button(
                        "Remove this point",
                        lambda checked=False, point_label=label, point_index=index - 1: self.remove_point(
                            point_label,
                            point_index,
                        ),
                    ),
                )
            plane_item.setExpanded(True)

        if current_item is not None:
            self.plane_tree.setCurrentItem(current_item)
        self._syncing_tree = False

        self.plane_tree.setStyleSheet("""
            QTreeWidget::item {
                min-height: 22px;
            }
            QTreeWidget::item:selected {
                background-color: #0a64d8;
                color: white;
            }
        """)

    def tree_item_clicked(self, item, column):
        if item is None or column != 1:
            return
        data = item.data(0, Qt.UserRole)
        if not data:
            return
        kind, label, index = data
        if kind == "plane":
            self.remove_plane(label)
        elif kind == "point":
            self.remove_point(label, index)

    def remove_plane(self, label):
        if label not in self.annotations:
            return

        del self.annotations[label]
        if self.current_plane == label:
            self.current_plane = next(iter(self.annotations), "plane 1")
            self.ensure_plane(self.current_plane)
            self.label_edit.setText(self.current_plane)
        self.refresh_plane_tree()
        self.save_annotations()
        self.canvas.show_image()

    def remove_point(self, label, index):
        data = self.annotations.get(label)
        if data is None:
            return

        points = data.get("points", [])
        if not (0 <= index < len(points)):
            return

        points.pop(index)
        if not points:
            data["label_pos"] = None
        self.current_plane = label
        self.label_edit.setText(label)
        self.refresh_plane_tree()
        self.save_annotations()
        self.canvas.show_image()

    def tree_selection_changed(self, current, previous):
        if self._syncing_tree or current is None:
            return
        data = current.data(0, Qt.UserRole)
        if not data:
            return
        _kind, label, _index = data
        self.current_plane = label
        self.label_edit.setText(label)

    def add_point(self, point):
        label = self.current_label()
        data = self.ensure_plane(label)
        data["points"].append((float(point[0]), float(point[1])))
        if data["label_pos"] is None:
            data["label_pos"] = (float(point[0]) + 45.0, float(point[1]) - 35.0)
        self.refresh_plane_tree()
        self.save_annotations()
        self.canvas.show_image()
        self.update_coordinate_label(point)

    def undo_last_point(self):
        label = self.current_label()
        data = self.annotations.get(label)
        if data is None or not data.get("points"):
            return
        data["points"].pop()
        self.refresh_plane_tree()
        self.save_annotations()
        self.canvas.show_image()

    def clear_annotations(self):
        self.annotations = {}
        self.current_plane = "plane 1"
        self.label_edit.setText(self.current_plane)
        self.ensure_plane(self.current_plane)
        self.refresh_plane_tree()
        self.save_annotations()
        self.canvas.show_image()

    def save_png(self):
        parent = self.parent()
        default_name = "annotated_image.png"
        default_folder = Path.home()
        current_file = getattr(parent, "current_file", None)
        if current_file is not None:
            default_folder = current_file.parent
            default_name = f"{current_file.stem}_annotated.png"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save annotated image",
            str(default_folder / default_name),
            "PNG image (*.png)",
        )
        if not file_path:
            return
        if not file_path.lower().endswith(".png"):
            file_path += ".png"

        try:
            self.canvas.draw()
            self.canvas.fig.savefig(
                file_path,
                dpi=300,
                bbox_inches="tight",
                pad_inches=0.02,
                facecolor="white",
            )
        except Exception as error:
            QMessageBox.warning(self, "Save error", f"Unable to save annotated image:\n{error}")

    def update_coordinate_label(self, point):
        x_index = int(round(point[0]))
        y_index = int(round(point[1]))
        ny, nx = self.raw_image.shape
        if not (0 <= x_index < nx and 0 <= y_index < ny):
            self.coordinate_label.setText("x = - | y = - | q = - | I = -")
            return

        value = self.raw_image[y_index, x_index]
        if np.isnan(value):
            value_text = "NaN"
        elif np.isposinf(value):
            value_text = "+Inf"
        elif np.isneginf(value):
            value_text = "-Inf"
        else:
            value_text = f"{value:.8g}"

        q_value = self.q_calculator(x_index, y_index) if self.q_calculator is not None else None
        q_text = "-" if q_value is None else f"{q_value:.6g} nm⁻¹"
        self.coordinate_label.setText(
            f"x = {x_index + 1} | y = {y_index + 1} | q = {q_text} | I = {value_text}"
        )

class CompositeImageDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.view_tab = parent
        self.settings = parent.settings
        self.entries = []
        self.multi_h5_path = None
        self.multi_dataset_name = None
        self.multi_shape = None
        self.multi_frame_axis = None
        self.multi_frame_count = 1
        self.preview_images = []
        self.preview_titles = []

        self.setWindowTitle("Composite image")
        self.resize(1100, 820)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        source_box = QGroupBox("Images")
        source_layout = QVBoxLayout(source_box)
        source_layout.setContentsMargins(10, 18, 10, 10)
        source_layout.setSpacing(6)

        browser_folder_layout = QHBoxLayout()
        self.composite_folder_path = QLineEdit(str(self.view_tab.current_folder))
        self.composite_folder_path.returnPressed.connect(self.refresh_source_files)
        self.composite_browse_button = QPushButton("Browse")
        self.composite_browse_button.clicked.connect(self.choose_source_folder)
        browser_folder_layout.addWidget(self.composite_folder_path, 1)
        browser_folder_layout.addWidget(self.composite_browse_button)
        source_layout.addLayout(browser_folder_layout)

        browser_filters_layout = QHBoxLayout()
        self.composite_name_filter = QLineEdit()
        self.composite_name_filter.setPlaceholderText("Name filter")
        self.composite_name_filter.textChanged.connect(self.refresh_source_files)
        self.composite_extension_filter = QLineEdit("*.h5 *.hdf5")
        self.composite_extension_filter.setPlaceholderText("Extensions")
        self.composite_extension_filter.textChanged.connect(self.refresh_source_files)
        self.composite_show_subfolders_checkbox = QCheckBox("Show subfolders")
        self.composite_show_subfolders_checkbox.stateChanged.connect(self.refresh_source_files)
        self.composite_refresh_button = QPushButton("Refresh")
        self.composite_refresh_button.clicked.connect(self.refresh_source_files)
        browser_filters_layout.addWidget(self.composite_name_filter)
        browser_filters_layout.addWidget(self.composite_extension_filter)
        browser_filters_layout.addWidget(self.composite_show_subfolders_checkbox)
        browser_filters_layout.addWidget(self.composite_refresh_button)
        source_layout.addLayout(browser_filters_layout)

        self.browser_file_list = QListWidget()
        self.browser_file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.browser_file_list.setMinimumHeight(150)
        self.browser_file_list.itemDoubleClicked.connect(lambda _item: self.add_selected_h5_files())
        source_layout.addWidget(self.browser_file_list)

        source_buttons = QHBoxLayout()
        self.add_h5_button = QPushButton("Add selected H5")
        self.add_h5_button.clicked.connect(self.add_selected_h5_files)
        self.choose_multi_h5_button = QPushButton("Use selected as multi H5")
        self.choose_multi_h5_button.clicked.connect(self.choose_selected_multi_h5)
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_entries)
        source_buttons.addWidget(self.add_h5_button)
        source_buttons.addWidget(self.choose_multi_h5_button)
        source_buttons.addWidget(self.clear_button)
        source_layout.addLayout(source_buttons)

        self.file_list = QListWidget()
        self.file_list.setMinimumHeight(90)
        source_layout.addWidget(self.file_list)

        range_layout = QHBoxLayout()
        self.start_spin = QSpinBox()
        self.start_spin.setRange(1, 1)
        self.start_spin.setValue(1)
        self.end_spin = QSpinBox()
        self.end_spin.setRange(1, 1)
        self.end_spin.setValue(1)
        self.step_spin = QSpinBox()
        self.step_spin.setRange(1, 9999)
        self.step_spin.setValue(1)
        self.build_range_button = QPushButton("Build range")
        self.build_range_button.clicked.connect(self.build_multi_h5_range)
        for label, widget in [
            ("Start", self.start_spin),
            ("End", self.end_spin),
            ("Every", self.step_spin),
        ]:
            range_layout.addWidget(QLabel(label))
            range_layout.addWidget(widget)
        range_layout.addWidget(self.build_range_button)
        range_layout.addStretch(1)
        source_layout.addLayout(range_layout)
        layout.addWidget(source_box)

        controls_box = QGroupBox("Composite settings")
        controls_layout = QGridLayout(controls_box)
        controls_layout.setContentsMargins(10, 18, 10, 10)
        controls_layout.setHorizontalSpacing(8)
        controls_layout.setVerticalSpacing(6)

        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(1, 20)
        self.rows_spin.setValue(self.settings.value("view/composite_rows", 2, type=int))
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(1, 20)
        self.cols_spin.setValue(self.settings.value("view/composite_cols", 3, type=int))

        self.cmap_combo = QComboBox()
        for label, cmap_name in getattr(parent, "colormap_options", [("Jet", "jet")]):
            self.cmap_combo.addItem(label, cmap_name)
        saved_cmap = self.settings.value("view/composite_colormap", parent.current_colormap(), type=str)
        saved_cmap_index = self.cmap_combo.findData(saved_cmap)
        self.cmap_combo.setCurrentIndex(max(saved_cmap_index, 0))

        self.log_checkbox = QCheckBox("Log")
        self.log_checkbox.setChecked(self.settings.value("view/composite_log", True, type=bool))
        self.nan_to_zero_checkbox = QCheckBox("NaN=0")
        self.nan_to_zero_checkbox.setChecked(self.settings.value("view/composite_nan_to_zero", True, type=bool))

        self.vmin_spin = QDoubleSpinBox()
        self.vmin_spin.setDecimals(4)
        self.vmin_spin.setRange(-1e12, 1e12)
        self.vmin_spin.setValue(self.settings.value("view/composite_vmin", 0.0, type=float))
        self.vmax_spin = QDoubleSpinBox()
        self.vmax_spin.setDecimals(4)
        self.vmax_spin.setRange(-1e12, 1e12)
        self.vmax_spin.setValue(self.settings.value("view/composite_vmax", 5.0, type=float))

        self.auto_button = QPushButton("Auto intensity")
        self.auto_button.clicked.connect(self.auto_intensity)
        self.update_preview_button = QPushButton("Update preview")
        self.update_preview_button.clicked.connect(self.update_preview)
        self.save_button = QPushButton("💾 Save composite")
        self.save_button.clicked.connect(self.save_composite)

        controls_layout.addWidget(QLabel("Rows"), 0, 0)
        controls_layout.addWidget(self.rows_spin, 0, 1)
        controls_layout.addWidget(QLabel("Columns"), 0, 2)
        controls_layout.addWidget(self.cols_spin, 0, 3)
        controls_layout.addWidget(QLabel("Colormap"), 0, 4)
        controls_layout.addWidget(self.cmap_combo, 0, 5)
        controls_layout.addWidget(self.log_checkbox, 1, 0)
        controls_layout.addWidget(self.nan_to_zero_checkbox, 1, 1)
        controls_layout.addWidget(QLabel("Min"), 1, 2)
        controls_layout.addWidget(self.vmin_spin, 1, 3)
        controls_layout.addWidget(QLabel("Max"), 1, 4)
        controls_layout.addWidget(self.vmax_spin, 1, 5)
        controls_layout.addWidget(self.auto_button, 2, 0, 1, 2)
        controls_layout.addWidget(self.update_preview_button, 2, 2, 1, 2)
        controls_layout.addWidget(self.save_button, 2, 4, 1, 2)
        layout.addWidget(controls_box)

        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas, 1)

        self.status_label = QLabel("Choose separate H5 files or a range from one multi-frame H5.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        for widget in [
            self.rows_spin,
            self.cols_spin,
            self.cmap_combo,
            self.log_checkbox,
            self.nan_to_zero_checkbox,
            self.vmin_spin,
            self.vmax_spin,
        ]:
            if hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self.update_preview)
            elif hasattr(widget, "currentIndexChanged"):
                widget.currentIndexChanged.connect(self.update_preview)
            elif hasattr(widget, "stateChanged"):
                widget.stateChanged.connect(self.update_preview)

        self.refresh_source_files()
        self.update_preview()

    def preferred_h5_dataset(self, path):
        candidates = []
        with h5py.File(path, "r") as handle:
            def visitor(name, obj):
                if not isinstance(obj, h5py.Dataset) or obj.ndim not in (2, 3):
                    return
                if not np.issubdtype(obj.dtype, np.number):
                    return
                frame_axis, n_frames, image_shape = self.view_tab.h5_dataset_image_info(obj.shape)
                score = self.view_tab.h5_dataset_image_score(name, obj, frame_axis, n_frames, image_shape)
                candidates.append((score, name, tuple(obj.shape), frame_axis, n_frames, image_shape))

            handle.visititems(visitor)

        if not candidates:
            raise ValueError("No 2D or 3D image dataset found.")
        return max(candidates, key=lambda item: item[0])

    def read_h5_frame(self, path, dataset_name, frame_axis, frame_index):
        with h5py.File(path, "r") as handle:
            dataset = handle[dataset_name]
            if dataset.ndim == 2:
                return np.asarray(dataset[:, :], dtype=float)
            if frame_axis == 0:
                return np.asarray(dataset[frame_index, :, :], dtype=float)
            if frame_axis == 1:
                return np.asarray(dataset[:, frame_index, :], dtype=float)
            return np.asarray(dataset[:, :, frame_index], dtype=float)

    def choose_source_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Choose folder",
            self.composite_folder_path.text() or str(Path.home()),
        )
        if folder:
            self.composite_folder_path.setText(folder)
            self.refresh_source_files()

    def extension_patterns(self):
        text = self.composite_extension_filter.text().strip()
        if not text:
            return ["*.h5", "*.hdf5"]
        return [part.strip() for part in re.split(r"[;,\s]+", text) if part.strip()]

    def refresh_source_files(self):
        if not hasattr(self, "browser_file_list"):
            return

        folder = Path(self.composite_folder_path.text()).expanduser()
        self.browser_file_list.clear()
        if not folder.exists():
            return

        name_filter = self.composite_name_filter.text().strip().lower()
        patterns = self.extension_patterns()
        iterator = folder.rglob("*") if self.composite_show_subfolders_checkbox.isChecked() else folder.glob("*")

        paths = []
        for path in iterator:
            if not path.is_file():
                continue
            if patterns and not any(fnmatch.fnmatch(path.name.lower(), pattern.lower()) for pattern in patterns):
                continue
            if name_filter and name_filter not in path.name.lower():
                continue
            paths.append(path)

        for path in sorted(paths, key=lambda item: str(item).lower()):
            display_name = str(path.relative_to(folder))
            item = QListWidgetItem(display_name)
            item.setData(Qt.UserRole, str(path))
            self.browser_file_list.addItem(item)

    def selected_browser_paths(self):
        paths = []
        for item in self.browser_file_list.selectedItems():
            stored_path = item.data(Qt.UserRole)
            if stored_path:
                paths.append(stored_path)
        return paths

    def add_selected_h5_files(self):
        paths = self.selected_browser_paths()
        if not paths:
            QMessageBox.information(self, "Composite H5", "Select one or more H5 files in the list first.")
            return

        errors = []
        for path in paths:
            try:
                _score, dataset_name, shape, frame_axis, _n_frames, _image_shape = self.preferred_h5_dataset(path)
                self.entries.append({
                    "path": path,
                    "dataset": dataset_name,
                    "frame_axis": frame_axis,
                    "frame": 0,
                    "title": Path(path).stem,
                })
                self.file_list.addItem(f"{Path(path).name}  {dataset_name}  {shape}")
            except Exception as error:
                errors.append(f"{Path(path).name}: {error}")

        if errors:
            QMessageBox.warning(self, "Composite H5", "\n".join(errors))
        self.load_preview_images()

    def choose_selected_multi_h5(self):
        paths = self.selected_browser_paths()
        if not paths:
            QMessageBox.information(self, "Composite H5", "Select one multi-frame H5 file in the list first.")
            return
        path = paths[0]

        try:
            _score, dataset_name, shape, frame_axis, n_frames, _image_shape = self.preferred_h5_dataset(path)
        except Exception as error:
            QMessageBox.warning(self, "Composite H5", f"Unable to read this H5 file:\n{error}")
            return

        self.multi_h5_path = path
        self.multi_dataset_name = dataset_name
        self.multi_shape = shape
        self.multi_frame_axis = frame_axis
        self.multi_frame_count = n_frames
        self.start_spin.setRange(1, n_frames)
        self.end_spin.setRange(1, n_frames)
        self.start_spin.setValue(1)
        self.end_spin.setValue(n_frames)
        self.status_label.setText(f"Multi H5 ready: {Path(path).name}, dataset {dataset_name}, {n_frames} frame(s).")

    def build_multi_h5_range(self):
        if not self.multi_h5_path:
            QMessageBox.information(self, "Composite H5", "Choose a multi H5 file first.")
            return

        start = min(self.start_spin.value(), self.end_spin.value())
        end = max(self.start_spin.value(), self.end_spin.value())
        step = max(1, self.step_spin.value())
        self.entries = []
        self.file_list.clear()

        for frame_number in range(start, end + 1, step):
            self.entries.append({
                "path": self.multi_h5_path,
                "dataset": self.multi_dataset_name,
                "frame_axis": self.multi_frame_axis,
                "frame": frame_number - 1,
                "title": f"{Path(self.multi_h5_path).stem} #{frame_number}",
            })
            self.file_list.addItem(
                f"{Path(self.multi_h5_path).name}  frame {frame_number}  {self.multi_dataset_name}  {self.multi_shape}"
            )

        self.load_preview_images()

    def clear_entries(self):
        self.entries = []
        self.preview_images = []
        self.preview_titles = []
        self.file_list.clear()
        self.update_preview()

    def transformed_image(self, image):
        display = np.asarray(image, dtype=float).copy()
        display[display > 4e9] = np.nan
        if self.log_checkbox.isChecked():
            display = np.log10(np.clip(display, 0, None) + 1)
        if self.nan_to_zero_checkbox.isChecked():
            display = np.nan_to_num(display, nan=0.0, copy=False)
        return display

    def load_preview_images(self):
        images = []
        titles = []
        errors = []
        max_images = self.rows_spin.value() * self.cols_spin.value()

        for entry in self.entries[:max_images]:
            try:
                images.append(self.read_h5_frame(entry["path"], entry["dataset"], entry["frame_axis"], entry["frame"]))
                titles.append(entry["title"])
            except Exception as error:
                errors.append(f"{entry['title']}: {error}")

        self.preview_images = images
        self.preview_titles = titles
        if errors:
            QMessageBox.warning(self, "Composite H5", "\n".join(errors))
        self.auto_intensity()

    def auto_intensity(self):
        values = []
        for image in self.preview_images:
            display = self.transformed_image(image)
            finite = display[np.isfinite(display)]
            if finite.size:
                values.append(finite)

        if values:
            merged = np.concatenate(values)
            vmin = float(np.nanpercentile(merged, 1))
            vmax = float(np.nanpercentile(merged, 99))
            if vmax <= vmin:
                vmax = vmin + 1.0
            self.vmin_spin.blockSignals(True)
            self.vmax_spin.blockSignals(True)
            self.vmin_spin.setValue(vmin)
            self.vmax_spin.setValue(vmax)
            self.vmin_spin.blockSignals(False)
            self.vmax_spin.blockSignals(False)

        self.update_preview()

    def render_composite(self):
        rows = self.rows_spin.value()
        cols = self.cols_spin.value()
        max_images = rows * cols
        images = self.preview_images[:max_images]

        self.fig.clear()
        if images:
            ny, nx = images[0].shape
            self.fig.set_size_inches(max(cols * nx / 300.0, 1.0), max(rows * ny / 300.0, 1.0), forward=True)
        axes = self.fig.subplots(rows, cols, squeeze=False)
        cmap = self.cmap_combo.currentData() or "jet"
        vmin = self.vmin_spin.value()
        vmax = self.vmax_spin.value()

        for index, ax in enumerate(axes.flat):
            ax.set_axis_off()
            if index >= len(images):
                continue
            ax.imshow(
                self.transformed_image(images[index]),
                cmap=cmap,
                origin="upper",
                vmin=vmin,
                vmax=vmax,
                aspect="equal",
            )
            ny, nx = images[index].shape
            ax.set_xlim(-0.5, nx - 0.5)
            ax.set_ylim(ny - 0.5, -0.5)

        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        return len(images), max_images

    def update_preview(self, *args):
        needed = min(len(self.entries), self.rows_spin.value() * self.cols_spin.value())
        if len(self.preview_images) != needed:
            self.load_preview_images()
            return
        count, max_images = self.render_composite()
        self.canvas.draw_idle()
        self.status_label.setText(f"Composite preview: {count} / {len(self.entries)} selected image(s), grid capacity {max_images}.")

    def save_settings(self):
        self.settings.setValue("view/composite_rows", self.rows_spin.value())
        self.settings.setValue("view/composite_cols", self.cols_spin.value())
        self.settings.setValue("view/composite_colormap", self.cmap_combo.currentData() or "jet")
        self.settings.setValue("view/composite_log", self.log_checkbox.isChecked())
        self.settings.setValue("view/composite_nan_to_zero", self.nan_to_zero_checkbox.isChecked())
        self.settings.setValue("view/composite_vmin", self.vmin_spin.value())
        self.settings.setValue("view/composite_vmax", self.vmax_spin.value())

    def save_composite(self):
        if not self.preview_images:
            QMessageBox.information(self, "Composite H5", "No composite image to save.")
            return

        start_folder = str(self.view_tab.current_file.parent) if self.view_tab.current_file is not None else str(Path.home())
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save composite",
            str(Path(start_folder) / "composite.png"),
            "PNG image (*.png);;JPEG image (*.jpg *.jpeg);;TIFF image (*.tif *.tiff);;All files (*)",
        )
        if not path:
            return

        lower_path = path.lower()
        if "JPEG" in selected_filter and not lower_path.endswith((".jpg", ".jpeg")):
            path += ".jpg"
        elif "TIFF" in selected_filter and not lower_path.endswith((".tif", ".tiff")):
            path += ".tif"
        elif not lower_path.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
            path += ".png"

        try:
            self.render_composite()
            save_kwargs = {
                "dpi": 300,
                "bbox_inches": "tight",
                "pad_inches": 0.02,
                "facecolor": "white",
            }
            if path.lower().endswith((".jpg", ".jpeg")):
                save_kwargs["pil_kwargs"] = {"quality": 95}
            self.fig.savefig(path, **save_kwargs)
            self.save_settings()
            self.status_label.setText(f"Composite saved: {path}")
        except Exception as error:
            QMessageBox.warning(self, "Composite save error", f"Unable to save composite:\n{error}")


class FilmDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.view_tab = parent
        self.settings = parent.settings
        self.frames = []
        self.frame_labels = []
        self.current_frame_index = 0
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self.next_frame)

        self.setWindowTitle("Film")
        self.resize(1050, 780)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        source_box = QGroupBox("Frames")
        source_layout = QVBoxLayout(source_box)
        source_layout.setContentsMargins(10, 18, 10, 10)
        source_layout.setSpacing(6)

        source_buttons = QHBoxLayout()
        self.add_files_button = QPushButton("Add files")
        self.add_files_button.clicked.connect(self.add_files)
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_frames)
        source_buttons.addWidget(self.add_files_button)
        source_buttons.addWidget(self.clear_button)
        source_layout.addLayout(source_buttons)

        self.file_list = QListWidget()
        self.file_list.setMinimumHeight(90)
        source_layout.addWidget(self.file_list)
        layout.addWidget(source_box)

        controls_box = QGroupBox("Film settings")
        controls_layout = QGridLayout(controls_box)
        controls_layout.setContentsMargins(10, 18, 10, 10)
        controls_layout.setHorizontalSpacing(8)
        controls_layout.setVerticalSpacing(6)

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(self.settings.value("view/film_fps", 12, type=int))
        self.fps_spin.valueChanged.connect(self.update_timer_interval)
        self.cmap_combo = QComboBox()
        for label, cmap_name in getattr(parent, "colormap_options", [("Jet", "jet")]):
            self.cmap_combo.addItem(label, cmap_name)
        saved_cmap = self.settings.value("view/film_colormap", parent.current_colormap(), type=str)
        saved_cmap_index = self.cmap_combo.findData(saved_cmap)
        self.cmap_combo.setCurrentIndex(max(saved_cmap_index, 0))
        self.cmap_combo.currentIndexChanged.connect(self.update_preview)
        self.log_checkbox = QCheckBox("Log")
        self.log_checkbox.setChecked(self.settings.value("view/film_log", True, type=bool))
        self.log_checkbox.stateChanged.connect(self.update_after_display_change)
        self.nan_to_zero_checkbox = QCheckBox("NaN=0")
        self.nan_to_zero_checkbox.setChecked(self.settings.value("view/film_nan_to_zero", True, type=bool))
        self.nan_to_zero_checkbox.stateChanged.connect(self.update_after_display_change)

        self.vmin_spin = QDoubleSpinBox()
        self.vmin_spin.setDecimals(4)
        self.vmin_spin.setRange(-1e12, 1e12)
        self.vmax_spin = QDoubleSpinBox()
        self.vmax_spin.setDecimals(4)
        self.vmax_spin.setRange(-1e12, 1e12)
        self.vmin_spin.valueChanged.connect(self.update_preview)
        self.vmax_spin.valueChanged.connect(self.update_preview)

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.play)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop)
        self.auto_button = QPushButton("Auto intensity")
        self.auto_button.clicked.connect(self.auto_intensity)
        self.save_button = QPushButton("💾 Save GIF/video")
        self.save_button.clicked.connect(self.save_film)

        controls_layout.addWidget(QLabel("FPS"), 0, 0)
        controls_layout.addWidget(self.fps_spin, 0, 1)
        controls_layout.addWidget(QLabel("Colormap"), 0, 2)
        controls_layout.addWidget(self.cmap_combo, 0, 3)
        controls_layout.addWidget(self.log_checkbox, 0, 4)
        controls_layout.addWidget(self.nan_to_zero_checkbox, 0, 5)
        controls_layout.addWidget(QLabel("Min"), 1, 0)
        controls_layout.addWidget(self.vmin_spin, 1, 1)
        controls_layout.addWidget(QLabel("Max"), 1, 2)
        controls_layout.addWidget(self.vmax_spin, 1, 3)
        controls_layout.addWidget(self.auto_button, 1, 4)
        controls_layout.addWidget(self.play_button, 2, 0, 1, 2)
        controls_layout.addWidget(self.stop_button, 2, 2, 1, 2)
        controls_layout.addWidget(self.save_button, 2, 4, 1, 2)
        layout.addWidget(controls_box)

        self.fig = Figure()
        self.fig.patch.set_alpha(0)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setAttribute(Qt.WA_TranslucentBackground, True)
        self.canvas.setStyleSheet("background: transparent;")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("none")
        self.ax.set_axis_off()
        layout.addWidget(self.canvas, 1)

        self.status_label = QLabel("Add EDF or H5 files to build a film.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.update_timer_interval()
        self.update_buttons()

    def preferred_h5_dataset(self, path):
        candidates = []
        with h5py.File(path, "r") as handle:
            def visitor(name, obj):
                if not isinstance(obj, h5py.Dataset) or obj.ndim not in (2, 3):
                    return
                if not np.issubdtype(obj.dtype, np.number):
                    return
                frame_axis, n_frames, image_shape = self.view_tab.h5_dataset_image_info(obj.shape)
                score = self.view_tab.h5_dataset_image_score(name, obj, frame_axis, n_frames, image_shape)
                candidates.append((score, name, frame_axis, n_frames))

            handle.visititems(visitor)
        if not candidates:
            raise ValueError("No 2D or 3D image dataset found.")
        return max(candidates, key=lambda item: item[0])

    def read_h5_frames(self, path):
        _score, dataset_name, frame_axis, n_frames = self.preferred_h5_dataset(path)
        frames = []
        with h5py.File(path, "r") as handle:
            dataset = handle[dataset_name]
            if dataset.ndim == 2:
                frames.append(np.asarray(dataset[:, :], dtype=float))
            elif frame_axis == 0:
                for index in range(n_frames):
                    frames.append(np.asarray(dataset[index, :, :], dtype=float))
            elif frame_axis == 1:
                for index in range(n_frames):
                    frames.append(np.asarray(dataset[:, index, :], dtype=float))
            else:
                for index in range(n_frames):
                    frames.append(np.asarray(dataset[:, :, index], dtype=float))
        return frames

    def read_edf_frames(self, path):
        try:
            import fabio
        except ImportError:
            raise ImportError("fabio is required to read EDF files.")

        edf = fabio.open(str(path))
        try:
            nframes = int(getattr(edf, "nframes", 1) or 1)
            if nframes <= 1:
                return [np.asarray(edf.data, dtype=float)]
            return [np.asarray(edf.getframe(index).data, dtype=float) for index in range(nframes)]
        finally:
            try:
                edf.close()
            except Exception:
                pass

    def add_files(self):
        start_folder = self.view_tab.current_file.parent if self.view_tab.current_file is not None else Path.home()
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Choose film files",
            str(start_folder),
            "Image data (*.edf *.h5 *.hdf5);;EDF files (*.edf);;HDF5 files (*.h5 *.hdf5);;All files (*)",
        )
        if not paths:
            return

        errors = []
        for path_text in paths:
            path = Path(path_text).expanduser()
            try:
                suffix = path.suffix.lower()
                if suffix == ".edf":
                    new_frames = self.read_edf_frames(path)
                elif suffix in {".h5", ".hdf5"}:
                    new_frames = self.read_h5_frames(path)
                else:
                    raise ValueError("Unsupported file format.")
                start_index = len(self.frames)
                self.frames.extend(new_frames)
                for offset, _frame in enumerate(new_frames):
                    self.frame_labels.append(f"{path.name} #{offset + 1}")
                self.file_list.addItem(f"{path.name}  {len(new_frames)} frame(s)  total {start_index + len(new_frames)}")
            except Exception as error:
                errors.append(f"{path.name}: {error}")

        if errors:
            QMessageBox.warning(self, "Film", "\n".join(errors))
        if self.frames:
            self.current_frame_index = min(self.current_frame_index, len(self.frames) - 1)
            self.auto_intensity()
        self.update_buttons()

    def clear_frames(self):
        self.stop()
        self.frames = []
        self.frame_labels = []
        self.current_frame_index = 0
        self.file_list.clear()
        self.ax.clear()
        self.ax.set_axis_off()
        self.canvas.draw_idle()
        self.status_label.setText("Add EDF or H5 files to build a film.")
        self.update_buttons()

    def transformed_image(self, image):
        display = np.asarray(image, dtype=float).copy()
        display[display > 4e9] = np.nan
        if self.log_checkbox.isChecked():
            display = np.log10(np.clip(display, 0, None) + 1)
        if self.nan_to_zero_checkbox.isChecked():
            display = np.nan_to_num(display, nan=0.0, copy=False)
        return display

    def update_after_display_change(self):
        self.auto_intensity()
        self.update_preview()

    def auto_intensity(self):
        values = []
        for frame in self.frames:
            display = self.transformed_image(frame)
            finite = display[np.isfinite(display)]
            if finite.size:
                values.append(finite)
        if not values:
            return
        merged = np.concatenate(values)
        vmin = float(np.nanpercentile(merged, 1))
        vmax = float(np.nanpercentile(merged, 99))
        if vmax <= vmin:
            vmax = vmin + 1.0
        self.vmin_spin.blockSignals(True)
        self.vmax_spin.blockSignals(True)
        self.vmin_spin.setValue(vmin)
        self.vmax_spin.setValue(vmax)
        self.vmin_spin.blockSignals(False)
        self.vmax_spin.blockSignals(False)
        self.update_preview()

    def update_timer_interval(self):
        self.play_timer.setInterval(max(1, int(1000 / max(1, self.fps_spin.value()))))
        self.settings.setValue("view/film_fps", self.fps_spin.value())

    def update_buttons(self):
        has_frames = bool(self.frames)
        self.play_button.setEnabled(has_frames and not self.play_timer.isActive())
        self.stop_button.setEnabled(self.play_timer.isActive())
        self.save_button.setEnabled(has_frames)

    def play(self):
        if not self.frames:
            return
        self.play_timer.start()
        self.update_buttons()

    def stop(self):
        self.play_timer.stop()
        self.update_buttons()

    def next_frame(self):
        if not self.frames:
            self.stop()
            return
        self.current_frame_index = (self.current_frame_index + 1) % len(self.frames)
        self.update_preview()

    def update_preview(self, *args):
        if not self.frames:
            return
        self.current_frame_index = self.current_frame_index % len(self.frames)
        image = self.transformed_image(self.frames[self.current_frame_index])
        self.ax.clear()
        self.ax.set_facecolor("none")
        self.ax.set_axis_off()
        self.ax.imshow(
            image,
            cmap=self.cmap_combo.currentData() or "jet",
            origin="upper",
            vmin=self.vmin_spin.value(),
            vmax=self.vmax_spin.value(),
            aspect="equal",
        )
        ny, nx = image.shape
        self.ax.set_xlim(-0.5, nx - 0.5)
        self.ax.set_ylim(ny - 0.5, -0.5)
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.canvas.draw_idle()
        self.status_label.setText(
            f"Frame {self.current_frame_index + 1} / {len(self.frames)}"
            f"  {self.frame_labels[self.current_frame_index]}"
        )
        self.settings.setValue("view/film_colormap", self.cmap_combo.currentData() or "jet")
        self.settings.setValue("view/film_log", self.log_checkbox.isChecked())
        self.settings.setValue("view/film_nan_to_zero", self.nan_to_zero_checkbox.isChecked())

    def frame_to_rgb(self, frame):
        image = self.transformed_image(frame)
        vmin = self.vmin_spin.value()
        vmax = self.vmax_spin.value()
        if vmax <= vmin:
            vmax = vmin + 1.0
        normalized = np.clip((image - vmin) / (vmax - vmin), 0, 1)
        normalized[~np.isfinite(normalized)] = 0
        rgba = plt.get_cmap(self.cmap_combo.currentData() or "jet")(normalized)
        return (rgba[:, :, :3] * 255).astype(np.uint8)

    def save_film(self):
        if not self.frames:
            return
        start_folder = self.view_tab.current_file.parent if self.view_tab.current_file is not None else Path.home()
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save film",
            str(Path(start_folder) / "film.gif"),
            "GIF animation (*.gif);;MP4 video (*.mp4);;All files (*)",
        )
        if not path:
            return
        lower_path = path.lower()
        if "MP4" in selected_filter and not lower_path.endswith(".mp4"):
            path += ".mp4"
        elif not lower_path.endswith((".gif", ".mp4")):
            path += ".gif"

        try:
            rgb_frames = [self.frame_to_rgb(frame) for frame in self.frames]
            if path.lower().endswith(".gif"):
                from PIL import Image
                images = [Image.fromarray(frame) for frame in rgb_frames]
                duration_ms = max(1, int(1000 / max(1, self.fps_spin.value())))
                images[0].save(
                    path,
                    save_all=True,
                    append_images=images[1:],
                    duration=duration_ms,
                    loop=0,
                )
            else:
                import imageio.v2 as imageio
                imageio.mimsave(path, rgb_frames, fps=self.fps_spin.value())
            self.status_label.setText(f"Film saved: {path}")
        except ImportError as error:
            QMessageBox.warning(self, "Film save error", f"Missing dependency:\n{error}")
        except Exception as error:
            QMessageBox.warning(self, "Film save error", f"Unable to save film:\n{error}")


class AllColormapPreviewDialog(QDialog):
    def __init__(self, parent, image, colormap_options, vmin, vmax, source_file=None, frame_index=0, total_frames=1):
        super().__init__(parent)
        self.setWindowTitle("All colors preview")
        self.resize(1150, 820)
        self.image = np.asarray(image, dtype=float)
        self.colormap_options = list(colormap_options)[:12]
        self.source_file = Path(source_file) if source_file is not None else None
        self.frame_index = int(frame_index)
        self.total_frames = int(total_frames)
        self.artists = []
        self._syncing_controls = False

        finite = self.image[np.isfinite(self.image)]
        if finite.size:
            self.data_min = float(np.nanmin(finite))
            self.data_max = float(np.nanmax(finite))
        else:
            self.data_min = 0.0
            self.data_max = 1.0
        if self.data_max <= self.data_min:
            self.data_max = self.data_min + 1.0

        if vmax <= vmin:
            vmin, vmax = self.data_min, self.data_max

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        controls = QGridLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setHorizontalSpacing(8)
        controls.setVerticalSpacing(4)

        self.vmin_spin = QDoubleSpinBox()
        self.vmin_spin.setDecimals(4)
        self.vmin_spin.setRange(-1e12, 1e12)
        self.vmin_spin.setValue(vmin)
        self.vmax_spin = QDoubleSpinBox()
        self.vmax_spin.setDecimals(4)
        self.vmax_spin.setRange(-1e12, 1e12)
        self.vmax_spin.setValue(vmax)

        self.min_slider = QSlider(Qt.Horizontal)
        self.min_slider.setRange(0, 1000)
        self.max_slider = QSlider(Qt.Horizontal)
        self.max_slider.setRange(0, 1000)
        self.auto_button = QPushButton("Auto intensity")
        self.auto_button.clicked.connect(self.auto_intensity)
        self.format_combo = QComboBox()
        self.format_combo.addItems(["PNG", "JPEG", "TIFF"])
        self.save_all_button = QPushButton("💾 Save all")
        self.save_all_button.clicked.connect(self.save_all_images)

        controls.addWidget(QLabel("Min"), 0, 0)
        controls.addWidget(self.vmin_spin, 0, 1)
        controls.addWidget(self.min_slider, 0, 2)
        controls.addWidget(QLabel("Max"), 1, 0)
        controls.addWidget(self.vmax_spin, 1, 1)
        controls.addWidget(self.max_slider, 1, 2)
        controls.addWidget(self.auto_button, 0, 3, 2, 1)
        controls.addWidget(QLabel("Format"), 0, 4)
        controls.addWidget(self.format_combo, 0, 5)
        controls.addWidget(self.save_all_button, 1, 4, 1, 2)
        controls.setColumnStretch(2, 1)
        layout.addLayout(controls)

        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas, 1)
        self.hover_label = QLabel(" ")
        self.hover_label.setAlignment(Qt.AlignCenter)
        self.hover_label.setMinimumHeight(22)
        layout.addWidget(self.hover_label)

        self.vmin_spin.valueChanged.connect(self.spin_intensity_changed)
        self.vmax_spin.valueChanged.connect(self.spin_intensity_changed)
        self.min_slider.valueChanged.connect(self.slider_intensity_changed)
        self.max_slider.valueChanged.connect(self.slider_intensity_changed)
        self.canvas.mpl_connect("motion_notify_event", self.on_hover)
        self.canvas.mpl_connect("figure_leave_event", self.on_leave)

        self.sync_sliders_from_spins()
        self.draw_preview()

    def value_to_slider(self, value):
        if self.data_max <= self.data_min:
            return 0
        slider_value = int(1000 * (value - self.data_min) / (self.data_max - self.data_min))
        return max(0, min(1000, slider_value))

    def slider_to_value(self, value):
        return self.data_min + (value / 1000.0) * (self.data_max - self.data_min)

    def sync_sliders_from_spins(self):
        self._syncing_controls = True
        self.min_slider.setValue(self.value_to_slider(self.vmin_spin.value()))
        self.max_slider.setValue(self.value_to_slider(self.vmax_spin.value()))
        self._syncing_controls = False

    def spin_intensity_changed(self):
        if self._syncing_controls:
            return
        if self.vmax_spin.value() <= self.vmin_spin.value():
            return
        self.sync_sliders_from_spins()
        self.update_intensity()

    def slider_intensity_changed(self):
        if self._syncing_controls:
            return
        vmin = self.slider_to_value(self.min_slider.value())
        vmax = self.slider_to_value(self.max_slider.value())
        if vmax <= vmin:
            return
        self._syncing_controls = True
        self.vmin_spin.setValue(vmin)
        self.vmax_spin.setValue(vmax)
        self._syncing_controls = False
        self.update_intensity()

    def auto_intensity(self):
        finite = self.image[np.isfinite(self.image)]
        if finite.size == 0:
            return
        vmin = float(np.nanpercentile(finite, 1))
        vmax = float(np.nanpercentile(finite, 99))
        if vmax <= vmin:
            vmax = vmin + 1.0
        self._syncing_controls = True
        self.vmin_spin.setValue(vmin)
        self.vmax_spin.setValue(vmax)
        self._syncing_controls = False
        self.sync_sliders_from_spins()
        self.update_intensity()

    def draw_preview(self):
        self.fig.clear()
        self.artists = []
        axes = self.fig.subplots(3, 4)
        vmin = self.vmin_spin.value()
        vmax = self.vmax_spin.value()

        for ax, (label, cmap_name) in zip(np.ravel(axes), self.colormap_options):
            ax._lrphoton_colormap_label = label
            artist = ax.imshow(
                self.image,
                cmap=cmap_name,
                origin="upper",
                vmin=vmin,
                vmax=vmax,
                aspect="equal",
            )
            ax.set_axis_off()
            self.artists.append(artist)

        for ax in np.ravel(axes)[len(self.colormap_options):]:
            ax.set_axis_off()

        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        self.canvas.draw_idle()

    def on_hover(self, event):
        label = getattr(event.inaxes, "_lrphoton_colormap_label", "") if event.inaxes is not None else ""
        self.hover_label.setText(label or " ")

    def on_leave(self, _event):
        self.hover_label.setText(" ")

    def save_all_images(self):
        start_folder = self.source_file.parent if self.source_file is not None else Path.home()
        output_folder = QFileDialog.getExistingDirectory(
            self,
            "Save all preview colors",
            str(start_folder),
        )
        if not output_folder:
            return

        selected_format = self.format_combo.currentText()
        extension = {
            "PNG": ".png",
            "JPEG": ".jpg",
            "TIFF": ".tif",
        }.get(selected_format, ".png")

        base_name = self.source_file.stem if self.source_file is not None else "preview_colors"
        if self.total_frames > 1:
            base_name = f"{base_name}_frame{self.frame_index + 1:04d}"

        vmin = self.vmin_spin.value()
        vmax = self.vmax_spin.value()
        if vmax <= vmin:
            return

        try:
            saved_count = 0
            for label, cmap_name in self.colormap_options:
                slug = re.sub(r"[^A-Za-z0-9_-]+", "_", label.strip()).strip("_").lower()
                path = Path(output_folder) / f"{base_name}_{slug}{extension}"
                save_kwargs = {}
                if extension == ".jpg":
                    save_kwargs["pil_kwargs"] = {"quality": 95}
                plt.imsave(
                    path,
                    self.image,
                    cmap=cmap_name,
                    vmin=vmin,
                    vmax=vmax,
                    origin="upper",
                    **save_kwargs,
                )
                saved_count += 1

            QMessageBox.information(
                self,
                "Save all preview colors",
                f"Saved {saved_count} image(s) in:\n{output_folder}",
            )
        except Exception as error:
            QMessageBox.critical(
                self,
                "Save all preview colors error",
                f"Unable to save all preview images:\n{error}",
            )

    def update_intensity(self):
        vmin = self.vmin_spin.value()
        vmax = self.vmax_spin.value()
        if vmax <= vmin:
            return
        for artist in self.artists:
            artist.set_clim(vmin, vmax)
        self.canvas.draw_idle()


class ViewTab(QWidget):
    folder_changed = Signal(Path)
    def __init__(self):
        super().__init__()

        self.settings = QSettings("LRP", "LRPhoton")

        self.current_folder = Path(
            "/Users/nathanpiaget/Documents/Thèse LRP/Expériences/XENOCS"
        )
        self.current_file = None
        self._syncing_folder = False

        self.images = None
        self.display_img = None
        self.raw_current_img = None
        self.headers = {}
        self.complementary_geometry_metadata = []
        self.h5_datasets = []
        self.is_lazy_h5 = False
        self.is_lazy_edf = False
        self.edf_path = None
        self.h5_file = None
        self.h5_dataset = None
        self.h5_frame_axis = None
        self.n_frames = 0
        self.image_shape = None

        self.current_index = 0
        self.image_artist = None
        self.colorbar = None
        self.center_artists = []

        self.intensity_min = 0.0
        self.intensity_max = 1.0

        self._is_panning = False
        self._pan_start_event = None
        self._pan_start_xlim = None
        self._pan_start_ylim = None

        self._saved_xlim = None
        self._saved_ylim = None
        self._trackpad_view_pushed = False

        self.q_geometry_mode = None
        self.q_geometry_source_tab = None
        self.custom_q_geometry = self.load_custom_q_geometry()
        self.current_file_type = None
        self.current_dataset_name = None
        self.annotation_dialog = None
        self.image_axis_bounds = None
        self.is_azimuthal_image = False

        self._build_ui()

    def create_matplotlib_toolbar_block(
        self,
        title,
        toolbar,
        option_widgets=None,
        save_callback=None,
        save_tooltip="Save",
        toolbar_width=340,
        remove_customize=False,
    ):
        return make_matplotlib_toolbar_block(self, title, toolbar, option_widgets=option_widgets, save_callback=save_callback, save_tooltip=save_tooltip, toolbar_width=toolbar_width, remove_customize=remove_customize)

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(*PAGE_MARGINS)
        main_layout.setSpacing(BLOCK_SPACING)

        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(BLOCK_SPACING)
        main_layout.addLayout(content_layout, stretch=1)

        # ============================================================
        # LEFT PANEL
        # ============================================================

        left_panel = QWidget()
        left_panel.setFixedWidth(FILE_BROWSER_WIDTH)

        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(*PANEL_MARGINS)
        left_layout.setSpacing(BLOCK_SPACING)

        file_box = QGroupBox("File browser")
        file_box.setMinimumHeight(220)
        file_box.setStyleSheet(GROUP_BOX_STYLE)
        file_layout = QVBoxLayout(file_box)
        file_layout.setContentsMargins(*GROUP_BOX_MARGINS)
        file_layout.setSpacing(6)

        self.folder_path = QLineEdit(str(self.current_folder))
        self.folder_path.returnPressed.connect(self.refresh_files)
        file_layout.addWidget(self.folder_path)

        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.choose_folder)
        file_layout.addWidget(browse_button)

        filters_layout = QGridLayout()

        self.extension_filter = QLineEdit("*.edf *.h5")
        self.name_filter = QLineEdit("**")
        self.extension_filter.textChanged.connect(self.refresh_files)
        self.name_filter.textChanged.connect(self.refresh_files)

        self.show_subfolders_checkbox = QCheckBox("Show subfolders")
        self.show_subfolders_checkbox.setChecked(False)
        self.show_subfolders_checkbox.stateChanged.connect(self.refresh_files)
        self.only_thumbs_up_checkbox = QCheckBox("Only 👍")
        self.only_thumbs_up_checkbox.setChecked(False)
        self.only_thumbs_up_checkbox.stateChanged.connect(self.refresh_files)

        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh_files)

        filters_layout.addWidget(QLabel("Name:"), 0, 0)
        filters_layout.addWidget(self.name_filter, 0, 1)
        filters_layout.addWidget(QLabel("Extensions:"), 1, 0)
        filters_layout.addWidget(self.extension_filter, 1, 1)
        file_layout.addLayout(filters_layout)
        file_options_layout = QHBoxLayout()
        file_options_layout.setContentsMargins(0, 0, 0, 0)
        file_options_layout.setSpacing(10)
        file_options_layout.addWidget(self.show_subfolders_checkbox)
        file_options_layout.addWidget(self.only_thumbs_up_checkbox)
        file_options_layout.addStretch(1)
        file_layout.addLayout(file_options_layout)
        file_layout.addWidget(refresh_button)

        self.file_list = QListWidget()
        install_file_rating_menu(self.file_list)
        self.file_list.currentItemChanged.connect(self.file_selection_changed)
        self.file_list.itemClicked.connect(self.open_selected_file)
        self.file_list.itemDoubleClicked.connect(self.open_selected_file)
        file_layout.addWidget(self.file_list, stretch=1)

        left_layout.addWidget(file_box, stretch=1)

        content_layout.addWidget(left_panel, stretch=0)

        # ============================================================
        # CENTER PANEL
        # ============================================================

        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(*PANEL_MARGINS)
        center_layout.setSpacing(BLOCK_SPACING)


        self.fig = Figure()
        self.fig.patch.set_alpha(0)

        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("none")
        self.ax.set_axis_off()
        self.ax.set_aspect("equal")

        self.canvas = ViewImageCanvas(self.fig, self)
        self.canvas.setFocus()

        self.toolbar = ImageOnlyToolbar(self.canvas, self)
        self.log_checkbox = QCheckBox("Log")
        self.log_checkbox.setChecked(True)
        self.log_checkbox.stateChanged.connect(self.update_image)

        self.nan_to_zero_checkbox = QCheckBox("NaN=0")
        self.nan_to_zero_checkbox.setToolTip("Display NaN pixels as 0 instead of leaving them blank")
        self.nan_to_zero_checkbox.setChecked(
            self.settings.value("view/nan_to_zero", True, type=bool)
        )
        self.nan_to_zero_checkbox.stateChanged.connect(self.change_nan_to_zero)

        self.colormap_combo = QComboBox()
        self.colormap_combo.setToolTip("Colormap")
        self.colormap_combo.setFixedWidth(125)
        self.colormap_options = [
            ("Gray", "gray"),
            ("Reversed Gray", "gray_r"),
            ("Red", "Reds"),
            ("Green", "Greens"),
            ("Blue", "Blues"),
            ("Viridis", "viridis"),
            ("Cividis", "cividis"),
            ("Magma", "magma"),
            ("Inferno", "inferno"),
            ("Plasma", "plasma"),
            ("Temperature", "turbo"),
            ("Jet", "jet"),
            ("Hsv", "hsv"),
        ]
        for label, cmap_name in self.colormap_options:
            self.colormap_combo.addItem(label, cmap_name)
        saved_colormap = self.settings.value("view/colormap", "jet", type=str)
        saved_index = self.colormap_combo.findData(saved_colormap)
        if saved_index < 0:
            saved_index = self.colormap_combo.findData("jet")
        self.colormap_combo.setCurrentIndex(max(saved_index, 0))
        self.colormap_combo.currentIndexChanged.connect(self.change_colormap)

        self.save_all_colors_button = QPushButton("Preview colors")
        self.save_all_colors_button.setToolTip("Preview 12 colormaps with a shared intensity scale")
        self.save_all_colors_button.setFixedWidth(125)
        self.save_all_colors_button.setFixedHeight(22)
        self.save_all_colors_button.clicked.connect(self.show_all_colormap_preview)

        self.keep_ratio_checkbox = QCheckBox("Keep ratio")
        self.keep_ratio_checkbox.setChecked(True)
        self.keep_ratio_checkbox.stateChanged.connect(self.update_image)

        self.keep_zoom_checkbox = QCheckBox("Keep zoom")
        self.keep_zoom_checkbox.setChecked(True)
        self.keep_zoom_checkbox.setToolTip("Keep current zoom and pan when changing file or frame")

        self.save_colorbar_checkbox = QCheckBox("Keep contrast")
        self.save_colorbar_checkbox.setChecked(
            self.settings.value("view/save_colorbar", False, type=bool)
        )
        self.save_colorbar_checkbox.stateChanged.connect(self.save_colorbar_setting)

        toolbar_box, self.toolbar_extra_layout, self.save_image_button = self.create_matplotlib_toolbar_block(
            title="Scattering pattern",
            toolbar=self.toolbar,
            option_widgets=[
                self.colormap_combo,
                self.log_checkbox,
                self.keep_ratio_checkbox,
                self.keep_zoom_checkbox,
                self.save_colorbar_checkbox,
            ],
            save_callback=self.save_png_image_only,
            save_tooltip="Save image only",
            toolbar_width=340,
            remove_customize=True,
        )

        center_layout.addWidget(toolbar_box, alignment=Qt.AlignTop)

        image_area = QHBoxLayout()
        image_area.setContentsMargins(0, 0, 0, 0)
        image_area.setSpacing(4)
        image_area.addWidget(self.canvas, stretch=1)

        slider_panel = QWidget()
        slider_panel.setFixedWidth(110)
        slider_panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)

        slider_box = QVBoxLayout(slider_panel)
        self.slider_box = slider_box
        slider_box.setContentsMargins(0, 0, 0, 0)
        slider_box.setSpacing(2)

        self.max_slider = QSlider(Qt.Vertical)
        self.max_slider.setMinimum(0)
        self.max_slider.setMaximum(1000)
        self.max_slider.setValue(1000)
        self.max_slider.valueChanged.connect(self.vertical_sliders_changed)

        self.min_slider = QSlider(Qt.Vertical)
        self.min_slider.setMinimum(0)
        self.min_slider.setMaximum(1000)
        self.min_slider.setValue(0)
        self.min_slider.valueChanged.connect(self.vertical_sliders_changed)

        self.vmin_spin = QDoubleSpinBox()
        self.vmin_spin.setDecimals(4)
        self.vmin_spin.setRange(-999999, 999999)
        self.vmin_spin.setFixedWidth(90)
        self.vmin_spin.setValue(self.settings.value("view/vmin", 0.0, type=float))
        self.vmin_spin.valueChanged.connect(self.spin_intensity_changed)

        self.vmax_spin = QDoubleSpinBox()
        self.vmax_spin.setDecimals(4)
        self.vmax_spin.setRange(-999999, 999999)
        self.vmax_spin.setFixedWidth(90)
        self.vmax_spin.setValue(self.settings.value("view/vmax", 5.0, type=float))
        self.vmax_spin.valueChanged.connect(self.spin_intensity_changed)

        autoscale_button = QPushButton("Auto")
        self.autoscale_button = autoscale_button
        autoscale_button.setFixedWidth(90)
        autoscale_button.clicked.connect(self.auto_intensity)

        max_label = QLabel("Max:")
        min_label = QLabel("Min:")

        max_label.setAlignment(Qt.AlignCenter)
        min_label.setAlignment(Qt.AlignCenter)

        slider_box.addWidget(max_label, alignment=Qt.AlignHCenter)
        slider_box.addWidget(self.vmax_spin, alignment=Qt.AlignHCenter)
        slider_box.addWidget(self.max_slider, alignment=Qt.AlignHCenter)
        slider_box.addSpacing(8)
        slider_box.addWidget(self.min_slider, alignment=Qt.AlignHCenter)
        slider_box.addWidget(min_label, alignment=Qt.AlignHCenter)
        slider_box.addWidget(self.vmin_spin, alignment=Qt.AlignHCenter)
        slider_box.addWidget(autoscale_button, alignment=Qt.AlignHCenter)

        image_area.addWidget(slider_panel, stretch=0, alignment=Qt.AlignRight)
        center_layout.addLayout(image_area)


        self.cursor_label = QLabel("x = - | y = - | q = - | I = -")
        self.cursor_label.setMinimumHeight(28)
        self.cursor_label.setAlignment(Qt.AlignCenter)
        self.cursor_label.setStyleSheet("""
            QLabel {
                background-color: #f4f4f4;
                border-radius: 8px;
                padding: 6px;
                font-family: Menlo, Monaco, monospace;
                font-size: 11px;
            }
        """)
        center_layout.addWidget(self.cursor_label)

        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas.mpl_connect("figure_leave_event", self.on_mouse_leave)
        self.canvas.mpl_connect("scroll_event", self.on_scroll_zoom)
        self.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)

        nav_layout = QHBoxLayout()
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(FRAME_NAV_SPACING)

        self.previous_button = QPushButton("<")
        self.next_button = QPushButton(">")
        self.previous_button.setFixedWidth(FRAME_BUTTON_WIDTH)
        self.next_button.setFixedWidth(FRAME_BUTTON_WIDTH)

        self.previous_button.clicked.connect(self.previous_image)
        self.next_button.clicked.connect(self.next_image)

        self.frame_start_spin = QSpinBox()
        self.frame_start_spin.setMinimum(1)
        self.frame_start_spin.setMaximum(1)
        self.frame_start_spin.setValue(1)
        self.frame_start_spin.setFixedWidth(FRAME_SPIN_WIDTH)
        self.frame_start_spin.valueChanged.connect(self.update_frame_slider_range)

        self.frame_end_spin = QSpinBox()
        self.frame_end_spin.setMinimum(1)
        self.frame_end_spin.setMaximum(1)
        self.frame_end_spin.setValue(1)
        self.frame_end_spin.setFixedWidth(FRAME_SPIN_WIDTH)
        self.frame_end_spin.valueChanged.connect(self.update_frame_slider_range)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.valueChanged.connect(self.slider_changed)

        self.frame_label = QLabel("0 / 0")
        self.frame_label.setMinimumWidth(FRAME_COUNTER_WIDTH)
        self.frame_label.setAlignment(Qt.AlignCenter)

        nav_layout.addWidget(QLabel("Start:"))
        nav_layout.addWidget(self.frame_start_spin)

        nav_layout.addWidget(self.previous_button)
        nav_layout.addWidget(self.frame_slider, stretch=1)
        nav_layout.addWidget(self.next_button)

        nav_layout.addWidget(QLabel("End:"))
        nav_layout.addWidget(self.frame_end_spin)
        nav_layout.addWidget(self.frame_label)


        content_layout.addWidget(center_panel, stretch=1)

        # ============================================================
        # RIGHT PANEL
        # ============================================================

        right_panel = QWidget()
        right_panel.setFixedWidth(FILE_BROWSER_WIDTH)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(*PANEL_MARGINS)
        right_layout.setSpacing(BLOCK_SPACING)

        info_box = QGroupBox("File information")
        self.info_box = info_box
        self.right_layout = right_layout
        info_box.setMinimumHeight(86)
        self.panel_box_style = GROUP_BOX_STYLE
        info_box.setStyleSheet(self.panel_box_style)
        info_box_layout = QVBoxLayout(info_box)
        self.info_box_layout = info_box_layout
        info_box_layout.setContentsMargins(*GROUP_BOX_MARGINS)
        info_box_layout.setSpacing(6)

        self.info_text = QTextEdit()
        self.info_text.setLineWrapMode(QTextEdit.WidgetWidth)
        self.info_text.setMinimumWidth(240)
        self.info_text.setReadOnly(True)
        self.info_text.setText("")

        self.dataset_list = QListWidget()
        self.dataset_list.itemDoubleClicked.connect(self.open_selected_dataset)
        self.dataset_list.hide()

        q_buttons_layout = QHBoxLayout()
        self.q_buttons_layout = q_buttons_layout
        q_buttons_layout.setSpacing(4)
        self.q_xenocs_button = QPushButton("XENOCS")
        self.q_id02_button = QPushButton("ID02")
        self.q_id13_button = QPushButton("ID13")
        self.q_custom_button = QPushButton("Custom")
        self.q_manual_button = QPushButton("+")
        self.q_manual_button.setFixedWidth(28)

        self.q_xenocs_button.clicked.connect(lambda: self.set_q_geometry_mode("XENOCS"))
        self.q_id02_button.clicked.connect(lambda: self.set_q_geometry_mode("ID02"))
        self.q_id13_button.clicked.connect(lambda: self.set_q_geometry_mode("ID13"))
        self.q_custom_button.clicked.connect(self.use_custom_q_geometry_from_source)
        self.q_manual_button.clicked.connect(self.open_q_geometry_dialog)

        for button in [
            self.q_xenocs_button,
            self.q_id02_button,
            self.q_id13_button,
            self.q_custom_button,
        ]:
            button.setCheckable(True)

        self.q_xenocs_button.setChecked(True)

        for button in [
            self.q_xenocs_button,
            self.q_id02_button,
            self.q_id13_button,
            self.q_custom_button,
            self.q_manual_button,
        ]:
            q_buttons_layout.addWidget(button)
            button.hide()

        self.line_geometry_selector = LineGeometrySelector(self, "XENOCS")
        self.line_geometry_selector.geometry_selected.connect(self.apply_line_geometry_selection)
        q_buttons_layout.addWidget(self.line_geometry_selector, 1)

        self.update_q_geometry_button_styles()
        self.set_q_geometry_mode("XENOCS")
        info_box_layout.addLayout(q_buttons_layout)
        info_box_layout.addWidget(self.info_text)
        annotation_buttons_layout = QHBoxLayout()
        annotation_buttons_layout.setContentsMargins(0, 0, 0, 0)
        annotation_buttons_layout.setSpacing(6)
        self.open_annotation_button = QPushButton("✏️ Annotate")
        self.open_annotation_button.clicked.connect(self.open_annotation_window)
        self.open_composite_button = QPushButton("🎞️ Composite")
        self.open_composite_button.clicked.connect(self.open_composite_window)
        annotation_buttons_layout.addWidget(self.open_annotation_button)
        annotation_buttons_layout.addWidget(self.open_composite_button)
        info_box_layout.addLayout(annotation_buttons_layout)
        right_layout.addWidget(info_box)

        content_layout.addWidget(right_panel, stretch=0)

        right_layout.setStretch(0, 1)
        main_layout.addLayout(nav_layout, stretch=0)
        self.update_frame_navigation_state()

        self.canvas.draw_idle()
        self.set_toolbar_options_enabled(False)

    def set_toolbar_options_enabled(self, enabled):
        set_matplotlib_toolbar_enabled(getattr(self, "toolbar", None), enabled)
        for widget in [
            getattr(self, "log_checkbox", None),
            getattr(self, "nan_to_zero_checkbox", None),
            getattr(self, "colormap_combo", None),
            getattr(self, "save_all_colors_button", None),
            getattr(self, "keep_ratio_checkbox", None),
            getattr(self, "keep_zoom_checkbox", None),
            getattr(self, "save_colorbar_checkbox", None),
            getattr(self, "save_image_button", None),
            getattr(self, "min_slider", None),
            getattr(self, "max_slider", None),
            getattr(self, "vmin_spin", None),
            getattr(self, "vmax_spin", None),
            getattr(self, "autoscale_button", None),
            getattr(self, "open_annotation_button", None),
        ]:
            if widget is not None:
                if widget is getattr(self, "save_image_button", None):
                    set_widget_enabled_with_opacity(widget, enabled)
                else:
                    widget.setEnabled(enabled)

    # ============================================================
    # SETTINGS
    # ============================================================

    def save_colorbar_setting(self):
        keep_colorbar = self.save_colorbar_checkbox.isChecked()
        self.settings.setValue("view/save_colorbar", keep_colorbar)

        if keep_colorbar:
            self.settings.setValue("view/vmin", self.vmin_spin.value())
            self.settings.setValue("view/vmax", self.vmax_spin.value())

    def change_nan_to_zero(self, *args):
        self.settings.setValue("view/nan_to_zero", self.nan_to_zero_checkbox.isChecked())
        self.update_image()

    def current_colormap(self):
        combo = getattr(self, "colormap_combo", None)
        if combo is None:
            return "jet"
        cmap_name = combo.currentData()
        return cmap_name or "jet"

    def change_colormap(self, *args):
        cmap_name = self.current_colormap()
        self.settings.setValue("view/colormap", cmap_name)

        if self.image_artist is not None:
            self.image_artist.set_cmap(cmap_name)
            if self.colorbar is not None:
                self.colorbar.update_normal(self.image_artist)
            self.canvas.draw_idle()

    def set_q_geometry_source_tab(self, tab):
        self.q_geometry_source_tab = tab

    def load_custom_q_geometry(self):
        keys = ["xc", "yc", "distance_m", "pixel_x_mm", "pixel_y_mm", "wavelength_a"]
        values = {}
        for key in keys:
            value = self.settings.value(f"view/custom_q_geometry/{key}", None, type=float)
            if value is None:
                return None
            values[key] = value
        return values

    def save_custom_q_geometry(self):
        if not self.custom_q_geometry:
            return

        for key, value in self.custom_q_geometry.items():
            self.settings.setValue(f"view/custom_q_geometry/{key}", value)

    def preset_q_geometry(self, mode):
        if mode == "ID02":
            return {
                "xc": 914.4,
                "yc": 996.5,
                "distance_m": 10.0002,
                "pixel_x_mm": 0.075000,
                "pixel_y_mm": 0.075000,
                "wavelength_a": 1.01402,
            }

        if mode == "XENOCS":
            return {
                "xc": 0.0,
                "yc": 0.0,
                "distance_m": 0.0,
                "pixel_x_mm": 0.075000,
                "pixel_y_mm": 0.075000,
                "wavelength_a": 0.0,
            }

        if mode == "ID13":
            return {
                "xc": ID13_DEFAULT_CENTER_X,
                "yc": ID13_DEFAULT_CENTER_Y,
                "distance_m": ID13_DEFAULT_DISTANCE_M,
                "pixel_x_mm": ID13_DEFAULT_PIXEL_MM,
                "pixel_y_mm": ID13_DEFAULT_PIXEL_MM,
                "wavelength_a": ID13_DEFAULT_WAVELENGTH_A,
            }

        if mode == "Custom":
            return self.custom_q_geometry

        return None

    def update_q_geometry_button_styles(self):
        buttons = {
            "XENOCS": self.q_xenocs_button,
            "ID02": self.q_id02_button,
            "ID13": self.q_id13_button,
            "Custom": self.q_custom_button,
        }
        style_q_geometry_buttons(buttons, self.q_geometry_mode, self.q_manual_button)

    def set_q_geometry_mode(self, mode):
        if mode == "Custom" and not self.custom_q_geometry:
            self.open_q_geometry_dialog()
            return

        self.q_geometry_mode = mode
        if hasattr(self, "line_geometry_selector") and mode in self.line_geometry_selector.geometries:
            self.line_geometry_selector.set_current_name(mode)
        self.update_q_geometry_button_styles()
        self.refresh_file_information()
        self.update_image()

    def apply_line_geometry_selection(self, name, geometry):
        values = line_geometry_to_lrphoton(geometry)
        self.custom_q_geometry = {
            "xc": values["xc"],
            "yc": values["yc"],
            "distance_m": values["distance_m"],
            "pixel_x_mm": values["pixel_x_mm"],
            "pixel_y_mm": values["pixel_y_mm"],
            "wavelength_a": values["wavelength_a"],
        }
        self.save_custom_q_geometry()
        self.q_geometry_mode = "Custom" if name not in {"XENOCS", "ID02", "ID13"} else name
        self.update_q_geometry_button_styles()
        self.refresh_file_information()
        self.update_image()

    def use_custom_q_geometry_from_source(self):
        if self.q_geometry_source_tab is not None:
            try:
                self.custom_q_geometry = {
                    "xc": self.q_geometry_source_tab.center_x.value(),
                    "yc": self.q_geometry_source_tab.center_y.value(),
                    "distance_m": self.q_geometry_source_tab.distance.value(),
                    "pixel_x_mm": self.q_geometry_source_tab.pixel_x.value(),
                    "pixel_y_mm": self.q_geometry_source_tab.pixel_y.value(),
                    "wavelength_a": self.q_geometry_source_tab.wavelength.value(),
                }
                self.save_custom_q_geometry()
            except Exception:
                pass

        self.set_q_geometry_mode("Custom")

    def open_q_geometry_dialog(self):
        geometry = self.preset_q_geometry(self.q_geometry_mode) or self.custom_q_geometry
        if geometry is None:
            geometry = self.preset_q_geometry("ID13")

        dialog = QDialog(self)
        dialog.setWindowTitle("Custom q geometry")
        layout = QVBoxLayout(dialog)
        form = QFormLayout()

        fields = {}
        labels = [
            ("xc", "Center X"),
            ("yc", "Center Y"),
            ("distance_m", "Distance (m)"),
            ("pixel_x_mm", "Pixel X (mm)"),
            ("pixel_y_mm", "Pixel Y (mm)"),
            ("wavelength_a", "Wavelength (Å)"),
        ]

        for key, label in labels:
            spin = QDoubleSpinBox()
            spin.setDecimals(6)
            spin.setRange(0, 1e9)
            spin.setValue(float(geometry.get(key, 0)))
            spin.setFixedWidth(130)
            fields[key] = spin
            form.addRow(label, spin)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        layout.addLayout(form)
        layout.addWidget(buttons)

        if dialog.exec() != QDialog.Accepted:
            return

        self.custom_q_geometry = {
            key: spin.value()
            for key, spin in fields.items()
        }
        self.save_custom_q_geometry()
        self.set_q_geometry_mode("Custom")

    # ============================================================
    # FILE BROWSER
    # ============================================================

    def set_folder_from_external_tab(self, folder):
        folder = Path(folder).expanduser().resolve()

        if self.current_folder.expanduser().resolve() == folder:
            return

        self._syncing_folder = True
        self.current_folder = folder
        self.folder_path.setText(str(self.current_folder))
        self.refresh_files()
        self._syncing_folder = False

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Choose folder",
            str(self.current_folder)
        )

        if folder:
            self.current_folder = Path(folder)
            self.folder_path.setText(str(self.current_folder))
            self.refresh_files()

    def should_show_file_in_browser(self, path):
        if should_hide_file_in_browser(path):
            return False

        lower_name = path.name.lower()

        if lower_name.endswith(".dat"):
            return False

        if lower_name.endswith("_ave.h5"):
            return False

        return True

    def refresh_files(self):
        folder = Path(self.folder_path.text()).expanduser()

        if not folder.exists():
            QMessageBox.warning(
                self,
                "Folder not found",
                "The selected folder does not exist."
            )
            return

        self.current_folder = folder

        if not self._syncing_folder:
            self.folder_changed.emit(self.current_folder)

        self.file_list.clear()

        extension_patterns = self.extension_filter.text().split()
        name_pattern = self.name_filter.text().strip() or "*"

        if self.show_subfolders_checkbox.isChecked():
            iterator = folder.rglob("*")
        else:
            iterator = folder.glob("*")

        files = []

        for path in iterator:
            if not path.is_file():
                continue

            lower_name = path.name.lower()

            if not self.should_show_file_in_browser(path):
                continue

            match_extension = any(
                fnmatch.fnmatch(lower_name, pattern.lower())
                for pattern in extension_patterns
            )

            match_name = fnmatch.fnmatch(path.name, name_pattern)

            if match_extension and match_name:
                if self.only_thumbs_up_checkbox.isChecked() and not is_file_rated_up(path):
                    continue
                files.append(path)

        for path in sorted(files):
            item_text = str(path.relative_to(folder))
            self.file_list.addItem(item_text)
            item = self.file_list.item(self.file_list.count() - 1)
            set_item_file_path(item, path)

    def file_selection_changed(self, current, previous):
        if current is None:
            return

        self.open_selected_file(current)

    def open_selected_file(self, item=None):
        if item is None:
            item = self.file_list.currentItem()

        if item is None:
            return

        stored_path = item.data(Qt.UserRole)

        if not stored_path:
            return

        path = Path(stored_path).expanduser().resolve()
        print("Clicked item:", item.text())
        print("Selected path:", path)
        self.open_file(path)

    # ============================================================
    # FILE OPENING
    # ============================================================

    def open_file(self, path):
        self.current_file = Path(path).expanduser().resolve()
        if should_hide_file_in_browser(self.current_file):
            self.current_file = None
            self.refresh_files()
            return

        self.set_toolbar_options_enabled(False)
        if hasattr(self, "keep_zoom_checkbox") and self.keep_zoom_checkbox.isChecked() and self.image_artist is not None:
            self._saved_xlim = self.ax.get_xlim()
            self._saved_ylim = self.ax.get_ylim()
        else:
            self._saved_xlim = None
            self._saved_ylim = None
        print("Loaded file:", self.current_file)

        self.images = None
        self.display_img = None
        self.raw_current_img = None
        self.headers = {}
        self.complementary_geometry_metadata = []
        self.h5_datasets = []
        self.is_lazy_h5 = False
        self.is_lazy_edf = False
        self.edf_path = None
        self.h5_dataset = None
        self.h5_frame_axis = None
        self.is_azimuthal_image = False

        if self.h5_file is not None:
            try:
                self.h5_file.close()
            except Exception:
                pass

        self.h5_file = None
        self.current_index = 0

        self.dataset_list.clear()

        # Important : reset figure completely when opening a new file.
        # This avoids cumulative colorbar/axis shrinking.
        self.reset_figure()

        suffix = self.current_file.suffix.lower()

        try:
            if suffix == ".edf":
                self.open_edf(self.current_file)

            elif suffix == ".h5":
                self.open_h5(self.current_file)

            else:
                QMessageBox.warning(
                    self,
                    "Unsupported file",
                    "Unsupported file format."
                )
                self.current_file = None
                self.current_file_type = None
                self.set_toolbar_options_enabled(False)

        except Exception as e:
            self.current_file = None
            self.current_file_type = None
            self.set_toolbar_options_enabled(False)
            QMessageBox.critical(self, "Error", str(e))

    def reset_figure(self):
        self.fig.clear()
        self.fig.patch.set_alpha(0)

        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("none")
        self.ax.set_axis_off()
        self.ax.set_aspect("equal")

        self.image_artist = None
        self.colorbar = None
        self.center_artists = []

        self.fig.subplots_adjust(
            left=0.005,
            right=0.995,
            top=0.995,
            bottom=0.005
        )

        self.canvas.draw_idle()

    def open_edf(self, path):
        try:
            import fabio
        except ImportError:
            raise ImportError(
                "fabio is required to open EDF files.\n"
                "Install it with: pip install fabio"
            )

        path = Path(path).expanduser().resolve()
        print("Reading EDF from disk:", path)

        self.images = None
        self.headers = {}
        self.raw_header_text = ""
        self.is_lazy_edf = True
        self.edf_path = path
        first_image = None

        edf = fabio.open(str(path))

        try:
            nframes = int(getattr(edf, "nframes", 1) or 1)

            header_text = ""
            try:
                with open(path, "rb") as handle:
                    header_bytes = handle.read(65536)
                    header_text = header_bytes.decode("latin-1", errors="ignore")

                header_match = re.search(r"EDF_HeaderSize\s*=\s*(\d+)", header_text)
                if header_match:
                    header_size = int(header_match.group(1))
                    with open(path, "rb") as handle:
                        header_bytes = handle.read(header_size)
                        header_text = header_bytes.decode("latin-1", errors="ignore")
            except Exception:
                header_text = ""

            if header_text:
                self.raw_header_text = header_text
                parsed_header = parse_header_text(header_text)
            else:
                parsed_header = {}

            if nframes <= 1:
                first_image = np.array(edf.data, dtype=float).copy()
                self.headers = dict(edf.header)
            else:
                first_frame = edf.getframe(0)
                first_image = np.array(first_frame.data, dtype=float).copy()
                self.headers = dict(first_frame.header)

            if parsed_header:
                self.headers = {**self.headers, **parsed_header}

        finally:
            try:
                edf.close()
            except Exception:
                pass

        if first_image is None:
            raise ValueError("No frame was found in this EDF file.")

        self.n_frames = max(1, nframes)
        self.image_shape = first_image.shape
        self.add_matching_geometry_to_headers()

        self.dataset_list.clear()
        for i in range(self.n_frames):
            self.dataset_list.addItem(f"Frame {i + 1}")

        print("EDF lazy loaded:", self.n_frames, "frame(s)", self.image_shape)
        print("EDF frame 1 intensity min/max:", np.nanmin(first_image), np.nanmax(first_image))

        self.configure_azimuthal_display_defaults()

        self.update_file_information(
            "EDF",
            "-",
            self.n_frames,
            self.image_shape
        )

        self.configure_slider()
        if self.save_colorbar_checkbox.isChecked():
            self.update_image()
        else:
            self.auto_intensity()
            self.update_image()

    def open_h5(self, path):
        datasets = []

        try:
            with h5py.File(path, "r") as h5:
                def visitor(name, obj):
                    if isinstance(obj, h5py.Dataset) and obj.ndim in [2, 3]:
                        frame_axis, n_frames, image_shape = self.h5_dataset_image_info(obj.shape)
                        score = self.h5_dataset_image_score(name, obj, frame_axis, n_frames, image_shape)
                        datasets.append((name, obj.shape, obj.dtype, frame_axis, n_frames, image_shape, score))

                h5.visititems(visitor)

        except Exception as e:
            raise RuntimeError(
                "Unable to read this H5 file.\n\n"
                "If it is a compressed HDF5 file, install hdf5plugin:\n"
                "python3 -m pip install hdf5plugin\n\n"
                f"Original error:\n{e}"
            )

        if not datasets:
            raise ValueError("No 2D or 3D image dataset found in this H5 file.")

        self.h5_datasets = datasets

        for name, shape, dtype, _frame_axis, _n_frames, _image_shape, _score in datasets:
            self.dataset_list.addItem(f"{name}   {shape}")

        preferred_row = max(range(len(datasets)), key=lambda index: datasets[index][6])
        self.dataset_list.setCurrentRow(preferred_row)
        self.open_h5_dataset(datasets[preferred_row][0])

    def h5_dataset_image_info(self, shape):
        shape = tuple(int(size) for size in shape)
        if len(shape) == 2:
            return None, 1, shape
        if len(shape) == 3:
            frame_axis = int(np.argmin(shape))
            n_frames = int(shape[frame_axis])
            image_shape = tuple(size for axis, size in enumerate(shape) if axis != frame_axis)
            return frame_axis, n_frames, image_shape
        raise ValueError("Dataset must be 2D or 3D.")

    def h5_dataset_image_score(self, name, dataset, frame_axis, n_frames, image_shape):
        lower_name = str(name).lower()
        score = float(image_shape[0]) * float(image_shape[1])

        if len(tuple(dataset.shape)) == 3:
            score *= 10.0

        if min(image_shape) >= 128:
            score *= 4.0
        elif min(image_shape) < 32:
            score *= 0.05

        if any(token in lower_name for token in ["data", "image", "eiger", "detector", "pilatus"]):
            score *= 3.0
        if any(token in lower_name for token in ["mcs", "spectrum", "spectra", "counter", "monitor"]):
            score *= 0.05

        interpretation = str(dataset.attrs.get("interpretation", "")).lower()
        if "image" in interpretation:
            score *= 3.0
        if "spectrum" in interpretation:
            score *= 0.02

        if n_frames > 1:
            score *= 1.5

        return score

    def open_selected_dataset(self):
        row = self.dataset_list.currentRow()

        if row < 0:
            return

        if self.current_file and self.current_file.suffix.lower() == ".h5":
            self.reset_figure()
            self.open_h5_dataset(self.h5_datasets[row][0])

    def open_h5_dataset(self, dataset_name):
        try:
            if self.h5_file is not None:
                try:
                    self.h5_file.close()
                except Exception:
                    pass

            self.h5_file = h5py.File(self.current_file, "r")
            self.h5_dataset = self.h5_file[dataset_name]

            shape = tuple(self.h5_dataset.shape)
            dtype = self.h5_dataset.dtype
            frame_axis, n_frames, image_shape = self.h5_dataset_image_info(shape)

            self.complementary_geometry_metadata = []
            self.headers = {
                "Dataset": dataset_name,
                "Shape": str(shape),
                "Dtype": str(dtype),
            }

            for key, value in self.h5_dataset.attrs.items():
                self.headers[key] = str(value)

            for key, value in self.h5_file.attrs.items():
                self.headers[f"File attribute - {key}"] = str(value)

            self.add_matching_geometry_to_headers()

        except Exception as e:
            raise RuntimeError(f"Unable to read this H5 dataset:\n{e}")

        self.is_lazy_h5 = True
        self.images = None
        self.h5_frame_axis = frame_axis
        self.n_frames = n_frames
        self.image_shape = image_shape

        if self.h5_frame_axis is not None:
            self.headers["Frame axis"] = str(self.h5_frame_axis)
            self.headers["Number of frames"] = str(self.n_frames)

        self.current_index = 0
        self.configure_azimuthal_display_defaults()

        self.update_file_information(
            "HDF5",
            dataset_name,
            self.n_frames,
            self.image_shape
        )

        self.configure_slider()
        if self.save_colorbar_checkbox.isChecked():
            self.update_image()
        else:
            self.auto_intensity()
            self.update_image()

    # ============================================================
    # INFORMATION
    # ============================================================

    def update_file_information(self, file_type, dataset_name, n_frames, image_shape):
        self.set_toolbar_options_enabled(True)
        self.current_file_type = file_type
        self.current_dataset_name = dataset_name
        self.n_frames = n_frames
        self.image_shape = image_shape

        height, width = image_shape

        lines = [
            f"File: {self.current_file.name}",
            "",
            f"Format: {file_type}",
            f"Dataset: {dataset_name}",
            f"Frames: {n_frames}",
            f"Image size: {width} x {height}",
        ]

        if self.headers:
            lines.extend([
                "",
                "Header / Metadata:",
            ])

            for key, value in self.headers.items():
                lines.append(f"{key}: {value}")

        q_geometry = self.get_q_geometry_from_header()
        if q_geometry is not None:
            xc, yc, distance_m, pixel_x_mm, pixel_y_mm, wavelength_nm = q_geometry
            source = self.q_geometry_mode or "header"
            if self.get_header_q_values():
                source = f"{source} + header"

            lines.extend([
                "",
                "q geometry:",
                f"Source: {source}",
                f"Center: X = {xc:.6g}, Y = {yc:.6g}",
                f"Distance: {distance_m:.6g} m",
                f"Pixel: {pixel_x_mm:.6g} x {pixel_y_mm:.6g} mm",
                f"Wavelength: {wavelength_nm:.6g} nm",
            ])

        if self.complementary_geometry_metadata:
            lines.extend([
                "",
                "Complementary geometry metadata:",
            ])
            for block in self.complementary_geometry_metadata:
                source = block.get("source", "")
                source_format = block.get("format", "")
                entries = block.get("entries", [])
                copied = block.get("copied", [])
                lines.append(f"Source: {source} ({source_format})")
                if copied:
                    lines.append(f"Used to complete: {', '.join(copied)}")
                for origin, key, value in entries:
                    origin_text = f"{origin} / " if origin else ""
                    lines.append(f"{origin_text}{key}: {value}")

        self.info_text.setPlainText("\n".join(lines))

    def refresh_file_information(self):
        if (
            self.current_file is None
            or self.current_file_type is None
            or self.current_dataset_name is None
            or self.image_shape is None
        ):
            return

        self.update_file_information(
            self.current_file_type,
            self.current_dataset_name,
            self.n_frames,
            self.image_shape,
        )

    def add_matching_geometry_to_headers(self):
        if self.current_file is None:
            return

        suffix = self.current_file.suffix.lower()
        if suffix == ".h5":
            matching_path = self.current_file.with_suffix(".edf")
            source_format = "EDF"
            reader = self.read_edf_geometry_metadata
        elif suffix == ".edf":
            matching_path = self.current_file.with_suffix(".h5")
            source_format = "HDF5"
            reader = self.read_h5_geometry_metadata
        else:
            return

        if not matching_path.exists():
            return

        entries = reader(matching_path)
        geometry_entries = self.geometry_metadata_entries(entries)
        if not geometry_entries:
            return

        copied = []
        for canonical_key, aliases in self.geometry_header_groups():
            if self.header_has_alias(aliases):
                continue
            match = self.first_entry_matching_aliases(geometry_entries, aliases)
            if match is None:
                continue
            _origin, key, value = match
            self.headers[canonical_key] = value
            copied.append(canonical_key)

        self.headers["Complementary geometry source"] = matching_path.name
        self.complementary_geometry_metadata.append({
            "source": matching_path.name,
            "format": source_format,
            "entries": geometry_entries,
            "copied": copied,
        })

    def geometry_header_groups(self):
        return [
            ("Center_1", [
                "Center_1", "center_1", "Center1", "BeamCenter_1",
                "BeamCenterX", "Center_X", "CenterX", "center_x",
                "Poni1", "Beam_x", "beam_x",
            ]),
            ("Center_2", [
                "Center_2", "center_2", "Center2", "BeamCenter_2",
                "BeamCenterY", "Center_Y", "CenterY", "center_y",
                "Poni2", "Beam_y", "beam_y",
            ]),
            ("SampleDistance", [
                "SampleDistance", "sampledistance", "sample_distance",
                "Distance", "DetectorDistance", "detector_distance",
            ]),
            ("PSize_1", [
                "PSize_1", "psize_1", "PSize_X", "PixelSizeX",
                "pixel_size_x", "x_pixel_size",
            ]),
            ("PSize_2", [
                "PSize_2", "psize_2", "PSize_Y", "PixelSizeY",
                "pixel_size_y", "y_pixel_size",
            ]),
            ("WaveLength", [
                "WaveLength", "Wavelength", "wavelength", "Lambda", "lambda",
            ]),
            ("BeamEnergy", [
                "BeamEnergy", "beamenergy", "beam_energy", "Energy", "energy",
            ]),
        ]

    def geometry_aliases(self):
        aliases = set()
        for _canonical_key, group_aliases in self.geometry_header_groups():
            aliases.update(self.normalized_header_key(alias) for alias in group_aliases)
        return aliases

    def geometry_metadata_entries(self, entries):
        aliases = self.geometry_aliases()
        geometry_entries = []
        seen = set()
        for origin, key, value in entries:
            normalized = self.normalized_header_key(key)
            if normalized not in aliases:
                continue
            item_key = (str(origin), str(key), str(value))
            if item_key in seen:
                continue
            seen.add(item_key)
            geometry_entries.append((origin, key, value))
        return geometry_entries

    def header_has_alias(self, aliases):
        normalized_aliases = {self.normalized_header_key(alias) for alias in aliases}
        for key, _value in self.expanded_header_items():
            if self.normalized_header_key(key) in normalized_aliases:
                return True
        return False

    def first_entry_matching_aliases(self, entries, aliases):
        normalized_aliases = {self.normalized_header_key(alias) for alias in aliases}
        for entry in entries:
            _origin, key, _value = entry
            if self.normalized_header_key(key) in normalized_aliases:
                return entry
        return None

    def read_edf_geometry_metadata(self, path):
        try:
            import fabio

            edf = fabio.open(str(path))
            try:
                header = dict(edf.header)
            finally:
                try:
                    edf.close()
                except Exception:
                    pass

            try:
                with open(path, "rb") as handle:
                    header_bytes = handle.read(65536)
                    header_text = header_bytes.decode("latin-1", errors="ignore")
                parsed = parse_header_text(header_text)
                header.update(parsed)
            except Exception:
                pass

            return [("", str(key), str(value)) for key, value in header.items()]
        except Exception:
            return []

    def read_h5_geometry_metadata(self, path):
        entries = []

        def add_entry(origin, key, value):
            try:
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                elif hasattr(value, "item"):
                    value = value.item()
            except Exception:
                pass
            entries.append((origin, str(key), str(value)))

        try:
            with h5py.File(path, "r") as h5:
                for key, value in h5.attrs.items():
                    add_entry("/", key, value)

                def visitor(name, obj):
                    if obj.attrs:
                        origin = f"/{name}"
                        for key, value in obj.attrs.items():
                            add_entry(origin, key, value)

                h5.visititems(visitor)
        except Exception:
            return []

        return entries

    def is_azimuthal_processed_file(self):
        if self.current_file is not None:
            stem = self.current_file.stem.lower()
            if "_azim" in stem or stem.endswith("azim"):
                return True

        for key, value in self.headers.items():
            key_text = str(key).lower()
            value_text = str(value).lower()
            if key_text == "processing" and "azim" in value_text:
                return True

        return False

    def configure_azimuthal_display_defaults(self):
        self.is_azimuthal_image = self.is_azimuthal_processed_file()

    # ============================================================
    # IMAGE DISPLAY
    # ============================================================

    def configure_slider(self):
        n = self.total_frame_count()

        self.frame_start_spin.blockSignals(True)
        self.frame_end_spin.blockSignals(True)

        self.frame_start_spin.setMinimum(1)
        self.frame_start_spin.setMaximum(n)
        self.frame_start_spin.setValue(1)

        self.frame_end_spin.setMinimum(1)
        self.frame_end_spin.setMaximum(n)
        self.frame_end_spin.setValue(n)

        self.frame_start_spin.blockSignals(False)
        self.frame_end_spin.blockSignals(False)

        self.frame_slider.blockSignals(True)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(n - 1)
        self.frame_slider.setValue(0)
        self.frame_slider.blockSignals(False)

        self.frame_label.setText(f"1 / {n}")
        self.update_frame_navigation_state()

    def update_frame_slider_range(self):
        n = self.total_frame_count()
        if n <= 0:
            return

        start = self.frame_start_spin.value() - 1
        end = self.frame_end_spin.value() - 1

        if start > end:
            sender = self.sender()
            if sender is self.frame_start_spin:
                self.frame_end_spin.setValue(self.frame_start_spin.value())
                end = start
            else:
                self.frame_start_spin.setValue(self.frame_end_spin.value())
                start = end

        self.frame_slider.setMinimum(start)
        self.frame_slider.setMaximum(end)

        if self.current_index < start:
            self.frame_slider.setValue(start)
        elif self.current_index > end:
            self.frame_slider.setValue(end)
        else:
            self.update_frame_navigation_state()

    def update_frame_navigation_state(self):
        total = self.total_frame_count()
        can_navigate = total > 1
        self.frame_start_spin.setEnabled(can_navigate)
        self.frame_end_spin.setEnabled(can_navigate)
        self.frame_slider.setEnabled(can_navigate)
        self.previous_button.setEnabled(can_navigate and self.current_index > self.frame_slider.minimum())
        self.next_button.setEnabled(can_navigate and self.current_index < self.frame_slider.maximum())

    def total_frame_count(self):
        if self.is_lazy_h5 or self.is_lazy_edf:
            return max(0, int(self.n_frames or 0))
        if self.images is not None:
            return int(self.images.shape[0])
        return 0

    def read_edf_frame(self, frame_index):
        try:
            import fabio
        except ImportError:
            return None

        if self.edf_path is None:
            return None

        edf = fabio.open(str(self.edf_path))
        try:
            nframes = int(getattr(edf, "nframes", 1) or 1)
            frame_index = max(0, min(int(frame_index), nframes - 1))
            if nframes <= 1:
                return np.array(edf.data, dtype=float)
            frame = edf.getframe(frame_index)
            return np.array(frame.data, dtype=float)
        finally:
            try:
                edf.close()
            except Exception:
                pass

    def get_current_image(self):
        if self.is_lazy_edf:
            return self.read_edf_frame(self.current_index)

        if self.is_lazy_h5:
            if self.h5_dataset is None:
                return None

            if self.h5_dataset.ndim == 2:
                return np.array(self.h5_dataset, dtype=float)

            frame_axis = 0 if self.h5_frame_axis is None else self.h5_frame_axis
            if frame_axis == 0:
                return np.array(self.h5_dataset[self.current_index, :, :], dtype=float)
            if frame_axis == 1:
                return np.array(self.h5_dataset[:, self.current_index, :], dtype=float)
            return np.array(self.h5_dataset[:, :, self.current_index], dtype=float)

        if self.images is None:
            return None

        return self.images[self.current_index]

    def prepare_display_image(self, img):
        img = np.array(img, dtype=float)
        img[img > 4e9] = np.nan

        if self.log_checkbox.isChecked():
            img = np.log10(np.clip(img, 0, None) + 1)

        if getattr(self, "nan_to_zero_checkbox", None) is not None and self.nan_to_zero_checkbox.isChecked():
            img = np.nan_to_num(img, nan=0.0, copy=False)

        return img

    def get_center_from_header(self):
        if not self.headers:
            return None

        possible_x_keys = [
            "Center_1",
            "center_1",
            "Center1",
            "BeamCenter_1",
            "BeamCenterX",
            "Center_X",
            "CenterX",
            "center_x",
            "Poni1",
            "Beam_x",
            "beam_x"
        ]

        possible_y_keys = [
            "Center_2",
            "center_2",
            "Center2",
            "BeamCenter_2",
            "BeamCenterY",
            "Center_Y",
            "CenterY",
            "center_y",
            "Poni2",
            "Beam_y",
            "beam_y"
        ]

        x = self.header_float_by_alias(possible_x_keys)
        y = self.header_float_by_alias(possible_y_keys)

        if x is None or y is None:
            return None

        return x, y

    def normalized_header_key(self, key):
        text = str(key)
        for prefix in ("File attribute - ", "edf_header_"):
            if text.startswith(prefix):
                text = text[len(prefix):]
        return re.sub(r"[^a-z0-9]", "", text.lower())

    def expanded_header_items(self):
        items = list(self.headers.items())

        for json_key in ("edf_header_json", "source_h5_attrs_json"):
            raw_json = self.headers.get(json_key)
            if not raw_json:
                continue
            try:
                payload = json.loads(str(raw_json))
            except (TypeError, json.JSONDecodeError):
                continue

            if json_key == "edf_header_json" and isinstance(payload, dict):
                items.extend(payload.items())
            elif json_key == "source_h5_attrs_json" and isinstance(payload, dict):
                for attrs in payload.values():
                    if isinstance(attrs, dict):
                        items.extend(attrs.items())

        return items

    def header_float_by_alias(self, aliases):
        normalized_aliases = {self.normalized_header_key(alias) for alias in aliases}

        for key, value in self.expanded_header_items():
            if self.normalized_header_key(key) not in normalized_aliases:
                continue
            parsed = self.parse_header_float(value)
            if parsed is not None:
                return parsed

        return None

    def parse_header_float(self, value):
        text = str(value).strip().replace(",", ".")
        if not text:
            return None

        try:
            return float(text)
        except ValueError:
            pass

        match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)
        if match is None:
            return None

        try:
            return float(match.group(0))
        except ValueError:
            return None

    def get_header_float(self, *keys):
        if not self.headers:
            return None

        return self.header_float_by_alias(keys)

    def get_header_q_values(self):
        center = self.get_center_from_header()
        distance_m = self.get_header_float(
            "SampleDistance",
            "sampledistance",
            "sample_distance",
            "Distance",
            "DetectorDistance",
            "detector_distance",
        )
        pixel_x = self.get_header_float(
            "PSize_1",
            "psize_1",
            "PSize_X",
            "PixelSizeX",
            "pixel_size_x",
            "x_pixel_size",
        )
        pixel_y = self.get_header_float(
            "PSize_2",
            "psize_2",
            "PSize_Y",
            "PixelSizeY",
            "pixel_size_y",
            "y_pixel_size",
        )
        wavelength = self.get_header_float(
            "WaveLength",
            "Wavelength",
            "wavelength",
            "Lambda",
            "lambda",
        )
        if wavelength is None:
            beam_energy = self.get_header_float(
                "BeamEnergy",
                "beamenergy",
                "beam_energy",
                "Energy",
                "energy",
            )
            wavelength = self.beam_energy_to_wavelength_a(beam_energy)

        values = {}
        if center is not None:
            values["xc"], values["yc"] = center
        if distance_m is not None:
            values["distance_m"] = distance_m
        if pixel_x is not None:
            values["pixel_x_mm"] = pixel_x * 1000.0 if pixel_x < 1e-3 else pixel_x
        if pixel_y is not None:
            values["pixel_y_mm"] = pixel_y * 1000.0 if pixel_y < 1e-3 else pixel_y
        if wavelength is not None:
            if wavelength < 1e-6:
                values["wavelength_a"] = wavelength * 1e10
            elif wavelength < 0.5:
                values["wavelength_a"] = wavelength * 10.0
            else:
                values["wavelength_a"] = wavelength

        return values

    def beam_energy_to_wavelength_a(self, beam_energy):
        if beam_energy is None or beam_energy <= 0:
            return None

        # ID02 headers commonly store beamenergy in eV; keV values are also accepted.
        energy_ev = beam_energy * 1000.0 if beam_energy < 100.0 else beam_energy
        return 12398.419843320026 / energy_ev

    def wavelength_to_nm(self, wavelength):
        if wavelength < 1e-6:
            return wavelength * 1e9
        if wavelength >= 0.5:
            return wavelength * 0.1
        return wavelength

    def get_header_q_geometry(self):
        if self.is_azimuthal_image:
            return None

        center = self.get_center_from_header()
        if center is None:
            return None

        xc, yc = center

        distance_m = self.get_header_float(
            "SampleDistance",
            "sampledistance",
            "sample_distance",
            "Distance",
            "DetectorDistance",
            "detector_distance",
        )

        pixel_x = self.get_header_float(
            "PSize_1",
            "psize_1",
            "PSize_X",
            "PixelSizeX",
            "pixel_size_x",
            "x_pixel_size",
        )

        pixel_y = self.get_header_float(
            "PSize_2",
            "psize_2",
            "PSize_Y",
            "PixelSizeY",
            "pixel_size_y",
            "y_pixel_size",
        )

        wavelength = self.get_header_float(
            "WaveLength",
            "Wavelength",
            "wavelength",
            "Lambda",
            "lambda",
        )
        if wavelength is None:
            beam_energy = self.get_header_float(
                "BeamEnergy",
                "beamenergy",
                "beam_energy",
                "Energy",
                "energy",
            )
            wavelength = self.beam_energy_to_wavelength_a(beam_energy)

        if distance_m is None or pixel_x is None or pixel_y is None or wavelength is None:
            return None

        # Pixel sizes may come from headers in meters, while the fallback is in mm.
        # Values below 1e-3 are assumed to be meters and converted to mm.
        pixel_x_mm = pixel_x * 1000.0 if pixel_x < 1e-3 else pixel_x
        pixel_y_mm = pixel_y * 1000.0 if pixel_y < 1e-3 else pixel_y

        wavelength_nm = self.wavelength_to_nm(wavelength)

        if distance_m <= 0 or pixel_x_mm <= 0 or pixel_y_mm <= 0 or wavelength_nm <= 0:
            return None

        return xc, yc, distance_m, pixel_x_mm, pixel_y_mm, wavelength_nm

    def get_preset_q_geometry(self):
        if self.is_azimuthal_image:
            return None

        geometry = self.q_geometry_values_for_mode()
        if not geometry:
            return None

        xc = geometry["xc"]
        yc = geometry["yc"]
        distance_m = geometry["distance_m"]
        pixel_x_mm = geometry["pixel_x_mm"]
        pixel_y_mm = geometry["pixel_y_mm"]
        wavelength_nm = self.wavelength_to_nm(geometry["wavelength_a"])

        if distance_m <= 0 or pixel_x_mm <= 0 or pixel_y_mm <= 0 or wavelength_nm <= 0:
            return None

        return xc, yc, distance_m, pixel_x_mm, pixel_y_mm, wavelength_nm

    def q_geometry_values_for_mode(self):
        if self.is_azimuthal_image:
            return None

        geometry = self.preset_q_geometry(self.q_geometry_mode)
        if geometry is None:
            return None

        geometry = dict(geometry)
        header_values = self.get_header_q_values()
        if header_values:
            geometry.update(header_values)

        return geometry

    def get_q_geometry_from_header(self):
        return self.get_preset_q_geometry()

    def calculate_q_at_pixel(self, x_index, y_index):
        if self.is_azimuthal_image:
            return self.calculate_azimuthal_q_at_pixel(x_index)

        geometry = self.get_q_geometry_from_header()
        if geometry is None:
            return None

        xc, yc, distance_m, pixel_x_mm, pixel_y_mm, wavelength_nm = geometry

        dx_px = float(x_index) - float(xc)
        dy_px = float(y_index) - float(yc)

        dx_m = dx_px * pixel_x_mm * 1e-3
        dy_m = dy_px * pixel_y_mm * 1e-3
        r_m = np.sqrt(dx_m ** 2 + dy_m ** 2)

        two_theta = np.arctan2(r_m, distance_m)
        q_nm = (4.0 * np.pi / wavelength_nm) * np.sin(two_theta / 2.0)

        return q_nm

    def calculate_azimuthal_angle_at_pixel(self, y_index):
        if self.raw_current_img is None:
            return None

        ny, _nx = self.raw_current_img.shape
        if ny <= 0:
            return None

        return ((float(y_index) + 0.5) / float(ny)) * 360.0

    def calculate_azimuthal_q_at_pixel(self, x_index):
        values = self.get_header_q_values()
        required_keys = ("xc", "distance_m", "pixel_x_mm", "wavelength_a")
        if not all(key in values for key in required_keys):
            return None

        xc = values["xc"]
        distance_m = values["distance_m"]
        pixel_x_mm = values["pixel_x_mm"]
        wavelength_nm = self.wavelength_to_nm(values["wavelength_a"])

        if distance_m <= 0 or pixel_x_mm <= 0 or wavelength_nm <= 0:
            return None

        dx_px = float(x_index) - float(xc)
        dx_m = abs(dx_px) * pixel_x_mm * 1e-3
        two_theta = np.arctan2(dx_m, distance_m)
        return (4.0 * np.pi / wavelength_nm) * np.sin(two_theta / 2.0)

    def draw_center_cross(self):
        for artist in self.center_artists:
            try:
                artist.remove()
            except Exception:
                pass

        self.center_artists = []

        if self.is_azimuthal_image:
            return

        values = self.q_geometry_values_for_mode()
        center = None
        if values is not None and values.get("xc") is not None and values.get("yc") is not None:
            center = values["xc"], values["yc"]

        if center is None:
            return

        xc, yc = center

        vline = self.ax.axvline(
            xc,
            color="red",
            linewidth=1,
            alpha=0.9
        )

        hline = self.ax.axhline(
            yc,
            color="red",
            linewidth=1,
            alpha=0.9
        )

        point = self.ax.plot(
            xc,
            yc,
            marker="o",
            markersize=5,
            markerfacecolor="white",
            markeredgecolor="red",
            markeredgewidth=1
        )[0]

        self.center_artists = [vline, hline, point]

    def update_image(self):
        img = self.get_current_image()

        if img is None:
            return

        self.raw_current_img = np.array(img, dtype=float).copy()
        self.display_img = self.prepare_display_image(img)
        self.image_axis_bounds = None

        aspect = "equal" if self.keep_ratio_checkbox.isChecked() else "auto"

        if self.image_artist is None:
            self.image_artist = self.ax.imshow(
                self.display_img,
                cmap=self.current_colormap(),
                origin="upper",
                vmin=self.vmin_spin.value(),
                vmax=self.vmax_spin.value(),
                aspect=aspect
            )

            self.ax.set_axis_off()
            self.ax.set_aspect(aspect)

            self.colorbar = self.fig.colorbar(
                self.image_artist,
                ax=self.ax,
                fraction=0.046,
                pad=0.04
            )

            self.fig.subplots_adjust(
                left=0.02,
                right=0.92,
                top=0.98,
                bottom=0.02
            )

        else:
            self.image_artist.set_data(self.display_img)
            self.image_artist.set_clim(
                self.vmin_spin.value(),
                self.vmax_spin.value()
            )
            self.ax.set_aspect(aspect)

        # Restore zoom/pan if needed
        if self.keep_zoom_checkbox.isChecked() and self._saved_xlim is not None and self._saved_ylim is not None:
            self.ax.set_xlim(self._saved_xlim)
            self.ax.set_ylim(self._saved_ylim)
            self.constrain_current_image_axes()

        self.ax.set_title("")

        self.draw_center_cross()

        total = self.total_frame_count()
        self.frame_label.setText(f"{self.current_index + 1} / {total}")

        self.ax.set_autoscale_on(False)
        if hasattr(self, "toolbar"):
            self.toolbar.push_current()
            self.toolbar.set_history_buttons()
        self.canvas.draw_idle()

    def auto_intensity(self):
        display_img = self.display_image_for_auto_intensity()
        if display_img is None:
            return

        finite = display_img[np.isfinite(display_img)]

        if finite.size == 0:
            return

        lower_percentile, upper_percentile = self.auto_intensity_percentiles()
        vmin = float(np.nanpercentile(finite, lower_percentile))
        vmax = float(np.nanpercentile(finite, upper_percentile))

        self.intensity_min = float(np.nanmin(finite))
        self.intensity_max = float(np.nanmax(finite))

        self.vmin_spin.blockSignals(True)
        self.vmax_spin.blockSignals(True)
        self.min_slider.blockSignals(True)
        self.max_slider.blockSignals(True)

        self.vmin_spin.setValue(vmin)
        self.vmax_spin.setValue(vmax)

        self.min_slider.setValue(self.value_to_slider(vmin))
        self.max_slider.setValue(self.value_to_slider(vmax))

        self.vmin_spin.blockSignals(False)
        self.vmax_spin.blockSignals(False)
        self.min_slider.blockSignals(False)
        self.max_slider.blockSignals(False)

        if self.save_colorbar_checkbox.isChecked():
            self.settings.setValue("view/vmin", vmin)
            self.settings.setValue("view/vmax", vmax)
        self.update_image()

    def display_image_for_auto_intensity(self):
        img = self.get_current_image()
        if img is None:
            return None
        return self.prepare_display_image(img)

    def auto_intensity_percentiles(self):
        return 1.0, 99.0

    def value_to_slider(self, value):
        if self.intensity_max == self.intensity_min:
            return 0

        slider_value = int(
            1000
            * (value - self.intensity_min)
            / (self.intensity_max - self.intensity_min)
        )

        return max(0, min(1000, slider_value))

    def slider_to_value(self, value):
        return self.intensity_min + (
            value / 1000
        ) * (self.intensity_max - self.intensity_min)

    def vertical_sliders_changed(self):
        vmin = self.slider_to_value(self.min_slider.value())
        vmax = self.slider_to_value(self.max_slider.value())

        if vmin >= vmax:
            return

        if self.save_colorbar_checkbox.isChecked():
            self.settings.setValue("view/vmin", vmin)
            self.settings.setValue("view/vmax", vmax)

        self.vmin_spin.blockSignals(True)
        self.vmax_spin.blockSignals(True)

        self.vmin_spin.setValue(vmin)
        self.vmax_spin.setValue(vmax)

        self.vmin_spin.blockSignals(False)
        self.vmax_spin.blockSignals(False)

        self.update_image()

    def spin_intensity_changed(self):
        vmin = self.vmin_spin.value()
        vmax = self.vmax_spin.value()

        if vmin >= vmax:
            return

        current_min = min(vmin, self.intensity_min)
        current_max = max(vmax, self.intensity_max)

        if current_max > current_min:
            self.intensity_min = current_min
            self.intensity_max = current_max

        if self.save_colorbar_checkbox.isChecked():
            self.settings.setValue("view/vmin", vmin)
            self.settings.setValue("view/vmax", vmax)

        self.min_slider.blockSignals(True)
        self.max_slider.blockSignals(True)

        self.min_slider.setValue(self.value_to_slider(vmin))
        self.max_slider.setValue(self.value_to_slider(vmax))

        self.min_slider.blockSignals(False)
        self.max_slider.blockSignals(False)

        self.update_image()

    # ============================================================
    # FRAME NAVIGATION
    # ============================================================

    def slider_changed(self, value):
        self.current_index = value
        self.update_image()
        self.update_frame_navigation_state()


    def previous_image(self):
        self.frame_slider.setValue(
            max(self.frame_slider.minimum(), self.current_index - 1)
        )

    def next_image(self):
        self.frame_slider.setValue(
            min(self.frame_slider.maximum(), self.current_index + 1)
        )


    # ============================================================
    # CURSOR READOUT
    # ============================================================

    def on_mouse_move(self, event):
        if self.pan_image_from_motion(event):
            return
        if self.raw_current_img is None or event.inaxes != self.ax:
            self.cursor_label.setText("x = - | y = - | q = - | I = -")
            return

        if event.xdata is None or event.ydata is None:
            self.cursor_label.setText("x = - | y = - | q = - | I = -")
            return

        x_index = int(round(event.xdata))
        y_index = int(round(event.ydata))

        ny, nx = self.raw_current_img.shape

        if not (0 <= x_index < nx and 0 <= y_index < ny):
            self.cursor_label.setText("x = - | y = - | q = - | I = -")
            return

        value = self.raw_current_img[y_index, x_index]

        if np.isnan(value):
            value_text = "NaN"
        elif np.isposinf(value):
            value_text = "+Inf"
        elif np.isneginf(value):
            value_text = "-Inf"
        else:
            value_text = f"{value:.8g}"

        q_value = self.calculate_q_at_pixel(x_index, y_index)
        q_text = "-" if q_value is None else f"{q_value:.6g} nm⁻¹"

        if self.is_azimuthal_image:
            angle_value = self.calculate_azimuthal_angle_at_pixel(y_index)
            angle_text = "-" if angle_value is None else f"{angle_value:.3f}°"
            self.cursor_label.setText(
                f"x = {x_index + 1} | y = {y_index + 1} | angle = {angle_text} | q = {q_text} | I = {value_text}"
            )
        else:
            self.cursor_label.setText(
                f"x = {x_index + 1} | y = {y_index + 1} | q = {q_text} | I = {value_text}"
            )

    def on_mouse_leave(self, event):
        self.cursor_label.setText("x = - | y = - | q = - | I = -")

    def open_annotation_window(self):
        raw_image = self.get_current_image()
        if raw_image is None:
            QMessageBox.information(self, "No image", "No image is currently loaded.")
            return

        raw_image = np.asarray(raw_image, dtype=float)
        display_image = self.prepare_display_image(raw_image)
        vmin, vmax = self.display_limits_for_save(display_image)
        title = "Annotation"
        if self.current_file is not None:
            title = f"Annotation - {self.current_file.name}"
        annotation_key = self.annotation_storage_key()

        dialog = PlaneAnnotationDialog(
            self,
            raw_image,
            display_image,
            vmin,
            vmax,
            self.calculate_q_at_pixel,
            title,
            annotation_key,
        )
        self.annotation_dialog = dialog
        dialog.show()

    def open_composite_window(self):
        dialog = getattr(self, "composite_dialog", None)
        if dialog is None or not dialog.isVisible():
            dialog = CompositeImageDialog(self)
            self.composite_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def open_film_window(self):
        dialog = getattr(self, "film_dialog", None)
        if dialog is None or not dialog.isVisible():
            dialog = FilmDialog(self)
            self.film_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def annotation_storage_key(self):
        if self.current_file is None:
            return None

        parts = [str(Path(self.current_file).expanduser().resolve())]
        if self.current_dataset_name:
            parts.append(f"dataset={self.current_dataset_name}")
        total_frames = self.total_frame_count()
        if total_frames > 1:
            parts.append(f"frame={self.current_index + 1}")
        return "#".join(parts)

    # ============================================================
    # SAVE
    # ============================================================

    def save_png_image_only(self):
        if self.display_img is None or self.current_file is None:
            QMessageBox.information(
                self,
                "No image",
                "No image is currently loaded."
            )
            return

        frame_suffix = ""
        total_frames = self.total_frame_count()
        if total_frames > 1:
            frame_suffix = f"_frame{self.current_index + 1:04d}"

        suggested_path = self.current_file.parent / f"{self.current_file.stem}{frame_suffix}.png"

        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save image",
            str(suggested_path),
            "PNG image (*.png);;TIFF image (*.tif);;PDF image (*.pdf);;EDF image (*.edf);;HDF5 image (*.h5);;All files (*)"
        )

        if not path:
            return

        lower_path = path.lower()
        if "EDF" in selected_filter and not lower_path.endswith(".edf"):
            path += ".edf"
        elif "HDF5" in selected_filter and not lower_path.endswith((".h5", ".hdf5")):
            path += ".h5"
        elif "PDF" in selected_filter and not lower_path.endswith(".pdf"):
            path += ".pdf"
        elif "TIFF" in selected_filter and not lower_path.endswith((".tif", ".tiff")):
            path += ".tif"
        elif not lower_path.endswith((".png", ".tif", ".tiff", ".pdf", ".edf", ".h5", ".hdf5")):
            path += ".png"

        try:
            lower_path = path.lower()
            if lower_path.endswith(".edf"):
                self.save_current_frame_as_edf(path)
            elif lower_path.endswith((".h5", ".hdf5")):
                self.save_current_frame_as_h5(path)
            elif lower_path.endswith(".pdf"):
                self.save_current_display_as_pdf(path)
            else:
                vmin, vmax = self.display_limits_for_save(self.display_img)
                plt.imsave(
                    path,
                    self.display_img,
                    cmap=self.current_colormap(),
                    vmin=vmin,
                    vmax=vmax,
                    origin="upper"
                )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Save error",
                f"Unable to save image:\n{e}"
            )

    def show_all_colormap_preview(self):
        if self.display_img is None or self.current_file is None:
            QMessageBox.information(
                self,
                "No image",
                "No image is currently loaded."
            )
            return

        try:
            image = self.current_display_image_for_save()
            vmin, vmax = self.display_limits_for_save(image)
            dialog = AllColormapPreviewDialog(
                self,
                image,
                self.colormap_options,
                vmin,
                vmax,
                source_file=self.current_file,
                frame_index=self.current_index,
                total_frames=self.total_frame_count(),
            )
            dialog.exec()
        except Exception as error:
            QMessageBox.critical(
                self,
                "Preview colors error",
                f"Unable to preview color images:\n{error}"
            )

    def display_limits_for_save(self, image):
        vmin = self.vmin_spin.value()
        vmax = self.vmax_spin.value()

        if vmax > vmin:
            return vmin, vmax

        finite = np.asarray(image, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return 0.0, 1.0

        vmin = float(np.nanpercentile(finite, 1))
        vmax = float(np.nanpercentile(finite, 99))
        if vmax <= vmin:
            delta = abs(vmin) * 0.01 or 1.0
            vmin -= delta
            vmax += delta

        return vmin, vmax

    def current_raw_image_for_save(self):
        image = self.get_current_image()
        if image is None:
            image = self.raw_current_img
        if image is None:
            raise ValueError("No raw image data is available.")

        return np.asarray(image, dtype=float)

    def current_display_image_for_save(self):
        image = self.current_raw_image_for_save()
        return self.prepare_display_image(image)

    def save_current_display_as_pdf(self, path):
        image = self.current_display_image_for_save()
        vmin, vmax = self.display_limits_for_save(image)
        ny, nx = image.shape
        dpi = 150
        width = max(nx / dpi, 1.0)
        height = max(ny / dpi, 1.0)

        fig = Figure(figsize=(width, height), dpi=dpi)
        canvas = FigureCanvas(fig)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        ax.imshow(
            image,
            cmap=self.current_colormap(),
            origin="upper",
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )
        canvas.draw()
        fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0)

    def metadata_for_saved_image(self):
        metadata = {
            "SourceFile": self.current_file.name if self.current_file is not None else "",
            "SourceFrame": str(self.current_index),
            "SavedBy": "LRPhoton View",
        }

        for key, value in self.headers.items():
            clean_key = str(key).replace(" ", "_")
            metadata[clean_key] = str(value)

        geometry = self.get_q_geometry_from_header()
        if geometry is not None:
            xc, yc, distance_m, pixel_x_mm, pixel_y_mm, wavelength_nm = geometry
            metadata.update({
                "Center_1": f"{xc:.12g}",
                "Center_2": f"{yc:.12g}",
                "SampleDistance": f"{distance_m:.12g}",
                "PSize_1": f"{pixel_x_mm / 1000.0:.12g}",
                "PSize_2": f"{pixel_y_mm / 1000.0:.12g}",
                "WaveLength": f"{wavelength_nm * 1e-9:.12g}",
            })

        return metadata

    def save_current_frame_as_h5(self, path):
        image = self.current_raw_image_for_save()

        metadata = self.metadata_for_saved_image()
        with h5py.File(path, "w") as h5:
            dataset = h5.create_dataset(
                "/entry_0000/instrument/detector/data",
                data=np.asarray(image, dtype=np.float32),
                compression="gzip",
            )
            for key, value in metadata.items():
                dataset.attrs[key] = value

    def save_current_frame_as_edf(self, path):
        image = self.current_raw_image_for_save().astype("<f4", copy=False)
        ny, nx = image.shape
        metadata = self.metadata_for_saved_image()
        metadata.update({
            "HeaderID": "EH:000001:000000:000000",
            "Image": "0",
            "ByteOrder": "LowByteFirst",
            "DataType": "FloatValue",
            "Dim_1": str(nx),
            "Dim_2": str(ny),
            "Size": str(nx * ny * 4),
            "EDF_BinarySize": str(nx * ny * 4),
        })

        header_size = 1024
        while True:
            metadata["EDF_HeaderSize"] = str(header_size)
            header_text = "{\n"
            for key, value in metadata.items():
                header_text += f"{key} = {value} ;\n"
            header_bytes = header_text.encode("latin-1", errors="ignore")
            closing_bytes = b"}\n"
            if len(header_bytes) + len(closing_bytes) <= header_size:
                break
            header_size += 1024

        padding = b" " * (header_size - len(header_bytes) - len(closing_bytes))
        header_bytes = header_bytes + padding + closing_bytes

        with open(path, "wb") as file:
            file.write(header_bytes)
            file.write(image.tobytes(order="C"))

    # ============================================================
    # TRACKPAD / MOUSE NAVIGATION
    # ============================================================

    def qpoint_to_data_pos(self, qpoint):
        try:
            x_widget = float(qpoint.x())
            y_widget = float(qpoint.y())
        except Exception:
            x_widget = self.canvas.width() / 2
            y_widget = self.canvas.height() / 2

        bbox = self.ax.get_window_extent()
        x_fig = bbox.x0 + (x_widget / max(self.canvas.width(), 1)) * bbox.width
        y_fig = bbox.y1 - (y_widget / max(self.canvas.height(), 1)) * bbox.height

        xdata, ydata = self.ax.transData.inverted().transform((x_fig, y_fig))

        if not np.isfinite(xdata) or not np.isfinite(ydata):
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            xdata = (xlim[0] + xlim[1]) / 2
            ydata = (ylim[0] + ylim[1]) / 2

        return xdata, ydata

    def zoom_at_qpoint(self, qpoint, zoom_factor):
        if zoom_factor <= 0 or self.image_artist is None:
            return

        xdata, ydata = self.qpoint_to_data_pos(qpoint)
        self.zoom_at_data_position(xdata, ydata, zoom_factor)

    def zoom_at_data_position(self, xdata, ydata, zoom_factor):
        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()

        if x_max == x_min or y_max == y_min:
            return

        new_width = (x_max - x_min) * zoom_factor
        new_height = (y_max - y_min) * zoom_factor

        rel_x = (xdata - x_min) / (x_max - x_min)
        rel_y = (ydata - y_min) / (y_max - y_min)

        self.ax.set_xlim(
            xdata - new_width * rel_x,
            xdata + new_width * (1.0 - rel_x),
        )
        self.ax.set_ylim(
            ydata - new_height * rel_y,
            ydata + new_height * (1.0 - rel_y),
        )
        self.constrain_current_image_axes()

        if self.keep_zoom_checkbox.isChecked():
            self._saved_xlim = self.ax.get_xlim()
            self._saved_ylim = self.ax.get_ylim()

        self.ax.set_autoscale_on(False)
        if hasattr(self, "toolbar"):
            self.toolbar.push_current()
            self.toolbar.set_history_buttons()
        self.canvas.draw_idle()

    def reset_image_view(self):
        if self.raw_current_img is None:
            return

        ny, nx = self.raw_current_img.shape
        self.ax.set_xlim(-0.5, nx - 0.5)
        self.ax.set_ylim(ny - 0.5, -0.5)

        if self.keep_zoom_checkbox.isChecked():
            self._saved_xlim = self.ax.get_xlim()
            self._saved_ylim = self.ax.get_ylim()

        self.ax.set_autoscale_on(False)
        self._trackpad_view_pushed = False
        if hasattr(self, "toolbar"):
            self.toolbar.set_history_buttons()
        self.canvas.draw_idle()

    def constrain_current_image_axes(self):
        if self.raw_current_img is None:
            return

        bounds = getattr(self, "image_axis_bounds", None)
        if bounds is not None:
            x_bounds, y_bounds = bounds
            constrain_image_axes(self.ax, x_bounds=x_bounds, y_bounds=y_bounds)
            return

        constrain_image_axes(self.ax, self.raw_current_img.shape)

    def pan_by_trackpad(self, dx, dy):
        if self.image_artist is None:
            return

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        xspan = xlim[1] - xlim[0]
        yspan = ylim[1] - ylim[0]

        shift_x = -dx * xspan * 0.08
        shift_y = dy * yspan * 0.08

        self.ax.set_xlim(xlim[0] + shift_x, xlim[1] + shift_x)
        self.ax.set_ylim(ylim[0] + shift_y, ylim[1] + shift_y)
        self.constrain_current_image_axes()

        if self.keep_zoom_checkbox.isChecked():
            self._saved_xlim = self.ax.get_xlim()
            self._saved_ylim = self.ax.get_ylim()

        self.ax.set_autoscale_on(False)
        if hasattr(self, "toolbar"):
            self.toolbar.push_current()
            self.toolbar.set_history_buttons()
        self.canvas.draw_idle()

    def on_scroll_zoom(self, event):
        if self.image_artist is None or event.inaxes != self.ax:
            return

        if self.toolbar_interaction_active():
            return

        if event.xdata is None or event.ydata is None:
            return

        zoom_factor = 0.85 if event.button == "up" else 1.18
        self.zoom_at_data_position(event.xdata, event.ydata, zoom_factor)

    def on_mouse_press(self, event):
        if self.image_artist is None or event.inaxes != self.ax:
            return

        if self.toolbar_interaction_active():
            return

        if event.button != 1:
            return

        self._is_panning = True
        self._pan_start_event = event
        self._pan_start_xlim = self.ax.get_xlim()
        self._pan_start_ylim = self.ax.get_ylim()
        self.canvas.setCursor(Qt.ClosedHandCursor)

    def on_mouse_release(self, event):
        if not self._is_panning:
            return

        self._is_panning = False
        self._pan_start_event = None
        self._pan_start_xlim = None
        self._pan_start_ylim = None
        self._trackpad_view_pushed = False
        if hasattr(self, "toolbar"):
            self.toolbar.push_current()
            self.toolbar.set_history_buttons()
        self.canvas.setCursor(Qt.ArrowCursor)

    def pan_image_from_motion(self, event):
        if self.toolbar_interaction_active():
            return False

        if not self._is_panning:
            return False

        if self._pan_start_event is None or event.xdata is None or event.ydata is None:
            return False

        if self._pan_start_event.xdata is None or self._pan_start_event.ydata is None:
            return False

        dx = self._pan_start_event.xdata - event.xdata
        dy = self._pan_start_event.ydata - event.ydata

        x0, x1 = self._pan_start_xlim
        y0, y1 = self._pan_start_ylim

        if hasattr(self, "toolbar") and not self._trackpad_view_pushed:
            self.toolbar.push_current()
            self._trackpad_view_pushed = True

        self.ax.set_xlim(x0 + dx, x1 + dx)
        self.ax.set_ylim(y0 + dy, y1 + dy)
        self.constrain_current_image_axes()

        if self.keep_zoom_checkbox.isChecked():
            self._saved_xlim = self.ax.get_xlim()
            self._saved_ylim = self.ax.get_ylim()

        self.ax.set_autoscale_on(False)
        if hasattr(self, "toolbar"):
            self.toolbar.set_history_buttons()
        self.canvas.draw_idle()
        return True

    def toolbar_interaction_active(self):
        toolbar = getattr(self, "toolbar", None)
        if toolbar is None:
            return False

        mode = getattr(toolbar, "mode", "")
        try:
            return bool(mode)
        except Exception:
            return str(mode).strip() != ""

        # (rest of code continues unchanged)
