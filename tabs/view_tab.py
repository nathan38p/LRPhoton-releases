import fnmatch
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
import matplotlib.pyplot as plt

from PySide6.QtCore import Qt, QEvent, QSettings, QSize, Signal

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
    QDoubleSpinBox,
    QSplitter,
    QMessageBox,
    QGroupBox,
    QSpinBox,
    QStyle,
    QDialog,
    QDialogButtonBox,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class ImageOnlyToolbar(NavigationToolbar):

    def __init__(self, canvas, parent):

        super().__init__(canvas, parent)

        save_icon = parent.style().standardIcon(QStyle.SP_DialogSaveButton)
        save_image_action = QAction(save_icon, "Save image only", self)
        save_image_action.setToolTip("Save image only")
        save_image_action.triggered.connect(parent.save_png_image_only)

        self.addSeparator()

        self.addAction(save_image_action)

    def save_figure(self, *args):

        view_tab = self.parent()

        if hasattr(view_tab, "save_png_image_only"):

            view_tab.save_png_image_only()

        else:

            super().save_figure(*args)


class ViewImageCanvas(FigureCanvas):
    def __init__(self, figure, view_tab):
        self.view_tab = view_tab
        super().__init__(figure)
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
        self.h5_datasets = []
        self.is_lazy_h5 = False
        self.h5_file = None
        self.h5_dataset = None
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

        self.q_geometry_mode = None
        self.q_geometry_source_tab = None
        self.custom_q_geometry = self.load_custom_q_geometry()
        self.current_file_type = None
        self.current_dataset_name = None

        self._build_ui()

    def _build_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(2, 2, 2, 2)
        main_layout.setSpacing(2)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # ============================================================
        # LEFT PANEL
        # ============================================================

        left_panel = QWidget()
        left_panel.setMinimumWidth(280)
        left_panel.setMaximumWidth(280)

        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(2, 2, 2, 2)
        left_layout.setSpacing(4)

        folder_box = QGroupBox("Folder browser")
        folder_box.setFixedHeight(86)
        folder_box.setStyleSheet("""
            QGroupBox {
                background-color: #f4f4f4;
                border: 0px;
                border-radius: 10px;
                margin-top: 14px;
                padding: 4px;
                font-size: 12px;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0px 4px;
                color: #222222;
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
        """)
        folder_layout = QVBoxLayout(folder_box)
        folder_layout.setContentsMargins(6, 18, 6, 4)
        folder_layout.setSpacing(4)

        self.folder_path = QLineEdit(str(self.current_folder))
        self.folder_path.returnPressed.connect(self.refresh_files)

        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.choose_folder)

        folder_layout.addWidget(self.folder_path)
        folder_layout.addWidget(browse_button)

        filter_box = QGroupBox("File filters")
        filter_box.setStyleSheet("""
            QPushButton {
                background-color: #e2e2e2;
                border: 0px;
                border-radius: 5px;
                padding: 4px;
            }

            QPushButton:hover {
                background-color: #d8d8d8;
            }
        """)
        filter_layout = QGridLayout(filter_box)

        self.extension_filter = QLineEdit("*.edf *.h5")
        self.name_filter = QLineEdit("**")

        self.show_subfolders_checkbox = QCheckBox("Show subfolders")
        self.show_subfolders_checkbox.setChecked(False)
        self.show_subfolders_checkbox.stateChanged.connect(self.refresh_files)

        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh_files)

        filter_layout.addWidget(QLabel("Extensions:"), 0, 0)
        filter_layout.addWidget(self.extension_filter, 0, 1)
        filter_layout.addWidget(QLabel("Name:"), 1, 0)
        filter_layout.addWidget(self.name_filter, 1, 1)
        filter_layout.addWidget(self.show_subfolders_checkbox, 2, 0, 1, 2)
        filter_layout.addWidget(refresh_button, 3, 0, 1, 2)

        files_box = QGroupBox("Files")
        files_layout = QVBoxLayout(files_box)

        self.file_list = QListWidget()
        self.file_list.currentItemChanged.connect(self.file_selection_changed)
        self.file_list.itemClicked.connect(self.open_selected_file)
        self.file_list.itemDoubleClicked.connect(self.open_selected_file)

        files_layout.addWidget(self.file_list)

        left_layout.addWidget(folder_box)
        left_layout.addWidget(filter_box)
        left_layout.addWidget(files_box)

        splitter.addWidget(left_panel)

        # ============================================================
        # CENTER PANEL
        # ============================================================

        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(2, 2, 2, 2)
        center_layout.setSpacing(4)

        toolbar_box = QGroupBox("Image tools")
        toolbar_box.setFixedHeight(86)
        toolbar_box.setStyleSheet("""
            QGroupBox {
                background-color: #f4f4f4;
                border: 0px;
                border-radius: 10px;
                margin-top: 14px;
                padding: 4px;
                font-size: 12px;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0px 4px;
                color: #222222;
                font-size: 12px;
            }

            QToolBar {
                background-color: #f4f4f4;
                border: 0px;
                spacing: 8px;
            }

            QToolButton {
                background-color: #f4f4f4;
                border: 0px;
                padding: 4px;
            }

            QToolButton:hover {
                background-color: #e5e5e5;
                border-radius: 5px;
            }
        """)

        toolbar_layout = QVBoxLayout(toolbar_box)
        toolbar_layout.setContentsMargins(6, 22, 6, 8)
        toolbar_layout.setSpacing(0)

        self.fig = Figure()
        self.fig.patch.set_facecolor("white")

        self.ax = self.fig.add_subplot(111)
        self.ax.set_axis_off()
        self.ax.set_aspect("equal")

        self.canvas = ViewImageCanvas(self.fig, self)
        self.canvas.setFocus()

        self.toolbar = ImageOnlyToolbar(self.canvas, self)
        self.toolbar.setIconSize(QSize(32, 32))
        self.toolbar.setFixedHeight(44)
        self.toolbar.setContentsMargins(0, 0, 0, 0)
        self.toolbar.coordinates = False
        # Remove Matplotlib save button for now
        for action in self.toolbar.actions():
            if action.text().lower() in ["save", "save the figure"]:
                self.toolbar.removeAction(action)

        toolbar_layout.addWidget(self.toolbar, alignment=Qt.AlignVCenter)

        center_layout.addWidget(toolbar_box, alignment=Qt.AlignTop)

        image_area = QHBoxLayout()
        image_area.setContentsMargins(0, 0, 0, 0)
        image_area.setSpacing(4)
        image_area.addWidget(self.canvas)

        slider_box = QVBoxLayout()
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

        slider_box.addWidget(QLabel("Max"))
        slider_box.addWidget(self.max_slider)
        slider_box.addWidget(QLabel("Min"))
        slider_box.addWidget(self.min_slider)

        image_area.addLayout(slider_box)
        center_layout.addLayout(image_area)

        self.cursor_label = QLabel("x = - | y = - | I = -")
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

        self.previous_button = QPushButton("<")
        self.next_button = QPushButton(">")
        self.previous_button.setFixedWidth(40)
        self.next_button.setFixedWidth(40)

        self.previous_button.clicked.connect(self.previous_image)
        self.next_button.clicked.connect(self.next_image)

        self.frame_start_spin = QSpinBox()
        self.frame_start_spin.setMinimum(1)
        self.frame_start_spin.setMaximum(1)
        self.frame_start_spin.setValue(1)
        self.frame_start_spin.valueChanged.connect(self.update_frame_slider_range)

        self.frame_end_spin = QSpinBox()
        self.frame_end_spin.setMinimum(1)
        self.frame_end_spin.setMaximum(1)
        self.frame_end_spin.setValue(1)
        self.frame_end_spin.valueChanged.connect(self.update_frame_slider_range)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.valueChanged.connect(self.slider_changed)

        self.frame_label = QLabel("0 / 0")

        nav_layout.addWidget(QLabel("From:"))
        nav_layout.addWidget(self.frame_start_spin)

        nav_layout.addWidget(self.previous_button)
        nav_layout.addWidget(self.frame_slider)
        nav_layout.addWidget(self.next_button)

        nav_layout.addWidget(QLabel("To:"))
        nav_layout.addWidget(self.frame_end_spin)
        nav_layout.addWidget(self.frame_label)

        display_layout = QHBoxLayout()

        self.log_checkbox = QCheckBox("Log")
        self.log_checkbox.setChecked(True)
        self.log_checkbox.stateChanged.connect(self.update_image)

        self.keep_ratio_checkbox = QCheckBox("Keep ratio")
        self.keep_ratio_checkbox.setChecked(True)
        self.keep_ratio_checkbox.stateChanged.connect(self.update_image)

        self.keep_zoom_checkbox = QCheckBox("Keep zoom")
        self.keep_zoom_checkbox.setChecked(False)
        self.keep_zoom_checkbox.setToolTip("Keep current zoom and pan when changing file or frame")

        self.save_colorbar_checkbox = QCheckBox("Save colorbar")
        self.save_colorbar_checkbox.setChecked(
            self.settings.value("view/save_colorbar", False, type=bool)
        )
        self.save_colorbar_checkbox.stateChanged.connect(self.save_colorbar_setting)

        self.vmin_spin = QDoubleSpinBox()
        self.vmin_spin.setDecimals(4)
        self.vmin_spin.setRange(-999999, 999999)
        self.vmin_spin.setValue(self.settings.value("view/vmin", 0.0, type=float))
        self.vmin_spin.valueChanged.connect(self.spin_intensity_changed)

        self.vmax_spin = QDoubleSpinBox()
        self.vmax_spin.setDecimals(4)
        self.vmax_spin.setRange(-999999, 999999)
        self.vmax_spin.setValue(self.settings.value("view/vmax", 5.0, type=float))
        self.vmax_spin.valueChanged.connect(self.spin_intensity_changed)

        autoscale_button = QPushButton("Auto intensity")
        autoscale_button.clicked.connect(self.auto_intensity)

        display_layout.addWidget(self.log_checkbox)
        display_layout.addWidget(self.keep_ratio_checkbox)
        display_layout.addWidget(self.keep_zoom_checkbox)
        display_layout.addWidget(self.save_colorbar_checkbox)
        display_layout.addWidget(QLabel("Min:"))
        display_layout.addWidget(self.vmin_spin)
        display_layout.addWidget(QLabel("Max:"))
        display_layout.addWidget(self.vmax_spin)
        display_layout.addWidget(autoscale_button)

        center_layout.addLayout(nav_layout)
        # center_layout.addLayout(display_layout)

        splitter.addWidget(center_panel)

        # ============================================================
        # RIGHT PANEL
        # ============================================================

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(2, 2, 2, 2)
        right_layout.setSpacing(4)

        info_box = QGroupBox("File information")
        info_box.setMinimumHeight(86)
        info_box.setStyleSheet("""
            QGroupBox {
                background-color: #f4f4f4;
                border: 0px;
                border-radius: 10px;
                margin-top: 14px;
                padding: 4px;
                font-size: 12px;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0px 4px;
                color: #222222;
                font-size: 12px;
            }
        """)
        info_box_layout = QVBoxLayout(info_box)
        info_box_layout.setContentsMargins(6, 18, 6, 4)
        info_box_layout.setSpacing(4)

        self.info_text = QTextEdit()
        self.info_text.setLineWrapMode(QTextEdit.WidgetWidth)
        self.info_text.setMinimumWidth(240)
        self.info_text.setReadOnly(True)
        self.info_text.setText("No file loaded.")

        self.dataset_list = QListWidget()
        self.dataset_list.itemDoubleClicked.connect(self.open_selected_dataset)
        self.dataset_list.hide()

        info_box_layout.addWidget(self.info_text)

        q_buttons_layout = QHBoxLayout()
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

        for button in [
            self.q_xenocs_button,
            self.q_id02_button,
            self.q_id13_button,
            self.q_custom_button,
            self.q_manual_button,
        ]:
            q_buttons_layout.addWidget(button)

        self.update_q_geometry_button_styles()
        info_box_layout.addLayout(q_buttons_layout)
        right_layout.addWidget(info_box)

        display_box = QGroupBox("Display settings")
        display_box.setStyleSheet("""
            QGroupBox {
                background-color: #f4f4f4;
                border: 0px;
                border-radius: 10px;
                margin-top: 14px;
                padding: 4px;
                font-size: 12px;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0px 4px;
                color: #222222;
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
        """)

        display_box_layout = QVBoxLayout(display_box)
        display_box_layout.setContentsMargins(10, 22, 10, 10)
        display_box_layout.setSpacing(12)

        display_checks_layout = QVBoxLayout()
        display_checks_layout.setSpacing(8)
        display_checks_layout.addWidget(self.log_checkbox)
        display_checks_layout.addWidget(self.keep_ratio_checkbox)
        display_checks_layout.addWidget(self.keep_zoom_checkbox)
        display_checks_layout.addWidget(self.save_colorbar_checkbox)

        min_layout = QHBoxLayout()
        min_layout.setSpacing(10)
        min_layout.addWidget(QLabel("Min:"))
        min_layout.addWidget(self.vmin_spin)

        max_layout = QHBoxLayout()
        max_layout.setSpacing(10)
        max_layout.addWidget(QLabel("Max:"))
        max_layout.addWidget(self.vmax_spin)

        display_box_layout.addLayout(display_checks_layout)
        display_box_layout.addLayout(min_layout)
        display_box_layout.addLayout(max_layout)
        display_box_layout.addWidget(autoscale_button)

        right_layout.addWidget(display_box)

        splitter.addWidget(right_panel)

        splitter.setSizes([280, 1000, 260])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)

        right_layout.setStretch(0, 1)

        self.canvas.draw_idle()

    # ============================================================
    # SETTINGS
    # ============================================================

    def save_colorbar_setting(self):
        keep_colorbar = self.save_colorbar_checkbox.isChecked()
        self.settings.setValue("view/save_colorbar", keep_colorbar)

        if keep_colorbar:
            self.settings.setValue("view/vmin", self.vmin_spin.value())
            self.settings.setValue("view/vmax", self.vmax_spin.value())

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
                "xc": 1294.689,
                "yc": 1310.290,
                "distance_m": 0.8,
                "pixel_x_mm": 0.075000,
                "pixel_y_mm": 0.075000,
                "wavelength_a": 0.826563,
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

        for mode, button in buttons.items():
            active = mode == self.q_geometry_mode
            button.blockSignals(True)
            button.setChecked(active)
            button.blockSignals(False)
            if active:
                button.setStyleSheet("""
                    QPushButton {
                        background-color: #007aff;
                        color: white;
                        border: 0px;
                        border-radius: 5px;
                        padding: 4px;
                    }
                """)
            else:
                button.setStyleSheet("""
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
                """)

        self.q_manual_button.setStyleSheet("""
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
        """)

    def set_q_geometry_mode(self, mode):
        if mode == "Custom" and not self.custom_q_geometry:
            self.open_q_geometry_dialog()
            return

        self.q_geometry_mode = mode
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
            ("xc", "Centre X"),
            ("yc", "Centre Y"),
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

            if lower_name.endswith(".dat"):
                continue

            if lower_name.endswith("_ave.h5"):
                continue

            match_extension = any(
                fnmatch.fnmatch(lower_name, pattern.lower())
                for pattern in extension_patterns
            )

            match_name = fnmatch.fnmatch(path.name, name_pattern)

            if match_extension and match_name:
                files.append(path)

        for path in sorted(files):
            item_text = str(path.relative_to(folder))
            self.file_list.addItem(item_text)
            item = self.file_list.item(self.file_list.count() - 1)
            resolved_path = path.expanduser().resolve()
            item.setData(Qt.UserRole, str(resolved_path))
            item.setToolTip(str(resolved_path))

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
        self.h5_datasets = []
        self.is_lazy_h5 = False
        self.h5_dataset = None

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

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def reset_figure(self):
        self.fig.clear()

        self.ax = self.fig.add_subplot(111)
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
        frames = []

        edf = fabio.open(str(path))

        try:
            nframes = int(getattr(edf, "nframes", 1) or 1)

            if nframes <= 1:
                frames.append(np.array(edf.data, dtype=float).copy())
                self.headers = dict(edf.header)

            else:
                for i in range(nframes):
                    frame = edf.getframe(i)
                    frames.append(np.array(frame.data, dtype=float).copy())

                    if i == 0:
                        self.headers = dict(frame.header)

        finally:
            try:
                edf.close()
            except Exception:
                pass

        if not frames:
            raise ValueError("No frame was found in this EDF file.")

        self.images = np.stack(frames, axis=0)

        self.dataset_list.clear()
        for i in range(self.images.shape[0]):
            self.dataset_list.addItem(f"Frame {i + 1}")

        print("EDF loaded shape:", self.images.shape)
        print("EDF intensity min/max:", np.nanmin(self.images), np.nanmax(self.images))

        self.update_file_information(
            "EDF",
            "-",
            self.images.shape[0],
            self.images.shape[1:]
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
                        datasets.append((name, obj.shape, obj.dtype))

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

        for name, shape, dtype in datasets:
            self.dataset_list.addItem(f"{name}   {shape}")

        self.dataset_list.setCurrentRow(0)
        self.open_h5_dataset(datasets[0][0])

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

            shape = self.h5_dataset.shape
            dtype = self.h5_dataset.dtype

            self.headers = {
                "Dataset": dataset_name,
                "Shape": str(shape),
                "Dtype": str(dtype),
            }

            for key, value in self.h5_dataset.attrs.items():
                self.headers[key] = str(value)

            for key, value in self.h5_file.attrs.items():
                self.headers[f"File attribute - {key}"] = str(value)

            self.add_matching_edf_center_to_headers()

        except Exception as e:
            raise RuntimeError(f"Unable to read this H5 dataset:\n{e}")

        if len(shape) == 2:
            self.is_lazy_h5 = True
            self.images = None
            self.n_frames = 1
            self.image_shape = shape

        elif len(shape) == 3:
            self.is_lazy_h5 = True
            self.images = None
            self.n_frames = shape[0]
            self.image_shape = shape[1:]

        else:
            raise ValueError("Dataset must be 2D or 3D.")

        self.current_index = 0

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
                f"Centre: X = {xc:.6g}, Y = {yc:.6g}",
                f"Distance: {distance_m:.6g} m",
                f"Pixel: {pixel_x_mm:.6g} x {pixel_y_mm:.6g} mm",
                f"Wavelength: {wavelength_nm:.6g} nm",
            ])

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

    def add_matching_edf_center_to_headers(self):
        if self.current_file is None:
            return

        edf_path = self.current_file.with_suffix(".edf")
        if not edf_path.exists():
            return

        try:
            import fabio

            edf = fabio.open(str(edf_path))
            try:
                edf_header = dict(edf.header)
            finally:
                try:
                    edf.close()
                except Exception:
                    pass
        except Exception:
            return

        copied = []
        for key in ["Center_1", "Center_2", "center_1", "center_2"]:
            if key in edf_header and key not in self.headers:
                self.headers[key] = edf_header[key]
                copied.append(key)

        if copied:
            self.headers["Center source"] = edf_path.name

    # ============================================================
    # IMAGE DISPLAY
    # ============================================================

    def configure_slider(self):
        n = self.n_frames if self.is_lazy_h5 else self.images.shape[0]

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

    def update_frame_slider_range(self):
        if self.is_lazy_h5:
            n = self.n_frames
        elif self.images is not None:
            n = self.images.shape[0]
        else:
            return

        start = self.frame_start_spin.value() - 1
        end = self.frame_end_spin.value() - 1

        if start > end:
            return

        self.frame_slider.setMinimum(start)
        self.frame_slider.setMaximum(end)

        if self.current_index < start:
            self.frame_slider.setValue(start)
        elif self.current_index > end:
            self.frame_slider.setValue(end)

    def get_current_image(self):
        if self.is_lazy_h5:
            if self.h5_dataset is None:
                return None

            if self.h5_dataset.ndim == 2:
                return np.array(self.h5_dataset, dtype=float)

            return np.array(self.h5_dataset[self.current_index], dtype=float)

        if self.images is None:
            return None

        return self.images[self.current_index]

    def prepare_display_image(self, img):
        img = np.array(img, dtype=float)
        img[img > 4e9] = np.nan

        if self.log_checkbox.isChecked():
            img = np.log10(np.clip(img, 0, None) + 1)

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

        x = None
        y = None

        for key in possible_x_keys:
            if key in self.headers:
                try:
                    x = float(str(self.headers[key]).replace(",", "."))
                    break
                except Exception:
                    pass

        for key in possible_y_keys:
            if key in self.headers:
                try:
                    y = float(str(self.headers[key]).replace(",", "."))
                    break
                except Exception:
                    pass

        if x is None or y is None:
            return None

        return x, y

    def get_header_float(self, *keys):
        if not self.headers:
            return None

        for key in keys:
            if key in self.headers:
                try:
                    return float(str(self.headers[key]).replace(",", "."))
                except Exception:
                    pass

        return None

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

    def wavelength_to_nm(self, wavelength):
        if wavelength < 1e-6:
            return wavelength * 1e9
        if wavelength >= 0.5:
            return wavelength * 0.1
        return wavelength

    def get_header_q_geometry(self):
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
        geometry = self.preset_q_geometry(self.q_geometry_mode)
        if geometry is None:
            return None

        geometry = dict(geometry)
        header_values = self.get_header_q_values()
        if self.q_geometry_mode != "Custom":
            geometry.update(header_values)
        else:
            for key, value in header_values.items():
                if geometry.get(key) in (None, 0):
                    geometry[key] = value

        return geometry

    def get_q_geometry_from_header(self):
        return self.get_preset_q_geometry()

    def calculate_q_at_pixel(self, x_index, y_index):
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

    def draw_center_cross(self):
        for artist in self.center_artists:
            try:
                artist.remove()
            except Exception:
                pass

        self.center_artists = []

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

        aspect = "equal" if self.keep_ratio_checkbox.isChecked() else "auto"

        if self.image_artist is None:
            self.image_artist = self.ax.imshow(
                self.display_img,
                cmap="jet",
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

        self.ax.set_title("")

        self.draw_center_cross()

        total = self.n_frames if self.is_lazy_h5 else self.images.shape[0]
        self.frame_label.setText(f"{self.current_index + 1} / {total}")

        self.ax.set_autoscale_on(False)
        self.canvas.draw_idle()

    def auto_intensity(self):
        img = self.get_current_image()

        if img is None:
            return

        display_img = self.prepare_display_image(img)
        finite = display_img[np.isfinite(display_img)]

        if finite.size == 0:
            return

        vmin = float(np.nanpercentile(finite, 1))
        vmax = float(np.nanpercentile(finite, 99))

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
            self.cursor_label.setText("x = - | y = - | I = - | q = -")
            return

        if event.xdata is None or event.ydata is None:
            self.cursor_label.setText("x = - | y = - | I = - | q = -")
            return

        x_index = int(round(event.xdata))
        y_index = int(round(event.ydata))

        ny, nx = self.raw_current_img.shape

        if not (0 <= x_index < nx and 0 <= y_index < ny):
            self.cursor_label.setText("x = - | y = - | I = - | q = -")
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

        self.cursor_label.setText(
            f"x = {x_index + 1} | y = {y_index + 1} | I = {value_text} | q = {q_text}"
        )

    def on_mouse_leave(self, event):
        self.cursor_label.setText("x = - | y = - | I = - | q = -")

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
        total_frames = self.n_frames if self.is_lazy_h5 else (self.images.shape[0] if self.images is not None else 1)
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
                    cmap="jet",
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
            cmap="jet",
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

        if self.keep_zoom_checkbox.isChecked():
            self._saved_xlim = self.ax.get_xlim()
            self._saved_ylim = self.ax.get_ylim()

        self.ax.set_autoscale_on(False)
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
        self.canvas.draw_idle()

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

        if self.keep_zoom_checkbox.isChecked():
            self._saved_xlim = self.ax.get_xlim()
            self._saved_ylim = self.ax.get_ylim()

        self.ax.set_autoscale_on(False)
        self.canvas.draw_idle()

    def on_scroll_zoom(self, event):
        if self.image_artist is None or event.inaxes != self.ax:
            return

        if event.xdata is None or event.ydata is None:
            return

        zoom_factor = 0.85 if event.button == "up" else 1.18
        self.zoom_at_data_position(event.xdata, event.ydata, zoom_factor)

    def on_mouse_press(self, event):
        if self.image_artist is None or event.inaxes != self.ax:
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
        self.canvas.setCursor(Qt.ArrowCursor)

    def pan_image_from_motion(self, event):
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

        self.ax.set_xlim(x0 + dx, x1 + dx)
        self.ax.set_ylim(y0 + dy, y1 + dy)

        if self.keep_zoom_checkbox.isChecked():
            self._saved_xlim = self.ax.get_xlim()
            self._saved_ylim = self.ax.get_ylim()

        self.ax.set_autoscale_on(False)
        self.canvas.draw_idle()
        return True
