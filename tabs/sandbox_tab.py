from pathlib import Path
import fnmatch
import re

import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QCheckBox,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import cm, colors

from tabs.file_ratings import install_file_rating_menu, set_item_file_path
from tabs.instrument_presets import (
    ID13_DEFAULT_CENTER_X,
    ID13_DEFAULT_CENTER_Y,
    ID13_DEFAULT_DISTANCE_M,
    ID13_DEFAULT_PIXEL_MM,
    ID13_DEFAULT_WAVELENGTH_A,
)
from tabs.radial_tab import radial_average
from tabs.ui_style import GROUP_BOX_STYLE, PAGE_MARGINS, PANEL_MARGINS


class SandboxTab(QWidget):
    folder_changed = Signal(object, object)

    def __init__(self):
        super().__init__()
        self.folder_path = None
        self.current_folder = None
        self.current_image = None
        self.current_path = None
        self.current_stack = None
        self.current_frame_index = 0
        self.current_frame_count = 1
        self.max_3d_points = 90
        self.imogolite_q = None
        self.imogolite_intensity = None
        self.imogolite_current_paths = []
        self.imogolite_gradient_start = "#1f77b4"
        self.imogolite_gradient_end = "#d62728"
        self.imogolite_results = []
        self.geometry_presets = {
            "XENOCS": {"center_x": 612.0, "center_y": 649.0, "pixel_size": 75.0, "distance": 900.0, "wavelength": 1.54189},
            "ID02": {"center_x": 0.0, "center_y": 0.0, "pixel_size": 75.0, "distance": 900.0, "wavelength": 1.0},
            "ID13": {
                "center_x": ID13_DEFAULT_CENTER_X,
                "center_y": ID13_DEFAULT_CENTER_Y,
                "pixel_size": ID13_DEFAULT_PIXEL_MM * 1000.0,
                "distance": ID13_DEFAULT_DISTANCE_M * 1000.0,
                "wavelength": ID13_DEFAULT_WAVELENGTH_A,
            },
            "Custom": {"center_x": None, "center_y": None, "pixel_size": None, "distance": None, "wavelength": None},
            "+": {"center_x": None, "center_y": None, "pixel_size": None, "distance": None, "wavelength": None},
        }
        self.current_geometry = self.geometry_presets["ID13"].copy()
        self.build_ui()

    def build_ui(self):
        self.setMinimumHeight(0)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(*PAGE_MARGINS)
        main_layout.setSpacing(6)

        self.project_stack = QStackedWidget()
        main_layout.addWidget(self.project_stack, 1)

        selector_page = QWidget()
        selector_layout = QVBoxLayout(selector_page)
        selector_layout.setContentsMargins(0, 0, 0, 0)
        selector_layout.setSpacing(16)
        selector_layout.addStretch(1)

        selector_title = QLabel("Sandbox projects")
        selector_title.setAlignment(Qt.AlignCenter)
        selector_title.setStyleSheet("font-size: 24px; font-weight: 700;")
        selector_layout.addWidget(selector_title)

        selector_buttons = QHBoxLayout()
        selector_buttons.setContentsMargins(0, 0, 0, 0)
        selector_buttons.setSpacing(14)
        selector_buttons.addStretch(1)

        self.open_3d_project_button = QPushButton("3D SAXS pattern")
        self.open_3d_project_button.setMinimumSize(190, 70)
        self.open_3d_project_button.clicked.connect(lambda: self.open_sandbox_project("3D SAXS pattern"))
        selector_buttons.addWidget(self.open_3d_project_button)

        self.open_imogolite_project_button = QPushButton("Imogolite distance")
        self.open_imogolite_project_button.setMinimumSize(190, 70)
        self.open_imogolite_project_button.clicked.connect(lambda: self.open_sandbox_project("Imogolite distance"))
        selector_buttons.addWidget(self.open_imogolite_project_button)

        selector_buttons.addStretch(1)
        selector_layout.addLayout(selector_buttons)
        selector_layout.addStretch(1)
        self.project_stack.addWidget(selector_page)

        splitter = QSplitter(Qt.Horizontal)
        self.project_stack.addWidget(splitter)

        file_box = QGroupBox("File browser")
        file_box.setStyleSheet(GROUP_BOX_STYLE)
        file_box.setMinimumWidth(280)
        file_box.setMaximumWidth(420)
        file_layout = QVBoxLayout(file_box)
        file_layout.setContentsMargins(*PANEL_MARGINS)
        file_layout.setSpacing(6)

        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText("Folder")
        self.folder_edit.returnPressed.connect(self.folder_from_text)
        file_layout.addWidget(self.folder_edit)

        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_folder)
        file_layout.addWidget(browse_button)

        filters_layout = QHBoxLayout()
        filters_layout.setContentsMargins(0, 0, 0, 0)
        filters_layout.setSpacing(4)
        filters_layout.addWidget(QLabel("Name:"))
        self.name_filter = QLineEdit("*")
        filters_layout.addWidget(self.name_filter, 1)
        file_layout.addLayout(filters_layout)

        ext_layout = QHBoxLayout()
        ext_layout.setContentsMargins(0, 0, 0, 0)
        ext_layout.setSpacing(4)
        ext_layout.addWidget(QLabel("Extensions:"))
        self.extension_filter = QLineEdit("*.edf *.h5 *.hdf5")
        ext_layout.addWidget(self.extension_filter, 1)
        file_layout.addLayout(ext_layout)

        options_layout = QHBoxLayout()
        options_layout.setContentsMargins(0, 0, 0, 0)
        options_layout.setSpacing(4)
        self.subfolders_checkbox = QCheckBox("Show subfolders")
        options_layout.addWidget(self.subfolders_checkbox)
        options_layout.addStretch(1)
        file_layout.addLayout(options_layout)

        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh_files)
        file_layout.addWidget(refresh_button)

        self.file_list = QListWidget()
        install_file_rating_menu(self.file_list)
        self.file_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.file_list.itemClicked.connect(self.open_selected_files)
        file_layout.addWidget(self.file_list, 1)

        self.plot_box = QGroupBox("3D SAXS pattern")
        self.plot_box.setStyleSheet(GROUP_BOX_STYLE)
        self.plot_box.setMinimumHeight(0)
        plot_layout = QVBoxLayout(self.plot_box)
        plot_layout.setContentsMargins(*PANEL_MARGINS)
        plot_layout.setSpacing(6)

        self.figure = Figure(figsize=(6, 5))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax = self.figure.add_subplot(111, projection="3d")
        self.canvas.mpl_connect("button_press_event", self.on_canvas_click)
        self.toolbar = NavigationToolbar(self.canvas, self)
        plot_layout.addWidget(self.toolbar, 0)

        plot_area_layout = QHBoxLayout()
        plot_area_layout.setContentsMargins(0, 0, 0, 0)
        plot_area_layout.setSpacing(6)
        plot_area_layout.addWidget(self.canvas, 1)

        side_box = QGroupBox("Line geometry")
        side_box.setStyleSheet(GROUP_BOX_STYLE)
        side_box.setFixedWidth(430)
        side_box.setMinimumHeight(0)
        side_box.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Ignored)
        side_layout = QVBoxLayout(side_box)
        side_layout.setContentsMargins(10, 14, 10, 6)
        side_layout.setSpacing(4)

        self.back_to_projects_button = QPushButton("← Projects")
        self.back_to_projects_button.clicked.connect(self.show_project_selector)
        side_layout.addWidget(self.back_to_projects_button)

        self.sandbox_project_combo = QComboBox()
        self.sandbox_project_combo.addItems([
            "3D SAXS pattern",
            "Imogolite distance",
        ])
        self.sandbox_project_combo.currentTextChanged.connect(self.apply_sandbox_project)
        side_layout.addLayout(self.form_row("Project", self.sandbox_project_combo))

        self.geometry_combo = QComboBox()
        self.geometry_combo.addItems(["XENOCS", "ID02", "ID13", "Custom", "+"])
        self.geometry_combo.setCurrentText("ID13")
        self.geometry_combo.currentTextChanged.connect(self.apply_geometry_preset)
        side_layout.addLayout(self.form_row("Geometry", self.geometry_combo))

        self.center_x_label = QLabel("Center X")
        self.center_x_edit = QLineEdit()
        self.center_x_edit.returnPressed.connect(self.apply_custom_geometry_from_fields)
        side_layout.addLayout(self.form_row(self.center_x_label, self.center_x_edit))

        self.center_y_label = QLabel("Center Y")
        self.center_y_edit = QLineEdit()
        self.center_y_edit.returnPressed.connect(self.apply_custom_geometry_from_fields)
        side_layout.addLayout(self.form_row(self.center_y_label, self.center_y_edit))

        self.pixel_size_label = QLabel("Pixel size (µm)")
        self.pixel_size_edit = QLineEdit()
        self.pixel_size_edit.returnPressed.connect(self.apply_custom_geometry_from_fields)
        side_layout.addLayout(self.form_row(self.pixel_size_label, self.pixel_size_edit))

        self.distance_label = QLabel("Distance (mm)")
        self.distance_edit = QLineEdit()
        self.distance_edit.returnPressed.connect(self.apply_custom_geometry_from_fields)
        side_layout.addLayout(self.form_row(self.distance_label, self.distance_edit))

        self.wavelength_label = QLabel("Wavelength (Å)")
        self.wavelength_edit = QLineEdit()
        self.wavelength_edit.returnPressed.connect(self.apply_custom_geometry_from_fields)
        side_layout.addLayout(self.form_row(self.wavelength_label, self.wavelength_edit))

        self.apply_geometry_button = QPushButton("Apply")
        self.apply_geometry_button.clicked.connect(self.apply_custom_geometry_from_fields)
        side_layout.addWidget(self.apply_geometry_button)

        self.imogolite_peak_box = QGroupBox("Imogolite peak")
        self.imogolite_peak_box.setStyleSheet(GROUP_BOX_STYLE)
        self.imogolite_peak_box.setMinimumHeight(0)
        self.imogolite_peak_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        peak_layout = QVBoxLayout(self.imogolite_peak_box)
        peak_layout.setContentsMargins(8, 14, 8, 6)
        peak_layout.setSpacing(3)

        self.imogolite_peak_radius_edit = QLineEdit()
        self.imogolite_peak_radius_edit.setPlaceholderText("from centre")
        self.imogolite_peak_radius_edit.returnPressed.connect(self.calculate_imogolite_distance)
        peak_layout.addLayout(self.form_row("Peak radius (px)", self.imogolite_peak_radius_edit))

        self.imogolite_peak_q_edit = QLineEdit()
        self.imogolite_peak_q_edit.setPlaceholderText("click I(q) peak")
        self.imogolite_peak_q_edit.returnPressed.connect(self.calculate_imogolite_distance)
        peak_layout.addLayout(self.form_row("Peak q (nm⁻¹)", self.imogolite_peak_q_edit))

        self.imogolite_bins_spinbox = QSpinBox()
        self.imogolite_bins_spinbox.setRange(50, 10000)
        self.imogolite_bins_spinbox.setValue(100)
        self.imogolite_bins_spinbox.valueChanged.connect(self.replot_current_image)
        peak_layout.addLayout(self.form_row("I(q) bins", self.imogolite_bins_spinbox))

        self.imogolite_file_stride_spinbox = QSpinBox()
        self.imogolite_file_stride_spinbox.setRange(1, 100000)
        self.imogolite_file_stride_spinbox.setValue(10)
        self.imogolite_file_stride_spinbox.setToolTip("Integrate one selected cave file every N files")
        peak_layout.addLayout(self.form_row("Every N files", self.imogolite_file_stride_spinbox))

        self.imogolite_plot_mode_combo = QComboBox()
        self.imogolite_plot_mode_combo.addItems([
            "lin lin",
            "log log",
            "lin log",
            "log lin",
        ])
        self.imogolite_plot_mode_combo.setCurrentText("lin lin")
        self.imogolite_plot_mode_combo.currentTextChanged.connect(self.replot_current_image)
        peak_layout.addLayout(self.form_row("Plot scale", self.imogolite_plot_mode_combo))

        self.imogolite_tube_diameter_edit = QLineEdit("4.4")
        self.imogolite_tube_diameter_edit.setToolTip("Tube outer diameter used for the swelling-law volume fraction")
        self.imogolite_tube_diameter_edit.returnPressed.connect(self.update_imogolite_table_only)
        peak_layout.addLayout(self.form_row("Tube diam. (nm)", self.imogolite_tube_diameter_edit))

        self.imogolite_frame_step_combo = QComboBox()
        self.imogolite_frame_step_combo.addItems(["5", "2"])
        self.imogolite_frame_step_combo.setToolTip("Sample distance between two consecutive frames")
        self.imogolite_frame_step_combo.currentTextChanged.connect(self.update_imogolite_table_only)
        peak_layout.addLayout(self.form_row("Frame step (µm)", self.imogolite_frame_step_combo))

        self.imogolite_origin_combo = QComboBox()
        self.imogolite_origin_combo.addItems([
            "first frame → last",
            "last frame → first",
        ])
        self.imogolite_origin_combo.currentTextChanged.connect(self.update_imogolite_table_only)
        peak_layout.addLayout(self.form_row("Origin side", self.imogolite_origin_combo))

        self.imogolite_legend_checkbox = QCheckBox("Legend")
        self.imogolite_legend_checkbox.setChecked(True)
        self.imogolite_legend_checkbox.stateChanged.connect(self.replot_current_image)
        peak_layout.addWidget(self.imogolite_legend_checkbox)

        peak_buttons_layout = QHBoxLayout()
        peak_buttons_layout.setContentsMargins(0, 0, 0, 0)
        peak_buttons_layout.setSpacing(4)
        self.use_max_peak_button = QPushButton("Use max")
        self.use_max_peak_button.setToolTip("Use the strongest finite I(q) point")
        self.use_max_peak_button.clicked.connect(self.use_current_max_as_imogolite_peak)
        peak_buttons_layout.addWidget(self.use_max_peak_button)
        self.integrate_imogolite_button = QPushButton("Integrate")
        self.integrate_imogolite_button.clicked.connect(self.integrate_imogolite_selection)
        peak_buttons_layout.addWidget(self.integrate_imogolite_button)
        self.calculate_imogolite_button = QPushButton("Calc")
        self.calculate_imogolite_button.clicked.connect(self.calculate_imogolite_distance)
        peak_buttons_layout.addWidget(self.calculate_imogolite_button)
        peak_layout.addLayout(peak_buttons_layout)

        self.imogolite_result_label = QLabel("q = -\nd = -\nφ = -")
        self.imogolite_result_label.setWordWrap(True)
        peak_layout.addWidget(self.imogolite_result_label)

        self.imogolite_results_table = QTableWidget(0, 5)
        self.imogolite_results_table.setHorizontalHeaderLabels(["Frame", "x (µm)", "qmax", "d", "φ"])
        self.imogolite_results_table.setMinimumHeight(70)
        self.imogolite_results_table.setMaximumHeight(120)
        self.imogolite_results_table.setAlternatingRowColors(True)
        self.imogolite_results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.imogolite_results_table.verticalHeader().setDefaultSectionSize(20)
        self.imogolite_results_table.verticalHeader().setMinimumSectionSize(18)
        self.imogolite_results_table.horizontalHeader().setStretchLastSection(True)
        peak_layout.addWidget(self.imogolite_results_table)

        self.copy_imogolite_table_button = QPushButton("Copy table")
        self.copy_imogolite_table_button.clicked.connect(self.copy_imogolite_results_table)
        peak_layout.addWidget(self.copy_imogolite_table_button)
        copy_shortcut = QShortcut(QKeySequence.Copy, self.imogolite_results_table)
        copy_shortcut.activated.connect(self.copy_imogolite_results_table)
        side_layout.addWidget(self.imogolite_peak_box)

        self.wireframe_checkbox = QCheckBox("Wireframe")
        self.wireframe_checkbox.setChecked(False)
        self.wireframe_checkbox.toggled.connect(self.replot_current_image)
        side_layout.addWidget(self.wireframe_checkbox)

        self.displayed_pixels_label = QLabel("Displayed pixels")
        self.displayed_pixels_spinbox = QSpinBox()
        self.displayed_pixels_spinbox.setRange(20, 600)
        self.displayed_pixels_spinbox.setSingleStep(10)
        self.displayed_pixels_spinbox.setValue(self.max_3d_points)
        self.displayed_pixels_spinbox.setToolTip("Maximum number of displayed points along the largest image dimension. Higher values are more precise but slower.")
        self.displayed_pixels_spinbox.valueChanged.connect(self.update_display_precision)
        side_layout.addLayout(self.form_row(self.displayed_pixels_label, self.displayed_pixels_spinbox))

        self.precision_step_label = QLabel("Precision step")
        self.precision_step_spinbox = QSpinBox()
        self.precision_step_spinbox.setRange(1, 100)
        self.precision_step_spinbox.setValue(1)
        self.precision_step_spinbox.setToolTip("Additional pixel skipping. 1 keeps the automatic display density; higher values show fewer pixels.")
        self.precision_step_spinbox.valueChanged.connect(self.replot_current_image)
        side_layout.addLayout(self.form_row(self.precision_step_label, self.precision_step_spinbox))

        self.z_min_label = QLabel("Intensity min")
        self.z_min_edit = QLineEdit()
        self.z_min_edit.setPlaceholderText("auto")
        self.z_min_edit.returnPressed.connect(self.replot_current_image)
        side_layout.addLayout(self.form_row(self.z_min_label, self.z_min_edit))

        self.z_max_label = QLabel("Intensity max")
        self.z_max_edit = QLineEdit()
        self.z_max_edit.setPlaceholderText("auto")
        self.z_max_edit.returnPressed.connect(self.replot_current_image)
        side_layout.addLayout(self.form_row(self.z_max_label, self.z_max_edit))

        self.reset_z_scale_button = QPushButton("Auto intensity scale")
        self.reset_z_scale_button.clicked.connect(self.reset_intensity_scale)
        side_layout.addWidget(self.reset_z_scale_button)

        self.saxs_3d_widgets = [
            self.wireframe_checkbox,
            self.displayed_pixels_label,
            self.displayed_pixels_spinbox,
            self.precision_step_label,
            self.precision_step_spinbox,
            self.z_min_label,
            self.z_min_edit,
            self.z_max_label,
            self.z_max_edit,
            self.reset_z_scale_button,
        ]

        side_layout.addStretch(1)
        self.update_geometry_fields()

        plot_area_layout.addWidget(side_box, 0)
        plot_layout.addLayout(plot_area_layout, 1)

        bottom_bar = QHBoxLayout()
        bottom_bar.setContentsMargins(0, 0, 0, 0)
        bottom_bar.setSpacing(6)

        self.previous_frame_button = QPushButton("◀")
        self.previous_frame_button.clicked.connect(self.previous_frame)
        bottom_bar.addWidget(self.previous_frame_button)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.valueChanged.connect(self.set_frame_from_slider)
        bottom_bar.addWidget(self.frame_slider, 1)

        self.next_frame_button = QPushButton("▶")
        self.next_frame_button.clicked.connect(self.next_frame)
        bottom_bar.addWidget(self.next_frame_button)

        self.frame_label = QLabel("Frame 1 / 1")
        self.frame_label.setMinimumWidth(90)
        bottom_bar.addWidget(self.frame_label)

        self.auto_contrast_button = QPushButton("Auto contrast")
        self.auto_contrast_button.clicked.connect(self.auto_contrast)
        bottom_bar.addWidget(self.auto_contrast_button)

        self.contrast_min_label = QLabel("Contrast min")
        bottom_bar.addWidget(self.contrast_min_label)
        self.contrast_min_slider = QSlider(Qt.Horizontal)
        self.contrast_min_slider.setRange(0, 999)
        self.contrast_min_slider.setValue(10)
        self.contrast_min_slider.valueChanged.connect(self.replot_current_image)
        bottom_bar.addWidget(self.contrast_min_slider, 1)

        self.contrast_max_label = QLabel("max")
        bottom_bar.addWidget(self.contrast_max_label)
        self.contrast_max_slider = QSlider(Qt.Horizontal)
        self.contrast_max_slider.setRange(1, 1000)
        self.contrast_max_slider.setValue(990)
        self.contrast_max_slider.valueChanged.connect(self.replot_current_image)
        bottom_bar.addWidget(self.contrast_max_slider, 1)


        plot_layout.addLayout(bottom_bar, 0)

        self.status_label = QLabel("Open an EDF/H5 SAXS image to display it as a 3D surface.")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setMinimumHeight(22)
        plot_layout.addWidget(self.status_label, 0)
        self.saxs_3d_bottom_widgets = [
            self.auto_contrast_button,
            self.contrast_min_label,
            self.contrast_min_slider,
            self.contrast_max_label,
            self.contrast_max_slider,
        ]
        self.apply_sandbox_project(self.sandbox_project_combo.currentText())

        splitter.addWidget(file_box)
        splitter.addWidget(self.plot_box)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

    def form_row(self, label, widget):
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        label_widget = label if isinstance(label, QLabel) else QLabel(str(label))
        label_widget.setMinimumWidth(120)
        widget.setMinimumWidth(140)
        row.addWidget(label_widget, 0)
        row.addWidget(widget, 1)
        return row

    def show_project_selector(self):
        self.project_stack.setCurrentIndex(0)

    def current_project(self):
        return self.sandbox_project_combo.currentText()

    def open_sandbox_project(self, name):
        self.sandbox_project_combo.blockSignals(True)
        self.sandbox_project_combo.setCurrentText(name)
        self.sandbox_project_combo.blockSignals(False)
        self.apply_sandbox_project(name)
        self.project_stack.setCurrentIndex(1)

    def apply_sandbox_project(self, name):
        is_imogolite = name == "Imogolite distance"

        self.file_list.setSelectionMode(
            QAbstractItemView.ExtendedSelection if is_imogolite else QAbstractItemView.SingleSelection
        )
        self.name_filter.blockSignals(True)
        self.name_filter.setText("*cave*" if is_imogolite else "*")
        self.name_filter.blockSignals(False)
        self.extension_filter.blockSignals(True)
        self.extension_filter.setText("*.h5 *.hdf5 *.edf *.dat" if is_imogolite else "*.h5 *.hdf5 *.edf")
        self.extension_filter.blockSignals(False)
        self.subfolders_checkbox.setChecked(is_imogolite)
        self.refresh_files()

        self.imogolite_peak_box.setVisible(is_imogolite)
        self.wavelength_label.setVisible(is_imogolite)
        self.wavelength_edit.setVisible(is_imogolite)

        for widget in self.saxs_3d_widgets:
            widget.setVisible(not is_imogolite)
        for widget in self.saxs_3d_bottom_widgets:
            widget.setVisible(not is_imogolite)

        if is_imogolite:
            self.plot_box.setTitle("Imogolite distance")
            paths = self.selected_files()
            if paths:
                self.imogolite_current_paths = paths
                self.status_label.setText(f"{len(paths)} imogolite file(s) selected. Choose Every N files, then click Integrate.")
            elif self.current_image is None:
                self.status_label.setText("Open one or more cave EDF/H5 images or saved .dat curves, then integrate qI(q).")
            else:
                self.status_label.setText("Choose cave files or .dat curves, then click Integrate.")
        elif self.current_image is None:
            self.plot_box.setTitle("3D SAXS pattern")
            self.status_label.setText("Open an EDF/H5 SAXS image to display it as a 3D surface.")
        else:
            self.plot_box.setTitle("3D SAXS pattern")
            self.plot_3d(self.current_image)

    def apply_geometry_preset(self, name):
        preset = self.geometry_presets.get(name)
        if preset is None:
            return
        self.current_geometry = preset.copy()
        self.update_geometry_fields()
        self.replot_current_image()

    def update_geometry_fields(self):
        geometry = self.current_geometry
        self.center_x_edit.setText("" if geometry.get("center_x") is None else f"{geometry['center_x']:.3f}")
        self.center_y_edit.setText("" if geometry.get("center_y") is None else f"{geometry['center_y']:.3f}")
        self.pixel_size_edit.setText("" if geometry.get("pixel_size") is None else f"{geometry['pixel_size']:.3f}")
        self.distance_edit.setText("" if geometry.get("distance") is None else f"{geometry['distance']:.3f}")
        self.wavelength_edit.setText("" if geometry.get("wavelength") is None else f"{geometry['wavelength']:.6f}")

    def read_geometry_from_fields(self):
        return {
            "center_x": self.optional_float_from_text(self.center_x_edit.text()),
            "center_y": self.optional_float_from_text(self.center_y_edit.text()),
            "pixel_size": self.optional_float_from_text(self.pixel_size_edit.text()),
            "distance": self.optional_float_from_text(self.distance_edit.text()),
            "wavelength": self.optional_float_from_text(self.wavelength_edit.text()),
        }

    def apply_custom_geometry_from_fields(self):
        self.current_geometry = self.read_geometry_from_fields()
        if self.geometry_combo.currentText() not in {"Custom", "+"}:
            self.geometry_combo.blockSignals(True)
            self.geometry_combo.setCurrentText("Custom")
            self.geometry_combo.blockSignals(False)
        self.replot_current_image()

    def optional_float_from_text(self, text):
        text = text.strip().replace(",", ".")
        if not text:
            return None
        return float(text)

    def geometry_value(self, key):
        value = self.current_geometry.get(key)
        if value is None:
            return None
        return float(value)

    def q_from_peak_radius_px(self, radius_px):
        pixel_size_um = self.geometry_value("pixel_size")
        distance_mm = self.geometry_value("distance")
        wavelength_a = self.geometry_value("wavelength")

        if radius_px is None or radius_px <= 0:
            raise ValueError("Peak radius must be > 0 px.")
        if pixel_size_um is None or pixel_size_um <= 0:
            raise ValueError("Pixel size must be > 0 µm.")
        if distance_mm is None or distance_mm <= 0:
            raise ValueError("Distance must be > 0 mm.")
        if wavelength_a is None or wavelength_a <= 0:
            raise ValueError("Wavelength must be > 0 Å.")

        radius_m = float(radius_px) * pixel_size_um * 1e-6
        distance_m = distance_mm * 1e-3
        wavelength_nm = wavelength_a * 0.1
        two_theta = np.arctan2(radius_m, distance_m)
        return (4.0 * np.pi / wavelength_nm) * np.sin(two_theta / 2.0)

    def peak_radius_px_from_q(self, q_nm):
        pixel_size_um = self.geometry_value("pixel_size")
        distance_mm = self.geometry_value("distance")
        wavelength_a = self.geometry_value("wavelength")

        if q_nm is None or q_nm <= 0:
            raise ValueError("Peak q must be > 0 nm⁻¹.")
        if pixel_size_um is None or pixel_size_um <= 0:
            raise ValueError("Pixel size must be > 0 µm.")
        if distance_mm is None or distance_mm <= 0:
            raise ValueError("Distance must be > 0 mm.")
        if wavelength_a is None or wavelength_a <= 0:
            raise ValueError("Wavelength must be > 0 Å.")

        wavelength_nm = wavelength_a * 0.1
        argument = float(q_nm) * wavelength_nm / (4.0 * np.pi)
        if abs(argument) > 1:
            raise ValueError("Peak q is incompatible with the wavelength.")
        two_theta = 2.0 * np.arcsin(argument)
        radius_m = distance_mm * 1e-3 * np.tan(two_theta)
        return radius_m / (pixel_size_um * 1e-6)

    def calculate_imogolite_distance(self):
        try:
            self.current_geometry = self.read_geometry_from_fields()
            q_nm = self.optional_float_from_text(self.imogolite_peak_q_edit.text())
            radius_px = self.optional_float_from_text(self.imogolite_peak_radius_edit.text())
            if q_nm is None:
                q_nm = self.q_from_peak_radius_px(radius_px)
            elif radius_px is None:
                radius_px = self.peak_radius_px_from_q(q_nm)
                self.imogolite_peak_radius_edit.setText(f"{radius_px:.6g}")
            d_nm = 2.0 * np.pi / q_nm
            volume_fraction = self.imogolite_volume_fraction_from_q(q_nm)
        except Exception as exc:
            self.imogolite_result_label.setText(f"Could not calculate:\n{exc}")
            return

        self.imogolite_peak_q_edit.setText(f"{q_nm:.6g}")
        self.imogolite_result_label.setText(
            f"q = {q_nm:.6g} nm⁻¹\n"
            f"r = {radius_px:.6g} px\n"
            f"d = 2π/q = {d_nm:.6g} nm\n"
            f"φ = {volume_fraction:.6g}"
        )

    def imogolite_tube_diameter_nm(self):
        diameter = self.optional_float_from_text(self.imogolite_tube_diameter_edit.text())
        if diameter is None or diameter <= 0:
            raise ValueError("Tube diameter must be > 0 nm.")
        return float(diameter)

    def imogolite_frame_step_um(self):
        step = self.optional_float_from_text(self.imogolite_frame_step_combo.currentText())
        if step is None or step <= 0:
            raise ValueError("Frame step must be > 0 µm.")
        return float(step)

    def imogolite_volume_fraction_from_q(self, q_nm):
        if q_nm is None or q_nm <= 0:
            raise ValueError("Peak q must be > 0 nm⁻¹.")
        d_nm = 2.0 * np.pi / float(q_nm)
        tube_diameter_nm = self.imogolite_tube_diameter_nm()
        swelling_prefactor = np.sqrt(np.pi * np.sqrt(3.0) / 8.0)
        return float(((swelling_prefactor * tube_diameter_nm) / d_nm) ** 2)

    def parabolic_peak_q(self, q, y):
        q = np.asarray(q, dtype=float)
        y = np.asarray(y, dtype=float)
        valid = np.isfinite(q) & np.isfinite(y) & (q > 0)
        if not np.any(valid):
            return None

        q_valid = q[valid]
        y_valid = y[valid]
        peak_index = int(np.nanargmax(y_valid))
        if peak_index == 0 or peak_index == len(q_valid) - 1:
            return float(q_valid[peak_index])

        x_fit = q_valid[peak_index - 1:peak_index + 2]
        y_fit = y_valid[peak_index - 1:peak_index + 2]
        try:
            a, b, _c = np.polyfit(x_fit, y_fit, 2)
        except Exception:
            return float(q_valid[peak_index])

        if not np.isfinite(a) or not np.isfinite(b) or a >= 0:
            return float(q_valid[peak_index])

        refined_q = float(-b / (2.0 * a))
        if refined_q < float(x_fit[0]) or refined_q > float(x_fit[-1]):
            return float(q_valid[peak_index])
        return refined_q

    def imogolite_metrics(self, q, intensity):
        q = np.asarray(q, dtype=float)
        intensity = np.asarray(intensity, dtype=float)
        qiq = q * intensity
        qmax = self.parabolic_peak_q(q, qiq)
        if qmax is None:
            return None

        d_nm = float(2.0 * np.pi / qmax)

        try:
            volume_fraction = self.imogolite_volume_fraction_from_q(qmax)
        except Exception:
            volume_fraction = np.nan

        return {
            "qmax": qmax,
            "d_nm": d_nm,
            "volume_fraction": volume_fraction,
        }

    def imogolite_output_folder(self):
        base_folder = self.current_folder or self.folder_path
        if base_folder is None and self.current_path is not None:
            base_folder = self.current_path.parent
        if base_folder is None:
            base_folder = Path.home()
        output_folder = Path(base_folder) / "imogolite_dat"
        output_folder.mkdir(exist_ok=True)
        return output_folder

    def frame_number_from_path(self, path):
        match = re.search(r"frame[_-]?(\d+)", Path(path).stem, re.IGNORECASE)
        if match:
            return int(match.group(1))
        match = re.search(r"(\d+)", Path(path).stem)
        return int(match.group(1)) if match else None

    def add_imogolite_sample_positions(self, results):
        results = [dict(result) for result in results]
        if not results:
            return results

        try:
            frame_step_um = self.imogolite_frame_step_um()
        except Exception:
            frame_step_um = 5.0

        frames = [result.get("frame") for result in results if result.get("frame") is not None]
        reverse_origin = self.imogolite_origin_combo.currentText() == "last frame → first"

        if frames:
            first_frame = min(frames)
            last_frame = max(frames)
            for result in results:
                frame = result.get("frame")
                if frame is None:
                    result["sample_distance_um"] = None
                elif reverse_origin:
                    result["sample_distance_um"] = float(last_frame - frame) * frame_step_um
                else:
                    result["sample_distance_um"] = float(frame - first_frame) * frame_step_um
            return results

        last_index = len(results) - 1
        for index, result in enumerate(results):
            position_index = last_index - index if reverse_origin else index
            result["sample_distance_um"] = float(position_index) * frame_step_um
        return results

    def imogolite_curve_color(self, index, total):
        total = max(1, int(total))
        t = 0.0 if total == 1 else float(index) / float(total - 1)
        start = QColor(self.imogolite_gradient_start)
        end = QColor(self.imogolite_gradient_end)
        red = round(start.red() + (end.red() - start.red()) * t)
        green = round(start.green() + (end.green() - start.green()) * t)
        blue = round(start.blue() + (end.blue() - start.blue()) * t)
        return QColor(red, green, blue).name()

    def imogolite_dat_path_for_source(self, source_path):
        source_path = Path(source_path)
        return self.imogolite_output_folder() / f"{source_path.stem}_imogolite_qI.dat"

    def save_imogolite_dat(self, source_path, q, intensity, counts, metrics):
        output_path = self.imogolite_dat_path_for_source(source_path)
        q = np.asarray(q, dtype=float)
        intensity = np.asarray(intensity, dtype=float)
        counts = np.asarray(counts, dtype=float)
        if counts.shape != q.shape:
            counts = np.ones_like(q)

        data = np.column_stack([q, intensity, q * intensity, counts])
        with output_path.open("w", encoding="utf-8") as handle:
            handle.write("# LRPhoton imogolite integrated curve\n")
            handle.write(f"# source = {Path(source_path)}\n")
            if metrics is not None:
                handle.write(f"# qmax_nm-1 = {metrics['qmax']:.10g}\n")
                handle.write(f"# d_nm = {metrics['d_nm']:.10g}\n")
                handle.write(f"# volume_fraction = {metrics['volume_fraction']:.10g}\n")
            handle.write("# columns: q_nm-1 I_q qI_q counts\n")
            np.savetxt(handle, data, fmt="%.10g")
        return output_path

    def load_imogolite_dat(self, path):
        data = np.loadtxt(path, comments="#")
        data = np.atleast_2d(data)
        if data.shape[1] < 2:
            raise ValueError("DAT file must contain at least q and I(q).")
        q = np.asarray(data[:, 0], dtype=float)
        intensity = np.asarray(data[:, 1], dtype=float)
        counts = np.asarray(data[:, 3], dtype=float) if data.shape[1] >= 4 else np.ones_like(q)
        valid = np.isfinite(q) & np.isfinite(intensity)
        return q[valid], intensity[valid], counts[valid]

    def update_imogolite_results_table(self, results):
        self.imogolite_results = self.add_imogolite_sample_positions(results)
        self.imogolite_results_table.setRowCount(len(self.imogolite_results))

        for row, result in enumerate(self.imogolite_results):
            frame = result.get("frame")
            sample_distance_um = result.get("sample_distance_um")
            qmax = result.get("qmax")
            d_nm = result.get("d_nm")
            volume_fraction = result.get("volume_fraction")
            values = [
                "-" if frame is None else str(frame),
                "-" if sample_distance_um is None else f"{sample_distance_um:.6g}",
                "-" if qmax is None else f"{qmax:.6g}",
                "-" if d_nm is None else f"{d_nm:.6g}",
                "-" if volume_fraction is None or not np.isfinite(volume_fraction) else f"{volume_fraction:.6g}",
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignCenter)
                self.imogolite_results_table.setItem(row, column, item)

        self.imogolite_results_table.resizeColumnsToContents()

    def copy_imogolite_results_table(self):
        table = self.imogolite_results_table
        if table.rowCount() == 0:
            self.status_label.setText("No imogolite table to copy.")
            return

        selected_ranges = table.selectedRanges()
        if selected_ranges:
            selected_range = selected_ranges[0]
            row_start = selected_range.topRow()
            row_end = selected_range.bottomRow()
            column_start = selected_range.leftColumn()
            column_end = selected_range.rightColumn()
        else:
            row_start = 0
            row_end = table.rowCount() - 1
            column_start = 0
            column_end = table.columnCount() - 1

        lines = []
        headers = [
            table.horizontalHeaderItem(column).text()
            for column in range(column_start, column_end + 1)
        ]
        lines.append("\t".join(headers))

        for row in range(row_start, row_end + 1):
            values = []
            for column in range(column_start, column_end + 1):
                item = table.item(row, column)
                values.append("" if item is None else item.text())
            lines.append("\t".join(values))

        QApplication.clipboard().setText("\n".join(lines))
        self.status_label.setText(f"Copied {row_end - row_start + 1} table row(s) to clipboard.")

    def update_imogolite_table_only(self):
        if not self.imogolite_results:
            return
        refreshed = []
        for result in self.imogolite_results:
            updated = dict(result)
            updated.pop("sample_distance_um", None)
            qmax = updated.get("qmax")
            if qmax is not None and qmax > 0:
                metrics = self.imogolite_metrics(np.asarray([qmax]), np.asarray([1.0 / qmax]))
                if metrics is not None:
                    updated.update(metrics)
            refreshed.append(updated)
        self.update_imogolite_results_table(refreshed)

    def use_current_max_as_imogolite_peak(self):
        if self.imogolite_q is None or self.imogolite_intensity is None:
            self.plot_imogolite_iq(self.current_image)
        if self.imogolite_q is None or self.imogolite_intensity is None:
            self.imogolite_result_label.setText("Integrate I(q) first.")
            return

        valid = np.isfinite(self.imogolite_q) & np.isfinite(self.imogolite_intensity)
        if not np.any(valid):
            self.imogolite_result_label.setText("I(q) contains no finite points.")
            return

        q = self.imogolite_q[valid]
        intensity = self.imogolite_intensity[valid]
        peak_q = float(q[int(np.nanargmax(q * intensity))])
        self.set_imogolite_peak_q(peak_q)

    def set_imogolite_peak_q(self, q_nm):
        self.imogolite_peak_q_edit.setText(f"{float(q_nm):.6g}")
        self.imogolite_peak_radius_edit.clear()
        self.calculate_imogolite_distance()

    def imogolite_paths_for_stride(self, paths):
        stride = max(1, int(self.imogolite_file_stride_spinbox.value()))
        return list(paths)[::stride]

    def integrate_imogolite_selection(self):
        paths = self.selected_files() or self.imogolite_current_paths
        if paths:
            self.imogolite_current_paths = paths
            paths_to_integrate = self.imogolite_paths_for_stride(paths)
            if paths_to_integrate:
                self.plot_imogolite_files(paths_to_integrate, total_selected=len(paths))
            return

        self.replot_current_image()

    def set_folder(self, folder):
        if folder is None:
            return
        folder = Path(folder)
        if not folder.exists():
            return
        self.folder_path = folder
        self.current_folder = folder
        self.folder_edit.setText(str(folder))
        self.refresh_files()

    def set_folder_from_external_tab(self, folder):
        self.set_folder(folder)

    def folder_from_text(self):
        self.set_folder(self.folder_edit.text().strip())

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose folder", self.folder_edit.text() or str(Path.home()))
        if folder:
            self.set_folder(folder)
            self.folder_changed.emit(Path(folder), self)

    def refresh_files(self):
        self.file_list.clear()
        if self.folder_path is None:
            return

        folder = Path(self.folder_path)
        if not folder.exists():
            return

        name_pattern = self.name_filter.text().strip() or "*"
        extension_patterns = self.extension_filter.text().split() or ["*.edf", "*.h5", "*.hdf5"]
        iterator = folder.rglob("*") if self.subfolders_checkbox.isChecked() else folder.glob("*")

        files = []
        for path in iterator:
            if not path.is_file():
                continue
            lower_name = path.name.lower()
            if not any(fnmatch.fnmatch(lower_name, pattern.lower()) for pattern in extension_patterns):
                continue
            if not fnmatch.fnmatch(path.name.lower(), name_pattern.lower()):
                continue
            files.append(path)

        for path in sorted(files):
            item = QListWidgetItem(str(path.relative_to(folder)))
            set_item_file_path(item, path)
            self.file_list.addItem(item)

    def selected_file(self):
        items = self.file_list.selectedItems()
        if not items:
            return None
        path = items[0].data(Qt.UserRole)
        if path is None and self.current_folder is not None:
            path = Path(self.current_folder) / items[0].text()
        return Path(path) if path is not None else None

    def selected_files(self):
        paths = []
        for item in self.file_list.selectedItems():
            path = item.data(Qt.UserRole)
            if path is None and self.current_folder is not None:
                path = Path(self.current_folder) / item.text()
            if path is not None:
                paths.append(Path(path))
        return paths

    def open_selected_files(self):
        if self.current_project() == "Imogolite distance":
            paths = self.selected_files()
            if not paths:
                return
            self.imogolite_current_paths = paths
            self.status_label.setText(f"{len(paths)} imogolite file(s) selected. Click Integrate to calculate qI(q).")
            return

        path = self.selected_file()
        if path is None:
            return
        try:
            stack = self.load_stack(path)
        except Exception as exc:
            self.status_label.setText(f"Could not open file: {exc}")
            return

        self.current_path = path
        self.current_stack = stack
        self.current_frame_count = stack.shape[0]
        self.current_frame_index = 0
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, max(0, self.current_frame_count - 1))
        self.frame_slider.setValue(0)
        self.frame_slider.blockSignals(False)
        self.update_frame_controls()
        self.set_current_image_from_stack()

    def load_stack(self, path):
        suffix = path.suffix.lower()
        if suffix == ".edf":
            return self.load_edf_stack(path)
        if suffix in {".h5", ".hdf5"}:
            import h5py

            return self.load_h5_stack(path, h5py)
        raise ValueError(f"Unsupported file type: {suffix}")

    def load_edf_stack(self, path):
        import fabio

        edf = fabio.open(str(path))
        frames = []
        index = 0
        while edf is not None:
            data = np.asarray(edf.data, dtype=float)
            frames.append(self.clean_image(data))
            index += 1
            try:
                edf = edf.next()
            except Exception:
                break
        if not frames:
            raise ValueError("No EDF frame found")
        return np.stack(frames, axis=0)

    def clean_image(self, data):
        data = np.asarray(data, dtype=float)
        if data.ndim != 2:
            raise ValueError(f"Expected a 2D image, got shape {data.shape}")
        data[data > 4e9] = np.nan
        return data

    def load_h5_stack(self, path, h5py):
        candidates = []

        with h5py.File(path, "r") as h5:
            def visitor(name, obj):
                if hasattr(obj, "shape") and hasattr(obj, "dtype"):
                    shape = tuple(obj.shape)
                    if len(shape) in {2, 3}:
                        candidates.append((name, shape))

            h5.visititems(visitor)
            if not candidates:
                raise ValueError("No 2D or 3D dataset found in H5 file")

            name, shape = sorted(candidates, key=lambda item: (len(item[1]) != 3, item[1]))[0]
            dataset = h5[name]
            data = np.asarray(dataset[...], dtype=float)

        if data.ndim == 2:
            data = data[np.newaxis, ...]
        elif data.ndim == 3:
            pass
        else:
            raise ValueError(f"Expected a 2D or 3D dataset, got shape {data.shape}")

        frames = [self.clean_image(frame) for frame in data]
        return np.stack(frames, axis=0)

    def set_current_image_from_stack(self):
        if self.current_stack is None:
            return
        self.current_frame_index = max(0, min(self.current_frame_index, self.current_frame_count - 1))
        self.current_image = self.current_stack[self.current_frame_index]
        self.update_frame_controls()
        self.replot_current_image()

    def update_frame_controls(self):
        self.frame_label.setText(f"Frame {self.current_frame_index + 1} / {self.current_frame_count}")
        enabled = self.current_frame_count > 1
        self.previous_frame_button.setEnabled(enabled and self.current_frame_index > 0)
        self.next_frame_button.setEnabled(enabled and self.current_frame_index < self.current_frame_count - 1)
        self.frame_slider.setEnabled(enabled)

    def set_frame_from_slider(self, value):
        if self.current_stack is None:
            return
        self.current_frame_index = int(value)
        self.set_current_image_from_stack()

    def previous_frame(self):
        if self.current_stack is None:
            return
        self.current_frame_index = max(0, self.current_frame_index - 1)
        self.frame_slider.setValue(self.current_frame_index)

    def next_frame(self):
        if self.current_stack is None:
            return
        self.current_frame_index = min(self.current_frame_count - 1, self.current_frame_index + 1)
        self.frame_slider.setValue(self.current_frame_index)

    def replot_current_image(self):
        if self.current_image is not None:
            if self.current_project() == "Imogolite distance":
                paths = self.selected_files() or self.imogolite_current_paths
                if paths:
                    self.plot_imogolite_files(self.imogolite_paths_for_stride(paths), total_selected=len(paths))
                else:
                    self.plot_imogolite_iq(self.current_image)
            else:
                self.plot_3d(self.current_image)

    def update_display_precision(self, value):
        self.max_3d_points = int(value)
        self.replot_current_image()

    def auto_contrast(self):
        self.contrast_min_slider.blockSignals(True)
        self.contrast_max_slider.blockSignals(True)
        self.contrast_min_slider.setValue(10)
        self.contrast_max_slider.setValue(990)
        self.contrast_min_slider.blockSignals(False)
        self.contrast_max_slider.blockSignals(False)
        self.z_max_edit.clear()
        self.replot_current_image()

    def reset_intensity_scale(self):
        self.z_min_edit.clear()
        self.z_max_edit.clear()
        self.replot_current_image()

    def imogolite_iq_profile(self, image):
        self.current_geometry = self.read_geometry_from_fields()
        center_x = self.geometry_value("center_x")
        center_y = self.geometry_value("center_y")
        pixel_size_um = self.geometry_value("pixel_size")
        distance_mm = self.geometry_value("distance")
        wavelength_a = self.geometry_value("wavelength")

        if center_x is None or center_y is None:
            raise ValueError("Set Center X and Center Y first.")
        if pixel_size_um is None or pixel_size_um <= 0:
            raise ValueError("Pixel size must be > 0 µm.")
        if distance_mm is None or distance_mm <= 0:
            raise ValueError("Distance must be > 0 mm.")
        if wavelength_a is None or wavelength_a <= 0:
            raise ValueError("Wavelength must be > 0 Å.")

        image = np.asarray(image, dtype=float)
        q_min_limit = 0.25
        q_max_limit = 2.0
        n_bins = int(self.imogolite_bins_spinbox.value())
        distance_m = distance_mm * 1e-3
        pixel_mm = pixel_size_um * 1e-3

        try:
            try:
                from pyFAI.integrator.azimuthal import AzimuthalIntegrator
            except Exception:
                from pyFAI.azimuthalIntegrator import AzimuthalIntegrator

            invalid_mask = ~np.isfinite(image) | (image <= 0) | (image >= 4e9)
            integrator = AzimuthalIntegrator(
                dist=float(distance_m),
                poni1=float(center_y) * pixel_mm * 1e-3,
                poni2=float(center_x) * pixel_mm * 1e-3,
                pixel1=pixel_mm * 1e-3,
                pixel2=pixel_mm * 1e-3,
                wavelength=wavelength_a * 1e-10,
            )
            result = integrator.integrate1d(
                image.astype(np.float64),
                n_bins,
                unit="q_nm^-1",
                radial_range=(q_min_limit, q_max_limit),
                mask=invalid_mask,
                method=("bbox", "csr", "cython"),
                correctSolidAngle=True,
            )
            q = np.asarray(getattr(result, "radial", result[0]), dtype=float)
            intensity = np.asarray(getattr(result, "intensity", result[1]), dtype=float)

            valid = np.isfinite(q) & np.isfinite(intensity) & (q >= q_min_limit) & (q <= q_max_limit) & (intensity > 0)
            if not np.any(valid):
                raise ValueError("pyFAI returned no finite I(q) points.")
            return q[valid], intensity[valid], np.ones(np.count_nonzero(valid), dtype=int)
        except Exception:
            q, intensity, counts, _mask = radial_average(
                image,
                center_x,
                center_y,
                distance_m,
                pixel_mm,
                pixel_mm,
                wavelength_a,
                q_min_limit,
                q_max_limit,
                n_bins,
                False,
                0,
                360,
                1,
            )
            return q, intensity, counts

    def first_frame_from_path(self, path):
        stack = self.load_stack(path)
        if stack.shape[0] < 1:
            raise ValueError("No frame found")
        return stack[0]

    def plot_imogolite_files(self, paths, total_selected=None):
        paths = list(paths)
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.ax = ax
        self.imogolite_q = None
        self.imogolite_intensity = None

        plot_mode = self.imogolite_plot_mode_combo.currentText()
        messages = []
        plotted_count = 0
        results = []
        saved_count = 0

        for path_index, path in enumerate(paths):
            image = None
            try:
                if path.suffix.lower() == ".dat":
                    q, intensity, counts = self.load_imogolite_dat(path)
                else:
                    image = self.first_frame_from_path(path)
                    q, intensity, counts = self.imogolite_iq_profile(image)
            except Exception as exc:
                messages.append(f"{path.name}: {exc}")
                continue

            plotted_intensity = q * intensity
            valid = np.isfinite(q) & np.isfinite(plotted_intensity)
            if plot_mode in {"log log", "log lin"}:
                valid &= q > 0
            if plot_mode in {"log log", "lin log"}:
                valid &= plotted_intensity > 0
            if not np.any(valid):
                messages.append(f"{path.name}: no finite qI(q) points for this scale")
                continue

            metrics = self.imogolite_metrics(q, intensity)
            if metrics is None:
                messages.append(f"{path.name}: could not determine qmax")
                continue

            if path.suffix.lower() != ".dat":
                try:
                    self.save_imogolite_dat(path, q, intensity, counts, metrics)
                    saved_count += 1
                except Exception as exc:
                    messages.append(f"{path.name}: could not save DAT ({exc})")

            color = self.imogolite_curve_color(path_index, len(paths))
            ax.plot(q[valid], plotted_intensity[valid], linewidth=1.1, color=color, label=path.stem)
            ax.axvline(metrics["qmax"], color=color, linestyle=":", linewidth=0.6, alpha=0.5)
            plotted_count += 1

            if self.imogolite_q is None:
                self.imogolite_q = q
                self.imogolite_intensity = intensity
                self.current_path = path
                if image is not None:
                    self.current_image = image

            results.append({
                "path": path,
                "frame": self.frame_number_from_path(path),
                "qmax": metrics["qmax"],
                "d_nm": metrics["d_nm"],
                "volume_fraction": metrics["volume_fraction"],
            })

        ax.set_xlabel("q / nm⁻¹")
        ax.set_ylabel("qI(q)")
        ax.set_xscale("log" if plot_mode in {"log log", "log lin"} else "linear")
        ax.set_yscale("log" if plot_mode in {"log log", "lin log"} else "linear")
        ax.set_xlim(0.25, 2.0)
        ax.set_title("Integrated qI(q)")
        ax.grid(True, alpha=0.25)
        if plotted_count and self.imogolite_legend_checkbox.isChecked():
            ax.legend(loc="best", frameon=True)

        peak_q = self.optional_float_from_text(self.imogolite_peak_q_edit.text())
        if peak_q is not None:
            ax.axvline(peak_q, color="#d62728", linestyle="--", linewidth=1.1)

        self.figure.tight_layout()
        self.canvas.draw_idle()
        self.update_imogolite_results_table(results)
        selected_count = total_selected if total_selected is not None else len(paths)
        stride = self.imogolite_file_stride_spinbox.value()
        if messages:
            self.status_label.setText(
                f"Integrated/plotted {plotted_count} / {selected_count} selected file(s), every {stride}. "
                + " | ".join(messages[:3])
            )
        else:
            self.status_label.setText(
                f"Integrated/plotted {plotted_count} / {selected_count} selected file(s), every {stride}. "
                f"Saved {saved_count} DAT curve(s) in imogolite_dat."
            )

    def plot_imogolite_iq(self, image):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.ax = ax
        self.imogolite_q = None
        self.imogolite_intensity = None

        if image is None:
            self.status_label.setText("Open a cave EDF/H5 image first.")
            self.canvas.draw_idle()
            return

        try:
            q, intensity, counts = self.imogolite_iq_profile(image)
        except Exception as exc:
            self.status_label.setText(f"Could not integrate I(q): {exc}")
            self.canvas.draw_idle()
            return

        self.imogolite_q = q
        self.imogolite_intensity = intensity
        plotted_intensity = q * intensity
        metrics = self.imogolite_metrics(q, intensity)
        if metrics is not None and self.current_path is not None:
            try:
                self.save_imogolite_dat(self.current_path, q, intensity, counts, metrics)
            except Exception as exc:
                self.status_label.setText(f"Integrated I(q), but could not save DAT: {exc}")
        plot_mode = self.imogolite_plot_mode_combo.currentText()
        valid = np.isfinite(q) & np.isfinite(plotted_intensity)
        if plot_mode in {"log log", "log lin"}:
            valid &= q > 0
        if plot_mode in {"log log", "lin log"}:
            valid &= plotted_intensity > 0

        ax.plot(q[valid], plotted_intensity[valid], color=self.imogolite_curve_color(0, 1), linewidth=1.2)
        if metrics is not None:
            ax.axvline(metrics["qmax"], color=self.imogolite_curve_color(0, 1), linestyle=":", linewidth=0.8, alpha=0.6)
        ax.set_xlabel("q / nm⁻¹")
        ax.set_ylabel("qI(q)")
        if plot_mode in {"log log", "log lin"}:
            ax.set_xscale("log")
        else:
            ax.set_xscale("linear")
        if plot_mode in {"log log", "lin log"}:
            ax.set_yscale("log")
        else:
            ax.set_yscale("linear")
        ax.set_xlim(0.25, 2.0)

        title = "Integrated qI(q)"
        if self.current_path is not None:
            title += f" - {self.current_path.name}"
        ax.set_title(title)
        ax.grid(True, alpha=0.25)

        peak_q = self.optional_float_from_text(self.imogolite_peak_q_edit.text())
        if peak_q is not None:
            ax.axvline(peak_q, color="#d62728", linestyle="--", linewidth=1.1)

        self.figure.tight_layout()
        self.canvas.draw_idle()
        if metrics is not None:
            self.update_imogolite_results_table([{
                "path": self.current_path,
                "frame": self.frame_number_from_path(self.current_path) if self.current_path is not None else None,
                "qmax": metrics["qmax"],
                "d_nm": metrics["d_nm"],
                "volume_fraction": metrics["volume_fraction"],
            }])
        self.status_label.setText("Integrated qI(q), saved DAT when possible, and calculated qmax.")

    def on_canvas_click(self, event):
        if self.current_project() != "Imogolite distance":
            return
        if event.inaxes is not self.ax or event.xdata is None:
            return
        if self.imogolite_q is None:
            return

        q_values = np.asarray(self.imogolite_q, dtype=float)
        if q_values.size == 0:
            return
        index = int(np.nanargmin(np.abs(q_values - float(event.xdata))))
        self.set_imogolite_peak_q(float(q_values[index]))
        paths = self.selected_files() or self.imogolite_current_paths
        if paths:
            self.plot_imogolite_files(self.imogolite_paths_for_stride(paths), total_selected=len(paths))
        else:
            self.plot_imogolite_iq(self.current_image)

    def plot_colored_wireframe(self, ax, xx, yy, z, vmin, vmax):
        z_data = np.asarray(z.filled(np.nan), dtype=float)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.get_cmap("jet")

        for row_index in range(z_data.shape[0]):
            row_z = z_data[row_index, :]
            finite = np.isfinite(row_z)
            if np.count_nonzero(finite) < 2:
                continue
            row_color = cmap(norm(np.nanmean(row_z[finite])))
            ax.plot(
                xx[row_index, finite],
                yy[row_index, finite],
                row_z[finite],
                color=row_color,
                linewidth=0.45,
            )

        for column_index in range(z_data.shape[1]):
            column_z = z_data[:, column_index]
            finite = np.isfinite(column_z)
            if np.count_nonzero(finite) < 2:
                continue
            column_color = cmap(norm(np.nanmean(column_z[finite])))
            ax.plot(
                xx[finite, column_index],
                yy[finite, column_index],
                column_z[finite],
                color=column_color,
                linewidth=0.45,
            )

    def plot_3d(self, image):
        self.figure.clear()
        ax = self.figure.add_subplot(111, projection="3d")
        self.ax = ax

        display = np.array(image, dtype=float)
        finite = np.isfinite(display)
        if not np.any(finite):
            self.status_label.setText("Image contains no finite intensity values.")
            self.canvas.draw_idle()
            return

        finite_display = np.isfinite(display)
        low_percentile = self.contrast_min_slider.value() / 10.0
        high_percentile = self.contrast_max_slider.value() / 10.0
        if high_percentile <= low_percentile:
            high_percentile = min(100.0, low_percentile + 0.1)
        vmin = np.nanpercentile(display[finite_display], low_percentile)
        vmax = np.nanpercentile(display[finite_display], high_percentile)
        if vmax <= vmin:
            vmax = vmin + 1e-12

        auto_z_min = float(np.nanmin(display[finite_display]))
        auto_z_max = float(np.nanmax(display[finite_display]))
        z_min = self.optional_float_from_text(self.z_min_edit.text())
        z_max = self.optional_float_from_text(self.z_max_edit.text())
        if z_min is None:
            z_min = auto_z_min
        if z_max is None:
            z_max = auto_z_max
        if z_max <= z_min:
            z_max = z_min + 1e-12

        max_points = self.max_3d_points
        precision_step = self.precision_step_spinbox.value()
        step_y = max(1, int(np.ceil(display.shape[0] / max_points))) * precision_step
        step_x = max(1, int(np.ceil(display.shape[1] / max_points))) * precision_step

        z = display[::step_y, ::step_x]
        y = np.arange(0, display.shape[0], step_y)
        x = np.arange(0, display.shape[1], step_x)
        xx, yy = np.meshgrid(x, y)

        z = np.asarray(z, dtype=float)
        z[(z > z_max) | (z < z_min)] = np.nan
        z = np.ma.masked_invalid(z)
        z_plot = z.filled(np.nan)
        if z.count() == 0:
            self.status_label.setText("3D downsample contains no finite intensity values.")
            self.canvas.draw_idle()
            return

        geometry = self.current_geometry
        center_x = geometry.get("center_x")
        center_y = geometry.get("center_y")
        if center_x is not None and center_y is not None:
            ax.set_xlabel("x - center (px)")
            ax.set_ylabel("y - center (px)")
            ax.set_xlim(float(np.nanmin(xx - center_x)), float(np.nanmax(xx - center_x)))
            ax.set_ylim(float(np.nanmin(yy - center_y)), float(np.nanmax(yy - center_y)))
            if self.wireframe_checkbox.isChecked():
                self.plot_colored_wireframe(ax, xx - center_x, yy - center_y, z, max(vmin, z_min), min(vmax, z_max))
            else:
                surface = ax.plot_surface(
                    xx - center_x,
                    yy - center_y,
                    z_plot,
                    cmap="jet",
                    vmin=max(vmin, z_min),
                    vmax=min(vmax, z_max),
                    linewidth=0,
                    antialiased=False,
                    rstride=1,
                    cstride=1,
                )
                surface.set_clim(max(vmin, z_min), min(vmax, z_max))
        else:
            ax.set_xlabel("x (px)")
            ax.set_ylabel("y (px)")
            if self.wireframe_checkbox.isChecked():
                self.plot_colored_wireframe(ax, xx, yy, z, max(vmin, z_min), min(vmax, z_max))
            else:
                surface = ax.plot_surface(
                    xx,
                    yy,
                    z_plot,
                    cmap="jet",
                    vmin=max(vmin, z_min),
                    vmax=min(vmax, z_max),
                    linewidth=0,
                    antialiased=False,
                    rstride=1,
                    cstride=1,
                )
                surface.set_clim(max(vmin, z_min), min(vmax, z_max))
        ax.set_zlim(z_min, z_max)
        ax.set_zlabel("Intensity")
        ax.view_init(elev=35, azim=-60)
        self.figure.tight_layout()
        self.canvas.draw_idle()
        self.status_label.clear()
