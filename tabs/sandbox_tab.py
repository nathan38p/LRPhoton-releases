from pathlib import Path
import fnmatch

import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QCheckBox,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from tabs.file_ratings import install_file_rating_menu, set_item_file_path, should_hide_file_in_browser
from tabs.line_geometry import LineGeometrySelector, line_geometry_to_lrphoton
from tabs.instrument_presets import (
    ID13_DEFAULT_CENTER_X,
    ID13_DEFAULT_CENTER_Y,
    ID13_DEFAULT_DISTANCE_M,
    ID13_DEFAULT_PIXEL_MM,
    ID13_DEFAULT_WAVELENGTH_A,
)
from tabs.ui_style import GROUP_BOX_STYLE, PAGE_MARGINS, PANEL_MARGINS, style_q_geometry_buttons
from tabs.sandbox_3d_view import Saxs3DProjectMixin
from tabs.sandbox_imogolite import ImogoliteProjectMixin
from tabs.sandbox_polynomials import PolynomialProjectMixin
from tabs.background_tab import BackgroundTab
from tabs.sandbox_header_editor import HeaderEditorTab
from tabs.tools_tab import ToolsTab


class SandboxTab(PolynomialProjectMixin, ImogoliteProjectMixin, Saxs3DProjectMixin, QWidget):
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
        self.geometry_mode = "ID13"
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

        selector_buttons = QGridLayout()
        selector_buttons.setContentsMargins(0, 0, 0, 0)
        selector_buttons.setHorizontalSpacing(14)
        selector_buttons.setVerticalSpacing(14)

        self.open_imogolite_project_button = self.make_project_button("↔️ Imogolite distance")
        self.open_imogolite_project_button.clicked.connect(lambda: self.open_sandbox_project("Imogolite distance"))
        selector_buttons.addWidget(self.open_imogolite_project_button, 0, 0)

        self.open_polynomial_project_button = self.make_project_button("🧪 Beidellite ODF")
        self.open_polynomial_project_button.clicked.connect(lambda: self.open_sandbox_project("Beidellite ODF"))
        selector_buttons.addWidget(self.open_polynomial_project_button, 0, 1)

        self.open_background_project_button = self.make_project_button("🧹 Background")
        self.open_background_project_button.clicked.connect(self.open_background_project)
        selector_buttons.addWidget(self.open_background_project_button, 1, 0)

        self.open_tools_project_button = self.make_project_button("🛠️ Tools")
        self.open_tools_project_button.clicked.connect(self.open_tools_project)
        selector_buttons.addWidget(self.open_tools_project_button, 1, 1)

        selector_layout.addLayout(selector_buttons)
        selector_layout.addStretch(1)
        self.project_stack.addWidget(selector_page)

        splitter = QSplitter(Qt.Horizontal)
        self.sandbox_workbench_page = self.wrap_sandbox_project(splitter)
        self.project_stack.addWidget(self.sandbox_workbench_page)

        file_box = QGroupBox("File browser")
        self.file_box = file_box
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

        self.plot_box = QGroupBox("Imogolite distance")
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

        self.sandbox_project_combo = QComboBox()
        self.sandbox_project_combo.addItems([
            "Imogolite distance",
            "Beidellite ODF",
        ])
        self.sandbox_project_combo.currentTextChanged.connect(self.apply_sandbox_project)
        side_layout.addLayout(self.form_row("Project", self.sandbox_project_combo))

        self.geometry_combo_label = QLabel("Geometry")
        geometry_buttons_layout = QHBoxLayout()
        geometry_buttons_layout.setContentsMargins(0, 0, 0, 0)
        geometry_buttons_layout.setSpacing(4)
        self.btn_xenocs = QPushButton("XENOCS")
        self.btn_id02 = QPushButton("ID02")
        self.btn_id13 = QPushButton("ID13")
        self.q_manual_button = QPushButton("+")
        self.btn_xenocs.clicked.connect(lambda: self.apply_geometry_preset("XENOCS"))
        self.btn_id02.clicked.connect(lambda: self.apply_geometry_preset("ID02"))
        self.btn_id13.clicked.connect(lambda: self.apply_geometry_preset("ID13"))
        self.q_manual_button.setToolTip("Edit geometry and wavelength parameters")
        self.q_manual_button.clicked.connect(self.open_sandbox_geometry_dialog)
        for button in [self.btn_xenocs, self.btn_id02, self.btn_id13, self.q_manual_button]:
            geometry_buttons_layout.addWidget(button)
            button.hide()
        self.line_geometry_selector = LineGeometrySelector(self, "ID13")
        self.line_geometry_selector.geometry_selected.connect(self.apply_line_geometry_selection)
        geometry_buttons_layout.addWidget(self.line_geometry_selector, 1)
        geometry_row = QHBoxLayout()
        geometry_row.setContentsMargins(0, 0, 0, 0)
        geometry_row.setSpacing(8)
        self.geometry_combo_label.setMinimumWidth(120)
        geometry_row.addWidget(self.geometry_combo_label, 0)
        geometry_row.addLayout(geometry_buttons_layout, 1)
        side_layout.addLayout(geometry_row)

        self.center_x_label = QLabel("Center X")
        self.center_x_edit = QLineEdit()
        self.center_x_edit.returnPressed.connect(self.apply_custom_geometry_from_fields)

        self.center_y_label = QLabel("Center Y")
        self.center_y_edit = QLineEdit()
        self.center_y_edit.returnPressed.connect(self.apply_custom_geometry_from_fields)

        self.pixel_size_label = QLabel("Pixel size (µm)")
        self.pixel_size_edit = QLineEdit()
        self.pixel_size_edit.returnPressed.connect(self.apply_custom_geometry_from_fields)

        self.distance_label = QLabel("Distance (mm)")
        self.distance_edit = QLineEdit()
        self.distance_edit.returnPressed.connect(self.apply_custom_geometry_from_fields)

        self.wavelength_label = QLabel("Wavelength (Å)")
        self.wavelength_edit = QLineEdit()
        self.wavelength_edit.returnPressed.connect(self.apply_custom_geometry_from_fields)

        self.apply_geometry_button = QPushButton("Apply")
        self.apply_geometry_button.clicked.connect(self.apply_custom_geometry_from_fields)
        self.geometry_widgets = [
            self.geometry_combo_label,
            self.btn_xenocs,
            self.btn_id02,
            self.btn_id13,
            self.q_manual_button,
        ]
        self.update_sandbox_geometry_buttons()

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

        self.build_polynomial_controls(side_layout)

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

        self.status_label = QLabel("Open one or more cave EDF/H5 images or saved .dat curves, then integrate qI(q).")
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
        self.frame_bottom_widgets = [
            self.previous_frame_button,
            self.frame_slider,
            self.next_frame_button,
            self.frame_label,
        ]
        self.apply_sandbox_project(self.sandbox_project_combo.currentText())

        splitter.addWidget(file_box)
        splitter.addWidget(self.plot_box)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self.background_project = BackgroundTab()
        self.background_project.folder_changed.connect(lambda folder: self.folder_changed.emit(Path(folder), self))
        self.background_project_page = self.wrap_sandbox_project(self.background_project)
        self.project_stack.addWidget(self.background_project_page)

        self.header_editor_project = HeaderEditorTab()
        self.header_editor_project.folder_changed.connect(lambda folder: self.folder_changed.emit(Path(folder), self))
        self.header_editor_project_page = self.wrap_sandbox_project(self.header_editor_project, show_back_button=False)
        self.project_stack.addWidget(self.header_editor_project_page)

        self.tools_project = ToolsTab()
        self.tools_project.folder_changed.connect(lambda folder: self.folder_changed.emit(Path(folder), self))
        self.tools_project_page = self.wrap_sandbox_project(self.tools_project)
        self.project_stack.addWidget(self.tools_project_page)

    def make_project_button(self, text):
        button = QPushButton(text)
        button.setMinimumSize(190, 70)
        button.setCursor(Qt.PointingHandCursor)
        button.setStyleSheet("""
            QPushButton {
                background-color: #fde68a;
                color: #222222;
                border: 0px;
                border-radius: 14px;
                padding: 10px 16px;
                font-size: 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #facc15;
            }
            QPushButton:pressed {
                background-color: #d97706;
                color: white;
            }
        """)
        return button

    def wrap_sandbox_project(self, content_widget, show_back_button=True):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        if show_back_button:
            layout.addWidget(self.make_back_to_projects_button(), 0)
        layout.addWidget(content_widget, 1)
        return page

    def make_back_to_projects_button(self):
        button = QPushButton("← Sandbox projects")
        button.clicked.connect(self.show_project_selector)
        button.setCursor(Qt.PointingHandCursor)
        return button

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

    def is_beidellite_odf_project(self):
        return self.current_project() in {"Beidellite ODF", "Polynomes"}

    def open_sandbox_project(self, name):
        self.sandbox_project_combo.blockSignals(True)
        self.sandbox_project_combo.setCurrentText(name)
        self.sandbox_project_combo.blockSignals(False)
        self.apply_sandbox_project(name)
        self.project_stack.setCurrentWidget(self.sandbox_workbench_page)

    def open_background_project(self):
        self.project_stack.setCurrentWidget(self.background_project_page)

    def open_header_editor_project(self):
        self.project_stack.setCurrentWidget(self.header_editor_project_page)

    def open_tools_project(self):
        self.project_stack.setCurrentWidget(self.tools_project_page)

    def apply_sandbox_project(self, name):
        is_imogolite = name == "Imogolite distance"
        is_polynomial = name in {"Beidellite ODF", "Polynomes"}

        self.file_list.setSelectionMode(
            QAbstractItemView.ExtendedSelection if is_imogolite else QAbstractItemView.SingleSelection
        )
        self.name_filter.blockSignals(True)
        self.name_filter.setText("*cave*" if (is_imogolite or is_polynomial) else "*")
        self.name_filter.blockSignals(False)
        self.extension_filter.blockSignals(True)
        self.extension_filter.setText("*.h5 *.hdf5 *.edf *.dat" if is_imogolite else "*.h5 *.hdf5 *.edf")
        self.extension_filter.blockSignals(False)
        self.subfolders_checkbox.setChecked(is_imogolite)
        self.file_box.setVisible(True)
        self.refresh_files()

        self.imogolite_peak_box.setVisible(is_imogolite)
        self.polynomial_box.setVisible(is_polynomial)
        for widget in self.geometry_widgets:
            widget.setVisible(True)

        for widget in self.saxs_3d_widgets:
            widget.setVisible(False)
        for widget in self.saxs_3d_bottom_widgets:
            widget.setVisible(False)
        for widget in self.frame_bottom_widgets:
            widget.setVisible(True)

        if is_polynomial:
            self.apply_polynomial_project()
        elif is_imogolite:
            self.plot_box.setTitle("Imogolite distance")
            paths = self.selected_files()
            if paths:
                self.imogolite_current_paths = paths
                self.status_label.setText(f"{len(paths)} imogolite file(s) selected. Choose Every N files, then click Integrate.")
            elif self.current_image is None:
                self.status_label.setText("Open one or more cave EDF/H5 images or saved .dat curves, then integrate qI(q).")
            else:
                self.status_label.setText("Choose cave files or .dat curves, then click Integrate.")
        else:
            self.plot_box.setTitle("Sandbox")
            self.status_label.setText("Choose a Sandbox project.")

    def apply_geometry_preset(self, name):
        if name == "+":
            self.open_sandbox_geometry_dialog()
            return
        preset = self.geometry_presets.get(name)
        if preset is None:
            return
        self.geometry_mode = name
        self.current_geometry = preset.copy()
        self.update_geometry_fields()
        self.update_sandbox_geometry_buttons()
        self.replot_current_image()

    def apply_line_geometry_selection(self, name, geometry):
        values = line_geometry_to_lrphoton(geometry)
        self.geometry_mode = "Custom" if name not in {"XENOCS", "ID02", "ID13"} else name
        self.current_geometry = {
            "center_x": values["xc"],
            "center_y": values["yc"],
            "pixel_size": values["pixel_x_mm"] * 1000.0,
            "distance": values["distance_m"] * 1000.0,
            "wavelength": values["wavelength_a"],
        }
        self.update_geometry_fields()
        self.update_sandbox_geometry_buttons()
        self.replot_current_image()

    def update_sandbox_geometry_buttons(self):
        buttons = {
            "XENOCS": self.btn_xenocs,
            "ID02": self.btn_id02,
            "ID13": self.btn_id13,
        }
        style_q_geometry_buttons(buttons, self.geometry_mode, self.q_manual_button)

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
        self.geometry_mode = "Custom"
        self.update_sandbox_geometry_buttons()
        self.replot_current_image()

    def open_sandbox_geometry_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Geometry")
        layout = QVBoxLayout(dialog)
        form = QFormLayout()

        fields = [
            ("Center X", self.center_x_edit.text()),
            ("Center Y", self.center_y_edit.text()),
            ("Pixel size (µm)", self.pixel_size_edit.text()),
            ("Distance (mm)", self.distance_edit.text()),
            ("Wavelength (Å)", self.wavelength_edit.text()),
        ]
        dialog_edits = []
        for label, value in fields:
            edit = QLineEdit(value)
            dialog_edits.append(edit)
            form.addRow(label, edit)
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() == QDialog.Accepted:
            (
                center_x_edit,
                center_y_edit,
                pixel_size_edit,
                distance_edit,
                wavelength_edit,
            ) = dialog_edits
            self.center_x_edit.setText(center_x_edit.text())
            self.center_y_edit.setText(center_y_edit.text())
            self.pixel_size_edit.setText(pixel_size_edit.text())
            self.distance_edit.setText(distance_edit.text())
            self.wavelength_edit.setText(wavelength_edit.text())
            self.apply_custom_geometry_from_fields()
        else:
            self.update_geometry_fields()

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
        if hasattr(self, "background_project"):
            self.background_project.set_folder_from_external_tab(str(folder))
        if hasattr(self, "header_editor_project"):
            self.header_editor_project.set_folder_from_external_tab(str(folder))
        if hasattr(self, "tools_project"):
            self.tools_project.set_folder_from_external_tab(str(folder))

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
            if should_hide_file_in_browser(path):
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
        if self.is_beidellite_odf_project():
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
            return
        if self.current_project() == "Imogolite distance":
            paths = self.selected_files()
            if not paths:
                return
            self.imogolite_current_paths = paths
            self.status_label.setText(f"{len(paths)} imogolite file(s) selected. Click Integrate to calculate qI(q).")
            return

        self.status_label.setText("Choose a Sandbox project.")

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
        if self.is_beidellite_odf_project():
            self.update_polynomial_analysis()
            return
        if self.current_image is not None:
            if self.current_project() == "Imogolite distance":
                paths = self.selected_files() or self.imogolite_current_paths
                if paths:
                    self.plot_imogolite_files(self.imogolite_paths_for_stride(paths), total_selected=len(paths))
                else:
                    self.plot_imogolite_iq(self.current_image)
            else:
                self.status_label.setText("Choose a Sandbox project.")
