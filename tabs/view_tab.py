import fnmatch
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
import matplotlib.pyplot as plt

from PySide6.QtCore import Qt, QSettings, QSize, Signal

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMessageBox

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
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
    QStyle
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

        self.canvas = FigureCanvas(self.fig)

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

        self.save_colorbar_checkbox = QCheckBox("Save colorbar")
        self.save_colorbar_checkbox.setChecked(
            self.settings.value("view/save_colorbar", False, type=bool)
        )
        self.save_colorbar_checkbox.stateChanged.connect(self.save_colorbar_setting)

        self.vmin_spin = QDoubleSpinBox()
        self.vmin_spin.setDecimals(4)
        self.vmin_spin.setRange(-999999, 999999)
        self.vmin_spin.setValue(0)
        self.vmin_spin.valueChanged.connect(self.spin_intensity_changed)

        self.vmax_spin = QDoubleSpinBox()
        self.vmax_spin.setDecimals(4)
        self.vmax_spin.setRange(-999999, 999999)
        self.vmax_spin.setValue(5)
        self.vmax_spin.valueChanged.connect(self.spin_intensity_changed)

        autoscale_button = QPushButton("Auto intensity")
        autoscale_button.clicked.connect(self.auto_intensity)

        display_layout.addWidget(self.log_checkbox)
        display_layout.addWidget(self.keep_ratio_checkbox)
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
        self.settings.setValue(
            "view/save_colorbar",
            self.save_colorbar_checkbox.isChecked()
        )

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

            match_extension = any(
                fnmatch.fnmatch(path.name.lower(), pattern.lower())
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
        self.auto_intensity()
        self.update_image()

    # ============================================================
    # INFORMATION
    # ============================================================

    def update_file_information(self, file_type, dataset_name, n_frames, image_shape):
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

        self.info_text.setPlainText("\n".join(lines))

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
            "Center1",
            "BeamCenter_1",
            "BeamCenterX",
            "Center_X",
            "CenterX",
            "Poni1",
            "Beam_x",
            "beam_x"
        ]

        possible_y_keys = [
            "Center_2",
            "Center2",
            "BeamCenter_2",
            "BeamCenterY",
            "Center_Y",
            "CenterY",
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

    def draw_center_cross(self):
        for artist in self.center_artists:
            try:
                artist.remove()
            except Exception:
                pass

        self.center_artists = []

        center = self.get_center_from_header()

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

        self.ax.set_title("")

        self.draw_center_cross()

        total = self.n_frames if self.is_lazy_h5 else self.images.shape[0]
        self.frame_label.setText(f"{self.current_index + 1} / {total}")

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

        self.update_image()

    def value_to_slider(self, value):
        if self.intensity_max == self.intensity_min:
            return 0

        return int(
            1000
            * (value - self.intensity_min)
            / (self.intensity_max - self.intensity_min)
        )

    def slider_to_value(self, value):
        return self.intensity_min + (
            value / 1000
        ) * (self.intensity_max - self.intensity_min)

    def vertical_sliders_changed(self):
        vmin = self.slider_to_value(self.min_slider.value())
        vmax = self.slider_to_value(self.max_slider.value())

        if vmin >= vmax:
            return

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
        if self.raw_current_img is None or event.inaxes != self.ax:
            self.cursor_label.setText("x = - | y = - | I = -")
            return

        if event.xdata is None or event.ydata is None:
            self.cursor_label.setText("x = - | y = - | I = -")
            return

        x_index = int(round(event.xdata))
        y_index = int(round(event.ydata))

        ny, nx = self.raw_current_img.shape

        if not (0 <= x_index < nx and 0 <= y_index < ny):
            self.cursor_label.setText("x = - | y = - | I = -")
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

        self.cursor_label.setText(
            f"x = {x_index + 1} | y = {y_index + 1} | I = {value_text}"
        )

    def on_mouse_leave(self, event):
        self.cursor_label.setText("x = - | y = - | I = -")

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

        suggested_path = self.current_file.parent / f"{self.current_file.stem}.png"

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save image only as PNG",
            str(suggested_path),
            "PNG image (*.png);;TIFF image (*.tif);;All files (*)"
        )

        if not path:
            return

        if not path.lower().endswith(".png"):
            path += ".png"

        try:
            plt.imsave(
                path,
                self.display_img,
                cmap="jet",
                vmin=self.vmin_spin.value(),
                vmax=self.vmax_spin.value(),
                origin="upper"
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Save error",
                f"Unable to save image:\n{e}"
            )
