from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QFileDialog,
    QLineEdit,
    QCheckBox,
    QDoubleSpinBox,
    QSpinBox,
    QTextEdit,
    QSlider,
    QScrollArea,
    QSizePolicy,
)

import os
import numpy as np

from .cave_tab import ImageCanvas

try:
    import fabio
    from fabio.edfimage import EdfImage
except Exception:
    fabio = None
    EdfImage = None

try:
    import h5py
except Exception:
    h5py = None

try:
    import hdf5plugin  # noqa: F401
except Exception:
    pass

try:
    from tabs.ui_style import (
        PAGE_MARGINS,
        PANEL_MARGINS,
        BLOCK_SPACING,
        FILE_BROWSER_WIDTH,
        FRAME_BUTTON_WIDTH,
        FRAME_COUNTER_WIDTH,
        FRAME_NAV_SPACING,
        FRAME_SPIN_WIDTH,
        GROUP_BOX_MARGINS,
        TOOL_GROUP_BOX_STYLE,
    )
except Exception:
    PAGE_MARGINS = (4, 4, 4, 4)
    PANEL_MARGINS = (0, 0, 0, 0)
    BLOCK_SPACING = 8
    FILE_BROWSER_WIDTH = 320
    FRAME_BUTTON_WIDTH = 44
    FRAME_COUNTER_WIDTH = 72
    FRAME_NAV_SPACING = 8
    FRAME_SPIN_WIDTH = 80
    GROUP_BOX_MARGINS = (8, 20, 8, 8)
    TOOL_GROUP_BOX_STYLE = ""


class LazyImageStack:
    def __init__(self, file_path, kind, data=None, dataset_path=None, frame_count=1, shape=None):
        self.file_path = file_path
        self.kind = kind
        self.data = data
        self.dataset_path = dataset_path
        self.frame_count = int(frame_count)
        self.shape = shape

    def get_frame(self, frame_index):
        frame_index = max(0, min(int(frame_index), self.frame_count - 1))

        if self.kind in ("edf", "text"):
            if self.data.ndim == 2:
                return self.data.astype(np.float64)
            return self.data[frame_index].astype(np.float64)

        if self.kind == "hdf5":
            if h5py is None:
                raise ImportError("h5py is required to read HDF5 files.")
            with h5py.File(self.file_path, "r") as handle:
                dataset = handle[self.dataset_path]
                if dataset.ndim == 2:
                    return np.asarray(dataset[:, :], dtype=np.float64)
                return np.asarray(dataset[frame_index, :, :], dtype=np.float64)

        return None


class BackgroundTab(QWidget):
    folder_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.sample_file_path = ""
        self.background_file_path = ""
        self.output_folder_path = ""
        self.current_folder = ""
        self.sample_stack = None
        self.background_stack = None
        self.result_data = None
        self.contrast_vmin = None
        self.contrast_vmax = None
        self.contrast_auto_initialized = False
        self.build_ui()

    def build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(*PAGE_MARGINS)
        main_layout.setSpacing(BLOCK_SPACING)

        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(*PANEL_MARGINS)
        content_layout.setSpacing(BLOCK_SPACING)

        original_box = QGroupBox("Original pattern")
        original_box.setStyleSheet(TOOL_GROUP_BOX_STYLE)
        original_box.setMinimumWidth(0)
        original_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        original_layout = QVBoxLayout(original_box)
        original_layout.setContentsMargins(*GROUP_BOX_MARGINS)
        original_layout.setSpacing(6)

        self.original_canvas = ImageCanvas()
        self.original_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.original_canvas.setMinimumWidth(0)
        self.original_canvas.setMinimumHeight(0)
        self.original_ax = self.original_canvas.ax
        self.original_coordinate_label = QLabel("x = - | y = - | I = -")
        self.original_coordinate_label.setMinimumHeight(28)
        self.original_coordinate_label.setAlignment(Qt.AlignCenter)
        self.original_coordinate_label.setStyleSheet(self.coordinate_label_style())
        self.original_canvas.set_coordinate_label(self.original_coordinate_label, "")
        original_layout.addWidget(self.original_canvas, 1)
        original_layout.addWidget(self.original_coordinate_label, 0)

        parameters_box = QGroupBox("Background tools")
        parameters_box.setStyleSheet(TOOL_GROUP_BOX_STYLE)
        parameters_box.setFixedWidth(FILE_BROWSER_WIDTH)
        parameters_layout = QVBoxLayout(parameters_box)
        parameters_layout.setContentsMargins(*GROUP_BOX_MARGINS)
        parameters_layout.setSpacing(6)

        self.sample_file_edit = QLineEdit()
        self.sample_file_edit.setPlaceholderText("Sample file")
        self.sample_file_edit.setReadOnly(True)
        self.sample_file_button = QPushButton("Open sample")
        self.sample_file_button.clicked.connect(self.select_sample_file)
        parameters_layout.addWidget(QLabel("Sample"))
        parameters_layout.addWidget(self.sample_file_edit)
        parameters_layout.addWidget(self.sample_file_button)

        self.background_file_edit = QLineEdit()
        self.background_file_edit.setPlaceholderText("Background file")
        self.background_file_edit.setReadOnly(True)
        self.background_file_button = QPushButton("Open background")
        self.background_file_button.clicked.connect(self.select_background_file)
        parameters_layout.addWidget(QLabel("Background"))
        parameters_layout.addWidget(self.background_file_edit)
        parameters_layout.addWidget(self.background_file_button)

        self.output_folder_edit = QLineEdit()
        self.output_folder_edit.setPlaceholderText("Output folder")
        self.output_folder_edit.setReadOnly(True)
        self.output_folder_button = QPushButton("Output folder")
        self.output_folder_button.clicked.connect(self.select_output_folder)
        parameters_layout.addWidget(QLabel("Output"))
        parameters_layout.addWidget(self.output_folder_edit)
        parameters_layout.addWidget(self.output_folder_button)

        self.background_scale_spin = QDoubleSpinBox()
        self.background_scale_spin.setDecimals(4)
        self.background_scale_spin.setRange(-999999.0, 999999.0)
        self.background_scale_spin.setSingleStep(0.01)
        self.background_scale_spin.setValue(1.0)
        self.background_scale_spin.valueChanged.connect(self.update_result_preview)
        parameters_layout.addWidget(QLabel("Background factor"))
        parameters_layout.addWidget(self.background_scale_spin)

        self.offset_spin = QDoubleSpinBox()
        self.offset_spin.setDecimals(4)
        self.offset_spin.setRange(-999999.0, 999999.0)
        self.offset_spin.setSingleStep(0.01)
        self.offset_spin.setValue(0.0)
        self.offset_spin.valueChanged.connect(self.update_result_preview)
        parameters_layout.addWidget(QLabel("Offset"))
        parameters_layout.addWidget(self.offset_spin)

        self.frame_spin = QSpinBox(self)
        self.frame_spin.setRange(1, 1)
        self.frame_spin.setValue(1)
        self.frame_spin.hide()
        self.frame_spin.valueChanged.connect(self.sync_frame_slider_from_spin)

        self.keep_negative_checkbox = QCheckBox("Keep negative values")
        self.keep_negative_checkbox.setChecked(True)
        self.keep_negative_checkbox.stateChanged.connect(self.update_result_preview)
        parameters_layout.addWidget(self.keep_negative_checkbox)

        contrast_box = QGroupBox("Contrast")
        contrast_box.setStyleSheet(TOOL_GROUP_BOX_STYLE)
        contrast_layout = QVBoxLayout(contrast_box)
        contrast_layout.setContentsMargins(*GROUP_BOX_MARGINS)
        contrast_layout.setSpacing(6)

        min_row = QHBoxLayout()
        min_row.addWidget(QLabel("Min"))
        self.intensity_min_spin = QDoubleSpinBox()
        self.intensity_min_spin.setDecimals(2)
        self.intensity_min_spin.setRange(-1e12, 1e12)
        self.intensity_min_spin.setSingleStep(100.0)
        self.intensity_min_spin.valueChanged.connect(self.update_contrast_from_spins)
        min_row.addWidget(self.intensity_min_spin)
        contrast_layout.addLayout(min_row)

        self.intensity_min_slider = QSlider(Qt.Horizontal)
        self.intensity_min_slider.setRange(0, 1000)
        self.intensity_min_slider.setValue(0)
        self.intensity_min_slider.valueChanged.connect(self.update_contrast_from_sliders)
        contrast_layout.addWidget(self.intensity_min_slider)

        max_row = QHBoxLayout()
        max_row.addWidget(QLabel("Max"))
        self.intensity_max_spin = QDoubleSpinBox()
        self.intensity_max_spin.setDecimals(2)
        self.intensity_max_spin.setRange(-1e12, 1e12)
        self.intensity_max_spin.setSingleStep(100.0)
        self.intensity_max_spin.valueChanged.connect(self.update_contrast_from_spins)
        max_row.addWidget(self.intensity_max_spin)
        contrast_layout.addLayout(max_row)

        self.intensity_max_slider = QSlider(Qt.Horizontal)
        self.intensity_max_slider.setRange(0, 1000)
        self.intensity_max_slider.setValue(1000)
        self.intensity_max_slider.valueChanged.connect(self.update_contrast_from_sliders)
        contrast_layout.addWidget(self.intensity_max_slider)

        self.auto_contrast_button = QPushButton("Auto contrast")
        self.auto_contrast_button.clicked.connect(self.auto_contrast)
        contrast_layout.addWidget(self.auto_contrast_button)
        parameters_layout.addWidget(contrast_box)

        self.save_preview_checkbox = QCheckBox("Save preview image")
        self.save_preview_checkbox.setChecked(False)
        parameters_layout.addWidget(self.save_preview_checkbox)

        self.save_current_button = QPushButton("Save current frame")
        self.save_current_button.clicked.connect(self.save_current_frame)
        parameters_layout.addWidget(self.save_current_button)

        self.run_button = QPushButton("Save all frames")
        self.run_button.clicked.connect(self.run_background_subtraction)
        parameters_layout.addWidget(self.run_button)

        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)
        parameters_layout.addWidget(self.status_label)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(80)
        self.log_text.setPlaceholderText("Background processing messages will appear here.")
        parameters_layout.addWidget(self.log_text)
        parameters_layout.addStretch(1)

        result_box = QGroupBox("Background-subtracted pattern")
        result_box.setStyleSheet(TOOL_GROUP_BOX_STYLE)
        result_box.setMinimumWidth(0)
        result_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        result_layout = QVBoxLayout(result_box)
        result_layout.setContentsMargins(*GROUP_BOX_MARGINS)
        result_layout.setSpacing(6)

        self.result_canvas = ImageCanvas()
        self.result_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.result_canvas.setMinimumWidth(0)
        self.result_canvas.setMinimumHeight(0)
        self.result_ax = self.result_canvas.ax
        self.result_coordinate_label = QLabel("x = - | y = - | I = -")
        self.result_coordinate_label.setMinimumHeight(28)
        self.result_coordinate_label.setAlignment(Qt.AlignCenter)
        self.result_coordinate_label.setStyleSheet(self.coordinate_label_style())
        self.result_canvas.set_coordinate_label(self.result_coordinate_label, "")
        result_layout.addWidget(self.result_canvas, 1)
        result_layout.addWidget(self.result_coordinate_label, 0)

        content_layout.addWidget(original_box, 2)

        parameters_scroll = QScrollArea()
        parameters_scroll.setWidgetResizable(True)
        parameters_scroll.setFrameShape(QScrollArea.NoFrame)
        parameters_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        parameters_scroll.setWidget(parameters_box)
        parameters_scroll.setFixedWidth(FILE_BROWSER_WIDTH)
        content_layout.addWidget(parameters_scroll, 0)

        content_layout.addWidget(result_box, 2)
        main_layout.addLayout(content_layout, 1)

        frame_slider_layout = QHBoxLayout()
        frame_slider_layout.setContentsMargins(0, 0, 0, 0)
        frame_slider_layout.setSpacing(FRAME_NAV_SPACING)

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
        self.frame_slider.valueChanged.connect(self.sync_frame_spin_from_slider)

        self.frame_counter_label = QLabel("1 / 1")
        self.frame_counter_label.setMinimumWidth(FRAME_COUNTER_WIDTH)
        self.frame_counter_label.setAlignment(Qt.AlignCenter)

        frame_slider_layout.addWidget(QLabel("Start:"))
        frame_slider_layout.addWidget(self.frame_start_spin)
        frame_slider_layout.addWidget(self.prev_frame_button)
        frame_slider_layout.addWidget(self.frame_slider, 1)
        frame_slider_layout.addWidget(self.next_frame_button)
        frame_slider_layout.addWidget(QLabel("End:"))
        frame_slider_layout.addWidget(self.frame_end_spin)
        frame_slider_layout.addWidget(self.frame_counter_label)
        main_layout.addLayout(frame_slider_layout)

        self.frame_start_spin.valueChanged.connect(self.update_frame_bounds)
        self.frame_end_spin.valueChanged.connect(self.update_frame_bounds)
        self.prev_frame_button.clicked.connect(self.previous_frame)
        self.next_frame_button.clicked.connect(self.next_frame)
        self.update_frame_bounds()

    def coordinate_label_style(self):
        return """
            QLabel {
                background-color: #f4f4f4;
                border-radius: 8px;
                padding: 6px;
                font-family: Menlo, Monaco, monospace;
                font-size: 11px;
            }
        """

    def open_image_stack(self, file_path):
        lower_path = file_path.lower()

        if lower_path.endswith(".edf"):
            if fabio is None:
                raise ImportError("fabio is required to read EDF files.")
            data = np.asarray(fabio.open(file_path).data)
            if data.ndim == 2:
                return LazyImageStack(file_path, "edf", data=data, frame_count=1, shape=data.shape)
            if data.ndim == 3:
                return LazyImageStack(file_path, "edf", data=data, frame_count=data.shape[0], shape=data.shape[-2:])
            raise ValueError("Unsupported EDF data dimensions.")

        if lower_path.endswith((".h5", ".hdf5")):
            if h5py is None:
                raise ImportError("h5py is required to read HDF5 files.")
            with h5py.File(file_path, "r") as handle:
                dataset = self.find_first_image_dataset(handle)
                if dataset is None:
                    raise ValueError("No 2D or 3D image dataset found in HDF5 file.")
                dataset_path = dataset.name
                if dataset.ndim == 2:
                    frame_count = 1
                    shape = dataset.shape
                elif dataset.ndim == 3:
                    frame_count = dataset.shape[0]
                    shape = dataset.shape[-2:]
                else:
                    raise ValueError("Unsupported HDF5 data dimensions.")
            return LazyImageStack(file_path, "hdf5", dataset_path=dataset_path, frame_count=frame_count, shape=shape)

        data = np.loadtxt(file_path)
        if data.ndim == 2:
            return LazyImageStack(file_path, "text", data=data, frame_count=1, shape=data.shape)
        raise ValueError("Only 2D text data can be displayed as an image.")

    def find_first_image_dataset(self, h5_group):
        best_dataset = None

        def visitor(name, obj):
            nonlocal best_dataset
            if best_dataset is not None:
                return
            if isinstance(obj, h5py.Dataset) and obj.ndim in (2, 3):
                shape = obj.shape
                if len(shape) == 2 and min(shape) > 16:
                    best_dataset = obj
                elif len(shape) == 3 and min(shape[-2:]) > 16:
                    best_dataset = obj

        h5_group.visititems(visitor)
        return best_dataset

    def update_frame_controls(self):
        frame_count = 1 if self.sample_stack is None else max(1, self.sample_stack.frame_count)

        self.frame_slider.blockSignals(True)
        self.frame_spin.blockSignals(True)
        self.frame_start_spin.blockSignals(True)
        self.frame_end_spin.blockSignals(True)

        self.frame_spin.setRange(1, frame_count)
        self.frame_start_spin.setRange(1, frame_count)
        self.frame_start_spin.setValue(1)
        self.frame_end_spin.setRange(1, frame_count)
        self.frame_end_spin.setValue(frame_count)
        self.frame_slider.setRange(1, frame_count)
        if self.frame_spin.value() > frame_count:
            self.frame_spin.setValue(frame_count)
        self.frame_slider.setValue(self.frame_spin.value())

        self.frame_end_spin.blockSignals(False)
        self.frame_start_spin.blockSignals(False)
        self.frame_slider.blockSignals(False)
        self.frame_spin.blockSignals(False)
        self.update_frame_navigation_state()

    def current_frame_index(self):
        return max(0, self.frame_spin.value() - 1)

    def update_frame_bounds(self):
        start = self.frame_start_spin.value()
        end = self.frame_end_spin.value()

        if start > end:
            sender = self.sender()
            if sender is self.frame_start_spin:
                self.frame_end_spin.setValue(start)
                end = start
            else:
                self.frame_start_spin.setValue(end)
                start = end

        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(start, end)
        self.frame_slider.blockSignals(False)

        current = self.frame_spin.value()
        if current < start:
            self.frame_spin.setValue(start)
        elif current > end:
            self.frame_spin.setValue(end)
        self.update_frame_navigation_state()

    def update_frame_navigation_state(self):
        frame_count = 1 if self.sample_stack is None else max(1, self.sample_stack.frame_count)
        current = self.frame_spin.value()
        can_navigate = frame_count > 1

        self.frame_counter_label.setText(f"{current} / {frame_count}")
        self.frame_start_spin.setEnabled(can_navigate)
        self.frame_end_spin.setEnabled(can_navigate)
        self.frame_slider.setEnabled(can_navigate)
        self.prev_frame_button.setEnabled(can_navigate and current > self.frame_slider.minimum())
        self.next_frame_button.setEnabled(can_navigate and current < self.frame_slider.maximum())

    def previous_frame(self):
        self.frame_spin.setValue(max(self.frame_slider.minimum(), self.frame_spin.value() - 1))

    def next_frame(self):
        self.frame_spin.setValue(min(self.frame_slider.maximum(), self.frame_spin.value() + 1))

    def display_image(self, ax, canvas, image, title):
        if image is None:
            ax.clear()
            ax.set_axis_off()
            if hasattr(canvas, "raw_image"):
                canvas.raw_image = None
                canvas.image_artist = None
            canvas.draw_idle()
            return

        image = np.asarray(image, dtype=np.float64)
        image = np.where(np.isfinite(image), image, np.nan)

        if hasattr(canvas, "show_image"):
            canvas.show_image(
                image,
                title="",
                vmin=self.contrast_vmin,
                vmax=self.contrast_vmax,
            )
            return

        ax.clear()
        ax.set_axis_off()
        ax.imshow(
            image,
            cmap="jet",
            origin="upper",
            vmin=self.contrast_vmin,
            vmax=self.contrast_vmax,
        )
        canvas.draw_idle()

    def block_contrast_signals(self, blocked):
        self.intensity_min_spin.blockSignals(blocked)
        self.intensity_max_spin.blockSignals(blocked)
        self.intensity_min_slider.blockSignals(blocked)
        self.intensity_max_slider.blockSignals(blocked)

    def set_contrast_values(self, vmin, vmax):
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return
        if vmax <= vmin:
            vmax = vmin + 1.0

        self.contrast_vmin = float(vmin)
        self.contrast_vmax = float(vmax)
        self.contrast_auto_initialized = True

        self.block_contrast_signals(True)
        self.intensity_min_spin.setValue(self.contrast_vmin)
        self.intensity_max_spin.setValue(self.contrast_vmax)
        self.intensity_min_slider.setValue(0)
        self.intensity_max_slider.setValue(1000)
        self.block_contrast_signals(False)

        self.refresh_displayed_images()

    def update_contrast_from_spins(self):
        vmin = self.intensity_min_spin.value()
        vmax = self.intensity_max_spin.value()
        if vmax <= vmin:
            vmax = vmin + 1.0
            self.intensity_max_spin.blockSignals(True)
            self.intensity_max_spin.setValue(vmax)
            self.intensity_max_spin.blockSignals(False)

        self.contrast_vmin = vmin
        self.contrast_vmax = vmax
        self.contrast_auto_initialized = True
        self.refresh_displayed_images()

    def update_contrast_from_sliders(self):
        if self.contrast_vmin is None or self.contrast_vmax is None:
            self.auto_contrast()
            return

        slider_min = self.intensity_min_slider.value()
        slider_max = self.intensity_max_slider.value()
        if slider_max <= slider_min:
            slider_max = slider_min + 1
            self.intensity_max_slider.blockSignals(True)
            self.intensity_max_slider.setValue(slider_max)
            self.intensity_max_slider.blockSignals(False)

        current_span = max(self.contrast_vmax - self.contrast_vmin, 1.0)
        center = (self.contrast_vmin + self.contrast_vmax) / 2.0
        global_span = max(abs(center), current_span, 1.0) * 4.0
        range_min = center - global_span
        range_max = center + global_span

        vmin = range_min + (range_max - range_min) * (slider_min / 1000.0)
        vmax = range_min + (range_max - range_min) * (slider_max / 1000.0)

        self.block_contrast_signals(True)
        self.intensity_min_spin.setValue(vmin)
        self.intensity_max_spin.setValue(vmax)
        self.block_contrast_signals(False)

        self.contrast_vmin = vmin
        self.contrast_vmax = vmax
        self.contrast_auto_initialized = True
        self.refresh_displayed_images()

    def auto_contrast(self):
        frame = self.result_data
        if frame is None and self.sample_stack is not None:
            frame = self.sample_stack.get_frame(self.current_frame_index())
        if frame is None:
            return

        finite_values = np.asarray(frame, dtype=np.float64)
        finite_values = finite_values[np.isfinite(finite_values)]
        if finite_values.size == 0:
            return

        vmin, vmax = np.nanpercentile(finite_values, [1, 99])
        self.set_contrast_values(vmin, vmax)

    def refresh_displayed_images(self):
        sample_frame = None
        if self.sample_stack is not None:
            sample_frame = self.sample_stack.get_frame(self.current_frame_index())
        self.display_image(self.original_ax, self.original_canvas, sample_frame, "Original file")

        if self.result_data is not None:
            self.display_image(self.result_ax, self.result_canvas, self.result_data, "Result")
        else:
            self.display_image(self.result_ax, self.result_canvas, None, "Result")

    def update_sample_preview(self):
        frame = None
        if self.sample_stack is not None:
            frame = self.sample_stack.get_frame(self.current_frame_index())
            if not self.contrast_auto_initialized:
                finite_values = np.asarray(frame, dtype=np.float64)
                finite_values = finite_values[np.isfinite(finite_values)]
                if finite_values.size:
                    vmin, vmax = np.nanpercentile(finite_values, [1, 99])
                    self.set_contrast_values(vmin, vmax)
        self.display_image(self.original_ax, self.original_canvas, frame, "Original file")
        self.update_result_preview()

    def update_result_preview(self):
        sample_frame = None
        background_frame = None
        frame_index = self.current_frame_index()

        if self.sample_stack is not None:
            sample_frame = self.sample_stack.get_frame(frame_index)
        if self.background_stack is not None:
            background_frame = self.background_stack.get_frame(frame_index)

        if sample_frame is None:
            self.result_data = None
            self.display_image(self.result_ax, self.result_canvas, None, "Result")
            return

        if background_frame is None:
            self.result_data = None
            self.display_image(self.result_ax, self.result_canvas, None, "Result")
            return

        if sample_frame.shape != background_frame.shape:
            self.result_data = None
            self.status_label.setText("Sample and background frames do not have the same shape.")
            self.display_image(self.result_ax, self.result_canvas, None, "Shape mismatch")
            return

        result = self.compute_result_frame(frame_index)
        self.result_data = result
        self.display_image(self.result_ax, self.result_canvas, result, "Result")
        self.status_label.setText("Preview updated.")

    def sync_frame_slider_from_spin(self, value):
        if self.frame_slider.value() != value:
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(value)
            self.frame_slider.blockSignals(False)
        self.update_sample_preview()
        self.update_frame_navigation_state()

    def sync_frame_spin_from_slider(self, value):
        if self.frame_spin.value() != value:
            self.frame_spin.blockSignals(True)
            self.frame_spin.setValue(value)
            self.frame_spin.blockSignals(False)
        self.update_sample_preview()
        self.update_frame_navigation_state()

    def set_working_folder(self, folder_path):
        self.current_folder = folder_path or ""
        if folder_path and hasattr(self, "output_folder_edit") and not self.output_folder_path:
            self.output_folder_path = folder_path
            self.output_folder_edit.setText(folder_path)

    def set_folder_from_external_tab(self, folder_path):
        self.set_working_folder(folder_path)

    def select_sample_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select sample file",
            self.current_folder,
            "Data files (*.edf *.h5 *.hdf5 *.dat *.txt);;All files (*)",
        )
        if file_path:
            self.sample_file_path = file_path
            self.sample_file_edit.setText(file_path)
            self.set_working_folder(file_path.rsplit("/", 1)[0])
            self.folder_changed.emit(self.current_folder)
            try:
                self.sample_stack = self.open_image_stack(file_path)
                self.contrast_auto_initialized = False
                self.contrast_vmin = None
                self.contrast_vmax = None
                self.update_frame_controls()
                self.update_sample_preview()
                self.status_label.setText(f"Sample loaded: {self.sample_stack.frame_count} frame(s).")
            except Exception as exc:
                self.sample_stack = None
                self.update_frame_controls()
                self.display_image(self.original_ax, self.original_canvas, None, "Original file")
                self.status_label.setText(f"Sample loading error: {exc}")

    def select_background_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select background file",
            self.current_folder,
            "Data files (*.edf *.h5 *.hdf5 *.dat *.txt);;All files (*)",
        )
        if file_path:
            self.background_file_path = file_path
            self.background_file_edit.setText(file_path)
            self.set_working_folder(file_path.rsplit("/", 1)[0])
            self.folder_changed.emit(self.current_folder)
            try:
                self.background_stack = self.open_image_stack(file_path)
                self.update_result_preview()
                self.status_label.setText(f"Background loaded: {self.background_stack.frame_count} frame(s).")
            except Exception as exc:
                self.background_stack = None
                self.update_result_preview()
                self.status_label.setText(f"Background loading error: {exc}")

    def select_output_folder(self):
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select output folder",
            self.current_folder,
        )
        if folder_path:
            self.output_folder_path = folder_path
            self.output_folder_edit.setText(folder_path)
            self.set_working_folder(folder_path)
            self.folder_changed.emit(self.current_folder)

    def get_output_base_name(self):
        if self.sample_file_path:
            base_name = os.path.splitext(os.path.basename(self.sample_file_path))[0]
        else:
            base_name = "background_subtracted"
        return f"{base_name}_background_subtracted"

    def compute_result_frame(self, frame_index):
        if self.sample_stack is None:
            raise ValueError("No sample file loaded.")
        if self.background_stack is None:
            raise ValueError("No background file loaded.")

        sample_frame = self.sample_stack.get_frame(frame_index)
        background_frame = self.background_stack.get_frame(frame_index)

        if sample_frame.shape != background_frame.shape:
            raise ValueError("Sample and background frames do not have the same shape.")

        result = sample_frame - self.background_scale_spin.value() * background_frame + self.offset_spin.value()
        if not self.keep_negative_checkbox.isChecked():
            result = np.maximum(result, 0)
        return result

    def save_current_frame(self):
        if not self.output_folder_path:
            self.status_label.setText("Select an output folder first.")
            return

        if EdfImage is None:
            self.status_label.setText("fabio EDF support is not available.")
            return

        try:
            frame_index = self.current_frame_index()
            result = self.compute_result_frame(frame_index)
            base_name = self.get_output_base_name()

            edf_path = os.path.join(
                self.output_folder_path,
                f"{base_name}_frame_{frame_index:04d}.edf",
            )

            edf_image = EdfImage(data=np.asarray(result, dtype=np.float32))
            edf_image.write(edf_path)

            if self.save_preview_checkbox.isChecked():
                png_path = os.path.join(
                    self.output_folder_path,
                    f"{base_name}_frame_{frame_index:04d}.png",
                )
                self.result_canvas.figure.savefig(
                    png_path,
                    dpi=300,
                    bbox_inches="tight",
                )
                self.log_text.append(f"Saved preview: {png_path}")

            self.log_text.append(f"Saved EDF: {edf_path}")
            self.status_label.setText("Current EDF frame saved.")

        except Exception as exc:
            self.status_label.setText(f"Save error: {exc}")

    def save_all_frames_as_npy(self):
        if not self.output_folder_path:
            self.status_label.setText("Select an output folder first.")
            return

        if self.sample_stack is None or self.background_stack is None:
            self.status_label.setText("Load both sample and background first.")
            return

        if EdfImage is None:
            self.status_label.setText("fabio EDF support is not available.")
            return

        try:
            frame_count = min(
                self.sample_stack.frame_count,
                self.background_stack.frame_count,
            )
            start_frame = max(0, self.frame_start_spin.value() - 1)
            end_frame = min(frame_count, self.frame_end_spin.value())

            base_name = self.get_output_base_name()

            for frame_index in range(start_frame, end_frame):
                result = self.compute_result_frame(frame_index)

                edf_path = os.path.join(
                    self.output_folder_path,
                    f"{base_name}_frame_{frame_index:04d}.edf",
                )

                edf_image = EdfImage(data=np.asarray(result, dtype=np.float32))
                edf_image.write(edf_path)

                self.log_text.append(f"Saved EDF: {edf_path}")

            self.status_label.setText(f"Saved {max(0, end_frame - start_frame)} EDF frame(s).")

        except Exception as exc:
            self.status_label.setText(f"Save all error: {exc}")

    def run_background_subtraction(self):
        if not self.sample_file_path:
            self.status_label.setText("Select a sample file first.")
            return

        if not self.background_file_path:
            self.status_label.setText("Select a background file first.")
            return

        if not self.output_folder_path:
            self.status_label.setText("Select an output folder first.")
            return

        self.update_result_preview()
        self.save_all_frames_as_npy()
