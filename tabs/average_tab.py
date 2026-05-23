from pathlib import Path

import h5py
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QGroupBox,
    QSpinBox,
    QTextEdit,
    QCheckBox,
    QGridLayout,
    QMessageBox,
    QSlider,
    QSizePolicy,
)

from .cave_tab import (
    ImageCanvas,
    inspect_h5_image_dataset,
    read_edf_file,
    read_h5_frame,
    write_edf_file,
)
from .ui_style import (
    BLOCK_SPACING,
    FILE_BROWSER_WIDTH,
    FRAME_BUTTON_WIDTH,
    FRAME_COUNTER_WIDTH,
    FRAME_NAV_SPACING,
    FRAME_SPIN_WIDTH,
    GROUP_BOX_MARGINS,
    PAGE_MARGINS,
)


def write_average_h5_file(filename: str, image: np.ndarray, source_files, start_frame: int, end_frame: int):
    filename = Path(filename)
    with h5py.File(filename, "w") as out:
        dataset = out.create_dataset("/entry_0000/instrument/eiger/data", data=image.astype(np.float32), compression="gzip")
        dataset.attrs["processing"] = "frame average"
        dataset.attrs["source_files"] = ", ".join(Path(path).name for path in source_files)
        dataset.attrs["start_frame"] = int(start_frame)
        dataset.attrs["end_frame"] = int(end_frame)


class AverageTab(QWidget):
    """Average tab: average EDF/H5 images over a selected frame range."""

    def __init__(self):
        super().__init__()

        self.sources = []
        self.frames = []
        self.current_frame_index = 0
        self.current_image = None
        self.average_image = None
        self.first_edf_header_text = ""
        self.first_edf_byte_order = "LowByteFirst"
        self.display_vmin = 0.0
        self.display_vmax = 1.0
        self.slider_scale = 1000
        self._syncing_frame_controls = False

        self.build_ui()
        self.set_controls_enabled(False)

    def build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(*PAGE_MARGINS)
        main_layout.setSpacing(BLOCK_SPACING)

        top_layout = QGridLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setHorizontalSpacing(BLOCK_SPACING)
        top_layout.setVerticalSpacing(BLOCK_SPACING)
        main_layout.addLayout(top_layout, stretch=1)

        original_box = QGroupBox("Original pattern")
        original_box.setMinimumWidth(0)
        original_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        original_layout = QVBoxLayout(original_box)
        original_layout.setContentsMargins(*GROUP_BOX_MARGINS)
        self.canvas_original = ImageCanvas()
        self.canvas_original.setMinimumWidth(0)
        self.canvas_original.setMinimumHeight(0)
        self.canvas_original.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.original_coordinate_label = QLabel("x = - | y = - | q = - | I = -")
        self.original_coordinate_label.setMinimumHeight(28)
        self.original_coordinate_label.setAlignment(Qt.AlignCenter)
        self.original_coordinate_label.setStyleSheet(self.coordinate_label_style())
        self.canvas_original.set_coordinate_label(self.original_coordinate_label, "")
        original_layout.addWidget(self.canvas_original, stretch=1)
        original_layout.addWidget(self.original_coordinate_label, stretch=0)

        controls_box = QGroupBox("Average tools")
        controls_box.setFixedWidth(FILE_BROWSER_WIDTH)
        controls_layout = QVBoxLayout(controls_box)
        controls_layout.setContentsMargins(*GROUP_BOX_MARGINS)
        controls_layout.setSpacing(6)

        average_box = QGroupBox("Average pattern")
        average_box.setMinimumWidth(0)
        average_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        average_layout = QVBoxLayout(average_box)
        average_layout.setContentsMargins(*GROUP_BOX_MARGINS)
        self.canvas_average = ImageCanvas()
        self.canvas_average.setMinimumWidth(0)
        self.canvas_average.setMinimumHeight(0)
        self.canvas_average.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.average_coordinate_label = QLabel("x = - | y = - | q = - | I = -")
        self.average_coordinate_label.setMinimumHeight(28)
        self.average_coordinate_label.setAlignment(Qt.AlignCenter)
        self.average_coordinate_label.setStyleSheet(self.coordinate_label_style())
        self.canvas_average.set_coordinate_label(self.average_coordinate_label, "")
        average_layout.addWidget(self.canvas_average, stretch=1)
        average_layout.addWidget(self.average_coordinate_label, stretch=0)

        top_layout.addWidget(original_box, 0, 0)
        top_layout.addWidget(controls_box, 0, 1, alignment=Qt.AlignHCenter)
        top_layout.addWidget(average_box, 0, 2)

        top_layout.setColumnMinimumWidth(0, 0)
        top_layout.setColumnMinimumWidth(1, FILE_BROWSER_WIDTH)
        top_layout.setColumnMinimumWidth(2, 0)

        top_layout.setColumnStretch(0, 1)
        top_layout.setColumnStretch(1, 0)
        top_layout.setColumnStretch(2, 1)

        self.open_button = QPushButton("Open multi EDF / multi H5")
        self.open_button.clicked.connect(self.open_files)
        controls_layout.addWidget(self.open_button)

        range_box = QGroupBox("Frame range")
        range_layout = QGridLayout(range_box)
        range_layout.setContentsMargins(*GROUP_BOX_MARGINS)
        range_layout.setSpacing(4)

        self.frame_start_spin = QSpinBox()
        self.frame_start_spin.setRange(1, 1)
        self.frame_start_spin.setValue(1)
        self.frame_start_spin.setFixedWidth(FRAME_SPIN_WIDTH)

        self.frame_start_spin_preview = QSpinBox()
        self.frame_start_spin_preview.setRange(1, 1)
        self.frame_start_spin_preview.setValue(1)
        self.frame_start_spin_preview.setFixedWidth(FRAME_SPIN_WIDTH)
        self.frame_start_spin_preview.valueChanged.connect(self.frame_start_spin.setValue)
        self.frame_start_spin.valueChanged.connect(self.frame_start_spin_preview.setValue)

        self.frame_end_spin = QSpinBox()
        self.frame_end_spin.setRange(1, 1)
        self.frame_end_spin.setValue(1)
        self.frame_end_spin.setFixedWidth(FRAME_SPIN_WIDTH)

        self.frame_end_spin_preview = QSpinBox()
        self.frame_end_spin_preview.setRange(1, 1)
        self.frame_end_spin_preview.setValue(1)
        self.frame_end_spin_preview.setFixedWidth(FRAME_SPIN_WIDTH)
        self.frame_end_spin_preview.valueChanged.connect(self.frame_end_spin.setValue)
        self.frame_end_spin.valueChanged.connect(self.frame_end_spin_preview.setValue)

        range_layout.addWidget(QLabel("Start frame:"), 0, 0)
        range_layout.addWidget(self.frame_start_spin_preview, 0, 1)
        range_layout.addWidget(QLabel("End frame:"), 1, 0)
        range_layout.addWidget(self.frame_end_spin_preview, 1, 1)
        range_layout.setColumnStretch(0, 1)
        range_layout.setColumnStretch(1, 0)
        controls_layout.addWidget(range_box)

        self.save_checkbox = QCheckBox("Save output after Run Average")
        self.save_checkbox.setChecked(False)
        controls_layout.addWidget(self.save_checkbox)

        intensity_box = QGroupBox("Display intensity")
        intensity_layout = QGridLayout(intensity_box)
        intensity_layout.setContentsMargins(*GROUP_BOX_MARGINS)
        intensity_layout.setSpacing(4)

        self.vmin_slider = QSlider(Qt.Horizontal)
        self.vmax_slider = QSlider(Qt.Horizontal)
        self.vmin_slider.setRange(0, self.slider_scale)
        self.vmax_slider.setRange(0, self.slider_scale)
        self.vmin_slider.setValue(0)
        self.vmax_slider.setValue(self.slider_scale)

        self.vmin_label = QLabel("Min: 0.000")
        self.vmax_label = QLabel("Max: 1.000")
        self.lock_intensity_checkbox = QCheckBox("Lock min/max")
        self.lock_intensity_checkbox.setChecked(False)

        intensity_layout.addWidget(self.vmin_label, 0, 0)
        intensity_layout.addWidget(self.vmin_slider, 0, 1)
        intensity_layout.addWidget(self.vmax_label, 1, 0)
        intensity_layout.addWidget(self.vmax_slider, 1, 1)
        intensity_layout.addWidget(self.lock_intensity_checkbox, 2, 0, 1, 2)
        controls_layout.addWidget(intensity_box)

        button_layout = QHBoxLayout()
        self.run_button = QPushButton("Run Average")
        self.run_button.clicked.connect(self.run_average)
        self.save_button = QPushButton("Save Average")
        self.save_button.clicked.connect(self.save_average)
        button_layout.addWidget(self.run_button)
        button_layout.addWidget(self.save_button)
        controls_layout.addLayout(button_layout)

        self.status = QTextEdit()
        self.status.setReadOnly(True)
        self.status.setPlaceholderText("")
        controls_layout.addWidget(self.status, stretch=1)

        self.vmin_slider.valueChanged.connect(self.update_display_limits_from_sliders)
        self.vmax_slider.valueChanged.connect(self.update_display_limits_from_sliders)
        self.frame_start_spin.valueChanged.connect(self.update_frame_bounds)
        self.frame_end_spin.valueChanged.connect(self.update_frame_bounds)

        frame_nav = QHBoxLayout()
        frame_nav.setContentsMargins(0, 0, 0, 0)
        frame_nav.setSpacing(FRAME_NAV_SPACING)
        self.prev_frame_button = QPushButton("<")
        self.next_frame_button = QPushButton(">")
        self.prev_frame_button.setFixedWidth(FRAME_BUTTON_WIDTH)
        self.next_frame_button.setFixedWidth(FRAME_BUTTON_WIDTH)
        self.frame_counter_label = QLabel("1 / 1")
        self.frame_counter_label.setMinimumWidth(FRAME_COUNTER_WIDTH)
        self.frame_counter_label.setAlignment(Qt.AlignCenter)
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(1, 1)
        self.frame_slider.setValue(1)

        frame_nav.addWidget(QLabel("Start:"))
        frame_nav.addWidget(self.frame_start_spin)
        frame_nav.addWidget(self.prev_frame_button)
        frame_nav.addWidget(self.frame_slider, stretch=1)
        frame_nav.addWidget(self.next_frame_button)
        frame_nav.addWidget(QLabel("End:"))
        frame_nav.addWidget(self.frame_end_spin)
        frame_nav.addWidget(self.frame_counter_label)
        main_layout.addLayout(frame_nav, stretch=0)

        self.frame_slider.valueChanged.connect(self.frame_slider_changed)
        self.prev_frame_button.clicked.connect(self.previous_frame)
        self.next_frame_button.clicked.connect(self.next_frame)

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

    def set_controls_enabled(self, enabled):
        for widget in [
            self.frame_start_spin,
            self.frame_end_spin,
            self.frame_start_spin_preview,
            self.frame_end_spin_preview,
            self.frame_slider,
            self.prev_frame_button,
            self.next_frame_button,
            self.lock_intensity_checkbox,
            self.vmin_slider,
            self.vmax_slider,
            self.run_button,
            self.save_button,
            self.save_checkbox,
        ]:
            widget.setEnabled(enabled)

        self.open_button.setEnabled(True)
        self.save_button.setEnabled(enabled and self.average_image is not None)
        self.update_frame_counter()

    def open_files(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Open EDF or H5 files",
            "",
            "Data files (*.edf *.h5 *.hdf5);;EDF (*.edf);;HDF5 (*.h5 *.hdf5);;All files (*)",
        )

        if not file_paths:
            return

        try:
            self.sources = []
            self.frames = []
            self.average_image = None
            self.clear_average_display()
            self.first_edf_header_text = ""
            self.first_edf_byte_order = "LowByteFirst"

            for file_path in file_paths:
                self.add_file(Path(file_path))

            if not self.frames:
                raise ValueError("No image frame was found in the selected files.")

            self.configure_frame_navigation(len(self.frames))
            self.current_frame_index = 0
            self.load_current_frame()
            self.set_controls_enabled(True)
            self.update_status()
        except Exception as error:
            QMessageBox.critical(self, "File reading error", str(error))

    def add_file(self, path: Path):
        suffix = path.suffix.lower()

        if suffix == ".edf":
            image, header, raw_header_text, byte_order = read_edf_file(path)
            if not self.first_edf_header_text:
                self.first_edf_header_text = raw_header_text
                self.first_edf_byte_order = byte_order
            source_index = len(self.sources)
            self.sources.append({"path": path, "type": "EDF", "header": header})
            self.frames.append({"source_index": source_index, "frame_index": 0, "shape": image.shape})
            return

        if suffix in [".h5", ".hdf5"]:
            dataset_name, dataset_shape, frame_axis, n_frames, header = inspect_h5_image_dataset(path)
            source_index = len(self.sources)
            self.sources.append({
                "path": path,
                "type": "H5",
                "dataset_name": dataset_name,
                "frame_axis": frame_axis,
                "n_frames": n_frames,
                "header": header,
                "shape": dataset_shape,
            })
            for frame_index in range(n_frames):
                self.frames.append({"source_index": source_index, "frame_index": frame_index})
            return

        raise ValueError(f"Unsupported file format: {path.name}")

    def configure_frame_navigation(self, n_frames):
        n_frames = max(1, int(n_frames))
        self._syncing_frame_controls = True
        for spin in [self.frame_start_spin, self.frame_end_spin, self.frame_start_spin_preview, self.frame_end_spin_preview]:
            spin.blockSignals(True)
        self.frame_slider.blockSignals(True)

        self.frame_slider.setRange(1, n_frames)
        self.frame_slider.setValue(1)
        self.frame_start_spin.setRange(1, n_frames)
        self.frame_start_spin.setValue(1)
        self.frame_end_spin.setRange(1, n_frames)
        self.frame_end_spin.setValue(n_frames)
        self.frame_start_spin_preview.setRange(1, n_frames)
        self.frame_start_spin_preview.setValue(1)
        self.frame_end_spin_preview.setRange(1, n_frames)
        self.frame_end_spin_preview.setValue(n_frames)

        for spin in [self.frame_start_spin, self.frame_end_spin, self.frame_start_spin_preview, self.frame_end_spin_preview]:
            spin.blockSignals(False)
        self.frame_slider.blockSignals(False)
        self._syncing_frame_controls = False

        self.update_frame_counter()
        self.invalidate_average()

    def frame_slider_changed(self, value):
        if self._syncing_frame_controls:
            return

        self.current_frame_index = max(0, min(int(value) - 1, len(self.frames) - 1))
        self.load_current_frame()

    def update_frame_bounds(self):
        if self._syncing_frame_controls:
            return

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

        current = self.current_frame_index + 1
        if current < start:
            self.set_current_frame_index(start - 1)
            return
        if current > end:
            self.set_current_frame_index(end - 1)
            return

        self.update_frame_counter()
        self.invalidate_average()

    def update_frame_counter(self):
        total = max(1, len(self.frames))
        current = min(total, self.current_frame_index + 1)
        self.frame_counter_label.setText(f"{current} / {total}")
        if hasattr(self, "prev_frame_button"):
            can_navigate = len(self.frames) > 1
            self.frame_start_spin.setEnabled(can_navigate)
            self.frame_end_spin.setEnabled(can_navigate)
            self.frame_slider.setEnabled(can_navigate)
            self.prev_frame_button.setEnabled(can_navigate and current > self.frame_slider.minimum())
            self.next_frame_button.setEnabled(can_navigate and current < self.frame_slider.maximum())

    def previous_frame(self):
        if self.current_frame_index + 1 <= self.frame_slider.minimum():
            return
        self.set_current_frame_index(self.current_frame_index - 1)

    def next_frame(self):
        if self.current_frame_index + 1 >= self.frame_slider.maximum():
            return
        self.set_current_frame_index(self.current_frame_index + 1)

    def set_current_frame_index(self, index):
        self.current_frame_index = max(0, min(int(index), len(self.frames) - 1))
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self.current_frame_index + 1)
        self.frame_slider.blockSignals(False)
        self.load_current_frame()

    def load_current_frame(self):
        if not self.frames:
            return

        image = self.read_frame(self.current_frame_index)
        self.current_image = image.astype(np.float64)

        if not self.lock_intensity_checkbox.isChecked():
            self.auto_set_display_limits(self.current_image)

        vmin, vmax = self.current_display_limits()
        self.canvas_original.setVisible(True)
        self.canvas_original.show_image(self.current_image, vmin=vmin, vmax=vmax)
        self.canvas_original.draw_idle()
        self.refresh_average_display()
        self.update_frame_counter()
        self.update_status()

    def read_frame(self, flat_index):
        frame = self.frames[flat_index]
        source = self.sources[frame["source_index"]]

        if source["type"] == "EDF":
            image, *_ = read_edf_file(source["path"])
            return image

        image, _ = read_h5_frame(source["path"], source["dataset_name"], frame["frame_index"])
        return image

    def auto_set_display_limits(self, image):
        display = image.astype(np.float64).copy()
        display[~np.isfinite(display)] = np.nan
        display[display < 0] = np.nan

        with np.errstate(invalid="ignore", divide="ignore"):
            display = np.log10(display + 1)

        finite_values = display[np.isfinite(display)]

        if finite_values.size == 0:
            self.display_vmin = 0.0
            self.display_vmax = 1.0
        else:
            self.display_vmin = float(np.nanpercentile(finite_values, 1))
            self.display_vmax = float(np.nanpercentile(finite_values, 99))
            if self.display_vmin >= self.display_vmax:
                self.display_vmin = float(np.nanmin(finite_values))
                self.display_vmax = float(np.nanmax(finite_values))
            if self.display_vmin >= self.display_vmax:
                self.display_vmax = self.display_vmin + 1.0

        self.vmin_slider.blockSignals(True)
        self.vmax_slider.blockSignals(True)
        self.vmin_slider.setValue(0)
        self.vmax_slider.setValue(self.slider_scale)
        self.vmin_slider.blockSignals(False)
        self.vmax_slider.blockSignals(False)
        self.update_display_labels()

    def current_display_limits(self):
        span = self.display_vmax - self.display_vmin
        if span <= 0:
            return self.display_vmin, self.display_vmax

        vmin = self.display_vmin + span * (self.vmin_slider.value() / self.slider_scale)
        vmax = self.display_vmin + span * (self.vmax_slider.value() / self.slider_scale)
        if vmin >= vmax:
            vmax = vmin + span / self.slider_scale
        return vmin, vmax

    def update_display_limits_from_sliders(self):
        self.update_display_labels()
        vmin, vmax = self.current_display_limits()
        if self.current_image is not None:
            self.canvas_original.show_image(self.current_image, vmin=vmin, vmax=vmax)
        self.refresh_average_display()

    def update_display_labels(self):
        vmin, vmax = self.current_display_limits()
        self.vmin_label.setText(f"Min: {vmin:.3f}")
        self.vmax_label.setText(f"Max: {vmax:.3f}")

    def refresh_average_display(self):
        if self.average_image is None:
            return

        vmin, vmax = self.current_display_limits()
        self.canvas_average.show_image(self.average_image, vmin=vmin, vmax=vmax)

    def clear_average_display(self):
        self.canvas_average.raw_image = None
        self.canvas_average.image_artist = None
        self.canvas_average.ax.clear()
        self.canvas_average.ax.set_axis_off()
        self.canvas_average.draw_idle()
        self.average_coordinate_label.setText("x = - | y = - | q = - | I = -")

    def invalidate_average(self):
        if self.average_image is None:
            return

        self.average_image = None
        self.save_button.setEnabled(False)
        self.clear_average_display()
        self.update_status()

    def run_average(self):
        if not self.frames:
            return

        start = self.frame_start_spin.value() - 1
        end = self.frame_end_spin.value() - 1

        try:
            accumulator = None
            valid_counts = None
            expected_shape = None
            used = 0

            for index in range(start, end + 1):
                image = self.read_frame(index).astype(np.float64)
                if expected_shape is None:
                    expected_shape = image.shape
                    accumulator = np.zeros(expected_shape, dtype=np.float64)
                    valid_counts = np.zeros(expected_shape, dtype=np.float64)
                elif image.shape != expected_shape:
                    raise ValueError("All selected frames must have the same image size.")

                finite = np.isfinite(image)
                accumulator[finite] += image[finite]
                valid_counts[finite] += 1
                used += 1

            with np.errstate(invalid="ignore", divide="ignore"):
                average = accumulator / valid_counts
            average[valid_counts == 0] = np.nan

            self.average_image = average
            self.save_button.setEnabled(True)
            if not self.lock_intensity_checkbox.isChecked():
                self.auto_set_display_limits(self.average_image)
            self.refresh_average_display()
            self.update_status()
            self.status.append(f"\nAverage computed from {used} frame(s): {start + 1} to {end + 1}.")

            if self.save_checkbox.isChecked():
                self.save_average()
        except Exception as error:
            QMessageBox.critical(self, "Average error", str(error))

    def save_average(self):
        if self.average_image is None or not self.sources:
            return

        first_path = self.sources[0]["path"]
        start = self.frame_start_spin.value()
        end = self.frame_end_spin.value()

        if self.first_edf_header_text:
            suggested_path = first_path.parent / f"{first_path.stem}_averaged.edf"
            output_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save average EDF",
                str(suggested_path),
                "EDF (*.edf);;HDF5 (*.h5);;All files (*)",
            )
        else:
            suggested_path = first_path.parent / f"{first_path.stem}_averaged.h5"
            output_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save average H5",
                str(suggested_path),
                "HDF5 (*.h5);;EDF (*.edf);;All files (*)",
            )

        if not output_path:
            return

        output_path = self._ensure_averaged_suffix(output_path)

        try:
            lower_path = output_path.lower()
            if lower_path.endswith(".edf"):
                if not self.first_edf_header_text:
                    raise ValueError("Saving as EDF requires at least one EDF source file for the header.")
                write_edf_file(output_path, self.average_image, self.first_edf_header_text, self.first_edf_byte_order)
                self.status.append(f"\nSaved average EDF:\n{output_path}")
            else:
                if not lower_path.endswith((".h5", ".hdf5")):
                    output_path += ".h5"
                write_average_h5_file(output_path, self.average_image, [source["path"] for source in self.sources], start, end)
                self.status.append(f"\nSaved average H5:\n{output_path}")
        except Exception as error:
            QMessageBox.critical(self, "Save error", str(error))

    def _ensure_averaged_suffix(self, output_path):
        path_obj = Path(output_path)
        suffix = path_obj.suffix.lower()
        name = path_obj.stem

        if suffix in {".edf", ".h5", ".hdf5"}:
            if not name.endswith("_averaged"):
                name += "_averaged"
            return str(path_obj.with_name(name + path_obj.suffix))

        if not name.endswith("_averaged"):
            name += "_averaged"
        return str(path_obj.with_name(name))

    def update_status(self):
        if not self.sources:
            return

        current = self.frames[self.current_frame_index]
        source = self.sources[current["source_index"]]
        lines = [
            f"Files: {len(self.sources)}",
            f"Frames: {len(self.frames)}",
            f"Current file: {source['path'].name}",
            f"Current frame: {self.current_frame_index + 1} / {len(self.frames)}",
        ]

        if source["type"] == "H5":
            lines.append(f"H5 dataset: {source['dataset_name']}")
            lines.append(f"H5 frame: {current['frame_index'] + 1} / {source['n_frames']}")

        if self.current_image is not None:
            lines.append(f"Image size: {self.current_image.shape[1]} x {self.current_image.shape[0]}")

        lines.append(f"Average range: {self.frame_start_spin.value()} to {self.frame_end_spin.value()}")
        if self.average_image is None:
            lines.append("Average: not computed yet")
        else:
            lines.append("Average: computed")

        self.status.setPlainText("\n".join(lines))
