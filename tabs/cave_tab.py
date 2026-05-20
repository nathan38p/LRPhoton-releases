import re
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
    QDoubleSpinBox,
    QTextEdit,
    QCheckBox,
    QGridLayout,
    QMessageBox,
    QSlider,
    QComboBox,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ============================================================
# ========================= FILE TOOLS ========================
# ============================================================

def parse_edf_header(header_text: str) -> dict:
    i1 = header_text.find("{")
    i2 = header_text.rfind("}")
    if i1 < 0 or i2 < 0:
        raise ValueError("Invalid EDF header: braces not found.")

    content = header_text[i1 + 1:i2]
    header = {}

    for part in content.split(";"):
        part = part.strip()
        if "=" in part:
            key, value = part.split("=", 1)
            header[key.strip()] = value.strip()

    return header


def edf_dtype_to_numpy(data_type: str):
    data_type = data_type.strip().lower()

    if data_type in ["floatvalue", "float"]:
        return np.float32
    if data_type in ["doublevalue", "double"]:
        return np.float64
    if data_type == "unsignedshort":
        return np.uint16
    if data_type == "signedshort":
        return np.int16
    if data_type in ["unsignedinteger", "uint32"]:
        return np.uint32
    if data_type in ["signedinteger", "int32"]:
        return np.int32
    if data_type in ["unsignedbyte", "uint8"]:
        return np.uint8
    if data_type in ["signedbyte", "int8"]:
        return np.int8

    raise ValueError(f"Unsupported EDF data type: {data_type}")


def read_edf_file(filename: str):
    filename = Path(filename)

    with open(filename, "rb") as file:
        first = file.read(8192).decode("latin-1", errors="ignore")

    match = re.search(r"EDF_HeaderSize\s*=\s*(\d+)", first)
    if not match:
        raise ValueError("EDF_HeaderSize not found in EDF header.")

    header_size = int(match.group(1))

    with open(filename, "rb") as file:
        raw_header_bytes = file.read(header_size)
        raw_header_text = raw_header_bytes.decode("latin-1", errors="ignore")

    header = parse_edf_header(raw_header_text)

    data_type = header.get("DataType", "FloatValue")
    byte_order = header.get("ByteOrder", "LowByteFirst")
    dim_1 = int(float(header["Dim_1"]))
    dim_2 = int(float(header["Dim_2"]))

    dtype = np.dtype(edf_dtype_to_numpy(data_type))
    dtype = dtype.newbyteorder(">" if byte_order.lower() == "highbytefirst" else "<")

    with open(filename, "rb") as file:
        file.seek(header_size)
        data = np.fromfile(file, dtype=dtype, count=dim_1 * dim_2)

    if data.size != dim_1 * dim_2:
        raise ValueError(f"Incorrect EDF data size: expected {dim_1 * dim_2}, read {data.size}.")

    image = data.reshape((dim_2, dim_1)).astype(np.float64)
    return image, header, raw_header_text, byte_order


def update_edf_header_value(header_text: str, key: str, new_value: str) -> str:
    expression = rf"{re.escape(key)}\s*=\s*[^;]*;"
    replacement = f"{key} = {new_value} ;"

    if re.search(expression, header_text):
        return re.sub(expression, replacement, header_text, count=1)

    closing = header_text.rfind("}")
    if closing < 0:
        raise ValueError("Unable to update EDF header: closing brace not found.")

    return header_text[:closing] + f"\n{key} = {new_value} ;" + header_text[closing:]


def write_edf_file(filename: str, image: np.ndarray, raw_header_text: str, byte_order: str):
    filename = Path(filename)
    ny, nx = image.shape

    header_text = raw_header_text
    header_text = update_edf_header_value(header_text, "Dim_1", str(nx))
    header_text = update_edf_header_value(header_text, "Dim_2", str(ny))
    header_text = update_edf_header_value(header_text, "DataType", "FloatValue")
    header_text = update_edf_header_value(header_text, "Size", str(nx * ny * 4))
    header_text = update_edf_header_value(header_text, "EDF_BinarySize", str(nx * ny * 4))

    match = re.search(r"EDF_HeaderSize\s*=\s*(\d+)", header_text)
    header_size = int(match.group(1)) if match else 1024

    if not match:
        header_text = update_edf_header_value(header_text, "EDF_HeaderSize", str(header_size))

    header_bytes = header_text.encode("latin-1", errors="ignore")

    if len(header_bytes) > header_size:
        header_size = int(np.ceil(len(header_bytes) / 1024) * 1024)
        header_text = update_edf_header_value(header_text, "EDF_HeaderSize", str(header_size))
        header_bytes = header_text.encode("latin-1", errors="ignore")

    header_bytes = header_bytes + b" " * (header_size - len(header_bytes))

    output = image.astype(np.float32)
    output_dtype = output.dtype.newbyteorder(">" if byte_order.lower() == "highbytefirst" else "<")
    output = output.astype(output_dtype, copy=False)

    with open(filename, "wb") as file:
        file.write(header_bytes)
        file.write(output.tobytes(order="C"))


def read_h5_first_image(filename: str):
    filename = Path(filename)
    datasets = []

    def collect_dataset(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.ndim >= 2:
            datasets.append(name)

    with h5py.File(filename, "r") as h5:
        h5.visititems(collect_dataset)

        if not datasets:
            raise ValueError("No 2D or 3D dataset found in this H5 file.")

        preferred = None
        for name in datasets:
            lower = name.lower()
            if "data" in lower or "eiger" in lower or "detector" in lower:
                preferred = name
                break

        dataset_name = preferred or datasets[0]
        dataset = h5[dataset_name]

        header = {
            "Dataset": dataset_name,
            "Shape": str(dataset.shape),
            "Dtype": str(dataset.dtype),
        }

        for key, value in dataset.attrs.items():
            header[key] = str(value)

        if dataset.ndim == 2:
            image = np.asarray(dataset[...], dtype=np.float64)
        elif dataset.ndim == 3:
            shape = dataset.shape
            frame_axis = int(np.argmin(shape))

            if frame_axis == 0:
                image = np.asarray(dataset[0, :, :], dtype=np.float64)
                header["Displayed frame"] = "0 from axis 0"
            elif frame_axis == 1:
                image = np.asarray(dataset[:, 0, :], dtype=np.float64)
                header["Displayed frame"] = "0 from axis 1"
            else:
                image = np.asarray(dataset[:, :, 0], dtype=np.float64)
                header["Displayed frame"] = "0 from axis 2"
        else:
            raise ValueError("Only 2D and 3D H5 datasets are supported here.")

    return image, header


def get_header_float(header: dict, *names):
    for name in names:
        if name in header:
            try:
                return float(header[name])
            except (TypeError, ValueError):
                return None
    return None


# ============================================================
# ========================= CAVE TOOLS ========================
# ============================================================

def apply_central_symmetry_cave(image, xc, yc, nan_operator=">=", nan_threshold=4e9, use_id13_beamstop=False, beamstop_y=1376):
    source = image.astype(np.float64).copy()
    cave_mask = np.zeros(source.shape, dtype=bool)

    if nan_operator == ">=":
        cave_mask |= source >= nan_threshold
    elif nan_operator == "<=":
        cave_mask |= source <= nan_threshold

    cave_mask |= ~np.isfinite(source)
    source[cave_mask] = np.nan
    filled = source.copy()

    ny, nx = source.shape

    if use_id13_beamstop:
        x1 = int(round(xc))
        x2 = nx
        y1 = int(round(yc))
        y2 = int(round(beamstop_y))

        x1 = max(0, min(x1, nx - 1))
        x2 = max(0, min(x2, nx))
        y1 = max(0, min(y1, ny - 1))
        y2 = max(0, min(y2, ny))

        if y2 < y1:
            y1, y2 = y2, y1

        cave_mask[y1:y2, x1:x2] = True
        source[y1:y2, x1:x2] = np.nan
        filled[y1:y2, x1:x2] = np.nan

    missing_y, missing_x = np.where(cave_mask)

    for y, x in zip(missing_y, missing_x):
        xs = int(round(2 * xc - x))
        ys = int(round(2 * yc - y))

        if 0 <= xs < nx and 0 <= ys < ny:
            value = source[ys, xs]
            if np.isfinite(value):
                filled[y, x] = value

    return source, filled, cave_mask


# ============================================================
# =========================== CANVAS ==========================
# ============================================================

class ImageCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)

        self.image_artist = None
        self.raw_image = None
        self.coordinate_label = None
        self.ax.set_axis_off()
        self.fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
        self.mpl_connect("motion_notify_event", self._on_motion)

    def set_coordinate_label(self, label, image_name):
        self.coordinate_label = label
        self.image_name = image_name

    def _on_motion(self, event):
        if self.coordinate_label is None:
            return

        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            self.coordinate_label.setText(f"{self.image_name} | x = - | y = - | I = -")
            return

        x = int(round(event.xdata + 1))
        y = int(round(event.ydata + 1))
        intensity_text = "I = -"

        if self.raw_image is not None:
            ny, nx = self.raw_image.shape
            if 1 <= x <= nx and 1 <= y <= ny:
                value = self.raw_image[y - 1, x - 1]
                if np.isfinite(value):
                    intensity_text = f"I = {value:.6g}"
                else:
                    intensity_text = "I = NaN"

        self.coordinate_label.setText(f"{self.image_name} | x = {x} | y = {y} | {intensity_text}")

    def show_image(self, image, xc=None, yc=None, title="", vmin=None, vmax=None, white_mask=None):
        self.raw_image = image
        self.ax.clear()
        self.ax.set_axis_off()

        display = image.astype(np.float64).copy()
        display[~np.isfinite(display)] = np.nan
        display[display < 0] = np.nan

        with np.errstate(invalid="ignore", divide="ignore"):
            display = np.log10(display + 1)

        if white_mask is not None:
            display = display.copy()
            display[white_mask] = np.nan
            cmap = self.fig.canvas.figure.axes[0].images[0].cmap.copy() if self.ax.images else "jet"
        else:
            cmap = "jet"

        self.image_artist = self.ax.imshow(
            display,
            origin="upper",
            cmap=cmap,
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )

        if white_mask is not None:
            self.image_artist.cmap.set_bad(color="white")

        if xc is not None and yc is not None:
            self.ax.axvline(xc, color="red", linewidth=1.0)
            self.ax.axhline(yc, color="red", linewidth=1.0)
            self.ax.plot(xc, yc, "wo", markersize=4)

        if title:
            self.ax.set_title(title, fontsize=10)

        self.ax.set_aspect("equal")
        self.draw_idle()


# ============================================================
# =========================== CAVE TAB ========================
# ============================================================

class CaveTab(QWidget):
    """Cave tab: fill masked detector zones by central symmetry."""

    def __init__(self):
        super().__init__()

        self.current_file = None
        self.file_type = None
        self.header = {}
        self.raw_header_text = ""
        self.byte_order = "LowByteFirst"

        self.image = None
        self.image_clean = None
        self.image_filled = None
        self.cave_mask = None
        self.display_vmin = 0.0
        self.display_vmax = 1.0
        self.slider_scale = 1000

        self.instrument_mode = "XENOCS"

        self.build_ui()
        self.set_controls_enabled(False)
        self.update_centre_warning_labels()
        self.update_beamstop_visibility()

    def build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(8)

        top_layout = QHBoxLayout()
        top_layout.setSpacing(8)
        main_layout.addLayout(top_layout)

        original_box = QGroupBox("Original pattern")
        original_layout = QVBoxLayout(original_box)
        original_layout.setContentsMargins(6, 18, 6, 6)
        self.canvas_original = ImageCanvas()
        self.original_coordinate_label = QLabel("Original | x = - | y = - | I = -")
        self.original_coordinate_label.setMinimumHeight(28)
        self.original_coordinate_label.setAlignment(Qt.AlignCenter)
        self.original_coordinate_label.setStyleSheet("""
            QLabel {
                background-color: #f4f4f4;
                border-radius: 8px;
                padding: 6px;
                font-family: Menlo, Monaco, monospace;
                font-size: 11px;
            }
        """)
        self.canvas_original.set_coordinate_label(self.original_coordinate_label, "Original")
        original_layout.addWidget(self.canvas_original, stretch=1)
        original_layout.addWidget(self.original_coordinate_label, stretch=0)

        controls_box = QGroupBox("Cave tools")
        controls_layout = QVBoxLayout(controls_box)
        controls_layout.setContentsMargins(8, 18, 8, 8)
        controls_layout.setSpacing(6)

        cave_box = QGroupBox("Cave-filled pattern")
        cave_layout = QVBoxLayout(cave_box)
        cave_layout.setContentsMargins(6, 18, 6, 6)
        self.canvas_cave = ImageCanvas()
        self.cave_coordinate_label = QLabel("Cave | x = - | y = - | I = -")
        self.cave_coordinate_label.setMinimumHeight(28)
        self.cave_coordinate_label.setAlignment(Qt.AlignCenter)
        self.cave_coordinate_label.setStyleSheet("""
            QLabel {
                background-color: #f4f4f4;
                border-radius: 8px;
                padding: 6px;
                font-family: Menlo, Monaco, monospace;
                font-size: 11px;
            }
        """)
        self.canvas_cave.set_coordinate_label(self.cave_coordinate_label, "Cave")
        cave_layout.addWidget(self.canvas_cave, stretch=1)
        cave_layout.addWidget(self.cave_coordinate_label, stretch=0)

        top_layout.addWidget(original_box, stretch=1)
        top_layout.addWidget(controls_box, stretch=0)
        top_layout.addWidget(cave_box, stretch=1)

        self.open_button = QPushButton("Open EDF / H5")
        self.open_button.clicked.connect(self.open_file)
        controls_layout.addWidget(self.open_button)

        controls_layout.addWidget(QLabel("Instrument preset:"))
        preset_layout = QHBoxLayout()
        preset_layout.setSpacing(4)
        self.btn_xenocs = QPushButton("XENOCS")
        self.btn_id02 = QPushButton("ID02")
        self.btn_id13 = QPushButton("ID13")
        self.btn_custom = QPushButton("Custom")

        for button in [self.btn_xenocs, self.btn_id02, self.btn_id13, self.btn_custom]:
            button.setCheckable(True)
            preset_layout.addWidget(button)

        self.btn_xenocs.setChecked(True)
        controls_layout.addLayout(preset_layout)

        self.xc_spin = QDoubleSpinBox()
        self.xc_spin.setRange(-100000, 100000)
        self.xc_spin.setDecimals(3)

        self.yc_spin = QDoubleSpinBox()
        self.yc_spin.setRange(-100000, 100000)
        self.yc_spin.setDecimals(3)

        self.beamstop_y_spin = QDoubleSpinBox()
        self.beamstop_y_spin.setRange(0, 100000)
        self.beamstop_y_spin.setDecimals(0)
        self.beamstop_y_spin.setValue(1376)

        self.centre_x_label = QLabel("Centre X:")
        self.centre_y_label = QLabel("Centre Y:")

        form_layout = QGridLayout()
        form_layout.addWidget(self.centre_x_label, 0, 0)
        form_layout.addWidget(self.xc_spin, 0, 1)
        form_layout.addWidget(self.centre_y_label, 1, 0)
        form_layout.addWidget(self.yc_spin, 1, 1)
        self.beamstop_y_label = QLabel("ID13 beamstop Y:")
        form_layout.addWidget(self.beamstop_y_label, 2, 0)
        form_layout.addWidget(self.beamstop_y_spin, 2, 1)
        controls_layout.addLayout(form_layout)

        self.nan_operator_combo = QComboBox()
        self.nan_operator_combo.addItems(["<=", ">="])

        self.nan_threshold_spin = QDoubleSpinBox()
        self.nan_threshold_spin.setRange(-1e12, 1e12)
        self.nan_threshold_spin.setDecimals(6)
        self.nan_threshold_spin.setValue(-14)

        nan_layout = QGridLayout()
        nan_layout.addWidget(QLabel("Set NaN if I"), 0, 0)
        nan_layout.addWidget(self.nan_operator_combo, 0, 1)
        nan_layout.addWidget(self.nan_threshold_spin, 0, 2)

        self.id13_beamstop_checkbox = QCheckBox("Add ID13 beamstop mask")
        self.id13_beamstop_checkbox.setChecked(False)

        self.save_checkbox = QCheckBox("Save output after Run Cave")
        self.save_checkbox.setChecked(True)

        controls_layout.addLayout(nan_layout)
        controls_layout.addWidget(self.id13_beamstop_checkbox)
        controls_layout.addWidget(self.save_checkbox)

        intensity_box = QGroupBox("Display intensity")
        intensity_layout = QGridLayout(intensity_box)
        intensity_layout.setContentsMargins(6, 18, 6, 6)
        intensity_layout.setSpacing(4)

        self.vmin_slider = QSlider(Qt.Horizontal)
        self.vmax_slider = QSlider(Qt.Horizontal)
        self.vmin_slider.setRange(0, self.slider_scale)
        self.vmax_slider.setRange(0, self.slider_scale)
        self.vmin_slider.setValue(0)
        self.vmax_slider.setValue(self.slider_scale)

        self.vmin_label = QLabel("Min: 0.000")
        self.vmax_label = QLabel("Max: 1.000")

        intensity_layout.addWidget(self.vmin_label, 0, 0)
        intensity_layout.addWidget(self.vmin_slider, 0, 1)
        intensity_layout.addWidget(self.vmax_label, 1, 0)
        intensity_layout.addWidget(self.vmax_slider, 1, 1)

        controls_layout.addWidget(intensity_box)

        button_layout = QHBoxLayout()
        self.run_button = QPushButton("Run Cave")
        self.run_button.clicked.connect(self.run_cave)
        self.save_button = QPushButton("Save Cave")
        self.save_button.clicked.connect(self.save_cave)
        button_layout.addWidget(self.run_button)
        button_layout.addWidget(self.save_button)
        controls_layout.addLayout(button_layout)

        self.status = QTextEdit()
        self.status.setReadOnly(True)
        self.status.setPlaceholderText("Cave processing information will appear here.")
        controls_layout.addWidget(self.status, stretch=1)

        self.btn_xenocs.clicked.connect(lambda: self.set_instrument_mode("XENOCS"))
        self.btn_id02.clicked.connect(lambda: self.set_instrument_mode("ID02"))
        self.btn_id13.clicked.connect(lambda: self.set_instrument_mode("ID13"))
        self.btn_custom.clicked.connect(lambda: self.set_instrument_mode("Custom"))

        self.xc_spin.valueChanged.connect(self.refresh_preview)
        self.yc_spin.valueChanged.connect(self.refresh_preview)
        self.beamstop_y_spin.valueChanged.connect(self.refresh_preview)
        self.nan_operator_combo.currentTextChanged.connect(self.refresh_preview)
        self.nan_threshold_spin.valueChanged.connect(self.refresh_preview)
        self.id13_beamstop_checkbox.stateChanged.connect(self.refresh_preview)
        self.vmin_slider.valueChanged.connect(self.update_display_limits_from_sliders)
        self.vmax_slider.valueChanged.connect(self.update_display_limits_from_sliders)

    def set_controls_enabled(self, enabled):
        for widget in [
            self.btn_xenocs,
            self.btn_id02,
            self.btn_id13,
            self.btn_custom,
            self.xc_spin,
            self.yc_spin,
            self.beamstop_y_spin,
            self.nan_operator_combo,
            self.nan_threshold_spin,
            self.save_checkbox,
            self.vmin_slider,
            self.vmax_slider,
            self.run_button,
            self.save_button,
        ]:
            widget.setEnabled(enabled)

        self.update_beamstop_visibility()
    def auto_set_display_limits(self):
        if self.image is None:
            return

        display = self.image.astype(np.float64).copy()
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
        self.refresh_preview()

    def update_display_labels(self):
        vmin, vmax = self.current_display_limits()
        self.vmin_label.setText(f"Min: {vmin:.3f}")
        self.vmax_label.setText(f"Max: {vmax:.3f}")

    def set_instrument_mode(self, mode):
        self.instrument_mode = mode

        buttons = {
            "XENOCS": self.btn_xenocs,
            "ID02": self.btn_id02,
            "ID13": self.btn_id13,
            "Custom": self.btn_custom,
        }

        for key, button in buttons.items():
            button.blockSignals(True)
            button.setChecked(key == mode)
            button.blockSignals(False)

        self.apply_instrument_preset()
        self.update_centre_warning_labels()
        self.update_beamstop_visibility()
        self.refresh_preview()

    def update_centre_warning_labels(self):
        warning = " ⚠️" if self.instrument_mode in ["ID02", "ID13", "Custom"] else ""
        self.centre_x_label.setText(f"Centre X:{warning}")
        self.centre_y_label.setText(f"Centre Y:{warning}")

    def update_beamstop_visibility(self):
        is_id13 = self.instrument_mode == "ID13"

        self.beamstop_y_label.setVisible(is_id13)
        self.beamstop_y_spin.setVisible(is_id13)
        self.id13_beamstop_checkbox.setVisible(is_id13)

        self.beamstop_y_spin.setEnabled(is_id13 and self.image is not None)
        self.id13_beamstop_checkbox.setEnabled(is_id13 and self.image is not None)

        self.id13_beamstop_checkbox.blockSignals(True)
        self.id13_beamstop_checkbox.setChecked(is_id13)
        self.id13_beamstop_checkbox.blockSignals(False)

    def apply_instrument_preset(self):
        if self.instrument_mode == "XENOCS":
            center_1 = get_header_float(self.header, "Center_1", "center_1")
            center_2 = get_header_float(self.header, "Center_2", "center_2")
            self.xc_spin.setValue(center_1 if center_1 is not None else 0)
            self.yc_spin.setValue(center_2 if center_2 is not None else 0)
            self.nan_operator_combo.setCurrentText("<=")
            self.nan_threshold_spin.setValue(-14)
            return

        if self.instrument_mode == "ID02":
            self.xc_spin.setValue(919.689)
            self.yc_spin.setValue(994.290)
            self.nan_operator_combo.setCurrentText("<=")
            self.nan_threshold_spin.setValue(-9)
            return

        if self.instrument_mode == "ID13":
            self.xc_spin.setValue(1294.689)
            self.yc_spin.setValue(1310.290)
            self.nan_operator_combo.setCurrentText(">=")
            self.nan_threshold_spin.setValue(4e9)
            return

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open EDF or H5 file",
            "",
            "Data files (*.edf *.h5 *.hdf5);;EDF (*.edf);;HDF5 (*.h5 *.hdf5);;All files (*)",
        )

        if not file_path:
            return

        try:
            path = Path(file_path)
            suffix = path.suffix.lower()

            if suffix == ".edf":
                image, header, raw_header_text, byte_order = read_edf_file(file_path)
                self.file_type = "EDF"
                self.raw_header_text = raw_header_text
                self.byte_order = byte_order
            elif suffix in [".h5", ".hdf5"]:
                image, header = read_h5_first_image(file_path)
                self.file_type = "H5"
                self.raw_header_text = ""
                self.byte_order = "LowByteFirst"
            else:
                raise ValueError("Unsupported file format. Please select an EDF, H5 or HDF5 file.")

            self.current_file = path
            self.header = header
            self.image = image.astype(np.float64)
            self.image_clean = None
            self.image_filled = None
            self.cave_mask = None

            self.set_controls_enabled(True)
            self.apply_instrument_preset()
            self.update_centre_warning_labels()
            self.update_beamstop_visibility()
            self.auto_set_display_limits()
            self.refresh_preview()
            self.update_status()

        except Exception as error:
            QMessageBox.critical(self, "File reading error", str(error))

    def refresh_preview(self):
        if self.image is None:
            return

        use_id13_beamstop = self.instrument_mode == "ID13" and self.id13_beamstop_checkbox.isChecked()

        clean, filled, cave_mask = apply_central_symmetry_cave(
            self.image,
            self.xc_spin.value(),
            self.yc_spin.value(),
            nan_operator=self.nan_operator_combo.currentText(),
            nan_threshold=self.nan_threshold_spin.value(),
            use_id13_beamstop=use_id13_beamstop,
            beamstop_y=self.beamstop_y_spin.value(),
        )

        self.image_clean = clean
        self.image_filled = filled
        self.cave_mask = cave_mask
        vmin, vmax = self.current_display_limits()
        self.canvas_original.show_image(self.image, self.xc_spin.value(), self.yc_spin.value(), vmin=vmin, vmax=vmax, white_mask=cave_mask)
        self.canvas_cave.show_image(filled, self.xc_spin.value(), self.yc_spin.value(), vmin=vmin, vmax=vmax)

    def run_cave(self):
        if self.image is None:
            return

        self.refresh_preview()
        self.update_status()

        if self.save_checkbox.isChecked():
            self.save_cave()

    def save_cave(self):
        if self.image_filled is None or self.current_file is None:
            return

        if self.file_type == "EDF":
            suggested_path = self.current_file.parent / f"{self.current_file.stem}_cave.edf"
            output_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save cave EDF",
                str(suggested_path),
                "EDF (*.edf);;All files (*)",
            )

            if not output_path:
                return

            if not output_path.lower().endswith(".edf"):
                output_path += ".edf"

            try:
                write_edf_file(output_path, self.image_filled, self.raw_header_text, self.byte_order)
                self.status.append(f"\nSaved cave EDF:\n{output_path}")
            except Exception as error:
                QMessageBox.critical(self, "Save error", str(error))

        else:
            suggested_path = self.current_file.parent / f"{self.current_file.stem}_cave.npy"
            output_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save cave array",
                str(suggested_path),
                "NumPy array (*.npy);;All files (*)",
            )

            if not output_path:
                return

            if not output_path.lower().endswith(".npy"):
                output_path += ".npy"

            try:
                np.save(output_path, self.image_filled)
                self.status.append(f"\nSaved cave array:\n{output_path}")
            except Exception as error:
                QMessageBox.critical(self, "Save error", str(error))

    def update_status(self):
        if self.current_file is None:
            return

        lines = [
            f"File: {self.current_file.name}",
            f"Format: {self.file_type}",
        ]

        if self.file_type == "H5" and "Dataset" in self.header:
            lines.append(f"Dataset: {self.header['Dataset']}")

        if self.image is not None:
            lines.append(f"Image size: {self.image.shape[1]} x {self.image.shape[0]}")

        self.status.setPlainText("\n".join(lines))
