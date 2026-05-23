import re
from pathlib import Path

import h5py
import numpy as np

from PySide6.QtCore import Qt, QEvent, QPoint
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QGroupBox,
    QDoubleSpinBox,
    QSpinBox,
    QTextEdit,
    QCheckBox,
    QGridLayout,
    QMessageBox,
    QSlider,
    QComboBox,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from .instrument_presets import (
    ID13_DEFAULT_CENTER_X,
    ID13_DEFAULT_CENTER_Y,
    ID13_DEFAULT_DISTANCE_M,
    ID13_DEFAULT_PIXEL_MM,
    ID13_DEFAULT_WAVELENGTH_A,
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
    style_q_geometry_buttons,
)


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


def add_matching_edf_center(header: dict, filename: str):
    edf_path = Path(filename).with_suffix(".edf")
    if not edf_path.exists():
        return header

    try:
        _, edf_header, *_ = read_edf_file(edf_path)
    except Exception:
        return header

    copied = False
    for key in ["Center_1", "Center_2", "center_1", "center_2"]:
        if key in edf_header and key not in header:
            header[key] = edf_header[key]
            copied = True

    if copied:
        header["Center source"] = edf_path.name

    return header


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


# New function for writing cave-filled H5 frames
def write_h5_frame_file(filename: str, image: np.ndarray, source_file: str, source_dataset_name: str, frame_index: int):
    filename = Path(filename)
    source_file = Path(source_file)

    with h5py.File(filename, "w") as out:
        dataset = out.create_dataset("/entry_0000/instrument/eiger/data", data=image.astype(np.float32), compression="gzip")
        dataset.attrs["source_file"] = str(source_file.name)
        dataset.attrs["source_dataset"] = str(source_dataset_name)
        dataset.attrs["source_frame"] = int(frame_index)
        dataset.attrs["processing"] = "central symmetry cave filling"


def inspect_h5_image_dataset(filename: str):
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
        shape = tuple(dataset.shape)

        header = {
            "Dataset": dataset_name,
            "Shape": str(shape),
            "Dtype": str(dataset.dtype),
        }

        for key, value in dataset.attrs.items():
            header[key] = str(value)

        add_matching_edf_center(header, filename)

        if dataset.ndim == 2:
            frame_axis = None
            n_frames = 1
        elif dataset.ndim == 3:
            frame_axis = int(np.argmin(shape))
            n_frames = int(shape[frame_axis])
            header["Frame axis"] = str(frame_axis)
            header["Number of frames"] = str(n_frames)
        else:
            raise ValueError("Only 2D and 3D H5 datasets are supported here.")

    return dataset_name, shape, frame_axis, n_frames, header


def read_h5_frame(filename: str, dataset_name: str, frame_index: int = 0):
    filename = Path(filename)

    with h5py.File(filename, "r") as h5:
        dataset = h5[dataset_name]

        header = {
            "Dataset": dataset_name,
            "Shape": str(tuple(dataset.shape)),
            "Dtype": str(dataset.dtype),
        }

        for key, value in dataset.attrs.items():
            header[key] = str(value)

        add_matching_edf_center(header, filename)

        if dataset.ndim == 2:
            image = np.asarray(dataset[...], dtype=np.float64)
            header["Displayed frame"] = "single 2D image"
        elif dataset.ndim == 3:
            shape = dataset.shape
            frame_axis = int(np.argmin(shape))
            n_frames = int(shape[frame_axis])
            frame_index = max(0, min(int(frame_index), n_frames - 1))

            if frame_axis == 0:
                image = np.asarray(dataset[frame_index, :, :], dtype=np.float64)
            elif frame_axis == 1:
                image = np.asarray(dataset[:, frame_index, :], dtype=np.float64)
            else:
                image = np.asarray(dataset[:, :, frame_index], dtype=np.float64)

            header["Frame axis"] = str(frame_axis)
            header["Displayed frame"] = f"{frame_index} from axis {frame_axis}"
            header["Number of frames"] = str(n_frames)
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


ID02_DEFAULT_CENTER_X = 914.4
ID02_DEFAULT_CENTER_Y = 996.5
ID02_DEFAULT_DISTANCE_M = 10.0002
ID02_DEFAULT_PIXEL_MM = 0.075
ID02_DEFAULT_WAVELENGTH_A = 1.01402
CENTER_X_KEYS = ("Center_1", "center_1", "CenterX", "center_x", "BeamCenterX", "Beam_x", "beam_x")
CENTER_Y_KEYS = ("Center_2", "center_2", "CenterY", "center_y", "BeamCenterY", "Beam_y", "beam_y")


# ============================================================
# ========================= CAVE TOOLS ========================
# ============================================================

def apply_central_symmetry_cave(
    image,
    xc,
    yc,
    nan_operator=">=",
    nan_threshold=4e9,
    use_id13_beamstop=False,
    beamstop_y=1376,
    expand_nan_neighbors=False,
):
    source = image.astype(np.float64).copy()
    cave_mask = np.zeros(source.shape, dtype=bool)

    if nan_operator == ">=":
        cave_mask |= source >= nan_threshold
    elif nan_operator == "<=":
        cave_mask |= source <= nan_threshold

    cave_mask |= ~np.isfinite(source)

    if expand_nan_neighbors:
        original_nan_mask = cave_mask.copy()
        radius = 2

        padded_mask = np.pad(
            original_nan_mask,
            radius,
            mode="constant",
            constant_values=False,
        )

        expanded_mask = np.zeros_like(original_nan_mask, dtype=bool)

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                expanded_mask |= padded_mask[
                    radius + dy:radius + dy + original_nan_mask.shape[0],
                    radius + dx:radius + dx + original_nan_mask.shape[1],
                ]

        cave_mask = expanded_mask

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
        self.image_artist = None
        self.raw_image = None
        self.coordinate_label = None
        self.q_calculator = None
        self.image_name = "Image"
        self._is_panning = False
        self._pan_start_pos = None
        self._pan_start_xlim = None
        self._pan_start_ylim = None
        self._data_xlim = None
        self._data_ylim = None

        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setFocusPolicy(Qt.StrongFocus)
        self.ax.set_axis_off()
        self.fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
        self.mpl_connect("motion_notify_event", self._on_motion)

        try:
            self.grabGesture(Qt.PinchGesture)
        except Exception:
            pass

    def set_coordinate_label(self, label, image_name):
        self.coordinate_label = label
        self.image_name = image_name

    def coordinate_text(self, text):
        if self.image_name:
            return f"{self.image_name} | {text}"
        return text

    def set_q_calculator(self, calculator):
        self.q_calculator = calculator

    def event(self, event):
        if getattr(self, "raw_image", None) is not None:
            try:
                if event.type() == QEvent.NativeGesture:
                    gesture_type = event.gestureType()
                    value = event.value()
                    if gesture_type == Qt.ZoomNativeGesture and value != 0:
                        scale = 1.0 / (1.0 + value) if value > -0.95 else 1.25
                        self._zoom_from_qpoint(self._event_center_point(event), scale)
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
                            self._zoom_from_qpoint(self._event_center_point(event), 1.0 / factor)
                            event.accept()
                            return True
            except Exception:
                pass

        return super().event(event)

    def wheelEvent(self, event):
        if self.raw_image is None:
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
                self._zoom_from_qpoint(event.position(), scale)
        else:
            self._pan_by_trackpad(dx, dy)
        event.accept()

    def _event_center_point(self, event):
        try:
            position = event.position()
            if position is not None:
                return position
        except Exception:
            pass

        return self.rect().center()

    def mousePressEvent(self, event):
        if self.raw_image is not None and event.button() == Qt.LeftButton:
            self._is_panning = True
            self._pan_start_pos = event.position()
            self._pan_start_xlim = self.ax.get_xlim()
            self._pan_start_ylim = self.ax.get_ylim()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._is_panning and self.raw_image is not None:
            start_x, start_y = self._qt_pos_to_data(self._pan_start_pos.x(), self._pan_start_pos.y())
            current_x, current_y = self._qt_pos_to_data(event.position().x(), event.position().y())

            if None not in (start_x, start_y, current_x, current_y):
                dx = start_x - current_x
                dy = start_y - current_y
                x0, x1 = self._pan_start_xlim
                y0, y1 = self._pan_start_ylim
                self.ax.set_xlim(x0 + dx, x1 + dx)
                self.ax.set_ylim(y0 + dy, y1 + dy)
                self.draw_idle()

            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._is_panning and event.button() == Qt.LeftButton:
            self._is_panning = False
            self._pan_start_pos = None
            self._pan_start_xlim = None
            self._pan_start_ylim = None
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.raw_image is not None:
            self.reset_view()
            event.accept()
            return

        super().mouseDoubleClickEvent(event)

    def _qt_pos_to_data(self, x, y):
        if self.ax is None:
            return None, None

        canvas_height = self.height()
        display_x = x
        display_y = canvas_height - y

        try:
            return self.ax.transData.inverted().transform((display_x, display_y))
        except Exception:
            return None, None

    def _zoom_at(self, xdata, ydata, zoom_factor):
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()

        new_width = (x1 - x0) * zoom_factor
        new_height = (y1 - y0) * zoom_factor

        rel_x = (xdata - x0) / (x1 - x0) if x1 != x0 else 0.5
        rel_y = (ydata - y0) / (y1 - y0) if y1 != y0 else 0.5

        self.ax.set_xlim(xdata - new_width * rel_x, xdata + new_width * (1 - rel_x))
        self.ax.set_ylim(ydata - new_height * rel_y, ydata + new_height * (1 - rel_y))
        self.draw_idle()

    def _zoom_from_qpoint(self, qpoint, zoom_factor):
        try:
            xdata, ydata = self._qt_pos_to_data(float(qpoint.x()), float(qpoint.y()))
        except Exception:
            xdata, ydata = None, None

        if xdata is None or ydata is None:
            return

        self._zoom_at(xdata, ydata, zoom_factor)

    def _pan_by_trackpad(self, dx, dy):
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        xspan = x1 - x0
        yspan = y1 - y0
        shift_x = -dx * xspan * 0.08
        shift_y = dy * yspan * 0.08
        self.ax.set_xlim(x0 + shift_x, x1 + shift_x)
        self.ax.set_ylim(y0 + shift_y, y1 + shift_y)
        self.draw_idle()

    def reset_view(self):
        if self._data_xlim is not None and self._data_ylim is not None:
            self.ax.set_xlim(self._data_xlim)
            self.ax.set_ylim(self._data_ylim)
            self.draw_idle()

    def _on_motion(self, event):
        if self.coordinate_label is None:
            return

        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            self.coordinate_label.setText(self.coordinate_text("x = - | y = - | I = -"))
            return

        x = int(round(event.xdata + 1))
        y = int(round(event.ydata + 1))
        intensity_text = "I = -"
        q_text = "q = -"

        if self.raw_image is not None:
            ny, nx = self.raw_image.shape
            if 1 <= x <= nx and 1 <= y <= ny:
                value = self.raw_image[y - 1, x - 1]
                if np.isfinite(value):
                    intensity_text = f"I = {value:.6g}"
                else:
                    intensity_text = "I = NaN"

                if self.q_calculator is not None:
                    q_value = self.q_calculator(x, y)
                    if q_value is not None:
                        q_text = f"q = {q_value:.6g} nm⁻¹"

        self.coordinate_label.setText(self.coordinate_text(f"x = {x} | y = {y} | {intensity_text} | {q_text}"))

    def show_image(self, image, xc=None, yc=None, title="", vmin=None, vmax=None, white_mask=None):
        previous_xlim = self.ax.get_xlim() if self.image_artist is not None else None
        previous_ylim = self.ax.get_ylim() if self.image_artist is not None else None
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

        ny, nx = image.shape
        self._data_xlim = (-0.5, nx - 0.5)
        self._data_ylim = (ny - 0.5, -0.5)

        if previous_xlim is not None and previous_ylim is not None:
            self.ax.set_xlim(previous_xlim)
            self.ax.set_ylim(previous_ylim)

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
        self.h5_dataset_name = None
        self.h5_frame_axis = None
        self.h5_n_frames = 1
        self._syncing_frame_controls = False

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
        main_layout.setContentsMargins(*PAGE_MARGINS)
        main_layout.setSpacing(BLOCK_SPACING)

        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(BLOCK_SPACING)
        main_layout.addLayout(top_layout, stretch=1)

        original_box = QGroupBox("Original pattern")
        original_layout = QVBoxLayout(original_box)
        original_layout.setContentsMargins(*GROUP_BOX_MARGINS)
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
        self.canvas_original.set_q_calculator(self.calculate_q_at_pixel)
        original_layout.addWidget(self.canvas_original, stretch=1)
        original_layout.addWidget(self.original_coordinate_label, stretch=0)

        controls_box = QGroupBox("Cave tools")
        controls_box.setFixedWidth(FILE_BROWSER_WIDTH)
        controls_layout = QVBoxLayout(controls_box)
        controls_layout.setContentsMargins(*GROUP_BOX_MARGINS)
        controls_layout.setSpacing(6)

        cave_box = QGroupBox("Cave-filled pattern")
        cave_layout = QVBoxLayout(cave_box)
        cave_layout.setContentsMargins(*GROUP_BOX_MARGINS)
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
        self.canvas_cave.set_q_calculator(self.calculate_q_at_pixel)
        cave_layout.addWidget(self.canvas_cave, stretch=1)
        cave_layout.addWidget(self.cave_coordinate_label, stretch=0)

        top_layout.addWidget(original_box, stretch=1)
        top_layout.addWidget(controls_box, stretch=0)
        top_layout.addWidget(cave_box, stretch=1)
        top_layout.setStretch(0, 1)
        top_layout.setStretch(1, 0)
        top_layout.setStretch(2, 1)

        self.open_button = QPushButton("Open EDF / H5")
        self.open_button.clicked.connect(self.open_file)
        controls_layout.addWidget(self.open_button)

        preset_layout = QHBoxLayout()
        preset_layout.setSpacing(4)
        self.btn_xenocs = QPushButton("XENOCS")
        self.btn_id02 = QPushButton("ID02")
        self.btn_id13 = QPushButton("ID13")
        self.btn_custom = QPushButton("Custom")
        self.q_manual_button = QPushButton("+")
        self.q_manual_button.clicked.connect(lambda: self.set_instrument_mode("Custom"))

        for button in [self.btn_xenocs, self.btn_id02, self.btn_id13, self.btn_custom]:
            button.setCheckable(True)
            preset_layout.addWidget(button)
        preset_layout.addWidget(self.q_manual_button)

        self.btn_xenocs.setChecked(True)
        style_q_geometry_buttons(
            {
                "XENOCS": self.btn_xenocs,
                "ID02": self.btn_id02,
                "ID13": self.btn_id13,
                "Custom": self.btn_custom,
            },
            "XENOCS",
            self.q_manual_button,
        )
        controls_layout.addLayout(preset_layout)

        self.xc_spin = QDoubleSpinBox()
        self.xc_spin.setRange(-100000, 100000)
        self.xc_spin.setDecimals(13)

        self.yc_spin = QDoubleSpinBox()
        self.yc_spin.setRange(-100000, 100000)
        self.yc_spin.setDecimals(13)

        self.beamstop_y_spin = QDoubleSpinBox()
        self.beamstop_y_spin.setRange(0, 100000)
        self.beamstop_y_spin.setDecimals(0)
        self.beamstop_y_spin.setValue(1376)

        self.centre_x_label = QLabel("Center X:")
        self.centre_y_label = QLabel("Center Y:")

        form_layout = QGridLayout()
        form_layout.addWidget(self.centre_x_label, 0, 0)
        form_layout.addWidget(self.xc_spin, 0, 1)
        form_layout.addWidget(self.centre_y_label, 1, 0)
        form_layout.addWidget(self.yc_spin, 1, 1)
        self.beamstop_y_label = QLabel("ID13 beamstop Y:")
        form_layout.addWidget(self.beamstop_y_label, 2, 0)
        form_layout.addWidget(self.beamstop_y_spin, 2, 1)

        self.frame_label = QLabel("H5 frame:")
        self.frame_spin = QSpinBox()
        self.frame_spin.setRange(1, 1)
        self.frame_spin.setValue(1)
        self.frame_spin.setEnabled(False)
        self.frame_spin.hide()

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

        self.expand_nan_neighbors_checkbox = QCheckBox("Test: expand NaN by 2 px")
        self.expand_nan_neighbors_checkbox.setChecked(False)
        self.expand_nan_neighbors_checkbox.setToolTip(
            "Expands the NaN mask by 2 pixels before central symmetry filling."
        )

        self.save_checkbox = QCheckBox("Save output after Run Cave")
        self.save_checkbox.setChecked(True)

        controls_layout.addLayout(nan_layout)
        controls_layout.addWidget(self.id13_beamstop_checkbox)
        controls_layout.addWidget(self.expand_nan_neighbors_checkbox)

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
        self.run_button = QPushButton("Run Cave")
        self.run_button.clicked.connect(self.run_cave)
        self.save_button = QPushButton("Save Cave")
        self.save_button.clicked.connect(self.save_cave)
        button_layout.addWidget(self.run_button)
        button_layout.addWidget(self.save_button)
        controls_layout.addLayout(button_layout)

        self.status = QTextEdit()
        self.status.setReadOnly(True)
        self.status.setPlaceholderText("")
        controls_layout.addWidget(self.status, stretch=1)

        self.btn_xenocs.clicked.connect(lambda: self.set_instrument_mode("XENOCS"))
        self.btn_id02.clicked.connect(lambda: self.set_instrument_mode("ID02"))
        self.btn_id13.clicked.connect(lambda: self.set_instrument_mode("ID13"))
        self.btn_custom.clicked.connect(lambda: self.set_instrument_mode("Custom"))

        self.xc_spin.valueChanged.connect(self.refresh_preview)
        self.yc_spin.valueChanged.connect(self.refresh_preview)
        self.beamstop_y_spin.valueChanged.connect(self.refresh_preview)
        self.frame_spin.valueChanged.connect(self.load_selected_h5_frame)
        self.nan_operator_combo.currentTextChanged.connect(self.refresh_preview)
        self.nan_threshold_spin.valueChanged.connect(self.refresh_preview)
        self.id13_beamstop_checkbox.stateChanged.connect(self.refresh_preview)
        self.expand_nan_neighbors_checkbox.stateChanged.connect(self.refresh_preview)
        self.vmin_slider.valueChanged.connect(self.update_display_limits_from_sliders)
        self.vmax_slider.valueChanged.connect(self.update_display_limits_from_sliders)

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
        self.frame_counter_label = QLabel("1 / 1")
        self.frame_counter_label.setMinimumWidth(FRAME_COUNTER_WIDTH)
        self.frame_counter_label.setAlignment(Qt.AlignCenter)

        frame_nav.addWidget(QLabel("Start:"))
        frame_nav.addWidget(self.frame_start_spin)
        frame_nav.addWidget(self.prev_frame_button)
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(1, 1)
        self.frame_slider.setValue(1)
        frame_nav.addWidget(self.frame_slider, stretch=1)
        frame_nav.addWidget(self.next_frame_button)
        frame_nav.addWidget(QLabel("End:"))
        frame_nav.addWidget(self.frame_end_spin)
        frame_nav.addWidget(self.frame_counter_label)
        main_layout.addLayout(frame_nav, stretch=0)

        self.frame_start_spin.valueChanged.connect(self.update_frame_bounds)
        self.frame_end_spin.valueChanged.connect(self.update_frame_bounds)
        self.frame_slider.valueChanged.connect(self.frame_slider_changed)
        self.prev_frame_button.clicked.connect(self.previous_frame)
        self.next_frame_button.clicked.connect(self.next_frame)

    def set_controls_enabled(self, enabled):
        for widget in [
            self.btn_xenocs,
            self.btn_id02,
            self.btn_id13,
            self.btn_custom,
            self.xc_spin,
            self.yc_spin,
            self.beamstop_y_spin,
            self.frame_spin,
            self.frame_slider,
            self.nan_operator_combo,
            self.nan_threshold_spin,
            self.id13_beamstop_checkbox,
            self.expand_nan_neighbors_checkbox,
            self.lock_intensity_checkbox,
            self.vmin_slider,
            self.vmax_slider,
            self.run_button,
            self.save_button,
        ]:
            widget.setEnabled(enabled)

        self.update_frame_selector_visibility()
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

        style_q_geometry_buttons(buttons, mode, self.q_manual_button)

        self.apply_instrument_preset()
        self.update_centre_warning_labels()
        self.update_beamstop_visibility()
        self.refresh_preview()

    def update_centre_warning_labels(self):
        self.centre_x_label.setText("Center X:")
        self.centre_y_label.setText("Center Y:")

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

    def update_frame_selector_visibility(self):
        self.frame_label.setVisible(False)
        self.frame_spin.setVisible(False)
        self.update_frame_counter()

    def configure_frame_navigation(self, n_frames):
        n_frames = max(1, int(n_frames))
        self._syncing_frame_controls = True
        for spin in [self.frame_spin, self.frame_start_spin, self.frame_end_spin]:
            spin.blockSignals(True)
        self.frame_slider.blockSignals(True)

        self.frame_spin.setRange(1, n_frames)
        self.frame_spin.setValue(1)
        self.frame_slider.setRange(1, n_frames)
        self.frame_slider.setValue(1)
        self.frame_start_spin.setRange(1, n_frames)
        self.frame_start_spin.setValue(1)
        self.frame_end_spin.setRange(1, n_frames)
        self.frame_end_spin.setValue(n_frames)

        for spin in [self.frame_spin, self.frame_start_spin, self.frame_end_spin]:
            spin.blockSignals(False)
        self.frame_slider.blockSignals(False)
        self._syncing_frame_controls = False

        self.update_frame_counter()

    def frame_slider_changed(self, value):
        if self._syncing_frame_controls:
            return

        start = self.frame_start_spin.value()
        end = self.frame_end_spin.value()
        value = max(start, min(int(value), end))

        if value != self.frame_slider.value():
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(value)
            self.frame_slider.blockSignals(False)

        self.frame_spin.setValue(value)

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

        current = self.frame_spin.value()
        if current < start:
            self.frame_spin.setValue(start)
        elif current > end:
            self.frame_spin.setValue(end)

        self.update_frame_counter()

    def update_frame_counter(self):
        current = self.frame_spin.value()
        total = max(1, self.h5_n_frames)
        self.frame_counter_label.setText(f"{current} / {total}")
        if hasattr(self, "prev_frame_button"):
            can_navigate = self.file_type == "H5" and total > 1
            self.frame_spin.setEnabled(can_navigate)
            self.frame_start_spin.setEnabled(can_navigate)
            self.frame_end_spin.setEnabled(can_navigate)
            self.frame_slider.setEnabled(can_navigate)
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(current)
            self.frame_slider.blockSignals(False)
            self.prev_frame_button.setEnabled(can_navigate and current > self.frame_start_spin.value())
            self.next_frame_button.setEnabled(can_navigate and current < self.frame_end_spin.value())

    def previous_frame(self):
        self.frame_spin.setValue(max(self.frame_start_spin.value(), self.frame_spin.value() - 1))

    def next_frame(self):
        self.frame_spin.setValue(min(self.frame_end_spin.value(), self.frame_spin.value() + 1))

    def wavelength_to_nm(self, wavelength):
        if wavelength < 1e-6:
            return wavelength * 1e9
        if wavelength >= 0.5:
            return wavelength * 0.1
        return wavelength

    def q_geometry(self):
        if self.image is None:
            return None

        xc = self.xc_spin.value()
        yc = self.yc_spin.value()

        distance_m = get_header_float(
            self.header,
            "SampleDistance",
            "sampledistance",
            "sample_distance",
            "Distance",
            "DetectorDistance",
            "detector_distance",
        )
        pixel_x = get_header_float(
            self.header,
            "PSize_1",
            "psize_1",
            "PSize_X",
            "PixelSizeX",
            "pixel_size_x",
            "x_pixel_size",
        )
        pixel_y = get_header_float(
            self.header,
            "PSize_2",
            "psize_2",
            "PSize_Y",
            "PixelSizeY",
            "pixel_size_y",
            "y_pixel_size",
        )
        wavelength = get_header_float(
            self.header,
            "WaveLength",
            "Wavelength",
            "wavelength",
            "Lambda",
            "lambda",
        )

        if self.instrument_mode == "ID02":
            distance_m = ID02_DEFAULT_DISTANCE_M if distance_m is None else distance_m
            pixel_x = ID02_DEFAULT_PIXEL_MM if pixel_x is None else pixel_x
            pixel_y = ID02_DEFAULT_PIXEL_MM if pixel_y is None else pixel_y
            wavelength = ID02_DEFAULT_WAVELENGTH_A if wavelength is None else wavelength
        elif self.instrument_mode == "ID13":
            distance_m = ID13_DEFAULT_DISTANCE_M
            pixel_x = ID13_DEFAULT_PIXEL_MM
            pixel_y = ID13_DEFAULT_PIXEL_MM
            wavelength = ID13_DEFAULT_WAVELENGTH_A

        if distance_m is None or pixel_x is None or pixel_y is None or wavelength is None:
            return None

        pixel_x_mm = pixel_x * 1000.0 if pixel_x < 1e-3 else pixel_x
        pixel_y_mm = pixel_y * 1000.0 if pixel_y < 1e-3 else pixel_y
        wavelength_nm = self.wavelength_to_nm(wavelength)

        if distance_m <= 0 or pixel_x_mm <= 0 or pixel_y_mm <= 0 or wavelength_nm <= 0:
            return None

        return xc, yc, distance_m, pixel_x_mm, pixel_y_mm, wavelength_nm

    def calculate_q_at_pixel(self, x_index, y_index):
        geometry = self.q_geometry()
        if geometry is None:
            return None

        xc, yc, distance_m, pixel_x_mm, pixel_y_mm, wavelength_nm = geometry
        dx_px = float(x_index) - float(xc)
        dy_px = float(y_index) - float(yc)
        dx_m = dx_px * pixel_x_mm * 1e-3
        dy_m = dy_px * pixel_y_mm * 1e-3
        r_m = np.sqrt(dx_m ** 2 + dy_m ** 2)
        two_theta = np.arctan2(r_m, distance_m)
        return (4.0 * np.pi / wavelength_nm) * np.sin(two_theta / 2.0)

    def apply_instrument_preset(self):
        if self.instrument_mode == "XENOCS":
            center_1 = get_header_float(self.header, *CENTER_X_KEYS)
            center_2 = get_header_float(self.header, *CENTER_Y_KEYS)
            self.xc_spin.setValue(center_1 if center_1 is not None else 0)
            self.yc_spin.setValue(center_2 if center_2 is not None else 0)
            self.nan_operator_combo.setCurrentText("<=")
            self.nan_threshold_spin.setValue(-14)
            return

        if self.instrument_mode == "ID02":
            center_1 = get_header_float(self.header, *CENTER_X_KEYS)
            center_2 = get_header_float(self.header, *CENTER_Y_KEYS)
            self.xc_spin.setValue(center_1 if center_1 is not None else ID02_DEFAULT_CENTER_X)
            self.yc_spin.setValue(center_2 if center_2 is not None else ID02_DEFAULT_CENTER_Y)
            self.nan_operator_combo.setCurrentText("<=")
            self.nan_threshold_spin.setValue(-9)
            return

        if self.instrument_mode == "ID13":
            self.xc_spin.setValue(ID13_DEFAULT_CENTER_X)
            self.yc_spin.setValue(ID13_DEFAULT_CENTER_Y)
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
                self.h5_dataset_name = None
                self.h5_frame_axis = None
                self.h5_n_frames = 1

                self.configure_frame_navigation(1)
            elif suffix in [".h5", ".hdf5"]:
                dataset_name, dataset_shape, frame_axis, n_frames, header = inspect_h5_image_dataset(file_path)
                image, header = read_h5_frame(file_path, dataset_name, 0)
                self.file_type = "H5"
                self.raw_header_text = ""
                self.byte_order = "LowByteFirst"
                self.h5_dataset_name = dataset_name
                self.h5_frame_axis = frame_axis
                self.h5_n_frames = n_frames

                self.configure_frame_navigation(n_frames)
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
            self.update_frame_selector_visibility()
            self.auto_set_display_limits()
            self.refresh_preview()
            self.update_status()

        except Exception as error:
            QMessageBox.critical(self, "File reading error", str(error))

    def load_selected_h5_frame(self):
        self.update_frame_counter()
        if self.file_type != "H5" or self.current_file is None or self.h5_dataset_name is None:
            return

        frame_index = self.frame_spin.value() - 1

        try:
            image, header = read_h5_frame(self.current_file, self.h5_dataset_name, frame_index)
            self.header = header
            self.image = image.astype(np.float64)
            self.image_clean = None
            self.image_filled = None
            self.cave_mask = None

            if not self.lock_intensity_checkbox.isChecked():
                self.auto_set_display_limits()
            self.apply_instrument_preset()
            self.update_beamstop_visibility()
            self.update_frame_selector_visibility()
            self.refresh_preview()
            self.update_status()
        except Exception as error:
            QMessageBox.critical(self, "H5 frame reading error", str(error))

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
            expand_nan_neighbors=self.expand_nan_neighbors_checkbox.isChecked(),
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
            frame_suffix = f"_frame{self.frame_spin.value():04d}" if self.h5_n_frames > 1 else ""
            suggested_path = self.current_file.parent / f"{self.current_file.stem}{frame_suffix}_cave.h5"
            output_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save cave H5",
                str(suggested_path),
                "HDF5 (*.h5);;All files (*)",
            )

            if not output_path:
                return

            if not output_path.lower().endswith((".h5", ".hdf5")):
                output_path += ".h5"

            try:
                write_h5_frame_file(
                    output_path,
                    self.image_filled,
                    self.current_file,
                    self.h5_dataset_name or "data",
                    self.frame_spin.value() - 1,
                )
                self.status.append(f"\nSaved cave H5:\n{output_path}")
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
            lines.append(f"Frame: {self.frame_spin.value()} / {self.h5_n_frames}")
            if self.h5_frame_axis is not None:
                lines.append(f"Frame axis: {self.h5_frame_axis}")

        if self.image is not None:
            lines.append(f"Image size: {self.image.shape[1]} x {self.image.shape[0]}")

        self.status.setPlainText("\n".join(lines))
