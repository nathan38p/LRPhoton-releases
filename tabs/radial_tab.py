import re
from pathlib import Path

import h5py
import numpy as np

from PySide6.QtCore import Qt, Signal, QEvent
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
    QListWidget,
    QLineEdit,
    QMessageBox,
    QScrollArea,
    QComboBox,
    QSlider,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
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
        raw_header = file.read(header_size).decode("latin-1", errors="ignore")

    header = parse_edf_header(raw_header)

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
    return image, header


def add_matching_edf_center(header: dict, filename: str):
    edf_path = Path(filename).with_suffix(".edf")
    if not edf_path.exists():
        return header

    try:
        _, edf_header = read_edf_file(edf_path)
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


def read_h5_frame(filename: str, dataset_name: str = None, frame_index: int = 0):
    filename = Path(filename)

    if dataset_name is None:
        dataset_name, _, _, _, _ = inspect_h5_image_dataset(filename)

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


def read_image_file(file_path, h5_dataset_name=None, h5_frame_index=0):
    suffix = Path(file_path).suffix.lower()
    if suffix == ".edf":
        return read_edf_file(file_path)
    if suffix in [".h5", ".hdf5"]:
        return read_h5_frame(file_path, h5_dataset_name, h5_frame_index)
    raise ValueError("Unsupported file format. Please select EDF, H5 or HDF5.")


def get_header_float(header: dict, *names):
    for name in names:
        if name in header:
            try:
                return float(header[name])
            except (TypeError, ValueError):
                return None
    return None


# ============================================================
# ==================== WAVELENGTH UTILS ======================
# ============================================================

def wavelength_to_nm(value: float):
    """
    Convert wavelength to nm with automatic unit detection.

    Typical cases:
    - EDF/H5 header in meters: 8.26563e-11 m -> 0.0826563 nm
    - Interface value in Å: 0.826563 Å -> 0.0826563 nm
    - Already in nm: 0.0826563 nm -> 0.0826563 nm
    """
    value = float(value)

    if value <= 0:
        raise ValueError("Wavelength must be > 0.")

    if value < 1e-6:
        return value * 1e9

    if value >= 0.5:
        return value * 0.1

    return value

# ------------------------------------------------------------
# q in nm⁻¹ to 2θ in degrees
# ------------------------------------------------------------
def q_nm_to_two_theta_deg(q_nm, wavelength_value):
    """Convert q in nm⁻¹ to 2θ in degrees using the wavelength field value."""
    wavelength_nm = wavelength_to_nm(wavelength_value)
    argument = np.asarray(q_nm, dtype=np.float64) * wavelength_nm / (4.0 * np.pi)
    argument = np.clip(argument, -1.0, 1.0)
    return np.degrees(2.0 * np.arcsin(argument))

# ------------------------------------------------------------
# q geometry diagnostics
# ------------------------------------------------------------

def q_geometry_diagnostics(image, xc, yc, distance_m, pixel_x_mm, pixel_y_mm, wavelength_value):
    """Return useful geometry diagnostics for q calibration checks."""
    ny, nx = image.shape
    wavelength_angstrom = float(wavelength_value)
    wavelength_nm = wavelength_angstrom * 0.1

    corners = np.array([
        [0, 0],
        [nx - 1, 0],
        [0, ny - 1],
        [nx - 1, ny - 1],
    ], dtype=float)

    dx_px = corners[:, 0] - xc
    dy_px = corners[:, 1] - yc
    dx_m = dx_px * pixel_x_mm * 1e-3
    dy_m = dy_px * pixel_y_mm * 1e-3
    r_m = np.sqrt(dx_m ** 2 + dy_m ** 2)
    two_theta = np.arctan2(r_m, distance_m)
    q_corners_angstrom = (4.0 * np.pi / wavelength_angstrom) * np.sin(two_theta / 2.0)
    q_corners = q_corners_angstrom * 10.0

    q_per_pixel_x = (
        (4.0 * np.pi / wavelength_angstrom)
        * np.sin(np.arctan2(pixel_x_mm * 1e-3, distance_m) / 2.0)
        * 10.0
    )

    q_per_pixel_y = (
        (4.0 * np.pi / wavelength_angstrom)
        * np.sin(np.arctan2(pixel_y_mm * 1e-3, distance_m) / 2.0)
        * 10.0
    )

    return {
        "image_shape": f"{ny} x {nx}",
        "center": f"({xc:.6g}, {yc:.6g}) px",
        "distance_m": distance_m,
        "pixel_x_mm": pixel_x_mm,
        "pixel_y_mm": pixel_y_mm,
        "wavelength_input": wavelength_value,
        "wavelength_nm": wavelength_nm,
        "q_per_pixel_x": q_per_pixel_x,
        "q_per_pixel_y": q_per_pixel_y,
        "q_corner_min": float(np.nanmin(q_corners)),
        "q_corner_max": float(np.nanmax(q_corners)),
    }


# ============================================================
# ======================= RADIAL TOOLS ========================
# ============================================================

def radial_average(
    image,
    xc,
    yc,
    distance_m,
    pixel_x_mm,
    pixel_y_mm,
    wavelength_a,
    q_min,
    q_max,
    n_bins,
    log_bins,
    sector_min=0,
    sector_max=360,
):
    """
    Clean radial integration I(q).

    Principle:
    - q = 0 at the beam centre.
    - q is calculated from detector geometry.
    - The intensity is the arithmetic mean of valid finite pixels inside each q bin.
    - NaN, Inf, negative values and detector-gap values >= 4e9 are excluded.
    """
    if distance_m <= 0:
        raise ValueError("Detector distance must be > 0.")
    if pixel_x_mm <= 0 or pixel_y_mm <= 0:
        raise ValueError("Pixel size must be > 0.")
    if wavelength_a <= 0:
        raise ValueError("Wavelength must be > 0.")
    if n_bins < 2:
        raise ValueError("Number of bins must be at least 2.")

    img = image.astype(np.float64)
    ny, nx = img.shape
    y, x = np.indices(img.shape)

    dx_px = x - float(xc)
    dy_px = y - float(yc)

    dx_m = dx_px * float(pixel_x_mm) * 1e-3
    dy_m = dy_px * float(pixel_y_mm) * 1e-3
    r_m = np.sqrt(dx_m ** 2 + dy_m ** 2)
    two_theta = np.arctan2(r_m, float(distance_m))
    wavelength_nm = wavelength_to_nm(float(wavelength_a))
    q = (4.0 * np.pi / wavelength_nm) * np.sin(two_theta / 2.0)

    psi = (np.degrees(np.arctan2(dy_px, dx_px)) + 360.0) % 360.0
    sector_min = sector_min % 360.0
    sector_max = sector_max % 360.0

    if abs((sector_max - sector_min) % 360.0) < 1e-9:
        sector_mask = np.ones_like(psi, dtype=bool)
    elif sector_min <= sector_max:
        sector_mask = (psi >= sector_min) & (psi <= sector_max)
    else:
        sector_mask = (psi >= sector_min) | (psi <= sector_max)

    intensity_valid = np.isfinite(img) & (img < 4e9) & (img > 0)
    geometry_valid = np.isfinite(q) & (q > 0) & sector_mask
    valid = geometry_valid & intensity_valid
    weights = img

    if q_min > 0:
        valid &= q >= q_min
    if q_max > 0:
        valid &= q <= q_max

    q_values = q[valid]
    i_values = weights[valid]

    if q_values.size == 0:
        raise ValueError("No valid pixel found in the selected q range / sector.")

    q_min_eff = float(q_min) if q_min > 0 else float(np.nanmin(q_values))
    q_max_eff = float(q_max) if q_max > 0 else float(np.nanmax(q_values))

    if q_max_eff <= q_min_eff:
        raise ValueError("q max must be greater than q min.")

    if log_bins:
        if q_min_eff <= 0:
            q_min_eff = float(np.nanmin(q_values[q_values > 0]))
        edges = np.logspace(np.log10(q_min_eff), np.log10(q_max_eff), int(n_bins) + 1)
        q_axis = np.sqrt(edges[:-1] * edges[1:])
    else:
        edges = np.linspace(q_min_eff, q_max_eff, int(n_bins) + 1)
        q_axis = 0.5 * (edges[:-1] + edges[1:])

    sums, _ = np.histogram(q_values, bins=edges, weights=i_values)
    counts, _ = np.histogram(q_values, bins=edges)

    with np.errstate(invalid="ignore", divide="ignore"):
        intensity = sums / counts

    valid_bins = (counts > 0) & np.isfinite(intensity) & (intensity > 0)
    q_axis = q_axis[valid_bins]
    intensity = intensity[valid_bins]
    counts = counts[valid_bins]

    return q_axis, intensity, counts, valid

# ============================================================
# =========================== CANVAS ==========================
# ============================================================



class PlotCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.fig.subplots_adjust(left=0.12, right=0.98, top=0.92, bottom=0.20)

        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.grabGesture(Qt.PinchGesture)

        self._base_zoom = 1.12

    def event(self, event):
        if event.type() == QEvent.Gesture:
            return self._handle_gesture_event(event)

        if event.type() == QEvent.NativeGesture:
            return self._handle_native_gesture_event(event)

        return super().event(event)

    def wheelEvent(self, event):
        """
        Trackpad behavior on the radial graph:
        - two-finger scroll/pan moves the graph,
        - Ctrl/Command + wheel or pinch-like wheel zooms around the cursor.
        """
        modifiers = event.modifiers()
        is_zoom = bool(modifiers & Qt.ControlModifier) or bool(modifiers & Qt.MetaModifier)

        pixel_delta = event.pixelDelta()
        angle_delta = event.angleDelta()

        if is_zoom:
            delta_y = pixel_delta.y() if not pixel_delta.isNull() else angle_delta.y() / 8.0
            if delta_y == 0:
                event.accept()
                return

            scale = self._base_zoom if delta_y < 0 else 1.0 / self._base_zoom
            position = event.position()
            self._zoom_at_canvas_position(position.x(), position.y(), scale)
            event.accept()
            return

        dx = pixel_delta.x() if not pixel_delta.isNull() else angle_delta.x() / 8.0
        dy = pixel_delta.y() if not pixel_delta.isNull() else angle_delta.y() / 8.0
        self._pan_from_pixels(dx, dy)
        event.accept()

    def _handle_gesture_event(self, event):
        pinch = event.gesture(Qt.PinchGesture)
        if pinch is None:
            return False

        scale = pinch.scaleFactor()
        if scale and scale > 0:
            center = pinch.centerPoint()
            self._zoom_at_canvas_position(center.x(), center.y(), 1.0 / scale)

        event.accept()
        return True

    def _handle_native_gesture_event(self, event):
        gesture_type = event.gestureType()

        if gesture_type == Qt.ZoomNativeGesture:
            value = event.value()
            if value != 0:
                scale = 1.0 / (1.0 + value)
                position = event.position()
                self._zoom_at_canvas_position(position.x(), position.y(), scale)
            event.accept()
            return True

        if gesture_type == Qt.PanNativeGesture:
            value = event.value()
            self._pan_from_pixels(0, value * 120.0)
            event.accept()
            return True

        return False

    def _zoom_at_canvas_position(self, canvas_x, canvas_y, scale):
        if scale <= 0:
            return

        xdata, ydata = self._canvas_position_to_data(canvas_x, canvas_y)
        if xdata is None or ydata is None:
            return

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        new_xlim = self._scaled_limits(xlim, xdata, scale, self.ax.get_xscale())
        new_ylim = self._scaled_limits(ylim, ydata, scale, self.ax.get_yscale())

        if new_xlim is not None:
            self.ax.set_xlim(new_xlim)
        if new_ylim is not None:
            self.ax.set_ylim(new_ylim)

        self.draw_idle()

    def _pan_from_pixels(self, dx_pixels, dy_pixels):
        if dx_pixels == 0 and dy_pixels == 0:
            return

        bbox = self.ax.bbox
        width = max(float(bbox.width), 1.0)
        height = max(float(bbox.height), 1.0)

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        new_xlim = self._shift_limits(xlim, -dx_pixels / width, self.ax.get_xscale())
        new_ylim = self._shift_limits(ylim, dy_pixels / height, self.ax.get_yscale())

        if new_xlim is not None:
            self.ax.set_xlim(new_xlim)
        if new_ylim is not None:
            self.ax.set_ylim(new_ylim)

        self.draw_idle()

    def _canvas_position_to_data(self, canvas_x, canvas_y):
        height = self.height()
        display_y = height - canvas_y
        try:
            xdata, ydata = self.ax.transData.inverted().transform((canvas_x, display_y))
        except Exception:
            return None, None

        if not np.isfinite(xdata) or not np.isfinite(ydata):
            return None, None

        return float(xdata), float(ydata)

    def _scaled_limits(self, limits, center, scale, axis_scale):
        low, high = float(limits[0]), float(limits[1])
        center = float(center)

        if axis_scale == "log":
            if low <= 0 or high <= 0 or center <= 0:
                return None
            log_low = np.log10(low)
            log_high = np.log10(high)
            log_center = np.log10(center)
            new_low = log_center + (log_low - log_center) * scale
            new_high = log_center + (log_high - log_center) * scale
            return 10 ** new_low, 10 ** new_high

        new_low = center + (low - center) * scale
        new_high = center + (high - center) * scale
        return new_low, new_high

    def _shift_limits(self, limits, fraction, axis_scale):
        low, high = float(limits[0]), float(limits[1])

        if axis_scale == "log":
            if low <= 0 or high <= 0:
                return None
            log_low = np.log10(low)
            log_high = np.log10(high)
            span = log_high - log_low
            shift = span * fraction
            return 10 ** (log_low + shift), 10 ** (log_high + shift)

        span = high - low
        shift = span * fraction
        return low + shift, high + shift


# ======================= IMAGE CANVAS =======================

class ImageCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.ax.set_axis_on()
        self.fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

        self._dragging = False
        self._drag_start = None
        self._xlim_start = None
        self._ylim_start = None
        self._base_scale = 1.18
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.grabGesture(Qt.PinchGesture)
        self.raw_image = None
        self.coordinate_label = None
        self.display_vmin = None
        self.display_vmax = None
        self.q_map = None

        self.mpl_connect("scroll_event", self._on_scroll)
        self.mpl_connect("button_press_event", self._on_press)
        self.mpl_connect("button_release_event", self._on_release)
        self.mpl_connect("motion_notify_event", self._on_motion)


    def event(self, event):
        if event.type() == QEvent.Gesture:
            return self._handle_gesture_event(event)

        if event.type() == QEvent.NativeGesture:
            return self._handle_native_gesture_event(event)

        return super().event(event)

    def wheelEvent(self, event):
        pixel_delta = event.pixelDelta()
        angle_delta = event.angleDelta()
        modifiers = event.modifiers()

        is_zoom = bool(modifiers & Qt.ControlModifier) or bool(modifiers & Qt.MetaModifier)

        if is_zoom:
            delta_y = pixel_delta.y() if not pixel_delta.isNull() else angle_delta.y() / 8.0

            if delta_y > 0:
                scale_factor = 1 / self._base_scale
            elif delta_y < 0:
                scale_factor = self._base_scale
            else:
                return

            canvas_pos = event.position()
            self._zoom_at_canvas_position(canvas_pos.x(), canvas_pos.y(), scale_factor)
            event.accept()
            return

        dx = pixel_delta.x() if not pixel_delta.isNull() else angle_delta.x() / 8.0
        dy = pixel_delta.y() if not pixel_delta.isNull() else angle_delta.y() / 8.0
        self._pan_from_pixels(dx, dy)
        event.accept()

    def _handle_gesture_event(self, event):
        pinch = event.gesture(Qt.PinchGesture)
        if pinch is None:
            return False

        scale = pinch.scaleFactor()
        if scale and scale > 0:
            center = pinch.centerPoint()
            self._zoom_at_canvas_position(center.x(), center.y(), 1.0 / scale)

        event.accept()
        return True

    def _handle_native_gesture_event(self, event):
        gesture_type = event.gestureType()

        if gesture_type == Qt.ZoomNativeGesture:
            value = event.value()
            if value != 0:
                scale = 1.0 / (1.0 + value)
                position = event.position()
                self._zoom_at_canvas_position(position.x(), position.y(), scale)
            event.accept()
            return True

        if gesture_type == Qt.PanNativeGesture:
            value = event.value()
            self._pan_from_pixels(0, value * 120.0)
            event.accept()
            return True

        return False

    def _zoom_at_canvas_position(self, canvas_x, canvas_y, scale_factor):
        if scale_factor <= 0:
            return

        height = self.height()
        display_y = height - canvas_y

        try:
            xdata, ydata = self.ax.transData.inverted().transform((canvas_x, display_y))
        except Exception:
            return

        if not np.isfinite(xdata) or not np.isfinite(ydata):
            return

        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

        self.ax.set_xlim([
            xdata - new_width * (1 - relx),
            xdata + new_width * relx,
        ])

        self.ax.set_ylim([
            ydata - new_height * (1 - rely),
            ydata + new_height * rely,
        ])

        self.draw_idle()

    def _pan_from_pixels(self, dx_pixels, dy_pixels):
        if dx_pixels == 0 and dy_pixels == 0:
            return

        bbox = self.ax.bbox
        width = max(float(bbox.width), 1.0)
        height = max(float(bbox.height), 1.0)

        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()

        dx_data = (cur_xlim[1] - cur_xlim[0]) * dx_pixels / width
        dy_data = (cur_ylim[1] - cur_ylim[0]) * dy_pixels / height

        self.ax.set_xlim(cur_xlim[0] - dx_data, cur_xlim[1] - dx_data)
        self.ax.set_ylim(cur_ylim[0] + dy_data, cur_ylim[1] + dy_data)

        self.draw_idle()

    def set_coordinate_label(self, label):
        self.coordinate_label = label

    def reset_display_limits(self):
        self.display_vmin = None
        self.display_vmax = None

    def set_q_map(self, q_map):
        self.q_map = q_map

    def _on_scroll(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()

        xdata = event.xdata
        ydata = event.ydata

        if event.button == "up":
            scale_factor = 1 / self._base_scale
        elif event.button == "down":
            scale_factor = self._base_scale
        else:
            return

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

        self.ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
        self.ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])
        self.draw_idle()

    def _on_press(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return

        self._dragging = True
        self._drag_start = (event.xdata, event.ydata)
        self._xlim_start = self.ax.get_xlim()
        self._ylim_start = self.ax.get_ylim()

    def _on_release(self, event):
        self._dragging = False
        self._drag_start = None
        self._xlim_start = None
        self._ylim_start = None

    def _on_motion(self, event):
        if self.coordinate_label is not None:
            if event.inaxes == self.ax and event.xdata is not None and event.ydata is not None:
                x_index = int(round(event.xdata))
                y_index = int(round(event.ydata))
                value_text = "-"
                q_text = "-"

                if self.raw_image is not None:
                    ny, nx = self.raw_image.shape
                    if 0 <= x_index < nx and 0 <= y_index < ny:
                        value = self.raw_image[y_index, x_index]
                        if np.isnan(value):
                            value_text = "NaN"
                        elif np.isposinf(value):
                            value_text = "+Inf"
                        elif np.isneginf(value):
                            value_text = "-Inf"
                        else:
                            value_text = f"{value:.8g}"

                        if self.q_map is not None:
                            q_value = self.q_map[y_index, x_index]
                            if np.isfinite(q_value):
                                q_text = f"{q_value:.6g} nm⁻¹"

                self.coordinate_label.setText(
                    f"x = {x_index + 1} | y = {y_index + 1}\nq = {q_text} | I = {value_text}"
                )
            else:
                self.coordinate_label.setText("x = - | y = -\nq = - | I = -")

        if not self._dragging or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None or self._drag_start is None:
            return

        dx = event.xdata - self._drag_start[0]
        dy = event.ydata - self._drag_start[1]

        self.ax.set_xlim(self._xlim_start[0] - dx, self._xlim_start[1] - dx)
        self.ax.set_ylim(self._ylim_start[0] - dy, self._ylim_start[1] - dy)
        self.draw_idle()

    def show_image(self, image, xc=None, yc=None, mask=None):
        current_xlim = self.ax.get_xlim()
        current_ylim = self.ax.get_ylim()
        had_image = len(self.ax.images) > 0
        self.raw_image = image

        self.ax.clear()
        self.ax.set_axis_on()

        display = image.astype(np.float64).copy()
        display[~np.isfinite(display)] = np.nan
        display[display < 0] = np.nan

        with np.errstate(invalid="ignore", divide="ignore"):
            display = np.log10(display + 1)

        if self.display_vmin is None or self.display_vmax is None:
            finite_display = display[np.isfinite(display)]
            if finite_display.size > 0:
                self.display_vmin = float(np.nanpercentile(finite_display, 1))
                self.display_vmax = float(np.nanpercentile(finite_display, 99))
            else:
                self.display_vmin = None
                self.display_vmax = None

        self.ax.imshow(
            display,
            origin="upper",
            cmap="jet",
            interpolation="nearest",
            vmin=self.display_vmin,
            vmax=self.display_vmax,
        )

        if mask is not None:
            overlay = np.zeros((*mask.shape, 4), dtype=float)
            overlay[~mask, :] = [0.55, 0.55, 0.55, 0.65]
            self.ax.imshow(overlay, origin="upper", interpolation="nearest")

        if xc is not None and yc is not None:
            self.ax.axvline(xc, color="red", linewidth=1.0)
            self.ax.axhline(yc, color="red", linewidth=1.0)
            self.ax.plot(xc, yc, "wo", markersize=4)

            ny, nx = image.shape
            radius = min(nx, ny) * 0.35
            angle_marks = [0, 90, 180, 270]
            for angle in angle_marks:
                rad = np.deg2rad(angle)
                x_text = xc + radius * np.cos(rad)
                y_text = yc + radius * np.sin(rad)
                self.ax.text(
                    x_text,
                    y_text,
                    f"{angle}°",
                    color="white",
                    fontsize=10,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=2),
                )

        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_xlabel("")
        self.ax.set_ylabel("")
        self.ax.set_aspect("equal")
        if had_image:
            self.ax.set_xlim(current_xlim)
            self.ax.set_ylim(current_ylim)
        else:
            ny, nx = image.shape
            self.ax.set_xlim(-0.5, nx - 0.5)
            self.ax.set_ylim(ny - 0.5, -0.5)

        self.draw_idle()


# ============================================================
# ========================== RADIAL TAB =======================
# ============================================================

class RadialTab(QWidget):
    """Radial tab: radial integration I(q) and Kratky plot."""

    folder_changed = Signal(Path)

    def __init__(self):
        super().__init__()

        self.current_folder = Path("/Users/nathanpiaget/Documents/Thèse LRP/Expériences/XENOCS")
        self.current_files = []
        self.instrument_mode = "XENOCS"
        self.last_results = {}
        self.h5_dataset_name = None
        self.h5_frame_axis = None
        self.h5_n_frames = 1
        self._syncing_folder = False
        self._changing_h5_frame = False
        self._syncing_frame_controls = False

        self.build_ui()
        self.refresh_files()
        self.set_controls_enabled(False)

    def build_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(8)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setFixedWidth(330)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)
        left_scroll.setWidget(left_panel)
        main_layout.addWidget(left_scroll, stretch=0)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)
        main_layout.addWidget(right_panel, stretch=1)

        view_row = QHBoxLayout()
        view_row.setContentsMargins(0, 0, 0, 0)
        view_row.setSpacing(8)
        right_layout.addLayout(view_row, stretch=1)

        graph_box = QGroupBox("I(q) graph")
        graph_layout = QVBoxLayout(graph_box)
        graph_layout.setContentsMargins(6, 18, 6, 6)
        view_row.addWidget(graph_box, stretch=3)

        image_box = QGroupBox("Image / selected integration area")
        image_layout = QVBoxLayout(image_box)
        image_layout.setContentsMargins(6, 18, 6, 6)
        image_box.setMinimumWidth(320)
        image_box.setMaximumWidth(430)
        view_row.addWidget(image_box, stretch=1)

        file_box = QGroupBox("File browser")
        file_layout = QVBoxLayout(file_box)
        file_layout.setContentsMargins(8, 18, 8, 8)
        file_layout.setSpacing(6)
        left_layout.addWidget(file_box, stretch=0)
        file_box.setFixedHeight(285)

        self.folder_path = QLineEdit(str(self.current_folder))
        file_layout.addWidget(self.folder_path)

        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.choose_folder)
        file_layout.addWidget(self.browse_button)

        filters_layout = QGridLayout()
        self.extensions_filter = QLineEdit("*.edf *.h5")
        self.name_filter = QLineEdit("**")
        self.extensions_filter.textChanged.connect(self.refresh_files)
        self.name_filter.textChanged.connect(self.refresh_files)
        filters_layout.addWidget(QLabel("Extensions:"), 0, 0)
        filters_layout.addWidget(self.extensions_filter, 0, 1)
        filters_layout.addWidget(QLabel("Name:"), 1, 0)
        filters_layout.addWidget(self.name_filter, 1, 1)
        file_layout.addLayout(filters_layout)

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_files)
        file_layout.addWidget(self.refresh_button)

        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.file_list.itemSelectionChanged.connect(self.selection_changed)
        file_layout.addWidget(self.file_list, stretch=1)

        params_box = QGroupBox("Radial parameters")
        params_layout = QVBoxLayout(params_box)
        params_layout.setContentsMargins(8, 18, 8, 8)
        params_layout.setSpacing(4)
        left_layout.addWidget(params_box)

        preset_layout = QHBoxLayout()
        self.btn_xenocs = QPushButton("XENOCS")
        self.btn_id02 = QPushButton("ID02")
        self.btn_id13 = QPushButton("ID13")
        self.btn_custom = QPushButton("Custom")
        for button in [self.btn_xenocs, self.btn_id02, self.btn_id13, self.btn_custom]:
            button.setCheckable(True)
            preset_layout.addWidget(button)
        self.btn_xenocs.setChecked(True)
        # params_layout.addWidget(QLabel("Instrument preset:"))
        params_layout.addLayout(preset_layout)

        form = QGridLayout()
        form.setVerticalSpacing(3)
        form.setHorizontalSpacing(6)
        self.center_x = self.double_spin(0, decimals=3)
        self.center_y = self.double_spin(0, decimals=3)
        self.distance = self.double_spin(0, decimals=6, minimum=0)
        self.pixel_x = self.double_spin(0.075000, decimals=6, minimum=0)
        self.pixel_y = self.double_spin(0.075000, decimals=6, minimum=0)
        self.wavelength = self.double_spin(0, decimals=6, minimum=0)
        self.use_q_range = QCheckBox("Use q range")
        self.use_q_range.setChecked(False)
        self.use_q_range.stateChanged.connect(self.update_mask_parameter_state)
        self.q_min = self.double_spin(0, decimals=6, minimum=0)
        self.q_max = self.double_spin(0, decimals=6, minimum=0)

        self.use_sector = QCheckBox("Use azimuthal sector")
        self.use_sector.setChecked(False)
        self.use_sector.stateChanged.connect(self.update_mask_parameter_state)
        self.sector_min = self.double_spin(0, decimals=3, minimum=-360)
        self.sector_max = self.double_spin(360, decimals=3, minimum=-360)
        self.n_bins = QSpinBox()
        self.n_bins.setRange(10, 10000)
        self.n_bins.setValue(300)
        self.plot_mode = QComboBox()
        self.plot_mode.addItems(["linear linear", "linear log", "log log", "log linear", "Kratky", "2θ linear", "2θ log"])
        self.plot_mode.setCurrentText("log log")
        self.plot_mode.currentTextChanged.connect(self.update_plot_mode)

        form.addWidget(QLabel("Centre X:"), 0, 0)
        form.addWidget(self.center_x, 0, 1)
        form.addWidget(QLabel("Centre Y:"), 1, 0)
        form.addWidget(self.center_y, 1, 1)
        form.addWidget(QLabel("Distance (m):"), 2, 0)
        form.addWidget(self.distance, 2, 1)
        form.addWidget(QLabel("Pixel X (mm):"), 3, 0)
        form.addWidget(self.pixel_x, 3, 1)
        form.addWidget(QLabel("Pixel Y (mm):"), 4, 0)
        form.addWidget(self.pixel_y, 4, 1)
        form.addWidget(QLabel("Wavelength (Å):"), 5, 0)
        form.addWidget(self.wavelength, 5, 1)

        self.frame_label = QLabel("H5 frame:")
        self.frame_spin = QSpinBox()
        self.frame_spin.setRange(1, 1)
        self.frame_spin.setValue(1)
        self.frame_spin.setEnabled(False)
        self.frame_label.hide()
        self.frame_spin.hide()
        self.frame_spin.valueChanged.connect(self.update_selected_h5_frame)

        form.addWidget(self.use_q_range, 7, 0, 1, 2)
        form.addWidget(QLabel("q min (nm⁻¹):"), 8, 0)
        form.addWidget(self.q_min, 8, 1)
        form.addWidget(QLabel("q max (nm⁻¹):"), 9, 0)
        form.addWidget(self.q_max, 9, 1)

        form.addWidget(self.use_sector, 10, 0, 1, 2)
        form.addWidget(QLabel("Sector min ψ (°):"), 11, 0)
        form.addWidget(self.sector_min, 11, 1)
        form.addWidget(QLabel("Sector max ψ (°):"), 12, 0)
        form.addWidget(self.sector_max, 12, 1)

        form.addWidget(QLabel("Bins:"), 13, 0)
        form.addWidget(self.n_bins, 13, 1)
        params_layout.addLayout(form)

        button_layout = QHBoxLayout()
        self.integrate_button = QPushButton("Integrate I(q)")
        self.integrate_button.clicked.connect(self.integrate_selected_files)
        self.save_button = QPushButton("Save .dat")
        self.save_button.clicked.connect(self.save_results)
        button_layout.addWidget(self.integrate_button)
        button_layout.addWidget(self.save_button)
        params_layout.addLayout(button_layout)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setVisible(False)

        self.canvas = PlotCanvas()
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setStyleSheet("""
            QToolBar {
                background: #f4f4f4;
                background-color: #f4f4f4;
                border: none;
                spacing: 6px;
                padding: 4px;
            }
            QToolButton {
                background: transparent;
                background-color: transparent;
            }
        """)

        toolbar_row = QHBoxLayout()
        toolbar_row.setContentsMargins(0, 0, 0, 0)
        toolbar_row.setSpacing(8)
        toolbar_row.addWidget(self.toolbar, stretch=1)
        toolbar_row.addWidget(self.plot_mode, stretch=0)
        graph_layout.addLayout(toolbar_row)

        self.graph_coordinate_label = QLabel("q = - | I = -")
        self.graph_coordinate_label.setMinimumHeight(26)
        self.graph_coordinate_label.setAlignment(Qt.AlignCenter)
        self.graph_coordinate_label.setStyleSheet("""
            QLabel {
                background-color: #f4f4f4;
                border-radius: 8px;
                padding: 4px;
                font-family: Menlo, Monaco, monospace;
                font-size: 10px;
            }
        """)

        graph_layout.addWidget(self.graph_coordinate_label, stretch=0)
        graph_layout.addWidget(self.canvas, stretch=1)

        self.image_canvas = ImageCanvas()
        self.image_coordinate_label = QLabel("x = - | y = -\nq = - | I = -")
        self.image_coordinate_label.setMinimumHeight(42)
        self.image_coordinate_label.setAlignment(Qt.AlignCenter)
        self.image_coordinate_label.setStyleSheet("""
            QLabel {
                background-color: #f4f4f4;
                border-radius: 8px;
                padding: 5px;
                font-family: Menlo, Monaco, monospace;
                font-size: 10px;
            }
        """)
        self.image_canvas.set_coordinate_label(self.image_coordinate_label)
        image_layout.addWidget(self.image_canvas, stretch=1)
        image_layout.addWidget(self.image_coordinate_label, stretch=0)

        self.canvas.mpl_connect("button_press_event", self.on_graph_right_click)
        self.canvas.mpl_connect("motion_notify_event", self.update_graph_coordinates)
        self.canvas.mpl_connect("axes_leave_event", self.clear_graph_coordinates)

        self.btn_xenocs.clicked.connect(lambda: self.set_instrument_mode("XENOCS"))
        self.btn_id02.clicked.connect(lambda: self.set_instrument_mode("ID02"))
        self.btn_id13.clicked.connect(lambda: self.set_instrument_mode("ID13"))
        self.btn_custom.clicked.connect(lambda: self.set_instrument_mode("Custom"))
        self.update_mask_parameter_state()

        frame_nav = QHBoxLayout()
        frame_nav.setContentsMargins(0, 0, 0, 0)
        frame_nav.setSpacing(6)
        self.frame_start_spin = QSpinBox()
        self.frame_start_spin.setRange(1, 1)
        self.frame_start_spin.setValue(1)
        self.frame_end_spin = QSpinBox()
        self.frame_end_spin.setRange(1, 1)
        self.frame_end_spin.setValue(1)
        self.prev_frame_button = QPushButton("<")
        self.next_frame_button = QPushButton(">")
        self.prev_frame_button.setFixedWidth(44)
        self.next_frame_button.setFixedWidth(44)
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(1, 1)
        self.frame_slider.setValue(1)
        self.frame_counter_label = QLabel("1 / 1")

        frame_nav.addWidget(QLabel("From:"))
        frame_nav.addWidget(self.frame_start_spin)
        frame_nav.addWidget(self.prev_frame_button)
        frame_nav.addWidget(self.frame_slider, stretch=1)
        frame_nav.addWidget(self.next_frame_button)
        frame_nav.addWidget(QLabel("To:"))
        frame_nav.addWidget(self.frame_end_spin)
        frame_nav.addWidget(self.frame_counter_label)
        right_layout.addLayout(frame_nav)

        self.frame_start_spin.valueChanged.connect(self.update_frame_bounds)
        self.frame_end_spin.valueChanged.connect(self.update_frame_bounds)
        self.frame_slider.valueChanged.connect(self.frame_slider_changed)
        self.prev_frame_button.clicked.connect(self.previous_frame)
        self.next_frame_button.clicked.connect(self.next_frame)

    def update_mask_parameter_state(self):
        use_q_range = self.use_q_range.isChecked()
        self.q_min.setEnabled(use_q_range)
        self.q_max.setEnabled(use_q_range)

        use_sector = self.use_sector.isChecked()
        self.sector_min.setEnabled(use_sector)
        self.sector_max.setEnabled(use_sector)

    def double_spin(self, value, decimals=3, minimum=-1e9):
        spin = QDoubleSpinBox()
        spin.setDecimals(decimals)
        spin.setRange(minimum, 1e12)
        spin.setValue(value)
        spin.setFixedHeight(24)
        return spin

    def set_controls_enabled(self, enabled):
        for widget in [
            self.btn_xenocs, self.btn_id02, self.btn_id13, self.btn_custom,
            self.center_x, self.center_y, self.distance, self.pixel_x, self.pixel_y,
            self.wavelength, self.frame_spin, self.frame_start_spin, self.frame_end_spin,
            self.frame_slider, self.prev_frame_button, self.next_frame_button,
            self.use_q_range, self.q_min, self.q_max, self.use_sector,
            self.n_bins, self.plot_mode, self.integrate_button, self.save_button,
        ]:
            widget.setEnabled(enabled)
        self.plot_mode.setCurrentText("log log")
        self.update_frame_selector_visibility()
        self.update_mask_parameter_state()

    def update_frame_selector_visibility(self):
        is_multiframe_h5 = self.h5_n_frames > 1
        self.frame_label.setVisible(False)
        self.frame_spin.setVisible(False)
        self.frame_spin.setEnabled(is_multiframe_h5)
        self.frame_start_spin.setVisible(is_multiframe_h5)
        self.frame_end_spin.setVisible(is_multiframe_h5)
        self.frame_slider.setVisible(is_multiframe_h5)
        self.prev_frame_button.setVisible(is_multiframe_h5)
        self.next_frame_button.setVisible(is_multiframe_h5)
        self.frame_counter_label.setVisible(is_multiframe_h5)
        self.update_frame_counter()

    def configure_frame_navigation(self, n_frames):
        n_frames = max(1, int(n_frames))
        self._syncing_frame_controls = True

        for widget in [self.frame_spin, self.frame_start_spin, self.frame_end_spin, self.frame_slider]:
            widget.blockSignals(True)

        self.frame_spin.setRange(1, n_frames)
        self.frame_spin.setValue(1)
        self.frame_start_spin.setRange(1, n_frames)
        self.frame_start_spin.setValue(1)
        self.frame_end_spin.setRange(1, n_frames)
        self.frame_end_spin.setValue(n_frames)
        self.frame_slider.setRange(1, n_frames)
        self.frame_slider.setValue(1)

        for widget in [self.frame_spin, self.frame_start_spin, self.frame_end_spin, self.frame_slider]:
            widget.blockSignals(False)

        self._syncing_frame_controls = False
        self.update_frame_counter()

    def frame_slider_changed(self, value):
        if self._syncing_frame_controls:
            return

        value = max(self.frame_start_spin.value(), min(int(value), self.frame_end_spin.value()))
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

        current = self.frame_spin.value()
        if current < start:
            self.frame_spin.setValue(start)
        elif current > end:
            self.frame_spin.setValue(end)
        else:
            self.update_frame_counter()

    def update_frame_counter(self):
        current = self.frame_spin.value()
        total = max(1, self.h5_n_frames)
        self.frame_counter_label.setText(f"{current} / {total}")
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(current)
        self.frame_slider.blockSignals(False)
        self.prev_frame_button.setEnabled(self.h5_n_frames > 1 and current > self.frame_start_spin.value())
        self.next_frame_button.setEnabled(self.h5_n_frames > 1 and current < self.frame_end_spin.value())

    def previous_frame(self):
        self.frame_spin.setValue(max(self.frame_start_spin.value(), self.frame_spin.value() - 1))

    def next_frame(self):
        self.frame_spin.setValue(min(self.frame_end_spin.value(), self.frame_spin.value() + 1))

    def update_selected_h5_frame(self):
        self.update_frame_counter()
        if self.h5_n_frames <= 1:
            return
        if not self.selected_files():
            return

        self._changing_h5_frame = True
        try:
            self.integrate_selected_files()
        finally:
            self._changing_h5_frame = False

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose folder", str(self.current_folder))
        if folder:
            self.current_folder = Path(folder)
            self.image_canvas.reset_display_limits()
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
            patterns = ["*.edf", "*.h5"]

        name_filter = self.name_filter.text().strip()
        if not name_filter:
            name_filter = "**"

        files = []
        for pattern in patterns:
            files.extend(folder.glob(pattern))

        from fnmatch import fnmatch
        files = sorted(set(files))
        files = [file for file in files if fnmatch(file.name, name_filter)]

        self.file_list.clear()
        for file in files:
            self.file_list.addItem(file.name)

        self.set_controls_enabled(bool(files))

    def selection_changed(self):
        selected = self.selected_files()
        self.image_canvas.reset_display_limits()
        self.set_controls_enabled(bool(selected))

        self.h5_dataset_name = None
        self.h5_frame_axis = None
        self.h5_n_frames = 1

        if selected:
            first_file = selected[0]
            if first_file.suffix.lower() in [".h5", ".hdf5"]:
                try:
                    dataset_name, dataset_shape, frame_axis, n_frames, header = inspect_h5_image_dataset(first_file)
                    self.h5_dataset_name = dataset_name
                    self.h5_frame_axis = frame_axis
                    self.h5_n_frames = n_frames

                    self.configure_frame_navigation(n_frames)
                except Exception as error:
                    QMessageBox.warning(self, "H5 inspection error", str(error))

            else:
                self.configure_frame_navigation(1)

            self.update_frame_selector_visibility()
            self.apply_preset_from_file(selected[0])
        else:
            self.update_frame_selector_visibility()

    def selected_files(self):
        return [self.current_folder / item.text() for item in self.file_list.selectedItems()]

    def set_instrument_mode(self, mode):
        self.instrument_mode = mode
        self.image_canvas.reset_display_limits()
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

        selected = self.selected_files()
        self.apply_preset_from_file(selected[0] if selected else None)

    def apply_preset_from_file(self, file_path=None):
        header = {}
        if file_path is not None and self.instrument_mode == "XENOCS":
            try:
                if file_path.suffix.lower() in [".h5", ".hdf5"]:
                    matching_edf = file_path.with_suffix(".edf")

                    if matching_edf.exists():
                        _, header = read_edf_file(matching_edf)
                        header["Parameter source"] = matching_edf.name
                    else:
                        _, header = read_image_file(file_path)
                else:
                    _, header = read_image_file(file_path)
            except Exception:
                header = {}

        if self.instrument_mode == "XENOCS":
            cx = get_header_float(header, "Center_1", "center_1")
            cy = get_header_float(header, "Center_2", "center_2")
            dist = get_header_float(header, "SampleDistance", "sample_distance")
            px = get_header_float(header, "PSize_1", "PSize_X", "PixelSizeX")
            py = get_header_float(header, "PSize_2", "PSize_Y", "PixelSizeY")
            wav = get_header_float(header, "WaveLength", "Wavelength", "wavelength")

            self.center_x.setValue(cx if cx is not None else 0)
            self.center_y.setValue(cy if cy is not None else 0)
            self.distance.setValue(dist if dist is not None else 0)
            self.pixel_x.setValue(px * 1000 if px is not None else 0.075000)
            self.pixel_y.setValue(py * 1000 if py is not None else 0.075000)
            if wav is not None:
                if wav < 1e-6:
                    self.wavelength.setValue(wav * 1e10)  # m -> Å for display
                elif wav < 0.5:
                    self.wavelength.setValue(wav * 10.0)  # nm -> Å for display
                else:
                    self.wavelength.setValue(wav)  # already Å
            else:
                self.wavelength.setValue(0)
            return

        if self.instrument_mode == "ID02":
            self.center_x.setValue(919.689)
            self.center_y.setValue(994.290)
            self.distance.setValue(1.0)
            self.pixel_x.setValue(0.075000)
            self.pixel_y.setValue(0.075000)
            self.wavelength.setValue(1.0)
            return

        if self.instrument_mode == "ID13":
            self.center_x.setValue(1294.689)
            self.center_y.setValue(1310.290)
            self.distance.setValue(0.8)
            self.pixel_x.setValue(0.075000)
            self.pixel_y.setValue(0.075000)
            self.wavelength.setValue(0.826563)
            return

    def integrate_selected_files(self):
        files = self.selected_files()
        if not files:
            return

        preserve_view = self._changing_h5_frame
        ax = self.canvas.ax

        previous_xlim = tuple(ax.get_xlim()) if preserve_view else None
        previous_ylim = tuple(ax.get_ylim()) if preserve_view else None
        previous_xscale = ax.get_xscale() if preserve_view else None
        previous_yscale = ax.get_yscale() if preserve_view else None

        self.last_results = {}
        ax.clear()

        messages = []
        for file_path in files:
            try:
                h5_dataset_name = self.h5_dataset_name if file_path.suffix.lower() in [".h5", ".hdf5"] else None
                h5_frame_index = self.frame_spin.value() - 1 if file_path.suffix.lower() in [".h5", ".hdf5"] else 0
                image, _ = read_image_file(file_path, h5_dataset_name, h5_frame_index)
                q_min = self.q_min.value() if self.use_q_range.isChecked() else 0
                q_max = self.q_max.value() if self.use_q_range.isChecked() else 0
                sector_min = self.sector_min.value() if self.use_sector.isChecked() else 0
                sector_max = self.sector_max.value() if self.use_sector.isChecked() else 360
                use_log_bins = self.plot_mode.currentText() in ["log log", "log linear", "Kratky"]
                wavelength_nm = wavelength_to_nm(self.wavelength.value())

                diagnostics = q_geometry_diagnostics(
                    image,
                    self.center_x.value(),
                    self.center_y.value(),
                    self.distance.value(),
                    self.pixel_x.value(),
                    self.pixel_y.value(),
                    self.wavelength.value(),
                )

                # --- q_map calculation ---
                ny, nx = image.shape
                yy, xx = np.indices(image.shape)

                dx_px = xx - float(self.center_x.value())
                dy_px = yy - float(self.center_y.value())

                dx_m = dx_px * float(self.pixel_x.value()) * 1e-3
                dy_m = dy_px * float(self.pixel_y.value()) * 1e-3

                r_m = np.sqrt(dx_m ** 2 + dy_m ** 2)
                two_theta_map = np.arctan2(r_m, float(self.distance.value()))
                wavelength_nm_map = wavelength_to_nm(float(self.wavelength.value()))
                q_map = (4.0 * np.pi / wavelength_nm_map) * np.sin(two_theta_map / 2.0)
                # --- end q_map calculation ---

                q, intensity, counts, mask = radial_average(
                    image,
                    self.center_x.value(),
                    self.center_y.value(),
                    self.distance.value(),
                    self.pixel_x.value(),
                    self.pixel_y.value(),
                    self.wavelength.value(),
                    q_min,
                    q_max,
                    self.n_bins.value(),
                    use_log_bins,
                    sector_min,
                    sector_max,
                )

                y = self.make_plot_y(q, intensity)
                x = self.make_plot_x(q)
                line, = ax.plot(x, y, linewidth=1.2, label=file_path.stem)
                self.last_results[file_path.stem] = (q, intensity, counts)

                if file_path == files[0]:
                    self.image_canvas.set_q_map(q_map)
                    self.image_canvas.show_image(image, self.center_x.value(), self.center_y.value(), mask=mask)
                frame_text = f" | H5 frame {self.frame_spin.value()} / {self.h5_n_frames}" if file_path.suffix.lower() in [".h5", ".hdf5"] and self.h5_n_frames > 1 else ""
                messages.append(
                    f"Integrated: {file_path.name}{frame_text} ({q.size} bins)\n"
                    f"  λ input/display = {diagnostics['wavelength_input']:.8g} Å ; λ used = {diagnostics['wavelength_nm']:.8g} nm\n"
                    f"  distance = {diagnostics['distance_m']:.8g} m ; pixel = {diagnostics['pixel_x_mm']:.8g} x {diagnostics['pixel_y_mm']:.8g} mm\n"
                    f"  centre = {diagnostics['center']} ; image = {diagnostics['image_shape']} px\n"
                    f"  q per pixel ≈ {diagnostics['q_per_pixel_x']:.8g} nm⁻¹/px ; q corner max ≈ {diagnostics['q_corner_max']:.8g} nm⁻¹\n"
                    f"  exported q range = {np.nanmin(q):.10g} -> {np.nanmax(q):.10g} nm⁻¹"
                    f" ; arithmetic mean ; invalid pixels excluded ; no smoothing"
                )

            except Exception as error:
                messages.append(f"Error: {file_path.name}: {error}")

        self.apply_plot_axes()

        if preserve_view and previous_xlim is not None and previous_ylim is not None:
            ax.set_autoscale_on(False)
            ax.set_xscale(previous_xscale)
            ax.set_yscale(previous_yscale)
            ax.set_xlim(previous_xlim[0], previous_xlim[1], auto=False)
            ax.set_ylim(previous_ylim[0], previous_ylim[1], auto=False)
        else:
            ax.set_autoscale_on(True)

        ax.grid(True)
        if self.last_results:
            self.legend = ax.legend(loc="best")
        self.canvas.draw_idle()
        self.log_box.setPlainText("\n".join(messages))


    def make_plot_x(self, q):
        mode = self.plot_mode.currentText()
        if mode in ["2θ linear", "2θ log"]:
            return q_nm_to_two_theta_deg(q, self.wavelength.value())
        return q


    def make_plot_y(self, q, intensity):
        if self.plot_mode.currentText() == "Kratky":
            return q ** 2 * intensity
        return intensity

    def apply_plot_axes(self):
        ax = self.canvas.ax
        mode = self.plot_mode.currentText()

        if mode in ["2θ linear", "2θ log"]:
            ax.set_xlabel("2θ / °")
        else:
            ax.set_xlabel("q / nm⁻¹")

        ax.xaxis.labelpad = 10
        ax.tick_params(axis="x", labelsize=9, pad=6)
        ax.set_ylabel("q²I(q)" if mode == "Kratky" else "I(q)")

        if mode == "linear linear":
            ax.set_xscale("linear")
            ax.set_yscale("linear")
        elif mode == "linear log":
            ax.set_xscale("linear")
            ax.set_yscale("log")
        elif mode == "log log":
            ax.set_xscale("log")
            ax.set_yscale("log")
        elif mode == "log linear":
            ax.set_xscale("log")
            ax.set_yscale("linear")
        elif mode == "Kratky":
            ax.set_xscale("log")
            ax.set_yscale("log")
        elif mode == "2θ linear":
            ax.set_xscale("linear")
            ax.set_yscale("linear")
        elif mode == "2θ log":
            ax.set_xscale("linear")
            ax.set_yscale("log")

    def update_plot_mode(self):
        # Kratky must be recomputed with logarithmic q bins, not only redrawn.
        # Otherwise the log-log display is based on linearly spaced bins and does not
        # match the expected SAXSutilities-style Kratky plot.
        if self.last_results and self.selected_files():
            self.integrate_selected_files()
            return

        ax = self.canvas.ax
        for line in ax.get_lines():
            label = line.get_label()
            if label in self.last_results:
                q, intensity, counts = self.last_results[label]
                line.set_xdata(self.make_plot_x(q))
                line.set_ydata(self.make_plot_y(q, intensity))

        self.apply_plot_axes()
        self.canvas.ax.relim()
        self.canvas.ax.autoscale_view()
        self.canvas.draw_idle()


    def update_graph_coordinates(self, event):
        if event.inaxes != self.canvas.ax or event.xdata is None or event.ydata is None:
            return

        try:
            self.graph_coordinate_label.setText(
                f"q = {event.xdata:.6g} | I = {event.ydata:.6g}"
            )
        except Exception:
            self.graph_coordinate_label.setText("q = - | I = -")

    def clear_graph_coordinates(self, event=None):
        self.graph_coordinate_label.setText("q = - | I = -")

    def on_graph_right_click(self, event):
        if event.button != 3 or event.inaxes != self.canvas.ax:
            return

        legend = self.canvas.ax.get_legend()
        if legend is None:
            return

        legend_lines = legend.get_lines()
        axis_lines = self.canvas.ax.get_lines()

        if not axis_lines:
            return

        labels = [line.get_label() for line in axis_lines if not line.get_label().startswith("_")]
        if not labels:
            return

        current = labels[0]
        new_label, ok = self.ask_text("Rename legend", "New legend label:", current)
        if not ok or not new_label.strip():
            return

        axis_lines[0].set_label(new_label.strip())
        self.legend = self.canvas.ax.legend(loc="best")
        self.canvas.draw_idle()

    def ask_text(self, title, label, text):
        from PySide6.QtWidgets import QInputDialog
        return QInputDialog.getText(self, title, label, text=text)

    def save_results(self):
        if not self.last_results:
            QMessageBox.warning(self, "No results", "No radial integration result to save.")
            return

        if self.use_q_range.isChecked():
            range_parts = [f"q{self.q_min.value():.4g}-{self.q_max.value():.4g}nm-1"]
        else:
            range_parts = ["qfull"]

        if self.use_sector.isChecked():
            range_parts.append(f"psi{self.sector_min.value():.3g}-{self.sector_max.value():.3g}deg")
        else:
            range_parts.append("psi360")

        range_suffix = "_" + "_".join(range_parts)

        for filename, (q, intensity, counts) in self.last_results.items():
            source_stem = Path(filename).stem
            frame_suffix = f"_frame{self.frame_spin.value():04d}" if self.h5_n_frames > 1 else ""
            out_file = self.current_folder / f"{source_stem}{frame_suffix}{range_suffix}_azimAvg.dat"
            data = np.column_stack([q, intensity, counts])
            with open(out_file, "w", encoding="utf-8") as file:
                file.write("# q_nm-1 I_q pixel_count\n")
                file.write("# averaging arithmetic_mean\n")
                file.write("# invalid_pixel_handling excluded\n")
                file.write("# smoothing none\n")
                np.savetxt(file, data, fmt="%.8e %.8e %d")

        QMessageBox.information(self, "Saved", "Radial profiles saved in the current folder.")
