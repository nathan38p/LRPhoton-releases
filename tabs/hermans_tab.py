
from pathlib import Path
import re

import h5py
import numpy as np

from PySide6.QtCore import Qt, Signal
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
    QGridLayout,
    QListWidget,
    QMessageBox,
    QSlider,
    QCheckBox,
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


# ============================================================
# =========================== TOOLS ===========================
# ============================================================

def wrap_to_180(angle):
    return (angle + 180) % 360 - 180


def read_azimuthal_file(file_path):
    file_path = Path(file_path)
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    text = text.replace(",", ".")

    data = []

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        if not any(char.isdigit() for char in line):
            continue

        for separator in [";", "\t", ","]:
            line = line.replace(separator, " ")

        parts = line.split()
        values = []

        for part in parts:
            try:
                values.append(float(part))
            except ValueError:
                pass

        if len(values) >= 2:
            data.append([values[0], values[1]])

    if not data:
        raise ValueError("No valid numerical data found in this file.")

    array = np.asarray(data, dtype=float)
    azimuth = array[:, 0]
    intensity = array[:, 1]

    valid = np.isfinite(azimuth) & np.isfinite(intensity)
    azimuth = azimuth[valid]
    intensity = intensity[valid]

    order = np.argsort(azimuth)
    return azimuth[order], intensity[order]


def parse_edf_header(header_text):
    header = {}
    for raw_line in header_text.replace("{", "").replace("}", "").split(";"):
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        header[key.strip()] = value.strip()
    return header


def read_edf_file(file_path):
    file_path = Path(file_path)
    with open(file_path, "rb") as file:
        content = file.read()

    header_end = content.find(b"}\n")
    if header_end < 0:
        header_end = content.find(b"}")
    if header_end < 0:
        raise ValueError("EDF header end not found.")

    header_bytes = content[:header_end + 1]
    header_text = header_bytes.decode("latin-1", errors="ignore")
    header = parse_edf_header(header_text)

    header_size = ((header_end + 1 + 511) // 512) * 512

    dim1 = int(float(header.get("Dim_1", header.get("dim_1"))))
    dim2 = int(float(header.get("Dim_2", header.get("dim_2"))))
    data_type = header.get("DataType", header.get("Data_Type", "UnsignedShort")).lower()

    if "unsignedshort" in data_type or "uint16" in data_type:
        dtype = np.uint16
    elif "signedshort" in data_type or "int16" in data_type:
        dtype = np.int16
    elif "unsignedlong" in data_type or "uint32" in data_type:
        dtype = np.uint32
    elif "signedlong" in data_type or "int32" in data_type:
        dtype = np.int32
    elif "float" in data_type:
        dtype = np.float32
    elif "double" in data_type:
        dtype = np.float64
    else:
        dtype = np.float32

    data = np.frombuffer(content[header_size:], dtype=dtype, count=dim1 * dim2)
    if data.size != dim1 * dim2:
        raise ValueError("EDF data size does not match header dimensions.")

    image = data.reshape((dim2, dim1)).astype(np.float64)
    image[image > 4e9] = np.nan
    return image, header


def find_h5_dataset(h5):
    datasets = []

    def collect(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.ndim >= 2:
            datasets.append(name)

    h5.visititems(collect)
    if not datasets:
        raise ValueError("No 2D or 3D dataset found in H5 file.")

    for name in datasets:
        lower = name.lower()
        if "data" in lower or "eiger" in lower or "detector" in lower:
            return name
    return datasets[0]


def read_h5_image(file_path, frame_index=0):
    with h5py.File(file_path, "r") as h5:
        dataset_name = find_h5_dataset(h5)
        dataset = h5[dataset_name]
        header = {"Dataset": dataset_name, "Shape": str(dataset.shape)}
        for key, value in dataset.attrs.items():
            header[key] = str(value)

        if dataset.ndim == 2:
            image = np.asarray(dataset[...], dtype=np.float64)
        elif dataset.ndim == 3:
            frame_axis = int(np.argmin(dataset.shape))
            n_frames = int(dataset.shape[frame_axis])
            frame_index = int(np.clip(frame_index, 0, n_frames - 1))
            if frame_axis == 0:
                image = np.asarray(dataset[frame_index, :, :], dtype=np.float64)
            elif frame_axis == 1:
                image = np.asarray(dataset[:, frame_index, :], dtype=np.float64)
            else:
                image = np.asarray(dataset[:, :, frame_index], dtype=np.float64)
            header["Frames"] = str(n_frames)
        else:
            raise ValueError("Only 2D and 3D H5 datasets are supported.")

    image[image > 4e9] = np.nan
    add_matching_edf_center(header, file_path)
    return image, header


def add_matching_edf_center(header, file_path):
    edf_path = Path(file_path).with_suffix(".edf")
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


def count_h5_frames(file_path):
    try:
        with h5py.File(file_path, "r") as h5:
            dataset = h5[find_h5_dataset(h5)]
            if dataset.ndim == 3:
                return int(np.min(dataset.shape))
    except Exception:
        pass
    return 1


def read_image_file(file_path, frame_index=0):
    suffix = Path(file_path).suffix.lower()
    if suffix == ".edf":
        return read_edf_file(file_path)
    if suffix in [".h5", ".hdf5"]:
        return read_h5_image(file_path, frame_index=frame_index)
    raise ValueError("Unsupported image file format.")


def header_float(header, keys, default):
    for key in keys:
        if key in header:
            try:
                match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(header[key]))
                if match:
                    return float(match.group(0))
            except Exception:
                pass
    return default


ID02_DEFAULT_CENTER_X = 914.4
ID02_DEFAULT_CENTER_Y = 996.5
ID02_DEFAULT_DISTANCE_M = 10.0002
ID02_DEFAULT_PIXEL_MM = 0.075
ID02_DEFAULT_WAVELENGTH_A = 1.01402
CENTER_X_KEYS = ["Center_1", "center_1", "CenterX", "center_x", "BeamCenterX", "Beam_x", "beam_x"]
CENTER_Y_KEYS = ["Center_2", "center_2", "CenterY", "center_y", "BeamCenterY", "Beam_y", "beam_y"]


def q_map_from_geometry(shape, xc, yc, distance_m, pixel_x_mm, pixel_y_mm, wavelength_a):
    y, x = np.indices(shape)
    dx_px = x + 1 - xc
    dy_px = y + 1 - yc
    dx_m = dx_px * pixel_x_mm * 1e-3
    dy_m = dy_px * pixel_y_mm * 1e-3
    r_m = np.sqrt(dx_m ** 2 + dy_m ** 2)
    two_theta = np.arctan2(r_m, distance_m)
    theta = two_theta / 2
    wavelength_nm = wavelength_a * 0.1
    return (4 * np.pi / wavelength_nm) * np.sin(theta)


def sector_iq_profiles(image, xc, yc, distance_m, pixel_x_mm, pixel_y_mm, wavelength_a, horizontal_ranges, vertical_ranges, n_bins=300, reference_angle=0.0):
    img = image.astype(np.float64)
    q = q_map_from_geometry(img.shape, xc, yc, distance_m, pixel_x_mm, pixel_y_mm, wavelength_a)

    y, x = np.indices(img.shape)
    dx_px = x + 1 - xc
    dy_px = y + 1 - yc
    psi = (np.degrees(np.arctan2(dy_px, dx_px)) - reference_angle + 360) % 360

    finite = np.isfinite(img) & np.isfinite(q)
    if not np.any(finite):
        raise ValueError("No valid image pixels for I(q) calculation.")

    q_values = q[finite]
    q_min = float(np.nanpercentile(q_values, 1))
    q_max = float(np.nanpercentile(q_values, 99))
    edges = np.linspace(q_min, q_max, n_bins + 1)
    q_centers = 0.5 * (edges[:-1] + edges[1:])

    def sector_mask(ranges):
        mask = np.zeros(img.shape, dtype=bool)
        for a0, a1 in ranges:
            a0 = a0 % 360
            a1 = a1 % 360
            if a0 <= a1:
                mask |= (psi >= a0) & (psi <= a1)
            else:
                mask |= (psi >= a0) | (psi <= a1)
        return mask & finite

    def integrate(mask):
        values = img[mask]
        q_sector = q[mask]
        bin_index = np.digitize(q_sector, edges) - 1
        valid = (bin_index >= 0) & (bin_index < n_bins) & np.isfinite(values)
        sums = np.bincount(bin_index[valid], weights=values[valid], minlength=n_bins)
        counts = np.bincount(bin_index[valid], minlength=n_bins)
        profile = np.full(n_bins, np.nan)
        ok = counts > 0
        profile[ok] = sums[ok] / counts[ok]
        return profile, counts

    h_mask = sector_mask(horizontal_ranges)
    v_mask = sector_mask(vertical_ranges)
    ih, h_counts = integrate(h_mask)
    iv, v_counts = integrate(v_mask)

    mask_overlay = h_mask | v_mask
    return q_centers, ih, iv, h_counts, v_counts, mask_overlay, h_mask, v_mask


def fit_gaussian_fixed_center(x, y, x0, window):
    if x.size < 10 or np.nanmax(y) <= 0:
        raise ValueError("Not enough valid points for Gaussian fit.")

    sigma_min = max(window / 200, 0.05)
    sigma_max = max(window * 2, sigma_min * 2)

    best = None

    for _ in range(4):
        sigmas = np.linspace(sigma_min, sigma_max, 240)

        for sigma in sigmas:
            g = np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))
            denom = np.sum(g ** 2)

            if denom <= 0:
                continue

            amplitude = np.sum(y * g) / denom
            amplitude = abs(amplitude)
            residual = amplitude * g - y
            sse = float(np.sum(residual ** 2))

            if best is None or sse < best[0]:
                best = (sse, amplitude, sigma)

        if best is None:
            raise ValueError("Gaussian fit failed.")

        best_sigma = best[2]
        span = (sigma_max - sigma_min) / 8
        sigma_min = max(best_sigma - span, 0.01)
        sigma_max = best_sigma + span

    return best[1], abs(best[2])


# ============================================================
# =========================== CANVAS ==========================
# ============================================================


class PlotCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.fig.subplots_adjust(left=0.12, right=0.98, top=0.94, bottom=0.20)

        self._dragging = False
        self._drag_start = None
        self._xlim_start = None
        self._ylim_start = None
        self._base_scale = 1.18

        self.mpl_connect("scroll_event", self._on_scroll)
        self.mpl_connect("button_press_event", self._on_press)
        self.mpl_connect("button_release_event", self._on_release)
        self.mpl_connect("motion_notify_event", self._on_motion)

    def _on_scroll(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        scale = 1 / self._base_scale if event.step > 0 else self._base_scale
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        x = event.xdata
        y = event.ydata

        if self.ax.get_xscale() == "log":
            if x <= 0 or xlim[0] <= 0 or xlim[1] <= 0:
                return
            log_x = np.log10(x)
            log_xlim = np.log10(xlim)
            new_log_xlim = [
                log_x - (log_x - log_xlim[0]) * scale,
                log_x + (log_xlim[1] - log_x) * scale,
            ]
            self.ax.set_xlim(10 ** new_log_xlim[0], 10 ** new_log_xlim[1])
        else:
            self.ax.set_xlim(
                x - (x - xlim[0]) * scale,
                x + (xlim[1] - x) * scale,
            )

        if self.ax.get_yscale() == "log":
            if y <= 0 or ylim[0] <= 0 or ylim[1] <= 0:
                return
            log_y = np.log10(y)
            log_ylim = np.log10(ylim)
            new_log_ylim = [
                log_y - (log_y - log_ylim[0]) * scale,
                log_y + (log_ylim[1] - log_y) * scale,
            ]
            self.ax.set_ylim(10 ** new_log_ylim[0], 10 ** new_log_ylim[1])
        else:
            self.ax.set_ylim(
                y - (y - ylim[0]) * scale,
                y + (ylim[1] - y) * scale,
            )

        self.draw_idle()

    def _on_press(self, event):
        if event.inaxes != self.ax or event.button not in [1, 2]:
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
        if not self._dragging or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None or self._drag_start is None:
            return

        dx = event.xdata - self._drag_start[0]
        dy = event.ydata - self._drag_start[1]

        self.ax.set_xlim(self._xlim_start[0] - dx, self._xlim_start[1] - dx)
        self.ax.set_ylim(self._ylim_start[0] - dy, self._ylim_start[1] - dy)
        self.draw_idle()


# ============================================================
# ======================== IMAGE CANVAS ======================
# ============================================================

class ImageCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.ax.set_axis_off()
        self.fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        self.raw_image = None
        self.q_map = None
        self.coordinate_label = None
        self._dragging = False
        self._drag_start = None
        self._xlim_start = None
        self._ylim_start = None
        self._base_scale = 1.18

        self.mpl_connect("scroll_event", self._on_scroll)
        self.mpl_connect("button_press_event", self._on_press)
        self.mpl_connect("button_release_event", self._on_release)
        self.mpl_connect("motion_notify_event", self._on_motion)

    def set_coordinate_label(self, label):
        self.coordinate_label = label

    def set_q_map(self, q_map):
        self.q_map = q_map

    def _on_scroll(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        scale = 1 / self._base_scale if event.step > 0 else self._base_scale
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        x = event.xdata
        y = event.ydata

        self.ax.set_xlim(
            x - (x - xlim[0]) * scale,
            x + (xlim[1] - x) * scale,
        )
        self.ax.set_ylim(
            y - (y - ylim[0]) * scale,
            y + (ylim[1] - y) * scale,
        )
        self.draw_idle()

    def _on_press(self, event):
        if event.inaxes != self.ax or event.button not in [1, 2]:
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
                    q_ny, q_nx = self.q_map.shape
                    if 0 <= x_index < q_nx and 0 <= y_index < q_ny:
                        q_value = self.q_map[y_index, x_index]
                        if np.isfinite(q_value):
                            q_text = f"{q_value:.6g}"

                self.coordinate_label.setText(
                    f"x = {x_index + 1} | y = {y_index + 1}\n"
                    f"q = {q_text} nm⁻¹ | I = {value_text}"
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

    def show_image(self, image, xc=None, yc=None, mask=None, h_mask=None, v_mask=None, reference_angle=0.0):
        previous_xlim = None
        previous_ylim = None
        preserve_view = self.raw_image is not None and self.raw_image.shape == image.shape

        if preserve_view:
            previous_xlim = self.ax.get_xlim()
            previous_ylim = self.ax.get_ylim()

        self.raw_image = image
        self.ax.clear()
        self.ax.set_axis_off()

        display = image.astype(np.float64).copy()
        display[~np.isfinite(display)] = np.nan
        display[display < 0] = np.nan

        with np.errstate(invalid="ignore", divide="ignore"):
            display = np.log10(display + 1)

        finite = display[np.isfinite(display)]
        if finite.size > 0:
            vmin = float(np.nanpercentile(finite, 1))
            vmax = float(np.nanpercentile(finite, 99))
        else:
            vmin = None
            vmax = None

        self.ax.imshow(
            display,
            origin="upper",
            cmap="jet",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )

        if h_mask is not None or v_mask is not None:
            selected = np.zeros(image.shape, dtype=bool)
            if h_mask is not None:
                selected |= h_mask
            if v_mask is not None:
                selected |= v_mask

            grey_overlay = np.zeros((*image.shape, 4), dtype=float)
            grey_overlay[:, :, :] = [0.30, 0.30, 0.30, 0.34]
            grey_overlay[selected, 3] = 0.0
            self.ax.imshow(grey_overlay, origin="upper", interpolation="nearest")

            color_overlay = np.zeros((*image.shape, 4), dtype=float)
            if h_mask is not None:
                color_overlay[h_mask, :] = [1.0, 0.10, 0.10, 0.48]
            if v_mask is not None:
                color_overlay[v_mask, :] = [0.10, 0.35, 1.0, 0.48]
            self.ax.imshow(color_overlay, origin="upper", interpolation="nearest")
        elif mask is not None:
            grey_overlay = np.zeros((*mask.shape, 4), dtype=float)
            grey_overlay[~mask, :] = [0.45, 0.45, 0.45, 0.58]
            self.ax.imshow(grey_overlay, origin="upper", interpolation="nearest")

        if xc is not None and yc is not None:
            x0 = xc - 1
            y0 = yc - 1
            ny, nx = image.shape
            radius = 0.55 * max(nx, ny)

            angle0 = np.deg2rad(reference_angle)
            angle90 = np.deg2rad(reference_angle + 90)

            self.ax.plot(
                [x0 - radius * np.cos(angle0), x0 + radius * np.cos(angle0)],
                [y0 - radius * np.sin(angle0), y0 + radius * np.sin(angle0)],
                color="red",
                linewidth=1.0,
            )
            self.ax.plot(
                [x0 - radius * np.cos(angle90), x0 + radius * np.cos(angle90)],
                [y0 - radius * np.sin(angle90), y0 + radius * np.sin(angle90)],
                color="blue",
                linewidth=1.0,
            )
            self.ax.plot(x0, y0, "wo", markersize=4)

            label_style = dict(
                fontsize=9,
                color="black",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#4a90e2", alpha=0.92),
            )

            label_radius = 0.48 * min(nx, ny)
            labels = [(0, "0°"), (90, "90°"), (180, "180°"), (270, "270°")]
            for angle_deg, text in labels:
                angle = np.deg2rad(reference_angle + angle_deg)
                x_label = x0 + label_radius * np.cos(angle)
                y_label = y0 + label_radius * np.sin(angle)
                x_label = min(max(x_label, 12), nx - 12)
                y_label = min(max(y_label, 12), ny - 12)
                self.ax.text(x_label, y_label, text, **label_style)

        self.ax.set_aspect("equal")

        if preserve_view and previous_xlim is not None and previous_ylim is not None:
            self.ax.set_xlim(previous_xlim)
            self.ax.set_ylim(previous_ylim)

        self.draw_idle()


# ============================================================
# ========================== HERMANS TAB ======================
# ============================================================

class HermansTab(QWidget):
    """Hermans tab: orientation factor, Pi and Gaussian fit from I(ψ) profiles."""
    folder_changed = Signal(Path)

    def __init__(self):
        super().__init__()

        self.folder = None
        self.available_files = []
        self.current_file = None
        self.azimuth = None
        self.intensity = None
        self.current_image = None
        self.current_header = {}
        self.last_fit = None
        self.last_anisotropy = None
        self.current_frame = 1
        self.total_frames = 1
        self._updating_frame_controls = False
        self.instrument_mode = "Custom"

        self.build_ui()
        self.init_default_folder()

    def build_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(8)

        left_panel = QWidget()
        controls_layout = QVBoxLayout(left_panel)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)
        main_layout.addWidget(left_panel, stretch=0)

        graph_wrapper = QWidget()
        graph_wrapper_layout = QVBoxLayout(graph_wrapper)
        graph_wrapper_layout.setContentsMargins(0, 0, 0, 0)
        graph_wrapper_layout.setSpacing(8)

        graph_content_layout = QHBoxLayout()
        graph_content_layout.setContentsMargins(0, 0, 0, 0)
        graph_content_layout.setSpacing(8)
        graph_wrapper_layout.addLayout(graph_content_layout, stretch=1)

        self.graph_box = QGroupBox("Azimuthal profile")
        graph_layout = QVBoxLayout(self.graph_box)
        graph_layout.setContentsMargins(6, 24, 6, 6)
        graph_layout.setSpacing(6)
        graph_content_layout.addWidget(self.graph_box, stretch=5)

        self.image_box = QGroupBox("Pattern")
        image_layout = QVBoxLayout(self.image_box)
        image_layout.setContentsMargins(6, 24, 6, 6)
        image_layout.setSpacing(6)
        self.image_box.setMinimumWidth(320)
        self.image_box.setMaximumWidth(430)
        graph_content_layout.addWidget(self.image_box, stretch=2)

        main_layout.addWidget(graph_wrapper, stretch=1)

        # --- Parameter selection box ---
        mode_box = QGroupBox("Parameter")
        mode_layout = QGridLayout(mode_box)
        mode_layout.setContentsMargins(8, 18, 8, 8)
        mode_layout.setSpacing(6)

        self.parameter_selector = QComboBox()
        self.parameter_selector.addItems([
            "Hermans factor",
            "Anisotropy factor (Iv - Ih) / (Iv + Ih)",
        ])
        self.parameter_selector.currentIndexChanged.connect(self.parameter_mode_changed)

        mode_layout.addWidget(QLabel("Parameter:"), 0, 0)
        mode_layout.addWidget(self.parameter_selector, 0, 1)
        mode_layout.setColumnStretch(1, 1)

        controls_layout.addWidget(mode_box, stretch=0)

        file_browser_box = QGroupBox("File browser")
        file_browser_layout = QVBoxLayout(file_browser_box)
        file_browser_layout.setContentsMargins(8, 18, 8, 8)
        file_browser_layout.setSpacing(6)

        self.open_folder_button = QPushButton("Open folder")
        self.open_folder_button.clicked.connect(self.open_folder)
        file_browser_layout.addWidget(self.open_folder_button)



        self.file_list = QListWidget()
        self.file_list.currentItemChanged.connect(self.load_selected_file)
        file_browser_layout.addWidget(self.file_list, stretch=1)

        controls_layout.addWidget(file_browser_box, stretch=1)
        file_browser_box.setMinimumHeight(320)

        results_box = QGroupBox("Results")
        results_layout = QVBoxLayout(results_box)
        results_layout.setContentsMargins(8, 18, 8, 8)
        results_layout.setSpacing(6)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)

        controls_layout.addWidget(results_box, stretch=0)
        results_box.setFixedHeight(190)

        params_box = QGroupBox("Parameters")
        params_layout = QGridLayout(params_box)
        params_layout.setContentsMargins(6, 18, 6, 6)
        params_layout.setSpacing(6)
        graph_wrapper_layout.addWidget(params_box, stretch=0)

        self.offset_spin, self.offset_slider = self.add_slider_control(
            params_layout, 0, "Baseline", 0.0, -1.0, 1.0
        )
        self.peak_spin, self.peak_slider = self.add_slider_control(
            params_layout, 1, "Peak ψ₀ (°)", 85.0, 70.0, 130.0
        )
        self.window_spin, self.window_slider = self.add_slider_control(
            params_layout, 2, "Window (°)", 90.0, 10.0, 180.0
        )

        self.use_fit_checkbox = QCheckBox("Fit")
        self.use_fit_checkbox.setChecked(True)
        self.use_fit_checkbox.stateChanged.connect(self.update_fit_mode)
        params_layout.addWidget(self.use_fit_checkbox, 3, 0, 1, 5)

        self.height_spin, self.height_slider = self.add_slider_control(
            params_layout, 4, "Height", 1.0, 0.0, 10.0
        )
        self.manual_fwhm_spin, self.manual_fwhm_slider = self.add_slider_control(
            params_layout, 5, "FWHM (°)", 90.0, 0.1, 180.0
        )

        self.fit_button = QPushButton("Fit")
        self.fit_button.clicked.connect(self.recenter_on_fit)
        self.save_fit_button = QPushButton("Save fit .dat")
        self.save_fit_button.clicked.connect(self.save_gaussian_fit)
        params_layout.addWidget(self.fit_button, 6, 0, 1, 2)
        params_layout.addWidget(self.save_fit_button, 6, 2, 1, 3)

        self.anisotropy_param_widgets = []

        h_box = QGroupBox("Horizontal sector")
        h_layout = QGridLayout(h_box)
        h_layout.setContentsMargins(8, 18, 8, 8)
        h_layout.setHorizontalSpacing(8)
        h_layout.setVerticalSpacing(6)

        v_box = QGroupBox("Vertical sector")
        reference_box = QGroupBox("Reference frame")
        reference_layout = QGridLayout(reference_box)
        reference_layout.setContentsMargins(8, 18, 8, 8)
        reference_layout.setHorizontalSpacing(8)
        reference_layout.setVerticalSpacing(6)
        v_layout = QGridLayout(v_box)
        v_layout.setContentsMargins(8, 18, 8, 8)
        v_layout.setHorizontalSpacing(8)
        v_layout.setVerticalSpacing(6)

        q_range_box = QGroupBox("q range")
        q_range_layout = QGridLayout(q_range_box)
        q_range_layout.setContentsMargins(8, 18, 8, 8)
        q_range_layout.setHorizontalSpacing(8)
        q_range_layout.setVerticalSpacing(6)

        def add_sector_spin(layout, row, label, value):
            label_widget = QLabel(label)
            spin = QDoubleSpinBox()
            spin.setDecimals(3)
            spin.setRange(0, 360)
            spin.setValue(value)
            spin.setSuffix(" °")
            spin.valueChanged.connect(self.calculate_anisotropy)
            layout.addWidget(label_widget, row, 0)
            layout.addWidget(spin, row, 1)
            layout.setColumnStretch(1, 1)
            self.anisotropy_param_widgets.extend([label_widget, spin])
            return spin

        self.h_psi_min = add_sector_spin(h_layout, 0, "ψ min", 350.0)
        self.h_psi_max = add_sector_spin(h_layout, 1, "ψ max", 10.0)
        self.v_psi_min = add_sector_spin(v_layout, 0, "ψ min", 80.0)
        self.v_psi_max = add_sector_spin(v_layout, 1, "ψ max", 100.0)

        h_box.setStyleSheet("""
            QGroupBox {
                background-color: #fff1f1;
                border: 1px solid #f0a0a0;
                border-radius: 8px;
                margin-top: 14px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0px 4px;
                color: #9b1111;
            }
        """)
        v_box.setStyleSheet("""
            QGroupBox {
                background-color: #eef4ff;
                border: 1px solid #9ebcff;
                border-radius: 8px;
                margin-top: 14px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0px 4px;
                color: #123f9a;
            }
        """)

        q_range_box = QGroupBox("q range")
        q_range_layout = QGridLayout(q_range_box)
        q_range_layout.setContentsMargins(8, 18, 8, 8)
        q_range_layout.setHorizontalSpacing(8)
        q_range_layout.setVerticalSpacing(6)

        self.use_q_range = QCheckBox("Use q range")
        self.use_q_range.setChecked(False)
        self.use_q_range.stateChanged.connect(self.calculate_anisotropy)

        self.q_min_filter = QDoubleSpinBox()
        self.q_min_filter.setDecimals(4)
        self.q_min_filter.setRange(0, 1000)
        self.q_min_filter.setValue(0.0)
        self.q_min_filter.setSuffix(" nm⁻¹")
        self.q_min_filter.valueChanged.connect(self.calculate_anisotropy)

        self.q_max_filter = QDoubleSpinBox()
        self.q_max_filter.setDecimals(4)
        self.q_max_filter.setRange(0, 1000)
        self.q_max_filter.setValue(10.0)
        self.q_max_filter.setSuffix(" nm⁻¹")
        self.q_max_filter.valueChanged.connect(self.calculate_anisotropy)

        q_range_layout.addWidget(QLabel("q min"), 0, 0)
        q_range_layout.addWidget(self.q_min_filter, 0, 1)
        q_range_layout.addWidget(QLabel("q max"), 1, 0)
        q_range_layout.addWidget(self.q_max_filter, 1, 1)
        q_range_layout.addWidget(self.use_q_range, 2, 0, 1, 2)
        q_range_layout.setColumnStretch(1, 1)

        self.reference_angle = QDoubleSpinBox()
        self.reference_angle.setDecimals(3)
        self.reference_angle.setRange(-180, 180)
        self.reference_angle.setValue(0.0)
        self.reference_angle.setSuffix(" °")
        self.reference_angle.valueChanged.connect(self.calculate_anisotropy)

        reference_layout.addWidget(QLabel("0° direction"), 0, 0)
        reference_layout.addWidget(self.reference_angle, 0, 1)
        reference_layout.setColumnStretch(1, 1)

        params_layout.addWidget(h_box, 7, 0, 3, 2)
        params_layout.addWidget(v_box, 7, 2, 3, 2)
        params_layout.addWidget(q_range_box, 7, 4, 3, 2)
        params_layout.addWidget(reference_box, 7, 6, 3, 2)
        params_layout.setColumnStretch(0, 1)
        params_layout.setColumnStretch(2, 1)
        params_layout.setColumnStretch(4, 1)
        params_layout.setColumnStretch(6, 1)

        self.anisotropy_param_widgets.extend([
            h_box, v_box, q_range_box, reference_box,
            self.use_q_range, self.q_min_filter, self.q_max_filter, self.reference_angle,
        ])

        frame_nav = QHBoxLayout()
        frame_nav.setContentsMargins(0, 0, 0, 0)
        frame_nav.setSpacing(6)

        self.frame_start_spin = QSpinBox()
        self.frame_start_spin.setRange(1, 1)
        self.frame_start_spin.setValue(1)
        self.frame_start_spin.setFixedWidth(70)

        self.frame_end_spin = QSpinBox()
        self.frame_end_spin.setRange(1, 1)
        self.frame_end_spin.setValue(1)
        self.frame_end_spin.setFixedWidth(70)

        self.prev_frame_button = QPushButton("<")
        self.next_frame_button = QPushButton(">")
        self.prev_frame_button.setFixedWidth(44)
        self.next_frame_button.setFixedWidth(44)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(1, 1)
        self.frame_slider.setValue(1)

        self.frame_counter_label = QLabel("1 / 1")
        self.frame_counter_label.setMinimumWidth(56)
        self.frame_counter_label.setAlignment(Qt.AlignCenter)

        frame_nav.addWidget(QLabel("From:"))
        frame_nav.addWidget(self.frame_start_spin)
        frame_nav.addWidget(self.prev_frame_button)
        frame_nav.addWidget(self.frame_slider, stretch=1)
        frame_nav.addWidget(self.next_frame_button)
        frame_nav.addWidget(QLabel("To:"))
        frame_nav.addWidget(self.frame_end_spin)
        frame_nav.addWidget(self.frame_counter_label)

        graph_wrapper_layout.addLayout(frame_nav)

        self.frame_start_spin.valueChanged.connect(self.update_frame_bounds)
        self.frame_end_spin.valueChanged.connect(self.update_frame_bounds)
        self.frame_slider.valueChanged.connect(self.frame_slider_changed)
        self.prev_frame_button.clicked.connect(self.previous_frame)
        self.next_frame_button.clicked.connect(self.next_frame)


        self.canvas = PlotCanvas()
        graph_layout.addWidget(self.canvas)

        anisotropy_graph_controls = QHBoxLayout()
        anisotropy_graph_controls.setContentsMargins(0, 0, 0, 0)
        anisotropy_graph_controls.setSpacing(6)

        self.anisotropy_plot_mode = QComboBox()
        self.anisotropy_plot_mode.addItems(["log-log", "lin-lin", "lin-log", "log-lin", "Kratky"])
        self.anisotropy_plot_mode.currentIndexChanged.connect(self.calculate_anisotropy)

        self.save_anisotropy_button = QPushButton("Save .dat")
        self.save_anisotropy_button.clicked.connect(self.save_anisotropy_profiles)

        self.anisotropy_plot_mode_label = QLabel("Scale")
        anisotropy_graph_controls.addWidget(self.anisotropy_plot_mode_label)
        anisotropy_graph_controls.addWidget(self.anisotropy_plot_mode)
        anisotropy_graph_controls.addStretch(1)
        anisotropy_graph_controls.addWidget(self.save_anisotropy_button)
        graph_layout.addLayout(anisotropy_graph_controls)

        self.anisotropy_param_widgets.extend([
            self.anisotropy_plot_mode_label,
            self.anisotropy_plot_mode,
            self.save_anisotropy_button,
        ])

        self.image_canvas = ImageCanvas()
        image_layout.addWidget(self.image_canvas, stretch=1)

        self.image_coordinate_label = QLabel("x = - | y = -\nq = - | I = -")
        self.image_coordinate_label.setAlignment(Qt.AlignCenter)
        self.image_coordinate_label.setStyleSheet("font-family: Menlo, monospace;")
        image_layout.addWidget(self.image_coordinate_label, stretch=0)
        self.image_canvas.set_coordinate_label(self.image_coordinate_label)

        # --- Instrument buttons and center controls ---
        instrument_layout = QGridLayout()
        instrument_layout.setContentsMargins(0, 0, 0, 0)
        instrument_layout.setHorizontalSpacing(6)
        instrument_layout.setVerticalSpacing(4)

        self.btn_xenocs = QPushButton("XENOCS")
        self.btn_id02 = QPushButton("ID02")
        self.btn_id13 = QPushButton("ID13")
        self.btn_custom = QPushButton("Custom")

        self.btn_xenocs.clicked.connect(lambda: self.set_instrument_mode("XENOCS"))
        self.btn_id02.clicked.connect(lambda: self.set_instrument_mode("ID02"))
        self.btn_id13.clicked.connect(lambda: self.set_instrument_mode("ID13"))
        self.btn_custom.clicked.connect(lambda: self.set_instrument_mode("Custom"))

        instrument_layout.addWidget(self.btn_xenocs, 0, 0)
        instrument_layout.addWidget(self.btn_id02, 0, 1)
        instrument_layout.addWidget(self.btn_id13, 0, 2)
        instrument_layout.addWidget(self.btn_custom, 0, 3)

        self.center_x_spin = QDoubleSpinBox()
        self.center_y_spin = QDoubleSpinBox()
        self.center_x_spin.setDecimals(3)
        self.center_y_spin.setDecimals(3)
        self.center_x_spin.setRange(-1e6, 1e6)
        self.center_y_spin.setRange(-1e6, 1e6)
        self.center_x_spin.valueChanged.connect(self.calculate_anisotropy)
        self.center_y_spin.valueChanged.connect(self.calculate_anisotropy)

        instrument_layout.addWidget(QLabel("Center X"), 1, 0)
        instrument_layout.addWidget(self.center_x_spin, 1, 1)
        instrument_layout.addWidget(QLabel("Center Y"), 1, 2)
        instrument_layout.addWidget(self.center_y_spin, 1, 3)

        image_layout.insertLayout(0, instrument_layout)

        for widget in [
            self.peak_spin,
            self.window_spin,
            self.offset_spin,
            self.height_spin,
            self.manual_fwhm_spin,
        ]:
            widget.valueChanged.connect(self.calculate)

        for slider in [
            self.peak_slider,
            self.window_slider,
            self.offset_slider,
            self.height_slider,
            self.manual_fwhm_slider,
        ]:
            slider.valueChanged.connect(self.slider_changed)

        self.update_fit_mode()
        self.parameter_mode_changed()
    def is_anisotropy_mode(self):
        return hasattr(self, "parameter_selector") and self.parameter_selector.currentIndex() == 1

    def parameter_mode_changed(self):
        anisotropy_mode = self.is_anisotropy_mode()

        self.graph_box.setTitle("I(q) profiles" if anisotropy_mode else "Azimuthal profile")
        self.image_box.setVisible(anisotropy_mode)

        hermans_widgets = [
            self.offset_slider.label_widget,
            self.peak_slider.label_widget,
            self.window_slider.label_widget,
            self.height_slider.label_widget,
            self.manual_fwhm_slider.label_widget,
            self.offset_spin, self.offset_slider, self.offset_slider.min_spin, self.offset_slider.max_spin,
            self.peak_spin, self.peak_slider, self.peak_slider.min_spin, self.peak_slider.max_spin,
            self.window_spin, self.window_slider, self.window_slider.min_spin, self.window_slider.max_spin,
            self.use_fit_checkbox, self.height_spin, self.height_slider,
            self.height_slider.min_spin, self.height_slider.max_spin,
            self.manual_fwhm_spin, self.manual_fwhm_slider,
            self.manual_fwhm_slider.min_spin, self.manual_fwhm_slider.max_spin,
            self.fit_button, self.save_fit_button,
        ]

        for widget in hermans_widgets:
            widget.setVisible(not anisotropy_mode)

        for widget in self.anisotropy_param_widgets:
            widget.setVisible(anisotropy_mode)

        for widget in [
            self.frame_start_spin, self.frame_end_spin, self.prev_frame_button,
            self.next_frame_button, self.frame_slider, self.frame_counter_label,
            self.btn_xenocs, self.btn_id02, self.btn_id13, self.btn_custom,
            self.center_x_spin, self.center_y_spin,
        ]:
            widget.setVisible(anisotropy_mode)

        if self.folder is not None:
            self.load_folder(self.folder)
        elif anisotropy_mode:
            self.calculate_anisotropy()
        else:
            self.calculate()

    def configure_frame_navigation(self, n_frames):
        self.total_frames = max(1, int(n_frames))
        self.current_frame = min(max(1, self.current_frame), self.total_frames)

        self._updating_frame_controls = True
        self.frame_start_spin.setRange(1, self.total_frames)
        self.frame_end_spin.setRange(1, self.total_frames)
        self.frame_slider.setRange(1, self.total_frames)
        self.frame_start_spin.setValue(1)
        self.frame_end_spin.setValue(self.total_frames)
        self.frame_slider.setValue(self.current_frame)
        self.frame_counter_label.setText(f"{self.current_frame} / {self.total_frames}")
        self._updating_frame_controls = False

    def update_frame_bounds(self):
        start = self.frame_start_spin.value()
        end = self.frame_end_spin.value()

        if start > end:
            if self.sender() == self.frame_start_spin:
                self.frame_end_spin.setValue(start)
                end = start
            else:
                self.frame_start_spin.setValue(end)
                start = end

        self.frame_slider.setRange(start, end)
        if self.frame_slider.value() < start:
            self.frame_slider.setValue(start)
        elif self.frame_slider.value() > end:
            self.frame_slider.setValue(end)

    def frame_slider_changed(self, value):
        self.current_frame = int(value)
        self.frame_counter_label.setText(f"{self.current_frame} / {self.total_frames}")

        if not self._updating_frame_controls and self.is_anisotropy_mode() and self.current_file is not None:
            self.load_file(self.current_file)

    def previous_frame(self):
        self.frame_slider.setValue(max(self.frame_slider.minimum(), self.frame_slider.value() - 1))

    def next_frame(self):
        self.frame_slider.setValue(min(self.frame_slider.maximum(), self.frame_slider.value() + 1))

    def set_instrument_mode(self, mode):
        self.instrument_mode = mode
        self.apply_instrument_preset()
        self.calculate_anisotropy()

    def set_center_spins(self, x_value, y_value):
        self.center_x_spin.blockSignals(True)
        self.center_y_spin.blockSignals(True)
        self.center_x_spin.setValue(float(x_value))
        self.center_y_spin.setValue(float(y_value))
        self.center_x_spin.blockSignals(False)
        self.center_y_spin.blockSignals(False)

    def apply_instrument_preset(self):
        if self.current_image is None:
            return

        ny, nx = self.current_image.shape
        header = self.current_header or {}

        if self.instrument_mode == "XENOCS":
            x = header_float(header, CENTER_X_KEYS, 612.0)
            y = header_float(header, CENTER_Y_KEYS, 649.0)
        elif self.instrument_mode == "ID02":
            x = header_float(header, CENTER_X_KEYS, ID02_DEFAULT_CENTER_X)
            y = header_float(header, CENTER_Y_KEYS, ID02_DEFAULT_CENTER_Y)
        elif self.instrument_mode == "ID13":
            x = header_float(header, CENTER_X_KEYS, ID13_DEFAULT_CENTER_X)
            y = header_float(header, CENTER_Y_KEYS, ID13_DEFAULT_CENTER_Y)
        else:
            x = self.center_x_spin.value() if self.center_x_spin.value() != 0 else header_float(header, CENTER_X_KEYS, nx / 2)
            y = self.center_y_spin.value() if self.center_y_spin.value() != 0 else header_float(header, CENTER_Y_KEYS, ny / 2)

        self.set_center_spins(x, y)

    def update_fit_mode(self):
        use_fit = self.use_fit_checkbox.isChecked()

        self.height_spin.setEnabled(not use_fit)
        self.height_slider.setEnabled(not use_fit)
        self.height_slider.min_spin.setEnabled(not use_fit)
        self.height_slider.max_spin.setEnabled(not use_fit)

        self.manual_fwhm_spin.setEnabled(not use_fit)
        self.manual_fwhm_slider.setEnabled(not use_fit)
        self.manual_fwhm_slider.min_spin.setEnabled(not use_fit)
        self.manual_fwhm_slider.max_spin.setEnabled(not use_fit)

        self.fit_button.setEnabled(use_fit)
        self.save_fit_button.setEnabled(True)

        self.calculate()

    def update_manual_fields_from_fit(self, amplitude, fwhm):
        def update_spin_and_slider(spin, slider, value):
            min_spin = slider.min_spin
            max_spin = slider.max_spin

            if value < min_spin.value():
                min_spin.blockSignals(True)
                min_spin.setValue(value)
                min_spin.blockSignals(False)

            if value > max_spin.value():
                max_spin.blockSignals(True)
                max_spin.setValue(value)
                max_spin.blockSignals(False)

            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)

            slider.blockSignals(True)
            slider.setValue(self.value_to_slider(value, min_spin.value(), max_spin.value()))
            slider.blockSignals(False)

        update_spin_and_slider(self.height_spin, self.height_slider, amplitude)
        update_spin_and_slider(self.manual_fwhm_spin, self.manual_fwhm_slider, fwhm)

    def add_slider_control(self, layout, row, label, default, minimum, maximum):
        label_widget = QLabel(label)

        min_spin = QDoubleSpinBox()
        min_spin.setDecimals(3)
        min_spin.setRange(-1e9, 1e9)
        min_spin.setValue(minimum)
        min_spin.setFixedWidth(75)

        max_spin = QDoubleSpinBox()
        max_spin.setDecimals(3)
        max_spin.setRange(-1e9, 1e9)
        max_spin.setValue(maximum)
        max_spin.setFixedWidth(75)

        value_spin = QDoubleSpinBox()
        value_spin.setDecimals(3)
        value_spin.setRange(minimum, maximum)
        value_spin.setValue(default)
        value_spin.setFixedWidth(75)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 1000)
        slider.setValue(self.value_to_slider(default, minimum, maximum))
        slider.setMinimumWidth(180)

        min_spin.valueChanged.connect(lambda: self.update_slider_limits(min_spin, max_spin, value_spin, slider))
        max_spin.valueChanged.connect(lambda: self.update_slider_limits(min_spin, max_spin, value_spin, slider))
        value_spin.valueChanged.connect(lambda: self.spin_changed(min_spin, max_spin, value_spin, slider))

        layout.addWidget(label_widget, row, 0)
        layout.addWidget(min_spin, row, 1)
        layout.addWidget(slider, row, 2)
        layout.addWidget(max_spin, row, 3)
        layout.addWidget(value_spin, row, 4)
        layout.setColumnStretch(2, 1)

        slider.min_spin = min_spin
        slider.max_spin = max_spin
        slider.value_spin = value_spin
        slider.label_widget = label_widget

        return value_spin, slider

    def value_to_slider(self, value, minimum, maximum):
        if maximum <= minimum:
            return 0
        ratio = (value - minimum) / (maximum - minimum)
        return int(round(max(0, min(1, ratio)) * 1000))

    def slider_to_value(self, slider):
        minimum = slider.min_spin.value()
        maximum = slider.max_spin.value()
        return minimum + (maximum - minimum) * slider.value() / 1000

    def slider_changed(self):
        slider = self.sender()
        slider.value_spin.blockSignals(True)
        slider.value_spin.setValue(self.slider_to_value(slider))
        slider.value_spin.blockSignals(False)
        self.calculate()

    def spin_changed(self, min_spin, max_spin, value_spin, slider):
        minimum = min_spin.value()
        maximum = max_spin.value()
        value = max(minimum, min(maximum, value_spin.value()))

        value_spin.blockSignals(True)
        value_spin.setValue(value)
        value_spin.blockSignals(False)

        slider.blockSignals(True)
        slider.setValue(self.value_to_slider(value, minimum, maximum))
        slider.blockSignals(False)

        self.calculate()

    def update_slider_limits(self, min_spin, max_spin, value_spin, slider):
        minimum = min_spin.value()
        maximum = max_spin.value()

        if maximum <= minimum:
            maximum = minimum + 1e-9
            max_spin.blockSignals(True)
            max_spin.setValue(maximum)
            max_spin.blockSignals(False)

        value_spin.setRange(minimum, maximum)
        self.spin_changed(min_spin, max_spin, value_spin, slider)

    def init_default_folder(self):
        folder = Path.cwd() / "2 AZIM SAXSUtilities"

        if folder.is_dir():
            self.load_folder(folder)
        else:
            self.current_file = None
            self.azimuth = None
            self.intensity = None
            self.last_fit = None
            self.file_list.clear()
            self.canvas.ax.clear()
            self.canvas.draw_idle()

    def set_folder_from_external_tab(self, folder):
        folder = Path(folder)
        if self.folder == folder:
            return
        self.load_folder(folder)

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Open azimuthal folder", "")
        if folder:
            self.load_folder(Path(folder))

    def load_folder(self, folder):
        self.folder = Path(folder)
        self.folder_changed.emit(self.folder)

        if self.is_anisotropy_mode():
            files = []
            for pattern in ["*.edf", "*.h5", "*.hdf5"]:
                files.extend(self.folder.glob(pattern))

            excluded_suffixes = ("_ave.h5", "_aveq_ave.h5", "polar.edf")
            unique_files = sorted(set(files))
            edf_stems = {file.stem for file in unique_files if file.suffix.lower() == ".edf"}

            self.available_files = []
            for file in unique_files:
                lower_name = file.name.lower()
                suffix = file.suffix.lower()

                if lower_name.endswith(excluded_suffixes):
                    continue

                if suffix in [".h5", ".hdf5"] and file.stem in edf_stems:
                    continue

                self.available_files.append(file.name)
        else:
            self.available_files = sorted(
                file.name for file in self.folder.glob("*_cave*_azimProf.dat")
            )

        self.file_list.blockSignals(True)
        self.file_list.clear()
        self.file_list.addItems(self.available_files)
        self.file_list.blockSignals(False)

        if self.available_files:
            self.file_list.setCurrentRow(0)
            self.load_file(self.folder / self.available_files[0])
        else:
            self.current_file = None
            self.azimuth = None
            self.intensity = None
            self.last_fit = None
            self.results_text.clear()
            self.canvas.ax.clear()
            self.canvas.draw_idle()

    

    def load_selected_file(self):
        item = self.file_list.currentItem()

        if item is None or self.folder is None:
            return

        self.load_file(self.folder / item.text())

    def load_file(self, file_path):
        self.current_file = Path(file_path)

        if self.is_anisotropy_mode():
            try:
                if self.current_file.suffix.lower() in [".h5", ".hdf5"]:
                    self.configure_frame_navigation(count_h5_frames(self.current_file))
                else:
                    self.configure_frame_navigation(1)

                image, header = read_image_file(self.current_file, frame_index=self.current_frame - 1)
            except Exception as error:
                QMessageBox.critical(self, "File reading error", str(error))
                return

            self.current_image = image
            self.current_header = header
            self.azimuth = None
            self.intensity = None
            self.last_fit = None
            self.apply_instrument_preset()
            self.calculate_anisotropy()
            return

        try:
            azimuth, intensity = read_azimuthal_file(file_path)
        except Exception as error:
            QMessageBox.critical(self, "File reading error", str(error))
            return

        self.configure_frame_navigation(1)
        self.azimuth = azimuth
        self.intensity = intensity
        self.current_image = None
        self.current_header = {}
        self.last_fit = None

        self.offset_spin.blockSignals(True)
        self.offset_spin.setValue(0)
        self.offset_spin.blockSignals(False)
        self.offset_slider.blockSignals(True)
        self.offset_slider.setValue(self.value_to_slider(0, self.offset_slider.min_spin.value(), self.offset_slider.max_spin.value()))
        self.offset_slider.blockSignals(False)

        self.calculate()

    def calculate(self):
        if self.is_anisotropy_mode():
            self.calculate_anisotropy()
            return

        if self.azimuth is None or self.intensity is None or self.current_file is None:
            return

        azimuth = self.azimuth
        intensity = self.intensity
        peak = self.peak_spin.value()
        window = self.window_spin.value()
        offset = self.offset_spin.value()

        mask = np.abs(wrap_to_180(azimuth - peak)) <= window / 2
        az_fit = azimuth[mask]

        if az_fit.size < 10:
            self.results_text.setPlainText("Not enough points in the fitting window.")
            return

        baseline_level = np.nanmin(intensity) + offset
        baseline = np.full_like(azimuth, baseline_level)
        corrected = intensity - baseline
        corrected[corrected < 0] = 0
        fit_intensity = corrected[mask]

        if np.nanmax(fit_intensity) <= 0:
            self.results_text.setPlainText(
                "Corrected signal is null or negative.\nAdjust the baseline."
            )
            return

        use_fit = self.use_fit_checkbox.isChecked()

        if use_fit:
            try:
                amplitude, sigma = fit_gaussian_fixed_center(az_fit, fit_intensity, peak, window)
            except Exception as error:
                self.results_text.setPlainText(str(error))
                return

            fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
            self.update_manual_fields_from_fit(amplitude, fwhm)
        else:
            amplitude = self.height_spin.value()
            fwhm = self.manual_fwhm_spin.value()
            sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

        pi = (180 - fwhm) / 180

        phi_deg = np.abs(wrap_to_180(azimuth - peak))
        phi_deg[phi_deg > 90] = 180 - phi_deg[phi_deg > 90]
        phi = np.deg2rad(phi_deg)

        den_raw = np.sum(intensity * np.sin(phi))
        num_raw = np.sum(intensity * np.sin(phi) * np.cos(phi) ** 2)
        cos2mean_raw = np.nan if den_raw == 0 else num_raw / den_raw
        hermans_raw = np.nan if den_raw == 0 else (3 * cos2mean_raw - 1) / 2

        den_corr = np.sum(corrected * np.sin(phi))
        num_corr = np.sum(corrected * np.sin(phi) * np.cos(phi) ** 2)
        cos2mean_corr = np.nan if den_corr == 0 else num_corr / den_corr
        hermans_corr = np.nan if den_corr == 0 else (3 * cos2mean_corr - 1) / 2

        self.last_fit = {
            "use_fit": use_fit,
            "peak": peak,
            "window": window,
            "offset": offset,
            "baseline": baseline_level,
            "amplitude": amplitude,
            "sigma": sigma,
            "fwhm": fwhm,
            "pi": pi,
            "cos2mean_corr": cos2mean_corr,
            "hermans_corr": hermans_corr,
            "cos2mean_raw": cos2mean_raw,
            "hermans_raw": hermans_raw,
        }

        self.update_plot(azimuth, intensity, baseline, peak, amplitude, sigma, fwhm, pi, hermans_corr)
        self.update_results_text()

    def calculate_anisotropy(self):
        if not self.is_anisotropy_mode():
            return

        if self.current_image is None or self.current_file is None:
            return

        image = self.current_image
        header = self.current_header or {}

        ny, nx = image.shape
        if self.center_x_spin.value() == 0 and self.center_y_spin.value() == 0:
            self.apply_instrument_preset()
        xc = self.center_x_spin.value()
        yc = self.center_y_spin.value()
        if self.instrument_mode == "ID02":
            default_distance = ID02_DEFAULT_DISTANCE_M
            default_pixel = ID02_DEFAULT_PIXEL_MM
            default_wavelength = ID02_DEFAULT_WAVELENGTH_A
        elif self.instrument_mode == "ID13":
            default_distance = ID13_DEFAULT_DISTANCE_M
            default_pixel = ID13_DEFAULT_PIXEL_MM
            default_wavelength = ID13_DEFAULT_WAVELENGTH_A
        else:
            default_distance = 0.9
            default_pixel = 0.075
            default_wavelength = 1.54189
        distance = header_float(header, ["SampleDistance", "sampledistance", "sample_distance", "Distance", "DetectorDistance"], default_distance)
        pixel_x = header_float(header, ["PSize_1", "psize_1", "PixelSizeX", "pixel_x"], default_pixel)
        pixel_y = header_float(header, ["PSize_2", "psize_2", "PixelSizeY", "pixel_y"], default_pixel)
        wavelength = header_float(header, ["Wavelength", "wavelength"], default_wavelength)

        if pixel_x < 1e-3:
            pixel_x *= 1000
        if pixel_y < 1e-3:
            pixel_y *= 1000
        if wavelength < 1e-6:
            wavelength *= 1e10

        horizontal_ranges = [(self.h_psi_min.value(), self.h_psi_max.value())]
        vertical_ranges = [(self.v_psi_min.value(), self.v_psi_max.value())]

        try:
            q, ih, iv, h_counts, v_counts, mask, h_mask, v_mask = sector_iq_profiles(
                image,
                xc,
                yc,
                distance,
                pixel_x,
                pixel_y,
                wavelength,
                horizontal_ranges,
                vertical_ranges,
                reference_angle=self.reference_angle.value(),
            )
        except Exception as error:
            self.results_text.setPlainText(str(error))
            return

        q_map = q_map_from_geometry(image.shape, xc, yc, distance, pixel_x, pixel_y, wavelength)
        self.image_canvas.set_q_map(q_map)

        display_h_mask = h_mask.copy()
        display_v_mask = v_mask.copy()

        valid = np.isfinite(ih) & np.isfinite(iv) & ((iv + ih) != 0)
        if self.use_q_range.isChecked():
            q_min = min(self.q_min_filter.value(), self.q_max_filter.value())
            q_max = max(self.q_min_filter.value(), self.q_max_filter.value())
            valid &= (q >= q_min) & (q <= q_max)

            q_range_image_mask = np.isfinite(q_map) & (q_map >= q_min) & (q_map <= q_max)
            display_h_mask &= q_range_image_mask
            display_v_mask &= q_range_image_mask
        anisotropy_curve = np.full_like(q, np.nan, dtype=float)
        anisotropy_curve[valid] = np.abs((iv[valid] - ih[valid]) / (iv[valid] + ih[valid]))
        anisotropy_factor = float(np.nanmean(anisotropy_curve[valid])) if np.any(valid) else np.nan

        self.last_anisotropy = {
            "q": q,
            "ih": ih,
            "iv": iv,
            "anisotropy_curve": anisotropy_curve,
            "anisotropy_factor": anisotropy_factor,
            "xc": xc,
            "yc": yc,
            "distance": distance,
            "pixel_x": pixel_x,
            "pixel_y": pixel_y,
            "wavelength": wavelength,
        }

        ax = self.canvas.ax
        ax.clear()

        plot_mode = self.anisotropy_plot_mode.currentText()
        x_values = q
        y_iv = iv
        y_ih = ih
        x_label = "q / nm⁻¹"
        y_label = "Intensity / a.u."
        x_scale = "linear"
        y_scale = "linear"

        if plot_mode == "Kratky":
            y_iv = iv * q ** 2
            y_ih = ih * q ** 2
            y_label = "q² I(q)"
        elif plot_mode == "log-log":
            x_scale = "log"
            y_scale = "log"
        elif plot_mode == "lin-log":
            y_scale = "log"
        elif plot_mode == "log-lin":
            x_scale = "log"

        plot_valid_iv = np.isfinite(x_values) & np.isfinite(y_iv)
        plot_valid_ih = np.isfinite(x_values) & np.isfinite(y_ih)
        if x_scale == "log":
            plot_valid_iv &= x_values > 0
            plot_valid_ih &= x_values > 0
        if y_scale == "log":
            plot_valid_iv &= y_iv > 0
            plot_valid_ih &= y_ih > 0

        if self.use_q_range.isChecked():
            q_min = min(self.q_min_filter.value(), self.q_max_filter.value())
            q_max = max(self.q_min_filter.value(), self.q_max_filter.value())
            plot_valid_iv &= (q >= q_min) & (q <= q_max)
            plot_valid_ih &= (q >= q_min) & (q <= q_max)

        ax.plot(x_values[plot_valid_ih], y_ih[plot_valid_ih], color="red", linewidth=1.4, label="Ih")
        ax.plot(x_values[plot_valid_iv], y_iv[plot_valid_iv], color="blue", linewidth=1.4, label="Iv")
        ax.set_xscale(x_scale)
        ax.set_yscale(y_scale)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, which="both")
        ax.legend(loc="best")
        self.canvas.draw_idle()

        self.image_canvas.show_image(
            image,
            xc,
            yc,
            mask=display_h_mask | display_v_mask,
            h_mask=display_h_mask,
            v_mask=display_v_mask,
            reference_angle=self.reference_angle.value(),
        )

        self.results_text.setPlainText(
            f"File = {self.current_file.name}\n"
            f"Frame = {self.current_frame} / {self.total_frames}\n"
            f"Horizontal ψ = {self.h_psi_min.value():.3f}° -> {self.h_psi_max.value():.3f}°\n"
            f"Vertical ψ = {self.v_psi_min.value():.3f}° -> {self.v_psi_max.value():.3f}°\n"
            f"Reference angle = {self.reference_angle.value():.3f}°\n"
            f"Center = ({xc:.3f}, {yc:.3f}) | {self.instrument_mode}\n"
            f"q range = {'on' if self.use_q_range.isChecked() else 'off'}"
            f" ({self.q_min_filter.value():.4f} -> {self.q_max_filter.value():.4f} nm⁻¹)\n"
            f"A = |Iv - Ih| / (Iv + Ih)\n"
            f"A mean = {anisotropy_factor:.5f}"
        )

        # (Removed accidental duplicated Hermans calculation block)

    def update_plot(self, azimuth, intensity, baseline, peak, amplitude, sigma, fwhm, pi, hermans_corr):
        ax = self.canvas.ax
        ax.clear()

        x_model = np.linspace(0, 360, 2000)
        y_model_corr = amplitude * np.exp(-((x_model - peak) ** 2) / (2 * sigma ** 2))
        y_model_raw = self.last_fit["baseline"] + y_model_corr

        half_max_raw = self.last_fit["baseline"] + amplitude / 2
        x_left = peak - fwhm / 2
        x_right = peak + fwhm / 2

        ax.plot(azimuth, intensity, "k-", linewidth=1.2, label="Raw data")
        ax.plot(azimuth, baseline, "r--", linewidth=1.3, label="Baseline")
        ax.plot(x_model, y_model_raw, "r-", linewidth=1.6, label="Gaussian fit")
        ax.plot([x_left, x_right], [half_max_raw, half_max_raw], "b-", linewidth=2, label="FWHM")
        ax.axvline(peak, color="blue", linestyle="--", linewidth=1, label="Peak centre")

        ax.set_title("")
        ax.set_xlabel("ψ / °")
        ax.set_ylabel("Intensity / a.u.")
        ax.set_xlim(0, 360)
        ax.grid(True)
        ax.legend(loc="best")
        self.canvas.draw_idle()

    def update_results_text(self):
        if not self.last_fit or self.current_file is None:
            return

        fit = self.last_fit
        self.results_text.setPlainText(
            f"File = {self.current_file.name}\n"
            f"Mode = {'Gaussian fit' if fit['use_fit'] else 'Manual height/FWHM'}\n"
            f"Baseline = {fit['baseline']:.5f}\n"
            f"Baseline offset = {fit['offset']:.5f}\n"
            f"Peak centre = {fit['peak']:.3f} °\n"
            f"Height = {fit['amplitude']:.5f}\n"
            f"FWHM = {fit['fwhm']:.3f} °\n"
            f"Π = {fit['pi']:.4f}\n"
            f"Π = {fit['pi'] * 100:.2f} %\n"
            f"<cos²ψ> = {fit['cos2mean_corr']:.5f}\n"
            f"Hermans factor f = {fit['hermans_corr']:.5f}"
        )

    def recenter_on_fit(self):
        if not self.use_fit_checkbox.isChecked():
            QMessageBox.warning(self, "Fit disabled", "Enable Fit mode before using this button.")
            return

        if self.last_fit is None:
            QMessageBox.warning(self, "Fit unavailable", "No fit is currently available.")
            return

        peak = self.last_fit["peak"]
        self.peak_spin.setValue(peak)
        self.calculate()

    def save_gaussian_fit(self):
        # Removed Fit mode check: always allow saving
        if self.current_file is None or self.last_fit is None:
            QMessageBox.warning(self, "Save unavailable", "No fit is currently available.")
            return

        fit = self.last_fit
        x_save = np.linspace(0, 360, 2000)
        y_fit_corr = fit["amplitude"] * np.exp(-((x_save - fit["peak"]) ** 2) / (2 * fit["sigma"] ** 2))
        y_fit_raw = fit["baseline"] + y_fit_corr

        output = np.column_stack([x_save, y_fit_corr, y_fit_raw])
        output_file = self.current_file.parent / f"{self.current_file.stem}_gaussian_fit.dat"

        with open(output_file, "w", encoding="utf-8") as file:
            file.write("# Gaussian/manual profile saved from Hermans\n")
            file.write(f"# Source file: {self.current_file}\n")
            file.write(f"# Mode = {'Gaussian fit' if fit['use_fit'] else 'Manual height/FWHM'}\n")
            file.write(f"# Peak centre = {fit['peak']:.6f} deg\n")
            file.write(f"# Amplitude = {fit['amplitude']:.10e}\n")
            file.write(f"# Sigma = {fit['sigma']:.10e} deg\n")
            file.write(f"# FWHM = {fit['fwhm']:.10e} deg\n")
            file.write(f"# Baseline = {fit['baseline']:.10e}\n")
            file.write("# Columns: azimuth_deg fit_corrected fit_raw\n")
            np.savetxt(file, output, fmt="%.6f %.10e %.10e")

        QMessageBox.information(self, "Fit saved", f"Fit saved:\n{output_file}")

    def save_anisotropy_profiles(self):
        if self.current_file is None:
            QMessageBox.warning(self, "Save unavailable", "No image file is currently loaded.")
            return

        if not self.is_anisotropy_mode():
            QMessageBox.warning(self, "Save unavailable", "Switch to anisotropy mode first.")
            return

        if self.last_anisotropy is None:
            self.calculate_anisotropy()

        if self.last_anisotropy is None:
            QMessageBox.warning(self, "Save unavailable", "No Iv/Ih profiles are currently available.")
            return

        data = self.last_anisotropy
        q = np.asarray(data["q"], dtype=float)
        ih = np.asarray(data["ih"], dtype=float)
        iv = np.asarray(data["iv"], dtype=float)

        frame_suffix = ""
        if self.total_frames > 1:
            frame_suffix = f"_frame{self.current_frame:04d}"

        ih_file = self.current_file.parent / f"{self.current_file.stem}{frame_suffix}_Ih.dat"
        iv_file = self.current_file.parent / f"{self.current_file.stem}{frame_suffix}_Iv.dat"

        def write_profile(path, intensity, label):
            valid = np.isfinite(q) & np.isfinite(intensity)
            output = np.column_stack([q[valid], intensity[valid]])

            with open(path, "w", encoding="utf-8") as file:
                file.write(f"# {label} profile saved from Hermans anisotropy\n")
                file.write(f"# Source file: {self.current_file}\n")
                file.write(f"# Frame: {self.current_frame} / {self.total_frames}\n")
                file.write(f"# Instrument mode: {self.instrument_mode}\n")
                file.write(f"# Center X = {data['xc']:.10g}\n")
                file.write(f"# Center Y = {data['yc']:.10g}\n")
                file.write(f"# Distance = {data['distance']:.10g} m\n")
                file.write(f"# Pixel X = {data['pixel_x']:.10g} mm\n")
                file.write(f"# Pixel Y = {data['pixel_y']:.10g} mm\n")
                file.write(f"# Wavelength = {data['wavelength']:.10g} Å\n")
                file.write("# Columns: q_nm^-1 intensity\n")
                np.savetxt(file, output, fmt="%.10e %.10e")

        write_profile(ih_file, ih, "Ih")
        write_profile(iv_file, iv, "Iv")

        QMessageBox.information(
            self,
            "Profiles saved",
            f"Profiles saved:\n{ih_file}\n{iv_file}",
        )
