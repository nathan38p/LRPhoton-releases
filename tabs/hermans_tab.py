
from pathlib import Path
import re

import h5py
import numpy as np

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QSpinBox,
    QTextEdit,
    QGridLayout,
    QFormLayout,
    QListWidget,
    QMessageBox,
    QSlider,
    QCheckBox,
    QComboBox,
    QLineEdit,
    QSizePolicy,
    QStackedLayout,
    QRadioButton,
    QButtonGroup,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from scipy.integrate import quad
from scipy.optimize import curve_fit

from .instrument_presets import (
    ID13_DEFAULT_CENTER_X,
    ID13_DEFAULT_CENTER_Y,
    ID13_DEFAULT_DISTANCE_M,
    ID13_DEFAULT_PIXEL_MM,
    ID13_DEFAULT_WAVELENGTH_A,
)
from .file_ratings import file_path_from_item, install_file_rating_menu, is_file_rated_up, set_item_file_path
from .ui_style import (
    BLOCK_SPACING,
    FILE_BROWSER_WIDTH,
    FlexibleDoubleSpinBox as QDoubleSpinBox,
    FRAME_BUTTON_WIDTH,
    FRAME_COUNTER_WIDTH,
    FRAME_NAV_SPACING,
    FRAME_SPIN_WIDTH,
    GROUP_BOX_MARGINS,
    GROUP_BOX_STYLE,
    clear_plot_canvas,
    install_selectable_legend,
    make_matplotlib_toolbar_block,
    normalize_decimal_text,
    PAGE_MARGINS,
    PANEL_MARGINS,
    style_q_geometry_buttons,
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
                match = re.search(r"[-+]?\d*[\.,]?\d+(?:[eE][-+]?\d+)?", str(header[key]))
                if match:
                    return float(normalize_decimal_text(match.group(0)))
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
# ========== Reciprocal Lorentzian Distribution ==============
# ============================================================

def reciprocal_lorentzian_distribution(theta, width, constant):
    return (1.0 + (0.7664 * theta / width) ** 2) ** -1.5 + constant


def reciprocal_lorentzian_intensity(array_x, amplitude, width, constant, q_a, center_rad, wavelength_a):
    x_values = np.asarray(array_x, dtype=float)
    width = max(float(abs(width)), 1e-12)

    theta_b_arg = wavelength_a * q_a / (4.0 * np.pi)
    theta_b_arg = np.clip(theta_b_arg, -1.0, 1.0)
    theta_b = np.arcsin(theta_b_arg)

    def p_tau_csi(tau, csi):
        inner = np.cos(theta_b) * np.cos(tau)
        inner = np.clip(inner, -1.0, 1.0)

        projected = np.cos(csi) * np.sin(np.arccos(inner))
        projected = np.clip(projected, -1.0, 1.0)

        angle = np.arccos(projected)
        return (1.0 + (0.7664 * angle / width) ** 2) ** (-1.5) + constant

    intensity = []
    for x_value in np.ravel(x_values):
        # Pascale-equivalent convention while keeping LRPhoton's center as the visible peak:
        # Pascale uses tau = x - cen and the visible detector peak is cen + 90°.
        # Here center_rad is the visible peak, so cen = center_rad - 90°.
        tau = x_value - center_rad + np.pi / 2.0
        result = quad(lambda csi: amplitude * p_tau_csi(tau, csi), 0.0, np.pi / 2.0)[0]
        intensity.append(result)

    return np.asarray(intensity, dtype=float).reshape(x_values.shape)


def estimate_profile_background(intensity, percentile=5.0):
    values = np.asarray(intensity, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    return float(np.nanpercentile(values, percentile))


def middle_profile_values(azimuth, intensity, background=0.0, middle_fraction=0.5):
    x = np.asarray(azimuth, dtype=float)
    y = np.asarray(intensity, dtype=float) - background
    valid = np.isfinite(x) & np.isfinite(y)
    if not np.any(valid):
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    x = x[valid]
    y = y[valid]
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    span = x_max - x_min
    if span > 0:
        margin = 0.5 * (1.0 - middle_fraction) * span
        middle = (x >= x_min + margin) & (x <= x_max - margin)
        if np.any(middle):
            x = x[middle]
            y = y[middle]

    return x, y


def estimate_profile_center(azimuth, intensity, background=0.0, middle_fraction=0.5):
    x, y = middle_profile_values(azimuth, intensity, background=background, middle_fraction=middle_fraction)
    if x.size == 0:
        return 0.0
    return float(x[int(np.nanargmax(y))])


def estimate_profile_peak_amplitude(azimuth, intensity, background=0.0, middle_fraction=0.5):
    _, y = middle_profile_values(azimuth, intensity, background=background, middle_fraction=middle_fraction)
    if y.size == 0:
        return 1.0
    amplitude = float(np.nanmax(y))
    return max(amplitude, 1e-9)


def _parse_q_number(text):
    return float(text.replace(",", ".").replace("p", ".").replace("P", ".").replace("v", ".").replace("V", "."))


def q_nm_from_filename(file_path):
    name = Path(file_path).name
    pattern = re.compile(
        r"[qQ]\s*([0-9]+(?:[.,pPvV][0-9]+)?)"
        r"(?:\s*[-–_]\s*([0-9]+(?:[.,pPvV][0-9]+)?))?"
        r"(?=\s*(?:nm\s*(?:[-^]?1|⁻¹)|A\s*(?:[-^]?1|⁻¹)|Å\s*(?:[-^]?1|⁻¹)|_|-|\.|$))"
    )
    match = pattern.search(name)
    if match is None:
        return None

    q_min = _parse_q_number(match.group(1))
    q_max = _parse_q_number(match.group(2)) if match.group(2) else q_min
    q_value = 0.5 * (q_min + q_max)

    unit_text = name[match.end():match.end() + 8].lower()
    if unit_text.startswith("a") or unit_text.startswith("å"):
        return q_value * 10.0
    return q_value


def calculate_reciprocal_order_parameter(width, constant, ratio):
    width = max(abs(width), 1e-12)
    ratio = float(np.clip(ratio, 0.0, 0.999999999))
    if ratio == 1.0:
        return 0.0, 1.0

    def distribution(angle):
        return reciprocal_lorentzian_distribution(angle, width, constant)

    denominator = quad(lambda angle: 4.0 * np.pi * distribution(angle) * np.sin(angle), 0.0, np.pi / 2.0)[0]
    if denominator == 0:
        return np.nan, np.nan

    integral_p1 = quad(lambda angle: distribution(angle) / denominator, 0.0, np.pi / 2.0)[0]
    isotropic_ratio = 8.0 * ratio / (1.0 - ratio) * integral_p1

    integral_cos2 = quad(
        lambda angle: distribution(angle) / denominator * np.sin(angle) * np.cos(angle) ** 2,
        0.0,
        np.pi / 2.0,
    )[0]

    order_parameter = 0.5 / (1.0 + isotropic_ratio) * (12.0 * np.pi * integral_cos2 - 1.0)
    return order_parameter, isotropic_ratio


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
        self.last_xc = None
        self.last_yc = None
        self.reference_angle = 0.0
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
                psi_text = "-"

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
                            q_text = f"{q_value:.6g} nm⁻¹"

                if self.last_xc is not None and self.last_yc is not None:
                    dx = (x_index + 1) - self.last_xc
                    dy = (y_index + 1) - self.last_yc
                    psi = (np.degrees(np.arctan2(dy, dx)) - self.reference_angle) % 360.0
                    psi_text = f"{psi:.3f}°"

                self.coordinate_label.setText(
                    f"ψ = {psi_text} | q = {q_text} | I = {value_text}"
                )
            else:
                self.coordinate_label.setText("ψ = - | q = - | I = -")

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
        self.last_xc = xc
        self.last_yc = yc
        self.reference_angle = float(reference_angle)
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
        self.last_order_parameter = None
        self.current_frame = 1
        self.total_frames = 1
        self._updating_frame_controls = False
        self.instrument_mode = "XENOCS"
        self.custom_anisotropy_geometry = None
        self.q_axis_unit = "nm"
        self._order_fit_timer = QTimer(self)
        self._order_fit_timer.setSingleShot(True)
        self._order_fit_timer.timeout.connect(self.calculate_order_parameter)

        self.build_ui()
        self.init_default_folder()

    def build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(*PAGE_MARGINS)
        main_layout.setSpacing(BLOCK_SPACING)

        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(BLOCK_SPACING)
        main_layout.addLayout(content_layout, stretch=1)

        left_panel = QWidget()
        left_panel.setFixedWidth(FILE_BROWSER_WIDTH)
        controls_layout = QVBoxLayout(left_panel)
        controls_layout.setContentsMargins(*PANEL_MARGINS)
        controls_layout.setSpacing(BLOCK_SPACING)
        content_layout.addWidget(left_panel, stretch=0)

        graph_wrapper = QWidget()
        graph_wrapper_layout = QVBoxLayout(graph_wrapper)
        graph_wrapper_layout.setContentsMargins(0, 0, 0, 0)
        graph_wrapper_layout.setSpacing(BLOCK_SPACING)

        graph_content_layout = QHBoxLayout()
        graph_content_layout.setContentsMargins(0, 0, 0, 0)
        graph_content_layout.setSpacing(BLOCK_SPACING)
        graph_wrapper_layout.addLayout(graph_content_layout, stretch=1)

        self.right_panel = QWidget()
        self.right_panel.setFixedWidth(FILE_BROWSER_WIDTH)
        self.right_layout = QVBoxLayout(self.right_panel)
        self.right_layout.setContentsMargins(*PANEL_MARGINS)
        self.right_layout.setSpacing(BLOCK_SPACING)

        self.center_column = QWidget()
        self.center_column_layout = QVBoxLayout(self.center_column)
        self.center_column_layout.setContentsMargins(0, 0, 0, 0)
        self.center_column_layout.setSpacing(4)
        graph_content_layout.addWidget(self.center_column, stretch=5)

        self.image_column = QWidget()
        self.image_column.setFixedWidth(FILE_BROWSER_WIDTH)
        self.image_column_layout = QVBoxLayout(self.image_column)
        self.image_column_layout.setContentsMargins(*PANEL_MARGINS)
        self.image_column_layout.setSpacing(BLOCK_SPACING)

        self.image_box = QGroupBox("Scattering pattern")
        image_layout = QVBoxLayout(self.image_box)
        image_layout.setContentsMargins(*GROUP_BOX_MARGINS)
        image_layout.setSpacing(6)
        self.image_column_layout.addWidget(self.image_box, stretch=1)

        self.side_panel = QWidget()
        self.side_panel.setFixedWidth(FILE_BROWSER_WIDTH)
        self.side_panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.side_layout = QStackedLayout(self.side_panel)
        self.side_layout.setContentsMargins(0, 0, 0, 0)
        self.side_layout.setSpacing(0)
        self.side_layout.addWidget(self.right_panel)
        self.side_layout.addWidget(self.image_column)
        graph_content_layout.addWidget(self.side_panel, stretch=0)

        content_layout.addWidget(graph_wrapper, stretch=1)

        file_browser_box = QGroupBox("File browser")
        file_browser_box.setMinimumHeight(220)
        file_browser_box.setStyleSheet(GROUP_BOX_STYLE)

        file_browser_layout = QVBoxLayout(file_browser_box)
        file_browser_layout.setContentsMargins(*GROUP_BOX_MARGINS)
        file_browser_layout.setSpacing(6)

        controls_layout.addWidget(file_browser_box, stretch=1)

        self.folder_path = QLineEdit("")
        self.folder_path.returnPressed.connect(self.refresh_files)
        file_browser_layout.addWidget(self.folder_path)

        self.open_folder_button = QPushButton("Browse")
        self.open_folder_button.clicked.connect(self.open_folder)
        file_browser_layout.addWidget(self.open_folder_button)

        filters_layout = QGridLayout()

        self.extensions_filter = QLineEdit("*azimProf.dat")
        self.name_filter = QLineEdit("*cave*")

        self.extensions_filter.textChanged.connect(self.refresh_files)
        self.name_filter.textChanged.connect(self.refresh_files)

        filters_layout.addWidget(QLabel("Name:"), 0, 0)
        filters_layout.addWidget(self.name_filter, 0, 1)

        filters_layout.addWidget(QLabel("Extensions:"), 1, 0)
        filters_layout.addWidget(self.extensions_filter, 1, 1)

        file_browser_layout.addLayout(filters_layout)

        self.show_subfolders_checkbox = QCheckBox("Show subfolders")
        self.show_subfolders_checkbox.setChecked(False)
        self.show_subfolders_checkbox.stateChanged.connect(self.refresh_files)
        self.only_thumbs_up_checkbox = QCheckBox("Only 👍")
        self.only_thumbs_up_checkbox.setChecked(False)
        self.only_thumbs_up_checkbox.stateChanged.connect(self.refresh_files)
        file_options_layout = QHBoxLayout()
        file_options_layout.setContentsMargins(0, 0, 0, 0)
        file_options_layout.addWidget(self.show_subfolders_checkbox)
        file_options_layout.addWidget(self.only_thumbs_up_checkbox)
        file_options_layout.addStretch(1)
        file_browser_layout.addLayout(file_options_layout)

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_files)
        file_browser_layout.addWidget(self.refresh_button)

        self.file_list = QListWidget()
        install_file_rating_menu(self.file_list)
        self.file_list.currentItemChanged.connect(self.load_selected_file)
        self.file_list.itemClicked.connect(self.load_selected_file)
        self.file_list.itemActivated.connect(self.load_selected_file)
        self.file_list.itemSelectionChanged.connect(self.load_selected_file)
        self.file_list.setMinimumHeight(180)

        file_browser_layout.addWidget(self.file_list, stretch=1)

        self.anisotropy_param_widgets = []
        self.order_param_widgets = []
        # --- Parameter selection box ---
        mode_box = QGroupBox("Parameter choice")
        mode_layout = QGridLayout(mode_box)
        mode_layout.setContentsMargins(*GROUP_BOX_MARGINS)
        mode_layout.setSpacing(6)

        self.parameter_selector = QComboBox()
        self.parameter_selector.addItems([
            "Hermans factor",
            "Anisotropy factor (Iv - Ih) / (Iv + Ih)",
            "Order parameter S (L3/2 reciprocal fit)",
        ])
        self.parameter_selector.currentIndexChanged.connect(self.parameter_mode_changed)
        self.parameter_selector.setCurrentIndex(1)

        mode_layout.addWidget(QLabel("Parameter:"), 0, 0)
        mode_layout.addWidget(self.parameter_selector, 0, 1)
        mode_layout.setColumnStretch(1, 1)
        mode_box.setFixedHeight(78)

        self.mode_box = mode_box
        controls_layout.addWidget(mode_box, stretch=0)

        results_box = QGroupBox("Results")
        results_layout = QVBoxLayout(results_box)
        results_layout.setContentsMargins(*GROUP_BOX_MARGINS)
        results_layout.setSpacing(6)

        # Ensure required matplotlib imports for formula rendering
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        except ImportError:
            pass
        try:
            from matplotlib.figure import Figure
        except ImportError:
            pass

        self.lorentzian_formula_figure = Figure(figsize=(5.2, 0.55))
        self.lorentzian_formula_canvas = FigureCanvasQTAgg(self.lorentzian_formula_figure)
        self.lorentzian_formula_ax = self.lorentzian_formula_figure.add_subplot(111)
        self.lorentzian_formula_ax.axis("off")

        self.lorentzian_formula_figure.subplots_adjust(left=0.0, right=1.0, top=0.95, bottom=0.05)
        self.lorentzian_formula_canvas.setFixedHeight(0)
        self.lorentzian_formula_canvas.setToolTip("Pascale model; LRPhoton keeps Center as the visible peak, so internal cen = center - 90°.")

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)

        self.results_box = results_box
        self.results_box.setMinimumHeight(190)
        controls_layout.addWidget(results_box, stretch=0)

        params_box = QGroupBox("Parameters")
        params_layout = QGridLayout(params_box)
        params_layout.setContentsMargins(*GROUP_BOX_MARGINS)
        params_layout.setSpacing(6)
        self.params_box = params_box
        self.center_column_layout.addWidget(params_box, stretch=0)
        self.graph_wrapper_layout = graph_wrapper_layout
        self.offset_spin, self.offset_slider = self.add_slider_control(
            params_layout, 0, "Baseline", 0.0, -1.0, 1.0
        )
        self.peak_spin, self.peak_slider = self.add_slider_control(
            params_layout, 1, "Peak ψ₀ (°)", 85.0, 0.0, 360.0
        )

        self.fit_mode_label = QLabel("Fit:")
        self.fit_with_data_radio = QRadioButton("With data")
        self.manual_fit_radio = QRadioButton("Manually")
        self.fit_with_data_radio.setChecked(True)

        self.fit_mode_group = QButtonGroup(self)
        self.fit_mode_group.setExclusive(True)
        self.fit_mode_group.addButton(self.fit_with_data_radio, 0)
        self.fit_mode_group.addButton(self.manual_fit_radio, 1)
        self.fit_mode_group.idClicked.connect(self.update_fit_mode)

        fit_mode_layout = QHBoxLayout()
        fit_mode_layout.setContentsMargins(0, 0, 0, 0)
        fit_mode_layout.setSpacing(12)
        fit_mode_layout.addWidget(self.fit_with_data_radio)
        fit_mode_layout.addWidget(self.manual_fit_radio)
        fit_mode_layout.addStretch(1)

        params_layout.addWidget(self.fit_mode_label, 2, 0)
        params_layout.addLayout(fit_mode_layout, 2, 1, 1, 4)

        self.window_spin, self.window_slider = self.add_slider_control(
            params_layout, 3, "Window (°)", 90.0, 10.0, 180.0
        )
        self.height_spin, self.height_slider = self.add_slider_control(
            params_layout, 4, "Height", 1.0, 0.0, 10.0
        )
        self.manual_fwhm_spin, self.manual_fwhm_slider = self.add_slider_control(
            params_layout, 5, "FWHM (°)", 90.0, 0.1, 180.0
        )

        self.order_panel = QWidget()
        order_layout = QGridLayout(self.order_panel)
        order_layout.setContentsMargins(0, 0, 0, 0)
        order_layout.setHorizontalSpacing(6)
        order_layout.setVerticalSpacing(4)

        def add_order_slider(row, label, value, minimum, maximum, decimals=3, suffix=""):
            label_widget = QLabel(label)
            min_spin = QDoubleSpinBox()
            min_spin.setDecimals(decimals)
            min_spin.setRange(-1e9, 1e9)
            min_spin.setValue(minimum)
            min_spin.setFixedWidth(64)
            min_spin.setKeyboardTracking(False)
            max_spin = QDoubleSpinBox()
            max_spin.setDecimals(decimals)
            max_spin.setRange(-1e9, 1e9)
            max_spin.setValue(maximum)
            max_spin.setFixedWidth(64)
            max_spin.setKeyboardTracking(False)
            value_spin = QDoubleSpinBox()
            value_spin.setDecimals(decimals)
            value_spin.setRange(minimum, maximum)
            value_spin.setValue(value)
            if suffix:
                value_spin.setSuffix(suffix)
            value_spin.setFixedWidth(92)
            value_spin.setFixedHeight(22)
            value_spin.setKeyboardTracking(False)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 1000)
            slider.setValue(self.value_to_slider(value, minimum, maximum))
            slider.setMinimumWidth(110)
            slider.setTracking(False)

            slider.min_spin = min_spin
            slider.max_spin = max_spin
            slider.value_spin = value_spin
            value_spin.order_slider = slider
            value_spin.order_min_spin = min_spin
            value_spin.order_max_spin = max_spin

            def sync_slider_limits(_value=None):
                min_value = min_spin.value()
                max_value = max_spin.value()
                if max_value <= min_value:
                    max_value = min_value + 1e-9
                    max_spin.blockSignals(True)
                    max_spin.setValue(max_value)
                    max_spin.blockSignals(False)
                value_spin.setRange(min_value, max_value)
                clamped = max(min_value, min(max_value, value_spin.value()))
                value_spin.blockSignals(True)
                value_spin.setValue(clamped)
                value_spin.blockSignals(False)
                slider.blockSignals(True)
                slider.setValue(self.value_to_slider(clamped, min_value, max_value))
                slider.blockSignals(False)
                self.order_parameter_changed()

            def sync_from_slider(slider_value=None, run_fit=True):
                if slider_value is None:
                    slider_value = slider.value()
                min_value = min_spin.value()
                max_value = max_spin.value()
                value = min_value + (max_value - min_value) * float(slider_value) / 1000.0
                value_spin.blockSignals(True)
                value_spin.setValue(value)
                value_spin.blockSignals(False)
                if run_fit:
                    self.order_parameter_changed()
                else:
                    self.update_order_preview()

            def begin_slider_drag():
                self._order_fit_timer.stop()

            def preview_from_slider(_value=None):
                sync_from_slider(slider_value=_value, run_fit=False)

            def finish_slider_drag():
                sync_from_slider(run_fit=True)

            def sync_from_spin(_value=None):
                min_value = min_spin.value()
                max_value = max_spin.value()
                value = max(min_value, min(max_value, value_spin.value()))
                value_spin.blockSignals(True)
                value_spin.setValue(value)
                value_spin.blockSignals(False)
                slider.blockSignals(True)
                slider.setValue(self.value_to_slider(value, min_value, max_value))
                slider.blockSignals(False)
                self.order_parameter_changed()

            min_spin.valueChanged.connect(sync_slider_limits)
            max_spin.valueChanged.connect(sync_slider_limits)
            slider.sliderPressed.connect(begin_slider_drag)
            slider.sliderMoved.connect(preview_from_slider)
            slider.sliderReleased.connect(finish_slider_drag)
            slider.valueChanged.connect(sync_from_slider)
            value_spin.valueChanged.connect(sync_from_spin)

            order_layout.addWidget(label_widget, row, 0)
            order_layout.addWidget(min_spin, row, 1)
            order_layout.addWidget(slider, row, 2)
            order_layout.addWidget(max_spin, row, 3)
            order_layout.addWidget(value_spin, row, 4)
            order_layout.setColumnStretch(2, 1)
            self.order_param_widgets.extend([label_widget, min_spin, slider, max_spin, value_spin])
            return value_spin

        def add_order_spin(row, column, label, value, minimum, maximum, decimals=4, suffix=""):
            label_widget = QLabel(label)
            spin = QDoubleSpinBox()
            spin.setDecimals(decimals)
            spin.setRange(minimum, maximum)
            spin.setValue(value)
            if suffix:
                spin.setSuffix(suffix)
            spin.setFixedWidth(96)
            spin.setFixedHeight(22)
            spin.setKeyboardTracking(False)
            spin.valueChanged.connect(self.order_parameter_changed)
            order_layout.addWidget(label_widget, row, column)
            order_layout.addWidget(spin, row, column + 1)
            self.order_param_widgets.extend([label_widget, spin])
            return spin

        self.order_background_spin = add_order_slider(0, "Background", 30.0, 20.0, 40.0, decimals=3, suffix=" a.u.")
        self.order_center_spin = add_order_slider(1, "Center", 1.0, 0.0, 360.0, decimals=3, suffix=" °")
        self.order_amp_spin = add_order_slider(2, "Amp init", 200.0, 0.0, 300.0, decimals=3, suffix=" a.u.")
        self.order_width_spin = add_order_slider(3, "Width init", 8.0, 0.001, 60.0, decimals=3, suffix=" °")
        self.order_q_spin = add_order_spin(4, 0, "Q", 6.35, 0.0, 20.0, decimals=4, suffix=" nm⁻¹")
        self.order_ratio_spin = add_order_spin(4, 2, "r", 0.001, 0.0, 0.999999, decimals=4)
        self.order_constant_spin = add_order_spin(5, 0, "C init", 0.0, -1e9, 1e9, decimals=5, suffix=" a.u.")
        self.order_wavelength_spin = add_order_spin(5, 2, "λ", ID13_DEFAULT_WAVELENGTH_A, 0.0001, 100.0, decimals=5, suffix=" Å")
        self.order_fit_min_spin = add_order_spin(6, 0, "Fit min", 0.0, 0.0, 360.0, decimals=2, suffix=" °")
        self.order_fit_max_spin = add_order_spin(6, 2, "Fit max", 360.0, 0.0, 360.0, decimals=2, suffix=" °")
        self.order_fit_button = QPushButton("Fit S")
        self.order_fit_button.setFixedWidth(96)
        self.order_fit_button.clicked.connect(self.calculate_order_parameter)
        order_layout.addWidget(self.order_fit_button, 6, 4)
        params_layout.addWidget(self.order_panel, 8, 0, 1, 5)
        self.order_param_widgets.extend([self.order_panel, self.order_fit_button])

        self.fit_button = None
        self.save_fit_button = None
        small_box_margins = (6, 16, 6, 6)
        small_box_height = 104
        small_spin_width = 86

        h_box = QGroupBox("Horizontal sector")
        h_layout = QGridLayout(h_box)
        h_layout.setContentsMargins(*small_box_margins)
        h_layout.setHorizontalSpacing(6)
        h_layout.setVerticalSpacing(4)

        v_box = QGroupBox("Vertical sector")
        v_layout = QGridLayout(v_box)
        v_layout.setContentsMargins(*small_box_margins)
        v_layout.setHorizontalSpacing(6)
        v_layout.setVerticalSpacing(4)

        q_range_box = QGroupBox("q range")
        q_range_layout = QGridLayout(q_range_box)
        q_range_layout.setContentsMargins(*small_box_margins)
        q_range_layout.setHorizontalSpacing(6)
        q_range_layout.setVerticalSpacing(4)

        reference_box = QGroupBox("Reference angle")
        reference_layout = QGridLayout(reference_box)
        reference_layout.setContentsMargins(*small_box_margins)
        reference_layout.setHorizontalSpacing(6)
        reference_layout.setVerticalSpacing(4)
        def add_sector_spin(layout, row, label, value):
            label_widget = QLabel(label)
            spin = QDoubleSpinBox()
            spin.setDecimals(3)
            spin.setRange(0, 360)
            spin.setValue(value)
            spin.setSuffix(" °")
            spin.setFixedWidth(small_spin_width)
            spin.setFixedHeight(22)
            spin.valueChanged.connect(self.calculate_anisotropy)
            layout.addWidget(label_widget, row, 0)
            layout.addWidget(spin, row, 1)
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
                border-radius: 10px;
                margin-top: 14px;
                padding: 4px;
                font-family: Arial;
                font-size: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 8px;
                padding: 0px 4px;
                color: #9b1111;
                font-family: Arial;
                font-size: 12px;
            }
        """)
        v_box.setStyleSheet("""
            QGroupBox {
                background-color: #eef4ff;
                border: 1px solid #9ebcff;
                border-radius: 10px;
                margin-top: 14px;
                padding: 4px;
                font-family: Arial;
                font-size: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 8px;
                padding: 0px 4px;
                color: #123f9a;
                font-family: Arial;
                font-size: 12px;
            }
        """)

        q_range_box.setStyleSheet("""
            QGroupBox {
                background-color: #fff8df;
                border: 1px solid #f0c85a;
                border-radius: 10px;
                margin-top: 14px;
                padding: 4px;
                font-family: Arial;
                font-size: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 8px;
                padding: 0px 4px;
                color: #8a6200;
                font-family: Arial;
                font-size: 12px;
            }
        """)

        reference_box.setStyleSheet("""
            QGroupBox {
                background-color: #fff8df;
                border: 1px solid #f0c85a;
                border-radius: 10px;
                margin-top: 14px;
                padding: 4px;
                font-family: Arial;
                font-size: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 8px;
                padding: 0px 4px;
                color: #8a6200;
                font-family: Arial;
                font-size: 12px;
            }
        """)
        # (Second q_range_box creation block deleted)
        self.use_q_range = QCheckBox("Use q range")
        self.use_q_range.setChecked(False)
        self.use_q_range.stateChanged.connect(self.calculate_anisotropy)
        self.q_min_filter = QDoubleSpinBox()
        self.q_min_filter.setDecimals(4)
        self.q_min_filter.setRange(0, 1000)
        self.q_min_filter.setValue(0.0)
        self.q_min_filter.setSuffix(" nm⁻¹")
        self.q_min_filter.setFixedWidth(small_spin_width)
        self.q_min_filter.setFixedHeight(22)
        self.q_min_filter.valueChanged.connect(self.calculate_anisotropy)
        self.q_max_filter = QDoubleSpinBox()
        self.q_max_filter.setDecimals(4)
        self.q_max_filter.setRange(0, 1000)
        self.q_max_filter.setValue(10.0)
        self.q_max_filter.setSuffix(" nm⁻¹")
        self.q_max_filter.setFixedWidth(small_spin_width)
        self.q_max_filter.setFixedHeight(22)
        self.q_max_filter.valueChanged.connect(self.calculate_anisotropy)
        q_range_layout.addWidget(self.use_q_range, 0, 0, 1, 2)
        q_range_layout.addWidget(QLabel("q min"), 1, 0)
        q_range_layout.addWidget(self.q_min_filter, 1, 1)
        q_range_layout.addWidget(QLabel("q max"), 2, 0)
        q_range_layout.addWidget(self.q_max_filter, 2, 1)
        self.reference_angle = QDoubleSpinBox()
        self.reference_angle.setDecimals(3)
        self.reference_angle.setRange(-180, 180)
        self.reference_angle.setValue(0.0)
        self.reference_angle.setSuffix(" °")
        self.reference_angle.setFixedWidth(small_spin_width)
        self.reference_angle.setFixedHeight(22)
        self.reference_angle.valueChanged.connect(self.calculate_anisotropy)
        reference_layout.addWidget(QLabel("0°"), 1, 0)
        reference_layout.addWidget(self.reference_angle, 1, 1)
        params_layout.addWidget(h_box, 7, 0, 1, 1)
        params_layout.addWidget(v_box, 7, 1, 1, 1)
        params_layout.addWidget(q_range_box, 7, 2, 1, 1)
        params_layout.addWidget(reference_box, 7, 3, 1, 1)
        params_layout.setColumnStretch(0, 1)
        params_layout.setColumnStretch(1, 1)
        params_layout.setColumnStretch(2, 1)
        params_layout.setColumnStretch(3, 1)
        for box in [h_box, v_box, q_range_box, reference_box]:
            box.setFixedHeight(small_box_height)
            box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.anisotropy_param_widgets.extend([
            h_box, v_box, q_range_box, reference_box,
            self.use_q_range, self.q_min_filter, self.q_max_filter, self.reference_angle,
        ])
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
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(1, 1)
        self.frame_slider.setValue(1)
        self.frame_counter_label = QLabel("1 / 1")
        self.frame_counter_label.setMinimumWidth(FRAME_COUNTER_WIDTH)
        self.frame_counter_label.setAlignment(Qt.AlignCenter)
        frame_nav.addWidget(QLabel("Start:"))
        frame_nav.addWidget(self.frame_start_spin)
        frame_nav.addWidget(self.prev_frame_button)
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
        self.canvas = PlotCanvas()

        self.plot_toolbar = NavigationToolbar(self.canvas, self)

        self.anisotropy_plot_mode = QComboBox()
        self.anisotropy_plot_mode.addItems(["log-log", "lin-lin", "lin-log", "log-lin", "Kratky"])
        self.anisotropy_plot_mode.setFixedWidth(120)
        self.anisotropy_plot_mode.currentIndexChanged.connect(self.calculate_anisotropy)

        self.show_legend_checkbox = QCheckBox("Legend")
        self.show_legend_checkbox.setChecked(True)
        self.show_legend_checkbox.stateChanged.connect(self.calculate)

        self.plot_control_bar, self.plot_toolbar_extra_layout, self.save_anisotropy_button = make_matplotlib_toolbar_block(
            self,
            "Azimuthal profile",
            self.plot_toolbar,
            option_widgets=[
                self.anisotropy_plot_mode,
                self.show_legend_checkbox,
            ],
            save_callback=self.save_current_profiles,
            save_tooltip="Save",
            toolbar_width=320,
        )

        self.anisotropy_param_widgets.extend([
            self.anisotropy_plot_mode,
        ])

        self.center_column_layout.insertWidget(0, self.plot_control_bar, stretch=0)
        self.center_column_layout.insertWidget(1, self.canvas, stretch=1)
        clear_plot_canvas(self.canvas)
        self.graph_coordinate_label = QLabel("ψ = - | I = -")
        self.graph_coordinate_label.setMinimumHeight(28)
        self.graph_coordinate_label.setAlignment(Qt.AlignCenter)
        self.graph_coordinate_label.setStyleSheet("""
            QLabel {
                background-color: #f4f4f4;
                border-radius: 8px;
                padding: 6px;
                font-family: Menlo, Monaco, monospace;
                font-size: 11px;
            }
        """)
        self.center_column_layout.insertWidget(2, self.graph_coordinate_label, stretch=0)
        self.update_plot_toolbar_enabled(False)
        self.canvas.mpl_connect("motion_notify_event", self.update_graph_coordinates)
        self.canvas.mpl_connect("button_press_event", self.on_graph_button_press)
        self.canvas.mpl_connect("axes_leave_event", self.clear_graph_coordinates)
        # (anisotropy_graph_controls block removed)
        self.image_canvas = ImageCanvas()
        self.image_canvas.setMinimumWidth(0)
        self.image_canvas.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        image_layout.addWidget(self.image_canvas, stretch=1)
        self.image_coordinate_label = QLabel("ψ = - | q = - | I = -")
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
        image_layout.addWidget(self.image_coordinate_label, stretch=0)
        self.image_canvas.set_coordinate_label(self.image_coordinate_label)
        # --- Instrument buttons and center controls ---
        instrument_layout = QVBoxLayout()
        instrument_layout.setContentsMargins(0, 0, 0, 0)
        instrument_layout.setSpacing(4)
        preset_layout = QHBoxLayout()
        preset_layout.setSpacing(4)
        self.btn_xenocs = QPushButton("XENOCS")
        self.btn_id02 = QPushButton("ID02")
        self.btn_id13 = QPushButton("ID13")
        self.btn_custom = QPushButton("Custom")
        self.btn_xenocs.setCheckable(True)
        self.btn_id02.setCheckable(True)
        self.btn_id13.setCheckable(True)
        self.btn_custom.setCheckable(True)
        self.btn_xenocs.clicked.connect(lambda: self.set_instrument_mode("XENOCS"))
        self.btn_id02.clicked.connect(lambda: self.set_instrument_mode("ID02"))
        self.btn_id13.clicked.connect(lambda: self.set_instrument_mode("ID13"))
        self.btn_custom.clicked.connect(lambda: self.set_instrument_mode("Custom"))
        self.line_params_button = QPushButton("+")
        self.line_params_button.setFixedWidth(28)
        self.line_params_button.setToolTip("Edit pyFAI geometry parameters")
        self.line_params_button.clicked.connect(self.open_geometry_dialog)

        for button in [
            self.btn_xenocs,
            self.btn_id02,
            self.btn_id13,
            self.btn_custom,
            self.line_params_button,
        ]:
            preset_layout.addWidget(button)
        instrument_layout.addLayout(preset_layout)

        self.center_x_spin = QDoubleSpinBox()
        self.center_y_spin = QDoubleSpinBox()
        self.center_x_spin.setDecimals(3)
        self.center_y_spin.setDecimals(3)
        self.center_x_spin.setRange(-1e6, 1e6)
        self.center_y_spin.setRange(-1e6, 1e6)
        self.center_x_spin.setMinimumWidth(90)
        self.center_y_spin.setMinimumWidth(90)
        self.center_x_spin.valueChanged.connect(self.calculate_anisotropy)
        self.center_y_spin.valueChanged.connect(self.calculate_anisotropy)
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
        self.set_instrument_mode("XENOCS")
        self.parameter_mode_changed()

    def move_widget_to_layout(self, widget, layout, stretch=0):
        if widget.parentWidget() is not None and widget.parentWidget().layout() is not None:
            widget.parentWidget().layout().removeWidget(widget)
        layout.addWidget(widget, stretch=stretch)
        widget.show()

    def apply_parameter_layout_mode(self):
        if not hasattr(self, "mode_box"):
            return

        anisotropy_mode = self.is_anisotropy_mode()

        if anisotropy_mode:
            self.side_layout.setCurrentWidget(self.image_column)

            self.move_widget_to_layout(self.mode_box, self.image_column_layout, stretch=0)
            self.image_column_layout.removeWidget(self.mode_box)
            self.image_column_layout.insertWidget(0, self.mode_box, stretch=0)
            self.mode_box.setVisible(True)

            self.move_widget_to_layout(self.results_box, self.image_column_layout, stretch=1)
            self.image_column_layout.removeWidget(self.results_box)
            self.image_column_layout.insertWidget(2, self.results_box, stretch=1)

            self.move_widget_to_layout(self.params_box, self.center_column_layout, stretch=0)
        else:
            self.side_layout.setCurrentWidget(self.right_panel)

            self.move_widget_to_layout(self.mode_box, self.right_layout, stretch=0)
            self.mode_box.setVisible(True)

            self.move_widget_to_layout(self.results_box, self.right_layout, stretch=1)
            self.move_widget_to_layout(self.params_box, self.center_column_layout, stretch=0)
    def is_hermans_mode(self):
        return hasattr(self, "parameter_selector") and self.parameter_selector.currentIndex() == 0

    def is_anisotropy_mode(self):
        return hasattr(self, "parameter_selector") and self.parameter_selector.currentIndex() == 1

    def is_order_mode(self):
        return hasattr(self, "parameter_selector") and self.parameter_selector.currentIndex() == 2

    def parameter_mode_changed(self, *args):
        hermans_mode = self.is_hermans_mode()
        anisotropy_mode = self.is_anisotropy_mode()
        order_mode = self.is_order_mode()

        if hasattr(self, "plot_control_bar"):
            if anisotropy_mode:
                self.plot_control_bar.setTitle("I(q) profiles")
            elif order_mode:
                self.plot_control_bar.setTitle("Reciprocal order fit")
            else:
                self.plot_control_bar.setTitle("Azimuthal profile")
        if hasattr(self, "save_anisotropy_button"):
            if anisotropy_mode:
                tooltip = "Save anisotropy profiles .dat"
            elif order_mode:
                tooltip = "Save reciprocal order fit .dat"
            else:
                tooltip = "Save Hermans fit .dat"
            self.save_anisotropy_button.setToolTip(tooltip)
        self.image_box.setVisible(anisotropy_mode)

        hermans_widgets = []
        for slider in [
            getattr(self, "offset_slider", None),
            getattr(self, "peak_slider", None),
            getattr(self, "window_slider", None),
            getattr(self, "height_slider", None),
            getattr(self, "manual_fwhm_slider", None),
        ]:
            if slider is None:
                continue
            hermans_widgets.append(slider)
            for attribute_name in ["label_widget", "min_spin", "max_spin"]:
                widget = getattr(slider, attribute_name, None)
                if widget is not None:
                    hermans_widgets.append(widget)

        for widget in [
            getattr(self, "offset_spin", None),
            getattr(self, "peak_spin", None),
            getattr(self, "window_spin", None),
            getattr(self, "fit_mode_label", None),
            getattr(self, "fit_with_data_radio", None),
            getattr(self, "manual_fit_radio", None),
            getattr(self, "height_spin", None),
            getattr(self, "manual_fwhm_spin", None),
        ]:
            if widget is not None:
                hermans_widgets.append(widget)

        for widget in hermans_widgets:
            if widget is not None:
                widget.setVisible(hermans_mode)

        for widget in getattr(self, "anisotropy_param_widgets", []):
            if widget is not None:
                widget.setVisible(anisotropy_mode)

        for widget in getattr(self, "order_param_widgets", []):
            if widget is not None:
                widget.setVisible(order_mode)

        for widget in [
            getattr(self, "btn_xenocs", None),
            getattr(self, "btn_id02", None),
            getattr(self, "btn_id13", None),
            getattr(self, "btn_custom", None),
            getattr(self, "line_params_button", None),
        ]:
            if widget is not None:
                widget.setVisible(anisotropy_mode)

        if hasattr(self, "plot_control_bar"):
            self.plot_control_bar.setVisible(True)
        if hasattr(self, "anisotropy_plot_mode"):
            self.anisotropy_plot_mode.setVisible(anisotropy_mode)

        self.update_file_filter_for_parameter()

        self.apply_parameter_layout_mode()

        if self.folder is not None:
            self.refresh_files()
        elif hasattr(self, "canvas"):
            if anisotropy_mode:
                self.calculate_anisotropy()
            elif order_mode:
                self.update_order_preview()
            else:
                self.calculate()

        if hasattr(self, "frame_slider"):
            self.update_frame_navigation_state()

    def update_file_filter_for_parameter(self):
        if not hasattr(self, "extensions_filter"):
            return

        if self.is_anisotropy_mode():
            desired_filter = "*.edf *.h5 *.hdf5"
        elif self.is_order_mode():
            desired_filter = "*azimProf.dat"
        else:
            desired_filter = "*azimProf.dat"
        if self.extensions_filter.text() == desired_filter:
            if hasattr(self, "name_filter") and self.is_anisotropy_mode() and self.name_filter.text().strip() in {"", "**", "*"}:
                self.name_filter.blockSignals(True)
                self.name_filter.setText("*cave*")
                self.name_filter.blockSignals(False)
            return

        self.extensions_filter.blockSignals(True)
        self.extensions_filter.setText(desired_filter)
        self.extensions_filter.blockSignals(False)
        if hasattr(self, "name_filter"):
            self.name_filter.blockSignals(True)
            self.name_filter.setText("*cave*" if self.is_anisotropy_mode() else "**")
            self.name_filter.blockSignals(False)

    def refresh_files(self, *args):
        if not hasattr(self, "folder_path"):
            return

        folder = Path(self.folder_path.text()).expanduser()
        if not folder.exists() or not folder.is_dir():
            return

        self.folder = folder
        self.folder_changed.emit(self.folder)

        patterns = self.extensions_filter.text().split()
        if not patterns:
            if self.is_anisotropy_mode():
                patterns = ["*.edf", "*.h5", "*.hdf5"]
            elif self.is_order_mode():
                patterns = ["*azimProf.dat"]
            else:
                patterns = ["*azimProf.dat"]

        name_filter = self.name_filter.text().strip()
        if not name_filter:
            name_filter = "**"

        search_method = folder.rglob if self.show_subfolders_checkbox.isChecked() else folder.glob
        files = []
        for pattern in patterns:
            files.extend(search_method(pattern))

        from fnmatch import fnmatch
        files = sorted(set(files))
        files = [file for file in files if fnmatch(file.name, name_filter)]
        if self.only_thumbs_up_checkbox.isChecked():
            files = [file for file in files if is_file_rated_up(file)]

        if self.is_anisotropy_mode():
            excluded_suffixes = (
                "_ave.h5",
                "_ave.hdf5",
                "_aveq_ave.h5",
                "_aveq_ave.hdf5",
                "_averaged.h5",
                "_averaged.hdf5",
                "polar.edf",
            )
            edf_stems = {file.stem for file in files if file.suffix.lower() == ".edf"}
            filtered_files = []
            for file in files:
                lower_name = file.name.lower()
                suffix = file.suffix.lower()

                if lower_name.endswith(excluded_suffixes):
                    continue

                if suffix in [".h5", ".hdf5"] and file.stem in edf_stems:
                    continue

                filtered_files.append(file)
            files = filtered_files

        self.available_files = [
            str(file.relative_to(folder)) if self.show_subfolders_checkbox.isChecked() else file.name
            for file in files
        ]

        current_item = self.file_list.currentItem()
        current_text = current_item.text() if current_item else None

        self.file_list.blockSignals(True)
        self.file_list.clear()
        for display_name, file in zip(self.available_files, files):
            self.file_list.addItem(display_name)
            item = self.file_list.item(self.file_list.count() - 1)
            set_item_file_path(item, file)
        self.file_list.blockSignals(False)

        if self.available_files:
            row = self.available_files.index(current_text) if current_text in self.available_files else 0
            self.file_list.setCurrentRow(row)
            item = self.file_list.item(row)
            self.load_file(file_path_from_item(item, self.folder))
        else:
            self.current_file = None
            self.azimuth = None
            self.intensity = None
            self.current_image = None
            self.current_header = {}
            self.last_fit = None
            self.last_anisotropy = None
            self.last_order_parameter = None
            self.update_plot_toolbar_enabled(False)
            self.results_text.clear()
            self.clear_graph_coordinates()
            clear_plot_canvas(self.canvas)

    def update_plot_toolbar_enabled(self, enabled=None):
        if enabled is None:
            enabled = self.current_file is not None
        for widget in [
            getattr(self, "anisotropy_plot_mode", None),
            getattr(self, "show_legend_checkbox", None),
            getattr(self, "save_anisotropy_button", None),
        ]:
            if widget is not None:
                widget.setEnabled(enabled)

    def save_current_profiles(self):
        if self.is_anisotropy_mode():
            self.save_anisotropy_profiles()
        elif self.is_order_mode():
            self.save_order_fit()
        else:
            self.save_gaussian_fit()

    def update_frame_navigation_state(self):
        can_navigate = self.is_anisotropy_mode() and self.current_file is not None and self.total_frames > 1
        current = self.frame_slider.value()
        self.frame_start_spin.setEnabled(can_navigate)
        self.frame_end_spin.setEnabled(can_navigate)
        self.frame_slider.setEnabled(can_navigate)
        self.prev_frame_button.setEnabled(can_navigate and current > self.frame_slider.minimum())
        self.next_frame_button.setEnabled(can_navigate and current < self.frame_slider.maximum())

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
        self.update_frame_navigation_state()

    def update_frame_bounds(self, *args):
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
        self.update_frame_navigation_state()

    def frame_slider_changed(self, value):
        self.current_frame = int(value)
        self.frame_counter_label.setText(f"{self.current_frame} / {self.total_frames}")
        self.update_frame_navigation_state()

        if not self._updating_frame_controls and self.is_anisotropy_mode() and self.current_file is not None:
            self.load_file(self.current_file)

    def previous_frame(self):
        self.frame_slider.setValue(max(self.frame_slider.minimum(), self.frame_slider.value() - 1))

    def next_frame(self):
        self.frame_slider.setValue(min(self.frame_slider.maximum(), self.frame_slider.value() + 1))

    def graph_coordinate_labels(self):
        if self.is_anisotropy_mode():
            return "q", "I"
        return "ψ", "I"

    def update_graph_coordinates(self, event):
        if event.inaxes != self.canvas.ax or event.xdata is None or event.ydata is None:
            return

        try:
            if self.is_anisotropy_mode():
                unit_label = "Å⁻¹" if self.q_axis_unit == "A" else "nm⁻¹"
                self.graph_coordinate_label.setText(
                    f"q = {event.xdata:.6g} {unit_label} | I = {event.ydata:.6g}"
                )
            else:
                x_name, y_name = self.graph_coordinate_labels()
                x_suffix = "°" if x_name == "ψ" else ""
                self.graph_coordinate_label.setText(
                    f"{x_name} = {event.xdata:.6g}{x_suffix} | {y_name} = {event.ydata:.6g}"
                )
        except Exception:
            self.clear_graph_coordinates()

    def clear_graph_coordinates(self, event=None):
        if not hasattr(self, "graph_coordinate_label"):
            return

        if self.is_anisotropy_mode():
            self.graph_coordinate_label.setText("q = - | I = -")
        else:
            x_name, y_name = self.graph_coordinate_labels()
            self.graph_coordinate_label.setText(f"{x_name} = - | {y_name} = -")

    def q_display_factor(self):
        return 0.1 if self.q_axis_unit == "A" else 1.0

    def q_axis_label(self):
        return "q / Å⁻¹" if self.q_axis_unit == "A" else "q / nm⁻¹"

    def on_graph_button_press(self, event):
        if event.button != 1 or not self.is_anisotropy_mode():
            return
        try:
            clicked_label = self.canvas.ax.xaxis.label.contains(event)[0]
        except Exception:
            clicked_label = False
        if not clicked_label:
            return

        self.q_axis_unit = "A" if self.q_axis_unit == "nm" else "nm"
        self.calculate_anisotropy()

    def set_instrument_mode(self, mode):
        self.instrument_mode = mode

        buttons = {
            "XENOCS": self.btn_xenocs,
            "ID02": self.btn_id02,
            "ID13": self.btn_id13,
            "Custom": self.btn_custom,
        }

        style_q_geometry_buttons(buttons, mode, self.line_params_button)

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
            x = ID13_DEFAULT_CENTER_X
            y = ID13_DEFAULT_CENTER_Y
        else:
            x = self.center_x_spin.value() if self.center_x_spin.value() != 0 else header_float(header, CENTER_X_KEYS, nx / 2)
            y = self.center_y_spin.value() if self.center_y_spin.value() != 0 else header_float(header, CENTER_Y_KEYS, ny / 2)

        self.set_center_spins(x, y)

    def current_anisotropy_geometry(self):
        header = self.current_header or {}
        image = self.current_image
        ny, nx = image.shape if image is not None else (0, 0)

        if self.instrument_mode == "Custom" and self.custom_anisotropy_geometry:
            geometry = dict(self.custom_anisotropy_geometry)
        elif self.instrument_mode == "ID13":
            geometry = {
                "center_x": ID13_DEFAULT_CENTER_X,
                "center_y": ID13_DEFAULT_CENTER_Y,
                "distance": ID13_DEFAULT_DISTANCE_M,
                "pixel_x": ID13_DEFAULT_PIXEL_MM,
                "pixel_y": ID13_DEFAULT_PIXEL_MM,
                "wavelength": ID13_DEFAULT_WAVELENGTH_A,
            }
        else:
            if self.instrument_mode == "ID02":
                default_center_x = ID02_DEFAULT_CENTER_X
                default_center_y = ID02_DEFAULT_CENTER_Y
                default_distance = ID02_DEFAULT_DISTANCE_M
                default_pixel = ID02_DEFAULT_PIXEL_MM
                default_wavelength = ID02_DEFAULT_WAVELENGTH_A
            else:
                default_center_x = 612.0
                default_center_y = 649.0
                default_distance = 0.9
                default_pixel = 0.075
                default_wavelength = 1.54189

            geometry = {
                "center_x": header_float(header, CENTER_X_KEYS, default_center_x),
                "center_y": header_float(header, CENTER_Y_KEYS, default_center_y),
                "distance": header_float(
                    header,
                    ["SampleDistance", "sampledistance", "sample_distance", "Distance", "DetectorDistance"],
                    default_distance,
                ),
                "pixel_x": header_float(header, ["PSize_1", "psize_1", "PixelSizeX", "pixel_x"], default_pixel),
                "pixel_y": header_float(header, ["PSize_2", "psize_2", "PixelSizeY", "pixel_y"], default_pixel),
                "wavelength": header_float(header, ["Wavelength", "wavelength"], default_wavelength),
            }

        if self.center_x_spin.value() != 0:
            geometry["center_x"] = self.center_x_spin.value()
        elif geometry.get("center_x", 0) == 0 and nx:
            geometry["center_x"] = nx / 2

        if self.center_y_spin.value() != 0:
            geometry["center_y"] = self.center_y_spin.value()
        elif geometry.get("center_y", 0) == 0 and ny:
            geometry["center_y"] = ny / 2

        if geometry["pixel_x"] < 1e-3:
            geometry["pixel_x"] *= 1000
        if geometry["pixel_y"] < 1e-3:
            geometry["pixel_y"] *= 1000
        if geometry["wavelength"] < 1e-6:
            geometry["wavelength"] *= 1e10

        return geometry

    def open_geometry_dialog(self):
        geometry = self.current_anisotropy_geometry()

        dialog = QDialog(self)
        dialog.setWindowTitle("pyFAI geometry + anisotropy")
        layout = QVBoxLayout(dialog)
        form = QFormLayout()

        def dialog_spin(value, decimals=6, minimum=0.0, maximum=1e9, suffix=""):
            spin = QDoubleSpinBox()
            spin.setDecimals(decimals)
            spin.setRange(minimum, maximum)
            spin.setValue(float(value))
            spin.setFixedWidth(150)
            spin.setKeyboardTracking(False)
            if suffix:
                spin.setSuffix(suffix)
            return spin

        dialog_spins = {
            "center_x": dialog_spin(geometry["center_x"], decimals=3, minimum=-1e6, maximum=1e6),
            "center_y": dialog_spin(geometry["center_y"], decimals=3, minimum=-1e6, maximum=1e6),
            "distance": dialog_spin(geometry["distance"], decimals=16, suffix=" m"),
            "pixel_x": dialog_spin(geometry["pixel_x"], decimals=6, suffix=" mm"),
            "pixel_y": dialog_spin(geometry["pixel_y"], decimals=6, suffix=" mm"),
            "wavelength": dialog_spin(geometry["wavelength"], decimals=16, suffix=" Å"),
        }

        form.addRow("Center X", dialog_spins["center_x"])
        form.addRow("Center Y", dialog_spins["center_y"])
        form.addRow("Distance", dialog_spins["distance"])
        form.addRow("Pixel X", dialog_spins["pixel_x"])
        form.addRow("Pixel Y", dialog_spins["pixel_y"])
        form.addRow("Wavelength", dialog_spins["wavelength"])

        settings_box = QGroupBox("Anisotropy settings")
        settings_form = QFormLayout(settings_box)
        settings_form.setContentsMargins(*GROUP_BOX_MARGINS)
        settings_form.setSpacing(6)

        h_min_spin = dialog_spin(self.h_psi_min.value(), decimals=3, maximum=360.0, suffix=" °")
        h_max_spin = dialog_spin(self.h_psi_max.value(), decimals=3, maximum=360.0, suffix=" °")
        v_min_spin = dialog_spin(self.v_psi_min.value(), decimals=3, maximum=360.0, suffix=" °")
        v_max_spin = dialog_spin(self.v_psi_max.value(), decimals=3, maximum=360.0, suffix=" °")
        q_min_spin = dialog_spin(self.q_min_filter.value(), decimals=4, maximum=1000.0, suffix=" nm⁻¹")
        q_max_spin = dialog_spin(self.q_max_filter.value(), decimals=4, maximum=1000.0, suffix=" nm⁻¹")
        reference_spin = dialog_spin(self.reference_angle.value(), decimals=3, minimum=-180.0, maximum=180.0, suffix=" °")
        q_range_checkbox = QCheckBox("Use q range")
        q_range_checkbox.setChecked(self.use_q_range.isChecked())

        settings_form.addRow("Horizontal ψ min", h_min_spin)
        settings_form.addRow("Horizontal ψ max", h_max_spin)
        settings_form.addRow("Vertical ψ min", v_min_spin)
        settings_form.addRow("Vertical ψ max", v_max_spin)
        settings_form.addRow("", q_range_checkbox)
        settings_form.addRow("q min", q_min_spin)
        settings_form.addRow("q max", q_max_spin)
        settings_form.addRow("Reference angle", reference_spin)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addLayout(form)
        layout.addWidget(settings_box)
        layout.addWidget(buttons)

        if dialog.exec() != QDialog.Accepted:
            return

        self.custom_anisotropy_geometry = {
            key: spin.value()
            for key, spin in dialog_spins.items()
        }
        self.set_center_spins(
            self.custom_anisotropy_geometry["center_x"],
            self.custom_anisotropy_geometry["center_y"],
        )
        self.h_psi_min.setValue(h_min_spin.value())
        self.h_psi_max.setValue(h_max_spin.value())
        self.v_psi_min.setValue(v_min_spin.value())
        self.v_psi_max.setValue(v_max_spin.value())
        self.use_q_range.setChecked(q_range_checkbox.isChecked())
        self.q_min_filter.setValue(q_min_spin.value())
        self.q_max_filter.setValue(q_max_spin.value())
        self.reference_angle.setValue(reference_spin.value())
        self.set_instrument_mode("Custom")

    def update_fit_mode(self, *args):
        use_fit = self.fit_with_data_radio.isChecked()

        self.window_spin.setEnabled(use_fit)
        self.window_slider.setEnabled(use_fit)
        self.window_slider.min_spin.setEnabled(use_fit)
        self.window_slider.max_spin.setEnabled(use_fit)

        self.height_spin.setEnabled(not use_fit)
        self.height_slider.setEnabled(not use_fit)
        self.height_slider.min_spin.setEnabled(not use_fit)
        self.height_slider.max_spin.setEnabled(not use_fit)

        self.manual_fwhm_spin.setEnabled(not use_fit)
        self.manual_fwhm_slider.setEnabled(not use_fit)
        self.manual_fwhm_slider.min_spin.setEnabled(not use_fit)
        self.manual_fwhm_slider.max_spin.setEnabled(not use_fit)

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

        min_spin.valueChanged.connect(lambda _value=None: self.update_slider_limits(min_spin, max_spin, value_spin, slider))
        max_spin.valueChanged.connect(lambda _value=None: self.update_slider_limits(min_spin, max_spin, value_spin, slider))
        value_spin.valueChanged.connect(lambda _value=None: self.spin_changed(min_spin, max_spin, value_spin, slider))

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

    def slider_changed(self, value=None):
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
            self.last_order_parameter = None
            self.file_list.clear()
            clear_plot_canvas(self.canvas)


    def set_folder_from_external_tab(self, folder):
        folder = Path(folder)
        if self.folder == folder:
            return
        self.load_folder(folder)

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Open folder", str(self.folder or Path.cwd()))
        if folder:
            self.load_folder(Path(folder))

    def load_folder(self, folder):
        self.folder = Path(folder)
        if hasattr(self, "folder_path"):
            self.folder_path.blockSignals(True)
            self.folder_path.setText(str(self.folder))
            self.folder_path.blockSignals(False)
        self.update_file_filter_for_parameter()
        self.refresh_files()

    

    def load_selected_file(self, current=None, previous=None):
        item = current or self.file_list.currentItem()

        if item is None or self.folder is None:
            self.current_file = None
            self.azimuth = None
            self.intensity = None
            self.current_image = None
            self.current_header = {}
            self.last_fit = None
            self.last_anisotropy = None
            self.last_order_parameter = None
            self.results_text.clear()
            self.clear_graph_coordinates()
            self.update_plot_toolbar_enabled(False)
            if hasattr(self, "canvas"):
                clear_plot_canvas(self.canvas)
            return

        self.load_file(file_path_from_item(item, self.folder))

    def load_file(self, file_path):
        self.current_file = Path(file_path)

        if self.is_anisotropy_mode():
            self.current_image = None
            self.current_header = {}
            self.last_anisotropy = None
            self.last_order_parameter = None
            self.clear_graph_coordinates()
            clear_plot_canvas(self.canvas)

            try:
                if self.current_file.suffix.lower() in [".h5", ".hdf5"]:
                    self.configure_frame_navigation(count_h5_frames(self.current_file))
                else:
                    self.configure_frame_navigation(1)

                image, header = read_image_file(self.current_file, frame_index=self.current_frame - 1)
            except Exception as error:
                self.current_file = None
                self.current_image = None
                self.current_header = {}
                self.last_anisotropy = None
                self.update_plot_toolbar_enabled(False)
                clear_plot_canvas(self.canvas)
                QMessageBox.critical(self, "File reading error", str(error))
                return

            self.current_image = image
            self.current_header = header
            self.azimuth = None
            self.intensity = None
            self.last_fit = None
            self.apply_instrument_preset()
            self.update_plot_toolbar_enabled(True)
            self.calculate_anisotropy()
            return

        try:
            azimuth, intensity = read_azimuthal_file(file_path)
        except Exception as error:
            self.current_file = None
            self.update_plot_toolbar_enabled(False)
            QMessageBox.critical(self, "File reading error", str(error))
            return

        self.configure_frame_navigation(1)
        self.azimuth = azimuth
        self.intensity = intensity
        self.current_image = None
        self.current_header = {}
        self.last_fit = None
        self.last_order_parameter = None
        self.update_plot_toolbar_enabled(True)

        if self.is_order_mode():
            self.set_order_q_from_filename()
            self.set_order_background_from_profile()
            self.set_order_center_from_profile()
            self.set_order_amplitude_from_profile()
            self.update_order_preview()
            self.schedule_order_fit()
            return

        self.offset_spin.blockSignals(True)
        self.offset_spin.setValue(0)
        self.offset_spin.blockSignals(False)
        self.offset_slider.blockSignals(True)
        self.offset_slider.setValue(self.value_to_slider(0, self.offset_slider.min_spin.value(), self.offset_slider.max_spin.value()))
        self.offset_slider.blockSignals(False)

        self.draw_raw_azimuthal_profile()
        self.calculate()

    def calculate(self, *args):
        if self.is_anisotropy_mode():
            self.calculate_anisotropy()
            return
        if self.is_order_mode():
            self.update_order_preview()
            return

        if self.azimuth is None or self.intensity is None or self.current_file is None:
            self.clear_graph_coordinates()
            clear_plot_canvas(self.canvas)
            return
        
        ax = self.canvas.ax
        previous_xlim = ax.get_xlim()
        previous_ylim = ax.get_ylim()
        preserve_view = len(ax.lines) > 0

        azimuth = self.azimuth
        intensity = self.intensity
        peak = self.peak_spin.value()
        window = self.window_spin.value()
        offset = self.offset_spin.value()

        mask = np.abs(wrap_to_180(azimuth - peak)) <= window / 2
        az_fit = azimuth[mask]

        if az_fit.size < 10:
            self.draw_raw_azimuthal_profile(peak=peak, window=window)
            self.results_text.setPlainText("Not enough points in the fitting window.")
            return

        baseline_level = np.nanmin(intensity) + offset
        baseline = np.full_like(azimuth, baseline_level)
        corrected = intensity - baseline
        corrected[corrected < 0] = 0
        fit_intensity = corrected[mask]

        if np.nanmax(fit_intensity) <= 0:
            self.draw_raw_azimuthal_profile(baseline=baseline, peak=peak, window=window)
            self.results_text.setPlainText(
                "Corrected signal is null or negative.\nAdjust the baseline."
            )
            return

        use_fit = self.fit_with_data_radio.isChecked()

        if use_fit:
            try:
                amplitude, sigma = fit_gaussian_fixed_center(az_fit, fit_intensity, peak, window)
            except Exception as error:
                self.draw_raw_azimuthal_profile(baseline=baseline, peak=peak, window=window)
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
        if preserve_view:
            self.canvas.ax.set_xlim(previous_xlim)
            self.canvas.ax.set_ylim(previous_ylim)
            self.canvas.draw_idle()
        self.update_results_text()

    def set_order_slider_value_and_limits(self, spin, value, minimum, maximum):
        slider = getattr(spin, "order_slider", None)
        min_spin = getattr(spin, "order_min_spin", None)
        max_spin = getattr(spin, "order_max_spin", None)
        if slider is None or min_spin is None or max_spin is None:
            self.set_order_control_value(spin, value)
            return

        if maximum <= minimum:
            maximum = minimum + 1e-9

        value = max(minimum, min(maximum, value))

        min_spin.blockSignals(True)
        max_spin.blockSignals(True)
        spin.blockSignals(True)
        slider.blockSignals(True)

        min_spin.setValue(minimum)
        max_spin.setValue(maximum)
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        slider.setValue(self.value_to_slider(value, minimum, maximum))

        min_spin.blockSignals(False)
        max_spin.blockSignals(False)
        spin.blockSignals(False)
        slider.blockSignals(False)

    def set_order_control_value(self, spin, value):
        slider = getattr(spin, "order_slider", None)
        min_spin = getattr(spin, "order_min_spin", None)
        max_spin = getattr(spin, "order_max_spin", None)
        if slider is not None and min_spin is not None and max_spin is not None:
            if value < min_spin.value():
                min_spin.blockSignals(True)
                min_spin.setValue(value)
                min_spin.blockSignals(False)
            if value > max_spin.value():
                max_spin.blockSignals(True)
                max_spin.setValue(value)
                max_spin.blockSignals(False)
            spin.setRange(min_spin.value(), max_spin.value())
            slider.blockSignals(True)
            slider.setValue(self.value_to_slider(value, min_spin.value(), max_spin.value()))
            slider.blockSignals(False)
        spin.blockSignals(True)
        spin.setValue(value)
        spin.blockSignals(False)

    def set_order_background_from_profile(self):
        if self.intensity is None or not hasattr(self, "order_background_spin"):
            return
        finite_intensity = np.asarray(self.intensity, dtype=float)
        finite_intensity = finite_intensity[np.isfinite(finite_intensity)]
        if finite_intensity.size == 0:
            return
        background = float(np.nanmin(finite_intensity))
        self.set_order_slider_value_and_limits(
            self.order_background_spin,
            background,
            background - 10.0,
            background + 10.0,
        )

    def set_order_q_from_filename(self):
        if self.current_file is None or not hasattr(self, "order_q_spin"):
            return
        q_nm = q_nm_from_filename(self.current_file)
        if q_nm is None:
            return
        self.set_order_control_value(self.order_q_spin, q_nm)

    def set_order_center_from_profile(self):
        if self.azimuth is None or self.intensity is None or not hasattr(self, "order_center_spin"):
            return
        center = estimate_profile_center(
            self.azimuth,
            self.intensity,
            background=self.order_background_spin.value(),
        )
        self.set_order_control_value(self.order_center_spin, center)

    def set_order_amplitude_from_profile(self):
        if self.azimuth is None or self.intensity is None or not hasattr(self, "order_amp_spin"):
            return
        corrected = np.asarray(self.intensity, dtype=float) - self.order_background_spin.value()
        corrected = corrected[np.isfinite(corrected)]
        if corrected.size == 0:
            return
        amplitude = float(np.nanmax(corrected))
        self.set_order_slider_value_and_limits(
            self.order_amp_spin,
            amplitude,
            amplitude - 10.0,
            max(300.0, amplitude),
        )

    def schedule_order_fit(self):
        if not self.is_order_mode() or self.current_file is None:
            return
        self._order_fit_timer.start(350)

    def order_parameter_changed(self, *args):
        self.update_order_preview()
        self.schedule_order_fit()

    def update_order_preview(self, *args):
        if not self.is_order_mode():
            return
        if self.azimuth is None or self.intensity is None or self.current_file is None:
            self.clear_graph_coordinates()
            clear_plot_canvas(self.canvas)
            return

        background = self.order_background_spin.value()
        center_deg = self.order_center_spin.value()

        azimuth = self.azimuth.astype(float)
        intensity = self.intensity.astype(float)
        y_corrected = intensity - background
        valid = np.isfinite(azimuth) & np.isfinite(y_corrected)
        x_deg = azimuth[valid]
        y_corrected_plot = y_corrected[valid]
        order = np.argsort(x_deg)
        x_deg = x_deg[order]
        y_corrected_plot = y_corrected_plot[order]

        self.last_order_parameter = None
        self.draw_order_raw_profile(x_deg, y_corrected_plot, center_deg=center_deg)
        self.results_text.setPlainText(
            f"File = {self.current_file.name}\n"
            f"Background = {background:.6g} a.u.\n"
            f"Q = {self.order_q_spin.value():.6g} nm⁻¹\n"
            f"λ = {self.order_wavelength_spin.value():.5g} Å\n"
            f"Center = {center_deg:.3f} °\n"
            "L3/2 fit updates automatically after parameter changes."
        )

    def calculate_order_parameter(self, *args):
        if not self.is_order_mode():
            return

        if self.azimuth is None or self.intensity is None or self.current_file is None:
            self.clear_graph_coordinates()
            clear_plot_canvas(self.canvas)
            return

        background = self.order_background_spin.value()
        q_nm = self.order_q_spin.value()
        q_value = q_nm * 0.1
        center_deg = self.order_center_spin.value()
        center_rad = np.deg2rad(center_deg)
        wavelength_a = self.order_wavelength_spin.value()
        ratio = self.order_ratio_spin.value()

        # Amplitude min/max from spin boxes if possible
        amplitude_min = getattr(self.order_amp_spin, "order_min_spin", None).value() if getattr(self.order_amp_spin, "order_min_spin", None) is not None else 0.0
        amplitude_max = getattr(self.order_amp_spin, "order_max_spin", None).value() if getattr(self.order_amp_spin, "order_max_spin", None) is not None else np.inf
        amplitude_min = float(amplitude_min)
        amplitude_max = float(amplitude_max)
        if amplitude_max <= amplitude_min:
            amplitude_max = amplitude_min + 1e-9
        initial_amplitude = max(amplitude_min, min(amplitude_max, self.order_amp_spin.value()))

        # Width min/max from spin boxes if possible
        width_min_deg = getattr(self.order_width_spin, "order_min_spin", None).value() if getattr(self.order_width_spin, "order_min_spin", None) is not None else 0.01
        width_max_deg = getattr(self.order_width_spin, "order_max_spin", None).value() if getattr(self.order_width_spin, "order_max_spin", None) is not None else 180.0
        width_min_deg = max(0.01, float(width_min_deg))
        width_max_deg = max(width_min_deg + 1e-6, float(width_max_deg))
        initial_width = np.deg2rad(max(width_min_deg, min(width_max_deg, self.order_width_spin.value())))

        # Constant min/max from spin box
        constant_min = self.order_constant_spin.minimum()
        constant_max = self.order_constant_spin.maximum()
        constant_min = float(constant_min)
        constant_max = float(constant_max)
        if constant_max <= constant_min:
            constant_max = constant_min + 1e-9
        initial_constant = max(constant_min, min(constant_max, self.order_constant_spin.value()))

        azimuth = self.azimuth.astype(float)
        intensity = self.intensity.astype(float)
        y_corrected_full = intensity - background
        valid = np.isfinite(azimuth) & np.isfinite(y_corrected_full)
        x_deg = azimuth[valid]
        y_corrected = y_corrected_full[valid]
        order = np.argsort(x_deg)
        x_deg = x_deg[order]
        y_corrected = y_corrected[order]

        if x_deg.size < 5:
            self.results_text.setPlainText("Not enough valid points for reciprocal order fit.")
            self.draw_order_raw_profile(x_deg, y_corrected, center_deg=center_deg)
            return

        fit_min_deg = self.order_fit_min_spin.value() if hasattr(self, "order_fit_min_spin") else 0.0
        fit_max_deg = self.order_fit_max_spin.value() if hasattr(self, "order_fit_max_spin") else 360.0
        if fit_max_deg <= fit_min_deg:
            self.results_text.setPlainText("Fit max must be greater than Fit min for Lorentzian fit.")
            self.draw_order_raw_profile(x_deg, y_corrected, center_deg=center_deg)
            return

        fit_mask = (x_deg >= fit_min_deg) & (x_deg <= fit_max_deg)
        if np.count_nonzero(fit_mask) < 8:
            self.results_text.setPlainText(
                f"Not enough valid points between {fit_min_deg:.3g}° and {fit_max_deg:.3g}° for Lorentzian fit."
            )
            self.draw_order_raw_profile(x_deg, y_corrected, center_deg=center_deg)
            return

        fit_x_plot_deg = x_deg[fit_mask]
        fit_y_corrected = y_corrected[fit_mask]
        x_rad = np.deg2rad(fit_x_plot_deg)

        try:
            best_values, covariance = curve_fit(
                lambda x_values, amplitude, width, constant: reciprocal_lorentzian_intensity(
                    x_values,
                    amplitude,
                    width,
                    constant,
                    q_value,
                    center_rad,
                    wavelength_a,
                ),
                x_rad,
                fit_y_corrected,
                p0=[initial_amplitude, initial_width, initial_constant],
                bounds=(
                    [amplitude_min, np.deg2rad(width_min_deg), constant_min],
                    [amplitude_max, np.deg2rad(width_max_deg), constant_max],
                ),
                maxfev=50000,
            )
            amplitude, width, constant = best_values
            width = abs(width)
            y_fit = reciprocal_lorentzian_intensity(x_rad, amplitude, width, constant, q_value, center_rad, wavelength_a)
            order_parameter, isotropic_ratio = calculate_reciprocal_order_parameter(width, constant, ratio)
        except Exception as error:
            self.last_order_parameter = None
            self.draw_order_raw_profile(x_deg, y_corrected)
            self.results_text.setPlainText(str(error))
            return

        theta = np.arange(0.0, np.pi / 2.0, 0.001)
        distribution = np.asarray([
            reciprocal_lorentzian_distribution(angle, width, constant)
            for angle in theta
        ])
        denominator = quad(
            lambda angle: 4.0 * np.pi * reciprocal_lorentzian_distribution(angle, width, constant) * np.sin(angle),
            0.0,
            np.pi / 2.0,
        )[0]
        if denominator != 0:
            distribution = distribution / denominator

        self.last_order_parameter = {
            "x_deg": azimuth[np.isfinite(azimuth) & np.isfinite(y_corrected_full)],
            "y_corrected": y_corrected_full[np.isfinite(azimuth) & np.isfinite(y_corrected_full)],
            "fit_x_deg": fit_x_plot_deg,
            "fit_y_corrected": fit_y_corrected,
            "y_fit": y_fit,
            "fit_min_deg": fit_min_deg,
            "fit_max_deg": fit_max_deg,
            "amplitude_min": amplitude_min,
            "amplitude_max": amplitude_max,
            "constant_min": constant_min,
            "constant_max": constant_max,
            "width_min_deg": width_min_deg,
            "width_max_deg": width_max_deg,
            "theta_deg": np.rad2deg(theta),
            "distribution": distribution,
            "background": background,
            "q_nm": q_nm,
            "q_a": q_value,
            "center_deg": center_deg,
            "wavelength_a": wavelength_a,
            "ratio": ratio,
            "amplitude": amplitude,
            "width_rad": width,
            "width_deg": np.rad2deg(width),
            "constant": constant,
            "order_parameter": order_parameter,
            "isotropic_ratio": isotropic_ratio,
            "covariance": covariance,
        }

        self.update_order_plot()
        self.results_text.setPlainText(
            f"File = {self.current_file.name}\n"
            f"Fit range = {fit_min_deg:.6g}° to {fit_max_deg:.6g}°\n"
            f"Background = {background:.6g} a.u.\n"
            f"Q = {q_nm:.6g} nm⁻¹ ({q_value:.6g} Å⁻¹ used in fit)\n"
            f"λ = {wavelength_a:.5g} Å\n"
            f"Center = {center_deg:.3f} °\n"
            f"r = {ratio:.6g}\n"
            f"Amplitude = {amplitude:.6g} a.u.\n"
            f"Amplitude bounds = {amplitude_min:.6g} to {amplitude_max:.6g} a.u.\n"
            f"Width = {np.rad2deg(width):.6g} °\n"
            f"Width bounds = {width_min_deg:.6g}° to {width_max_deg:.6g}°\n"
            f"Constant = {constant:.6g} a.u.\n"
            f"Constant bounds = {constant_min:.6g} to {constant_max:.6g} a.u.\n"
            f"f = {isotropic_ratio:.6g}\n"
            f"Order parameter S = {order_parameter:.6g}"
        )

    def draw_order_raw_profile(self, x_deg, y_corrected, center_deg=None):
        ax = self.canvas.ax
        ax.clear()
        ax.set_axis_on()
        if x_deg.size:
            ax.plot(x_deg, y_corrected, "k-", linewidth=1.2, label="Background-corrected data")
        if center_deg is not None:
            ax.axvline(center_deg, color="#777777", linestyle="--", linewidth=1.0, label="Center")
        ax.set_xlabel("ψ / °")
        ax.set_ylabel("Intensity / a.u.")
        ax.set_xlim(0, 360)
        ax.grid(True)
        if self.show_legend_checkbox.isChecked():
            install_selectable_legend(ax, ax.legend(loc="best"))
        self.canvas.draw_idle()

    def update_order_plot(self):
        data = self.last_order_parameter
        if data is None:
            return

        ax = self.canvas.ax
        ax.clear()
        ax.set_axis_on()
        order = np.argsort(data["x_deg"])
        ax.plot(data["x_deg"][order], data["y_corrected"][order], "k-", linewidth=1.2, label="Background-corrected data")

        fit_order = np.argsort(data["fit_x_deg"])
        ax.plot(data["fit_x_deg"][fit_order], data["y_fit"][fit_order], color="#d71920", linewidth=1.6, label="L3/2 fit")

        center_deg = data.get("center_deg")
        if center_deg is not None:
            ax.axvline(center_deg, color="#777777", linestyle="--", linewidth=1.0, label="Center")


        ax.set_xlabel("ψ / °")
        ax.set_ylabel("Intensity / a.u.")
        ax.set_xlim(0, 360)
        ax.grid(True)
        if self.show_legend_checkbox.isChecked():
            install_selectable_legend(ax, ax.legend(loc="best"))
        self.canvas.draw_idle()

    def calculate_anisotropy(self, *args):
        if not self.is_anisotropy_mode():
            return

        if self.current_image is None or self.current_file is None:
            self.clear_graph_coordinates()
            clear_plot_canvas(self.canvas)
            return

        image = self.current_image
        if self.center_x_spin.value() == 0 and self.center_y_spin.value() == 0:
            self.apply_instrument_preset()
        geometry = self.current_anisotropy_geometry()
        xc = geometry["center_x"]
        yc = geometry["center_y"]
        distance = geometry["distance"]
        pixel_x = geometry["pixel_x"]
        pixel_y = geometry["pixel_y"]
        wavelength = geometry["wavelength"]

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
            self.last_anisotropy = None
            self.clear_graph_coordinates()
            clear_plot_canvas(self.canvas)
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
        preserve_view = len(ax.lines) > 0
        ax.clear()
        ax.set_axis_on()

        plot_mode = self.anisotropy_plot_mode.currentText()
        q_display = q * self.q_display_factor()
        x_values = q_display
        y_iv = iv
        y_ih = ih
        x_label = self.q_axis_label()
        y_label = "Intensity / a.u."
        x_scale = "linear"
        y_scale = "linear"

        if plot_mode == "Kratky":
            y_iv = iv * q_display ** 2
            y_ih = ih * q_display ** 2
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
        if self.use_q_range.isChecked():
            q_min = min(self.q_min_filter.value(), self.q_max_filter.value())
            q_max = max(self.q_min_filter.value(), self.q_max_filter.value())
            if q_max > q_min:
                if x_scale == "log":
                    positive_x = x_values[np.isfinite(x_values) & (x_values > 0)]
                    lower = q_min * self.q_display_factor() if q_min > 0 else (float(np.nanmin(positive_x)) if positive_x.size else None)
                    upper = q_max * self.q_display_factor()
                    if lower is not None and upper > lower:
                        ax.set_xlim(lower, upper)
                else:
                    ax.set_xlim(q_min * self.q_display_factor(), q_max * self.q_display_factor())
        ax.grid(True, which="both")
        install_selectable_legend(ax, ax.legend(loc="best"))
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

    def draw_raw_azimuthal_profile(self, baseline=None, peak=None, window=None):
        if self.azimuth is None or self.intensity is None:
            return

        ax = self.canvas.ax
        ax.clear()
        ax.set_axis_on()
        ax.plot(self.azimuth, self.intensity, "k-", linewidth=1.2, label="Raw data")

        if baseline is not None:
            ax.plot(self.azimuth, baseline, "r--", linewidth=1.3, label="Baseline")
        if peak is not None:
            ax.axvline(peak, color="blue", linestyle="--", linewidth=1, label="Peak centre")
        if peak is not None and window is not None:
            ax.axvspan(peak - window / 2, peak + window / 2, color="#4a90e2", alpha=0.12, label="Fit window")

        ax.set_title("")
        ax.set_xlabel("ψ / °")
        ax.set_ylabel("Intensity / a.u.")
        ax.set_xlim(0, 360)
        ax.grid(True)
        install_selectable_legend(ax, ax.legend(loc="best"))
        self.canvas.draw_idle()

    def update_plot(self, azimuth, intensity, baseline, peak, amplitude, sigma, fwhm, pi, hermans_corr):
        ax = self.canvas.ax
        ax.clear()
        ax.set_axis_on()

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
        install_selectable_legend(ax, ax.legend(loc="best"))
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
        if self.fit_mode_selector.currentIndex() != 0:
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

    def save_order_fit(self):
        if self.current_file is None:
            QMessageBox.warning(self, "Save unavailable", "No file is currently loaded.")
            return

        if self.last_order_parameter is None:
            self.calculate_order_parameter()

        if self.last_order_parameter is None:
            QMessageBox.warning(self, "Save unavailable", "No reciprocal order fit is currently available.")
            return

        data = self.last_order_parameter
        subtracted_file = self.current_file.parent / f"{self.current_file.stem}_order_subtracted.dat"
        fit_file = self.current_file.parent / f"{self.current_file.stem}_order_lorentzian_fit.dat"

        subtracted_output = np.column_stack([
            data["x_deg"],
            data["y_corrected"],
        ])
        fit_output = np.column_stack([
            data["fit_x_deg"],
            data["y_fit"],
        ])

        with open(subtracted_file, "w", encoding="utf-8") as file:
            file.write("# Background-subtracted azimuthal profile saved from LRPhoton\n")
            file.write(f"# Source file: {self.current_file}\n")
            file.write(f"# Background = {data['background']:.10g} a.u.\n")
            file.write("# Columns: angle_deg subtracted_intensity\n")
            np.savetxt(file, subtracted_output, fmt="%.10g %.10e")

        with open(fit_file, "w", encoding="utf-8") as file:
            file.write("# Reciprocal L3/2 Lorentzian fit saved from LRPhoton\n")
            file.write(f"# Source file: {self.current_file}\n")
            file.write(f"# Background = {data['background']:.10g} a.u.\n")
            file.write(f"# Fit min = {data.get('fit_min_deg', 0.0):.10g} deg\n")
            file.write(f"# Fit max = {data.get('fit_max_deg', 360.0):.10g} deg\n")
            file.write(f"# Amplitude min = {data.get('amplitude_min', 0.0):.10g} a.u.\n")
            file.write(f"# Amplitude max = {data.get('amplitude_max', np.inf):.10g} a.u.\n")
            file.write(f"# Width min = {data.get('width_min_deg', 0.01):.10g} deg\n")
            file.write(f"# Width max = {data.get('width_max_deg', 180.0):.10g} deg\n")
            file.write(f"# Constant min = {data.get('constant_min', -np.inf):.10g} a.u.\n")
            file.write(f"# Constant max = {data.get('constant_max', np.inf):.10g} a.u.\n")
            file.write(f"# Q = {data['q_nm']:.10g} nm^-1\n")
            file.write(f"# Q used in fit = {data['q_a']:.10g} A^-1\n")
            file.write(f"# Center = {data['center_deg']:.10g} deg\n")
            file.write(f"# Wavelength = {data['wavelength_a']:.10g} A\n")
            file.write(f"# r = {data['ratio']:.10g}\n")
            file.write(f"# Amplitude = {data['amplitude']:.10g} a.u.\n")
            file.write(f"# Width = {data['width_deg']:.10g} deg\n")
            file.write(f"# Constant = {data['constant']:.10g} a.u.\n")
            file.write(f"# f = {data['isotropic_ratio']:.10g}\n")
            file.write(f"# S = {data['order_parameter']:.10g}\n")
            file.write("# Columns: angle_deg lorentzian_fit\n")
            np.savetxt(file, fit_output, fmt="%.10g %.10e")

        QMessageBox.information(
            self,
            "Order files saved",
            f"Files saved:\n{subtracted_file}\n{fit_file}",
        )

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
