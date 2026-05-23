import json
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
    QFrame,
    QComboBox,
    QSlider,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from .instrument_presets import (
    ID13_DEFAULT_CENTER_X,
    ID13_DEFAULT_CENTER_Y,
    ID13_DEFAULT_DISTANCE_M,
    ID13_DEFAULT_PIXEL_MM,
    ID13_DEFAULT_WAVELENGTH_A,
)
from .file_ratings import file_path_from_item, install_file_rating_menu, set_item_file_path
from .ui_style import (
    BLOCK_SPACING,
    FILE_BROWSER_WIDTH,
    FRAME_BUTTON_WIDTH,
    FRAME_COUNTER_WIDTH,
    FRAME_NAV_SPACING,
    FRAME_SPIN_WIDTH,
    GROUP_BOX_MARGINS,
    GROUP_BOX_STYLE,
    apply_plot_display_style,
    clear_plot_canvas,
    finalize_plot_canvas,
    make_plot_legend,
    make_matplotlib_toolbar_block,
    PAGE_MARGINS,
    PANEL_MARGINS,
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


ID02_DEFAULT_CENTER_X = 914.4
ID02_DEFAULT_CENTER_Y = 996.5
ID02_DEFAULT_DISTANCE_M = 10.0002
ID02_DEFAULT_PIXEL_MM = 0.075
ID02_DEFAULT_WAVELENGTH_A = 1.01402
CENTER_X_KEYS = ("Center_1", "center_1", "CenterX", "center_x", "BeamCenterX", "Beam_x", "beam_x")
CENTER_Y_KEYS = ("Center_2", "center_2", "CenterY", "center_y", "BeamCenterY", "Beam_y", "beam_y")
ID13_PYFAI_CONFIG = {
    "application": "pyfai-integrate",
    "version": 5,
    "poni": {
        "poni_version": 2.1,
        "dist": 0.2635206714029405,
        "poni1": 0.0982514482559307,
        "poni2": 0.09702116862200875,
        "rot1": 0.0,
        "rot2": 0.0,
        "rot3": 0.0,
        "detector": "Eiger4M",
        "detector_config": {
            "orientation": 3,
        },
        "wavelength": 8.265613228880018e-11,
    },
    "nbpt_rad": 1400,
    "nbpt_azim": 360,
    "unit": "q_nm^-1",
    "chi_discontinuity_at_0": False,
    "polarization_description": [
        0.99,
        0.0,
    ],
    "normalization_factor": 1.0,
    "val_dummy": None,
    "delta_dummy": None,
    "correct_solid_angle": True,
    "dark_current": None,
    "flat_field": None,
    "mask_file": "fabio:///gpfs/gb/data/visitor/sc5729/id13/20251202/PROCESSED_DATA/mask_udetx_m800_.edf",
    "error_model": "no",
    "method": [
        "bbox",
        "csr",
        "opencl",
    ],
    "opencl_device": "gpu",
    "azimuth_range": None,
    "radial_range": None,
    "integrator_class": "AzimuthalIntegrator",
    "integrator_method": None,
    "extra_options": None,
    "monitor_name": None,
    "shape": None,
}


# ============================================================
# ==================== WAVELENGTH UTILS ======================
# ============================================================

def wavelength_to_nm(value: float):
    """
    Convert wavelength to nm with automatic unit detection.

    Typical cases:
    - EDF/H5 header in meters: 8.26563e-11 m -> 0.0826563 nm
    - Interface value in Å: 0.8265613228880018 Å -> 0.08265613228880018 nm
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


def load_id13_pyfai_config():
    """Return the embedded ID13 pyFAI configuration without reading an external file."""
    fallback = {
        "poni": {
            "dist": ID13_DEFAULT_DISTANCE_M,
            "poni1": ID13_DEFAULT_CENTER_Y * ID13_DEFAULT_PIXEL_MM * 1e-3,
            "poni2": ID13_DEFAULT_CENTER_X * ID13_DEFAULT_PIXEL_MM * 1e-3,
            "rot1": 0.0,
            "rot2": 0.0,
            "rot3": 0.0,
            "wavelength": ID13_DEFAULT_WAVELENGTH_A * 1e-10,
        },
        "nbpt_rad": 1400,
        "unit": "q_nm^-1",
        "polarization_description": [0.99, 0.0],
        "correct_solid_angle": True,
        "radial_range": None,
        "azimuth_range": None,
    }

    config = dict(ID13_PYFAI_CONFIG)
    merged = dict(fallback)
    merged.update(config)
    merged["poni"] = {**fallback["poni"], **config.get("poni", {})}
    return merged


def id13_pyfai_q_map(image_shape, config):
    """
    Compute the q map from the pyFAI PONI geometry used by the ID13 workflow.

    The provided ID13 config has no detector rotations, so the workflow q vector
    reduces to pyFAI's PONI metric geometry in detector space.
    """
    poni = config["poni"]
    distance_m = float(poni["dist"])
    poni1_m = float(poni["poni1"])
    poni2_m = float(poni["poni2"])
    wavelength_nm = wavelength_to_nm(float(poni["wavelength"]))

    ny, nx = image_shape
    y, x = np.indices((ny, nx))
    pixel_m = ID13_DEFAULT_PIXEL_MM * 1e-3
    pixel1_m = float(config.get("pixel1_m", pixel_m))
    pixel2_m = float(config.get("pixel2_m", pixel_m))

    d1_m = y * pixel1_m - poni1_m
    d2_m = x * pixel2_m - poni2_m
    r_m = np.sqrt(d1_m ** 2 + d2_m ** 2)
    two_theta = np.arctan2(r_m, distance_m)
    q = (4.0 * np.pi / wavelength_nm) * np.sin(two_theta / 2.0)
    chi = np.arctan2(d1_m, d2_m)
    return q, two_theta, chi


def id13_solid_angle_correction(two_theta):
    # pyFAI's flat-detector solid angle term is normalized to 1 at the PONI.
    return np.cos(two_theta) ** 3


def id13_polarization_correction(two_theta, chi, config):
    description = config.get("polarization_description")
    if not description:
        return 1.0

    factor = float(description[0])
    axis_offset = float(description[1]) if len(description) > 1 else 0.0
    sin_tth = np.sin(two_theta)
    cos_tth_sq = np.cos(two_theta) ** 2
    cos_2chi = np.cos(2.0 * (chi - axis_offset))
    correction = 0.5 * (1.0 + cos_tth_sq - factor * cos_2chi * sin_tth ** 2)
    return np.clip(correction, 1e-12, None)


def id13_pyfai_like_average(image, q_min, q_max, sector_min=0, sector_max=360):
    """
    Local comparison profile based on the ID13 pyFAI JSON.

    It mirrors the relevant workflow options available in config_udetx_m800.json:
    PONI geometry, q_nm^-1 unit, nbpt_rad, optional radial range, solid-angle
    correction and polarization correction. Pixel splitting/OpenCL are pyFAI
    implementation details and are approximated here with center-of-pixel bins.
    """
    config = load_id13_pyfai_config()
    exact = id13_pyfai_exact_average(image, q_min, q_max, sector_min, sector_max, config)
    if exact is not None:
        return exact

    if config.get("unit") != "q_nm^-1":
        raise ValueError(f"Unsupported ID13 pyFAI unit: {config.get('unit')}")

    q, two_theta, chi = id13_pyfai_q_map(image.shape, config)
    intensity = image.astype(np.float64)

    if config.get("correct_solid_angle", False):
        intensity = intensity / np.clip(id13_solid_angle_correction(two_theta), 1e-12, None)

    intensity = intensity / id13_polarization_correction(two_theta, chi, config)

    psi = (np.degrees(chi) + 360.0) % 360.0
    sector_min = sector_min % 360.0
    sector_max = sector_max % 360.0
    if abs((sector_max - sector_min) % 360.0) < 1e-9:
        sector_mask = np.ones_like(psi, dtype=bool)
    elif sector_min <= sector_max:
        sector_mask = (psi >= sector_min) & (psi <= sector_max)
    else:
        sector_mask = (psi >= sector_min) | (psi <= sector_max)

    radial_range = config.get("radial_range")
    if radial_range and len(radial_range) == 2:
        q_min_eff, q_max_eff = map(float, radial_range)
    else:
        q_min_eff, q_max_eff = float(q_min), float(q_max)

    valid = np.isfinite(q) & (q > 0) & np.isfinite(intensity) & (intensity < 4e9) & (intensity > 0) & sector_mask
    if q_min_eff > 0:
        valid &= q >= q_min_eff
    if q_max_eff > 0:
        valid &= q <= q_max_eff

    q_values = q[valid]
    i_values = intensity[valid]
    if q_values.size == 0:
        raise ValueError("No valid pixel found for the ID13 pyFAI comparison.")

    q_min_eff = q_min_eff if q_min_eff > 0 else float(np.nanmin(q_values))
    q_max_eff = q_max_eff if q_max_eff > 0 else float(np.nanmax(q_values))
    if q_max_eff <= q_min_eff:
        raise ValueError("ID13 pyFAI q max must be greater than q min.")

    n_bins = int(config.get("nbpt_rad", 1400))
    edges = np.linspace(q_min_eff, q_max_eff, n_bins + 1)
    q_axis = 0.5 * (edges[:-1] + edges[1:])
    sums, _ = np.histogram(q_values, bins=edges, weights=i_values)
    counts, _ = np.histogram(q_values, bins=edges)

    with np.errstate(invalid="ignore", divide="ignore"):
        averaged = sums / counts

    valid_bins = (counts > 0) & np.isfinite(averaged) & (averaged > 0)
    return q_axis[valid_bins], averaged[valid_bins], counts[valid_bins], config


def id13_pyfai_exact_average(image, q_min, q_max, sector_min, sector_max, config):
    """Use pyFAI itself when it is installed; otherwise let the local fallback run."""
    try:
        import pyFAI
    except Exception:
        return None

    # No external ID13 config file is used here; the embedded config is already
    # available through load_id13_pyfai_config(). If pyFAI is installed, this
    # function could be extended to build an integrator from the embedded config.
    return None


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
        self.fig = Figure(dpi=150)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.fig.subplots_adjust(left=0.12, right=0.98, top=0.92, bottom=0.18)

        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(620, 420)
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
        self.ax.set_axis_off()
        self.ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
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
        self.display_data_min = 0.0
        self.display_data_max = 1.0
        self.q_map = None
        self.last_xc = None
        self.last_yc = None

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
                            q_value = self.q_map[y_index, x_index]
                            if np.isfinite(q_value):
                                q_text = f"{q_value:.6g} nm⁻¹"

                        if self.last_xc is not None and self.last_yc is not None:
                            dx = (x_index + 1) - self.last_xc
                            dy = (y_index + 1) - self.last_yc
                            psi = np.degrees(np.arctan2(dy, dx)) % 360.0
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

    def show_image(self, image, xc=None, yc=None, mask=None):
        current_xlim = self.ax.get_xlim()
        current_ylim = self.ax.get_ylim()
        had_image = len(self.ax.images) > 0
        self.raw_image = image
        self.last_xc = xc
        self.last_yc = yc

        self.ax.clear()
        self.ax.set_axis_off()
        self.ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        display = image.astype(np.float64).copy()
        display[~np.isfinite(display)] = np.nan
        display[display < 0] = np.nan

        with np.errstate(invalid="ignore", divide="ignore"):
            display = np.log10(display + 1)

        finite_display = display[np.isfinite(display)]
        if finite_display.size > 0:
            self.display_data_min = float(np.nanmin(finite_display))
            self.display_data_max = float(np.nanmax(finite_display))
        else:
            self.display_data_min = 0.0
            self.display_data_max = 1.0

        if self.display_vmin is None or self.display_vmax is None:
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

        self.ax.set_axis_off()
        self.ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
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
        self.last_comparison_results = {}
        self.h5_dataset_name = None
        self.h5_frame_axis = None
        self.h5_n_frames = 1
        self._syncing_folder = False
        self._changing_h5_frame = False
        self._syncing_frame_controls = False

        self.build_ui()
        self.refresh_files()
        self.set_controls_enabled(False)

    def update_image_intensity_limits(self):
        if not hasattr(self, "image_canvas") or self.image_canvas.raw_image is None:
            return

        data_min = self.image_canvas.display_data_min
        data_max = self.image_canvas.display_data_max
        span = data_max - data_min
        if span <= 0:
            return

        min_pos = self.image_vmin_slider.value()
        max_pos = self.image_vmax_slider.value()

        if min_pos >= max_pos:
            sender = self.sender()
            if sender is self.image_vmin_slider:
                max_pos = min(1000, min_pos + 1)
                self.image_vmax_slider.blockSignals(True)
                self.image_vmax_slider.setValue(max_pos)
                self.image_vmax_slider.blockSignals(False)
            else:
                min_pos = max(0, max_pos - 1)
                self.image_vmin_label.setVisible(True)
                self.image_vmax_label.setVisible(True)
                self.image_vmin_slider.setVisible(True)
                self.image_vmax_slider.setVisible(True)
                self.image_vmin_label.setVisible(True)
                self.image_vmax_label.setVisible(True)
                self.image_vmin_slider.setVisible(True)
                self.image_vmax_slider.setVisible(True)
                self.image_vmin_slider.blockSignals(True)
                self.image_vmin_slider.setValue(min_pos)
                self.image_vmin_slider.blockSignals(False)

        vmin = data_min + span * min_pos / 1000.0
        vmax = data_min + span * max_pos / 1000.0

        self.image_canvas.display_vmin = vmin
        self.image_canvas.display_vmax = vmax
        self.image_vmin_label.setText(f"Min: {vmin:.3g}")
        self.image_vmax_label.setText(f"Max: {vmax:.3g}")

        self.image_canvas.show_image(
            self.image_canvas.raw_image,
            self.center_x.value(),
            self.center_y.value(),
            None,
        )

    def sync_image_intensity_sliders(self):
        data_min = self.image_canvas.display_data_min
        data_max = self.image_canvas.display_data_max
        span = data_max - data_min
        if span <= 0 or self.image_canvas.display_vmin is None or self.image_canvas.display_vmax is None:
            self.image_vmin_label.setText("Min: -")
            self.image_vmax_label.setText("Max: -")
            return

        min_pos = int(round((self.image_canvas.display_vmin - data_min) / span * 1000))
        max_pos = int(round((self.image_canvas.display_vmax - data_min) / span * 1000))
        min_pos = max(0, min(1000, min_pos))
        max_pos = max(0, min(1000, max_pos))

        self.image_vmin_slider.blockSignals(True)
        self.image_vmax_slider.blockSignals(True)
        self.image_vmin_slider.setValue(min_pos)
        self.image_vmax_slider.setValue(max_pos)
        self.image_vmin_slider.blockSignals(False)
        self.image_vmax_slider.blockSignals(False)

        self.image_vmin_label.setText(f"Min: {self.image_canvas.display_vmin:.3g}")
        self.image_vmax_label.setText(f"Max: {self.image_canvas.display_vmax:.3g}")

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
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(*PANEL_MARGINS)
        left_layout.setSpacing(BLOCK_SPACING)
        content_layout.addWidget(left_panel, stretch=0)

        right_panel = QWidget()
        right_layout = QHBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(BLOCK_SPACING)
        content_layout.addWidget(right_panel, stretch=1)

        # ============================================================
        # COLUMN 2: I(q) GRAPH
        # ============================================================
        graph_panel = QWidget()
        graph_layout = QVBoxLayout(graph_panel)
        graph_layout.setContentsMargins(0, 0, 0, 0)
        graph_layout.setSpacing(4)
        right_layout.addWidget(graph_panel, stretch=1)

        # ============================================================
        # COLUMN 3: PARAMETERS + SELECTED AREA (IMAGE)
        # ============================================================
        right_side_panel = QWidget()
        right_side_panel.setFixedWidth(FILE_BROWSER_WIDTH)
        right_side_layout = QVBoxLayout(right_side_panel)
        right_side_layout.setContentsMargins(0, 0, 0, 0)
        right_side_layout.setSpacing(BLOCK_SPACING)
        right_layout.addWidget(right_side_panel, stretch=0)

        image_box = QGroupBox("Selected area")
        image_layout = QVBoxLayout(image_box)
        image_layout.setContentsMargins(*GROUP_BOX_MARGINS)
        right_side_layout.addWidget(image_box, stretch=1)

        file_box = QGroupBox("File browser")
        file_box.setMinimumHeight(220)
        file_box.setStyleSheet(GROUP_BOX_STYLE)

        file_layout = QVBoxLayout(file_box)
        file_layout.setContentsMargins(*GROUP_BOX_MARGINS)
        file_layout.setSpacing(6)

        left_layout.addWidget(file_box, stretch=1)

        self.folder_path = QLineEdit(str(self.current_folder))
        self.folder_path.returnPressed.connect(self.refresh_files)
        file_layout.addWidget(self.folder_path)

        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.choose_folder)
        file_layout.addWidget(self.browse_button)

        filters_layout = QGridLayout()

        self.extensions_filter = QLineEdit("*.edf *.h5")
        self.name_filter = QLineEdit("**")

        self.extensions_filter.textChanged.connect(self.refresh_files)
        self.name_filter.textChanged.connect(self.refresh_files)

        filters_layout.addWidget(QLabel("Name:"), 0, 0)
        filters_layout.addWidget(self.name_filter, 0, 1)

        filters_layout.addWidget(QLabel("Extensions:"), 1, 0)
        filters_layout.addWidget(self.extensions_filter, 1, 1)

        file_layout.addLayout(filters_layout)

        self.show_subfolders_checkbox = QCheckBox("Show subfolders")
        self.show_subfolders_checkbox.setChecked(False)
        self.show_subfolders_checkbox.stateChanged.connect(self.refresh_files)
        file_layout.addWidget(self.show_subfolders_checkbox)

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_files)
        file_layout.addWidget(self.refresh_button)

        self.file_list = QListWidget()
        install_file_rating_menu(self.file_list)
        self.file_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.file_list.itemSelectionChanged.connect(self.selection_changed)
        self.file_list.setMinimumHeight(180)

        file_layout.addWidget(self.file_list, stretch=1)

        params_box = QGroupBox("Radial parameters")
        params_layout = QVBoxLayout(params_box)
        params_layout.setContentsMargins(*GROUP_BOX_MARGINS)
        params_layout.setSpacing(4)
        right_side_layout.insertWidget(0, params_box, stretch=0)

        preset_layout = QHBoxLayout()
        self.btn_xenocs = QPushButton("XENOCS")
        self.btn_id02 = QPushButton("ID02")
        self.btn_id13 = QPushButton("ID13")
        self.btn_custom = QPushButton("Custom")
        self.q_manual_button = QPushButton("+")
        self.q_manual_button.clicked.connect(self.open_geometry_dialog)
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
        params_layout.addLayout(preset_layout)

        form = QGridLayout()
        form.setVerticalSpacing(6)
        form.setHorizontalSpacing(10)
        form.setContentsMargins(0, 0, 0, 0)
        form.setColumnStretch(0, 0)
        form.setColumnStretch(1, 1)
        self.center_x = self.double_spin(0, decimals=13)
        self.center_y = self.double_spin(0, decimals=13)
        self.distance = self.double_spin(0, decimals=16, minimum=0)
        self.pixel_x = self.double_spin(0.075000, decimals=6, minimum=0)
        self.pixel_y = self.double_spin(0.075000, decimals=6, minimum=0)
        self.wavelength = self.double_spin(0, decimals=16, minimum=0)
        self.use_id13_pyfai_comparison = QCheckBox("ID13 pyFAI comparison")
        self.use_id13_pyfai_comparison.setChecked(False)
        self.use_id13_pyfai_comparison.setVisible(False)

        self.use_sector = QCheckBox("Use azimuthal sector")
        self.use_sector.setChecked(False)
        self.use_sector.stateChanged.connect(self.update_mask_parameter_state)
        self.sector_min = self.double_spin(0, decimals=3, minimum=-360)
        self.sector_max = self.double_spin(360, decimals=3, minimum=-360)
        self.n_bins = QSpinBox()
        self.n_bins.setRange(10, 10000)
        self.n_bins.setValue(200)
        self.n_bins.setMinimumWidth(130)
        self.n_bins.setFixedHeight(24)
        self.plot_mode = QComboBox()
        self.plot_mode.addItems(["linear linear", "linear log", "log log", "log linear", "Kratky", "2θ linear", "2θ log"])
        self.plot_mode.setCurrentText("log log")
        self.plot_mode.setFixedWidth(120)
        self.plot_mode.currentTextChanged.connect(self.update_plot_mode)
        self.show_legend = QCheckBox("Legend")
        self.show_legend.setChecked(True)
        self.show_legend.stateChanged.connect(self.update_legend_visibility)

        self.frame_label = QLabel("H5 frame:")
        self.frame_spin = QSpinBox()
        self.frame_spin.setRange(1, 1)
        self.frame_spin.setValue(1)
        self.frame_spin.setEnabled(False)
        self.frame_label.hide()
        self.frame_spin.hide()
        self.frame_spin.valueChanged.connect(self.update_selected_h5_frame)

        form.addWidget(self.use_sector, 0, 0, 1, 2)
        form.addWidget(QLabel("Sector min ψ (°):"), 1, 0)
        form.addWidget(self.sector_min, 1, 1)
        form.addWidget(QLabel("Sector max ψ (°):"), 2, 0)
        form.addWidget(self.sector_max, 2, 1)

        form.addWidget(QLabel("Bins:"), 3, 0)
        form.addWidget(self.n_bins, 3, 1)
        params_layout.addLayout(form)

        self.integrate_button = QPushButton("Integrate I(q)")
        self.integrate_button.clicked.connect(self.integrate_selected_files)
        params_layout.addWidget(self.integrate_button)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setVisible(False)

        self.canvas = PlotCanvas()
        self.canvas.setContentsMargins(0, 0, 0, 0)
        clear_plot_canvas(self.canvas)
        self.toolbar = NavigationToolbar(self.canvas, self)
        toolbar_box, self.toolbar_extra_layout, self.save_graph_button = make_matplotlib_toolbar_block(
            self,
            "I(q) graph",
            self.toolbar,
            option_widgets=[
                self.plot_mode,
                self.show_legend,
            ],
            save_callback=self.save_results,
            save_tooltip="Save graph (.dat, .png or .tiff)",
            toolbar_width=320,
        )
        graph_layout.addWidget(toolbar_box, alignment=Qt.AlignTop)

        self.graph_coordinate_label = QLabel("q = - | I = -")
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

        graph_layout.addWidget(self.canvas, stretch=1)
        graph_layout.addWidget(self.graph_coordinate_label, stretch=0)

        # ============================================================
        # IMAGE CANVAS
        # ============================================================
        self.image_canvas = ImageCanvas()
        self.image_coordinate_label = QLabel("ψ = - | q = - | I = -")
        self.image_coordinate_label.setMinimumHeight(28)
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
        
        image_limits_layout = QGridLayout()
        image_limits_layout.setContentsMargins(0, 0, 0, 0)
        image_limits_layout.setHorizontalSpacing(6)
        image_limits_layout.setVerticalSpacing(2)

        self.image_vmin_label = QLabel("Min: -")
        self.image_vmax_label = QLabel("Max: -")
        self.image_vmin_label.setAlignment(Qt.AlignCenter)
        self.image_vmax_label.setAlignment(Qt.AlignCenter)

        self.image_vmin_slider = QSlider(Qt.Horizontal)
        self.image_vmax_slider = QSlider(Qt.Horizontal)
        self.image_vmin_slider.setRange(0, 1000)
        self.image_vmax_slider.setRange(0, 1000)
        self.image_vmin_slider.setValue(0)
        self.image_vmax_slider.setValue(1000)

        image_limits_layout.addWidget(self.image_vmin_label, 0, 0)
        image_limits_layout.addWidget(self.image_vmin_slider, 0, 1)
        image_limits_layout.addWidget(self.image_vmax_label, 1, 0)
        image_limits_layout.addWidget(self.image_vmax_slider, 1, 1)

        image_layout.addLayout(image_limits_layout)

        self.image_vmin_slider.valueChanged.connect(self.update_image_intensity_limits)
        self.image_vmax_slider.valueChanged.connect(self.update_image_intensity_limits)

        self.canvas.mpl_connect("button_press_event", self.on_graph_right_click)
        self.canvas.mpl_connect("motion_notify_event", self.update_graph_coordinates)
        self.canvas.mpl_connect("axes_leave_event", self.clear_graph_coordinates)

        # ============================================================
        # FRAME NAVIGATION (at bottom)
        # ============================================================
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

        self.btn_xenocs.clicked.connect(lambda: self.set_instrument_mode("XENOCS"))
        self.btn_id02.clicked.connect(lambda: self.set_instrument_mode("ID02"))
        self.btn_id13.clicked.connect(lambda: self.set_instrument_mode("ID13"))
        self.btn_custom.clicked.connect(self.open_geometry_dialog)
        self.update_mask_parameter_state()

    def update_mask_parameter_state(self):
        use_sector = self.use_sector.isChecked()
        self.sector_min.setEnabled(use_sector)
        self.sector_max.setEnabled(use_sector)

    def double_spin(self, value, decimals=3, minimum=-1e9):
        spin = QDoubleSpinBox()
        spin.setDecimals(decimals)
        spin.setRange(minimum, 1e12)
        spin.setValue(value)
        spin.setFixedHeight(24)
        spin.setMinimumWidth(130)
        return spin

    def set_controls_enabled(self, enabled):
        for widget in [
            self.center_x, self.center_y, self.distance, self.pixel_x, self.pixel_y,
            self.wavelength, self.frame_spin, self.frame_start_spin, self.frame_end_spin,
            self.frame_slider, self.prev_frame_button, self.next_frame_button,
            self.use_sector,
            self.sector_min, self.sector_max,
            self.n_bins, self.plot_mode, self.show_legend, self.integrate_button,
            self.image_vmin_label, self.image_vmax_label,
            self.image_vmin_slider, self.image_vmax_slider,
        ]:
            widget.setEnabled(enabled)

        for widget in [
            self.btn_xenocs,
            self.btn_id02,
            self.btn_id13,
            self.btn_custom,
            self.q_manual_button,
        ]:
            widget.setEnabled(True)

        if hasattr(self, "save_graph_button"):
            self.save_graph_button.setEnabled(enabled)

        self.plot_mode.setCurrentText("log log")
        self.update_frame_selector_visibility()
        self.update_mask_parameter_state()

        if not enabled:
            self.sector_min.setEnabled(False)
            self.sector_max.setEnabled(False)
            self.prev_frame_button.setEnabled(False)
            self.next_frame_button.setEnabled(False)

    def update_frame_selector_visibility(self):
        is_multiframe_h5 = self.h5_n_frames > 1
        self.frame_label.setVisible(False)
        self.frame_spin.setVisible(False)
        self.frame_spin.setEnabled(is_multiframe_h5)

        self.frame_start_spin.setVisible(True)
        self.frame_end_spin.setVisible(True)
        self.frame_slider.setVisible(True)
        self.prev_frame_button.setVisible(True)
        self.next_frame_button.setVisible(True)
        self.frame_counter_label.setVisible(True)

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

        self.frame_spin.blockSignals(True)
        self.frame_spin.setValue(value)
        self.frame_spin.blockSignals(False)

        self.update_frame_counter()

        if self.h5_n_frames > 1 and self.selected_files():
            self._changing_h5_frame = True
            try:
                self.integrate_selected_files()
            finally:
                self._changing_h5_frame = False

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
        else:
            self.update_frame_counter()

    def update_frame_counter(self):
        current = self.frame_spin.value()
        total = max(1, self.h5_n_frames)
        self.frame_counter_label.setText(f"{current} / {total}")
        can_navigate = self.h5_n_frames > 1
        self.frame_start_spin.setEnabled(can_navigate)
        self.frame_end_spin.setEnabled(can_navigate)
        self.frame_slider.setEnabled(can_navigate)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(current)
        self.frame_slider.blockSignals(False)
        self.prev_frame_button.setEnabled(can_navigate and current > self.frame_start_spin.value())
        self.next_frame_button.setEnabled(can_navigate and current < self.frame_end_spin.value())

    def previous_frame(self):
        self.frame_slider.setValue(max(self.frame_start_spin.value(), self.frame_slider.value() - 1))

    def next_frame(self):
        self.frame_slider.setValue(min(self.frame_end_spin.value(), self.frame_slider.value() + 1))

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
        search_method = folder.rglob if getattr(self, "show_subfolders_checkbox", None) and self.show_subfolders_checkbox.isChecked() else folder.glob
        for pattern in patterns:
            files.extend(search_method(pattern))

        from fnmatch import fnmatch
        files = sorted(set(files))
        files = [file for file in files if fnmatch(file.name, name_filter)]

        self.file_list.clear()
        for file in files:
            display_name = str(file.relative_to(folder)) if getattr(self, "show_subfolders_checkbox", None) and self.show_subfolders_checkbox.isChecked() else file.name
            self.file_list.addItem(display_name)
            item = self.file_list.item(self.file_list.count() - 1)
            set_item_file_path(item, file)

        selected = self.selected_files()
        self.set_controls_enabled(bool(selected))
        if not selected:
            self.last_results = {}
            self.clear_graph_coordinates()
            clear_plot_canvas(self.canvas)

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
            self.display_selected_file_preview(selected[0])
        else:
            self.configure_frame_navigation(1)
            self.update_frame_selector_visibility()
            self.last_results = {}
            self.clear_graph_coordinates()
            self.image_canvas.raw_image = None
            self.image_canvas.set_q_map(None)
            self.image_coordinate_label.setText("ψ = - | q = - | I = -")
            clear_plot_canvas(self.canvas)
            clear_plot_canvas(self.image_canvas)

    def selected_files(self):
        return [file_path_from_item(item, self.current_folder) for item in self.file_list.selectedItems()]

    def set_instrument_mode(self, mode):
        self.instrument_mode = mode
        self.image_canvas.reset_display_limits()
        buttons = {
            "XENOCS": self.btn_xenocs,
            "ID02": self.btn_id02,
            "ID13": self.btn_id13,
            "Custom": self.btn_custom,
        }
        style_q_geometry_buttons(buttons, mode, self.q_manual_button)

        selected = self.selected_files()
        self.apply_preset_from_file(selected[0] if selected else None)

    def open_geometry_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Geometry")
        layout = QVBoxLayout(dialog)
        form = QFormLayout()

        fields = [
            ("center_x", "Center X", self.center_x),
            ("center_y", "Center Y", self.center_y),
            ("distance", "Distance (m)", self.distance),
            ("pixel_x", "Pixel X (mm)", self.pixel_x),
            ("pixel_y", "Pixel Y (mm)", self.pixel_y),
            ("wavelength", "Wavelength (Å)", self.wavelength),
        ]
        dialog_spins = {}
        for key, label, source in fields:
            spin = self.double_spin(source.value(), decimals=source.decimals(), minimum=source.minimum())
            spin.setFixedWidth(150)
            dialog_spins[key] = spin
            form.addRow(label, spin)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addLayout(form)
        layout.addWidget(buttons)

        if dialog.exec() != QDialog.Accepted:
            return

        self.center_x.setValue(dialog_spins["center_x"].value())
        self.center_y.setValue(dialog_spins["center_y"].value())
        self.distance.setValue(dialog_spins["distance"].value())
        self.pixel_x.setValue(dialog_spins["pixel_x"].value())
        self.pixel_y.setValue(dialog_spins["pixel_y"].value())
        self.wavelength.setValue(dialog_spins["wavelength"].value())
        self.set_instrument_mode("Custom")

    def apply_preset_from_file(self, file_path=None):
        header = {}
        if file_path is not None and self.instrument_mode in ("XENOCS", "ID02", "ID13"):
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
            cx = get_header_float(header, *CENTER_X_KEYS)
            cy = get_header_float(header, *CENTER_Y_KEYS)
            dist = get_header_float(header, "SampleDistance", "sampledistance", "sample_distance")
            px = get_header_float(header, "PSize_1", "psize_1", "PSize_X", "PixelSizeX")
            py = get_header_float(header, "PSize_2", "psize_2", "PSize_Y", "PixelSizeY")
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
            cx = get_header_float(header, *CENTER_X_KEYS)
            cy = get_header_float(header, *CENTER_Y_KEYS)
            dist = get_header_float(header, "SampleDistance", "sampledistance", "sample_distance")
            px = get_header_float(header, "PSize_1", "psize_1", "PSize_X", "PixelSizeX")
            py = get_header_float(header, "PSize_2", "psize_2", "PSize_Y", "PixelSizeY")
            wav = get_header_float(header, "WaveLength", "Wavelength", "wavelength")
            self.center_x.setValue(cx if cx is not None else ID02_DEFAULT_CENTER_X)
            self.center_y.setValue(cy if cy is not None else ID02_DEFAULT_CENTER_Y)
            self.distance.setValue(dist if dist is not None else ID02_DEFAULT_DISTANCE_M)
            self.pixel_x.setValue(px * 1000 if px is not None else ID02_DEFAULT_PIXEL_MM)
            self.pixel_y.setValue(py * 1000 if py is not None else ID02_DEFAULT_PIXEL_MM)
            self.wavelength.setValue(wav * 1e10 if wav is not None else ID02_DEFAULT_WAVELENGTH_A)
            return

        if self.instrument_mode == "ID13":
            self.center_x.setValue(ID13_DEFAULT_CENTER_X)
            self.center_y.setValue(ID13_DEFAULT_CENTER_Y)
            self.distance.setValue(ID13_DEFAULT_DISTANCE_M)
            self.pixel_x.setValue(ID13_DEFAULT_PIXEL_MM)
            self.pixel_y.setValue(ID13_DEFAULT_PIXEL_MM)
            self.wavelength.setValue(ID13_DEFAULT_WAVELENGTH_A)
            return

    def display_selected_file_preview(self, file_path):
        try:
            h5_dataset_name = self.h5_dataset_name if file_path.suffix.lower() in [".h5", ".hdf5"] else None
            h5_frame_index = self.frame_spin.value() - 1 if file_path.suffix.lower() in [".h5", ".hdf5"] else 0
            image, _ = read_image_file(file_path, h5_dataset_name, h5_frame_index)

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

            self.image_canvas.set_q_map(q_map)
            self.image_canvas.show_image(image, self.center_x.value(), self.center_y.value(), mask=None)
            self.sync_image_intensity_sliders()
            self.image_coordinate_label.setText("ψ = - | q = - | I = -")
        except Exception as error:
            self.image_canvas.raw_image = None
            self.image_canvas.set_q_map(None)
            self.image_coordinate_label.setText("ψ = - | q = - | I = -")

    def integrate_selected_files(self):
        files = self.selected_files()
        if not files:
            self.last_results = {}
            self.clear_graph_coordinates()
            clear_plot_canvas(self.canvas)
            return

        preserve_view = self._changing_h5_frame
        ax = self.canvas.ax
        ax.set_axis_on()

        previous_xlim = tuple(ax.get_xlim()) if preserve_view else None
        previous_ylim = tuple(ax.get_ylim()) if preserve_view else None
        previous_xscale = ax.get_xscale() if preserve_view else None
        previous_yscale = ax.get_yscale() if preserve_view else None

        self.last_results = {}
        self.last_comparison_results = {}
        ax.clear()

        messages = []
        for file_path in files:
            try:
                h5_dataset_name = self.h5_dataset_name if file_path.suffix.lower() in [".h5", ".hdf5"] else None
                h5_frame_index = self.frame_spin.value() - 1 if file_path.suffix.lower() in [".h5", ".hdf5"] else 0
                image, _ = read_image_file(file_path, h5_dataset_name, h5_frame_index)
                q_min = 0
                q_max = 0
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

                comparison_message = None
                if False:
                    try:
                        q_id13, intensity_id13, counts_id13, id13_config = id13_pyfai_like_average(
                            image,
                            q_min,
                            q_max,
                            sector_min,
                            sector_max,
                        )
                        id13_label = f"{file_path.stem} ID13 pyFAI"
                        id13_wavelength = id13_config["poni"]["wavelength"]
                        ax.plot(
                            self.make_plot_x(q_id13, id13_wavelength),
                            self.make_plot_y(q_id13, intensity_id13),
                            linewidth=1.1,
                            linestyle="--",
                            color=line.get_color(),
                            label=id13_label,
                        )
                        self.last_comparison_results[id13_label] = (
                            q_id13,
                            intensity_id13,
                            counts_id13,
                            id13_wavelength,
                        )
                        comparison_source = "pyFAI integrate1d" if id13_config.get("used_pyfai") else "local pyFAI-compatible"
                        comparison_message = (
                            f"  ID13 pyFAI comparison = {q_id13.size} bins from embedded config"
                            f" ; q range = {np.nanmin(q_id13):.10g} -> {np.nanmax(q_id13):.10g} nm⁻¹"
                            f" ; {comparison_source} ; solid angle/polarization corrected"
                        )
                    except Exception as comparison_error:
                        comparison_message = f"  ID13 pyFAI comparison error: {comparison_error}"

                if file_path == files[0]:
                    self.image_canvas.set_q_map(q_map)
                    self.image_canvas.show_image(image, self.center_x.value(), self.center_y.value(), mask=mask)
                    self.sync_image_intensity_sliders()
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
                if comparison_message is not None:
                    messages.append(comparison_message)

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

        apply_plot_display_style(ax)
        if self.last_results and self.show_legend.isChecked():
            self.legend = make_plot_legend(ax)
        finalize_plot_canvas(self.canvas)
        self.log_box.setPlainText("\n".join(messages))


    def make_plot_x(self, q, wavelength_value=None):
        mode = self.plot_mode.currentText()
        if mode in ["2θ linear", "2θ log"]:
            return q_nm_to_two_theta_deg(q, self.wavelength.value() if wavelength_value is None else wavelength_value)
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
            elif label in self.last_comparison_results:
                q, intensity, counts, wavelength = self.last_comparison_results[label]
                line.set_xdata(self.make_plot_x(q, wavelength))
                line.set_ydata(self.make_plot_y(q, intensity))

        self.apply_plot_axes()
        apply_plot_display_style(self.canvas.ax)
        self.update_legend_visibility(redraw=False)
        self.canvas.ax.relim()
        self.canvas.ax.autoscale_view()
        finalize_plot_canvas(self.canvas)

    def update_legend_visibility(self, redraw=True):
        legend = self.canvas.ax.get_legend()
        if self.show_legend.isChecked():
            lines = [
                line for line in self.canvas.ax.get_lines()
                if not line.get_label().startswith("_")
            ]
            if lines:
                self.legend = make_plot_legend(self.canvas.ax)
        elif legend is not None:
            legend.remove()
            self.legend = None

        if redraw:
            finalize_plot_canvas(self.canvas)

    def update_id13_comparison(self):
        if self.selected_files() and self.last_results:
            self.integrate_selected_files()


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
        self.legend = make_plot_legend(self.canvas.ax)
        finalize_plot_canvas(self.canvas)

    def ask_text(self, title, label, text):
        from PySide6.QtWidgets import QInputDialog
        return QInputDialog.getText(self, title, label, text=text)

    def save_results(self):
        if not self.last_results:
            QMessageBox.warning(self, "No results", "No radial integration result to save.")
            return

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
