import re
from pathlib import Path

import h5py
import numpy as np

from PySide6.QtCore import Qt, Signal, QTimer, QEvent
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
from .file_ratings import file_path_from_item, install_file_rating_menu, is_file_rated_up, set_item_file_path, should_hide_file_in_browser
from .line_geometry import LineGeometrySelector, line_geometry_to_lrphoton
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
    apply_plot_display_style,
    clear_plot_canvas,
    constrain_image_axes,
    finalize_plot_canvas,
    make_plot_legend,
    make_matplotlib_toolbar_block,
    normalize_decimal_text,
    PAGE_MARGINS,
    PANEL_MARGINS,
    style_q_geometry_buttons,
)


# ============================================================
# FILE TOOLS
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

    match = re.search(r"EDF_HeaderSize\s*[:=]\s*(\d+)", first)
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


def read_h5_first_image(filename: str, frame_index: int = 0):
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

        add_matching_edf_center(header, filename)

        if dataset.ndim == 2:
            image = np.asarray(dataset[...], dtype=np.float64)
            header["Number of frames"] = "1"
        elif dataset.ndim == 3:
            shape = dataset.shape
            frame_axis = int(np.argmin(shape))
            n_frames = int(shape[frame_axis])
            frame_index = int(np.clip(frame_index, 0, n_frames - 1))
            header["Frame axis"] = str(frame_axis)
            header["Number of frames"] = str(n_frames)

            if frame_axis == 0:
                image = np.asarray(dataset[frame_index, :, :], dtype=np.float64)
                header["Displayed frame"] = f"{frame_index} from axis 0"
            elif frame_axis == 1:
                image = np.asarray(dataset[:, frame_index, :], dtype=np.float64)
                header["Displayed frame"] = f"{frame_index} from axis 1"
            else:
                image = np.asarray(dataset[:, :, frame_index], dtype=np.float64)
                header["Displayed frame"] = f"{frame_index} from axis 2"
        else:
            raise ValueError("Only 2D and 3D H5 datasets are supported here.")

    return image, header


def read_image_file(file_path, frame_index: int = 0):
    suffix = Path(file_path).suffix.lower()
    if suffix == ".edf":
        return read_edf_file(file_path)
    if suffix in [".h5", ".hdf5"]:
        return read_h5_first_image(file_path, frame_index=frame_index)
    raise ValueError("Unsupported file format. Please select EDF, H5 or HDF5.")


def get_header_float(header: dict, *names):
    for name in names:
        if name in header:
            try:
                return float(normalize_decimal_text(header[name]))
            except (TypeError, ValueError):
                continue
    return None


def get_header_center_values(header: dict):
    cx = get_header_float(header, "Center_1", "center_1", "CenterX", "center_x")
    cy = get_header_float(header, "Center_2", "center_2", "CenterY", "center_y")

    if cx is not None and cy is not None:
        return cx, cy, "Center_1/Center_2"

    tx = get_header_float(header, "Theoretical_Center_1", "theoretical_center_1")
    ty = get_header_float(header, "Theoretical_Center_2", "theoretical_center_2")
    return tx, ty, "Theoretical fallback"


def read_dat_header_float(file_path: Path, key: str):
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            for line in file:
                stripped = line.strip()
                if not stripped.startswith("#"):
                    continue
                parts = stripped[1:].strip().split()
                if len(parts) >= 2 and parts[0].lower() == key.lower():
                    return float(parts[1].replace(",", "."))
    except Exception:
        return None
    return None


def matching_manufacturer_azim_profile(file_path: Path):
    path = Path(file_path)
    candidates = [
        path.with_name(f"{path.stem}_azimProf.dat"),
    ]
    if path.stem.endswith("_cave"):
        candidates.append(path.with_name(f"{path.stem[:-5]}_cave_azimProf.dat"))
    else:
        candidates.append(path.with_name(f"{path.stem}_cave_azimProf.dat"))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def matching_manufacturer_q_range(file_path: Path):
    profile = matching_manufacturer_azim_profile(file_path)
    if profile is None:
        return None

    q_min = read_dat_header_float(profile, "QMin")
    q_max = read_dat_header_float(profile, "QMax")
    if q_min is None or q_max is None or q_max <= q_min:
        return None
    return q_min, q_max


def is_azimuthal_processed_image(file_path: Path, header: dict):
    stem = Path(file_path).stem.lower()
    if "_azim" in stem or stem.endswith("azim"):
        return True

    for key, value in header.items():
        key_text = str(key).lower()
        value_text = str(value).lower()
        if key_text == "processing" and "azim" in value_text:
            return True
        if "azim" in key_text and value_text not in ("", "none"):
            return True

    return False


# ============================================================
# AZIMUTHAL PROCESSED IMAGE PROFILE
# ============================================================
def azimuthal_processed_psi_profile(image, q_values, q_min=0.0, q_max=np.inf):
    """Return I(ψ) from an already azimuthal image.

    For *_azim images:
    - x/columns are q positions,
    - y/rows are azimuthal angles ψ,
    - pixel values are intensities.

    The azimuthal profile is therefore the horizontal mean intensity
    of each ψ row over the selected q columns.
    """
    img = image.astype(np.float64)
    ny, nx = img.shape

    q_values = np.asarray(q_values, dtype=float)
    if q_values.ndim != 1 or q_values.size != nx:
        raise ValueError("The q axis must contain one value per image column.")

    q_mask = (q_values >= float(q_min)) & (q_values <= float(q_max))
    if not np.any(q_mask):
        raise ValueError("No q column found in the selected q range.")

    selected = img[:, q_mask]
    valid = np.isfinite(selected) & (selected < 4e9)
    counts = np.sum(valid, axis=1)

    with np.errstate(invalid="ignore"):
        intensity = np.nanmean(np.where(valid, selected, np.nan), axis=1)

    psi_values = np.linspace(0.0, 360.0, ny, endpoint=False)

    keep = np.isfinite(psi_values) & np.isfinite(intensity) & (counts > 0)
    return psi_values[keep], intensity[keep], counts[keep]
def azimuthal_processed_q_mask(shape, q_values, q_min=0.0, q_max=np.inf):
    ny, nx = shape
    q_values = np.asarray(q_values, dtype=float)
    if q_values.ndim != 1 or q_values.size != nx:
        return np.ones((ny, nx), dtype=bool)
    q_mask = (q_values >= float(q_min)) & (q_values <= float(q_max))
    return np.tile(q_mask, (ny, 1))


def azimuthal_normalization_factor(header: dict, mode: str):
    if mode == "raw":
        return 1.0, "raw"

    exposure = get_header_float(header, "ExposureTime", "Exposure", "exposure_time", "count_time", "CountTime")
    flux = get_header_float(header, "TransmittedFlux", "Monitor", "monitor", "Flux", "IncidentFlux")

    if mode == "exposure":
        if exposure and exposure > 0:
            return 1.0 / exposure, f"/ ExposureTime ({exposure:.6g})"
        return 1.0, "raw, missing ExposureTime"

    if mode == "flux":
        if flux and flux > 0:
            return 1.0 / flux, f"/ TransmittedFlux ({flux:.6g})"
        return 1.0, "raw, missing TransmittedFlux"

    if mode == "exposure_flux":
        if exposure and exposure > 0 and flux and flux > 0:
            return 1.0 / (exposure * flux), f"/ ExposureTime / TransmittedFlux ({exposure:.6g}, {flux:.6g})"
        return 1.0, "raw, missing ExposureTime or TransmittedFlux"

    return 1.0, "raw"

def rotate_psi_reference(psi_values, reference_angle_deg):
    psi = np.asarray(psi_values, dtype=float)
    return (psi - float(reference_angle_deg)) % 360.0

ID02_DEFAULT_CENTER_X = 914.4
ID02_DEFAULT_CENTER_Y = 996.5
ID02_DEFAULT_DISTANCE_M = 10.0002
ID02_DEFAULT_PIXEL_MM = 0.075
ID02_DEFAULT_WAVELENGTH_A = 1.01402
CENTER_X_KEYS = (
    "Center_1", "center_1",
    "CenterX", "center_x",
    "BeamCenterX", "Beam_x", "beam_x",
    "Theoretical_Center_1", "theoretical_center_1",
)

CENTER_Y_KEYS = (
    "Center_2", "center_2",
    "CenterY", "center_y",
    "BeamCenterY", "Beam_y", "beam_y",
    "Theoretical_Center_2", "theoretical_center_2",
)


# ============================================================
# AZIMUTHAL INTEGRATION TOOLS
# ============================================================

def azimuthal_average(
    image,
    xc,
    yc,
    distance_m,
    pixel_x_mm,
    pixel_y_mm,
    wavelength_a,
    q_min,
    q_max,
    psi_points,
    min_pixels_per_bin=1,
    axis_mask_px=0,
):
    if distance_m <= 0:
        raise ValueError("Detector distance must be > 0.")
    if pixel_x_mm <= 0 or pixel_y_mm <= 0:
        raise ValueError("Pixel size must be > 0.")
    if wavelength_a <= 0:
        raise ValueError("Wavelength must be > 0.")
    if q_max <= q_min:
        raise ValueError("q max must be greater than q min.")
    if psi_points < 2:
        raise ValueError("Number of ψ points must be at least 2.")

    img = image.astype(np.float64)
    y, x = np.indices(img.shape)

    dx_px = x + 1 - xc
    dy_px = y + 1 - yc

    # Always use geometric q calculation:
    dx_m = dx_px * pixel_x_mm * 1e-3
    dy_m = dy_px * pixel_y_mm * 1e-3
    r_m = np.sqrt(dx_m ** 2 + dy_m ** 2)

    two_theta = np.arctan2(r_m, distance_m)
    theta = two_theta / 2
    wavelength_nm = wavelength_a * 0.1
    q = (4 * np.pi / wavelength_nm) * np.sin(theta)

    psi = (np.degrees(np.arctan2(dy_px, dx_px)) + 360) % 360

    valid = np.isfinite(img) & np.isfinite(q) & np.isfinite(psi)
    valid &= img > 0
    valid &= img < 4e9
    valid &= q >= q_min
    valid &= q <= q_max
    if axis_mask_px > 0:
        valid &= (np.abs(dx_px) > axis_mask_px) & (np.abs(dy_px) > axis_mask_px)

    psi_values = psi[valid]
    i_values = img[valid]

    if psi_values.size == 0:
        raise ValueError("No valid pixel found in the selected q crown.")

    edges = np.linspace(0, 360, psi_points + 1)
    sums, _ = np.histogram(psi_values, bins=edges, weights=i_values)
    counts, _ = np.histogram(psi_values, bins=edges)
    psi_sums, _ = np.histogram(psi_values, bins=edges, weights=psi_values)

    with np.errstate(invalid="ignore", divide="ignore"):
        intensity = sums / counts
        psi_mean = psi_sums / counts

    valid_bins = counts >= max(1, int(min_pixels_per_bin))
    return psi_mean[valid_bins], intensity[valid_bins], counts[valid_bins], valid, q


def pyfai_azimuthal_average(
    image,
    xc,
    yc,
    distance_m,
    pixel_x_mm,
    pixel_y_mm,
    wavelength_a,
    q_min,
    q_max,
    psi_points,
    min_pixels_per_bin=1,
    axis_mask_px=0,
):
    try:
        from pyFAI.integrator.azimuthal import AzimuthalIntegrator
    except Exception:
        from pyFAI.azimuthalIntegrator import AzimuthalIntegrator

    img = image.astype(np.float64)
    invalid_mask = ~np.isfinite(img) | (img <= 0) | (img >= 4e9)
    pixel1_m = float(pixel_y_mm) * 1e-3
    pixel2_m = float(pixel_x_mm) * 1e-3
    wavelength_m = float(wavelength_a) * 1e-10

    integrator = AzimuthalIntegrator(
        dist=float(distance_m),
        poni1=float(yc) * pixel1_m,
        poni2=float(xc) * pixel2_m,
        pixel1=pixel1_m,
        pixel2=pixel2_m,
        wavelength=wavelength_m,
    )

    radial_range = None
    if q_min > 0 and np.isfinite(q_max) and q_max > q_min:
        radial_range = (float(q_min), float(q_max))

    try:
        q_map = np.asarray(integrator.qArray(img.shape), dtype=float)
    except Exception:
        y, x = np.indices(img.shape)
        dx_px = x + 1 - float(xc)
        dy_px = y + 1 - float(yc)
        dx_m = dx_px * float(pixel_x_mm) * 1e-3
        dy_m = dy_px * float(pixel_y_mm) * 1e-3
        r_m = np.sqrt(dx_m ** 2 + dy_m ** 2)
        two_theta = np.arctan2(r_m, float(distance_m))
        q_map = (4 * np.pi / (float(wavelength_a) * 0.1)) * np.sin(two_theta / 2)

    try:
        chi_map = np.asarray(integrator.center_array(img.shape, "chi_rad"), dtype=float)
    except Exception:
        chi_map = np.asarray(integrator.chiArray(img.shape), dtype=float)

    psi_map = (np.degrees(chi_map) + 360.0) % 360.0
    weights = img.copy()
    try:
        solid_angle = np.asarray(integrator.solidAngleArray(img.shape), dtype=float)
        weights = np.divide(weights, solid_angle, out=weights, where=np.isfinite(solid_angle) & (solid_angle > 0))
    except Exception:
        pass

    y, x = np.indices(img.shape)
    dx_px = x + 1 - float(xc)
    dy_px = y + 1 - float(yc)

    valid_mask = (~invalid_mask) & np.isfinite(q_map) & np.isfinite(psi_map) & np.isfinite(weights)
    valid_mask &= q_map >= q_min
    valid_mask &= q_map <= q_max
    if axis_mask_px > 0:
        valid_mask &= (np.abs(dx_px) > axis_mask_px) & (np.abs(dy_px) > axis_mask_px)

    psi_values = psi_map[valid_mask]
    intensity_values = weights[valid_mask]
    if psi_values.size == 0:
        raise ValueError("No valid pixel found in the selected q crown.")

    edges = np.linspace(0, 360, int(psi_points) + 1)
    sums, _ = np.histogram(psi_values, bins=edges, weights=intensity_values)
    counts, _ = np.histogram(psi_values, bins=edges)
    psi_sums, _ = np.histogram(psi_values, bins=edges, weights=psi_values)

    with np.errstate(invalid="ignore", divide="ignore"):
        intensity = sums / counts
        psi = psi_sums / counts

    valid_bins = np.isfinite(psi) & np.isfinite(intensity) & (counts >= max(1, int(min_pixels_per_bin)))
    return psi[valid_bins], intensity[valid_bins], counts[valid_bins], valid_mask, q_map


# ============================================================
# PYFAI AZIMUTHAL TEST DIALOG
# ============================================================

class PyFAIAzimuthalTestDialog(QDialog):
    """Interactive pyFAI azimuthal integration tester with pyFAI-like options."""

    def __init__(self, parent, image, file_path, geometry, header=None, frame_count=1, frame_index=None):
        super().__init__(parent)
        self.setWindowTitle("pyFAI azimuthal integration test")
        self.resize(1250, 780)

        self.image = np.asarray(image, dtype=np.float64)
        self.file_path = Path(file_path)
        self.geometry = dict(geometry)
        self.header = dict(header or {})
        self.frame_count = int(frame_count or 1)
        self.frame_index = frame_index
        self.reference_path = None

        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(10)

        left_panel = QWidget()
        left_panel.setFixedWidth(390)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)
        root_layout.addWidget(left_panel, stretch=0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        left_layout.addWidget(scroll, stretch=1)
        controls = QWidget()
        scroll.setWidget(controls)
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(4, 4, 4, 4)
        controls_layout.setSpacing(8)

        reference_box = QGroupBox("Reference I(ψ)")
        reference_layout = QVBoxLayout(reference_box)
        self.reference_combo = QComboBox()
        self.reference_combo.addItem("No reference", None)
        self.populate_reference_combo()
        self.reference_combo.currentIndexChanged.connect(self.recalculate)
        self.reference_browse_button = QPushButton("Choose reference .dat...")
        self.reference_browse_button.clicked.connect(self.choose_reference_file)
        reference_layout.addWidget(self.reference_combo)
        reference_layout.addWidget(self.reference_browse_button)
        controls_layout.addWidget(reference_box)

        geometry_box = QGroupBox("Geometry used")
        geometry_layout = QVBoxLayout(geometry_box)
        self.geometry_info_label = QLabel(self.geometry_summary_text())
        self.geometry_info_label.setWordWrap(True)
        self.geometry_info_label.setStyleSheet("font-size: 11px; color: #333; padding: 4px;")
        geometry_layout.addWidget(self.geometry_info_label)
        controls_layout.addWidget(geometry_box)

        method_box = QGroupBox("pyFAI method")
        method_layout = QFormLayout(method_box)
        self.split_combo = QComboBox()
        for label, value in [("Any", "*"), ("No splitting", "no"), ("Bounding box", "bbox"), ("Pseudo split (2D)", "pseudo"), ("Full splitting", "full")]:
            self.split_combo.addItem(label, value)
        self.split_combo.setCurrentText("Bounding box")
        self.split_combo.currentIndexChanged.connect(self.recalculate)

        self.algorithm_combo = QComboBox()
        for label, value in [("Any", "*"), ("Histogram", "histogram"), ("LUT", "lut"), ("CSR", "csr"), ("CSC", "csc")]:
            self.algorithm_combo.addItem(label, value)
        self.algorithm_combo.setCurrentText("Histogram")
        self.algorithm_combo.currentIndexChanged.connect(self.recalculate)

        self.implementation_combo = QComboBox()
        for label, value in [("Any", "*"), ("Python", "python"), ("Cython", "cython"), ("OpenCL", "opencl")]:
            self.implementation_combo.addItem(label, value)
        self.implementation_combo.setCurrentText("Cython")
        self.implementation_combo.currentIndexChanged.connect(self.recalculate)

        method_layout.addRow("Pixel splitting:", self.split_combo)
        method_layout.addRow("Algorithm:", self.algorithm_combo)
        method_layout.addRow("Implementation:", self.implementation_combo)
        controls_layout.addWidget(method_box)

        preprocessing_box = QGroupBox("Preprocessing")
        preprocessing_layout = QFormLayout(preprocessing_box)
        self.solid_angle_checkbox = QCheckBox("Solid angle correction")
        self.solid_angle_checkbox.setChecked(False)
        self.solid_angle_checkbox.stateChanged.connect(self.recalculate)
        self.mask_combo = QComboBox()
        self.mask_combo.addItem("Strict: finite, >0, <4e9", "strict")
        self.mask_combo.addItem("Finite only", "finite_only")
        self.mask_combo.addItem("Finite and positive", "finite_positive")
        self.mask_combo.addItem("No mask", "none")
        self.mask_combo.setCurrentIndex(3)
        self.mask_combo.currentIndexChanged.connect(self.recalculate)
        self.dummy_edit = QLineEdit()
        self.dummy_edit.setPlaceholderText("none")
        self.dummy_edit.textChanged.connect(self.recalculate)
        self.delta_dummy_edit = QLineEdit()
        self.delta_dummy_edit.setPlaceholderText("none")
        self.delta_dummy_edit.textChanged.connect(self.recalculate)
        self.polarization_edit = QLineEdit()
        self.polarization_edit.setPlaceholderText("none, e.g. 0.99")
        self.polarization_edit.textChanged.connect(self.recalculate)
        preprocessing_layout.addRow(self.solid_angle_checkbox)
        preprocessing_layout.addRow("Auto mask:", self.mask_combo)
        preprocessing_layout.addRow("Dummy:", self.dummy_edit)
        preprocessing_layout.addRow("Delta dummy:", self.delta_dummy_edit)
        preprocessing_layout.addRow("Polarization:", self.polarization_edit)
        controls_layout.addWidget(preprocessing_box)

        integration_box = QGroupBox("Azimuthal integration")
        integration_layout = QFormLayout(integration_box)
        self.psi_points_spin = QSpinBox()
        self.psi_points_spin.setRange(10, 10000)
        self.psi_points_spin.setValue(int(self.geometry.get("psi_points", 360)))
        self.psi_points_spin.valueChanged.connect(self.recalculate)
        self.q_min_edit = QLineEdit(str(self.geometry.get("q_min", "") or ""))
        self.q_min_edit.setPlaceholderText("auto")
        self.q_min_edit.textChanged.connect(self.recalculate)
        self.q_max_edit = QLineEdit(str(self.geometry.get("q_max", "") or ""))
        self.q_max_edit.setPlaceholderText("auto")
        self.q_max_edit.textChanged.connect(self.recalculate)
        self.axis_mask_spin = QSpinBox()
        self.axis_mask_spin.setRange(0, 100)
        self.axis_mask_spin.setValue(int(self.geometry.get("axis_mask_px", 0)))
        self.axis_mask_spin.valueChanged.connect(self.recalculate)
        self.min_pixels_spin = QSpinBox()
        self.min_pixels_spin.setRange(1, 1000000)
        self.min_pixels_spin.setValue(int(self.geometry.get("min_pixels", 1)))
        self.min_pixels_spin.valueChanged.connect(self.recalculate)
        integration_layout.addRow("ψ points:", self.psi_points_spin)
        integration_layout.addRow("q min nm⁻¹:", self.q_min_edit)
        integration_layout.addRow("q max nm⁻¹:", self.q_max_edit)
        integration_layout.addRow("Axis mask px:", self.axis_mask_spin)
        integration_layout.addRow("Min pixels/bin:", self.min_pixels_spin)
        controls_layout.addWidget(integration_box)

        geometry_offset_box = QGroupBox("Manual center offsets")
        geometry_offset_layout = QFormLayout(geometry_offset_box)
        self.center_x_shift = QDoubleSpinBox()
        self.center_x_shift.setDecimals(3)
        self.center_x_shift.setRange(-50, 50)
        self.center_x_shift.setSingleStep(0.1)
        self.center_x_shift.setValue(0)
        self.center_x_shift.valueChanged.connect(self.recalculate)
        self.center_y_shift = QDoubleSpinBox()
        self.center_y_shift.setDecimals(3)
        self.center_y_shift.setRange(-50, 50)
        self.center_y_shift.setSingleStep(0.1)
        self.center_y_shift.setValue(0)
        self.center_y_shift.valueChanged.connect(self.recalculate)
        geometry_offset_layout.addRow("Center Δx px:", self.center_x_shift)
        geometry_offset_layout.addRow("Center Δy px:", self.center_y_shift)
        controls_layout.addWidget(geometry_offset_box)

        display_box = QGroupBox("Display / correction")
        display_layout = QFormLayout(display_box)
        self.x_scale_combo = QComboBox()
        self.x_scale_combo.addItems(["linear", "log"])
        self.x_scale_combo.currentTextChanged.connect(self.recalculate)
        self.y_scale_combo = QComboBox()
        self.y_scale_combo.addItems(["linear", "log"])
        self.y_scale_combo.currentTextChanged.connect(self.recalculate)
        self.normalization_combo = QComboBox()
        self.normalization_combo.addItem("Raw detector intensity", "raw")
        self.normalization_combo.addItem("Counts/s: I / ExposureTime", "exposure")
        self.normalization_combo.addItem("I / TransmittedFlux", "flux")
        self.normalization_combo.addItem("Counts/s/flux", "exposure_flux")
        self.normalization_combo.currentIndexChanged.connect(self.recalculate)
        self.intensity_scale_spin = QDoubleSpinBox()
        self.intensity_scale_spin.setDecimals(6)
        self.intensity_scale_spin.setRange(1e-12, 1e12)
        self.intensity_scale_spin.setValue(1.0)
        self.intensity_scale_spin.valueChanged.connect(self.recalculate)
        display_layout.addRow("x scale:", self.x_scale_combo)
        display_layout.addRow("y scale:", self.y_scale_combo)
        display_layout.addRow("Normalization:", self.normalization_combo)
        display_layout.addRow("I scale:", self.intensity_scale_spin)
        controls_layout.addWidget(display_box)

        self.save_button = QPushButton("Save tested I(ψ) .dat")
        self.save_button.clicked.connect(self.save_current_curve)
        controls_layout.addWidget(self.save_button)
        controls_layout.addStretch(1)

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        left_layout.addWidget(self.close_button)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)
        root_layout.addWidget(right_panel, stretch=1)

        self.canvas = PlotCanvas()
        self.toolbar = NavigationToolbar(self.canvas, self)
        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas, stretch=1)

        self.log_label = QLabel("")
        self.log_label.setWordWrap(True)
        right_layout.addWidget(self.log_label)

        self.current_psi = None
        self.current_i = None
        self.current_counts = None
        self.current_label = None
        self.recalculate()

    def geometry_summary_text(self):
        used_cx = float(self.geometry.get("xc", 0.0))
        used_cy = float(self.geometry.get("yc", 0.0))
        used_dist = float(self.geometry.get("distance_m", 0.0))
        used_px = float(self.geometry.get("pixel_x_mm", 0.0))
        used_py = float(self.geometry.get("pixel_y_mm", 0.0))
        used_wav_a = float(self.geometry.get("wavelength_a", 0.0))
        return (
            f"Used for integration: center=({used_cx:.6g}, {used_cy:.6g}) px ; "
            f"distance={used_dist:.6g} m ; pixel=({used_px:.6g}, {used_py:.6g}) mm ; "
            f"λ={used_wav_a:.6g} Å"
        )

    def populate_reference_combo(self):
        folder = self.file_path.parent
        for dat_path in sorted(folder.glob("*.dat")):
            lower = dat_path.stem.lower()
            if "azim" in lower or "psi" in lower:
                self.reference_combo.addItem(dat_path.name, str(dat_path))

    def choose_reference_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose reference I(ψ) DAT", str(self.file_path.parent), "DAT files (*.dat);;All files (*)")
        if not path:
            return
        path = str(Path(path))
        for index in range(self.reference_combo.count()):
            if self.reference_combo.itemData(index) == path:
                self.reference_combo.setCurrentIndex(index)
                return
        self.reference_combo.addItem(Path(path).name, path)
        self.reference_combo.setCurrentIndex(self.reference_combo.count() - 1)

    def parse_float_or_zero(self, text):
        text = normalize_decimal_text(text).strip()
        if not text:
            return 0.0
        try:
            return float(text)
        except ValueError:
            return 0.0

    def parse_optional_float(self, text):
        text = normalize_decimal_text(text).strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def build_mask(self):
        mode = self.mask_combo.currentData()
        if mode == "none":
            return None
        if mode == "finite_only":
            return ~np.isfinite(self.image)
        if mode == "finite_positive":
            return ~np.isfinite(self.image) | (self.image <= 0)
        return ~np.isfinite(self.image) | (self.image <= 0) | (self.image >= 4e9)

    def make_integrator(self):
        try:
            try:
                from pyFAI.integrator.azimuthal import AzimuthalIntegrator
            except Exception:
                from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
        except Exception as exc:
            raise RuntimeError(f"pyFAI unavailable: {exc}") from exc

        dx = float(self.center_x_shift.value())
        dy = float(self.center_y_shift.value())
        pixel1_m = float(self.geometry["pixel_y_mm"]) * 1e-3
        pixel2_m = float(self.geometry["pixel_x_mm"]) * 1e-3
        wavelength_m = float(self.geometry["wavelength_a"]) * 1e-10
        return AzimuthalIntegrator(
            dist=float(self.geometry["distance_m"]),
            poni1=(float(self.geometry["yc"]) + dy) * pixel1_m,
            poni2=(float(self.geometry["xc"]) + dx) * pixel2_m,
            pixel1=pixel1_m,
            pixel2=pixel2_m,
            wavelength=wavelength_m,
        )

    def load_reference_curve(self):
        path = self.reference_combo.currentData()
        if not path:
            return None, None, None
        psi_values = []
        intensity_values = []
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#"):
                        continue
                    parts = stripped.replace(",", ".").split()
                    if len(parts) < 2:
                        continue
                    try:
                        psi_values.append(float(parts[0]))
                        intensity_values.append(float(parts[1]))
                    except ValueError:
                        continue
            if not psi_values:
                raise ValueError("no numeric ψ/I rows found")
            psi = np.asarray(psi_values, dtype=float)
            intensity = np.asarray(intensity_values, dtype=float)
            valid = np.isfinite(psi) & np.isfinite(intensity)
            return psi[valid], intensity[valid], Path(path).name
        except Exception:
            return None, None, None

    def compute_test_curve(self):
        q_min = self.parse_float_or_zero(self.q_min_edit.text())
        q_max = self.parse_float_or_zero(self.q_max_edit.text())
        if q_max <= q_min:
            raise ValueError("q max must be greater than q min for azimuthal integration.")

        integrator = self.make_integrator()
        try:
            q_map = np.asarray(integrator.qArray(self.image.shape), dtype=float)
        except Exception:
            y, x = np.indices(self.image.shape)
            dx_px = x + 1 - float(self.geometry["xc"])
            dy_px = y + 1 - float(self.geometry["yc"])
            dx_m = dx_px * float(self.geometry["pixel_x_mm"]) * 1e-3
            dy_m = dy_px * float(self.geometry["pixel_y_mm"]) * 1e-3
            r_m = np.sqrt(dx_m ** 2 + dy_m ** 2)
            two_theta = np.arctan2(r_m, float(self.geometry["distance_m"]))
            q_map = (4 * np.pi / (float(self.geometry["wavelength_a"]) * 0.1)) * np.sin(two_theta / 2)

        try:
            chi_map = np.asarray(integrator.center_array(self.image.shape, "chi_rad"), dtype=float)
        except Exception:
            chi_map = np.asarray(integrator.chiArray(self.image.shape), dtype=float)

        psi_map = (np.degrees(chi_map) + 360.0) % 360.0
        weights = self.image.astype(np.float64).copy()
        if self.solid_angle_checkbox.isChecked():
            try:
                solid_angle = np.asarray(integrator.solidAngleArray(self.image.shape), dtype=float)
                weights = np.divide(weights, solid_angle, out=weights, where=np.isfinite(solid_angle) & (solid_angle > 0))
            except Exception:
                pass

        dummy = self.parse_optional_float(self.dummy_edit.text())
        delta_dummy = self.parse_optional_float(self.delta_dummy_edit.text())
        if dummy is not None:
            tolerance = delta_dummy if delta_dummy is not None else 0.0
            if tolerance > 0:
                dummy_mask = np.abs(weights - dummy) <= tolerance
            else:
                dummy_mask = weights == dummy
        else:
            dummy_mask = np.zeros_like(weights, dtype=bool)

        mask = self.build_mask()
        valid = np.isfinite(q_map) & np.isfinite(psi_map) & np.isfinite(weights)
        valid &= q_map >= q_min
        valid &= q_map <= q_max
        if mask is not None:
            valid &= ~mask
        if dummy is not None:
            valid &= ~dummy_mask

        axis_mask_px = int(self.axis_mask_spin.value())
        if axis_mask_px > 0:
            y, x = np.indices(self.image.shape)
            dx_px = x + 1 - float(self.geometry["xc"])
            dy_px = y + 1 - float(self.geometry["yc"])
            valid &= (np.abs(dx_px) > axis_mask_px) & (np.abs(dy_px) > axis_mask_px)

        psi_values = psi_map[valid]
        intensity_values = weights[valid]
        if psi_values.size == 0:
            raise ValueError("No valid pixel found in selected q crown.")

        edges = np.linspace(0.0, 360.0, int(self.psi_points_spin.value()) + 1)
        sums, _ = np.histogram(psi_values, bins=edges, weights=intensity_values)
        counts, _ = np.histogram(psi_values, bins=edges)
        psi_sums, _ = np.histogram(psi_values, bins=edges, weights=psi_values)
        with np.errstate(invalid="ignore", divide="ignore"):
            intensity = sums / counts
            psi = psi_sums / counts
        valid_bins = np.isfinite(psi) & np.isfinite(intensity) & (counts >= max(1, int(self.min_pixels_spin.value())))
        psi = psi[valid_bins]
        intensity = intensity[valid_bins]
        counts = counts[valid_bins]

        normalization_factor, _label = azimuthal_normalization_factor(self.header, self.normalization_combo.currentData())
        intensity = intensity * normalization_factor * float(self.intensity_scale_spin.value())
        return psi, intensity, counts

    def recalculate(self):
        try:
            psi, intensity, counts = self.compute_test_curve()
            self.current_psi = psi
            self.current_i = intensity
            self.current_counts = counts
            self.current_label = self.file_path.name
            self.geometry_info_label.setText(self.geometry_summary_text())

            ax = self.canvas.ax
            ax.clear()
            ax.plot(psi, intensity, linewidth=1.4, label=self.current_label)
            ref_psi, ref_i, ref_name = self.load_reference_curve()
            if ref_psi is not None and ref_i is not None:
                ax.plot(ref_psi, ref_i, linewidth=1.4, linestyle="--", label=ref_name)
            ax.set_xscale(self.x_scale_combo.currentText())
            ax.set_yscale(self.y_scale_combo.currentText())
            ax.set_xlabel("ψ / °")
            ax.set_ylabel("Intensity / a.u.")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="lower left")
            self.canvas.draw_idle()
            self.log_label.clear()
        except Exception:
            self.log_label.clear()

    def save_current_curve(self):
        if self.current_psi is None or self.current_i is None:
            return
        parent = self.parent()
        if parent is None:
            return
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"azim_test_{self.split_combo.currentData()}_{self.algorithm_combo.currentData()}_{self.implementation_combo.currentData()}").strip("_")
        out_file = self.file_path.with_name(f"{self.file_path.stem}_{safe_label}.dat")
        try:
            with open(out_file, "w", encoding="utf-8") as file:
                file.write("# psi_deg I_psi pixel_count\n")
                file.write(f"# source {self.file_path.name}\n")
                file.write(f"# method {self.split_combo.currentData()} {self.algorithm_combo.currentData()} {self.implementation_combo.currentData()}\n")
                file.write(f"# q_range_nm-1 {self.q_min_edit.text()} {self.q_max_edit.text()}\n")
                for psi, intensity, count in zip(self.current_psi, self.current_i, self.current_counts):
                    file.write(f"{psi:.10g} {intensity:.10g} {int(count)}\n")
        except Exception:
            pass


# ============================================================
# CANVAS
# ============================================================

class PlotCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(dpi=150)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.fig.subplots_adjust(left=0.12, right=0.98, top=0.92, bottom=0.18)
        self.setMinimumSize(620, 420)


class ImageCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setStyleSheet("background: transparent;")
        self.fig.patch.set_alpha(0)
        self.ax.set_facecolor("none")
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
        self.last_xc = None
        self.last_yc = None
        self.last_mask = None
        self.q_map = None
        self.coordinate_mode = "detector"
        self.reference_angle_deg = 0.0

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
        if scale_factor <= 0 or self.raw_image is None:
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
        constrain_image_axes(self.ax, self.raw_image.shape)

        self.draw_idle()

    def _pan_from_pixels(self, dx_pixels, dy_pixels):
        if self.raw_image is None or (dx_pixels == 0 and dy_pixels == 0):
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
        constrain_image_axes(self.ax, self.raw_image.shape)

        self.draw_idle()

    def set_coordinate_label(self, label):
        self.coordinate_label = label

    def set_q_map(self, q_map):
        self.q_map = q_map

    def set_coordinate_mode(self, mode):
        self.coordinate_mode = mode

    def set_reference_angle(self, angle_deg):
        self.reference_angle_deg = float(angle_deg)
        if self.raw_image is not None:
            self.show_image(self.raw_image, self.last_xc, self.last_yc, self.last_mask)

    def reset_display_limits(self):
        self.display_vmin = None
        self.display_vmax = None

    def set_display_limits(self, vmin, vmax):
        self.display_vmin = float(vmin)
        self.display_vmax = float(vmax)
        if self.display_vmax <= self.display_vmin:
            self.display_vmax = self.display_vmin + 1e-6

        if self.raw_image is not None:
            self.show_image(self.raw_image, self.last_xc, self.last_yc, self.last_mask)

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
        constrain_image_axes(self.ax, self.raw_image.shape)
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
                    q_ny, q_nx = self.q_map.shape
                    if 0 <= x_index < q_nx and 0 <= y_index < q_ny:
                        q_value = self.q_map[y_index, x_index]
                        if np.isfinite(q_value):
                            q_text = f"{q_value:.6g} nm⁻¹"

                if self.coordinate_mode == "azimuthal_image" and self.raw_image is not None:
                    ny, _nx = self.raw_image.shape
                    if ny > 0:
                        psi = (y_index / max(1, ny - 1)) * 360.0
                        psi = rotate_psi_reference(psi, self.reference_angle_deg)
                        psi_text = f"{psi:.3f}°"    
                elif self.last_xc is not None and self.last_yc is not None:
                    dx = (x_index + 1) - self.last_xc
                    dy = (y_index + 1) - self.last_yc
                    psi = np.degrees(np.arctan2(dy, dx)) % 360.0
                    psi = rotate_psi_reference(psi, self.reference_angle_deg)
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
        constrain_image_axes(self.ax, self.raw_image.shape)
        self.draw_idle()

    def show_image(self, image, xc=None, yc=None, mask=None):
        current_xlim = self.ax.get_xlim()
        current_ylim = self.ax.get_ylim()
        had_image = len(self.ax.images) > 0
        self.raw_image = image
        self.last_xc = xc
        self.last_yc = yc
        self.last_mask = mask

        self.ax.clear()
        self.fig.patch.set_alpha(0)
        self.ax.set_facecolor("none")
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
            overlay[~mask, :] = [0.55, 0.55, 0.55, 0.72]
            self.ax.imshow(overlay, origin="upper", interpolation="nearest")

            if self.coordinate_mode == "azimuthal_image" and np.any(mask):
                selected_columns = np.where(np.any(mask, axis=0))[0]
                if selected_columns.size > 0:
                    self.ax.axvline(selected_columns[0] - 0.5, color="white", linewidth=1.1)
                    self.ax.axvline(selected_columns[-1] + 0.5, color="white", linewidth=1.1)

        if xc is not None and yc is not None:
            ny, nx = image.shape

            if self.coordinate_mode == "azimuthal_image":
                self.ax.axvline(0, color="red", linewidth=1.0)
                self.ax.axhline(0, color="red", linewidth=1.0)
                self.ax.plot(0, 0, "wo", markersize=4)

                for angle in [0, 90, 180, 270, 360]:
                    y_text = (angle / 360.0) * (ny - 1)
                    self.ax.text(
                        0,
                        y_text,
                        f"{angle}°",
                        color="white",
                        fontsize=10,
                        fontweight="bold",
                        ha="left",
                        va="center",
                        bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=2),
                    )
            else:
                center_x = float(xc) - 1.0
                center_y = float(yc) - 1.0

                line_length = max(nx, ny) * 0.75

                angle0 = np.deg2rad(float(self.reference_angle_deg))
                angle90 = np.deg2rad(float(self.reference_angle_deg) + 90.0)

                self.ax.plot(
                    [center_x - line_length * np.cos(angle0), center_x + line_length * np.cos(angle0)],
                    [center_y - line_length * np.sin(angle0), center_y + line_length * np.sin(angle0)],
                    color="red",
                    linewidth=1.0,
                )

                self.ax.plot(
                    [center_x - line_length * np.cos(angle90), center_x + line_length * np.cos(angle90)],
                    [center_y - line_length * np.sin(angle90), center_y + line_length * np.sin(angle90)],
                    color="red",
                    linewidth=1.0,
                )

                self.ax.plot(center_x, center_y, "wo", markersize=4)

                radius = min(nx, ny) * 0.35
                for angle in [0, 90, 180, 270]:
                    detector_angle = (float(angle) + float(self.reference_angle_deg)) % 360.0
                    rad = np.deg2rad(detector_angle)
                    x_text = center_x + radius * np.cos(rad)
                    y_text = center_y + radius * np.sin(rad)
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
        self.ax.set_aspect("auto" if self.coordinate_mode == "azimuthal_image" else "equal")

        if had_image:
            self.ax.set_xlim(current_xlim)
            self.ax.set_ylim(current_ylim)
            constrain_image_axes(self.ax, self.raw_image.shape)

        self.draw_idle()


# ============================================================
# AZIMUTHAL TAB
# ============================================================

class AzimuthalTab(QWidget):
    """Azimuthal tab: azimuthal integration I(ψ) on a q crown."""

    folder_changed = Signal(Path)

    def __init__(self):
        super().__init__()

        self.current_folder = Path("/Users/nathanpiaget/Documents/Thèse LRP/Expériences/XENOCS")
        self.instrument_mode = "XENOCS"
        self.last_results = {}
        self.last_result_paths = {}
        self.last_result_frame_counts = {}
        self._syncing_folder = False
        self.current_frame = 1
        self.total_frames = 1

        self.build_ui()
        self.set_controls_enabled(False)

    def build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(*PAGE_MARGINS)
        main_layout.setSpacing(BLOCK_SPACING)

        page_layout = QHBoxLayout()
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(BLOCK_SPACING)
        main_layout.addLayout(page_layout, stretch=1)

        left_panel = QWidget()
        left_panel.setFixedWidth(FILE_BROWSER_WIDTH)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(*PANEL_MARGINS)
        left_layout.setSpacing(BLOCK_SPACING)
        page_layout.addWidget(left_panel, stretch=0)

        right_panel = QWidget()
        right_layout = QHBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(BLOCK_SPACING)
        page_layout.addWidget(right_panel, stretch=1)

        # ============================================================
        # COLUMN 2: I(ψ) GRAPH
        # ============================================================
        center_column = QWidget()
        center_column_layout = QVBoxLayout(center_column)
        center_column_layout.setContentsMargins(0, 0, 0, 0)
        center_column_layout.setSpacing(4)
        right_layout.addWidget(center_column, stretch=1)

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
        self.name_filter = QLineEdit("*cave*")

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
        self.only_thumbs_up_checkbox = QCheckBox("Only 👍")
        self.only_thumbs_up_checkbox.setChecked(False)
        self.only_thumbs_up_checkbox.stateChanged.connect(self.refresh_files)
        file_options_layout = QHBoxLayout()
        file_options_layout.setContentsMargins(0, 0, 0, 0)
        file_options_layout.setSpacing(10)
        file_options_layout.addWidget(self.show_subfolders_checkbox)
        file_options_layout.addWidget(self.only_thumbs_up_checkbox)
        file_options_layout.addStretch(1)
        file_layout.addLayout(file_options_layout)

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_files)
        file_layout.addWidget(self.refresh_button)

        self.file_list = QListWidget()
        install_file_rating_menu(self.file_list)
        self.file_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.file_list.itemSelectionChanged.connect(self.selection_changed)
        self.file_list.currentItemChanged.connect(self.current_file_changed)
        self.file_list.itemClicked.connect(self.file_item_clicked)
        self.file_list.setMinimumHeight(180)
        file_layout.addWidget(self.file_list, stretch=1)

        params_box = QGroupBox("Azimuthal parameters")
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
        for button in [self.btn_xenocs, self.btn_id02, self.btn_id13, self.btn_custom, self.q_manual_button]:
            button.hide()
        self.line_geometry_selector = LineGeometrySelector(self, "XENOCS")
        self.line_geometry_selector.geometry_selected.connect(self.apply_line_geometry_selection)
        preset_layout.addWidget(self.line_geometry_selector, 1)
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
        self.use_q_range = QCheckBox("Use q range")
        self.use_q_range.setChecked(True)
        self.use_q_range.stateChanged.connect(self.update_q_range_state)
        self.q_min = self.double_spin(0.1, decimals=8, minimum=0)
        self.q_max = self.double_spin(1.0, decimals=8, minimum=0)
        parameter_field_width = 130
        self.n_points = QSpinBox()
        self.n_points.setRange(10, 10000)
        self.n_points.setValue(360)
        self.n_points.setFixedHeight(24)
        self.integration_engine = QComboBox()
        self.integration_engine.addItems(["pyFAI", "LRPhoton mean"])
        self.integration_engine.setCurrentText("pyFAI")
        self.integration_engine.setFixedWidth(150)
        self.min_pixels_per_bin = QSpinBox()
        self.min_pixels_per_bin.setRange(1, 1000000)
        self.min_pixels_per_bin.setValue(1)
        self.min_pixels_per_bin.setFixedHeight(24)
        self.min_pixels_per_bin.setMinimumWidth(parameter_field_width)
        self.axis_mask_pixels = QSpinBox()
        self.axis_mask_pixels.setRange(0, 20)
        self.axis_mask_pixels.setValue(0)
        self.axis_mask_pixels.setFixedHeight(24)
        self.axis_mask_pixels.setMinimumWidth(parameter_field_width)
        self.reference_angle = self.double_spin(0.0, decimals=3, minimum=-360.0)
        self.reference_angle.setRange(-360.0, 360.0)
        self.reference_angle.setSingleStep(1.0)
        self.reference_angle.valueChanged.connect(self.reference_angle_changed)
        self.normalization_mode = QComboBox()
        self.normalization_mode.addItem("Raw detector intensity", "raw")
        self.normalization_mode.addItem("Counts/s: I / ExposureTime", "exposure")
        self.normalization_mode.addItem("I / TransmittedFlux", "flux")
        self.normalization_mode.addItem("Counts/s/flux", "exposure_flux")
        self.normalization_mode.setCurrentIndex(0)
        self.normalization_mode.setFixedWidth(220)
        self.intensity_scale = QDoubleSpinBox()
        self.intensity_scale.setDecimals(6)
        self.intensity_scale.setRange(1e-9, 1e9)
        self.intensity_scale.setValue(1.0)
        self.intensity_scale.setSingleStep(0.1)
        self.intensity_scale.setFixedHeight(24)
        self.intensity_scale.setMinimumWidth(parameter_field_width)

        self.q_min.setMinimumWidth(parameter_field_width)
        self.q_max.setMinimumWidth(parameter_field_width)
        self.n_points.setMinimumWidth(parameter_field_width)

        form.addWidget(self.use_q_range, 0, 0, 1, 2)
        form.addWidget(QLabel("q min (nm⁻¹):"), 1, 0)
        form.addWidget(self.q_min, 1, 1)
        form.addWidget(QLabel("q max (nm⁻¹):"), 2, 0)
        form.addWidget(self.q_max, 2, 1)
        form.addWidget(QLabel("ψ reference angle (°):"), 3, 0)
        form.addWidget(self.reference_angle, 3, 1)
        params_layout.addLayout(form)

        integrate_buttons_layout = QHBoxLayout()
        self.integrate_button = QPushButton("Integrate I(ψ)")
        self.integrate_button.clicked.connect(self.integrate_selected_files)
        self.azimuthal_test_button = QPushButton("Test")
        self.azimuthal_test_button.setToolTip("Open an interactive pyFAI azimuthal integration test window")
        self.azimuthal_test_button.clicked.connect(self.open_azimuthal_test_dialog)
        integrate_buttons_layout.addWidget(self.integrate_button, stretch=1)
        integrate_buttons_layout.addWidget(self.azimuthal_test_button, stretch=0)
        params_layout.addLayout(integrate_buttons_layout)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setVisible(False)

        self.show_legend = QCheckBox("Legend")
        self.show_legend.setChecked(True)
        self.show_legend.stateChanged.connect(self.update_legend_visibility)

        self.canvas = PlotCanvas()
        self.canvas.setContentsMargins(0, 0, 0, 0)
        clear_plot_canvas(self.canvas)
        self.toolbar = NavigationToolbar(self.canvas, self)
        toolbar_box, self.toolbar_extra_layout, self.save_graph_button = make_matplotlib_toolbar_block(
            self,
            "I(ψ) graph",
            self.toolbar,
            option_widgets=[
                self.show_legend,
            ],
            save_callback=self.save_results,
            save_tooltip="Save .dat",
            toolbar_width=320,
        )
        center_column_layout.addWidget(toolbar_box, stretch=0)

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

        center_column_layout.addWidget(self.canvas, stretch=1)
        center_column_layout.addWidget(self.graph_coordinate_label, stretch=0)

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
        image_limits_box = QGroupBox("Contrast")
        image_limits_box.setStyleSheet(GROUP_BOX_STYLE)
        image_limits_layout = QGridLayout(image_limits_box)
        image_limits_layout.setContentsMargins(*GROUP_BOX_MARGINS)
        image_limits_layout.setHorizontalSpacing(6)
        image_limits_layout.setVerticalSpacing(2)

        self.image_vmin_label = QLabel("Min: -")
        self.image_vmax_label = QLabel("Max: -")
        self.image_vmin_label.setAlignment(Qt.AlignCenter)
        self.image_vmax_label.setAlignment(Qt.AlignCenter)
        self.image_lock_contrast_checkbox = QCheckBox("Lock min/max")
        self.image_lock_contrast_checkbox.setToolTip("Keep current contrast limits when changing files or recalculating")
        self.image_auto_contrast_button = QPushButton("Auto")
        self.image_auto_contrast_button.setFixedWidth(54)
        self.image_auto_contrast_button.clicked.connect(self.auto_image_intensity_limits)

        self.image_vmin_slider = QSlider(Qt.Horizontal)
        self.image_vmax_slider = QSlider(Qt.Horizontal)
        self.image_vmin_slider.setRange(0, 1000)
        self.image_vmax_slider.setRange(0, 1000)
        self.image_vmin_slider.setValue(0)
        self.image_vmax_slider.setValue(1000)

        image_limits_layout.addWidget(self.image_vmin_label, 0, 0)
        image_limits_layout.addWidget(self.image_vmin_slider, 0, 1)
        image_limits_layout.addWidget(self.image_auto_contrast_button, 0, 2, 2, 1)
        image_limits_layout.addWidget(self.image_vmax_label, 1, 0)
        image_limits_layout.addWidget(self.image_vmax_slider, 1, 1)
        image_limits_layout.addWidget(self.image_lock_contrast_checkbox, 2, 0, 1, 3)

        image_layout.addWidget(image_limits_box)
        self.image_vmin_slider.valueChanged.connect(self.update_image_intensity_limits)
        self.image_vmax_slider.valueChanged.connect(self.update_image_intensity_limits)

        self.canvas.mpl_connect("button_press_event", self.on_graph_right_click)
        self.canvas.mpl_connect("motion_notify_event", self.update_graph_coordinates)
        self.canvas.mpl_connect("axes_leave_event", self.clear_graph_coordinates)

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

    def current_file_changed(self, current, previous):
        if current is None:
            self.selection_changed()
            return
        QTimer.singleShot(0, lambda item=current: self.refresh_selected_file_preview(item))

    def file_item_clicked(self, item):
        if item is None:
            return
        self.file_list.setCurrentItem(item)
        QTimer.singleShot(0, lambda item=item: self.refresh_selected_file_preview(item))

    def refresh_selected_file_preview(self, item):
        if item is None:
            return

        current_file = file_path_from_item(item, self.current_folder)
        if current_file is None:
            return

        self.set_controls_enabled(True)
        if hasattr(self, "image_lock_contrast_checkbox") and not self.image_lock_contrast_checkbox.isChecked():
            self.image_canvas.reset_display_limits()

        self.last_results = {}
        self.last_result_paths = {}
        self.last_result_frame_counts = {}
        self.clear_graph_coordinates()
        clear_plot_canvas(self.canvas)

        self.apply_preset_from_file(current_file)
        self.update_frame_controls_from_file(current_file)
        self.display_selected_file_preview(current_file)

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
            self.wavelength, self.use_q_range, self.q_min, self.q_max, self.n_points,
            self.integration_engine, self.min_pixels_per_bin, self.axis_mask_pixels,
            self.normalization_mode, self.intensity_scale,
            self.integrate_button, self.azimuthal_test_button, self.show_legend,
            self.frame_start_spin, self.frame_end_spin, self.prev_frame_button,
            self.next_frame_button, self.frame_slider,
            self.image_vmin_label, self.image_vmax_label,
            self.image_vmin_slider, self.image_vmax_slider, self.image_lock_contrast_checkbox, self.image_auto_contrast_button,
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
        self.update_frame_navigation_state()
        self.update_q_range_state()

    def reference_angle_changed(self):
        if hasattr(self, "image_canvas"):
            self.image_canvas.set_reference_angle(float(self.reference_angle.value()))
        selected = self.selected_files()
        if selected:
            self.display_selected_file_preview(selected[0])
        if self.last_results:
            self.integrate_selected_files()

    def update_q_range_state(self):
        use_q_range = self.use_q_range.isChecked()
        enabled = self.use_q_range.isEnabled() and use_q_range
        self.q_min.setEnabled(enabled)
        self.q_max.setEnabled(enabled)

    def update_frame_navigation_state(self):
        can_navigate = bool(self.selected_files()) and self.total_frames > 1
        current = self.frame_slider.value()
        self.frame_start_spin.setEnabled(can_navigate)
        self.frame_end_spin.setEnabled(can_navigate)
        self.frame_slider.setEnabled(can_navigate)
        self.prev_frame_button.setEnabled(can_navigate and current > self.frame_slider.minimum())
        self.next_frame_button.setEnabled(can_navigate and current < self.frame_slider.maximum())
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
                self.image_vmin_slider.blockSignals(True)
                self.image_vmin_slider.setValue(min_pos)
                self.image_vmin_slider.blockSignals(False)

        vmin = data_min + span * min_pos / 1000.0
        vmax = data_min + span * max_pos / 1000.0

        self.image_canvas.set_display_limits(vmin, vmax)
        self.image_vmin_label.setText(f"Min: {vmin:.3g}")
        self.image_vmax_label.setText(f"Max: {vmax:.3g}")
        self.canvas.draw_idle()

    def auto_image_intensity_limits(self):
        if not hasattr(self, "image_canvas") or self.image_canvas.raw_image is None:
            return

        self.image_canvas.reset_display_limits()
        self.image_canvas.show_image(
            self.image_canvas.raw_image,
            self.image_canvas.last_xc,
            self.image_canvas.last_yc,
            self.image_canvas.last_mask,
        )
        self.sync_image_intensity_sliders()

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

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose folder", str(self.current_folder))
        if folder:
            self.current_folder = Path(folder)
            if not self.image_lock_contrast_checkbox.isChecked():
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
        files = [
            file for file in files
            if not should_hide_file_in_browser(file)
            and fnmatch(file.name, name_filter)
        ]

        # In the Azimuthal tab, hide already-unfolded azimuthal images and
        # averaged H5 files from the browser. This tab should integrate detector
        # images, not the intermediate *_azim.h5 or *_ave.h5 outputs.
        files = [
            file for file in files
            if "_azim" not in file.stem.lower()
            and not file.stem.lower().endswith("_ave")
            and "_ave_" not in file.stem.lower()
        ]

        if self.only_thumbs_up_checkbox.isChecked():
            files = [file for file in files if is_file_rated_up(file)]

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
            self.last_result_paths = {}
            self.last_result_frame_counts = {}
            self.clear_graph_coordinates()
            clear_plot_canvas(self.canvas)

    def selection_changed(self):
        selected = self.selected_files()
        self.set_controls_enabled(bool(selected))

        if selected:
            current_item = self.file_list.currentItem()
            if current_item is not None and current_item.isSelected():
                QTimer.singleShot(0, lambda item=current_item: self.refresh_selected_file_preview(item))
                return

            selected_items = self.file_list.selectedItems()
            if selected_items:
                self.file_list.setCurrentItem(selected_items[-1])
                QTimer.singleShot(0, lambda item=selected_items[-1]: self.refresh_selected_file_preview(item))
                return

        self.update_frame_controls_from_file(None)
        self.last_results = {}
        self.last_result_paths = {}
        self.last_result_frame_counts = {}
        self.clear_graph_coordinates()
        self.image_canvas.raw_image = None
        self.image_canvas.set_q_map(None)
        self.image_coordinate_label.setText("ψ = - | q = - | I = -")
        clear_plot_canvas(self.canvas)
        clear_plot_canvas(self.image_canvas)

    def selected_files(self):
        selected_items = list(self.file_list.selectedItems())
        current_item = self.file_list.currentItem()

        if current_item is not None and current_item in selected_items:
            ordered_items = [current_item] + [item for item in selected_items if item is not current_item]
        else:
            ordered_items = selected_items

        return [file_path_from_item(item, self.current_folder) for item in ordered_items]

    def current_preview_file(self):
        current_item = self.file_list.currentItem()
        if current_item is not None:
            current_file = file_path_from_item(current_item, self.current_folder)
            if current_file is not None:
                return current_file

        selected = self.selected_files()
        return selected[0] if selected else None

    def update_frame_controls_from_file(self, file_path):
        self.total_frames = 1
        self.current_frame = 1

        try:
            if file_path is None:
                raise ValueError("No file selected")

            suffix = Path(file_path).suffix.lower()
            if suffix in [".h5", ".hdf5"]:
                with h5py.File(file_path, "r") as h5:
                    datasets = []

                    def collect_dataset(name, obj):
                        if isinstance(obj, h5py.Dataset) and obj.ndim >= 2:
                            datasets.append(name)

                    h5.visititems(collect_dataset)

                    if datasets:
                        preferred = None
                        for name in datasets:
                            lower = name.lower()
                            if "data" in lower or "eiger" in lower or "detector" in lower:
                                preferred = name
                                break

                        dataset = h5[preferred or datasets[0]]
                        if dataset.ndim == 3:
                            self.total_frames = int(np.min(dataset.shape))
        except Exception:
            self.total_frames = 1

        self.frame_start_spin.blockSignals(True)
        self.frame_end_spin.blockSignals(True)
        self.frame_slider.blockSignals(True)

        self.frame_start_spin.setRange(1, self.total_frames)
        self.frame_end_spin.setRange(1, self.total_frames)
        self.frame_slider.setRange(1, self.total_frames)

        self.frame_start_spin.setValue(1)
        self.frame_end_spin.setValue(self.total_frames)
        self.frame_slider.setValue(1)
        self.frame_counter_label.setText(f"1 / {self.total_frames}")

        self.frame_start_spin.blockSignals(False)
        self.frame_end_spin.blockSignals(False)
        self.frame_slider.blockSignals(False)
        self.update_frame_navigation_state()

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
        self.update_frame_navigation_state()

    def frame_slider_changed(self, value):
        self.current_frame = value
        self.frame_counter_label.setText(f"{value} / {self.total_frames}")
        self.update_frame_navigation_state()

        current_file = self.current_preview_file()
        if current_file is not None:
            self.display_selected_file_preview(current_file)
            if self.last_results:
                self.integrate_selected_files()

    def previous_frame(self):
        value = max(self.frame_slider.minimum(), self.frame_slider.value() - 1)
        self.frame_slider.setValue(value)

    def next_frame(self):
        value = min(self.frame_slider.maximum(), self.frame_slider.value() + 1)
        self.frame_slider.setValue(value)

    def set_instrument_mode(self, mode):
        self.instrument_mode = mode
        if hasattr(self, "line_geometry_selector") and mode in self.line_geometry_selector.geometries:
            self.line_geometry_selector.set_current_name(mode)
        buttons = {
            "XENOCS": self.btn_xenocs,
            "ID02": self.btn_id02,
            "ID13": self.btn_id13,
            "Custom": self.btn_custom,
        }
        style_q_geometry_buttons(buttons, mode, self.q_manual_button)

        selected = self.selected_files()
        self.apply_preset_from_file(selected[0] if selected else None)
        if selected:
            self.display_selected_file_preview(selected[0])

    def apply_line_geometry_selection(self, name, geometry):
        values = line_geometry_to_lrphoton(geometry)
        self.center_x.setValue(values["xc"])
        self.center_y.setValue(values["yc"])
        self.distance.setValue(values["distance_m"])
        self.pixel_x.setValue(values["pixel_x_mm"])
        self.pixel_y.setValue(values["pixel_y_mm"])
        self.wavelength.setValue(values["wavelength_a"])
        self.instrument_mode = "Custom" if name not in {"XENOCS", "ID02", "ID13"} else name
        buttons = {
            "XENOCS": self.btn_xenocs,
            "ID02": self.btn_id02,
            "ID13": self.btn_id13,
            "Custom": self.btn_custom,
        }
        style_q_geometry_buttons(buttons, self.instrument_mode, self.q_manual_button)
        selected = self.selected_files()
        if selected:
            self.display_selected_file_preview(selected[0])

    def open_geometry_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Geometry + integration")
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

        settings_box = QGroupBox("Integration settings")
        settings_form = QFormLayout(settings_box)
        settings_form.setContentsMargins(*GROUP_BOX_MARGINS)
        settings_form.setSpacing(6)

        engine_combo = QComboBox()
        for index in range(self.integration_engine.count()):
            engine_combo.addItem(self.integration_engine.itemText(index))
        engine_combo.setCurrentText(self.integration_engine.currentText())
        engine_combo.setFixedWidth(150)

        points_spin = QSpinBox()
        points_spin.setRange(self.n_points.minimum(), self.n_points.maximum())
        points_spin.setValue(self.n_points.value())
        points_spin.setFixedWidth(150)

        min_pixels_spin = QSpinBox()
        min_pixels_spin.setRange(self.min_pixels_per_bin.minimum(), self.min_pixels_per_bin.maximum())
        min_pixels_spin.setValue(self.min_pixels_per_bin.value())
        min_pixels_spin.setFixedWidth(150)

        axis_mask_spin = QSpinBox()
        axis_mask_spin.setRange(self.axis_mask_pixels.minimum(), self.axis_mask_pixels.maximum())
        axis_mask_spin.setValue(self.axis_mask_pixels.value())
        axis_mask_spin.setFixedWidth(150)

        normalize_combo = QComboBox()
        for index in range(self.normalization_mode.count()):
            normalize_combo.addItem(
                self.normalization_mode.itemText(index),
                self.normalization_mode.itemData(index),
            )
        normalize_combo.setCurrentIndex(self.normalization_mode.currentIndex())
        normalize_combo.setFixedWidth(220)

        scale_spin = QDoubleSpinBox()
        scale_spin.setDecimals(self.intensity_scale.decimals())
        scale_spin.setRange(self.intensity_scale.minimum(), self.intensity_scale.maximum())
        scale_spin.setSingleStep(self.intensity_scale.singleStep())
        scale_spin.setValue(self.intensity_scale.value())
        scale_spin.setFixedWidth(150)

        settings_form.addRow("Engine", engine_combo)
        settings_form.addRow("ψ points", points_spin)
        settings_form.addRow("Min pixels/bin", min_pixels_spin)
        settings_form.addRow("Axis mask (px)", axis_mask_spin)
        settings_form.addRow("Intensity correction", normalize_combo)
        settings_form.addRow("I scale", scale_spin)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addLayout(form)
        layout.addWidget(settings_box)
        layout.addWidget(buttons)

        if dialog.exec() != QDialog.Accepted:
            return

        self.center_x.setValue(dialog_spins["center_x"].value())
        self.center_y.setValue(dialog_spins["center_y"].value())
        self.distance.setValue(dialog_spins["distance"].value())
        self.pixel_x.setValue(dialog_spins["pixel_x"].value())
        self.pixel_y.setValue(dialog_spins["pixel_y"].value())
        self.wavelength.setValue(dialog_spins["wavelength"].value())
        self.integration_engine.setCurrentText(engine_combo.currentText())
        self.n_points.setValue(points_spin.value())
        self.min_pixels_per_bin.setValue(min_pixels_spin.value())
        self.axis_mask_pixels.setValue(axis_mask_spin.value())
        self.normalization_mode.setCurrentIndex(normalize_combo.currentIndex())
        self.intensity_scale.setValue(scale_spin.value())
        self.set_instrument_mode("Custom")
        selected = self.selected_files()
        if selected:
            self.display_selected_file_preview(selected[0])

    def apply_preset_from_file(self, file_path=None):
        header = {}
        if file_path is not None and self.instrument_mode in ("XENOCS", "ID02", "ID13"):
            try:
                _, header = read_image_file(file_path)
            except Exception:
                header = {}
        self.current_header_for_line_geometry = header

        if self.instrument_mode == "XENOCS":
            cx = get_header_float(header, *CENTER_X_KEYS)
            cy = get_header_float(header, *CENTER_Y_KEYS)
            dist = get_header_float(header, "SampleDistance", "sampledistance", "sample_distance", "Distance", "DetectorDistance")
            px = get_header_float(header, "PSize_1", "psize_1", "PSize_X", "PixelSizeX", "pixel_size_x", "x_pixel_size")
            py = get_header_float(header, "PSize_2", "psize_2", "PSize_Y", "PixelSizeY", "pixel_size_y", "y_pixel_size")
            wav = get_header_float(header, "WaveLength", "Wavelength", "wavelength", "Lambda", "lambda")

            self.center_x.setValue(cx if cx is not None else 0)
            self.center_y.setValue(cy if cy is not None else 0)
            self.distance.setValue(dist if dist is not None else 0)
            self.pixel_x.setValue(px * 1000 if px is not None else 0.075000)
            self.pixel_y.setValue(py * 1000 if py is not None else 0.075000)
            self.wavelength.setValue(wav * 1e10 if wav is not None else 0)
            return

        if self.instrument_mode == "ID02":
            cx = get_header_float(header, *CENTER_X_KEYS)
            cy = get_header_float(header, *CENTER_Y_KEYS)
            dist = get_header_float(header, "SampleDistance", "sampledistance", "sample_distance", "Distance", "DetectorDistance")
            px = get_header_float(header, "PSize_1", "psize_1", "PSize_X", "PixelSizeX", "pixel_size_x", "x_pixel_size")
            py = get_header_float(header, "PSize_2", "psize_2", "PSize_Y", "PixelSizeY", "pixel_size_y", "y_pixel_size")
            wav = get_header_float(header, "WaveLength", "Wavelength", "wavelength", "Lambda", "lambda")
            self.center_x.setValue(cx if cx is not None else ID02_DEFAULT_CENTER_X)
            self.center_y.setValue(cy if cy is not None else ID02_DEFAULT_CENTER_Y)
            self.distance.setValue(dist if dist is not None else ID02_DEFAULT_DISTANCE_M)
            self.pixel_x.setValue(px * 1000 if px is not None else ID02_DEFAULT_PIXEL_MM)
            self.pixel_y.setValue(py * 1000 if py is not None else ID02_DEFAULT_PIXEL_MM)
            self.wavelength.setValue(wav * 1e10 if wav is not None else ID02_DEFAULT_WAVELENGTH_A)
            return

        if self.instrument_mode == "ID13":
            cx = get_header_float(header, *CENTER_X_KEYS)
            cy = get_header_float(header, *CENTER_Y_KEYS)
            dist = get_header_float(header, "SampleDistance", "sampledistance", "sample_distance", "Distance", "DetectorDistance")
            px = get_header_float(header, "PSize_1", "psize_1", "PSize_X", "PixelSizeX", "pixel_size_x", "x_pixel_size")
            py = get_header_float(header, "PSize_2", "psize_2", "PSize_Y", "PixelSizeY", "pixel_size_y", "y_pixel_size")
            wav = get_header_float(header, "WaveLength", "Wavelength", "wavelength", "Lambda", "lambda")
            self.center_x.setValue(cx if cx is not None else ID13_DEFAULT_CENTER_X)
            self.center_y.setValue(cy if cy is not None else ID13_DEFAULT_CENTER_Y)
            self.distance.setValue(dist if dist is not None else ID13_DEFAULT_DISTANCE_M)
            self.pixel_x.setValue(px * 1000 if px is not None else ID13_DEFAULT_PIXEL_MM)
            self.pixel_y.setValue(py * 1000 if py is not None else ID13_DEFAULT_PIXEL_MM)
            self.wavelength.setValue(wav * 1e10 if wav is not None else ID13_DEFAULT_WAVELENGTH_A)
            return

    def azimuthal_image_q_axis(self, shape, header):
        _ny, nx = shape

        selected = self.selected_files() if hasattr(self, "file_list") else []
        file_path = Path(selected[0]) if selected else None

        def scaled_axis(q_values):
            q_values = np.asarray(q_values, dtype=float)
            if q_values.size != nx:
                return None
            finite = q_values[np.isfinite(q_values)]
            if finite.size == 0:
                return None

            # LRPhoton *_azim files store the radial axis 10 times lower than the
            # nm⁻¹ values used in the Azimuthal tab q-range boxes.
            return q_values * 10.0

        def h5_scalar(value):
            try:
                return float(np.asarray(value).ravel()[0])
            except Exception:
                return None

        q_min = get_header_float(
            header,
            "QMin", "Q Min", "q_min", "qmin", "q min",
            "RadialMin", "radial_min", "radial min",
            "q_min_nm-1", "q_min_nm^-1",
        )
        q_max = get_header_float(
            header,
            "QMax", "Q Max", "q_max", "qmax", "q max",
            "RadialMax", "radial_max", "radial max",
            "q_max_nm-1", "q_max_nm^-1",
        )
        if q_min is not None and q_max is not None and q_max > q_min:
            axis = scaled_axis(np.linspace(float(q_min), float(q_max), nx))
            if axis is not None:
                return axis

        if file_path is not None and file_path.suffix.lower() in [".h5", ".hdf5"] and file_path.exists():
            try:
                with h5py.File(file_path, "r") as h5:
                    attrs_list = [h5.attrs]

                    def collect_attrs(_name, obj):
                        if hasattr(obj, "attrs"):
                            attrs_list.append(obj.attrs)

                    h5.visititems(collect_attrs)
                    found_min = None
                    found_max = None
                    for attrs in attrs_list:
                        for key, value in attrs.items():
                            key_text = str(key).lower().replace("_", " ").replace("-", " ")
                            scalar = h5_scalar(value)
                            if scalar is None:
                                continue
                            if ("q" in key_text or "radial" in key_text) and any(token in key_text for token in ["min", "start", "first"]):
                                found_min = scalar
                            if ("q" in key_text or "radial" in key_text) and any(token in key_text for token in ["max", "end", "last"]):
                                found_max = scalar

                    if found_min is not None and found_max is not None and found_max > found_min:
                        axis = scaled_axis(np.linspace(float(found_min), float(found_max), nx))
                        if axis is not None:
                            return axis
            except Exception:
                pass

        if file_path is not None and file_path.suffix.lower() in [".h5", ".hdf5"] and file_path.exists():
            try:
                with h5py.File(file_path, "r") as h5:
                    q_candidates = []

                    def collect_q_axis(name, obj):
                        if not isinstance(obj, h5py.Dataset) or obj.ndim != 1 or obj.size != nx:
                            return
                        lower = name.lower()
                        if "q" in lower:
                            axis = scaled_axis(obj[...])
                            if axis is not None:
                                q_candidates.append(axis)

                    h5.visititems(collect_q_axis)
                    if q_candidates:
                        return q_candidates[0]
            except Exception:
                pass

        if file_path is not None:
            q_range = matching_manufacturer_q_range(file_path)
            if q_range is not None:
                q_min, q_max = q_range
                axis = scaled_axis(np.linspace(float(q_min), float(q_max), nx))
                if axis is not None:
                    return axis

        fallback_max = self.q_max.value() if hasattr(self, "q_max") and self.q_max.value() > 0 else float(nx - 1)
        return np.linspace(0.0, float(fallback_max), nx)

    def display_selected_file_preview(self, file_path):
        if file_path is None:
            return

        try:
            image, header = read_image_file(file_path, frame_index=self.current_frame - 1)
        except Exception as exc:
            self.log_box.setPlainText(f"Preview error: {Path(file_path).name}: {exc}")
            self.image_canvas.raw_image = None
            self.image_canvas.set_q_map(None)
            self.image_coordinate_label.setText("ψ = - | q = - | I = -")
            clear_plot_canvas(self.image_canvas)
            return

        if self.use_q_range.isChecked():
            q_min = self.q_min.value()
            q_max = self.q_max.value()
        else:
            q_min = 0.0
            q_max = np.inf

        try:
            if is_azimuthal_processed_image(Path(file_path), header):
                q_values_full = self.azimuthal_image_q_axis(image.shape, header)
                q_step = float(np.nanmedian(np.diff(q_values_full))) if q_values_full.size > 1 else np.nan
                if np.isfinite(q_step) and q_step > 0 and np.isfinite(q_max):
                    q_min_for_mask = q_min - q_step / 2.0
                    q_max_for_mask = q_max + q_step / 2.0
                else:
                    q_min_for_mask = q_min
                    q_max_for_mask = q_max

                q_map = np.tile(q_values_full, (image.shape[0], 1))
                mask = azimuthal_processed_q_mask(
                    image.shape,
                    q_values_full,
                    q_min=q_min_for_mask,
                    q_max=q_max_for_mask,
                )
                self.image_canvas.set_coordinate_mode("azimuthal_image")
                self.image_canvas.set_q_map(q_map)
                self.image_canvas.show_image(image, 0.0, 0.0, mask=mask)
                self.sync_image_intensity_sliders()
                return

            engine = self.integration_engine.currentText()
            try:
                if engine == "pyFAI":
                    _psi, _intensity, _counts, mask, q_map = pyfai_azimuthal_average(
                        image,
                        self.center_x.value(),
                        self.center_y.value(),
                        self.distance.value(),
                        self.pixel_x.value(),
                        self.pixel_y.value(),
                        self.wavelength.value(),
                        q_min,
                        q_max,
                        self.n_points.value(),
                        self.min_pixels_per_bin.value(),
                        self.axis_mask_pixels.value(),
                    )
                else:
                    raise RuntimeError("Use LRPhoton mean preview")
            except Exception:
                _psi, _intensity, _counts, mask, q_map = azimuthal_average(
                    image,
                    self.center_x.value(),
                    self.center_y.value(),
                    self.distance.value(),
                    self.pixel_x.value(),
                    self.pixel_y.value(),
                    self.wavelength.value(),
                    q_min,
                    q_max,
                    self.n_points.value(),
                    self.min_pixels_per_bin.value(),
                    self.axis_mask_pixels.value(),
                )

            self.image_canvas.set_coordinate_mode("detector")
            self.image_canvas.set_q_map(q_map)
            self.image_canvas.show_image(image, self.center_x.value(), self.center_y.value(), mask=mask)
            self.sync_image_intensity_sliders()
        except Exception as exc:
            self.log_box.setPlainText(f"Preview error: {Path(file_path).name}: {exc}")
            self.image_canvas.set_coordinate_mode("detector")
            self.image_canvas.set_q_map(None)
            self.image_canvas.show_image(image, self.center_x.value(), self.center_y.value(), mask=None)
            self.sync_image_intensity_sliders()

    def integrate_selected_files(self):
        files = self.selected_files()
        if not files:
            self.last_results = {}
            self.last_result_paths = {}
            self.last_result_frame_counts = {}
            self.clear_graph_coordinates()
            clear_plot_canvas(self.canvas)
            return

        current_item = self.file_list.currentItem()
        if current_item is not None:
            current_file = file_path_from_item(current_item, self.current_folder)
            if current_file in files:
                files = [current_file] + [file for file in files if file != current_file]

        self.last_results = {}
        self.last_result_paths = {}
        self.last_result_frame_counts = {}
        ax = self.canvas.ax
        ax.clear()
        ax.set_axis_on()

        messages = []
        for file_path in files:
            try:
                image, header = read_image_file(file_path, frame_index=self.current_frame - 1)
                if is_azimuthal_processed_image(file_path, header):
                    q_values_full = self.azimuthal_image_q_axis(image.shape, header)
                    q_step = float(np.nanmedian(np.diff(q_values_full))) if q_values_full.size > 1 else np.nan
                    q_map = np.tile(q_values_full, (image.shape[0], 1))

                    if self.use_q_range.isChecked():
                        q_min = self.q_min.value()
                        q_max = self.q_max.value()
                    else:
                        q_min = 0
                        q_max = np.inf

                    if np.isfinite(q_step) and q_step > 0 and np.isfinite(q_max):
                        q_min_for_mask = q_min - q_step / 2.0
                        q_max_for_mask = q_max + q_step / 2.0
                    else:
                        q_min_for_mask = q_min
                        q_max_for_mask = q_max

                    psi_values, intensity, counts = azimuthal_processed_psi_profile(
                        image,
                        q_values_full,
                        q_min=q_min_for_mask,
                        q_max=q_max_for_mask,
                    )

                    if psi_values.size == 0:
                        raise ValueError("No valid ψ row found in the selected azimuthal image range.")

                    psi = rotate_psi_reference(psi_values, float(self.reference_angle.value()))
                    order = np.argsort(psi)
                    psi = psi[order]
                    intensity = intensity[order]
                    counts = counts[order]
                    normalization_factor, normalization_label = azimuthal_normalization_factor(
                        header,
                        self.normalization_mode.currentData(),
                    )
                    intensity = intensity * normalization_factor * self.intensity_scale.value()

                    ax.plot(psi, intensity, linewidth=1.2, label=file_path.stem)
                    self.last_results[file_path.stem] = (psi, intensity, counts)
                    self.last_result_paths[file_path.stem] = file_path
                    self.last_result_frame_counts[file_path.stem] = int(header.get("Number of frames", 1) or 1)

                    if file_path == files[0]:
                        self.image_canvas.set_coordinate_mode("azimuthal_image")
                        self.image_canvas.set_q_map(q_map)
                        selected_q_mask = azimuthal_processed_q_mask(
                            image.shape,
                            q_values_full,
                            q_min=q_min_for_mask,
                            q_max=q_max_for_mask,
                        )
                        self.image_canvas.show_image(image, 0.0, 0.0, mask=selected_q_mask)
                        self.sync_image_intensity_sliders()

                    messages.append(
                        f"Integrated azimuthal image as I(ψ): {file_path.name} ({psi_values.size} ψ rows)"
                        f" | q range = {q_min:.8g} -> {q_max:.8g} nm⁻¹"
                        f" | I(ψ) = average intensity of each ψ row over selected q columns, not a sum"
                        f" | intensity = {normalization_label} ; scale = {self.intensity_scale.value():.8g}"
                    )
                    continue

                if self.use_q_range.isChecked():
                    q_min = self.q_min.value()
                    q_max = self.q_max.value()
                else:
                    q_min = 0
                    q_max = np.inf

                engine = self.integration_engine.currentText()
                if engine == "pyFAI":
                    try:
                        psi, intensity, counts, mask, q_map = pyfai_azimuthal_average(
                            image,
                            self.center_x.value(),
                            self.center_y.value(),
                            self.distance.value(),
                            self.pixel_x.value(),
                            self.pixel_y.value(),
                            self.wavelength.value(),
                            q_min,
                            q_max,
                            self.n_points.value(),
                            self.min_pixels_per_bin.value(),
                            self.axis_mask_pixels.value(),
                        )
                    except Exception as pyfai_error:
                        psi, intensity, counts, mask, q_map = azimuthal_average(
                            image,
                            self.center_x.value(),
                            self.center_y.value(),
                            self.distance.value(),
                            self.pixel_x.value(),
                            self.pixel_y.value(),
                            self.wavelength.value(),
                            q_min,
                            q_max,
                            self.n_points.value(),
                            self.min_pixels_per_bin.value(),
                            self.axis_mask_pixels.value(),
                        )
                        engine = f"LRPhoton mean fallback, pyFAI error: {pyfai_error}"
                else:
                    psi, intensity, counts, mask, q_map = azimuthal_average(
                        image,
                        self.center_x.value(),
                        self.center_y.value(),
                        self.distance.value(),
                        self.pixel_x.value(),
                        self.pixel_y.value(),
                        self.wavelength.value(),
                        q_min,
                        q_max,
                        self.n_points.value(),
                        self.min_pixels_per_bin.value(),
                        self.axis_mask_pixels.value(),
                    )

                psi = rotate_psi_reference(psi, float(self.reference_angle.value()))
                order = np.argsort(psi)
                psi = psi[order]
                intensity = intensity[order]
                counts = counts[order]

                normalization_factor, normalization_label = azimuthal_normalization_factor(
                    header,
                    self.normalization_mode.currentData(),
                )
                intensity = intensity * normalization_factor * self.intensity_scale.value()

                ax.plot(psi, intensity, linewidth=1.2, label=file_path.stem)
                self.last_results[file_path.stem] = (psi, intensity, counts)
                self.last_result_paths[file_path.stem] = file_path
                self.last_result_frame_counts[file_path.stem] = int(header.get("Number of frames", 1) or 1)

                if file_path == files[0]:
                    self.image_canvas.set_q_map(q_map)
                    self.image_canvas.show_image(image, self.center_x.value(), self.center_y.value(), mask=mask)
                    self.sync_image_intensity_sliders()

                messages.append(
                    f"Integrated: {file_path.name} ({psi.size} ψ points) | q crown = {q_min:.8g} -> {q_max:.8g} nm⁻¹"
                    f" | engine = {engine} ; min {self.min_pixels_per_bin.value()} pixels/bin"
                    f" ; axis mask = {self.axis_mask_pixels.value()} px ; I<=0 masked"
                    f" | intensity = {normalization_label} ; scale = {self.intensity_scale.value():.8g}"
                )

            except Exception as error:
                messages.append(f"Error: {file_path.name}: {error}")

        first_result_path = next(iter(self.last_result_paths.values()), None)
        if first_result_path is not None:
            try:
                first_image, first_header = read_image_file(first_result_path, frame_index=self.current_frame - 1)
                if is_azimuthal_processed_image(first_result_path, first_header):
                    self.apply_plot_axes()
                else:
                    self.apply_plot_axes()
            except Exception:
                self.apply_plot_axes()
        else:
            self.apply_plot_axes()
        apply_plot_display_style(ax)
        if ax.get_xlabel().startswith("ψ"):
            ax.set_xlim(0, 360)
        if self.last_results and self.show_legend.isChecked():
            self.legend = make_plot_legend(ax)
        finalize_plot_canvas(self.canvas)
        self.canvas.draw()
        self.canvas.flush_events()
        self.log_box.setPlainText("\n".join(messages))

    def apply_plot_axes(self):
        ax = self.canvas.ax
        ax.set_xlabel("ψ / °")
        ax.set_ylabel("Intensity / a.u.")
        ax.set_xscale("linear")
        ax.set_yscale("linear")

    def apply_iq_plot_axes(self):
        ax = self.canvas.ax
        ax.set_xlabel("q / nm⁻¹")
        ax.set_ylabel("Intensity / a.u.")
        ax.set_xscale("linear")
        ax.set_yscale("linear")

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
            self.canvas.draw()
            self.canvas.flush_events()

    def update_graph_coordinates(self, event):
        if event.inaxes != self.canvas.ax or event.xdata is None or event.ydata is None:
            return

        try:
            self.graph_coordinate_label.setText(
                f"ψ = {event.xdata:.6g}° | I = {event.ydata:.6g}"
            )
        except Exception:
            self.graph_coordinate_label.setText("ψ = - | I = -")

    def clear_graph_coordinates(self, event=None):
        self.graph_coordinate_label.setText("ψ = - | I = -")

    def on_graph_right_click(self, event):
        if event.button != 3 or event.inaxes != self.canvas.ax:
            return

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
        self.canvas.draw()
        self.canvas.flush_events()

    def ask_text(self, title, label, text):
        from PySide6.QtWidgets import QInputDialog
        return QInputDialog.getText(self, title, label, text=text)

    def save_results(self):
        if not self.last_results:
            QMessageBox.warning(self, "No results", "No azimuthal integration result to save.")
            return

        if self.use_q_range.isChecked():
            range_suffix = f"_q{self.q_min.value():.8g}-{self.q_max.value():.8g}nm-1"
        else:
            range_suffix = "_qfull"

        for source_stem, (psi, intensity, counts) in self.last_results.items():
            source_path = self.last_result_paths.get(source_stem)
            frame_count = self.last_result_frame_counts.get(source_stem, 1)
            is_h5 = source_path is not None and source_path.suffix.lower() in [".h5", ".hdf5"]
            frame_suffix = f"_frame{self.current_frame:04d}" if is_h5 and frame_count > 1 else ""
            out_file = self.current_folder / f"{source_stem}{frame_suffix}{range_suffix}_azimProf.dat"
            data = np.column_stack([psi, intensity, counts])
            with open(out_file, "w", encoding="utf-8") as file:
                file.write(f"# integration_engine {self.integration_engine.currentText()}\n")
                file.write(f"# psi_points {self.n_points.value()}\n")
                file.write(f"# min_pixels_per_bin {self.min_pixels_per_bin.value()}\n")
                file.write(f"# axis_mask_px {self.axis_mask_pixels.value()}\n")
                file.write("# nonpositive_pixels masked\n")
                file.write(f"# intensity_correction {self.normalization_mode.currentText()}\n")
                file.write(f"# intensity_scale {self.intensity_scale.value():.8g}\n")
                source_path = self.last_result_paths.get(source_stem)
                try:
                    _image, source_header = read_image_file(source_path, frame_index=self.current_frame - 1) if source_path is not None else (None, {})
                    if source_path is not None and is_azimuthal_processed_image(source_path, source_header):
                        file.write("# For *_azim input images, saved I_psi is the average intensity of each ψ row over selected q columns, not a sum.\n")
                        file.write("# psi_deg I_psi pixel_count\n")
                    else:
                        file.write("# psi_deg I_psi pixel_count\n")
                except Exception:
                    file.write("# psi_deg I_psi pixel_count\n")
                np.savetxt(file, data, fmt="%.8e %.8e %d")

    def open_azimuthal_test_dialog(self):
        files = self.selected_files()
        if not files:
            return

        file_path = Path(files[0])
        try:
            image, header = read_image_file(file_path, frame_index=self.current_frame - 1)
        except Exception as exc:
            QMessageBox.warning(self, "Test", f"Could not read selected image:\n{exc}")
            return

        try:
            header_xc = get_header_float(header, "Center_1", "center_1", "CenterX", "center_x")
            header_yc = get_header_float(header, "Center_2", "center_2", "CenterY", "center_y")
            center_source = "Center_1/Center_2"

            if header_xc is None or header_yc is None:
                header_xc = get_header_float(header, "Theoretical_Center_1", "theoretical_center_1")
                header_yc = get_header_float(header, "Theoretical_Center_2", "theoretical_center_2")
                center_source = "Theoretical fallback"

            geometry = {
                "xc": float(header_xc) if header_xc is not None else float(self.center_x.value()),
                "yc": float(header_yc) if header_yc is not None else float(self.center_y.value()),
                "center_source": center_source,
                "distance_m": float(self.distance.value()),
                "pixel_x_mm": float(self.pixel_x.value()),
                "pixel_y_mm": float(self.pixel_y.value()),
                "wavelength_a": float(self.wavelength.value()),
                "q_min": float(self.q_min.value()) if self.use_q_range.isChecked() else 0.0,
                "q_max": float(self.q_max.value()) if self.use_q_range.isChecked() else 1.0,
                "psi_points": int(self.n_points.value()),
                "axis_mask_px": int(self.axis_mask_pixels.value()),
                "min_pixels": int(self.min_pixels_per_bin.value()),
                "reference_angle_deg": float(self.reference_angle.value()),
            }
        except Exception as exc:
            QMessageBox.warning(self, "Test", f"Invalid azimuthal parameters:\n{exc}")
            return

        dialog = PyFAIAzimuthalTestDialog(
            self,
            image,
            file_path,
            geometry,
            header=header,
            frame_count=self.total_frames,
            frame_index=self.current_frame - 1,
        )
        self._azimuthal_test_dialog = dialog
        dialog.show()
