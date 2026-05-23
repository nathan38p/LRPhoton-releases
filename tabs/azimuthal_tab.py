import re
from pathlib import Path

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
        elif dataset.ndim == 3:
            shape = dataset.shape
            frame_axis = int(np.argmin(shape))
            n_frames = int(shape[frame_axis])
            frame_index = int(np.clip(frame_index, 0, n_frames - 1))

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
# AZIMUTHAL INTEGRATION TOOLS
# ============================================================

def azimuthal_average(image, xc, yc, distance_m, pixel_x_mm, pixel_y_mm, wavelength_a, q_min, q_max, psi_points):
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
    valid &= img < 4e9
    valid &= q >= q_min
    valid &= q <= q_max

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

    valid_bins = counts > 0
    return psi_mean[valid_bins], intensity[valid_bins], counts[valid_bins], valid, q


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
        self.ax.set_axis_off()
        self.ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        self.fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

        self._dragging = False
        self._drag_start = None
        self._xlim_start = None
        self._ylim_start = None
        self._base_scale = 1.18
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

        self.mpl_connect("scroll_event", self._on_scroll)
        self.mpl_connect("button_press_event", self._on_press)
        self.mpl_connect("button_release_event", self._on_release)
        self.mpl_connect("motion_notify_event", self._on_motion)

    def set_coordinate_label(self, label):
        self.coordinate_label = label

    def set_q_map(self, q_map):
        self.q_map = q_map

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
        self.last_mask = mask

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
            self.ax.axvline(xc - 1, color="red", linewidth=1.0)
            self.ax.axhline(yc - 1, color="red", linewidth=1.0)
            self.ax.plot(xc - 1, yc - 1, "wo", markersize=4)

            ny, nx = image.shape
            radius = min(nx, ny) * 0.35
            for angle in [0, 90, 180, 270]:
                rad = np.deg2rad(angle)
                x_text = (xc - 1) + radius * np.cos(rad)
                y_text = (yc - 1) + radius * np.sin(rad)
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
        self.ax.set_aspect("equal")

        if had_image:
            self.ax.set_xlim(current_xlim)
            self.ax.set_ylim(current_ylim)

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
        self._syncing_folder = False
        self.current_frame = 1
        self.total_frames = 1

        self.build_ui()
        self.refresh_files()
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
        self.n_points = QSpinBox()
        self.n_points.setRange(10, 10000)
        self.n_points.setValue(360)
        self.n_points.setFixedHeight(24)

        parameter_field_width = 130
        self.q_min.setMinimumWidth(parameter_field_width)
        self.q_max.setMinimumWidth(parameter_field_width)
        self.n_points.setMinimumWidth(parameter_field_width)

        form.addWidget(self.use_q_range, 0, 0, 1, 2)
        form.addWidget(QLabel("q min (nm⁻¹):"), 1, 0)
        form.addWidget(self.q_min, 1, 1)
        form.addWidget(QLabel("q max (nm⁻¹):"), 2, 0)
        form.addWidget(self.q_max, 2, 1)
        form.addWidget(QLabel("ψ points:"), 3, 0)
        form.addWidget(self.n_points, 3, 1)
        params_layout.addLayout(form)

        self.integrate_button = QPushButton("Integrate I(ψ)")
        self.integrate_button.clicked.connect(self.integrate_selected_files)
        params_layout.addWidget(self.integrate_button)

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
            save_callback=self.toolbar.save_figure,
            save_tooltip="Save graph",
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
            self.integrate_button, self.show_legend,
            self.frame_start_spin, self.frame_end_spin, self.prev_frame_button,
            self.next_frame_button, self.frame_slider,
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
        self.update_frame_navigation_state()
        self.update_q_range_state()

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
        self.set_controls_enabled(bool(selected))
        self.image_canvas.reset_display_limits()
        if selected:
            self.apply_preset_from_file(selected[0])
            self.update_frame_controls_from_file(selected[0])
            self.display_selected_file_preview(selected[0])
        else:
            self.update_frame_controls_from_file(None)
            self.last_results = {}
            self.clear_graph_coordinates()
            self.image_canvas.raw_image = None
            self.image_canvas.set_q_map(None)
            self.image_coordinate_label.setText("ψ = - | q = - | I = -")
            clear_plot_canvas(self.canvas)
            clear_plot_canvas(self.image_canvas)

    def selected_files(self):
        return [file_path_from_item(item, self.current_folder) for item in self.file_list.selectedItems()]

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

        if self.selected_files():
            self.integrate_selected_files()

    def previous_frame(self):
        value = max(self.frame_slider.minimum(), self.frame_slider.value() - 1)
        self.frame_slider.setValue(value)

    def next_frame(self):
        value = min(self.frame_slider.maximum(), self.frame_slider.value() + 1)
        self.frame_slider.setValue(value)

    def set_instrument_mode(self, mode):
        self.instrument_mode = mode
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
            self.wavelength.setValue(wav * 1e10 if wav is not None else 0)
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

    def integrate_selected_files(self):
        files = self.selected_files()
        if not files:
            self.last_results = {}
            self.clear_graph_coordinates()
            clear_plot_canvas(self.canvas)
            return

        self.last_results = {}
        ax = self.canvas.ax
        ax.clear()
        ax.set_axis_on()

        messages = []
        for file_path in files:
            try:
                image, _ = read_image_file(file_path, frame_index=self.current_frame - 1)
                if self.use_q_range.isChecked():
                    q_min = self.q_min.value()
                    q_max = self.q_max.value()
                else:
                    q_min = 0
                    q_max = np.inf

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
                )

                ax.plot(psi, intensity, linewidth=1.2, label=file_path.stem)
                self.last_results[file_path.stem] = (psi, intensity, counts)

                if file_path == files[0]:
                    self.image_canvas.set_q_map(q_map)
                    self.image_canvas.show_image(image, self.center_x.value(), self.center_y.value(), mask=mask)
                    self.sync_image_intensity_sliders()

                messages.append(
                    f"Integrated: {file_path.name} ({psi.size} ψ points) | q crown = {q_min:.8g} -> {q_max:.8g} nm⁻¹"
                )

            except Exception as error:
                messages.append(f"Error: {file_path.name}: {error}")

        self.apply_plot_axes()
        apply_plot_display_style(ax)
        ax.set_xlim(0, 360)
        if self.last_results and self.show_legend.isChecked():
            self.legend = make_plot_legend(ax)
        finalize_plot_canvas(self.canvas)
        self.log_box.setPlainText("\n".join(messages))

    def apply_plot_axes(self):
        ax = self.canvas.ax
        ax.set_xlabel("ψ / °")
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

        for filename, (psi, intensity, counts) in self.last_results.items():
            source_stem = Path(filename).stem
            out_file = self.current_folder / f"{source_stem}{range_suffix}_azimProf.dat"
            data = np.column_stack([psi, intensity, counts])
            with open(out_file, "w", encoding="utf-8") as file:
                file.write("# psi_deg I_psi pixel_count\n")
                np.savetxt(file, data, fmt="%.8e %.8e %d")

        QMessageBox.information(self, "Saved", "Azimuthal profiles saved in the current folder.")

    def display_selected_file_preview(self, file_path):
        try:
            image, _ = read_image_file(file_path, frame_index=self.current_frame - 1)

            y, x = np.indices(image.shape)
            dx_px = x + 1 - self.center_x.value()
            dy_px = y + 1 - self.center_y.value()
            dx_m = dx_px * self.pixel_x.value() * 1e-3
            dy_m = dy_px * self.pixel_y.value() * 1e-3
            r_m = np.sqrt(dx_m ** 2 + dy_m ** 2)
            two_theta = np.arctan2(r_m, self.distance.value())
            theta = two_theta / 2
            wavelength_nm = self.wavelength.value() * 0.1
            q_map = (4 * np.pi / wavelength_nm) * np.sin(theta)

            self.image_canvas.set_q_map(q_map)
            self.image_canvas.show_image(image, self.center_x.value(), self.center_y.value(), mask=None)
            self.sync_image_intensity_sliders()
            self.image_coordinate_label.setText("ψ = - | q = - | I = -")
        except Exception as error:
            self.image_canvas.raw_image = None
            self.image_canvas.set_q_map(None)
            self.image_coordinate_label.setText("ψ = - | q = - | I = -")
            QMessageBox.warning(self, "Preview error", str(error))
