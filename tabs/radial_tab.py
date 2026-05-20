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
    QComboBox,
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


def read_image_file(file_path):
    suffix = Path(file_path).suffix.lower()
    if suffix == ".edf":
        return read_edf_file(file_path)
    if suffix in [".h5", ".hdf5"]:
        return read_h5_first_image(file_path)
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
# ======================= RADIAL TOOLS ========================
# ============================================================

def radial_average(image, xc, yc, distance_m, pixel_x_mm, pixel_y_mm, wavelength_a, q_min, q_max, n_bins, log_bins, sector_min=0, sector_max=360):
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
    dx_px = x + 1 - xc
    dy_px = y + 1 - yc
    dx_m = dx_px * pixel_x_mm * 1e-3
    dy_m = dy_px * pixel_y_mm * 1e-3
    r_m = np.sqrt(dx_m ** 2 + dy_m ** 2)

    two_theta = np.arctan2(r_m, distance_m)
    theta = two_theta / 2
    wavelength_nm = wavelength_a * 0.1
    q = (4 * np.pi / wavelength_nm) * np.sin(theta)

    psi = (np.degrees(np.arctan2(dy_px, dx_px)) + 360) % 360
    sector_min = sector_min % 360
    sector_max = sector_max % 360

    if abs((sector_max - sector_min) % 360) < 1e-9:
        sector_mask = np.ones_like(psi, dtype=bool)
    elif sector_min <= sector_max:
        sector_mask = (psi >= sector_min) & (psi <= sector_max)
    else:
        sector_mask = (psi >= sector_min) | (psi <= sector_max)

    valid = np.isfinite(img) & np.isfinite(q) & sector_mask
    valid &= img < 4e9

    if q_min > 0:
        valid &= q >= q_min
    if q_max > 0:
        valid &= q <= q_max

    q_values = q[valid]
    i_values = img[valid]

    if q_values.size == 0:
        raise ValueError("No valid pixel found in the selected q range / sector.")

    if q_min <= 0:
        q_min_eff = float(np.nanmin(q_values[q_values > 0]))
    else:
        q_min_eff = q_min

    if q_max <= 0:
        q_max_eff = float(np.nanmax(q_values))
    else:
        q_max_eff = q_max

    if q_max_eff <= q_min_eff:
        raise ValueError("q max must be greater than q min.")

    # Match SAXSutilities-style radial profiles more closely:
    # use a very fine internal binning, then reduce to the requested number of points.
    # This avoids under-sampling the steep low-q rise when the final curve is displayed in log scale.
    internal_bins = max(int(n_bins) * 8, 4000)

    if log_bins:
        edges = np.logspace(np.log10(q_min_eff), np.log10(q_max_eff), internal_bins + 1)
    else:
        edges = np.linspace(q_min_eff, q_max_eff, internal_bins + 1)

    sums, _ = np.histogram(q_values, bins=edges, weights=i_values)
    counts, _ = np.histogram(q_values, bins=edges)

    with np.errstate(invalid="ignore", divide="ignore"):
        intensity = sums / counts

    if log_bins:
        q_axis = np.sqrt(edges[:-1] * edges[1:])
    else:
        q_axis = 0.5 * (edges[:-1] + edges[1:])

    valid_bins = counts > 0
    q_axis = q_axis[valid_bins]
    intensity = intensity[valid_bins]
    counts = counts[valid_bins]

    if q_axis.size > n_bins:
        index = np.linspace(0, q_axis.size - 1, int(n_bins))
        index = np.unique(np.round(index).astype(int))
        q_axis = q_axis[index]
        intensity = intensity[index]
        counts = counts[index]

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
        self.raw_image = None
        self.coordinate_label = None

        self.mpl_connect("scroll_event", self._on_scroll)
        self.mpl_connect("button_press_event", self._on_press)
        self.mpl_connect("button_release_event", self._on_release)
        self.mpl_connect("motion_notify_event", self._on_motion)

    def set_coordinate_label(self, label):
        self.coordinate_label = label

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

                self.coordinate_label.setText(
                    f"x = {x_index + 1} | y = {y_index + 1} | I = {value_text}"
                )
            else:
                self.coordinate_label.setText("x = - | y = - | I = -")

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

        self.ax.imshow(display, origin="upper", cmap="jet", interpolation="nearest")

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
            angle_marks = [0, 90, 180, 270]
            for angle in angle_marks:
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

        self.ax.set_xlabel("x / px")
        self.ax.set_ylabel("y / px")
        self.ax.tick_params(axis="both", colors="black", labelsize=8)
        self.ax.set_aspect("equal")
        if had_image:
            self.ax.set_xlim(current_xlim)
            self.ax.set_ylim(current_ylim)

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
        self._syncing_folder = False

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

        graph_box = QGroupBox("I(q) graph")
        graph_layout = QVBoxLayout(graph_box)
        graph_layout.setContentsMargins(6, 18, 6, 6)
        right_layout.addWidget(graph_box, stretch=2)

        image_box = QGroupBox("Image / selected integration area")
        image_layout = QVBoxLayout(image_box)
        image_layout.setContentsMargins(6, 18, 6, 6)
        right_layout.addWidget(image_box, stretch=1)
        image_box.setMaximumHeight(260)

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
        self.use_q_crown = QCheckBox("Use q crown")
        self.use_q_crown.setChecked(False)
        self.use_q_crown.stateChanged.connect(self.update_mask_parameter_state)

        self.q_min = self.double_spin(0, decimals=6, minimum=0)
        self.q_max = self.double_spin(0, decimals=6, minimum=0)

        self.use_sector = QCheckBox("Use azimuthal sector")
        self.use_sector.setChecked(False)
        self.use_sector.stateChanged.connect(self.update_mask_parameter_state)

        self.sector_min = self.double_spin(0, decimals=3, minimum=-360)
        self.sector_max = self.double_spin(360, decimals=3, minimum=-360)
        self.n_bins = QSpinBox()
        self.n_bins.setRange(10, 10000)
        self.n_bins.setValue(1500)
        self.plot_mode = QComboBox()
        self.plot_mode.addItems(["linear linear", "linear log", "log log", "log linear", "Kratky"])
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

        form.addWidget(self.use_q_crown, 6, 0, 1, 2)
        form.addWidget(QLabel("q min (nm⁻¹):"), 7, 0)
        form.addWidget(self.q_min, 7, 1)
        form.addWidget(QLabel("q max (nm⁻¹):"), 8, 0)
        form.addWidget(self.q_max, 8, 1)

        form.addWidget(self.use_sector, 9, 0, 1, 2)
        form.addWidget(QLabel("Sector min ψ (°):"), 10, 0)
        form.addWidget(self.sector_min, 10, 1)
        form.addWidget(QLabel("Sector max ψ (°):"), 11, 0)
        form.addWidget(self.sector_max, 11, 1)

        form.addWidget(QLabel("Bins:"), 12, 0)
        form.addWidget(self.n_bins, 12, 1)
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
        graph_layout.addWidget(self.canvas)

        self.image_canvas = ImageCanvas()
        self.image_coordinate_label = QLabel("x = - | y = - | I = -")
        self.image_coordinate_label.setMinimumHeight(26)
        self.image_coordinate_label.setAlignment(Qt.AlignCenter)
        self.image_coordinate_label.setStyleSheet("""
            QLabel {
                background-color: #f4f4f4;
                border-radius: 8px;
                padding: 5px;
                font-family: Menlo, Monaco, monospace;
                font-size: 11px;
            }
        """)
        self.image_canvas.set_coordinate_label(self.image_coordinate_label)
        image_layout.addWidget(self.image_canvas, stretch=1)
        image_layout.addWidget(self.image_coordinate_label, stretch=0)

        self.canvas.mpl_connect("button_press_event", self.on_graph_right_click)

        self.btn_xenocs.clicked.connect(lambda: self.set_instrument_mode("XENOCS"))
        self.btn_id02.clicked.connect(lambda: self.set_instrument_mode("ID02"))
        self.btn_id13.clicked.connect(lambda: self.set_instrument_mode("ID13"))
        self.btn_custom.clicked.connect(lambda: self.set_instrument_mode("Custom"))
        self.update_mask_parameter_state()
    def update_mask_parameter_state(self):
        use_q = self.use_q_crown.isChecked()
        self.q_min.setEnabled(use_q)
        self.q_max.setEnabled(use_q)

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
            self.wavelength, self.use_q_crown, self.use_sector,
            self.n_bins, self.plot_mode, self.integrate_button, self.save_button,
        ]:
            widget.setEnabled(enabled)
        self.update_mask_parameter_state()

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
        self.set_controls_enabled(bool(selected))
        if selected:
            self.apply_preset_from_file(selected[0])

    def selected_files(self):
        return [self.current_folder / item.text() for item in self.file_list.selectedItems()]

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

        selected = self.selected_files()
        self.apply_preset_from_file(selected[0] if selected else None)

    def apply_preset_from_file(self, file_path=None):
        header = {}
        if file_path is not None and self.instrument_mode == "XENOCS":
            try:
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
            self.wavelength.setValue(wav * 1e10 if wav is not None else 0)
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

        self.last_results = {}
        ax = self.canvas.ax
        ax.clear()

        messages = []
        for file_path in files:
            try:
                image, _ = read_image_file(file_path)
                q_min = self.q_min.value() if self.use_q_crown.isChecked() else 0
                q_max = self.q_max.value() if self.use_q_crown.isChecked() else 0
                sector_min = self.sector_min.value() if self.use_sector.isChecked() else 0
                sector_max = self.sector_max.value() if self.use_sector.isChecked() else 360
                use_log_bins = self.plot_mode.currentText() in ["log log", "log linear"]

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
                line, = ax.plot(q, y, linewidth=1.2, label=file_path.stem)
                self.last_results[file_path.stem] = (q, intensity, counts)

                if file_path == files[0]:
                    self.image_canvas.show_image(image, self.center_x.value(), self.center_y.value(), mask=mask)
                messages.append(f"Integrated: {file_path.name} ({q.size} bins)")

            except Exception as error:
                messages.append(f"Error: {file_path.name}: {error}")

        self.apply_plot_axes()
        ax.grid(True)
        if self.last_results:
            self.legend = ax.legend(loc="best")
        self.canvas.draw_idle()
        self.log_box.setPlainText("\n".join(messages))


    def make_plot_y(self, q, intensity):
        if self.plot_mode.currentText() == "Kratky":
            return q ** 2 * intensity
        return intensity

    def apply_plot_axes(self):
        ax = self.canvas.ax
        mode = self.plot_mode.currentText()

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
            ax.set_xscale("linear")
            ax.set_yscale("linear")

    def update_plot_mode(self):
        ax = self.canvas.ax
        mode = self.plot_mode.currentText()

        for line in ax.get_lines():
            label = line.get_label()
            if label in self.last_results:
                q, intensity, counts = self.last_results[label]
                line.set_ydata(self.make_plot_y(q, intensity))

        self.apply_plot_axes()
        self.canvas.draw_idle()


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

        range_parts = []

        if self.use_q_crown.isChecked():
            range_parts.append(f"q{self.q_min.value():.4g}-{self.q_max.value():.4g}nm-1")
        else:
            range_parts.append("qfull")

        if self.use_sector.isChecked():
            range_parts.append(f"psi{self.sector_min.value():.3g}-{self.sector_max.value():.3g}deg")
        else:
            range_parts.append("psi360")

        range_suffix = "_" + "_".join(range_parts)

        for filename, (q, intensity, counts) in self.last_results.items():
            source_stem = Path(filename).stem
            out_file = self.current_folder / f"{source_stem}{range_suffix}_azimAvg.dat"
            data = np.column_stack([q, intensity, counts])
            with open(out_file, "w", encoding="utf-8") as file:
                file.write("# q_nm-1 I_q pixel_count\n")
                np.savetxt(file, data, fmt="%.8e %.8e %d")

        QMessageBox.information(self, "Saved", "Radial profiles saved in the current folder.")
