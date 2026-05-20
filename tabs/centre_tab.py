from pathlib import Path
import re
import h5py

import numpy as np
from PySide6.QtCore import Qt, QEvent
from PySide6.QtWidgets import (
    QWidget, QFileDialog, QMessageBox, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QDoubleSpinBox, QSpinBox, QTextEdit, QGroupBox
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ============================================================
# ======================= EDF TOOLS ==========================
# Everything is kept in this tab: no utils folder.
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
            k, v = part.split("=", 1)
            header[k.strip()] = v.strip()

    return header


def edf_dtype_to_numpy(data_type: str):
    s = data_type.strip().lower()

    if s in ("floatvalue", "float"):
        return np.float32
    if s in ("doublevalue", "double"):
        return np.float64
    if s in ("unsignedshort",):
        return np.uint16
    if s in ("signedshort",):
        return np.int16
    if s in ("unsignedinteger", "uint32"):
        return np.uint32
    if s in ("signedinteger", "int32"):
        return np.int32
    if s in ("unsignedbyte", "uint8"):
        return np.uint8
    if s in ("signedbyte", "int8"):
        return np.int8

    raise ValueError(f"Unsupported EDF data type: {data_type}")


def read_edf_file(filename: str):
    filename = Path(filename)

    with open(filename, "rb") as f:
        first = f.read(8192).decode("latin-1", errors="ignore")

    m = re.search(r"EDF_HeaderSize\s*=\s*(\d+)", first)
    if not m:
        raise ValueError("EDF_HeaderSize not found in header.")

    header_size = int(m.group(1))

    with open(filename, "rb") as f:
        raw_header_bytes = f.read(header_size)
        raw_header_text = raw_header_bytes.decode("latin-1", errors="ignore")

    header = parse_edf_header(raw_header_text)

    byte_order = header.get("ByteOrder", "LowByteFirst")
    data_type_str = header.get("DataType", "FloatValue")
    dim1 = int(float(header.get("Dim_1")))
    dim2 = int(float(header.get("Dim_2")))

    dtype = np.dtype(edf_dtype_to_numpy(data_type_str))
    if byte_order.lower() == "highbytefirst":
        dtype = dtype.newbyteorder(">")
    else:
        dtype = dtype.newbyteorder("<")

    with open(filename, "rb") as f:
        f.seek(header_size)
        data = np.fromfile(f, dtype=dtype, count=dim1 * dim2)

    if data.size != dim1 * dim2:
        raise ValueError(f"Incorrect data size: expected {dim1 * dim2}, read {data.size}.")

    img = data.reshape((dim2, dim1)).astype(np.float64)
    return img, header, raw_header_text, byte_order, data_type_str


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
            img = np.asarray(dataset[...], dtype=np.float64)
        elif dataset.ndim == 3:
            shape = dataset.shape
            frame_axis = int(np.argmin(shape))

            if frame_axis == 0:
                img = np.asarray(dataset[0, :, :], dtype=np.float64)
                header["Displayed frame"] = "0 from axis 0"
            elif frame_axis == 1:
                img = np.asarray(dataset[:, 0, :], dtype=np.float64)
                header["Displayed frame"] = "0 from axis 1"
            else:
                img = np.asarray(dataset[:, :, 0], dtype=np.float64)
                header["Displayed frame"] = "0 from axis 2"
        else:
            raise ValueError("Only 2D and 3D H5 datasets are supported here.")

    return img, header


# ============================================================
# ======================= I(q) COMPUTATION ===================
# ============================================================

def compute_directional_iq_nm_general(img, xc, yc, angle_deg, half_width,
                                      d_m, px_x_mm, px_y_mm, lambda_a):
    ny, nx = img.shape

    px_x_m = px_x_mm * 1e-3
    px_y_m = px_y_mm * 1e-3
    lambda_m = lambda_a * 1e-10

    theta = np.deg2rad(angle_deg)

    ux = np.cos(theta)
    uy = np.sin(theta)
    vx = -np.sin(theta)
    vy = np.cos(theta)

    corners_x = np.array([1, nx, 1, nx], dtype=float) - xc
    corners_y = np.array([1, 1, ny, ny], dtype=float) - yc

    t_corners = corners_x * ux + corners_y * uy
    t_max = int(np.floor(np.max(t_corners)))

    if t_max < 0:
        return np.array([]), np.array([])

    t_vals = np.arange(0, t_max + 1, dtype=float)
    q_vals = np.full_like(t_vals, np.nan, dtype=float)
    i_vals = np.full_like(t_vals, np.nan, dtype=float)

    half_width = int(round(half_width))

    for i, t in enumerate(t_vals):
        if half_width == 0:
            x = int(round(xc + t * ux))
            y = int(round(yc + t * uy))

            if 1 <= x <= nx and 1 <= y <= ny:
                i_vals[i] = img[y - 1, x - 1]
        else:
            vals = []
            for w in range(-half_width, half_width + 1):
                x = int(round(xc + t * ux + w * vx))
                y = int(round(yc + t * uy + w * vy))

                if 1 <= x <= nx and 1 <= y <= ny:
                    vals.append(img[y - 1, x - 1])

            if vals:
                vals = np.asarray(vals, dtype=float)
                if np.any(np.isfinite(vals)):
                    i_vals[i] = np.nanmean(vals)

        dx_m = t * ux * px_x_m
        dy_m = t * uy * px_y_m
        r_m = np.sqrt(dx_m ** 2 + dy_m ** 2)
        two_theta = np.arctan(r_m / d_m)
        q_vals[i] = ((4 * np.pi / lambda_m) * np.sin(two_theta / 2)) / 1e9

    return q_vals, i_vals


# ============================================================
# ======================= IMAGE CANVAS =======================
# ============================================================

class ImageCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(5, 5), tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)

        self.raw_image = None
        self.im = None
        self.cbar = None
        self.coordinate_label = None

        self._dragging = False
        self._drag_start = None
        self._xlim_start = None
        self._ylim_start = None
        self.setFocusPolicy(Qt.StrongFocus)

        self.mpl_connect("button_press_event", self._on_press)
        self.mpl_connect("button_release_event", self._on_release)
        self.mpl_connect("motion_notify_event", self._on_motion)

        try:
            self.grabGesture(Qt.PinchGesture)
        except Exception:
            pass

    def set_coordinate_label(self, label):
        self.coordinate_label = label

    def event(self, event):
        if event.type() == QEvent.NativeGesture and self.raw_image is not None:
            try:
                gesture_type = event.gestureType()
                value = event.value()

                if gesture_type == Qt.ZoomNativeGesture and value != 0:
                    scale = 1.0 / (1.0 + value) if value > -0.95 else 1.25
                    self._zoom_from_qpoint(self._event_center_point(event), scale)
                    event.accept()
                    return True

                if gesture_type == Qt.SmartZoomNativeGesture:
                    self._reset_zoom()
                    event.accept()
                    return True
            except Exception:
                pass

        if event.type() == QEvent.Gesture and self.raw_image is not None:
            try:
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

        modifiers = event.modifiers()
        delta = event.angleDelta()
        dx = delta.x() / 120.0
        dy = delta.y() / 120.0

        if modifiers & Qt.ControlModifier or modifiers & Qt.MetaModifier:
            self._zoom_from_qt_event(event, dy)
        else:
            self._pan_from_wheel(dx, dy)

        event.accept()

    def _event_center_point(self, event):
        try:
            position = event.position()
            if position is not None:
                return position
        except Exception:
            pass

        return self.rect().center()

    def _qpoint_to_data_pos(self, qpoint):
        try:
            x_widget = float(qpoint.x())
            y_widget = float(qpoint.y())
        except Exception:
            x_widget = self.width() / 2
            y_widget = self.height() / 2

        bbox = self.ax.get_window_extent()
        x_fig = bbox.x0 + (x_widget / max(self.width(), 1)) * bbox.width
        y_fig = bbox.y1 - (y_widget / max(self.height(), 1)) * bbox.height

        xdata, ydata = self.ax.transData.inverted().transform((x_fig, y_fig))

        if not np.isfinite(xdata) or not np.isfinite(ydata):
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            xdata = (xlim[0] + xlim[1]) / 2
            ydata = (ylim[0] + ylim[1]) / 2

        return xdata, ydata

    def _zoom_from_qt_event(self, event, dy):
        if dy == 0:
            return
        scale = 0.88 if dy > 0 else 1.14
        self._zoom_from_qpoint(event.position(), scale)

    def _zoom_from_qpoint(self, qpoint, scale):
        if scale <= 0:
            return

        xdata, ydata = self._qpoint_to_data_pos(qpoint)

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        new_xlim = [xdata + (xlim[0] - xdata) * scale,
                    xdata + (xlim[1] - xdata) * scale]
        new_ylim = [ydata + (ylim[0] - ydata) * scale,
                    ydata + (ylim[1] - ydata) * scale]

        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.draw_idle()

    def _reset_zoom(self):
        if self.raw_image is None:
            return
        ny, nx = self.raw_image.shape
        self.ax.set_xlim(-0.5, nx - 0.5)
        self.ax.set_ylim(ny - 0.5, -0.5)
        self.draw_idle()

    def _pan_from_wheel(self, dx, dy):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        xspan = xlim[1] - xlim[0]
        yspan = ylim[1] - ylim[0]

        shift_x = -dx * xspan * 0.08
        shift_y = dy * yspan * 0.08

        self.ax.set_xlim(xlim[0] + shift_x, xlim[1] + shift_x)
        self.ax.set_ylim(ylim[0] + shift_y, ylim[1] + shift_y)
        self.draw_idle()

    def _on_press(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        self._dragging = True
        self._drag_start = (event.xdata, event.ydata)
        self._xlim_start = self.ax.get_xlim()
        self._ylim_start = self.ax.get_ylim()

    def _on_release(self, event):
        self._dragging = False
        self._drag_start = None

    def _on_motion(self, event):
        if self.coordinate_label is not None:
            if event.inaxes == self.ax and event.xdata is not None and event.ydata is not None:
                x = int(round(event.xdata + 1))
                y = int(round(event.ydata + 1))
                intensity_text = ""

                if self.raw_image is not None:
                    ny, nx = self.raw_image.shape
                    if 1 <= x <= nx and 1 <= y <= ny:
                        value = self.raw_image[y - 1, x - 1]
                        if np.isfinite(value):
                            intensity_text = f" | I = {value:.6g}"
                        else:
                            intensity_text = " | I = NaN"

                dx_text = ""
                dy_text = ""
                r_text = ""

                try:
                    parent = self.parent()
                    while parent is not None and not hasattr(parent, "edit_xc"):
                        parent = parent.parent()

                    if parent is not None:
                        dx = x - parent.edit_xc.value()
                        dy = y - parent.edit_yc.value()
                        r = np.sqrt(dx ** 2 + dy ** 2)
                        dx_text = f" | Δx = {dx:.2f}"
                        dy_text = f" | Δy = {dy:.2f}"
                        r_text = f" | r = {r:.2f} px"
                except Exception:
                    pass

                self.coordinate_label.setText(f"x = {x} | y = {y}{dx_text}{dy_text}{r_text}{intensity_text}")
            else:
                self.coordinate_label.setText("x = - | y = - | Δx = - | Δy = - | r = - | I = -")

        if not self._dragging or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None or self._drag_start is None:
            return

        dx = event.xdata - self._drag_start[0]
        dy = event.ydata - self._drag_start[1]

        self.ax.set_xlim(self._xlim_start[0] - dx, self._xlim_start[1] - dx)
        self.ax.set_ylim(self._ylim_start[0] - dy, self._ylim_start[1] - dy)
        self.draw_idle()

    def show_image(self, img, xc, yc, title, clim_min=None, clim_max=None):
        self.raw_image = img

        old_xlim = self.ax.get_xlim() if self.im is not None else None
        old_ylim = self.ax.get_ylim() if self.im is not None else None

        self.ax.clear()

        img_disp = img.astype(float).copy()
        img_disp[~np.isfinite(img_disp)] = np.nan
        img_disp[img_disp < 0] = np.nan

        with np.errstate(invalid="ignore", divide="ignore"):
            log_img = np.log10(img_disp + 1)

        self.im = self.ax.imshow(log_img, origin="upper", cmap="jet", interpolation="nearest")

        if clim_min is not None and clim_max is not None and clim_min < clim_max:
            self.im.set_clim(clim_min, clim_max)

        self.ax.axvline(xc - 1, color="red", linewidth=1.2)
        self.ax.axhline(yc - 1, color="red", linewidth=1.2)
        if title:
            self.ax.set_title(title)
        self.ax.set_aspect("equal")

        # Important: otherwise each update adds a new colorbar and the image shrinks.
        if self.cbar is not None:
            try:
                self.cbar.remove()
            except Exception:
                pass
            self.cbar = None

        # No colorbar here: it changes the axes position and makes trackpad zoom feel offset.
        self.cbar = None

        if old_xlim is not None and old_ylim is not None:
            self.ax.set_xlim(old_xlim)
            self.ax.set_ylim(old_ylim)

        self.draw_idle()


class PlotCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(5, 3), tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)


# ============================================================
# ======================= FIND CENTRE TAB ====================
# ============================================================

class CentreTab(QWidget):
    def __init__(self, user_email="", default_xc=1294.689, default_yc=1310.290):
        super().__init__()

        self.user_email = user_email
        self.input_file = None
        self.img_raw = None
        self.img = None
        self.xc = default_xc
        self.yc = default_yc
        self.theta_deg = 0.0
        self.beamstop_radius = 0.0
        self.instrument_mode = "XENOCS"
        self.current_header = {}
        self.file_loaded = False

        self._build_ui(default_xc, default_yc)

    def _build_ui(self, default_xc, default_yc):
        main = QGridLayout(self)
        main.setContentsMargins(4, 4, 4, 4)
        main.setSpacing(6)

        image_box = QGroupBox("Scattering pattern")
        control_box = QGroupBox("Centre tools")
        graph_box = QGroupBox("I(q) directions, log-log")

        main.addWidget(image_box, 0, 0)
        main.addWidget(control_box, 0, 1)
        main.addWidget(graph_box, 0, 2)

        main.setColumnStretch(0, 2)
        main.setColumnStretch(1, 0)
        main.setColumnStretch(2, 1)

        image_layout = QVBoxLayout(image_box)
        image_layout.setContentsMargins(6, 18, 6, 6)

        self.canvas_img = ImageCanvas()
        self.coordinate_label = QLabel("x = - | y = - | Δx = - | Δy = - | r = - | I = -")
        self.coordinate_label.setMinimumHeight(28)
        self.coordinate_label.setAlignment(Qt.AlignCenter)
        self.coordinate_label.setStyleSheet("""
            QLabel {
                background-color: #f4f4f4;
                border-radius: 8px;
                padding: 6px;
                font-family: Menlo, Monaco, monospace;
                font-size: 11px;
            }
        """)
        self.canvas_img.set_coordinate_label(self.coordinate_label)

        image_layout.addWidget(self.canvas_img, stretch=1)
        image_layout.addWidget(self.coordinate_label, stretch=0)

        adjust = QHBoxLayout()
        adjust.setSpacing(12)
        image_layout.addLayout(adjust)

        move_box = QGroupBox("Centre position")
        move_layout = QGridLayout(move_box)
        move_layout.setContentsMargins(6, 18, 6, 6)
        move_layout.setSpacing(4)

        rotate_box = QGroupBox("Axes rotation")
        rotate_layout = QGridLayout(rotate_box)
        rotate_layout.setContentsMargins(6, 18, 6, 6)
        rotate_layout.setSpacing(4)

        self.btn_up = QPushButton("↑")
        self.btn_down = QPushButton("↓")
        self.btn_left = QPushButton("←")
        self.btn_right = QPushButton("→")
        self.btn_rot_left = QPushButton("⟲")
        self.btn_rot_right = QPushButton("⟳")

        self.label_theta = QLabel("θ = 0.00°")

        self.step_px = QDoubleSpinBox()
        self.step_px.setRange(0.01, 1e9)
        self.step_px.setDecimals(2)
        self.step_px.setValue(1)

        self.step_deg = QDoubleSpinBox()
        self.step_deg.setRange(0.01, 360)
        self.step_deg.setDecimals(2)
        self.step_deg.setValue(0.2)

        move_layout.addWidget(self.btn_up, 0, 1)
        move_layout.addWidget(self.btn_left, 1, 0)
        move_layout.addWidget(self.btn_down, 1, 1)
        move_layout.addWidget(self.btn_right, 1, 2)
        move_layout.addWidget(QLabel("Step px"), 0, 3)
        move_layout.addWidget(self.step_px, 0, 4)

        rotate_layout.addWidget(self.btn_rot_left, 0, 0)
        rotate_layout.addWidget(self.btn_rot_right, 1, 0)
        rotate_layout.addWidget(self.label_theta, 0, 1, 1, 2)
        rotate_layout.addWidget(QLabel("Step deg"), 1, 1)
        rotate_layout.addWidget(self.step_deg, 1, 2)

        adjust.addWidget(move_box, stretch=1)
        adjust.addStretch(1)
        adjust.addWidget(rotate_box, stretch=1)

        control = QVBoxLayout(control_box)
        control.setContentsMargins(8, 18, 8, 8)
        control.setSpacing(6)

        self.btn_open = QPushButton("Open EDF / H5")
        self.btn_plot = QPushButton("Plot I(q) directions")

        preset_layout = QHBoxLayout()
        preset_layout.setSpacing(4)

        self.btn_xenocs = QPushButton("XENOCS")
        self.btn_id02 = QPushButton("ID02")
        self.btn_id13 = QPushButton("ID13")
        self.btn_custom = QPushButton("Custom")

        for btn in [self.btn_xenocs, self.btn_id02, self.btn_id13, self.btn_custom]:
            btn.setCheckable(True)
            preset_layout.addWidget(btn)

        self.btn_xenocs.setChecked(True)

        self.edit_xc = self._double_spin(0, decimals=3)
        self.edit_yc = self._double_spin(0, decimals=3)
        self.edit_distance = self._double_spin(0, decimals=6, minimum=0)
        self.edit_px_x = self._double_spin(0.075000, decimals=6, minimum=0)
        self.edit_px_y = self._double_spin(0.075000, decimals=6, minimum=0)
        self.edit_lambda = self._double_spin(0, decimals=6, minimum=0)


        self.edit_beamstop_radius = self._double_spin(0, decimals=3, minimum=0)

        self.status = QTextEdit()
        self.status.setReadOnly(True)
        self.status.setText(f"No file loaded\nUser: {self.user_email}")

        control.addWidget(self.btn_open)
        control.addWidget(QLabel("Instrument preset:"))
        control.addLayout(preset_layout)
        self.centre_x_label = QLabel("Centre X:")
        self.centre_y_label = QLabel("Centre Y:")
        self._add_control(control, self.centre_x_label, self.edit_xc)
        self._add_control(control, self.centre_y_label, self.edit_yc)
        self._add_control(control, "Detector distance (m):", self.edit_distance)
        self._add_control(control, "Pixel X (mm):", self.edit_px_x)
        self._add_control(control, "Pixel Y (mm):", self.edit_px_y)
        self._add_control(control, "Wavelength (Å):", self.edit_lambda)
        self._add_control(control, "Beamstop radius (px):", self.edit_beamstop_radius)
        control.addWidget(self.btn_plot)
        control.addWidget(self.status)

        graph_layout = QVBoxLayout(graph_box)
        graph_layout.setContentsMargins(6, 18, 6, 6)
        self.canvas_h = PlotCanvas()
        self.canvas_v = PlotCanvas()
        graph_layout.addWidget(self.canvas_h)
        graph_layout.addWidget(self.canvas_v)

        self.btn_open.clicked.connect(self.open_image_file)
        self.btn_plot.clicked.connect(self.update_plots)

        self.btn_xenocs.clicked.connect(lambda: self.set_instrument_mode("XENOCS"))
        self.btn_id02.clicked.connect(lambda: self.set_instrument_mode("ID02"))
        self.btn_id13.clicked.connect(lambda: self.set_instrument_mode("ID13"))
        self.btn_custom.clicked.connect(lambda: self.set_instrument_mode("Custom"))

        self.update_custom_editing_state()

        self.btn_up.clicked.connect(lambda: self.move_center(0, -self.step_px.value()))
        self.btn_down.clicked.connect(lambda: self.move_center(0, self.step_px.value()))
        self.btn_left.clicked.connect(lambda: self.move_center(-self.step_px.value(), 0))
        self.btn_right.clicked.connect(lambda: self.move_center(self.step_px.value(), 0))

        self.btn_rot_left.clicked.connect(lambda: self.rotate_center(-self.step_deg.value()))
        self.btn_rot_right.clicked.connect(lambda: self.rotate_center(self.step_deg.value()))

        self.edit_xc.valueChanged.connect(self.manual_params_changed)
        self.edit_yc.valueChanged.connect(self.manual_params_changed)
        self.edit_beamstop_radius.valueChanged.connect(self.manual_params_changed)

        self.set_controls_enabled(False)
        self.update_centre_warning_labels()

    def _double_spin(self, value, decimals=3, minimum=-1e9):
        spin = QDoubleSpinBox()
        spin.setRange(minimum, 1e9)
        spin.setDecimals(decimals)
        spin.setValue(value)
        return spin

    def _add_control(self, layout, text, widget):
        if isinstance(text, QLabel):
            layout.addWidget(text)
        else:
            layout.addWidget(QLabel(text))
        layout.addWidget(widget)

    def set_controls_enabled(self, enabled):
        self.btn_plot.setEnabled(enabled)

        self.btn_xenocs.setEnabled(enabled)
        self.btn_id02.setEnabled(enabled)
        self.btn_id13.setEnabled(enabled)
        self.btn_custom.setEnabled(enabled)

        self.btn_up.setEnabled(enabled)
        self.btn_down.setEnabled(enabled)
        self.btn_left.setEnabled(enabled)
        self.btn_right.setEnabled(enabled)
        self.btn_rot_left.setEnabled(enabled)
        self.btn_rot_right.setEnabled(enabled)

        self.step_px.setEnabled(enabled)
        self.step_deg.setEnabled(enabled)

        self.edit_xc.setEnabled(enabled)
        self.edit_yc.setEnabled(enabled)
        self.edit_distance.setEnabled(enabled)
        self.edit_lambda.setEnabled(enabled)
        self.edit_beamstop_radius.setEnabled(enabled)

        if enabled:
            self.update_custom_editing_state()
        else:
            self.edit_px_x.setEnabled(False)
            self.edit_px_y.setEnabled(False)

    def set_instrument_mode(self, mode):
        self.instrument_mode = mode

        buttons = {
            "XENOCS": self.btn_xenocs,
            "ID02": self.btn_id02,
            "ID13": self.btn_id13,
            "Custom": self.btn_custom,
        }

        for key, btn in buttons.items():
            btn.blockSignals(True)
            btn.setChecked(key == mode)
            btn.blockSignals(False)

        self.update_custom_editing_state()
        self.update_centre_warning_labels()
        self.apply_instrument_preset()
        self.refresh_after_preset_change()

    def update_custom_editing_state(self):
        self.edit_px_x.setEnabled(False)
        self.edit_px_y.setEnabled(False)

    def update_centre_warning_labels(self):
        warning = " ⚠️" if self.instrument_mode in ["ID02", "ID13", "Custom"] else ""
        self.centre_x_label.setText(f"Centre X:{warning}")
        self.centre_y_label.setText(f"Centre Y:{warning}")

    def refresh_after_preset_change(self):
        if self.img_raw is None:
            return

        self.xc = self.edit_xc.value()
        self.yc = self.edit_yc.value()
        self.beamstop_radius = self.edit_beamstop_radius.value()

        self.update_masked_image()
        self.show_center_image()
        self.update_plots()

    def apply_instrument_preset(self):
        header = self.current_header or {}

        if self.instrument_mode == "Custom":
            self.xc = self.edit_xc.value()
            self.yc = self.edit_yc.value()
            return

        if self.instrument_mode == "XENOCS":
            self.apply_header_values(header)
            return

        if self.instrument_mode == "ID02":
            self.edit_xc.blockSignals(True)
            self.edit_yc.blockSignals(True)
            self.edit_distance.blockSignals(True)
            self.edit_lambda.blockSignals(True)
            self.edit_xc.setValue(919.689)
            self.edit_yc.setValue(994.290)
            self.edit_distance.setValue(1.0)
            self.edit_lambda.setValue(1.0)
            self.edit_xc.blockSignals(False)
            self.edit_yc.blockSignals(False)
            self.edit_distance.blockSignals(False)
            self.edit_lambda.blockSignals(False)
            self.xc = self.edit_xc.value()
            self.yc = self.edit_yc.value()
            return

        if self.instrument_mode == "ID13":
            self.edit_xc.blockSignals(True)
            self.edit_yc.blockSignals(True)
            self.edit_distance.blockSignals(True)
            self.edit_lambda.blockSignals(True)
            self.edit_xc.setValue(1294.689)
            self.edit_yc.setValue(1310.290)
            self.edit_distance.setValue(0.8)
            self.edit_lambda.setValue(0.826563)
            self.edit_xc.blockSignals(False)
            self.edit_yc.blockSignals(False)
            self.edit_distance.blockSignals(False)
            self.edit_lambda.blockSignals(False)
            self.xc = self.edit_xc.value()
            self.yc = self.edit_yc.value()
            return

    def apply_header_values(self, header):
        def get_header_float(*names):
            for name in names:
                if name in header:
                    try:
                        return float(header[name])
                    except (TypeError, ValueError):
                        return None
            return None

        center_1 = get_header_float("Center_1", "center_1")
        center_2 = get_header_float("Center_2", "center_2")
        sample_distance = get_header_float("SampleDistance", "sample_distance")
        pixel_x_m = get_header_float("PSize_1", "PSize_X", "PixelSizeX")
        pixel_y_m = get_header_float("PSize_2", "PSize_Y", "PixelSizeY")
        wavelength_m = get_header_float("WaveLength", "Wavelength", "wavelength")

        self.xc = center_1 if center_1 is not None else 0
        self.yc = center_2 if center_2 is not None else 0

        self.edit_xc.blockSignals(True)
        self.edit_yc.blockSignals(True)
        self.edit_distance.blockSignals(True)
        self.edit_px_x.blockSignals(True)
        self.edit_px_y.blockSignals(True)
        self.edit_lambda.blockSignals(True)

        self.edit_xc.setValue(self.xc)
        self.edit_yc.setValue(self.yc)
        self.edit_distance.setValue(sample_distance if sample_distance is not None else 0)
        self.edit_px_x.setValue(pixel_x_m * 1000 if pixel_x_m is not None else 0.075000)
        self.edit_px_y.setValue(pixel_y_m * 1000 if pixel_y_m is not None else 0.075000)
        self.edit_lambda.setValue(wavelength_m * 1e10 if wavelength_m is not None else 0)

        self.edit_xc.blockSignals(False)
        self.edit_yc.blockSignals(False)
        self.edit_distance.blockSignals(False)
        self.edit_px_x.blockSignals(False)
        self.edit_px_y.blockSignals(False)
        self.edit_lambda.blockSignals(False)

    def open_image_file(self):
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Open EDF or H5 file",
            "",
            "Image data (*.edf *.h5 *.hdf5);;EDF (*.edf);;HDF5 (*.h5 *.hdf5);;All files (*)"
        )
        if not file:
            return

        try:
            path = Path(file)
            suffix = path.suffix.lower()

            if suffix == ".edf":
                img, header, *_ = read_edf_file(file)
                file_type = "EDF"
            elif suffix in [".h5", ".hdf5"]:
                img, header = read_h5_first_image(file)
                file_type = "H5"
            else:
                raise ValueError("Unsupported file format. Please select an EDF, H5 or HDF5 file.")

            self.input_file = path
            self.current_header = header
            self.file_loaded = True
            self.set_controls_enabled(True)

            self.img_raw = img.astype(float)
            self.img_raw[self.img_raw > 4e9] = np.nan
            self.img_raw[self.img_raw < 0] = np.nan

            self.apply_instrument_preset()

            self.theta_deg = 0
            self.beamstop_radius = self.edit_beamstop_radius.value()
            self.label_theta.setText(f"θ = {self.theta_deg:.2f}°")

            self.update_masked_image()
            self.show_center_image()
            self.update_plots()

            header_lines = [
                f"Loaded file: {self.input_file.name}",
                "",
                "Header values found:",
            ]

            for key in ["PSize_1", "PSize_2", "Center_1", "Center_2", "SampleDistance", "WaveLength", "Wavelength"]:
                if key in header:
                    header_lines.append(f"{key}: {header[key]}")

            if len(header_lines) == 3:
                header_lines.append("No relevant header value found.")

            self.status.setText("\n".join(header_lines))

        except Exception as e:
            QMessageBox.critical(self, "File reading error", str(e))

    def update_masked_image(self):
        if self.img_raw is None:
            return

        self.img = self.img_raw.copy()
        r = self.beamstop_radius

        if r > 0:
            ny, nx = self.img.shape
            y, x = np.ogrid[1:ny + 1, 1:nx + 1]
            mask = (x - self.xc) ** 2 + (y - self.yc) ** 2 <= r ** 2
            self.img[mask] = np.nan

    def show_center_image(self):
        if self.img is None:
            return

        self.canvas_img.show_image(self.img, self.xc, self.yc, "", None, None)
        ax = self.canvas_img.ax

        theta = np.deg2rad(self.theta_deg)
        ux, uy = np.cos(theta), np.sin(theta)
        vx, vy = -np.sin(theta), np.cos(theta)

        ny, nx = self.img.shape
        length = max(nx, ny)

        ax.plot(
            [self.xc - 1 - length * ux, self.xc - 1 + length * ux],
            [self.yc - 1 - length * uy, self.yc - 1 + length * uy],
            color="red", linewidth=1.2
        )
        ax.plot(
            [self.xc - 1 - length * vx, self.xc - 1 + length * vx],
            [self.yc - 1 - length * vy, self.yc - 1 + length * vy],
            color="blue", linewidth=1.2
        )

        if self.beamstop_radius > 0:
            ang = np.linspace(0, 2 * np.pi, 300)
            ax.plot(
                self.xc - 1 + self.beamstop_radius * np.cos(ang),
                self.yc - 1 + self.beamstop_radius * np.sin(ang),
                color="white", linewidth=1.2
            )

        ax.plot(self.xc - 1, self.yc - 1, "wo", markersize=5)
        self.canvas_img.draw_idle()

    def manual_params_changed(self):
        if self.img_raw is None:
            return

        self.xc = self.edit_xc.value()
        self.yc = self.edit_yc.value()
        self.beamstop_radius = self.edit_beamstop_radius.value()

        self.update_masked_image()
        self.show_center_image()
        self.update_plots()

    def move_center(self, dx, dy):
        if self.img_raw is None:
            return

        self.xc += dx
        self.yc += dy
        self.edit_xc.blockSignals(True)
        self.edit_yc.blockSignals(True)
        self.edit_xc.setValue(self.xc)
        self.edit_yc.setValue(self.yc)
        self.edit_xc.blockSignals(False)
        self.edit_yc.blockSignals(False)

        self.update_masked_image()
        self.show_center_image()
        self.update_plots()

    def rotate_center(self, dtheta):
        if self.img_raw is None:
            return

        self.theta_deg += dtheta
        self.label_theta.setText(f"θ = {self.theta_deg:.2f}°")
        self.show_center_image()
        self.update_plots()

    def update_plots(self):
        if self.img is None:
            return

        self.xc = self.edit_xc.value()
        self.yc = self.edit_yc.value()
        self.beamstop_radius = self.edit_beamstop_radius.value()

        d_m = self.edit_distance.value()
        px_x = self.edit_px_x.value()
        px_y = self.edit_px_y.value()
        lam = self.edit_lambda.value()
        half_width = 0

        ang_h = self.theta_deg
        ang_h2 = self.theta_deg + 180
        ang_v = self.theta_deg + 90
        ang_v2 = self.theta_deg + 270

        q1, i1 = compute_directional_iq_nm_general(self.img, self.xc, self.yc, ang_h, half_width, d_m, px_x, px_y, lam)
        q2, i2 = compute_directional_iq_nm_general(self.img, self.xc, self.yc, ang_h2, half_width, d_m, px_x, px_y, lam)
        q3, i3 = compute_directional_iq_nm_general(self.img, self.xc, self.yc, ang_v, half_width, d_m, px_x, px_y, lam)
        q4, i4 = compute_directional_iq_nm_general(self.img, self.xc, self.yc, ang_v2, half_width, d_m, px_x, px_y, lam)

        self.canvas_h.ax.clear()
        self.canvas_v.ax.clear()

        valid = np.isfinite(q1) & np.isfinite(i1) & (q1 > 0) & (i1 > 0)
        self.canvas_h.ax.loglog(q1[valid], i1[valid], "-", color="red", linewidth=1.3, label="H +")

        valid = np.isfinite(q2) & np.isfinite(i2) & (q2 > 0) & (i2 > 0)
        self.canvas_h.ax.loglog(q2[valid], i2[valid], "--", color="darkred", linewidth=1.1, label="H -")

        valid = np.isfinite(q3) & np.isfinite(i3) & (q3 > 0) & (i3 > 0)
        self.canvas_v.ax.loglog(q3[valid], i3[valid], "-", color="blue", linewidth=1.3, label="V +")

        valid = np.isfinite(q4) & np.isfinite(i4) & (q4 > 0) & (i4 > 0)
        self.canvas_v.ax.loglog(q4[valid], i4[valid], "--", color="navy", linewidth=1.1, label="V -")

        for ax, title in [(self.canvas_h.ax, "Horizontal I(q)"), (self.canvas_v.ax, "Vertical I(q)")]:
            ax.set_xlabel("q (nm$^{-1}$)")
            ax.set_ylabel("I(q)")
            ax.set_title(title)
            ax.grid(True, which="both")
            ax.legend(loc="best")

        self.canvas_h.draw_idle()
        self.canvas_v.draw_idle()

    # Replace any remaining occurrence of "Tous les fichiers (*)" with "All files (*)"
