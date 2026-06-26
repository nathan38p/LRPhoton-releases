import fnmatch
from pathlib import Path
import re
import h5py

import numpy as np
from PySide6.QtCore import Qt, QEvent, Signal, QTimer
from PySide6.QtWidgets import (
    QWidget, QFileDialog, QMessageBox, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QSpinBox, QTextEdit, QGroupBox, QSlider, QSizePolicy,
    QLineEdit, QListWidget, QListWidgetItem, QAbstractItemView, QScrollArea, QSplitter,
    QCheckBox,
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
    FlexibleDoubleSpinBox as QDoubleSpinBox,
    FRAME_BUTTON_WIDTH,
    FRAME_COUNTER_WIDTH,
    FRAME_NAV_SPACING,
    FRAME_SPIN_WIDTH,
    GROUP_BOX_MARGINS,
    GROUP_BOX_STYLE,
    PAGE_MARGINS,
    PANEL_MARGINS,
    clear_plot_canvas,
    constrain_image_axes,
    install_selectable_legend,
    normalize_decimal_text,
    style_q_geometry_buttons,
)
from .file_ratings import install_file_rating_menu, is_file_rated_up, set_item_file_path, should_hide_file_in_browser
from .line_geometry import LineGeometrySelector, line_geometry_to_lrphoton


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


# Helper to infer EDF header size from a string containing the start of the file
def infer_edf_header_size(first_chunk: str) -> int:
    match = re.search(r"EDF_HeaderSize\s*[:=]\s*(\d+)", first_chunk)
    if match:
        return int(match.group(1))

    closing = first_chunk.find("}")
    if closing < 0:
        raise ValueError("EDF header size not found and closing brace not found.")

    header_end = closing + 1
    for boundary in (1024, 512, 256):
        padded_size = int(np.ceil(header_end / boundary) * boundary)
        if padded_size <= len(first_chunk):
            return padded_size

    return header_end


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

    header_size = infer_edf_header_size(first)

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


ID02_DEFAULT_CENTER_X = 914.4
ID02_DEFAULT_CENTER_Y = 996.5
ID02_DEFAULT_DISTANCE_M = 10.0002
ID02_DEFAULT_PIXEL_MM = 0.075
ID02_DEFAULT_WAVELENGTH_A = 1.01402
CENTER_X_KEYS = ("Center_1", "center_1", "CenterX", "center_x", "BeamCenterX", "Beam_x", "beam_x")
CENTER_Y_KEYS = ("Center_2", "center_2", "CenterY", "center_y", "BeamCenterY", "Beam_y", "beam_y")


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

        add_matching_edf_center(header, filename)

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
        self.fig = Figure(figsize=(4.2, 4.2), tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)

        self.raw_image = None
        self.im = None
        self.cbar = None
        self.coordinate_label = None
        self.reset_view_on_next_show = False

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

    def clear_image(self):
        self.raw_image = None
        self.im = None
        if self.cbar is not None:
            try:
                self.cbar.remove()
            except Exception:
                pass
            self.cbar = None
        self.fig.patch.set_facecolor("white")
        self.ax.clear()
        self.ax.set_facecolor("white")
        self.ax.set_axis_off()
        if self.coordinate_label is not None:
            self.coordinate_label.setText("x = - | y = - | r = - | I = -")
        self.draw_idle()

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
        constrain_image_axes(self.ax, self.raw_image.shape)
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
        constrain_image_axes(self.ax, self.raw_image.shape)
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

                r_text = ""

                try:
                    parent = self.parent()
                    while parent is not None and not hasattr(parent, "edit_xc"):
                        parent = parent.parent()

                    if parent is not None:
                        dx = x - parent.edit_xc.value()
                        dy = y - parent.edit_yc.value()
                        r = np.sqrt(dx ** 2 + dy ** 2)
                        r_text = f" | r = {r:.2f} px"
                except Exception:
                    pass

                self.coordinate_label.setText(f"x = {x} | y = {y}{r_text}{intensity_text}")
            else:
                self.coordinate_label.setText("x = - | y = - | r = - | I = -")

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

    def show_image(self, img, xc, yc, title, clim_min=None, clim_max=None):
        self.raw_image = img

        old_xlim = self.ax.get_xlim() if self.im is not None and not self.reset_view_on_next_show else None
        old_ylim = self.ax.get_ylim() if self.im is not None and not self.reset_view_on_next_show else None

        self.ax.clear()

        img_disp = img.astype(float).copy()
        img_disp[~np.isfinite(img_disp)] = np.nan
        img_disp[img_disp < 0] = np.nan

        with np.errstate(invalid="ignore", divide="ignore"):
            log_img = np.log10(img_disp + 1)

        self.im = self.ax.imshow(log_img, origin="upper", cmap="jet", interpolation="nearest")

        if clim_min is not None and clim_max is not None and clim_min < clim_max:
            self.im.set_clim(clim_min, clim_max)

        if title:
            self.ax.set_title(title)
        self.ax.set_aspect("equal")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_xlabel("")
        self.ax.set_ylabel("")

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
            constrain_image_axes(self.ax, self.raw_image.shape)

        self.reset_view_on_next_show = False
        self.draw_idle()


class PlotCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(4.2, 2.6), tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setFocusPolicy(Qt.StrongFocus)

        try:
            self.grabGesture(Qt.PinchGesture)
        except Exception:
            pass

    def event(self, event):
        try:
            if event.type() == QEvent.NativeGesture:
                gesture_type = event.gestureType()
                value = event.value()
                if gesture_type == Qt.ZoomNativeGesture and value != 0:
                    scale = 1.0 / (1.0 + value) if value > -0.95 else 1.25
                    self.zoom_at_qpoint(self._event_center_point(event), scale)
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
                        self.zoom_at_qpoint(self._event_center_point(event), 1.0 / factor)
                        event.accept()
                        return True
        except Exception:
            pass

        return super().event(event)

    def wheelEvent(self, event):
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
                self.zoom_at_qpoint(event.position(), scale)
        else:
            self.pan_by_trackpad(dx, dy)

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

    def _scaled_limits(self, limits, center, scale, is_log):
        left, right = limits
        if is_log and left > 0 and right > 0 and center > 0:
            left_l, right_l, center_l = np.log10([left, right, center])
            return (
                10 ** (center_l + (left_l - center_l) * scale),
                10 ** (center_l + (right_l - center_l) * scale),
            )

        return (
            center + (left - center) * scale,
            center + (right - center) * scale,
        )

    def zoom_at_qpoint(self, qpoint, scale):
        if scale <= 0:
            return

        xdata, ydata = self._qpoint_to_data_pos(qpoint)
        self.ax.set_xlim(
            self._scaled_limits(self.ax.get_xlim(), xdata, scale, self.ax.get_xscale() == "log")
        )
        self.ax.set_ylim(
            self._scaled_limits(self.ax.get_ylim(), ydata, scale, self.ax.get_yscale() == "log")
        )
        self.draw_idle()

    def _panned_limits(self, limits, delta, is_log):
        left, right = limits
        if is_log and left > 0 and right > 0:
            left_l, right_l = np.log10([left, right])
            span = right_l - left_l
            shift = -delta * span * 0.08
            return 10 ** (left_l + shift), 10 ** (right_l + shift)

        span = right - left
        shift = -delta * span * 0.08
        return left + shift, right + shift

    def pan_by_trackpad(self, dx, dy):
        self.ax.set_xlim(
            self._panned_limits(self.ax.get_xlim(), dx, self.ax.get_xscale() == "log")
        )
        self.ax.set_ylim(
            self._panned_limits(self.ax.get_ylim(), -dy, self.ax.get_yscale() == "log")
        )
        self.draw_idle()

    def reset_view(self):
        self.ax.relim()
        self.ax.autoscale_view()
        self.draw_idle()


# ============================================================
# ======================= FIND CENTRE TAB ====================
# ============================================================

class CentreTab(QWidget):
    folder_changed = Signal(Path)

    def __init__(self, user_email="", default_xc=ID13_DEFAULT_CENTER_X, default_yc=ID13_DEFAULT_CENTER_Y):
        super().__init__()

        self.user_email = user_email
        self.input_file = None
        self.current_folder = Path.home()
        self.img_raw = None
        self.img = None
        self.xc = default_xc
        self.yc = default_yc
        self.theta_deg = 0.0
        self.beamstop_radius = 0.0
        self.instrument_mode = "XENOCS"
        self.current_header = {}
        self.file_loaded = False
        self.q_axis_unit = "nm"
        self.display_vmin = None
        self.display_vmax = None
        self.display_data_min = 0.0
        self.display_data_max = 1.0
        self._syncing_folder = False

        self._build_ui(default_xc, default_yc)

    def _build_ui(self, default_xc, default_yc):
        main = QGridLayout(self)
        main.setContentsMargins(*PAGE_MARGINS)
        main.setSpacing(BLOCK_SPACING)

        image_box = QGroupBox("Scattering pattern")
        graph_box = QGroupBox("I(q) directions")

        self.image_box = image_box
        self.graph_box = graph_box

        image_box.setMinimumWidth(0)
        image_box.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        graph_box.setMinimumWidth(0)
        graph_box.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)

        center_panel = QWidget()
        center_panel.setFixedWidth(FILE_BROWSER_WIDTH)
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(*PANEL_MARGINS)
        center_layout.setSpacing(BLOCK_SPACING)

        center_splitter = QSplitter(Qt.Vertical)
        self.center_splitter = center_splitter
        center_splitter.setChildrenCollapsible(True)
        center_splitter.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        center_layout.addWidget(center_splitter, stretch=1)

        main.addWidget(image_box, 0, 0)
        main.addWidget(center_panel, 0, 1, alignment=Qt.AlignHCenter)
        main.addWidget(graph_box, 0, 2)

        self.main_grid = main

        main.setColumnMinimumWidth(0, 0)
        main.setColumnMinimumWidth(1, FILE_BROWSER_WIDTH)
        main.setColumnMinimumWidth(2, 0)

        main.setColumnStretch(0, 1)
        main.setColumnStretch(1, 0)
        main.setColumnStretch(2, 1)

        image_layout = QVBoxLayout(image_box)
        image_layout.setContentsMargins(*GROUP_BOX_MARGINS)

        self.canvas_img = ImageCanvas()
        self.canvas_img.setMinimumWidth(0)
        self.canvas_img.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas_img.clear_image()
        self.coordinate_label = QLabel("x = - | y = - | Δx = - | Δy = - | r = - | I = -")
        self.coordinate_label.setMinimumWidth(0)
        self.coordinate_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
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
        adjust.setContentsMargins(0, 0, 0, 0)
        adjust.setSpacing(BLOCK_SPACING)
        image_layout.addLayout(adjust)

        move_box = QGroupBox("Center position")
        move_layout = QGridLayout(move_box)
        move_layout.setContentsMargins(*GROUP_BOX_MARGINS)
        move_layout.setHorizontalSpacing(8)
        move_layout.setVerticalSpacing(10)

        rotate_box = QGroupBox("Axes rotation")
        rotate_layout = QGridLayout(rotate_box)
        rotate_layout.setContentsMargins(*GROUP_BOX_MARGINS)
        rotate_layout.setHorizontalSpacing(10)
        rotate_layout.setVerticalSpacing(10)

        self.btn_up = QPushButton("↑")
        self.btn_down = QPushButton("↓")
        self.btn_left = QPushButton("←")
        self.btn_right = QPushButton("→")
        self.btn_rot_left = QPushButton("⟲")
        self.btn_rot_right = QPushButton("⟳")

        for button in [self.btn_up, self.btn_down, self.btn_left, self.btn_right]:
            button.setFixedSize(44, 32)

        for button in [self.btn_rot_left, self.btn_rot_right]:
            button.setFixedSize(54, 32)

        self.label_theta = QLabel("θ = 0.00°")

        self.step_px = QDoubleSpinBox()
        self.step_px.setRange(0.01, 1e9)
        self.step_px.setDecimals(2)
        self.step_px.setValue(1)
        self.step_px.setFixedWidth(96)

        self.step_deg = QDoubleSpinBox()
        self.step_deg.setRange(0.01, 360)
        self.step_deg.setDecimals(2)
        self.step_deg.setValue(0.2)
        self.step_deg.setFixedWidth(96)

        move_layout.addWidget(self.btn_up, 0, 1, alignment=Qt.AlignCenter)
        move_layout.addWidget(self.btn_left, 1, 0, alignment=Qt.AlignCenter)
        move_layout.addWidget(self.btn_down, 1, 1, alignment=Qt.AlignCenter)
        move_layout.addWidget(self.btn_right, 1, 2, alignment=Qt.AlignCenter)
        step_px_label = QLabel("Step px")
        step_px_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        move_layout.addWidget(step_px_label, 2, 0, 1, 1)
        move_layout.addWidget(self.step_px, 2, 1, 1, 2, alignment=Qt.AlignLeft)
        move_layout.setColumnStretch(0, 1)
        move_layout.setColumnStretch(1, 1)
        move_layout.setColumnStretch(2, 1)

        rotate_layout.addWidget(self.btn_rot_left, 0, 0, alignment=Qt.AlignCenter)
        self.label_theta.setMinimumWidth(100)
        self.label_theta.setAlignment(Qt.AlignCenter)
        rotate_layout.addWidget(self.label_theta, 0, 1, alignment=Qt.AlignCenter)
        rotate_layout.addWidget(self.btn_rot_right, 0, 2, alignment=Qt.AlignCenter)
        step_deg_label = QLabel("Step deg")
        step_deg_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        rotate_layout.addWidget(step_deg_label, 1, 0, 1, 1)
        rotate_layout.addWidget(self.step_deg, 1, 1, 1, 2, alignment=Qt.AlignLeft)
        rotate_layout.setColumnStretch(0, 1)
        rotate_layout.setColumnStretch(1, 1)
        rotate_layout.setColumnStretch(2, 1)

        adjust.addWidget(move_box, stretch=1)
        adjust.addWidget(rotate_box, stretch=1)

        file_box = QGroupBox("File browser")
        file_box.setStyleSheet(GROUP_BOX_STYLE)
        file_box.setMinimumHeight(0)
        file_box.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        file_layout = QVBoxLayout(file_box)
        file_layout.setContentsMargins(*GROUP_BOX_MARGINS)
        file_layout.setSpacing(6)

        self.folder_path = QLineEdit(str(self.current_folder))
        self.folder_path.returnPressed.connect(self.refresh_files)
        file_layout.addWidget(self.folder_path)

        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.choose_folder)
        file_layout.addWidget(browse_button)

        filters_layout = QGridLayout()
        self.name_filter = QLineEdit("*")
        self.extension_filter = QLineEdit("*.edf *.h5 *.hdf5")
        self.name_filter.textChanged.connect(self.refresh_files)
        self.extension_filter.textChanged.connect(self.refresh_files)
        filters_layout.addWidget(QLabel("Name:"), 0, 0)
        filters_layout.addWidget(self.name_filter, 0, 1)
        filters_layout.addWidget(QLabel("Extensions:"), 1, 0)
        filters_layout.addWidget(self.extension_filter, 1, 1)
        file_layout.addLayout(filters_layout)

        file_options_layout = QHBoxLayout()
        file_options_layout.setContentsMargins(0, 0, 0, 0)
        file_options_layout.setSpacing(10)
        self.show_subfolders_checkbox = QCheckBox("Show subfolders")
        self.show_subfolders_checkbox.setChecked(False)
        self.show_subfolders_checkbox.stateChanged.connect(self.refresh_files)
        self.only_thumbs_up_checkbox = QCheckBox("Only 👍")
        self.only_thumbs_up_checkbox.setChecked(False)
        self.only_thumbs_up_checkbox.stateChanged.connect(self.refresh_files)
        file_options_layout.addWidget(self.show_subfolders_checkbox)
        file_options_layout.addWidget(self.only_thumbs_up_checkbox)
        file_options_layout.addStretch(1)
        file_layout.addLayout(file_options_layout)

        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh_files)
        file_layout.addWidget(refresh_button)

        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.SingleSelection)
        install_file_rating_menu(self.file_list)
        self.file_list.itemClicked.connect(self.open_selected_file)
        file_layout.addWidget(self.file_list, stretch=1)

        tools_box = QGroupBox("Center tools")
        self.control_box = tools_box
        tools_box.setStyleSheet(GROUP_BOX_STYLE)
        tools_box.setMinimumHeight(0)
        tools_box.setMinimumWidth(0)
        tools_box.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

        tools_box_layout = QVBoxLayout(tools_box)
        tools_box_layout.setContentsMargins(6, 18, 6, 6)
        tools_box_layout.setSpacing(0)
        tools_content = QWidget()
        tools_content.setStyleSheet("background-color: #eeeeee;")
        control = QVBoxLayout(tools_content)
        control.setContentsMargins(0, 0, 0, 0)
        control.setSpacing(4)

        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QScrollArea.NoFrame)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        controls_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        controls_scroll.setMinimumHeight(0)
        controls_scroll.setMinimumWidth(0)
        controls_scroll.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        controls_scroll.setStyleSheet("""
            QScrollArea {
                background-color: #eeeeee;
                border: 0px;
            }
            QScrollArea > QWidget > QWidget {
                background-color: #eeeeee;
            }
        """)
        controls_scroll.viewport().setStyleSheet("background-color: #eeeeee;")
        controls_scroll.setWidget(tools_content)
        tools_box_layout.addWidget(controls_scroll)

        preset_layout = QHBoxLayout()
        preset_layout.setSpacing(4)

        self.btn_xenocs = QPushButton("XENOCS")
        self.btn_id02 = QPushButton("ID02")
        self.btn_id13 = QPushButton("ID13")
        self.btn_custom = QPushButton("Custom")
        self.q_manual_button = QPushButton("+")
        self.q_manual_button.clicked.connect(lambda: self.set_instrument_mode("Custom"))

        for btn in [self.btn_xenocs, self.btn_id02, self.btn_id13, self.btn_custom]:
            btn.setCheckable(True)
            preset_layout.addWidget(btn)
        preset_layout.addWidget(self.q_manual_button)
        for btn in [self.btn_xenocs, self.btn_id02, self.btn_id13, self.btn_custom, self.q_manual_button]:
            btn.hide()
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

        self.edit_xc = self._double_spin(0, decimals=13)
        self.edit_yc = self._double_spin(0, decimals=13)
        self.edit_distance = self._double_spin(0, decimals=16, minimum=0)
        self.edit_px_x = self._double_spin(0.075000, decimals=6, minimum=0)
        self.edit_px_y = self._double_spin(0.075000, decimals=6, minimum=0)
        self.edit_lambda = self._double_spin(0, decimals=16, minimum=0)


        self.edit_beamstop_radius = self._double_spin(0, decimals=3, minimum=0)

        self.status = QTextEdit()
        self.status.setReadOnly(True)
        self.status.setText("")

        control.addLayout(preset_layout)

        contrast_box = QGroupBox("Contrast")
        contrast_box.setStyleSheet(GROUP_BOX_STYLE)
        contrast_layout = QGridLayout(contrast_box)
        contrast_layout.setContentsMargins(*GROUP_BOX_MARGINS)
        contrast_layout.setSpacing(4)
        self.vmin_label = QLabel("Min: -")
        self.vmax_label = QLabel("Max: -")
        self.vmin_slider = QSlider(Qt.Horizontal)
        self.vmax_slider = QSlider(Qt.Horizontal)
        self.vmin_slider.setRange(0, 1000)
        self.vmax_slider.setRange(0, 1000)
        self.vmin_slider.setValue(0)
        self.vmax_slider.setValue(1000)
        self.auto_contrast_button = QPushButton("Auto")
        self.auto_contrast_button.setFixedWidth(54)
        self.auto_contrast_button.clicked.connect(self.auto_current_image_contrast)
        self.lock_contrast_checkbox = QCheckBox("Lock min/max")
        self.lock_contrast_checkbox.setChecked(False)
        contrast_layout.addWidget(self.vmin_label, 0, 0)
        contrast_layout.addWidget(self.vmin_slider, 0, 1)
        contrast_layout.addWidget(self.auto_contrast_button, 0, 2, 2, 1)
        contrast_layout.addWidget(self.vmax_label, 1, 0)
        contrast_layout.addWidget(self.vmax_slider, 1, 1)
        contrast_layout.addWidget(self.lock_contrast_checkbox, 2, 0, 1, 3)
        control.addWidget(contrast_box)

        self.centre_x_label = QLabel("Center X:")
        self.centre_y_label = QLabel("Center Y:")
        self._add_control(control, self.centre_x_label, self.edit_xc)
        self._add_control(control, self.centre_y_label, self.edit_yc)
        self._add_control(control, "Detector distance (m):", self.edit_distance)
        self._add_control(control, "Pixel X (mm):", self.edit_px_x)
        self._add_control(control, "Pixel Y (mm):", self.edit_px_y)
        self._add_control(control, "Wavelength (Å):", self.edit_lambda)
        self._add_control(control, "Beamstop radius (px):", self.edit_beamstop_radius)
        self.status.hide()
        control.addStretch(1)

        center_splitter.addWidget(file_box)
        center_splitter.addWidget(tools_box)
        center_splitter.setStretchFactor(0, 1)
        center_splitter.setStretchFactor(1, 1)
        self.set_initial_center_splitter_sizes()
        QTimer.singleShot(0, self.set_initial_center_splitter_sizes)
        QTimer.singleShot(100, self.set_initial_center_splitter_sizes)

        graph_layout = QVBoxLayout(graph_box)
        graph_layout.setContentsMargins(*GROUP_BOX_MARGINS)
        self.canvas_h = PlotCanvas()
        self.canvas_v = PlotCanvas()
        clear_plot_canvas(self.canvas_h)
        clear_plot_canvas(self.canvas_v)
        self.canvas_h.setMinimumWidth(0)
        self.canvas_v.setMinimumWidth(0)
        self.canvas_h.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas_v.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas_h.mpl_connect("button_press_event", self.on_graph_button_press)
        self.canvas_v.mpl_connect("button_press_event", self.on_graph_button_press)
        graph_layout.addWidget(self.canvas_h)
        graph_layout.addWidget(self.canvas_v)

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
        main.addLayout(frame_nav, 1, 0, 1, 3)

        for widget in [
            self.frame_start_spin, self.frame_end_spin, self.prev_frame_button,
            self.next_frame_button, self.frame_slider,
        ]:
            widget.setEnabled(False)

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
        self.edit_distance.valueChanged.connect(self.manual_params_changed)
        self.edit_px_x.valueChanged.connect(self.manual_params_changed)
        self.edit_px_y.valueChanged.connect(self.manual_params_changed)
        self.edit_lambda.valueChanged.connect(self.manual_params_changed)
        self.edit_beamstop_radius.valueChanged.connect(self.manual_params_changed)
        self.vmin_slider.valueChanged.connect(self.update_image_contrast_from_sliders)
        self.vmax_slider.valueChanged.connect(self.update_image_contrast_from_sliders)

        self.set_controls_enabled(False)
        self.update_centre_warning_labels()
        self.update_side_block_widths()

    def _double_spin(self, value, decimals=3, minimum=-1e9):
        spin = QDoubleSpinBox()
        spin.setRange(minimum, 1e9)
        spin.setDecimals(decimals)
        spin.setValue(value)
        return spin

    def _add_control(self, layout, text, widget):
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)

        if isinstance(text, QLabel):
            label = text
        else:
            label = QLabel(text)

        label.setMinimumWidth(142)
        label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        widget.setMinimumWidth(0)
        row_layout.addWidget(label, stretch=0)
        row_layout.addWidget(widget, stretch=1)
        layout.addWidget(row)

    def set_controls_enabled(self, enabled):
        for widget in [
            self.btn_xenocs,
            self.btn_id02,
            self.btn_id13,
            self.btn_custom,
            self.q_manual_button,
        ]:
            widget.setEnabled(True)

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
        self.vmin_slider.setEnabled(enabled)
        self.vmax_slider.setEnabled(enabled)
        self.auto_contrast_button.setEnabled(enabled)
        self.lock_contrast_checkbox.setEnabled(enabled)

        if enabled:
            self.update_custom_editing_state()
        else:
            self.edit_px_x.setEnabled(False)
            self.edit_px_y.setEnabled(False)

    def set_initial_center_splitter_sizes(self):
        if not hasattr(self, "center_splitter"):
            return
        height = max(2, self.center_splitter.height())
        self.center_splitter.setSizes([height // 2, height - height // 2])

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

        self.update_custom_editing_state()
        self.update_centre_warning_labels()
        self.apply_instrument_preset()
        self.refresh_after_preset_change()

    def apply_line_geometry_selection(self, name, geometry):
        values = line_geometry_to_lrphoton(geometry)
        self.edit_xc.setValue(values["xc"])
        self.edit_yc.setValue(values["yc"])
        self.edit_distance.setValue(values["distance_m"])
        self.edit_px_x.setValue(values["pixel_x_mm"])
        self.edit_px_y.setValue(values["pixel_y_mm"])
        if hasattr(self, "edit_lambda"):
            self.edit_lambda.setValue(values["wavelength_a"])
        self.instrument_mode = "Custom" if name not in {"XENOCS", "ID02", "ID13"} else name
        buttons = {
            "XENOCS": self.btn_xenocs,
            "ID02": self.btn_id02,
            "ID13": self.btn_id13,
            "Custom": self.btn_custom,
        }
        style_q_geometry_buttons(buttons, self.instrument_mode, self.q_manual_button)
        self.refresh_after_preset_change()

    def update_custom_editing_state(self):
        self.edit_px_x.setEnabled(False)
        self.edit_px_y.setEnabled(False)

    def update_centre_warning_labels(self):
        self.centre_x_label.setText("Center X:")
        self.centre_y_label.setText("Center Y:")

    def refresh_after_preset_change(self):
        if self.img_raw is None:
            self.canvas_img.clear_image()
            clear_plot_canvas(self.canvas_h)
            clear_plot_canvas(self.canvas_v)
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
            center_1 = self.get_header_float(header, *CENTER_X_KEYS)
            center_2 = self.get_header_float(header, *CENTER_Y_KEYS)
            sample_distance = self.get_header_float(header, "SampleDistance", "sampledistance", "sample_distance", "Distance", "DetectorDistance")
            pixel_x_m = self.get_header_float(header, "PSize_1", "psize_1", "PSize_X", "PixelSizeX", "pixel_size_x", "x_pixel_size")
            pixel_y_m = self.get_header_float(header, "PSize_2", "psize_2", "PSize_Y", "PixelSizeY", "pixel_size_y", "y_pixel_size")
            wavelength_m = self.get_header_float(header, "WaveLength", "Wavelength", "wavelength", "Lambda", "lambda")
            self.edit_xc.blockSignals(True)
            self.edit_yc.blockSignals(True)
            self.edit_distance.blockSignals(True)
            self.edit_px_x.blockSignals(True)
            self.edit_px_y.blockSignals(True)
            self.edit_lambda.blockSignals(True)
            self.edit_xc.setValue(center_1 if center_1 is not None else ID02_DEFAULT_CENTER_X)
            self.edit_yc.setValue(center_2 if center_2 is not None else ID02_DEFAULT_CENTER_Y)
            self.edit_distance.setValue(sample_distance if sample_distance is not None else ID02_DEFAULT_DISTANCE_M)
            self.edit_px_x.setValue(pixel_x_m * 1000 if pixel_x_m is not None else ID02_DEFAULT_PIXEL_MM)
            self.edit_px_y.setValue(pixel_y_m * 1000 if pixel_y_m is not None else ID02_DEFAULT_PIXEL_MM)
            self.edit_lambda.setValue(wavelength_m * 1e10 if wavelength_m is not None else ID02_DEFAULT_WAVELENGTH_A)
            self.edit_xc.blockSignals(False)
            self.edit_yc.blockSignals(False)
            self.edit_distance.blockSignals(False)
            self.edit_px_x.blockSignals(False)
            self.edit_px_y.blockSignals(False)
            self.edit_lambda.blockSignals(False)
            self.xc = self.edit_xc.value()
            self.yc = self.edit_yc.value()
            return

        if self.instrument_mode == "ID13":
            center_1 = self.get_header_float(header, *CENTER_X_KEYS)
            center_2 = self.get_header_float(header, *CENTER_Y_KEYS)
            sample_distance = self.get_header_float(header, "SampleDistance", "sampledistance", "sample_distance", "Distance", "DetectorDistance")
            pixel_x_m = self.get_header_float(header, "PSize_1", "psize_1", "PSize_X", "PixelSizeX", "pixel_size_x", "x_pixel_size")
            pixel_y_m = self.get_header_float(header, "PSize_2", "psize_2", "PSize_Y", "PixelSizeY", "pixel_size_y", "y_pixel_size")
            wavelength_m = self.get_header_float(header, "WaveLength", "Wavelength", "wavelength", "Lambda", "lambda")
            self.edit_xc.blockSignals(True)
            self.edit_yc.blockSignals(True)
            self.edit_distance.blockSignals(True)
            self.edit_px_x.blockSignals(True)
            self.edit_px_y.blockSignals(True)
            self.edit_lambda.blockSignals(True)
            self.edit_xc.setValue(center_1 if center_1 is not None else ID13_DEFAULT_CENTER_X)
            self.edit_yc.setValue(center_2 if center_2 is not None else ID13_DEFAULT_CENTER_Y)
            self.edit_distance.setValue(sample_distance if sample_distance is not None else ID13_DEFAULT_DISTANCE_M)
            self.edit_px_x.setValue(pixel_x_m * 1000 if pixel_x_m is not None else ID13_DEFAULT_PIXEL_MM)
            self.edit_px_y.setValue(pixel_y_m * 1000 if pixel_y_m is not None else ID13_DEFAULT_PIXEL_MM)
            self.edit_lambda.setValue(wavelength_m * 1e10 if wavelength_m is not None else ID13_DEFAULT_WAVELENGTH_A)
            self.edit_xc.blockSignals(False)
            self.edit_yc.blockSignals(False)
            self.edit_distance.blockSignals(False)
            self.edit_px_x.blockSignals(False)
            self.edit_px_y.blockSignals(False)
            self.edit_lambda.blockSignals(False)
            self.xc = self.edit_xc.value()
            self.yc = self.edit_yc.value()
            return

    def get_header_float(self, header, *names):
        for name in names:
            if name in header:
                try:
                    return float(normalize_decimal_text(header[name]))
                except (TypeError, ValueError):
                    return None
        return None

    def apply_header_values(self, header):
        center_1 = self.get_header_float(header, *CENTER_X_KEYS)
        center_2 = self.get_header_float(header, *CENTER_Y_KEYS)
        sample_distance = self.get_header_float(header, "SampleDistance", "sampledistance", "sample_distance", "Distance", "DetectorDistance")
        pixel_x_m = self.get_header_float(header, "PSize_1", "psize_1", "PSize_X", "PixelSizeX", "pixel_size_x", "x_pixel_size")
        pixel_y_m = self.get_header_float(header, "PSize_2", "psize_2", "PSize_Y", "PixelSizeY", "pixel_size_y", "y_pixel_size")
        wavelength_m = self.get_header_float(header, "WaveLength", "Wavelength", "wavelength", "Lambda", "lambda")

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

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose folder", str(self.current_folder))
        if folder:
            self.current_folder = Path(folder)
            self.folder_path.setText(str(self.current_folder))
            self.refresh_files()

    def refresh_files(self):
        if not hasattr(self, "file_list"):
            return

        folder = Path(self.folder_path.text()).expanduser()
        if not folder.exists() or not folder.is_dir():
            return

        self.current_folder = folder
        if not self._syncing_folder:
            self.folder_changed.emit(self.current_folder)
        self.file_list.clear()

        extension_patterns = self.extension_filter.text().split() or ["*.edf", "*.h5", "*.hdf5"]
        name_pattern = self.name_filter.text().strip() or "*"
        iterator = folder.rglob("*") if self.show_subfolders_checkbox.isChecked() else folder.glob("*")

        files = []
        for path in iterator:
            if not path.is_file():
                continue
            if should_hide_file_in_browser(path):
                continue
            lower_name = path.name.lower()
            if not any(fnmatch.fnmatch(lower_name, pattern.lower()) for pattern in extension_patterns):
                continue
            if not fnmatch.fnmatch(path.name, name_pattern):
                continue
            if self.only_thumbs_up_checkbox.isChecked() and not is_file_rated_up(path):
                continue
            files.append(path)

        for path in sorted(files):
            item = QListWidgetItem(str(path.relative_to(folder)))
            set_item_file_path(item, path)
            self.file_list.addItem(item)

    def set_folder_from_external_tab(self, folder):
        folder = Path(folder).expanduser()
        if folder.exists() and folder.is_dir():
            self._syncing_folder = True
            self.current_folder = folder
            if hasattr(self, "folder_path"):
                self.folder_path.setText(str(self.current_folder))
                self.refresh_files()
            self._syncing_folder = False

    def open_selected_file(self, item=None):
        if item is None:
            item = self.file_list.currentItem() if hasattr(self, "file_list") else None
        if item is None:
            self.open_image_file()
            return

        path = Path(item.data(Qt.UserRole) or self.current_folder / item.text())
        self.open_image_file(path)

    def open_image_file(self, file_path=None):
        if file_path is None:
            file, _ = QFileDialog.getOpenFileName(
                self,
                "Open EDF or H5 file",
                str(self.current_folder),
                "Image data (*.edf *.h5 *.hdf5);;EDF (*.edf);;HDF5 (*.h5 *.hdf5);;All files (*)"
            )
            if not file:
                return
            file_path = file

        try:
            path = Path(file_path)
            suffix = path.suffix.lower()

            if suffix == ".edf":
                img, header, *_ = read_edf_file(path)
                file_type = "EDF"
            elif suffix in [".h5", ".hdf5"]:
                img, header = read_h5_first_image(path)
                file_type = "H5"
            else:
                raise ValueError("Unsupported file format. Please select an EDF, H5 or HDF5 file.")

            self.input_file = path
            self.current_folder = path.parent
            if hasattr(self, "folder_path"):
                self.folder_path.setText(str(self.current_folder))
            self.folder_changed.emit(self.current_folder)
            self.current_header = header
            self.file_loaded = True
            self.set_controls_enabled(True)

            self.img_raw = img.astype(float)
            self.img_raw[self.img_raw > 4e9] = np.nan
            self.img_raw[self.img_raw < 0] = np.nan
            if self.display_vmin is None or not self.lock_contrast_checkbox.isChecked():
                self.auto_set_image_contrast(self.img_raw)

            self.apply_instrument_preset()

            self.theta_deg = 0
            self.beamstop_radius = self.edit_beamstop_radius.value()
            self.label_theta.setText(f"θ = {self.theta_deg:.2f}°")

            self.update_masked_image()
            self.canvas_img.reset_view_on_next_show = True
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
            self.img = None
            self.canvas_img.clear_image()
            clear_plot_canvas(self.canvas_h)
            clear_plot_canvas(self.canvas_v)
            return

        self.img = self.img_raw.copy()
        r = self.beamstop_radius

        if r > 0:
            ny, nx = self.img.shape
            y, x = np.ogrid[1:ny + 1, 1:nx + 1]
            mask = (x - self.xc) ** 2 + (y - self.yc) ** 2 <= r ** 2
            self.img[mask] = np.nan

    def auto_set_image_contrast(self, image):
        img_disp = image.astype(float).copy()
        img_disp[~np.isfinite(img_disp)] = np.nan
        img_disp[img_disp < 0] = np.nan
        with np.errstate(invalid="ignore", divide="ignore"):
            display = np.log10(img_disp + 1)

        finite = display[np.isfinite(display)]
        if finite.size == 0:
            self.display_data_min = 0.0
            self.display_data_max = 1.0
        else:
            self.display_data_min = float(np.nanpercentile(finite, 1))
            self.display_data_max = float(np.nanpercentile(finite, 99.5))
            if self.display_data_max <= self.display_data_min:
                self.display_data_max = self.display_data_min + 1.0

        self.display_vmin = self.display_data_min
        self.display_vmax = self.display_data_max
        self.sync_image_contrast_sliders()

    def auto_current_image_contrast(self):
        if self.img_raw is None:
            return
        self.auto_set_image_contrast(self.img_raw)
        self.show_center_image()

    def sync_image_contrast_sliders(self):
        span = self.display_data_max - self.display_data_min
        if span <= 0:
            self.vmin_label.setText("Min: -")
            self.vmax_label.setText("Max: -")
            return

        min_pos = int(round((self.display_vmin - self.display_data_min) / span * 1000))
        max_pos = int(round((self.display_vmax - self.display_data_min) / span * 1000))
        min_pos = max(0, min(1000, min_pos))
        max_pos = max(0, min(1000, max_pos))

        self.vmin_slider.blockSignals(True)
        self.vmax_slider.blockSignals(True)
        self.vmin_slider.setValue(min_pos)
        self.vmax_slider.setValue(max_pos)
        self.vmin_slider.blockSignals(False)
        self.vmax_slider.blockSignals(False)
        self.vmin_label.setText(f"Min: {self.display_vmin:.3g}")
        self.vmax_label.setText(f"Max: {self.display_vmax:.3g}")

    def update_image_contrast_from_sliders(self):
        span = self.display_data_max - self.display_data_min
        if span <= 0:
            return

        min_pos = self.vmin_slider.value()
        max_pos = self.vmax_slider.value()
        if min_pos >= max_pos:
            sender = self.sender()
            if sender is self.vmin_slider:
                max_pos = min(1000, min_pos + 1)
                self.vmax_slider.blockSignals(True)
                self.vmax_slider.setValue(max_pos)
                self.vmax_slider.blockSignals(False)
            else:
                min_pos = max(0, max_pos - 1)
                self.vmin_slider.blockSignals(True)
                self.vmin_slider.setValue(min_pos)
                self.vmin_slider.blockSignals(False)

        self.display_vmin = self.display_data_min + span * min_pos / 1000.0
        self.display_vmax = self.display_data_min + span * max_pos / 1000.0
        self.vmin_label.setText(f"Min: {self.display_vmin:.3g}")
        self.vmax_label.setText(f"Max: {self.display_vmax:.3g}")
        self.show_center_image()

    def show_center_image(self):
        if self.img is None:
            self.canvas_img.clear_image()
            return

        self.canvas_img.show_image(self.img, self.xc, self.yc, "", self.display_vmin, self.display_vmax)
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

    def q_display_factor(self):
        return 0.1 if self.q_axis_unit == "A" else 1.0

    def q_axis_label(self):
        return "q (Å⁻¹)" if self.q_axis_unit == "A" else "q (nm⁻¹)"

    def on_graph_button_press(self, event):
        if event.button != 1:
            return
        try:
            clicked_label = (
                self.canvas_h.ax.xaxis.label.contains(event)[0]
                or self.canvas_v.ax.xaxis.label.contains(event)[0]
            )
        except Exception:
            clicked_label = False
        if not clicked_label:
            return

        self.q_axis_unit = "A" if self.q_axis_unit == "nm" else "nm"
        self.update_plots()

    def update_plots(self):
        if self.img is None:
            clear_plot_canvas(self.canvas_h)
            clear_plot_canvas(self.canvas_v)
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
        q_factor = self.q_display_factor()

        self.canvas_h.ax.clear()
        self.canvas_v.ax.clear()
        self.canvas_h.ax.set_axis_on()
        self.canvas_v.ax.set_axis_on()

        valid = np.isfinite(q1) & np.isfinite(i1) & (q1 > 0) & (i1 > 0)
        self.canvas_h.ax.loglog(q1[valid] * q_factor, i1[valid], "-", color="red", linewidth=1.3, label="H +")

        valid = np.isfinite(q2) & np.isfinite(i2) & (q2 > 0) & (i2 > 0)
        self.canvas_h.ax.loglog(q2[valid] * q_factor, i2[valid], "--", color="darkred", linewidth=1.1, label="H -")

        valid = np.isfinite(q3) & np.isfinite(i3) & (q3 > 0) & (i3 > 0)
        self.canvas_v.ax.loglog(q3[valid] * q_factor, i3[valid], "-", color="blue", linewidth=1.3, label="V +")

        valid = np.isfinite(q4) & np.isfinite(i4) & (q4 > 0) & (i4 > 0)
        self.canvas_v.ax.loglog(q4[valid] * q_factor, i4[valid], "--", color="navy", linewidth=1.1, label="V -")

        for ax, title in [(self.canvas_h.ax, "Horizontal I(q)"), (self.canvas_v.ax, "Vertical I(q)")]:
            ax.set_xlabel(self.q_axis_label())
            ax.set_ylabel("I(q)")
            ax.set_title(title)
            ax.grid(True, which="both")
            install_selectable_legend(ax, ax.legend(loc="best"))

        self.canvas_h.draw_idle()
        self.canvas_v.draw_idle()

    # Replace any remaining occurrence of "Tous les fichiers (*)" with "All files (*)"

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_side_block_widths()

    def update_side_block_widths(self):
        if not hasattr(self, "image_box") or not hasattr(self, "graph_box"):
            return

        margins = self.main_grid.contentsMargins()
        available_width = (
            self.width()
            - margins.left()
            - margins.right()
            - FILE_BROWSER_WIDTH
            - 2 * BLOCK_SPACING
        )
        side_width = max(0, available_width // 2)

        self.image_box.setMinimumWidth(0)
        self.graph_box.setMinimumWidth(0)

        self.image_box.setMaximumWidth(side_width)
        self.graph_box.setMaximumWidth(side_width)
