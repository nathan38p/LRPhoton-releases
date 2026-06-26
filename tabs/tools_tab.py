from pathlib import Path


import numpy as np

try:
    import h5py
except Exception:
    h5py = None

try:
    import fabio
except Exception:
    fabio = None

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
except Exception:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from matplotlib.figure import Figure

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSlider,
    QPlainTextEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)

try:
    from tabs.file_utils import set_item_file_path
except Exception:
    def set_item_file_path(item, path):
        item.setData(Qt.UserRole, str(path))

try:
    from tabs.file_ratings import should_hide_file_in_browser
except Exception:
    def should_hide_file_in_browser(_path):
        return False

try:
    from tabs.ui_style import FILE_BROWSER_WIDTH, PAGE_MARGINS
except Exception:
    FILE_BROWSER_WIDTH = 320
    PAGE_MARGINS = (4, 4, 4, 4)

from tabs.line_geometry import LineGeometrySelector, line_geometry_to_lrphoton


class PreviewCanvas(FigureCanvas):
    def __init__(self):
        self.figure = Figure(figsize=(3.2, 3.2), tight_layout=True)
        self.ax = self.figure.add_subplot(111)
        super().__init__(self.figure)
        self.draw_empty()

    def draw_empty(self):
        self.ax.clear()
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.draw_idle()

    def draw_image(self, image, vmin=None, vmax=None):
        self.ax.clear()
        display_image = np.asarray(image, dtype=float)
        display_image = np.where(display_image > 1e9, np.nan, display_image)
        self.ax.imshow(display_image, cmap="jet", origin="upper", vmin=vmin, vmax=vmax)        
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.draw_idle()

class RangeSlider(QWidget):
    valuesChanged = Signal(int, int)

    def __init__(self):
        super().__init__()
        self._minimum = 1
        self._maximum = 1
        self._lower = 1
        self._upper = 1
        self._active_handle = None
        self.setMinimumHeight(30)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def setRange(self, minimum, maximum):
        self._minimum = int(minimum)
        self._maximum = max(int(maximum), self._minimum)
        self._lower = max(self._minimum, min(self._lower, self._maximum))
        self._upper = max(self._lower, min(self._upper, self._maximum))
        self.update()

    def setValues(self, lower, upper, emit_signal=True):
        lower = max(self._minimum, min(int(lower), self._maximum))
        upper = max(self._minimum, min(int(upper), self._maximum))
        if upper < lower:
            upper = lower

        changed = lower != self._lower or upper != self._upper
        self._lower = lower
        self._upper = upper
        self.update()

        if changed and emit_signal:
            self.valuesChanged.emit(self._lower, self._upper)

    def value_to_x(self, value):
        if self._maximum <= self._minimum:
            return 14
        width = max(1, self.width() - 28)
        ratio = (value - self._minimum) / (self._maximum - self._minimum)
        return int(14 + ratio * width)

    def x_to_value(self, x):
        if self._maximum <= self._minimum:
            return self._minimum
        width = max(1, self.width() - 28)
        ratio = (x - 14) / width
        ratio = max(0.0, min(1.0, ratio))
        return int(round(self._minimum + ratio * (self._maximum - self._minimum)))

    def paintEvent(self, event):
        from PySide6.QtGui import QColor, QPainter, QPen

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        y = self.height() // 2
        lower_x = self.value_to_x(self._lower)
        upper_x = self.value_to_x(self._upper)

        painter.setPen(QPen(QColor("#cfcfcfcf"), 5, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(14, y, self.width() - 14, y)

        painter.setPen(QPen(QColor("#2f80ed"), 5, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(lower_x, y, upper_x, y)

        painter.setPen(QPen(QColor("#999999"), 1))
        painter.setBrush(QColor("#ffffff"))
        painter.drawEllipse(QPoint(lower_x, y), 8, 8)
        painter.drawEllipse(QPoint(upper_x, y), 8, 8)

    def mousePressEvent(self, event):
        pos_x = event.position().x() if hasattr(event, "position") else event.x()
        lower_x = self.value_to_x(self._lower)
        upper_x = self.value_to_x(self._upper)
        self._active_handle = "lower" if abs(pos_x - lower_x) <= abs(pos_x - upper_x) else "upper"
        self.update_from_mouse(pos_x)

    def mouseMoveEvent(self, event):
        pos_x = event.position().x() if hasattr(event, "position") else event.x()
        self.update_from_mouse(pos_x)

    def mouseReleaseEvent(self, event):
        self._active_handle = None

    def update_from_mouse(self, pos_x):
        value = self.x_to_value(pos_x)
        if self._active_handle == "lower":
            self.setValues(min(value, self._upper), self._upper)
        elif self._active_handle == "upper":
            self.setValues(self._lower, max(value, self._lower))


class GeometryFieldsDialog(QDialog):
    def __init__(self, values, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Custom geometry")
        self.values = dict(values or {})

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.sample_distance_spin = self.make_spin(0.0, 100.0, 6, self.values.get("SampleDistance", 0.9))
        self.wavelength_spin = self.make_spin(0.0, 1e-6, 14, self.values.get("Wavelength", 1.54189e-10))
        self.psize_1_spin = self.make_spin(0.0, 1.0, 10, self.values.get("PSize_1", 7.5e-05))
        self.psize_2_spin = self.make_spin(0.0, 1.0, 10, self.values.get("PSize_2", 7.5e-05))
        self.center_1_spin = self.make_spin(-100000.0, 100000.0, 6, self.values.get("Center_1", 0.0))
        self.center_2_spin = self.make_spin(-100000.0, 100000.0, 6, self.values.get("Center_2", 0.0))

        form.addRow("SampleDistance (m):", self.sample_distance_spin)
        form.addRow("Wavelength (m):", self.wavelength_spin)
        form.addRow("PSize_1 (m):", self.psize_1_spin)
        form.addRow("PSize_2 (m):", self.psize_2_spin)
        form.addRow("Center_1 (px):", self.center_1_spin)
        form.addRow("Center_2 (px):", self.center_2_spin)
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def make_spin(self, minimum, maximum, decimals, value):
        spin = QDoubleSpinBox()
        spin.setRange(float(minimum), float(maximum))
        spin.setDecimals(int(decimals))
        spin.setValue(float(value))
        spin.setKeyboardTracking(False)
        return spin

    def geometry_values(self):
        return {
            "SampleDistance": str(self.sample_distance_spin.value()),
            "Wavelength": str(self.wavelength_spin.value()),
            "PSize_1": str(self.psize_1_spin.value()),
            "PSize_2": str(self.psize_2_spin.value()),
            "Center_1": str(self.center_1_spin.value()),
            "Center_2": str(self.center_2_spin.value()),
        }
class ToolsTab(QWidget):
    folder_changed = Signal(str)

    def __init__(self):
        super().__init__()
        self.folder_path = ""
        self.current_folder = None
        self.name_filter = ""
        self.include_subfolders = False

        self.current_file_path = None
        self.current_h5_dataset_path = None
        self.current_header = {}
        self.current_frame_count = 0
        self.contrast_min = None
        self.contrast_max = None
        self._updating_contrast_controls = False
        self.geometry_mode = "XENOCS"
        self.custom_geometry_values = {
            "SampleDistance": "0.9",
            "Wavelength": "1.54189e-10",
            "PSize_1": "7.5e-05",
            "PSize_2": "7.5e-05",
            "Center_1": "0.0",
            "Center_2": "0.0",
        }

        self.build_ui()

    def build_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(*PAGE_MARGINS)
        main_layout.setSpacing(8)

        self.file_panel = QWidget()
        self.file_panel.setFixedWidth(FILE_BROWSER_WIDTH)
        self.file_panel.setStyleSheet(
            "QWidget { background-color: #eeeeee; border: 1px solid #d0d0d0; border-radius: 8px; }"
            "QLabel { border: none; background: transparent; }"
            "QLineEdit { background-color: #ffffff; border: 1px solid #d7d7d7; border-radius: 5px; padding: 2px 6px; }"
            "QPushButton { background-color: #e2e2e2; border: none; border-radius: 5px; padding: 4px 8px; }"
            "QPushButton:hover { background-color: #d8d8d8; }"
            "QListWidget { background-color: #eeeeee; border: none; }"
            "QCheckBox { border: none; background: transparent; }"
            "QSlider { border: none; background: transparent; }"
        )

        file_layout = QVBoxLayout(self.file_panel)
        file_layout.setContentsMargins(10, 10, 10, 10)
        file_layout.setSpacing(7)

        title = QLabel("File browser")
        title.setStyleSheet("font-weight: 600;")
        file_layout.addWidget(title)

        self.folder_label = QLineEdit()
        self.folder_label.setReadOnly(True)
        self.folder_label.setPlaceholderText("No folder selected")
        file_layout.addWidget(self.folder_label)

        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_folder)
        file_layout.addWidget(self.browse_button)

        name_row = QHBoxLayout()
        name_label = QLabel("Name:")
        name_label.setFixedWidth(72)
        name_row.addWidget(name_label)

        self.filter_edit = QLineEdit("**")
        self.filter_edit.textChanged.connect(self.on_filter_changed)
        name_row.addWidget(self.filter_edit, 1)
        file_layout.addLayout(name_row)

        ext_row = QHBoxLayout()
        ext_label = QLabel("Extensions:")
        ext_label.setFixedWidth(72)
        ext_row.addWidget(ext_label)

        self.extensions_edit = QLineEdit("*.h5 *.hdf5 *.edf")
        self.extensions_edit.setReadOnly(True)
        ext_row.addWidget(self.extensions_edit, 1)
        file_layout.addLayout(ext_row)

        options_row = QHBoxLayout()
        self.subfolders_checkbox = QCheckBox("Show subfolders")
        self.subfolders_checkbox.toggled.connect(self.on_subfolders_toggled)
        options_row.addWidget(self.subfolders_checkbox)
        options_row.addStretch(1)
        file_layout.addLayout(options_row)

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_files)
        file_layout.addWidget(self.refresh_button)

        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.SingleSelection)
        self.file_list.currentItemChanged.connect(self.on_selected_file_changed)
        file_layout.addWidget(self.file_list, 1)

        geometry_group = QGroupBox("Geometry")
        geometry_group.setStyleSheet(
            "QGroupBox { background-color: #eeeeee; border: 1px solid #d0d0d0; border-radius: 8px; margin-top: 12px; font-weight: 600; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }"
            "QPushButton { background-color: #f5f5f5; border: 1px solid #d7d7d7; border-radius: 5px; padding: 3px 6px; }"
            "QPushButton:checked { background-color: #2f80ed; color: white; border: 1px solid #2f80ed; }"
        )

        geometry_layout = QHBoxLayout(geometry_group)
        geometry_layout.setContentsMargins(10, 18, 10, 10)
        geometry_layout.setSpacing(4)

        self.geometry_button_group = QButtonGroup(self)
        self.geometry_button_group.setExclusive(True)
        self.geometry_buttons = {}

        for name in ["XENOCS", "ID02", "ID13", "Custom"]:
            button = QPushButton(name)
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False, n=name: self.set_geometry_mode(n))
            self.geometry_button_group.addButton(button)
            self.geometry_buttons[name] = button
            geometry_layout.addWidget(button)
            button.hide()

        self.geometry_buttons["XENOCS"].setChecked(True)

        self.geometry_plus_button = QPushButton("+")
        self.geometry_plus_button.clicked.connect(self.open_custom_geometry_dialog)
        geometry_layout.addWidget(self.geometry_plus_button)
        self.geometry_plus_button.hide()

        self.line_geometry_selector = LineGeometrySelector(self, "XENOCS")
        self.line_geometry_selector.geometry_selected.connect(self.apply_line_geometry_selection)
        geometry_layout.addWidget(self.line_geometry_selector, 1)

        file_layout.addWidget(geometry_group)

        contrast_group = QGroupBox("Contrast")
        contrast_group.setStyleSheet(
            "QGroupBox { background-color: #eeeeee; border: 1px solid #d0d0d0; border-radius: 8px; margin-top: 12px; font-weight: 600; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }"
            "QLabel { border: none; background: transparent; }"
            "QPushButton { background-color: #e2e2e2; border: none; border-radius: 5px; padding: 4px 8px; }"
            "QPushButton:hover { background-color: #d8d8d8; }"
            "QSlider { border: none; background: transparent; }"
        )
        contrast_layout = QVBoxLayout(contrast_group)
        contrast_layout.setContentsMargins(10, 18, 10, 10)
        contrast_layout.setSpacing(6)

        contrast_min_row = QHBoxLayout()
        contrast_min_row.addWidget(QLabel("Min:"))
        self.contrast_min_slider = QSlider(Qt.Horizontal)
        self.contrast_min_slider.setRange(0, 1000)
        self.contrast_min_slider.setValue(0)
        self.contrast_min_slider.valueChanged.connect(self.on_contrast_changed)
        contrast_min_row.addWidget(self.contrast_min_slider, 1)
        self.contrast_min_label = QLabel("auto")
        self.contrast_min_label.setFixedWidth(70)
        contrast_min_row.addWidget(self.contrast_min_label)
        contrast_layout.addLayout(contrast_min_row)

        contrast_max_row = QHBoxLayout()
        contrast_max_row.addWidget(QLabel("Max:"))
        self.contrast_max_slider = QSlider(Qt.Horizontal)
        self.contrast_max_slider.setRange(0, 1000)
        self.contrast_max_slider.setValue(1000)
        self.contrast_max_slider.valueChanged.connect(self.on_contrast_changed)
        contrast_max_row.addWidget(self.contrast_max_slider, 1)
        self.contrast_max_label = QLabel("auto")
        self.contrast_max_label.setFixedWidth(70)
        contrast_max_row.addWidget(self.contrast_max_label)
        contrast_layout.addLayout(contrast_max_row)

        self.auto_contrast_button = QPushButton("Auto")
        self.auto_contrast_button.clicked.connect(self.auto_contrast)
        contrast_layout.addWidget(self.auto_contrast_button)

        file_layout.addWidget(contrast_group)
        main_layout.addWidget(self.file_panel)

        self.content_panel = QWidget()
        self.content_panel.setStyleSheet(
            "QWidget { background-color: transparent; }"
            "QGroupBox { background-color: #eeeeee; border: 1px solid #d0d0d0; border-radius: 8px; margin-top: 14px; font-weight: 600; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }"
            "QLineEdit, QSpinBox, QComboBox, QPlainTextEdit { background-color: #ffffff; border: 1px solid #d7d7d7; border-radius: 5px; padding: 2px 6px; }"
            "QPushButton { background-color: #e2e2e2; border: none; border-radius: 5px; padding: 5px 10px; }"
            "QPushButton:hover { background-color: #d8d8d8; }"
        )

        content_layout = QVBoxLayout(self.content_panel)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(8)

        preview_group = QGroupBox("1. Frame range and preview")
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.setContentsMargins(12, 18, 12, 12)
        preview_layout.setSpacing(10)

        range_controls = QHBoxLayout()
        range_controls.setSpacing(8)
        range_controls.addWidget(QLabel("Start frame:"))

        self.start_frame_spin = QSpinBox()
        self.start_frame_spin.setMinimum(1)
        self.start_frame_spin.setMaximum(1)
        self.start_frame_spin.valueChanged.connect(self.on_frame_spin_changed)
        range_controls.addWidget(self.start_frame_spin)

        range_controls.addWidget(QLabel("Range:"))
        self.frame_range_slider = RangeSlider()
        self.frame_range_slider.valuesChanged.connect(self.on_frame_range_slider_changed)
        range_controls.addWidget(self.frame_range_slider, 1)

        range_controls.addWidget(QLabel("End frame:"))
        self.end_frame_spin = QSpinBox()
        self.end_frame_spin.setMinimum(1)
        self.end_frame_spin.setMaximum(1)
        self.end_frame_spin.valueChanged.connect(self.on_frame_spin_changed)
        range_controls.addWidget(self.end_frame_spin)

        self.frame_count_label = QLabel("0 frame(s)")
        self.frame_count_label.setStyleSheet("color: #666666;")
        range_controls.addWidget(self.frame_count_label)
        preview_layout.addLayout(range_controls)

        preview_images_layout = QHBoxLayout()
        preview_images_layout.setSpacing(10)

        self.start_preview = PreviewCanvas()
        preview_images_layout.addWidget(self.start_preview, 1)

        self.end_preview = PreviewCanvas()
        preview_images_layout.addWidget(self.end_preview, 1)

        preview_layout.addLayout(preview_images_layout, 1)
        content_layout.addWidget(preview_group, 1)

        header_group = QGroupBox("2. Header used for exported frames")
        header_layout = QVBoxLayout(header_group)
        header_layout.setContentsMargins(12, 18, 12, 12)

        self.header_text = QPlainTextEdit()
        self.header_text.setReadOnly(True)
        self.header_text.setMinimumHeight(105)
        header_layout.addWidget(self.header_text)

        content_layout.addWidget(header_group)

        export_group = QGroupBox("3. Export individual frames")
        export_layout = QHBoxLayout(export_group)
        export_layout.setContentsMargins(12, 18, 12, 12)
        export_layout.setSpacing(8)

        export_layout.addWidget(QLabel("Format:"))

        self.output_format_combo = QComboBox()
        self.output_format_combo.addItems(["H5", "EDF"])
        export_layout.addWidget(self.output_format_combo)

        self.export_button = QPushButton("Export frames")
        self.export_button.clicked.connect(self.export_frames)
        export_layout.addWidget(self.export_button)

        export_layout.addStretch(1)
        content_layout.addWidget(export_group)

        main_layout.addWidget(self.content_panel, 1)

    def browse_folder(self):
        start_folder = self.folder_path or str(Path.home())
        folder = QFileDialog.getExistingDirectory(self, "Select folder", start_folder)
        if not folder:
            return
        self.set_folder(folder, emit_signal=True)

    def set_folder(self, folder_path, emit_signal=False):
        self.folder_path = str(folder_path) if folder_path else ""
        self.current_folder = Path(self.folder_path) if self.folder_path else None
        self.folder_label.setText(self.folder_path)
        self.refresh_files()

        if emit_signal and self.folder_path:
            self.folder_changed.emit(self.folder_path)

    def set_folder_path(self, folder_path):
        self.set_folder(folder_path, emit_signal=True)

    def set_folder_from_external_tab(self, folder_path):
        self.set_folder(folder_path, emit_signal=False)

    def set_geometry_mode(self, mode):
        self.geometry_mode = mode
        if mode in getattr(self, "geometry_buttons", {}):
            self.geometry_buttons[mode].setChecked(True)
        if hasattr(self, "line_geometry_selector") and mode in self.line_geometry_selector.geometries:
            self.line_geometry_selector.set_current_name(mode)
        self.update_header_preview()
        self.update_previews()

    def apply_line_geometry_selection(self, name, geometry):
        values = line_geometry_to_lrphoton(geometry)
        self.custom_geometry_values = {
            "SampleDistance": str(values["distance_m"]),
            "Wavelength": str(values["wavelength_a"] * 1e-10),
            "PSize_1": str(values["pixel_x_mm"] * 1e-3),
            "PSize_2": str(values["pixel_y_mm"] * 1e-3),
            "Center_1": str(values["xc"]),
            "Center_2": str(values["yc"]),
        }
        self.geometry_mode = "Custom" if name not in {"XENOCS", "ID02", "ID13"} else name
        self.update_header_preview()
        self.update_previews()

    def open_custom_geometry_dialog(self):
        dialog = GeometryFieldsDialog(self.custom_geometry_values, self)
        if dialog.exec() != QDialog.Accepted:
            return
        self.custom_geometry_values = dialog.geometry_values()
        self.set_geometry_mode("Custom")

    def current_geometry_mode(self):
        return self.geometry_mode

    def on_filter_changed(self, text):
        text = text.strip().lower()
        if text in ("", "*", "**"):
            self.name_filter = ""
        else:
            self.name_filter = text.replace("*", "")
        self.refresh_files()

    def on_subfolders_toggled(self, checked):
        self.include_subfolders = bool(checked)
        self.refresh_files()

    def on_contrast_changed(self):
        if self._updating_contrast_controls:
            return
        self.update_contrast_from_sliders()
        self.update_previews()

    def update_contrast_from_sliders(self):
        if self.contrast_min is None or self.contrast_max is None:
            return
        data_min = float(self.contrast_min)
        data_max = float(self.contrast_max)
        if data_max <= data_min:
            return

        min_ratio = self.contrast_min_slider.value() / 1000.0
        max_ratio = self.contrast_max_slider.value() / 1000.0
        vmin = data_min + min_ratio * (data_max - data_min)
        vmax = data_min + max_ratio * (data_max - data_min)
        if vmax <= vmin:
            vmax = vmin + max((data_max - data_min) / 1000.0, 1e-12)
        self.contrast_min_label.setText(f"{vmin:.3g}")
        self.contrast_max_label.setText(f"{vmax:.3g}")

    def current_contrast_limits(self):
        if self.contrast_min is None or self.contrast_max is None:
            return None, None
        data_min = float(self.contrast_min)
        data_max = float(self.contrast_max)
        if data_max <= data_min:
            return None, None
        min_ratio = self.contrast_min_slider.value() / 1000.0
        max_ratio = self.contrast_max_slider.value() / 1000.0
        vmin = data_min + min_ratio * (data_max - data_min)
        vmax = data_min + max_ratio * (data_max - data_min)
        if vmax <= vmin:
            vmax = vmin + max((data_max - data_min) / 1000.0, 1e-12)
        return vmin, vmax

    def auto_contrast(self):
        if self.current_file_path is None or self.current_frame_count <= 0:
            return
        try:
            start_image = self.read_frame(self.start_frame_spin.value() - 1)
            end_image = self.read_frame(self.end_frame_spin.value() - 1)
            values = np.concatenate([
                np.asarray(start_image, dtype=float).ravel(),
                np.asarray(end_image, dtype=float).ravel(),
            ])
            values = values[np.isfinite(values)]
            values = values[values < 1e9]
            if values.size == 0:
                return
            vmin = float(np.nanpercentile(values, 1))
            vmax = float(np.nanpercentile(values, 99))
            if vmax <= vmin:
                vmin = float(np.nanmin(values))
                vmax = float(np.nanmax(values))
            if vmax <= vmin:
                vmax = vmin + 1.0
            self.contrast_min = vmin
            self.contrast_max = vmax
            self._updating_contrast_controls = True
            self.contrast_min_slider.setValue(0)
            self.contrast_max_slider.setValue(1000)
            self._updating_contrast_controls = False
            self.contrast_min_label.setText(f"{vmin:.3g}")
            self.contrast_max_label.setText(f"{vmax:.3g}")
            self.update_previews()
        except Exception as exc:
            QMessageBox.warning(self, "Contrast", f"Unable to compute automatic contrast:\n{exc}")

    def is_multi_file(self, path):
        name = path.name.lower()
        suffix = path.suffix.lower()

        if suffix in (".h5", ".hdf5"):
            return self.is_multi_h5_file(path)

        if suffix == ".edf":
            return "multi" in name

        return False

    def is_multi_h5_file(self, path):
        if h5py is None:
            return "multi" in path.name.lower()

        try:
            with h5py.File(path, "r") as h5:
                return self.find_first_h5_image_stack_path(h5) is not None
        except Exception:
            return False

    def find_first_h5_image_stack_path(self, h5_object):
        found_path = None

        def visitor(name, obj):
            nonlocal found_path
            if found_path is not None:
                return
            if not hasattr(obj, "shape"):
                return

            shape = tuple(int(v) for v in obj.shape)
            if len(shape) >= 3 and shape[-1] >= 16 and shape[-2] >= 16:
                found_path = name

        h5_object.visititems(visitor)
        return found_path

    def refresh_files(self):
        self.file_list.clear()

        folder = Path(self.folder_path) if self.folder_path else None
        self.current_folder = folder

        if folder is None or not folder.exists() or not folder.is_dir():
            return

        iterator = folder.rglob("*") if self.include_subfolders else folder.glob("*")
        candidate_suffixes = {".h5", ".hdf5", ".edf"}
        files = []

        for path in iterator:
            if not path.is_file():
                continue
            if should_hide_file_in_browser(path):
                continue
            if path.suffix.lower() not in candidate_suffixes:
                continue
            if not self.is_multi_file(path):
                continue

            relative_name = str(path.relative_to(folder)).lower()
            if self.name_filter and self.name_filter not in path.name.lower() and self.name_filter not in relative_name:
                continue

            files.append(path)

        for path in sorted(files, key=lambda p: str(p.relative_to(folder)).lower()):
            item = QListWidgetItem(str(path.relative_to(folder)))
            set_item_file_path(item, path)
            self.file_list.addItem(item)

    def selected_files(self):
        files = []
        folder = self.current_folder or (Path(self.folder_path) if self.folder_path else None)

        for item in self.file_list.selectedItems():
            path = item.data(Qt.UserRole)
            if path is None and folder is not None:
                path = folder / item.text()
            if path is not None:
                files.append(Path(path))

        return files

    def on_selected_file_changed(self, current, previous=None):
        if current is None:
            self.clear_source_information()
            return

        path = current.data(Qt.UserRole)
        if path is None and self.current_folder is not None:
            path = self.current_folder / current.text()

        self.load_source_file(Path(path))

    def clear_source_information(self):
        self.current_file_path = None
        self.current_h5_dataset_path = None
        self.current_header = {}
        self.current_frame_count = 0
        self.contrast_min = None
        self.contrast_max = None

        self.header_text.clear()
        self.frame_count_label.setText("0 frame(s)")
        self._updating_contrast_controls = True
        self.contrast_min_slider.setValue(0)
        self.contrast_max_slider.setValue(1000)
        self._updating_contrast_controls = False
        self.contrast_min_label.setText("auto")
        self.contrast_max_label.setText("auto")

        self.start_frame_spin.blockSignals(True)
        self.end_frame_spin.blockSignals(True)
        self.start_frame_spin.setMaximum(1)
        self.end_frame_spin.setMaximum(1)
        self.start_frame_spin.setValue(1)
        self.end_frame_spin.setValue(1)
        self.start_frame_spin.blockSignals(False)
        self.end_frame_spin.blockSignals(False)

        self.frame_range_slider.setRange(1, 1)
        self.frame_range_slider.setValues(1, 1, emit_signal=False)

        self.start_preview.draw_empty()
        self.end_preview.draw_empty()

    def load_source_file(self, path):
        self.clear_source_information()
        self.current_file_path = path

        suffix = path.suffix.lower()

        try:
            if suffix in (".h5", ".hdf5"):
                self.load_h5_source(path)
            elif suffix == ".edf":
                self.load_edf_source(path)
            else:
                return
        except Exception as exc:
            QMessageBox.warning(self, "Source file", f"Unable to read source file:\n{exc}")
            self.clear_source_information()
            return

        self.configure_frame_range()
        self.auto_contrast()
        self.update_header_preview()
        self.update_previews()

    def load_h5_source(self, path):
        if h5py is None:
            raise RuntimeError("h5py is not installed.")

        with h5py.File(path, "r") as h5:
            dataset_path = self.find_first_h5_image_stack_path(h5)

            if dataset_path is None:
                raise RuntimeError("No image stack dataset found in this H5 file.")

            dataset = h5[dataset_path]
            shape = tuple(int(v) for v in dataset.shape)

            self.current_h5_dataset_path = dataset_path
            self.current_frame_count = shape[0]
            self.current_header = self.extract_h5_header(h5, dataset)

    def load_edf_source(self, path):
        if fabio is None:
            raise RuntimeError("fabio is not installed.")

        edf = fabio.open(str(path))
        frame_count = int(getattr(edf, "nframes", 1) or 1)

        self.current_frame_count = max(1, frame_count)
        self.current_header = dict(getattr(edf, "header", {}) or {})

    def configure_frame_range(self):
        frame_count = max(1, int(self.current_frame_count or 1))

        self.frame_count_label.setText(f"{frame_count} frame(s)")

        self.start_frame_spin.blockSignals(True)
        self.end_frame_spin.blockSignals(True)

        self.start_frame_spin.setMinimum(1)
        self.end_frame_spin.setMinimum(1)
        self.start_frame_spin.setMaximum(frame_count)
        self.end_frame_spin.setMaximum(frame_count)
        self.start_frame_spin.setValue(1)
        self.end_frame_spin.setValue(frame_count)

        self.start_frame_spin.blockSignals(False)
        self.end_frame_spin.blockSignals(False)

        self.frame_range_slider.setRange(1, frame_count)
        self.frame_range_slider.setValues(1, frame_count, emit_signal=False)

    def on_frame_range_slider_changed(self, start_frame, end_frame):
        self.start_frame_spin.blockSignals(True)
        self.end_frame_spin.blockSignals(True)

        self.start_frame_spin.setValue(start_frame)
        self.end_frame_spin.setValue(end_frame)

        self.start_frame_spin.blockSignals(False)
        self.end_frame_spin.blockSignals(False)

        self.update_previews()

    def on_frame_spin_changed(self):
        start_frame = self.start_frame_spin.value()
        end_frame = self.end_frame_spin.value()

        if end_frame < start_frame:
            end_frame = start_frame
            self.end_frame_spin.blockSignals(True)
            self.end_frame_spin.setValue(end_frame)
            self.end_frame_spin.blockSignals(False)

        self.frame_range_slider.setValues(start_frame, end_frame, emit_signal=False)
        self.update_previews()

    def extract_h5_header(self, h5, dataset):
        header = {}

        def add_attrs(prefix, obj):
            for key, value in getattr(obj, "attrs", {}).items():
                header[f"{prefix}{key}"] = self.header_value_to_text(value)

        add_attrs("file/", h5)
        add_attrs("dataset/", dataset)

        wanted_aliases = {
            "SampleDistance": [
                "SampleDistance", "sampledistance", "sample_distance", "Distance", "distance",
                "detector_distance", "detectorDistance", "DetectorDistance", "sdd", "SDD",
                "LDet", "Ldet",
            ],
            "PSize_1": [
                "PSize_1", "PSize1", "pixel_size_1", "pixel_size_x", "PixelSize1",
                "PixelSizeX", "x_pixel_size", "pixelsize_x", "x_pixel_size_m",
                "detector_pixel_size_x", "x_pixel_size_meter",
            ],
            "PSize_2": [
                "PSize_2", "PSize2", "pixel_size_2", "pixel_size_y", "PixelSize2",
                "PixelSizeY", "y_pixel_size", "pixelsize_y", "y_pixel_size_m",
                "detector_pixel_size_y", "y_pixel_size_meter",
            ],
            "Center_1": [
                "Center_1", "Center1", "center_1", "center_x", "CenterX", "BeamCenterX",
                "beam_center_x", "beam_center_1", "direct_beam_x", "poni1", "PONI1",
            ],
            "Center_2": [
                "Center_2", "Center2", "center_2", "center_y", "CenterY", "BeamCenterY",
                "beam_center_y", "beam_center_2", "direct_beam_y", "poni2", "PONI2",
            ],
            "Wavelength": [
                "Wavelength", "wavelength", "lambda", "Lambda", "wave_length",
                "incident_wavelength", "beam_wavelength", "energy_wavelength",
            ],
            "Tube_HT": ["Tube_HT", "tube_ht", "TubeHT", "tube_voltage", "Voltage", "voltage"],
            "Tube_anode": ["Tube_anode", "tube_anode", "TubeAnode", "anode"],
            "Tube_current": ["Tube_current", "tube_current", "TubeCurrent", "current"],
            "Pressure": ["Pressure", "pressure"],
            "TransmittedFlux": ["TransmittedFlux", "transmitted_flux", "Transmission", "transmission"],
            "PixelSolidAngle": ["PixelSolidAngle", "pixel_solid_angle", "solid_angle"],
        }

        for output_key, aliases in wanted_aliases.items():
            found_value, found_path = self.find_h5_value_by_aliases(h5, aliases)
            if found_value is not None:
                header[output_key] = self.header_value_to_text(found_value)
                header[f"{output_key}_source"] = found_path

        return header

    def find_h5_value_by_aliases(self, h5, aliases):
        alias_lowers = {alias.lower() for alias in aliases}

        for key, value in h5.attrs.items():
            if key.lower() in alias_lowers:
                return value, f"file attribute: {key}"

        found_value = None
        found_path = None

        def visitor(name, obj):
            nonlocal found_value, found_path
            if found_value is not None:
                return

            for key, value in getattr(obj, "attrs", {}).items():
                if key.lower() in alias_lowers:
                    found_value = value
                    found_path = f"/{name} attribute: {key}"
                    return

            short_name = name.split("/")[-1].lower()
            if short_name in alias_lowers and hasattr(obj, "shape"):
                try:
                    if obj.shape == () or obj.size == 1:
                        found_value = obj[()]
                        found_path = f"/{name} dataset"
                        return
                    if obj.size > 1 and len(obj.shape) == 1:
                        found_value = obj[()]
                        found_path = f"/{name} dataset"
                        return
                except Exception:
                    pass

        h5.visititems(visitor)
        return found_value, found_path

    def header_value_to_text(self, value):
        if isinstance(value, bytes):
            return value.decode(errors="replace")

        if isinstance(value, np.ndarray):
            if value.shape == ():
                return self.header_value_to_text(value.item())
            if value.size == 1:
                return self.header_value_to_text(value.reshape(-1)[0])
            return np.array2string(value, separator=", ")

        return str(value)

    def geometry_header(self):
        geometry = self.current_geometry_mode()
        header = dict(self.current_header)

        header["Geometry"] = geometry

        if self.current_file_path is not None:
            header["SourceFile"] = str(self.current_file_path)

        if self.current_h5_dataset_path:
            header["SourceDataset"] = f"/{self.current_h5_dataset_path}"

        if self.current_frame_count:
            header["SourceFrameCount"] = str(self.current_frame_count)

        if geometry == "XENOCS":
            header.setdefault("Instrument", "XENOCS")
            header.setdefault("SampleDistance", "0.9")
            header.setdefault("Wavelength", "1.54189e-10")
            header.setdefault("PSize_1", "7.5e-05")
            header.setdefault("PSize_2", "7.5e-05")

        elif geometry == "ID02":
            header.setdefault("Instrument", "ESRF ID02")
            header.setdefault("Wavelength", "1.0e-10")

        elif geometry == "ID13":
            header.setdefault("Instrument", "ESRF ID13")
            header.setdefault("SampleDistance", "0.9")
            header.setdefault("Wavelength", "1.0e-10")
            header.setdefault("Center_1", "1294.689")
            header.setdefault("Center_2", "1310.29")
            header.setdefault("PSize_1", "7.5e-05")
            header.setdefault("PSize_2", "7.5e-05")

        elif geometry == "Custom":
            header.setdefault("Instrument", "Custom")
            for key, value in self.custom_geometry_values.items():
                header.setdefault(key, value)

        elif geometry == "+":
            header.setdefault("Instrument", "+")

        return self.normalized_detector_header(header)

    def normalized_detector_header(self, header):
        normalized = dict(header)

        aliases = {
            "Center_1": ["Center_1", "center_1", "BeamCenterX", "beam_center_x", "PONI1", "poni1"],
            "Center_2": ["Center_2", "center_2", "BeamCenterY", "beam_center_y", "PONI2", "poni2"],
            "SampleDistance": ["SampleDistance", "sample_distance", "Distance", "distance", "detector_distance", "SDD", "sdd"],
            "Wavelength": ["Wavelength", "wavelength", "lambda", "Lambda", "wave_length"],
            "PSize_1": ["PSize_1", "pixel_size_1", "PixelSize1", "PixelSizeX", "x_pixel_size", "pixel_size_x"],
            "PSize_2": ["PSize_2", "pixel_size_2", "PixelSize2", "PixelSizeY", "y_pixel_size", "pixel_size_y"],
        }

        lower_key_map = {str(key).lower(): key for key in normalized.keys()}

        for target_key, possible_keys in aliases.items():
            if target_key in normalized:
                continue

            for possible_key in possible_keys:
                real_key = lower_key_map.get(possible_key.lower())
                if real_key is not None:
                    normalized[target_key] = normalized[real_key]
                    break

        return normalized

    def update_header_preview(self):
        header = self.geometry_header()

        priority_keys = [
            "Instrument",
            "Geometry",
            "SampleDistance",
            "Wavelength",
            "PSize_1",
            "PSize_2",
            "Center_1",
            "Center_2",
            "SourceFile",
            "SourceDataset",
            "SourceFrameCount",
        ]

        lines = []

        for key in priority_keys:
            if key in header:
                lines.append(f"{key}: {header[key]}")
                source_key = f"{key}_source"
                if source_key in header:
                    lines.append(f"  ↳ from {header[source_key]}")

        remaining_keys = [
            key for key in sorted(header.keys())
            if key not in priority_keys and not key.endswith("_source")
        ]

        if remaining_keys:
            lines.append("")
            lines.append("Other header entries:")
            for key in remaining_keys:
                lines.append(f"{key}: {header[key]}")

        self.header_text.setPlainText("\n".join(lines))

    def update_previews(self):
        if self.current_file_path is None or self.current_frame_count <= 0:
            self.start_preview.draw_empty()
            self.end_preview.draw_empty()
            return

        start_idx = self.start_frame_spin.value() - 1
        end_idx = self.end_frame_spin.value() - 1

        try:
            vmin, vmax = self.current_contrast_limits()
            self.start_preview.draw_image(self.read_frame(start_idx), vmin=vmin, vmax=vmax)
            self.end_preview.draw_image(self.read_frame(end_idx), vmin=vmin, vmax=vmax)
        except Exception as exc:
            QMessageBox.warning(self, "Preview", f"Unable to preview frames:\n{exc}")

    def read_frame(self, frame_index):
        if self.current_file_path is None:
            raise RuntimeError("No source file selected.")

        suffix = self.current_file_path.suffix.lower()

        if suffix in (".h5", ".hdf5"):
            return self.read_h5_frame(frame_index)

        if suffix == ".edf":
            return self.read_edf_frame(frame_index)

        raise RuntimeError("Unsupported source format.")

    def read_h5_frame(self, frame_index):
        if h5py is None:
            raise RuntimeError("h5py is not installed.")

        if not self.current_h5_dataset_path:
            raise RuntimeError("No H5 dataset selected.")

        with h5py.File(self.current_file_path, "r") as h5:
            dataset = h5[self.current_h5_dataset_path]
            image = np.asarray(dataset[frame_index])

        return self.squeeze_image(image)

    def read_edf_frame(self, frame_index):
        if fabio is None:
            raise RuntimeError("fabio is not installed.")

        edf = fabio.open(str(self.current_file_path))

        if int(getattr(edf, "nframes", 1) or 1) > 1:
            frame = edf.getframe(frame_index)
            image = np.asarray(frame.data)
        else:
            image = np.asarray(edf.data)

        return self.squeeze_image(image)

    def squeeze_image(self, image):
        image = np.asarray(image)
        image = np.squeeze(image)

        if image.ndim != 2:
            raise RuntimeError(f"Frame is not a 2D image after squeeze: shape={image.shape}")

        return image

    def export_frames(self):
        if self.current_file_path is None:
            QMessageBox.warning(self, "Export", "Select a multifile first.")
            return

        start_frame = self.start_frame_spin.value()
        end_frame = self.end_frame_spin.value()

        if end_frame < start_frame:
            QMessageBox.warning(self, "Export", "End frame must be greater than or equal to start frame.")
            return

        output_folder = self.current_file_path.with_suffix("")
        output_folder.mkdir(parents=True, exist_ok=True)

        output_format = self.output_format_combo.currentText().lower()
        header = self.geometry_header()

        exported = 0

        try:
            for frame_number in range(start_frame, end_frame + 1):
                image = self.read_frame(frame_number - 1)

                if output_format == "h5":
                    self.save_single_h5(output_folder, frame_number, image, header)
                elif output_format == "edf":
                    self.save_single_edf(output_folder, frame_number, image, header)

                exported += 1

        except Exception as exc:
            QMessageBox.warning(self, "Export", f"Export stopped after {exported} frame(s):\n{exc}")
            return

        QMessageBox.information(
            self,
            "Export",
            f"Exported {exported} frame(s) to:\n{output_folder}",
        )

    def output_stem(self, frame_number):
        stem = self.current_file_path.stem if self.current_file_path is not None else "frame"
        return f"{stem}_frame{frame_number:04d}"

    def save_single_h5(self, output_folder, frame_number, image, header):
        if h5py is None:
            raise RuntimeError("h5py is not installed.")

        output_path = output_folder / f"{self.output_stem(frame_number)}.h5"

        with h5py.File(output_path, "w") as h5:
            image = np.where(np.asarray(image, dtype=float) > 1e9, np.nan, image)
            dataset = h5.create_dataset("/entry/data", data=image, compression="gzip")

            for key, value in header.items():
                safe_key = str(key).replace("/", "_")
                h5.attrs[safe_key] = str(value)
                dataset.attrs[safe_key] = str(value)

            h5.attrs["SourceFile"] = str(self.current_file_path)
            h5.attrs["SourceFrame"] = int(frame_number)

    def save_single_edf(self, output_folder, frame_number, image, header):
        if fabio is None:
            raise RuntimeError("fabio is not installed.")

        output_path = output_folder / f"{self.output_stem(frame_number)}.edf"
        image = np.where(np.asarray(image, dtype=float) > 1e9, np.nan, image)

        edf_header = {str(key): str(value) for key, value in header.items()}
        edf_header["SourceFile"] = str(self.current_file_path)
        edf_header["SourceFrame"] = str(frame_number)

        edf_image = fabio.edfimage.edfimage(data=image, header=edf_header)
        edf_image.write(str(output_path))
