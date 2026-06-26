import json
from pathlib import Path

import numpy as np

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from tabs.instrument_presets import (
    ID13_DEFAULT_CENTER_X,
    ID13_DEFAULT_CENTER_Y,
    ID13_DEFAULT_DISTANCE_M,
    ID13_DEFAULT_PIXEL_MM,
    ID13_DEFAULT_WAVELENGTH_M,
)


GEOMETRY_FILE = Path.home() / ".lrphoton" / "line_geometries.json"
SELECT_GEOMETRY_ITEM = "Beamline"
ADD_GEOMETRY_ITEM = "+ Ajouter une ligne..."

LINE_GEOMETRY_SELECTOR_STYLE = """
    QComboBox {
        background-color: #dddddd;
        color: #222222;
        border: 0px;
        border-radius: 7px;
        padding: 3px 24px 3px 8px;
        min-height: 20px;
        font-size: 14px;
    }
    QComboBox:hover {
        background-color: #d4d4d4;
    }
    QComboBox::drop-down {
        border: 0px;
        width: 24px;
    }
    QPushButton {
        background-color: #dddddd;
        color: #111111;
        border: 0px;
        border-radius: 7px;
        padding: 3px 12px;
        min-height: 20px;
        font-size: 14px;
    }
    QPushButton:hover {
        background-color: #d4d4d4;
    }
"""


def default_center_text(size):
    return f"{(float(size) - 1.0) / 2.0:.1f}"


def default_line_geometries():
    return {
        "SALS default": {
            "name": "SALS default",
            "center_x": default_center_text(796),
            "center_y": default_center_text(796),
            "pixel_x_m": "5,5e-6",
            "pixel_y_m": "5,5e-6",
            "distance_m": "0,00477",
            "wavelength_m": "632,8e-9",
        },
        "XENOCS": {
            "name": "XENOCS",
            "center_x": "0",
            "center_y": "0",
            "pixel_x_m": "75e-6",
            "pixel_y_m": "75e-6",
            "distance_m": "0",
            "wavelength_m": "0",
        },
        "ID02": {
            "name": "ID02",
            "center_x": "914,4",
            "center_y": "996,5",
            "pixel_x_m": "75e-6",
            "pixel_y_m": "75e-6",
            "distance_m": "10,0002",
            "wavelength_m": "1,01402e-10",
        },
        "ID13": {
            "name": "ID13",
            "center_x": f"{ID13_DEFAULT_CENTER_X:.10g}",
            "center_y": f"{ID13_DEFAULT_CENTER_Y:.10g}",
            "pixel_x_m": f"{ID13_DEFAULT_PIXEL_MM * 1e-3:.10g}",
            "pixel_y_m": f"{ID13_DEFAULT_PIXEL_MM * 1e-3:.10g}",
            "distance_m": f"{ID13_DEFAULT_DISTANCE_M:.10g}",
            "wavelength_m": f"{ID13_DEFAULT_WAVELENGTH_M:.10g}",
        },
    }


def load_line_geometries():
    defaults = default_line_geometries()
    try:
        if not GEOMETRY_FILE.exists():
            save_line_geometries(defaults)
            return defaults
        data = json.loads(GEOMETRY_FILE.read_text(encoding="utf-8"))
        geometries = dict(defaults)
        for item in data.get("geometries", []):
            geometry = normalized_line_geometry(item)
            if geometry["name"]:
                geometries[geometry["name"]] = geometry
        return geometries
    except Exception:
        return defaults


def save_line_geometries(geometries):
    GEOMETRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {"geometries": [geometries[name] for name in sorted(geometries)]}
    GEOMETRY_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def normalized_line_geometry(item):
    return {
        "name": str(item.get("name", "")).strip(),
        "center_x": str(item.get("center_x", "")).strip(),
        "center_y": str(item.get("center_y", "")).strip(),
        "pixel_x_m": str(item.get("pixel_x_m", "")).strip(),
        "pixel_y_m": str(item.get("pixel_y_m", "")).strip(),
        "distance_m": str(item.get("distance_m", "")).strip(),
        "wavelength_m": str(item.get("wavelength_m", "")).strip(),
    }


def number_value(text):
    text = str(text).strip().replace(",", ".")
    if not text:
        return 0.0
    return float(text)


def _optional_number_text(value):
    if value is None:
        return ""
    return f"{float(value):.10g}"


def _merge_geometry(base, override):
    merged = normalized_line_geometry(base or {})
    for key, value in (override or {}).items():
        if key not in merged:
            continue
        text = str(value).strip()
        if text:
            merged[key] = text
    if not merged["name"]:
        merged["name"] = str((base or {}).get("name") or (override or {}).get("name") or "").strip()
    return merged


def read_poni_line_geometry(path, fallback=None, name="PONI"):
    values = {}
    detector_config = {}
    path = Path(path)
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, value = line.split(":", 1)
        elif "=" in line:
            key, value = line.split("=", 1)
        else:
            continue

        key = key.strip().lower()
        value = value.strip()
        if key == "detector_config":
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    detector_config = {str(k).lower(): v for k, v in parsed.items()}
            except (TypeError, json.JSONDecodeError):
                pass
            continue

        try:
            values[key] = number_value(value)
        except (TypeError, ValueError):
            continue

    def get_value(*keys):
        for key in keys:
            if key in values:
                return values[key]
        return None

    def get_detector_value(*keys):
        for key in keys:
            if key in detector_config:
                try:
                    return float(detector_config[key])
                except (TypeError, ValueError):
                    return None
        return None

    distance_m = get_value("distance", "sampledistance", "detectordistance", "dist")
    poni1_m = get_value("poni1")
    poni2_m = get_value("poni2")
    pixel1_m = get_value("pixelsize1", "pixelsize_1", "pixelsizey", "pixel_size_y", "pixel1")
    pixel2_m = get_value("pixelsize2", "pixelsize_2", "pixelsizex", "pixel_size_x", "pixel2")
    wavelength_m = get_value("wavelength", "wave_length", "lambda")

    if pixel1_m is None:
        pixel1_m = get_detector_value("pixel1", "pixelsize1", "pixel_size_y")
    if pixel2_m is None:
        pixel2_m = get_detector_value("pixel2", "pixelsize2", "pixel_size_x")

    pyfai_center = None
    try:
        import pyFAI

        integrator = pyFAI.load(str(path.expanduser()))
        detector = getattr(integrator, "detector", None)
        if detector is not None:
            pixel1_m = pixel1_m or getattr(detector, "pixel1", None)
            pixel2_m = pixel2_m or getattr(detector, "pixel2", None)
            pyfai_center = _pyfai_beam_center_pixels(integrator)
    except Exception:
        pass

    missing = [
        label for label, value in {
            "Poni1": poni1_m,
            "Poni2": poni2_m,
            "PixelSize1": pixel1_m,
            "PixelSize2": pixel2_m,
        }.items()
        if value is None or value == 0
    ]
    if missing:
        raise ValueError(f"Missing PONI geometry field(s): {', '.join(missing)}")

    poni_geometry = {
        "name": name,
        "center_x": _optional_number_text((pyfai_center[0] + 1.0) if pyfai_center is not None else (poni2_m / pixel2_m) + 0.5),
        "center_y": _optional_number_text((pyfai_center[1] + 1.0) if pyfai_center is not None else (poni1_m / pixel1_m) + 0.5),
        "pixel_x_m": _optional_number_text(pixel2_m),
        "pixel_y_m": _optional_number_text(pixel1_m),
        "distance_m": _optional_number_text(distance_m),
        "wavelength_m": _optional_number_text(wavelength_m),
    }
    return _merge_geometry(fallback or {}, poni_geometry)


def _pyfai_beam_center_pixels(integrator):
    detector = getattr(integrator, "detector", None)
    shape = getattr(detector, "shape", None) or getattr(detector, "max_shape", None)
    if not shape or len(shape) != 2:
        return None

    shape = tuple(int(value) for value in shape)
    if shape[0] <= 1 or shape[1] <= 1:
        return None

    try:
        q_map = np.asarray(integrator.center_array(shape, unit="q_nm^-1"), dtype=float)
    except Exception:
        q_map = np.asarray(integrator.qArray(shape), dtype=float)

    if q_map.shape != shape or not np.any(np.isfinite(q_map)):
        return None

    y_index, x_index = np.unravel_index(np.nanargmin(q_map), q_map.shape)

    def subpixel_offset(values):
        left, center, right = (float(value) for value in values)
        denom = left - 2.0 * center + right
        if not np.isfinite(denom) or abs(denom) < 1e-30:
            return 0.0
        offset = 0.5 * (left - right) / denom
        return float(np.clip(offset, -0.5, 0.5))

    x_offset = 0.0
    if 0 < x_index < q_map.shape[1] - 1:
        x_values = q_map[y_index, x_index - 1:x_index + 2]
        if np.all(np.isfinite(x_values)):
            x_offset = subpixel_offset(x_values)

    y_offset = 0.0
    if 0 < y_index < q_map.shape[0] - 1:
        y_values = q_map[y_index - 1:y_index + 2, x_index]
        if np.all(np.isfinite(y_values)):
            y_offset = subpixel_offset(y_values)

    return float(x_index) + x_offset, float(y_index) + y_offset


def parse_header_text(header_text):
    if header_text is None:
        return {}

    if isinstance(header_text, bytes):
        header_text = header_text.decode("latin-1", errors="ignore")

    mapping = {}
    for raw_line in str(header_text).splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", ";")):
            continue

        if "=" in line:
            key, value = line.split("=", 1)
        elif ":" in line:
            key, value = line.split(":", 1)
        else:
            continue

        key = key.strip()
        value = value.strip()
        if key:
            mapping[key] = value

    return mapping


def header_number(header, *keys):
    if header is None:
        return None

    if isinstance(header, (str, bytes)):
        header = parse_header_text(header)
    elif not isinstance(header, dict):
        try:
            header = dict(header)
        except Exception:
            return None

    normalized = {str(k): str(v) for k, v in header.items()}
    for key in keys:
        if key in normalized:
            try:
                return number_value(normalized[key])
            except (TypeError, ValueError):
                return None
    return None


def header_to_line_geometry(header, fallback=None, name="XENOCS"):
    fallback = dict(fallback or {})
    if isinstance(header, (str, bytes)):
        header = parse_header_text(header)
    elif not isinstance(header, dict):
        try:
            header = dict(header)
        except Exception:
            header = {}

    cx = header_number(header, "Center_1", "center_1", "Center X", "CenterX", "center_x", "BeamCenterX", "Beam_x", "beam_x")
    cy = header_number(header, "Center_2", "center_2", "Center Y", "CenterY", "center_y", "BeamCenterY", "Beam_y", "beam_y")
    px = header_number(header, "PSize_1", "psize_1", "PSize_X", "PixelSizeX", "pixel_size_x", "x_pixel_size", "pixel_x")
    py = header_number(header, "PSize_2", "psize_2", "PSize_Y", "PixelSizeY", "pixel_size_y", "y_pixel_size", "pixel_y")
    dist = header_number(header, "SampleDistance", "sampledistance", "sample_distance", "Distance", "DetectorDistance", "detector_distance")
    wavelength = header_number(header, "WaveLength", "Wavelength", "wavelength", "Lambda", "lambda")

    geometry = normalized_line_geometry({
        "name": name,
        "center_x": fallback.get("center_x", ""),
        "center_y": fallback.get("center_y", ""),
        "pixel_x_m": fallback.get("pixel_x_m", ""),
        "pixel_y_m": fallback.get("pixel_y_m", ""),
        "distance_m": fallback.get("distance_m", ""),
        "wavelength_m": fallback.get("wavelength_m", ""),
    })
    if cx is not None:
        geometry["center_x"] = f"{cx:.10g}"
    if cy is not None:
        geometry["center_y"] = f"{cy:.10g}"
    if px is not None:
        geometry["pixel_x_m"] = f"{px:.10g}" if px < 1e-3 else f"{px * 1e-3:.10g}"
    if py is not None:
        geometry["pixel_y_m"] = f"{py:.10g}" if py < 1e-3 else f"{py * 1e-3:.10g}"
    if dist is not None:
        geometry["distance_m"] = f"{dist:.10g}"
    if wavelength is not None:
        geometry["wavelength_m"] = f"{wavelength:.10g}" if wavelength < 1e-6 else f"{wavelength * 1e-10:.10g}"
    return geometry


def line_geometry_to_lrphoton(geometry):
    return {
        "name": geometry.get("name", ""),
        "xc": number_value(geometry.get("center_x")),
        "yc": number_value(geometry.get("center_y")),
        "distance_m": number_value(geometry.get("distance_m")),
        "pixel_x_mm": number_value(geometry.get("pixel_x_m")) * 1000.0,
        "pixel_y_mm": number_value(geometry.get("pixel_y_m")) * 1000.0,
        "wavelength_a": number_value(geometry.get("wavelength_m")) * 1e10,
    }


class LineGeometrySelector(QWidget):
    geometry_selected = Signal(str, dict)

    def __init__(self, parent=None, current_name="SALS default"):
        super().__init__(parent)
        self.context_owner = parent
        self.geometries = load_line_geometries()
        self.current_name = current_name if current_name in self.geometries else next(iter(self.geometries))
        self.has_explicit_selection = False
        self.context_geometry_provider = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        selector_layout = QHBoxLayout()
        selector_layout.setContentsMargins(0, 0, 0, 0)
        selector_layout.setSpacing(4)
        self.combo = QComboBox()
        self.combo.currentTextChanged.connect(self.on_combo_changed)
        self.edit_button = QPushButton("Edit")
        self.edit_button.clicked.connect(self.open_editor)
        selector_layout.addWidget(self.combo, 1)
        selector_layout.addWidget(self.edit_button, 0)
        layout.addLayout(selector_layout)

        poni_layout = QHBoxLayout()
        poni_layout.setContentsMargins(0, 0, 0, 0)
        poni_layout.setSpacing(4)
        poni_layout.addWidget(QLabel("PONI:"))
        self.poni_path = QLineEdit()
        self.poni_path.setPlaceholderText("Optional .poni path")
        self.poni_path.setToolTip("Optional pyFAI .poni file. Its values complete and override the selected beamline/header geometry.")
        self.poni_path.editingFinished.connect(self.on_poni_changed)
        self.poni_button = QPushButton("Browse")
        self.poni_button.clicked.connect(self.choose_poni_file)
        poni_layout.addWidget(self.poni_path, 1)
        poni_layout.addWidget(self.poni_button, 0)
        layout.addLayout(poni_layout)

        self.setStyleSheet(LINE_GEOMETRY_SELECTOR_STYLE)
        self.refresh()

    def refresh(self):
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.setPlaceholderText(SELECT_GEOMETRY_ITEM)
        for name in sorted(self.geometries):
            self.combo.addItem(name)
        self.combo.addItem(ADD_GEOMETRY_ITEM)
        if self.has_explicit_selection:
            self.combo.setCurrentText(self.current_name)
        else:
            self.combo.setCurrentIndex(-1)
        self.combo.blockSignals(False)

    def current_geometry(self):
        return self.geometry_for_name(self.current_name)

    def set_context_geometry_provider(self, provider):
        self.context_geometry_provider = provider

    def geometry_for_name(self, name):
        base = normalized_line_geometry(self.geometries.get(name, {}))
        geometry = base
        if self.context_geometry_provider is not None:
            try:
                provided = self.context_geometry_provider(name, base)
                if provided:
                    geometry = _merge_geometry(geometry, normalized_line_geometry(provided))
            except Exception:
                pass
        parent = self.context_owner or self.parent()
        for attr in ("current_header_for_line_geometry", "headers", "current_header", "header"):
            header = getattr(parent, attr, None)
            if isinstance(header, dict) and header:
                geometry = _merge_geometry(geometry, header_to_line_geometry(header, geometry, name))
                break

        poni_path = self.selected_poni_path()
        if poni_path is not None:
            geometry = read_poni_line_geometry(poni_path, geometry, name)
        return geometry

    def selected_poni_path(self):
        text = self.poni_path.text().strip()
        if not text:
            return None
        path = Path(text).expanduser()
        if not path.exists() or not path.is_file():
            return None
        return path

    def choose_poni_file(self):
        start_folder = Path.home()
        current_path = self.selected_poni_path()
        if current_path is not None:
            start_folder = current_path.parent
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose PONI file",
            str(start_folder),
            "PONI files (*.poni);;All files (*)",
        )
        if path:
            self.poni_path.setText(path)
            self.on_poni_changed()

    def on_poni_changed(self):
        if not self.poni_path.text().strip():
            self.geometry_selected.emit(self.current_name, self.geometry_for_name(self.current_name))
            return
        path = self.selected_poni_path()
        if path is None:
            return
        try:
            geometry = self.geometry_for_name(self.current_name)
        except Exception as error:
            QMessageBox.warning(self, "PONI reading error", f"{path.name}\n\n{error}")
            return
        self.geometry_selected.emit(self.current_name, geometry)

    def set_current_name(self, name, explicit=False):
        if name in self.geometries:
            self.current_name = name
            self.has_explicit_selection = bool(explicit)
            self.refresh()

    def on_combo_changed(self, name):
        if not name:
            return
        if name == SELECT_GEOMETRY_ITEM:
            return
        if name == ADD_GEOMETRY_ITEM:
            self.open_editor(new_geometry=True)
            return
        self.current_name = name
        self.has_explicit_selection = True
        self.geometry_selected.emit(name, self.geometry_for_name(name))

    def open_editor(self, new_geometry=False):
        source = dict(self.current_geometry())
        if new_geometry:
            source["name"] = self.unique_name("Nouvelle ligne")

        dialog = QDialog(self)
        dialog.setWindowTitle("Configuration ligne")
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        edits = {}
        fields = [
            ("name", "Nom"),
            ("center_x", "Center X"),
            ("center_y", "Center Y"),
            ("pixel_x_m", "Pixel X (m)"),
            ("pixel_y_m", "Pixel Y (m)"),
            ("distance_m", "Distance sample-detector (m)"),
            ("wavelength_m", "Wavelength (m)"),
        ]
        for key, label in fields:
            edit = QLineEdit(str(source.get(key, "")))
            edits[key] = edit
            form.addRow(label, edit)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() != QDialog.Accepted:
            self.refresh()
            return

        geometry = normalized_line_geometry({key: edit.text() for key, edit in edits.items()})
        if not geometry["name"]:
            self.refresh()
            return
        self.geometries[geometry["name"]] = geometry
        self.current_name = geometry["name"]
        self.has_explicit_selection = True
        save_line_geometries(self.geometries)
        self.refresh()
        self.geometry_selected.emit(self.current_name, geometry)

    def unique_name(self, base_name):
        if base_name not in self.geometries:
            return base_name
        index = 2
        while f"{base_name} {index}" in self.geometries:
            index += 1
        return f"{base_name} {index}"
