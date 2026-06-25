import json
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QLineEdit, QPushButton, QVBoxLayout, QWidget

from tabs.instrument_presets import (
    ID13_DEFAULT_CENTER_X,
    ID13_DEFAULT_CENTER_Y,
    ID13_DEFAULT_DISTANCE_M,
    ID13_DEFAULT_PIXEL_MM,
    ID13_DEFAULT_WAVELENGTH_M,
)


GEOMETRY_FILE = Path.home() / ".lrphoton" / "line_geometries.json"
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


def header_number(header, *keys):
    for key in keys:
        if key in header:
            try:
                return number_value(header[key])
            except (TypeError, ValueError):
                return None
    return None


def header_to_line_geometry(header, fallback=None, name="XENOCS"):
    fallback = dict(fallback or {})
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
        self.context_geometry_provider = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.combo = QComboBox()
        self.combo.currentTextChanged.connect(self.on_combo_changed)
        self.edit_button = QPushButton("Edit")
        self.edit_button.clicked.connect(self.open_editor)
        layout.addWidget(self.combo, 1)
        layout.addWidget(self.edit_button, 0)
        self.setStyleSheet(LINE_GEOMETRY_SELECTOR_STYLE)
        self.refresh()

    def refresh(self):
        self.combo.blockSignals(True)
        self.combo.clear()
        for name in sorted(self.geometries):
            self.combo.addItem(name)
        self.combo.addItem(ADD_GEOMETRY_ITEM)
        self.combo.setCurrentText(self.current_name)
        self.combo.blockSignals(False)

    def current_geometry(self):
        return self.geometry_for_name(self.current_name)

    def set_context_geometry_provider(self, provider):
        self.context_geometry_provider = provider

    def geometry_for_name(self, name):
        base = self.geometries.get(name, {})
        if self.context_geometry_provider is not None:
            try:
                geometry = self.context_geometry_provider(name, base)
                if geometry:
                    return normalized_line_geometry(geometry)
            except Exception:
                pass
        parent = self.context_owner or self.parent()
        for attr in ("current_header_for_line_geometry", "headers", "current_header", "header"):
            header = getattr(parent, attr, None)
            if isinstance(header, dict) and header:
                return header_to_line_geometry(header, base, name)
        return base

    def set_current_name(self, name):
        if name in self.geometries:
            self.current_name = name
            self.refresh()

    def on_combo_changed(self, name):
        if not name:
            return
        if name == ADD_GEOMETRY_ITEM:
            self.open_editor(new_geometry=True)
            return
        self.current_name = name
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
