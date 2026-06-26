from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np

from tabs.cave_tab import parse_edf_header, read_edf_file, set_h5_attr, write_edf_file
from tabs.radial_tab import read_image_file


PONI_KEY_ALIASES = {
    "PixelSize1": ["PixelSize1", "PixelSize_1", "PixelSizeX", "pixel_size_x"],
    "PixelSize2": ["PixelSize2", "PixelSize_2", "PixelSizeY", "pixel_size_y"],
    "Distance": ["Distance", "SampleDistance", "DetectorDistance", "dist"],
    "Poni1": ["Poni1", "CenterX", "BeamCenterX", "center_x"],
    "Poni2": ["Poni2", "CenterY", "BeamCenterY", "center_y"],
    "Wavelength": ["Wavelength", "WaveLength", "wavelength", "lambda"],
}


def parse_poni_file(path) -> Dict[str, float]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Poni file not found: {path}")

    values = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line and "=" not in line:
            continue
        sep = ":" if ":" in line else "="
        key, value = line.split(sep, 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            continue
        if key in PONI_KEY_ALIASES["PixelSize1"]:
            values["pixel_size_x_m"] = float(value)
        elif key in PONI_KEY_ALIASES["PixelSize2"]:
            values["pixel_size_y_m"] = float(value)
        elif key in PONI_KEY_ALIASES["Distance"]:
            values["distance_m"] = float(value)
        elif key in PONI_KEY_ALIASES["Poni1"]:
            values["center_x_px"] = float(value)
        elif key in PONI_KEY_ALIASES["Poni2"]:
            values["center_y_px"] = float(value)
        elif key in PONI_KEY_ALIASES["Wavelength"]:
            values["wavelength_m"] = float(value)

    if "center_x_px" not in values or "center_y_px" not in values:
        raise ValueError("Poni file does not contain Poni1/Poni2 center coordinates.")
    if "distance_m" not in values:
        raise ValueError("Poni file does not contain a distance.")
    if "pixel_size_x_m" not in values or "pixel_size_y_m" not in values:
        raise ValueError("Poni file does not contain pixel size values.")
    if "wavelength_m" not in values:
        raise ValueError("Poni file does not contain a wavelength.")

    return values


def _build_header_from_poni(metadata: Dict[str, float], source_path: Optional[Path] = None) -> Dict[str, str]:
    header = {
        "Center_1": str(float(metadata["center_x_px"])),
        "Center_2": str(float(metadata["center_y_px"])),
        "Distance": str(float(metadata["distance_m"])),
        "PixelSizeX": str(float(metadata["pixel_size_x_m"])),
        "PixelSizeY": str(float(metadata["pixel_size_y_m"])),
        "Wavelength": str(float(metadata["wavelength_m"])),
    }
    if source_path is not None:
        header["PoniSource"] = str(source_path)
    return header


def _write_edf_with_poni_metadata(path: Path, image: np.ndarray, header: Dict[str, str]) -> Path:
    raw_header = "{\n"
    raw_header += "DataType = FloatValue ;\n"
    raw_header += "ByteOrder = LowByteFirst ;\n"
    raw_header += f"Dim_1 = {image.shape[1]} ;\n"
    raw_header += f"Dim_2 = {image.shape[0]} ;\n"
    raw_header += "EDF_HeaderSize = 1024 ;\n"
    raw_header += "}\n"

    for key, value in header.items():
        raw_header = raw_header.replace("}\n", f"\n{key} = {value} ;\n}}\n", 1)

    write_edf_file(str(path), image, raw_header, "LowByteFirst")
    return path


def _write_h5_with_poni_metadata(path: Path, image: np.ndarray, header: Dict[str, str]) -> Path:
    with h5py.File(path, "w") as h5:
        h5.create_dataset("data", data=image.astype(np.float32), compression="gzip")
        for key, value in header.items():
            try:
                h5.attrs[str(key)] = value
            except TypeError:
                h5.attrs[str(key)] = str(value)
    return path


def apply_poni_to_files(poni_path, target_paths, output_dir=None, output_format=".edf"):
    metadata = parse_poni_file(poni_path)
    header = _build_header_from_poni(metadata, Path(poni_path).resolve())

    results = []
    for target in target_paths:
        target_path = Path(target)
        if not target_path.exists():
            raise FileNotFoundError(f"Target file not found: {target_path}")

        image, existing_header = read_image_file(target_path)
        if existing_header is None:
            existing_header = {}

        combined_header = dict(existing_header)
        combined_header.update(header)

        if output_dir is None:
            output_path = target_path.with_suffix(output_format)
        else:
            output_path = Path(output_dir) / target_path.name
            output_path = output_path.with_suffix(output_format)

        if output_format == ".edf":
            _write_edf_with_poni_metadata(output_path, image, combined_header)
        else:
            _write_h5_with_poni_metadata(output_path, image, combined_header)

        results.append(output_path)

    return results
