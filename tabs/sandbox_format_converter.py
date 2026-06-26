from pathlib import Path
import json
import re

import h5py
import numpy as np

from tabs.cave_tab import write_edf_file, parse_edf_header, read_edf_file
from tabs.radial_tab import read_image_file


def _ensure_suffix(path, suffix):
    path = Path(path)
    if path.suffix.lower() == suffix.lower():
        return path
    return path.with_suffix(suffix)


def _load_image_and_header(source_path):
    image, header = read_image_file(source_path)
    return image, header


def convert_image_to_format(source_path, output_path, output_format=None):
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    output_format = (output_format or source_path.suffix).lower()
    if output_format not in {".edf", ".h5", ".hdf5"}:
        raise ValueError("Output format must be .edf, .h5 or .hdf5")

    image, header = _load_image_and_header(source_path)
    output_path = Path(output_path)
    if not output_path.suffix:
        output_path = output_path.with_suffix(".edf" if output_format == ".edf" else ".h5")

    if output_format == ".edf":
        return _write_edf_with_header(source_path, output_path, image, header)

    return _write_h5_with_header(source_path, output_path, image, header)


def _write_edf_with_header(source_path, output_path, image, header):
    source_path = Path(source_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_header_text = ""
    if source_path.suffix.lower() == ".edf":
        try:
            _, _, raw_header_text, byte_order, _ = read_edf_file(source_path)
        except Exception:
            raw_header_text = ""
    else:
        raw_header_text = ""

    if not raw_header_text:
        raw_header_text = "{\n"
        raw_header_text += "DataType = FloatValue ;\n"
        raw_header_text += "ByteOrder = LowByteFirst ;\n"
        raw_header_text += f"Dim_1 = {image.shape[1]} ;\n"
        raw_header_text += f"Dim_2 = {image.shape[0]} ;\n"
        raw_header_text += "EDF_HeaderSize = 1024 ;\n"
        raw_header_text += "}\n"

    if not header:
        header = {}

    # Preserve the most useful metadata in the EDF header.
    for key, value in header.items():
        if key in {"Dim_1", "Dim_2", "DataType", "ByteOrder", "EDF_HeaderSize", "Size", "EDF_BinarySize"}:
            continue
        raw_header_text = raw_header_text.replace("}\n", f"\n{key} = {value} ;\n}}\n", 1)

    write_edf_file(str(output_path), image, raw_header_text, "LowByteFirst")
    return output_path


def _write_h5_with_header(source_path, output_path, image, header):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as h5:
        h5.create_dataset("data", data=image.astype(np.float32), compression="gzip")
        for key, value in header.items():
            if key in {"Dataset", "Shape", "Dtype", "Frame axis", "Displayed frame", "Number of frames"}:
                continue
            try:
                h5.attrs[str(key)] = value
            except TypeError:
                h5.attrs[str(key)] = str(value)

        h5.attrs["source_file"] = str(source_path)
        h5.attrs["converted_from"] = source_path.suffix.lower()
    return output_path
