import fnmatch
import json
import re
from pathlib import Path

import h5py
import numpy as np

from PySide6.QtCore import Qt, QEvent, QPoint, QSize, QCoreApplication, Signal, QTimer
from PySide6.QtWidgets import (
    QWidget,
    QDialog,
    QAbstractItemView,
    QAbstractSpinBox,
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
    QMessageBox,
    QSlider,
    QComboBox,
    QListWidget,
    QListWidgetItem,
    QButtonGroup,
    QLineEdit,
    QScrollArea,
    QProgressBar,
    QSizePolicy,
    QSplitter,
    QMenu,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.path import Path as MplPath
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import Rectangle as MplRectangle

from .instrument_presets import (
    ID13_DEFAULT_CENTER_X,
    ID13_DEFAULT_CENTER_Y,
    ID13_DEFAULT_DISTANCE_M,
    ID13_DEFAULT_PIXEL_MM,
    ID13_DEFAULT_WAVELENGTH_A,
)
from .ui_style import (
    ACTION_BUTTON_STYLE,
    BLOCK_SPACING,
    COMPACT_COMBO_STYLE,
    FILE_BROWSER_WIDTH,
    FlexibleDoubleSpinBox as QDoubleSpinBox,
    FRAME_BUTTON_WIDTH,
    FRAME_COUNTER_WIDTH,
    FRAME_NAV_SPACING,
    FRAME_SPIN_WIDTH,
    GROUP_BOX_STYLE,
    GROUP_BOX_MARGINS,
    PAGE_MARGINS,
    PANEL_MARGINS,
    constrain_image_axes,
    emojiize_matplotlib_toolbar,
    make_matplotlib_toolbar_block,
    normalize_decimal_text,
    style_q_geometry_buttons,
)
from .file_ratings import install_file_rating_menu, is_file_rated_up, set_item_file_path, should_hide_file_in_browser
from .line_geometry import LineGeometrySelector, line_geometry_to_lrphoton


CAVE_MASK_PRESET_VERSION = 1


def cave_mask_preset_dir():
    path = Path.home() / ".lrphoton" / "cave_masks"
    path.mkdir(parents=True, exist_ok=True)
    return path


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


# Helper to infer EDF header size if not explicitly present
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
    image, header, raw_header_text, byte_order, _n_frames = read_edf_frame(filename, 0)
    return image, header, raw_header_text, byte_order


def read_edf_header_only(filename: str):
    filename = Path(filename)
    with open(filename, "rb") as file:
        first = file.read(8192).decode("latin-1", errors="ignore")
        header_size = infer_edf_header_size(first)
        file.seek(0)
        raw_header_text = file.read(header_size).decode("latin-1", errors="ignore")
    return parse_edf_header(raw_header_text)


def read_edf_frame(filename: str, frame_index: int = 0):
    filename = Path(filename)
    frame_index = max(0, int(frame_index))

    selected_image = None
    selected_header = None
    selected_raw_header_text = ""
    selected_byte_order = "LowByteFirst"
    first_header = None
    first_raw_header_text = ""
    first_byte_order = "LowByteFirst"
    total_frames = 0
    offset = 0
    file_size = filename.stat().st_size

    with open(filename, "rb") as file:
        while offset < file_size:
            file.seek(offset)
            first = file.read(8192).decode("latin-1", errors="ignore")
            if not first.strip():
                break

            try:
                header_size = infer_edf_header_size(first)
            except ValueError:
                if total_frames:
                    break
                raise ValueError("EDF_HeaderSize not found in EDF header.")
            file.seek(offset)
            raw_header_bytes = file.read(header_size)
            raw_header_text = raw_header_bytes.decode("latin-1", errors="ignore")
            header = parse_edf_header(raw_header_text)

            data_type = header.get("DataType", "FloatValue")
            byte_order = header.get("ByteOrder", "LowByteFirst")
            dim_1 = int(float(header["Dim_1"]))
            dim_2 = int(float(header["Dim_2"]))
            block_frame_count = 1
            for frame_key in ["Dim_3", "Dim_4", "NumberOfFrames", "NFrames", "NumFrames"]:
                if frame_key in header:
                    try:
                        block_frame_count = max(1, int(float(header[frame_key])))
                        break
                    except (TypeError, ValueError):
                        continue

            dtype = np.dtype(edf_dtype_to_numpy(data_type))
            dtype = dtype.newbyteorder(">" if byte_order.lower() == "highbytefirst" else "<")
            pixels_per_frame = dim_1 * dim_2
            bytes_per_frame = pixels_per_frame * dtype.itemsize
            data_offset = file.tell()

            if first_header is None:
                first_header = header
                first_raw_header_text = raw_header_text
                first_byte_order = byte_order

            block_start = total_frames
            block_end = block_start + block_frame_count
            if selected_image is None and block_start <= frame_index < block_end:
                local_frame = frame_index - block_start
                file.seek(data_offset + local_frame * bytes_per_frame)
                data = np.fromfile(file, dtype=dtype, count=pixels_per_frame)
                if data.size != pixels_per_frame:
                    raise ValueError(f"Incorrect EDF data size: expected {pixels_per_frame}, read {data.size}.")
                selected_image = data.reshape((dim_2, dim_1)).astype(np.float64)
                selected_header = header
                selected_raw_header_text = raw_header_text
                selected_byte_order = byte_order

            total_frames = block_end
            offset = data_offset + block_frame_count * bytes_per_frame

    if total_frames <= 0:
        raise ValueError("No image found in EDF file.")

    if selected_image is None:
        return read_edf_frame(filename, total_frames - 1)

    if selected_header is None:
        selected_header = first_header
        selected_raw_header_text = first_raw_header_text
        selected_byte_order = first_byte_order

    return selected_image, selected_header, selected_raw_header_text, selected_byte_order, total_frames


def read_edf_frames(filename: str):
    filename = Path(filename)

    frames = []
    first_header = None
    first_raw_header_text = ""
    first_byte_order = "LowByteFirst"
    offset = 0
    file_size = filename.stat().st_size

    with open(filename, "rb") as file:
        while offset < file_size:
            file.seek(offset)
            first = file.read(8192).decode("latin-1", errors="ignore")
            if not first.strip():
                break

            try:
                header_size = infer_edf_header_size(first)
            except ValueError:
                if frames:
                    break
                raise ValueError("EDF_HeaderSize not found in EDF header.")
            file.seek(offset)
            raw_header_bytes = file.read(header_size)
            raw_header_text = raw_header_bytes.decode("latin-1", errors="ignore")
            header = parse_edf_header(raw_header_text)

            data_type = header.get("DataType", "FloatValue")
            byte_order = header.get("ByteOrder", "LowByteFirst")
            dim_1 = int(float(header["Dim_1"]))
            dim_2 = int(float(header["Dim_2"]))
            frame_count = 1
            for frame_key in ["Dim_3", "Dim_4", "NumberOfFrames", "NFrames", "NumFrames"]:
                if frame_key in header:
                    try:
                        frame_count = max(1, int(float(header[frame_key])))
                        break
                    except (TypeError, ValueError):
                        continue

            dtype = np.dtype(edf_dtype_to_numpy(data_type))
            dtype = dtype.newbyteorder(">" if byte_order.lower() == "highbytefirst" else "<")

            expected_count = dim_1 * dim_2 * frame_count
            data = np.fromfile(file, dtype=dtype, count=expected_count)

            if data.size != expected_count:
                raise ValueError(f"Incorrect EDF data size: expected {expected_count}, read {data.size}.")

            block_frames = data.reshape((frame_count, dim_2, dim_1)).astype(np.float64)
            frames.extend(block_frames)

            if first_header is None:
                first_header = header
                first_raw_header_text = raw_header_text
                first_byte_order = byte_order

            offset = file.tell()

    if not frames:
        raise ValueError("No image found in EDF file.")

    return np.asarray(frames, dtype=np.float64), first_header, first_raw_header_text, first_byte_order


def add_matching_edf_center(header: dict, filename: str):
    edf_path = Path(filename).with_suffix(".edf")
    if not edf_path.exists():
        return header

    try:
        edf_header = read_edf_header_only(edf_path)
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


def update_edf_header_value(header_text: str, key: str, new_value: str) -> str:
    expression = rf"{re.escape(key)}\s*=\s*[^;]*;"
    replacement = f"{key} = {new_value} ;"

    if re.search(expression, header_text):
        return re.sub(expression, replacement, header_text, count=1)

    closing = header_text.rfind("}")
    if closing < 0:
        raise ValueError("Unable to update EDF header: closing brace not found.")

    return header_text[:closing] + f"\n{key} = {new_value} ;" + header_text[closing:]


def write_edf_file(filename: str, image: np.ndarray, raw_header_text: str, byte_order: str):
    filename = Path(filename)
    ny, nx = image.shape

    header_text = raw_header_text
    header_text = update_edf_header_value(header_text, "Dim_1", str(nx))
    header_text = update_edf_header_value(header_text, "Dim_2", str(ny))
    header_text = update_edf_header_value(header_text, "DataType", "FloatValue")
    header_text = update_edf_header_value(header_text, "Size", str(nx * ny * 4))
    header_text = update_edf_header_value(header_text, "EDF_BinarySize", str(nx * ny * 4))
    for frame_key in ["Dim_3", "Dim_4", "NumberOfFrames", "NFrames", "NumFrames"]:
        if re.search(rf"{re.escape(frame_key)}\s*=", header_text):
            header_text = update_edf_header_value(header_text, frame_key, "1")

    match = re.search(r"EDF_HeaderSize\s*=\s*(\d+)", header_text)
    header_size = int(match.group(1)) if match else 1024

    if not match:
        header_text = update_edf_header_value(header_text, "EDF_HeaderSize", str(header_size))

    header_bytes = header_text.encode("latin-1", errors="ignore")

    if len(header_bytes) > header_size:
        header_size = int(np.ceil(len(header_bytes) / 1024) * 1024)
        header_text = update_edf_header_value(header_text, "EDF_HeaderSize", str(header_size))
        header_bytes = header_text.encode("latin-1", errors="ignore")

    header_bytes = header_bytes + b" " * (header_size - len(header_bytes))

    output = image.astype(np.float32)
    output_dtype = output.dtype.newbyteorder(">" if byte_order.lower() == "highbytefirst" else "<")
    output = output.astype(output_dtype, copy=False)

    with open(filename, "wb") as file:
        file.write(header_bytes)
        file.write(output.tobytes(order="C"))


def sanitize_cave_output_image(image: np.ndarray):
    output = image.astype(np.float32, copy=True)
    output[~np.isfinite(output)] = np.nan
    output[output < 0] = np.nan
    return output


def h5_attr_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("latin-1", errors="ignore")
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value


def h5_attr_json_value(value):
    value = h5_attr_value(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bytes_,)):
        return bytes(value).decode("latin-1", errors="ignore")
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def set_h5_attr(attrs, key, value):
    try:
        attrs[str(key)] = h5_attr_value(value)
    except TypeError:
        attrs[str(key)] = str(value)


def copy_h5_attrs(source_attrs, target_attrs, prefix=None, overwrite=False):
    for key, value in source_attrs.items():
        target_key = f"{prefix}{key}" if prefix else str(key)
        if not overwrite and target_key in target_attrs:
            continue
        set_h5_attr(target_attrs, target_key, value)


def matching_edf_header(filename: str):
    edf_path = Path(filename).with_suffix(".edf")
    if not edf_path.exists():
        return {}

    try:
        _image, header, *_ = read_edf_file(edf_path)
        return dict(header)
    except Exception:
        return {}


def collect_h5_attrs_json(source):
    collected = {
        "/": {str(key): h5_attr_json_value(value) for key, value in source.attrs.items()}
    }

    def collect(name, obj):
        if obj.attrs:
            collected[f"/{name}"] = {
                str(key): h5_attr_json_value(value)
                for key, value in obj.attrs.items()
            }

    source.visititems(collect)
    return collected


def add_h5_output_metadata(out, dataset, source_file: str, source_dataset_name: str):
    source_file = Path(source_file)

    set_h5_attr(out.attrs, "source_file", source_file.name)
    set_h5_attr(out.attrs, "source_dataset", source_dataset_name)
    set_h5_attr(out.attrs, "processing", "central symmetry cave filling")
    set_h5_attr(dataset.attrs, "source_file", source_file.name)
    set_h5_attr(dataset.attrs, "source_dataset", source_dataset_name)
    set_h5_attr(dataset.attrs, "processing", "central symmetry cave filling")

    try:
        with h5py.File(source_file, "r") as source:
            copy_h5_attrs(source.attrs, out.attrs)
            source_attrs_json = json.dumps(collect_h5_attrs_json(source), ensure_ascii=False)
            set_h5_attr(out.attrs, "source_h5_attrs_json", source_attrs_json)
            set_h5_attr(dataset.attrs, "source_h5_attrs_json", source_attrs_json)
            if source_dataset_name in source:
                copy_h5_attrs(source[source_dataset_name].attrs, dataset.attrs)
    except Exception:
        pass

    edf_header = matching_edf_header(source_file)
    if not edf_header:
        return

    set_h5_attr(out.attrs, "edf_header_source", source_file.with_suffix(".edf").name)
    set_h5_attr(dataset.attrs, "edf_header_source", source_file.with_suffix(".edf").name)
    edf_header_json = json.dumps(edf_header, ensure_ascii=False)
    set_h5_attr(out.attrs, "edf_header_json", edf_header_json)
    set_h5_attr(dataset.attrs, "edf_header_json", edf_header_json)

    for key, value in edf_header.items():
        if key not in out.attrs:
            set_h5_attr(out.attrs, key, value)
        if key not in dataset.attrs:
            set_h5_attr(dataset.attrs, key, value)
        set_h5_attr(out.attrs, f"edf_header_{key}", value)
        set_h5_attr(dataset.attrs, f"edf_header_{key}", value)


# New function for writing cave-filled H5 frames
def write_h5_frame_file(filename: str, image: np.ndarray, source_file: str, source_dataset_name: str, frame_index: int):
    filename = Path(filename)
    source_file = Path(source_file)

    with h5py.File(filename, "w") as out:
        dataset = out.create_dataset("/entry_0000/instrument/eiger/data", data=sanitize_cave_output_image(image), compression="gzip")
        add_h5_output_metadata(out, dataset, source_file, source_dataset_name)
        set_h5_attr(dataset.attrs, "source_frame", int(frame_index))


def h5_stack_shape(frame_shape, n_frames, frame_axis):
    ny, nx = tuple(int(value) for value in frame_shape)
    n_frames = int(n_frames)

    if frame_axis == 1:
        return ny, n_frames, nx
    if frame_axis == 2:
        return ny, nx, n_frames
    return n_frames, ny, nx


def h5_stack_chunks(frame_shape, frame_axis):
    ny, nx = tuple(int(value) for value in frame_shape)

    if frame_axis == 1:
        return ny, 1, nx
    if frame_axis == 2:
        return ny, nx, 1
    return 1, ny, nx


def write_h5_stack_frame(dataset, frame_axis, frame_index, image):
    output = sanitize_cave_output_image(image)

    if frame_axis == 1:
        dataset[:, frame_index, :] = output
    elif frame_axis == 2:
        dataset[:, :, frame_index] = output
    else:
        dataset[frame_index, :, :] = output


def create_h5_cave_stack_file(
    filename: str,
    frame_shape,
    n_frames: int,
    frame_axis,
    source_file: str,
    source_dataset_name: str,
):
    filename = Path(filename)
    source_file = Path(source_file)
    frame_axis = 0 if frame_axis is None else int(frame_axis)
    n_frames = int(n_frames)

    out = h5py.File(filename, "w")
    try:
        dataset = out.create_dataset(
            "/entry_0000/instrument/eiger/data",
            shape=h5_stack_shape(frame_shape, n_frames, frame_axis),
            dtype=np.float32,
            chunks=h5_stack_chunks(frame_shape, frame_axis),
            compression="gzip",
        )
        add_h5_output_metadata(out, dataset, source_file, source_dataset_name)
        set_h5_attr(out.attrs, "source_frames", int(n_frames))
        set_h5_attr(out.attrs, "source_frame_axis", int(frame_axis))
        set_h5_attr(dataset.attrs, "source_frames", int(n_frames))
        set_h5_attr(dataset.attrs, "source_frame_axis", int(frame_axis))
        return out, dataset
    except Exception:
        out.close()
        raise


def inspect_h5_image_dataset(filename: str):
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
        shape = tuple(dataset.shape)

        header = {
            "Dataset": dataset_name,
            "Shape": str(shape),
            "Dtype": str(dataset.dtype),
        }

        for key, value in dataset.attrs.items():
            header[key] = str(value)

        add_matching_edf_center(header, filename)

        if dataset.ndim == 2:
            frame_axis = None
            n_frames = 1
        elif dataset.ndim == 3:
            frame_axis = int(np.argmin(shape))
            n_frames = int(shape[frame_axis])
            header["Frame axis"] = str(frame_axis)
            header["Number of frames"] = str(n_frames)
        else:
            raise ValueError("Only 2D and 3D H5 datasets are supported here.")

    return dataset_name, shape, frame_axis, n_frames, header


def read_h5_frame(filename: str, dataset_name: str, frame_index: int = 0, add_matching_center=True):
    filename = Path(filename)

    with h5py.File(filename, "r") as h5:
        dataset = h5[dataset_name]

        header = {
            "Dataset": dataset_name,
            "Shape": str(tuple(dataset.shape)),
            "Dtype": str(dataset.dtype),
        }

        for key, value in dataset.attrs.items():
            header[key] = str(value)

        if add_matching_center:
            add_matching_edf_center(header, filename)

        if dataset.ndim == 2:
            image = np.asarray(dataset[...], dtype=np.float64)
            header["Displayed frame"] = "single 2D image"
        elif dataset.ndim == 3:
            shape = dataset.shape
            frame_axis = int(np.argmin(shape))
            n_frames = int(shape[frame_axis])
            frame_index = max(0, min(int(frame_index), n_frames - 1))

            if frame_axis == 0:
                image = np.asarray(dataset[frame_index, :, :], dtype=np.float64)
            elif frame_axis == 1:
                image = np.asarray(dataset[:, frame_index, :], dtype=np.float64)
            else:
                image = np.asarray(dataset[:, :, frame_index], dtype=np.float64)

            header["Frame axis"] = str(frame_axis)
            header["Displayed frame"] = f"{frame_index} from axis {frame_axis}"
            header["Number of frames"] = str(n_frames)
        else:
            raise ValueError("Only 2D and 3D H5 datasets are supported here.")

    return image, header


def get_header_float(header: dict, *names):
    for name in names:
        if name in header:
            try:
                return float(normalize_decimal_text(header[name]))
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
# ========================= CAVE TOOLS ========================
# ============================================================

def apply_central_symmetry_cave(
    image,
    xc,
    yc,
    nan_operator=">=",
    nan_threshold=4e9,
    nan_operator_2=None,
    nan_threshold_2=None,
    use_id13_beamstop=False,
    beamstop_y=1376,
    reference_angle_deg=0.0,
    expand_nan_neighbors=False,
    pre_nan_mask=None,
    extra_mask=None,
    exclude_mask=None,
):
    source = image.astype(np.float64).copy()
    if pre_nan_mask is not None:
        pre_nan_mask = np.asarray(pre_nan_mask, dtype=bool)
        if pre_nan_mask.shape == source.shape:
            source[pre_nan_mask] = np.nan

    cave_threshold_mask = np.zeros(source.shape, dtype=bool)

    if nan_operator == ">=":
        cave_threshold_mask |= source >= nan_threshold
    elif nan_operator == "<=":
        cave_threshold_mask |= source <= nan_threshold

    if nan_operator_2 == ">=" and nan_threshold_2 is not None:
        cave_threshold_mask |= source >= nan_threshold_2
    elif nan_operator_2 == "<=" and nan_threshold_2 is not None:
        cave_threshold_mask |= source <= nan_threshold_2

    cave_threshold_mask |= ~np.isfinite(source)

    if expand_nan_neighbors:
        original_nan_mask = cave_threshold_mask.copy()
        radius = 2

        padded_mask = np.pad(
            original_nan_mask,
            radius,
            mode="constant",
            constant_values=False,
        )

        expanded_mask = np.zeros_like(original_nan_mask, dtype=bool)

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                expanded_mask |= padded_mask[
                    radius + dy:radius + dy + original_nan_mask.shape[0],
                    radius + dx:radius + dx + original_nan_mask.shape[1],
                ]

        cave_threshold_mask = expanded_mask

    ny, nx = source.shape
    cave_mask = cave_threshold_mask.copy()

    if use_id13_beamstop:
        angle = np.deg2rad(float(reference_angle_deg))
        u = np.array([np.cos(angle), np.sin(angle)], dtype=float)
        v = np.array([-np.sin(angle), np.cos(angle)], dtype=float)
        height = float(beamstop_y) - float(yc)
        length = 2.0 * float(np.hypot(nx, ny))
        polygon = np.array(
            [
                (xc, yc),
                (xc + u[0] * length, yc + u[1] * length),
                (xc + u[0] * length + v[0] * height, yc + u[1] * length + v[1] * height),
                (xc + v[0] * height, yc + v[1] * height),
            ],
            dtype=float,
        )
        xmin = max(0, int(np.floor(np.nanmin(polygon[:, 0]))))
        xmax = min(nx, int(np.ceil(np.nanmax(polygon[:, 0]))) + 1)
        ymin = max(0, int(np.floor(np.nanmin(polygon[:, 1]))))
        ymax = min(ny, int(np.ceil(np.nanmax(polygon[:, 1]))) + 1)
        if xmin < xmax and ymin < ymax:
            yy, xx = np.mgrid[ymin:ymax, xmin:xmax]
            points = np.column_stack((xx.ravel(), yy.ravel()))
            path = MplPath(polygon)
            cave_mask[ymin:ymax, xmin:xmax] |= path.contains_points(points).reshape((ymax - ymin, xmax - xmin))

    if extra_mask is not None:
        extra_mask = np.asarray(extra_mask, dtype=bool)
        if extra_mask.shape == source.shape:
            cave_mask |= extra_mask

    if exclude_mask is not None:
        exclude_mask = np.asarray(exclude_mask, dtype=bool)
        if exclude_mask.shape == source.shape:
            cave_mask &= ~exclude_mask

    source[cave_mask] = np.nan
    filled = source.copy()
    filled[cave_mask] = np.nan

    missing_y, missing_x = np.where(cave_mask)
    if missing_y.size:
        symmetric_x = np.rint(2 * float(xc) - missing_x).astype(int)
        symmetric_y = np.rint(2 * float(yc) - missing_y).astype(int)
        valid = (
            (symmetric_x >= 0)
            & (symmetric_x < nx)
            & (symmetric_y >= 0)
            & (symmetric_y < ny)
        )
        if np.any(valid):
            target_y = missing_y[valid]
            target_x = missing_x[valid]
            source_values = source[symmetric_y[valid], symmetric_x[valid]]
            finite = np.isfinite(source_values)
            filled[target_y[finite], target_x[finite]] = source_values[finite]

    final_threshold_mask = np.zeros(filled.shape, dtype=bool)
    if nan_operator == ">=":
        final_threshold_mask |= filled >= nan_threshold
    elif nan_operator == "<=":
        final_threshold_mask |= filled <= nan_threshold

    if nan_operator_2 == ">=" and nan_threshold_2 is not None:
        final_threshold_mask |= filled >= nan_threshold_2
    elif nan_operator_2 == "<=" and nan_threshold_2 is not None:
        final_threshold_mask |= filled <= nan_threshold_2

    final_threshold_mask |= ~np.isfinite(filled)
    filled[final_threshold_mask] = np.nan

    return source, filled, cave_mask | final_threshold_mask


# ============================================================
# =========================== CANVAS ==========================
# ============================================================

def draw_reference_axes(ax, shape, xc, yc, angle_deg=0.0):
    ny, nx = shape
    angle = np.deg2rad(float(angle_deg or 0.0))
    length = 2.0 * float(np.hypot(nx, ny))
    u = np.array([np.cos(angle), np.sin(angle)], dtype=float)
    v = np.array([-np.sin(angle), np.cos(angle)], dtype=float)
    center = np.array([float(xc), float(yc)], dtype=float)
    for direction in (u, v):
        p0 = center - direction * length
        p1 = center + direction * length
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="red", linewidth=1.0)
    ax.plot(xc, yc, "wo", markersize=4)


class ImageCanvas(FigureCanvas):
    def __init__(self):
        self.image_artist = None
        self.raw_image = None
        self.coordinate_label = None
        self.q_calculator = None
        self.image_name = "Image"
        self._is_panning = False
        self._pan_start_pos = None
        self._pan_start_xlim = None
        self._pan_start_ylim = None
        self._data_xlim = None
        self._data_ylim = None
        self.sync_partner = None
        self._syncing_view = False

        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setStyleSheet("background: transparent;")
        self.fig.patch.set_alpha(0)
        self.ax.set_facecolor("none")
        self.setFocusPolicy(Qt.StrongFocus)
        self.ax.set_axis_off()
        self.fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
        self.mpl_connect("motion_notify_event", self._on_motion)

        try:
            self.grabGesture(Qt.PinchGesture)
        except Exception:
            pass

    def set_coordinate_label(self, label, image_name):
        self.coordinate_label = label
        self.image_name = image_name

    def coordinate_text(self, text):
        if self.image_name and not text.endswith("= -"):
            return f"{self.image_name} | {text}"
        return text

    def set_q_calculator(self, calculator):
        self.q_calculator = calculator

    def set_sync_partner(self, partner):
        self.sync_partner = partner

    def apply_synced_limits_from(self, source):
        if self._syncing_view or source is self or self.raw_image is None:
            return
        self._syncing_view = True
        try:
            self.ax.set_xlim(source.ax.get_xlim())
            self.ax.set_ylim(source.ax.get_ylim())
            self.draw_idle()
        finally:
            self._syncing_view = False

    def sync_partner_view(self):
        if self._syncing_view or self.sync_partner is None:
            return
        self.sync_partner.apply_synced_limits_from(self)

    def event(self, event):
        if getattr(self, "raw_image", None) is not None:
            try:
                if event.type() == QEvent.NativeGesture:
                    gesture_type = event.gestureType()
                    value = event.value()
                    if gesture_type == Qt.ZoomNativeGesture and value != 0:
                        scale = 1.0 / (1.0 + value) if value > -0.95 else 1.25
                        self._zoom_from_qpoint(self._event_center_point(event), scale)
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
                            self._zoom_from_qpoint(self._event_center_point(event), 1.0 / factor)
                            event.accept()
                            return True
            except Exception:
                pass

        return super().event(event)

    def wheelEvent(self, event):
        if self.raw_image is None:
            return super().wheelEvent(event)

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
                self._zoom_from_qpoint(event.position(), scale)
        else:
            self._pan_by_trackpad(dx, dy)
        event.accept()

    def _event_center_point(self, event):
        try:
            position = event.position()
            if position is not None:
                return position
        except Exception:
            pass

        return self.rect().center()

    def mousePressEvent(self, event):
        if self.raw_image is not None and event.button() == Qt.LeftButton:
            self._is_panning = True
            self._pan_start_pos = event.position()
            self._pan_start_xlim = self.ax.get_xlim()
            self._pan_start_ylim = self.ax.get_ylim()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._is_panning and self.raw_image is not None:
            start_x, start_y = self._qt_pos_to_data(self._pan_start_pos.x(), self._pan_start_pos.y())
            current_x, current_y = self._qt_pos_to_data(event.position().x(), event.position().y())

            if None not in (start_x, start_y, current_x, current_y):
                dx = start_x - current_x
                dy = start_y - current_y
                x0, x1 = self._pan_start_xlim
                y0, y1 = self._pan_start_ylim
                self.ax.set_xlim(x0 + dx, x1 + dx)
                self.ax.set_ylim(y0 + dy, y1 + dy)
                constrain_image_axes(self.ax, self.raw_image.shape)
                self.draw_idle()
                self.sync_partner_view()

            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._is_panning and event.button() == Qt.LeftButton:
            self._is_panning = False
            self._pan_start_pos = None
            self._pan_start_xlim = None
            self._pan_start_ylim = None
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.raw_image is not None:
            self.reset_view()
            event.accept()
            return

        super().mouseDoubleClickEvent(event)

    def _qt_pos_to_data(self, x, y):
        if self.ax is None:
            return None, None

        canvas_height = self.height()
        display_x = x
        display_y = canvas_height - y

        try:
            return self.ax.transData.inverted().transform((display_x, display_y))
        except Exception:
            return None, None

    def _zoom_at(self, xdata, ydata, zoom_factor):
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()

        new_width = (x1 - x0) * zoom_factor
        new_height = (y1 - y0) * zoom_factor

        rel_x = (xdata - x0) / (x1 - x0) if x1 != x0 else 0.5
        rel_y = (ydata - y0) / (y1 - y0) if y1 != y0 else 0.5

        self.ax.set_xlim(xdata - new_width * rel_x, xdata + new_width * (1 - rel_x))
        self.ax.set_ylim(ydata - new_height * rel_y, ydata + new_height * (1 - rel_y))
        constrain_image_axes(self.ax, self.raw_image.shape)
        self.draw_idle()
        self.sync_partner_view()

    def _zoom_from_qpoint(self, qpoint, zoom_factor):
        try:
            xdata, ydata = self._qt_pos_to_data(float(qpoint.x()), float(qpoint.y()))
        except Exception:
            xdata, ydata = None, None

        if xdata is None or ydata is None:
            return

        self._zoom_at(xdata, ydata, zoom_factor)

    def _pan_by_trackpad(self, dx, dy):
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        xspan = x1 - x0
        yspan = y1 - y0
        shift_x = -dx * xspan * 0.08
        shift_y = dy * yspan * 0.08
        self.ax.set_xlim(x0 + shift_x, x1 + shift_x)
        self.ax.set_ylim(y0 + shift_y, y1 + shift_y)
        constrain_image_axes(self.ax, self.raw_image.shape)
        self.draw_idle()
        self.sync_partner_view()

    def reset_view(self):
        if self._data_xlim is not None and self._data_ylim is not None:
            self.ax.set_xlim(self._data_xlim)
            self.ax.set_ylim(self._data_ylim)
            self.draw_idle()
            self.sync_partner_view()

    def _on_motion(self, event):
        if self.coordinate_label is None:
            return

        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            self.coordinate_label.setText("x = - | y = - | q = - | I = -")
            return

        x = int(round(event.xdata + 1))
        y = int(round(event.ydata + 1))
        intensity_text = "I = -"
        q_text = "q = -"

        if self.raw_image is not None:
            ny, nx = self.raw_image.shape
            if 1 <= x <= nx and 1 <= y <= ny:
                value = self.raw_image[y - 1, x - 1]
                if np.isfinite(value):
                    intensity_text = f"I = {value:.6g}"
                else:
                    intensity_text = "I = NaN"

                if self.q_calculator is not None:
                    q_value = self.q_calculator(x, y)
                    if q_value is not None:
                        q_text = f"q = {q_value:.6g} nm⁻¹"

        self.coordinate_label.setText(self.coordinate_text(f"x = {x} | y = {y} | {q_text} | {intensity_text}"))

    def show_image(self, image, xc=None, yc=None, title="", vmin=None, vmax=None, white_mask=None, reference_angle_deg=0.0):
        previous_xlim = self.ax.get_xlim() if self.image_artist is not None else None
        previous_ylim = self.ax.get_ylim() if self.image_artist is not None else None
        self.raw_image = image
        self.ax.clear()
        self.fig.patch.set_alpha(0)
        self.ax.set_facecolor("none")
        self.ax.set_axis_off()

        display = image.astype(np.float64).copy()
        display[~np.isfinite(display)] = np.nan
        display[display < 0] = np.nan

        with np.errstate(invalid="ignore", divide="ignore"):
            display = np.log10(display + 1)

        if white_mask is not None:
            display = display.copy()
            display[white_mask] = np.nan
            cmap = self.fig.canvas.figure.axes[0].images[0].cmap.copy() if self.ax.images else "jet"
        else:
            cmap = "jet"

        self.image_artist = self.ax.imshow(
            display,
            origin="upper",
            cmap=cmap,
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )

        ny, nx = image.shape
        self._data_xlim = (-0.5, nx - 0.5)
        self._data_ylim = (ny - 0.5, -0.5)

        if previous_xlim is not None and previous_ylim is not None:
            self.ax.set_xlim(previous_xlim)
            self.ax.set_ylim(previous_ylim)
            constrain_image_axes(self.ax, self.raw_image.shape)

        if white_mask is not None:
            self.image_artist.cmap.set_bad(color="white")

        if xc is not None and yc is not None:
            draw_reference_axes(self.ax, image.shape, xc, yc, reference_angle_deg)

        if title:
            self.ax.set_title(title, fontsize=10)

        self.ax.set_aspect("equal")
        self.draw_idle()


class ManualCaveCanvas(FigureCanvas):
    def __init__(self, dialog, title):
        self.dialog = dialog
        self.title = title
        self.image = None
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setStyleSheet("background: transparent;")
        self.fig.patch.set_alpha(0)
        self.ax.set_facecolor("none")
        self.ax.set_axis_off()
        self.fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
        self._drag_start = None
        self._last_pan_point = None
        self._edit_state = None
        self._preview_patch = None
        self.coordinate_label = None
        self.image_name = ""
        self.mpl_connect("motion_notify_event", self.update_coordinate_label)
        self.mpl_connect("figure_leave_event", lambda event: self.update_coordinate_label(None))
        
    def set_coordinate_label(self, label, image_name=""):
        self.coordinate_label = label
        self.image_name = image_name
        self.update_coordinate_label(None)

    def update_coordinate_label(self, event):
        if self.coordinate_label is None:
            return

        if event is None or event.inaxes != self.ax or event.xdata is None or event.ydata is None or self.image is None:
            self.coordinate_label.setText("x = - | y = - | q = - | I = -")
            return

        x = int(round(event.xdata + 1))
        y = int(round(event.ydata + 1))
        intensity_text = "I = -"
        q_text = "q = -"

        ny, nx = self.image.shape
        if 1 <= x <= nx and 1 <= y <= ny:
            value = self.image[y - 1, x - 1]
            if np.isfinite(value):
                intensity_text = f"I = {value:.6g}"
            else:
                intensity_text = "I = NaN"

            parent = self.dialog.parent()
            if parent is not None and hasattr(parent, "calculate_q_at_pixel"):
                q_value = parent.calculate_q_at_pixel(x, y)
                if q_value is not None:
                    q_text = f"q = {q_value:.6g} nm⁻¹"

        prefix = f"{self.image_name} | " if self.image_name else ""
        self.coordinate_label.setText(f"{prefix}x = {x} | y = {y} | {q_text} | {intensity_text}")
        self.mpl_connect("button_press_event", self.on_press)
        self.mpl_connect("button_release_event", self.on_release)
        self.mpl_connect("motion_notify_event", self.on_motion)
        self.mpl_connect("scroll_event", self.on_scroll)
        try:
            self.grabGesture(Qt.PinchGesture)
        except Exception:
            pass

    def show_image(self, image, vmin=None, vmax=None, shapes=None, active_polygon=None, xc=None, yc=None, reference_angle_deg=0.0):
        self.image = image
        self.ax.clear()
        self.ax.set_axis_off()

        if image is not None:
            display = image.astype(np.float64).copy()
            display[~np.isfinite(display)] = np.nan
            display[display < 0] = np.nan
            with np.errstate(invalid="ignore", divide="ignore"):
                display = np.log10(display + 1)
            self.ax.imshow(display, origin="upper", cmap="jet", interpolation="nearest", vmin=vmin, vmax=vmax)

            if xc is not None and yc is not None:
                draw_reference_axes(self.ax, image.shape, xc, yc, reference_angle_deg)

        for shape in shapes or []:
            self.add_shape_patch(shape, alpha=0.22)

        if self is self.dialog.before_canvas and self.dialog.selected_shape_index is not None:
            if 0 <= self.dialog.selected_shape_index < len(self.dialog.shapes):
                self.draw_selection_handles(self.dialog.shapes[self.dialog.selected_shape_index])

        if active_polygon and len(active_polygon) > 1:
            color = self.dialog.mode_color(self.dialog.current_mask_mode)
            patch = MplPolygon(active_polygon, closed=False, fill=False, edgecolor=color, linewidth=1.5)
            self.ax.add_patch(patch)

        self.ax.set_aspect("equal")
        self.draw_idle()
        self.dialog.apply_synced_view(source=self)

    def add_shape_patch(self, shape, alpha=0.22):
        self.ax.add_patch(self.shape_to_patch(shape, alpha=alpha))

    def draw_selection_handles(self, shape):
        self.add_shape_patch(shape, alpha=0.08)
        for x, y in self.dialog.shape_handles(shape):
            self.ax.plot(x, y, "s", ms=7, mfc="white", mec=self.dialog.mode_edge_color(shape), mew=1.6)

    def on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None or event.button != 1:
            return

        if self.dialog.current_tool == "Select":
            hit = self.dialog.hit_test_shape(event.xdata, event.ydata)
            if hit is not None:
                self.dialog.select_shape(hit[0])
                self._edit_state = {
                    "shape_index": hit[0],
                    "handle": hit[1],
                    "last": (event.xdata, event.ydata),
                }
            return

        if self is not self.dialog.before_canvas:
            return

        if self.dialog.current_tool == "Rectangle":
            self._drag_start = (event.xdata, event.ydata)
        elif self.dialog.current_tool in ("Vertical band", "Horizontal band"):
            self._drag_start = (event.xdata, event.ydata)
        else:
            if event.dblclick:
                self.dialog.finish_polygon()
                return
            self.dialog.active_polygon.append((event.xdata, event.ydata))
            self.dialog.refresh_preview()

    def on_motion(self, event):
        self.update_coordinate_label(event)
        if self._edit_state is not None:
            if event.inaxes == self.ax and event.xdata is not None and event.ydata is not None:
                last_x, last_y = self._edit_state["last"]
                self.dialog.edit_shape(
                    self._edit_state["shape_index"],
                    self._edit_state["handle"],
                    event.xdata,
                    event.ydata,
                    event.xdata - last_x,
                    event.ydata - last_y,
                )
                self._edit_state["last"] = (event.xdata, event.ydata)
            return

        if self._drag_start is None or event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        self.dialog.refresh_preview()
        x0, y0 = self._drag_start
        patch = self.preview_patch(x0, y0, event.xdata, event.ydata)
        self.ax.add_patch(patch)
        self.draw_idle()

    def on_release(self, event):
        if self._edit_state is not None:
            self._edit_state = None
            return

        if self._drag_start is None:
            self._last_pan_point = None
            return
        if event.inaxes == self.ax and event.xdata is not None and event.ydata is not None:
            x0, y0 = self._drag_start
            if self.dialog.current_tool == "Rectangle" and abs(event.xdata - x0) >= 2 and abs(event.ydata - y0) >= 2:
                self.dialog.add_shape("rect", (x0, y0, event.xdata, event.ydata))
            elif self.dialog.current_tool == "Vertical band" and abs(event.xdata - x0) >= 2:
                self.dialog.add_shape("vband", (x0, y0, event.xdata, event.ydata))
            elif self.dialog.current_tool == "Horizontal band" and abs(event.ydata - y0) >= 2:
                self.dialog.add_shape("hband", (x0, y0, event.xdata, event.ydata))
        self._drag_start = None
        self._last_pan_point = None

    def on_scroll(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        scale = 0.85 if event.button == "up" else 1.18
        self.zoom_at(event.xdata, event.ydata, scale)

    def event(self, event):
        if self.image is not None:
            try:
                if event.type() == QEvent.NativeGesture:
                    gesture_type = event.gestureType()
                    value = event.value()
                    if gesture_type == Qt.ZoomNativeGesture and value != 0:
                        scale = 1.0 / (1.0 + value) if value > -0.95 else 1.25
                        xdata, ydata = self.qt_pos_to_data(self.event_center_point(event))
                        if xdata is not None and ydata is not None:
                            self.zoom_at(xdata, ydata, scale)
                            event.accept()
                            return True

                    if gesture_type == Qt.SmartZoomNativeGesture:
                        self.dialog.reset_synced_view()
                        event.accept()
                        return True

                if event.type() == QEvent.Gesture:
                    pinch = event.gesture(Qt.PinchGesture)
                    if pinch is not None:
                        factor = pinch.scaleFactor()
                        if factor and factor > 0:
                            xdata, ydata = self.qt_pos_to_data(self.event_center_point(event))
                            if xdata is not None and ydata is not None:
                                self.zoom_at(xdata, ydata, 1.0 / factor)
                                event.accept()
                                return True
            except Exception:
                pass

        return super().event(event)

    def wheelEvent(self, event):
        if self.image is None:
            return super().wheelEvent(event)

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
                xdata, ydata = self.qt_pos_to_data(event.position())
                if xdata is not None and ydata is not None:
                    scale = 0.88 if dy > 0 else 1.14
                    self.zoom_at(xdata, ydata, scale)
        else:
            x0, x1 = self.ax.get_xlim()
            y0, y1 = self.ax.get_ylim()
            xspan = x1 - x0
            yspan = y1 - y0
            shift_x = -dx * xspan * 0.08
            shift_y = dy * yspan * 0.08
            self.dialog.set_synced_limits((x0 + shift_x, x1 + shift_x), (y0 + shift_y, y1 + shift_y))

        event.accept()

    def event_center_point(self, event):
        try:
            position = event.position()
            if position is not None:
                return position
        except Exception:
            pass
        return self.rect().center()

    def qt_pos_to_data(self, qpoint):
        try:
            display_x = float(qpoint.x())
            display_y = self.height() - float(qpoint.y())
            return self.ax.transData.inverted().transform((display_x, display_y))
        except Exception:
            return None, None

    def zoom_at(self, xdata, ydata, scale):
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        new_width = (x1 - x0) * scale
        new_height = (y1 - y0) * scale
        rel_x = (xdata - x0) / (x1 - x0) if x1 != x0 else 0.5
        rel_y = (ydata - y0) / (y1 - y0) if y1 != y0 else 0.5
        self.dialog.set_synced_limits(
            (xdata - new_width * rel_x, xdata + new_width * (1 - rel_x)),
            (ydata - new_height * rel_y, ydata + new_height * (1 - rel_y)),
        )

    def preview_patch(self, x0, y0, x1, y1):
        shape = {"type": "rect", "points": (x0, y0, x1, y1), "mode": self.dialog.current_mask_mode}
        if self.dialog.current_tool == "Vertical band":
            shape = {"type": "vband", "points": (x0, y0, x1, y1, 10.0), "mode": self.dialog.current_mask_mode}
        elif self.dialog.current_tool == "Horizontal band":
            shape = {"type": "hband", "points": (x0, y0, x1, y1, 10.0), "mode": self.dialog.current_mask_mode}
        return self.shape_to_patch(shape, alpha=0.18)

    def shape_to_patch(self, shape, alpha=0.22):
        facecolor = self.dialog.mode_color(shape.get("mode", "include"))
        edgecolor = self.dialog.mode_edge_color(shape)
        if shape["type"] == "rect":
            x0, y0, x1, y1 = shape["points"]
            return MplRectangle(
                (min(x0, x1), min(y0, y1)),
                abs(x1 - x0),
                abs(y1 - y0),
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=1.2,
                alpha=alpha,
            )

        if shape["type"] in ("vband", "hband"):
            return MplPolygon(
                self.dialog.band_polygon(shape),
                closed=True,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=1.2,
                alpha=alpha,
            )

        return MplPolygon(
            shape["points"],
            closed=True,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1.2,
            alpha=alpha,
        )


def connected_nan_region(nan_mask, x, y):
    nan_mask = np.asarray(nan_mask, dtype=bool)
    ny, nx = nan_mask.shape
    x = int(round(x))
    y = int(round(y))
    if not (0 <= x < nx and 0 <= y < ny) or not nan_mask[y, x]:
        return np.zeros_like(nan_mask, dtype=bool)

    region = np.zeros_like(nan_mask, dtype=bool)
    stack = [(y, x)]
    region[y, x] = True

    while stack:
        cy, cx = stack.pop()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nyi = cy + dy
            nxi = cx + dx
            if 0 <= nyi < ny and 0 <= nxi < nx and nan_mask[nyi, nxi] and not region[nyi, nxi]:
                region[nyi, nxi] = True
                stack.append((nyi, nxi))

    return region


def visual_blank_mask(image):
    image = np.asarray(image, dtype=np.float64)
    return (~np.isfinite(image)) | (image < 0)


def fill_region_by_symmetry(image, source_image, region_mask, mode, xc, yc):
    output = np.asarray(image, dtype=np.float64).copy()
    source = np.asarray(source_image, dtype=np.float64)
    region_mask = np.asarray(region_mask, dtype=bool)
    ny, nx = output.shape
    ys, xs = np.where(region_mask)
    if ys.size == 0:
        return output

    if mode == "central":
        sx = np.rint(2 * float(xc) - xs).astype(int)
        sy = np.rint(2 * float(yc) - ys).astype(int)
    elif mode == "horizontal":
        sx = xs
        sy = np.rint(2 * float(yc) - ys).astype(int)
    elif mode == "vertical":
        sx = np.rint(2 * float(xc) - xs).astype(int)
        sy = ys
    else:
        return output

    valid = (sx >= 0) & (sx < nx) & (sy >= 0) & (sy < ny)
    if np.any(valid):
        source_values = source[sy[valid], sx[valid]]
        finite = np.isfinite(source_values)
        output[ys[valid][finite], xs[valid][finite]] = source_values[finite]

    return output


class CustomCaveCanvas(FigureCanvas):
    def __init__(self, dialog, role):
        self.dialog = dialog
        self.role = role
        self.image = None
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setStyleSheet("background: transparent;")
        self.fig.patch.set_alpha(0)
        self.ax.set_facecolor("none")
        self.ax.set_axis_off()
        self.fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
        self.coordinate_label = None
        self._selection_start = None
        try:
            self.grabGesture(Qt.PinchGesture)
        except Exception:
            pass
        self.mpl_connect("button_press_event", self.on_press)
        self.mpl_connect("button_release_event", self.on_release)
        self.mpl_connect("motion_notify_event", self.on_motion)
        self.mpl_connect("scroll_event", self.on_scroll)
        self.mpl_connect("figure_leave_event", lambda event: self.update_coordinate_label(None))

    def set_coordinate_label(self, label):
        self.coordinate_label = label
        self.update_coordinate_label(None)

    def update_coordinate_label(self, event):
        if self.coordinate_label is None:
            return
        if event is None or event.inaxes != self.ax or event.xdata is None or event.ydata is None or self.image is None:
            self.coordinate_label.setText("x = - | y = - | q = - | I = -")
            return

        x = int(round(event.xdata + 1))
        y = int(round(event.ydata + 1))
        intensity_text = "I = -"
        q_text = "q = -"
        ny, nx = self.image.shape
        if 1 <= x <= nx and 1 <= y <= ny:
            value = self.image[y - 1, x - 1]
            intensity_text = f"I = {value:.6g}" if np.isfinite(value) else "I = NaN"
            parent = self.dialog.parent()
            if parent is not None and hasattr(parent, "calculate_q_at_pixel"):
                q_value = parent.calculate_q_at_pixel(x, y)
                if q_value is not None:
                    q_text = f"q = {q_value:.6g} nm⁻¹"
        self.coordinate_label.setText(f"x = {x} | y = {y} | {q_text} | {intensity_text}")

    def event_data_position(self, event):
        if self.image is None:
            return None

        if event.xdata is not None and event.ydata is not None:
            xdata, ydata = event.xdata, event.ydata
        else:
            try:
                xdata, ydata = self.ax.transData.inverted().transform((event.x, event.y))
            except Exception:
                return None

        ny, nx = self.image.shape
        xdata = min(max(float(xdata), -0.5), nx - 0.5)
        ydata = min(max(float(ydata), -0.5), ny - 0.5)
        return xdata, ydata

    def show_image(self, image, vmin=None, vmax=None, region_mask=None, xc=None, yc=None, reference_angle_deg=0.0):
        self.image = image
        self.ax.clear()
        self.ax.set_axis_off()
        self.fig.patch.set_alpha(0)
        self.ax.set_facecolor("none")

        if image is not None:
            display = np.asarray(image, dtype=np.float64).copy()
            display[~np.isfinite(display)] = np.nan
            display[display < 0] = np.nan
            with np.errstate(invalid="ignore", divide="ignore"):
                display = np.log10(display + 1)
            self.ax.imshow(display, origin="upper", cmap="jet", interpolation="nearest", vmin=vmin, vmax=vmax)
            if region_mask is not None and np.any(region_mask):
                overlay = np.zeros((*region_mask.shape, 4), dtype=float)
                overlay[region_mask] = (0.0, 1.0, 1.0, 0.35)
                self.ax.imshow(overlay, origin="upper", interpolation="nearest")
            if xc is not None and yc is not None:
                draw_reference_axes(self.ax, image.shape, xc, yc, reference_angle_deg)

        self.ax.set_aspect("equal")
        if self.dialog.synced_xlim is not None and self.dialog.synced_ylim is not None and self.image is not None:
            self.ax.set_xlim(self.dialog.synced_xlim)
            self.ax.set_ylim(self.dialog.synced_ylim)
            constrain_image_axes(self.ax, self.image.shape)
        self.draw_idle()
        self.dialog.apply_synced_view(source=self)

    def on_press(self, event):
        if self.role not in ("original", "result") or event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        if event.button == 1:
            self._selection_start = (event.xdata, event.ydata)
        elif event.button == 3:
            self.dialog.open_symmetry_menu(event, self.image)

    def on_motion(self, event):
        self.update_coordinate_label(event)
        if self.role not in ("original", "result") or self._selection_start is None:
            return
        position = self.event_data_position(event)
        if position is None:
            return
        xdata, ydata = position
        self.dialog.refresh_images()
        x0, y0 = self._selection_start
        patch = MplRectangle(
            (min(x0, xdata), min(y0, ydata)),
            abs(xdata - x0),
            abs(ydata - y0),
            facecolor="none",
            edgecolor="#00ffff",
            linewidth=1.5,
            linestyle="--",
        )
        self.ax.add_patch(patch)
        self.draw_idle()

    def on_release(self, event):
        if self.role not in ("original", "result") or self._selection_start is None:
            self._selection_start = None
            return
        position = self.event_data_position(event)
        if position is not None:
            self.dialog.select_nan_rectangle(self._selection_start, position, self.image)
        self._selection_start = None

    def on_scroll(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        scale = 0.85 if event.button == "up" else 1.18
        self.zoom_at(event.xdata, event.ydata, scale)

    def event(self, event):
        if self.image is not None:
            try:
                if event.type() == QEvent.NativeGesture:
                    gesture_type = event.gestureType()
                    value = event.value()
                    if gesture_type == Qt.ZoomNativeGesture and value != 0:
                        scale = 1.0 / (1.0 + value) if value > -0.95 else 1.25
                        xdata, ydata = self.qt_pos_to_data(self.event_center_point(event))
                        if xdata is not None and ydata is not None:
                            self.zoom_at(xdata, ydata, scale)
                            event.accept()
                            return True

                    if gesture_type == Qt.SmartZoomNativeGesture:
                        self.dialog.reset_synced_view()
                        event.accept()
                        return True

                if event.type() == QEvent.Gesture:
                    pinch = event.gesture(Qt.PinchGesture)
                    if pinch is not None:
                        factor = pinch.scaleFactor()
                        if factor and factor > 0:
                            xdata, ydata = self.qt_pos_to_data(self.event_center_point(event))
                            if xdata is not None and ydata is not None:
                                self.zoom_at(xdata, ydata, 1.0 / factor)
                                event.accept()
                                return True
            except Exception:
                pass

        return super().event(event)

    def wheelEvent(self, event):
        if self.image is None:
            return super().wheelEvent(event)

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
                xdata, ydata = self.qt_pos_to_data(event.position())
                if xdata is not None and ydata is not None:
                    scale = 0.88 if dy > 0 else 1.14
                    self.zoom_at(xdata, ydata, scale)
        else:
            x0, x1 = self.ax.get_xlim()
            y0, y1 = self.ax.get_ylim()
            xspan = x1 - x0
            yspan = y1 - y0
            shift_x = -dx * xspan * 0.08
            shift_y = dy * yspan * 0.08
            self.dialog.set_synced_limits((x0 + shift_x, x1 + shift_x), (y0 + shift_y, y1 + shift_y))

        event.accept()

    def event_center_point(self, event):
        try:
            position = event.position()
            if position is not None:
                return position
        except Exception:
            pass
        return self.rect().center()

    def qt_pos_to_data(self, qpoint):
        try:
            display_x = float(qpoint.x())
            display_y = self.height() - float(qpoint.y())
            return self.ax.transData.inverted().transform((display_x, display_y))
        except Exception:
            return None, None

    def zoom_at(self, xdata, ydata, scale):
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        new_width = (x1 - x0) * scale
        new_height = (y1 - y0) * scale
        rel_x = (xdata - x0) / (x1 - x0) if x1 != x0 else 0.5
        rel_y = (ydata - y0) / (y1 - y0) if y1 != y0 else 0.5
        self.dialog.set_synced_limits(
            (xdata - new_width * rel_x, xdata + new_width * (1 - rel_x)),
            (ydata - new_height * rel_y, ydata + new_height * (1 - rel_y)),
        )


class CustomCaveDialog(QDialog):
    def __init__(self, parent, source_image, caved_image, display_limits):
        super().__init__(parent)
        self.setWindowTitle("Custom cave")
        self.resize(1400, 640)
        self.source_image = np.asarray(source_image, dtype=np.float64)
        self.caved_image = np.asarray(caved_image, dtype=np.float64)
        self.final_image = self.caved_image.copy()
        self.display_limits = display_limits
        self.display_data_min, self.display_data_max = self.compute_display_range()
        self.selected_region = None
        self._syncing_view = False
        self.synced_xlim = None
        self.synced_ylim = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        self.save_button = QPushButton("💾 Save cave+")
        self.save_button.clicked.connect(self.save_cave_plus)
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_custom_fills)
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        toolbar.addStretch(1)
        toolbar.addWidget(self.reset_button)
        toolbar.addWidget(self.save_button)
        toolbar.addWidget(self.close_button)
        layout.addLayout(toolbar)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(6)
        self.original_canvas = CustomCaveCanvas(self, "original")
        self.result_canvas = CustomCaveCanvas(self, "result")
        for canvas in [self.original_canvas, self.result_canvas]:
            panel = QVBoxLayout()
            panel.setContentsMargins(0, 0, 0, 0)
            panel.setSpacing(2)
            coord = QLabel("x = - | y = - | q = - | I = -")
            coord.setMinimumHeight(28)
            coord.setAlignment(Qt.AlignCenter)
            coord.setStyleSheet("""
                QLabel {
                    background-color: #f4f4f4;
                    border-radius: 8px;
                    padding: 6px;
                    font-family: Menlo, Monaco, monospace;
                    font-size: 11px;
                }
            """)
            canvas.set_coordinate_label(coord)
            panel.addWidget(canvas, 1)
            panel.addWidget(coord, 0)
            body.addLayout(panel, 1)
            if canvas is not self.result_canvas:
                arrow = QLabel("→")
                arrow.setAlignment(Qt.AlignCenter)
                arrow.setFixedWidth(24)
                arrow.setStyleSheet("font-size: 22px; font-weight: 700; color: #444444;")
                body.addWidget(arrow, 0)

        layout.addLayout(body, 1)

        intensity_layout = QGridLayout()
        intensity_layout.setContentsMargins(0, 0, 0, 0)
        intensity_layout.setHorizontalSpacing(8)
        intensity_layout.setVerticalSpacing(2)
        self.min_label = QLabel()
        self.max_label = QLabel()
        self.min_slider = QSlider(Qt.Horizontal)
        self.max_slider = QSlider(Qt.Horizontal)
        self.min_slider.setRange(0, 1000)
        self.max_slider.setRange(0, 1000)
        self.auto_button = QPushButton("Auto")
        intensity_layout.addWidget(self.min_label, 0, 0)
        intensity_layout.addWidget(self.min_slider, 0, 1)
        intensity_layout.addWidget(self.max_label, 1, 0)
        intensity_layout.addWidget(self.max_slider, 1, 1)
        intensity_layout.addWidget(self.auto_button, 0, 2, 2, 1)
        layout.addLayout(intensity_layout)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.min_slider.valueChanged.connect(self.update_display_limits_from_sliders)
        self.max_slider.valueChanged.connect(self.update_display_limits_from_sliders)
        self.auto_button.clicked.connect(self.auto_display_limits)
        self.set_display_sliders_from_limits()
        self.refresh_images()

    def image_canvases(self):
        return [
            getattr(self, "original_canvas", None),
            getattr(self, "result_canvas", None),
        ]

    def reset_synced_view(self):
        self.synced_xlim = None
        self.synced_ylim = None
        self.refresh_images()

    def set_synced_limits(self, xlim, ylim):
        self.synced_xlim = tuple(xlim)
        self.synced_ylim = tuple(ylim)
        for canvas in self.image_canvases():
            if canvas is None or canvas.image is None:
                continue
            canvas.ax.set_xlim(self.synced_xlim)
            canvas.ax.set_ylim(self.synced_ylim)
            constrain_image_axes(canvas.ax, canvas.image.shape)
            canvas.draw_idle()

    def apply_synced_view(self, source=None):
        if self._syncing_view or self.synced_xlim is None or self.synced_ylim is None:
            return

        self._syncing_view = True
        try:
            for canvas in self.image_canvases():
                if canvas is None or canvas is source or canvas.image is None:
                    continue
                canvas.ax.set_xlim(self.synced_xlim)
                canvas.ax.set_ylim(self.synced_ylim)
                constrain_image_axes(canvas.ax, canvas.image.shape)
                canvas.draw_idle()
        finally:
            self._syncing_view = False

    def remaining_nan_mask(self):
        return visual_blank_mask(self.final_image)

    def compute_display_range(self):
        values = []
        for image in (self.source_image, self.caved_image):
            display = np.asarray(image, dtype=np.float64).copy()
            display[~np.isfinite(display)] = np.nan
            display[display < 0] = np.nan
            with np.errstate(invalid="ignore", divide="ignore"):
                display = np.log10(display + 1)
            finite = display[np.isfinite(display)]
            if finite.size:
                values.append(finite)

        if not values:
            return 0.0, 1.0

        merged = np.concatenate(values)
        data_min = float(np.nanmin(merged))
        data_max = float(np.nanmax(merged))
        if data_max <= data_min:
            data_max = data_min + 1.0
        return data_min, data_max

    def set_display_sliders_from_limits(self):
        data_min, data_max = self.display_data_min, self.display_data_max
        span = max(data_max - data_min, 1e-12)
        vmin, vmax = self.display_limits
        min_value = int(np.clip((vmin - data_min) / span * 1000.0, 0, 1000))
        max_value = int(np.clip((vmax - data_min) / span * 1000.0, 0, 1000))
        if max_value <= min_value:
            max_value = min(1000, min_value + 1)
        self.min_slider.blockSignals(True)
        self.max_slider.blockSignals(True)
        self.min_slider.setValue(min_value)
        self.max_slider.setValue(max_value)
        self.min_slider.blockSignals(False)
        self.max_slider.blockSignals(False)
        self.update_intensity_labels()

    def update_intensity_labels(self):
        vmin, vmax = self.display_limits
        self.min_label.setText(f"Min: {vmin:.3g}")
        self.max_label.setText(f"Max: {vmax:.3g}")

    def update_display_limits_from_sliders(self):
        min_value = self.min_slider.value()
        max_value = self.max_slider.value()
        if max_value <= min_value:
            max_value = min(1000, min_value + 1)
            self.max_slider.blockSignals(True)
            self.max_slider.setValue(max_value)
            self.max_slider.blockSignals(False)

        data_min, data_max = self.display_data_min, self.display_data_max
        span = max(data_max - data_min, 1e-12)
        self.display_limits = (
            data_min + span * min_value / 1000.0,
            data_min + span * max_value / 1000.0,
        )
        self.update_intensity_labels()
        self.refresh_images()

    def auto_display_limits(self):
        self.display_limits = self.parent().current_display_limits()
        self.set_display_sliders_from_limits()
        self.refresh_images()

    def select_nan_rectangle(self, start, end, mask_source_image=None):
        x0, y0 = start
        x1, y1 = end
        ny, nx = self.final_image.shape
        xmin = max(0, int(np.floor(min(x0, x1))))
        xmax = min(nx, int(np.ceil(max(x0, x1))) + 1)
        ymin = max(0, int(np.floor(min(y0, y1))))
        ymax = min(ny, int(np.ceil(max(y0, y1))) + 1)
        region = np.zeros(self.final_image.shape, dtype=bool)
        source_mask = visual_blank_mask(mask_source_image) if mask_source_image is not None else self.remaining_nan_mask()
        if xmin < xmax and ymin < ymax:
            region[ymin:ymax, xmin:xmax] = source_mask[ymin:ymax, xmin:xmax]

        if np.any(region):
            self.selected_region = region
            count = int(np.count_nonzero(region))
            self.status_label.setText(f"Selected {count} white/NaN pixel(s). Right-click the selection to choose a symmetry.")
        else:
            self.selected_region = None
            self.status_label.setText("No white/NaN pixels in this selection.")
        self.refresh_images()

    def open_symmetry_menu(self, event, mask_source_image=None):
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        nan_mask = visual_blank_mask(mask_source_image) if mask_source_image is not None else self.remaining_nan_mask()
        clicked_nan = 0 <= y < nan_mask.shape[0] and 0 <= x < nan_mask.shape[1] and nan_mask[y, x]

        if self.selected_region is None or not np.any(self.selected_region):
            if not clicked_nan:
                self.status_label.setText("No white/NaN zone at this position.")
                return
            self.selected_region = connected_nan_region(nan_mask, x, y)
            self.refresh_images()
        elif not (0 <= y < self.selected_region.shape[0] and 0 <= x < self.selected_region.shape[1] and self.selected_region[y, x]):
            if not clicked_nan:
                self.status_label.setText("Right-click inside the selected zone, or directly on another white/NaN zone.")
                return
            self.selected_region = connected_nan_region(nan_mask, x, y)
            self.refresh_images()

        menu = QMenu(self)
        central_action = menu.addAction("Central symmetry")
        horizontal_action = menu.addAction("Horizontal axial symmetry")
        vertical_action = menu.addAction("Vertical axial symmetry")
        chosen = menu.exec(event.guiEvent.globalPos() if event.guiEvent is not None else self.mapToGlobal(self.rect().center()))
        if chosen is central_action:
            self.apply_symmetry_to_selected_region("central")
        elif chosen is horizontal_action:
            self.apply_symmetry_to_selected_region("horizontal")
        elif chosen is vertical_action:
            self.apply_symmetry_to_selected_region("vertical")

    def apply_symmetry_to_selected_region(self, mode):
        if self.selected_region is None or not np.any(self.selected_region):
            return
        parent = self.parent()
        self.final_image = fill_region_by_symmetry(
            self.final_image,
            self.final_image,
            self.selected_region,
            mode,
            parent.xc_spin.value(),
            parent.yc_spin.value(),
        )
        remaining = int(np.count_nonzero(~np.isfinite(self.final_image)))
        self.status_label.setText(f"Applied {mode} symmetry. Remaining NaN pixels: {remaining}.")
        self.selected_region = None
        self.refresh_images()

    def reset_custom_fills(self):
        self.final_image = self.caved_image.copy()
        self.selected_region = None
        self.refresh_images()

    def refresh_images(self):
        vmin, vmax = self.display_limits
        parent = self.parent()
        xc = parent.xc_spin.value()
        yc = parent.yc_spin.value()
        angle = parent.cave_angle_spin.value()
        original_nan_view = self.source_image.copy()
        original_nan_view[~np.isfinite(self.caved_image)] = np.nan
        self.original_canvas.show_image(original_nan_view, vmin=vmin, vmax=vmax, region_mask=self.selected_region, xc=xc, yc=yc, reference_angle_deg=angle)
        self.result_canvas.show_image(self.final_image, vmin=vmin, vmax=vmax, region_mask=self.selected_region, xc=xc, yc=yc, reference_angle_deg=angle)

    def save_cave_plus(self):
        parent = self.parent()
        if parent.current_file is None:
            return
        try:
            frame_suffix = f"_frame{parent.frame_spin.value():04d}" if getattr(parent, "h5_n_frames", 1) > 1 else ""
            if parent.file_type == "EDF":
                output_path = parent.current_file.parent / f"{parent.current_file.stem}{frame_suffix}_cave+.edf"
                write_edf_file(output_path, sanitize_cave_output_image(self.final_image), parent.raw_header_text, parent.byte_order)
            else:
                output_path = parent.current_file.parent / f"{parent.current_file.stem}{frame_suffix}_cave+.h5"
                write_h5_frame_file(
                    output_path,
                    self.final_image,
                    parent.current_file,
                    parent.h5_dataset_name or "data",
                    parent.frame_spin.value() - 1,
                )
            parent.image_filled = self.final_image.copy()
            parent.canvas_cave.show_image(
                parent.image_filled,
                parent.xc_spin.value(),
                parent.yc_spin.value(),
                vmin=self.display_limits[0],
                vmax=self.display_limits[1],
                reference_angle_deg=parent.cave_angle_spin.value(),
            )
            parent.status.append(f"\nSaved cave+:\n{output_path}")
            self.status_label.setText(f"Saved cave+: {output_path}")
        except Exception as error:
            QMessageBox.critical(self, "Save cave+ error", str(error))


class ManualCaveDialog(QDialog):
    def __init__(self, parent, image, filled_image, shapes, exclusion_shapes, pre_nan_shapes, display_limits):
        super().__init__(parent)
        self.setWindowTitle("Manual cave mask")
        self.resize(1100, 620)
        self.source_image = np.asarray(image, dtype=np.float64)
        self.base_filled_image = np.asarray(filled_image, dtype=np.float64)
        self.shapes = [self.copy_shape(shape, mode="include") for shape in shapes]
        self.shapes.extend(self.copy_shape(shape, mode="exclude") for shape in exclusion_shapes)
        self.shapes.extend(self.copy_shape(shape, mode="pre_nan") for shape in pre_nan_shapes)
        self.current_tool = "Rectangle"
        self.current_mask_mode = "include"
        self.selected_shape_index = None
        self.active_polygon = []
        self._updating_shape_list = False
        self.display_limits = display_limits
        self.display_data_min, self.display_data_max = self.compute_display_range()
        self._syncing_view = False
        self.synced_xlim = None
        self.synced_ylim = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(6)
        self.select_button = QPushButton("↔")
        self.select_button.setToolTip("Move or resize selected shape")
        self.rect_button = QPushButton("▭")
        self.rect_button.setToolTip("Rectangle")
        self.vband_button = QPushButton("▏")
        self.vband_button.setToolTip("Vertical band")
        self.hband_button = QPushButton("▔")
        self.hband_button.setToolTip("Horizontal band")
        self.poly_button = QPushButton("⬠")
        self.poly_button.setToolTip("Polygon")
        self.finish_poly_button = QPushButton("✓")
        self.finish_poly_button.setToolTip("Finish polygon")
        self.include_mode_button = QPushButton("Cave +")
        self.include_mode_button.setToolTip("Draw zones to cave-fill")
        self.exclude_mode_button = QPushButton("Exclude -")
        self.exclude_mode_button.setToolTip("Draw zones that must not be cave-filled")
        self.pre_nan_mode_button = QPushButton("NaN")
        self.pre_nan_mode_button.setToolTip("Draw zones forced to NaN before cave filling")
        self.clear_button = QPushButton("Clear")
        self.load_mask_button = QPushButton("Load mask")
        self.save_mask_button = QPushButton("💾 Save mask")
        self.apply_button = QPushButton("Apply")
        self.multicave_button = QPushButton("MultiCave")
        self.multicave_button.setToolTip(
            "Apply the ID13 cave frame by frame on the current H5 file and save each caved frame in a folder."
        )
        self.multicave_progress = QProgressBar()
        self.multicave_progress.setRange(0, 1)
        self.multicave_progress.setValue(0)
        self.multicave_progress.setTextVisible(True)
        self.close_button = QPushButton("Close")

        self.select_button.setCheckable(True)
        self.rect_button.setCheckable(True)
        self.vband_button.setCheckable(True)
        self.hband_button.setCheckable(True)
        self.poly_button.setCheckable(True)
        self.rect_button.setChecked(True)
        self.include_mode_button.setCheckable(True)
        self.exclude_mode_button.setCheckable(True)
        self.pre_nan_mode_button.setCheckable(True)
        self.include_mode_button.setChecked(True)

        for widget in [
            self.select_button,
            self.rect_button,
            self.vband_button,
            self.hband_button,
            self.poly_button,
            self.finish_poly_button,
        ]:
            widget.setFixedSize(36, 30)
        self.include_mode_button.setFixedSize(64, 30)
        self.exclude_mode_button.setFixedSize(76, 30)
        self.pre_nan_mode_button.setFixedSize(54, 30)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(4)
        left_panel = QVBoxLayout()
        left_panel.setContentsMargins(0, 0, 0, 0)
        left_panel.setSpacing(2)
        self.before_canvas = ManualCaveCanvas(self, "")
        self.after_canvas = ManualCaveCanvas(self, "")
        self.before_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.after_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        toolbar = NavigationToolbar(self.before_canvas, self)
        toolbar.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        for action in list(toolbar.actions()):
            text = action.text().lower()
            if "save" in text or "subplots" in text:
                toolbar.removeAction(action)
        emojiize_matplotlib_toolbar(toolbar, remove_customize=True)

        if hasattr(toolbar, "locLabel"):
            toolbar.locLabel.hide()
        if hasattr(toolbar, "_message_label"):
            toolbar._message_label.hide()

        for label in toolbar.findChildren(QLabel):
            label.hide()

        toolbar.set_message = lambda message: None

        toolbar_box = QWidget()
        toolbar_box.setFixedHeight(48)
        toolbar_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        toolbar_box.setStyleSheet("""
            QWidget {
                background-color: #eeeeee;
                border: none;
                border-radius: 8px;
            }
            QToolButton {
                background-color: transparent;
                border: none;
                padding: 2px;
            }
            QToolButton:hover {
                background-color: #dddddd;
                border-radius: 4px;
            }
            QToolButton:pressed {
                background-color: #d0d0d0;
                border-radius: 4px;
            }
            QPushButton {
                background-color: #dddddd;
                border: none;
                border-radius: 4px;
                padding: 2px 8px;
            }
            QPushButton:hover {
                background-color: #d4d4d4;
            }
            QPushButton:pressed, QPushButton:checked {
                background-color: #cfcfcf;
            }
        """)
        toolbar_row = QHBoxLayout(toolbar_box)
        toolbar_row.setContentsMargins(8, 4, 8, 4)
        toolbar_row.setSpacing(6)
        toolbar_row.addWidget(toolbar, 0)
        toolbar_row.addStretch(1)
        toolbar_row.addWidget(self.select_button)
        toolbar_row.addWidget(self.rect_button)
        toolbar_row.addWidget(self.vband_button)
        toolbar_row.addWidget(self.hband_button)
        toolbar_row.addWidget(self.poly_button)
        toolbar_row.addWidget(self.finish_poly_button)
        toolbar_row.addWidget(self.include_mode_button)
        toolbar_row.addWidget(self.exclude_mode_button)
        toolbar_row.addWidget(self.pre_nan_mode_button)

        layout.addWidget(toolbar_box, 0)

        self.before_coordinate_label = QLabel("x = - | y = - | q = - | I = -")
        self.before_coordinate_label.setMinimumHeight(28)
        self.before_coordinate_label.setAlignment(Qt.AlignCenter)
        self.before_coordinate_label.setStyleSheet("""
            QLabel {
                background-color: #f4f4f4;
                border-radius: 8px;
                padding: 6px;
                font-family: Menlo, Monaco, monospace;
                font-size: 11px;
            }
        """)
        self.before_canvas.set_coordinate_label(self.before_coordinate_label, "")
        left_panel.addWidget(self.before_canvas, 1)
        left_panel.addWidget(self.before_coordinate_label, 0)
        body.addLayout(left_panel, 1)
        arrow_label = QLabel("→")
        arrow_label.setAlignment(Qt.AlignCenter)
        arrow_label.setFixedWidth(24)
        arrow_label.setStyleSheet("""
            QLabel {
                color: #444444;
                font-size: 22px;
                font-weight: 700;
            }
        """)
        body.addWidget(arrow_label, 0)

        right_panel = QVBoxLayout()
        right_panel.setContentsMargins(0, 0, 0, 0)
        right_panel.setSpacing(2)
        self.after_coordinate_label = QLabel("x = - | y = - | q = - | I = -")
        self.after_coordinate_label.setMinimumHeight(28)
        self.after_coordinate_label.setAlignment(Qt.AlignCenter)
        self.after_coordinate_label.setStyleSheet("""
            QLabel {
                background-color: #f4f4f4;
                border-radius: 8px;
                padding: 6px;
                font-family: Menlo, Monaco, monospace;
                font-size: 11px;
            }
        """)
        self.after_canvas.set_coordinate_label(self.after_coordinate_label, "")
        right_panel.addWidget(self.after_canvas, 1)
        right_panel.addWidget(self.after_coordinate_label, 0)
        body.addLayout(right_panel, 1)

        side = QVBoxLayout()
        side.setContentsMargins(0, 0, 0, 0)
        side.setSpacing(8)
        side.addWidget(QLabel("Shapes"))
        self.shape_list = QListWidget()
        self.shape_list.currentRowChanged.connect(self.shape_list_row_changed)
        side.addWidget(self.shape_list, 1)
        for action_button in [
            self.clear_button,
            self.load_mask_button,
            self.save_mask_button,
            self.apply_button,
            self.multicave_button,
            self.close_button,
        ]:
            action_button.setMinimumHeight(28)
            action_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        side.addWidget(self.clear_button)
        side.addWidget(self.load_mask_button)
        side.addWidget(self.save_mask_button)
        side.addWidget(self.apply_button)
        side.addWidget(self.multicave_button)
        side.addWidget(self.multicave_progress)
        side.addWidget(self.close_button)
        body.addLayout(side)
        layout.addLayout(body, 1)

        intensity_layout = QGridLayout()
        intensity_layout.setContentsMargins(0, 0, 0, 0)
        intensity_layout.setHorizontalSpacing(8)
        intensity_layout.setVerticalSpacing(2)
        self.min_label = QLabel()
        self.max_label = QLabel()
        self.min_slider = QSlider(Qt.Horizontal)
        self.max_slider = QSlider(Qt.Horizontal)
        self.min_slider.setRange(0, 1000)
        self.max_slider.setRange(0, 1000)
        self.auto_button = QPushButton("Auto")
        intensity_layout.addWidget(self.min_label, 0, 0)
        intensity_layout.addWidget(self.min_slider, 0, 1)
        intensity_layout.addWidget(self.max_label, 1, 0)
        intensity_layout.addWidget(self.max_slider, 1, 1)
        intensity_layout.addWidget(self.auto_button, 0, 2, 2, 1)
        layout.addLayout(intensity_layout)

        self.select_button.clicked.connect(lambda: self.set_tool("Select"))
        self.rect_button.clicked.connect(lambda: self.set_tool("Rectangle"))
        self.vband_button.clicked.connect(lambda: self.set_tool("Vertical band"))
        self.hband_button.clicked.connect(lambda: self.set_tool("Horizontal band"))
        self.poly_button.clicked.connect(lambda: self.set_tool("Polygon"))
        self.finish_poly_button.clicked.connect(self.finish_polygon)
        self.include_mode_button.clicked.connect(lambda: self.set_mask_mode("include"))
        self.exclude_mode_button.clicked.connect(lambda: self.set_mask_mode("exclude"))
        self.pre_nan_mode_button.clicked.connect(lambda: self.set_mask_mode("pre_nan"))
        self.clear_button.clicked.connect(self.clear_shapes)
        self.load_mask_button.clicked.connect(self.load_mask_preset)
        self.save_mask_button.clicked.connect(self.save_mask_preset)
        self.apply_button.clicked.connect(self.apply_to_parent)
        self.multicave_button.clicked.connect(self.run_multicave_current_file)
        self.close_button.clicked.connect(self.reject)
        self.min_slider.valueChanged.connect(self.update_display_limits_from_sliders)
        self.max_slider.valueChanged.connect(self.update_display_limits_from_sliders)
        self.auto_button.clicked.connect(self.auto_display_limits)

        self.refresh_shape_list()
        self.set_display_sliders_from_limits()
        self.refresh_preview()

    def combined_manual_mask_for_shape(self, shape, mode="include"):
        mask = np.zeros(shape, dtype=bool)

        for manual_shape in self.shapes:
            if manual_shape.get("mode", "include") == mode:
                self.shape_to_mask_single(mask, manual_shape)

        return mask

    def run_multicave_current_file(self):
        parent = self.parent()

        if parent.current_file is None or str(parent.file_type).upper() != "H5":
            QMessageBox.warning(self, "MultiCave", "MultiCave is only available for the current H5 file.")
            return

        if parent.h5_dataset_name is None:
            QMessageBox.warning(self, "MultiCave", "No H5 dataset is currently loaded.")
            return

        total_frames = int(getattr(parent, "h5_n_frames", 1) or 1)
        if total_frames <= 0:
            QMessageBox.warning(self, "MultiCave", "No frame found in the current H5 file.")
            return

        source_path = Path(parent.current_file)
        output_dir = source_path.with_name(f"{source_path.stem}_cave_frames")
        output_dir.mkdir(parents=True, exist_ok=True)

        self.multicave_button.setEnabled(False)
        self.multicave_progress.setRange(0, total_frames)
        self.multicave_progress.setValue(0)
        self.multicave_progress.setFormat(f"0 / {total_frames}")
        QCoreApplication.processEvents()

        previous_frame = parent.frame_slider.value() if hasattr(parent, "frame_slider") else 1
        saved_count = 0

        try:
            for frame_index in range(total_frames):
                image, _ = read_h5_frame(parent.current_file, parent.h5_dataset_name, frame_index, add_matching_center=False)
                manual_mask = self.combined_manual_mask_for_shape(image.shape, "include")
                exclusion_mask = self.combined_manual_mask_for_shape(image.shape, "exclude")
                pre_nan_mask = self.combined_manual_mask_for_shape(image.shape, "pre_nan")
                extra_operator, extra_threshold = parent.extra_nan_condition()

                _, filled, _ = apply_central_symmetry_cave(
                    image,
                    parent.xc_spin.value(),
                    parent.yc_spin.value(),
                    nan_operator=parent.nan_operator_combo.currentText(),
                    nan_threshold=parent.nan_threshold_spin.value(),
                    nan_operator_2=extra_operator,
                    nan_threshold_2=extra_threshold,
                    use_id13_beamstop=True,
                    beamstop_y=parent.beamstop_y_spin.value(),
                    reference_angle_deg=parent.cave_angle_spin.value(),
                    expand_nan_neighbors=parent.expand_nan_neighbors_checkbox.isChecked(),
                    pre_nan_mask=pre_nan_mask,
                    extra_mask=manual_mask,
                    exclude_mask=exclusion_mask,
                )

                output_file = output_dir / f"{source_path.stem}_frame{frame_index + 1:04d}_cave.h5"
                write_h5_frame_file(
                    output_file,
                    filled,
                    parent.current_file,
                    parent.h5_dataset_name,
                    frame_index,
                )
                saved_count += 1

                self.multicave_progress.setValue(frame_index + 1)
                self.multicave_progress.setFormat(f"{frame_index + 1} / {total_frames}")
                QCoreApplication.processEvents()

            if hasattr(parent, "frame_slider"):
                parent.frame_slider.setValue(previous_frame)

            parent.status.append(
                f"MultiCave done: {saved_count} H5 frames saved in {output_dir}"
            )
            QMessageBox.information(
                self,
                "MultiCave",
                f"MultiCave finished.\n\n{saved_count} H5 files saved in:\n{output_dir}",
            )
        except Exception as exc:
            QMessageBox.critical(self, "MultiCave error", str(exc))
        finally:
            self.multicave_button.setEnabled(True)

    def copy_shape(self, shape, mode=None):
        if shape["type"] == "rect":
            points = tuple(float(value) for value in shape["points"])
        elif shape["type"] in ("vband", "hband"):
            points = tuple(float(value) for value in shape["points"])
        else:
            points = [(float(x), float(y)) for x, y in shape["points"]]
        return {"type": shape["type"], "points": points, "mode": mode or shape.get("mode", "include")}

    def shape_to_serializable(self, shape):
        copied = self.copy_shape(shape)
        if isinstance(copied["points"], tuple):
            points = list(copied["points"])
        else:
            points = [[float(x), float(y)] for x, y in copied["points"]]
        return {
            "type": copied["type"],
            "mode": copied.get("mode", "include"),
            "points": points,
        }

    def save_mask_preset(self):
        if not self.shapes:
            QMessageBox.warning(self, "Save mask", "No mask shape to save.")
            return

        default_path = cave_mask_preset_dir() / "cave_mask.json"
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save cave mask",
            str(default_path),
            "Cave mask JSON (*.json);;All files (*)",
        )
        if not output_path:
            return

        output_path = Path(output_path).expanduser()
        if output_path.suffix.lower() != ".json":
            output_path = output_path.with_suffix(".json")

        payload = {
            "format": "LRPhoton cave mask",
            "version": CAVE_MASK_PRESET_VERSION,
            "image_shape": list(self.source_image.shape),
            "shapes": [self.shape_to_serializable(shape) for shape in self.shapes],
        }

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as file:
                json.dump(payload, file, indent=2)
            self.parent().status.append(f"Saved cave mask preset:\n{output_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save mask error", str(exc))

    def load_mask_preset(self):
        input_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load cave mask",
            str(cave_mask_preset_dir()),
            "Cave mask JSON (*.json);;All files (*)",
        )
        if not input_path:
            return

        try:
            with open(input_path, "r", encoding="utf-8") as file:
                payload = json.load(file)

            if payload.get("format") != "LRPhoton cave mask":
                raise ValueError("This file is not an LRPhoton cave mask preset.")

            shapes = payload.get("shapes", [])
            if not isinstance(shapes, list):
                raise ValueError("Invalid mask preset: shapes must be a list.")

            self.shapes = [self.copy_shape(shape) for shape in shapes]
            self.selected_shape_index = 0 if self.shapes else None
            self.active_polygon = []
            self.refresh_shape_list()
            if self.selected_shape_index is not None:
                self.shape_list.setCurrentRow(self.selected_shape_index)
                self.set_tool("Select")
            self.refresh_preview()
            self.parent().status.append(f"Loaded cave mask preset:\n{input_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Load mask error", str(exc))

    def mode_color(self, mode):
        if mode == "exclude":
            return "#ff2d55"
        if mode == "pre_nan":
            return "#ffd400"
        return "#00ffff"

    def mode_edge_color(self, shape):
        mode = shape.get("mode", "include")
        if mode == "exclude":
            return "#b00030"
        if mode == "pre_nan":
            return "#b08900"
        return "#00a0a0"

    def set_mask_mode(self, mode):
        self.current_mask_mode = mode
        self.include_mode_button.setChecked(mode == "include")
        self.exclude_mode_button.setChecked(mode == "exclude")
        self.pre_nan_mode_button.setChecked(mode == "pre_nan")
        self.refresh_preview()

    def set_tool(self, tool):
        self.current_tool = tool
        self.select_button.setChecked(tool == "Select")
        self.rect_button.setChecked(tool == "Rectangle")
        self.vband_button.setChecked(tool == "Vertical band")
        self.hband_button.setChecked(tool == "Horizontal band")
        self.poly_button.setChecked(tool == "Polygon")
        self.active_polygon = []
        self.refresh_preview()

    def compute_display_range(self):
        display = self.source_image.astype(np.float64).copy()
        display[~np.isfinite(display)] = np.nan
        display[display < 0] = np.nan
        with np.errstate(invalid="ignore", divide="ignore"):
            display = np.log10(display + 1)
        finite = display[np.isfinite(display)]
        if finite.size == 0:
            return 0.0, 1.0
        return float(np.nanmin(finite)), float(np.nanmax(finite))

    def set_display_sliders_from_limits(self):
        data_min, data_max = self.display_data_min, self.display_data_max
        span = max(data_max - data_min, 1e-12)
        vmin, vmax = self.display_limits
        min_value = int(np.clip((vmin - data_min) / span * 1000.0, 0, 1000))
        max_value = int(np.clip((vmax - data_min) / span * 1000.0, 0, 1000))
        if max_value <= min_value:
            max_value = min(1000, min_value + 1)
        self.min_slider.blockSignals(True)
        self.max_slider.blockSignals(True)
        self.min_slider.setValue(min_value)
        self.max_slider.setValue(max_value)
        self.min_slider.blockSignals(False)
        self.max_slider.blockSignals(False)
        self.update_intensity_labels()

    def update_intensity_labels(self):
        vmin, vmax = self.display_limits
        self.min_label.setText(f"Min: {vmin:.3g}")
        self.max_label.setText(f"Max: {vmax:.3g}")

    def update_display_limits_from_sliders(self):
        min_value = self.min_slider.value()
        max_value = self.max_slider.value()
        if max_value <= min_value:
            max_value = min(1000, min_value + 1)
            self.max_slider.blockSignals(True)
            self.max_slider.setValue(max_value)
            self.max_slider.blockSignals(False)

        data_min, data_max = self.display_data_min, self.display_data_max
        span = max(data_max - data_min, 1e-12)
        self.display_limits = (
            data_min + span * min_value / 1000.0,
            data_min + span * max_value / 1000.0,
        )
        self.update_intensity_labels()
        self.refresh_preview()

    def auto_display_limits(self):
        self.display_limits = self.parent().current_display_limits()
        self.set_display_sliders_from_limits()
        self.refresh_preview()

    def add_shape(self, shape_type, points):
        if shape_type == "vband" and len(points) == 4:
            x0, y0, x1, y1 = points
            points = (x0, 0.0, x1, float(self.source_image.shape[0]))
        elif shape_type == "hband" and len(points) == 4:
            x0, y0, x1, y1 = points
            points = (0.0, y0, float(self.source_image.shape[1]), y1)

        self.shapes.append({"type": shape_type, "points": points, "mode": self.current_mask_mode})
        self.selected_shape_index = len(self.shapes) - 1
        self.refresh_shape_list()
        self.shape_list.setCurrentRow(self.selected_shape_index)
        self.set_tool("Select")
        self.refresh_preview()

    def finish_polygon(self):
        if len(self.active_polygon) >= 3:
            self.add_shape("poly", list(self.active_polygon))
        self.active_polygon = []
        self.refresh_preview()

    def delete_selected_shape(self):
        row = self.shape_list.currentRow()
        self.delete_shape(row)

    def delete_shape(self, row):
        if 0 <= row < len(self.shapes):
            del self.shapes[row]
            if not self.shapes:
                self.selected_shape_index = None
            elif self.selected_shape_index is None:
                self.selected_shape_index = min(row, len(self.shapes) - 1)
            elif self.selected_shape_index >= len(self.shapes):
                self.selected_shape_index = len(self.shapes) - 1
            self.refresh_shape_list()
            self.refresh_preview()

    def clear_shapes(self):
        self.shapes = []
        self.active_polygon = []
        self.selected_shape_index = None
        self.refresh_shape_list()
        self.refresh_preview()

    def refresh_shape_list(self):
        previous_row = self.selected_shape_index
        self._updating_shape_list = True
        self.shape_list.clear()
        for index, shape in enumerate(self.shapes, 1):
            labels = {
                "rect": "Rectangle",
                "poly": "Polygon",
                "vband": "Vertical band",
                "hband": "Horizontal band",
            }
            label = labels.get(shape["type"], "Shape")
            mode_names = {
                "include": "Cave",
                "exclude": "Exclude",
                "pre_nan": "NaN",
            }
            mode_label = mode_names.get(shape.get("mode", "include"), "Cave")
            item = QListWidgetItem()
            item.setSizeHint(QSize(240, 28))
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(4, 2, 4, 2)
            row_layout.setSpacing(6)

            label_widget = QLabel(f"{index:02d} - {mode_label} - {label}")
            row_layout.addWidget(label_widget, 1)

            remove_button = QPushButton("−")
            remove_button.setFixedSize(22, 18)
            remove_button.setToolTip("Remove this shape")
            remove_button.setStyleSheet("""
                QPushButton {
                    background: #ffecec;
                    color: #b00020;
                    border: 1px solid #ffb3b3;
                    border-radius: 8px;
                    font-weight: bold;
                    font-size: 11px;
                    padding: 0px;
                }
                QPushButton:hover {
                    background: #ffd6d6;
                }
            """)
            remove_button.clicked.connect(lambda checked=False, row=index - 1: self.delete_shape(row))
            row_layout.addWidget(remove_button, 0, Qt.AlignCenter)

            self.shape_list.addItem(item)
            self.shape_list.setItemWidget(item, row_widget)
        self._updating_shape_list = False
        if previous_row is not None and 0 <= previous_row < len(self.shapes):
            self.shape_list.setCurrentRow(previous_row)

    def shape_list_row_changed(self, row):
        if self._updating_shape_list:
            return
        if 0 <= row < len(self.shapes):
            self.set_tool("Select")
            self.selected_shape_index = row
        else:
            self.selected_shape_index = None
        self.refresh_preview()

    def select_shape(self, index):
        if 0 <= index < len(self.shapes):
            self.selected_shape_index = index
            self.shape_list.setCurrentRow(index)
        else:
            self.selected_shape_index = None
        self.refresh_preview()

    def shape_handles(self, shape):
        if shape["type"] == "rect":
            x0, y0, x1, y1 = shape["points"]
            return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

        polygon = self.band_polygon(shape) if shape["type"] in ("vband", "hband") else shape["points"]
        return list(polygon)

    def hit_test_shape(self, x, y):
        for index in range(len(self.shapes) - 1, -1, -1):
            shape = self.shapes[index]
            for handle_index, (hx, hy) in enumerate(self.shape_handles(shape)):
                if abs(x - hx) <= 8 and abs(y - hy) <= 8:
                    return index, handle_index

            mask = np.zeros(self.source_image.shape, dtype=bool)
            self.shape_to_mask_single(mask, shape)
            xi = int(round(x))
            yi = int(round(y))
            if 0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1] and mask[yi, xi]:
                return index, None

        return None

    def edit_shape(self, index, handle, x, y, dx, dy):
        if not 0 <= index < len(self.shapes):
            return

        shape = self.shapes[index]
        if shape["type"] == "rect":
            x0, y0, x1, y1 = shape["points"]
            if handle is None:
                shape["points"] = (x0 + dx, y0 + dy, x1 + dx, y1 + dy)
            elif handle == 0:
                shape["points"] = (x, y, x1, y1)
            elif handle == 1:
                shape["points"] = (x0, y, x, y1)
            elif handle == 2:
                shape["points"] = (x0, y0, x, y)
            else:
                shape["points"] = (x, y0, x1, y)
        elif shape["type"] == "vband":
            x0, _y0, x1, _y1 = shape["points"][:4]
            ny = float(self.source_image.shape[0])
            if handle is None:
                shape["points"] = (x0 + dx, 0.0, x1 + dx, ny)
            elif handle in (0, 3):
                shape["points"] = (x, 0.0, x1, ny)
            else:
                shape["points"] = (x0, 0.0, x, ny)
        elif shape["type"] == "hband":
            _x0, y0, _x1, y1 = shape["points"][:4]
            nx = float(self.source_image.shape[1])
            if handle is None:
                shape["points"] = (0.0, y0 + dy, nx, y1 + dy)
            elif handle in (0, 1):
                shape["points"] = (0.0, y, nx, y1)
            else:
                shape["points"] = (0.0, y0, nx, y)
        else:
            points = list(shape["points"])
            if handle is None:
                shape["points"] = [(px + dx, py + dy) for px, py in points]
            elif 0 <= handle < len(points):
                points[handle] = (x, y)
                shape["points"] = points

        self.refresh_shape_list()
        self.shape_list.setCurrentRow(index)
        self.selected_shape_index = index
        self.refresh_preview()

    def band_polygon(self, shape):
        ny, nx = self.source_image.shape

        if shape["type"] == "vband":
            x0, _y0, x1, _y1 = shape["points"][:4]
            xmin = max(0.0, min(float(x0), float(x1)))
            xmax = min(float(nx), max(float(x0), float(x1)))
            return [
                (xmin, 0.0),
                (xmax, 0.0),
                (xmax, float(ny)),
                (xmin, float(ny)),
            ]

        if shape["type"] == "hband":
            _x0, y0, _x1, y1 = shape["points"][:4]
            ymin = max(0.0, min(float(y0), float(y1)))
            ymax = min(float(ny), max(float(y0), float(y1)))
            return [
                (0.0, ymin),
                (float(nx), ymin),
                (float(nx), ymax),
                (0.0, ymax),
            ]

        return shape["points"]

    def set_synced_limits(self, xlim, ylim):
        self.synced_xlim = tuple(xlim)
        self.synced_ylim = tuple(ylim)
        for canvas in (self.before_canvas, self.after_canvas):
            canvas.ax.set_xlim(self.synced_xlim)
            canvas.ax.set_ylim(self.synced_ylim)
            if canvas.image is not None:
                constrain_image_axes(canvas.ax, canvas.image.shape)
                self.synced_xlim = tuple(canvas.ax.get_xlim())
                self.synced_ylim = tuple(canvas.ax.get_ylim())
            canvas.draw_idle()

    def reset_synced_view(self):
        self.synced_xlim = None
        self.synced_ylim = None
        self.refresh_preview()

    def apply_synced_view(self, source=None):
        if self.synced_xlim is None or self.synced_ylim is None:
            if source is not None:
                self.synced_xlim = tuple(source.ax.get_xlim())
                self.synced_ylim = tuple(source.ax.get_ylim())
            return

        for canvas in (self.before_canvas, self.after_canvas):
            if canvas is source:
                continue
            canvas.ax.set_xlim(self.synced_xlim)
            canvas.ax.set_ylim(self.synced_ylim)
            if canvas.image is not None:
                constrain_image_axes(canvas.ax, canvas.image.shape)

    def shape_mask(self, mode="include"):
        mask = np.zeros(self.source_image.shape, dtype=bool)

        for shape in self.shapes:
            if shape.get("mode", "include") == mode:
                self.shape_to_mask_single(mask, shape)

        return mask

    def shape_to_mask_single(self, mask, shape):
        ny, nx = mask.shape
        if shape["type"] == "rect":
            x0, y0, x1, y1 = shape["points"]
            xmin = max(0, int(np.floor(min(x0, x1))))
            xmax = min(nx, int(np.ceil(max(x0, x1))))
            ymin = max(0, int(np.floor(min(y0, y1))))
            ymax = min(ny, int(np.ceil(max(y0, y1))))
            mask[ymin:ymax, xmin:xmax] = True
            return

        polygon_points = self.band_polygon(shape) if shape["type"] in ("vband", "hband") else shape["points"]
        polygon = np.asarray(polygon_points, dtype=float)
        if polygon.size == 0:
            return
        xmin = max(0, int(np.floor(np.nanmin(polygon[:, 0]))))
        xmax = min(nx, int(np.ceil(np.nanmax(polygon[:, 0]))) + 1)
        ymin = max(0, int(np.floor(np.nanmin(polygon[:, 1]))))
        ymax = min(ny, int(np.ceil(np.nanmax(polygon[:, 1]))) + 1)
        if xmin >= xmax or ymin >= ymax:
            return
        yy, xx = np.mgrid[ymin:ymax, xmin:xmax]
        points = np.column_stack((xx.ravel(), yy.ravel()))
        path = MplPath(polygon_points)
        mask[ymin:ymax, xmin:xmax] |= path.contains_points(points).reshape((ymax - ymin, xmax - xmin))

    def filled_image(self):
        parent = self.parent()
        parent.commit_nan_threshold_edits()
        use_id13_beamstop = parent.instrument_mode == "ID13" and parent.id13_beamstop_checkbox.isChecked()
        extra_operator, extra_threshold = parent.extra_nan_condition()
        _, filled, _ = apply_central_symmetry_cave(
            self.source_image,
            parent.xc_spin.value(),
            parent.yc_spin.value(),
            nan_operator=parent.nan_operator_combo.currentText(),
            nan_threshold=parent.nan_threshold_spin.value(),
            nan_operator_2=extra_operator,
            nan_threshold_2=extra_threshold,
            use_id13_beamstop=use_id13_beamstop,
            beamstop_y=parent.beamstop_y_spin.value(),
            reference_angle_deg=parent.cave_angle_spin.value(),
            expand_nan_neighbors=parent.expand_nan_neighbors_checkbox.isChecked(),
            pre_nan_mask=self.shape_mask("pre_nan"),
            extra_mask=self.shape_mask("include"),
            exclude_mask=self.shape_mask("exclude"),
        )
        return filled

    def refresh_preview(self):
        vmin, vmax = self.display_limits
        xc = self.parent().xc_spin.value()
        yc = self.parent().yc_spin.value()
        self.before_canvas.show_image(
            self.source_image,
            vmin=vmin,
            vmax=vmax,
            shapes=self.shapes,
            active_polygon=self.active_polygon,
            xc=xc,
            yc=yc,
            reference_angle_deg=self.parent().cave_angle_spin.value(),
        )
        self.after_canvas.show_image(
            self.filled_image(),
            vmin=vmin,
            vmax=vmax,
            shapes=[],
            active_polygon=None,
            xc=xc,
            yc=yc,
            reference_angle_deg=self.parent().cave_angle_spin.value(),
        )

        self.after_canvas.ax.set_xlim(self.before_canvas.ax.get_xlim())
        self.after_canvas.ax.set_ylim(self.before_canvas.ax.get_ylim())
        self.after_canvas.draw_idle()

    def apply_to_parent(self):

        self.parent().manual_cave_shapes = [
            self.parent().copy_shape_data(shape)
            for shape in self.shapes
            if shape.get("mode", "include") == "include"
        ]
        self.parent().manual_cave_exclusion_shapes = [
            self.parent().copy_shape_data(shape)
            for shape in self.shapes
            if shape.get("mode", "include") == "exclude"
        ]
        self.parent().manual_cave_pre_nan_shapes = [
            self.parent().copy_shape_data(shape)
            for shape in self.shapes
            if shape.get("mode", "include") == "pre_nan"
        ]

        self.parent().image_filled = self.filled_image()
        self.parent().update_manual_mask_status_label()

        self.parent().refresh_preview()

        self.accept()


# ============================================================
# =========================== CAVE TAB ========================
# ============================================================

class CaveTab(QWidget):
    """Cave tab: fill masked detector zones by central symmetry."""

    folder_changed = Signal(Path)

    def __init__(self):
        super().__init__()

        self.current_file = None
        self.current_folder = Path.home()
        self.file_type = None
        self.header = {}
        self.raw_header_text = ""
        self.byte_order = "LowByteFirst"
        self.h5_dataset_name = None
        self.h5_frame_axis = None
        self.h5_n_frames = 1
        self._edf_frames = None
        self._syncing_folder = False
        self._syncing_frame_controls = False
        self._batch_cave_running = False

        self.image = None
        self.image_clean = None
        self.image_filled = None
        self.cave_mask = None
        self.manual_cave_shapes = []
        self.manual_cave_exclusion_shapes = []
        self.manual_cave_pre_nan_shapes = []
        self.display_vmin = 0.0
        self.display_vmax = 1.0
        self.slider_scale = 1000

        self.instrument_mode = "XENOCS"

        self.build_ui()
        self.set_controls_enabled(False)
        self.update_centre_warning_labels()
        self.update_beamstop_visibility()
        self.update_manual_mask_status_label()

    def build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(*PAGE_MARGINS)
        main_layout.setSpacing(BLOCK_SPACING)

        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(BLOCK_SPACING)
        main_layout.addLayout(top_layout, stretch=1)

        original_box = QGroupBox("Original pattern")
        original_box.setMinimumHeight(0)
        original_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        original_layout = QVBoxLayout(original_box)
        original_layout.setContentsMargins(*GROUP_BOX_MARGINS)
        self.canvas_original = ImageCanvas()
        self.canvas_original.setMinimumHeight(0)
        self.canvas_original.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        self.original_coordinate_label = QLabel("x = - | y = - | q = - | I = -")
        self.original_coordinate_label.setMinimumHeight(28)
        self.original_coordinate_label.setAlignment(Qt.AlignCenter)
        self.original_coordinate_label.setStyleSheet("""
            QLabel {
                background-color: #f4f4f4;
                border-radius: 8px;
                padding: 6px;
                font-family: Menlo, Monaco, monospace;
                font-size: 11px;
            }
        """)
        self.canvas_original.set_coordinate_label(self.original_coordinate_label, "")
        self.canvas_original.set_q_calculator(self.calculate_q_at_pixel)
        original_layout.addWidget(self.canvas_original, stretch=1)
        original_layout.addWidget(self.original_coordinate_label, stretch=0)

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

        file_box = QGroupBox("File browser")
        file_box.setStyleSheet(GROUP_BOX_STYLE)
        file_box.setMinimumHeight(0)
        file_box.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        file_box.setMinimumHeight(120)
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

        self.show_subfolders_checkbox = QCheckBox("Show subfolders")
        self.show_subfolders_checkbox.setChecked(False)
        self.show_subfolders_checkbox.stateChanged.connect(self.refresh_files)
        self.only_thumbs_up_checkbox = QCheckBox("Only 👍")
        self.only_thumbs_up_checkbox.setChecked(False)
        self.only_thumbs_up_checkbox.stateChanged.connect(self.refresh_files)

        refresh_button = QPushButton("Refresh")
        cave_action_button_style = """
            QPushButton {
                background-color: #dddddd;
                border: none;
                border-radius: 6px;
                padding: 4px 10px;
            }
            QPushButton:hover {
                background-color: #d4d4d4;
            }
            QPushButton:pressed {
                background-color: #c8c8c8;
            }
            QPushButton:disabled {
                background-color: #eeeeee;
                color: #aaaaaa;
                border: none;
            }
        """
        refresh_button.setStyleSheet(cave_action_button_style)
        cave_action_button_height = refresh_button.sizeHint().height()
        refresh_button.setFixedHeight(cave_action_button_height)
        refresh_button.clicked.connect(self.refresh_files)

        filters_layout.addWidget(QLabel("Name:"), 0, 0)
        filters_layout.addWidget(self.name_filter, 0, 1)
        filters_layout.addWidget(QLabel("Extensions:"), 1, 0)
        filters_layout.addWidget(self.extension_filter, 1, 1)
        file_layout.addLayout(filters_layout)
        file_options_layout = QHBoxLayout()
        file_options_layout.setContentsMargins(0, 0, 0, 0)
        file_options_layout.setSpacing(10)
        file_options_layout.addWidget(self.show_subfolders_checkbox)
        file_options_layout.addWidget(self.only_thumbs_up_checkbox)
        file_options_layout.addStretch(1)
        file_layout.addLayout(file_options_layout)
        file_layout.addWidget(refresh_button)

        self.file_list = QListWidget()
        install_file_rating_menu(self.file_list)
        self.file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.file_list.itemClicked.connect(self.open_selected_file)
        self.file_list.setMinimumHeight(0)
        file_layout.addWidget(self.file_list, stretch=1)

        controls_box = QGroupBox("Cave tools")
        controls_box.setStyleSheet(GROUP_BOX_STYLE)
        controls_box.setMinimumHeight(0)
        controls_box.setMinimumWidth(0)
        controls_box.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        controls_box.setMinimumHeight(120)
        controls_box_layout = QVBoxLayout(controls_box)
        controls_box_layout.setContentsMargins(6, 18, 6, 6)
        controls_box_layout.setSpacing(0)
        controls_content = QWidget()
        controls_content.setStyleSheet("background-color: #eeeeee;")
        controls_content.setMinimumWidth(0)
        controls_content.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        controls_layout = QVBoxLayout(controls_content)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(4)

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
        controls_scroll.setWidget(controls_content)
        controls_box_layout.addWidget(controls_scroll)

        cave_box = QGroupBox("Cave-filled pattern")
        cave_box.setMinimumHeight(0)
        cave_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        cave_layout = QVBoxLayout(cave_box)
        cave_layout.setContentsMargins(*GROUP_BOX_MARGINS)
        self.canvas_cave = ImageCanvas()
        self.canvas_cave.setMinimumHeight(0)
        self.canvas_cave.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        self.cave_coordinate_label = QLabel("x = - | y = - | q = - | I = -")
        self.cave_coordinate_label.setMinimumHeight(28)
        self.cave_coordinate_label.setAlignment(Qt.AlignCenter)
        self.cave_coordinate_label.setStyleSheet("""
            QLabel {
                background-color: #f4f4f4;
                border-radius: 8px;
                padding: 6px;
                font-family: Menlo, Monaco, monospace;
                font-size: 11px;
            }
        """)
        self.canvas_cave.set_coordinate_label(self.cave_coordinate_label, "")
        self.canvas_cave.set_q_calculator(self.calculate_q_at_pixel)
        cave_layout.addWidget(self.canvas_cave, stretch=1)
        cave_layout.addWidget(self.cave_coordinate_label, stretch=0)
        self.canvas_original.set_sync_partner(self.canvas_cave)
        self.canvas_cave.set_sync_partner(self.canvas_original)

        top_layout.addWidget(original_box, stretch=1)
        center_splitter.addWidget(file_box)
        center_splitter.addWidget(controls_box)
        center_splitter.setStretchFactor(0, 1)
        center_splitter.setStretchFactor(1, 1)
        center_splitter.setSizes([1, 1])
        QTimer.singleShot(0, self.set_initial_center_splitter_sizes)

        top_layout.addWidget(center_panel, stretch=0)
        top_layout.addWidget(cave_box, stretch=1)
        top_layout.setStretch(0, 1)
        top_layout.setStretch(1, 0)
        top_layout.setStretch(2, 1)

        preset_layout = QHBoxLayout()
        preset_layout.setContentsMargins(0, 0, 0, 0)
        preset_layout.setSpacing(3)
        self.btn_xenocs = QPushButton("XENOCS")
        self.btn_id02 = QPushButton("ID02")
        self.btn_id13 = QPushButton("ID13")
        self.btn_custom = QPushButton("Custom")
        self.q_manual_button = QPushButton("+")
        self.q_manual_button.clicked.connect(self.use_custom_cave_mask)

        for button in [self.btn_xenocs, self.btn_id02, self.btn_id13, self.btn_custom]:
            button.setCheckable(True)
            preset_layout.addWidget(button)
        preset_layout.addWidget(self.q_manual_button)
        for button in [self.btn_xenocs, self.btn_id02, self.btn_id13, self.btn_custom, self.q_manual_button]:
            button.hide()
        self.line_geometry_selector = LineGeometrySelector(self, "XENOCS")
        self.line_geometry_selector.geometry_selected.connect(self.apply_line_geometry_selection)
        self.line_geometry_selector.setMinimumWidth(0)
        self.line_geometry_selector.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.line_geometry_selector.combo.setMinimumWidth(0)
        self.line_geometry_selector.combo.setMinimumContentsLength(7)
        self.line_geometry_selector.combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
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
        compact_widths = {
            self.btn_xenocs: 66,
            self.btn_id02: 48,
            self.btn_id13: 48,
            self.btn_custom: 60,
            self.q_manual_button: 24,
        }
        self.compact_preset_buttons(compact_widths)
        controls_layout.addLayout(preset_layout)

        self.xc_spin = QDoubleSpinBox()
        self.xc_spin.setRange(-100000, 100000)
        self.xc_spin.setDecimals(13)
        self.xc_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.xc_spin.setMaximumWidth(148)

        self.yc_spin = QDoubleSpinBox()
        self.yc_spin.setRange(-100000, 100000)
        self.yc_spin.setDecimals(13)
        self.yc_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.yc_spin.setMaximumWidth(148)

        self.beamstop_y_spin = QDoubleSpinBox()
        self.beamstop_y_spin.setRange(0, 100000)
        self.beamstop_y_spin.setDecimals(0)
        self.beamstop_y_spin.setValue(1376)
        self.beamstop_y_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.beamstop_y_spin.setMaximumWidth(148)

        self.cave_angle_spin = QDoubleSpinBox()
        self.cave_angle_spin.setRange(-180.0, 180.0)
        self.cave_angle_spin.setDecimals(3)
        self.cave_angle_spin.setValue(0.0)
        self.cave_angle_spin.setSuffix(" deg")
        self.cave_angle_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.cave_angle_spin.setMaximumWidth(148)

        self.centre_x_label = QLabel("Center X:")
        self.centre_y_label = QLabel("Center Y:")

        form_layout = QGridLayout()
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.setHorizontalSpacing(4)
        form_layout.setVerticalSpacing(4)
        form_layout.setColumnStretch(0, 0)
        form_layout.setColumnStretch(1, 1)
        form_layout.addWidget(self.centre_x_label, 0, 0)
        form_layout.addWidget(self.xc_spin, 0, 1)
        form_layout.addWidget(self.centre_y_label, 1, 0)
        form_layout.addWidget(self.yc_spin, 1, 1)
        self.beamstop_y_label = QLabel("ID13 beamstop Y:")
        self.beamstop_y_label.setWordWrap(True)
        form_layout.addWidget(self.beamstop_y_label, 2, 0)
        form_layout.addWidget(self.beamstop_y_spin, 2, 1)
        self.cave_angle_label = QLabel("Cave angle:")
        form_layout.addWidget(self.cave_angle_label, 3, 0)
        form_layout.addWidget(self.cave_angle_spin, 3, 1)
        # Hide Center X, Center Y, and Cave angle controls in the UI, but keep their spinboxes and values for internal use
        for widget in [
            self.centre_x_label,
            self.xc_spin,
            self.centre_y_label,
            self.yc_spin,
            self.cave_angle_label,
            self.cave_angle_spin,
        ]:
            widget.hide()

        self.frame_label = QLabel("Frame:")
        self.frame_spin = QSpinBox()
        self.frame_spin.setRange(1, 1)
        self.frame_spin.setValue(1)
        self.frame_spin.setEnabled(False)
        self.frame_spin.hide()
        form_layout.addWidget(self.frame_label, 4, 0)
        form_layout.addWidget(self.frame_spin, 4, 1)

        controls_layout.addLayout(form_layout)

        self.nan_operator_combo = QComboBox()
        self.nan_operator_combo.addItems(["<=", ">="])
        self.nan_operator_combo.setFixedWidth(64)
        self.nan_operator_combo.setStyleSheet(COMPACT_COMBO_STYLE)

        self.nan_threshold_spin = QDoubleSpinBox()
        self.nan_threshold_spin.setRange(-1e12, 1e12)
        self.nan_threshold_spin.setDecimals(6)
        self.nan_threshold_spin.setValue(-14)
        self.nan_threshold_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.nan_threshold_spin.setMaximumWidth(136)

        self.nan_extra_checkbox = QCheckBox("Or")
        self.nan_extra_operator_combo = QComboBox()
        self.nan_extra_operator_combo.addItems([">=", "<="])
        self.nan_extra_operator_combo.setFixedWidth(64)
        self.nan_extra_operator_combo.setStyleSheet(COMPACT_COMBO_STYLE)
        self.nan_extra_threshold_spin = QDoubleSpinBox()
        self.nan_extra_threshold_spin.setRange(-1e12, 1e12)
        self.nan_extra_threshold_spin.setDecimals(6)
        self.nan_extra_threshold_spin.setValue(4e9)
        self.nan_extra_threshold_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.nan_extra_threshold_spin.setMaximumWidth(136)
        self.nan_extra_operator_combo.setEnabled(False)
        self.nan_extra_threshold_spin.setEnabled(False)

        nan_layout = QGridLayout()
        nan_layout.setContentsMargins(0, 0, 0, 0)
        nan_layout.setHorizontalSpacing(4)
        nan_layout.setVerticalSpacing(4)
        nan_layout.setColumnStretch(2, 1)
        nan_layout.addWidget(QLabel("NaN if I"), 0, 0)
        nan_layout.addWidget(self.nan_operator_combo, 0, 1)
        nan_layout.addWidget(self.nan_threshold_spin, 0, 2)
        nan_layout.addWidget(self.nan_extra_checkbox, 1, 0)
        nan_layout.addWidget(self.nan_extra_operator_combo, 1, 1)
        nan_layout.addWidget(self.nan_extra_threshold_spin, 1, 2)

        self.id13_beamstop_checkbox = QCheckBox("Add ID13 beamstop mask")
        self.id13_beamstop_checkbox.setChecked(False)

        self.expand_nan_neighbors_checkbox = QCheckBox("Expand NaN 2 px")
        self.expand_nan_neighbors_checkbox.setChecked(False)
        self.expand_nan_neighbors_checkbox.setToolTip(
            "Expands the NaN mask by 2 pixels before central symmetry filling."
        )

        self.manual_mask_button = QPushButton("Manual mask")
        self.custom_cave_button = QPushButton("Custom cave")
        self.manual_mask_status_label = QLabel("Manual mask: none")
        self.manual_mask_status_label.setWordWrap(True)
        self.manual_mask_status_label.setStyleSheet("""
            QLabel {
                color: #666666;
                font-size: 11px;
                padding-left: 4px;
            }
        """)

        self.cave_scope_combo = QComboBox()
        self.cave_scope_combo.addItems(["Current image", "All frames/scans in file"])
        self.cave_scope_combo.setToolTip("Process only the current image or every frame/scan in the loaded file.")
        self.cave_scope_combo.setFixedWidth(220)
        self.manual_mask_button.setStyleSheet(cave_action_button_style)
        self.manual_mask_button.setFixedHeight(cave_action_button_height)
        self.manual_mask_button.setCursor(Qt.PointingHandCursor)
        self.manual_mask_button.setVisible(True)
        self.manual_mask_button.setEnabled(True)
        self.manual_mask_button.clicked.connect(self.open_manual_cave_dialog)
        self.custom_cave_button.setStyleSheet(cave_action_button_style)
        self.custom_cave_button.setFixedHeight(cave_action_button_height)
        self.custom_cave_button.setCursor(Qt.PointingHandCursor)
        self.custom_cave_button.setVisible(True)
        self.custom_cave_button.setEnabled(True)
        self.custom_cave_button.clicked.connect(self.open_custom_cave_dialog)
        self.update_manual_mask_button_state()

        self.save_checkbox = QCheckBox("Save output after Run Cave")
        self.save_checkbox.setChecked(True)

        controls_layout.addLayout(nan_layout)
        controls_layout.addWidget(self.id13_beamstop_checkbox)
        controls_layout.addWidget(self.expand_nan_neighbors_checkbox)
        mask_button_layout = QHBoxLayout()
        mask_button_layout.setContentsMargins(0, 0, 0, 0)
        mask_button_layout.setSpacing(4)
        mask_button_layout.addWidget(self.manual_mask_button)
        mask_button_layout.addWidget(self.custom_cave_button)
        controls_layout.addLayout(mask_button_layout)
        controls_layout.addWidget(self.manual_mask_status_label)
        controls_layout.addWidget(QLabel("Apply cave to:"))
        controls_layout.addWidget(self.cave_scope_combo)

        intensity_box = QGroupBox("Contrast")
        intensity_box.setMinimumWidth(0)
        intensity_box.setStyleSheet(GROUP_BOX_STYLE)
        intensity_layout = QGridLayout(intensity_box)
        intensity_layout.setContentsMargins(*GROUP_BOX_MARGINS)
        intensity_layout.setSpacing(4)

        self.vmin_slider = QSlider(Qt.Horizontal)
        self.vmax_slider = QSlider(Qt.Horizontal)
        self.vmin_slider.setRange(0, self.slider_scale)
        self.vmax_slider.setRange(0, self.slider_scale)
        self.vmin_slider.setValue(0)
        self.vmax_slider.setValue(self.slider_scale)

        self.vmin_label = QLabel("Min: 0.000")
        self.vmax_label = QLabel("Max: 1.000")
        self.auto_intensity_button = QPushButton("Auto")
        self.auto_intensity_button.setFixedWidth(54)
        self.auto_intensity_button.clicked.connect(self.auto_display_limits)
        self.lock_intensity_checkbox = QCheckBox("Lock min/max")
        self.lock_intensity_checkbox.setChecked(False)

        intensity_layout.addWidget(self.vmin_label, 0, 0)
        intensity_layout.addWidget(self.vmin_slider, 0, 1)
        intensity_layout.addWidget(self.auto_intensity_button, 0, 2, 2, 1)
        intensity_layout.addWidget(self.vmax_label, 1, 0)
        intensity_layout.addWidget(self.vmax_slider, 1, 1)
        intensity_layout.addWidget(self.lock_intensity_checkbox, 2, 0, 1, 3)

        controls_layout.addWidget(intensity_box)

        cave_action_button_style = """
            QPushButton {
                background-color: #dddddd;
                border: none;
                border-radius: 6px;
                padding: 4px 10px;
            }
            QPushButton:hover {
                background-color: #d4d4d4;
            }
            QPushButton:pressed {
                background-color: #c8c8c8;
            }
            QPushButton:disabled {
                background-color: #eeeeee;
                color: #aaaaaa;
                border: none;
            }
        """

        button_layout = QGridLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(4)

        self.run_button = QPushButton("▶️ Run Cave")
        self.run_button.setFixedHeight(cave_action_button_height)
        self.run_button.setStyleSheet(cave_action_button_style)
        self.run_button.clicked.connect(self.run_cave)

        self.save_button = QPushButton("💾 Save Cave")
        self.batch_cave_button = QPushButton("Cave selected")
        self.batch_cave_button.setToolTip("Apply the current cave settings to all selected files")
        self.save_button.setFixedHeight(cave_action_button_height)
        self.save_button.setStyleSheet(cave_action_button_style)
        self.save_button.clicked.connect(self.save_cave)
        self.batch_cave_button.clicked.connect(self.cave_selected_files)
        for button in [self.run_button, self.save_button, self.batch_cave_button]:
            button.setMinimumWidth(0)
            button.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)

        button_layout.addWidget(self.run_button, 0, 0)
        button_layout.addWidget(self.save_button, 0, 1)
        button_layout.addWidget(self.batch_cave_button, 1, 0, 1, 2)
        button_layout.setColumnStretch(0, 1)
        button_layout.setColumnStretch(1, 1)
        controls_layout.addLayout(button_layout)

        self.status = QTextEdit()
        self.status.setReadOnly(True)
        self.status.setPlaceholderText("")
        self.status.hide()
        controls_layout.addStretch(1)

        self.btn_xenocs.clicked.connect(lambda: self.set_instrument_mode("XENOCS"))
        self.btn_id02.clicked.connect(lambda: self.set_instrument_mode("ID02"))
        self.btn_id13.clicked.connect(lambda: self.set_instrument_mode("ID13"))
        self.btn_custom.clicked.connect(self.use_custom_cave_mask)

        self.xc_spin.valueChanged.connect(self.refresh_preview)
        self.yc_spin.valueChanged.connect(self.refresh_preview)
        self.beamstop_y_spin.valueChanged.connect(self.refresh_preview)
        self.cave_angle_spin.valueChanged.connect(self.refresh_preview)
        self.frame_spin.valueChanged.connect(self.load_selected_frame)
        self.nan_operator_combo.currentTextChanged.connect(self.refresh_preview)
        self.nan_threshold_spin.valueChanged.connect(self.refresh_preview)
        self.nan_threshold_spin.editingFinished.connect(self.refresh_preview)
        self.nan_extra_checkbox.stateChanged.connect(self.update_extra_nan_condition)
        self.nan_extra_operator_combo.currentTextChanged.connect(self.refresh_preview)
        self.nan_extra_threshold_spin.valueChanged.connect(self.refresh_preview)
        self.nan_extra_threshold_spin.editingFinished.connect(self.refresh_preview)
        self.id13_beamstop_checkbox.stateChanged.connect(self.refresh_preview)
        self.expand_nan_neighbors_checkbox.stateChanged.connect(self.refresh_preview)
        self.vmin_slider.valueChanged.connect(self.update_display_limits_from_sliders)
        self.vmax_slider.valueChanged.connect(self.update_display_limits_from_sliders)

        self.frame_nav_container = QWidget()
        frame_nav = QHBoxLayout(self.frame_nav_container)
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
        self.frame_counter_label = QLabel("1 / 1")
        self.frame_counter_label.setMinimumWidth(FRAME_COUNTER_WIDTH)
        self.frame_counter_label.setAlignment(Qt.AlignCenter)

        frame_nav.addWidget(QLabel("Start:"))
        frame_nav.addWidget(self.frame_start_spin)
        frame_nav.addWidget(self.prev_frame_button)
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(1, 1)
        self.frame_slider.setValue(1)
        frame_nav.addWidget(self.frame_slider, stretch=1)
        frame_nav.addWidget(self.next_frame_button)
        frame_nav.addWidget(QLabel("End:"))
        frame_nav.addWidget(self.frame_end_spin)
        frame_nav.addWidget(self.frame_counter_label)
        main_layout.addWidget(self.frame_nav_container, stretch=0)

        self.frame_start_spin.valueChanged.connect(self.update_frame_bounds)
        self.frame_end_spin.valueChanged.connect(self.update_frame_bounds)
        self.frame_slider.valueChanged.connect(self.frame_slider_changed)
        self.prev_frame_button.clicked.connect(self.previous_frame)
        self.next_frame_button.clicked.connect(self.next_frame)

        self.batch_progress = QProgressBar()
        self.batch_progress.setRange(0, 1)
        self.batch_progress.setValue(0)
        self.batch_progress.setVisible(False)
        main_layout.addWidget(self.batch_progress)

    def set_controls_enabled(self, enabled):
        for widget in [
            self.beamstop_y_spin,
            self.frame_spin,
            self.frame_slider,
            self.nan_operator_combo,
            self.nan_threshold_spin,
            self.nan_extra_checkbox,
            self.id13_beamstop_checkbox,
            self.expand_nan_neighbors_checkbox,
            self.lock_intensity_checkbox,
            self.auto_intensity_button,
            self.vmin_slider,
            self.vmax_slider,
            self.run_button,
            self.save_button,
            self.batch_cave_button,
        ]:
            widget.setEnabled(enabled)

        self.update_frame_selector_visibility()
        self.update_beamstop_visibility()
        self.update_manual_mask_button_state()
        self.update_extra_nan_condition(refresh=False)

        for button in [
            self.btn_xenocs,
            self.btn_id02,
            self.btn_id13,
            self.btn_custom,
            self.q_manual_button,
        ]:
            button.setEnabled(True)
        self.update_manual_mask_button_state()

    def set_initial_center_splitter_sizes(self):
        if not hasattr(self, "center_splitter"):
            return
        height = max(2, self.center_splitter.height())
        half_height = height // 2
        self.center_splitter.setSizes([half_height, height - half_height])

    def is_development_copy(self):
        return (Path(__file__).resolve().parents[1] / ".git").exists()

    def update_manual_mask_button_state(self):
        if not hasattr(self, "manual_mask_button"):
            return

        self.manual_mask_button.setText("Manual mask")
        self.manual_mask_button.setVisible(True)
        self.manual_mask_button.setEnabled(True)
        self.manual_mask_button.setToolTip("Open the manual cave mask.")
        if hasattr(self, "custom_cave_button"):
            self.custom_cave_button.setVisible(True)
            self.custom_cave_button.setEnabled(True)
            self.custom_cave_button.setToolTip("Open the custom cave+ editor.")
        
    def update_extra_nan_condition(self, refresh=True):
        self.commit_nan_threshold_edits()
        enabled = self.nan_extra_checkbox.isEnabled() and self.nan_extra_checkbox.isChecked()
        self.nan_extra_operator_combo.setEnabled(enabled)
        self.nan_extra_threshold_spin.setEnabled(enabled)
        if refresh:
            self.refresh_preview()

    def extra_nan_condition(self):
        self.commit_nan_threshold_edits()
        if not self.nan_extra_checkbox.isChecked():
            return None, None
        return self.nan_extra_operator_combo.currentText(), self.nan_extra_threshold_spin.value()

    def commit_nan_threshold_edits(self):
        for spin in (self.nan_threshold_spin, self.nan_extra_threshold_spin):
            try:
                spin.interpretText()
            except Exception:
                pass

    def manual_cave_mask(self):
        if self.image is None or not self.manual_cave_shapes:
            return None

        mask = np.zeros(self.image.shape, dtype=bool)
        ny, nx = mask.shape

        for shape in self.manual_cave_shapes:
            self.shape_to_mask(mask, shape)

        return mask

    def manual_cave_exclusion_mask(self):
        if self.image is None or not self.manual_cave_exclusion_shapes:
            return None

        mask = np.zeros(self.image.shape, dtype=bool)

        for shape in self.manual_cave_exclusion_shapes:
            self.shape_to_mask(mask, shape)

        return mask

    def manual_cave_pre_nan_mask(self):
        if self.image is None or not self.manual_cave_pre_nan_shapes:
            return None

        mask = np.zeros(self.image.shape, dtype=bool)

        for shape in self.manual_cave_pre_nan_shapes:
            self.shape_to_mask(mask, shape)

        return mask

    def contiguous_ranges(self, flags):
        ranges = []
        start = None

        for index, flag in enumerate(flags):
            if flag and start is None:
                start = index
            elif not flag and start is not None:
                ranges.append((start, index))
                start = None

        if start is not None:
            ranges.append((start, len(flags)))

        return ranges

    def current_bad_pixel_mask(self):
        if self.image is None:
            return None

        self.commit_nan_threshold_edits()
        source = self.image.astype(np.float64)
        mask = ~np.isfinite(source)

        if self.nan_operator_combo.currentText() == ">=":
            mask |= source >= self.nan_threshold_spin.value()
        else:
            mask |= source <= self.nan_threshold_spin.value()

        extra_operator, extra_threshold = self.extra_nan_condition()
        if extra_operator == ">=":
            mask |= source >= extra_threshold
        elif extra_operator == "<=":
            mask |= source <= extra_threshold

        if self.expand_nan_neighbors_checkbox.isChecked():
            original_mask = mask.copy()
            radius = 2
            padded_mask = np.pad(original_mask, radius, mode="constant", constant_values=False)
            expanded_mask = np.zeros_like(original_mask, dtype=bool)
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    expanded_mask |= padded_mask[
                        radius + dy:radius + dy + original_mask.shape[0],
                        radius + dx:radius + dx + original_mask.shape[1],
                    ]
            mask = expanded_mask

        return mask

    def id13_custom_shapes(self):
        if self.image is None:
            return []

        bad_mask = self.current_bad_pixel_mask()
        if bad_mask is None:
            return []

        ny, nx = bad_mask.shape
        shapes = []

        y1 = int(round(self.yc_spin.value()))
        y2 = int(round(self.beamstop_y_spin.value()))
        x1 = int(round(self.xc_spin.value()))
        x2 = nx
        x1 = max(0, min(x1, nx))
        y1 = max(0, min(y1, ny))
        y2 = max(0, min(y2, ny))
        if y2 < y1:
            y1, y2 = y2, y1
        if y1 != y2 and x1 != x2:
            shapes.append({"type": "rect", "points": (x1, y1, x2, y2)})

        row_threshold = max(8, int(nx * 0.35))
        col_threshold = max(8, int(ny * 0.35))
        row_ranges = self.contiguous_ranges(np.sum(bad_mask, axis=1) >= row_threshold)
        col_ranges = self.contiguous_ranges(np.sum(bad_mask, axis=0) >= col_threshold)

        for start, end in row_ranges:
            if end - start >= 1:
                shapes.append({"type": "rect", "points": (0, start, nx, end)})

        for start, end in col_ranges:
            if end - start >= 1:
                shapes.append({"type": "rect", "points": (start, 0, end, ny)})

        return shapes

    def use_custom_cave_mask(self):
        self.set_instrument_mode("Custom")

        if self.image is None:
            return

        if not self.manual_cave_shapes:
            self.manual_cave_shapes = [self.copy_shape_data(shape) for shape in self.id13_custom_shapes()]

        self.open_manual_cave_dialog()

    def copy_shape_data(self, shape):
        if shape["type"] in ("rect", "vband", "hband"):
            points = tuple(float(value) for value in shape["points"])
        else:
            points = [(float(x), float(y)) for x, y in shape["points"]]
        return {"type": shape["type"], "points": points}

    def manual_band_polygon(self, shape):
        if self.image is None:
            return shape["points"]

        ny, nx = self.image.shape

        if shape["type"] == "vband":
            x0, _y0, x1, _y1 = shape["points"][:4]
            xmin = max(0.0, min(float(x0), float(x1)))
            xmax = min(float(nx), max(float(x0), float(x1)))
            return [
                (xmin, 0.0),
                (xmax, 0.0),
                (xmax, float(ny)),
                (xmin, float(ny)),
            ]

        if shape["type"] == "hband":
            _x0, y0, _x1, y1 = shape["points"][:4]
            ymin = max(0.0, min(float(y0), float(y1)))
            ymax = min(float(ny), max(float(y0), float(y1)))
            return [
                (0.0, ymin),
                (float(nx), ymin),
                (float(nx), ymax),
                (0.0, ymax),
            ]

        return shape["points"]

    def shape_to_mask(self, mask, shape):
        ny, nx = mask.shape
        if shape["type"] == "rect":
            x0, y0, x1, y1 = shape["points"]
            xmin = max(0, int(np.floor(min(x0, x1))))
            xmax = min(nx, int(np.ceil(max(x0, x1))))
            ymin = max(0, int(np.floor(min(y0, y1))))
            ymax = min(ny, int(np.ceil(max(y0, y1))))
            mask[ymin:ymax, xmin:xmax] = True
            return

        polygon_points = self.manual_band_polygon(shape) if shape["type"] in ("vband", "hband") else shape["points"]
        polygon = np.asarray(polygon_points, dtype=float)
        if polygon.size == 0:
            return

        xmin = max(0, int(np.floor(np.nanmin(polygon[:, 0]))))
        xmax = min(nx, int(np.ceil(np.nanmax(polygon[:, 0]))) + 1)
        ymin = max(0, int(np.floor(np.nanmin(polygon[:, 1]))))
        ymax = min(ny, int(np.ceil(np.nanmax(polygon[:, 1]))) + 1)
        if xmin >= xmax or ymin >= ymax:
            return

        yy, xx = np.mgrid[ymin:ymax, xmin:xmax]
        points = np.column_stack((xx.ravel(), yy.ravel()))
        path = MplPath(polygon_points)
        mask[ymin:ymax, xmin:xmax] |= path.contains_points(points).reshape((ymax - ymin, xmax - xmin))

    def open_manual_cave_dialog(self):
        try:
            image = self.image_clean if self.image_clean is not None else self.image

            if image is None:
                QMessageBox.information(
                    self,
                    "Manual cave mask",
                    "Load an EDF or H5 image before opening the manual cave mask editor."
                )
                return

            filled_image = self.image_filled if self.image_filled is not None else image

            dialog = ManualCaveDialog(
                self,
                image,
                filled_image,
                self.manual_cave_shapes,
                self.manual_cave_exclusion_shapes,
                self.manual_cave_pre_nan_shapes,
                self.current_display_limits(),
            )
            dialog.exec()
            self.refresh_preview()

        except Exception as error:
            QMessageBox.critical(self, "Manual cave mask error", str(error))

    def open_custom_cave_dialog(self):
        try:
            if self.image is None:
                QMessageBox.information(
                    self,
                    "Custom cave",
                    "Load an EDF or H5 image before opening the custom cave editor."
                )
                return

            self.refresh_preview()
            image = self.image_clean if self.image_clean is not None else self.image
            filled_image = self.image_filled if self.image_filled is not None else image
            dialog = CustomCaveDialog(
                self,
                image,
                filled_image,
                self.current_display_limits(),
            )
            dialog.exec()
            self.refresh_preview()

        except Exception as error:
            QMessageBox.critical(self, "Custom cave error", str(error))

    def auto_set_display_limits(self):
        if self.image is None:
            return

        display = self.image.astype(np.float64).copy()
        display[~np.isfinite(display)] = np.nan
        display[display < 0] = np.nan

        with np.errstate(invalid="ignore", divide="ignore"):
            display = np.log10(display + 1)

        finite_values = display[np.isfinite(display)]

        if finite_values.size == 0:
            self.display_vmin = 0.0
            self.display_vmax = 1.0
        else:
            self.display_vmin = float(np.nanpercentile(finite_values, 1))
            self.display_vmax = float(np.nanpercentile(finite_values, 99))

            if self.display_vmin >= self.display_vmax:
                self.display_vmin = float(np.nanmin(finite_values))
                self.display_vmax = float(np.nanmax(finite_values))

            if self.display_vmin >= self.display_vmax:
                self.display_vmax = self.display_vmin + 1.0

        self.vmin_slider.blockSignals(True)
        self.vmax_slider.blockSignals(True)
        self.vmin_slider.setValue(0)
        self.vmax_slider.setValue(self.slider_scale)
        self.vmin_slider.blockSignals(False)
        self.vmax_slider.blockSignals(False)

        self.update_display_labels()

    def auto_display_limits(self):
        self.auto_set_display_limits()
        self.update_display_limits_from_sliders()

    def current_display_limits(self):
        span = self.display_vmax - self.display_vmin
        if span <= 0:
            return self.display_vmin, self.display_vmax

        vmin = self.display_vmin + span * (self.vmin_slider.value() / self.slider_scale)
        vmax = self.display_vmin + span * (self.vmax_slider.value() / self.slider_scale)

        if vmin >= vmax:
            vmax = vmin + span / self.slider_scale

        return vmin, vmax

    def update_display_limits_from_sliders(self):
        self.update_display_labels()
        self.refresh_preview()

    def update_display_labels(self):
        vmin, vmax = self.current_display_limits()
        self.vmin_label.setText(f"Min: {vmin:.3f}")
        self.vmax_label.setText(f"Max: {vmax:.3f}")

    def compact_preset_buttons(self, widths=None):
        widths = widths or {
            self.btn_xenocs: 66,
            self.btn_id02: 48,
            self.btn_id13: 48,
            self.btn_custom: 60,
            self.q_manual_button: 24,
        }

        for button, width in widths.items():
            button.setMinimumWidth(0)
            button.setFixedWidth(width)

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
        self.compact_preset_buttons()

        self.apply_instrument_preset()
        self.update_centre_warning_labels()
        self.update_beamstop_visibility()
        self.refresh_preview()

    def apply_line_geometry_selection(self, name, geometry):
        values = line_geometry_to_lrphoton(geometry)
        self.custom_line_geometry_values = values
        self.xc_spin.setValue(values["xc"])
        self.yc_spin.setValue(values["yc"])
        self.instrument_mode = "Custom" if name not in {"XENOCS", "ID02", "ID13"} else name
        buttons = {
            "XENOCS": self.btn_xenocs,
            "ID02": self.btn_id02,
            "ID13": self.btn_id13,
            "Custom": self.btn_custom,
        }
        style_q_geometry_buttons(buttons, self.instrument_mode, self.q_manual_button)
        self.compact_preset_buttons()
        self.update_centre_warning_labels()
        self.update_beamstop_visibility()
        self.refresh_preview()

    def update_centre_warning_labels(self):
        self.centre_x_label.setText("Center X:")
        self.centre_y_label.setText("Center Y:")

    def update_beamstop_visibility(self):
        is_id13 = self.instrument_mode == "ID13"

        self.beamstop_y_label.setVisible(is_id13)
        self.beamstop_y_spin.setVisible(is_id13)
        self.id13_beamstop_checkbox.setVisible(is_id13)

        self.beamstop_y_spin.setEnabled(is_id13 and self.image is not None)
        self.id13_beamstop_checkbox.setEnabled(is_id13 and self.image is not None)
        self.cave_angle_spin.setEnabled(False)
        self.cave_angle_spin.setVisible(False)
        if hasattr(self, "cave_angle_label"):
            self.cave_angle_label.setVisible(False)

        self.id13_beamstop_checkbox.blockSignals(True)
        self.id13_beamstop_checkbox.setChecked(is_id13)
        self.id13_beamstop_checkbox.blockSignals(False)

    def update_frame_selector_visibility(self):
        has_multiple_frames = max(1, int(getattr(self, "h5_n_frames", 1) or 1)) > 1
        self.frame_label.setVisible(has_multiple_frames)
        self.frame_spin.setVisible(has_multiple_frames)
        if hasattr(self, "frame_nav_container"):
            self.frame_nav_container.setVisible(has_multiple_frames)
        self.update_frame_counter()

    def configure_frame_navigation(self, n_frames):
        n_frames = max(1, int(n_frames))
        self._syncing_frame_controls = True
        for spin in [self.frame_spin, self.frame_start_spin, self.frame_end_spin]:
            spin.blockSignals(True)
        self.frame_slider.blockSignals(True)

        self.frame_spin.setRange(1, n_frames)
        self.frame_spin.setValue(1)
        self.frame_slider.setRange(1, n_frames)
        self.frame_slider.setValue(1)
        self.frame_start_spin.setRange(1, n_frames)
        self.frame_start_spin.setValue(1)
        self.frame_end_spin.setRange(1, n_frames)
        self.frame_end_spin.setValue(n_frames)

        for spin in [self.frame_spin, self.frame_start_spin, self.frame_end_spin]:
            spin.blockSignals(False)
        self.frame_slider.blockSignals(False)
        self._syncing_frame_controls = False
        self._batch_cave_running = False

        self.update_frame_counter()

    def frame_slider_changed(self, value):
        if self._syncing_frame_controls:
            return

        start = self.frame_start_spin.value()
        end = self.frame_end_spin.value()
        value = max(start, min(int(value), end))

        if value != self.frame_slider.value():
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(value)
            self.frame_slider.blockSignals(False)

        self.frame_spin.setValue(value)

    def update_frame_bounds(self):
        if self._syncing_frame_controls:
            return

        start = self.frame_start_spin.value()
        end = self.frame_end_spin.value()

        if start > end:
            sender = self.sender()
            if sender is self.frame_start_spin:
                self.frame_end_spin.setValue(start)
                end = start
            else:
                self.frame_start_spin.setValue(end)
                start = end

        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(start, end)
        self.frame_slider.blockSignals(False)

        current = self.frame_spin.value()
        if current < start:
            self.frame_spin.setValue(start)
        elif current > end:
            self.frame_spin.setValue(end)

        self.update_frame_counter()

    def update_frame_counter(self):
        current = self.frame_spin.value()
        total = max(1, self.h5_n_frames)
        self.frame_counter_label.setText(f"{current} / {total}")
        if hasattr(self, "prev_frame_button"):
            can_navigate = total > 1 and self.file_type in {"H5", "EDF"}
            self.frame_spin.setEnabled(can_navigate)
            self.frame_start_spin.setEnabled(can_navigate)
            self.frame_end_spin.setEnabled(can_navigate)
            self.frame_slider.setEnabled(can_navigate)
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(current)
            self.frame_slider.blockSignals(False)
            self.prev_frame_button.setEnabled(can_navigate and current > self.frame_start_spin.value())
            self.next_frame_button.setEnabled(can_navigate and current < self.frame_end_spin.value())

    def previous_frame(self):
        self.frame_spin.setValue(max(self.frame_start_spin.value(), self.frame_spin.value() - 1))

    def next_frame(self):
        self.frame_spin.setValue(min(self.frame_end_spin.value(), self.frame_spin.value() + 1))

    def wavelength_to_nm(self, wavelength):
        if wavelength < 1e-6:
            return wavelength * 1e9
        if wavelength >= 0.5:
            return wavelength * 0.1
        return wavelength

    def q_geometry(self):
        if self.image is None:
            return None

        xc = self.xc_spin.value()
        yc = self.yc_spin.value()
        if self.instrument_mode == "Custom" and hasattr(self, "custom_line_geometry_values"):
            geometry = self.custom_line_geometry_values
            return (
                xc,
                yc,
                geometry["distance_m"],
                geometry["pixel_x_mm"],
                geometry["pixel_y_mm"],
                geometry["wavelength_a"] * 0.1,
            )

        distance_m = get_header_float(
            self.header,
            "SampleDistance",
            "sampledistance",
            "sample_distance",
            "Distance",
            "DetectorDistance",
            "detector_distance",
        )
        pixel_x = get_header_float(
            self.header,
            "PSize_1",
            "psize_1",
            "PSize_X",
            "PixelSizeX",
            "pixel_size_x",
            "x_pixel_size",
        )
        pixel_y = get_header_float(
            self.header,
            "PSize_2",
            "psize_2",
            "PSize_Y",
            "PixelSizeY",
            "pixel_size_y",
            "y_pixel_size",
        )
        wavelength = get_header_float(
            self.header,
            "WaveLength",
            "Wavelength",
            "wavelength",
            "Lambda",
            "lambda",
        )

        if self.instrument_mode == "ID02":
            distance_m = ID02_DEFAULT_DISTANCE_M if distance_m is None else distance_m
            pixel_x = ID02_DEFAULT_PIXEL_MM if pixel_x is None else pixel_x
            pixel_y = ID02_DEFAULT_PIXEL_MM if pixel_y is None else pixel_y
            wavelength = ID02_DEFAULT_WAVELENGTH_A if wavelength is None else wavelength
        elif self.instrument_mode == "ID13":
            distance_m = ID13_DEFAULT_DISTANCE_M if distance_m is None else distance_m
            pixel_x = ID13_DEFAULT_PIXEL_MM if pixel_x is None else pixel_x
            pixel_y = ID13_DEFAULT_PIXEL_MM if pixel_y is None else pixel_y
            wavelength = ID13_DEFAULT_WAVELENGTH_A if wavelength is None else wavelength

        if distance_m is None or pixel_x is None or pixel_y is None or wavelength is None:
            return None

        pixel_x_mm = pixel_x * 1000.0 if pixel_x < 1e-3 else pixel_x
        pixel_y_mm = pixel_y * 1000.0 if pixel_y < 1e-3 else pixel_y
        wavelength_nm = self.wavelength_to_nm(wavelength)

        if distance_m <= 0 or pixel_x_mm <= 0 or pixel_y_mm <= 0 or wavelength_nm <= 0:
            return None

        return xc, yc, distance_m, pixel_x_mm, pixel_y_mm, wavelength_nm

    def calculate_q_at_pixel(self, x_index, y_index):
        geometry = self.q_geometry()
        if geometry is None:
            return None

        xc, yc, distance_m, pixel_x_mm, pixel_y_mm, wavelength_nm = geometry
        dx_px = float(x_index) - float(xc)
        dy_px = float(y_index) - float(yc)
        dx_m = dx_px * pixel_x_mm * 1e-3
        dy_m = dy_px * pixel_y_mm * 1e-3
        r_m = np.sqrt(dx_m ** 2 + dy_m ** 2)
        two_theta = np.arctan2(r_m, distance_m)
        return (4.0 * np.pi / wavelength_nm) * np.sin(two_theta / 2.0)

    def apply_instrument_preset(self):
        if self.instrument_mode != "Custom":
            self.nan_extra_checkbox.blockSignals(True)
            self.nan_extra_checkbox.setChecked(False)
            self.nan_extra_checkbox.blockSignals(False)
            self.nan_extra_operator_combo.setCurrentText(">=")
            self.nan_extra_threshold_spin.setValue(4e9)
            self.update_extra_nan_condition(refresh=False)

        if self.instrument_mode == "XENOCS":
            center_1 = get_header_float(self.header, *CENTER_X_KEYS)
            center_2 = get_header_float(self.header, *CENTER_Y_KEYS)
            self.xc_spin.setValue(center_1 if center_1 is not None else 0)
            self.yc_spin.setValue(center_2 if center_2 is not None else 0)
            self.nan_operator_combo.setCurrentText("<=")
            self.nan_threshold_spin.setValue(-14)
            return

        if self.instrument_mode == "ID02":
            center_1 = get_header_float(self.header, *CENTER_X_KEYS)
            center_2 = get_header_float(self.header, *CENTER_Y_KEYS)
            self.xc_spin.setValue(center_1 if center_1 is not None else ID02_DEFAULT_CENTER_X)
            self.yc_spin.setValue(center_2 if center_2 is not None else ID02_DEFAULT_CENTER_Y)
            self.nan_operator_combo.setCurrentText("<=")
            self.nan_threshold_spin.setValue(-9)
            return

        if self.instrument_mode == "ID13":
            self.xc_spin.setValue(ID13_DEFAULT_CENTER_X)
            self.yc_spin.setValue(ID13_DEFAULT_CENTER_Y)
            self.nan_operator_combo.setCurrentText(">=")
            self.nan_threshold_spin.setValue(4e9)
            return

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Choose folder",
            str(self.current_folder),
        )

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
        if hasattr(self, "folder_path"):
            self.folder_path.setText(str(self.current_folder))
        self.refresh_files()
        self._syncing_folder = False

    def refresh_files(self):
        if not hasattr(self, "file_list"):
            return

        folder = Path(self.folder_path.text()).expanduser()

        if not folder.exists():
            QMessageBox.warning(
                self,
                "Folder not found",
                "The selected folder does not exist."
            )
            return

        self.current_folder = folder
        if not self._syncing_folder:
            self.folder_changed.emit(self.current_folder)
        self.file_list.clear()

        extension_patterns = self.extension_filter.text().split()
        name_pattern = self.name_filter.text().strip() or "*"
        iterator = folder.rglob("*") if self.show_subfolders_checkbox.isChecked() else folder.glob("*")

        files = []
        cave_outputs_by_base = set()

        for path in iterator:
            if not path.is_file():
                continue
            if should_hide_file_in_browser(path):
                continue

            lower_name = path.name.lower()
            lower_stem = path.stem.lower()

            match_extension = any(
                fnmatch.fnmatch(lower_name, pattern.lower())
                for pattern in extension_patterns
            )
            match_name = fnmatch.fnmatch(path.name, name_pattern)

            if not (match_extension and match_name):
                continue

            if "_cave" in lower_stem:
                base_stem = re.sub(r"_cave.*$", "", path.stem, flags=re.IGNORECASE)
                cave_outputs_by_base.add((path.parent.resolve(), base_stem))
                continue

            if path.suffix.lower() in [".h5", ".hdf5"] and "_ave" in lower_stem:
                continue

            if self.only_thumbs_up_checkbox.isChecked() and not is_file_rated_up(path):
                continue

            files.append(path)

        for path in sorted(files):
            display_name = str(path.relative_to(folder))
            has_matching_cave = (path.parent.resolve(), path.stem) in cave_outputs_by_base

            if has_matching_cave:
                display_name = f"✅ {display_name}"

            item = QListWidgetItem(display_name)
            set_item_file_path(item, path)
            if has_matching_cave:
                item.setToolTip("A matching cave output already exists for this file.")
            self.file_list.addItem(item)

    def open_selected_file(self, item=None):
        if item is None:
            item = self.file_list.currentItem()

        if item is None:
            return

        stored_path = item.data(Qt.UserRole)

        if not stored_path:
            return

        self.open_file(Path(stored_path).expanduser().resolve())

    def open_file(self, file_path=None):
        if isinstance(file_path, bool):
            file_path = None

        if file_path is None:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Open EDF or H5 file",
                str(self.current_folder),
                "Data files (*.edf *.h5 *.hdf5);;EDF (*.edf);;HDF5 (*.h5 *.hdf5);;All files (*)",
            )

        if not file_path:
            return

        try:
            path = Path(file_path)
            suffix = path.suffix.lower()

            if suffix == ".edf":
                image, header, raw_header_text, byte_order, n_frames = read_edf_frame(file_path, 0)
                self.file_type = "EDF"
                self.raw_header_text = raw_header_text
                self.byte_order = byte_order
                self.h5_dataset_name = None
                self.h5_frame_axis = None
                self.h5_n_frames = n_frames
                self._edf_frames = None

                self.configure_frame_navigation(self.h5_n_frames)
            elif suffix in [".h5", ".hdf5"]:
                dataset_name, dataset_shape, frame_axis, n_frames, header = inspect_h5_image_dataset(file_path)
                image, frame_header = read_h5_frame(file_path, dataset_name, 0, add_matching_center=False)
                header.update(frame_header)
                self.file_type = "H5"
                self.raw_header_text = ""
                self.byte_order = "LowByteFirst"
                self.h5_dataset_name = dataset_name
                self.h5_frame_axis = frame_axis
                self.h5_n_frames = n_frames

                self.configure_frame_navigation(n_frames)
            else:
                raise ValueError("Unsupported file format. Please select an EDF, H5 or HDF5 file.")

            self.current_file = path
            self.current_folder = path.parent
            if hasattr(self, "folder_path"):
                self.folder_path.setText(str(self.current_folder))
            self.header = header
            self.image = image.astype(np.float64)
            self.image_clean = None
            self.image_filled = None
            self.cave_mask = None
            self.manual_cave_shapes = []
            self.manual_cave_exclusion_shapes = []
            self.manual_cave_pre_nan_shapes = []
            self.update_manual_mask_status_label()

            self.set_controls_enabled(True)
            self.apply_instrument_preset()
            self.update_centre_warning_labels()
            self.update_beamstop_visibility()
            self.update_frame_selector_visibility()
            self.auto_set_display_limits()
            self.refresh_preview()
            self.update_manual_mask_button_state()
            self.update_status()

        except Exception as error:
            QMessageBox.critical(self, "File reading error", str(error))

    def load_selected_frame(self):
        self.update_frame_counter()
        if self.current_file is None:
            return

        if self.file_type == "EDF":
            frame_index = self.frame_spin.value() - 1
            try:
                image, header, raw_header_text, byte_order, n_frames = read_edf_frame(self.current_file, frame_index)
                self.h5_n_frames = n_frames
                self.header = header
                self.raw_header_text = raw_header_text
                self.byte_order = byte_order
                self.image = image.astype(np.float64)
                self.image_clean = None
                self.image_filled = None
                self.cave_mask = None
                self.manual_cave_shapes = []
                self.manual_cave_exclusion_shapes = []
                self.manual_cave_pre_nan_shapes = []
                self.update_manual_mask_status_label()

                if not self.lock_intensity_checkbox.isChecked():
                    self.auto_set_display_limits()

                self.apply_instrument_preset()
                self.update_beamstop_visibility()
                self.update_frame_selector_visibility()
                self.refresh_preview()
                self.update_manual_mask_button_state()
                self.update_status()

            except Exception as error:
                QMessageBox.critical(self, "Frame reading error", str(error))
            return

        if self.file_type != "H5" or self.h5_dataset_name is None:
            return

        frame_index = self.frame_spin.value() - 1

        try:
            image, header = read_h5_frame(self.current_file, self.h5_dataset_name, frame_index, add_matching_center=False)
            for key in ["Center_1", "Center_2", "center_1", "center_2", "Center source"]:
                if key in self.header and key not in header:
                    header[key] = self.header[key]
            self.header = header
            self.image = image.astype(np.float64)
            self.image_clean = None
            self.image_filled = None
            self.cave_mask = None
            self.manual_cave_shapes = []
            self.manual_cave_exclusion_shapes = []
            self.manual_cave_pre_nan_shapes = []
            self.update_manual_mask_status_label()

            if not self.lock_intensity_checkbox.isChecked():
                self.auto_set_display_limits()

            self.apply_instrument_preset()
            self.update_beamstop_visibility()
            self.update_frame_selector_visibility()
            self.refresh_preview()
            self.update_manual_mask_button_state()
            self.update_status()

        except Exception as error:
            QMessageBox.critical(self, "Frame reading error", str(error))

    def load_selected_h5_frame(self):
        self.load_selected_frame()

    def manual_mask_for_shape(self, shape, mode="include"):
        shapes_by_mode = {
            "include": self.manual_cave_shapes,
            "exclude": self.manual_cave_exclusion_shapes,
            "pre_nan": self.manual_cave_pre_nan_shapes,
        }
        shapes = shapes_by_mode.get(mode, [])
        if not shapes:
            return None

        mask = np.zeros(shape, dtype=bool)
        original_image = self.image
        try:
            self.image = np.zeros(shape, dtype=np.float64)
            for manual_shape in shapes:
                self.shape_to_mask(mask, manual_shape)
        finally:
            self.image = original_image

        return mask

    def cave_filled_image_for(self, image):
        image = image.astype(np.float64)
        extra_operator, extra_threshold = self.extra_nan_condition()
        use_id13_beamstop = self.instrument_mode == "ID13" and self.id13_beamstop_checkbox.isChecked()

        _clean, filled, _mask = apply_central_symmetry_cave(
            image,
            self.xc_spin.value(),
            self.yc_spin.value(),
            nan_operator=self.nan_operator_combo.currentText(),
            nan_threshold=self.nan_threshold_spin.value(),
            nan_operator_2=extra_operator,
            nan_threshold_2=extra_threshold,
            use_id13_beamstop=use_id13_beamstop,
            beamstop_y=self.beamstop_y_spin.value(),
            reference_angle_deg=self.cave_angle_spin.value(),
            expand_nan_neighbors=self.expand_nan_neighbors_checkbox.isChecked(),
            pre_nan_mask=self.manual_mask_for_shape(image.shape, "pre_nan"),
            extra_mask=self.manual_mask_for_shape(image.shape, "include"),
            exclude_mask=self.manual_mask_for_shape(image.shape, "exclude"),
        )

        return filled

    def write_cave_h5_stack(self, path, dataset_name, frame_axis, n_frames, output_path, progress_callback=None):
        first_image, _header = read_h5_frame(path, dataset_name, 0, add_matching_center=False)
        output_h5, output_dataset = create_h5_cave_stack_file(
            output_path,
            first_image.shape,
            n_frames,
            frame_axis,
            path,
            dataset_name,
        )
        try:
            for frame_index in range(int(n_frames)):
                image = first_image if frame_index == 0 else read_h5_frame(path, dataset_name, frame_index, add_matching_center=False)[0]
                filled = self.cave_filled_image_for(image)
                write_h5_stack_frame(output_dataset, frame_axis, frame_index, filled)
                if progress_callback is not None:
                    progress_callback(frame_index + 1)
        finally:
            output_h5.close()

        return output_path

    def batch_cave_single_file_fast(self, path, progress_callback=None):
        path = Path(path)
        suffix = path.suffix.lower()
        self.commit_nan_threshold_edits()

        extra_operator, extra_threshold = self.extra_nan_condition()
        use_id13_beamstop = self.instrument_mode == "ID13" and self.id13_beamstop_checkbox.isChecked()

        if suffix == ".edf":
            targets = self.cave_processing_targets(path)
            _, _, raw_header_text, byte_order = read_edf_frames(path) if self.cave_scope_is_all() else read_edf_file(path)
            saved_paths = []
            for frame_number, image in targets:
                image = image.astype(np.float64)

                _clean, filled, _mask = apply_central_symmetry_cave(
                    image,
                    self.xc_spin.value(),
                    self.yc_spin.value(),
                    nan_operator=self.nan_operator_combo.currentText(),
                    nan_threshold=self.nan_threshold_spin.value(),
                    nan_operator_2=extra_operator,
                    nan_threshold_2=extra_threshold,
                    use_id13_beamstop=use_id13_beamstop,
                    beamstop_y=self.beamstop_y_spin.value(),
                    reference_angle_deg=self.cave_angle_spin.value(),
                    expand_nan_neighbors=self.expand_nan_neighbors_checkbox.isChecked(),
                    pre_nan_mask=self.manual_mask_for_shape(image.shape, "pre_nan"),
                    extra_mask=self.manual_mask_for_shape(image.shape, "include"),
                    exclude_mask=self.manual_mask_for_shape(image.shape, "exclude"),
                )

                is_current_loaded_file = (
                    self.current_file is not None
                    and Path(self.current_file).resolve() == path.resolve()
                )
                frame_suffix = (
                    f"_frame{frame_number + 1:04d}"
                    if (self.cave_scope_is_all() and len(targets) > 1)
                    or (is_current_loaded_file and self.h5_n_frames > 1)
                    else ""
                )
                output_path = path.parent / f"{path.stem}{frame_suffix}_cave.edf"
                write_edf_file(output_path, sanitize_cave_output_image(filled), raw_header_text, byte_order)
                saved_paths.append(output_path)
                if progress_callback is not None:
                    progress_callback()

            return saved_paths[-1] if saved_paths else None

        if suffix in [".h5", ".hdf5"]:
            dataset_name, _dataset_shape, frame_axis, n_frames, _header = inspect_h5_image_dataset(path)
            if self.cave_scope_is_all() and int(n_frames) > 1:
                output_path = path.parent / f"{path.stem}_cave.h5"
                self.write_cave_h5_stack(
                    path,
                    dataset_name,
                    frame_axis,
                    n_frames,
                    output_path,
                    progress_callback=lambda _current: progress_callback() if progress_callback is not None else None,
                )
                return output_path

            saved_paths = []
            for frame_number, image in self.cave_processing_targets(path):
                filled = self.cave_filled_image_for(image)

                frame_suffix = f"_frame{frame_number:04d}" if int(n_frames) > 1 else ""
                output_path = path.parent / f"{path.stem}{frame_suffix}_cave.h5"
                write_h5_frame_file(output_path, filled, path, dataset_name, frame_number - 1)
                saved_paths.append(output_path)
                if progress_callback is not None:
                    progress_callback()

            return saved_paths[-1] if saved_paths else None

        raise ValueError(f"Unsupported file format: {path.suffix}")

    def selected_file_paths_for_batch(self):
        paths = []
        for item in self.file_list.selectedItems():
            path = item.data(Qt.UserRole)
            if path is None:
                path = self.current_folder / item.text()
            paths.append(Path(path))
        return paths

    def cave_progress_total_for_path(self, path):
        path = Path(path)
        if not self.cave_scope_is_all():
            return 1

        suffix = path.suffix.lower()
        if suffix == ".edf":
            frames, *_ = read_edf_frames(path)
            return max(1, len(frames))

        if suffix in [".h5", ".hdf5"]:
            _dataset_name, _dataset_shape, _frame_axis, n_frames, _header = inspect_h5_image_dataset(path)
            return max(1, int(n_frames))

        return 1

    def start_cave_progress(self, total, text="0 / {total}"):
        total = max(1, int(total))
        self.batch_progress.setVisible(True)
        self.batch_progress.setRange(0, total)
        self.batch_progress.setValue(0)
        self.batch_progress.setFormat(text.format(current=0, total=total))
        QCoreApplication.processEvents()

    def update_cave_progress(self, current, total, text="{current} / {total}"):
        total = max(1, int(total))
        current = max(0, min(int(current), total))
        self.batch_progress.setValue(current)
        self.batch_progress.setFormat(text.format(current=current, total=total))
        QCoreApplication.processEvents()

    def stop_cave_progress(self):
        self.batch_progress.setVisible(False)

    def current_cave_scope(self):
        if hasattr(self, "cave_scope_combo"):
            return self.cave_scope_combo.currentText()
        return "Current image"

    def cave_scope_is_all(self):
        return self.current_cave_scope() == "All frames/scans in file"

    def cave_processing_targets(self, path):
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix == ".edf":
            if self.cave_scope_is_all():
                frames, *_ = read_edf_frames(path)
                return [(index, frame) for index, frame in enumerate(frames)]

            if self.current_file is not None and Path(self.current_file).resolve() == path.resolve() and self.image is not None:
                return [(max(0, self.frame_spin.value() - 1), self.image)]

            image, *_ = read_edf_file(path)
            return [(0, image)]

        if suffix in [".h5", ".hdf5"]:
            dataset_name, _dataset_shape, _frame_axis, n_frames, _header = inspect_h5_image_dataset(path)
            if self.cave_scope_is_all():
                frames = []
                for frame_number in range(1, int(n_frames) + 1):
                    image, _header = read_h5_frame(path, dataset_name, frame_number - 1, add_matching_center=False)
                    frames.append((frame_number, image))
                return frames

            current_frame = 0
            if self.current_file is not None and Path(self.current_file).resolve() == path.resolve() and self.file_type == "H5":
                current_frame = max(0, min(self.frame_spin.value() - 1, int(n_frames) - 1))
            image, _header = read_h5_frame(path, dataset_name, current_frame, add_matching_center=False)
            return [(current_frame + 1, image)]

        raise ValueError(f"Unsupported file format: {path.suffix}")

    def cave_selected_files(self):
        paths = self.selected_file_paths_for_batch()
        if not paths:
            QMessageBox.warning(self, "Cave selected", "No file selected.")
            return

        if self._batch_cave_running:
            return

        self._batch_cave_running = True
        self.batch_cave_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.save_button.setEnabled(False)

        original_file = self.current_file
        original_frame = self.frame_spin.value() if hasattr(self, "frame_spin") else 1
        saved_count = 0
        errors = []
        total_units = 0
        completed_units = 0

        try:
            for path in paths:
                try:
                    total_units += self.cave_progress_total_for_path(path)
                except Exception:
                    total_units += 1
            self.start_cave_progress(total_units, "Cave: {current} / {total}")

            def advance_progress():
                nonlocal completed_units
                completed_units += 1
                self.update_cave_progress(completed_units, total_units, "Cave: {current} / {total}")

            for i, path in enumerate(paths, 1):
                try:
                    self.batch_cave_single_file_fast(path, progress_callback=advance_progress)
                    saved_count += 1
                except Exception as error:
                    errors.append(f"{path.name}: {error}")
                    if not self.cave_scope_is_all():
                        advance_progress()

            if original_file is not None and Path(original_file).exists():
                try:
                    self.open_file(original_file)
                    if hasattr(self, "frame_spin"):
                        self.frame_spin.setValue(original_frame)
                except Exception:
                    pass

            message = f"Batch cave finished: {saved_count} / {len(paths)} files saved."
            self.status.append("\n" + message)
            if errors:
                self.status.append("\nErrors:\n" + "\n".join(errors))
                QMessageBox.warning(
                    self,
                    "Cave selected",
                    message + "\n\nSome files failed:\n" + "\n".join(errors[:8]),
                )
            else:
                QMessageBox.information(self, "Cave selected", message)
        finally:
            self._batch_cave_running = False
            self.stop_cave_progress()
            self.run_button.setEnabled(self.image is not None)
            self.save_button.setEnabled(self.image is not None)
            self.batch_cave_button.setEnabled(self.image is not None)

    def refresh_preview(self):
        if self.image is None:
            return

        self.commit_nan_threshold_edits()
        use_id13_beamstop = self.instrument_mode == "ID13" and self.id13_beamstop_checkbox.isChecked()
        extra_operator, extra_threshold = self.extra_nan_condition()

        clean, filled, cave_mask = apply_central_symmetry_cave(
            self.image,
            self.xc_spin.value(),
            self.yc_spin.value(),
            nan_operator=self.nan_operator_combo.currentText(),
            nan_threshold=self.nan_threshold_spin.value(),
            nan_operator_2=extra_operator,
            nan_threshold_2=extra_threshold,
            use_id13_beamstop=use_id13_beamstop,
            beamstop_y=self.beamstop_y_spin.value(),
            reference_angle_deg=self.cave_angle_spin.value(),
            expand_nan_neighbors=self.expand_nan_neighbors_checkbox.isChecked(),
            pre_nan_mask=self.manual_cave_pre_nan_mask(),
            extra_mask=self.manual_cave_mask(),
            exclude_mask=self.manual_cave_exclusion_mask(),
        )

        self.image_clean = clean
        self.image_filled = filled
        self.cave_mask = cave_mask
        vmin, vmax = self.current_display_limits()
        angle = self.cave_angle_spin.value()
        self.canvas_original.show_image(
            self.image,
            self.xc_spin.value(),
            self.yc_spin.value(),
            vmin=vmin,
            vmax=vmax,
            white_mask=cave_mask,
            reference_angle_deg=angle,
        )
        self.canvas_cave.show_image(
            filled,
            self.xc_spin.value(),
            self.yc_spin.value(),
            vmin=vmin,
            vmax=vmax,
            reference_angle_deg=angle,
        )
        self.canvas_cave.apply_synced_limits_from(self.canvas_original)

    def run_cave(self):
        if self.image is None:
            return

        self.refresh_preview()
        self.update_status()

        if self.save_checkbox.isChecked():
            self.save_cave()

    def save_cave(self, show_message=True):
        if self.image_filled is None or self.current_file is None:
            return

        if self.file_type == "EDF":
            try:
                if self.cave_scope_is_all():
                    frames, _, raw_header_text, byte_order = read_edf_frames(self.current_file)
                    total_frames = max(1, len(frames))
                    self.start_cave_progress(total_frames, "Cave: {current} / {total}")
                    saved_paths = []
                    for frame_number, frame in enumerate(frames):
                        _, filled, _ = apply_central_symmetry_cave(
                            frame.astype(np.float64),
                            self.xc_spin.value(),
                            self.yc_spin.value(),
                            nan_operator=self.nan_operator_combo.currentText(),
                            nan_threshold=self.nan_threshold_spin.value(),
                            nan_operator_2=self.extra_nan_condition()[0],
                            nan_threshold_2=self.extra_nan_condition()[1],
                            use_id13_beamstop=self.instrument_mode == "ID13" and self.id13_beamstop_checkbox.isChecked(),
                            beamstop_y=self.beamstop_y_spin.value(),
                            reference_angle_deg=self.cave_angle_spin.value(),
                            expand_nan_neighbors=self.expand_nan_neighbors_checkbox.isChecked(),
                            pre_nan_mask=self.manual_mask_for_shape(frame.shape, "pre_nan"),
                            extra_mask=self.manual_mask_for_shape(frame.shape, "include"),
                            exclude_mask=self.manual_mask_for_shape(frame.shape, "exclude"),
                        )
                        frame_suffix = f"_frame{frame_number + 1:04d}" if len(frames) > 1 else ""
                        output_path = self.current_file.parent / f"{self.current_file.stem}{frame_suffix}_cave.edf"
                        write_edf_file(output_path, sanitize_cave_output_image(filled), raw_header_text, byte_order)
                        saved_paths.append(output_path)
                        self.update_cave_progress(frame_number + 1, total_frames, "Cave: {current} / {total}")
                    self.stop_cave_progress()
                    self.status.append(f"\nSaved cave EDF(s):\n" + "\n".join(str(path) for path in saved_paths))
                else:
                    frame_suffix = f"_frame{self.frame_spin.value():04d}" if self.h5_n_frames > 1 else ""
                    output_path = self.current_file.parent / f"{self.current_file.stem}{frame_suffix}_cave.edf"
                    write_edf_file(output_path, sanitize_cave_output_image(self.image_filled), self.raw_header_text, self.byte_order)
                    self.status.append(f"\nSaved cave EDF:\n{output_path}")
            except Exception as error:
                self.stop_cave_progress()
                QMessageBox.critical(self, "Save error", str(error))

        else:
            try:
                if self.cave_scope_is_all():
                    dataset_name, _dataset_shape, frame_axis, n_frames, _header = inspect_h5_image_dataset(self.current_file)
                    total_frames = max(1, int(self.h5_n_frames))
                    self.start_cave_progress(total_frames, "Cave: {current} / {total}")
                    output_path = self.current_file.parent / f"{self.current_file.stem}_cave.h5"
                    self.write_cave_h5_stack(
                        self.current_file,
                        dataset_name,
                        frame_axis,
                        n_frames,
                        output_path,
                        progress_callback=lambda current: self.update_cave_progress(
                            current,
                            total_frames,
                            "Cave: {current} / {total}",
                        ),
                    )
                    self.stop_cave_progress()
                    self.status.append(f"\nSaved cave H5:\n{output_path}")
                else:
                    frame_suffix = f"_frame{self.frame_spin.value():04d}" if self.h5_n_frames > 1 else ""
                    output_path = self.current_file.parent / f"{self.current_file.stem}{frame_suffix}_cave.h5"
                    write_h5_frame_file(
                        output_path,
                        self.image_filled,
                        self.current_file,
                        self.h5_dataset_name or "data",
                        self.frame_spin.value() - 1,
                    )
                    self.status.append(f"\nSaved cave H5:\n{output_path}")
            except Exception as error:
                self.stop_cave_progress()
                QMessageBox.critical(self, "Save error", str(error))

    def update_status(self):
        if self.current_file is None:
            return

        lines = [
            f"File: {self.current_file.name}",
            f"Format: {self.file_type}",
        ]

        if self.file_type == "H5" and "Dataset" in self.header:
            lines.append(f"Dataset: {self.header['Dataset']}")

        if self.file_type in {"EDF", "H5"} and self.h5_n_frames > 1:
            lines.append(f"Frame: {self.frame_spin.value()} / {self.h5_n_frames}")
            if self.h5_frame_axis is not None:
                lines.append(f"Frame axis: {self.h5_frame_axis}")

        if self.image is not None:
            lines.append(f"Image size: {self.image.shape[1]} x {self.image.shape[0]}")

        self.status.setPlainText("\n".join(lines))

    def update_manual_mask_status_label(self):
        include_count = len(self.manual_cave_shapes)
        exclude_count = len(self.manual_cave_exclusion_shapes)
        pre_nan_count = len(self.manual_cave_pre_nan_shapes)

        if include_count + exclude_count + pre_nan_count == 0:
            self.manual_mask_status_label.setText("Manual mask: none")
        else:
            self.manual_mask_status_label.setText(
                f"Manual mask: Cave + {include_count} | "
                f"Exclude - {exclude_count} | "
                f"NaN {pre_nan_count}"
            )
