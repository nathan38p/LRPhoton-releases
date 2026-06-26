import json
from pathlib import Path

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QAction, QBrush, QColor, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QListWidget,
    QMenu,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
)


FILE_PATH_ROLE = Qt.UserRole
FILE_RATING_ROLE = Qt.UserRole + 10


def _ratings_file_path():
    return Path.home() / ".lrphoton" / "file_ratings.json"


def _rating_key(file_path):
    name = Path(str(file_path)).name
    return name or str(file_path)


class FileRatingStore:
    def __init__(self, path=None):
        self.path = Path(path) if path else _ratings_file_path()
        self.ratings = {}
        self.load()

    def load(self):
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.ratings = {}
            return

        ratings = data.get("ratings", {}) if isinstance(data, dict) else {}
        self.ratings = {
            _rating_key(path): rating
            for path, rating in ratings.items()
            if rating in {"up", "down"}
        }

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        data = {"version": 1, "ratings": self.ratings}
        tmp_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(self.path)

    def get(self, file_path):
        return self.ratings.get(_rating_key(file_path))

    def set(self, file_path, rating):
        key = _rating_key(file_path)
        if rating in {"up", "down"}:
            self.ratings[key] = rating
        else:
            self.ratings.pop(key, None)
        self.save()


_STORE = FileRatingStore()
_DEFAULT_FILE_ICON = "💿"
_GRAPH_FILE_ICON = "📈"
_THUMBS_UP_ICON = "👍"
_THUMBS_DOWN_ICON = "👎"


def file_prefix_icon(file_path, rating=None):
    if rating == "up":
        return _THUMBS_UP_ICON
    if rating == "down":
        return _THUMBS_DOWN_ICON

    name = Path(str(file_path)).name.lower()
    if (
        name.endswith(".dat")
        or name.endswith(("_ave.h5", "_ave.hdf5", "_averaged.h5", "_averaged.hdf5"))
        or "averaged" in name
    ):
        return _GRAPH_FILE_ICON
    if name.endswith((".h5", ".hdf5", ".edf")):
        return _DEFAULT_FILE_ICON
    return _DEFAULT_FILE_ICON


def should_hide_file_in_browser(file_path):
    path = Path(str(file_path))
    name = path.name.lower()
    if name.startswith("._") or name == ".ds_store":
        return True
    if name.endswith("cleannan.edf"):
        return True
    return path.suffix.lower() in {".h5", ".hdf5"} and "_composite" in name


class FileRatingDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        rating = index.data(FILE_RATING_ROLE)
        item_option = QStyleOptionViewItem(option)
        self.initStyleOption(item_option, index)
        file_path = index.data(FILE_PATH_ROLE) or item_option.text
        item_option.text = f"{file_prefix_icon(file_path, rating)} {item_option.text}"

        widget = option.widget
        style = widget.style() if widget is not None else QApplication.style()
        style.drawControl(QStyle.CE_ItemViewItem, item_option, painter, widget)


def _normalize_path(file_path):
    try:
        return str(Path(file_path).expanduser().resolve())
    except (OSError, RuntimeError):
        return str(file_path)


def _path_for_item(item):
    stored_path = item.data(FILE_PATH_ROLE)
    if stored_path:
        return stored_path
    return item.toolTip() or item.text()


def apply_file_rating_to_item(item, file_path=None):
    path = file_path or _path_for_item(item)
    rating = _STORE.get(path)
    item.setData(FILE_RATING_ROLE, rating)
    item.setIcon(QIcon())

    base_tooltip = _normalize_path(path)
    if rating == "up":
        item.setBackground(QColor("#eaf7ee"))
        item.setForeground(QColor("#146c2e"))
        item.setToolTip(f"{base_tooltip}\nRating: thumbs up")
    elif rating == "down":
        item.setBackground(QColor("#fdecec"))
        item.setForeground(QColor("#9b1c1c"))
        item.setToolTip(f"{base_tooltip}\nRating: thumbs down")
    else:
        item.setBackground(QBrush())
        item.setForeground(QBrush())
        item.setToolTip(base_tooltip)


def set_item_file_path(item, file_path):
    normalized = _normalize_path(file_path)
    item.setData(FILE_PATH_ROLE, normalized)
    apply_file_rating_to_item(item, normalized)


def is_file_rated_up(file_path):
    return _STORE.get(file_path) == "up"


def file_path_from_item(item, fallback_folder=None):
    stored_path = item.data(FILE_PATH_ROLE)
    if stored_path:
        return Path(stored_path)
    if fallback_folder is not None:
        return Path(fallback_folder) / item.text()
    return Path(item.text())


def install_file_rating_menu(file_list: QListWidget):
    file_list.setContextMenuPolicy(Qt.CustomContextMenu)
    file_list.setItemDelegate(FileRatingDelegate(file_list))

    def open_menu(position: QPoint):
        item = file_list.itemAt(position)
        if item is None:
            return

        path = _path_for_item(item)
        menu = QMenu(file_list)

        thumbs_up_action = QAction("👍  Mark as good", menu)
        thumbs_down_action = QAction("👎  Mark as bad", menu)
        clear_action = QAction("Clear rating", menu)

        menu.addAction(thumbs_up_action)
        menu.addAction(thumbs_down_action)
        menu.addSeparator()
        menu.addAction(clear_action)

        action = menu.exec(file_list.viewport().mapToGlobal(position))
        if action is thumbs_up_action:
            _STORE.set(path, "up")
        elif action is thumbs_down_action:
            _STORE.set(path, "down")
        elif action is clear_action:
            _STORE.set(path, None)
        else:
            return

        apply_file_rating_to_item(item, path)

    file_list.customContextMenuRequested.connect(open_menu)
