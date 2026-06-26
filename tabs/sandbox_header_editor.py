from pathlib import Path
import fnmatch
import shutil

import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QCheckBox,
    QFrame,
    QGridLayout,
    QSizePolicy,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from tabs.ui_style import (
    BLOCK_SPACING,
    FILE_BROWSER_WIDTH,
    GROUP_BOX_MARGINS,
    GROUP_BOX_STYLE,
    PAGE_MARGINS,
)
from tabs.file_ratings import install_file_rating_menu, set_item_file_path, should_hide_file_in_browser

try:
    import fabio
    from fabio.edfimage import EdfImage
except Exception:
    fabio = None
    EdfImage = None


class HeaderEditorTab(QWidget):
    folder_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_folder = Path.home()
        self._syncing_folder = False
        self.current_file = None
        self.current_data = None
        self.current_header = {}
        self.build_ui()

    def build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(*PAGE_MARGINS)
        main_layout.setSpacing(BLOCK_SPACING)

        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(BLOCK_SPACING)
        main_layout.addLayout(content_layout, 1)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setFixedWidth(FILE_BROWSER_WIDTH)
        left_scroll.setFrameShape(QFrame.NoFrame)
        content_layout.addWidget(left_scroll, 0)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(BLOCK_SPACING)
        left_scroll.setWidget(left_panel)

        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(BLOCK_SPACING)
        content_layout.addWidget(center_panel, 1)

        tools_scroll = QScrollArea()
        tools_scroll.setWidgetResizable(True)
        tools_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        tools_scroll.setFixedWidth(FILE_BROWSER_WIDTH)
        tools_scroll.setFrameShape(QFrame.NoFrame)
        content_layout.addWidget(tools_scroll, 0)

        tools_panel = QWidget()
        tools_layout = QVBoxLayout(tools_panel)
        tools_layout.setContentsMargins(0, 0, 0, 0)
        tools_layout.setSpacing(BLOCK_SPACING)
        tools_scroll.setWidget(tools_panel)

        file_box = QGroupBox("File browser")
        file_box.setStyleSheet(GROUP_BOX_STYLE)
        file_layout = QVBoxLayout(file_box)
        file_layout.setContentsMargins(*GROUP_BOX_MARGINS)
        file_layout.setSpacing(6)
        left_layout.addWidget(file_box, 1)

        self.folder_edit = QLineEdit(str(self.current_folder))
        self.folder_edit.setPlaceholderText("Folder")
        self.folder_edit.returnPressed.connect(self.folder_from_text)
        file_layout.addWidget(self.folder_edit)

        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_folder)
        file_layout.addWidget(browse_button)

        filters_layout = QGridLayout()
        filters_layout.setContentsMargins(0, 0, 0, 0)
        filters_layout.setHorizontalSpacing(4)
        filters_layout.setVerticalSpacing(6)
        self.name_filter = QLineEdit("*")
        self.name_filter.returnPressed.connect(self.refresh_files)
        self.extension_filter = QLineEdit("*.edf")
        self.extension_filter.returnPressed.connect(self.refresh_files)
        filters_layout.addWidget(QLabel("Name:"), 0, 0)
        filters_layout.addWidget(self.name_filter, 0, 1)
        filters_layout.addWidget(QLabel("Extensions:"), 1, 0)
        filters_layout.addWidget(self.extension_filter, 1, 1)
        file_layout.addLayout(filters_layout)

        options_layout = QHBoxLayout()
        options_layout.setContentsMargins(0, 0, 0, 0)
        options_layout.setSpacing(4)
        self.subfolders_checkbox = QCheckBox("Show subfolders")
        self.subfolders_checkbox.toggled.connect(self.refresh_files)
        options_layout.addWidget(self.subfolders_checkbox)
        options_layout.addStretch(1)
        file_layout.addLayout(options_layout)

        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh_files)
        file_layout.addWidget(refresh_button)

        self.file_list = QListWidget()
        install_file_rating_menu(self.file_list)
        self.file_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.file_list.itemClicked.connect(self.open_selected_file)
        file_layout.addWidget(self.file_list, 1)

        editor_box = QGroupBox("Header")
        editor_box.setStyleSheet(GROUP_BOX_STYLE)
        editor_layout = QVBoxLayout(editor_box)
        editor_layout.setContentsMargins(*GROUP_BOX_MARGINS)
        editor_layout.setSpacing(6)
        center_layout.addWidget(editor_box, 1)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Key", "Value"])
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.verticalHeader().setDefaultSectionSize(24)
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        editor_layout.addWidget(self.table, 1)

        self.status_label = QLabel("")
        self.status_label.setMinimumHeight(24)
        self.status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        center_layout.addWidget(self.status_label, 0)

        tools_box = QGroupBox("Header tools")
        tools_box.setStyleSheet(GROUP_BOX_STYLE)
        tools_box_layout = QVBoxLayout(tools_box)
        tools_box_layout.setContentsMargins(*GROUP_BOX_MARGINS)
        tools_box_layout.setSpacing(6)
        tools_layout.addWidget(tools_box, 0)

        self.reload_button = QPushButton("Reload")
        self.reload_button.clicked.connect(self.reload_current_file)
        tools_box_layout.addWidget(self.reload_button)

        self.add_row_button = QPushButton("Add header key")
        self.add_row_button.clicked.connect(self.add_header_row)
        tools_box_layout.addWidget(self.add_row_button)

        self.remove_row_button = QPushButton("Remove selected key")
        self.remove_row_button.clicked.connect(self.remove_selected_rows)
        tools_box_layout.addWidget(self.remove_row_button)

        self.save_copy_button = QPushButton("Save as copy")
        self.save_copy_button.clicked.connect(self.save_as_copy)
        tools_box_layout.addWidget(self.save_copy_button)

        self.overwrite_button = QPushButton("Overwrite with backup")
        self.overwrite_button.clicked.connect(self.overwrite_with_backup)
        tools_box_layout.addWidget(self.overwrite_button)

        info_box = QGroupBox("File information")
        info_box.setStyleSheet(GROUP_BOX_STYLE)
        info_layout = QVBoxLayout(info_box)
        info_layout.setContentsMargins(*GROUP_BOX_MARGINS)
        info_layout.setSpacing(6)
        tools_layout.addWidget(info_box, 1)

        self.info_label = QLabel("Open an EDF file to edit its header.")
        self.info_label.setWordWrap(True)
        self.info_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        info_layout.addWidget(self.info_label, 1)

        self.update_buttons()

    def folder_from_text(self):
        self.set_folder(self.folder_edit.text().strip())

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose folder", self.folder_edit.text() or str(Path.home()))
        if folder:
            self.set_folder(folder)

    def set_folder(self, folder):
        if folder is None:
            return
        path = Path(folder).expanduser()
        if not path.exists():
            return
        self.current_folder = path
        self.folder_edit.setText(str(path))
        self.refresh_files()

    def refresh_files(self):
        self.file_list.clear()
        folder = Path(self.folder_edit.text()).expanduser()
        if not folder.exists():
            return

        self.current_folder = folder
        if not self._syncing_folder:
            self.folder_changed.emit(str(folder))

        name_pattern = self.name_filter.text().strip() or "*"
        extension_patterns = self.extension_filter.text().split() or ["*.edf"]
        iterator = folder.rglob("*") if self.subfolders_checkbox.isChecked() else folder.glob("*")

        files = []
        for path in iterator:
            if not path.is_file():
                continue
            if should_hide_file_in_browser(path):
                continue
            lower_name = path.name.lower()
            if not any(fnmatch.fnmatch(lower_name, pattern.lower()) for pattern in extension_patterns):
                continue
            if not fnmatch.fnmatch(path.name.lower(), name_pattern.lower()):
                continue
            files.append(path)

        for path in sorted(files):
            display_name = str(path.relative_to(folder)) if self.subfolders_checkbox.isChecked() else path.name
            item = QListWidgetItem(display_name)
            set_item_file_path(item, path)
            self.file_list.addItem(item)

    def open_selected_file(self):
        path = self.selected_file()
        if path is not None:
            self.load_file(path)

    def selected_file(self):
        item = self.file_list.currentItem()
        if item is None:
            selected = self.file_list.selectedItems()
            item = selected[0] if selected else None
        if item is None:
            return None
        path = item.data(Qt.UserRole)
        if path is None:
            path = Path(self.current_folder) / item.text()
        return Path(path)

    def reload_current_file(self):
        if self.current_file is not None:
            self.load_file(self.current_file)

    def load_file(self, filename):
        if fabio is None:
            self.status_label.setText("fabio is required to read EDF files.")
            return

        path = Path(filename)
        try:
            edf = fabio.open(str(path))
            try:
                data = np.asarray(edf.data)
                header = {str(key): str(value) for key, value in dict(edf.header).items()}
            finally:
                try:
                    edf.close()
                except Exception:
                    pass
        except Exception as exc:
            QMessageBox.warning(self, "EDF read error", str(exc))
            return

        self.current_file = path
        self.current_folder = path.parent
        self.current_data = data
        self.current_header = header
        self.populate_table(header)
        self.info_label.setText(f"{path.name}\nImage: {data.shape} / {data.dtype}\nHeader keys: {len(header)}")
        self.status_label.setText("Header loaded.")
        self.folder_changed.emit(str(path.parent))
        self.update_buttons()

    def populate_table(self, header):
        self.table.setRowCount(0)
        for key in sorted(header):
            self.add_header_row(key, header[key])

    def add_header_row(self, key="", value=""):
        row = self.table.rowCount()
        self.table.insertRow(row)
        key_item = QTableWidgetItem(str(key))
        value_item = QTableWidgetItem(str(value))
        self.table.setItem(row, 0, key_item)
        self.table.setItem(row, 1, value_item)
        self.table.setCurrentCell(row, 0 if not key else 1)
        self.update_buttons()

    def remove_selected_rows(self):
        rows = sorted({index.row() for index in self.table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.table.removeRow(row)
        self.update_buttons()

    def header_from_table(self):
        header = {}
        duplicates = []
        for row in range(self.table.rowCount()):
            key_item = self.table.item(row, 0)
            value_item = self.table.item(row, 1)
            key = "" if key_item is None else key_item.text().strip()
            value = "" if value_item is None else value_item.text().strip()
            if not key:
                continue
            if key in header:
                duplicates.append(key)
            header[key] = value
        if duplicates:
            duplicate_text = ", ".join(sorted(set(duplicates)))
            raise ValueError(f"Duplicate header key(s): {duplicate_text}")
        return header

    def save_as_copy(self):
        if self.current_file is None:
            return
        default_path = self.current_file.with_name(f"{self.current_file.stem}_header.edf")
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save EDF copy",
            str(default_path),
            "EDF files (*.edf);;All files (*)",
        )
        if filename:
            self.write_edf(Path(filename))

    def overwrite_with_backup(self):
        if self.current_file is None:
            return
        answer = QMessageBox.question(
            self,
            "Overwrite EDF",
            f"Overwrite {self.current_file.name}?\nA .bak copy will be created first.",
        )
        if answer != QMessageBox.Yes:
            return

        backup_path = self.next_backup_path(self.current_file)
        try:
            shutil.copy2(self.current_file, backup_path)
            self.write_edf(self.current_file)
            self.status_label.setText(f"Saved. Backup: {backup_path.name}")
        except Exception as exc:
            QMessageBox.warning(self, "EDF save error", str(exc))

    def write_edf(self, output_path):
        if EdfImage is None:
            self.status_label.setText("fabio EDF support is not available.")
            return
        if self.current_data is None:
            return
        try:
            header = self.header_from_table()
            edf = EdfImage(data=np.asarray(self.current_data), header=header)
            edf.write(str(output_path))
        except Exception as exc:
            QMessageBox.warning(self, "EDF save error", str(exc))
            return

        self.status_label.setText(f"Saved EDF: {output_path}")

    def next_backup_path(self, path):
        backup = path.with_suffix(path.suffix + ".bak")
        if not backup.exists():
            return backup
        index = 1
        while True:
            candidate = path.with_suffix(path.suffix + f".bak{index}")
            if not candidate.exists():
                return candidate
            index += 1

    def set_folder_from_external_tab(self, folder):
        if folder:
            path = Path(folder)
            if path.exists():
                self._syncing_folder = True
                self.current_folder = path
                self.folder_edit.setText(str(path))
                self.refresh_files()
                self._syncing_folder = False

    def update_buttons(self):
        has_file = self.current_file is not None and self.current_data is not None
        for button in [
            self.reload_button,
            self.add_row_button,
            self.remove_row_button,
            self.save_copy_button,
            self.overwrite_button,
        ]:
            button.setEnabled(has_file)

    def show_project_selector(self):
        parent = self.parentWidget()
        while parent is not None:
            if hasattr(parent, "setCurrentIndex"):
                parent.setCurrentIndex(0)
                return
            parent = parent.parentWidget()
