
from pathlib import Path

import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QListWidget,
    QGroupBox,
    QCheckBox,
    QGridLayout,
    QLineEdit,
    QDoubleSpinBox,
    QScrollArea,
    QComboBox,
    QColorDialog,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QMessageBox,
)
from PySide6.QtGui import QColor

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


# ============================================================
# ========================== TOOLS ============================
# ============================================================

def read_dat_curve(file_path):
    file_path = Path(file_path)
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    text = text.replace(",", ".")

    data = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        if not any(char.isdigit() for char in line):
            continue

        for separator in [";", "\t", ","]:
            line = line.replace(separator, " ")

        values = []
        for part in line.split():
            try:
                values.append(float(part))
            except ValueError:
                pass

        if len(values) >= 2:
            data.append([values[0], values[1]])

    if not data:
        raise ValueError("No valid numerical data found in this file.")

    array = np.asarray(data, dtype=float)
    valid = np.isfinite(array[:, 0]) & np.isfinite(array[:, 1])
    array = array[valid]

    if array.size == 0:
        raise ValueError("No finite numerical data found in this file.")

    order = np.argsort(array[:, 0])
    array = array[order]
    return array[:, 0], array[:, 1]


def default_color(index):
    palette = [
        "#e91e63", "#9c27b0", "#f44336", "#4caf50", "#2196f3",
        "#000000", "#ff9800", "#009688", "#795548", "#607d8b",
    ]
    return palette[index % len(palette)]


# ============================================================
# =========================== CANVAS ==========================
# ============================================================

class PlotCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(dpi=150)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setMinimumSize(620, 420)
        self.fig.subplots_adjust(left=0.12, right=0.98, top=0.92, bottom=0.18)


# ============================================================
# ========================= DAT PLOT TAB ======================
# ============================================================

class DatPlotTab(QWidget):
    """Plot tab: display and compare .dat curves."""

    folder_changed = Signal(Path)

    def __init__(self):
        super().__init__()

        self.current_folder = Path("/Users/nathanpiaget/Documents/Thèse LRP/Expériences/XENOCS")
        self.curves = {}
        self._syncing_folder = False

        self.build_ui()
        self.refresh_files()

    def build_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(8)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setFixedWidth(280)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)
        left_scroll.setWidget(left_panel)
        main_layout.addWidget(left_scroll, stretch=0)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)
        main_layout.addWidget(right_panel, stretch=1)

        file_box = QGroupBox("File browser")
        file_layout = QVBoxLayout(file_box)
        file_layout.setContentsMargins(8, 18, 8, 8)
        file_layout.setSpacing(6)
        left_layout.addWidget(file_box, stretch=0)
        file_box.setFixedHeight(260)

        self.folder_path = QLineEdit(str(self.current_folder))
        file_layout.addWidget(self.folder_path)

        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.choose_folder)
        file_layout.addWidget(self.browse_button)

        filters_layout = QGridLayout()
        self.extensions_filter = QLineEdit("*.dat")
        self.name_filter = QLineEdit("**")
        self.extensions_filter.textChanged.connect(self.refresh_files)
        self.name_filter.textChanged.connect(self.refresh_files)
        filters_layout.addWidget(QLabel("Extensions:"), 0, 0)
        filters_layout.addWidget(self.extensions_filter, 0, 1)
        filters_layout.addWidget(QLabel("Name:"), 1, 0)
        filters_layout.addWidget(self.name_filter, 1, 1)
        file_layout.addLayout(filters_layout)

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_files)
        file_layout.addWidget(self.refresh_button)

        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.file_list.itemSelectionChanged.connect(self.selection_changed)
        file_layout.addWidget(self.file_list, stretch=1)

        settings_box = QGroupBox("Plot settings")
        settings_layout = QGridLayout(settings_box)
        settings_layout.setContentsMargins(8, 18, 8, 8)
        settings_layout.setVerticalSpacing(5)
        settings_layout.setHorizontalSpacing(6)
        left_layout.addWidget(settings_box, stretch=0)

        self.plot_mode = QComboBox()
        self.plot_mode.addItems(["linear linear", "linear log", "log linear", "log log", "Kratky"])
        self.plot_mode.currentTextChanged.connect(self.update_plot)

        self.auto_limits = QCheckBox("Auto limits")
        self.auto_limits.setChecked(True)
        self.auto_limits.stateChanged.connect(self.update_limit_state)

        self.x_min = self.double_spin(0.0)
        self.x_max = self.double_spin(1.0)
        self.y_min = self.double_spin(0.0)
        self.y_max = self.double_spin(1.0)

        self.x_label = QLineEdit("q / nm⁻¹")
        self.y_label = QLineEdit("Intensity / a.u.")
        self.title_edit = QLineEdit("")

        self.x_label.textChanged.connect(self.update_plot)
        self.y_label.textChanged.connect(self.update_plot)
        self.title_edit.textChanged.connect(self.update_plot)

        for spin in [self.x_min, self.x_max, self.y_min, self.y_max]:
            spin.valueChanged.connect(self.update_plot)

        settings_layout.addWidget(QLabel("Mode:"), 0, 0)
        settings_layout.addWidget(self.plot_mode, 0, 1)
        settings_layout.addWidget(self.auto_limits, 1, 0, 1, 2)
        settings_layout.addWidget(QLabel("x min:"), 2, 0)
        settings_layout.addWidget(self.x_min, 2, 1)
        settings_layout.addWidget(QLabel("x max:"), 3, 0)
        settings_layout.addWidget(self.x_max, 3, 1)
        settings_layout.addWidget(QLabel("y min:"), 4, 0)
        settings_layout.addWidget(self.y_min, 4, 1)
        settings_layout.addWidget(QLabel("y max:"), 5, 0)
        settings_layout.addWidget(self.y_max, 5, 1)
        settings_layout.addWidget(QLabel("x label:"), 6, 0)
        settings_layout.addWidget(self.x_label, 6, 1)
        settings_layout.addWidget(QLabel("y label:"), 7, 0)
        settings_layout.addWidget(self.y_label, 7, 1)
        settings_layout.addWidget(QLabel("Title:"), 8, 0)
        settings_layout.addWidget(self.title_edit, 8, 1)

        curve_box = QGroupBox("Curves / legend")
        curve_layout = QVBoxLayout(curve_box)
        curve_layout.setContentsMargins(8, 18, 8, 8)
        curve_layout.setSpacing(6)
        left_layout.addWidget(curve_box, stretch=1)

        self.curve_table = QTableWidget(0, 3)
        self.curve_table.setMinimumHeight(140)
        self.curve_table.setMaximumHeight(220)
        self.curve_table.setHorizontalHeaderLabels(["File", "Legend", "Color"])
        self.curve_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.curve_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.curve_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.curve_table.cellChanged.connect(self.curve_table_changed)
        self.curve_table.cellDoubleClicked.connect(self.curve_table_double_clicked)
        curve_layout.addWidget(self.curve_table)

        buttons_layout = QHBoxLayout()
        self.update_button = QPushButton("Update plot")
        self.update_button.clicked.connect(self.update_plot)
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_curves)
        buttons_layout.addWidget(self.update_button)
        buttons_layout.addWidget(self.clear_button)
        curve_layout.addLayout(buttons_layout)

        graph_box = QGroupBox("Plot")
        graph_layout = QVBoxLayout(graph_box)
        graph_layout.setContentsMargins(6, 18, 6, 6)
        right_layout.addWidget(graph_box, stretch=1)

        self.canvas = PlotCanvas()
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setIconSize(self.toolbar.iconSize() * 0.8)
        self.toolbar.setMaximumHeight(42)
        self.toolbar.setStyleSheet("""
            QToolBar {
                background: #f4f4f4;
                background-color: #f4f4f4;
                border: none;
                spacing: 6px;
                padding: 4px;
            }
            QToolButton {
                background: transparent;
                background-color: transparent;
            }
        """)
        graph_layout.addWidget(self.toolbar)
        graph_layout.addWidget(self.canvas)

        self.update_limit_state()

    def double_spin(self, value):
        spin = QDoubleSpinBox()
        spin.setDecimals(6)
        spin.setRange(-1e12, 1e12)
        spin.setValue(value)
        spin.setFixedHeight(24)
        spin.setFixedWidth(90)
        return spin

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose folder", str(self.current_folder))
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
        self.folder_path.setText(str(self.current_folder))
        self.refresh_files()
        self._syncing_folder = False

    def refresh_files(self):
        folder = Path(self.folder_path.text()).expanduser()
        if not folder.exists():
            return

        self.current_folder = folder
        if not self._syncing_folder:
            self.folder_changed.emit(self.current_folder)

        patterns = self.extensions_filter.text().split()
        if not patterns:
            patterns = ["*.dat"]

        name_filter = self.name_filter.text().strip()
        if not name_filter:
            name_filter = "**"

        from fnmatch import fnmatch
        files = []
        for pattern in patterns:
            files.extend(folder.glob(pattern))

        files = sorted(set(files))
        files = [file for file in files if fnmatch(file.name, name_filter)]

        self.file_list.blockSignals(True)
        self.file_list.clear()
        for file in files:
            self.file_list.addItem(file.name)
        self.file_list.blockSignals(False)

    def selection_changed(self):
        selected = self.selected_files()
        if not selected:
            return

        for file_path in selected:
            key = file_path.name
            if key in self.curves:
                continue

            try:
                x, y = read_dat_curve(file_path)
            except Exception as error:
                QMessageBox.warning(self, "File reading error", f"{file_path.name}\n\n{error}")
                continue

            index = len(self.curves)
            self.curves[key] = {
                "path": file_path,
                "x": x,
                "y": y,
                "legend": file_path.stem,
                "color": default_color(index),
            }

        self.refresh_curve_table()
        self.update_plot()

    def selected_files(self):
        return [self.current_folder / item.text() for item in self.file_list.selectedItems()]

    def refresh_curve_table(self):
        self.curve_table.blockSignals(True)
        self.curve_table.setRowCount(0)

        for row, (key, curve) in enumerate(self.curves.items()):
            self.curve_table.insertRow(row)

            file_item = QTableWidgetItem(key)
            file_item.setFlags(file_item.flags() & ~Qt.ItemIsEditable)
            self.curve_table.setItem(row, 0, file_item)

            legend_item = QTableWidgetItem(curve["legend"])
            self.curve_table.setItem(row, 1, legend_item)

            color_item = QTableWidgetItem(curve["color"])
            color_item.setBackground(QColor(curve["color"]))
            self.curve_table.setItem(row, 2, color_item)

        self.curve_table.blockSignals(False)

    def curve_table_changed(self, row, column):
        file_item = self.curve_table.item(row, 0)
        if file_item is None:
            return

        key = file_item.text()
        if key not in self.curves:
            return

        if column == 1:
            item = self.curve_table.item(row, column)
            self.curves[key]["legend"] = item.text() if item else self.curves[key]["legend"]
        elif column == 2:
            item = self.curve_table.item(row, column)
            if item:
                self.curves[key]["color"] = item.text()

        self.update_plot()

    def curve_table_double_clicked(self, row, column):
        if column != 2:
            return

        file_item = self.curve_table.item(row, 0)
        if file_item is None:
            return

        key = file_item.text()
        if key not in self.curves:
            return

        color = QColorDialog.getColor(QColor(self.curves[key]["color"]), self, "Choose curve color")
        if not color.isValid():
            return

        self.curves[key]["color"] = color.name()
        self.refresh_curve_table()
        self.update_plot()

    def clear_curves(self):
        self.curves.clear()
        self.refresh_curve_table()
        self.canvas.ax.clear()
        self.canvas.draw_idle()

    def make_plot_y(self, x, y):
        if self.plot_mode.currentText() == "Kratky":
            return x ** 2 * y
        return y

    def update_limit_state(self):
        auto = self.auto_limits.isChecked()

        if auto:
            self.update_limit_fields_from_current_data()

        for widget in [self.x_min, self.x_max, self.y_min, self.y_max]:
            widget.setEnabled(not auto)

        self.update_plot()

    def update_limit_fields_from_current_data(self):
        if not self.curves:
            return

        all_x = []
        all_y = []

        for curve in self.curves.values():
            x = curve["x"]
            y = self.make_plot_y(x, curve["y"])

            valid = np.isfinite(x) & np.isfinite(y)
            if np.any(valid):
                all_x.append(x[valid])
                all_y.append(y[valid])

        if not all_x or not all_y:
            return

        x_values = np.concatenate(all_x)
        has_azim_profile = any(
            curve["path"].name.endswith("azimProf.dat")
            for curve in self.curves.values()
        )
        y_values = np.concatenate(all_y)

        mode = self.plot_mode.currentText()
        if mode in ["log linear", "log log"]:
            x_values = x_values[x_values > 0]
        if mode in ["linear log", "log log"]:
            y_values = y_values[y_values > 0]

        if x_values.size == 0 or y_values.size == 0:
            return

        if has_azim_profile:
            x_min = 0.0
            x_max = 360.0
        else:
            x_min = float(np.nanmin(x_values))
            x_max = float(np.nanmax(x_values))
        y_min = float(np.nanmin(y_values))
        y_max = float(np.nanmax(y_values))

        if x_max == x_min:
            x_max = x_min + 1
        if y_max == y_min:
            y_max = y_min + 1

        for spin, value in [
            (self.x_min, x_min),
            (self.x_max, x_max),
            (self.y_min, y_min),
            (self.y_max, y_max),
        ]:
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)

    def update_plot(self):
        ax = self.canvas.ax
        ax.clear()

        if not self.curves:
            self.canvas.draw_idle()
            return

        mode = self.plot_mode.currentText()
        if self.auto_limits.isChecked():
            self.update_limit_fields_from_current_data()

        for curve in self.curves.values():
            x = curve["x"]
            y = self.make_plot_y(x, curve["y"])
            ax.plot(
                x,
                y,
                linewidth=1.6,
                label=curve["legend"],
                color=curve["color"],
                antialiased=True,
                solid_capstyle="round",
                solid_joinstyle="round",
            )

        if mode == "linear linear" or mode == "Kratky":
            ax.set_xscale("linear")
            ax.set_yscale("linear")
        elif mode == "linear log":
            ax.set_xscale("linear")
            ax.set_yscale("log")
        elif mode == "log linear":
            ax.set_xscale("log")
            ax.set_yscale("linear")
        elif mode == "log log":
            ax.set_xscale("log")
            ax.set_yscale("log")

        has_azim_profile = any(
            curve["path"].name.endswith("azimProf.dat")
            for curve in self.curves.values()
        )

        if has_azim_profile:
            default_x_label = "ψ / °"
            ax.set_xlim(0, 360)
            ax.set_xlabel(default_x_label)
        else:
            default_x_label = "q / nm⁻¹"
            ax.set_xlabel(self.x_label.text() or default_x_label)
        ax.set_ylabel("q²I(q)" if mode == "Kratky" else (self.y_label.text() or "Intensity / a.u."))
        ax.set_title(self.title_edit.text())
        ax.grid(True, linewidth=0.5, alpha=0.35)
        ax.tick_params(axis="both", labelsize=10)
        ax.legend(loc="best", frameon=True, fontsize=9)

        if not self.auto_limits.isChecked():
            if self.x_max.value() > self.x_min.value():
                ax.set_xlim(self.x_min.value(), self.x_max.value())
            if self.y_max.value() > self.y_min.value():
                ax.set_ylim(self.y_min.value(), self.y_max.value())

        self.canvas.fig.tight_layout()
        self.canvas.draw_idle()
