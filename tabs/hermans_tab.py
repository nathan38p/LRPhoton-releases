
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
    QGroupBox,
    QDoubleSpinBox,
    QTextEdit,
    QGridLayout,
    QListWidget,
    QMessageBox,
    QSlider,
    QCheckBox,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ============================================================
# =========================== TOOLS ===========================
# ============================================================

def wrap_to_180(angle):
    return (angle + 180) % 360 - 180


def read_azimuthal_file(file_path):
    file_path = Path(file_path)
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    text = text.replace(",", ".")

    data = []

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        if not any(char.isdigit() for char in line):
            continue

        for separator in [";", "\t", ","]:
            line = line.replace(separator, " ")

        parts = line.split()
        values = []

        for part in parts:
            try:
                values.append(float(part))
            except ValueError:
                pass

        if len(values) >= 2:
            data.append([values[0], values[1]])

    if not data:
        raise ValueError("No valid numerical data found in this file.")

    array = np.asarray(data, dtype=float)
    azimuth = array[:, 0]
    intensity = array[:, 1]

    valid = np.isfinite(azimuth) & np.isfinite(intensity)
    azimuth = azimuth[valid]
    intensity = intensity[valid]

    order = np.argsort(azimuth)
    return azimuth[order], intensity[order]


def fit_gaussian_fixed_center(x, y, x0, window):
    if x.size < 10 or np.nanmax(y) <= 0:
        raise ValueError("Not enough valid points for Gaussian fit.")

    sigma_min = max(window / 200, 0.05)
    sigma_max = max(window * 2, sigma_min * 2)

    best = None

    for _ in range(4):
        sigmas = np.linspace(sigma_min, sigma_max, 240)

        for sigma in sigmas:
            g = np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))
            denom = np.sum(g ** 2)

            if denom <= 0:
                continue

            amplitude = np.sum(y * g) / denom
            amplitude = abs(amplitude)
            residual = amplitude * g - y
            sse = float(np.sum(residual ** 2))

            if best is None or sse < best[0]:
                best = (sse, amplitude, sigma)

        if best is None:
            raise ValueError("Gaussian fit failed.")

        best_sigma = best[2]
        span = (sigma_max - sigma_min) / 8
        sigma_min = max(best_sigma - span, 0.01)
        sigma_max = best_sigma + span

    return best[1], abs(best[2])


# ============================================================
# =========================== CANVAS ==========================
# ============================================================

class PlotCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.10)


# ============================================================
# ========================== HERMANS TAB ======================
# ============================================================

class HermansTab(QWidget):
    """Hermans tab: orientation factor, Pi and Gaussian fit from I(ψ) profiles."""
    folder_changed = Signal(Path)

    def __init__(self):
        super().__init__()

        self.folder = None
        self.available_files = []
        self.current_file = None
        self.azimuth = None
        self.intensity = None
        self.last_fit = None

        self.build_ui()
        self.init_default_folder()

    def build_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(8)

        left_panel = QWidget()
        controls_layout = QVBoxLayout(left_panel)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)
        main_layout.addWidget(left_panel, stretch=0)

        graph_wrapper = QWidget()
        graph_wrapper_layout = QVBoxLayout(graph_wrapper)
        graph_wrapper_layout.setContentsMargins(0, 0, 0, 0)
        graph_wrapper_layout.setSpacing(0)

        graph_box = QGroupBox("Azimuthal profile")
        graph_layout = QVBoxLayout(graph_box)
        graph_layout.setContentsMargins(6, 24, 6, 6)
        graph_layout.setSpacing(6)
        graph_wrapper_layout.addWidget(graph_box)
        main_layout.addWidget(graph_wrapper, stretch=1)

        file_browser_box = QGroupBox("File browser")
        file_browser_layout = QVBoxLayout(file_browser_box)
        file_browser_layout.setContentsMargins(8, 18, 8, 8)
        file_browser_layout.setSpacing(6)

        self.open_folder_button = QPushButton("Open folder")
        self.open_folder_button.clicked.connect(self.open_folder)
        file_browser_layout.addWidget(self.open_folder_button)



        self.file_list = QListWidget()
        self.file_list.currentItemChanged.connect(self.load_selected_file)
        file_browser_layout.addWidget(self.file_list, stretch=1)

        controls_layout.addWidget(file_browser_box, stretch=0)
        file_browser_box.setFixedHeight(240)

        params_box = QGroupBox("Parameters")
        params_layout = QGridLayout(params_box)
        params_layout.setContentsMargins(6, 18, 6, 6)
        params_layout.setSpacing(6)
        controls_layout.addWidget(params_box)

        self.offset_spin, self.offset_slider = self.add_slider_control(
            params_layout, 0, "Baseline", 0.0, -1.0, 1.0
        )
        self.peak_spin, self.peak_slider = self.add_slider_control(
            params_layout, 1, "Peak ψ₀ (°)", 85.0, 70.0, 130.0
        )
        self.window_spin, self.window_slider = self.add_slider_control(
            params_layout, 2, "Window (°)", 90.0, 10.0, 180.0
        )

        self.use_fit_checkbox = QCheckBox("Fit")
        self.use_fit_checkbox.setChecked(True)
        self.use_fit_checkbox.stateChanged.connect(self.update_fit_mode)
        params_layout.addWidget(self.use_fit_checkbox, 3, 0, 1, 5)

        self.height_spin, self.height_slider = self.add_slider_control(
            params_layout, 4, "Height", 1.0, 0.0, 10.0
        )
        self.manual_fwhm_spin, self.manual_fwhm_slider = self.add_slider_control(
            params_layout, 5, "FWHM (°)", 90.0, 0.1, 180.0
        )

        self.fit_button = QPushButton("Fit")
        self.fit_button.clicked.connect(self.recenter_on_fit)
        self.save_fit_button = QPushButton("Save fit .dat")
        self.save_fit_button.clicked.connect(self.save_gaussian_fit)
        params_layout.addWidget(self.fit_button, 6, 0, 1, 2)
        params_layout.addWidget(self.save_fit_button, 6, 2, 1, 3)

        results_box = QGroupBox("Results")
        results_layout = QVBoxLayout(results_box)
        results_layout.setContentsMargins(8, 18, 8, 8)
        results_layout.setSpacing(6)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setPlaceholderText("Hermans results will appear here.")
        results_layout.addWidget(self.results_text)
        controls_layout.addWidget(results_box, stretch=1)

        self.canvas = PlotCanvas()
        graph_layout.addWidget(self.canvas)

        for widget in [
            self.peak_spin,
            self.window_spin,
            self.offset_spin,
            self.height_spin,
            self.manual_fwhm_spin,
        ]:
            widget.valueChanged.connect(self.calculate)

        for slider in [
            self.peak_slider,
            self.window_slider,
            self.offset_slider,
            self.height_slider,
            self.manual_fwhm_slider,
        ]:
            slider.valueChanged.connect(self.slider_changed)

        self.update_fit_mode()
    def update_fit_mode(self):
        use_fit = self.use_fit_checkbox.isChecked()

        self.height_spin.setEnabled(not use_fit)
        self.height_slider.setEnabled(not use_fit)
        self.height_slider.min_spin.setEnabled(not use_fit)
        self.height_slider.max_spin.setEnabled(not use_fit)

        self.manual_fwhm_spin.setEnabled(not use_fit)
        self.manual_fwhm_slider.setEnabled(not use_fit)
        self.manual_fwhm_slider.min_spin.setEnabled(not use_fit)
        self.manual_fwhm_slider.max_spin.setEnabled(not use_fit)

        self.fit_button.setEnabled(use_fit)
        self.save_fit_button.setEnabled(True)

        self.calculate()

    def update_manual_fields_from_fit(self, amplitude, fwhm):
        def update_spin_and_slider(spin, slider, value):
            min_spin = slider.min_spin
            max_spin = slider.max_spin

            if value < min_spin.value():
                min_spin.blockSignals(True)
                min_spin.setValue(value)
                min_spin.blockSignals(False)

            if value > max_spin.value():
                max_spin.blockSignals(True)
                max_spin.setValue(value)
                max_spin.blockSignals(False)

            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)

            slider.blockSignals(True)
            slider.setValue(self.value_to_slider(value, min_spin.value(), max_spin.value()))
            slider.blockSignals(False)

        update_spin_and_slider(self.height_spin, self.height_slider, amplitude)
        update_spin_and_slider(self.manual_fwhm_spin, self.manual_fwhm_slider, fwhm)

    def add_slider_control(self, layout, row, label, default, minimum, maximum):
        label_widget = QLabel(label)

        min_spin = QDoubleSpinBox()
        min_spin.setDecimals(3)
        min_spin.setRange(-1e9, 1e9)
        min_spin.setValue(minimum)
        min_spin.setFixedWidth(75)

        max_spin = QDoubleSpinBox()
        max_spin.setDecimals(3)
        max_spin.setRange(-1e9, 1e9)
        max_spin.setValue(maximum)
        max_spin.setFixedWidth(75)

        value_spin = QDoubleSpinBox()
        value_spin.setDecimals(3)
        value_spin.setRange(minimum, maximum)
        value_spin.setValue(default)
        value_spin.setFixedWidth(75)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 1000)
        slider.setValue(self.value_to_slider(default, minimum, maximum))
        slider.setMinimumWidth(180)

        min_spin.valueChanged.connect(lambda: self.update_slider_limits(min_spin, max_spin, value_spin, slider))
        max_spin.valueChanged.connect(lambda: self.update_slider_limits(min_spin, max_spin, value_spin, slider))
        value_spin.valueChanged.connect(lambda: self.spin_changed(min_spin, max_spin, value_spin, slider))

        layout.addWidget(label_widget, row, 0)
        layout.addWidget(min_spin, row, 1)
        layout.addWidget(slider, row, 2)
        layout.addWidget(max_spin, row, 3)
        layout.addWidget(value_spin, row, 4)
        layout.setColumnStretch(2, 1)

        slider.min_spin = min_spin
        slider.max_spin = max_spin
        slider.value_spin = value_spin

        return value_spin, slider

    def value_to_slider(self, value, minimum, maximum):
        if maximum <= minimum:
            return 0
        ratio = (value - minimum) / (maximum - minimum)
        return int(round(max(0, min(1, ratio)) * 1000))

    def slider_to_value(self, slider):
        minimum = slider.min_spin.value()
        maximum = slider.max_spin.value()
        return minimum + (maximum - minimum) * slider.value() / 1000

    def slider_changed(self):
        slider = self.sender()
        slider.value_spin.blockSignals(True)
        slider.value_spin.setValue(self.slider_to_value(slider))
        slider.value_spin.blockSignals(False)
        self.calculate()

    def spin_changed(self, min_spin, max_spin, value_spin, slider):
        minimum = min_spin.value()
        maximum = max_spin.value()
        value = max(minimum, min(maximum, value_spin.value()))

        value_spin.blockSignals(True)
        value_spin.setValue(value)
        value_spin.blockSignals(False)

        slider.blockSignals(True)
        slider.setValue(self.value_to_slider(value, minimum, maximum))
        slider.blockSignals(False)

        self.calculate()

    def update_slider_limits(self, min_spin, max_spin, value_spin, slider):
        minimum = min_spin.value()
        maximum = max_spin.value()

        if maximum <= minimum:
            maximum = minimum + 1e-9
            max_spin.blockSignals(True)
            max_spin.setValue(maximum)
            max_spin.blockSignals(False)

        value_spin.setRange(minimum, maximum)
        self.spin_changed(min_spin, max_spin, value_spin, slider)

    def init_default_folder(self):
        folder = Path.cwd() / "2 AZIM SAXSUtilities"

        if folder.is_dir():
            self.load_folder(folder)
        else:
            self.current_file = None
            self.azimuth = None
            self.intensity = None
            self.last_fit = None
            self.file_list.clear()
            self.canvas.ax.clear()
            self.canvas.draw_idle()

    def set_folder_from_external_tab(self, folder):
        folder = Path(folder)
        if self.folder == folder:
            return
        self.load_folder(folder)

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Open azimuthal folder", "")
        if folder:
            self.load_folder(Path(folder))

    def load_folder(self, folder):
        self.folder = Path(folder)
        self.folder_changed.emit(self.folder)
        self.available_files = sorted(
            file.name for file in self.folder.glob("*_cave*_azimProf.dat")
        )

        self.file_list.blockSignals(True)
        self.file_list.clear()
        self.file_list.addItems(self.available_files)
        self.file_list.blockSignals(False)

        if self.available_files:
            self.file_list.setCurrentRow(0)
            self.load_file(self.folder / self.available_files[0])
        else:
            self.current_file = None
            self.azimuth = None
            self.intensity = None
            self.last_fit = None
            self.results_text.clear()
            self.canvas.ax.clear()
            self.canvas.draw_idle()

    

    def load_selected_file(self):
        item = self.file_list.currentItem()

        if item is None or self.folder is None:
            return

        self.load_file(self.folder / item.text())

    def load_file(self, file_path):
        try:
            azimuth, intensity = read_azimuthal_file(file_path)
        except Exception as error:
            QMessageBox.critical(self, "File reading error", str(error))
            return

        self.current_file = Path(file_path)
        self.azimuth = azimuth
        self.intensity = intensity
        self.last_fit = None

        self.offset_spin.blockSignals(True)
        self.offset_spin.setValue(0)
        self.offset_spin.blockSignals(False)
        self.offset_slider.blockSignals(True)
        self.offset_slider.setValue(self.value_to_slider(0, self.offset_slider.min_spin.value(), self.offset_slider.max_spin.value()))
        self.offset_slider.blockSignals(False)

        self.calculate()

    def calculate(self):
        if self.azimuth is None or self.intensity is None or self.current_file is None:
            return

        azimuth = self.azimuth
        intensity = self.intensity
        peak = self.peak_spin.value()
        window = self.window_spin.value()
        offset = self.offset_spin.value()

        mask = np.abs(wrap_to_180(azimuth - peak)) <= window / 2
        az_fit = azimuth[mask]

        if az_fit.size < 10:
            self.results_text.setPlainText("Not enough points in the fitting window.")
            return

        baseline_level = np.nanmin(intensity) + offset
        baseline = np.full_like(azimuth, baseline_level)
        corrected = intensity - baseline
        corrected[corrected < 0] = 0
        fit_intensity = corrected[mask]

        if np.nanmax(fit_intensity) <= 0:
            self.results_text.setPlainText(
                "Corrected signal is null or negative.\nAdjust the baseline."
            )
            return

        use_fit = self.use_fit_checkbox.isChecked()

        if use_fit:
            try:
                amplitude, sigma = fit_gaussian_fixed_center(az_fit, fit_intensity, peak, window)
            except Exception as error:
                self.results_text.setPlainText(str(error))
                return

            fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
            self.update_manual_fields_from_fit(amplitude, fwhm)
        else:
            amplitude = self.height_spin.value()
            fwhm = self.manual_fwhm_spin.value()
            sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

        pi = (180 - fwhm) / 180

        phi_deg = np.abs(wrap_to_180(azimuth - peak))
        phi_deg[phi_deg > 90] = 180 - phi_deg[phi_deg > 90]
        phi = np.deg2rad(phi_deg)

        den_raw = np.sum(intensity * np.sin(phi))
        num_raw = np.sum(intensity * np.sin(phi) * np.cos(phi) ** 2)
        cos2mean_raw = np.nan if den_raw == 0 else num_raw / den_raw
        hermans_raw = np.nan if den_raw == 0 else (3 * cos2mean_raw - 1) / 2

        den_corr = np.sum(corrected * np.sin(phi))
        num_corr = np.sum(corrected * np.sin(phi) * np.cos(phi) ** 2)
        cos2mean_corr = np.nan if den_corr == 0 else num_corr / den_corr
        hermans_corr = np.nan if den_corr == 0 else (3 * cos2mean_corr - 1) / 2

        self.last_fit = {
            "use_fit": use_fit,
            "peak": peak,
            "window": window,
            "offset": offset,
            "baseline": baseline_level,
            "amplitude": amplitude,
            "sigma": sigma,
            "fwhm": fwhm,
            "pi": pi,
            "cos2mean_corr": cos2mean_corr,
            "hermans_corr": hermans_corr,
            "cos2mean_raw": cos2mean_raw,
            "hermans_raw": hermans_raw,
        }

        self.update_plot(azimuth, intensity, baseline, peak, amplitude, sigma, fwhm, pi, hermans_corr)
        self.update_results_text()

    def update_plot(self, azimuth, intensity, baseline, peak, amplitude, sigma, fwhm, pi, hermans_corr):
        ax = self.canvas.ax
        ax.clear()

        x_model = np.linspace(0, 360, 2000)
        y_model_corr = amplitude * np.exp(-((x_model - peak) ** 2) / (2 * sigma ** 2))
        y_model_raw = self.last_fit["baseline"] + y_model_corr

        half_max_raw = self.last_fit["baseline"] + amplitude / 2
        x_left = peak - fwhm / 2
        x_right = peak + fwhm / 2

        ax.plot(azimuth, intensity, "k-", linewidth=1.2, label="Raw data")
        ax.plot(azimuth, baseline, "r--", linewidth=1.3, label="Baseline")
        ax.plot(x_model, y_model_raw, "r-", linewidth=1.6, label="Gaussian fit")
        ax.plot([x_left, x_right], [half_max_raw, half_max_raw], "b-", linewidth=2, label="FWHM")
        ax.axvline(peak, color="blue", linestyle="--", linewidth=1, label="Peak centre")

        ax.set_title(self.current_file.name)
        ax.set_xlabel("ψ / °")
        ax.set_ylabel("Intensity / a.u.")
        ax.set_xlim(0, 360)
        ax.grid(True)
        ax.legend(loc="best")
        self.canvas.draw_idle()

    def update_results_text(self):
        if not self.last_fit or self.current_file is None:
            return

        fit = self.last_fit
        self.results_text.setPlainText(
            f"File = {self.current_file.name}\n"
            f"Mode = {'Gaussian fit' if fit['use_fit'] else 'Manual height/FWHM'}\n"
            f"Baseline = {fit['baseline']:.5f}\n"
            f"Baseline offset = {fit['offset']:.5f}\n"
            f"Peak centre = {fit['peak']:.3f} °\n"
            f"Height = {fit['amplitude']:.5f}\n"
            f"FWHM = {fit['fwhm']:.3f} °\n"
            f"Π = {fit['pi']:.4f}\n"
            f"Π = {fit['pi'] * 100:.2f} %\n"
            f"<cos²ψ> = {fit['cos2mean_corr']:.5f}\n"
            f"Hermans factor f = {fit['hermans_corr']:.5f}"
        )

    def recenter_on_fit(self):
        if not self.use_fit_checkbox.isChecked():
            QMessageBox.warning(self, "Fit disabled", "Enable Fit mode before using this button.")
            return

        if self.last_fit is None:
            QMessageBox.warning(self, "Fit unavailable", "No fit is currently available.")
            return

        peak = self.last_fit["peak"]
        self.peak_spin.setValue(peak)
        self.calculate()

    def save_gaussian_fit(self):
        # Removed Fit mode check: always allow saving
        if self.current_file is None or self.last_fit is None:
            QMessageBox.warning(self, "Save unavailable", "No fit is currently available.")
            return

        fit = self.last_fit
        x_save = np.linspace(0, 360, 2000)
        y_fit_corr = fit["amplitude"] * np.exp(-((x_save - fit["peak"]) ** 2) / (2 * fit["sigma"] ** 2))
        y_fit_raw = fit["baseline"] + y_fit_corr

        output = np.column_stack([x_save, y_fit_corr, y_fit_raw])
        output_file = self.current_file.parent / f"{self.current_file.stem}_gaussian_fit.dat"

        with open(output_file, "w", encoding="utf-8") as file:
            file.write("# Gaussian/manual profile saved from Hermans\n")
            file.write(f"# Source file: {self.current_file}\n")
            file.write(f"# Mode = {'Gaussian fit' if fit['use_fit'] else 'Manual height/FWHM'}\n")
            file.write(f"# Peak centre = {fit['peak']:.6f} deg\n")
            file.write(f"# Amplitude = {fit['amplitude']:.10e}\n")
            file.write(f"# Sigma = {fit['sigma']:.10e} deg\n")
            file.write(f"# FWHM = {fit['fwhm']:.10e} deg\n")
            file.write(f"# Baseline = {fit['baseline']:.10e}\n")
            file.write("# Columns: azimuth_deg fit_corrected fit_raw\n")
            np.savetxt(file, output, fmt="%.6f %.10e %.10e")

        QMessageBox.information(self, "Fit saved", f"Fit saved:\n{output_file}")
