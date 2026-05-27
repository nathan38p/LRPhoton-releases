import numpy as np

from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
    QPushButton,
    QMessageBox,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from .ui_style import FlexibleDoubleSpinBox as QDoubleSpinBox, style_q_geometry_buttons
from .view_tab import ViewTab


class UnfoldTab(ViewTab):
    """View-like tab prepared for pattern unfolding workflows."""

    def _build_ui(self):
        self.q_axis_unit = "nm"
        super()._build_ui()
        self._configure_unfold_toolbar()
        self._configure_unfold_right_panel()

    def q_display_factor(self):
        return 0.1 if self.q_axis_unit == "A" else 1.0

    def q_axis_label(self):
        return "q (Å⁻¹)" if self.q_axis_unit == "A" else "q (nm⁻¹)"

    def _configure_unfold_toolbar(self):
        self.toolbar.coordinates = False

        if hasattr(self, "save_colorbar_checkbox"):
            self.save_colorbar_checkbox.setText("Save colorbar")
            self.save_colorbar_checkbox.setEnabled(self.current_file is not None)
            self.save_colorbar_checkbox.setVisible(True)
            self.save_colorbar_checkbox.setStyleSheet("")
        self.unfold_save_button = getattr(self, "save_image_button", None)
        if self.unfold_save_button is not None:
            self.unfold_save_button.setToolTip("Save unfolded image")

    def _configure_unfold_right_panel(self):
        self.info_box.setTitle("Parameters")
        self.info_box.setMinimumHeight(0)
        self.info_box.setFixedHeight(72)
        self.info_text.hide()
        
        # Reorganize geometry buttons to match radial_tab style
        self._reorganize_geometry_buttons()
        
        self.display_box = QGroupBox("Scattering pattern")
        self.display_box.setStyleSheet(self.panel_box_style)
        self.display_box_layout = QVBoxLayout(self.display_box)
        self.display_box_layout.setContentsMargins(10, 22, 10, 10)
        self.display_box_layout.setSpacing(12)
        self._build_pattern_panel()

        self._build_geometry_fields()

        self.right_layout.removeWidget(self.info_box)
        self.right_layout.insertWidget(0, self.info_box, stretch=0)
        self.right_layout.insertWidget(1, self.display_box, stretch=1)

        self.set_q_geometry_mode("ID13")

    def _reorganize_geometry_buttons(self):
        """Reorganize geometry buttons to match radial_tab layout."""
        # Clear the existing info_box_layout
        self._clear_layout(self.info_box_layout)
        
        # Create a fresh layout for geometry buttons like radial_tab
        preset_layout = QHBoxLayout()
        preset_layout.setSpacing(4)
        preset_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add the buttons in order
        preset_layout.addWidget(self.q_xenocs_button)
        preset_layout.addWidget(self.q_id02_button)
        preset_layout.addWidget(self.q_id13_button)
        preset_layout.addWidget(self.q_custom_button)
        preset_layout.addWidget(self.q_manual_button)
        
        # Apply the same styling as radial_tab
        style_q_geometry_buttons(
            {
                "XENOCS": self.q_xenocs_button,
                "ID02": self.q_id02_button,
                "ID13": self.q_id13_button,
                "Custom": self.q_custom_button,
            },
            "ID13",
            self.q_manual_button,
        )
        
        # Add layout to info_box
        self.info_box_layout.setContentsMargins(8, 20, 8, 8)
        self.info_box_layout.setSpacing(6)
        self.info_box_layout.addLayout(preset_layout)

    def _clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            child_layout = item.layout()
            widget = item.widget()
            if child_layout is not None:
                self._clear_layout(child_layout)
            if widget is not None:
                widget.setParent(None)

    def _build_pattern_panel(self):
        self.pattern_fig = Figure(figsize=(3, 3))
        self.pattern_fig.patch.set_facecolor("#f4f4f4")
        self.pattern_ax = self.pattern_fig.add_subplot(111)
        self.pattern_ax.set_axis_off()
        self.pattern_canvas = FigureCanvas(self.pattern_fig)
        self.pattern_canvas.setMinimumHeight(220)
        self.display_box_layout.addWidget(self.pattern_canvas)
        self.update_pattern_preview()

    def _build_geometry_fields(self):
        self.geometry_fields_box = QWidget()
        form = QFormLayout(self.geometry_fields_box)
        form.setContentsMargins(0, 4, 0, 0)
        form.setSpacing(5)

        self.geometry_fields = {}
        labels = [
            ("xc", "Center X"),
            ("yc", "Center Y"),
            ("distance_m", "Distance (m)"),
            ("pixel_x_mm", "Pixel X (mm)"),
            ("pixel_y_mm", "Pixel Y (mm)"),
            ("wavelength_a", "Wavelength (Å)"),
        ]

        for key, label in labels:
            spin = QDoubleSpinBox()
            spin.setDecimals(16)
            spin.setRange(0, 1e12)
            spin.setFixedHeight(24)
            spin.valueChanged.connect(self._geometry_field_changed)
            self.geometry_fields[key] = spin
            form.addRow(label, spin)

        self.geometry_fields_box.hide()
        self.sync_geometry_fields()

    def _geometry_field_changed(self):
        if getattr(self, "_syncing_geometry_fields", False):
            return

        self.custom_q_geometry = {
            key: spin.value()
            for key, spin in self.geometry_fields.items()
        }
        self.save_custom_q_geometry()
        if self.q_geometry_mode != "Custom":
            self.q_geometry_mode = "Custom"
            self.update_q_geometry_button_styles()
        self.refresh_file_information()
        self.update_image()

    def set_q_geometry_mode(self, mode):
        super().set_q_geometry_mode(mode)
        self.sync_geometry_fields()
        self.update_pattern_preview()

    def open_q_geometry_dialog(self):
        super().open_q_geometry_dialog()
        self.sync_geometry_fields()
        self.update_pattern_preview()

    def use_custom_q_geometry_from_source(self):
        super().use_custom_q_geometry_from_source()
        self.sync_geometry_fields()
        self.update_pattern_preview()

    def update_file_information(self, file_type, dataset_name, n_frames, image_shape):
        super().update_file_information(file_type, dataset_name, n_frames, image_shape)
        self.sync_geometry_fields()
        self.update_pattern_preview()

    def update_image(self):
        image = self.get_current_image()
        if image is None:
            return

        unfolded_raw, unfolded_display, q_min, q_max = self.make_unfolded_images(image)
        if unfolded_raw is None:
            super().update_image()
            self.update_pattern_preview()
            return

        self.raw_current_img = unfolded_raw
        self.display_img = unfolded_display
        self.unfold_q_min = q_min
        self.unfold_q_max = q_max
        q_factor = self.q_display_factor()
        q_display_min = q_min * q_factor
        q_display_max = q_max * q_factor
        aspect = "auto"
        extent = [q_display_min, q_display_max, 0.0, 360.0]

        if self.image_artist is None:
            self.image_artist = self.ax.imshow(
                self.display_img,
                cmap="jet",
                origin="lower",
                extent=extent,
                vmin=self.vmin_spin.value(),
                vmax=self.vmax_spin.value(),
                aspect=aspect,
            )
            self.ax.set_aspect(aspect)
            self.ax.set_xlabel(self.q_axis_label())
            self.ax.set_ylabel("ψ (°)")
            self.ax.set_axis_on()
            self.colorbar = self.fig.colorbar(
                self.image_artist,
                ax=self.ax,
                fraction=0.046,
                pad=0.04,
            )
        else:
            self.image_artist.set_data(self.display_img)
            self.image_artist.set_extent(extent)
            self.image_artist.set_clim(self.vmin_spin.value(), self.vmax_spin.value())
            self.ax.set_aspect(aspect)
            self.ax.set_xlabel(self.q_axis_label())
            self.ax.set_ylabel("ψ (°)")
            self.ax.set_axis_on()

        self._apply_unfold_figure_margins()

        if self.keep_zoom_checkbox.isChecked() and self._saved_xlim is not None and self._saved_ylim is not None:
            self.ax.set_xlim(self._saved_xlim)
            self.ax.set_ylim(self._saved_ylim)
        else:
            self.ax.set_xlim(q_display_min, q_display_max)
            self.ax.set_ylim(0.0, 360.0)

        total = self.n_frames if self.is_lazy_h5 else self.images.shape[0]
        self.frame_label.setText(f"{self.current_index + 1} / {total}")

        self.ax.set_autoscale_on(False)
        self.canvas.draw_idle()
        self.update_pattern_preview()

    def current_raw_image_for_save(self):
        if self.raw_current_img is None:
            raise ValueError("No unfolded image data is available.")
        return np.asarray(self.raw_current_img, dtype=float)

    def automatic_unfold_save_path(self):
        frame_suffix = ""
        total_frames = self.n_frames if self.is_lazy_h5 else (self.images.shape[0] if self.images is not None else 1)
        if total_frames > 1:
            frame_suffix = f"_frame{self.current_index + 1:04d}"

        suffix = ".edf" if self.current_file.suffix.lower() == ".edf" else ".h5"
        return self.current_file.parent / f"{self.current_file.stem}{frame_suffix}_unfold{suffix}"

    def save_png_image_only(self):
        if self.raw_current_img is None or self.current_file is None:
            QMessageBox.information(self, "No image", "No unfolded image is currently loaded.")
            return

        output_path = self.automatic_unfold_save_path()
        try:
            if output_path.suffix.lower() == ".edf":
                self.save_current_frame_as_edf(output_path)
            else:
                self.save_current_frame_as_h5(output_path)
        except Exception as error:
            QMessageBox.critical(self, "Save error", f"Unable to save unfolded image:\n{error}")

    def _apply_unfold_figure_margins(self):
        self.fig.subplots_adjust(left=0.13, right=0.90, top=0.98, bottom=0.14)

    def draw_center_cross(self):
        super().draw_center_cross()
        self.draw_unfold_axes()

    def make_unfolded_images(self, image):
        geometry = self.get_q_geometry_from_header()
        if geometry is None:
            return None, None, None, None

        xc, yc, distance_m, pixel_x_mm, pixel_y_mm, wavelength_nm = geometry
        if distance_m <= 0 or pixel_x_mm <= 0 or pixel_y_mm <= 0 or wavelength_nm <= 0:
            return None, None, None, None

        raw = np.asarray(image, dtype=float)
        raw = raw.copy()
        raw[raw > 4e9] = np.nan
        display = self.prepare_display_image(raw)
        ny, nx = raw.shape

        yy, xx = np.indices(raw.shape, dtype=float)
        finite_raw = np.isfinite(raw)
        dx_m = (xx - xc) * pixel_x_mm * 1e-3
        dy_m = (yy - yc) * pixel_y_mm * 1e-3
        r_m = np.sqrt(dx_m ** 2 + dy_m ** 2)
        two_theta = np.arctan2(r_m, distance_m)
        q_map = (4.0 * np.pi / wavelength_nm) * np.sin(two_theta / 2.0)
        valid_q = q_map[finite_raw & np.isfinite(q_map)]
        if valid_q.size == 0:
            return None, None, None, None

        q_min = float(np.nanmin(valid_q))
        q_max = float(np.nanmax(valid_q))
        if not np.isfinite(q_min) or not np.isfinite(q_max) or q_max <= q_min:
            return None, None, None, None

        angle_bins = 720
        q_bins = int(np.clip(max(nx, ny), 256, 1400))
        angles = np.linspace(0.0, 360.0, angle_bins, endpoint=False)
        q_values = np.linspace(q_min, q_max, q_bins)

        theta = -np.deg2rad(angles)[None, :]
        q_grid = q_values[:, None]
        argument = np.clip(q_grid * wavelength_nm / (4.0 * np.pi), -1.0, 1.0)
        two_theta_grid = 2.0 * np.arcsin(argument)
        radius_m = distance_m * np.tan(two_theta_grid)

        x = xc + (radius_m * np.cos(theta)) / (pixel_x_mm * 1e-3)
        y = yc + (radius_m * np.sin(theta)) / (pixel_y_mm * 1e-3)

        x_index = np.rint(x).astype(int)
        y_index = np.rint(y).astype(int)
        inside = (x_index >= 0) & (x_index < nx) & (y_index >= 0) & (y_index < ny)

        unfolded_raw = np.full((q_bins, angle_bins), np.nan, dtype=float)
        unfolded_display = np.full((q_bins, angle_bins), np.nan, dtype=float)
        unfolded_raw[inside] = raw[y_index[inside], x_index[inside]]
        unfolded_display[inside] = display[y_index[inside], x_index[inside]]
        return unfolded_raw.T, unfolded_display.T, q_min, q_max

    def sync_geometry_fields(self):
        if not hasattr(self, "geometry_fields"):
            return

        values = self.q_geometry_values_for_mode() or self.preset_q_geometry(self.q_geometry_mode) or {}
        self._syncing_geometry_fields = True
        try:
            for key, spin in self.geometry_fields.items():
                spin.blockSignals(True)
                spin.setValue(float(values.get(key, 0.0) or 0.0))
                spin.blockSignals(False)
        finally:
            self._syncing_geometry_fields = False

    def q_geometry_values_for_mode(self):
        if self.q_geometry_mode == "ID13":
            return self.preset_q_geometry("ID13")
        return super().q_geometry_values_for_mode()

    def draw_unfold_axes(self):
        values = self.q_geometry_values_for_mode()
        if not values:
            return

        xc = values.get("xc")
        yc = values.get("yc")
        if xc is None or yc is None:
            return

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        radius = 0.48 * min(abs(xlim[1] - xlim[0]), abs(ylim[1] - ylim[0]))
        if not np.isfinite(radius) or radius <= 0:
            return

        for angle in range(0, 360, 45):
            theta = np.deg2rad(angle)
            x2 = xc + radius * np.cos(theta)
            y2 = yc + radius * np.sin(theta)
            line = self.ax.plot([xc, x2], [yc, y2], color="white", linewidth=0.8, alpha=0.75)[0]
            label = self.ax.text(
                x2,
                y2,
                f"{angle}°",
                color="white",
                fontsize=8,
                ha="center",
                va="center",
                bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 1.5},
            )
            self.center_artists.extend([line, label])

    def update_pattern_preview(self):
        if not hasattr(self, "pattern_ax"):
            return

        self.pattern_ax.clear()
        self.pattern_ax.set_axis_off()
        image = self.get_current_image()

        if image is None:
            self.pattern_canvas.draw_idle()
            return

        preview = np.asarray(image, dtype=float)
        if self.log_checkbox.isChecked():
            preview = np.log10(np.clip(preview, 0, None) + 1)

        self.pattern_ax.imshow(
            preview,
            cmap="jet",
            origin="upper",
            vmin=self.vmin_spin.value(),
            vmax=self.vmax_spin.value(),
        )

        values = self.q_geometry_values_for_mode()
        if values:
            xc = values.get("xc")
            yc = values.get("yc")
            if xc is not None and yc is not None:
                ny, nx = preview.shape
                radius = 0.45 * min(nx, ny)
                self.pattern_ax.plot(xc, yc, "o", ms=4, mfc="white", mec="#d71920", mew=1)
                for angle in range(0, 360, 45):
                    theta = np.deg2rad(angle)
                    x2 = xc + radius * np.cos(theta)
                    y2 = yc + radius * np.sin(theta)
                    self.pattern_ax.plot([xc, x2], [yc, y2], color="#d71920", lw=0.8, alpha=0.9)
                    self.pattern_ax.text(
                        x2,
                        y2,
                        f"{angle}°",
                        color="white",
                        fontsize=7,
                        ha="center",
                        va="center",
                        bbox={"facecolor": "black", "alpha": 0.5, "edgecolor": "none", "pad": 1},
                    )

        self.pattern_fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.pattern_canvas.draw_idle()

    def on_mouse_move(self, event):
        if self.pan_image_from_motion(event):
            return
        if self.raw_current_img is None or event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            self.cursor_label.setText("q = - | ψ = - | I = -")
            return

        q_display_value = float(event.xdata)
        q_value = q_display_value / self.q_display_factor()
        angle = float(event.ydata) % 360.0
        angle_bins, q_bins = self.raw_current_img.shape
        q_min = getattr(self, "unfold_q_min", 0.0)
        y_index = int(np.floor(angle / 360.0 * angle_bins))
        q_max = getattr(self, "unfold_q_max", None)
        if q_max is None or q_max <= q_min:
            self.cursor_label.setText("q = - | ψ = - | I = -")
            return

        x_index = int(round((q_value - q_min) / (q_max - q_min) * (q_bins - 1)))
        if not (0 <= x_index < q_bins and 0 <= y_index < angle_bins):
            self.cursor_label.setText("q = - | ψ = - | I = -")
            return

        value = self.raw_current_img[y_index, x_index]
        if np.isnan(value):
            value_text = "NaN"
        elif np.isposinf(value):
            value_text = "+Inf"
        elif np.isneginf(value):
            value_text = "-Inf"
        else:
            value_text = f"{value:.8g}"

        unit_label = "Å⁻¹" if self.q_axis_unit == "A" else "nm⁻¹"
        self.cursor_label.setText(
            f"q = {q_display_value:.6g} {unit_label} | ψ = {angle:.3f}° | I = {value_text}"
        )

    def on_mouse_press(self, event):
        try:
            clicked_label = self.ax.xaxis.label.contains(event)[0]
        except Exception:
            clicked_label = False
        if event.button == 1 and clicked_label:
            self.q_axis_unit = "A" if self.q_axis_unit == "nm" else "nm"
            self._saved_xlim = None
            self._saved_ylim = None
            self.update_image()
            return

        super().on_mouse_press(event)

    def on_mouse_leave(self, event):
        self.cursor_label.setText("q = - | ψ = - | I = -")
