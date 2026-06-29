from datetime import datetime
import os
from pathlib import Path
import re
import sys

import numpy as np

from PySide6.QtCore import QEvent, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QAbstractItemView,
    QApplication,
    QPlainTextEdit,
    QPushButton,
    QComboBox,
    QSpinBox,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from tabs.line_geometry import LRP_SALS_DEFAULT_NAME, LineGeometrySelector, default_center_text
from tabs.ui_style import (
    BLOCK_SPACING,
    FILE_BROWSER_WIDTH,
    GROUP_BOX_MARGINS,
    GROUP_BOX_STYLE,
    PAGE_MARGINS,
    PANEL_MARGINS,
    make_matplotlib_toolbar_block,
    set_matplotlib_toolbar_enabled,
    set_widget_enabled_with_opacity,
)


class SALSPreviewToolbar(NavigationToolbar):
    def __init__(self, canvas, parent):
        super().__init__(canvas, parent)
        self.sals_widget = parent

    def home(self, *args):
        self.sals_widget.reset_preview_zoom()
        self.push_current()
        self.set_history_buttons()

    def release_pan(self, event):
        super().release_pan(event)
        self.sals_widget.sync_preview_limits_from_axes()

    def release_zoom(self, event):
        super().release_zoom(event)
        self.sals_widget.sync_preview_limits_from_axes()

    def back(self, *args):
        super().back(*args)
        self.sals_widget.sync_preview_limits_from_axes()

    def forward(self, *args):
        super().forward(*args)
        self.sals_widget.sync_preview_limits_from_axes()


class VimbaSALSWidget(QWidget):
    back_requested = Signal()
    FIELD_HEIGHT = 22
    FIELD_SPACING = 6
    INNER_GROUP_MARGINS = (8, 10, 8, 8)
    CAMERA_SPIN_WIDTH = 70
    CAMERA_LABEL_WIDTH = 112
    CAMERA_SHORT_LABEL_WIDTH = 58
    CAMERA_OFFSET_LABEL_WIDTH = 66
    SALS_LABEL_WIDTH = 100
    APP_ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
    CAMERA_ID = "DEV_000F315BB2BF"
    CAMERA_MODEL = "Mako G-419B"
    DEFAULT_ROI_WIDTH = 796
    DEFAULT_ROI_HEIGHT = 796
    DEFAULT_PIXEL_FORMAT = "Mono12Packed"
    DEFAULT_OFFSET_X = 626
    DEFAULT_OFFSET_Y = 626
    DEFAULT_EXPOSURE_US = "60"
    DEFAULT_GAIN = "0"
    DEFAULT_PREVIEW_FPS = 26
    DEFAULT_ACQUISITION_FRAME_RATE = 26.36992
    DEFAULT_REVERSE_Y = False
    DEFAULT_CAMERA_IP = "169.254.242.220"
    DEFAULT_DISTANCE_M = "0,00477"
    DEFAULT_PIXEL_SIZE_M = "5,5e-6"
    DEFAULT_WAVELENGTH_M = "632,8e-9"
    VIMBA_SETTINGS_PATH = APP_ROOT / "assets" / "camera" / "settingsvimba.xml"
    DEFAULT_CAMERA_FEATURES = (
        ("Acquisition", "AcquisitionMode", "Continuous"),
        ("Acquisition", "AcquisitionFrameCount", 1),
        ("Acquisition", "AcquisitionFrameRate", DEFAULT_ACQUISITION_FRAME_RATE),
        ("Acquisition", "RecorderPreEventCount", 0),
        ("Acquisition / Trigger", "TriggerActivation", "RisingEdge"),
        ("Acquisition / Trigger", "TriggerDelayAbs", 0.0),
        ("Acquisition / Trigger", "TriggerMode", "On"),
        ("Acquisition / Trigger", "TriggerOverlap", "Off"),
        ("Acquisition / Trigger", "TriggerSelector", "FrameStart"),
        ("Acquisition / Trigger", "TriggerSource", "Freerun"),
        ("GigE", "BandwidthControlMode", "StreamBytesPerSecond"),
        ("GigE", "ChunkModeActive", False),
        ("GigE / GVCP", "GVCPCmdRetries", 5),
        ("GigE / GVCP", "GVCPCmdTimeout", 250),
        ("GigE / GVCP", "GevHeartbeatInterval", 1450),
        ("GigE / GVCP", "GevHeartbeatTimeout", 3000),
        ("GigE", "GevSCPSPacketSize", 8999),
        ("GigE", "StreamBytesPerSecond", 115000000),
        ("GigE", "StreamFrameRateConstrain", True),
        ("GigE / StreamHold", "StreamHoldEnable", "Off"),
        ("Controls / BlackLevel", "BlackLevel", 4.0),
        ("Controls / BlackLevel", "BlackLevelSelector", "All"),
        ("Controls / DSPSubregion", "DSPSubregionBottom", 796),
        ("Controls / DSPSubregion", "DSPSubregionLeft", 0),
        ("Controls / DSPSubregion", "DSPSubregionRight", 796),
        ("Controls / DSPSubregion", "DSPSubregionTop", 0),
        ("Controls", "DefectMaskEnable", True),
        ("Controls / Exposure", "ExposureAuto", "Off"),
        ("Controls / Exposure", "ExposureAutoAdjustTol", 5),
        ("Controls / Exposure", "ExposureAutoAlg", "Mean"),
        ("Controls / Exposure", "ExposureAutoMax", 500000),
        ("Controls / Exposure", "ExposureAutoMin", 60),
        ("Controls / Exposure", "ExposureAutoOutliers", 0),
        ("Controls / Exposure", "ExposureAutoRate", 100),
        ("Controls / Exposure", "ExposureAutoTarget", 50),
        ("Controls / Exposure", "ExposureMode", "Timed"),
        ("Controls / Exposure", "ExposureTimePWL1", 15000.0),
        ("Controls / Exposure", "ExposureTimePWL2", 15000.0),
        ("Controls / Exposure", "ThresholdPWL1", 63),
        ("Controls / Exposure", "ThresholdPWL2", 63),
        ("Controls / Gain", "GainAuto", "Off"),
        ("Controls / Gain", "GainAutoAdjustTol", 5),
        ("Controls / Gain", "GainAutoMax", 26.0),
        ("Controls / Gain", "GainAutoMin", 0.0),
        ("Controls / Gain", "GainAutoOutliers", 0),
        ("Controls / Gain", "GainAutoRate", 100),
        ("Controls / Gain", "GainAutoTarget", 50),
        ("Controls / Gain", "GainSelector", "All"),
        ("Controls", "Gamma", 1.0),
        ("Controls / LUT", "LUTEnable", False),
        ("Controls / LUT", "LUTIndex", 0),
        ("Controls / LUT", "LUTMode", "Luminance"),
        ("Controls / LUT", "LUTSelector", "LUT1"),
        ("Controls / LUT", "LUTValue", 4095),
        ("ImageMode", "DecimationHorizontal", 1),
        ("ImageMode", "DecimationVertical", 1),
        ("ImageMode", "ReverseX", False),
        ("ImageMode", "ReverseY", DEFAULT_REVERSE_Y),
        ("EventControl / EventData", "EventNotification", "Off"),
        ("EventControl / EventData", "EventSelector", "AcquisitionStart"),
        ("EventControl / EventData", "EventsEnable1", 0),
        ("Stream / Settings", "GVSPBurstSize", 1),
        ("Stream / Settings", "GVSPDriver", "Socket"),
        ("Stream / Settings", "GVSPHostReceiveBufferSize", 67108864),
        ("Stream / Settings", "GVSPHostReceiveBuffers", 512),
        ("Stream / Settings", "GVSPMaxLookBack", 30),
        ("Stream / Settings", "GVSPMaxRequests", 1),
        ("Stream / Settings", "GVSPMaxWaitSize", 100),
        ("Stream / Settings", "GVSPMissingSize", 256),
        ("Stream / Settings", "GVSPPacketSize", 1500),
        ("Stream / Settings", "GVSPProtocol", "UDP"),
        ("Stream / Settings", "GVSPTiltingSize", 100),
        ("Stream / Settings", "GVSPTimeout", 70),
        ("Stream / Multicast", "MulticastEnable", False),
        ("BufferHandlingControl", "StreamAnnouncedBufferCount", 7),
        ("BufferHandlingControl", "StreamBufferHandlingMode", "Default"),
        ("IO / Strobe", "StrobeDelay", 0),
        ("IO / Strobe", "StrobeDuration", 0),
        ("IO / Strobe", "StrobeDurationMode", "Source"),
        ("IO / Strobe", "StrobeSource", "FrameTrigger"),
        ("IO / SyncIn", "SyncInGlitchFilter", 0),
        ("IO / SyncIn", "SyncInSelector", "SyncIn1"),
        ("IO / SyncOut", "SyncOutPolarity", "Normal"),
        ("IO / SyncOut", "SyncOutSelector", "SyncOut1"),
        ("IO / SyncOut", "SyncOutSource", "Exposing"),
    )
    CAMERA_SETTING_ALIASES = {
        "AcquisitionFrameRate": ("AcquisitionFrameRate", "AcquisitionFrameRateAbs"),
        "ExposureTime": ("ExposureTime", "ExposureTimeAbs"),
    }
    def __init__(self):
        super().__init__()
        self.vmb = None
        self.camera = None
        self.current_frame = None
        self.frame_index = 0
        self.output_folder = Path.home() / "LRPhoton_SALS"
        self.current_geometry_name = LRP_SALS_DEFAULT_NAME
        self.is_closing = False
        self.is_grabbing_frame = False
        self.preview_xlim = None
        self.preview_ylim = None
        self.preview_hover_pixel = None
        self.is_recording_frames = False
        self.recording_started_at = None
        self.recording_frame_count = 0
        self.recording_output_folder = None

        self.live_timer = QTimer(self)
        self.live_timer.timeout.connect(self.grab_live_frame)
        self.record_timer = QTimer(self)
        self.record_timer.timeout.connect(self.record_current_frame)
        self.record_title_timer = QTimer(self)
        self.record_title_timer.timeout.connect(self.update_recording_title)

        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.shutdown_camera)

        self.build_ui()
        self.update_connection_state(False)

    def build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(*PAGE_MARGINS)
        main_layout.setSpacing(BLOCK_SPACING)

        body_layout = QHBoxLayout()
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(BLOCK_SPACING)
        main_layout.addLayout(body_layout, 1)

        controls_box = QGroupBox("SALS acquisition")
        controls_box.setStyleSheet(GROUP_BOX_STYLE)
        controls_box.setFixedWidth(FILE_BROWSER_WIDTH)
        controls_layout = QVBoxLayout(controls_box)
        controls_layout.setContentsMargins(*GROUP_BOX_MARGINS)
        controls_layout.setSpacing(self.FIELD_SPACING)
        body_layout.addWidget(controls_box, 0)

        self.connect_button = QPushButton("Connect camera")
        self.connect_button.setFixedHeight(self.FIELD_HEIGHT)
        self.connect_button.clicked.connect(self.connect_camera)
        controls_layout.addWidget(self.connect_button)

        live_buttons_layout = QHBoxLayout()
        live_buttons_layout.setContentsMargins(0, 0, 0, 0)
        live_buttons_layout.setSpacing(self.FIELD_SPACING)
        self.start_button = QPushButton("Start live")
        self.start_button.setFixedHeight(self.FIELD_HEIGHT)
        self.start_button.clicked.connect(self.start_live)
        live_buttons_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop live")
        self.stop_button.setFixedHeight(self.FIELD_HEIGHT)
        self.stop_button.clicked.connect(self.stop_live)
        live_buttons_layout.addWidget(self.stop_button)
        controls_layout.addLayout(live_buttons_layout)

        self.exposure_edit = QLineEdit(self.DEFAULT_EXPOSURE_US)
        self.exposure_edit.setToolTip("Vimba ExposureTime in microseconds when available.")
        self.gain_edit = QLineEdit(self.DEFAULT_GAIN)
        self.gain_edit.setPlaceholderText("auto/unchanged")
        self.width_spinbox = QSpinBox()
        self.width_spinbox.setRange(1, 10000)
        self.width_spinbox.setValue(self.DEFAULT_ROI_WIDTH)
        self.height_spinbox = QSpinBox()
        self.height_spinbox.setRange(1, 10000)
        self.height_spinbox.setValue(self.DEFAULT_ROI_HEIGHT)
        self.offset_x_spinbox = QSpinBox()
        self.offset_x_spinbox.setRange(0, 10000)
        self.offset_x_spinbox.setValue(self.DEFAULT_OFFSET_X)
        self.offset_y_spinbox = QSpinBox()
        self.offset_y_spinbox.setRange(0, 10000)
        self.offset_y_spinbox.setValue(self.DEFAULT_OFFSET_Y)
        self.pixel_format_combo = QComboBox()
        self.pixel_format_combo.setEditable(True)
        self.pixel_format_combo.setMaxVisibleItems(3)
        self.pixel_format_combo.addItems([
            self.DEFAULT_PIXEL_FORMAT,
            "Mono12",
            "Mono12p",
            "Mono16",
            "Mono8",
        ])
        self.pixel_format_combo.setCurrentText(self.DEFAULT_PIXEL_FORMAT)
        self.pixel_format_combo.setToolTip("Camera PixelFormat. Packed formats are converted internally for display.")
        self.pixel_format_combo.view().setStyleSheet("""
            QListView {
                background: #ffffff;
                border: 1px solid #d7d7d7;
            }
            QListView::item {
                min-height: 22px;
                padding: 1px 8px;
            }
        """)
        self.fps_spinbox = QSpinBox()
        self.fps_spinbox.setRange(1, 60)
        self.fps_spinbox.setValue(self.DEFAULT_PREVIEW_FPS)
        self.reverse_y_checkbox = QCheckBox("Reverse Y")
        self.reverse_y_checkbox.setChecked(self.DEFAULT_REVERSE_Y)
        self.camera_ip_edit = QLineEdit(self.DEFAULT_CAMERA_IP)
        self.camera_ip_edit.setToolTip(
            "Direct GigE camera IP. Useful on macOS when Wi-Fi prevents automatic Ethernet discovery."
        )
        self.style_acquisition_fields()

        camera_box = QGroupBox("Camera settings")
        camera_box.setStyleSheet(GROUP_BOX_STYLE)
        camera_layout = QVBoxLayout(camera_box)
        camera_layout.setContentsMargins(*self.INNER_GROUP_MARGINS)
        camera_layout.setSpacing(self.FIELD_SPACING)
        controls_layout.addWidget(camera_box)

        camera_layout.addLayout(self.camera_field_row("Camera IP", self.camera_ip_edit))
        camera_layout.addLayout(self.camera_field_row("Exposure (µs)", self.exposure_edit))
        camera_layout.addLayout(
            self.camera_double_field_row("Width", self.width_spinbox, "Offset X", self.offset_x_spinbox)
        )
        camera_layout.addLayout(
            self.camera_double_field_row("Height", self.height_spinbox, "Offset Y", self.offset_y_spinbox)
        )
        camera_layout.addLayout(self.camera_field_row("Pixel format", self.pixel_format_combo))
        camera_layout.addLayout(self.camera_field_row("Preview fps", self.fps_spinbox, add_stretch=True))

        camera_buttons_layout = QHBoxLayout()
        camera_buttons_layout.setContentsMargins(0, 0, 0, 0)
        camera_buttons_layout.setSpacing(self.FIELD_SPACING)
        self.camera_settings_button = QPushButton("All camera settings")
        self.camera_settings_button.setFixedHeight(self.FIELD_HEIGHT)
        self.camera_settings_button.clicked.connect(self.show_camera_settings_dialog)
        camera_buttons_layout.addWidget(self.camera_settings_button, 1)
        self.apply_camera_button = QPushButton("Apply settings")
        self.apply_camera_button.setFixedHeight(self.FIELD_HEIGHT)
        self.apply_camera_button.clicked.connect(self.apply_camera_settings)
        camera_buttons_layout.addWidget(self.apply_camera_button, 1)
        camera_layout.addLayout(camera_buttons_layout)

        file_box = QGroupBox("File settings")
        file_box.setStyleSheet(GROUP_BOX_STYLE)
        file_layout = QVBoxLayout(file_box)
        file_layout.setContentsMargins(*self.INNER_GROUP_MARGINS)
        file_layout.setSpacing(self.FIELD_SPACING)
        controls_layout.addWidget(file_box)

        output_layout = QHBoxLayout()
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.setSpacing(self.FIELD_SPACING)
        self.output_edit = QLineEdit(str(self.output_folder))
        self.output_edit.setFixedHeight(self.FIELD_HEIGHT)
        output_layout.addWidget(self.output_edit, 1)
        self.output_button = QPushButton("Browse")
        self.output_button.setFixedHeight(self.FIELD_HEIGHT)
        self.output_button.clicked.connect(self.choose_output_folder)
        output_layout.addWidget(self.output_button)
        file_layout.addWidget(self.section_label("Output folder"))
        file_layout.addLayout(output_layout)

        self.save_button = QPushButton("💾 Save current EDF")
        self.save_button.setFixedHeight(self.FIELD_HEIGHT)
        self.save_button.clicked.connect(self.save_current_edf)
        file_layout.addWidget(self.save_button)

        sals_box = QGroupBox("EDF header")
        sals_box.setStyleSheet(GROUP_BOX_STYLE)
        sals_layout = QGridLayout(sals_box)
        sals_layout.setContentsMargins(*self.INNER_GROUP_MARGINS)
        sals_layout.setHorizontalSpacing(self.FIELD_SPACING)
        sals_layout.setVerticalSpacing(self.FIELD_SPACING)
        sals_layout.setColumnStretch(1, 1)
        controls_layout.addWidget(sals_box)

        self.geometry_selector = LineGeometrySelector(self, LRP_SALS_DEFAULT_NAME, show_poni=False)
        self.geometry_selector.set_current_name(LRP_SALS_DEFAULT_NAME, explicit=True)
        self.geometry_selector.geometry_selected.connect(self.apply_line_geometry)

        self.distance_edit = QLineEdit("")
        self.pixel_x_edit = QLineEdit("")
        self.pixel_y_edit = QLineEdit("")
        self.wavelength_edit = QLineEdit("")
        self.sample_name_edit = QLineEdit("sals")
        self.center_x_edit = QLineEdit(self.default_center_text(self.DEFAULT_ROI_WIDTH))
        self.center_y_edit = QLineEdit(self.default_center_text(self.DEFAULT_ROI_HEIGHT))
        self.width_spinbox.valueChanged.connect(self.update_default_center_x)
        self.height_spinbox.valueChanged.connect(self.update_default_center_y)

        self.distance_edit.setPlaceholderText("m")
        self.pixel_x_edit.setPlaceholderText("m")
        self.pixel_y_edit.setPlaceholderText("m")
        self.wavelength_edit.setPlaceholderText("m")
        self.sample_name_edit.setPlaceholderText("Sample name")
        self.style_sals_fields()
        self.add_labeled_field(sals_layout, 0, "Ligne", self.geometry_selector)
        self.add_labeled_field(sals_layout, 1, "Sample name", self.sample_name_edit)
        self.add_labeled_field(sals_layout, 2, "Distance (m)", self.distance_edit)
        self.add_labeled_field(sals_layout, 3, "Pixel X (m)", self.pixel_x_edit)
        self.add_labeled_field(sals_layout, 4, "Pixel Y (m)", self.pixel_y_edit)
        self.add_labeled_field(sals_layout, 5, "Wavelength (m)", self.wavelength_edit)
        self.add_labeled_field(sals_layout, 6, "Center X", self.center_x_edit)
        self.add_labeled_field(sals_layout, 7, "Center Y", self.center_y_edit)
        self.apply_line_geometry(self.geometry_selector.current_name, self.geometry_selector.current_geometry())

        controls_layout.addStretch(1)

        preview_box = QWidget()
        preview_layout = QVBoxLayout(preview_box)
        preview_layout.setContentsMargins(*PANEL_MARGINS)
        preview_layout.setSpacing(BLOCK_SPACING)
        body_layout.addWidget(preview_box, 1)

        self.figure = Figure(figsize=(6, 5))
        self.figure.patch.set_alpha(0)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setAttribute(Qt.WA_TranslucentBackground, True)
        self.canvas.setStyleSheet("background: transparent;")
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.canvas.setMouseTracking(True)
        self.canvas.grabGesture(Qt.PinchGesture)
        self.canvas.installEventFilter(self)
        self.canvas.mpl_connect("motion_notify_event", self.update_preview_coordinates)
        self.canvas.mpl_connect("figure_leave_event", self.clear_preview_coordinates)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor("none")
        self.ax.set_axis_off()
        self.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.image_artist = None
        self.toolbar = SALSPreviewToolbar(self.canvas, self)
        self.record_play_button = QToolButton(self)
        self.record_play_button.setText("▶️")
        self.record_play_button.setFixedSize(32, 32)
        self.record_play_button.setStyleSheet("""
            QToolButton {
                background: transparent;
                border: none;
                padding: 0px;
                margin: 0px;
                font-size: 28px;
            }
        """)
        self.record_play_button.setToolTip("Start saving EDF frames at the selected preview fps")
        self.record_play_button.clicked.connect(self.start_recording_frames)
        self.record_stop_button = QToolButton(self)
        self.record_stop_button.setText("⏹️")
        self.record_stop_button.setFixedSize(32, 32)
        self.record_stop_button.setStyleSheet("""
            QToolButton {
                background: transparent;
                border: none;
                padding: 0px;
                margin: 0px;
                font-size: 28px;
            }
        """)
        self.record_stop_button.setToolTip("Stop saving EDF frames")
        self.record_stop_button.clicked.connect(self.stop_recording_frames)
        self.status_label = QLabel("Vimba is not connected.")
        self.status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.status_label.setMinimumWidth(280)
        self.status_label.setMinimumHeight(22)
        self.status_label.setWordWrap(False)
        self.status_label.setStyleSheet("""
            QLabel {
                background: transparent;
                color: #111111;
                padding: 0px 6px;
            }
        """)
        toolbar_box, _, self.preview_save_button = make_matplotlib_toolbar_block(
            self,
            "Live EDF preview",
            self.toolbar,
            option_widgets=[self.status_label, self.record_play_button, self.record_stop_button],
            save_callback=self.save_current_edf,
            save_tooltip="Save current EDF",
            toolbar_width=340,
            remove_customize=True,
        )
        self.preview_toolbar_box = toolbar_box
        preview_layout.addWidget(toolbar_box, 0)
        preview_layout.addWidget(self.canvas, 1)

        self.preview_coordinate_label = QLabel("x = - | y = - | q = - | angle = - | I = -")
        self.preview_coordinate_label.setAlignment(Qt.AlignCenter)
        self.preview_coordinate_label.setFixedHeight(26)
        self.preview_coordinate_label.setStyleSheet("""
            QLabel {
                background: #f3f3f3;
                border: none;
                border-radius: 8px;
                padding: 3px 8px;
                color: #111111;
                font-family: Menlo, Consolas, monospace;
            }
        """)
        preview_layout.addWidget(self.preview_coordinate_label, 0)

    def style_acquisition_fields(self):
        for widget in [
            self.camera_ip_edit,
            self.exposure_edit,
            self.width_spinbox,
            self.height_spinbox,
            self.offset_x_spinbox,
            self.offset_y_spinbox,
            self.fps_spinbox,
        ]:
            widget.setFixedHeight(self.FIELD_HEIGHT)
        for widget in [
            self.width_spinbox,
            self.height_spinbox,
            self.offset_x_spinbox,
            self.offset_y_spinbox,
            self.fps_spinbox,
        ]:
            widget.setMinimumWidth(self.CAMERA_SPIN_WIDTH)
        self.pixel_format_combo.setFixedHeight(self.FIELD_HEIGHT)

    def style_sals_fields(self):
        for widget in [
            self.distance_edit,
            self.pixel_x_edit,
            self.pixel_y_edit,
            self.wavelength_edit,
            self.center_x_edit,
            self.center_y_edit,
        ]:
            widget.setFixedHeight(self.FIELD_HEIGHT)
            widget.setMinimumWidth(0)
        self.geometry_selector.setMinimumWidth(0)
        self.geometry_selector.setFixedHeight(self.FIELD_HEIGHT)
        self.geometry_selector.combo.setMinimumWidth(0)
        self.geometry_selector.combo.setFixedHeight(self.FIELD_HEIGHT)
        self.geometry_selector.edit_button.setFixedHeight(self.FIELD_HEIGHT)
        self.geometry_selector.edit_button.setFixedWidth(54)

    def form_label(self, text, minimum_width=0):
        label = QLabel(f"{text}:")
        label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        if minimum_width:
            label.setMinimumWidth(minimum_width)
            label.setMaximumWidth(minimum_width)
        return label

    def section_label(self, text):
        label = QLabel(text)
        label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        return label

    def camera_field_row(self, label_text, field, add_stretch=False):
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(self.FIELD_SPACING)
        layout.addWidget(self.form_label(label_text, self.CAMERA_LABEL_WIDTH))
        layout.addWidget(field, 0 if add_stretch else 1)
        if add_stretch:
            layout.addStretch(1)
        return layout

    def camera_double_field_row(self, left_label_text, left_field, right_label_text, right_field):
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(self.FIELD_SPACING)
        layout.addWidget(self.form_label(left_label_text, self.CAMERA_SHORT_LABEL_WIDTH))
        layout.addWidget(left_field)
        layout.addWidget(self.form_label(right_label_text, self.CAMERA_OFFSET_LABEL_WIDTH))
        layout.addWidget(right_field)
        layout.addStretch(1)
        return layout

    def add_labeled_field(self, layout, row, label_text, widget):
        label = self.form_label(label_text, self.SALS_LABEL_WIDTH)
        layout.addWidget(label, row, 0)
        layout.addWidget(widget, row, 1)

    def update_connection_state(self, connected):
        has_frame = self.current_frame is not None
        self.connect_button.setEnabled(not connected)
        self.start_button.setEnabled(connected and not self.live_timer.isActive())
        self.stop_button.setEnabled(self.live_timer.isActive())
        self.apply_camera_button.setEnabled(connected)
        self.save_button.setEnabled(has_frame)
        set_matplotlib_toolbar_enabled(getattr(self, "toolbar", None), has_frame)
        if hasattr(self, "preview_save_button"):
            set_widget_enabled_with_opacity(self.preview_save_button, has_frame)
        if hasattr(self, "record_play_button"):
            set_widget_enabled_with_opacity(self.record_play_button, has_frame and not self.is_recording_frames)
        if hasattr(self, "record_stop_button"):
            set_widget_enabled_with_opacity(self.record_stop_button, has_frame and self.is_recording_frames)

    def request_back(self):
        if self.live_timer.isActive():
            self.stop_live()
        self.back_requested.emit()

    def connect_camera(self):
        try:
            from vmbpy import VmbSystem
        except ImportError:
            self.status_label.setText("VmbPy is missing. Reinstall or update LRPhoton with the bundled camera support.")
            return

        try:
            self.vmb = VmbSystem.get_instance()
            self.vmb.__enter__()
            self.camera = self.connect_direct_vimba_camera()
            cameras = []
            if self.camera is None:
                cameras = self.discover_vimba_cameras()
            if self.camera is None and not cameras:
                raise RuntimeError(self.no_vimba_camera_message())

            if self.camera is None:
                self.camera = self.select_mako_camera(cameras)
            self.camera.__enter__()
            self.apply_camera_settings()
            self.sync_fields_from_camera()
            self.status_label.setText(f"Connected: {self.camera.get_id()}")
            self.update_connection_state(True)
        except Exception as exc:
            self.disconnect_camera()
            self.status_label.setText(f"Camera connection failed: {exc}")

    def direct_camera_identifiers(self):
        identifiers = []
        ip_text = getattr(self, "camera_ip_edit", None)
        if ip_text is not None:
            value = ip_text.text().strip()
            if value:
                identifiers.append(value)
        identifiers.append(self.CAMERA_ID)
        return list(dict.fromkeys(identifiers))

    def connect_direct_vimba_camera(self):
        if self.vmb is None:
            return None

        last_error = None
        for identifier in self.direct_camera_identifiers():
            try:
                camera = self.vmb.get_camera_by_id(identifier)
                self.status_label.setText(f"Direct Vimba lookup succeeded: {identifier}")
                return camera
            except Exception as exc:
                last_error = exc

        if last_error is not None:
            self.status_label.setText(f"Direct Vimba lookup failed: {last_error}")
        return None

    def discover_vimba_cameras(self):
        if self.vmb is None:
            return []
        try:
            discover_cameras = getattr(self.vmb, "_Impl__discover_cameras", None)
            if callable(discover_cameras):
                discover_cameras()
        except Exception:
            pass
        try:
            return list(self.vmb.get_all_cameras())
        except Exception:
            return []

    def no_vimba_camera_message(self):
        gentl_paths = self.vimba_gentl_paths_text()
        transport_layers = self.vimba_transport_layers_text()
        message = (
            "No Vimba camera detected. Quit Vimba X Viewer, unplug/replug the camera, "
            "then retry. On macOS with a GigE camera, allow LRPhoton/Python in "
            "System Settings > Privacy & Security > Local Network. If Wi-Fi interferes "
            "with GigE discovery, keep the Camera IP field set to the Force IP address "
            "shown in Vimba X Viewer."
        )
        if transport_layers:
            message += f" Transport layers loaded: {transport_layers}."
        if gentl_paths:
            message += f" GENICAM_GENTL64_PATH={gentl_paths}."
        return message

    def vimba_transport_layers_text(self):
        if self.vmb is None:
            return ""
        try:
            layers = list(self.vmb.get_all_transport_layers())
        except Exception:
            return ""
        layer_ids = []
        for layer in layers:
            try:
                layer_ids.append(Path(str(layer.get_id())).name)
            except Exception:
                pass
        return ", ".join(layer_ids)

    def vimba_gentl_paths_text(self):
        paths = os.environ.get("GENICAM_GENTL64_PATH", "")
        if not paths:
            return ""
        return os.pathsep.join(
            str(Path(path).expanduser().resolve())
            for path in paths.split(os.pathsep)
            if path
        )

    def select_mako_camera(self, cameras):
        for camera in cameras:
            if camera.get_id() == self.CAMERA_ID:
                return camera
            text = f"{camera.get_id()} {getattr(camera, 'get_name', lambda: '')()}".lower()
            if "g419" in text or "mako" in text:
                return camera
        return cameras[0]

    def apply_camera_settings(self):
        if self.camera is None:
            return
        loaded_settings = self.load_vimba_settings_file()
        if not loaded_settings:
            self.apply_default_camera_features()
        self.set_camera_feature("ReverseY", self.reverse_y_checkbox.isChecked())
        self.set_camera_feature("Width", self.width_spinbox.value())
        self.set_camera_feature("Height", self.height_spinbox.value())
        self.set_camera_feature("OffsetX", self.offset_x_spinbox.value())
        self.set_camera_feature("OffsetY", self.offset_y_spinbox.value())
        requested_pixel_format = self.pixel_format_combo.currentText().strip()
        self.set_camera_feature("PixelFormat", requested_pixel_format)
        exposure = self.optional_float(self.exposure_edit.text())
        self.set_camera_feature("ExposureTime", exposure)
        self.set_camera_feature("ExposureTimeAbs", exposure)
        self.set_camera_feature("Gain", self.optional_float(self.gain_edit.text()))
        self.set_camera_feature("AcquisitionFrameRate", self.DEFAULT_ACQUISITION_FRAME_RATE)
        self.set_camera_feature("AcquisitionFrameRateAbs", self.DEFAULT_ACQUISITION_FRAME_RATE)

    def load_vimba_settings_file(self):
        if self.camera is None or not self.VIMBA_SETTINGS_PATH.exists():
            return False
        try:
            from vmbpy import ModulePersistFlags, PersistType

            persist_flags = (
                ModulePersistFlags.LocalDevice
                | ModulePersistFlags.RemoteDevice
                | ModulePersistFlags.Streams
            )
            self.camera.load_settings(
                str(self.VIMBA_SETTINGS_PATH),
                PersistType.NoLUT,
                persist_flags,
                max_iterations=10,
            )
            return True
        except Exception as exc:
            if not self.is_closing:
                self.status_label.setText(f"Could not load Vimba settings XML: {exc}")
            return False

    def apply_default_camera_features(self):
        for _, name, value in self.DEFAULT_CAMERA_FEATURES:
            self.set_first_available_camera_feature(self.CAMERA_SETTING_ALIASES.get(name, (name,)), value)

    def show_camera_settings_dialog(self):
        if self.camera is None:
            self.connect_camera()

        dialog = QDialog(self)
        dialog.setWindowTitle("Réglages caméra")
        dialog.resize(980, 760)

        layout = QVBoxLayout(dialog)
        title = QLabel(self.camera_settings_title())
        title.setStyleSheet("font-weight: 600;")
        layout.addWidget(title)

        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter pattern:"))
        filter_edit = QLineEdit()
        filter_edit.setFixedHeight(self.FIELD_HEIGHT)
        filter_edit.setPlaceholderText("Example: Gain|Width")
        filter_layout.addWidget(filter_edit, 1)
        search_button = QPushButton("Search")
        search_button.setFixedHeight(self.FIELD_HEIGHT)
        filter_layout.addWidget(search_button)
        layout.addLayout(filter_layout)

        tree = QTreeWidget(dialog)
        tree.setColumnCount(2)
        tree.setHeaderLabels(["Feature", "Value"])
        tree.setAlternatingRowColors(True)
        tree.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        tree.setSortingEnabled(False)
        self.populate_camera_settings_tree(tree)

        header = tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        layout.addWidget(tree, 1)

        show_description_checkbox = QCheckBox("Show Description")
        show_description_checkbox.setChecked(True)
        layout.addWidget(show_description_checkbox)

        description_edit = QPlainTextEdit()
        description_edit.setReadOnly(True)
        description_edit.setMinimumHeight(60)
        description_edit.setMaximumHeight(90)
        layout.addWidget(description_edit)

        message_label = QLabel("")
        message_label.setWordWrap(True)
        layout.addWidget(message_label)

        apply_button = QPushButton("Appliquer à la caméra Vimba")
        apply_button.setFixedHeight(self.FIELD_HEIGHT)
        apply_button.clicked.connect(lambda: self.apply_camera_settings_tree(tree, message_label))
        layout.addWidget(apply_button)
        search_button.clicked.connect(lambda: self.filter_camera_settings_tree(tree, filter_edit.text()))
        filter_edit.returnPressed.connect(search_button.click)
        tree.currentItemChanged.connect(
            lambda current, previous: self.update_camera_feature_description(
                current,
                description_edit,
                show_description_checkbox.isChecked(),
            )
        )
        show_description_checkbox.toggled.connect(
            lambda checked: self.update_camera_feature_description(tree.currentItem(), description_edit, checked)
        )

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        dialog.exec()

    def populate_camera_settings_tree(self, tree):
        group_items = {}
        rows = self.vimba_camera_feature_rows() or self.camera_settings_rows()
        root_item = QTreeWidgetItem(tree.invisibleRootItem(), ["Camera", ""])
        root_item.setFlags(root_item.flags() & ~Qt.ItemIsEditable)
        group_items["Camera"] = root_item
        for section, name, value, feature in rows:
            parent = self.camera_settings_group_item(tree, group_items, section)
            item = QTreeWidgetItem(parent, [name, ""])
            item.setData(0, Qt.UserRole, name)
            item.setData(0, Qt.UserRole + 1, feature)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.set_camera_feature_value_widget(tree, item, feature, value)
        root_item.setExpanded(True)

    def camera_settings_group_item(self, tree, group_items, section):
        parent = group_items["Camera"]
        path = ["Camera"]
        for part in [item.strip() for item in section.split("/") if item.strip()]:
            path.append(part)
            key = " / ".join(path)
            if key not in group_items:
                group_item = QTreeWidgetItem(parent, [part, ""])
                group_item.setFlags(group_item.flags() & ~Qt.ItemIsEditable)
                group_items[key] = group_item
            parent = group_items[key]
        return parent

    def camera_settings_title(self):
        if self.camera is None:
            return "Controller for Mako G-419B"
        camera_id = self.safe_call(self.camera, "get_id") or self.CAMERA_ID
        camera_name = self.safe_call(self.camera, "get_name") or self.CAMERA_MODEL
        return f"Controller for {camera_name} ({camera_id})"

    def vimba_camera_feature_rows(self):
        if self.camera is None:
            return []
        rows = []
        for container_name, container in self.vimba_feature_containers():
            try:
                features = list(container.get_all_features())
            except Exception:
                continue
            for feature in features:
                name = self.safe_call(feature, "get_name")
                if not name:
                    continue
                category = self.safe_call(feature, "get_category") or ""
                category = str(category).strip().strip("/")
                if container_name:
                    category = f"{container_name} / {category}" if category else container_name
                rows.append((category, name, self.vimba_feature_value(feature), feature))
        rows.sort(key=lambda row: (row[0].lower(), row[1].lower()))
        return rows

    def vimba_feature_containers(self):
        if self.camera is None:
            return []
        containers = [("", self.camera)]
        try:
            streams = list(self.camera.get_streams())
        except Exception:
            streams = []
        for index, stream in enumerate(streams, start=1):
            containers.append((f"Stream {index}", stream))
        return containers

    def set_camera_feature_value_widget(self, tree, item, feature, value):
        if self.vimba_feature_is_command(feature):
            button = QPushButton("Command")
            button.setFixedHeight(self.FIELD_HEIGHT)
            button.setEnabled(self.vimba_feature_writeable(feature))
            button.clicked.connect(lambda: self.run_camera_command_feature(feature))
            tree.setItemWidget(item, 1, button)
            return

        if self.vimba_feature_is_enum(feature):
            widget = QComboBox()
            widget.setFixedHeight(self.FIELD_HEIGHT)
            values = self.vimba_enum_values(feature)
            if values:
                widget.addItems(values)
            current = str(value)
            if widget.findText(current) < 0:
                widget.addItem(current)
            widget.setCurrentText(current)
        else:
            widget = QLineEdit(str(value))
            widget.setFixedHeight(self.FIELD_HEIGHT)
        widget.setEnabled(feature is None or self.vimba_feature_writeable(feature))
        tree.setItemWidget(item, 1, widget)

    def camera_settings_rows(self):
        rows = [
            ("ImageFormat", "Width", self.width_spinbox.value(), None),
            ("ImageFormat", "Height", self.height_spinbox.value(), None),
            ("ImageFormat", "OffsetX", self.offset_x_spinbox.value(), None),
            ("ImageFormat", "OffsetY", self.offset_y_spinbox.value(), None),
            ("ImageFormat", "PixelFormat", self.pixel_format_combo.currentText().strip(), None),
            ("ImageMode", "ReverseY", self.reverse_y_checkbox.isChecked(), None),
            ("Controls / Exposure", "ExposureTime", self.exposure_edit.text().strip(), None),
            ("Controls / Gain", "Gain", self.gain_edit.text().strip(), None),
            ("Acquisition", "Preview fps", self.fps_spinbox.value(), None),
        ]
        for section, name, value in self.DEFAULT_CAMERA_FEATURES:
            rows.append((section, name, value, None))
        rows.extend([
            ("ImageMode", "SensorWidth", 2048, None),
            ("ImageMode", "SensorHeight", 2048, None),
            ("Info", "DeviceModelName", self.CAMERA_MODEL, None),
            ("Info", "SensorBits", 12, None),
            ("SavedUserSets", "UserSetDefaultSelector", "Default", None),
            ("SavedUserSets", "UserSetSelector", "Default", None),
        ])
        return rows

    def apply_camera_settings_tree(self, tree, message_label):
        values = self.camera_settings_tree_values(tree)

        self.apply_camera_settings_to_ui(values)
        if self.camera is None:
            self.connect_camera()
            self.apply_camera_settings_to_ui(values)
            if self.camera is None:
                message_label.setText("Impossible de connecter la caméra. Les valeurs de l'interface ont été mises à jour.")
                self.update_connection_state(False)
                return

        failures = []
        for name, raw_value in self.ordered_camera_settings(values):
            if name == "Preview fps":
                continue
            value = self.camera_setting_value(name, raw_value)
            feature_names = self.CAMERA_SETTING_ALIASES.get(name, (name,))
            if not self.set_first_available_camera_feature(feature_names, value):
                failures.append(name)

        self.sync_fields_from_camera()
        if failures:
            message_label.setText(
                f"Réglages envoyés. Non acceptés par Vimba: {', '.join(failures[:8])}"
                + ("..." if len(failures) > 8 else "")
            )
        else:
            message_label.setText("Réglages appliqués à la caméra Vimba.")

    def camera_settings_tree_values(self, tree):
        values = {}

        def collect(item):
            name = item.data(0, Qt.UserRole)
            if name:
                value_widget = tree.itemWidget(item, 1)
                feature = item.data(0, Qt.UserRole + 1)
                if feature is not None and self.vimba_feature_is_command(feature):
                    pass
                elif feature is not None and not self.vimba_feature_writeable(feature):
                    pass
                elif isinstance(value_widget, QComboBox):
                    values[str(name)] = value_widget.currentText().strip()
                elif value_widget is not None:
                    values[str(name)] = value_widget.text().strip()
                else:
                    values[str(name)] = item.text(1).strip()
            for child_index in range(item.childCount()):
                collect(item.child(child_index))

        root = tree.invisibleRootItem()
        for index in range(root.childCount()):
            collect(root.child(index))
        return values

    def filter_camera_settings_tree(self, tree, pattern):
        pattern = pattern.strip()
        root = tree.invisibleRootItem()
        if not pattern:
            self.set_tree_item_hidden_recursive(root, False)
            return
        try:
            import re

            matcher = re.compile(pattern, re.IGNORECASE)
            matches = lambda text: bool(matcher.search(text))
        except Exception:
            lowered = pattern.lower()
            matches = lambda text: lowered in text.lower()

        def update(item):
            own_match = matches(item.text(0)) or matches(self.camera_tree_item_value_text(tree, item))
            child_match = False
            for child_index in range(item.childCount()):
                if update(item.child(child_index)):
                    child_match = True
            visible = own_match or child_match
            item.setHidden(not visible)
            if child_match:
                item.setExpanded(True)
            return visible

        for index in range(root.childCount()):
            update(root.child(index))

    def camera_tree_item_value_text(self, tree, item):
        widget = tree.itemWidget(item, 1)
        if isinstance(widget, QComboBox):
            return widget.currentText()
        if isinstance(widget, QLineEdit):
            return widget.text()
        if isinstance(widget, QPushButton):
            return widget.text()
        return item.text(1)

    def set_tree_item_hidden_recursive(self, item, hidden):
        for child_index in range(item.childCount()):
            child = item.child(child_index)
            child.setHidden(hidden)
            self.set_tree_item_hidden_recursive(child, hidden)

    def update_camera_feature_description(self, item, description_edit, show_description):
        if not show_description or item is None:
            description_edit.clear()
            return
        feature = item.data(0, Qt.UserRole + 1)
        if feature is None:
            description_edit.setPlainText(item.text(0))
            return
        lines = [
            self.safe_call(feature, "get_display_name") or self.safe_call(feature, "get_name") or "",
            self.safe_call(feature, "get_tooltip") or "",
            self.safe_call(feature, "get_description") or "",
        ]
        access = self.safe_call(feature, "get_access_mode")
        visibility = self.safe_call(feature, "get_visibility")
        if access is not None:
            lines.append(f"Access: {access}")
        if visibility is not None:
            lines.append(f"Visibility: {visibility}")
        description_edit.setPlainText("\n".join(str(line) for line in lines if line))

    def vimba_feature_value(self, feature):
        if feature is None or not self.vimba_feature_readable(feature) or self.vimba_feature_is_command(feature):
            return ""
        try:
            value = feature.get()
        except Exception:
            return ""
        if hasattr(value, "get_name"):
            try:
                return value.get_name()
            except Exception:
                return str(value)
        return value

    def vimba_feature_readable(self, feature):
        if feature is None:
            return True
        try:
            return feature.is_readable()
        except Exception:
            return False

    def vimba_feature_writeable(self, feature):
        if feature is None:
            return True
        try:
            return feature.is_writeable()
        except Exception:
            return False

    def vimba_feature_is_enum(self, feature):
        return feature is not None and feature.__class__.__name__ == "EnumFeature"

    def vimba_feature_is_command(self, feature):
        return feature is not None and feature.__class__.__name__ == "CommandFeature"

    def vimba_enum_values(self, feature):
        try:
            entries = feature.get_available_entries()
        except Exception:
            return []
        values = []
        for entry in entries:
            if hasattr(entry, "get_name"):
                try:
                    values.append(entry.get_name())
                    continue
                except Exception:
                    pass
            values.append(str(entry))
        return values

    def run_camera_command_feature(self, feature):
        try:
            feature.run()
            self.status_label.setText(f"Command sent: {feature.get_name()}")
        except Exception as exc:
            self.status_label.setText(f"Command failed: {exc}")

    def safe_call(self, obj, method_name):
        try:
            return getattr(obj, method_name)()
        except Exception:
            return None

    def apply_camera_settings_to_ui(self, values):
        if "Width" in values:
            self.width_spinbox.setValue(self.int_camera_setting(values["Width"], self.width_spinbox.value()))
        if "Height" in values:
            self.height_spinbox.setValue(self.int_camera_setting(values["Height"], self.height_spinbox.value()))
        if "OffsetX" in values:
            self.offset_x_spinbox.setValue(self.int_camera_setting(values["OffsetX"], self.offset_x_spinbox.value()))
        if "OffsetY" in values:
            self.offset_y_spinbox.setValue(self.int_camera_setting(values["OffsetY"], self.offset_y_spinbox.value()))
        if "PixelFormat" in values:
            self.set_pixel_format_text(values["PixelFormat"])
        if "ReverseY" in values:
            self.reverse_y_checkbox.setChecked(self.bool_camera_setting(values["ReverseY"], self.reverse_y_checkbox.isChecked()))
        if "ExposureTime" in values:
            self.exposure_edit.setText(values["ExposureTime"])
        if "Gain" in values:
            self.gain_edit.setText(values["Gain"])
        if "Preview fps" in values:
            self.fps_spinbox.setValue(self.int_camera_setting(values["Preview fps"], self.fps_spinbox.value()))

    def ordered_camera_settings(self, values):
        ordered_names = [
            "AcquisitionMode",
            "ExposureAuto",
            "GainAuto",
            "PixelFormat",
            "Width",
            "Height",
            "OffsetX",
            "OffsetY",
            "ExposureTime",
            "Gain",
        ]
        ordered_names.extend(name for _, name, _ in self.DEFAULT_CAMERA_FEATURES if name not in ordered_names)
        ordered_names.extend(name for name in values if name not in ordered_names)
        for name in ordered_names:
            if name in values:
                yield name, values[name]

    def camera_setting_value(self, name, raw_value):
        raw_value = str(raw_value).strip()
        if raw_value.lower() in {"true", "on", "yes", "1"} and self.boolean_camera_setting(name):
            return True
        if raw_value.lower() in {"false", "off", "no", "0"} and self.boolean_camera_setting(name):
            return False
        numeric_defaults = self.camera_setting_numeric_defaults()
        if name not in numeric_defaults:
            return raw_value
        default_value = numeric_defaults[name]
        number = self.optional_float(raw_value)
        if number is None:
            return raw_value
        if isinstance(default_value, int) and not isinstance(default_value, bool):
            return int(round(number))
        return number

    def camera_setting_numeric_defaults(self):
        defaults = {
            "Width": self.DEFAULT_ROI_WIDTH,
            "Height": self.DEFAULT_ROI_HEIGHT,
            "OffsetX": self.DEFAULT_OFFSET_X,
            "OffsetY": self.DEFAULT_OFFSET_Y,
            "Preview fps": self.DEFAULT_PREVIEW_FPS,
            "ExposureTime": float(self.DEFAULT_EXPOSURE_US),
            "Gain": float(self.DEFAULT_GAIN),
        }
        defaults.update({name: value for _, name, value in self.DEFAULT_CAMERA_FEATURES if isinstance(value, (int, float))})
        return defaults

    def boolean_camera_setting(self, name):
        for _, feature_name, value in self.DEFAULT_CAMERA_FEATURES:
            if feature_name == name:
                return isinstance(value, bool)
        return False

    def int_camera_setting(self, raw_value, fallback):
        value = self.optional_float(raw_value)
        if value is None:
            return fallback
        return int(round(value))

    def bool_camera_setting(self, raw_value, fallback):
        value = str(raw_value).strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
        return fallback

    def set_camera_feature(self, name, value):
        if value is None:
            return False
        for _, container in self.vimba_feature_containers():
            try:
                feature = container.get_feature_by_name(name)
                feature.set(value)
                return True
            except Exception:
                continue
        return False

    def set_first_available_camera_feature(self, names, value):
        for name in names:
            if self.set_camera_feature(name, value):
                return True
        return False

    def sync_fields_from_camera(self):
        width = self.camera_feature_value("Width")
        height = self.camera_feature_value("Height")
        offset_x = self.camera_feature_value("OffsetX")
        offset_y = self.camera_feature_value("OffsetY")
        pixel_format = self.camera_feature_value("PixelFormat")
        reverse_y = self.camera_feature_value("ReverseY")
        exposure = self.camera_feature_value_any(("ExposureTime", "ExposureTimeAbs"))
        gain = self.camera_feature_value("Gain")
        fps = self.camera_feature_value_any(("AcquisitionFrameRate", "AcquisitionFrameRateAbs"))
        self.refresh_pixel_format_choices()

        if width is not None:
            self.width_spinbox.setValue(int(width))
        if height is not None:
            self.height_spinbox.setValue(int(height))
        if offset_x is not None:
            self.offset_x_spinbox.setValue(int(offset_x))
        if offset_y is not None:
            self.offset_y_spinbox.setValue(int(offset_y))
        if pixel_format is not None:
            self.set_pixel_format_text(str(pixel_format))
        if reverse_y is not None:
            self.reverse_y_checkbox.setChecked(bool(reverse_y))
        if exposure is not None:
            self.exposure_edit.setText(f"{float(exposure):.10g}")
        if gain is not None:
            self.gain_edit.setText(f"{float(gain):.10g}")
        if fps is not None:
            self.fps_spinbox.setValue(max(1, min(60, int(round(float(fps))))))

    def camera_feature_value(self, name):
        if self.camera is None:
            return None
        try:
            feature = self.camera.get_feature_by_name(name)
            return feature.get()
        except Exception:
            return None

    def camera_feature_value_any(self, names):
        for name in names:
            value = self.camera_feature_value(name)
            if value is not None:
                return value
        return None

    def refresh_pixel_format_choices(self):
        if self.camera is None:
            return
        current_text = self.pixel_format_combo.currentText().strip() or self.DEFAULT_PIXEL_FORMAT
        try:
            formats = [str(pixel_format) for pixel_format in self.camera.get_pixel_formats()]
        except Exception:
            return
        if not formats:
            return
        self.pixel_format_combo.blockSignals(True)
        self.pixel_format_combo.clear()
        for pixel_format in formats:
            self.pixel_format_combo.addItem(pixel_format)
        if current_text in formats:
            self.pixel_format_combo.setCurrentText(current_text)
        elif self.DEFAULT_PIXEL_FORMAT in formats:
            self.pixel_format_combo.setCurrentText(self.DEFAULT_PIXEL_FORMAT)
        else:
            self.pixel_format_combo.setCurrentIndex(0)
        self.pixel_format_combo.blockSignals(False)

    def set_pixel_format_text(self, text):
        if self.pixel_format_combo.findText(text) < 0:
            self.pixel_format_combo.addItem(text)
        self.pixel_format_combo.setCurrentText(text)

    def apply_line_geometry(self, name, geometry):
        self.current_geometry_name = name
        self.center_x_edit.setText(str(geometry.get("center_x", "")))
        self.center_y_edit.setText(str(geometry.get("center_y", "")))
        self.pixel_x_edit.setText(str(geometry.get("pixel_x_m", "")))
        self.pixel_y_edit.setText(str(geometry.get("pixel_y_m", "")))
        distance_m = self.DEFAULT_DISTANCE_M if name == LRP_SALS_DEFAULT_NAME else geometry.get("distance_m", "")
        self.distance_edit.setText(str(distance_m))
        self.wavelength_edit.setText(str(geometry.get("wavelength_m", "")))
        if self.current_frame is not None:
            self.update_preview()

    def update_default_center_x(self, value):
        if self.current_geometry_name == LRP_SALS_DEFAULT_NAME:
            self.center_x_edit.setText(self.default_center_text(value))

    def update_default_center_y(self, value):
        if self.current_geometry_name == LRP_SALS_DEFAULT_NAME:
            self.center_y_edit.setText(self.default_center_text(value))

    def start_live(self):
        if self.camera is None:
            self.connect_camera()
        if self.camera is None:
            return
        self.apply_camera_settings()
        interval_ms = max(1, int(1000 / max(1, self.fps_spinbox.value())))
        self.live_timer.start(interval_ms)
        self.status_label.setText("Live acquisition running.")
        self.update_connection_state(True)

    def stop_live(self, update_status=True):
        if self.is_recording_frames:
            self.stop_recording_frames(update_status=False)
        self.live_timer.stop()
        if update_status and not self.is_closing:
            self.status_label.setText("Live acquisition stopped.")
            self.update_connection_state(self.camera is not None)

    def grab_live_frame(self):
        if self.is_closing:
            return
        if self.camera is None:
            self.stop_live(update_status=False)
            return
        self.is_grabbing_frame = True
        try:
            frame = self.camera.get_frame(timeout_ms=1000)
            if self.is_closing:
                return
            image = np.asarray(self.frame_to_numpy(frame))
            if image.ndim == 3 and image.shape[-1] == 1:
                image = image[:, :, 0]
            elif image.ndim == 3:
                image = image.mean(axis=2)
            image = np.flipud(image)
            self.current_frame = np.asarray(image)
            self.frame_index += 1
            self.update_preview()
            self.update_connection_state(self.camera is not None)
        except Exception as exc:
            if not self.is_closing:
                self.status_label.setText(f"Frame grab failed: {exc}")
        finally:
            self.is_grabbing_frame = False

    def frame_to_numpy(self, frame):
        source_format = frame.get_pixel_format()
        source_name = str(source_format)
        if "Packed" in source_name or source_name.endswith("p"):
            return self.convert_frame_to_numpy(frame, source_format)

        try:
            return frame.as_numpy_ndarray()
        except Exception as error:
            if "PixelFormat" not in str(error):
                raise

        return self.convert_frame_to_numpy(frame, source_format)

    def convert_frame_to_numpy(self, frame, source_format):
        from vmbpy import PixelFormat

        convertible = source_format.get_convertible_formats()
        preferred_formats = (PixelFormat.Mono16, PixelFormat.Mono12, PixelFormat.Mono8)
        for target_format in preferred_formats:
            if target_format not in convertible:
                continue
            try:
                converted_frame = frame.convert_pixel_format(target_format)
                return converted_frame.as_numpy_ndarray()
            except Exception:
                continue

        raise ValueError(f"PixelFormat {source_format} cannot be converted to Mono16, Mono12 or Mono8.")

    def update_preview(self):
        if self.current_frame is None:
            return
        self.canvas.setVisible(True)
        image = np.asarray(self.current_frame, dtype=float)
        display_image, vmin, vmax = self.preview_display_image(image)
        self.ax.clear()
        self.figure.patch.set_alpha(0)
        self.ax.set_facecolor("none")
        self.ax.imshow(display_image, cmap="jet", origin="upper", vmin=vmin, vmax=vmax)
        center_x = self.optional_float(self.center_x_edit.text())
        center_y = self.optional_float(self.center_y_edit.text())
        if center_x is None:
            center_x = (image.shape[1] - 1.0) / 2.0
        if center_y is None:
            center_y = (image.shape[0] - 1.0) / 2.0
        self.ax.axvline(center_x, color="white", linewidth=0.8, alpha=0.9)
        self.ax.axhline(center_y, color="white", linewidth=0.8, alpha=0.9)
        self.ax.set_axis_off()
        self.apply_preview_zoom(image.shape)
        self.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.refresh_preview_coordinate_label()
        self.canvas.draw_idle()

    def eventFilter(self, obj, event):
        if obj is self.canvas and self.current_frame is not None:
            if event.type() == QEvent.NativeGesture:
                try:
                    gesture_type = event.gestureType()
                    value = event.value()
                    if gesture_type == Qt.ZoomNativeGesture and value != 0:
                        scale = 1.0 / (1.0 + value) if value > -0.95 else 1.25
                        self.zoom_preview_at(event.position(), scale)
                        event.accept()
                        return True
                    if gesture_type == Qt.SmartZoomNativeGesture:
                        self.reset_preview_zoom()
                        event.accept()
                        return True
                except Exception:
                    pass
            if event.type() == QEvent.Gesture:
                gesture = event.gesture(Qt.PinchGesture)
                if gesture is not None:
                    scale = gesture.scaleFactor()
                    last_scale = gesture.lastScaleFactor()
                    if last_scale > 0:
                        scale = scale / last_scale
                    if scale > 0:
                        self.zoom_preview_at(self.pinch_gesture_position(gesture), 1.0 / scale)
                    return True
            if event.type() == QEvent.Wheel:
                delta = event.pixelDelta()
                if delta.isNull():
                    angle_delta = event.angleDelta()
                    delta_x = angle_delta.x() / 8.0
                    delta_y = angle_delta.y() / 8.0
                else:
                    delta_x = delta.x()
                    delta_y = delta.y()
                if delta_x or delta_y:
                    if event.modifiers() & (Qt.ControlModifier | Qt.MetaModifier):
                        if delta_y:
                            scale = 0.88 if delta_y > 0 else 1.14
                            self.zoom_preview_at(event.position(), scale)
                    else:
                        self.pan_preview_by(delta_x, delta_y)
                    return True
        return super().eventFilter(obj, event)

    def zoom_preview_at(self, qt_position, scale):
        if self.current_frame is None:
            return
        image_shape = np.asarray(self.current_frame).shape
        if len(image_shape) < 2:
            return

        self.apply_preview_zoom(image_shape)
        if self.preview_xlim is None or self.preview_ylim is None:
            xlim, ylim = self.full_preview_limits(image_shape)
        else:
            xlim, ylim = self.preview_xlim, self.preview_ylim

        x_data, y_data = self.preview_data_position(qt_position)
        if not np.isfinite(x_data) or not np.isfinite(y_data):
            x_data = (xlim[0] + xlim[1]) / 2.0
            y_data = (ylim[0] + ylim[1]) / 2.0

        full_xlim, full_ylim = self.full_preview_limits(image_shape)
        full_width = abs(full_xlim[1] - full_xlim[0])
        full_height = abs(full_ylim[1] - full_ylim[0])
        new_width = min(full_width, abs(xlim[1] - xlim[0]) * scale)
        new_height = min(full_height, abs(ylim[1] - ylim[0]) * scale)

        if new_width >= full_width * 0.995 and new_height >= full_height * 0.995:
            self.reset_preview_zoom(draw=False)
            self.canvas.draw_idle()
            return

        x_fraction = 0.5 if xlim[1] == xlim[0] else (x_data - xlim[0]) / (xlim[1] - xlim[0])
        y_fraction = 0.5 if ylim[1] == ylim[0] else (y_data - ylim[0]) / (ylim[1] - ylim[0])
        new_xlim = (x_data - new_width * x_fraction, x_data + new_width * (1.0 - x_fraction))
        new_ylim = (y_data - (ylim[1] - ylim[0]) / abs(ylim[1] - ylim[0]) * new_height * y_fraction,
                    y_data + (ylim[1] - ylim[0]) / abs(ylim[1] - ylim[0]) * new_height * (1.0 - y_fraction))

        self.preview_xlim = self.clamp_axis_limits(new_xlim, full_xlim)
        self.preview_ylim = self.clamp_axis_limits(new_ylim, full_ylim)
        self.apply_preview_zoom(image_shape)
        self.canvas.draw_idle()

    def pan_preview_by(self, delta_x, delta_y):
        if self.current_frame is None:
            return
        image_shape = np.asarray(self.current_frame).shape
        if len(image_shape) < 2:
            return
        self.apply_preview_zoom(image_shape)
        if self.preview_xlim is None or self.preview_ylim is None:
            xlim, ylim = self.full_preview_limits(image_shape)
        else:
            xlim, ylim = self.preview_xlim, self.preview_ylim

        bbox = self.ax.bbox
        if bbox.width <= 0 or bbox.height <= 0:
            return
        dx = -delta_x * abs(xlim[1] - xlim[0]) / bbox.width
        dy = delta_y * (ylim[1] - ylim[0]) / bbox.height
        full_xlim, full_ylim = self.full_preview_limits(image_shape)
        self.preview_xlim = self.clamp_axis_limits((xlim[0] + dx, xlim[1] + dx), full_xlim)
        self.preview_ylim = self.clamp_axis_limits((ylim[0] + dy, ylim[1] + dy), full_ylim)
        self.apply_preview_zoom(image_shape)
        self.canvas.draw_idle()

    def preview_data_position(self, qt_position):
        x = qt_position.x()
        y = self.canvas.height() - qt_position.y()
        return self.ax.transData.inverted().transform((x, y))

    def update_preview_coordinates(self, event):
        if self.current_frame is None or event.inaxes is not self.ax or event.xdata is None or event.ydata is None:
            self.clear_preview_coordinates()
            return
        image = np.asarray(self.current_frame)
        if image.ndim < 2:
            self.clear_preview_coordinates()
            return
        height, width = image.shape[:2]
        x_index = int(round(event.xdata))
        y_index = int(round(event.ydata))
        if x_index < 0 or x_index >= width or y_index < 0 or y_index >= height:
            self.clear_preview_coordinates()
            return
        self.preview_hover_pixel = (x_index, y_index)
        self.refresh_preview_coordinate_label()

    def clear_preview_coordinates(self, event=None):
        self.preview_hover_pixel = None
        if hasattr(self, "preview_coordinate_label"):
            self.preview_coordinate_label.setText("x = - | y = - | q = - | angle = - | I = -")

    def refresh_preview_coordinate_label(self):
        if self.current_frame is None or self.preview_hover_pixel is None or not hasattr(self, "preview_coordinate_label"):
            return
        image = np.asarray(self.current_frame)
        if image.ndim < 2:
            self.clear_preview_coordinates()
            return
        height, width = image.shape[:2]
        x_index, y_index = self.preview_hover_pixel
        if x_index < 0 or x_index >= width or y_index < 0 or y_index >= height:
            self.clear_preview_coordinates()
            return

        value = np.asarray(image[y_index, x_index]).squeeze()
        if value.size == 1:
            intensity = float(value)
        else:
            intensity = float(np.nanmean(value))
        intensity_text = self.format_preview_value(intensity)
        q_value, angle_value = self.calculate_preview_q_angle(x_index, y_index, image.shape)
        q_text = "-" if q_value is None else f"{q_value:.5g} nm⁻¹"
        angle_text = "-" if angle_value is None else f"{angle_value:.2f}°"
        self.preview_coordinate_label.setText(
            f"x = {x_index + 1} | y = {y_index + 1} | q = {q_text} | angle = {angle_text} | I = {intensity_text}"
        )

    def calculate_preview_q_angle(self, x_index, y_index, image_shape):
        height, width = image_shape[:2]
        center_x = self.optional_float(self.center_x_edit.text())
        center_y = self.optional_float(self.center_y_edit.text())
        if center_x is None:
            center_x = (width - 1.0) / 2.0
        if center_y is None:
            center_y = (height - 1.0) / 2.0

        dx_pixels = x_index - center_x
        dy_pixels = y_index - center_y
        angle = (np.degrees(np.arctan2(dy_pixels, dx_pixels)) + 360.0) % 360.0

        distance_m = self.optional_float(self.distance_edit.text())
        pixel_x_m = self.optional_float(self.pixel_x_edit.text())
        pixel_y_m = self.optional_float(self.pixel_y_edit.text())
        wavelength_m = self.optional_float(self.wavelength_edit.text())
        if (
            distance_m is None or distance_m <= 0
            or pixel_x_m is None or pixel_x_m <= 0
            or pixel_y_m is None or pixel_y_m <= 0
            or wavelength_m is None or wavelength_m <= 0
        ):
            return None, angle

        radius_m = np.hypot(dx_pixels * pixel_x_m, dy_pixels * pixel_y_m)
        two_theta = np.arctan2(radius_m, distance_m)
        wavelength_nm = wavelength_m * 1e9
        q_nm = (4.0 * np.pi / wavelength_nm) * np.sin(two_theta / 2.0)
        return q_nm, angle

    def format_preview_value(self, value):
        if not np.isfinite(value):
            return "-"
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        return f"{value:.6g}"

    def pinch_gesture_position(self, gesture):
        try:
            return gesture.centerPoint()
        except Exception:
            pass
        try:
            return self.canvas.mapFromGlobal(gesture.hotSpot().toPoint())
        except Exception:
            return self.canvas.rect().center()

    def full_preview_limits(self, image_shape):
        height, width = image_shape[:2]
        return (-0.5, width - 0.5), (height - 0.5, -0.5)

    def apply_preview_zoom(self, image_shape):
        if self.preview_xlim is None or self.preview_ylim is None:
            xlim, ylim = self.full_preview_limits(image_shape)
        else:
            xlim = self.preview_xlim
            ylim = self.preview_ylim
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)

    def sync_preview_limits_from_axes(self):
        if self.current_frame is None:
            return
        self.preview_xlim = tuple(self.ax.get_xlim())
        self.preview_ylim = tuple(self.ax.get_ylim())

    def reset_preview_zoom(self, draw=True):
        self.preview_xlim = None
        self.preview_ylim = None
        if self.current_frame is not None:
            self.apply_preview_zoom(np.asarray(self.current_frame).shape)
        if draw:
            self.canvas.draw_idle()

    def clamp_axis_limits(self, limits, full_limits):
        start, end = limits
        full_start, full_end = full_limits
        direction = 1.0 if full_end >= full_start else -1.0
        width = min(abs(end - start), abs(full_end - full_start))
        low = min(full_start, full_end)
        high = max(full_start, full_end)
        center = (start + end) / 2.0
        center = min(max(center, low + width / 2.0), high - width / 2.0)
        return center - direction * width / 2.0, center + direction * width / 2.0

    def preview_display_image(self, image):
        finite = np.isfinite(image)
        if np.any(finite):
            display_image = image
            display_finite = np.isfinite(display_image)
            vmin, vmax = np.nanpercentile(display_image[display_finite], [0.1, 99.9])
            if vmax <= vmin:
                vmax = vmin + 1
        else:
            display_image = image
            vmin, vmax = 0, 1
        return display_image, vmin, vmax

    def choose_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose EDF output folder", self.output_edit.text())
        if folder:
            self.output_folder = Path(folder)
            self.output_edit.setText(str(self.output_folder))

    def start_recording_frames(self):
        if self.is_recording_frames:
            return
        if self.camera is None:
            self.connect_camera()
        if self.camera is None:
            return
        if not self.live_timer.isActive():
            self.start_live()
        if self.camera is None:
            return

        self.is_recording_frames = True
        self.recording_started_at = datetime.now()
        self.recording_frame_count = 0
        self.recording_output_folder = self.create_recording_output_folder(self.recording_started_at)
        interval_ms = max(1, int(1000 / max(1, self.fps_spinbox.value())))
        self.record_timer.start(interval_ms)
        self.record_title_timer.start(250)
        self.update_recording_title()
        self.update_connection_state(True)
        self.status_label.setText(
            f"Recording EDF frames at {self.fps_spinbox.value()} fps in {self.recording_output_folder.name}."
        )
        if self.current_frame is not None:
            self.record_current_frame()

    def stop_recording_frames(self, update_status=True):
        if not self.is_recording_frames and not self.record_timer.isActive():
            return
        self.record_timer.stop()
        self.record_title_timer.stop()
        self.is_recording_frames = False
        saved_count = self.recording_frame_count
        output_folder = self.recording_output_folder
        self.recording_started_at = None
        self.recording_output_folder = None
        self.update_recording_title()
        self.update_connection_state(self.camera is not None)
        if update_status and not self.is_closing:
            folder_text = f" in {output_folder}" if output_folder is not None else ""
            self.status_label.setText(f"Recording stopped. Saved {saved_count} EDF frame(s){folder_text}.")

    def update_recording_title(self):
        if not hasattr(self, "preview_toolbar_box"):
            return
        if not self.is_recording_frames or self.recording_started_at is None:
            self.preview_toolbar_box.setTitle("Live EDF preview")
            return
        elapsed = datetime.now() - self.recording_started_at
        total_seconds = max(0, int(elapsed.total_seconds()))
        minutes, seconds = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        self.preview_toolbar_box.setTitle(
            f"Live EDF preview - REC {hours:02d}:{minutes:02d}:{seconds:02d}"
        )

    def record_current_frame(self):
        if not self.is_recording_frames or self.current_frame is None:
            return
        try:
            self.recording_frame_count += 1
            output_path = self.write_current_edf_file(record_index=self.recording_frame_count)
            self.status_label.setText(
                f"Recording EDF: {self.recording_frame_count} frame(s) saved - {output_path.name}"
            )
        except Exception as exc:
            self.stop_recording_frames(update_status=False)
            self.status_label.setText(f"Recording stopped: {exc}")

    def create_recording_output_folder(self, started_at):
        base_folder = Path(self.output_edit.text()).expanduser()
        sample_name = self.sample_name_edit.text().strip() or "sals"
        prefix = self.filename_prefix_from_sample_name(sample_name)
        timestamp = started_at.strftime("%Y%m%d_%H%M%S")
        folder = base_folder / f"{prefix}_recording_{timestamp}"
        suffix = 2
        while folder.exists():
            folder = base_folder / f"{prefix}_recording_{timestamp}_{suffix}"
            suffix += 1
        folder.mkdir(parents=True, exist_ok=False)
        return folder

    def save_current_edf(self):
        if self.current_frame is None:
            self.status_label.setText("No live frame to save.")
            return
        try:
            output_path = self.write_current_edf_file()
        except ImportError:
            self.status_label.setText("fabio EDF support is not available.")
            return
        except Exception as exc:
            self.status_label.setText(f"Save EDF failed: {exc}")
            return
        self.status_label.setText(f"Saved EDF: {output_path.name}")

    def write_current_edf_file(self, record_index=None):
        try:
            from fabio.edfimage import EdfImage
        except ImportError:
            raise

        if record_index is not None and self.recording_output_folder is not None:
            folder = self.recording_output_folder
        else:
            folder = Path(self.output_edit.text()).expanduser()
        folder.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        sample_name = self.sample_name_edit.text().strip() or "sals"
        prefix = self.filename_prefix_from_sample_name(sample_name)
        record_suffix = f"_rec{record_index:06d}" if record_index is not None else ""
        output_path = folder / f"{prefix}_{timestamp}{record_suffix}.edf"

        image = np.asarray(self.current_frame)
        header = self.edf_header(image, timestamp)
        try:
            edf = EdfImage(data=image, header=header)
            edf.write(str(output_path))
        except TypeError:
            edf = EdfImage(data=image)
            edf.header.update(header)
            edf.write(str(output_path))
        return output_path

    def edf_header(self, image, timestamp):
        ny, nx = image.shape[:2]
        distance_m = self.optional_float(self.distance_edit.text())
        pixel_x_m = self.optional_float(self.pixel_x_edit.text())
        pixel_y_m = self.optional_float(self.pixel_y_edit.text())
        wavelength_m = self.optional_float(self.wavelength_edit.text())
        center_x = self.optional_float(self.center_x_edit.text())
        center_y = self.optional_float(self.center_y_edit.text())
        sample_name = self.sample_name_edit.text().strip()
        header = {
            "HeaderID": "EH:000001:000000:000000",
            "Image": str(self.frame_index),
            "ByteOrder": "LowByteFirst",
            "DataType": self.edf_data_type(image),
            "Dim_1": str(nx),
            "Dim_2": str(ny),
            "Camera": f"Allied Vision {self.CAMERA_MODEL}",
            "CameraID": self.CAMERA_ID,
            "PixelFormat": self.pixel_format_combo.currentText().strip(),
            "ROIWidth": str(nx),
            "ROIHeight": str(ny),
            "OffsetX": str(self.offset_x_spinbox.value()),
            "OffsetY": str(self.offset_y_spinbox.value()),
            "AcquisitionDate": timestamp,
            "ExposureTime": self.exposure_edit.text().strip(),
            "Gain": self.gain_edit.text().strip(),
            "SampleName": sample_name,
            "LRPhotonModule": "VimbaSALS",
            "LineGeometry": self.current_geometry_name,
            "SampleDistance": self.format_header_number(distance_m),
            "PSize_1": self.format_header_number(pixel_x_m),
            "PSize_2": self.format_header_number(pixel_y_m),
            "Wavelength": self.format_header_number(wavelength_m),
            "Center_1": self.format_header_number(center_x),
            "Center_2": self.format_header_number(center_y),
            "YReversed": "1" if self.reverse_y_checkbox.isChecked() else "0",
        }
        return {key: value for key, value in header.items() if value not in {None, ""}}

    def filename_prefix_from_sample_name(self, sample_name):
        text = str(sample_name).strip() or "sals"
        text = re.sub(r"\s+", "_", text)
        text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", text)
        text = text.strip("._")
        return text or "sals"

    def edf_data_type(self, image):
        dtype = np.asarray(image).dtype
        if dtype == np.dtype("uint16"):
            return "UnsignedShort"
        if dtype == np.dtype("int16"):
            return "SignedShort"
        if dtype == np.dtype("uint32"):
            return "UnsignedInteger"
        if dtype == np.dtype("int32"):
            return "SignedInteger"
        if dtype == np.dtype("uint8"):
            return "UnsignedByte"
        if dtype == np.dtype("int8"):
            return "SignedByte"
        if dtype == np.dtype("float32"):
            return "FloatValue"
        if dtype == np.dtype("float64"):
            return "DoubleValue"
        return str(dtype)

    def format_header_number(self, value):
        if value is None:
            return ""
        return f"{value:.12g}"

    def optional_float(self, text):
        text = str(text).strip().replace(",", ".")
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def scaled_optional_float(self, text, scale):
        value = self.optional_float(text)
        if value is None:
            return None
        return value * scale

    def default_center_text(self, size):
        return default_center_text(size)

    def shutdown_camera(self):
        self.is_closing = True
        self.disconnect_camera(closing=True)

    def disconnect_camera(self, closing=False):
        if closing:
            self.is_closing = True
        if self.is_recording_frames:
            self.stop_recording_frames(update_status=False)
        self.live_timer.stop()
        if closing:
            QApplication.processEvents()
        if self.camera is not None:
            self.stop_camera_acquisition()
            try:
                self.camera.__exit__(None, None, None)
            except Exception:
                pass
            self.camera = None
        if self.vmb is not None:
            try:
                self.vmb.__exit__(None, None, None)
            except Exception:
                pass
            self.vmb = None
        if not closing:
            self.update_connection_state(False)

    def stop_camera_acquisition(self):
        if self.camera is None:
            return
        try:
            if self.camera.is_streaming():
                self.camera.stop_streaming()
        except Exception:
            pass
        try:
            feature = self.camera.get_feature_by_name("AcquisitionStop")
            feature.run()
        except Exception:
            pass

    def closeEvent(self, event):
        self.shutdown_camera()
        super().closeEvent(event)
