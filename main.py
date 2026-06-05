import sys
import os
import json
import platform
import webbrowser
import shutil
import tempfile
import zipfile
import subprocess
import base64
import importlib.util
from datetime import datetime
from pathlib import Path


REQUIRED_PYTHON_MODULES = [
    ("PySide6", "PySide6"),
    ("numpy", "numpy"),
    ("matplotlib", "matplotlib"),
    ("h5py", "h5py"),
    ("requests", "requests"),
    ("hdf5plugin", "hdf5plugin"),
    ("fabio", "fabio"),
    ("scipy", "scipy"),
    ("pyFAI", "pyFAI"),
]


def ensure_required_python_modules():
    if (
        "--skip-dependency-check" in sys.argv
        or os.getenv("LRPHOTON_SKIP_DEPENDENCY_CHECK") in ("1", "true", "True", "TRUE")
    ):
        return

    missing_packages = [
        package_name
        for import_name, package_name in REQUIRED_PYTHON_MODULES
        if importlib.util.find_spec(import_name) is None
    ]
    if not missing_packages:
        return

    pip_command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        *missing_packages,
    ]

    try:
        subprocess.check_call(pip_command)
    except subprocess.CalledProcessError:
        try:
            subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])
            subprocess.check_call(pip_command)
        except Exception as error:
            package_list = ", ".join(missing_packages)
            raise RuntimeError(
                f"Impossible to install missing LRPhoton dependencies: {package_list}.\n"
                f"Run manually: {sys.executable} -m pip install {' '.join(missing_packages)}"
            ) from error


ensure_required_python_modules()


import requests

from PySide6.QtCore import Qt, QTimer, QSize, QByteArray, QBuffer, QIODevice

from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMessageBox,
    QFrame,
    QDialog,
    QTextEdit,
    QPushButton,
    QHBoxLayout,
    QMainWindow,
    QStackedWidget,
    QScrollArea,
    QSizePolicy,
    QTabBar,
    QVBoxLayout,
    QWidget
)

from PySide6.QtGui import QColor, QPainter, QPixmap, QIcon


# Application version and author
APP_NAME = "LRPhoton"
APP_VERSION = "2026.05"
APP_AUTHOR = "Nathan Piaget - Laboratoire Rhéologie et Procédés"
APP_AFFILIATION = "CNRS - Université Grenoble Alpes"
# Constants
REPORT_EMAIL = "nathan.piaget@univ-grenoble-alpes.fr"
# update test6
GITHUB_OWNER = "nathan38p"
GITHUB_REPO = "LRPhoton-releases"
GITHUB_BRANCH = "main"
GITHUB_URL = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}"
UPDATE_INFO_URL = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/commits/{GITHUB_BRANCH}"
RAW_FILE_URL = f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/{GITHUB_BRANCH}"
SOURCE_ZIP_URL = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/archive/refs/heads/{GITHUB_BRANCH}.zip"
LOCAL_VERSION_FILE = Path(__file__).resolve().parent / ".lrphoton_commit"
USER_SETTINGS_FILE = Path.home() / ".lrphoton" / "settings.json"

def local_asset_path(file_name):
    return Path(__file__).resolve().parent / "assets" / file_name


def ensure_asset_file(file_name):
    asset_path = local_asset_path(file_name)
    if asset_path.exists() and asset_path.stat().st_size > 0:
        return asset_path

    try:
        asset_path.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(f"{RAW_FILE_URL}/assets/{file_name}", timeout=10)
        response.raise_for_status()
        asset_path.write_bytes(response.content)
    except Exception:
        pass

    return asset_path


APP_ICON_FILE = ensure_asset_file("LRPhoton.ico")
if not APP_ICON_FILE.exists() or APP_ICON_FILE.stat().st_size <= 0:
    APP_ICON_FILE = ensure_asset_file("LRPhoton.png")

# Application icon creation utility for window manager (with macOS padding)
def make_application_icon():
    """
    Build the application icon used by the window manager.
    On macOS, the Dock icon needs visual padding, otherwise a full-canvas PNG looks
    oversized compared with normal macOS app icons.
    """
    if not APP_ICON_FILE.exists():
        return QIcon()

    if sys.platform != "darwin":
        return QIcon(str(APP_ICON_FILE))

    source_pixmap = QPixmap(str(APP_ICON_FILE))
    if source_pixmap.isNull():
        return QIcon(str(APP_ICON_FILE))

    canvas_size = 1024
    icon_size = 830
    canvas = QPixmap(canvas_size, canvas_size)
    canvas.fill(Qt.transparent)

    painter = QPainter(canvas)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setRenderHint(QPainter.SmoothPixmapTransform)
    painter.drawPixmap(
        (canvas_size - icon_size) // 2,
        (canvas_size - icon_size) // 2,
        source_pixmap.scaled(
            icon_size,
            icon_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        ),
    )
    painter.end()

    return QIcon(canvas)

class ColoredTabBar(QTabBar):
    TAB_COLORS = {
        0: ("#dbeafe", "#2563eb"),  # View 2D
        1: ("#dbeafe", "#2563eb"),  # Plot 1D
        2: ("#e5e7eb", "#6b7280"),  # Tools
        3: ("#ffedd5", "#f97316"),  # Center
        4: ("#ffedd5", "#f97316"),  # Background
        5: ("#ffedd5", "#f97316"),  # Average
        6: ("#ffedd5", "#f97316"),  # Cave
        7: ("#ffedd5", "#f97316"),  # Unfold
        8: ("#dcfce7", "#16a34a"),  # Radial
        9: ("#dcfce7", "#16a34a"),  # Azimuthal
        10: ("#f3e8ff", "#9333ea"),  # Anisotropy
        11: ("#fde68a", "#d97706"),  # Sandbox
    }

    def tabSizeHint(self, index):
        size = super().tabSizeHint(index)
        return QSize(size.width() + 4, size.height() + 2)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        for index in range(self.count()):
            rect = self.tabRect(index).adjusted(1, 1, -1, -1)
            pastel_color, selected_color = self.TAB_COLORS.get(index, ("#eeeeee", "#007aff"))
            selected = index == self.currentIndex()
            enabled = self.isTabEnabled(index)

            background = QColor(selected_color if selected else pastel_color)
            text_color = QColor("#ffffff" if selected else "#222222")

            if not enabled:
                background = QColor("#f5f5f5")
                text_color = QColor("#9a9a9a")

            painter.setPen(Qt.NoPen)
            painter.setBrush(background)
            painter.drawRoundedRect(rect, 8, 8)

            painter.setPen(text_color)
            painter.drawText(rect, Qt.AlignCenter, self.tabText(index))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(make_application_icon())

        container = QWidget()
        container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(22, 18, 22, 10)
        self.main_layout_left_margin = 22
        self.main_layout_right_margin = 22
        main_layout.setSpacing(8)

        # ============================================================
        # HEADER
        # ============================================================

        header = QFrame()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(4, 0, 4, 0)
        header_layout.setSpacing(8)

        logo_path = ensure_asset_file("LRPhoton.png")

        logo = QLabel()
        logo_pixmap = QPixmap(str(logo_path))
        if not logo_pixmap.isNull():
            logo.setPixmap(
                logo_pixmap.scaled(
                    42,
                    42,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )
        else:
            logo.setText("◉")
            logo.setAlignment(Qt.AlignCenter)
            logo.setStyleSheet("font-size: 30px; color: #2563eb;")
        logo.setFixedSize(42, 42)
        logo.setCursor(Qt.PointingHandCursor)
        logo.mousePressEvent = lambda event: self.open_about_dialog()

        title = QLabel(APP_NAME)
        title.setCursor(Qt.PointingHandCursor)
        title.mousePressEvent = lambda event: self.open_about_dialog()
        title.setStyleSheet("""
            QLabel {
                font-size: 20px;
                font-weight: 700;
                color: #333333;
            }
        """)

        self.dev_label = QLabel("BETA")
        self.dev_label.setStyleSheet("""
            QLabel {
                font-size: 10px;
                font-weight: 700;
                color: #155724;
                background: #d4edda;
                border: 1px solid #c3e6cb;
                border-radius: 5px;
                padding: 0px 5px;
                min-height: 18px;
            }
        """)
        self.dev_label.setFixedHeight(18)
        self.dev_label.setVisible(self.is_development_copy())

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(8)
        title_row.addWidget(title)
        title_row.addWidget(self.dev_label)
        title_row.addStretch()

        subtitle = QLabel("SAXS / WAXS data processing")
        subtitle.setStyleSheet("""
            QLabel {
                font-size: 12px;
                color: #777777;
            }
        """)

        title_box = QVBoxLayout()
        title_box.setContentsMargins(0, 0, 0, 0)
        title_box.setSpacing(0)
        title_box.addLayout(title_row)
        title_box.addWidget(subtitle)

        header_layout.addWidget(logo)
        header_layout.addLayout(title_box)

        self.header_balance_spacer = QWidget()
        self.header_balance_spacer.setFixedWidth(120)

        # ============================================================
        # TAB BAR IN HEADER
        # ============================================================

        self.tab_bar = ColoredTabBar()
        self.tab_bar.setExpanding(False)
        self.tab_bar.setMovable(False)
        self.tab_bar.setUsesScrollButtons(True)

        self.tab_bar.addTab("🖼️ View 2D")
        self.tab_bar.addTab("📈 Plot 1D")
        self.tools_tab_index = self.tab_bar.addTab("🛠️ Tools")
        self.tab_bar.addTab("🎯 Center")
        self.background_tab_index = self.tab_bar.addTab("🧹 Background")
        self.tab_bar.addTab("🧮 Average")
        self.tab_bar.addTab("🕳️ Cave")
        self.unfold_tab_index = self.tab_bar.addTab("Unfold")
        self.radial_tab_index = self.tab_bar.addTab("⭕ Radial")
        self.tab_bar.addTab("〰️ Azimuthal")
        self.tab_bar.addTab("🧬 Anisotropy")
        self.sandbox_tab_index = self.tab_bar.addTab("🧪 Sandbox")

        is_development_copy = self.is_development_copy()

        if is_development_copy:
            self.tab_bar.setTabText(self.background_tab_index, "🧹 Background")
            self.tab_bar.setTabEnabled(self.background_tab_index, True)
        else:
            self.tab_bar.setTabText(self.background_tab_index, "🔒 Background")
            self.tab_bar.setTabEnabled(self.background_tab_index, False)

        self.tab_bar.setTabText(self.tools_tab_index, "🛠️ Tools")
        self.tab_bar.setTabEnabled(self.tools_tab_index, True)
        self.tab_bar.setTabText(self.sandbox_tab_index, "🧪 Sandbox")
        self.tab_bar.setTabVisible(self.sandbox_tab_index, is_development_copy)
        self.tab_bar.setTabEnabled(self.sandbox_tab_index, is_development_copy)
        self.tab_bar.setTabVisible(self.unfold_tab_index, False)

        header_layout.addStretch()
        header_layout.addWidget(self.tab_bar)
        header_layout.addStretch()
        header_layout.addWidget(self.header_balance_spacer)

        self.report_button = QPushButton("💬 Feedback")
        self.report_button.setFixedHeight(28)
        self.report_button.setCursor(Qt.PointingHandCursor)
        self.report_button.setStyleSheet("""
            QPushButton {
                font-size: 11px;
                color: #444444;
                padding: 4px 10px;
                border-radius: 8px;
                border: 1px solid #dddddd;
                background: #f8f8f8;
            }
            QPushButton:hover {
                background: #eeeeee;
            }
        """)
        self.report_button.clicked.connect(self.open_issue_report_dialog)
        self.report_button.setVisible(not self.is_development_copy())
        header_layout.addWidget(self.report_button)

        self.version_label = QPushButton()
        self.version_label.setFixedHeight(28)
        self.version_label.setEnabled(False)
        self.version_label.setCursor(Qt.PointingHandCursor)
        self.version_label.clicked.connect(self.on_update_button_clicked)
        header_layout.addWidget(self.version_label)

        self.available_update_sha = None
        self.set_update_button_state("disabled", "Checking for updates…")
        self.silent_update_test = (
            "--silent-update-test" in sys.argv
            or os.getenv("LRPHOTON_SILENT_UPDATE_TEST") in ("1", "true", "True", "TRUE")
        )

        main_layout.addWidget(header)

        # ============================================================
        # PAGE CONTENT FULL WIDTH
        # ============================================================

        self.pages = QStackedWidget()
        loading_label = QLabel("Loading...")
        loading_label.setAlignment(Qt.AlignCenter)
        loading_label.setStyleSheet("""
            QLabel {
                color: #555555;
                font-size: 16px;
                padding: 40px;
            }
        """)
        self.pages.addWidget(loading_label)

        self.pages.setMinimumWidth(0)
        self.pages.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)

        self.pages_scroll_area = QScrollArea()
        self.pages_scroll_area.setWidgetResizable(True)
        self.pages_scroll_area.setFrameShape(QFrame.NoFrame)
        self.pages_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.pages_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.pages_scroll_area.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self.pages_scroll_area.setWidget(self.pages)

        main_layout.addWidget(self.pages_scroll_area, 1)

        self.setCentralWidget(container)
        self.build_tabs()
        self.resize_to_available_screen()
        self.sync_pages_width_to_window()
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.sync_pages_width_to_window()

    def sync_pages_width_to_window(self):
        if not hasattr(self, "pages") or not hasattr(self, "pages_scroll_area"):
            return

        central_widget = self.centralWidget()
        if central_widget is None:
            return

        available_width = max(
            0,
            central_widget.width()
            - self.main_layout_left_margin
            - self.main_layout_right_margin,
        )
        if available_width <= 0:
            return

        available_height = max(0, self.pages_scroll_area.viewport().height())
        self.pages_scroll_area.setFixedWidth(available_width)
        self.pages.setFixedWidth(available_width)
        if available_height > 0:
            self.pages.setFixedHeight(available_height)

        current_page = self.pages.currentWidget()
        if current_page is not None:
            current_page.setFixedWidth(available_width)
            if available_height > 0:
                current_page.setFixedHeight(available_height)

    def resize_to_available_screen(self):
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(1300, 700)
            return

        geometry = screen.availableGeometry()

        # On Windows, force a real maximized window so LRPhoton opens in
        # fullscreen/maximized mode like a normal desktop app.
        if sys.platform.startswith("win"):
            self.setMinimumSize(900, 620)
            self.setGeometry(geometry)
            self.showMaximized()
            QTimer.singleShot(0, self.sync_pages_width_to_window)
            QTimer.singleShot(200, self.sync_pages_width_to_window)
            return

        # macOS fullscreen/maximized behaviour is cleaner with the exact
        # available geometry.
        if sys.platform == "darwin":
            self.setMinimumSize(900, 620)
            self.setGeometry(geometry)
            self.showMaximized()
            QTimer.singleShot(0, self.sync_pages_width_to_window)
            return

        width = min(1300, max(900, geometry.width() - 80))
        height = min(760, max(620, geometry.height() - 80))
        self.resize(width, height)
        self.move(
            geometry.x() + max(0, (geometry.width() - width) // 2),
            geometry.y() + max(0, (geometry.height() - height) // 2),
        )
        self.show()
        QTimer.singleShot(0, self.sync_pages_width_to_window)

    def resolve_background_tab_class(self):
        import tabs.background_tab as background_tab_module

        background_tab = getattr(background_tab_module, "BackgroundTab", None)
        if background_tab is None:
            for candidate_name in dir(background_tab_module):
                candidate = getattr(background_tab_module, candidate_name)
                if (
                    isinstance(candidate, type)
                    and issubclass(candidate, QWidget)
                    and candidate is not QWidget
                    and getattr(candidate, "__module__", "") == background_tab_module.__name__
                ):
                    background_tab = candidate
                    break

        if background_tab is None:
            raise ImportError("No QWidget tab class found in tabs/background_tab.py")
        return background_tab

    def build_tabs(self):
        from tabs.view_tab import ViewTab
        from tabs.centre_tab import CentreTab
        from tabs.cave_tab import CaveTab
        from tabs.average_tab import AverageTab
        from tabs.radial_tab import RadialTab
        from tabs.azimuthal_tab import AzimuthalTab
        from tabs.unfold_tab import UnfoldTab
        from tabs.hermans_tab import HermansTab
        from tabs.datplot_tab import DatPlotTab
        from tabs.tools_tab import ToolsTab
        from tabs.sandbox_tab import SandboxTab

        BackgroundTab = self.resolve_background_tab_class()

        self.view_tab = ViewTab()
        self.datplot_tab = DatPlotTab()
        self.tools_tab = ToolsTab()

        self.centre_tab = CentreTab()
        self.background_tab = BackgroundTab()
        self.cave_tab = CaveTab()
        self.average_tab = AverageTab()
        self.radial_tab = RadialTab()
        self.azimuthal_tab = AzimuthalTab()
        self.unfold_tab = UnfoldTab()
        self.hermans_tab = HermansTab()
        self.sandbox_tab = SandboxTab()
        self.view_tab.set_q_geometry_source_tab(self.azimuthal_tab)
        self.unfold_tab.set_q_geometry_source_tab(self.azimuthal_tab)

        while self.pages.count():
            old_widget = self.pages.widget(0)
            self.pages.removeWidget(old_widget)
            old_widget.deleteLater()

        self.pages.addWidget(self.view_tab)
        self.pages.addWidget(self.datplot_tab)
        self.pages.addWidget(self.tools_tab)
        self.pages.addWidget(self.centre_tab)
        self.pages.addWidget(self.background_tab)
        self.pages.addWidget(self.average_tab)
        self.pages.addWidget(self.cave_tab)
        self.pages.addWidget(self.unfold_tab)
        self.pages.addWidget(self.radial_tab)
        self.pages.addWidget(self.azimuthal_tab)
        self.pages.addWidget(self.hermans_tab)
        self.pages.addWidget(self.sandbox_tab)

        # Folder synchronisation between tabs using a folder browser.
        # When one tab changes folder, all the others are updated too.
        self.folder_synced_tabs = [
            self.view_tab,
            self.datplot_tab,
            self.tools_tab,
            self.centre_tab,
            self.background_tab,
            self.average_tab,
            self.cave_tab,
            self.radial_tab,
            self.azimuthal_tab,
            self.unfold_tab,
            self.hermans_tab,
            self.sandbox_tab,
        ]

        default_folder = str(self.load_last_folder())
        for tab in self.folder_synced_tabs:
            if hasattr(tab, "set_folder_from_external_tab"):
                tab.set_folder_from_external_tab(default_folder)

        for source_tab in self.folder_synced_tabs:
            source_tab.folder_changed.connect(self.save_last_folder)
            for target_tab in self.folder_synced_tabs:
                if source_tab is target_tab:
                    continue
                source_tab.folder_changed.connect(target_tab.set_folder_from_external_tab)

        for index in range(self.pages.count()):
            page = self.pages.widget(index)
            if page is not None:
                page.setMinimumWidth(0)
                page.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)

        self.tab_bar.currentChanged.connect(self.pages.setCurrentIndex)
        self.tab_bar.currentChanged.connect(lambda _index: self.sync_pages_width_to_window())
        self.pages.setCurrentIndex(self.tab_bar.currentIndex())
        self.sync_pages_width_to_window()

    def load_last_folder(self):
        try:
            data = json.loads(USER_SETTINGS_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return Path.home()

        folder = Path(str(data.get("last_folder", ""))).expanduser()
        if folder.exists() and folder.is_dir():
            return folder
        return Path.home()

    def save_last_folder(self, folder):
        folder = Path(folder).expanduser()
        if not folder.exists() or not folder.is_dir():
            return

        try:
            USER_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            if USER_SETTINGS_FILE.exists():
                try:
                    data = json.loads(USER_SETTINGS_FILE.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    data = {}
            data["last_folder"] = str(folder.resolve())
            USER_SETTINGS_FILE.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        except OSError:
            pass

    def get_application_build(self):
        if self.is_development_copy():
            return "development"

        local_commit = self.get_local_commit().strip()
        if local_commit:
            return local_commit[:7]

        return "unknown"

    def get_build_name(self):
        return f"{APP_NAME} {APP_VERSION} - {self.get_application_build()}"

    def get_build_datetime(self):
        app_dir = Path(__file__).resolve().parent

        if self.is_development_copy():
            try:
                result = subprocess.run(
                    ["git", "log", "-1", "--format=%cI"],
                    cwd=str(app_dir),
                    check=True,
                    capture_output=True,
                    text=True,
                )
                commit_date = result.stdout.strip()
                if commit_date:
                    parsed_date = datetime.fromisoformat(commit_date.replace("Z", "+00:00"))
                    return parsed_date.astimezone().strftime("%d/%m/%Y %H:%M")
            except Exception:
                pass

        candidates = [
            path
            for path in app_dir.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix.lower() in {".py", ".png", ".svg", ".ico", ".icns", ".md", ".txt", ".bat", ".command"}
        ]
        if not candidates:
            return "unknown"

        latest_mtime = max(path.stat().st_mtime for path in candidates)
        return datetime.fromtimestamp(latest_mtime).strftime("%d/%m/%Y %H:%M")

    def add_about_logo(self, layout, image_path, label_text):
        logo_label = QLabel()
        logo_label.setAlignment(Qt.AlignCenter)
        logo_label.setMinimumSize(110, 72)

        pixmap = QPixmap(str(image_path))
        if not pixmap.isNull():
            logo_label.setPixmap(
                pixmap.scaled(
                    120,
                    72,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )
        else:
            logo_label.setText(label_text)
            logo_label.setStyleSheet("""
                QLabel {
                    color: #777777;
                    border: 1px dashed #cccccc;
                    border-radius: 8px;
                    padding: 12px;
                    background: #f7f7f7;
                }
            """)

        layout.addWidget(logo_label)

    def open_about_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle(f"About {APP_NAME}")
        dialog.setModal(True)
        dialog.setMinimumWidth(420)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)

        logos_layout = QHBoxLayout()
        logos_layout.setSpacing(18)
        logos_layout.addStretch(1)
        assets_dir = Path(__file__).resolve().parent / "assets"
        self.add_about_logo(logos_layout, assets_dir / "LRPhoton.png", APP_NAME)
        self.add_about_logo(logos_layout, assets_dir / "CNRS.png", "CNRS.png")
        logos_layout.addStretch(1)
        layout.addLayout(logos_layout)

        title_label = QLabel(APP_NAME)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                font-size: 22px;
                font-weight: 700;
                color: #222222;
            }
        """)
        layout.addWidget(title_label)

        if self.is_development_copy():
            build_info_html = f"Build: {self.get_build_name()}"
        else:
            build_info_html = (
                f"Build: {self.get_build_name()}<br>"
                f"Last build: {self.get_build_datetime()}"
            )

        info_label = QLabel(
            f"<div style='text-align:center;'>"
            f"<b>{APP_AUTHOR}</b><br><br>"
            f"{APP_AFFILIATION}<br><br>"
            f"<a href='{GITHUB_URL}'>{GITHUB_URL}</a><br><br>"
            f"{build_info_html}"
            f"</div>"
        )
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setOpenExternalLinks(True)
        info_label.setTextInteractionFlags(Qt.TextBrowserInteraction)
        info_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                color: #333333;
            }
        """)
        layout.addWidget(info_label)

        if not self.is_development_copy():
            close_button = QPushButton("Close")
            close_button.clicked.connect(dialog.accept)
            close_button.setFixedHeight(30)
            close_button.setCursor(Qt.PointingHandCursor)
            layout.addWidget(close_button)

        dialog.exec()

    def open_issue_report_dialog(self):
        from urllib.parse import quote

        active_tab = self.tab_bar.tabText(self.tab_bar.currentIndex())
        report_payload = (
            "\n\n"
            f"Application build: {self.get_application_build()}\n"
            f"Python: {platform.python_version()}\n"
            f"System: {platform.platform()}\n"
            f"Active tab: {active_tab}\n\n"
        )

        subject = quote("LRPhoton issue report")
        body = quote(report_payload)
        mailto_url = f"mailto:{REPORT_EMAIL}?subject={subject}&body={body}"

        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", mailto_url])

            elif sys.platform.startswith("win"):
                os.startfile(mailto_url)

            else:
                webbrowser.open(mailto_url)

        except Exception:
            webbrowser.open(mailto_url)

    def set_update_button_state(self, state, text=None):
        if text is not None:
            self.version_label.setText(text)

        if state == "available":
            self.version_label.setEnabled(True)
            self.version_label.setStyleSheet("""
                QPushButton {
                    font-size: 11px;
                    color: #856404;
                    padding: 4px 10px;
                    border-radius: 8px;
                    border: 1px solid #ffeeba;
                    background: #fff3cd;
                }
                QPushButton:disabled {
                    color: #856404;
                    background: #fff3cd;
                    border-color: #ffeeba;
                }
            """)
        else:
            self.version_label.setEnabled(False)
            self.version_label.setStyleSheet("""
                QPushButton {
                    font-size: 11px;
                    color: #444444;
                    padding: 4px 10px;
                    border-radius: 8px;
                    border: 1px solid #dddddd;
                    background: #f8f8f8;
                }
                QPushButton:disabled {
                    color: #999999;
                    background: #f2f2f2;
                    border-color: #dddddd;
                }
            """)

    def is_development_copy(self):
        """
        Enable the private development mode only for Nathan's local working copy.
        In development mode, locked tabs are available and GitHub auto-update checks
        are disabled.
        """
        if os.getenv("LRPHOTON_DEVELOPMENT_COPY") in ("1", "true", "True", "TRUE"):
            return True

        app_dir = Path(__file__).resolve().parent
        normalized_path = str(app_dir).replace("\\", "/").lower()

        developer_path_markers = (
            "/users/nathanpiaget/documents/thèse lrp/programmes/lrphoton",
            "/users/nathanpiaget/documents/thèse lrp/programmes/lrphoton",
            "/users/nathanpiaget/documents/these lrp/programmes/lrphoton",
        )

        return any(marker in normalized_path for marker in developer_path_markers)

    def get_app_dir_write_error(self):
        app_dir = Path(__file__).resolve().parent
        probe_path = app_dir / ".lrphoton_write_test"

        try:
            probe_path.write_text("ok", encoding="utf-8")
            probe_path.unlink(missing_ok=True)
            return None
        except OSError as error:
            try:
                probe_path.unlink(missing_ok=True)
            except OSError:
                pass
            return error

    def update_permission_message(self, error=None):
        detail = f"\n\nSystem error: {error}" if error else ""
        return (
            "LRPhoton found an update, but the installation folder is not writable.\n\n"
            "If LRPhoton cannot write to its installation folder, install the update manually:\n"
            f"1. Open {GITHUB_URL}\n"
            "2. Click the green Code button, then Download ZIP.\n"
            "3. Extract the ZIP.\n"
            "4. Open the extracted LRPhoton folder.\n"
            f"5. Copy its contents into {Path(__file__).resolve().parent}, "
            "replacing the existing files.\n\n"
            "If Windows refuses the copy, check that LRPhoton is closed and that "
            "the installation folder is writable."
            f"{detail}"
        )

    def get_local_commit(self):
        if LOCAL_VERSION_FILE.exists():
            return LOCAL_VERSION_FILE.read_text(encoding="utf-8").strip()
        return ""

    def save_local_commit(self, commit_sha):
        if self.is_development_copy():
            return

        LOCAL_VERSION_FILE.write_text(str(commit_sha).strip(), encoding="utf-8")

    def format_github_commit_date(self, data):
        commit_date = (
            data.get("commit", {})
            .get("committer", {})
            .get("date", "")
        )

        if not commit_date:
            return ""

        try:
            parsed_date = datetime.fromisoformat(commit_date.replace("Z", "+00:00"))
            local_date = parsed_date.astimezone()
            return local_date.strftime("%d/%m/%Y %H:%M")
        except Exception:
            return ""

    def current_sources_match_github(self):
        """
        Avoid false update alerts when LRPhoton was freshly downloaded as a GitHub ZIP,
        or when only the local .lrphoton_commit marker is stale.
        """
        app_dir = Path(__file__).resolve().parent

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            zip_path = temporary_path / "lrphoton_compare.zip"

            response = requests.get(SOURCE_ZIP_URL, timeout=20)
            response.raise_for_status()
            zip_path.write_bytes(response.content)

            with zipfile.ZipFile(zip_path, "r") as archive:
                archive.extractall(temporary_path)

            extracted_roots = [path for path in temporary_path.iterdir() if path.is_dir()]
            if not extracted_roots:
                return False

            source_root = extracted_roots[0]

            allowed_extensions = {
                ".py",
                ".json",
                ".txt",
                ".md",
                ".svg",
                ".png",
                ".jpg",
                ".jpeg",
                ".ico",
                ".icns",
                ".command",
                ".bat",
            }

            ignored_dirs = {
                ".git",
                "__pycache__",
                ".venv",
                "venv",
                "dist",
                "build",
            }

            ignored_files = {
                ".lrphoton_commit",
                "LRPhoton.bat",
                "Lancer LRPhoton.bat",
            }

            remote_files = {
                path.relative_to(source_root)
                for path in source_root.rglob("*")
                if path.is_file()
                and path.name not in ignored_files
                and path.suffix.lower() in allowed_extensions
                and not any(part in ignored_dirs for part in path.relative_to(source_root).parts)
            }

            for relative_name in remote_files:
                local_file = app_dir / relative_name
                remote_file = source_root / relative_name

                if not local_file.exists():
                    return False

                if local_file.read_bytes() != remote_file.read_bytes():
                    return False

        return True

    def install_update_from_github(self, remote_sha):
        app_dir = Path(__file__).resolve().parent
        write_error = self.get_app_dir_write_error()
        if write_error is not None:
            raise PermissionError(self.update_permission_message(write_error))

        allowed_extensions = {
            ".py",
            ".pyw",
            ".json",
            ".txt",
            ".md",
            ".svg",
            ".png",
            ".jpg",
            ".jpeg",
            ".ico",
            ".icns",
            ".command",
            ".bat",
        }

        ignored_dirs = {
            ".git",
            "__pycache__",
            ".venv",
            "venv",
            "dist",
            "build",
        }

        ignored_files = {
            ".lrphoton_commit",
            "LRPhoton.bat",
            "Lancer LRPhoton.bat",
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            zip_path = temporary_path / "lrphoton_update.zip"

            response = requests.get(SOURCE_ZIP_URL, timeout=20)
            response.raise_for_status()
            zip_path.write_bytes(response.content)

            with zipfile.ZipFile(zip_path, "r") as archive:
                archive.extractall(temporary_path)

            extracted_roots = [path for path in temporary_path.iterdir() if path.is_dir()]
            if not extracted_roots:
                raise RuntimeError("Downloaded update archive is empty.")

            source_root = extracted_roots[0]

            for source_path in source_root.rglob("*"):
                relative_path = source_path.relative_to(source_root)

                if any(part in ignored_dirs for part in relative_path.parts):
                    continue

                if source_path.name in ignored_files:
                    continue

                destination_path = app_dir / relative_path

                if source_path.is_dir():
                    destination_path.mkdir(parents=True, exist_ok=True)
                    continue

                if source_path.suffix.lower() not in allowed_extensions:
                    continue

                destination_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, destination_path)

        self.save_local_commit(remote_sha)

    def can_check_for_updates(self):
        app_dir = Path(__file__).resolve().parent
        return app_dir.exists() and not self.is_development_copy()

    def check_for_updates(self, silent=False):
        try:
            if not self.can_check_for_updates():
                self.version_label.setVisible(False)
                self.version_label.setEnabled(False)
                self.available_update_sha = None
                return
            response = requests.get(UPDATE_INFO_URL, timeout=5)
            response.raise_for_status()
            data = response.json()

            remote_sha = str(data.get("sha", "")).strip()
            if not remote_sha:
                self.set_update_button_state("disabled", "Update status unavailable")
                self.available_update_sha = None
                return

            up_to_date_text = "Up to date"
            local_sha = self.get_local_commit()

            if local_sha == remote_sha:
                self.version_label.setVisible(False)
                self.version_label.setEnabled(False)
                self.available_update_sha = None
                return

            if self.current_sources_match_github():
                self.save_local_commit(remote_sha)
                self.version_label.setVisible(False)
                self.version_label.setEnabled(False)
                self.available_update_sha = None
                return

            short_sha = remote_sha[:7]
            self.available_update_sha = remote_sha
            write_error = self.get_app_dir_write_error()
            if write_error is not None:
                self.set_update_button_state("available", "Update needs write access")
                if not silent and not self.silent_update_test:
                    QMessageBox.warning(
                        self,
                        "Update needs write access",
                        self.update_permission_message(write_error),
                    )
                return

            if silent or self.silent_update_test:
                self.set_update_button_state("disabled", "Updating…")
                QApplication.processEvents()
                try:
                    self.install_update_from_github(remote_sha)
                    self.set_update_button_state("disabled", "Update installed")
                except Exception:
                    self.set_update_button_state("disabled", "Update failed")
                self.available_update_sha = None
                return

            self.set_update_button_state("available", f"Update available")

            box = QMessageBox(self)
            box.setWindowTitle("Update available")
            box.setIcon(QMessageBox.Information)
            box.setText("A new version of LRPhoton is available.")
            box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            box.button(QMessageBox.Yes).setText("Install")
            box.button(QMessageBox.No).setText("Not Now")
            box.setDefaultButton(QMessageBox.Yes)

            result = box.exec()
            if result == QMessageBox.Yes:
                self.set_update_button_state("disabled", "Updating…")
                QApplication.processEvents()
                try:
                    self.install_update_from_github(remote_sha)
                    self.set_update_button_state("disabled", "Update installed")
                    self.available_update_sha = None
                    self.relaunch_application()
                    return
                except Exception:
                    self.set_update_button_state("disabled", "Update failed")
                    self.available_update_sha = None
            return

        except Exception as error:
            self.set_update_button_state("disabled", "Update status unavailable")
            self.available_update_sha = None
            QMessageBox.warning(
                self,
                "Update error",
                f"Impossible to check or install the update:\n{error}"
            )

    def on_update_button_clicked(self):
        if not self.available_update_sha:
            return

        write_error = self.get_app_dir_write_error()
        if write_error is not None:
            QMessageBox.warning(
                self,
                "Update needs write access",
                self.update_permission_message(write_error),
            )
            self.set_update_button_state("available", "Update needs write access")
            return

        self.set_update_button_state("disabled", "Updating…")
        QApplication.processEvents()

        try:
            self.install_update_from_github(self.available_update_sha)
            self.set_update_button_state("disabled", "Update installed")
            self.available_update_sha = None
            self.relaunch_application()
        except Exception as error:
            self.set_update_button_state("disabled", "Update failed")
            QMessageBox.warning(
                self,
                "Update error",
                f"Impossible to install the update:\n{error}"
            )
        except Exception as error:
            self.set_update_button_state("disabled", "Update failed")
            QMessageBox.warning(
                self,
                "Update error",
                f"Impossible to install the update:\n{error}"
            )

    def relaunch_application(self):
        app_dir = Path(__file__).resolve().parent
        try:
            if sys.platform == "darwin":
                app_launcher = app_dir.parent / "LRPhoton.app"
                command_launcher = app_dir / "LRPhoton.command"
                if app_launcher.exists():
                    subprocess.Popen(["open", str(app_launcher)])
                elif command_launcher.exists():
                    subprocess.Popen(["open", str(command_launcher)])
                else:
                    subprocess.Popen([sys.executable, str(app_dir / "main.py")], cwd=str(app_dir))

            elif sys.platform.startswith("win"):
                launcher = app_dir / "Lancer LRPhoton.bat"
                if launcher.exists():
                    subprocess.Popen([str(launcher)], cwd=str(app_dir), shell=True)
                else:
                    subprocess.Popen([sys.executable, str(app_dir / "main.py")], cwd=str(app_dir))

            else:
                subprocess.Popen([sys.executable, str(app_dir / "main.py")], cwd=str(app_dir))
        except Exception:
            pass

        self.close()
        QApplication.instance().quit()


def main():
    app = QApplication(sys.argv)
    app.setWindowIcon(make_application_icon())

    app.setStyleSheet("""
        QFrame {
            background: transparent;
        }

        QStackedWidget {
            background: transparent;
        }

        QTabBar {
            background: transparent;
        }

        QTabWidget::pane {
            border: none;
            background: transparent;
        }

        QGroupBox {
            background-color: #eeeeee;
            border: 1px solid #d8d8d8;
            border-radius: 10px;
            margin-top: 14px;
            padding: 4px;
            font-family: Arial;
            font-size: 12px;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 8px;
            padding: 0px 4px;
            background-color: transparent;
            color: #222222;
            font-family: Arial;
            font-size: 12px;
        }

        QTabBar::tab {
            padding: 7px 12px;
            margin-right: 2px;
            border-radius: 8px;
            border: none;
            background: #eeeeee;
            color: #222222;
            font-size: 13px;
        }

        QTabBar::tab:disabled {
            background: #f5f5f5;
            color: #9a9a9a;
        }

        QTabBar::tab:selected {
            background: #007aff;
            color: white;
        }

        QTabBar::tab:hover:!selected {
            background: #dddddd;
        }
    """)

    window = MainWindow()
    if "--force-update-check" in sys.argv:
        QTimer.singleShot(400, lambda: window.check_for_updates(silent=False))
    elif window.can_check_for_updates():
        QTimer.singleShot(1200, lambda: window.check_for_updates(silent=window.silent_update_test))
    else:
        window.version_label.setVisible(False)
        window.version_label.setEnabled(False)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
