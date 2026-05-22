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
from datetime import datetime
from pathlib import Path


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
    QTabBar,
    QVBoxLayout,
    QWidget
)

from PySide6.QtGui import QColor, QPainter, QPixmap

from tabs.view_tab import ViewTab
from tabs.centre_tab import CentreTab
from tabs.cave_tab import CaveTab
from tabs.average_tab import AverageTab
from tabs.radial_tab import RadialTab
from tabs.azimuthal_tab import AzimuthalTab
from tabs.unfold_tab import UnfoldTab
from tabs.hermans_tab import HermansTab
from tabs.datplot_tab import DatPlotTab


APP_NAME = "LRPhoton"
APP_VERSION = "1.0.2"
# Constants
REPORT_EMAIL = "nathan.piaget@univ-grenoble-alpes.fr"
# update test4
GITHUB_OWNER = "nathan38p"
GITHUB_REPO = "LRPhoton-releases"
GITHUB_BRANCH = "main"
UPDATE_INFO_URL = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/commits/{GITHUB_BRANCH}"
SOURCE_ZIP_URL = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/archive/refs/heads/{GITHUB_BRANCH}.zip"
LOCAL_VERSION_FILE = Path(__file__).resolve().parent / ".lrphoton_commit"

class ColoredTabBar(QTabBar):
    TAB_COLORS = {
        0: ("#dbeafe", "#2563eb"),  # View 2D
        1: ("#dbeafe", "#2563eb"),  # Plot 1D
        2: ("#fee2e2", "#dc2626"),  # Center
        3: ("#fee2e2", "#dc2626"),  # Average
        4: ("#fee2e2", "#dc2626"),  # Cave
        5: ("#fee2e2", "#dc2626"),  # Unfold
        6: ("#dcfce7", "#16a34a"),  # Radial
        7: ("#dcfce7", "#16a34a"),  # Azimuthal
        8: ("#f3e8ff", "#9333ea"),  # Anisotropy
    }

    def tabSizeHint(self, index):
        size = super().tabSizeHint(index)
        return QSize(size.width() + 10, size.height() + 2)

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
        self.resize(1300, 700)

        container = QWidget()
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(24, 18, 24, 24)
        main_layout.setSpacing(10)

        # ============================================================
        # HEADER
        # ============================================================

        header = QFrame()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 0, 8, 0)
        header_layout.setSpacing(12)

        logo_path = Path(__file__).parent / "assets" / "LRPhoton.png"

        logo = QLabel()
        logo_pixmap = QPixmap(str(logo_path))
        logo.setPixmap(
            logo_pixmap.scaled(
                42,
                42,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )
        logo.setFixedSize(42, 42)

        title = QLabel(APP_NAME)
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

        # ============================================================
        # TAB BAR IN HEADER
        # ============================================================

        self.tab_bar = ColoredTabBar()
        self.tab_bar.setExpanding(False)
        self.tab_bar.setMovable(False)
        self.tab_bar.setUsesScrollButtons(True)

        self.tab_bar.addTab("View 2D")
        self.tab_bar.addTab("Plot 1D")
        self.tab_bar.addTab("Center")
        self.tab_bar.addTab("Average")
        self.tab_bar.addTab("Cave")
        self.tab_bar.addTab("Unfold")
        self.radial_tab_index = self.tab_bar.addTab("Radial")
        self.tab_bar.addTab("Azimuthal")
        self.tab_bar.addTab("Anisotropy")

        header_layout.addStretch()
        header_layout.addWidget(self.tab_bar)
        header_layout.addStretch()

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

        self.view_tab = ViewTab()
        self.datplot_tab = DatPlotTab()
        self.centre_tab = CentreTab()
        self.cave_tab = CaveTab()
        self.average_tab = AverageTab()
        self.radial_tab = RadialTab()
        self.azimuthal_tab = AzimuthalTab()
        self.unfold_tab = UnfoldTab()
        self.hermans_tab = HermansTab()
        self.view_tab.set_q_geometry_source_tab(self.azimuthal_tab)
        self.unfold_tab.set_q_geometry_source_tab(self.azimuthal_tab)

        self.pages.addWidget(self.view_tab)
        self.pages.addWidget(self.datplot_tab)
        self.pages.addWidget(self.centre_tab)
        self.pages.addWidget(self.average_tab)
        self.pages.addWidget(self.cave_tab)
        self.pages.addWidget(self.unfold_tab)
        self.pages.addWidget(self.radial_tab)
        self.pages.addWidget(self.azimuthal_tab)
        self.pages.addWidget(self.hermans_tab)

        # Folder synchronisation between tabs using a folder browser.
        # When one tab changes folder, all the others are updated too.
        self.folder_synced_tabs = [
            self.view_tab,
            self.datplot_tab,
            self.radial_tab,
            self.azimuthal_tab,
            self.unfold_tab,
            self.hermans_tab,
        ]

        default_folder = str(Path.home())
        for tab in self.folder_synced_tabs:
            if hasattr(tab, "set_folder_from_external_tab"):
                tab.set_folder_from_external_tab(default_folder)

        for source_tab in self.folder_synced_tabs:
            for target_tab in self.folder_synced_tabs:
                if source_tab is target_tab:
                    continue
                source_tab.folder_changed.connect(target_tab.set_folder_from_external_tab)

        self.tab_bar.currentChanged.connect(self.pages.setCurrentIndex)

        main_layout.addWidget(self.pages)

        self.setCentralWidget(container)

    def get_application_build(self):
        if self.is_development_copy():
            return "development"

        local_commit = self.get_local_commit().strip()
        if local_commit:
            return local_commit[:7]

        return "unknown"

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
        Disable GitHub auto-update checks when running from the developer Git repository.
        ZIP downloads do not contain .git, so normal users still get update checks.
        """
        app_dir = Path(__file__).resolve().parent
        return (app_dir / ".git").exists()

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
                launcher = app_dir / "LRPhoton.command"
                if launcher.exists():
                    subprocess.Popen(["open", str(launcher)])
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
            background-color: #f4f4f4;
            border: 0px;
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
            color: #222222;
            font-family: Arial;
            font-size: 12px;
        }

        QTabBar::tab {
            padding: 7px 18px;
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
    window.show()
    if window.can_check_for_updates():
        QTimer.singleShot(1200, lambda: window.check_for_updates(silent=window.silent_update_test))
    else:
        window.version_label.setVisible(False)
        window.version_label.setEnabled(False)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()