import sys
import json
import platform
import webbrowser
import shutil
import tempfile
import zipfile
import subprocess
from datetime import datetime
from pathlib import Path

import requests

from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtCore import Qt, QTimer

from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMessageBox,
    QFrame,
    QHBoxLayout,
    QMainWindow,
    QStackedWidget,
    QTabBar,
    QVBoxLayout,
    QWidget
)

from tabs.view_tab import ViewTab
from tabs.centre_tab import CentreTab
from tabs.cave_tab import CaveTab
from tabs.radial_tab import RadialTab
from tabs.azimuthal_tab import AzimuthalTab
from tabs.hermans_tab import HermansTab
from tabs.datplot_tab import DatPlotTab


APP_NAME = "LRPhoton 1"
APP_VERSION = "1.0.2"
# update test
GITHUB_OWNER = "nathan38p"
GITHUB_REPO = "LRPhoton-releases"
GITHUB_BRANCH = "main"
UPDATE_INFO_URL = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/commits/{GITHUB_BRANCH}"
SOURCE_ZIP_URL = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/archive/refs/heads/{GITHUB_BRANCH}.zip"
LOCAL_VERSION_FILE = Path(__file__).resolve().parent / ".lrphoton_commit"


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
        header_layout.setSpacing(20)

        logo_path = Path(__file__).parent / "assets" / "LRP.svg"

        logo = QSvgWidget(str(logo_path))
        logo.setFixedSize(42, 42)

        title = QLabel(APP_NAME)
        title.setStyleSheet("""
            QLabel {
                font-size: 20px;
                font-weight: 700;
                color: #333333;
            }
        """)

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
        title_box.addWidget(title)
        title_box.addWidget(subtitle)

        header_layout.addWidget(logo)
        header_layout.addLayout(title_box)

        # ============================================================
        # TAB BAR IN HEADER
        # ============================================================

        self.tab_bar = QTabBar()
        self.tab_bar.setExpanding(False)
        self.tab_bar.setMovable(False)
        self.tab_bar.setUsesScrollButtons(True)

        self.tab_bar.addTab("View")
        self.tab_bar.addTab("Plot")
        self.tab_bar.addTab("Centre")
        self.tab_bar.addTab("Cave")
        self.radial_tab_index = self.tab_bar.addTab("Radial")
        self.tab_bar.addTab("Azimuthal")
        self.tab_bar.addTab("Anisotropy")

        header_layout.addStretch()
        header_layout.addWidget(self.tab_bar)
        header_layout.addStretch()

        self.version_label = QLabel("Checking for updates…")
        self.version_label.setStyleSheet("""
            QLabel {
                font-size: 11px;
                color: #777777;
                padding: 4px 8px;
                border-radius: 8px;
                background: #f2f2f2;
            }
        """)
        header_layout.addWidget(self.version_label)

        main_layout.addWidget(header)

        # ============================================================
        # PAGE CONTENT FULL WIDTH
        # ============================================================

        self.pages = QStackedWidget()

        self.view_tab = ViewTab()
        self.datplot_tab = DatPlotTab()
        self.centre_tab = CentreTab()
        self.cave_tab = CaveTab()
        self.radial_tab = RadialTab()
        self.azimuthal_tab = AzimuthalTab()
        self.hermans_tab = HermansTab()
        self.view_tab.set_q_geometry_source_tab(self.azimuthal_tab)

        self.pages.addWidget(self.view_tab)
        self.pages.addWidget(self.datplot_tab)
        self.pages.addWidget(self.centre_tab)
        self.pages.addWidget(self.cave_tab)
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
                "install.bat",
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
            "install.bat",
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

    def check_for_updates(self):
        try:
            if self.is_development_copy():
                self.version_label.setText("Development mode")
                return
            response = requests.get(UPDATE_INFO_URL, timeout=5)
            response.raise_for_status()
            data = response.json()

            remote_sha = str(data.get("sha", "")).strip()
            if not remote_sha:
                self.version_label.setText("Update status unavailable")
                return

            up_to_date_text = "Up to date"

            local_sha = self.get_local_commit()

            if local_sha == remote_sha:
                self.version_label.setText(up_to_date_text)
                return

            # If the source files already match GitHub, only the local marker is stale.
            # This happens easily after manually downloading/extracting a ZIP over an old folder.
            if self.current_sources_match_github():
                self.save_local_commit(remote_sha)
                self.version_label.setText(up_to_date_text)
                return

            short_sha = remote_sha[:7]
            self.version_label.setText(f"Update available: {short_sha}")

            box = QMessageBox(self)
            box.setWindowTitle("Update available")
            box.setIcon(QMessageBox.Information)
            box.setText("A new version of LRPhoton is available on GitHub.")
            box.setInformativeText(
                "LRPhoton can download the updated source files automatically.\n\n"
                "After the update, close LRPhoton and open it again."
            )
            box.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
            box.button(QMessageBox.Yes).setText("Update now")
            box.setDefaultButton(QMessageBox.Yes)

            result = box.exec()
            if result != QMessageBox.Yes:
                return

            self.version_label.setText("Updating…")
            QApplication.processEvents()

            self.install_update_from_github(remote_sha)

            self.version_label.setText(up_to_date_text)

            close_box = QMessageBox(self)
            close_box.setWindowTitle("LRPhoton updated")
            close_box.setIcon(QMessageBox.Information)
            close_box.setText("The update has been installed.")
            close_box.setInformativeText("Click Close to apply the new files.")
            close_box.setStandardButtons(QMessageBox.Ok)
            close_box.button(QMessageBox.Ok).setText("Close")

            close_box.exec()

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

        except Exception as error:
            self.version_label.setText("Update status unavailable")
            QMessageBox.warning(
                self,
                "Update error",
                f"Impossible to check or install the update:\n{error}"
            )


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
    QTimer.singleShot(1200, window.check_for_updates)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
