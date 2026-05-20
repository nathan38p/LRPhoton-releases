import sys
from pathlib import Path

from PySide6.QtSvgWidgets import QSvgWidget

from PySide6.QtWidgets import (
    QApplication,
    QLabel,
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
        self.tab_bar.addTab("Radial")
        self.tab_bar.addTab("Azimuthal")
        self.tab_bar.addTab("Anisotropy")

        header_layout.addStretch()
        header_layout.addWidget(self.tab_bar)
        header_layout.addStretch()

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

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
