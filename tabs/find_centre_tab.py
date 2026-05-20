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
    QGridLayout
)


class FindCentreTab(QWidget):
    """Onglet Find Centre : recherche/ajustement futur du centre du faisceau."""

    def __init__(self):
        super().__init__()
        self.current_file = None
        self.build_ui()

    def build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        title = QLabel("Find Centre")
        title.setStyleSheet("font-size: 22px; font-weight: bold;")
        subtitle = QLabel("Chargement d'une image, affichage et ajustement du centre.")
        subtitle.setStyleSheet("color: #555;")
        main_layout.addWidget(title)
        main_layout.addWidget(subtitle)

        controls = QHBoxLayout()
        self.open_button = QPushButton("Open image")
        self.open_button.clicked.connect(self.open_file)
        controls.addWidget(self.open_button)
        controls.addStretch()
        main_layout.addLayout(controls)

        centre_box = QGroupBox("Centre coordinates")
        centre_layout = QGridLayout(centre_box)

        self.xc_spin = QDoubleSpinBox()
        self.xc_spin.setRange(-100000, 100000)
        self.xc_spin.setDecimals(3)
        self.xc_spin.setValue(0.000)

        self.yc_spin = QDoubleSpinBox()
        self.yc_spin.setRange(-100000, 100000)
        self.yc_spin.setDecimals(3)
        self.yc_spin.setValue(0.000)

        centre_layout.addWidget(QLabel("Center X:"), 0, 0)
        centre_layout.addWidget(self.xc_spin, 0, 1)
        centre_layout.addWidget(QLabel("Center Y:"), 1, 0)
        centre_layout.addWidget(self.yc_spin, 1, 1)

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(QPushButton("←"))
        buttons_layout.addWidget(QPushButton("→"))
        buttons_layout.addWidget(QPushButton("↑"))
        buttons_layout.addWidget(QPushButton("↓"))
        centre_layout.addLayout(buttons_layout, 2, 0, 1, 2)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setPlaceholderText("Affichage image + centre à coder ici.")

        main_layout.addWidget(centre_box)
        main_layout.addWidget(self.log_box, stretch=1)

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open image file",
            "",
            "Data files (*.edf *.h5 *.hdf5);;All files (*)"
        )
        if not file_path:
            return
        self.current_file = file_path
        self.log_box.setPlainText(f"Selected file:\n{file_path}\n\nFind centre logic à ajouter.")
