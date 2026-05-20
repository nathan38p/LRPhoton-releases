from pathlib import Path
import numpy as np

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QTextEdit, QDoubleSpinBox, QGroupBox
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class XenocsCaveTab(QWidget):
    """Onglet autonome XENOCS Cave. Tout le code utile est dans ce fichier."""

    def __init__(self, user_email=""):
        super().__init__()
        self.user_email = user_email
        self.file_path = None
        self.image = None
        self.corrected = None
        self.colorbar = None
        self._build_ui()

    def _build_ui(self):
        main = QHBoxLayout(self)
        plot_box = QGroupBox("Prévisualisation XENOCS")
        plot_layout = QVBoxLayout(plot_box)
        self.fig = Figure(figsize=(7, 6))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        plot_layout.addWidget(self.canvas)

        ctrl_box = QGroupBox("Contrôles")
        ctrl = QVBoxLayout(ctrl_box)
        self.btn_choose = QPushButton("Choisir EDF")
        self.btn_apply = QPushButton("Appliquer cave")
        self.btn_save = QPushButton("Enregistrer _cave.edf")
        self.center_x = self._spin(612.0)
        self.center_y = self._spin(649.0)
        ctrl.addWidget(self.btn_choose)
        ctrl.addWidget(QLabel("Centre X :")); ctrl.addWidget(self.center_x)
        ctrl.addWidget(QLabel("Centre Y :")); ctrl.addWidget(self.center_y)
        ctrl.addWidget(self.btn_apply)
        ctrl.addWidget(self.btn_save)
        self.log = QTextEdit()
        self.log.setPlainText(f"Onglet XENOCS Cave\nUtilisateur : {self.user_email}")
        ctrl.addWidget(self.log, stretch=1)

        main.addWidget(plot_box, stretch=4)
        main.addWidget(ctrl_box, stretch=1)

        self.btn_choose.clicked.connect(self.choose_file)
        self.btn_apply.clicked.connect(self.apply_cave)
        self.btn_save.clicked.connect(self.save_corrected)
        self.refresh()

    def _spin(self, value):
        spin = QDoubleSpinBox()
        spin.setRange(-1_000_000, 1_000_000)
        spin.setDecimals(3)
        spin.setValue(value)
        return spin

    def choose_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choisir EDF", "", "EDF (*.edf);;Tous les fichiers (*)")
        if not path:
            return
        self.file_path = Path(path)
        self.image = self.read_edf_image(self.file_path)
        self.corrected = None
        self.log.setPlainText(f"Fichier chargé :\n{self.file_path}")
        self.refresh()

    def read_edf_image(self, path):
        raw = path.read_bytes()
        header_end = raw.find(b"}\n")
        if header_end == -1: header_end = raw.find(b"}")
        header = raw[:header_end + 1].decode(errors="ignore")
        def get_int(keys, default):
            for key in keys:
                if key in header:
                    part = header.split(key, 1)[1].split(";", 1)[0]
                    digits = "".join(c for c in part if c.isdigit())
                    if digits: return int(digits)
            return default
        dim1 = get_int(["Dim_1", "Dim1"], 1024)
        dim2 = get_int(["Dim_2", "Dim2"], 1024)
        data_start = ((header_end // 512) + 1) * 512
        arr = np.frombuffer(raw[data_start:], dtype=np.float32)
        if arr.size < dim1 * dim2:
            arr = np.frombuffer(raw[data_start:], dtype=np.uint16).astype(float)
        arr = arr[:dim1 * dim2].reshape((dim2, dim1))
        arr[arr > 4e9] = np.nan
        return arr

    def apply_cave(self):
        if self.image is None:
            self.log.append("Aucun fichier chargé.")
            return
        img = self.image.copy()
        h, w = img.shape
        xc = self.center_x.value()
        yc = self.center_y.value()
        mask = ~np.isfinite(img)
        yy, xx = np.where(mask)
        xs = np.rint(2 * xc - xx).astype(int)
        ys = np.rint(2 * yc - yy).astype(int)
        ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        img[yy[ok], xx[ok]] = img[ys[ok], xs[ok]]
        self.corrected = img
        self.log.append("Correction cave appliquée.")
        self.refresh()

    def save_corrected(self):
        if self.corrected is None or self.file_path is None:
            self.log.append("Rien à enregistrer.")
            return
        out = self.file_path.with_name(self.file_path.stem + "_cave.npy")
        np.save(out, self.corrected)
        self.log.append(f"Enregistré provisoirement en NPY :\n{out}")

    def refresh(self):
        self.ax.clear()
        if self.colorbar is not None:
            try: self.colorbar.remove()
            except Exception: pass
            self.colorbar = None
        img = self.corrected if self.corrected is not None else self.image
        if img is None:
            self.ax.set_xlim(0, 1); self.ax.set_ylim(0, 1)
        else:
            data = np.log10(np.where(np.isfinite(img), img, np.nan) + 1)
            im = self.ax.imshow(data, origin="upper", cmap="jet")
            self.colorbar = self.fig.colorbar(im, ax=self.ax)
            self.ax.axvline(self.center_x.value(), color="red", linewidth=1)
            self.ax.axhline(self.center_y.value(), color="red", linewidth=1)
            self.ax.set_title("XENOCS log10(I+1)")
        self.fig.tight_layout()
        self.canvas.draw_idle()
