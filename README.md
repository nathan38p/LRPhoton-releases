# LRPhoton

## 🔎 Overview

<p align="center">
  <img src="assets/LRPhoton.png" alt="LRPhoton" height="140">
</p>

LRPhoton is a Python application for SAXS/WAXS data processing and visualization dedicated to laboratory and synchrotron scattering data, including XENOCS laboratory SAXS systems and ESRF beamlines ID13 and ID02.

## 🪟 Windows installation

1. Click the green Code button, then select Download ZIP.
2. Extract the ZIP archive.
3. Open the extracted folder.
4. Double click `Install on Windows.bat`.

The installer will:

* install Python automatically if it is missing
* install all required Python dependencies
* copy the application into `%LOCALAPPDATA%\Programs\LRPhoton`
* create a desktop shortcut automatically

Important:

* If Python has just been installed, relaunch `Install on Windows.bat` once after the Python installation finishes.

The installed application folder is usually:

```text
C:\Users\<your-user-name>\AppData\Local\Programs\LRPhoton
```

* If the installation does not work, install Python manually and move the extracted LRPhoton folder into the directory shown above. You can create a shortcut on your Desktop.

## 🍎 MacOS Installation

1. Install Python 3.14.5 from: https://www.python.org/downloads/
2. Click the green Code button, then select Download ZIP.
3. Extract the ZIP archive.
4. Open the extracted LRPhoton folder.
5. Double click `Install on MacOS.command`.
6. Launch LRPhoton using the `LRPhoton.app` application created in /Applications.

The macOS installer will:

* copy the LRPhoton folder into `/Applications/LRPhoton`
* install all required Python dependencies
* create `/Applications/LRPhoton.app`
* apply the LRPhoton icon to the application launcher

`LRPhoton.app` is only a small launcher that opens:

```text
/Applications/LRPhoton/main.py
```

The LRPhoton files stay in `/Applications/LRPhoton`.

## 🛠️ Update system

Development is ongoing, new features, improvements and bug fixes are added regularly.

At startup, LRPhoton automatically checks for updates from the GitHub repository and downloads modified internal files when a newer version is available.

## 🧰 Troubleshooting

If LRPhoton does not open, closes immediately, or prints an error such as `ModuleNotFoundError: No module named ...`, install the Python dependencies manually from a terminal.

The required Python modules are listed in:

```text
requirements.txt
```

They are currently:

```text
PySide6
numpy
matplotlib
h5py
fabio
requests
pyserial
hdf5plugin
scipy
pyFAI
```

### Install all Python modules

First update `pip`, then install the requirements.

On Windows:

```text
python -m ensurepip --upgrade
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS:

```text
python3 -m ensurepip --upgrade
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

If you are not running the command from the LRPhoton folder, use the full path to `requirements.txt`.

### Manual one-line install

If needed, the same dependencies can be installed directly.

On Windows:

```text
python -m pip install PySide6 numpy matplotlib h5py fabio requests pyserial hdf5plugin scipy pyFAI
```

On macOS:

```text
python3 -m pip install PySide6 numpy matplotlib h5py fabio requests pyserial hdf5plugin scipy pyFAI
```

### Vimba / SALS camera module

The SALS camera tab also needs Allied Vision Vimba support. LRPhoton ships a local wheel for `vmbpy` in:

```text
assets/wheels/
```

If the camera tab reports that `vmbpy` is missing, install it manually from the LRPhoton folder.

On Windows:

```text
python -m pip install assets/wheels/vmbpy-1.0.4-py3-none-any.whl
```

On macOS:

```text
python3 -m pip install assets/wheels/vmbpy-1.0.4-py3-none-any.whl
```

For real camera acquisition, Allied Vision Vimba X must also be installed on the computer, and no other program should already have full access to the camera.

### Test the installation

From the LRPhoton folder:

On Windows:

```text
python main.py
```

On macOS:

```text
python3 main.py
```


## ℹ️ Credits

<p align="center">
  <a href="https://lrp.univ-grenoble-alpes.fr/"><img src="assets/LRP.svg" alt="LRP" height="44"></a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://www.cnrs.fr/"><img src="assets/CNRS.png" alt="CNRS" height="44"></a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://www.univ-grenoble-alpes.fr/"><img src="assets/UGA.png" alt="Université Grenoble Alpes" height="44"></a>
</p>

LRPhoton was developped during Nathan Piaget's PhD at the Laboratoire Rhéologie et Procédés (LRP, UMR 5520, CNRS / Université Grenoble Alpes).

If LRPhoton contributes to results presented in a scientific publication, please cite the software.
