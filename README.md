# LRPhoton

## Overview

LRPhoton is a Python application for SAXS/WAXS data processing and visualization dedicated to laboratory and synchrotron scattering data, including XENOCS laboratory SAXS systems and ESRF beamlines ID13 and ID02.

## Windows installation

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

* Administrator rights are not required.
* If Python has just been installed, relaunch `Install on Windows.bat` once after the Python installation finishes.

The installed application folder is usually:

```text
C:\Users\<your-user-name>\AppData\Local\Programs\LRPhoton
```

If the installation does not work, install Python manually and move the extracted LRPhoton folder into the directory shown above. You can create a shortcut on your Desktop.

## MacOS Installation

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

On Apple Silicon Macs, install the macOS universal2/arm64 Python build. If an Intel-only Python is used, macOS may show a Rosetta compatibility warning.

`LRPhoton.app` is only a small launcher that opens:

```text
/Applications/LRPhoton/main.py
```

The real LRPhoton files stay in `/Applications/LRPhoton`, so the automatic update system can still replace the Python files in that folder.

## Update System

At startup, LRPhoton automatically checks for updates from the GitHub repository and downloads modified internal files when a newer version is available.
