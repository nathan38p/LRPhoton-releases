#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="/Applications/LRPhoton"
LAUNCHER_APP="/Applications/LRPhoton.app"
LOG_FILE="$HOME/Library/Logs/LRPhoton.log"

if [ ! -f "$SCRIPT_DIR/main.py" ] && [ ! -f "$APP_DIR/main.py" ]; then
    echo "ERROR: main.py was not found."
    echo "Run this script from the extracted LRPhoton folder."
    echo
    read -r -p "Press Enter to close..."
    exit 1
fi

if [ "$SCRIPT_DIR" != "$APP_DIR" ]; then
    echo "Copying LRPhoton into /Applications..."
    mkdir -p "$APP_DIR"
    rsync -a --delete \
        --exclude ".git" \
        --exclude "__pycache__" \
        --exclude ".venv" \
        --exclude "venv" \
        --exclude "build" \
        --exclude "dist" \
        "$SCRIPT_DIR/" "$APP_DIR/"
else
    echo "LRPhoton is already in /Applications."
fi

if [ ! -f "$APP_DIR/main.py" ]; then
    echo "ERROR: $APP_DIR/main.py was not found after copy."
    echo
    read -r -p "Press Enter to close..."
    exit 1
fi

PYTHON_BIN="/Library/Frameworks/Python.framework/Versions/3.14/bin/python3"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="$(command -v python3 || true)"
fi

if [ -z "${PYTHON_BIN:-}" ]; then
    echo "ERROR: python3 was not found."
    echo "Install Python 3.14.5 from https://www.python.org/downloads/ then relaunch this installer."
    echo
    read -r -p "Press Enter to close..."
    exit 1
fi

echo "Installing Python dependencies with $PYTHON_BIN..."
VMBPY_WHEEL="$APP_DIR/assets/wheels/vmbpy-1.0.4-py3-none-any.whl"
if /usr/bin/arch -arm64 "$PYTHON_BIN" -c "import platform; raise SystemExit(platform.machine() != 'arm64')" >/dev/null 2>&1; then
    /usr/bin/arch -arm64 "$PYTHON_BIN" -m pip install --upgrade pip
    /usr/bin/arch -arm64 "$PYTHON_BIN" -m pip install --upgrade \
        PySide6 \
        numpy \
        matplotlib \
        h5py \
        requests \
        pyserial \
        hdf5plugin \
        fabio \
        scipy \
        pyFAI
    if [ -f "$VMBPY_WHEEL" ]; then
        /usr/bin/arch -arm64 "$PYTHON_BIN" -m pip install --upgrade "$VMBPY_WHEEL"
    fi
else
    "$PYTHON_BIN" -m pip install --upgrade pip
    "$PYTHON_BIN" -m pip install --upgrade \
        PySide6 \
        numpy \
        matplotlib \
        h5py \
        requests \
        pyserial \
        hdf5plugin \
        fabio \
        scipy \
        pyFAI
    if [ -f "$VMBPY_WHEEL" ]; then
        "$PYTHON_BIN" -m pip install --upgrade "$VMBPY_WHEEL"
    fi
fi

echo "Creating LRPhoton.app launcher..."
rm -rf "$LAUNCHER_APP"

mkdir -p "$LAUNCHER_APP/Contents/MacOS" "$LAUNCHER_APP/Contents/Resources"

cat > "$LAUNCHER_APP/Contents/Info.plist" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>LRPhoton</string>
    <key>CFBundleIconFile</key>
    <string>LRPhoton</string>
    <key>CFBundleIdentifier</key>
    <string>fr.lrp.lrphoton.launcher</string>
    <key>CFBundleName</key>
    <string>LRPhoton</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.13</string>
    <key>LSUIElement</key>
    <true/>
</dict>
</plist>
EOF

cat > "$LAUNCHER_APP/Contents/MacOS/LRPhoton" <<'EOF'
#!/bin/bash
APP_DIR="/Applications/LRPhoton"
LOG_FILE="$HOME/Library/Logs/LRPhoton.log"

cd "$APP_DIR" || exit 1
export PATH="/Library/Frameworks/Python.framework/Versions/3.14/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

PYTHON_BIN="/Library/Frameworks/Python.framework/Versions/3.14/bin/python3"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="$(command -v python3)"
fi

if [ -z "${PYTHON_BIN:-}" ]; then
    echo "python3 was not found." >> "$LOG_FILE"
    exit 1
fi

echo "Launching LRPhoton with $PYTHON_BIN" >> "$LOG_FILE"

if /usr/bin/arch -arm64 "$PYTHON_BIN" -c "import platform; raise SystemExit(platform.machine() != 'arm64')" >/dev/null 2>&1; then
    /usr/bin/arch -arm64 "$PYTHON_BIN" -c "import platform; print('Python architecture:', platform.machine())" >> "$LOG_FILE" 2>&1 || true
    nohup /usr/bin/arch -arm64 "$PYTHON_BIN" main.py >> "$LOG_FILE" 2>&1 < /dev/null &
else
    "$PYTHON_BIN" -c "import platform; print('Python architecture:', platform.machine())" >> "$LOG_FILE" 2>&1 || true
    nohup "$PYTHON_BIN" main.py >> "$LOG_FILE" 2>&1 < /dev/null &
fi

exit 0
EOF

chmod +x "$LAUNCHER_APP/Contents/MacOS/LRPhoton"

if [ -f "$APP_DIR/assets/LRPhoton.icns" ]; then
    cp "$APP_DIR/assets/LRPhoton.icns" "$LAUNCHER_APP/Contents/Resources/LRPhoton.icns"
    touch "$LAUNCHER_APP"
fi

echo "LRPhoton launcher created:"
echo "$LAUNCHER_APP"
echo
echo "You can now launch LRPhoton from /Applications."
echo
read -r -p "Press Enter to close..."
