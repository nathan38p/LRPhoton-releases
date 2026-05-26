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

echo "Creating LRPhoton.app launcher..."
rm -rf "$LAUNCHER_APP"

osacompile -o "$LAUNCHER_APP" \
    -e "on run" \
    -e "do shell script \"cd '$APP_DIR' && PATH=/Library/Frameworks/Python.framework/Versions/3.14/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin nohup /usr/bin/env python3 main.py >> '$LOG_FILE' 2>&1 < /dev/null &\"" \
    -e "quit" \
    -e "end run"

if [ -f "$APP_DIR/assets/LRPhoton.icns" ]; then
    cp "$APP_DIR/assets/LRPhoton.icns" "$LAUNCHER_APP/Contents/Resources/applet.icns"
    /usr/libexec/PlistBuddy -c "Set :CFBundleIconFile applet" "$LAUNCHER_APP/Contents/Info.plist" >/dev/null 2>&1 || true
    touch "$LAUNCHER_APP"
fi

echo "LRPhoton launcher created:"
echo "$LAUNCHER_APP"
echo
echo "You can now launch LRPhoton from /Applications."
echo
read -r -p "Press Enter to close..."
