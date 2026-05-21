@echo off
title LRPhoton Installer
setlocal

echo ==========================================
echo          INSTALLATION LRPhoton
echo ==========================================
echo.

set "PYTHON_EXE=python"

echo Installation / update of Python 3.14.5 x64...
winget install --id Python.Python.3.14 -v 3.14.5 --architecture x64 -e

if errorlevel 1 (
    echo.
    echo ERROR: Python 3.14.5 x64 could not be installed with winget.
    echo Install Python 3.14.5 x64 manually, then relaunch Install.bat.
    echo.
    echo Press any key to close Install.bat...
    pause
    exit /b 1
)

echo.
echo Checking Python...

%PYTHON_EXE% --version >nul 2>&1

if errorlevel 1 (
    echo.
    echo Refreshing environment variables...

    set "PATH=%PATH%;%LocalAppData%\Microsoft\WindowsApps"

    %PYTHON_EXE% --version >nul 2>&1

    if errorlevel 1 (
        echo.
        echo ERROR: Python is still not detected after installation.
        echo Restart Windows then relaunch Install.bat.
        echo.
        echo Press any key to close and relaunch Install.bat...
        pause
        exit /b 1
    )
)

echo Python detected.
echo.

echo Python version:
%PYTHON_EXE% --version

echo.
echo Python architecture:
%PYTHON_EXE% -c "import platform; print(platform.machine())"

echo.

set "SOURCE=%~dp0"
set "DEST=C:\Program Files\LRPhoton"

echo Creating installation folder...
mkdir "%DEST%" >nul 2>&1

echo.
echo Copying files...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Copy-Item -Path '%SOURCE%*' -Destination '%DEST%' -Recurse -Force"

if errorlevel 1 (
    echo.
    echo ERROR: File copy failed.
    echo Make sure Install.bat is running as administrator.
    echo.
    echo Press any key to close Install.bat...
    pause
    exit /b 1
)

echo.
echo Installing dependencies...

%PYTHON_EXE% -m pip install --upgrade pip

%PYTHON_EXE% -m pip install ^
PySide6 ^
numpy ^
matplotlib ^
h5py ^
requests ^
hdf5plugin ^
fabio

if errorlevel 1 (
    echo.
    echo ERROR: Some dependencies could not be installed.
    echo.
    echo Press any key to close Install.bat...
    pause
    exit /b 1
)

echo.
echo Creating desktop launcher...

(
echo @echo off
echo cd /d "C:\Program Files\LRPhoton"
echo python main.py
) > "%USERPROFILE%\Desktop\LRPhoton.bat"

if errorlevel 1 (
    echo.
    echo ERROR: Could not create LRPhoton.bat on the desktop.
    echo.
    echo Press any key to close Install.bat...
    pause
    exit /b 1
)

echo.
echo Creating desktop shortcut icon...

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
"$WshShell = New-Object -ComObject WScript.Shell; ^
$Shortcut = $WshShell.CreateShortcut('%USERPROFILE%\Desktop\LRPhoton.lnk'); ^
$Shortcut.TargetPath = '%USERPROFILE%\Desktop\LRPhoton.bat'; ^
$Shortcut.WorkingDirectory = 'C:\Program Files\LRPhoton'; ^
$Shortcut.IconLocation = 'C:\Program Files\LRPhoton\assets\LRPhoton.ico'; ^
$Shortcut.Save()"

echo.
echo ==========================================
echo Installation complete
echo ==========================================
echo.
echo Software:
echo C:\Program Files\LRPhoton
echo.
echo Desktop shortcut created:
echo LRPhoton
echo.
echo Press any key to close Install.bat...
pause
