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
    echo Install Python 3.14.5 x64 manually, then relaunch Install on Windows.bat.
    echo.
    echo Press any key to close Install on Windows.bat...
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
        echo Restart Windows then relaunch Install on Windows.bat.
        echo.
        echo Press any key to close and relaunch Install on Windows.bat...
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
set "DEST=%LOCALAPPDATA%\Programs\LRPhoton"

echo LRPhoton will be installed for the current user.
echo Administrator rights are not required.

for %%I in ("%SOURCE%.") do set "SOURCE_FULL=%%~fI"
for %%I in ("%DEST%") do set "DEST_FULL=%%~fI"

echo Creating installation folder...
mkdir "%DEST%" >nul 2>&1

echo.
if /I "%SOURCE_FULL%"=="%DEST_FULL%" (
    echo Install on Windows.bat is already running from the installation folder.
    echo Skipping file copy.
) else (
    echo Copying files...
    robocopy "%SOURCE_FULL%" "%DEST_FULL%" /E /XD .git __pycache__ .venv venv build dist /XF .DS_Store /NFL /NDL /NJH /NJS /NP

    if errorlevel 8 (
        echo.
        echo ERROR: File copy failed.
        echo Make sure the destination folder is writable:
        echo %DEST_FULL%
        echo.
        echo You can install LRPhoton manually:
        echo 1. Go to the LRPhoton GitHub page.
        echo 2. Click the green Code button, then Download ZIP.
        echo 3. Extract the ZIP.
        echo 4. Copy the contents of the LRPhoton folder into:
        echo    %DEST_FULL%
        echo.
        echo Press any key to close Install on Windows.bat...
        pause
        exit /b 1
    )
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
pyserial ^
hdf5plugin ^
fabio ^
scipy ^
pyFAI

if exist "%DEST_FULL%\assets\wheels\vmbpy-1.0.4-py3-none-any.whl" (
    %PYTHON_EXE% -m pip install --upgrade "%DEST_FULL%\assets\wheels\vmbpy-1.0.4-py3-none-any.whl"
)

if errorlevel 1 (
    echo.
    echo ERROR: Some dependencies could not be installed.
    echo.
    echo Press any key to close Install on Windows.bat...
    pause
    exit /b 1
)

echo.
echo Creating desktop shortcut...

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
"$WshShell = New-Object -ComObject WScript.Shell; ^
$Shortcut = $WshShell.CreateShortcut('%USERPROFILE%\Desktop\LRPhoton.lnk'); ^
$Shortcut.TargetPath = 'pythonw.exe'; ^
$Shortcut.Arguments = 'main.py'; ^
$Shortcut.WorkingDirectory = '%DEST_FULL%'; ^
$Shortcut.IconLocation = '%DEST_FULL%\assets\LRPhoton.ico'; ^
$Shortcut.Save()"

if errorlevel 1 (
    echo.
    echo ERROR: Could not create desktop shortcut.
    echo.
    echo Press any key to close Install on Windows.bat...
    pause
    exit /b 1
)

echo.
echo ==========================================
echo Installation complete
echo ==========================================
echo.
echo Software:
echo %DEST_FULL%
echo.
echo Desktop shortcut created:
echo LRPhoton
echo.
echo Press any key to close Install on Windows.bat...
pause
