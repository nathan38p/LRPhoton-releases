@echo off
title LRPhoton Installer

echo ==========================================
echo          INSTALLATION LRPhoton
echo ==========================================
echo.

:: =========================================================
:: INSTALLATION PYTHON MANAGER
:: =========================================================

winget install -e --id Python.PythonManager
if errorlevel 1 echo Python Manager non trouve par winget, installation ignoree.

:: =========================================================
:: INSTALLATION PYTHON 3.14 x64 SI ABSENT
:: =========================================================

echo Verification de Python x64...

py -3.14-64 --version >nul 2>&1

if errorlevel 1 (
    echo.
    echo Python 3.14 x64 non detecte.
    echo Installation de Python 3.14 x64...

    winget install -e --id Python.Python.3.14 --architecture x64

    echo.
    echo Python installe.
    echo Relancez Install.bat.
    pause
    exit
)

echo Python 3.14 x64 detecte.
echo.

:: =========================================================
:: DOSSIER INSTALLATION
:: =========================================================

set "SOURCE=%~dp0"
set "DEST=C:\Program Files\LRPhoton"

echo Creation du dossier...
mkdir "%DEST%" >nul 2>&1

echo.
echo Copie des fichiers...

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
"Copy-Item -Path '%SOURCE%*' -Destination '%DEST%' -Recurse -Force"

echo.
echo Installation des dependances...

py -3.14-64 -m pip install --upgrade pip

py -3.14-64 -m pip install ^
PySide6 ^
numpy ^
matplotlib ^
h5py ^
fabio ^
requests ^
hdf5plugin

echo.
echo Creation du raccourci bureau...

(
echo @echo off
echo cd /d "C:\Program Files\LRPhoton"
echo py -3.14-64 main.py
) > "%USERPROFILE%\Desktop\LRPhoton.bat"

echo.
echo Creation icone...

powershell -Command ^
"$WshShell = New-Object -comObject WScript.Shell; ^
$Shortcut = $WshShell.CreateShortcut('%USERPROFILE%\Desktop\LRPhoton.lnk'); ^
$Shortcut.TargetPath = '%USERPROFILE%\Desktop\LRPhoton.bat'; ^
$Shortcut.IconLocation = 'C:\Program Files\LRPhoton\assets\LRPhoton.ico'; ^
$Shortcut.Save()"

echo.
echo ==========================================
echo Installation terminee
echo ==========================================
echo.
echo Logiciel :
echo C:\Program Files\LRPhoton
echo.
echo Raccourci cree sur le bureau :
echo LRPhoton
echo.

pause