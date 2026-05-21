@echo off
title LRPhoton Installer
setlocal

echo ==========================================
echo          INSTALLATION LRPhoton
echo ==========================================
echo.

echo Installation de Python Install Manager...
winget install -e --id Python.PythonInstallManager

echo.

:: =========================================================
:: PYTHON 3.14 x64 DEDIE A LRPhoton
:: =========================================================

set "PYTHON_DIR=%LOCALAPPDATA%\Programs\LRPhotonPython314x64"
set "PYTHON_EXE=%PYTHON_DIR%\python.exe"
set "PYTHON_INSTALLER=%TEMP%\python-3.14.5-amd64.exe"

echo Verification de Python 3.14 x64 dedie a LRPhoton...

if not exist "%PYTHON_EXE%" (
    echo.
    echo Python 3.14 x64 dedie non detecte.
    echo Telechargement de Python 3.14 x64 depuis python.org...

    powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.14.5/python-3.14.5-amd64.exe' -OutFile '%PYTHON_INSTALLER%'"

    echo.
    echo Installation de Python 3.14 x64 dans :
    echo %PYTHON_DIR%

    "%PYTHON_INSTALLER%" /quiet InstallAllUsers=0 TargetDir="%PYTHON_DIR%" PrependPath=0 Include_launcher=0 Include_pip=1 Include_test=0
)

if not exist "%PYTHON_EXE%" (
    echo.
    echo ERREUR : Python x64 n'a pas ete installe correctement.
    pause
    exit /b 1
)

echo.
echo Verification architecture Python...
"%PYTHON_EXE%" -c "import platform, sys; print(platform.machine()); sys.exit(0 if platform.machine().lower() in ('amd64','x86_64') else 1)"

if errorlevel 1 (
    echo.
    echo ERREUR : le Python detecte n'est pas x64/AMD64.
    pause
    exit /b 1
)

echo Python 3.14 x64 detecte.
echo.

set "SOURCE=%~dp0"
set "DEST=C:\Program Files\LRPhoton"

echo Creation du dossier...
mkdir "%DEST%" >nul 2>&1

echo.
echo Copie des fichiers...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Copy-Item -Path '%SOURCE%*' -Destination '%DEST%' -Recurse -Force"

echo.
echo Installation des dependances...

"%PYTHON_EXE%" -m pip install --upgrade pip

"%PYTHON_EXE%" -m pip install ^
PySide6 ^
numpy ^
matplotlib ^
h5py ^
fabio ^
requests ^
hdf5plugin

if errorlevel 1 (
    echo.
    echo ERREUR : certaines dependances n'ont pas pu etre installees.
    pause
    exit /b 1
)

echo.
echo Creation du lanceur bureau...

(
echo @echo off
echo cd /d "C:\Program Files\LRPhoton"
echo "%PYTHON_EXE%" main.py
) > "%USERPROFILE%\Desktop\LRPhoton.bat"

echo.
echo Creation icone...

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
"$WshShell = New-Object -ComObject WScript.Shell; ^
$Shortcut = $WshShell.CreateShortcut('%USERPROFILE%\Desktop\LRPhoton.lnk'); ^
$Shortcut.TargetPath = '%USERPROFILE%\Desktop\LRPhoton.bat'; ^
$Shortcut.WorkingDirectory = 'C:\Program Files\LRPhoton'; ^
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
echo Python dedie :
echo %PYTHON_EXE%
echo.
echo Raccourci cree sur le bureau :
echo LRPhoton
echo.

pause