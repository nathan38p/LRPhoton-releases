@echo off
setlocal

cd /d "%~dp0"

set "DEST=C:\Program Files\LRPhoton"

py -3.13-64 --version >nul 2>nul
if %errorlevel% neq 0 (
    echo.
    echo Python 3.13 x64 non detecte.
    echo Installation automatique de Python 3.13 x64...

    winget install -e --id Python.Python.3.13 --architecture x64

    py -3.13-64 --version >nul 2>nul
    if %errorlevel% neq 0 (
        echo.
        echo ERREUR : Python 3.13 x64 n'a pas pu etre installe automatiquement.
        echo Installez Python 3.13 x64 manuellement depuis :
        echo https://www.python.org/downloads/windows/
        echo Puis relancez Install.bat.
        pause
        exit
    )
)

echo.
echo =====================================
echo        Installation de LRPhoton
echo =====================================
echo.

echo Creation du dossier...
mkdir "%DEST%" 2>nul

echo.
echo Copie des fichiers...
xcopy "%CD%" "%DEST%" /E /I /Y

cd /d "%DEST%"

echo.
echo Installation des dependances Python...
py -3.13-64 -m pip install --upgrade pip
py -3.13-64 -m pip install PySide6 numpy matplotlib h5py fabio requests hdf5plugin

echo.
echo Creation du lanceur LRPhoton.bat...

echo @echo off > "%DEST%\LRPhoton.bat"
echo cd /d "C:\Program Files\LRPhoton" >> "%DEST%\LRPhoton.bat"
echo start "" py -3.13-64 "main.py" >> "%DEST%\LRPhoton.bat"
echo exit >> "%DEST%\LRPhoton.bat"

echo.
echo Creation du raccourci bureau...

powershell -Command ^
"$WshShell = New-Object -ComObject WScript.Shell; ^
$Shortcut = $WshShell.CreateShortcut('%USERPROFILE%\Desktop\LRPhoton.lnk'); ^
$Shortcut.TargetPath = 'C:\Program Files\LRPhoton\LRPhoton.bat'; ^
$Shortcut.WorkingDirectory = 'C:\Program Files\LRPhoton'; ^
$Shortcut.IconLocation = 'C:\Program Files\LRPhoton\assets\LRPhoton.ico'; ^
$Shortcut.Save()"

echo.
echo Lancement de LRPhoton...
start "" py -3.13-64 "main.py"

echo.
echo =====================================
echo     Installation terminee
echo =====================================
echo.
echo Programme installe dans :
echo C:\Program Files\LRPhoton
echo.
echo Un raccourci LRPhoton a ete cree
echo sur le bureau.
echo.

pause