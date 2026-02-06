@echo off
echo Installing requirements...
pip install -r requirements.txt

echo Cleaning up previous builds...
if exist "Release" rmdir /s /q "Release"
mkdir "Release"

echo Building Application...
pyinstaller --onefile --noconsole --name "LAIZER" --icon "LAIZER.ico" --add-data "README.md;." main.py

echo Moving executable to Release folder...
move "dist\LAIZER.exe" "Release\"

echo Checking for Inno Setup Compiler...
set "ISCC=C:\Program Files (x86)\Inno Setup 6\ISCC.exe"

if exist "%ISCC%" (
    echo Compiling Installer...
    "%ISCC%" setup.iss
    echo Installer created in Installers folder.
) else (
    echo Inno Setup not found at "%ISCC%". Skipping installer creation.
    echo Please compile setup.iss manually if you want an installer.
)


echo Build Complete.
echo Executable: Release\LAIZER.exe
if exist "Installers\LAIZER_Setup.exe" echo Installer: Installers\LAIZER_Setup.exe
