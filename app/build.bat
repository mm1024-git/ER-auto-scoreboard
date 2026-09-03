@echo off
rem ER score tool - build a Windows executable.
rem ASCII only on purpose: cmd.exe reads .bat in the system code page.

setlocal

echo [1/3] create virtual environment
if not exist .venv (
    python -m venv .venv
    if errorlevel 1 goto fail
)
call .venv\Scripts\activate.bat

echo [2/3] install requirements
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 goto fail
python -m pip install pyinstaller
if errorlevel 1 goto fail

echo [3/3] build
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
pyinstaller er_scoreboard.spec
if errorlevel 1 goto fail

rem The build folder holds intermediate files only. Remove it so nobody
rem runs the half-built exe inside it by mistake.
if exist build rmdir /s /q build

echo.
echo Done.
echo Run this one:  dist\ER_score\ER_score.exe
echo Put config.json and digits.npz in that same folder.
echo.
echo If it says "Failed to load Python DLL", you are running the wrong file
echo or your Python came from the Microsoft Store. Store Python does not work
echo with PyInstaller. Install Python from python.org and build again.
pause
exit /b 0

:fail
echo.
echo Build failed.
pause
exit /b 1
