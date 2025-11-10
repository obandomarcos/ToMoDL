@echo off
:: ================================================
:: PyTorch Installer for Napari – Auto-Detect ANY version
:: Works with hidden folders, spaces, multiple installs
:: ================================================

setlocal EnableDelayedExpansion
set "PYTORCH_VERSION=2.5.1"
set "CHANNELS=-c pytorch -c nvidia -c conda-forge --override-channels"

echo.
echo ========================================
echo  PyTorch Installer for Napari (Auto-Detect)
echo ========================================
echo.

rem -------------------------------------------------
rem 1. Resolve %LOCALAPPDATA% (expand % variables)
rem -------------------------------------------------
set "APPDATA_ROOT=%LOCALAPPDATA%"
if "%APPDATA_ROOT:~-1%"=="\" set "APPDATA_ROOT=%APPDATA_ROOT:~0,-1%"

echo Scanning for Napari folder in:
echo   %APPDATA_ROOT%
echo.

rem -------------------------------------------------
rem 2. Use DIR /B /AD to list *all* napari-* folders
rem -------------------------------------------------
set "NAPARI_ENV="
set "CONDA_EXE="

for /f "delims=" %%F in ('dir "%APPDATA_ROOT%\napari-*" /b /ad 2^>nul') do (
    set "FULLDIR=%APPDATA_ROOT%\%%F"
    set "TEST_EXE=!FULLDIR!\envs\%%F\Scripts\conda.exe"
    echo   [CHECK] "!FULLDIR!"
    if exist "!TEST_EXE!" (
        set "NAPARI_ENV=!FULLDIR!"
        set "CONDA_EXE=!TEST_EXE!"
        echo   [FOUND] Napari environment: %%F
        goto :FOUND_ENV
    ) else (
        echo   [MISS] conda.exe not present in this folder
    )
)

rem -------------------------------------------------
rem No napari folder at all
rem -------------------------------------------------
echo.
echo [ERROR] No Napari installation found!
echo Expected folder pattern: napari-x.x.x  (e.g. napari-0.6.4)
echo.
echo Make sure Napari was installed with the official Windows installer.
pause
exit /b 1


:FOUND_ENV
echo.
echo [OK] Using Napari environment: %NAPARI_ENV%
echo      Conda executable: %CONDA_EXE%
echo.

rem -------------------------------------------------
rem 3. NVIDIA / CUDA detection
rem -------------------------------------------------
echo Checking for NVIDIA GPU and CUDA version...

set "HAS_NVIDIA="
for /f "tokens=*" %%i in ('nvidia-smi --query-gpu^=driver_version --format^=csv,noheader 2^>nul') do set "HAS_NVIDIA=1"

if not defined HAS_NVIDIA (
    echo [INFO] No NVIDIA GPU detected. Using CPU-only PyTorch.
    goto :CPU_INSTALL
)

rem ----- get CUDA version (FIXED BLOCK) -----
set "CUDA_VERSION="
for /f "tokens=9 delims= " %%A in ('nvidia-smi ^| findstr /C:"CUDA Version"') do set "CUDA_VERSION=%%A"

if not defined CUDA_VERSION (
    echo [WARN] nvidia-smi found but no CUDA version detected -> CPU fallback.
    goto :CPU_INSTALL
)

echo [OK] Detected CUDA Version: %CUDA_VERSION%

rem ----- choose highest supported pytorch-cuda -----
set "SELECTED_CUDA="
if %CUDA_VERSION% GEQ 12.4 set "SELECTED_CUDA=12.4"
if %CUDA_VERSION% GEQ 12.1 if not defined SELECTED_CUDA set "SELECTED_CUDA=12.1"
if %CUDA_VERSION% GEQ 11.8 if not defined SELECTED_CUDA set "SELECTED_CUDA=11.8"

if not defined SELECTED_CUDA (
    echo [INFO] CUDA %CUDA_VERSION% is too old for GPU PyTorch -> CPU install.
    goto :CPU_INSTALL
)

echo [OK] Selected pytorch-cuda=%SELECTED_CUDA%
goto :GPU_INSTALL


rem ================================================
:GPU_INSTALL
echo.
echo Installing PyTorch %PYTORCH_VERSION% with CUDA %SELECTED_CUDA% ...
call "%CONDA_EXE%" install -y pytorch==%PYTORCH_VERSION% pytorch-cuda=%SELECTED_CUDA% %CHANNELS%
if errorlevel 1 (
    echo.
    echo [ERROR] GPU install failed -> falling back to CPU.
    goto :CPU_INSTALL
)
echo.
echo [SUCCESS] PyTorch + CUDA %SELECTED_CUDA% installed!
goto :END


rem ================================================
:CPU_INSTALL
echo.
echo Installing CPU-only PyTorch %PYTORCH_VERSION% ...
call "%CONDA_EXE%" install -y pytorch==%PYTORCH_VERSION% cpuonly %CHANNELS%
if errorlevel 1 (
    echo.
    echo [ERROR] CPU-only install failed!
    pause
    exit /b 1
)
echo.
echo [SUCCESS] CPU-only PyTorch installed!
goto :END


rem ================================================
:END
echo.
echo ========================================
echo Installation complete!
echo Napari folder : %NAPARI_ENV%
echo PyTorch ready for use.
echo ========================================
echo.
pause
exit /b 0