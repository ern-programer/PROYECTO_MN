@echo off
setlocal
cd /d "%~dp0"

title GammaSync - Inicio
set "PYTHON=C:\Users\Ernesto\AppData\Local\Programs\Python\Python313\python.exe"
set "APP=%~dp0main.py"
set "LOG=%~dp0inicio_gammasync_error.log"

if not exist "%PYTHON%" (
    echo ERROR: No se encontro Python en:
    echo %PYTHON%
    echo.
    pause
    exit /b 2
)

if not exist "%APP%" (
    echo ERROR: No se encontro GammaSync en:
    echo %APP%
    echo.
    pause
    exit /b 3
)

"%PYTHON%" "%APP%"
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo [%DATE% %TIME%] GammaSync termino con codigo %EXIT_CODE%.>>"%LOG%"
    echo.
    echo GammaSync no pudo iniciarse o termino con un error.
    echo Codigo: %EXIT_CODE%
    echo Registro: %LOG%
    echo.
    pause
)

exit /b %EXIT_CODE%
