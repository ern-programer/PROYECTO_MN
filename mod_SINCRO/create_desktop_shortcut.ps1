# Script para crear/actualizar acceso directo de GammaSync en el escritorio
# Ejecutar como: .\create_desktop_shortcut.ps1

$shortcutPath = "$env:USERPROFILE\Desktop\GammaSync.lnk"
$pythonPath = "C:\Users\Ernesto\AppData\Local\Programs\Python\Python313\python.exe"
$scriptPath = "D:\- PROGRAMACIÓN\PROYECTO_MN\mod_SINCRO\main.py"
$workingDir = "D:\- PROGRAMACIÓN\PROYECTO_MN\mod_SINCRO"
$iconPath = "D:\- PROGRAMACIÓN\PROYECTO_MN\mod_SINCRO\assets\logo_gammasync_ern_02.ico"

# Verificar que Python existe
if (-not (Test-Path $pythonPath)) {
    Write-Host "ERROR: Python no encontrado en $pythonPath" -ForegroundColor Red
    Write-Host "Buscando Python..."
    
    # Buscar Python en el sistema
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd) {
        $pythonPath = $pythonCmd.Source
        Write-Host "Python encontrado en: $pythonPath" -ForegroundColor Green
    } else {
        Write-Host "No se encontró Python en el sistema. Instalá Python 3.13 primero." -ForegroundColor Red
        exit 1
    }
}

# Verificar que el script existe
if (-not (Test-Path $scriptPath)) {
    Write-Host "ERROR: Script no encontrado en $scriptPath" -ForegroundColor Red
    exit 1
}

# Verificar PyQt6
Write-Host "Verificando PyQt6..." -ForegroundColor Cyan
$pyqtCheck = & $pythonPath -c "import PyQt6; print('OK')" 2>&1
if ($pyqtCheck -ne "OK") {
    Write-Host "ADVERTENCIA: PyQt6 no está instalado en el Python del sistema." -ForegroundColor Yellow
    Write-Host "Instalando PyQt6..." -ForegroundColor Cyan
    & $pythonPath -m pip install PyQt6
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: No se pudo instalar PyQt6. Instalalo manualmente:" -ForegroundColor Red
        Write-Host "  $pythonPath -m pip install PyQt6" -ForegroundColor Yellow
        exit 1
    }
    Write-Host "PyQt6 instalado correctamente." -ForegroundColor Green
} else {
    Write-Host "PyQt6 verificado OK." -ForegroundColor Green
}

# Crear acceso directo
Write-Host "Creando acceso directo en el escritorio..." -ForegroundColor Cyan

$WshShell = New-Object -comObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut($shortcutPath)
$Shortcut.TargetPath = $pythonPath
$Shortcut.Arguments = "`"$scriptPath`""
$Shortcut.WorkingDirectory = $workingDir
$Shortcut.Description = "GammaSync - Análisis de disincronía cardíaca SPECT"

# Usar icono si existe
if (Test-Path $iconPath) {
    $Shortcut.IconLocation = $iconPath
    Write-Host "Icono configurado: $iconPath" -ForegroundColor Green
} else {
    Write-Host "ADVERTENCIA: Icono no encontrado en $iconPath" -ForegroundColor Yellow
}

$Shortcut.Save()

if (Test-Path $shortcutPath) {
    Write-Host "✅ Acceso directo creado exitosamente:" -ForegroundColor Green
    Write-Host "   $shortcutPath" -ForegroundColor White
    Write-Host ""
    Write-Host "Para usar GammaSync:" -ForegroundColor Cyan
    Write-Host "   1. Doble click en 'GammaSync' en el escritorio" -ForegroundColor White
    Write-Host "   2. O ejecutá: python `"$scriptPath`"" -ForegroundColor White
} else {
    Write-Host "❌ ERROR: No se pudo crear el acceso directo" -ForegroundColor Red
    exit 1
}
