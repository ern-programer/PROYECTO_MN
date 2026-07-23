# Script simple para crear acceso directo de GammaSync
# Sin caracteres especiales problematicos

$desktop = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktop "GammaSync.lnk"

# Buscar Python
$pythonPath = "C:\Users\Ernesto\AppData\Local\Programs\Python\Python313\python.exe"
if (-not (Test-Path $pythonPath)) {
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd) {
        $pythonPath = $pythonCmd.Source
    } else {
        Write-Host "ERROR: Python no encontrado" -ForegroundColor Red
        exit 1
    }
}

# Rutas del proyecto
$scriptPath = "D:\- PROGRAMACIÓN\PROYECTO_MN\mod_SINCRO\main.py"
$workingDir = "D:\- PROGRAMACIÓN\PROYECTO_MN\mod_SINCRO"
$iconPath = "D:\- PROGRAMACIÓN\PROYECTO_MN\mod_SINCRO\assets\logo_gammasync_ern_02.ico"

# Verificar script
if (-not (Test-Path $scriptPath)) {
    Write-Host "ERROR: Script no encontrado: $scriptPath" -ForegroundColor Red
    exit 1
}

# Verificar PyQt6
Write-Host "Verificando PyQt6..."
$pyqtCheck = & $pythonPath -c "import PyQt6; print('OK')" 2>&1
if ($pyqtCheck -ne "OK") {
    Write-Host "Instalando PyQt6..."
    & $pythonPath -m pip install PyQt6
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: No se pudo instalar PyQt6" -ForegroundColor Red
        exit 1
    }
}
Write-Host "PyQt6 OK" -ForegroundColor Green

# Crear acceso directo
Write-Host "Creando acceso directo..."
$WshShell = New-Object -comObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut($shortcutPath)
$Shortcut.TargetPath = $pythonPath
$Shortcut.Arguments = "`"$scriptPath`""
$Shortcut.WorkingDirectory = $workingDir
$Shortcut.Description = "GammaSync - Analisis de disincronia cardiaca"

if (Test-Path $iconPath) {
    $Shortcut.IconLocation = $iconPath
}

$Shortcut.Save()

if (Test-Path $shortcutPath) {
    Write-Host "Acceso directo creado: $shortcutPath" -ForegroundColor Green
} else {
    Write-Host "ERROR: No se pudo crear acceso directo" -ForegroundColor Red
}
