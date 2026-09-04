# Limpieza de artefactos generados durante pruebas (regenerables).
# Preserva fuente, presets/ (config), data_test/ (DICOMs de entrada),
# docs/ (manuales de referencia) y tests/ (fixtures).
# Uso:
#   .\clean_outputs.ps1            -> muestra que borraria y cuanto libera (dry-run)
#   .\clean_outputs.ps1 -Apply     -> borra de verdad
param([switch]$Apply)

$ErrorActionPreference = 'SilentlyContinue'
Set-Location -Path $PSScriptRoot

# Carpetas de salida de demos/harness de I+D (se borran completas).
$dirs = @(
    'output_demo',
    'output',
    '_nitida_audit_out',
    '_xel3_out',
    '.pytest_cache',
    '.mypy_cache',
    '.ruff_cache'
)

# Archivos sueltos regenerables en el root.
$files = @(
    'crash_error.txt',
    'crash_native.txt',
    'grilla_cine.xlsx',
    'grilla_cine_ern.xlsx'
)

$targets = New-Object System.Collections.Generic.List[System.IO.FileSystemInfo]

foreach ($d in $dirs) {
    if (Test-Path $d) { $targets.Add((Get-Item $d)) }
}
foreach ($f in $files) {
    if (Test-Path $f) { $targets.Add((Get-Item $f)) }
}
# Logs sueltos (se conserva la carpeta logs/).
Get-ChildItem -Path 'logs' -Filter '*.log' -File | ForEach-Object { $targets.Add($_) }
# Caches de bytecode en todo el arbol (menos .venv).
Get-ChildItem -Recurse -Directory -Filter '__pycache__' |
    Where-Object { $_.FullName -notmatch '\\\.venv\\' } |
    ForEach-Object { $targets.Add($_) }

$total = 0
foreach ($t in $targets) {
    if ($t.PSIsContainer) {
        $sz = (Get-ChildItem $t.FullName -Recurse -File | Measure-Object Length -Sum).Sum
    } else {
        $sz = $t.Length
    }
    if (-not $sz) { $sz = 0 }
    $total += $sz
    $rel = $t.FullName -replace [regex]::Escape($PSScriptRoot + '\'), ''
    '{0,9:N1} MB  {1}' -f ($sz / 1MB), $rel
}

Write-Host ''
Write-Host ('Total a liberar: {0:N1} MB' -f ($total / 1MB)) -ForegroundColor Cyan

if ($Apply) {
    foreach ($t in $targets) {
        Remove-Item -LiteralPath $t.FullName -Recurse -Force
    }
    Write-Host 'Listo: artefactos borrados.' -ForegroundColor Green
} else {
    Write-Host 'Dry-run. Volve a correr con -Apply para borrar.' -ForegroundColor Yellow
}
