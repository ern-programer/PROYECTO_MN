"""Crea acceso directo de GammaSync en el escritorio usando Python."""
import os
import sys
import win32com.client

def create_shortcut():
    # Rutas
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    shortcut_path = os.path.join(desktop, "GammaSync.lnk")
    
    # Python path
    python_path = sys.executable
    print(f"Python: {python_path}")
    
    # Project paths
    script_path = r"D:\- PROGRAMACIÓN\PROYECTO_MN\mod_SINCRO\main.py"
    working_dir = r"D:\- PROGRAMACIÓN\PROYECTO_MN\mod_SINCRO"
    icon_path = r"D:\- PROGRAMACIÓN\PROYECTO_MN\mod_SINCRO\assets\logo_gammasync_ern_02.ico"
    
    # Verificar que el script existe
    if not os.path.exists(script_path):
        print(f"ERROR: Script no encontrado: {script_path}")
        return False
    
    # Verificar PyQt6
    print("Verificando PyQt6...")
    try:
        import PyQt6
        print("PyQt6 OK")
    except ImportError:
        print("ERROR: PyQt6 no instalado. Instalando...")
        import subprocess
        result = subprocess.run([python_path, "-m", "pip", "install", "PyQt6"], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ERROR instalando PyQt6: {result.stderr}")
            return False
        print("PyQt6 instalado")
    
    # Crear acceso directo
    print("Creando acceso directo...")
    shell = win32com.client.Dispatch("WScript.Shell")
    shortcut = shell.CreateShortCut(shortcut_path)
    shortcut.TargetPath = python_path
    shortcut.Arguments = f'"{script_path}"'
    shortcut.WorkingDirectory = working_dir
    shortcut.Description = "GammaSync - Analisis de disincronia cardiaca"
    
    if os.path.exists(icon_path):
        shortcut.IconLocation = icon_path
        print(f"Icono: {icon_path}")
    
    shortcut.save()
    
    if os.path.exists(shortcut_path):
        print(f"Acceso directo creado: {shortcut_path}")
        return True
    else:
        print("ERROR: No se pudo crear acceso directo")
        return False

if __name__ == "__main__":
    try:
        import win32com.client
    except ImportError:
        print("ERROR: pywin32 no instalado. Instalando...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "pywin32"], check=True)
        import win32com.client
    
    success = create_shortcut()
    sys.exit(0 if success else 1)
