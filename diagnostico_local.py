import os
import glob
import requests
import json
from colorama import Fore, Style, init

init(autoreset=True)

OLLAMA_URL = "http://127.0.0.1:11435/api/generate"
OLLAMA_MODEL = "qwen2.5-coder:14b"

def get_last_n_lines(file_path, n=50):
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            # Ignorar lineas en blanco
            lines = [l for l in lines if l.strip()]
            return "".join(lines[-n:])
    except Exception as e:
        return f"Error leyendo {file_path}: {e}"

def analyze_log_with_ollama(log_name, log_content):
    if not log_content.strip():
        return None
        
    print(f"{Fore.CYAN}[DIAGNOSTICO IA] Consultando a {OLLAMA_MODEL}...")
    
    prompt = f"""
Actúa como un ingeniero DevOps/SRE experto en Python y C++.
Se ha producido el siguiente error o Stack Trace en el archivo de log correspondiente a '{log_name}':

```text
{log_content}
```

Tu tarea es:
1. Identificar la causa raíz exacta del fallo.
2. Proporcionar la solución concreta (línea de código, comando a ejecutar o dependencia a instalar).
3. Ser directo y conciso. Responde en español.
"""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2
        }
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json().get("response", "Sin respuesta.")
        else:
            return f"Error HTTP de Ollama: {response.status_code}"
    except Exception as e:
        return f"Error conectando con Ollama: {e}"

def main():
    print(f"{Fore.MAGENTA}=============================================")
    print(f"{Fore.MAGENTA}   DIAGNOSTICO LOCAL POR IA (OLLAMA SRE)     ")
    print(f"{Fore.MAGENTA}=============================================")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(script_dir, "logs")
    
    if not os.path.exists(logs_dir):
        print(f"{Fore.YELLOW}No existe la carpeta logs/. No hay errores recientes que analizar.")
        return

    log_files = glob.glob(os.path.join(logs_dir, "*_error.log"))
    
    if not log_files:
        print(f"{Fore.GREEN}No se encontraron archivos de error. El sistema parece estable.")
        return
        
    found_errors = False
    for log_file in log_files:
        size = os.path.getsize(log_file)
        if size == 0:
            continue
            
        found_errors = True
        base_name = os.path.basename(log_file)
        print(f"{Fore.YELLOW}\n--- Detectado log con contenido: {base_name} ({size} bytes) ---")
        
        content = get_last_n_lines(log_file, n=50)
        analysis = analyze_log_with_ollama(base_name, content)
        
        if analysis:
            print(f"\n{Fore.GREEN}[REPORTE SRE - {base_name}]:")
            print(f"{Style.RESET_ALL}{analysis}")
            print(f"{Fore.CYAN}" + "-"*50)
            
    if not found_errors:
        print(f"{Fore.GREEN}Los archivos de log estan vacios (0 errores). El sistema es estable.")

if __name__ == "__main__":
    main()
