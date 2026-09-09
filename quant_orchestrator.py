import subprocess
import time
import sys
import os
import signal
import mmap
from colorama import Fore, Style, init

init(autoreset=True)

SHM_NAME = "QuantSolana_V10_SHM"
SHM_SIZE = 448  # sizeof(SharedMemoryBlock)

def get_exec_env():
    env = dict(os.environ)
    conda_bin = r"C:\Users\pepel\miniconda3\envs\cerebro_v10\Library\bin"
    if os.path.exists(conda_bin):
        env["PATH"] = conda_bin + os.pathsep + env.get("PATH", "")
    return env

def get_log_file(name):
    root_cwd = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(root_cwd, "logs")
    os.makedirs(log_dir, exist_ok=True)
    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
    log_path = os.path.join(log_dir, f"{safe_name}_error.log")
    return open(log_path, "a")

def run_native_binary(binary_path, name):
    print(f"{Fore.CYAN}[ORQUESTADOR] Iniciando binario nativo C++ {name} ({binary_path})...")
    try:
        root_cwd = os.path.dirname(os.path.abspath(__file__))
        log_file = get_log_file(name)
        process = subprocess.Popen([binary_path], cwd=root_cwd, env=get_exec_env(), stderr=log_file)
        print(f"{Fore.GREEN}[ORQUESTADOR] {name} iniciado con PID: {process.pid}. Errores guardados en logs/")
        return process
    except Exception as e:
        print(f"{Fore.RED}[ORQUESTADOR ERROR] No se pudo iniciar {name}: {e}")
        return None

def run_python_script(script_path, name):
    print(f"{Fore.CYAN}[ORQUESTADOR] Iniciando script Python {name} ({script_path})...")
    try:
        root_cwd = os.path.dirname(os.path.abspath(__file__))
        python_exe = sys.executable
        conda_python = r"C:\Users\pepel\miniconda3\envs\cerebro_v10\python.exe"
        if os.path.exists(conda_python):
            python_exe = conda_python
        
        log_file = get_log_file(name)
        process = subprocess.Popen([python_exe, script_path], cwd=root_cwd, env=get_exec_env(), stderr=log_file)
        print(f"{Fore.GREEN}[ORQUESTADOR] {name} iniciado con PID: {process.pid}. Errores guardados en logs/")
        return process
    except Exception as e:
        print(f"{Fore.RED}[ORQUESTADOR ERROR] No se pudo iniciar {name}: {e}")
        return None

def wait_for_shared_memory(shm_name, timeout=30.0):
    print(f"{Fore.YELLOW}[ORQUESTADOR IPC] Esperando barrera de memoria compartida '{shm_name}' (Timeout: {timeout}s)...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            mm = mmap.mmap(-1, SHM_SIZE, tagname=shm_name, access=mmap.ACCESS_WRITE)
            mm.close()
            print(f"{Fore.GREEN}[ORQUESTADOR IPC] Memoria compartida '{shm_name}' detectada y sincronizada.")
            return True
        except Exception:
            time.sleep(0.5)
            
    print(f"{Fore.RED}[ORQUESTADOR ERROR CRITICO] Timeout esperando la memoria compartida '{shm_name}'.")
    return False

def terminate_process(process, name):
    if process and process.poll() is None:
        print(f"{Fore.YELLOW}[ORQUESTADOR] Terminando subproceso {name} (PID: {process.pid})...")
        try:
            process.terminate()
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            print(f"{Fore.RED}[ORQUESTADOR] Forzando kill en {name} (PID: {process.pid})...")
            process.kill()

USE_LIVE_FEED = True  # True: binance_ingestion.exe (WebSocket real), False: mock_writer.exe (Datos sintéticos)

def main():
    print(f"{Fore.MAGENTA}=============================================")
    print(f"{Fore.MAGENTA}   MASTER QUANT ORCHESTRATOR V10 (HIBRIDO)  ")
    print(f"{Fore.MAGENTA}=============================================")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # --- PROMPT DE REINICIO DE ESTADO ---
    ans = input(f"{Fore.YELLOW}¿Deseas empezar desde cero (borrar historial)? (s/n): {Style.RESET_ALL}").strip().lower()
    if ans == 's':
        state_file = os.path.join(BASE_DIR, "sim_state_sol.json")
        log_file = os.path.join(BASE_DIR, "paper_trading_log.txt")
        term_log_file = os.path.join(BASE_DIR, "log_terminal_data.json")
        for f in [state_file, log_file, term_log_file]:
            if os.path.exists(f):
                os.remove(f)
                print(f"{Fore.GREEN}[OK] Eliminado {os.path.basename(f)}")
        print(f"{Fore.GREEN}[SISTEMA] Estado reiniciado con exito.")
    else:
        print(f"{Fore.CYAN}[SISTEMA] Continuando con el estado actual.")
    # -------------------------------------
    
    # Búsqueda del ejecutable C++ compilado según el modo seleccionado
    if USE_LIVE_FEED:
        cpp_candidates = [
            os.path.join(BASE_DIR, "QuantSolana_V10", "build", "binance_ingestion.exe"),
            os.path.join(BASE_DIR, "QuantSolana_V10", "build", "Release", "binance_ingestion.exe"),
            os.path.join(BASE_DIR, "QuantSolana_V10", "build", "binance_ingestion"),
            os.path.join(BASE_DIR, "QuantSolana_V10", "build", "mock_writer.exe"),
        ]
    else:
        cpp_candidates = [
            os.path.join(BASE_DIR, "QuantSolana_V10", "build", "mock_writer.exe"),
            os.path.join(BASE_DIR, "QuantSolana_V10", "build", "Release", "mock_writer.exe"),
            os.path.join(BASE_DIR, "QuantSolana_V10", "build", "binance_ingestion.exe"),
        ]
    
    CPP_BINARY_PATH = None
    for cand in cpp_candidates:
        if os.path.exists(cand):
            CPP_BINARY_PATH = cand
            break
            
    if not CPP_BINARY_PATH:
        target_name = "binance_ingestion.exe" if USE_LIVE_FEED else "mock_writer.exe"
        print(f"{Fore.RED}[ORQUESTADOR ERROR] No se encontro el ejecutable C++ '{target_name}' en QuantSolana_V10/build.")
        print(f"{Fore.YELLOW}[SOLUCION] Compila el proyecto C++ en QuantSolana_V10/build")
        sys.exit(1)

    MODO_PRODUCCION = False
    ans_inverse = input(f"{Fore.YELLOW}¿Deseas ejecutar el MODO TRAMPA INVERSO (Contrarian)? (s/n): {Style.RESET_ALL}").strip().lower()
    MODO_INVERSO = (ans_inverse == 's')

    if MODO_PRODUCCION:
        bot_sol_path = os.path.join(BASE_DIR, "bot_sol", "bot_sol_live.py")
        bot_name_label = "Bot SOL (LIVE)"
    elif MODO_INVERSO:
        bot_sol_path = os.path.join(BASE_DIR, "bot_sol", "bot_sol_inverse.py")
        bot_name_label = "Bot SOL (Trampa Inverso)"
    else:
        bot_sol_path = os.path.join(BASE_DIR, "bot_sol", "bot_sol_paper.py")
        bot_name_label = "Bot SOL (Paper Estándar)"
    
    sentiment_path = os.path.join(BASE_DIR, "cerebro_sentimiento.py")
    cuda_path = os.path.join(BASE_DIR, "cerebro_cuda.py")
    logger_path = os.path.join(BASE_DIR, "python_engine", "tick_logger.py")

    cpp_process = None
    bot_process = None
    sentiment_process = None
    cuda_process = None
    logger_process = None

    def cleanup(signum=None, frame=None):
        print(f"\n{Fore.YELLOW}[ORQUESTADOR] Apagando todos los subprocesos de forma limpia (Graceful Shutdown)...")
        terminate_process(bot_process, bot_name_label)
        terminate_process(cpp_process, "Ingestor C++")
        terminate_process(logger_process, "Tick Logger")
        terminate_process(cuda_process, "Cerebro CUDA")
        terminate_process(sentiment_process, "Cerebro Sentimiento")
        print(f"{Fore.GREEN}[ORQUESTADOR] Apagado completo. Sin procesos huerfanos.")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        # 1. Iniciar subprocesos secundarios (NLP / CUDA / Logger)
        sentiment_process = run_python_script(sentiment_path, "Cerebro Sentimiento (NLP)")
        cuda_process = run_python_script(cuda_path, "Cerebro CUDA (Optimizador)")
        logger_process = run_python_script(logger_path, "Tick Logger (SQLite)")

        # 2. Arranque Secuencial HFT
        cpp_process = run_native_binary(CPP_BINARY_PATH, "Ingestor C++")
        if not cpp_process:
            cleanup()

        # 3. Barrera IPC Shared Memory
        if not wait_for_shared_memory(SHM_NAME, timeout=20.0):
            print(f"{Fore.RED}[ORQUESTADOR ERROR] Falla en barrera IPC. Abortando...")
            cleanup()

        # 4. Iniciar Bot de trading Python
        bot_process = run_python_script(bot_sol_path, bot_name_label)
        if not bot_process:
            cleanup()

        print(f"{Fore.GREEN}[ORQUESTADOR] Todos los procesos iniciados correctamente. Entrando en bucle Watchdog.")

        # 5. Watchdog Activo con Regla de Cascada + Heartbeat SHM
        last_shm_seq = 0
        last_shm_change_time = time.time()
        SHM_HEARTBEAT_TIMEOUT = 30  # segundos sin actualizacion = proceso zombie

        while True:
            # Heartbeat SHM: detectar proceso C++ zombie (vivo pero sin datos)
            try:
                mm = mmap.mmap(-1, SHM_SIZE, tagname=SHM_NAME, access=mmap.ACCESS_READ)
                mm.seek(0)
                current_seq = int.from_bytes(mm.read(8), byteorder='little')
                mm.close()

                if current_seq != last_shm_seq:
                    last_shm_seq = current_seq
                    last_shm_change_time = time.time()
                elif time.time() - last_shm_change_time > SHM_HEARTBEAT_TIMEOUT:
                    if cpp_process.poll() is None:
                        print(f"{Fore.RED}[ORQUESTADOR WATCHDOG] HEARTBEAT MUERTO: SHM congelada {SHM_HEARTBEAT_TIMEOUT}s. C++ zombie detectado.")
                        print(f"{Fore.YELLOW}[ORQUESTADOR WATCHDOG] Matando C++ zombie y Bot Python...")
                        terminate_process(cpp_process, "Ingestor C++")
                        terminate_process(bot_process, bot_name_label)

                        print(f"{Fore.CYAN}[ORQUESTADOR WATCHDOG] Reiniciando Ingestor C++...")
                        cpp_process = run_native_binary(CPP_BINARY_PATH, "Ingestor C++")
                        if not cpp_process:
                            print(f"{Fore.RED}[ORQUESTADOR ERROR] No se pudo reiniciar C++. Abortando...")
                            cleanup()

                        if not wait_for_shared_memory(SHM_NAME, timeout=20.0):
                            print(f"{Fore.RED}[ORQUESTADOR ERROR] Falla en barrera IPC tras reinicio C++. Abortando...")
                            cleanup()

                        print(f"{Fore.CYAN}[ORQUESTADOR WATCHDOG] Reiniciando Bot Python...")
                        bot_process = run_python_script(bot_sol_path, bot_name_label)
                        last_shm_change_time = time.time()
                        last_shm_seq = 0
            except Exception:
                pass

            # Regla de Cascada Critica: Si C++ cae, la Shared Memory se congela
            if cpp_process.poll() is not None:
                print(f"{Fore.RED}[ORQUESTADOR WATCHDOG] ALERTA CRITICA: El Ingestor C++ ha caido!")
                print(f"{Fore.YELLOW}[ORQUESTADOR WATCHDOG] Aplicando regla de cascada: Matando Bot Python...")
                terminate_process(bot_process, bot_name_label)
                
                print(f"{Fore.CYAN}[ORQUESTADOR WATCHDOG] Reiniciando Ingestor C++...")
                cpp_process = run_native_binary(CPP_BINARY_PATH, "Ingestor C++")
                if not cpp_process:
                    print(f"{Fore.RED}[ORQUESTADOR ERROR] No se pudo reiniciar C++. Abortando...")
                    cleanup()
                    
                if not wait_for_shared_memory(SHM_NAME, timeout=20.0):
                    print(f"{Fore.RED}[ORQUESTADOR ERROR] Falla en barrera IPC tras reinicio C++. Abortando...")
                    cleanup()
                    
                print(f"{Fore.CYAN}[ORQUESTADOR WATCHDOG] Reiniciando Bot Python...")
                bot_process = run_python_script(bot_sol_path, bot_name_label)
                last_shm_change_time = time.time()

            # Si el bot de Python cae de forma independiente
            if bot_process and bot_process.poll() is not None:
                print(f"{Fore.RED}[ORQUESTADOR WATCHDOG] Bot Python ha caido. Reiniciando...")
                bot_process = run_python_script(bot_sol_path, bot_name_label)

            # Monitoreo de secundarios (no afectan al trading principal)
            if logger_process and logger_process.poll() is not None:
                print(f"{Fore.YELLOW}[ORQUESTADOR WATCHDOG] Tick Logger ha caido. Reiniciando...")
                logger_process = run_python_script(logger_path, "Tick Logger (SQLite)")

            if sentiment_process and sentiment_process.poll() is not None:
                print(f"{Fore.YELLOW}[ORQUESTADOR WATCHDOG] Cerebro Sentimiento ha caido. Reiniciando...")
                sentiment_process = run_python_script(sentiment_path, "Cerebro Sentimiento (NLP)")

            if cuda_process and cuda_process.poll() is not None:
                print(f"{Fore.YELLOW}[ORQUESTADOR WATCHDOG] Cerebro CUDA ha caido. Reiniciando...")
                cuda_process = run_python_script(cuda_path, "Cerebro CUDA (Optimizador)")

            time.sleep(3)

    except KeyboardInterrupt:
        cleanup()
    except Exception as e:
        print(f"{Fore.RED}[ORQUESTADOR ERROR] Excepcion no controlada: {e}")
        cleanup()

if __name__ == "__main__":
    main()