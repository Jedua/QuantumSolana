import time
from shm_reader import SharedMemoryReader

def main():
    print("Iniciando test de lectura IPC Lock-Free...")
    try:
        reader = SharedMemoryReader("QuantSolana_V10_SHM")
    except Exception as e:
        print(f"Error al conectar con la shared memory: {e}")
        return

    reads_count = 0
    last_print_time = time.time()
    
    while True:
        try:
            state = reader.read_latest_state(max_retries=5000)
            
            # ABI Integrity Validation
            if state.bids[0].qty != 10.5:
                raise RuntimeError(f"FATAL ERROR: Corrupción de memoria (ABI Mismatch). Esperado 10.5, recibido {state.bids[0].qty}")
            
            reads_count += 1
            
            now = time.time()
            if now - last_print_time >= 1.0:
                print(f"[Validacion OK] Lecturas por segundo: {reads_count} | Mid Price: {state.mid_price:.2f} | Last Update ID: {state.last_update_id}")
                reads_count = 0
                last_print_time = now
                
        except RuntimeError as e:
            if "Torn Read" in str(e):
                # El C++ está escribiendo en bucle infinito sin sleeps. Es esperado perder ciclos de reloj
                # intentando leer cuando el C++ monopoliza el lock de memoria. Hacemos bypass pasivo.
                pass
            else:
                raise e

if __name__ == "__main__":
    main()
