import asyncio
import aiohttp
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
import os
import time
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import xml.etree.ElementTree as ET
from colorama import Fore, Style, init
init(autoreset=True)

# --- Configuracion ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SENTIMENT_FILE = os.path.join(SCRIPT_DIR, "macro_sentiment.json")
POLLING_INTERVAL = 60 # Segundos entre llamadas. 60s es ideal para no saturar los servidores RSS.

# Fuentes RSS Gratuitas de Alta Confianza
RSS_FEEDS = [
    "https://cointelegraph.com/rss",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cryptoslate.com/feed/",
    "https://www.newsbtc.com/feed/",
    "https://cryptopotato.com/feed/",
    "https://u.today/rss"
]

# Configuracion Ollama Local
OLLAMA_URL = "http://127.0.0.1:11435/api/generate"
OLLAMA_MODEL = "qwen2.5-coder:14b"
print(f"{Fore.CYAN}[SENTIMENT] Usando modelo local de Ollama: {OLLAMA_MODEL} en {OLLAMA_URL}")

def guardar_sentimiento(score):
    """Guarda el score en un archivo JSON para que el bot de trading lo lea."""
    state = {
        "global_sentiment_score": float(score),
        "timestamp": time.time()
    }
    try:
        temp_file = SENTIMENT_FILE + ".tmp"
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(state, f)
        os.replace(temp_file, SENTIMENT_FILE)
    except Exception as e:
        print(f"{Fore.RED}[SENTIMENT ERROR] No se pudo escribir {SENTIMENT_FILE}: {e}")

def obtener_noticias_sync():
    """
    Realiza web scraping sincrono a los feeds RSS de los portales principales.
    """
    titulares = []
    
    # Headers para simular un navegador real y evitar bloqueos basicos anti-bot
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    # Configuracion de Sesion con Retries (Tolerancia a fallos de red)
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1.5, status_forcelist=[ 500, 502, 503, 504 ])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    session.mount('http://', HTTPAdapter(max_retries=retries))

    for url in RSS_FEEDS:
        try:
            # Aumentamos el timeout a 25s ya que Cointelegraph aveces es muy lento
            response = session.get(url, headers=headers, timeout=25)
            if response.status_code == 200:
                content = response.text
                try:
                    root = ET.fromstring(content)
                    # Extraemos los 4 titulares mas recientes de cada fuente
                    for item in root.findall('.//item')[:4]: 
                        title = item.find('title')
                        if title is not None and title.text:
                            titulares.append(title.text.strip())
                except ET.ParseError:
                    print(f"{Fore.YELLOW}[SENTIMENT ERROR] Error parseando XML de {url}")
            else:
                print(f"{Fore.YELLOW}[SENTIMENT] Error HTTP {response.status_code} en {url}")
        except requests.exceptions.Timeout:
            print(f"{Fore.YELLOW}[SENTIMENT WARNING] Timeout de conexion con {url}. Saltando fuente por este ciclo.")
        except requests.exceptions.RequestException as e:
            print(f"{Fore.YELLOW}[SENTIMENT WARNING] Falla temporal de red con {url}: {e}")
        except Exception as e:
            print(f"{Fore.RED}[SENTIMENT ERROR] Fallo critico con {url}: {e}")
            
    return titulares

async def obtener_noticias():
    # Ejecutamos la funcion sincrona en un thread para evitar bloquear el event loop
    return await asyncio.to_thread(obtener_noticias_sync)

def analizar_textos(textos):
    """
    Pasa la lista de titulares por el modelo local de Ollama y extrae un float de sentimiento.
    """
    if not textos:
        return 0.0
    
    # Preparamos el texto a analizar
    titulares_str = "\n".join([f"- {t}" for t in textos])
    
    prompt = f"""
Actúa como un analista cuantitativo de criptomonedas.
Tu objetivo es analizar el sentimiento general del mercado de Solana (SOL) y cripto basándote en los siguientes titulares de noticias recientes.

Titulares:
{titulares_str}

Instrucción estricta:
Responde ÚNICAMENTE con un número decimal entre -1.0 (pánico extremo / mercado muy bajista) y 1.0 (euforia extrema / mercado muy alcista). No incluyas palabras, explicaciones, markdown ni texto adicional. Solo el número.
"""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0, # Determinístico
            "num_predict": 10   # Respuesta corta
        }
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=20)
        if response.status_code == 200:
            data = response.json()
            respuesta_texto = data.get("response", "").strip()
            score = float(respuesta_texto)
            return max(min(score, 1.0), -1.0)
        else:
            print(f"{Fore.YELLOW}[SENTIMENT ERROR] Error HTTP de Ollama: {response.status_code}")
            return 0.0
    except ValueError:
        print(f"{Fore.YELLOW}[SENTIMENT ERROR] Ollama no devolvio un numero valido: '{respuesta_texto}'")
        return 0.0
    except Exception as e:
        print(f"{Fore.RED}[SENTIMENT ERROR] Falla en conexion con Ollama: {e}")
        return 0.0

async def main_loop():
    print(f"{Fore.MAGENTA}=============================================")
    print(f"{Fore.MAGENTA}      MOTOR NLP INICIADO (RSS SCRAPER)       ")
    print(f"{Fore.MAGENTA}=============================================")
    
    # Asegurar que el archivo existe con un valor neutro inicial
    guardar_sentimiento(0.0)
    
    while True:
        try:
            titulares = await obtener_noticias()
            if titulares:
                score = analizar_textos(titulares)
                guardar_sentimiento(score)
                
                color = Fore.GREEN if score > 0.2 else (Fore.RED if score < -0.2 else Fore.YELLOW)
                # Mostramos el puntaje global y un ejemplo aleatorio (el primero de la lista)
                print(f"[NLP] Score: {color}{score:+.2f}{Style.RESET_ALL} | Muestra: {titulares[0][:75]}...")
            
            await asyncio.sleep(POLLING_INTERVAL)
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"{Fore.RED}[SENTIMENT ERROR] Fallo en el loop principal: {e}")
            await asyncio.sleep(5)

if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        print("\n[SENTIMENT] Motor detenido manualmente.")