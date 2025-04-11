import sys
import os
import threading
import webview
import socket
from waitress import serve
import platform # Add platform import

# --- Set Matplotlib Cache Directory --- 
def get_persistent_mpl_configdir():
    app_name = "EcoStatsApp"
    if platform.system() == "Windows":
        path = os.path.join(os.getenv('LOCALAPPDATA', ''), app_name)
    elif platform.system() == "Darwin": # macOS
        path = os.path.join(os.path.expanduser('~'), 'Library', 'Application Support', app_name)
    else: # Linux/Other
        path = os.path.join(os.getenv('XDG_CONFIG_HOME', os.path.join(os.path.expanduser('~'), '.config')), app_name)
    os.makedirs(path, exist_ok=True)
    print(f"--- run_webview.py: Setting MPLCONFIGDIR to: {path} ---")
    return path

if getattr(sys, 'frozen', False):
    os.environ['MPLCONFIGDIR'] = get_persistent_mpl_configdir()
# --- End Set Matplotlib Cache --- 

print("--- run_webview.py: Starting ---")

# Ensure the main app directory is in sys.path if not running bundled
if not getattr(sys, 'frozen', False):
    print("--- run_webview.py: Adding project dir to path ---")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("--- run_webview.py: Importing app ---")
    from app import app, resource_path # Import the Flask app instance and helper
    print("--- run_webview.py: App imported successfully ---")
except ImportError as e:
    print(f"--- run_webview.py: FAILED TO IMPORT APP: {e} ---")
    sys.exit(1) # Exit if app cannot be imported

# Use resource_path from app module now
# STATIC_FOLDER = resource_path('static')

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def run_server(flask_app, port):
    print(f"--- run_webview.py: Starting Waitress on port {port} ---")
    try:
        serve(flask_app, host='127.0.0.1', port=port)
        print("--- run_webview.py: Waitress server finished. ---")
    except Exception as e:
        print(f"--- run_webview.py: Waitress server FAILED: {e} ---")


if __name__ == '__main__':
    print("--- run_webview.py: In __main__ block ---")
    port = find_free_port()
    print(f"--- run_webview.py: Found free port {port} ---")
    
    server_thread = threading.Thread(target=run_server, args=(app, port))
    server_thread.daemon = True
    print("--- run_webview.py: Starting server thread ---")
    server_thread.start()

    # Give the server a moment to start
    # time.sleep(1) # Optional short delay

    window_title = "EcoStats App"
    url = f"http://127.0.0.1:{port}"
    
    print(f"--- run_webview.py: Creating webview window for URL: {url} ---")
    try:
        webview.create_window(window_title, url, width=1024, height=768, resizable=True)
        print("--- run_webview.py: Webview window created ---")
        print("--- run_webview.py: Starting webview event loop ---")
        webview.start(debug=True) # Enable debug for more potential output
        print("--- run_webview.py: Webview event loop finished ---")
    except Exception as e:
         print(f"--- run_webview.py: Webview FAILED: {e} ---")

    print("--- run_webview.py: Script finished ---") 