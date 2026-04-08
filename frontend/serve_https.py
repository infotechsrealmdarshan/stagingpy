"""
Simple HTTPS static file server using existing cert.pem + key.pem.
Serves all files in the current directory over HTTPS on port 8443.
Access from any device on the same Wi-Fi via https://<your-ip>:8443
"""
import http.server
import ssl
import socket
import os
from datetime import datetime

PORT = 8443
CERT = os.path.join(os.path.dirname(__file__), "cert.pem")
KEY  = os.path.join(os.path.dirname(__file__), "key.pem")

# Get local IP
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

os.chdir(os.path.dirname(os.path.abspath(__file__)))


class LoggingHandler(http.server.SimpleHTTPRequestHandler):
    """Prints a clean log line for every request."""

    def log_message(self, format, *args):
        ts     = datetime.now().strftime('%H:%M:%S')
        client = self.client_address[0]
        # args = (method+path, status_code, size)
        request_line = args[0] if args else '-'
        status       = args[1] if len(args) > 1 else '-'
        size         = args[2] if len(args) > 2 else '-'

        # Pick an icon based on status code
        try:
            code = int(status)
            icon = '✅' if code < 300 else ('↪️ ' if code < 400 else '❌')
        except (ValueError, TypeError):
            icon = '📋'

        print(f"[HTTPS] {icon}  [{ts}]  {client}  {status}  {request_line}  ({size} bytes)")

    def log_error(self, format, *args):
        ts     = datetime.now().strftime('%H:%M:%S')
        client = self.client_address[0]
        print(f"[HTTPS] ⚠️   [{ts}]  {client}  ERROR: {format % args}")


handler = LoggingHandler
handler.extensions_map.update({
    ".js":   "application/javascript",
    ".mjs":  "application/javascript",
    ".html": "text/html",
    ".css":  "text/css",
    ".json": "application/json",
    ".webp": "image/webp",
    ".jpg":  "image/jpeg",
    ".png":  "image/png",
})

httpd = http.server.HTTPServer(("0.0.0.0", PORT), handler)

ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ctx.load_cert_chain(certfile=CERT, keyfile=KEY)
httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)

ip = get_local_ip()
print(f"\n{'='*52}")
print(f"  🔒 HTTPS Frontend Server running!")
print(f"  Local:    https://localhost:{PORT}")
print(f"  Network:  https://{ip}:{PORT}")
print(f"{'='*52}\n")
print("  Open on phone → Accept cert warning → Use Network URL")
print("  Watching for requests...\n")

httpd.serve_forever()

