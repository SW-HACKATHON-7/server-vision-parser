
import http.server
import socketserver
import os

PORT = 8080
DIRECTORY = "results"

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.path.join(os.getcwd(), DIRECTORY), **kwargs)

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print("serving at port", PORT)
    print(f"Serving files from the '{DIRECTORY}' directory.")
    print(f"Access it at: http://localhost:{PORT}")
    httpd.serve_forever()
