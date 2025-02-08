from http.server import HTTPServer, SimpleHTTPRequestHandler
import os
import webbrowser
import logging
from functools import partial

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CORSHTTPRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.path.dirname(os.path.abspath(__file__)), **kwargs)

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.send_header(
            'Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()


def run_server(port=8080):
    """Run the website server on the specified port."""
    server_address = ('', port)

    try:
        handler = partial(CORSHTTPRequestHandler)
        httpd = HTTPServer(server_address, handler)
        logger.info(f"Server started at http://localhost:{port}")

        # Open the website in the default browser
        webbrowser.open(f"http://localhost:{port}")

        # Start the server
        httpd.serve_forever()
    except Exception as e:
        logger.error(f"Error running server: {str(e)}")
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        httpd.server_close()


if __name__ == "__main__":
    run_server()
