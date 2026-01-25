#!/usr/bin/env python3
"""Minimal server with COOP/COEP headers for SharedArrayBuffer support."""

import http.server
import sys

PORT = 8080

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Redirect /pkg/ to /pkg/wasm_test.js for dynamic imports
        if self.path == '/pkg/' or self.path == '/pkg':
            self.send_response(301)
            self.send_header('Location', '/pkg/wasm_test.js')
            self.end_headers()
            return
        super().do_GET()

    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        super().end_headers()

if __name__ == '__main__':
    print(f'Serving at http://localhost:{PORT}')
    print('Press Ctrl+C to stop')
    try:
        http.server.HTTPServer(('', PORT), Handler).serve_forever()
    except KeyboardInterrupt:
        sys.exit(0)
