#!/usr/bin/env python3
import os
import sys
import urllib.parse
import mimetypes

def main():
    try:
        # 1. Get filename from query string
        query = os.environ.get("QUERY_STRING", "")
        params = urllib.parse.parse_qs(query)
        filename = params.get("file", [""])[0].strip()

        # 2. Validate filename
        if not filename or ".." in filename or filename.startswith('/'):
            raise ValueError("Invalid filename")

        # 3. Secure file path resolution
        root_dir = os.path.dirname(os.path.abspath(__file__))
        safe_path = os.path.normpath(os.path.join(root_dir, filename))
        
        if not os.path.commonpath([safe_path, root_dir]) == os.path.normpath(root_dir):
            raise ValueError("Access denied")

        # 4. Check if file exists
        if not os.path.isfile(safe_path):
            raise FileNotFoundError("File not found")

        # 5. Get MIME type
        mime_type = mimetypes.guess_type(safe_path)[0] or "application/octet-stream"

        # 6. Send headers - CRITICAL: must complete before any binary data
        sys.stdout.write(f"Content-Type: {mime_type}\n")
        sys.stdout.write("Access-Control-Allow-Origin: *\n")
        sys.stdout.write(f"Content-Length: {os.path.getsize(safe_path)}\n")
        sys.stdout.write(f"Content-Disposition: inline; filename=\"{os.path.basename(safe_path)}\"\n")
        sys.stdout.write("\n")  # REQUIRED blank line after headers
        sys.stdout.flush()  # Ensure headers are sent

        # 7. Send file content in binary mode
        with open(safe_path, "rb") as f:
            while True:
                chunk = f.read(4096)  # Read in chunks
                if not chunk:
                    break
                sys.stdout.buffer.write(chunk)
        sys.stdout.flush()

    except Exception as e:
        sys.stdout.write("Content-Type: text/plain\n\n")
        sys.stdout.write(f"Error: {str(e)}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()