#!/usr/bin/env python3
import os
import sys
import urllib.parse
import mimetypes

print("Access-Control-Allow-Origin: *")

# Get query string
query = os.environ.get("QUERY_STRING", "")
params = urllib.parse.parse_qs(query)
filename = params.get("file", [None])[0]

# Path validation
if not filename or ".." in filename:
    print("Content-Type: text/plain\n")
    print("Invalid filename.")
    sys.exit(1)

# Root directory (change if needed)
root_dir = os.path.dirname(os.path.realpath(__file__))  # Only allow serving from script dir

# Normalize the full path
safe_path = os.path.normpath(os.path.join(root_dir, filename))

# Ensure it's within the root directory
if not safe_path.startswith(root_dir):
    print("Content-Type: text/plain\n")
    print("Access denied.")
    sys.exit(1)

# Check file exists
if not os.path.isfile(safe_path):
    print("Content-Type: text/plain\n")
    print("File not found.")
    sys.exit(1)

# Get MIME type
mime_type, _ = mimetypes.guess_type(safe_path)
if mime_type is None:
    mime_type = "application/octet-stream"

# Return file
print(f"Content-Type: {mime_type}")
print(f"Content-Disposition: inline; filename=\"{os.path.basename(safe_path)}\"")
print("")  # blank line between headers and content

with open(safe_path, "rb") as f:
    sys.stdout.buffer.write(f.read())
