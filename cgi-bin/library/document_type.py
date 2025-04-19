#!/usr/bin/env python3
import json

# Create a list of document types
document_types = [
    {"value": "All", "name": "All Types"},
    {"value": "tidecalendar", "name": "Tide Calendar"},
    {"value": "waveclimate", "name": "Wave Climate Report"}
]

# Set the content type to JSON
print("Content-Type: application/json")
print()

# Output the JSON
print(json.dumps({
    "document_types": document_types
}, indent=2))