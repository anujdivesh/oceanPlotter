#!/usr/bin/env python3
import json

# Create a list of year options
years = [
    {"value": "All", "name": "All Years"},
    {"value": "2025", "name": "2025"},
    {"value": "2024", "name": "2024"},
    {"value": "2023", "name": "2023"},
    {"value": "2019", "name": "2019"}
]

# Set the content type to JSON
print("Content-Type: application/json")
print()

# Output the JSON
print(json.dumps({
    "years": years
}, indent=2))