#!/usr/bin/python3

import json

# Create a dictionary of the countries
countries = {
    "American_Samoa": "American Samoa",
    "Cook_Islands": "Cook Islands",
    "Fiji": "Fiji",
    "French_Polynesia": "French Polynesia",
    "FSM": "Federated States of Micronesia",
    "Guam": "Guam",
    "Kiribati": "Kiribati",
    "Marshall_Islands": "Marshall Islands",
    "Nauru": "Nauru",
    "Niue": "Niue",
    "Northern_Mariana": "Northern Mariana",
    "Noumea": "Noumea",
    "Palau": "Palau",
    "PNG": "Papua New Guinea",
    "Solomon_Islands": "Solomon Islands",
    "Samoa": "Samoa",
    "Tokelau": "Tokelau",
    "Tonga": "Tonga",
    "Tuvalu": "Tuvalu",
    "Vanuatu": "Vanuatu",
    "Wallis": "Wallis and Futuna"
}

# Convert to list of dictionaries if you prefer that format
countries_list = [{"value": k, "name": v} for k, v in countries.items()]

# Set the content type to JSON
print("Content-Type: application/json")
print()

# Output the JSON
print(json.dumps({
    "countries_list": countries_list
}, indent=2))