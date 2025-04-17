#!/usr/bin/env python3
import json
import os
from urllib.parse import parse_qs

# Set the content type to JSON
print("Content-Type: application/json")
print()

try:
    # Get query parameters
    query_string = os.environ.get('QUERY_STRING', '')
    params = parse_qs(query_string)
    
    # Load the JSON data
    with open('doc.json', 'r') as f:
        documents = json.load(f)
    
    # Apply filters if parameters are provided
    filtered_docs = documents.copy()
    
    if 'category' in params:
        category_filter = params['category'][0]
        if category_filter != 'All':
            filtered_docs = [doc for doc in filtered_docs if doc['category'] == category_filter]
    
    if 'document' in params:
        document_filter = params['document'][0]
        if document_filter != 'All':
            filtered_docs = [doc for doc in filtered_docs if doc['document'] == document_filter]
    
    if 'year' in params:
        year_filter = params['year'][0]
        if year_filter != 'All':
            filtered_docs = [doc for doc in filtered_docs if doc['year'] == year_filter]
    else:
        # Sort by year (descending) only when no year filter is applied
        filtered_docs.sort(key=lambda x: x['year'], reverse=True)
    
    # Return the filtered results
    print(json.dumps({
        "status": "success",
        "count": len(filtered_docs),
        "documents": filtered_docs
    }, indent=2))

except FileNotFoundError:
    print(json.dumps({
        "status": "error",
        "message": "doc.json file not found"
    }))
except json.JSONDecodeError:
    print(json.dumps({
        "status": "error",
        "message": "Invalid JSON in doc.json"
    }))
except Exception as e:
    print(json.dumps({
        "status": "error",
        "message": str(e)
    }))