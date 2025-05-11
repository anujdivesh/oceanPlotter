#!/bin/bash
for f in *; do
  # Clean the filename:
  newname=$(echo "$f" | sed 's/&/and/g' | tr -s ' ' '_' | tr -d "'")
  if [[ "$f" != "$newname" ]]; then
    mv -- "$f" "$newname"
  fi
done