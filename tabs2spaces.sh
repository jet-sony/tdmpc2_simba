#!/bin/bash

# Define the number of spaces to replace each tab with
TAB_WIDTH=4
SPACES=$(printf ' %.0s' $(seq 1 $TAB_WIDTH))

# Find all .py files and replace tabs with spaces
find "./tdmpc2" -name '*.py' -type f | while read -r file; do
    echo "Converting tabs to spaces in: $file"
    sed -i.bak "s/\t/$SPACES/g" "$file"
    rm "${file}.bak" # Remove backup file created by sed
done

echo "All tabs have been replaced with spaces in all .py files."
