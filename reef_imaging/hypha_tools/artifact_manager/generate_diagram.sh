#!/bin/bash

# Generate workflow diagram from DOT file
echo "Generating upload process workflow diagram..."

# Check if Graphviz is installed
if ! command -v dot >/dev/null 2>&1; then
    echo "Graphviz not found. Please install it first:"
    echo "  Ubuntu/Debian: sudo apt-get install graphviz"
    echo "  macOS: brew install graphviz"
    exit 1
fi

# Generate PNG image
dot -Tpng upload_process.dot -o upload_process_diagram.png

# Check if generation was successful
if [ $? -eq 0 ]; then
    echo "Diagram generated successfully: upload_process_diagram.png"
    echo "You can now include this diagram in documentation."
else
    echo "Failed to generate diagram. Please check if the DOT file is valid."
    exit 1
fi 