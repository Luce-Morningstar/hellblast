#!/bin/bash
# Install hellblast
# Run from the directory containing this script

set -e

echo "Installing hellblast..."

# Check for pip
if ! command -v pip &> /dev/null; then
    echo "Error: pip not found. Please install Python first."
    exit 1
fi

# Install in editable mode
pip install -e ".[all]" 2>/dev/null || pip install -e .

echo ""
echo "✓ Hellblast installed!"
echo ""
echo "Usage:"
echo "  import hellblast.hp_compat as hp"
echo "  # or"
echo "  from hellblast import map2alm, alm2map, anafast"
