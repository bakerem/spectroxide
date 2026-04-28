#!/bin/bash
# Download the CosmoTherm Green's function database (~5 MB compressed).
# Source: https://www.jb.man.ac.uk/~jchluba/Science/CosmoTherm/Download.html
#
# The tarball contains the Greens package (C++/Python code + precomputed GF database).
# We extract only the data file.

set -e
cd "$(dirname "$0")"

URL="http://www.cita.utoronto.ca/~jchluba/CosmoTherm/_Downloads_/Greens.tar.gz"
TARBALL="Greens.tar.gz"

echo "Downloading CosmoTherm Green's function package..."
curl -L -o "$TARBALL" "$URL"

echo "Extracting Green's function database..."
tar xzf "$TARBALL"

# The tarball extracts to a Greens/ directory with the data file inside
if [ -f "Greens/Greens_data.dat" ]; then
    mv Greens/Greens_data.dat .
    rm -rf Greens "$TARBALL"
    echo "Done: Greens_data.dat ($(du -h Greens_data.dat | cut -f1))"
elif [ -f "Greens_data.dat" ]; then
    rm -f "$TARBALL"
    echo "Done: Greens_data.dat ($(du -h Greens_data.dat | cut -f1))"
else
    echo "Extracting all files from tarball..."
    ls -la
    echo "Warning: Greens_data.dat not found in expected location."
    echo "Check extracted files above and move manually."
    rm -f "$TARBALL"
fi
