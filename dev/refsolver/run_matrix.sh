#!/bin/bash
# Production run matrix for the R3 reference solver.  Sequential: 7 GB box.
set -e
P=/home/bakerem/miniforge3/bin/python
cd /home/bakerem/spectroxide/dev/refsolver
L=outputs/run_matrix.log
: > $L
run() { echo "### $*" >> $L; $P refsolver.py "$@" >> $L 2>&1; }

run --N 2049 --refine 1 --write-spectra                                   # baseline
run --N 4097 --refine 1 --tag _N4097                                      # grid x2
run --N 2049 --refine 2 --tag _refine2                                    # steps x2
run --N 2049 --refine 1 --drho 1e-5 --cases heat_z2e6,heat_z2e5,heat_z5e3 --tag _drho1e-5
run --N 2049 --refine 1 --z-end 1 --tag _zend1
run --N 2049 --refine 1 --no-br --tag _nobr
run --N 1025 --refine 1 --tag _N1025                                      # grid /2
run --N 2049 --refine 0.5 --tag _refine0.5                                # steps /2
echo "ALL DONE" >> $L
