#!/bin/bash
#
# Quick visualization viewer for comparative diagnostics
# Usage: ./view_diagnostics.sh
#

DIAG_DIR="/data2/users/lr4617/data_twobody_tries/diagnostics_comparative"

echo "=================================================="
echo "Comparative Diagnostics Visualization Viewer"
echo "=================================================="
echo ""
echo "Available visualizations:"
echo ""
echo "1. Extracted trajectories (from DDLP):"
ls -1 "$DIAG_DIR/extracted/"*.png | nl
echo ""
echo "2. Ground-truth trajectories (from physics simulation):"
ls -1 "$DIAG_DIR/groundtruth/"*.png | nl
echo ""
echo "3. Side-by-side comparisons:"
ls -1 "$DIAG_DIR/comparison/"*.png | nl
echo ""
echo "=================================================="
echo "Full diagnostic log:"
echo "  /tmp/comparative_diagnostic.log"
echo ""
echo "Summary documents:"
echo "  /data2/users/lr4617/ddlp/eval/SUMMARY_COMPARATIVE_DIAGNOSTICS.md"
echo "  /data2/users/lr4617/ddlp/eval/HAMILTONIAN_DOCUMENTATION.md"
echo "  /data2/users/lr4617/ddlp/eval/PHYSICS_PARAMETER_REGRESSION_ANALYSIS.md"
echo ""
echo "=================================================="
echo ""
echo "To view images (if display available):"
echo "  display $DIAG_DIR/comparison/comparison_run_0.png"
echo ""
echo "To copy to local machine:"
echo "  scp -r <username>@<host>:$DIAG_DIR ./diagnostics_comparative"
echo ""
