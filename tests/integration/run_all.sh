#!/bin/bash
# Run all integration test scenarios.
#
# Usage:
#   ./tests/integration/run_all.sh              # new pipeline only
#   ./tests/integration/run_all.sh --compare-old # also compare with old pipeline
#   ./tests/integration/run_all.sh --skip-nsm    # skip NSM/BScore (faster)

set -e
cd "$(dirname "$0")/../.."

EXTRA_ARGS="$@"
PYTHON="/home/gattia/miniconda3/envs/kneepipeline/bin/python"
export LD_LIBRARY_PATH="/home/gattia/miniconda3/envs/kneepipeline/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
OUTPUT_BASE="/tmp/kneepipeline_integration_$(date +%Y%m%d_%H%M%S)"

echo "============================================================"
echo "Integration tests — output base: $OUTPUT_BASE"
echo "============================================================"
echo ""

# Test 1: NRRD + ananya (default DOSMA model)
echo ">>> Test 1/3: NRRD + ananya (acl_qdess_bone_july_2024)"
$PYTHON tests/integration/run_pipeline_test.py \
    data/anthonys_knee.nrrd \
    "$OUTPUT_BASE/nrrd_ananya" \
    $EXTRA_ARGS
echo ""

# Test 2: NRRD + nnU-Net
echo ">>> Test 2/3: NRRD + nnU-Net"
$PYTHON tests/integration/run_pipeline_test.py \
    data/anthonys_knee.nrrd \
    "$OUTPUT_BASE/nrrd_nnunet" \
    --model nnunet_knee \
    $EXTRA_ARGS
echo ""

# Test 3: qDESS DICOM + ananya (tests T2 mapping)
echo ">>> Test 3/3: qDESS DICOM + ananya"
$PYTHON tests/integration/run_pipeline_test.py \
    data/012_knee1 \
    "$OUTPUT_BASE/qdess_ananya" \
    $EXTRA_ARGS
echo ""

echo "============================================================"
echo "All integration tests complete."
echo "Results in: $OUTPUT_BASE"
echo "============================================================"
