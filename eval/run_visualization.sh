#!/usr/bin/env bash
set -euo pipefail

# -------- user config --------
REPO_DIR="/data2/users/lr4617/ddlp"
SCRIPT="${REPO_DIR}/eval/eval_bounding_boxes.py"
CHECKPOINT="/data2/users/lr4617/ddlp/outputs/040426_084653_twobody_ddlp_minimal_off_cnt_BIG"
CHECKPOINT_NAME="best"
DEVICE="cuda:0"
BATCH_SIZE=20
LATENT_RECENTER_SOURCE="patch_alpha"
LATENT_RECENTER_NMS_SOURCE="nominal"
# ----------------------------

declare -A EVAL_SEQ_LEN_BY_MODE=(
  [train]=60
  [valid]=360
)

COMMON_ARGS=(
  --checkpoint "${CHECKPOINT}"
  --checkpoint_name "${CHECKPOINT_NAME}"
  --device "${DEVICE}"
  --batch_size "${BATCH_SIZE}"
  --convert_to_ghnn 0
  --visualize_trajectories 1
  --extract_coordinates 0
  --max_batches 1
  --latent_recenter_source "${LATENT_RECENTER_SOURCE}"
  --latent_recenter_nms_source "${LATENT_RECENTER_NMS_SOURCE}"
)

VIS_TARGET="${1:-help}"

print_usage() {
  cat <<'EOF'
Usage:
  bash eval/run_visualization.sh <target>

Targets:
  latent_nominal        Latent visualizations (nominal positions)
  latent_recentered     Latent visualizations (recentered positions)
  latent_both_compare   Latent nominal-vs-recentered comparison visualizations
  bbox_full             Full bbox visualizations (mask overlays, ordered/unordered/smoothed trajectories)
  all                   Run all targets above sequentially
  help                  Show this help

Notes:
  - Each target runs separate eval_bounding_boxes.py calls for train and valid.
  - Default target is 'help' to avoid launching all visualizations unintentionally.
EOF
}

run_visualization_target() {
  local target_name="$1"
  shift
  for mode in train valid; do
    echo "Running target='${target_name}' mode='${mode}' eval_seq_len='${EVAL_SEQ_LEN_BY_MODE[$mode]}'"
    python3 "${SCRIPT}" \
      "${COMMON_ARGS[@]}" \
      --mode "${mode}" \
      --eval_seq_len "${EVAL_SEQ_LEN_BY_MODE[$mode]}" \
      "$@"
  done
}

case "${VIS_TARGET}" in
  latent_nominal)
    run_visualization_target "latent_nominal" \
      --extraction_method latent \
      --latent_position_variant nominal
    ;;
  latent_recentered)
    run_visualization_target "latent_recentered" \
      --extraction_method latent \
      --latent_position_variant recentered
    ;;
  latent_both_compare)
    run_visualization_target "latent_both_compare" \
      --extraction_method latent \
      --latent_position_variant both
    ;;
  bbox_full)
    run_visualization_target "bbox_full" \
      --extraction_method bbox \
      --latent_position_variant nominal
    ;;
  all)
    run_visualization_target "latent_nominal" \
      --extraction_method latent \
      --latent_position_variant nominal
    run_visualization_target "latent_recentered" \
      --extraction_method latent \
      --latent_position_variant recentered
    run_visualization_target "latent_both_compare" \
      --extraction_method latent \
      --latent_position_variant both
    run_visualization_target "bbox_full" \
      --extraction_method bbox \
      --latent_position_variant nominal
    ;;
  help|-h|--help)
    print_usage
    ;;
  *)
    echo "[ERROR] Unknown target: ${VIS_TARGET}" >&2
    print_usage
    exit 1
    ;;
esac
