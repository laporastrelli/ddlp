#!/usr/bin/env bash
set -euo pipefail

# -------- user config --------
REPO_DIR="/data2/users/lr4617/ddlp"
SCRIPT="${REPO_DIR}/eval/eval_bounding_boxes.py"
CHECKPOINT="/data2/users/lr4617/ddlp/outputs/290126_164237_twobody_ddlp_minimal_off_cnt"
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
  --max_batches all
  --extraction_method latent
  --latent_position_variant both
  --latent_recenter_source "${LATENT_RECENTER_SOURCE}"
  --latent_recenter_nms_source "${LATENT_RECENTER_NMS_SOURCE}"
)

for mode in train valid; do
  python3 "${SCRIPT}" \
    "${COMMON_ARGS[@]}" \
    --mode "${mode}" \
    --eval_seq_len "${EVAL_SEQ_LEN_BY_MODE[$mode]}" \
    --visualize_trajectories 1 \
    --extract_coordinates 0 \
    --evaluate_latent_alignment 0
done
