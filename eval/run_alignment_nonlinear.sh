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
TRAIN_SEQ_LEN=60
VALID_SEQ_LEN=360
# ----------------------------

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
  --visualize_trajectories 0
  --extract_coordinates 1
  --evaluate_latent_alignment_nonlinear 1
  --reorder_method smallest_consecutive_distance
  --nonlinear_train_seq_len "${TRAIN_SEQ_LEN}"
  --nonlinear_valid_seq_len "${VALID_SEQ_LEN}"
)

for use_h in 0 1; do
  python3 "${SCRIPT}" \
    "${COMMON_ARGS[@]}" \
    --use_hungarian_for_correlation "${use_h}"
done
