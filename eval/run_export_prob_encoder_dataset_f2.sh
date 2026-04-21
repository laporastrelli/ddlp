#!/usr/bin/env bash
set -euo pipefail

# Export Route-F.2 (recentering-free / nominal) probabilistic encoder
# trajectories to GHNN/DEL-compatible HDF5 files using eval_bounding_boxes.py.

# -------- user config --------
REPO_DIR="/data2/users/lr4617/ddlp"
SCRIPT="${REPO_DIR}/eval/eval_bounding_boxes.py"
CHECKPOINT="/data2/users/lr4617/ddlp/outputs/040426_084653_twobody_ddlp_minimal_off_cnt_BIG"
CHECKPOINT_NAME="best"
PROB_ENCODER_CHECKPOINT="${CHECKPOINT}/oracle/f2/prob_encoder_best.pth"
DEVICE="cuda:0"
BATCH_SIZE=32
GHNN_OUTPUT_DIR="${CHECKPOINT}/oracle/extracted_datasets"
GHNN_DATASET_NAME="prob_encoder_f2"
GHNN_TRAIN_SEQ_LEN=60
EVAL_SEQ_LEN=360
GHNN_STEP_SIZE=1
GHNN_DT=1
LATENT_RECENTER_SOURCE="patch_alpha"
LATENT_RECENTER_NMS_SOURCE="nominal"
ALIGNMENT_SPLIT="test"
# ----------------------------

echo "============================================================"
echo "  Exporting probabilistic encoder dataset: Route F.2"
echo "  Base checkpoint : ${CHECKPOINT}"
echo "  Prob checkpoint : ${PROB_ENCODER_CHECKPOINT}"
echo "  Output dir      : ${GHNN_OUTPUT_DIR}"
echo "  Dataset name    : ${GHNN_DATASET_NAME}"
echo "============================================================"

if [[ ! -f "${PROB_ENCODER_CHECKPOINT}" ]]; then
  echo "Probabilistic encoder checkpoint not found: ${PROB_ENCODER_CHECKPOINT}" >&2
  exit 1
fi

python3 "${SCRIPT}" \
  --checkpoint "${CHECKPOINT}" \
  --checkpoint_name "${CHECKPOINT_NAME}" \
  --convert_to_ghnn 1 \
  --extraction_method latent \
  --prob_encoder_checkpoint "${PROB_ENCODER_CHECKPOINT}" \
  --prob_encoder_route f2 \
  --ghnn_output_dir "${GHNN_OUTPUT_DIR}" \
  --ghnn_dataset_name "${GHNN_DATASET_NAME}" \
  --device "${DEVICE}" \
  --batch_size "${BATCH_SIZE}" \
  --max_batches all \
  --ghnn_train_seq_len "${GHNN_TRAIN_SEQ_LEN}" \
  --eval_seq_len "${EVAL_SEQ_LEN}" \
  --ghnn_step_size "${GHNN_STEP_SIZE}" \
  --ghnn_dt "${GHNN_DT}" \
  --latent_recenter_source "${LATENT_RECENTER_SOURCE}" \
  --latent_recenter_nms_source "${LATENT_RECENTER_NMS_SOURCE}" \
  --write_prob_encoder_alignment_metrics 1 \
  --alignment_split "${ALIGNMENT_SPLIT}" \
  --use_hungarian_for_correlation 1

echo ""
echo "============================================================"
echo "  Route F.2 export complete."
echo "  Expected dataset:"
echo "    ${GHNN_OUTPUT_DIR}/${GHNN_DATASET_NAME}_training.h5.1"
echo "============================================================"
