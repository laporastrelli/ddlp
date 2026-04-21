#!/usr/bin/env bash
set -euo pipefail

# -------- user config --------
REPO_DIR="/data2/users/lr4617/ddlp"
SCRIPT="${REPO_DIR}/eval/eval_bounding_boxes.py"
CHECKPOINT="/data2/users/lr4617/ddlp/outputs/040426_084653_twobody_ddlp_minimal_off_cnt_BIG"
CHECKPOINT_NAME="best"
DEVICE="cuda:0"
BATCH_SIZE=10
LATENT_RECENTER_SOURCE="patch_alpha"
LATENT_RECENTER_NMS_SOURCE="nominal"
GHNN_DATASET_NAME="ddlp_extracted"
EVAL_SEQ_LEN=360
GHNN_TRAIN_SEQ_LEN=360
GHNN_STEP_SIZE=1
GHNN_DT=1
# ----------------------------

EXTRACTED_DATASETS_DIR="${CHECKPOINT}/extraction_evaluation/${CHECKPOINT_NAME}/extracted_datasets"

NOMINAL_ALL_RUNS_PATH="${EXTRACTED_DATASETS_DIR}/${GHNN_DATASET_NAME}_all_runs.h5.1"
NOMINAL_TRAINING_PATH="${EXTRACTED_DATASETS_DIR}/${GHNN_DATASET_NAME}_training.h5.1"
RECENTERED_ALL_RUNS_PATH="${EXTRACTED_DATASETS_DIR}/${GHNN_DATASET_NAME}_recentered_all_runs.h5.1"
RECENTERED_TRAINING_PATH="${EXTRACTED_DATASETS_DIR}/${GHNN_DATASET_NAME}_recentered_training.h5.1"

COMMON_ARGS=(
  --checkpoint "${CHECKPOINT}"
  --checkpoint_name "${CHECKPOINT_NAME}"
  --device "${DEVICE}"
  --batch_size "${BATCH_SIZE}"
  --mode train
  --convert_to_ghnn 1
  --extraction_method latent
  --latent_recenter_source "${LATENT_RECENTER_SOURCE}"
  --latent_recenter_nms_source "${LATENT_RECENTER_NMS_SOURCE}"
  --eval_seq_len "${EVAL_SEQ_LEN}"
  --ghnn_train_seq_len "${GHNN_TRAIN_SEQ_LEN}"
  --ghnn_dataset_name "${GHNN_DATASET_NAME}"
  --ghnn_output_dir "${EXTRACTED_DATASETS_DIR}"
  --ghnn_step_size "${GHNN_STEP_SIZE}"
  --ghnn_dt "${GHNN_DT}"
)

echo "Saving extracted datasets under: ${EXTRACTED_DATASETS_DIR}"
echo "Nominal outputs:"
echo "  ${NOMINAL_ALL_RUNS_PATH}"
echo "  ${NOMINAL_TRAINING_PATH}"
echo "Recentered outputs:"
echo "  ${RECENTERED_ALL_RUNS_PATH}"
echo "  ${RECENTERED_TRAINING_PATH}"

echo
echo "=== Saving NOMINAL extracted trajectories dataset ==="
python3 "${SCRIPT}" \
  "${COMMON_ARGS[@]}" \
  --latent_position_variant nominal

echo
echo "=== Saving RECENTERED extracted trajectories dataset ==="
python3 "${SCRIPT}" \
  "${COMMON_ARGS[@]}" \
  --latent_position_variant recentered
