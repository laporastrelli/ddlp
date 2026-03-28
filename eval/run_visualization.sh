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
GHNN_DATASET_NAME="ddlp_extracted"
# ----------------------------

python3 "${SCRIPT}" \
  --checkpoint "${CHECKPOINT}" \
  --checkpoint_name "${CHECKPOINT_NAME}" \
  --device "${DEVICE}" \
  --batch_size "${BATCH_SIZE}" \
  --mode train \
  --convert_to_ghnn 1 \
  --extraction_method latent \
  --latent_position_variant recentered \
  --latent_recenter_source "${LATENT_RECENTER_SOURCE}" \
  --latent_recenter_nms_source "${LATENT_RECENTER_NMS_SOURCE}" \
  --eval_seq_len 360 \
  --ghnn_train_seq_len 60 \
  --ghnn_dataset_name "${GHNN_DATASET_NAME}" \
  --ghnn_step_size 1 \
  --ghnn_dt 1

python3 "${SCRIPT}" \
  --checkpoint "${CHECKPOINT}" \
  --checkpoint_name "${CHECKPOINT_NAME}" \
  --device "${DEVICE}" \
  --batch_size "${BATCH_SIZE}" \
  --mode valid \
  --convert_to_ghnn 1 \
  --extraction_method latent \
  --latent_position_variant recentered \
  --latent_recenter_source "${LATENT_RECENTER_SOURCE}" \
  --latent_recenter_nms_source "${LATENT_RECENTER_NMS_SOURCE}" \
  --eval_seq_len 360 \
  --ghnn_train_seq_len 60 \
  --ghnn_dataset_name "${GHNN_DATASET_NAME}" \
  --ghnn_step_size 1 \
  --ghnn_dt 1