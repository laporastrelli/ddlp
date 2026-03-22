#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${1:-/data2/users/lr4617/ddlp/outputs/290126_164237_twobody_ddlp_minimal_off_cnt}"
CHECKPOINT_NAME="${2:-best}"
EXTRACTION_METHOD="${3:-latent}"
NOISY_GT_ALPHAS="${4:-0.05,0.1,0.3,0.5,0.7}"
NOISY_GT_SEED="${5:-0}"

for MODE in train valid; do
  if [[ "${MODE}" == "train" ]]; then
    EVAL_SEQ_LEN=60
  else
    EVAL_SEQ_LEN=360
  fi
  echo ""
  echo "-----------------------------------------------------------------------------------"
  echo "Running noisy-GT reference eval | mode=${MODE} | extraction_method=${EXTRACTION_METHOD} | noise_alphas=${NOISY_GT_ALPHAS} | noise_seed=${NOISY_GT_SEED}"
  echo "-----------------------------------------------------------------------------------"
  python eval/eval_bounding_boxes.py \
    --checkpoint "${CHECKPOINT}" \
    --checkpoint_name "${CHECKPOINT_NAME}" \
    --convert_to_ghnn 0 \
    --mode "${MODE}" \
    --eval_seq_len "${EVAL_SEQ_LEN}" \
    --extract_coordinates 0 \
    --evaluate_latent_alignment 0 \
    --evaluate_noisy_gt_reference 1 \
    --noisy_gt_noise_mode relative_step \
    --noisy_gt_noise_alphas "${NOISY_GT_ALPHAS}" \
    --noisy_gt_noise_seed "${NOISY_GT_SEED}" \
    --visualize_trajectories 0 \
    --extraction_method "${EXTRACTION_METHOD}" \
    --reorder_method smallest_consecutive_distance \
    --use_hungarian_for_correlation 0
done

for REORDER_METHOD in smallest_consecutive_distance hungarian; do
  for USE_HUNGARIAN in 1 0; do
    for MODE in train valid; do
      if [[ "${MODE}" == "train" ]]; then
        EVAL_SEQ_LEN=60
      else
        EVAL_SEQ_LEN=360
      fi
      echo ""
      echo "-----------------------------------------------------------------------------------"
      echo "Running latent alignment eval | mode=${MODE} | reorder_method=${REORDER_METHOD} | use_hungarian_for_correlation=${USE_HUNGARIAN} | eval_seq_len=${EVAL_SEQ_LEN}"
      echo "-----------------------------------------------------------------------------------"
      python eval/eval_bounding_boxes.py \
        --checkpoint "${CHECKPOINT}" \
        --checkpoint_name "${CHECKPOINT_NAME}" \
        --convert_to_ghnn 0 \
        --mode "${MODE}" \
        --eval_seq_len "${EVAL_SEQ_LEN}" \
        --extract_coordinates 1 \
        --evaluate_latent_alignment 1 \
        --evaluate_noisy_gt_reference 0 \
        --visualize_trajectories 0 \
        --extraction_method "${EXTRACTION_METHOD}" \
        --reorder_method "${REORDER_METHOD}" \
        --use_hungarian_for_correlation "${USE_HUNGARIAN}"
    done
  done
done

echo "Sweep completed."
