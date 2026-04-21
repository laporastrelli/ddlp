#!/usr/bin/env bash
set -euo pipefail

# -------- user config --------
REPO_DIR="/data2/users/lr4617/ddlp"
SCRIPT="${REPO_DIR}/eval/eval_bounding_boxes.py"
CHECKPOINT="/data2/users/lr4617/ddlp/outputs/040426_084653_twobody_ddlp_minimal_off_cnt_BIG"
CHECKPOINT_NAME="best"
PROB_ENCODER_ROUTE_REQUESTED="${PROB_ENCODER_ROUTE:-c1}"
case "${PROB_ENCODER_ROUTE_REQUESTED}" in
  c1)
    PROB_ENCODER_EVAL_ROUTE="c1"
    PROB_ENCODER_ROUTE_DIR="c1"
    ;;
  c1_dyn|c1-dyn)
    PROB_ENCODER_EVAL_ROUTE="c1"
    PROB_ENCODER_ROUTE_DIR="c1_dyn"
    ;;
  c1_dyn_full|c1-dyn-full)
    PROB_ENCODER_EVAL_ROUTE="c1-dyn-attrs"
    PROB_ENCODER_ROUTE_DIR="c1_dyn_full"
    ;;
  c1_dyn_attrs|c1-dyn-attrs)
    PROB_ENCODER_EVAL_ROUTE="c1-dyn-attrs"
    PROB_ENCODER_ROUTE_DIR="c1_dyn_attrs"
    ;;
  *)
    echo "Unsupported PROB_ENCODER_ROUTE=${PROB_ENCODER_ROUTE_REQUESTED}. Use c1, c1_dyn, c1_dyn_full, or c1-dyn-attrs." >&2
    exit 1
    ;;
esac
PROB_ENCODER_CHECKPOINT="${PROB_ENCODER_CHECKPOINT:-${CHECKPOINT}/oracle/${PROB_ENCODER_ROUTE_DIR}/prob_encoder_best.pth}"
PROB_ENCODER_OUTPUT_DIR="${CHECKPOINT}/extraction_evaluation/${CHECKPOINT_NAME}/latent_alignment_eval/prob_encoder_${PROB_ENCODER_ROUTE_DIR}"
DEVICE="cuda:0"
BATCH_SIZE=20
LATENT_RECENTER_SOURCE="patch_alpha"
LATENT_RECENTER_NMS_SOURCE="nominal"
# ----------------------------

declare -A EVAL_SEQ_LEN_BY_MODE=(
  [train]=60
  [valid]=360
)

if [[ ! -f "${SCRIPT}" ]]; then
  echo "Missing evaluator script: ${SCRIPT}" >&2
  exit 1
fi

if [[ ! -d "${CHECKPOINT}" ]]; then
  echo "Missing checkpoint directory: ${CHECKPOINT}" >&2
  exit 1
fi

if [[ ! -f "${PROB_ENCODER_CHECKPOINT}" ]]; then
  echo "Missing probabilistic encoder checkpoint: ${PROB_ENCODER_CHECKPOINT}" >&2
  exit 1
fi

COMMON_ARGS=(
  --checkpoint "${CHECKPOINT}"
  --checkpoint_name "${CHECKPOINT_NAME}"
  --device "${DEVICE}"
  --batch_size "${BATCH_SIZE}"
  --max_batches all
  --extraction_method latent
  --latent_position_variant nominal
  --latent_recenter_source "${LATENT_RECENTER_SOURCE}"
  --latent_recenter_nms_source "${LATENT_RECENTER_NMS_SOURCE}"
  --prob_encoder_route "${PROB_ENCODER_EVAL_ROUTE}"
  --prob_encoder_output_tag "${PROB_ENCODER_ROUTE_DIR}"
  --prob_encoder_checkpoint "${PROB_ENCODER_CHECKPOINT}"
  --visualize_trajectories 0
  --extract_coordinates 1
  --evaluate_latent_alignment 1
  --reorder_method smallest_consecutive_distance
)

for mode in train valid; do
  for use_h in 0 1; do
    echo ""
    echo "=================================================================="
    echo "Running oracle latent alignment | mode=${mode} | hungarian=${use_h}"
    echo "Probabilistic encoder route=${PROB_ENCODER_ROUTE_REQUESTED} (eval route=${PROB_ENCODER_EVAL_ROUTE})"
    echo "Probabilistic encoder ckpt=${PROB_ENCODER_CHECKPOINT}"
    echo "Latent-alignment metrics dir=${PROB_ENCODER_OUTPUT_DIR}"
    echo "=================================================================="
    python3 "${SCRIPT}" \
      "${COMMON_ARGS[@]}" \
      --mode "${mode}" \
      --eval_seq_len "${EVAL_SEQ_LEN_BY_MODE[$mode]}" \
      --use_hungarian_for_correlation "${use_h}"
  done
done

echo ""
echo "Done: oracle latent alignment runs complete."
