#!/usr/bin/env bash
set -euo pipefail

# ---------------- user config ----------------
TRAIN_SCRIPT="/data2/users/lr4617/ddlp/eval/train_probabilistic_encoder_route_f.py"
checkpoint_name="/data2/users/lr4617/ddlp/outputs/040426_084653_twobody_ddlp_minimal_off_cnt_BIG"
CHECKPOINT_NAME_ARG="best"
LAGRANGIAN_CHECKPOINT="/data2/users/lr4617/discrete_lagrangian/del_pytorch/outputs/regularized/ghnn_generated_BIG_TRUE_matched_lam0p1"
EPOCHS=50
DEVICE="cuda:0"
BATCH_SIZE=32
OUTPUT_BASE="${checkpoint_name}/oracle"
RUN_NAME="c2_dyn_full"
GT_NORMALIZATION="ddlp_similarity"
GT_SIMILARITY_REFERENCE="${checkpoint_name}/extraction_evaluation/best/extracted_datasets/ddlp_extracted_recentered_training.h5.1"
MONITOR_EVERY_EPOCHS=1
EXACT_EVAL_MONITOR_EVERY_EPOCHS=5
MONITOR_NUM_VIDEOS=10
MONITOR_MODES="train,valid,test"
MONITOR_MAX_BATCHES=30
MONITOR_EVAL_SEQ_LEN=60
# --------------------------------------------

# train_probabilistic_encoder_route_f.py expects a real filename for --checkpoint_name.
RESOLVED_CHECKPOINT_NAME="${CHECKPOINT_NAME_ARG}"
if [[ "${CHECKPOINT_NAME_ARG}" == "best" ]]; then
  RESOLVED_CHECKPOINT_NAME="twobody_ddlp_minimal_off_cnt_BIG_best.pth"
fi

if [[ ! -f "${checkpoint_name}/saves/${RESOLVED_CHECKPOINT_NAME}" ]]; then
  echo "Checkpoint file not found: ${checkpoint_name}/saves/${RESOLVED_CHECKPOINT_NAME}" >&2
  exit 1
fi

if [[ ! -f "${GT_SIMILARITY_REFERENCE}" ]]; then
  echo "GT similarity reference not found: ${GT_SIMILARITY_REFERENCE}" >&2
  exit 1
fi

python3 "${TRAIN_SCRIPT}" \
  --checkpoint "${checkpoint_name}" \
  --checkpoint_name "${RESOLVED_CHECKPOINT_NAME}" \
  --route c2 \
  --objective c1_dyn \
  --beta 1 \
  --trainable_attribute_set position_scale_depth_obj_on \
  --position_head_training full \
  --lagrangian_checkpoint "${LAGRANGIAN_CHECKPOINT}" \
  --dyn_weak_window 0 \
  --dyn_latent_to_pixel_mode ddlp_similarity_inverse \
  --gt_normalization "${GT_NORMALIZATION}" \
  --gt_similarity_reference "${GT_SIMILARITY_REFERENCE}" \
  --epochs "${EPOCHS}" \
  --device "${DEVICE}" \
  --output_dir "${OUTPUT_BASE}/${RUN_NAME}" \
  --batch_size "${BATCH_SIZE}" \
  --swap_gt_xy 1 \
  --monitor_visualizations 1 \
  --monitor_every_epochs "${MONITOR_EVERY_EPOCHS}" \
  --exact_eval_monitor_every_epochs "${EXACT_EVAL_MONITOR_EVERY_EPOCHS}" \
  --monitor_num_videos "${MONITOR_NUM_VIDEOS}" \
  --monitor_modes "${MONITOR_MODES}" \
  --monitor_max_batches "${MONITOR_MAX_BATCHES}" \
  --monitor_eval_seq_len "${MONITOR_EVAL_SEQ_LEN}"
