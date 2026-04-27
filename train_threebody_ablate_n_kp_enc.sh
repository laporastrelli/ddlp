#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/data2/users/lr4617/ddlp"
BASE_CONFIG="${REPO_ROOT}/outputs/230426_093625_threebody_ddlp_minimal_off_cnt_BIG_r3_t20/hparams.json"
GENERATED_CONFIG_DIR="${REPO_ROOT}/configs/generated_ablations"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_ddlp}"

# 230426 is the baseline control run with n_kp_enc=6 and timestep_horizon=10.
# This script only launches the ablated variants to compare against that run.
N_KP_ENC_VALUES=(5)

if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "Baseline config not found: ${BASE_CONFIG}" >&2
  exit 1
fi

mkdir -p "${GENERATED_CONFIG_DIR}" "${MPLCONFIGDIR}"
cd "${REPO_ROOT}"

for N_KP_ENC in "${N_KP_ENC_VALUES[@]}"; do
  CONFIG_PATH="${GENERATED_CONFIG_DIR}/threebody_minimal_off_cnt_t10_nkpe${N_KP_ENC}.json"

  "${PYTHON_BIN}" - "${BASE_CONFIG}" "${CONFIG_PATH}" "${N_KP_ENC}" <<'PY'
import json
import pathlib
import sys

base_config_path = pathlib.Path(sys.argv[1])
out_config_path = pathlib.Path(sys.argv[2])
n_kp_enc = int(sys.argv[3])

with base_config_path.open() as f:
    config = json.load(f)

# Keep the 230426 run as the control and only ablate posterior slot count.
config["n_kp_enc"] = n_kp_enc
config["timestep_horizon"] = 10
config["cond_steps"] = 10
config["num_static_frames"] = 6

# Leave topk/n_kp_prior at the 230426 baseline values. train_ddlp.py already
# clamps plotting topk to min(topk, n_kp_enc).
config["run_prefix"] = f"_minimal_off_cnt_BIG_r3_t10_ablate_nkpe{n_kp_enc}"

out_config_path.parent.mkdir(parents=True, exist_ok=True)
with out_config_path.open("w") as f:
    json.dump(config, f, indent=2)
    f.write("\n")

print(out_config_path)
PY

  echo "Launching three-body ablation with n_kp_enc=${N_KP_ENC}"
  MPLCONFIGDIR="${MPLCONFIGDIR}" "${PYTHON_BIN}" train_ddlp.py -d "${CONFIG_PATH}"
done
