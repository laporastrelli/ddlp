#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/data2/users/lr4617/ddlp"
RUN_DIR="$PROJECT_ROOT/outputs/250426_231916_threebody_ddlp_minimal_off_cnt_BIG_r3_t10_ablate_nkpe5"
BASE_CONFIG="$RUN_DIR/hparams.json"
SAVES_DIR="$RUN_DIR/saves"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_ddlp}"

if [[ ! -f "$BASE_CONFIG" ]]; then
  echo "Missing base config: $BASE_CONFIG" >&2
  exit 1
fi

if [[ ! -d "$SAVES_DIR" ]]; then
  echo "Missing saves directory: $SAVES_DIR" >&2
  exit 1
fi

CKPT="$(ls -t "$SAVES_DIR"/*.pth 2>/dev/null | grep -v '_best' | head -n1 || true)"
if [[ -z "$CKPT" ]]; then
  echo "No non-best checkpoint found in $SAVES_DIR" >&2
  exit 1
fi

echo "Resuming from checkpoint: $CKPT"

TMP_CONFIG="$("$PYTHON_BIN" - "$BASE_CONFIG" "$CKPT" <<'PY'
import json
import os
import sys
import tempfile

base_config = sys.argv[1]
ckpt = sys.argv[2]

with open(base_config, "r") as f:
    cfg = json.load(f)

cfg["load_model"] = True
cfg["pretrained_path"] = ckpt

fd, tmp_path = tempfile.mkstemp(prefix="ddlp_resume_threebody_nkpe5_", suffix=".json")
os.close(fd)
with open(tmp_path, "w") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")

print(tmp_path)
PY
)"

cleanup() {
  rm -f "$TMP_CONFIG"
}
trap cleanup EXIT

mkdir -p "$MPLCONFIGDIR"
cd "$PROJECT_ROOT"
MPLCONFIGDIR="$MPLCONFIGDIR" "$PYTHON_BIN" train_ddlp.py -d "$TMP_CONFIG"
