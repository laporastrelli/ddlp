#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/data2/users/lr4617/ddlp"
BASE_CONFIG="$PROJECT_ROOT/configs/twobody_minimal_off_cnt.json"
RUN_DIR="$PROJECT_ROOT/outputs/160326_112120_twobody_ddlp_minimal_off_cnt_BIG"
SAVES_DIR="$RUN_DIR/saves"

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

TMP_CONFIG="$(BASE_CONFIG="$BASE_CONFIG" CKPT="$CKPT" python - <<'PY'
import json
import os
import tempfile

base_config = os.environ["BASE_CONFIG"]
ckpt = os.environ["CKPT"]

with open(base_config, "r") as f:
    cfg = json.load(f)

cfg["load_model"] = True
cfg["pretrained_path"] = ckpt

fd, tmp_path = tempfile.mkstemp(prefix="ddlp_resume_", suffix=".json")
os.close(fd)
with open(tmp_path, "w") as f:
    json.dump(cfg, f)

print(tmp_path)
PY
)"

cleanup() {
  rm -f "$TMP_CONFIG"
}
trap cleanup EXIT

cd "$PROJECT_ROOT"
python train_ddlp.py -d "$TMP_CONFIG"
