#!/usr/bin/env python3
"""
Compute initialization (step-0) Pearson and R^2 metrics for Route-F probabilistic
encoder using the exact same model/dataloader/matching logic as
train_probabilistic_encoder_route_f.py.
"""

import argparse
import json
import os
import sys
from copy import deepcopy
from datetime import datetime
from typing import Dict

import torch
from torch.utils.data import DataLoader

# Allow running from any working directory.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from train_probabilistic_encoder_route_f import (  # noqa: E402
    _append_monitor_metrics_txt,
    _extract_state_dict,
    build_model_from_config,
    collect_visualization_samples_and_metrics,
    freeze_for_route_f,
    load_dataset_with_valid_alias_fallback,
    monitor_seq_len_for_mode,
    parse_monitor_modes,
    resolve_checkpoint_path,
    set_seed,
)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Evaluate Route-F probabilistic encoder initialization metrics "
            "(Pearson/R^2) with the same logic used in training monitor passes."
        )
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint directory containing hparams.json + saves/*.pth, or direct .pth path.",
    )
    p.add_argument(
        "--checkpoint_name",
        type=str,
        default=None,
        help="Checkpoint filename under <checkpoint>/saves. If omitted, auto-selects latest best/last.",
    )
    p.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Optional explicit config JSON path (defaults to <checkpoint_dir>/hparams.json).",
    )

    p.add_argument("--route", type=str, default="f1", choices=["f1", "f2"])
    p.add_argument(
        "--f1_delta_mode",
        type=str,
        default="relative_to_frozen_nominal",
        choices=["relative_to_frozen_nominal", "direct_trainable_nominal"],
    )
    p.add_argument(
        "--recenter_source",
        type=str,
        default="patch_alpha",
        choices=["patch_alpha", "global_alpha"],
    )
    p.add_argument("--recenter_eps", type=float, default=1e-6)

    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--seq_len",
        type=int,
        default=None,
        help="If omitted, uses config timestep_horizon.",
    )
    p.add_argument(
        "--num_static_frames",
        type=int,
        default=None,
        help="If omitted, uses config num_static_frames.",
    )
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument(
        "--temporal_reorder",
        type=str,
        default="auto",
        choices=["auto", "none", "smallest_consecutive_distance", "hungarian"],
    )
    p.add_argument("--use_in_camera_mask", type=int, default=1)
    p.add_argument(
        "--gt_normalization",
        type=str,
        default="auto",
        choices=["auto", "pixel", "zero_one", "minus_one_one"],
    )
    p.add_argument(
        "--swap_gt_xy",
        type=int,
        default=1,
        help=(
            "Swap normalized GT from dataset [x,y] to DDLP latent [y,x] before "
            "matching/metrics. Default 1 matches DDLP latent coordinate convention."
        ),
    )

    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--monitor_modes", type=str, default="train,valid,test")
    p.add_argument("--monitor_max_batches", type=int, default=30)
    p.add_argument(
        "--monitor_eval_seq_len",
        type=int,
        default=60,
        help="For monitor splits valid/val/test, use this seq_len instead of training seq_len.",
    )
    p.add_argument(
        "--global_step_tag",
        type=int,
        default=0,
        help="Step tag written to the metrics txt row (default 0 for initialization).",
    )
    p.add_argument(
        "--metrics_txt_path",
        type=str,
        default=None,
        help=(
            "Optional output txt path. Default: "
            "<checkpoint_dir>/oracle/visualizations/monitor_metrics_route_<route>_init.txt"
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()
    args.use_in_camera_mask = bool(args.use_in_camera_mask)
    args.swap_gt_xy = bool(args.swap_gt_xy)

    if args.monitor_max_batches <= 0:
        raise ValueError(f"monitor_max_batches must be > 0, got {args.monitor_max_batches}")
    if args.monitor_eval_seq_len <= 0:
        raise ValueError(f"monitor_eval_seq_len must be > 0, got {args.monitor_eval_seq_len}")

    monitor_modes = parse_monitor_modes(args.monitor_modes)

    ckpt_path, checkpoint_dir = resolve_checkpoint_path(args.checkpoint, args.checkpoint_name)
    config_path = args.config_path if args.config_path is not None else os.path.join(checkpoint_dir, "hparams.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config: Dict = json.load(f)

    if args.batch_size is None:
        args.batch_size = int(config["batch_size"])
    if args.seq_len is None:
        args.seq_len = int(config["timestep_horizon"])
    if args.num_static_frames is None:
        args.num_static_frames = int(config.get("num_static_frames", min(4, args.seq_len)))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    print("\n" + "=" * 90)
    print("Route-F Initialization Metrics")
    print("=" * 90)
    print(f"Checkpoint          : {ckpt_path}")
    print(f"Config              : {config_path}")
    print(f"Route               : {args.route}")
    print(f"Temporal reorder    : {args.temporal_reorder}")
    print(f"GT normalization    : {args.gt_normalization}")
    print(f"Swap GT x/y         : {int(args.swap_gt_xy)}")
    print(f"Train seq_len       : {args.seq_len}")
    print(f"Eval seq_len (v/t)  : {args.monitor_eval_seq_len}")
    print(f"Max batches/split   : {args.monitor_max_batches}")
    print(f"Device              : {device}")
    print("=" * 90 + "\n")

    model = build_model_from_config(config, device)
    checkpoint_obj = torch.load(ckpt_path, map_location=device)
    state_dict, _ = _extract_state_dict(checkpoint_obj)
    model.load_state_dict(state_dict, strict=True)
    freeze_for_route_f(model)

    frozen_model = None
    if args.route == "f1":
        frozen_model = deepcopy(model).to(device)
        frozen_model.eval()
        for p in frozen_model.parameters():
            p.requires_grad = False

    monitor_loaders = {}
    for requested_mode in monitor_modes:
        seq_len_for_mode = monitor_seq_len_for_mode(
            mode=requested_mode,
            train_seq_len=args.seq_len,
            eval_seq_len=args.monitor_eval_seq_len,
        )
        ds, resolved_mode = load_dataset_with_valid_alias_fallback(
            ds_name=config["ds"],
            root=config["root"],
            seq_len=seq_len_for_mode,
            mode=requested_mode,
            image_size=config["image_size"],
        )
        monitor_loaders[requested_mode] = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
            pin_memory=True,
        )
        alias_note = ""
        if resolved_mode != requested_mode:
            alias_note = f" (resolved as '{resolved_mode}')"
        print(
            f"[init-metrics] loader mode='{requested_mode}'{alias_note}, "
            f"seq_len={seq_len_for_mode}, num_samples={len(ds)}"
        )

    records = []
    for mode, loader in monitor_loaders.items():
        out = collect_visualization_samples_and_metrics(
            model=model,
            frozen_model=frozen_model,
            loader=loader,
            device=device,
            args=args,
            config=config,
            k=0,
            max_batches=args.monitor_max_batches,
            compute_metrics=True,
        )
        print(
            f"[init-metrics] mode={mode}: videos={out['num_videos']}, "
            f"batches={out['num_batches']}, "
            f"pearson={out['pearson_mean']:.4f}±{out['pearson_std']:.4f}, "
            f"r2={out['r2_mean']:.4f}±{out['r2_std']:.4f}"
        )
        records.append(
            {
                "mode": mode,
                "num_videos": out["num_videos"],
                "num_batches": out["num_batches"],
                "pearson_mean": out["pearson_mean"],
                "pearson_std": out["pearson_std"],
                "r2_mean": out["r2_mean"],
                "r2_std": out["r2_std"],
            }
        )

    if args.metrics_txt_path is None:
        vis_root = os.path.join(checkpoint_dir, "oracle", "visualizations")
        metrics_txt_path = os.path.join(
            vis_root,
            f"monitor_metrics_route_{args.route}_init.txt",
        )
    else:
        metrics_txt_path = os.path.abspath(args.metrics_txt_path)

    _append_monitor_metrics_txt(
        metrics_txt_path=metrics_txt_path,
        route=args.route,
        global_step=int(args.global_step_tag),
        records=records,
    )

    summary_json_path = os.path.splitext(metrics_txt_path)[0] + "_summary.json"
    payload = {
        "written_at_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checkpoint_source": ckpt_path,
        "config_source": config_path,
        "route": args.route,
        "global_step_tag": int(args.global_step_tag),
        "monitor_modes": monitor_modes,
        "monitor_max_batches": int(args.monitor_max_batches),
        "monitor_eval_seq_len": int(args.monitor_eval_seq_len),
        "temporal_reorder": args.temporal_reorder,
        "gt_normalization": args.gt_normalization,
        "swap_gt_xy": bool(args.swap_gt_xy),
        "use_in_camera_mask": bool(args.use_in_camera_mask),
        "records": records,
    }
    with open(summary_json_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved initialization metrics txt to: {metrics_txt_path}")
    print(f"Saved initialization metrics summary to: {summary_json_path}")


if __name__ == "__main__":
    main()
