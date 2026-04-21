#!/usr/bin/env python3
"""
Supervised probabilistic encoder trainer for Route F.

Implements:
- Route F.1 / C1: recentering-biased sequential Gaussian encoder
- C1-dyn: C1 position objective plus a fixed DEL weak residual in D6-TRUE coordinates

Design choices follow:
- guidance_ablation_experimental_plan.md, Route F + Phase 0
- freeze all pretrained DDLP components except selected attribute posterior heads
- preserve sequential DDLP recurrence and burn-in/proposal behavior
- deterministic recurrence based on posterior means
- sequence-level Hungarian identity handling
- detached temporal reordering prior to supervised matching
- Gaussian NLL with DDLP position posterior variance (no extra covariance head)
- optional fixed separable Lagrangian residual after C1->D6-TRUE coordinate transform
"""

import argparse
import contextlib
import glob
import io
import json
import math
import os
import random
import sys
from copy import deepcopy
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import h5py
import matplotlib
import numpy as np
import torch
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import imageio
except ImportError:
    imageio = None

# Allow running from any working directory.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from datasets.get_dataset import get_video_dataset
from models import ObjectDynamicsDLP

DEL_PARENT = "/data2/users/lr4617/discrete_lagrangian"
if os.path.isdir(DEL_PARENT) and DEL_PARENT not in sys.path:
    sys.path.insert(0, DEL_PARENT)

try:
    from del_pytorch.models.separable_lagrangian import SeparableLagrangianMLP
except Exception:
    SeparableLagrangianMLP = None


LOG_2PI = math.log(2.0 * math.pi)
_EVAL_MONITOR_HELPERS = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _extract_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict):
        if 'model_state_dict' in checkpoint_obj:
            return checkpoint_obj['model_state_dict'], checkpoint_obj
        if 'model' in checkpoint_obj:
            return checkpoint_obj['model'], checkpoint_obj
        if 'state_dict' in checkpoint_obj:
            return checkpoint_obj['state_dict'], checkpoint_obj
    return checkpoint_obj, checkpoint_obj if isinstance(checkpoint_obj, dict) else {}


def resolve_checkpoint_path(checkpoint: str, checkpoint_name: Optional[str]) -> Tuple[str, str]:
    """
    Returns:
        ckpt_path, checkpoint_dir
    """
    checkpoint = os.path.abspath(checkpoint)
    if checkpoint.endswith('.pth'):
        ckpt_path = checkpoint
        if os.path.basename(os.path.dirname(ckpt_path)) == 'saves':
            checkpoint_dir = os.path.dirname(os.path.dirname(ckpt_path))
        else:
            checkpoint_dir = os.path.dirname(ckpt_path)
        return ckpt_path, checkpoint_dir

    checkpoint_dir = checkpoint
    if checkpoint_name is not None:
        ckpt_path = os.path.join(checkpoint_dir, 'saves', checkpoint_name)
        if os.path.exists(ckpt_path):
            return ckpt_path, checkpoint_dir
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    saves_dir = os.path.join(checkpoint_dir, 'saves')
    if not os.path.isdir(saves_dir):
        raise FileNotFoundError(f"Could not find saves directory: {saves_dir}")

    candidates = [f for f in os.listdir(saves_dir) if f.endswith('.pth')]
    if not candidates:
        raise FileNotFoundError(f"No .pth checkpoints found under: {saves_dir}")

    # Prefer best checkpoints, then most recent by mtime.
    best_candidates = [f for f in candidates if 'best' in f.lower()]
    chosen_pool = best_candidates if best_candidates else candidates
    chosen_pool = sorted(chosen_pool, key=lambda n: os.path.getmtime(os.path.join(saves_dir, n)), reverse=True)
    ckpt_path = os.path.join(saves_dir, chosen_pool[0])
    return ckpt_path, checkpoint_dir


def build_model_from_config(config: Dict, device: torch.device) -> ObjectDynamicsDLP:
    model = ObjectDynamicsDLP(
        cdim=config['ch'],
        enc_channels=config['enc_channels'],
        prior_channels=config['prior_channels'],
        image_size=config['image_size'],
        n_kp=config['n_kp'],
        learned_feature_dim=config['learned_feature_dim'],
        pad_mode=config['pad_mode'],
        sigma=config['sigma'],
        dropout=config['dropout'],
        patch_size=config['patch_size'],
        n_kp_enc=config['n_kp_enc'],
        n_kp_prior=config['n_kp_prior'],
        kp_range=tuple(config['kp_range']),
        kp_activation=config['kp_activation'],
        anchor_s=config['anchor_s'],
        use_resblock=config.get('use_resblock', False),
        timestep_horizon=config['timestep_horizon'],
        predict_delta=config['predict_delta'],
        scale_std=config['scale_std'],
        offset_std=config['offset_std'],
        obj_on_alpha=config['obj_on_alpha'],
        obj_on_beta=config['obj_on_beta'],
        pint_layers=config.get('pint_layers', 2),
        pint_heads=config.get('pint_heads', 4),
        pint_dim=config.get('pint_dim', 128),
        use_correlation_heatmaps=config.get('use_correlation_heatmaps', False),
        enable_enc_attn=config.get('enable_enc_attn', False),
        filtering_heuristic=config.get('filtering_heuristic', 'variance'),
    ).to(device)
    return model


def _set_module_trainable(module: torch.nn.Module, trainable: bool) -> None:
    for p in module.parameters():
        p.requires_grad = trainable


def _build_linear_mean_only_grad_masks(layer: torch.nn.Linear) -> Dict[torch.nn.Parameter, torch.Tensor]:
    if layer.out_features != 2:
        raise ValueError(
            "Mean-only masking expects a 2-output linear layer "
            f"(mu, logvar), got out_features={layer.out_features}."
        )
    weight_mask = torch.zeros_like(layer.weight)
    weight_mask[0] = 1.0
    bias_mask = torch.zeros_like(layer.bias)
    bias_mask[0] = 1.0
    return {
        layer.weight: weight_mask,
        layer.bias: bias_mask,
    }


def _apply_gradient_masks(grad_masks: Optional[Dict[torch.nn.Parameter, torch.Tensor]]) -> None:
    if not grad_masks:
        return
    for param, mask in grad_masks.items():
        if param.grad is not None:
            param.grad.mul_(mask)


def _ema_update(prev: Optional[float], value: float, decay: float) -> float:
    value = float(value)
    if prev is None or (not math.isfinite(prev)):
        return value
    return float(decay * prev + (1.0 - decay) * value)


def _masked_grad_norm(
    loss: torch.Tensor,
    params: List[torch.nn.Parameter],
    grad_masks: Optional[Dict[torch.nn.Parameter, torch.Tensor]] = None,
) -> torch.Tensor:
    if not torch.is_tensor(loss) or (not loss.requires_grad):
        if len(params) == 0:
            return torch.tensor(0.0)
        return params[0].new_tensor(0.0)

    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=True,
        allow_unused=True,
    )
    total_sq = None
    for param, grad in zip(params, grads):
        if grad is None:
            continue
        if grad_masks and param in grad_masks:
            grad = grad * grad_masks[param]
        sq = grad.pow(2).sum()
        total_sq = sq if total_sq is None else total_sq + sq
    if total_sq is None:
        return loss.new_tensor(0.0)
    return total_sq.sqrt()


def _update_adaptive_beta_state(
    adaptive_beta_state: Dict[str, float],
    *,
    c: float,
    ema_decay: float,
    eps: float,
    position_mse: float,
    pos_grad_norm: float,
    del_grad_norm: float,
) -> Dict[str, float]:
    pos_grad_ema = _ema_update(adaptive_beta_state.get('pos_grad_ema'), pos_grad_norm, ema_decay)
    del_grad_ema = _ema_update(adaptive_beta_state.get('del_grad_ema'), del_grad_norm, ema_decay)
    fit_mse_ema = _ema_update(adaptive_beta_state.get('fit_mse_ema'), position_mse, ema_decay)

    fit_mse_ref = adaptive_beta_state.get('fit_mse_ref')
    if fit_mse_ref is None or (not math.isfinite(fit_mse_ref)):
        fit_mse_ref = max(float(fit_mse_ema), float(eps))
    else:
        fit_mse_ref = max(float(fit_mse_ref), float(eps))

    fit_quality = 1.0 - float(fit_mse_ema) / fit_mse_ref
    fit_quality = max(0.0, min(1.0, fit_quality))
    gate = fit_quality
    beta_ratio = float(c) * float(pos_grad_ema) / max(float(del_grad_ema), float(eps))
    beta_eff = gate * beta_ratio

    adaptive_beta_state.update({
        'pos_grad_ema': float(pos_grad_ema),
        'del_grad_ema': float(del_grad_ema),
        'fit_mse_ema': float(fit_mse_ema),
        'fit_mse_ref': float(fit_mse_ref),
        'fit_quality': float(fit_quality),
        'gate': float(gate),
        'beta_ratio': float(beta_ratio),
        'beta_eff': float(beta_eff),
    })
    return adaptive_beta_state


def _current_adaptive_beta_terms(
    adaptive_beta_state: Optional[Dict[str, float]],
) -> Dict[str, float]:
    if not adaptive_beta_state:
        return {
            'beta_eff': 0.0,
            'gate': 0.0,
            'fit_quality': 0.0,
            'pos_grad_ema': 0.0,
            'del_grad_ema': 0.0,
            'fit_mse_ema': 0.0,
            'fit_mse_ref': 0.0,
            'beta_ratio': 0.0,
        }
    return {
        'beta_eff': float(adaptive_beta_state.get('beta_eff', 0.0)),
        'gate': float(adaptive_beta_state.get('gate', 0.0)),
        'fit_quality': float(adaptive_beta_state.get('fit_quality', 0.0)),
        'pos_grad_ema': float(adaptive_beta_state.get('pos_grad_ema', 0.0)),
        'del_grad_ema': float(adaptive_beta_state.get('del_grad_ema', 0.0)),
        'fit_mse_ema': float(adaptive_beta_state.get('fit_mse_ema', 0.0)),
        'fit_mse_ref': float(adaptive_beta_state.get('fit_mse_ref', 0.0)),
        'beta_ratio': float(adaptive_beta_state.get('beta_ratio', 0.0)),
    }


def freeze_for_route_f(
    model: ObjectDynamicsDLP,
    trainable_attribute_set: str,
    position_head_training: str,
) -> Optional[Dict[torch.nn.Parameter, torch.Tensor]]:
    for p in model.parameters():
        p.requires_grad = False

    attr_enc = model.fg_module.particle_attribute_enc

    # Train the original DDLP position posterior outputs in all C1 variants.
    _set_module_trainable(attr_enc.x_head, True)
    _set_module_trainable(attr_enc.y_head, True)
    grad_masks = None
    if position_head_training == 'mean_only':
        grad_masks = {}
        grad_masks.update(_build_linear_mean_only_grad_masks(attr_enc.x_head))
        grad_masks.update(_build_linear_mean_only_grad_masks(attr_enc.y_head))
    elif position_head_training != 'full':
        raise ValueError(f"Unknown position_head_training: {position_head_training}")

    if trainable_attribute_set == 'position_only':
        return grad_masks

    if trainable_attribute_set == 'position_scale_depth_obj_on':
        _set_module_trainable(attr_enc.scale_xy_head, True)
        _set_module_trainable(attr_enc.depth_head, True)
        _set_module_trainable(attr_enc.obj_on_head, True)
        return grad_masks

    raise ValueError(f"Unknown trainable_attribute_set: {trainable_attribute_set}")


def get_trainable_parameters(model: ObjectDynamicsDLP):
    return [p for p in model.parameters() if p.requires_grad]


def parse_monitor_modes(modes_csv: str) -> List[str]:
    allowed = {'train', 'valid', 'val', 'test'}
    out = []
    for tok in str(modes_csv).split(','):
        mode = tok.strip().lower()
        if mode == '':
            continue
        if mode not in allowed:
            raise ValueError(
                f"Unknown mode '{mode}' in monitor_modes. Allowed: train, valid, val, test."
            )
        if mode not in out:
            out.append(mode)
    if len(out) == 0:
        raise ValueError("monitor_modes is empty after parsing.")
    return out


def monitor_seq_len_for_mode(mode: str, train_seq_len: int, eval_seq_len: int) -> int:
    mode = str(mode).strip().lower()
    if mode in {'valid', 'val', 'test'}:
        return int(eval_seq_len)
    return int(train_seq_len)


def load_dataset_with_valid_alias_fallback(
    ds_name: str,
    root: str,
    seq_len: int,
    mode: str,
    image_size: int,
):
    """
    Some datasets expose validation split as 'valid' and others as 'val'.
    Try the user-requested mode first, then alias fallback when applicable.
    """
    candidates = [mode]
    if mode == 'valid':
        candidates.append('val')
    elif mode == 'val':
        candidates.append('valid')

    last_err = None
    for candidate_mode in candidates:
        try:
            ds = get_video_dataset(
                ds_name,
                root=root,
                seq_len=seq_len,
                mode=candidate_mode,
                image_size=image_size,
            )
            return ds, candidate_mode
        except Exception as exc:
            last_err = exc
            continue

    raise RuntimeError(
        f"Failed to load dataset split for mode='{mode}'. Tried {candidates}. Last error: {last_err}"
    )


def _create_pred_vs_gt_trajectory_video(
    pred_positions: np.ndarray,
    gt_positions: np.ndarray,
    save_path: str,
) -> None:
    """
    Create trajectory GIF in the same visualization style as eval_bounding_boxes.py
    with predicted vs target trajectories over time.
    """
    if imageio is None:
        raise ImportError(
            "imageio is required for monitor GIF visualizations. "
            "Install imageio in this environment or pass --monitor_visualizations 0."
        )

    pred_positions = np.asarray(pred_positions, dtype=np.float32)
    gt_positions = np.asarray(gt_positions, dtype=np.float32)

    if pred_positions.ndim != 3 or pred_positions.shape[-1] != 2:
        raise ValueError(f"Expected pred_positions [T,N,2], got shape={pred_positions.shape}")
    if gt_positions.ndim != 3 or gt_positions.shape[-1] != 2:
        raise ValueError(f"Expected gt_positions [T,N,2], got shape={gt_positions.shape}")

    T_pred, N_pred, _ = pred_positions.shape
    T_gt, N_gt, _ = gt_positions.shape
    T = min(T_pred, T_gt)
    if T == 0:
        raise ValueError("Cannot visualize empty trajectories.")

    pred_positions = pred_positions[:T]
    gt_positions = gt_positions[:T]

    all_pos = np.concatenate(
        [pred_positions.reshape(-1, 2), gt_positions.reshape(-1, 2)],
        axis=0,
    )
    x_min, x_max = all_pos[:, 0].min(), all_pos[:, 0].max()
    y_min, y_max = all_pos[:, 1].min(), all_pos[:, 1].max()

    x_range = x_max - x_min if x_max > x_min else 0.1
    y_range = y_max - y_min if y_max > y_min else 0.1
    max_range = max(x_range, y_range) * 1.2

    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    x_lim = [x_center - max_range / 2.0, x_center + max_range / 2.0]
    y_lim = [y_center - max_range / 2.0, y_center + max_range / 2.0]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_xlabel('X Position', fontsize=13, fontweight='bold')
    ax.set_ylabel('Y Position', fontsize=13, fontweight='bold')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=0.4, linewidth=0.8)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.82)

    frames = []
    colors = plt.cm.Set1(np.linspace(0, 1, max(N_pred, N_gt, 1)))

    for t in range(T):
        ax.clear()

        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_xlabel('X Position', fontsize=13, fontweight='bold')
        ax.set_ylabel('Y Position', fontsize=13, fontweight='bold')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(alpha=0.4, linewidth=0.8)
        ax.set_title(
            f'Predicted vs Ground-Truth Trajectories: Frame {t+1}/{T}\n(Circle=GT, X=Pred)',
            fontsize=14,
            fontweight='bold',
            pad=15,
        )

        for obj_idx in range(N_gt):
            gt_traj = gt_positions[:t+1, obj_idx, :]
            ax.plot(
                gt_traj[:, 0],
                gt_traj[:, 1],
                '-',
                color=colors[obj_idx],
                alpha=0.8,
                linewidth=3,
                label=f'GT {obj_idx+1}',
            )
            ax.scatter(
                gt_traj[-1, 0],
                gt_traj[-1, 1],
                c=[colors[obj_idx]],
                s=350,
                marker='o',
                alpha=0.95,
                edgecolors='black',
                linewidths=3,
                zorder=10,
            )

        for obj_idx in range(N_pred):
            pred_traj = pred_positions[:t+1, obj_idx, :]
            if t > 0:
                ax.plot(
                    pred_traj[:, 0],
                    pred_traj[:, 1],
                    '--',
                    color=colors[obj_idx],
                    alpha=0.85,
                    linewidth=3,
                    label=f'Pred {obj_idx+1}',
                )
            ax.scatter(
                pred_traj[-1, 0],
                pred_traj[-1, 1],
                c=[colors[obj_idx]],
                s=350,
                marker='x',
                alpha=0.95,
                linewidths=5,
                zorder=10,
            )

        ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)

    imageio.mimsave(save_path, frames, duration=100, loop=0)
    plt.close(fig)


def _pearson_1d_np(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"pearson inputs must have same shape, got {a.shape} vs {b.shape}")
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b)) + eps
    if denom <= eps:
        return 0.0
    return float(np.sum(a * b) / denom)


def _r2_score_np(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"R2 inputs must have same shape, got {y_true.shape} vs {y_pred.shape}")
    sse = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - y_true.mean()) ** 2) + eps
    return float(1.0 - sse / sst)


def _per_video_quality_metrics(pred_positions: np.ndarray, gt_positions: np.ndarray, eps: float = 1e-8) -> Tuple[float, float]:
    pred_positions = np.asarray(pred_positions, dtype=np.float64)
    gt_positions = np.asarray(gt_positions, dtype=np.float64)
    if pred_positions.ndim != 3 or gt_positions.ndim != 3:
        raise ValueError(
            f"Expected pred/gt shapes [T,N,2], got pred={pred_positions.shape}, gt={gt_positions.shape}"
        )
    if pred_positions.shape[-1] != 2 or gt_positions.shape[-1] != 2:
        raise ValueError(
            f"Expected last dim=2 for pred/gt, got pred={pred_positions.shape}, gt={gt_positions.shape}"
        )

    # Be robust to small shape mismatches by truncating to common support.
    t = min(pred_positions.shape[0], gt_positions.shape[0])
    n = min(pred_positions.shape[1], gt_positions.shape[1])
    if t <= 0 or n <= 0:
        return 0.0, 0.0

    pred = pred_positions[:t, :n]
    gt = gt_positions[:t, :n]

    rs = []
    for obj_idx in range(n):
        rs.append(_pearson_1d_np(pred[:, obj_idx, 0], gt[:, obj_idx, 0], eps=eps))
        rs.append(_pearson_1d_np(pred[:, obj_idx, 1], gt[:, obj_idx, 1], eps=eps))
    pearson_mean = float(np.mean(rs)) if len(rs) > 0 else 0.0
    r2 = _r2_score_np(gt, pred, eps=eps)
    return pearson_mean, r2


def _mean_std(values: List[float]) -> Tuple[float, float]:
    if len(values) == 0:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def _load_eval_monitor_helpers():
    global _EVAL_MONITOR_HELPERS
    if _EVAL_MONITOR_HELPERS is None:
        from eval.eval_bounding_boxes import evaluate_latent_alignment_metrics, video_to_trajectory

        _EVAL_MONITOR_HELPERS = {
            'evaluate_latent_alignment_metrics': evaluate_latent_alignment_metrics,
            'video_to_trajectory': video_to_trajectory,
        }
    return _EVAL_MONITOR_HELPERS


def _compute_monitor_alignment_metrics(
    pred_coordinates: np.ndarray,
    gt_coordinates: np.ndarray,
    mode: str,
    matching_mode: str,
    reorder_method: str,
):
    helpers = _load_eval_monitor_helpers()
    evaluate_latent_alignment_metrics = helpers['evaluate_latent_alignment_metrics']
    use_hungarian = matching_mode == 'hungarian'
    with contextlib.redirect_stdout(io.StringIO()):
        return evaluate_latent_alignment_metrics(
            pred_coordinates,
            gt_coordinates,
            save_dir=None,
            mode=mode,
            extraction_method='latent',
            use_hungarian_for_correlation=use_hungarian,
            reorder_method=reorder_method,
        )


def _monitor_prob_encoder_eval_route(args) -> str:
    if (
        args.objective == 'c1_dyn'
        and args.trainable_attribute_set == 'position_scale_depth_obj_on'
        and args.position_head_training == 'full'
    ):
        return 'c1-dyn-attrs'
    return 'c1'


def _alignment_metrics_to_record(
    *,
    metric_family: str,
    matching: str,
    fit_family: str,
    mode: str,
    num_videos: int,
    num_batches: int,
    metrics_out: Dict[str, object],
) -> Dict[str, object]:
    block = metrics_out[fit_family]
    return {
        'metric_family': metric_family,
        'matching': matching,
        'fit_family': fit_family,
        'mode': mode,
        'num_videos': int(num_videos),
        'num_batches': int(num_batches),
        'pearson_pos_mean': float(block['mean_pearson_pos_mean']),
        'pearson_pos_std': float(block['mean_pearson_pos_std']),
        'pearson_vel_mean': float(block['mean_pearson_vel_mean']),
        'pearson_vel_std': float(block['mean_pearson_vel_std']),
        'pearson_acc_mean': float(block['mean_pearson_acc_mean']),
        'pearson_acc_std': float(block['mean_pearson_acc_std']),
        'acc_rms_ratio_mean': float(block['smoothness_ratio_acc_rms_mean']),
        'acc_rms_ratio_std': float(block['smoothness_ratio_acc_rms_std']),
        'r2_mean': float(block['r2_mean']),
        'r2_std': float(block['r2_std']),
    }


def normalize_gt_positions(gt_pos: torch.Tensor, image_size: int, mode: str) -> torch.Tensor:
    """
    Map ground-truth positions to DDLP coordinate range [-1, 1].
    """
    if mode not in {'auto', 'pixel', 'zero_one', 'minus_one_one', 'none', 'ddlp_similarity'}:
        raise ValueError(f"Unknown gt normalization mode: {mode}")

    gt = gt_pos.float()

    if mode in {'minus_one_one', 'none'}:
        return gt
    if mode == 'zero_one':
        return gt * 2.0 - 1.0
    if mode == 'ddlp_similarity':
        raise ValueError(
            "mode='ddlp_similarity' requires a fitted transform. "
            "Call apply_gt_similarity_transform(...) instead."
        )
    if mode == 'pixel':
        denom = max(float(image_size - 1), 1.0)
        return 2.0 * (gt / denom) - 1.0

    # auto
    gmin = float(gt.min().item())
    gmax = float(gt.max().item())
    if gmin >= -1.2 and gmax <= 1.2:
        return gt
    if gmin >= -0.1 and gmax <= 1.1:
        return gt * 2.0 - 1.0
    denom = max(float(image_size - 1), 1.0)
    return 2.0 * (gt / denom) - 1.0


def apply_gt_similarity_transform(
    gt_pos: torch.Tensor,
    similarity_transform: Optional[Dict[str, object]],
) -> torch.Tensor:
    if similarity_transform is None:
        raise ValueError("DDLP similarity transform is required for mode='ddlp_similarity'.")
    src_center = torch.as_tensor(
        similarity_transform['source_center'],
        device=gt_pos.device,
        dtype=gt_pos.dtype,
    )
    ref_center = torch.as_tensor(
        similarity_transform['reference_center'],
        device=gt_pos.device,
        dtype=gt_pos.dtype,
    )
    scale = float(similarity_transform['scale'])
    return ref_center + scale * (gt_pos - src_center)


def _compute_similarity_transform(
    source_points: np.ndarray,
    reference_points: np.ndarray,
    context: str,
) -> Dict[str, object]:
    if source_points.ndim != 2 or source_points.shape[1] != 2:
        raise ValueError(f"{context}: expected source points with shape [N,2], got {source_points.shape}")
    if reference_points.ndim != 2 or reference_points.shape[1] != 2:
        raise ValueError(f"{context}: expected reference points with shape [N,2], got {reference_points.shape}")
    if source_points.shape[0] == 0 or reference_points.shape[0] == 0:
        raise ValueError(f"{context}: source/reference point clouds must be non-empty.")

    source_center = source_points.mean(axis=0)
    reference_center = reference_points.mean(axis=0)
    source_rms = float(np.sqrt(np.mean(np.sum((source_points - source_center) ** 2, axis=1))))
    reference_rms = float(np.sqrt(np.mean(np.sum((reference_points - reference_center) ** 2, axis=1))))
    if not np.isfinite(source_rms) or source_rms <= 0.0:
        raise ValueError(f"{context}: invalid source RMS scale {source_rms!r}.")
    if not np.isfinite(reference_rms) or reference_rms <= 0.0:
        raise ValueError(f"{context}: invalid reference RMS scale {reference_rms!r}.")

    return {
        'source_center': source_center.astype(np.float64).tolist(),
        'reference_center': reference_center.astype(np.float64).tolist(),
        'scale': float(reference_rms / source_rms),
    }


def _resolve_gt_similarity_reference(
    requested_reference: Optional[str],
    checkpoint_dir: str,
) -> Optional[str]:
    if requested_reference not in {None, ''}:
        return os.path.abspath(requested_reference)
    candidate = os.path.join(
        checkpoint_dir,
        'extraction_evaluation',
        'best',
        'extracted_datasets',
        'ddlp_extracted_recentered_training.h5.1',
    )
    if os.path.exists(candidate):
        return candidate
    return None


def _load_reference_points_from_hdf(reference_path: str) -> np.ndarray:
    with h5py.File(reference_path, 'r') as f:
        grp = f['features']
        cols = [c.decode() if isinstance(c, bytes) else str(c) for c in grp['block0_items'][()]]
        vals = np.asarray(grp['block0_values'][()], dtype=np.float64)
    q_idx = [idx for idx, name in enumerate(cols) if name.startswith('q_')]
    if not q_idx:
        raise ValueError(
            f"Reference dataset '{reference_path}' does not contain any q_* position columns."
        )
    q_cols = [cols[idx] for idx in q_idx]
    name_to_local = {name: idx for idx, name in enumerate(q_cols)}
    x_cols = [name for name in q_cols if name.endswith('_x') and f"{name[:-2]}_y" in name_to_local]
    if not x_cols:
        raise ValueError(
            f"Reference dataset '{reference_path}' does not contain any paired q_*_x/q_*_y columns."
        )
    x_idx = [cols.index(name) for name in x_cols]
    y_idx = [cols.index(f"{name[:-2]}_y") for name in x_cols]
    pts = np.stack([vals[:, x_idx], vals[:, y_idx]], axis=-1).reshape(-1, 2)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] == 0:
        raise ValueError(f"Reference dataset '{reference_path}' did not yield any finite x/y points.")
    return pts


def _load_dataset_points_for_similarity(dataset) -> np.ndarray:
    if hasattr(dataset, 'file') and isinstance(getattr(dataset, 'file'), str) and os.path.exists(dataset.file):
        try:
            with h5py.File(dataset.file, 'r') as f:
                if 'positions' in f:
                    pos = np.asarray(f['positions'][()], dtype=np.float64)
                    in_camera = np.asarray(f['in_camera'][()], dtype=np.float64) if 'in_camera' in f else None
                    if getattr(dataset, 'mode', None) == 'train':
                        if getattr(dataset, 'use_subsequences', False):
                            max_frames = int(getattr(dataset, 'max_train_frames', pos.shape[1]))
                        else:
                            max_frames = int(getattr(dataset, 'max_eval_frames', pos.shape[1]))
                    else:
                        max_frames = int(getattr(dataset, 'max_eval_frames', pos.shape[1]))
                    pos = pos[:, :max_frames]
                    if in_camera is not None:
                        in_camera = in_camera[:, :max_frames]
                    pts = pos.reshape(-1, pos.shape[-1])
                    if in_camera is not None and in_camera.shape == pos.shape[:-1]:
                        pts = pts[in_camera.reshape(-1) > 0.5]
                    pts = pts[np.isfinite(pts).all(axis=1)]
                    if pts.shape[0] > 0:
                        return pts
        except Exception:
            pass

    pts_list = []
    for idx in range(len(dataset)):
        batch = dataset[idx]
        if len(batch) < 2:
            continue
        pos = np.asarray(batch[1], dtype=np.float64)
        if pos.ndim == 2 and pos.shape[-1] == 2:
            pts = pos.reshape(-1, 2)
        elif pos.ndim >= 3 and pos.shape[-1] == 2:
            pts = pos.reshape(-1, 2)
        else:
            continue
        if len(batch) >= 5:
            in_camera = np.asarray(batch[4], dtype=np.float64)
            if in_camera.shape == pos.shape[:-1]:
                pts = pts[in_camera.reshape(-1) > 0.5]
        pts = pts[np.isfinite(pts).all(axis=1)]
        if pts.shape[0] > 0:
            pts_list.append(pts)

    if not pts_list:
        raise ValueError("Failed to collect any finite dataset points for DDLP similarity fitting.")
    return np.concatenate(pts_list, axis=0)


def fit_ddlp_similarity_transform(
    dataset,
    reference_path: str,
) -> Dict[str, object]:
    source_points = _load_dataset_points_for_similarity(dataset)
    reference_points = _load_reference_points_from_hdf(reference_path)
    transform = _compute_similarity_transform(
        source_points=source_points,
        reference_points=reference_points,
        context='train_probabilistic_encoder_route_f',
    )
    transform['reference_path'] = os.path.abspath(reference_path)
    return transform


def maybe_swap_xy(xy: torch.Tensor, swap_xy: bool) -> torch.Tensor:
    if not swap_xy:
        return xy
    return xy[..., [1, 0]]


def _build_local_grid(h: int, w: int, device: torch.device, dtype: torch.dtype):
    y_lin = torch.linspace(-1.0, 1.0, steps=h, device=device, dtype=dtype)
    x_lin = torch.linspace(-1.0, 1.0, steps=w, device=device, dtype=dtype)
    try:
        yy, xx = torch.meshgrid(y_lin, x_lin, indexing='ij')
    except TypeError:
        yy, xx = torch.meshgrid(y_lin, x_lin)
    yy = yy.view(1, 1, 1, h, w)
    xx = xx.view(1, 1, 1, h, w)
    return yy, xx


def _alpha_centroid_yx(alpha_maps: torch.Tensor, eps: float = 1e-6):
    if alpha_maps.ndim != 5:
        raise ValueError(f"Expected alpha_maps [B,T,N,H,W], got shape={tuple(alpha_maps.shape)}")
    h, w = alpha_maps.shape[-2], alpha_maps.shape[-1]
    yy, xx = _build_local_grid(h, w, alpha_maps.device, alpha_maps.dtype)
    mass = alpha_maps.sum(dim=(-2, -1))
    denom = mass + eps
    mu_y = (alpha_maps * yy).sum(dim=(-2, -1)) / denom
    mu_x = (alpha_maps * xx).sum(dim=(-2, -1)) / denom
    centroid_yx = torch.stack([mu_y, mu_x], dim=-1)
    valid_mask = mass > eps
    return centroid_yx, valid_mask


def recenter_patch_alpha(mu_tot: torch.Tensor, mu_scale: torch.Tensor, dec_objects_original: torch.Tensor, eps: float = 1e-6):
    if dec_objects_original.ndim != 6:
        raise ValueError(f"Expected dec_objects_original [B,T,N,4,H,W], got shape={tuple(dec_objects_original.shape)}")
    alpha_patch = dec_objects_original[:, :, :, 0, :, :]
    mu_local_yx, valid_mask = _alpha_centroid_yx(alpha_patch, eps=eps)
    scale_norm = torch.sigmoid(mu_scale)
    delta = scale_norm * mu_local_yx
    delta = torch.where(valid_mask.unsqueeze(-1), delta, torch.zeros_like(delta))
    return mu_tot + delta, valid_mask


def recenter_global_alpha(mu_tot: torch.Tensor, alpha_masks: torch.Tensor, eps: float = 1e-6):
    if alpha_masks.ndim != 6:
        raise ValueError(f"Expected alpha_masks [B,T,N,1,H,W], got shape={tuple(alpha_masks.shape)}")
    alpha_global = alpha_masks.squeeze(3)
    mu_global_yx, valid_mask = _alpha_centroid_yx(alpha_global, eps=eps)
    mu_tot_rec = torch.where(valid_mask.unsqueeze(-1), mu_global_yx, mu_tot)
    return mu_tot_rec, valid_mask


def compute_temporal_permutations(pred: torch.Tensor, method: str) -> torch.Tensor:
    """
    pred: [B, T, N, 2] (detached)
    returns perms: [B, T, N] of indices applied to slot dimension.
    """
    if method == 'none':
        b, t, n, _ = pred.shape
        base = torch.arange(n, device=pred.device, dtype=torch.long)
        return base.view(1, 1, n).repeat(b, t, 1)

    if method not in {'smallest_consecutive_distance', 'hungarian'}:
        raise ValueError(f"Unknown temporal reorder method: {method}")

    bsz, tsz, nslots, _ = pred.shape
    perms = []

    for b in range(bsz):
        seq = pred[b]  # [T, N, 2]
        perm_seq = [torch.arange(nslots, device=pred.device, dtype=torch.long)]

        if method == 'smallest_consecutive_distance':
            # two-body-oriented heuristic, generalized with swap against slot 0
            tracked = seq[0, 0]
            for t in range(1, tsz):
                d = torch.norm(seq[t] - tracked, dim=-1)
                closest_idx = int(torch.argmin(d).item())
                p = torch.arange(nslots, device=pred.device, dtype=torch.long)
                if closest_idx != 0:
                    p[0], p[closest_idx] = p[closest_idx].clone(), p[0].clone()
                perm_seq.append(p)
                tracked = seq[t, p[0]]
        else:
            reordered_prev = seq[0]
            for t in range(1, tsz):
                curr = seq[t]
                cost = torch.cdist(reordered_prev.unsqueeze(0), curr.unsqueeze(0)).squeeze(0)
                row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
                p = torch.as_tensor(col_ind, device=pred.device, dtype=torch.long)
                perm_seq.append(p)
                reordered_prev = curr[p]

        perms.append(torch.stack(perm_seq, dim=0))

    return torch.stack(perms, dim=0)  # [B, T, N]


def apply_temporal_permutation(x: torch.Tensor, perms: torch.Tensor) -> torch.Tensor:
    """x: [B,T,N,D], perms: [B,T,N]"""
    if x.ndim != 4:
        raise ValueError(f"Expected x [B,T,N,D], got {tuple(x.shape)}")
    idx = perms.unsqueeze(-1).expand(-1, -1, -1, x.shape[-1])
    return torch.gather(x, dim=2, index=idx)


def sequence_level_hungarian_assign(
    pred: torch.Tensor,
    gt: torch.Tensor,
    gt_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    pred: [B,T,Np,2]
    gt:   [B,T,Ng,2]
    gt_mask: [B,T,Ng] or None

    returns assignment indices in pred-space ordered by gt index:
      assign[b, j] = predicted slot index matched to gt slot j
    """
    bsz, tsz, npred, _ = pred.shape
    _, _, ngt, _ = gt.shape

    if npred < ngt:
        raise ValueError(f"Need npred >= ngt for matching, got npred={npred}, ngt={ngt}")

    assign = torch.empty((bsz, ngt), dtype=torch.long, device=pred.device)

    for b in range(bsz):
        pb = pred[b]  # [T,Np,2]
        gb = gt[b]    # [T,Ng,2]

        # [Np,Ng]
        cost = torch.empty((npred, ngt), dtype=pred.dtype, device=pred.device)

        for i in range(npred):
            # [T,Ng,2]
            diff = pb[:, i].unsqueeze(1) - gb
            sq = (diff ** 2).sum(dim=-1)  # [T,Ng]
            if gt_mask is not None:
                mb = gt_mask[b].float()  # [T,Ng]
                num = (sq * mb).sum(dim=0)
                den = mb.sum(dim=0).clamp_min(1.0)
                c = num / den
            else:
                c = sq.mean(dim=0)
            cost[i] = c

        row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
        row_ind = torch.as_tensor(row_ind, device=pred.device, dtype=torch.long)
        col_ind = torch.as_tensor(col_ind, device=pred.device, dtype=torch.long)

        # Re-index so order is by gt index [0..Ng-1].
        gt_to_pred = torch.empty((ngt,), dtype=torch.long, device=pred.device)
        gt_to_pred[col_ind] = row_ind
        assign[b] = gt_to_pred

    return assign


def gather_slots(x: torch.Tensor, slot_idx: torch.Tensor) -> torch.Tensor:
    """
    x: [B,T,N,D], slot_idx: [B,K]
    returns: [B,T,K,D]
    """
    if x.ndim != 4:
        raise ValueError(f"Expected x [B,T,N,D], got {tuple(x.shape)}")
    idx = slot_idx.unsqueeze(1).unsqueeze(-1).expand(-1, x.shape[1], -1, x.shape[-1])
    return torch.gather(x, dim=2, index=idx)


def gaussian_nll(mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None):
    """
    mean/logvar/target: [B,T,N,2]
    mask: [B,T,N] or None
    """
    inv_var = torch.exp(-logvar)
    nll_dim = 0.5 * ((target - mean) ** 2 * inv_var + logvar + LOG_2PI)
    nll = nll_dim.sum(dim=-1)  # [B,T,N]

    if mask is not None:
        m = mask.float()
        denom = m.sum().clamp_min(1.0)
        loss = (nll * m).sum() / denom
        mse = (((target - mean) ** 2).sum(dim=-1) * m).sum() / denom
    else:
        loss = nll.mean()
        mse = ((target - mean) ** 2).sum(dim=-1).mean()
    return loss, mse


def resolve_lagrangian_checkpoint_path(path: str) -> Tuple[str, str]:
    path = os.path.abspath(path)
    if os.path.isfile(path):
        config_path = os.path.join(os.path.dirname(path), 'config.json')
        return path, config_path
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Fixed Lagrangian path not found: {path}")

    preferred = os.path.join(path, 'separable_hessian', 'seed_42', 'best_model.pt')
    if os.path.exists(preferred):
        return preferred, os.path.join(os.path.dirname(preferred), 'config.json')

    candidates = sorted(glob.glob(os.path.join(path, '**', 'best_model.pt'), recursive=True))
    if not candidates:
        raise FileNotFoundError(f"No best_model.pt found under fixed Lagrangian directory: {path}")

    ckpt_path = candidates[0]
    return ckpt_path, os.path.join(os.path.dirname(ckpt_path), 'config.json')


def load_fixed_lagrangian(
    checkpoint_path_or_dir: str,
    explicit_config_path: Optional[str],
    device: torch.device,
):
    if SeparableLagrangianMLP is None:
        raise ImportError(
            "Could not import del_pytorch.models.separable_lagrangian.SeparableLagrangianMLP. "
            f"Expected DEL parent path: {DEL_PARENT}"
        )

    ckpt_path, inferred_config_path = resolve_lagrangian_checkpoint_path(checkpoint_path_or_dir)
    config_path = os.path.abspath(explicit_config_path) if explicit_config_path else inferred_config_path
    cfg = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            cfg = json.load(f)

    checkpoint = torch.load(ckpt_path, map_location=device)
    if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint:
        raise ValueError(f"Fixed Lagrangian checkpoint does not contain model_state_dict: {ckpt_path}")

    state_dim = int(cfg.get('state_dim', checkpoint.get('state_dim', 4)))
    model_class = str(cfg.get('model_class', checkpoint.get('model_class', 'SeparableLagrangianMLP')))
    if model_class != 'SeparableLagrangianMLP':
        raise ValueError(
            f"C1-dyn expects a SeparableLagrangianMLP checkpoint, got model_class={model_class!r}"
        )

    model = SeparableLagrangianMLP(
        state_dim=state_dim,
        hidden_dims=tuple(cfg.get('hidden_dims', [128, 128, 128])),
        activation=cfg.get('activation', 'softplus'),
        dropout=float(cfg.get('dropout', 0.0)),
        h=float(cfg.get('mass_h', cfg.get('h', 1.0))),
        mass_eps=float(cfg.get('mass_eps', 1e-4)),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, ckpt_path, config_path, cfg


def latent_yx_to_d6_true_world(
    q_yx: torch.Tensor,
    args,
    gt_similarity_transform: Optional[Dict[str, object]] = None,
) -> torch.Tensor:
    """
    Fixed T_C1->phys used by C1-dyn.

    DDLP positions are [y,x] in [-1,1]. The transform first maps to
    D6-TRUE-PIX pixel centres, then applies the inverse renderer map to
    D6-TRUE world coordinates.
    """
    if q_yx.shape[-1] != 2:
        raise ValueError(f"Expected q_yx last dim=2, got shape={tuple(q_yx.shape)}")

    height = float(args.dyn_image_height)
    width = float(args.dyn_image_width)
    radius = float(args.dyn_render_radius_px)
    denom_x = max(width - 1.0 - 2.0 * radius, 1e-6)
    denom_y = max(height - 1.0 - 2.0 * radius, 1e-6)

    mode = getattr(args, 'dyn_latent_to_pixel_mode', 'auto')
    use_similarity_inverse = (
        mode == 'ddlp_similarity_inverse'
        or (mode == 'auto' and gt_similarity_transform is not None)
    )
    if use_similarity_inverse:
        if gt_similarity_transform is None:
            raise ValueError(
                "dyn_latent_to_pixel_mode='ddlp_similarity_inverse' requires "
                "gt_normalization='ddlp_similarity' with a fitted transform."
            )
        q_xy = q_yx[..., [1, 0]]
        source_center = torch.as_tensor(
            gt_similarity_transform['source_center'],
            device=q_yx.device,
            dtype=q_yx.dtype,
        )
        reference_center = torch.as_tensor(
            gt_similarity_transform['reference_center'],
            device=q_yx.device,
            dtype=q_yx.dtype,
        )
        scale = float(gt_similarity_transform['scale'])
        pixel_xy = source_center + (q_xy - reference_center) / max(scale, 1e-12)
        x_px = pixel_xy[..., 0]
        y_px = pixel_xy[..., 1]
    elif mode in {'auto', 'normalized_grid'}:
        y_norm = q_yx[..., 0]
        x_norm = q_yx[..., 1]
        x_px = 0.5 * (x_norm + 1.0) * (width - 1.0)
        y_px = 0.5 * (y_norm + 1.0) * (height - 1.0)
    else:
        raise ValueError(f"Unknown dyn_latent_to_pixel_mode: {mode}")

    x_min = float(args.dyn_world_x_min)
    x_max = float(args.dyn_world_x_max)
    y_min = float(args.dyn_world_y_min)
    y_max = float(args.dyn_world_y_max)

    x_world = x_min + ((x_px - radius) / denom_x) * (x_max - x_min)
    y_world = y_min + (1.0 - ((y_px - radius) / denom_y)) * (y_max - y_min)
    return torch.stack([x_world, y_world], dim=-1)


def matched_latent_yx_to_d6_true_state(
    q_yx: torch.Tensor,
    args,
    gt_similarity_transform: Optional[Dict[str, object]] = None,
) -> torch.Tensor:
    """
    Convert matched [B,T,N,2] DDLP y/x latents to [B,T,4] D6-TRUE state
    ordered as [q_A_x, q_A_y, q_B_x, q_B_y].
    """
    if q_yx.ndim != 4:
        raise ValueError(f"Expected q_yx [B,T,N,2], got shape={tuple(q_yx.shape)}")
    if int(q_yx.shape[2]) != 2:
        raise ValueError(
            "The fixed D6-TRUE Lagrangian checkpoint expects exactly two matched bodies. "
            f"Got N={int(q_yx.shape[2])}."
        )
    world_xy = latent_yx_to_d6_true_world(
        q_yx,
        args,
        gt_similarity_transform=gt_similarity_transform,
    )
    return world_xy.reshape(world_xy.shape[0], world_xy.shape[1], -1)


def differentiable_del_residual(
    lagrangian_model: torch.nn.Module,
    q_prev: torch.Tensor,
    q_curr: torch.Tensor,
    q_next: torch.Tensor,
) -> torch.Tensor:
    ld_prev = lagrangian_model(q_prev, q_curr)
    d2_prev = torch.autograd.grad(
        ld_prev.sum(),
        q_curr,
        create_graph=True,
        retain_graph=True,
    )[0]

    ld_next = lagrangian_model(q_curr, q_next)
    d1_next = torch.autograd.grad(
        ld_next.sum(),
        q_curr,
        create_graph=True,
        retain_graph=True,
    )[0]
    return d2_prev + d1_next


def weak_del_residual_loss(
    lagrangian_model: torch.nn.Module,
    q_seq: torch.Tensor,
    weak_window: int = 0,
) -> torch.Tensor:
    if q_seq.ndim != 3:
        raise ValueError(f"Expected q_seq [B,T,D], got shape={tuple(q_seq.shape)}")
    if q_seq.shape[1] < 3:
        return q_seq.new_tensor(0.0)

    q_prev = q_seq[:, :-2, :]
    q_curr = q_seq[:, 1:-1, :]
    q_next = q_seq[:, 2:, :]
    residual = differentiable_del_residual(lagrangian_model, q_prev, q_curr, q_next)

    if weak_window > 0 and residual.shape[1] > 1:
        kernel_size = 2 * int(weak_window) + 1
        pad = int(weak_window)
        channels = residual.shape[-1]
        res_ch = residual.transpose(1, 2)
        padded = torch.nn.functional.pad(res_ch, (pad, pad), mode='replicate')
        kernel = residual.new_full((channels, 1, kernel_size), 1.0 / float(kernel_size))
        residual = torch.nn.functional.conv1d(padded, kernel, groups=channels).transpose(1, 2)

    return residual.pow(2).sum(dim=-1).mean()


def encode_sequential(model: ObjectDynamicsDLP, x: torch.Tensor, num_static_frames: int):
    """
    Uses DDLP sequential encoder scaffold directly, deterministic mode.
    Returns tensors in [B,T,N,...] (reshape=False).
    """
    return model.fg_sequential_opt(
        x,
        deterministic=True,
        x_prior=x,
        warmup=False,
        noisy=False,
        reshape=False,
        train_prior=False,
        num_static_frames=num_static_frames,
        continuation_state=None,
    )


def infer_temporal_reorder_method(method_arg: str, gt_n: int) -> str:
    if method_arg != 'auto':
        return method_arg
    if gt_n == 2:
        return 'smallest_consecutive_distance'
    return 'hungarian'


def build_route_mean(
    route: str,
    train_out: Dict[str, torch.Tensor],
    frozen_out: Optional[Dict[str, torch.Tensor]],
    recenter_source: str,
    recenter_eps: float,
    f1_delta_mode: str,
    kp_min: float,
    kp_max: float,
) -> torch.Tensor:
    mu_nom_train = train_out['z_base'] + train_out['mu_offset']

    if route != 'f1':
        raise ValueError("Route F.2/C2 has been removed; only route='f1' is supported.")

    if frozen_out is None:
        raise RuntimeError("Route F.1 requires frozen_out")

    mu_nom_frozen = frozen_out['z_base'] + frozen_out['mu_offset']

    if recenter_source == 'patch_alpha':
        c0, _ = recenter_patch_alpha(
            mu_nom_frozen,
            frozen_out['mu_scale'],
            frozen_out['dec_objects_original'],
            eps=recenter_eps,
        )
    elif recenter_source == 'global_alpha':
        c0, _ = recenter_global_alpha(
            mu_nom_frozen,
            frozen_out['alpha_masks'],
            eps=recenter_eps,
        )
    else:
        raise ValueError(f"Unknown recenter_source: {recenter_source}")

    c0 = c0.detach()

    if f1_delta_mode == 'relative_to_frozen_nominal':
        delta = mu_nom_train - mu_nom_frozen.detach()
    elif f1_delta_mode == 'direct_trainable_nominal':
        delta = mu_nom_train
    else:
        raise ValueError(f"Unknown f1_delta_mode: {f1_delta_mode}")

    return (c0 + delta).clamp(min=kp_min, max=kp_max)


def one_pass(
    model: ObjectDynamicsDLP,
    frozen_model: Optional[ObjectDynamicsDLP],
    batch,
    device: torch.device,
    args,
    config: Dict,
    gt_similarity_transform: Optional[Dict[str, object]] = None,
    lagrangian_model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_masks: Optional[Dict[torch.nn.Parameter, torch.Tensor]] = None,
    adaptive_beta_state: Optional[Dict[str, float]] = None,
    return_trajectories: bool = False,
    compute_dynamics_loss: bool = True,
):
    train_mode = optimizer is not None

    x = batch[0].to(device)
    if len(batch) < 2:
        raise RuntimeError(
            "Dataset batch does not contain ground-truth positions at index 1. "
            "This trainer requires video+trajectory datasets."
        )
    gt_pos = torch.as_tensor(batch[1], device=device, dtype=torch.float32)
    in_cam = None
    if args.use_in_camera_mask and len(batch) >= 5:
        in_cam = torch.as_tensor(batch[4], device=device, dtype=torch.float32)

    if args.gt_normalization == 'ddlp_similarity':
        gt_pos = apply_gt_similarity_transform(gt_pos, similarity_transform=gt_similarity_transform)
    else:
        gt_pos = normalize_gt_positions(gt_pos, image_size=config['image_size'], mode=args.gt_normalization)
    gt_pos = maybe_swap_xy(gt_pos, args.swap_gt_xy)

    if train_mode:
        optimizer.zero_grad(set_to_none=True)

    train_out = encode_sequential(model, x, num_static_frames=args.num_static_frames)

    frozen_out = None
    if args.route == 'f1':
        if frozen_model is None:
            raise RuntimeError("Route F.1 requested but frozen model is missing")
        with torch.no_grad():
            frozen_out = encode_sequential(frozen_model, x, num_static_frames=args.num_static_frames)

    kp_min, kp_max = float(config['kp_range'][0]), float(config['kp_range'][1])
    pred_mean = build_route_mean(
        route=args.route,
        train_out=train_out,
        frozen_out=frozen_out,
        recenter_source=args.recenter_source,
        recenter_eps=args.recenter_eps,
        f1_delta_mode=args.f1_delta_mode,
        kp_min=kp_min,
        kp_max=kp_max,
    )
    pred_logvar = train_out['logvar_offset']

    # Optional detached temporal reordering (before sequence-level Hungarian).
    gt_n = int(gt_pos.shape[2])
    reorder_method = infer_temporal_reorder_method(args.temporal_reorder, gt_n)
    if reorder_method != 'none':
        perms = compute_temporal_permutations(pred_mean.detach(), reorder_method)
        pred_mean = apply_temporal_permutation(pred_mean, perms)
        pred_logvar = apply_temporal_permutation(pred_logvar, perms)

    # Sequence-level Hungarian matching (pred slots -> GT identities).
    seq_assign = sequence_level_hungarian_assign(pred_mean.detach(), gt_pos.detach(), gt_mask=in_cam)
    pred_mean = gather_slots(pred_mean, seq_assign)
    pred_logvar = gather_slots(pred_logvar, seq_assign)

    # Gaussian NLL with DDLP position posterior variance.
    position_loss, mse = gaussian_nll(pred_mean, pred_logvar, gt_pos, mask=in_cam)
    dynamics_loss = pred_mean.new_tensor(0.0)
    beta_eff = 0.0
    beta_gate = 0.0
    fit_quality = 0.0
    pos_grad_norm = 0.0
    del_grad_norm = 0.0
    beta_ratio = 0.0
    pos_grad_ema = 0.0
    del_grad_ema = 0.0
    fit_mse_ema = 0.0
    fit_mse_ref = 0.0
    if args.objective == 'c1_dyn' and compute_dynamics_loss:
        if lagrangian_model is None:
            raise RuntimeError("objective='c1_dyn' requires a loaded fixed Lagrangian model")
        q_dyn = matched_latent_yx_to_d6_true_state(
            pred_mean,
            args,
            gt_similarity_transform=gt_similarity_transform,
        )
        dynamics_loss = weak_del_residual_loss(
            lagrangian_model=lagrangian_model,
            q_seq=q_dyn,
            weak_window=args.dyn_weak_window,
        )

    if args.objective == 'c1_dyn':
        if train_mode:
            trainable_params = get_trainable_parameters(model)
            pos_grad_norm = float(
                _masked_grad_norm(position_loss, trainable_params, grad_masks=grad_masks).detach().cpu().item()
            )
            del_grad_norm = float(
                _masked_grad_norm(dynamics_loss, trainable_params, grad_masks=grad_masks).detach().cpu().item()
            )
            if adaptive_beta_state is None:
                raise RuntimeError("objective='c1_dyn' requires adaptive_beta_state during training.")
            beta_terms = _update_adaptive_beta_state(
                adaptive_beta_state,
                c=float(args.beta),
                ema_decay=float(args.beta_ema_decay),
                eps=float(args.beta_eps),
                position_mse=float(mse.detach().cpu().item()),
                pos_grad_norm=pos_grad_norm,
                del_grad_norm=del_grad_norm,
            )
        else:
            beta_terms = _current_adaptive_beta_terms(adaptive_beta_state)

        beta_eff = float(beta_terms['beta_eff'])
        beta_gate = float(beta_terms['gate'])
        fit_quality = float(beta_terms['fit_quality'])
        beta_ratio = float(beta_terms['beta_ratio'])
        pos_grad_ema = float(beta_terms['pos_grad_ema'])
        del_grad_ema = float(beta_terms['del_grad_ema'])
        fit_mse_ema = float(beta_terms['fit_mse_ema'])
        fit_mse_ref = float(beta_terms['fit_mse_ref'])
        loss = position_loss + beta_eff * dynamics_loss
    else:
        loss = position_loss

    if train_mode:
        loss.backward()
        _apply_gradient_masks(grad_masks)
        if args.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(get_trainable_parameters(model), args.grad_clip)
        optimizer.step()

    with torch.no_grad():
        mean_abs_err = torch.abs(pred_mean - gt_pos)
        if in_cam is not None:
            denom = in_cam.sum().clamp_min(1.0)
            mae = (mean_abs_err.sum(dim=-1) * in_cam).sum() / denom
        else:
            mae = mean_abs_err.sum(dim=-1).mean()
        mean_std = torch.exp(0.5 * pred_logvar).mean()

    out = {
        'loss': float(loss.detach().cpu().item()),
        'position_loss': float(position_loss.detach().cpu().item()),
        'dynamics_loss': float(dynamics_loss.detach().cpu().item()),
        'mse': float(mse.detach().cpu().item()),
        'mae_l1_xy': float(mae.detach().cpu().item()),
        'mean_pred_std': float(mean_std.detach().cpu().item()),
        'reorder_method': reorder_method,
        'beta_eff': float(beta_eff),
        'beta_gate': float(beta_gate),
        'fit_quality': float(fit_quality),
        'beta_ratio': float(beta_ratio),
        'pos_grad_norm': float(pos_grad_norm),
        'del_grad_norm': float(del_grad_norm),
        'pos_grad_ema': float(pos_grad_ema),
        'del_grad_ema': float(del_grad_ema),
        'fit_mse_ema': float(fit_mse_ema),
        'fit_mse_ref': float(fit_mse_ref),
    }
    if return_trajectories:
        out['pred_mean'] = pred_mean.detach().cpu()
        out['gt_pos'] = gt_pos.detach().cpu()
    return out


def collect_visualization_samples(
    model: ObjectDynamicsDLP,
    frozen_model: Optional[ObjectDynamicsDLP],
    loader: DataLoader,
    device: torch.device,
    args,
    config: Dict,
    gt_similarity_transform: Optional[Dict[str, object]],
    k: int,
):
    samples = []
    if k <= 0:
        return samples

    model.eval()
    with torch.no_grad():
        for batch in loader:
            metrics = one_pass(
                model=model,
                frozen_model=frozen_model,
                batch=batch,
                device=device,
                args=args,
                config=config,
                gt_similarity_transform=gt_similarity_transform,
                optimizer=None,
                return_trajectories=True,
                compute_dynamics_loss=False,
            )
            pred_batch = metrics['pred_mean'].numpy()
            gt_batch = metrics['gt_pos'].numpy()
            bsz = pred_batch.shape[0]
            for bi in range(bsz):
                samples.append((pred_batch[bi], gt_batch[bi]))
                if len(samples) >= k:
                    return samples
    return samples


def collect_visualization_samples_and_metrics(
    model: ObjectDynamicsDLP,
    frozen_model: Optional[ObjectDynamicsDLP],
    loader: DataLoader,
    device: torch.device,
    args,
    config: Dict,
    gt_similarity_transform: Optional[Dict[str, object]],
    k: int,
    monitor_mode: str,
    max_batches: Optional[int] = None,
    compute_metrics: bool = True,
):
    samples = []
    pred_batches = []
    gt_batches = []
    num_batches = 0

    model.eval()
    with torch.no_grad():
        for batch in loader:
            if max_batches is not None and num_batches >= max_batches:
                break
            num_batches += 1
            metrics = one_pass(
                model=model,
                frozen_model=frozen_model,
                batch=batch,
                device=device,
                args=args,
                config=config,
                gt_similarity_transform=gt_similarity_transform,
                optimizer=None,
                return_trajectories=True,
                compute_dynamics_loss=False,
            )
            pred_batch = metrics['pred_mean'].numpy()
            gt_batch = metrics['gt_pos'].numpy()
            if compute_metrics:
                pred_batches.append(pred_batch)
                gt_batches.append(gt_batch)
            bsz = pred_batch.shape[0]
            for bi in range(bsz):
                pred_pos = pred_batch[bi]
                gt_pos = gt_batch[bi]
                if len(samples) < k:
                    samples.append((pred_pos, gt_pos))

            # For step-0 monitor snapshots, we only need GIF samples and can stop early.
            if (not compute_metrics) and len(samples) >= k:
                break

    records = []
    if compute_metrics:
        if len(pred_batches) == 0:
            raise RuntimeError(
                f"Monitoring expected at least one prediction batch for mode='{monitor_mode}', found none."
            )
        pred_all = np.concatenate(pred_batches, axis=0)
        gt_all = np.concatenate(gt_batches, axis=0)
        metrics_out = _compute_monitor_alignment_metrics(
            pred_all,
            gt_all,
            mode=monitor_mode,
            matching_mode='index_to_index',
            reorder_method='supervised_train_matching',
        )
        records.append(
            _alignment_metrics_to_record(
                metric_family='train_matching',
                matching='supervised_matched',
                fit_family='uniform_forward',
                mode=monitor_mode,
                num_videos=pred_all.shape[0],
                num_batches=num_batches,
                metrics_out=metrics_out,
            )
        )
    else:
        pred_all = None
    return {
        'samples': samples,
        'num_batches': int(num_batches),
        'num_videos': int(pred_all.shape[0]) if pred_all is not None else int(len(samples)),
        'records': records,
        'compute_metrics': bool(compute_metrics),
    }


def collect_exact_eval_monitor_metrics(
    model: ObjectDynamicsDLP,
    frozen_model: Optional[ObjectDynamicsDLP],
    device: torch.device,
    args,
    config: Dict,
    mode: str,
):
    helpers = _load_eval_monitor_helpers()
    video_to_trajectory = helpers['video_to_trajectory']

    eval_seq_len = monitor_seq_len_for_mode(
        mode=mode,
        train_seq_len=args.seq_len,
        eval_seq_len=args.monitor_eval_seq_len,
    )
    payload = video_to_trajectory(
        model=model,
        config=config,
        device=device,
        mode=mode,
        batch_size=args.batch_size,
        max_batches=args.monitor_max_batches,
        eval_seq_len=eval_seq_len,
        save_dir=None,
        latent_eval_save_dir=None,
        visualize_trajectories=False,
        extract_coordinates=True,
        returns=False,
        evaluate_latent_alignment=False,
        evaluate_noisy_gt_reference=False,
        use_hungarian_for_correlation=False,
        reorder_method='smallest_consecutive_distance',
        sg_window_length=15,
        sg_polyorder=2,
        extraction_method='latent',
        filtering_report_path=None,
        latent_position_variant='nominal',
        latent_recenter_source=args.recenter_source,
        latent_recenter_nms_source='nominal',
        latent_recenter_eps=args.recenter_eps,
        collect_nonlinear_probe_inputs=True,
        return_nonlinear_probe_payload=True,
        prob_encoder_route=_monitor_prob_encoder_eval_route(args),
        prob_encoder_frozen_model=frozen_model,
        raw_physical_npz_root='none',
        respect_max_batches_for_extraction=True,
    )

    pred = np.asarray(payload['nominal']['p'], dtype=np.float64)
    gt = np.asarray(payload['gt'], dtype=np.float64)
    num_videos = int(pred.shape[0])
    num_batches = int(payload.get('num_batches_processed', args.monitor_max_batches))

    records = []
    for matching_mode in ['index_to_index', 'hungarian']:
        metrics_out = _compute_monitor_alignment_metrics(
            pred,
            gt,
            mode=mode,
            matching_mode=matching_mode,
            reorder_method='smallest_consecutive_distance',
        )
        records.append(
            _alignment_metrics_to_record(
                metric_family='eval_extraction',
                matching=matching_mode,
                fit_family='uniform_forward',
                mode=mode,
                num_videos=num_videos,
                num_batches=num_batches,
                metrics_out=metrics_out,
            )
        )

    return {
        'num_batches': num_batches,
        'num_videos': num_videos,
        'records': records,
    }


def _append_monitor_metrics_txt(
    metrics_txt_path: str,
    run_label: str,
    global_step: int,
    records: List[Dict[str, object]],
) -> None:
    is_new_file = not os.path.exists(metrics_txt_path)
    os.makedirs(os.path.dirname(metrics_txt_path), exist_ok=True)

    with open(metrics_txt_path, 'a') as f:
        if is_new_file:
            f.write(
                "# utc_timestamp\tstep\trun_label\tmetric_family\tmatching\tfit_family\tmode\t"
                "num_videos\tnum_batches\tpearson_pos_mean\tpearson_pos_std\t"
                "pearson_vel_mean\tpearson_vel_std\tpearson_acc_mean\tpearson_acc_std\t"
                "acc_rms_ratio_mean\tacc_rms_ratio_std\tr2_mean\tr2_std\n"
            )
        now_utc = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
        for rec in records:
            f.write(
                f"{now_utc}\t{global_step}\t{run_label}\t"
                f"{rec['metric_family']}\t{rec['matching']}\t{rec['fit_family']}\t{rec['mode']}\t"
                f"{rec['num_videos']}\t{rec['num_batches']}\t"
                f"{rec['pearson_pos_mean']:.8f}\t{rec['pearson_pos_std']:.8f}\t"
                f"{rec['pearson_vel_mean']:.8f}\t{rec['pearson_vel_std']:.8f}\t"
                f"{rec['pearson_acc_mean']:.8f}\t{rec['pearson_acc_std']:.8f}\t"
                f"{rec['acc_rms_ratio_mean']:.8f}\t{rec['acc_rms_ratio_std']:.8f}\t"
                f"{rec['r2_mean']:.8f}\t{rec['r2_std']:.8f}\n"
            )


def _monitor_run_label(args) -> str:
    if getattr(args, 'output_dir', None):
        return os.path.basename(os.path.normpath(args.output_dir))
    return f"{args.route}_{args.objective}_{args.trainable_attribute_set}"


def _default_output_label(args) -> str:
    if (
        args.objective == 'position_nll'
        and args.trainable_attribute_set == 'position_only'
        and args.position_head_training == 'mean_only'
    ):
        return 'c1'
    if (
        args.objective == 'c1_dyn'
        and args.trainable_attribute_set == 'position_only'
        and args.position_head_training == 'mean_only'
    ):
        return 'c1_dyn'
    if (
        args.objective == 'c1_dyn'
        and args.trainable_attribute_set == 'position_scale_depth_obj_on'
        and args.position_head_training == 'full'
    ):
        return 'c1_dyn_full'
    return (
        f"{args.objective}_{args.trainable_attribute_set}"
        f"_poshead_{args.position_head_training}"
    )


def write_periodic_visualization_snapshot(
    model: ObjectDynamicsDLP,
    frozen_model: Optional[ObjectDynamicsDLP],
    monitor_loaders: Dict[str, DataLoader],
    device: torch.device,
    args,
    config: Dict,
    gt_similarity_transform: Optional[Dict[str, object]],
    checkpoint_dir: str,
    global_step: int,
) -> None:
    run_label = _monitor_run_label(args)
    route_root = getattr(args, 'output_dir', None)
    if route_root is None:
        route_root = os.path.join(checkpoint_dir, 'oracle', run_label)
    vis_root = os.path.join(route_root, 'visualizations')
    step_dir = os.path.join(
        vis_root,
        f"step_{global_step}",
    )
    os.makedirs(step_dir, exist_ok=True)

    print(
        f"\n[monitor] Writing trajectory visualizations at step={global_step} "
        f"to: {step_dir}"
    )

    records = []
    compute_metrics = global_step > 0
    for mode, loader in monitor_loaders.items():
        mode_dir = os.path.join(step_dir, mode)
        os.makedirs(mode_dir, exist_ok=True)

        monitor_out = collect_visualization_samples_and_metrics(
            model=model,
            frozen_model=frozen_model,
            loader=loader,
            device=device,
            args=args,
            config=config,
            gt_similarity_transform=gt_similarity_transform,
            k=args.monitor_num_videos,
            monitor_mode=mode,
            max_batches=args.monitor_max_batches,
            compute_metrics=compute_metrics,
        )
        samples = monitor_out['samples']
        if len(samples) == 0:
            print(f"[monitor] mode={mode}: no samples available, skipped.")
            continue

        for idx, (pred_pos, gt_pos) in enumerate(samples):
            save_path = os.path.join(
                mode_dir,
                f"pred_vs_gt_route_{run_label}_{mode}_{idx:03d}.gif",
            )
            # DDLP latents are y/x internally; plot monitor GIFs in conventional x/y axes.
            pred_pos_plot = pred_pos[..., [1, 0]]
            gt_pos_plot = gt_pos[..., [1, 0]] if args.swap_gt_xy else gt_pos
            _create_pred_vs_gt_trajectory_video(
                pred_positions=pred_pos_plot,
                gt_positions=gt_pos_plot,
                save_path=save_path,
            )

        if compute_metrics:
            train_records = list(monitor_out['records'])
            records.extend(train_records)
            for rec in train_records:
                print(
                    f"[monitor] mode={mode} | family={rec['metric_family']} | "
                    f"matching={rec['matching']} | videos={rec['num_videos']} | "
                    f"batches={rec['num_batches']} | "
                    f"pearson_acc={rec['pearson_acc_mean']:.4f}±{rec['pearson_acc_std']:.4f} | "
                    f"acc_rms={rec['acc_rms_ratio_mean']:.4f}±{rec['acc_rms_ratio_std']:.4f} | "
                    f"r2={rec['r2_mean']:.4f}±{rec['r2_std']:.4f}"
                )

            try:
                eval_monitor_out = collect_exact_eval_monitor_metrics(
                    model=model,
                    frozen_model=frozen_model,
                    device=device,
                    args=args,
                    config=config,
                    mode=mode,
                )
            except Exception as exc:
                print(
                    f"[monitor] Warning: exact evaluation-path metrics failed for mode={mode}. "
                    f"Error: {exc}"
                )
                eval_monitor_out = None

            if eval_monitor_out is not None:
                records.extend(eval_monitor_out['records'])
                for rec in eval_monitor_out['records']:
                    print(
                        f"[monitor] mode={mode} | family={rec['metric_family']} | "
                        f"matching={rec['matching']} | videos={rec['num_videos']} | "
                        f"batches={rec['num_batches']} | "
                        f"pearson_acc={rec['pearson_acc_mean']:.4f}±{rec['pearson_acc_std']:.4f} | "
                        f"acc_rms={rec['acc_rms_ratio_mean']:.4f}±{rec['acc_rms_ratio_std']:.4f} | "
                        f"r2={rec['r2_mean']:.4f}±{rec['r2_std']:.4f}"
                    )
        else:
            print(
                f"[monitor] mode={mode}: wrote {len(samples)} GIFs | "
                f"batches={monitor_out['num_batches']} | metrics=skipped(step 0)"
            )

    if len(records) > 0:
        metrics_txt_path = os.path.join(
            vis_root,
            f"monitor_metrics_v2_route_{run_label}.txt",
        )
        _append_monitor_metrics_txt(
            metrics_txt_path=metrics_txt_path,
            run_label=run_label,
            global_step=global_step,
            records=records,
        )
        print(f"[monitor] updated metrics log: {metrics_txt_path}")


def run_epoch(
    model: ObjectDynamicsDLP,
    frozen_model: Optional[ObjectDynamicsDLP],
    loader: DataLoader,
    device: torch.device,
    args,
    config: Dict,
    gt_similarity_transform: Optional[Dict[str, object]] = None,
    lagrangian_model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_masks: Optional[Dict[torch.nn.Parameter, torch.Tensor]] = None,
    adaptive_beta_state: Optional[Dict[str, float]] = None,
    max_batches: Optional[int] = None,
    desc: str = 'train',
    step_state: Optional[Dict[str, int]] = None,
    step_callback: Optional[Callable[[int], None]] = None,
):
    # Keep module in eval-mode so frozen backbone BN/dropout stats never drift.
    model.eval()

    agg = {
        'loss': 0.0,
        'position_loss': 0.0,
        'dynamics_loss': 0.0,
        'mse': 0.0,
        'mae_l1_xy': 0.0,
        'mean_pred_std': 0.0,
        'beta_eff': 0.0,
        'beta_gate': 0.0,
        'fit_quality': 0.0,
        'beta_ratio': 0.0,
        'pos_grad_norm': 0.0,
        'del_grad_norm': 0.0,
        'pos_grad_ema': 0.0,
        'del_grad_ema': 0.0,
        'fit_mse_ema': 0.0,
        'fit_mse_ref': 0.0,
    }
    n = 0
    reorder_method_seen = None

    pbar = tqdm(loader, desc=desc)
    for bi, batch in enumerate(pbar):
        if max_batches is not None and bi >= max_batches:
            break

        metrics = one_pass(
            model=model,
            frozen_model=frozen_model,
            batch=batch,
            device=device,
            args=args,
            config=config,
            gt_similarity_transform=gt_similarity_transform,
            lagrangian_model=lagrangian_model,
            optimizer=optimizer,
            grad_masks=grad_masks,
            adaptive_beta_state=adaptive_beta_state,
        )

        n += 1
        for k in agg:
            agg[k] += metrics[k]
        reorder_method_seen = metrics['reorder_method']

        if optimizer is not None and step_state is not None:
            step_state['global_step'] = int(step_state.get('global_step', 0)) + 1
            if step_callback is not None:
                step_callback(step_state['global_step'])

        pbar.set_postfix(
            loss=f"{metrics['loss']:.5f}",
            pos=f"{metrics['position_loss']:.5f}",
            dyn=f"{metrics['dynamics_loss']:.5f}",
            beta=f"{metrics['beta_eff']:.5f}",
            gate=f"{metrics['beta_gate']:.3f}",
            mse=f"{metrics['mse']:.5f}",
            mae=f"{metrics['mae_l1_xy']:.5f}",
            std=f"{metrics['mean_pred_std']:.5f}",
        )

    if n == 0:
        raise RuntimeError(f"No batches processed for {desc}. Check dataset and loader settings.")

    out = {k: v / n for k, v in agg.items()}
    out['reorder_method'] = reorder_method_seen
    return out


def save_checkpoint(
    save_path: str,
    model: ObjectDynamicsDLP,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict,
    args,
    config: Dict,
):
    payload = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'route_f_variant': args.route,
        'args': vars(args),
        'source_config': config,
        'saved_at_utc': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
    }
    torch.save(payload, save_path)


def parse_args():
    p = argparse.ArgumentParser(
        description='Train C1 probabilistic oracle encoder from a pretrained DDLP checkpoint.'
    )
    p.add_argument('--checkpoint', type=str, required=True,
                   help='Checkpoint directory containing hparams.json + saves/*.pth, or direct .pth path.')
    p.add_argument('--checkpoint_name', type=str, default=None,
                   help='Checkpoint filename under <checkpoint>/saves. If omitted, auto-selects latest best/last.')
    p.add_argument('--config_path', type=str, default=None,
                   help='Optional explicit config JSON path (defaults to <checkpoint_dir>/hparams.json).')

    p.add_argument('--route', type=str, default='f1', choices=['f1'],
                   help="Only route='f1' is supported. The former C2/F2 whole-latent route was removed.")
    p.add_argument('--objective', type=str, default='position_nll', choices=['position_nll', 'c1_dyn'],
                   help='Training objective. c1_dyn adds a fixed DEL residual in D6-TRUE coordinates.')
    p.add_argument('--beta', type=float, default=1.0,
                   help='Adaptive-beta scale c in beta_eff = gate(fit_quality) * c * EMA(||grad L_pos||) / (EMA(||grad L_del||) + eps).')
    p.add_argument('--beta_ema_decay', type=float, default=0.95,
                   help='EMA decay used for fit-quality and gradient-norm tracking in adaptive beta_eff.')
    p.add_argument('--beta_eps', type=float, default=1e-8,
                   help='Numerical epsilon used in adaptive beta_eff and fit-quality computations.')
    p.add_argument('--trainable_attribute_set', type=str, default='position_only',
                   choices=['position_only', 'position_scale_depth_obj_on'],
                   help='Attribute heads to finetune. Features partition always remains fixed.')
    p.add_argument('--position_head_training', type=str, default='mean_only',
                   choices=['mean_only', 'full'],
                   help='Whether x/y heads finetune only the mean row or the full (mean, logvar) outputs.')
    p.add_argument('--f1_delta_mode', type=str, default='relative_to_frozen_nominal',
                   choices=['relative_to_frozen_nominal', 'direct_trainable_nominal'],
                   help='How to form delta_theta in F.1. Default keeps F.1 initialized at frozen recentered baseline.')
    p.add_argument('--recenter_source', type=str, default='patch_alpha', choices=['patch_alpha', 'global_alpha'],
                   help='Frozen recentering source for F.1.')
    p.add_argument('--recenter_eps', type=float, default=1e-6,
                   help='Numerical epsilon for recentering centroid computations.')

    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=None,
                   help='If omitted, uses batch_size from source config.')
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=0)

    p.add_argument('--train_mode', type=str, default='train', choices=['train', 'valid', 'val', 'test'])
    p.add_argument('--valid_mode', type=str, default='valid', choices=['train', 'valid', 'val', 'test'])
    p.add_argument('--seq_len', type=int, default=None,
                   help='Sequence length for supervised training. If omitted, uses config timestep_horizon.')
    p.add_argument('--num_static_frames', type=int, default=None,
                   help='Burn-in frames for prior proposal stage. If omitted, uses config num_static_frames.')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--max_train_batches', type=int, default=None)
    p.add_argument('--max_valid_batches', type=int, default=None)

    p.add_argument('--temporal_reorder', type=str, default='auto',
                   choices=['auto', 'none', 'smallest_consecutive_distance', 'hungarian'],
                   help='Detached temporal reordering before sequence-level Hungarian matching.')
    p.add_argument('--use_in_camera_mask', type=int, default=1,
                   help='Use in_camera mask (if provided by dataset) when computing matching/NLL.')

    p.add_argument('--lagrangian_checkpoint', type=str,
                   default='/data2/users/lr4617/discrete_lagrangian/del_pytorch/outputs/regularized/ghnn_generated_BIG_TRUE_matched_lam0p1',
                   help='Fixed SeparableLagrangianMLP checkpoint file or directory used by objective=c1_dyn.')
    p.add_argument('--lagrangian_config_path', type=str, default=None,
                   help='Optional explicit config.json for the fixed Lagrangian checkpoint.')
    p.add_argument('--dyn_weak_window', type=int, default=0,
                   help='Temporal half-window for weak residual smoothing. 0 uses pointwise DEL residuals.')
    p.add_argument('--dyn_latent_to_pixel_mode', type=str, default='auto',
                   choices=['auto', 'normalized_grid', 'ddlp_similarity_inverse'],
                   help='How C1-dyn maps DDLP latents to D6-TRUE-PIX before world conversion.')
    p.add_argument('--dyn_image_width', type=int, default=64,
                   help='D6-TRUE-PIX image width for T_C1->phys.')
    p.add_argument('--dyn_image_height', type=int, default=64,
                   help='D6-TRUE-PIX image height for T_C1->phys.')
    p.add_argument('--dyn_render_radius_px', type=float, default=5.0,
                   help='Renderer pixel radius used by the inverse D6-TRUE-PIX -> D6-TRUE transform.')
    p.add_argument('--dyn_world_x_min', type=float, default=-8.80335912,
                   help='D6-TRUE renderer world x_min.')
    p.add_argument('--dyn_world_x_max', type=float, default=8.44361367,
                   help='D6-TRUE renderer world x_max.')
    p.add_argument('--dyn_world_y_min', type=float, default=-10.50628237,
                   help='D6-TRUE renderer world y_min.')
    p.add_argument('--dyn_world_y_max', type=float, default=9.81283697,
                   help='D6-TRUE renderer world y_max.')

    p.add_argument('--gt_normalization', type=str, default='ddlp_similarity',
                   choices=['auto', 'pixel', 'zero_one', 'minus_one_one', 'none', 'ddlp_similarity'],
                   help='Ground-truth position normalization into DDLP coord range. Default matches labels to extracted DDLP latent scale.')
    p.add_argument('--gt_similarity_reference', type=str, default=None,
                   help='Reference DEL/DDLP HDF5 dataset used to fit the GT->DDLP similarity transform.')
    p.add_argument('--swap_gt_xy', type=int, default=0,
                   help='Swap GT coordinate order (x,y) <-> (y,x) after normalization.')

    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--output_dir', type=str, default=None,
                   help='Output directory for probabilistic encoder checkpoints/logs.')
    p.add_argument('--monitor_visualizations', type=int, default=1,
                   help='Enable periodic trajectory GIF snapshots during training (1/0).')
    p.add_argument('--monitor_every_steps', type=int, default=300,
                   help='Write monitor visualizations every this many optimizer steps.')
    p.add_argument('--monitor_num_videos', type=int, default=10,
                   help='Number of videos (k) per dataset split for each monitor snapshot.')
    p.add_argument('--monitor_modes', type=str, default='train,valid,test',
                   help='Comma-separated dataset splits to visualize (train,valid,val,test).')
    p.add_argument('--monitor_max_batches', type=int, default=30,
                   help='Maximum batches to process per split during each monitor snapshot.')
    p.add_argument('--monitor_eval_seq_len', type=int, default=60,
                   help='For monitor splits valid/val/test, use this seq_len instead of training seq_len.')
    return p.parse_args()


def main():
    args = parse_args()
    args.use_in_camera_mask = bool(args.use_in_camera_mask)
    args.swap_gt_xy = bool(args.swap_gt_xy)
    args.monitor_visualizations = bool(args.monitor_visualizations)
    if args.beta < 0.0:
        raise ValueError(f"beta must be non-negative, got {args.beta}")
    if not (0.0 <= args.beta_ema_decay < 1.0):
        raise ValueError(f"beta_ema_decay must be in [0,1), got {args.beta_ema_decay}")
    if args.beta_eps <= 0.0:
        raise ValueError(f"beta_eps must be > 0, got {args.beta_eps}")
    if args.dyn_weak_window < 0:
        raise ValueError(f"dyn_weak_window must be >= 0, got {args.dyn_weak_window}")
    if args.dyn_image_width <= 1 or args.dyn_image_height <= 1:
        raise ValueError(
            f"dyn_image_width/height must be > 1, got {args.dyn_image_width}x{args.dyn_image_height}"
        )

    monitor_modes = []
    if args.monitor_visualizations:
        if imageio is None:
            raise ImportError(
                "imageio is required because --monitor_visualizations 1. "
                "Install imageio in this environment or pass --monitor_visualizations 0."
            )
        if args.monitor_every_steps <= 0:
            raise ValueError(f"monitor_every_steps must be > 0, got {args.monitor_every_steps}")
        if args.monitor_num_videos <= 0:
            raise ValueError(f"monitor_num_videos must be > 0, got {args.monitor_num_videos}")
        if args.monitor_max_batches <= 0:
            raise ValueError(f"monitor_max_batches must be > 0, got {args.monitor_max_batches}")
        if args.monitor_eval_seq_len <= 0:
            raise ValueError(f"monitor_eval_seq_len must be > 0, got {args.monitor_eval_seq_len}")
        monitor_modes = parse_monitor_modes(args.monitor_modes)

    ckpt_path, checkpoint_dir = resolve_checkpoint_path(args.checkpoint, args.checkpoint_name)
    config_path = args.config_path if args.config_path is not None else os.path.join(checkpoint_dir, 'hparams.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    if args.batch_size is None:
        args.batch_size = int(config['batch_size'])
    if args.seq_len is None:
        args.seq_len = int(config['timestep_horizon'])
    if args.num_static_frames is None:
        args.num_static_frames = int(config.get('num_static_frames', min(4, args.seq_len)))

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    print('\n' + '=' * 90)
    print('Route-F Probabilistic Encoder Training')
    print('=' * 90)
    print(f"Checkpoint          : {ckpt_path}")
    print(f"Config              : {config_path}")
    print(f"Dataset             : {config['ds']}")
    print(f"Data root           : {config['root']}")
    print(f"Route               : {args.route} (C1 recentering-biased)")
    print(f"Objective           : {args.objective}")
    print(f"Adaptive beta c     : {args.beta}")
    print(f"Adaptive beta EMA   : {args.beta_ema_decay}")
    print(f"Adaptive beta eps   : {args.beta_eps}")
    print(f"Trainable attrs     : {args.trainable_attribute_set}")
    print(f"Position head mode  : {args.position_head_training}")
    print(f"Temporal reorder    : {args.temporal_reorder}")
    print(f"GT normalization    : {args.gt_normalization}")
    print(f"Sequence length     : {args.seq_len}")
    print(f"Num static frames   : {args.num_static_frames}")
    print(f"Device              : {device}")
    print('=' * 90 + '\n')

    model = build_model_from_config(config, device)
    checkpoint_obj = torch.load(ckpt_path, map_location=device)
    state_dict, ckpt_meta = _extract_state_dict(checkpoint_obj)
    model.load_state_dict(state_dict, strict=True)

    lagrangian_model = None
    lagrangian_ckpt_path = None
    lagrangian_config_path = None
    lagrangian_config = None
    if args.objective == 'c1_dyn':
        lagrangian_model, lagrangian_ckpt_path, lagrangian_config_path, lagrangian_config = load_fixed_lagrangian(
            checkpoint_path_or_dir=args.lagrangian_checkpoint,
            explicit_config_path=args.lagrangian_config_path,
            device=device,
        )
        args.lagrangian_checkpoint = lagrangian_ckpt_path
        args.lagrangian_config_path = lagrangian_config_path if os.path.exists(lagrangian_config_path) else None
        print(f"Fixed DEL checkpoint: {lagrangian_ckpt_path}")
        print(f"Fixed DEL config    : {args.lagrangian_config_path}")
        print(
            "C1->phys transform  : "
            f"latent y/x -> {args.dyn_image_width}x{args.dyn_image_height} D6-TRUE-PIX "
            f"-> D6-TRUE world (latent_to_pixel={args.dyn_latent_to_pixel_mode})"
        )

    grad_masks = freeze_for_route_f(
        model,
        trainable_attribute_set=args.trainable_attribute_set,
        position_head_training=args.position_head_training,
    )
    trainable_params = get_trainable_parameters(model)
    if not trainable_params:
        raise RuntimeError('No trainable parameters found after Route-F freezing.')

    print('Trainable parameter groups (Route F):')
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(f"  - {name}: shape={tuple(p.shape)}")

    # Frozen reference checkpoint for Route F.1 recentering bias computation.
    frozen_model = None
    if args.route == 'f1':
        frozen_model = deepcopy(model).to(device)
        frozen_model.eval()
        for p in frozen_model.parameters():
            p.requires_grad = False

    optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    train_ds = get_video_dataset(
        config['ds'],
        root=config['root'],
        seq_len=args.seq_len,
        mode=args.train_mode,
        image_size=config['image_size'],
    )
    valid_ds = get_video_dataset(
        config['ds'],
        root=config['root'],
        seq_len=args.seq_len,
        mode=args.valid_mode,
        image_size=config['image_size'],
    )

    gt_similarity_transform = None
    if args.gt_normalization == 'ddlp_similarity':
        resolved_ref = _resolve_gt_similarity_reference(args.gt_similarity_reference, checkpoint_dir)
        if resolved_ref is None:
            raise FileNotFoundError(
                "gt_normalization='ddlp_similarity' requires a reference dataset. "
                "Pass --gt_similarity_reference or place "
                "'extraction_evaluation/best/extracted_datasets/ddlp_extracted_recentered_training.h5.1' "
                f"under the checkpoint directory: {checkpoint_dir}"
            )
        args.gt_similarity_reference = resolved_ref
        gt_similarity_transform = fit_ddlp_similarity_transform(
            dataset=train_ds,
            reference_path=resolved_ref,
        )
        print(f"GT similarity ref   : {resolved_ref}")
        print(
            "GT similarity fit   : "
            f"scale={gt_similarity_transform['scale']:.6f}, "
            f"source_center={gt_similarity_transform['source_center']}, "
            f"reference_center={gt_similarity_transform['reference_center']}"
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
    )

    monitor_loaders = {}
    if args.monitor_visualizations:
        for requested_mode in monitor_modes:
            seq_len_for_mode = monitor_seq_len_for_mode(
                mode=requested_mode,
                train_seq_len=args.seq_len,
                eval_seq_len=args.monitor_eval_seq_len,
            )
            try:
                monitor_ds, resolved_mode = load_dataset_with_valid_alias_fallback(
                    ds_name=config['ds'],
                    root=config['root'],
                    seq_len=seq_len_for_mode,
                    mode=requested_mode,
                    image_size=config['image_size'],
                )
            except Exception as exc:
                print(
                    f"[monitor] Warning: failed to build loader for mode='{requested_mode}'. "
                    f"Skipping this split. Error: {exc}"
                )
                continue

            monitor_loaders[requested_mode] = DataLoader(
                monitor_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                drop_last=False,
                pin_memory=True,
            )
            if resolved_mode != requested_mode:
                print(
                    f"[monitor] mode alias fallback: requested '{requested_mode}', "
                    f"loaded dataset split '{resolved_mode}'."
                )
            print(
                f"[monitor] loader ready: mode='{requested_mode}', seq_len={seq_len_for_mode}, "
                f"num_samples={len(monitor_ds)}"
            )

    route_tag = _default_output_label(args)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(checkpoint_dir, 'oracle', route_tag)
    args.output_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'run_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    with open(os.path.join(output_dir, 'source_hparams.json'), 'w') as f:
        json.dump(config, f, indent=2)

    if args.monitor_visualizations:
        print(
            f"[monitor] Enabled periodic trajectory snapshots: "
            f"every={args.monitor_every_steps} train steps, "
            f"k={args.monitor_num_videos}, "
            f"max_batches={args.monitor_max_batches}, "
            f"modes={list(monitor_loaders.keys())}"
        )

    history = []
    best_val = float('inf')
    best_epoch = -1
    step_state = {'global_step': 0}
    adaptive_beta_state = {} if args.objective == 'c1_dyn' else None

    def _maybe_write_monitor_snapshot(global_step: int) -> None:
        if not args.monitor_visualizations:
            return
        if len(monitor_loaders) == 0:
            return
        if global_step % args.monitor_every_steps != 0:
            return
        write_periodic_visualization_snapshot(
            model=model,
            frozen_model=frozen_model,
            monitor_loaders=monitor_loaders,
            device=device,
            args=args,
            config=config,
            gt_similarity_transform=gt_similarity_transform,
            checkpoint_dir=checkpoint_dir,
            global_step=global_step,
        )

    # Also evaluate at initialization (step 0) before any optimizer update.
    _maybe_write_monitor_snapshot(step_state['global_step'])

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            frozen_model=frozen_model,
            loader=train_loader,
            device=device,
            args=args,
            config=config,
            gt_similarity_transform=gt_similarity_transform,
            lagrangian_model=lagrangian_model,
            optimizer=optimizer,
            grad_masks=grad_masks,
            adaptive_beta_state=adaptive_beta_state,
            max_batches=args.max_train_batches,
            desc=f'train e{epoch:03d}',
            step_state=step_state,
            step_callback=_maybe_write_monitor_snapshot,
        )

        val_metrics = run_epoch(
            model=model,
            frozen_model=frozen_model,
            loader=valid_loader,
            device=device,
            args=args,
            config=config,
            gt_similarity_transform=gt_similarity_transform,
            lagrangian_model=lagrangian_model,
            optimizer=None,
            grad_masks=None,
            adaptive_beta_state=adaptive_beta_state,
            max_batches=args.max_valid_batches,
            desc=f'valid e{epoch:03d}',
        )

        summary = {
            'epoch': epoch,
            'train': train_metrics,
            'valid': val_metrics,
            'lr': float(optimizer.param_groups[0]['lr']),
        }
        history.append(summary)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.6f}, val_loss={val_metrics['loss']:.6f}, "
            f"train_pos={train_metrics['position_loss']:.6f}, val_pos={val_metrics['position_loss']:.6f}, "
            f"train_dyn={train_metrics['dynamics_loss']:.6f}, val_dyn={val_metrics['dynamics_loss']:.6f}, "
            f"train_beta={train_metrics['beta_eff']:.6f}, val_beta={val_metrics['beta_eff']:.6f}, "
            f"train_gate={train_metrics['beta_gate']:.6f}, val_gate={val_metrics['beta_gate']:.6f}, "
            f"train_mse={train_metrics['mse']:.6f}, val_mse={val_metrics['mse']:.6f}, "
            f"reorder={train_metrics['reorder_method']}"
        )

        last_ckpt = os.path.join(output_dir, 'prob_encoder_last.pth')
        save_checkpoint(last_ckpt, model, optimizer, epoch, summary, args, config)

        if val_metrics['loss'] < best_val:
            best_val = val_metrics['loss']
            best_epoch = epoch
            best_ckpt = os.path.join(output_dir, 'prob_encoder_best.pth')
            save_checkpoint(best_ckpt, model, optimizer, epoch, summary, args, config)

        with open(os.path.join(output_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)

    final_report = {
        'best_val_loss': best_val,
        'best_epoch': best_epoch,
        'num_epochs': args.epochs,
        'route': args.route,
        'objective': args.objective,
        'beta': args.beta,
        'beta_role': 'adaptive_c',
        'beta_ema_decay': args.beta_ema_decay,
        'beta_eps': args.beta_eps,
        'adaptive_beta_state': adaptive_beta_state,
        'trainable_attribute_set': args.trainable_attribute_set,
        'gt_normalization': args.gt_normalization,
        'gt_similarity_reference': args.gt_similarity_reference,
        'gt_similarity_transform': gt_similarity_transform,
        'lagrangian_checkpoint': lagrangian_ckpt_path,
        'lagrangian_config_path': args.lagrangian_config_path,
        'lagrangian_config': lagrangian_config,
        'checkpoint_source': ckpt_path,
        'config_source': config_path,
        'completed_at_utc': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'checkpoint_meta': {
            'epoch': ckpt_meta.get('epoch', None) if isinstance(ckpt_meta, dict) else None,
        },
    }
    with open(os.path.join(output_dir, 'final_report.json'), 'w') as f:
        json.dump(final_report, f, indent=2)

    print('\n' + '=' * 90)
    print('Training complete.')
    print(f"Output dir          : {output_dir}")
    print(f"Best epoch          : {best_epoch}")
    print(f"Best validation NLL : {best_val:.6f}")
    print('=' * 90)


if __name__ == '__main__':
    main()
