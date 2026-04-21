"""
Generate visualizations for DDLP model including trajectories, bounding boxes,
and inter-particle distance plots with re-ordering detection.

IMPORTANT: This script uses FULL MODEL with temporal tracking and dynamics.
Video sequences are processed with temporal context to extract position latents
and learned features for visualization.

The 'mode' parameter selects which dataset split (train/valid/test) to evaluate on.
"""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np
import os
import sys
import json
import argparse
import math
from copy import deepcopy
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
from scipy.signal import savgol_filter
from scipy.optimize import linear_sum_assignment
import pandas as pd
from itertools import product
import time as time_module

# Add parent directory to path for local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision.utils as vutils

# datasets
from datasets.get_dataset import get_video_dataset
from utils.util_func import (
    plot_keypoints_on_image_batch,
    plot_bb_on_image_batch_from_z_scale_nms,
    plot_bb_on_image_batch_from_masks_nms,
)

# models
from models import ObjectDynamicsDLP

import sys

#####################################################################################
def _extract_model_state_dict(checkpoint_obj):
    """Return the state dict from common DDLP / probabilistic-encoder checkpoint payloads."""
    if isinstance(checkpoint_obj, dict):
        if 'model_state_dict' in checkpoint_obj:
            return checkpoint_obj['model_state_dict']
        if 'model' in checkpoint_obj:
            return checkpoint_obj['model']
        if 'state_dict' in checkpoint_obj:
            return checkpoint_obj['state_dict']
    return checkpoint_obj


def _load_checkpoint_into_model(model, checkpoint_path, device, strict=True):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(_extract_model_state_dict(checkpoint), strict=strict)
    return checkpoint


#####################################################################################
def create_pred_only_trajectory_videos(video_data_list, save_dir='./', mode='valid'):
    """
    Generate videos showing ONLY predicted trajectories in their original scale.
    No ground-truth is displayed, and scaling is based only on predicted positions.
    
    Args:
        video_data_list: List of video data dictionaries from evaluation
        save_dir: Directory to save videos
        mode: Dataset split name for filename
    """
    num_videos = len(video_data_list)
    print(f"\n🎯 Generating {num_videos} prediction-only trajectory videos...")
    print(f"   [DEBUG] Received {num_videos} video data entries")
    
    for video_idx, video_data in enumerate(video_data_list):
        try:
            pred_positions_active = video_data  # (T, N_active, 2)
            
            print(f"   [DEBUG] Video {video_idx}: pred shape={pred_positions_active.shape}")
            print(f"   [DEBUG] Video {video_idx}: pred range=[{pred_positions_active.min():.4f}, {pred_positions_active.max():.4f}]")
            
            # Create prediction-only trajectory visualization
            save_path = os.path.join(save_dir, f'pred_only_trajectory_{mode}_{video_idx:03d}.gif')
            _create_pred_only_trajectory_video(
                pred_positions_active,
                save_path
            )
            
            if (video_idx + 1) % 5 == 0:
                print(f"  Generated {video_idx + 1}/{num_videos} prediction-only trajectory videos")
        
        except Exception as e:
            print(f"  Error generating prediction-only trajectory video {video_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"✅ Completed prediction-only trajectory video generation")

def _create_pred_only_trajectory_video(pred_positions, save_path):
    """
    Create a single trajectory visualization video showing ONLY predicted trajectories.
    
    Args:
        pred_positions: (T, N_pred, 2) predicted positions in model range
        save_path: Path to save the GIF
    """
    T = pred_positions.shape[0]
    N_pred = pred_positions.shape[1]
    
    print(f"   [DEBUG] Creating pred-only trajectory video: T={T}, N_pred={N_pred}")
    
    # Debug: Print statistics for each particle
    for i in range(N_pred):
        pred_particle = pred_positions[:, i, :]
        print(f"   [DEBUG] Pred particle {i}: mean={pred_particle.mean(axis=0)}, std={pred_particle.std(axis=0)}, range=({pred_particle.min():.4f}, {pred_particle.max():.4f})")
    
    # Compute ranges for scaling based ONLY on predicted positions
    x_min, x_max = pred_positions[:, :, 0].min(), pred_positions[:, :, 0].max()
    y_min, y_max = pred_positions[:, :, 1].min(), pred_positions[:, :, 1].max()
    
    # Compute ranges
    x_range = x_max - x_min if x_max > x_min else 0.1
    y_range = y_max - y_min if y_max > y_min else 0.1
    
    # Use the larger range for both axes to ensure equal aspect ratio
    max_range = max(x_range, y_range)
    
    # Add 20% padding
    max_range = max_range * 1.2
    
    # Center the range around the data
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    
    x_lim = [x_center - max_range / 2, x_center + max_range / 2]
    y_lim = [y_center - max_range / 2, y_center + max_range / 2]
    
    print(f"   [DEBUG] Plot ranges (pred-only): x={x_lim}, y={y_lim}, max_range={max_range:.4f}")
    
    # Prepare figure - single panel showing only predicted trajectories
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Set fixed axes properties once to prevent shifting
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_xlabel('X Position', fontsize=13, fontweight='bold')
    ax.set_ylabel('Y Position', fontsize=13, fontweight='bold')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=0.4, linewidth=0.8)
    
    # Apply tight_layout once before the loop
    plt.tight_layout()
    
    frames = []
    
    # Color palette for particles
    colors = plt.cm.Set1(np.linspace(0, 1, N_pred))
    
    for t in range(T):
        ax.clear()
        
        # Re-apply fixed settings after clear
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_xlabel('X Position', fontsize=13, fontweight='bold')
        ax.set_ylabel('Y Position', fontsize=13, fontweight='bold')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(alpha=0.4, linewidth=0.8)
        
        # Plot predicted trajectories only
        for kp_idx in range(N_pred):
            pred_traj = pred_positions[:t+1, kp_idx, :]  # (t+1, 2)
            if t > 0:
                ax.plot(pred_traj[:, 0], pred_traj[:, 1],
                       '-', color=colors[kp_idx], alpha=0.7, linewidth=3,
                       label=f'Pred {kp_idx+1}')
            # Current position
            ax.scatter(pred_traj[-1, 0], pred_traj[-1, 1],
                      c=[colors[kp_idx]], s=350, marker='x', alpha=0.9,
                      linewidths=5, zorder=10)
        
        ax.set_title(f'Predicted Trajectories: Frame {t+1}/{T}\n(Original Scale)', 
                    fontsize=14, fontweight='bold', pad=15)
        
        # Always show legend on every frame
        ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
        
        # Render to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
    
    # Save as GIF (duration in ms: 100ms = 10fps)
    imageio.mimsave(save_path, frames, duration=100, loop=0)
    plt.close(fig)
#####################################################################################

#####################################################################################
def create_trajectory_videos(
    video_data_list, 
    save_dir='./', 
    mode='valid', 
    keys_to_plot=['pred_positions', 'pred_ordered_positions'], 
    window_length=None,
    legend_label_first=None,
    legend_label_second=None,
    title_override_template=None,
):
    """
    Generate videos showing low-dimensional trajectories from evaluated video data.
    
    Args:
        video_data_list: List of video data dictionaries from evaluation
        save_dir: Directory to save videos
        mode: Dataset split name for filename
        keys_to_plot: List of keys indicating which trajectories to plot (e.g., ['pred_positions', 'pred_ordered_positions'])
        window_length: Window length for smoothing filter (if applicable)
    """
    num_videos = len(video_data_list)

    
    for video_idx, video_data in enumerate(video_data_list):
        try:
            pred_positions = video_data[keys_to_plot[0]]  # (T, N_active, 2)
            pred_ordered_positions = video_data[keys_to_plot[1]]  # (T, N_pred_ordered, 2)

            if keys_to_plot[1] == 'smoothed_pred_ordered_positions':
                which = "smoothed_ordered_comparison"
            elif keys_to_plot[1] == 'pred_ordered_positions':
                which = "ordering_comparison"
            else:
                raise NotImplementedError(f"Unknown keys_to_plot combination: {keys_to_plot}")
            
            # Create trajectory visualization
            if window_length is not None and keys_to_plot[1] == 'smoothed_pred_ordered_positions':
                save_path = os.path.join(save_dir, f'{which}_wl_{window_length}_{mode}_{video_idx:03d}.gif')
            else:
                save_path = os.path.join(save_dir, f'{which}_{mode}_{video_idx:03d}.gif')
            _create_trajectory_video(
                pred_positions, 
                pred_ordered_positions,
                save_path, 
                keys_to_plot=keys_to_plot,
                legend_label_first=legend_label_first,
                legend_label_second=legend_label_second,
                title_override_template=title_override_template,
            )
            
            if (video_idx + 1) % 5 == 0:
                print(f"  Generated {video_idx + 1}/{num_videos} trajectory videos")
        
        except Exception as e:
            print(f"  Error generating trajectory video {video_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"✅ Completed trajectory video generation")

def _create_trajectory_video(
    pred_positions, pred_ordered_positions, 
    save_path, keys_to_plot=['pred_positions', 'pred_ordered_positions'],
    show_pairwise_connectors=False,
    title_override_template=None,
    annotation_text=None,
    legend_label_first=None,
    legend_label_second=None,
):
    """
    Create a single trajectory visualization video showing predicted vs predicted ordered trajectories.
    
    Args:
        pred_positions: (T, N_pred, 2) predicted positions in model range
        pred_ordered_positions: (T, N_pred_ordered, 2) predicted ordered positions in model range
        save_path: Path to save the GIF
        keys_to_plot: List of keys indicating which trajectories to plot
        show_pairwise_connectors: If True, draw line segments connecting each object
            at current frame between first and second trajectory set (same index).
        title_override_template: Optional format string for title, supporting {t} and {T}.
        annotation_text: Optional annotation text rendered inside the axes.
    """
    T = pred_positions.shape[0]
    N_pred = pred_positions.shape[1]
    N_pred_ordered = pred_ordered_positions.shape[1]
    
    # Compute global ranges for consistent scaling with equal aspect ratio
    all_pos = np.concatenate([pred_positions.reshape(-1, 2), pred_ordered_positions.reshape(-1, 2)], axis=0)
    x_min, x_max = all_pos[:, 0].min(), all_pos[:, 0].max()
    y_min, y_max = all_pos[:, 1].min(), all_pos[:, 1].max()
    
    # Compute ranges
    x_range = x_max - x_min if x_max > x_min else 0.1
    y_range = y_max - y_min if y_max > y_min else 0.1
    
    # Use the larger range for both axes to ensure equal aspect ratio doesn't shift plots
    max_range = max(x_range, y_range)
    
    # Add 20% padding
    max_range = max_range * 1.2
    
    # Center the range around the data
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    
    x_lim = [x_center - max_range / 2, x_center + max_range / 2]
    y_lim = [y_center - max_range / 2, y_center + max_range / 2]
        
    # Prepare figure - single panel showing only trajectories
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Set fixed axes properties once to prevent shifting
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_xlabel('X Position', fontsize=13, fontweight='bold')
    ax.set_ylabel('Y Position', fontsize=13, fontweight='bold')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=0.4, linewidth=0.8)
    
    # Keep extra top margin so longer titles are not clipped.
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.82)
    
    frames = []
    
    # Color palette(s) for particles.
    # For supervised-comparison diagnostics, use distinct source palettes so
    # "before" and "after" are visually separable even when trajectories overlap.
    if show_pairwise_connectors:
        colors_after = plt.cm.Reds(np.linspace(0.45, 0.95, max(N_pred_ordered, 1)))
        colors_before = plt.cm.Blues(np.linspace(0.45, 0.95, max(N_pred, 1)))
    else:
        shared_colors = plt.cm.Set1(np.linspace(0, 1, max(N_pred, N_pred_ordered)))
        colors_after = shared_colors
        colors_before = shared_colors
    
    for t in range(T):
        ax.clear()
        
        # Re-apply fixed settings after clear
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_xlabel('X Position', fontsize=13, fontweight='bold')
        ax.set_ylabel('Y Position', fontsize=13, fontweight='bold')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(alpha=0.4, linewidth=0.8)
        
        if keys_to_plot[1] == 'smoothed_pred_ordered_positions':
            if title_override_template is None:
                ax.set_title(
                    f'Predicted Ordered Trajectories vs Smoothed (Savitzky-Golay)\nFrame {t+1}/{T}\n(Circle=Pred-Or, X=Pred-Or-Smoothed)', 
                    fontsize=14, fontweight='bold', pad=15
                )
            label_smoothed = 'Pred-Or-Smoothed'
            label_noisy = 'Pred-Or'
        elif keys_to_plot[1] == 'pred_ordered_positions':
            if title_override_template is None:
                ax.set_title(
                    f'Predicted Trajectories vs Predicted Ordered Trajectories: Frame {t+1}/{T}\n(Circle=Pred-Or, X=Pred)', 
                    fontsize=14, fontweight='bold', pad=15
                )
            label_smoothed = 'Pred-Or'
            label_noisy = 'Pred'
        else:
            raise NotImplementedError(f"Unknown keys_to_plot combination: {keys_to_plot}")

        if show_pairwise_connectors:
            label_smoothed = 'After-Hungarian'
            label_noisy = 'Before-Hungarian'

        if legend_label_second is not None:
            label_smoothed = str(legend_label_second)
        if legend_label_first is not None:
            label_noisy = str(legend_label_first)

        if title_override_template is not None:
            ax.set_title(
                title_override_template.format(t=t+1, T=T),
                fontsize=14, fontweight='bold', pad=15
            )

        if annotation_text is not None:
            ax.text(
                0.01, 0.99, annotation_text,
                transform=ax.transAxes, ha='left', va='top',
                fontsize=10, color='black',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.8, edgecolor='black')
            )

        # Plot predicted (ordered) trajectories
        for obj_idx in range(N_pred_ordered):
            pred_ordered_traj = pred_ordered_positions[:t+1, obj_idx, :]  # (t+1, 2)
            ax.plot(pred_ordered_traj[:, 0], pred_ordered_traj[:, 1], 
                   '-', color=colors_after[obj_idx], alpha=0.8, linewidth=3,
                   label=f'{label_smoothed} {obj_idx+1}')
            # Current position
            ax.scatter(pred_ordered_traj[-1, 0], pred_ordered_traj[-1, 1], 
                      c=[colors_after[obj_idx]], s=350, marker='o', alpha=0.95,
                      edgecolors='black', linewidths=3, zorder=10)
        
        # Plot predicted trajectories
        for kp_idx in range(N_pred):
            pred_traj = pred_positions[:t+1, kp_idx, :]  # (t+1, 2)
            if t > 0:
                ax.plot(pred_traj[:, 0], pred_traj[:, 1],
                       '--', color=colors_before[kp_idx], alpha=0.85, linewidth=3,
                       label=f'{label_noisy} {kp_idx+1}')
            # Current position
            ax.scatter(pred_traj[-1, 0], pred_traj[-1, 1],
                      c=[colors_before[kp_idx]], s=350, marker='x', alpha=0.95,
                      linewidths=5, zorder=10)

        # Optional diagnostic overlay: visualize index-wise displacement
        # between the two trajectory sets at the current frame.
        if show_pairwise_connectors:
            n_pairs = min(N_pred, N_pred_ordered)
            for obj_idx in range(n_pairs):
                x_before, y_before = pred_positions[t, obj_idx, 0], pred_positions[t, obj_idx, 1]
                x_after, y_after = pred_ordered_positions[t, obj_idx, 0], pred_ordered_positions[t, obj_idx, 1]
                ax.plot(
                    [x_before, x_after], [y_before, y_after],
                    color='black', linestyle=':', linewidth=1.8, alpha=0.55, zorder=8
                )
        
        # Always show legend on every frame
        ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
        
        # Render to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
    
    # Save as GIF (duration in ms: 100ms = 10fps)
    imageio.mimsave(save_path, frames, duration=100, loop=0)
    plt.close(fig)
#####################################################################################

#####################################################################################
def reorder_predictions(
    pred_coords, 
    method='smallest_consecutive_distance',
    print_debug=False,
    return_permutations=False,
):
    
    reordered_predictions = []
    applied_permutations = []
    for b in range(pred_coords.shape[0]):
        pred_seq = pred_coords[b]  # (T, N_pred, 2)
        
        re_ordered_seq = pred_seq.clone()  # Initialize with original order
        permutation_seq = []
        identity_perm = np.arange(pred_seq.shape[1], dtype=np.int64)
        permutation_seq.append(identity_perm.copy())

        if method == 'smallest_consecutive_distance':
            # For 2-body systems: track first particle, second is automatically tracked
            # This is efficient and sufficient when N=2
            
            to_compare = pred_seq[0, 0]  # (2,)

            for t in range(1, pred_seq.shape[0]):
                
                dist_2_obj0 = torch.norm(pred_seq[t] - to_compare, dim=1)  # (N_pred, )
                closest_idx = torch.argmin(dist_2_obj0).item()
                
                if closest_idx != 0:
                    # Swap the closest particle with the first one
                    re_ordered_seq[t, [0, closest_idx]] = pred_seq[t, [closest_idx, 0]]
                    perm_t = identity_perm.copy()
                    perm_t[[0, closest_idx]] = perm_t[[closest_idx, 0]]
                else:
                    perm_t = identity_perm.copy()

                to_compare = re_ordered_seq[t, 0]  # Update to the newly assigned first particle for next comparison
                permutation_seq.append(perm_t)
        
        elif method == 'hungarian':
            # For N>2 systems: use Hungarian algorithm for optimal matching
            # This finds the best global assignment between particles at consecutive timesteps
            
            for t in range(1, pred_seq.shape[0]):
                # For each particle at t-1, find its closest match at time t
                prev_positions = re_ordered_seq[t-1]  # (N, 2)
                curr_positions = pred_seq[t]  # (N, 2)
                
                # Compute pairwise distances: dist[i,j] = distance from prev[i] to curr[j]
                # Shape: (N, N)
                dists = torch.cdist(prev_positions.unsqueeze(0), curr_positions.unsqueeze(0)).squeeze(0)
                
                # Use Hungarian algorithm to find optimal matching
                row_ind, col_ind = linear_sum_assignment(dists.cpu().numpy())
                
                # Apply the permutation to current timestep
                re_ordered_seq[t] = curr_positions[col_ind]
                permutation_seq.append(np.asarray(col_ind, dtype=np.int64))

        else:
            raise ValueError(f"Unknown re-ordering method: {method}. Available methods: 'smallest_consecutive_distance' (for N=2), 'hungarian' (for N>2).")
        
        # check whether any permutation has been applied
        if print_debug:
            print()
            print(f"Video {b}: Checking for re-ordering ...")
            print(f"[DEBUG] Applied re-ordering? {not torch.equal(re_ordered_seq[:, 0, :], pred_coords[b][:, 0, :])}")
            print()

        reordered_predictions.append(re_ordered_seq.cpu().numpy())
        applied_permutations.append(np.stack(permutation_seq, axis=0))

    if return_permutations:
        return reordered_predictions, applied_permutations
    return reordered_predictions
#####################################################################################

#####################################################################################
def _apply_temporal_permutation(array_like, permutation_seq):
    arr = np.asarray(array_like)
    perms = np.asarray(permutation_seq, dtype=np.int64)
    if arr.ndim < 2:
        raise ValueError(f"Expected arr.ndim >= 2 for temporal permutation, got shape={arr.shape}")
    if perms.ndim != 2:
        raise ValueError(f"Expected permutation_seq ndim=2 [T,N], got shape={perms.shape}")
    if arr.shape[0] != perms.shape[0] or arr.shape[1] != perms.shape[1]:
        raise ValueError(
            f"Temporal permutation shape mismatch: arr.shape={arr.shape}, permutation_seq.shape={perms.shape}"
        )
    reordered = np.empty_like(arr)
    for t in range(arr.shape[0]):
        reordered[t] = arr[t, perms[t], ...]
    return reordered


def _pearson_1d_np(a, b, eps=1e-8):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b)) + eps
    if denom <= eps:
        return 0.0
    return float(np.sum(a * b) / denom)


def _build_alignment_cost_matrix(P, G, eps=1e-8):
    if P.shape != G.shape:
        raise ValueError(f"Expected matching shapes, got P={P.shape}, G={G.shape}")
    _, N, D = P.shape
    if D != 2:
        raise ValueError(f"Expected last dim D=2, got D={D}")
    cost = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        px = P[:, i, 0]
        py = P[:, i, 1]
        for j in range(N):
            gx = G[:, j, 0]
            gy = G[:, j, 1]
            rx = _pearson_1d_np(px, gx, eps=eps)
            ry = _pearson_1d_np(py, gy, eps=eps)
            sim = 0.5 * (rx + ry)
            cost[i, j] = 1.0 - sim
    return cost


def _hungarian_perm_pred_for_gt(P, G, eps=1e-8):
    cost = _build_alignment_cost_matrix(P, G, eps=eps)
    row_ind, col_ind = linear_sum_assignment(cost)
    perm_pred_for_gt = np.empty(P.shape[1], dtype=np.int64)
    for pred_i, gt_j in zip(row_ind, col_ind):
        perm_pred_for_gt[gt_j] = pred_i
    return perm_pred_for_gt


def _compute_prediction_metrics(pred_coordinates, gt_coordinates_, eps=1e-8):
    pred_coordinates = np.asarray(pred_coordinates, dtype=np.float64)
    gt_coordinates_ = np.asarray(gt_coordinates_, dtype=np.float64)
    if pred_coordinates.shape != gt_coordinates_.shape:
        raise ValueError(
            f"Shape mismatch pred {pred_coordinates.shape} vs gt {gt_coordinates_.shape}. "
            "Expected both to be [V, T, N, 2]."
        )

    V, T, N, D = pred_coordinates.shape
    if D != 2:
        raise ValueError(f"Expected last dimension D=2, got D={D}.")

    def _rmse(A, B):
        return float(np.sqrt(np.mean((A - B) ** 2)))

    def _r2_score(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
        sse = np.sum((y_true - y_pred) ** 2)
        sst = np.sum((y_true - y_true.mean()) ** 2) + eps
        return float(1.0 - sse / sst)

    def _mean_std(xs):
        xs = np.asarray(xs, dtype=np.float64)
        return float(xs.mean()), float(xs.std())

    per_video_mean_pearson = []
    per_video_mean_pearson_vel = []
    per_video_rmse = []
    per_video_nrmse_bbox = []
    per_video_nrmse_std = []
    per_video_r2 = []

    for v in range(V):
        pred_v = pred_coordinates[v]
        gt_v = gt_coordinates_[v]

        rs = []
        for n in range(N):
            rs.append(_pearson_1d_np(pred_v[:, n, 0], gt_v[:, n, 0], eps=eps))
            rs.append(_pearson_1d_np(pred_v[:, n, 1], gt_v[:, n, 1], eps=eps))
        per_video_mean_pearson.append(float(np.mean(rs)) if len(rs) > 0 else 0.0)

        if T >= 2:
            d_pred = np.diff(pred_v, axis=0)
            d_gt = np.diff(gt_v, axis=0)
            rvs = []
            for n in range(N):
                rvs.append(_pearson_1d_np(d_pred[:, n, 0], d_gt[:, n, 0], eps=eps))
                rvs.append(_pearson_1d_np(d_pred[:, n, 1], d_gt[:, n, 1], eps=eps))
            per_video_mean_pearson_vel.append(float(np.mean(rvs)) if len(rvs) > 0 else 0.0)
        else:
            per_video_mean_pearson_vel.append(0.0)

        rmse_v = _rmse(pred_v, gt_v)
        per_video_rmse.append(rmse_v)

        gt_flat = gt_v.reshape(-1, 2)
        bbox_diag_v = float(np.linalg.norm(gt_flat.max(axis=0) - gt_flat.min(axis=0)))
        std_v = float(gt_v.reshape(-1).std())

        per_video_nrmse_bbox.append(rmse_v / (bbox_diag_v + eps))
        per_video_nrmse_std.append(rmse_v / (std_v + eps))
        per_video_r2.append(_r2_score(gt_v, pred_v))

    mean_r, std_r = _mean_std(per_video_mean_pearson)
    mean_rv, std_rv = _mean_std(per_video_mean_pearson_vel)
    mean_rmse, std_rmse = _mean_std(per_video_rmse)
    mean_nrmse_bbox, std_nrmse_bbox = _mean_std(per_video_nrmse_bbox)
    mean_nrmse_std, std_nrmse_std = _mean_std(per_video_nrmse_std)
    mean_r2, std_r2 = _mean_std(per_video_r2)

    return {
        "mean_pearson_pos_mean": mean_r,
        "mean_pearson_pos_std": std_r,
        "mean_pearson_vel_mean": mean_rv,
        "mean_pearson_vel_std": std_rv,
        "rmse_mean": mean_rmse,
        "rmse_std": std_rmse,
        "nrmse_bbox_mean": mean_nrmse_bbox,
        "nrmse_bbox_std": std_nrmse_bbox,
        "nrmse_std_mean": mean_nrmse_std,
        "nrmse_std_std": std_nrmse_std,
        "r2_mean": mean_r2,
        "r2_std": std_r2,
    }


def _set_random_seed(seed):
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class NonlinearProbeMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_hidden_layers=2, output_dim=2):
        super().__init__()
        layers = []
        in_dim = int(input_dim)
        hidden_dim = int(hidden_dim)
        num_hidden_layers = int(num_hidden_layers)
        if num_hidden_layers < 1:
            raise ValueError(f"num_hidden_layers must be >= 1, got {num_hidden_layers}")
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, int(output_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _fit_nonlinear_probe(
    train_inputs,
    train_targets,
    device,
    hidden_dim=64,
    num_hidden_layers=2,
    epochs=25,
    batch_size=4096,
    lr=1e-3,
    weight_decay=1e-6,
    seed=0,
):
    train_inputs = np.asarray(train_inputs, dtype=np.float32)
    train_targets = np.asarray(train_targets, dtype=np.float32)
    epochs = int(epochs)
    if train_inputs.ndim != 2:
        raise ValueError(f"Expected train_inputs ndim=2 [M,D], got shape={train_inputs.shape}")
    if train_targets.ndim != 2 or train_targets.shape[1] != 2:
        raise ValueError(f"Expected train_targets shape [M,2], got {train_targets.shape}")
    if train_inputs.shape[0] != train_targets.shape[0]:
        raise ValueError(
            f"Input/target sample mismatch: {train_inputs.shape[0]} vs {train_targets.shape[0]}"
        )
    if train_inputs.shape[0] == 0:
        raise ValueError("Cannot fit nonlinear probe with zero training samples.")
    if epochs < 1:
        raise ValueError(f"nonlinear probe epochs must be >= 1, got {epochs}.")

    _set_random_seed(seed)
    device = torch.device(device)

    x_cpu = torch.from_numpy(train_inputs)
    y_cpu = torch.from_numpy(train_targets)
    x_mean = x_cpu.mean(dim=0)
    x_std = x_cpu.std(dim=0, unbiased=False).clamp_min(1e-6)
    y_mean = y_cpu.mean(dim=0)
    y_std = y_cpu.std(dim=0, unbiased=False).clamp_min(1e-6)

    dataset = TensorDataset((x_cpu - x_mean) / x_std, (y_cpu - y_mean) / y_std)
    effective_batch_size = min(int(batch_size), len(dataset))
    if effective_batch_size <= 0:
        effective_batch_size = len(dataset)
    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        drop_last=False,
        generator=generator,
    )

    model = NonlinearProbeMLP(
        input_dim=train_inputs.shape[1],
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        output_dim=2,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    criterion = nn.MSELoss()

    loss_history = []
    for epoch_idx in range(epochs):
        model.train()
        running_loss = 0.0
        sample_count = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            batch_size_curr = xb.shape[0]
            running_loss += float(loss.detach().cpu().item()) * batch_size_curr
            sample_count += batch_size_curr

        epoch_loss = running_loss / max(sample_count, 1)
        loss_history.append(epoch_loss)

    num_parameters = int(sum(p.numel() for p in model.parameters()))
    return {
        "model": model,
        "x_mean": x_mean.to(device),
        "x_std": x_std.to(device),
        "y_mean": y_mean.to(device),
        "y_std": y_std.to(device),
        "training_history": {
            "loss": [float(v) for v in loss_history],
            "final_loss": float(loss_history[-1]),
        },
        "num_parameters": num_parameters,
        "config": {
            "hidden_dim": int(hidden_dim),
            "num_hidden_layers": int(num_hidden_layers),
            "epochs": epochs,
            "batch_size": int(effective_batch_size),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "seed": int(seed),
        },
    }


def _predict_with_nonlinear_probe(probe_bundle, inputs):
    inputs = np.asarray(inputs, dtype=np.float32)
    if inputs.ndim != 2:
        raise ValueError(f"Expected inputs ndim=2 [M,D], got shape={inputs.shape}")

    model = probe_bundle["model"]
    device = next(model.parameters()).device
    x_mean = probe_bundle["x_mean"]
    x_std = probe_bundle["x_std"]
    y_mean = probe_bundle["y_mean"]
    y_std = probe_bundle["y_std"]

    with torch.no_grad():
        x = torch.from_numpy(inputs).to(device)
        pred_norm = model((x - x_mean) / x_std)
        pred = pred_norm * y_std + y_mean
    return pred.detach().cpu().numpy()


def _build_nonlinear_probe_feature_tensor(variant_payload, probe_name):
    if probe_name == "p_only":
        return np.asarray(variant_payload["p"], dtype=np.float32)
    if probe_name == "p_s":
        return np.concatenate(
            [
                np.asarray(variant_payload["p"], dtype=np.float32),
                np.asarray(variant_payload["s"], dtype=np.float32),
            ],
            axis=-1,
        )
    if probe_name == "p_s_d_t":
        return np.concatenate(
            [
                np.asarray(variant_payload["p"], dtype=np.float32),
                np.asarray(variant_payload["s"], dtype=np.float32),
                np.asarray(variant_payload["d"], dtype=np.float32),
                np.asarray(variant_payload["tau"], dtype=np.float32),
            ],
            axis=-1,
        )
    if probe_name == "p_s_d_t_feat":
        return np.concatenate(
            [
                np.asarray(variant_payload["p"], dtype=np.float32),
                np.asarray(variant_payload["s"], dtype=np.float32),
                np.asarray(variant_payload["d"], dtype=np.float32),
                np.asarray(variant_payload["tau"], dtype=np.float32),
                np.asarray(variant_payload["feat"], dtype=np.float32),
            ],
            axis=-1,
        )
    raise ValueError(f"Unknown nonlinear probe '{probe_name}'")


def _truncate_nonlinear_probe_split_payload(split_payload, seq_len):
    seq_len = int(seq_len)
    out = {
        "mode": split_payload["mode"],
        "matching": split_payload.get("matching", "index_to_index"),
        "source_seq_len": int(split_payload["gt"].shape[1]),
        "eval_seq_len": seq_len,
    }
    for variant_name in ["nominal", "recentered"]:
        if variant_name not in split_payload:
            continue
        variant_payload = split_payload[variant_name]
        out[variant_name] = {
            "p": variant_payload["p"][:, :seq_len].copy(),
            "s": variant_payload["s"][:, :seq_len].copy(),
            "d": variant_payload["d"][:, :seq_len].copy(),
            "tau": variant_payload["tau"][:, :seq_len].copy(),
            "feat": variant_payload["feat"][:, :seq_len].copy(),
            "gt": variant_payload["gt"][:, :seq_len].copy(),
        }
    out["gt"] = split_payload["gt"][:, :seq_len].copy()
    return out


def _apply_supervised_matching_to_variant_payload(variant_payload):
    gt = np.asarray(variant_payload["gt"], dtype=np.float32)
    p = np.asarray(variant_payload["p"], dtype=np.float32)
    s = np.asarray(variant_payload["s"], dtype=np.float32)
    d = np.asarray(variant_payload["d"], dtype=np.float32)
    tau = np.asarray(variant_payload["tau"], dtype=np.float32)
    feat = np.asarray(variant_payload["feat"], dtype=np.float32)

    V = gt.shape[0]
    p_matched = np.empty_like(p)
    s_matched = np.empty_like(s)
    d_matched = np.empty_like(d)
    tau_matched = np.empty_like(tau)
    feat_matched = np.empty_like(feat)
    permutations = []

    for v in range(V):
        perm = _hungarian_perm_pred_for_gt(p[v], gt[v])
        permutations.append(perm.tolist())
        p_matched[v] = p[v][:, perm, :]
        s_matched[v] = s[v][:, perm, :]
        d_matched[v] = d[v][:, perm, :]
        tau_matched[v] = tau[v][:, perm, :]
        feat_matched[v] = feat[v][:, perm, :]

    return {
        "p": p_matched,
        "s": s_matched,
        "d": d_matched,
        "tau": tau_matched,
        "feat": feat_matched,
        "gt": gt.copy(),
        "supervised_matching_permutations": permutations,
    }


def _evaluate_nonlinear_probe_on_split(probe_bundle, variant_payload, probe_name):
    features = _build_nonlinear_probe_feature_tensor(variant_payload, probe_name)
    gt = np.asarray(variant_payload["gt"], dtype=np.float32)
    pred = _predict_with_nonlinear_probe(
        probe_bundle,
        features.reshape(-1, features.shape[-1]),
    ).reshape(gt.shape)
    metrics = _compute_prediction_metrics(pred, gt)
    metrics.update(
        {
            "num_videos": int(gt.shape[0]),
            "T": int(gt.shape[1]),
            "N": int(gt.shape[2]),
            "num_samples": int(np.prod(gt.shape[:3])),
        }
    )
    return metrics


def evaluate_latent_alignment_metrics(
    pred_coordinates, 
    gt_coordinates_, 
    save_dir, 
    mode='valid', 
    name_out_str='smoothed',
    extraction_method='bbox',
    use_hungarian_for_correlation=False,
    reorder_method='smallest_consecutive_distance',
    metrics_filename_prefix='latent_alignment_metrics',
    metrics_filename_extra_parts=None,
    include_bbox_smoothing_tag_in_filename=True,
    include_reorder_hungarian_tags_in_filename=True,
    extra_metadata=None,
    raw_physical_gt_coordinates=None,
    raw_physical_gt_metadata=None,
):
    if pred_coordinates is None:
        raise RuntimeError("pred_coordinates is None. Make sure extract_coordinates=True and all_coordinates is concatenated.")
    if pred_coordinates.shape != gt_coordinates_.shape:
        raise ValueError(
            f"Shape mismatch pred {pred_coordinates.shape} vs gt {gt_coordinates_.shape}. "
            "Expected both to be [V, T, N, 2]."
        )

    V, T, N, D = pred_coordinates.shape
    if D != 2:
        raise ValueError(f"Expected last dimension D=2, got D={D}.")

    eps = 1e-8

    def _pearson_1d(a, b, eps=1e-8):
        """Fast Pearson corr for 1D arrays; returns 0 if variance is ~0."""
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        a = a - a.mean()
        b = b - b.mean()
        denom = np.sqrt(np.sum(a * a) * np.sum(b * b)) + eps
        if denom <= eps:
            return 0.0
        return float(np.sum(a * b) / denom)

    def _build_cost_matrix(P, G):
        """
        Cost matrix C[i,j] for matching pred object i to gt object j.
        Uses mean Pearson over x/y trajectories.
        P, G: [T, N, 2]
        """
        C = np.zeros((N, N), dtype=np.float64)
        for i in range(N):
            px = P[:, i, 0]
            py = P[:, i, 1]
            for j in range(N):
                gx = G[:, j, 0]
                gy = G[:, j, 1]
                rx = _pearson_1d(px, gx, eps=eps)
                ry = _pearson_1d(py, gy, eps=eps)
                sim = 0.5 * (rx + ry)
                C[i, j] = 1.0 - sim
        return C

    def _hungarian_match(P, G):
        """
        Returns P_matched where objects are reordered to match GT ordering (GT index order).
        P_matched: [T, N, 2]
        """
        C = _build_cost_matrix(P, G)
        row_ind, col_ind = linear_sum_assignment(C)  # row: pred idx, col: gt idx
        perm_pred_for_gt = np.empty(N, dtype=np.int64)
        for pred_i, gt_j in zip(row_ind, col_ind):
            perm_pred_for_gt[gt_j] = pred_i
        P_matched = P[:, perm_pred_for_gt, :]
        return P_matched

    def _fit_uniform_scale_translation(P, G, eps=1e-8):
        """
        Fit s (scalar) and b (2,) minimizing || s*P + b - G ||_F^2 over all (t,n).
        P, G: [T, N, 2]
        """
        P_flat = P.reshape(-1, 2).astype(np.float64)
        G_flat = G.reshape(-1, 2).astype(np.float64)

        mu_P = P_flat.mean(axis=0)
        mu_G = G_flat.mean(axis=0)

        P0 = P_flat - mu_P
        G0 = G_flat - mu_G

        denom = np.sum(P0 * P0) + eps
        s = float(np.sum(P0 * G0) / denom)
        b = mu_G - s * mu_P
        return s, b

    def _apply_uniform_transform(P, s, b):
        """Apply P' = s*P + b."""
        return P * s + b.reshape(1, 1, 2)

    def _fit_affine_transform(P, G):
        """
        Fit full affine map Y = XW + b from P->G over all (v,t,n).
        Returns:
            W: (2, 2), b: (2,)
        """
        P_flat = P.reshape(-1, 2).astype(np.float64)
        G_flat = G.reshape(-1, 2).astype(np.float64)
        X_aug = np.concatenate(
            [P_flat, np.ones((P_flat.shape[0], 1), dtype=np.float64)],
            axis=1
        )  # [M,3]
        B, _, _, _ = np.linalg.lstsq(X_aug, G_flat, rcond=None)  # [3,2]
        W = B[:2, :]  # [2,2]
        b = B[2, :]   # [2]
        return W, b

    def _apply_affine_transform(P, W, b):
        """Apply P' = P@W + b for arrays with trailing dim=2."""
        out = np.einsum('...i,ij->...j', P, W)
        return out + b.reshape(*([1] * (out.ndim - 1)), 2)

    def _rmse(A, B):
        return float(np.sqrt(np.mean((A - B) ** 2)))

    def _r2_score(y_true, y_pred, eps=1e-8):
        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
        sse = np.sum((y_true - y_pred) ** 2)
        sst = np.sum((y_true - y_true.mean()) ** 2) + eps
        return float(1.0 - sse / sst)

    def _mean_std(xs):
        xs = np.asarray(xs, dtype=np.float64)
        return float(xs.mean()), float(xs.std())

    def _compute_alignment_metrics(aligned, target):
        """
        Compute per-video metrics and aggregate mean/std.
        aligned, target: [V, T, N, 2]
        """
        per_video_mean_pearson = []
        per_video_mean_pearson_vel = []
        per_video_mean_pearson_acc = []
        per_video_rmse = []
        per_video_nrmse_bbox = []
        per_video_nrmse_std = []
        per_video_r2 = []
        per_video_smoothness_ratio = []
        per_video_acc_rms_aligned = []
        per_video_acc_rms_target = []

        for v in range(V):
            A = aligned[v]
            Y = target[v]

            rs = []
            for n in range(N):
                rs.append(_pearson_1d(A[:, n, 0], Y[:, n, 0], eps=eps))
                rs.append(_pearson_1d(A[:, n, 1], Y[:, n, 1], eps=eps))
            per_video_mean_pearson.append(float(np.mean(rs)) if len(rs) > 0 else 0.0)

            if T >= 2:
                dA = np.diff(A, axis=0)
                dY = np.diff(Y, axis=0)
                rvs = []
                for n in range(N):
                    rvs.append(_pearson_1d(dA[:, n, 0], dY[:, n, 0], eps=eps))
                    rvs.append(_pearson_1d(dA[:, n, 1], dY[:, n, 1], eps=eps))
                per_video_mean_pearson_vel.append(float(np.mean(rvs)) if len(rvs) > 0 else 0.0)
            else:
                per_video_mean_pearson_vel.append(0.0)

            if T >= 3:
                ddA = A[2:] - 2.0 * A[1:-1] + A[:-2]
                ddY = Y[2:] - 2.0 * Y[1:-1] + Y[:-2]
                ras = []
                for n in range(N):
                    ras.append(_pearson_1d(ddA[:, n, 0], ddY[:, n, 0], eps=eps))
                    ras.append(_pearson_1d(ddA[:, n, 1], ddY[:, n, 1], eps=eps))
                per_video_mean_pearson_acc.append(float(np.mean(ras)) if len(ras) > 0 else 0.0)

                acc_rms_A = float(np.sqrt(np.mean(ddA ** 2)))
                acc_rms_Y = float(np.sqrt(np.mean(ddY ** 2)))
                per_video_acc_rms_aligned.append(acc_rms_A)
                per_video_acc_rms_target.append(acc_rms_Y)
                per_video_smoothness_ratio.append(acc_rms_A / (acc_rms_Y + eps))
            else:
                per_video_mean_pearson_acc.append(0.0)
                per_video_acc_rms_aligned.append(0.0)
                per_video_acc_rms_target.append(0.0)
                per_video_smoothness_ratio.append(0.0)

            rmse_v = _rmse(A, Y)
            per_video_rmse.append(rmse_v)

            Y_flat2 = Y.reshape(-1, 2)
            bbox_diag_v = float(np.linalg.norm(Y_flat2.max(axis=0) - Y_flat2.min(axis=0)))
            std_v = float(Y.reshape(-1).std())

            per_video_nrmse_bbox.append(rmse_v / (bbox_diag_v + eps))
            per_video_nrmse_std.append(rmse_v / (std_v + eps))
            per_video_r2.append(_r2_score(Y, A, eps=eps))

        mean_r, std_r = _mean_std(per_video_mean_pearson)
        mean_rv, std_rv = _mean_std(per_video_mean_pearson_vel)
        mean_ra, std_ra = _mean_std(per_video_mean_pearson_acc)
        mean_rmse, std_rmse = _mean_std(per_video_rmse)
        mean_nrmse_bbox, std_nrmse_bbox = _mean_std(per_video_nrmse_bbox)
        mean_nrmse_std, std_nrmse_std = _mean_std(per_video_nrmse_std)
        mean_r2, std_r2 = _mean_std(per_video_r2)
        mean_smooth, std_smooth = _mean_std(per_video_smoothness_ratio)
        mean_acc_A, std_acc_A = _mean_std(per_video_acc_rms_aligned)
        mean_acc_Y, std_acc_Y = _mean_std(per_video_acc_rms_target)

        return {
            "mean_pearson_pos_mean": mean_r,
            "mean_pearson_pos_std": std_r,
            "mean_pearson_vel_mean": mean_rv,
            "mean_pearson_vel_std": std_rv,
            "mean_pearson_acc_mean": mean_ra,
            "mean_pearson_acc_std": std_ra,
            "smoothness_ratio_acc_rms_mean": mean_smooth,
            "smoothness_ratio_acc_rms_std": std_smooth,
            "acc_rms_aligned_mean": mean_acc_A,
            "acc_rms_aligned_std": std_acc_A,
            "acc_rms_target_mean": mean_acc_Y,
            "acc_rms_target_std": std_acc_Y,
            "rmse_mean": mean_rmse,
            "rmse_std": std_rmse,
            "nrmse_bbox_mean": mean_nrmse_bbox,
            "nrmse_bbox_std": std_nrmse_bbox,
            "nrmse_std_mean": mean_nrmse_std,
            "nrmse_std_std": std_nrmse_std,
            "r2_mean": mean_r2,
            "r2_std": std_r2,
        }

    def _compute_linear_fit_metric_bundle(target_coordinates):
        """Compute all linear alignment blocks for the already selected pred_eval."""
        if target_coordinates.shape != pred_eval.shape:
            raise ValueError(
                f"Shape mismatch pred {pred_eval.shape} vs target {target_coordinates.shape}."
            )

        s_fwd_local, b_fwd_local = _fit_uniform_scale_translation(pred_eval, target_coordinates, eps=eps)
        pred_uniform_fwd_local = _apply_uniform_transform(pred_eval, s_fwd_local, b_fwd_local)
        metrics_uniform_fwd_local = _compute_alignment_metrics(pred_uniform_fwd_local, target_coordinates)
        metrics_uniform_fwd_local["fit_params"] = {
            "scale": float(s_fwd_local),
            "translation": [float(b_fwd_local[0]), float(b_fwd_local[1])],
        }

        s_rev_local, b_rev_local = _fit_uniform_scale_translation(target_coordinates, pred_eval, eps=eps)
        target_uniform_rev_local = _apply_uniform_transform(target_coordinates, s_rev_local, b_rev_local)
        metrics_uniform_rev_local = _compute_alignment_metrics(target_uniform_rev_local, pred_eval)
        metrics_uniform_rev_local["fit_params"] = {
            "scale": float(s_rev_local),
            "translation": [float(b_rev_local[0]), float(b_rev_local[1])],
        }

        W_fwd_local, t_fwd_local = _fit_affine_transform(pred_eval, target_coordinates)
        pred_affine_fwd_local = _apply_affine_transform(pred_eval, W_fwd_local, t_fwd_local)
        metrics_affine_fwd_local = _compute_alignment_metrics(pred_affine_fwd_local, target_coordinates)
        metrics_affine_fwd_local["fit_params"] = {
            "matrix": [[float(W_fwd_local[0, 0]), float(W_fwd_local[0, 1])],
                       [float(W_fwd_local[1, 0]), float(W_fwd_local[1, 1])]],
            "translation": [float(t_fwd_local[0]), float(t_fwd_local[1])],
            "determinant": float(np.linalg.det(W_fwd_local)),
        }

        W_rev_local, t_rev_local = _fit_affine_transform(target_coordinates, pred_eval)
        target_affine_rev_local = _apply_affine_transform(target_coordinates, W_rev_local, t_rev_local)
        metrics_affine_rev_local = _compute_alignment_metrics(target_affine_rev_local, pred_eval)
        metrics_affine_rev_local["fit_params"] = {
            "matrix": [[float(W_rev_local[0, 0]), float(W_rev_local[0, 1])],
                       [float(W_rev_local[1, 0]), float(W_rev_local[1, 1])]],
            "translation": [float(t_rev_local[0]), float(t_rev_local[1])],
            "determinant": float(np.linalg.det(W_rev_local)),
        }

        return {
            "uniform_forward": metrics_uniform_fwd_local,
            "uniform_reverse": metrics_uniform_rev_local,
            "affine_forward": metrics_affine_fwd_local,
            "affine_reverse": metrics_affine_rev_local,
        }

    def _print_metrics_block(title, metrics, params=None):
        print(f"{title}")
        print(f"  Mean Pearson (pos)       : {metrics['mean_pearson_pos_mean']:.4f} ± {metrics['mean_pearson_pos_std']:.4f}")
        print(f"  Mean Pearson (vel, diff) : {metrics['mean_pearson_vel_mean']:.4f} ± {metrics['mean_pearson_vel_std']:.4f}")
        print(f"  Mean Pearson (acc, diff2): {metrics['mean_pearson_acc_mean']:.4f} ± {metrics['mean_pearson_acc_std']:.4f}")
        print(f"  Smoothness ratio (acc)   : {metrics['smoothness_ratio_acc_rms_mean']:.4f} ± {metrics['smoothness_ratio_acc_rms_std']:.4f}")
        print(f"  RMSE (aligned)           : {metrics['rmse_mean']:.6f} ± {metrics['rmse_std']:.6f}")
        print(f"  NRMSE (bbox diag)        : {metrics['nrmse_bbox_mean']:.6f} ± {metrics['nrmse_bbox_std']:.6f}")
        print(f"  NRMSE (target std)       : {metrics['nrmse_std_mean']:.6f} ± {metrics['nrmse_std_std']:.6f}")
        print(f"  R^2 (aligned)            : {metrics['r2_mean']:.4f} ± {metrics['r2_std']:.4f}")
        if params is not None:
            for k, v in params.items():
                print(f"  {k:24s}: {v}")

    if use_hungarian_for_correlation:
        pred_hungarian = []
        for v in range(V):
            pred_hungarian.append(_hungarian_match(pred_coordinates[v], gt_coordinates_[v]))
        pred_hungarian = np.stack(pred_hungarian, axis=0)
        pred_eval = pred_hungarian
        matching_mode = "hungarian"
    else:
        pred_hungarian = None
        pred_eval = pred_coordinates
        matching_mode = "index_to_index"

    image_space_metrics = _compute_linear_fit_metric_bundle(gt_coordinates_)
    metrics_uniform_fwd = image_space_metrics["uniform_forward"]
    metrics_uniform_rev = image_space_metrics["uniform_reverse"]
    metrics_affine_fwd = image_space_metrics["affine_forward"]
    metrics_affine_rev = image_space_metrics["affine_reverse"]

    raw_physical_metrics = None
    if raw_physical_gt_coordinates is not None:
        raw_physical_gt_coordinates = np.asarray(raw_physical_gt_coordinates, dtype=np.float64)
        if raw_physical_gt_coordinates.shape != gt_coordinates_.shape:
            raise ValueError(
                f"Shape mismatch raw physical gt {raw_physical_gt_coordinates.shape} "
                f"vs image-space gt {gt_coordinates_.shape}."
            )
        raw_physical_metrics = _compute_linear_fit_metric_bundle(raw_physical_gt_coordinates)
        raw_physical_metrics["target_coordinate_system"] = "raw_physical_yx"
        raw_physical_metrics["target_coordinate_note"] = (
            "Raw physical coordinates are stored in latent-compatible [y, x] order. "
            "For two-body data this preserves object order while exposing the vertical-axis "
            "flip and physical-unit scaling relative to rendered image coordinates."
        )
        raw_physical_metrics["matching_note"] = (
            "If Hungarian matching is enabled, predicted objects are matched once using "
            "the rendered-image latent target; the same matched prediction ordering is "
            "then reused for raw-physical metrics."
        )
        if raw_physical_gt_metadata is not None:
            raw_physical_metrics["metadata"] = raw_physical_gt_metadata

    print("\n" + "=" * 80)
    print(f"LATENT ALIGNMENT METRICS (matching={matching_mode})")
    print(f"Videos evaluated: {V}, T={T}, N={N}")
    print("-" * 80)
    _print_metrics_block(
        "UNIFORM FORWARD (pred -> gt)",
        metrics_uniform_fwd,
        params=metrics_uniform_fwd["fit_params"]
    )
    print("-" * 80)
    _print_metrics_block(
        "UNIFORM REVERSE (gt -> pred)",
        metrics_uniform_rev,
        params=metrics_uniform_rev["fit_params"]
    )
    print("-" * 80)
    _print_metrics_block(
        "AFFINE FORWARD (pred -> gt)",
        metrics_affine_fwd,
        params=metrics_affine_fwd["fit_params"]
    )
    print("-" * 80)
    _print_metrics_block(
        "AFFINE REVERSE (gt -> pred)",
        metrics_affine_rev,
        params=metrics_affine_rev["fit_params"]
    )
    if raw_physical_metrics is not None:
        print("-" * 80)
        print("RAW PHYSICAL TARGET METRICS available in JSON under key: raw_physical")
    print("=" * 80 + "\n")

    # Optional: save top-k visual comparisons of supervised reordering effect
    if use_hungarian_for_correlation and save_dir is not None:
        try:
            supervised_comp_dir = os.path.join(save_dir, "supervised_reordering_comparison")
            os.makedirs(supervised_comp_dir, exist_ok=True)

            # Measure how strongly supervised reordering changes predicted trajectories:
            # mean L2 displacement between pre/post-Hungarian trajectories.
            per_video_reorder_delta = np.linalg.norm(pred_coordinates - pred_hungarian, axis=-1).mean(axis=(1, 2))
            k = min(10, V)
            top_video_ids = np.argsort(-per_video_reorder_delta)[:k]

            comparison_payload = []
            for rank, vid in enumerate(top_video_ids, start=1):
                vid_int = int(vid)
                before_traj = pred_coordinates[vid_int]
                after_traj = pred_hungarian[vid_int]

                # Infer explicit index mapping between "after" and "before" trajectories.
                n_before = before_traj.shape[1]
                n_after = after_traj.shape[1]
                idx_cost = np.zeros((n_after, n_before), dtype=np.float64)
                for after_idx in range(n_after):
                    for before_idx in range(n_before):
                        idx_cost[after_idx, before_idx] = float(
                            np.linalg.norm(after_traj[:, after_idx, :] - before_traj[:, before_idx, :], axis=-1).mean()
                        )
                map_rows, map_cols = linear_sum_assignment(idx_cost)
                mapping_pairs = sorted((int(r), int(c)) for r, c in zip(map_rows, map_cols))
                mapping_text = ", ".join([f"A{r+1}->B{c+1}" for r, c in mapping_pairs])
                mapping_mean_l2 = float(idx_cost[map_rows, map_cols].mean()) if len(map_rows) > 0 else 0.0

                save_path = os.path.join(
                    supervised_comp_dir,
                    (
                        f"supervised_reorder_comparison_{extraction_method}_{mode}"
                        f"_reorder_{reorder_method}_rank{rank:02d}_vid{vid_int:04d}.gif"
                    )
                )
                _create_trajectory_video(
                    before_traj,
                    after_traj,
                    save_path,
                    keys_to_plot=['pred_positions', 'pred_ordered_positions'],
                    show_pairwise_connectors=True,
                    title_override_template=(
                        "Before vs After Supervised Hungarian\n"
                        "Frame {t}/{T} (Circle=After, X=Before, dotted line=index-wise delta)"
                    ),
                    annotation_text=f"Index map (After->Before): {mapping_text}\nMean matched L2: {mapping_mean_l2:.4f}",
                )
                comparison_payload.append(
                    {
                        "rank": rank,
                        "video_index": vid_int,
                        "mean_l2_delta_before_vs_after_hungarian": float(per_video_reorder_delta[vid_int]),
                        "index_map_after_to_before": [
                            {"after_index": r, "before_index": c} for r, c in mapping_pairs
                        ],
                        "mean_l2_for_index_map": mapping_mean_l2,
                        "video_path": save_path,
                    }
                )

            summary_path = os.path.join(
                supervised_comp_dir,
                (
                    f"supervised_reordering_comparison_summary_{extraction_method}_{mode}"
                    f"_reorder_{reorder_method}.json"
                ),
            )
            with open(summary_path, "w") as f:
                json.dump(
                    {
                        "num_videos": int(V),
                        "top_k": int(k),
                        "ranking_metric": "mean_l2_delta_before_vs_after_hungarian",
                        "videos": comparison_payload,
                    },
                    f,
                    indent=2,
                )
            print(f"Saved supervised reordering comparison videos to: {supervised_comp_dir}")
            print(f"Saved supervised reordering summary to: {summary_path}")
        except Exception as e:
            print(f"[WARN] Could not save supervised reordering comparisons: {e}")

    metrics_out = {
        "num_videos": V,
        "T": T,
        "N": N,
        "matching": matching_mode,
        "target_coordinate_system": "rendered_image_latent_yx",
        "metric_definitions": {
            "velocity": "first temporal difference q[t+1] - q[t]",
            "discrete_acceleration": (
                "second temporal difference q[t+2] - 2*q[t+1] + q[t]; "
                "the common dt^-2 factor is omitted"
            ),
            "smoothness_ratio_acc_rms": (
                "RMS discrete acceleration of the aligned prediction divided by "
                "RMS discrete acceleration of the target"
            ),
        },
        "uniform_forward": metrics_uniform_fwd,
        "uniform_reverse": metrics_uniform_rev,
        "affine_forward": metrics_affine_fwd,
        "affine_reverse": metrics_affine_rev,
    }
    if raw_physical_metrics is not None:
        metrics_out["raw_physical"] = raw_physical_metrics
    if extra_metadata is not None:
        metrics_out["extra_metadata"] = extra_metadata

    # Optional: save metrics to JSON in save_dir
    if save_dir is not None:
        try:
            file_parts = [metrics_filename_prefix, extraction_method, mode]
            if extraction_method == "bbox" and include_bbox_smoothing_tag_in_filename:
                smooth_tag = "smoothed" if name_out_str == "smoothed" else "non_smoothed"
                file_parts.append(smooth_tag)
            if metrics_filename_extra_parts is not None:
                file_parts.extend([str(p) for p in metrics_filename_extra_parts if str(p) != ""])
            if include_reorder_hungarian_tags_in_filename:
                file_parts.append(f"reorder_{reorder_method}")
                file_parts.append("hungarian_on" if use_hungarian_for_correlation else "hungarian_off")
            metrics_filename = "_".join(file_parts) + ".json"
            metrics_path = os.path.join(save_dir, metrics_filename)
            with open(metrics_path, "w") as f:
                json.dump(metrics_out, f, indent=2)
            print(f"Saved latent alignment metrics to: {metrics_path}")
        except Exception as e:
            print(f"[WARN] Could not save metrics JSON: {e}")

    return metrics_out
#####################################################################################

#####################################################################################
def _format_float_for_filename(value):
    return f"{float(value):.12g}".replace("-", "neg_").replace(".", "p")


def _apply_relative_step_noise_to_gt(
    gt_coordinates, noise_alpha, noise_seed=0
):
    """
    Replicate DEL's `relative_step` noise:
      noise_std = noise_alpha * median(||x_{t+1}-x_t||) / sqrt(state_dim)
    with i.i.d. Gaussian noise per coordinate.
    """
    gt = np.asarray(gt_coordinates, dtype=np.float32)
    if gt.ndim != 4 or gt.shape[-1] != 2:
        raise ValueError(f"Expected gt shape [V, T, N, 2], got {gt.shape}")

    V, T, N, D = gt.shape
    state_dim = N * D
    gt_flat = torch.as_tensor(gt.reshape(V, T, state_dim), dtype=torch.float32)

    if T >= 2:
        step_norms = torch.linalg.norm(gt_flat[:, 1:, :] - gt_flat[:, :-1, :], dim=-1).reshape(-1)
        step_disp_median = float(step_norms.median().item()) if step_norms.numel() > 0 else 0.0
    else:
        step_disp_median = 0.0

    noise_std = float(float(noise_alpha) * step_disp_median / math.sqrt(float(state_dim)))

    noisy_flat = gt_flat.clone()
    if noise_std > 0.0:
        generator = torch.Generator()
        if noise_seed is not None:
            generator.manual_seed(int(noise_seed))
        noise = torch.randn(noisy_flat.shape, dtype=noisy_flat.dtype, generator=generator) * noise_std
        noisy_flat = noisy_flat + noise

    noisy_gt = noisy_flat.reshape(V, T, N, D).cpu().numpy()
    return noisy_gt, step_disp_median, noise_std

def _compute_raw_no_fit_alignment_metrics(
    pred_coordinates, target_coordinates, eps=1e-8
):
    """
    Compute raw (no-fit) alignment metrics directly between pred and target.
    Both inputs must be [V, T, N, 2].
    """
    pred = np.asarray(pred_coordinates, dtype=np.float64)
    target = np.asarray(target_coordinates, dtype=np.float64)
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch for raw metrics: pred {pred.shape} vs target {target.shape}")
    if pred.ndim != 4 or pred.shape[-1] != 2:
        raise ValueError(f"Expected [V, T, N, 2], got {pred.shape}")

    V, T, N, _ = pred.shape

    def _pearson_1d(a, b):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        a = a - a.mean()
        b = b - b.mean()
        denom = np.sqrt(np.sum(a * a) * np.sum(b * b)) + eps
        if denom <= eps:
            return 0.0
        return float(np.sum(a * b) / denom)

    def _rmse(A, B):
        return float(np.sqrt(np.mean((A - B) ** 2)))

    def _r2_score(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
        sse = np.sum((y_true - y_pred) ** 2)
        sst = np.sum((y_true - y_true.mean()) ** 2) + eps
        return float(1.0 - sse / sst)

    def _mean_std(xs):
        xs = np.asarray(xs, dtype=np.float64)
        return float(xs.mean()), float(xs.std())

    per_video_mean_pearson = []
    per_video_mean_pearson_vel = []
    per_video_rmse = []
    per_video_nrmse_bbox = []
    per_video_nrmse_std = []
    per_video_r2 = []

    for v in range(V):
        A = pred[v]
        Y = target[v]

        rs = []
        for n in range(N):
            rs.append(_pearson_1d(A[:, n, 0], Y[:, n, 0]))
            rs.append(_pearson_1d(A[:, n, 1], Y[:, n, 1]))
        per_video_mean_pearson.append(float(np.mean(rs)) if len(rs) > 0 else 0.0)

        if T >= 2:
            dA = np.diff(A, axis=0)
            dY = np.diff(Y, axis=0)
            rvs = []
            for n in range(N):
                rvs.append(_pearson_1d(dA[:, n, 0], dY[:, n, 0]))
                rvs.append(_pearson_1d(dA[:, n, 1], dY[:, n, 1]))
            per_video_mean_pearson_vel.append(float(np.mean(rvs)) if len(rvs) > 0 else 0.0)
        else:
            per_video_mean_pearson_vel.append(0.0)

        rmse_v = _rmse(A, Y)
        per_video_rmse.append(rmse_v)

        Y_flat2 = Y.reshape(-1, 2)
        bbox_diag_v = float(np.linalg.norm(Y_flat2.max(axis=0) - Y_flat2.min(axis=0)))
        std_v = float(Y.reshape(-1).std())

        per_video_nrmse_bbox.append(rmse_v / (bbox_diag_v + eps))
        per_video_nrmse_std.append(rmse_v / (std_v + eps))
        per_video_r2.append(_r2_score(Y, A))

    mean_r, std_r = _mean_std(per_video_mean_pearson)
    mean_rv, std_rv = _mean_std(per_video_mean_pearson_vel)
    mean_rmse, std_rmse = _mean_std(per_video_rmse)
    mean_nrmse_bbox, std_nrmse_bbox = _mean_std(per_video_nrmse_bbox)
    mean_nrmse_std, std_nrmse_std = _mean_std(per_video_nrmse_std)
    mean_r2, std_r2 = _mean_std(per_video_r2)

    return {
        "mean_pearson_pos_mean": mean_r,
        "mean_pearson_pos_std": std_r,
        "mean_pearson_vel_mean": mean_rv,
        "mean_pearson_vel_std": std_rv,
        "rmse_mean": mean_rmse,
        "rmse_std": std_rmse,
        "nrmse_bbox_mean": mean_nrmse_bbox,
        "nrmse_bbox_std": std_nrmse_bbox,
        "nrmse_std_mean": mean_nrmse_std,
        "nrmse_std_std": std_nrmse_std,
        "r2_mean": mean_r2,
        "r2_std": std_r2,
    }

def evaluate_noisy_gt_reference_metrics(
    gt_coordinates,
    save_dir,
    mode='valid',
    extraction_method='latent',
    noise_mode='relative_step',
    noise_alphas=None,
    noise_seed=0,
):
    """
    Compute latent-alignment metrics for clean GT vs noisy GT baselines.
    """
    if noise_alphas is None:
        noise_alphas = [0.05, 0.1, 0.3, 0.5, 0.7]
    if len(noise_alphas) == 0:
        raise ValueError("noise_alphas is empty. Provide at least one alpha value.")
    if noise_mode != 'relative_step':
        raise ValueError(f"Unsupported noise_mode '{noise_mode}'. Expected 'relative_step'.")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    summary_entries = []
    print("\n" + "=" * 80)
    print(f"NOISY-GT REFERENCE EVALUATION ({mode}, extraction={extraction_method})")
    print(f"noise_mode={noise_mode}, noise_alphas={noise_alphas}, noise_seed={noise_seed}")
    print("=" * 80)

    for alpha in noise_alphas:
        alpha = float(alpha)
        noisy_gt, step_disp_median, noise_std = _apply_relative_step_noise_to_gt(
            gt_coordinates=gt_coordinates,
            noise_alpha=alpha,
            noise_seed=noise_seed,
        )

        alpha_tag = _format_float_for_filename(alpha)
        raw_metrics = _compute_raw_no_fit_alignment_metrics(
            pred_coordinates=noisy_gt,
            target_coordinates=gt_coordinates,
        )
        metrics_out = {
            "num_videos": int(gt_coordinates.shape[0]),
            "T": int(gt_coordinates.shape[1]),
            "N": int(gt_coordinates.shape[2]),
            "reference_type": "clean_gt_vs_noisy_gt_raw_no_fit",
            "comparison": "noisy_gt_vs_clean_gt",
            "metrics": raw_metrics,
            "extra_metadata": {
                "noise_mode": noise_mode,
                "noise_alpha": alpha,
                "noise_seed": int(noise_seed),
                "step_disp_median": float(step_disp_median),
                "noise_std": float(noise_std),
            },
        }
        print(
            f"[RAW NO-FIT] alpha={alpha:.6g} | "
            f"pearson_pos={raw_metrics['mean_pearson_pos_mean']:.4f}±{raw_metrics['mean_pearson_pos_std']:.4f} | "
            f"pearson_vel={raw_metrics['mean_pearson_vel_mean']:.4f}±{raw_metrics['mean_pearson_vel_std']:.4f} | "
            f"r2={raw_metrics['r2_mean']:.4f}±{raw_metrics['r2_std']:.4f}"
        )

        metrics_filename = (
            f"latent_alignment_noisy_gt_reference_{extraction_method}_{mode}"
            f"_noise_{noise_mode}_alpha_{alpha_tag}_seed_{int(noise_seed)}.json"
        )
        metrics_path = None if save_dir is None else os.path.join(save_dir, metrics_filename)
        if metrics_path is not None:
            with open(metrics_path, "w") as f:
                json.dump(metrics_out, f, indent=2)
            print(f"Saved noisy-GT reference metrics to: {metrics_path}")

        summary_entries.append({
            "noise_alpha": alpha,
            "noise_std": float(noise_std),
            "step_disp_median": float(step_disp_median),
            "metrics_path": metrics_path,
            "raw_mean_pearson_pos_mean": raw_metrics["mean_pearson_pos_mean"],
            "raw_mean_pearson_vel_mean": raw_metrics["mean_pearson_vel_mean"],
            "raw_r2_mean": raw_metrics["r2_mean"],
        })

    if save_dir is not None:
        summary_path = os.path.join(
            save_dir,
            (
                f"latent_alignment_noisy_gt_reference_summary_{extraction_method}_{mode}"
                f"_noise_{noise_mode}_seed_{int(noise_seed)}.json"
            ),
        )
        with open(summary_path, "w") as f:
            json.dump(
                {
                    "reference_type": "clean_gt_vs_noisy_gt_raw_no_fit",
                    "mode": mode,
                    "extraction_method": extraction_method,
                    "noise_mode": noise_mode,
                    "noise_seed": int(noise_seed),
                    "noise_alphas": [float(a) for a in noise_alphas],
                    "entries": summary_entries,
                },
                f,
                indent=2,
            )
        print(f"Saved noisy-GT reference summary to: {summary_path}")

    return summary_entries
#####################################################################################

#####################################################################################
def _update_filtering_report_json(
    report_path,
    mode,
    extraction_method,
    total_seen,
    kept,
    filtered_ids,
    filtered_details,
):
    """Write or update filtering report JSON at a shared path."""
    if report_path is None:
        return

    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    payload = {}
    if os.path.exists(report_path):
        try:
            with open(report_path, "r") as f:
                payload = json.load(f)
            if not isinstance(payload, dict):
                payload = {}
        except Exception:
            payload = {}

    payload.setdefault("schema_version", "1.0")
    payload.setdefault("modes", {})
    payload["last_updated_utc"] = time_module.strftime("%Y-%m-%dT%H:%M:%SZ", time_module.gmtime())
    payload["modes"][mode] = {
        "extraction_method": extraction_method,
        "total_trajectories_seen": int(total_seen),
        "kept_trajectories": int(kept),
        "filtered_trajectories_count": int(len(filtered_ids)),
        "filtered_trajectory_ids": [int(v) for v in filtered_ids],
        "filtered_fraction": float(len(filtered_ids) / max(total_seen, 1)),
        "filtered_trajectories": filtered_details,
    }

    with open(report_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved filtering report to: {report_path}")

#####################################################################################

#####################################################################################
def _build_local_grid(h, w, device, dtype):
    """
    Build normalized local coordinate grids in [-1, 1] with y/x ordering.
    Returns:
        yy, xx: each with shape [1, 1, 1, h, w]
    """
    y_lin = torch.linspace(-1.0, 1.0, steps=h, device=device, dtype=dtype)
    x_lin = torch.linspace(-1.0, 1.0, steps=w, device=device, dtype=dtype)
    try:
        yy, xx = torch.meshgrid(y_lin, x_lin, indexing='ij')
    except TypeError:
        yy, xx = torch.meshgrid(y_lin, x_lin)
    yy = yy.view(1, 1, 1, h, w)
    xx = xx.view(1, 1, 1, h, w)
    return yy, xx

def _alpha_centroid_yx(alpha_maps, eps=1e-6):
    """
    Compute alpha centroid in normalized y/x coordinates.
    alpha_maps: [B, T, N, H, W]
    Returns:
        centroid_yx: [B, T, N, 2]
        valid_mask: [B, T, N] (alpha mass > eps)
    """
    if alpha_maps.ndim != 5:
        raise ValueError(f"Expected alpha_maps ndim=5 [B,T,N,H,W], got shape={tuple(alpha_maps.shape)}")

    h, w = alpha_maps.shape[-2], alpha_maps.shape[-1]
    yy, xx = _build_local_grid(h, w, alpha_maps.device, alpha_maps.dtype)

    mass = alpha_maps.sum(dim=(-2, -1))
    denom = mass + eps
    mu_y = (alpha_maps * yy).sum(dim=(-2, -1)) / denom
    mu_x = (alpha_maps * xx).sum(dim=(-2, -1)) / denom
    centroid_yx = torch.stack([mu_y, mu_x], dim=-1)
    valid_mask = mass > eps
    return centroid_yx, valid_mask

def _recenter_positions_patch_alpha(
    mu_tot,
    mu_scale,
    dec_objects_original,
    eps=1e-6
):
    """
    Recenter via local decoded patch alpha:
      p_can = p + sigmoid(scale) * mu_local_alpha
    Inputs:
      mu_tot: [B, T, N, 2]
      mu_scale: [B, T, N, 2]
      dec_objects_original: [B, T, N, 4, H_patch, W_patch]
    Returns:
      p_can: [B, T, N, 2]
      valid_mask: [B, T, N]
    """
    if dec_objects_original.ndim != 6:
        raise ValueError(
            f"Expected dec_objects_original ndim=6 [B,T,N,4,H,W], got shape={tuple(dec_objects_original.shape)}"
        )
    if dec_objects_original.shape[3] != 4:
        raise ValueError(
            f"Expected dec_objects_original channel dim=4 (RGBA), got shape={tuple(dec_objects_original.shape)}"
        )

    alpha_patch = dec_objects_original[:, :, :, 0, :, :]
    mu_local_yx, valid_mask = _alpha_centroid_yx(alpha_patch, eps=eps)
    scale_norm = torch.sigmoid(mu_scale)
    delta = scale_norm * mu_local_yx
    delta = torch.where(valid_mask.unsqueeze(-1), delta, torch.zeros_like(delta))
    p_can = mu_tot + delta
    return p_can, valid_mask

def _recenter_positions_global_alpha(mu_tot, alpha_masks, eps=1e-6):
    """
    Recenter via global per-particle alpha masks by directly taking image-space centroid.
    Inputs:
      mu_tot: [B, T, N, 2]
      alpha_masks: [B, T, N, 1, H, W]
    Returns:
      p_can: [B, T, N, 2]
      valid_mask: [B, T, N]
    """
    if alpha_masks.ndim != 6:
        raise ValueError(f"Expected alpha_masks ndim=6 [B,T,N,1,H,W], got shape={tuple(alpha_masks.shape)}")
    alpha_global = alpha_masks.squeeze(3)
    mu_global_yx, valid_mask = _alpha_centroid_yx(alpha_global, eps=eps)
    p_can = torch.where(valid_mask.unsqueeze(-1), mu_global_yx, mu_tot)
    return p_can, valid_mask


def _convert_gt_xy_pixels_to_latent_yx(gt_xy_pixels, image_height, image_width, kp_range):
    """
    Convert dataset GT positions from pixel [x, y] into DDLP latent [y, x].
    The output uses the same kp_range normalization as DDLP keypoints.
    """
    gt_xy = np.asarray(gt_xy_pixels, dtype=np.float32)
    if gt_xy.ndim < 2 or gt_xy.shape[-1] != 2:
        raise ValueError(f"Expected GT positions with trailing dim=2 [x,y], got shape={gt_xy.shape}")

    image_height = int(image_height)
    image_width = int(image_width)
    if image_height <= 1 or image_width <= 1:
        raise ValueError(
            f"Image height/width must be > 1 for pixel normalization, got H={image_height}, W={image_width}"
        )

    kp_min, kp_max = float(kp_range[0]), float(kp_range[1])
    kp_span = kp_max - kp_min
    x_latent = (gt_xy[..., 0] / float(image_width - 1)) * kp_span + kp_min
    y_latent = (gt_xy[..., 1] / float(image_height - 1)) * kp_span + kp_min
    return np.stack([y_latent, x_latent], axis=-1).astype(np.float32, copy=False)


def _looks_like_twobody_physical_npz_root(path):
    if path is None:
        return False
    return (
        os.path.isdir(path)
        and os.path.isdir(os.path.join(path, "training_trajectory"))
        and os.path.isdir(os.path.join(path, "validation_trajectory"))
    )


def _resolve_twobody_physical_npz_root(dataset_root, raw_physical_npz_root="auto"):
    """
    Resolve the synchronized raw-physics NPZ root for two-body DDLP HDF5 datasets.

    Existing DDLP HDF5 files store rendered pixel positions only. The companion
    *_npz directory stores the synchronized raw physical trajectories and the
    global render bounds used to produce those pixels.
    """
    if raw_physical_npz_root is None:
        return None

    raw_arg = str(raw_physical_npz_root).strip()
    if raw_arg == "" or raw_arg.lower() in {"none", "off", "false", "0"}:
        return None

    if raw_arg.lower() != "auto":
        return os.path.abspath(os.path.expanduser(raw_arg))

    if dataset_root is None:
        return None

    dataset_root = os.path.abspath(os.path.expanduser(str(dataset_root)))
    parent = os.path.dirname(dataset_root)
    base = os.path.basename(dataset_root)
    candidates = []

    def _add_candidate(path):
        if path is not None and path not in candidates:
            candidates.append(path)

    _add_candidate(dataset_root.replace("_hdf5", "_npz"))
    if base.endswith("_hdf5"):
        _add_candidate(os.path.join(parent, base[:-5] + "_npz"))
    if "_hdf5" in base:
        _add_candidate(os.path.join(parent, base.replace("_hdf5", "_npz")))

    for candidate in candidates:
        if _looks_like_twobody_physical_npz_root(candidate):
            return candidate
    return None


def _twobody_dataset_index_to_npz_slice(dataset, item_index):
    """
    Mirror TwoBodyDataset.__getitem__ indexing so raw NPZ trajectories align with
    the exact video subsequence fed through the DDLP model.
    """
    mode = getattr(dataset, "mode", None)
    sample_length = int(getattr(dataset, "sample_length", 0))

    if mode == "train":
        split_dir = "training_trajectory"
        if getattr(dataset, "use_subsequences", False):
            seq_per_episode = int(getattr(dataset, "seq_per_episode", 1))
            episode_idx = int(item_index) // seq_per_episode
            subseq_idx = int(item_index) % seq_per_episode
            start_frame = subseq_idx * sample_length
            end_frame = start_frame + sample_length
        else:
            episode_idx = int(item_index)
            start_frame = 0
            end_frame = int(getattr(dataset, "max_eval_frames", sample_length))
    else:
        # The current BIG DDLP test.hdf5 is a copy of val.hdf5, and the
        # synchronized raw trajectories are stored under validation_trajectory.
        split_dir = "validation_trajectory"
        episode_idx = int(item_index)
        start_frame = 0
        end_frame = int(getattr(dataset, "max_eval_frames", sample_length))

    return split_dir, episode_idx, start_frame, end_frame


def _load_twobody_raw_physical_yx_sequence(
    npz_root,
    dataset,
    item_index,
    expected_T=None,
    cache=None,
):
    """
    Load raw physical coordinates for one TwoBodyDataset item.

    Returns coordinates in [T, N, 2] with DDLP-compatible trailing order [y, x].
    The raw NPZ trajectory columns are [x_1, y_1, x_2, y_2, ...].
    """
    if npz_root is None:
        return None, "raw physical NPZ root is not configured"

    split_dir, episode_idx, start_frame, end_frame = _twobody_dataset_index_to_npz_slice(dataset, item_index)
    npz_path = os.path.join(npz_root, split_dir, f"trajectory_{episode_idx}.npz")
    if not os.path.exists(npz_path):
        return None, f"missing raw physical NPZ file: {npz_path}"

    cache_key = (split_dir, episode_idx)
    if cache is not None and cache_key in cache:
        trajectory = cache[cache_key]
    else:
        with np.load(npz_path, allow_pickle=True) as payload:
            if "trajectory" not in payload:
                return None, f"NPZ file has no 'trajectory' array: {npz_path}"
            trajectory = np.asarray(payload["trajectory"], dtype=np.float32)
        if cache is not None:
            cache[cache_key] = trajectory

    if trajectory.ndim != 2 or trajectory.shape[1] % 2 != 0:
        return None, f"raw physical trajectory has invalid shape {trajectory.shape}: {npz_path}"

    if expected_T is not None:
        end_frame = start_frame + int(expected_T)

    if start_frame < 0 or end_frame > trajectory.shape[0]:
        return None, (
            f"requested raw physical slice [{start_frame}:{end_frame}] exceeds "
            f"trajectory length {trajectory.shape[0]} for {npz_path}"
        )

    xy = trajectory[start_frame:end_frame].reshape(end_frame - start_frame, -1, 2)
    yx = xy[..., [1, 0]]
    return yx.astype(np.float32, copy=False), None


def _reshape_sequence_model_output(raw_output, sequence_model, x):
    """Reshape one DDLP forward output from [B*T, ...] tensors to [B,T,...]."""
    B = x.shape[0]
    T = x.shape[1]
    n_kp = sequence_model.n_kp_enc

    kp_p_flat = raw_output.get('kp_p', None)
    if (
        kp_p_flat is None
        or (not torch.is_tensor(kp_p_flat))
        or kp_p_flat.ndim != 3
        or kp_p_flat.shape[0] != (B * T)
        or kp_p_flat.shape[-1] != 2
    ):
        kp_p_reshaped = (raw_output['z_base'] + raw_output['mu_offset']).view(B, T, n_kp, 2)
    else:
        kp_p_reshaped = kp_p_flat.view(B, T, kp_p_flat.shape[1], 2)

    return {
        'kp_p': kp_p_reshaped,
        'z_base': raw_output['z_base'].view(B, T, n_kp, 2),
        'z': raw_output['z'].view(B, T, n_kp, 2),
        'mu_offset': raw_output['mu_offset'].view(B, T, n_kp, 2),
        'logvar_offset': raw_output['logvar_offset'].view(B, T, n_kp, 2),
        'mu_depth': raw_output['mu_depth'].view(B, T, n_kp, 1),
        'z_depth': raw_output['z_depth'].view(B, T, n_kp, 1),
        'mu_scale': raw_output['mu_scale'].view(B, T, n_kp, 2),
        'z_scale': raw_output['z_scale'].view(B, T, n_kp, 2),
        'obj_on': raw_output['obj_on'].view(B, T, n_kp),
        'z_features': raw_output['z_features'].view(B, T, n_kp, -1),
        'z_bg': raw_output['z_bg'].view(B, T, -1),
        'alpha_masks': raw_output['alpha_masks'].view(B, T, n_kp, 1, x.shape[3], x.shape[4]),
        'dec_objects_original': raw_output['dec_objects_original'].view(B, T, n_kp, *raw_output['dec_objects_original'].shape[2:]),
        'dec_objects': raw_output['dec_objects'].view(B, T, x.shape[2], x.shape[3], x.shape[4]),
        'bg': raw_output['bg'].view(B, T, x.shape[2], x.shape[3], x.shape[4]),
        'rec': raw_output['rec'].view(B, T, x.shape[2], x.shape[3], x.shape[4]),
    }


def _encode_sequence_model(sequence_model, x, config, label="model"):
    """Run DDLP temporal encoding and return outputs shaped [B,T,...]."""
    T = x.shape[1]
    timestep_horizon = config['timestep_horizon']

    if T <= timestep_horizon:
        raw_output = sequence_model(x, x_prior=x, deterministic=True, forward_dyn=True)
        return _reshape_sequence_model_output(raw_output, sequence_model, x)

    print(f"Autoregressive encoding ({label}): {T} frames in chunks of {timestep_horizon}")

    all_outputs = []
    continuation_state = None
    for chunk_start in range(0, T, timestep_horizon):
        print("" + "=" * 60)
        print(
            f"  Processing {label} chunk: frames "
            f"{chunk_start} to {min(chunk_start + timestep_horizon, T) - 1}"
        )
        print("" + "=" * 60)

        chunk_end = min(chunk_start + timestep_horizon, T)
        x_chunk = x[:, chunk_start:chunk_end]
        assert x_chunk.shape[1] == timestep_horizon, (
            f"Autoregressive chunk merge requires fixed chunk length equal to timestep_horizon. "
            f"Got chunk_len={x_chunk.shape[1]} vs timestep_horizon={timestep_horizon}. "
            f"Use eval_seq_len as a multiple of timestep_horizon."
        )

        if chunk_start == 0:
            x_prior_chunk = x_chunk
        else:
            x_prior_chunk = torch.cat(
                [x[:, chunk_start - 1:chunk_start], x_chunk[:, :-1]],
                dim=1,
            )

        chunk_output = sequence_model(
            x_chunk.contiguous(),
            x_prior=x_prior_chunk.contiguous(),
            deterministic=True,
            forward_dyn=True,
            continuation_state=continuation_state,
        )
        all_outputs.append(chunk_output)
        continuation_state = chunk_output['continuation_state']

    num_chunks = len(all_outputs)
    B = x.shape[0]
    chunk_T = timestep_horizon

    def reshape_chunk_outputs(outputs_list):
        stacked = torch.stack(outputs_list, dim=0)
        remaining_dims = stacked.shape[2:]
        stacked = stacked.view(num_chunks, B, chunk_T, *remaining_dims)
        stacked = stacked.transpose(0, 1).contiguous()
        total_T = num_chunks * chunk_T
        return stacked.view(B, total_T, *remaining_dims)

    kp_p_chunks = []
    for out in all_outputs:
        kp_chunk = out.get('kp_p', None)
        expected_shape = out['z_base'].shape
        if (
            kp_chunk is None
            or (not torch.is_tensor(kp_chunk))
            or kp_chunk.shape != expected_shape
        ):
            kp_chunk = out['z_base'] + out['mu_offset']
        kp_p_chunks.append(kp_chunk)

    return {
        'kp_p': reshape_chunk_outputs(kp_p_chunks),
        'z_base': reshape_chunk_outputs([out['z_base'] for out in all_outputs]),
        'z': reshape_chunk_outputs([out['z'] for out in all_outputs]),
        'mu_offset': reshape_chunk_outputs([out['mu_offset'] for out in all_outputs]),
        'logvar_offset': reshape_chunk_outputs([out['logvar_offset'] for out in all_outputs]),
        'mu_depth': reshape_chunk_outputs([out['mu_depth'] for out in all_outputs]),
        'z_depth': reshape_chunk_outputs([out['z_depth'] for out in all_outputs]),
        'mu_scale': reshape_chunk_outputs([out['mu_scale'] for out in all_outputs]),
        'z_scale': reshape_chunk_outputs([out['z_scale'] for out in all_outputs]),
        'obj_on': reshape_chunk_outputs([out['obj_on'] for out in all_outputs]),
        'z_features': reshape_chunk_outputs([out['z_features'] for out in all_outputs]),
        'z_bg': reshape_chunk_outputs([out['z_bg'] for out in all_outputs]),
        'alpha_masks': reshape_chunk_outputs([out['alpha_masks'] for out in all_outputs]),
        'dec_objects_original': reshape_chunk_outputs([out['dec_objects_original'] for out in all_outputs]),
        'dec_objects': reshape_chunk_outputs([out['dec_objects'] for out in all_outputs]),
        'bg': reshape_chunk_outputs([out['bg'] for out in all_outputs]),
        'rec': reshape_chunk_outputs([out['rec'] for out in all_outputs]),
    }


def _build_prob_encoder_route_mean(
    route,
    train_out,
    frozen_out,
    latent_recenter_source,
    latent_recenter_eps,
    kp_range,
):
    """Reconstruct the probabilistic encoder mean for C1-family checkpoints."""
    kp_min, kp_max = float(kp_range[0]), float(kp_range[1])
    mu_nom_train = train_out['z_base'] + train_out['mu_offset']
    if route not in {'c1', 'c1-dyn-attrs'}:
        raise ValueError(
            f"Unknown probabilistic encoder route: {route}. "
            "Expected one of ['c1','c1-dyn-attrs']."
        )
    if frozen_out is None:
        raise RuntimeError(
            "prob_encoder_route in {'c1','c1-dyn-attrs'} requires a frozen base DDLP output."
        )

    mu_nom_frozen = frozen_out['z_base'] + frozen_out['mu_offset']
    if latent_recenter_source == 'patch_alpha':
        c0, valid_mask = _recenter_positions_patch_alpha(
            mu_tot=mu_nom_frozen,
            mu_scale=frozen_out['mu_scale'],
            dec_objects_original=frozen_out['dec_objects_original'],
            eps=latent_recenter_eps,
        )
    elif latent_recenter_source == 'global_alpha':
        c0, valid_mask = _recenter_positions_global_alpha(
            mu_tot=mu_nom_frozen,
            alpha_masks=frozen_out['alpha_masks'],
            eps=latent_recenter_eps,
        )
    else:
        raise ValueError(f"Unknown recenter source: {latent_recenter_source}")

    delta = mu_nom_train - mu_nom_frozen.detach()
    return (c0.detach() + delta).clamp(min=kp_min, max=kp_max), valid_mask


def _normalize_prob_encoder_route(route):
    """Return the canonical probabilistic-encoder route label."""
    route_key = "none" if route is None else str(route).strip()
    aliases = {
        "none": "none",
        "c1": "c1",
        "c1-dyn-attrs": "c1-dyn-attrs",
        "c1_dyn_attrs": "c1-dyn-attrs",
    }
    if route_key not in aliases:
        raise ValueError(
            "prob_encoder_route must be one of "
            "['none','c1','c1-dyn-attrs','c1_dyn_attrs'], "
            f"got '{route}'"
        )
    return aliases[route_key]


def _prob_encoder_output_dirname(route, output_tag=None):
    """Directory name used for outputs produced from probabilistic encoders."""
    if output_tag is not None:
        tag = str(output_tag).strip()
        if tag != "" and tag.lower() != "auto":
            return f"prob_encoder_{tag.replace('-', '_')}"
    route = _normalize_prob_encoder_route(route)
    if route == "none":
        return None
    return f"prob_encoder_{route.replace('-', '_')}"
#####################################################################################

#####################################################################################
def video_to_trajectory(
    model, config, device=torch.device('cpu'), 
    mode='valid', batch_size=32, max_batches=None,
    eval_seq_len=None, save_dir=None, 
    latent_eval_save_dir=None,
    visualize_trajectories=True,
    extract_coordinates=False, 
    returns=False,
    evaluate_latent_alignment=False,
    evaluate_noisy_gt_reference=False,
    noisy_gt_noise_mode='relative_step',
    noisy_gt_noise_alphas=None,
    noisy_gt_noise_seed=0,
    use_hungarian_for_correlation=False,
    reorder_method='smallest_consecutive_distance',
    sg_window_length=15,
    sg_polyorder=2,
    extraction_method='bbox',
    filtering_report_path=None,
    latent_position_variant='nominal',
    latent_recenter_source='patch_alpha',
    latent_recenter_nms_source='nominal',
    latent_recenter_eps=1e-6,
    collect_nonlinear_probe_inputs=False,
    return_nonlinear_probe_payload=False,
    respect_max_batches_for_extraction=False,
    prob_encoder_route='none',
    prob_encoder_frozen_model=None,
    raw_physical_npz_root='auto',
):
    """
    Collect video data for visualization purposes.
    Uses temporal tracking across sequences (full model with dynamics).
    
    Args:
        model: Trained DDLP model with temporal tracking
        config: Configuration dictionary
        device: torch device
        mode: Dataset split to evaluate ('train', 'valid', or 'test')
        batch_size: Batch size for evaluation
        max_batches: Maximum number of batches to evaluate in preview mode (extract_coordinates=False).
                     Ignored when extract_coordinates=True.
        eval_seq_len: Number of timesteps to encode from sequence (None = use full sequence from dataset)
        save_dir: Directory to save visualization outputs (if None, outputs are not saved)
        latent_eval_save_dir: Directory to save latent alignment diagnostics and metrics
        extract_coordinates: If True, returns extracted predicted coordinates as numpy array after processing all batches
        returns: If extract_coordinates=True, returns a numpy array of shape [V, T, N, 2] containing predicted coordinates for all videos in the evaluated set. Otherwise, returns None.
        evaluate_latent_alignment: If True, evaluates alignment of predicted latents with ground-truth positions (requires extract_coordinates=True)
        evaluate_noisy_gt_reference: If True, computes latent-alignment metrics on clean GT vs noisy GT baselines.
        noisy_gt_noise_mode: Noise mode for noisy GT reference evaluation. Currently supports 'relative_step'.
        noisy_gt_noise_alphas: List of noise alpha values used for noisy GT reference evaluation.
        noisy_gt_noise_seed: Random seed used for noisy GT generation.
        use_hungarian_for_correlation: If True, apply Hungarian object matching before latent-alignment correlation metrics
        reorder_method: Method used by reorder_predictions. Must be either
            'smallest_consecutive_distance' or 'hungarian'
        sg_window_length: Window length for Savitzky-Golay smoothing (must be odd, default: 15)
        sg_polyorder: Polynomial order for Savitzky-Golay smoothing (default: 2)
        extraction_method: Method for trajectory extraction ('bbox' or 'latent').
            - 'bbox': Extract from bounding box centers in pixel space (current method, includes smoothing)
            - 'latent': Extract directly from continuous latent positions z in kp_range (no smoothing)
        filtering_report_path: Path to JSON report where filtered trajectory IDs and counts
            are written for this mode.
        latent_position_variant: For latent extraction, choose emitted coordinates:
            'nominal', 'recentered', or 'both' (paired comparison mode).
        latent_recenter_source: Source for recentering offset, either:
            'patch_alpha' (decoded local patch alpha centroid) or
            'global_alpha' (global per-particle alpha centroid).
        latent_recenter_nms_source: Which latent coordinates drive NMS/filtering in latent mode:
            'nominal' or 'recentered'.
        latent_recenter_eps: Numerical epsilon for centroid computation stability.
        collect_nonlinear_probe_inputs: If True, also collect re-ordered latent
            position/scale/depth/transparency tensors for nonlinear probe fitting.
        return_nonlinear_probe_payload: If True, return the structured nonlinear
            probe payload instead of the standard coordinate array.
        respect_max_batches_for_extraction: If True, keep max_batches as an
            active limiter even when extract_coordinates=True. Intended for
            lightweight monitoring paths rather than full evaluation runs.
        prob_encoder_route: Optional probabilistic encoder source:
            'none' keeps the standard DDLP extraction path.
            'c1' and 'c1-dyn-attrs' load probabilistic encoder checkpoints and
            reconstruct the recentering-biased C1-family mean using
            prob_encoder_frozen_model.
        prob_encoder_frozen_model: Frozen base DDLP model required for
            prob_encoder_route in {'c1', 'c1-dyn-attrs'}.
        raw_physical_npz_root: Optional synchronized NPZ root containing raw
            physical trajectories. Use 'auto' to infer it from the DDLP HDF5
            root by replacing *_hdf5 with *_npz. Use 'none' to disable.

    returns:
        If extract_coordinates=True, returns a numpy array of shape [V, T, N, 2] containing predicted coordinates for all videos in the evaluated set. Otherwise, returns None.
    
    """
    # Validate extraction/reordering and latent recentering options
    if extraction_method not in ['bbox', 'latent']:
        raise ValueError(f"extraction_method must be 'bbox' or 'latent', got '{extraction_method}'")
    prob_encoder_route = _normalize_prob_encoder_route(prob_encoder_route)
    if prob_encoder_route != 'none' and extraction_method != 'latent':
        raise ValueError("prob_encoder_route requires extraction_method='latent'.")
    if prob_encoder_route in ['c1', 'c1-dyn-attrs'] and prob_encoder_frozen_model is None:
        raise ValueError(
            "prob_encoder_route in {'c1','c1-dyn-attrs'} requires prob_encoder_frozen_model."
        )
    if reorder_method not in ['smallest_consecutive_distance', 'hungarian']:
        raise ValueError(
            f"reorder_method must be either 'smallest_consecutive_distance' or 'hungarian', got '{reorder_method}'"
        )
    if evaluate_latent_alignment and reorder_method != 'smallest_consecutive_distance':
        raise ValueError(
            "evaluate_latent_alignment=True requires reorder_method='smallest_consecutive_distance'. "
            "Use --use_hungarian_for_correlation=1 for supervised global-ID matching in metrics."
        )
    if latent_position_variant not in ['nominal', 'recentered', 'both']:
        raise ValueError(
            f"latent_position_variant must be one of ['nominal','recentered','both'], got '{latent_position_variant}'"
        )
    if latent_recenter_source not in ['patch_alpha', 'global_alpha']:
        raise ValueError(
            f"latent_recenter_source must be one of ['patch_alpha','global_alpha'], got '{latent_recenter_source}'"
        )
    if latent_recenter_nms_source not in ['nominal', 'recentered']:
        raise ValueError(
            f"latent_recenter_nms_source must be one of ['nominal','recentered'], got '{latent_recenter_nms_source}'"
        )
    latent_recenter_eps = float(latent_recenter_eps)
    if latent_recenter_eps <= 0.0:
        raise ValueError(f"latent_recenter_eps must be > 0, got {latent_recenter_eps}")
    if return_nonlinear_probe_payload and not collect_nonlinear_probe_inputs:
        raise ValueError(
            "return_nonlinear_probe_payload=True requires collect_nonlinear_probe_inputs=True."
        )

    if extraction_method != 'latent':
        if latent_position_variant != 'nominal' or latent_recenter_nms_source != 'nominal' or latent_recenter_source != 'patch_alpha':
            print(
                "[WARN] Latent recentering options are ignored when extraction_method='bbox'. "
                "Using nominal bbox extraction behavior."
            )

    use_recentered_latent_output = (
        extraction_method == 'latent' and latent_position_variant in ['recentered', 'both']
    )
    compute_recentered_latents = (
        extraction_method == 'latent' and (use_recentered_latent_output or latent_recenter_nms_source == 'recentered')
    )

    # Print extraction method
    print(f"\nUsing extraction method: {extraction_method.upper()}")
    if extraction_method == 'latent':
        print("  → Extracting continuous positions from latent space (kp_range coordinates)")
        print("  → Converting GT from pixel [x,y] to DDLP latent [y,x] coordinates")
        print("  → No Savitzky-Golay smoothing applied (preserves model dynamics)")
        print(f"  → Latent position variant: {latent_position_variant}")
        print(f"  → Recenter source: {latent_recenter_source}")
        print(f"  → NMS source: {latent_recenter_nms_source}")
        print(f"  → Recenter epsilon: {latent_recenter_eps:.3e}")
    else:
        print("  → Extracting from bounding box centers (pixel space)")
        print(f"  → Savitzky-Golay smoothing: window={sg_window_length}, polyorder={sg_polyorder}")
    print(f"  → Reordering method: {reorder_method}")

    # assert method arguments compatibility
    if evaluate_latent_alignment and not extract_coordinates:
        raise ValueError("evaluate_latent_alignment=True requires extract_coordinates=True to access predicted coordinates for alignment evaluation.")
    if collect_nonlinear_probe_inputs and extraction_method != 'latent':
        raise ValueError("collect_nonlinear_probe_inputs=True is supported only for extraction_method='latent'.")
    if collect_nonlinear_probe_inputs and not extract_coordinates:
        raise ValueError(
            "collect_nonlinear_probe_inputs=True requires extract_coordinates=True so the same "
            "re-ordered latent trajectories can be reused for nonlinear probe fitting."
        )
    if returns and not extract_coordinates:
        raise ValueError("returns=True requires extract_coordinates=True to return predicted coordinates.")
    if returns and evaluate_latent_alignment:
        raise ValueError("returns=True is not compatible with evaluate_latent_alignment=True " \
                         "since the latter already returns metrics. Set returns=False when evaluate_latent_alignment=True.")
    if returns and return_nonlinear_probe_payload:
        raise ValueError(
            "returns=True is not compatible with return_nonlinear_probe_payload=True. "
            "Use the structured nonlinear payload return instead."
        )
    if returns and eval_seq_len is None:
        raise ValueError("returns=True requires eval_seq_len to be specified to ensure consistent output shape." \
                         "Set eval_seq_len to an integer value.")

    if latent_position_variant == 'both' and visualize_trajectories and extraction_method != 'latent':
        raise ValueError(
            "latent_position_variant='both' with visualize_trajectories=True is supported only for extraction_method='latent'."
        )
    if evaluate_latent_alignment and filtering_report_path is not None:
        print("[INFO] evaluate_latent_alignment=True: disabling filtering_report.json output.")
        filtering_report_path = None

    noisy_gt_reference_only = (
        evaluate_noisy_gt_reference
        and (not evaluate_latent_alignment)
        and (not visualize_trajectories)
        and (not returns)
    )

    # max_batches is a preview/debug limiter only for non-extraction runs
    if noisy_gt_reference_only:
        # GT-only noisy-reference mode does not use extraction; keep user-provided limit.
        effective_max_batches = max_batches
    elif extract_coordinates and respect_max_batches_for_extraction:
        effective_max_batches = max_batches
    else:
        effective_max_batches = None if extract_coordinates else max_batches
    if (
        extract_coordinates
        and max_batches is not None
        and not noisy_gt_reference_only
        and not respect_max_batches_for_extraction
    ):
        print(f"extract_coordinates=True: ignoring max_batches={max_batches} and processing all batches.")
    if noisy_gt_noise_alphas is None:
        noisy_gt_noise_alphas = [0.05, 0.1, 0.3, 0.5, 0.7]
    noisy_gt_noise_alphas = [float(a) for a in noisy_gt_noise_alphas]
    needs_alignment_targets = bool(
        evaluate_latent_alignment or evaluate_noisy_gt_reference or collect_nonlinear_probe_inputs
    )

    # Load useful config parameters 
    ds = config['ds']
    image_size = config['image_size']
    root = config['root']
        
    # Determine sequence length for evaluation
    if eval_seq_len is None:
        eval_seq_len = 60 if mode == 'train' else 360
    
    # Load video dataset for temporal tracking evaluation
    print(f"Using video dataset with temporal tracking (seq_len={eval_seq_len})")
    if not os.path.exists(root):
        root = root + "_small"  # Try alternative root if original doesn't exist
        print(f"Original root not found. Trying alternative root: {root}")
    if not os.path.exists(root):
        raise FileNotFoundError(f"Dataset root not found: {root}")
    print(f"Dataset path used for mode='{mode}': {root}")
    dataset = get_video_dataset(
        ds, root, seq_len=eval_seq_len, 
        mode=mode, image_size=image_size
    )
    dataloader = DataLoader(
        dataset, shuffle=False, batch_size=batch_size, 
        num_workers=4, drop_last=False
    )
    gt_conversion_kp_range = getattr(model, "kp_range", config.get("kp_range", (-1, 1)))

    raw_physical_npz_root_resolved = None
    raw_physical_cache = {}
    raw_physical_unavailable_reason = None
    collect_raw_physical_targets = bool(
        extraction_method == 'latent'
        and evaluate_latent_alignment
        and ds == 'twobody'
        and str(raw_physical_npz_root).strip().lower() not in {'none', 'off', 'false', '0', ''}
    )
    if collect_raw_physical_targets:
        raw_physical_npz_root_resolved = _resolve_twobody_physical_npz_root(
            root,
            raw_physical_npz_root=raw_physical_npz_root,
        )
        if raw_physical_npz_root_resolved is None:
            collect_raw_physical_targets = False
            raw_physical_unavailable_reason = (
                f"could not resolve synchronized raw-physics NPZ root from dataset root {root!r}"
            )
            print(f"[WARN] Raw physical latent-alignment metrics disabled: {raw_physical_unavailable_reason}")
        elif not _looks_like_twobody_physical_npz_root(raw_physical_npz_root_resolved):
            collect_raw_physical_targets = False
            raw_physical_unavailable_reason = (
                f"resolved path does not look like a two-body raw-physics NPZ root: "
                f"{raw_physical_npz_root_resolved}"
            )
            print(f"[WARN] Raw physical latent-alignment metrics disabled: {raw_physical_unavailable_reason}")
        else:
            print(f"Raw physical latent-alignment target root: {raw_physical_npz_root_resolved}")

    if noisy_gt_reference_only:
        print(
            "\nRunning noisy-GT reference in GT-only mode: "
            "skipping model forward pass and trajectory extraction."
        )
        if extract_coordinates:
            print("  Note: extract_coordinates is ignored in noisy-GT reference-only mode.")
        gt_only_batches = []
        pbar_gt = tqdm(dataloader, desc=f"Collecting GT ({mode}) for noisy reference")
        for batch_idx, batch in enumerate(pbar_gt):
            if effective_max_batches is not None and batch_idx >= effective_max_batches:
                print(f"Reached max_batches={effective_max_batches}, stopping GT-only collection.")
                break
            gt_batch = batch[1]
            if torch.is_tensor(gt_batch):
                gt_batch = gt_batch.detach().cpu().numpy()
            else:
                gt_batch = np.asarray(gt_batch)
            if extraction_method == 'latent':
                img_batch = batch[0]
                gt_batch = _convert_gt_xy_pixels_to_latent_yx(
                    gt_batch,
                    image_height=int(img_batch.shape[-2]),
                    image_width=int(img_batch.shape[-1]),
                    kp_range=gt_conversion_kp_range,
                )
            gt_only_batches.append(gt_batch)

        if len(gt_only_batches) == 0:
            raise ValueError(
                f"No ground-truth trajectories available for noisy GT reference (mode={mode})."
            )
        gt_coordinates_gt_only = np.concatenate(gt_only_batches, axis=0)
        print(
            f"Collected GT trajectories for noisy reference: "
            f"{gt_coordinates_gt_only.shape[0]} videos, shape={gt_coordinates_gt_only.shape}"
        )

        evaluate_noisy_gt_reference_metrics(
            gt_coordinates=gt_coordinates_gt_only,
            save_dir=latent_eval_save_dir if latent_eval_save_dir is not None else save_dir,
            mode=mode,
            extraction_method=extraction_method,
            noise_mode=noisy_gt_noise_mode,
            noise_alphas=noisy_gt_noise_alphas,
            noise_seed=noisy_gt_noise_seed,
        )
        _update_filtering_report_json(
            report_path=filtering_report_path,
            mode=mode,
            extraction_method=extraction_method,
            total_seen=gt_coordinates_gt_only.shape[0],
            kept=gt_coordinates_gt_only.shape[0],
            filtered_ids=[],
            filtered_details=[],
        )
        return

    # The remaining logic uses model outputs and extraction-specific settings.
    model.eval()
    kp_range = model.kp_range
    iou_thresh = config['iou_thresh']
    topk = int(min(config.get('topk', model.n_kp_enc), model.n_kp_enc))

    # latent evaluation output directories
    if evaluate_latent_alignment and latent_eval_save_dir is not None:
        os.makedirs(latent_eval_save_dir, exist_ok=True)
        latent_failure_dir = os.path.join(latent_eval_save_dir, "encoding_failures")
        os.makedirs(latent_failure_dir, exist_ok=True)
        latent_success_dir = os.path.join(latent_eval_save_dir, "encoding_non_failures")
        os.makedirs(latent_success_dir, exist_ok=True)
    else:
        latent_failure_dir = None
        latent_success_dir = None

    # create output directory for visualizations
    if visualize_trajectories and save_dir is not None:
        # Save visualizations under extraction-method-specific directories.
        root_out = os.path.join(save_dir, extraction_method)
        os.makedirs(root_out, exist_ok=True)
        if extraction_method == 'bbox':
            os.makedirs(os.path.join(root_out, "masks_overlay_videos"), exist_ok=True)
            os.makedirs(os.path.join(root_out, "masks_overlay_videos_alpha"), exist_ok=True)

    # initialize helpers
    all_coordinates = []  # To store predicted coordinates for all videos in the batch for trajectory plotting
    all_coordinates_smoothed = []  # To store smoothed predicted coordinates for all videos in the batch for trajectory plotting
    all_coordinates_recentered = []  # secondary coordinates for latent_position_variant='both'
    gt_coordinates = []  # To store ground-truth coordinates for latent alignment evaluation
    gt_raw_physical_coordinates = []  # Raw physical GT in [y,x] order for optional latent alignment metrics
    all_frame_mse_per_video = []  # frame-wise reconstruction MSE, list of [T]
    nonlinear_probe_nominal = {"p": [], "s": [], "d": [], "tau": [], "feat": []}
    nonlinear_probe_recentered = {"p": [], "s": [], "d": [], "tau": [], "feat": []}
    recenter_shift_l2_means = []  # mean ||p_can - p|| over kept trajectories
    recenter_valid_fractions = []  # fraction of valid alpha-mass centroids over kept trajectories
    max_failed_videos_to_plot = 10
    max_success_videos_to_plot = 10
    max_frames_per_failure_plot = 10
    num_failed_videos_plotted = 0
    num_success_videos_plotted = 0
    total_trajectories_seen = 0
    kept_trajectories = 0
    filtered_trajectory_ids = []
    filtered_trajectory_details = []

    def _register_filtered_trajectory(trajectory_id, batch_idx, video_idx, reason, gt_n_particles, detected_counts, first_faulty_t=None):
        filtered_trajectory_ids.append(int(trajectory_id))
        filtered_trajectory_details.append({
            "trajectory_id": int(trajectory_id),
            "batch_idx": int(batch_idx),
            "video_idx_in_batch": int(video_idx),
            "reason": str(reason),
            "gt_n_particles": int(gt_n_particles),
            "detected_particles_per_timestep": [int(v) for v in detected_counts],
            "first_faulty_timestep": None if first_faulty_t is None else int(first_faulty_t),
        })

    def _get_diagnostic_frame_indices(total_timesteps, anchor_t, window_size):
        half_window = window_size // 2
        start_t = max(0, anchor_t - half_window)
        end_t = min(total_timesteps, start_t + window_size)
        start_t = max(0, end_t - window_size)
        return list(range(start_t, end_t))

    def _save_latent_encoding_diagnostic(
        output_dir,
        diagnostic_tag,
        title,
        batch_idx,
        video_idx,
        anchor_t,
        frame_indices,
        nms_indices_per_t,
        x_batch,
        z_batch,
        z_features_batch,
        z_bg_batch,
        obj_on_batch,
        z_depth_batch,
        z_scale_batch,
    ):
        num_frames = len(frame_indices)
        fig, axes = plt.subplots(2, num_frames, figsize=(2.6 * num_frames, 5.0), squeeze=False)
        fig.suptitle(title, fontsize=12, fontweight='bold')

        for col, t in enumerate(frame_indices):
            idx_list = nms_indices_per_t[t]
            try:
                # decode_all expects the full fixed-width particle tensor [bs, n_kp_enc, ...].
                # Keep all particle slots and mask inactive ones via obj_on so selected subsets
                # can be reconstructed without breaking the decoder's internal reshape logic.
                obj_on_selected = torch.zeros_like(obj_on_batch[t])
                if len(idx_list) > 0:
                    idx_tensor = torch.as_tensor(idx_list, dtype=torch.long, device=x_batch.device)
                    obj_on_selected[idx_tensor] = obj_on_batch[t, idx_tensor]

                dec_out = model.decode_all(
                    z=z_batch[t].unsqueeze(0),
                    z_features=z_features_batch[t].unsqueeze(0),
                    z_bg=z_bg_batch[t].unsqueeze(0),
                    obj_on=obj_on_selected.unsqueeze(0),
                    z_depth=z_depth_batch[t].unsqueeze(0),
                    noisy=False,
                    z_scale=z_scale_batch[t].unsqueeze(0),
                )
                rec_frame = dec_out['rec'][0].detach().cpu().permute(1, 2, 0).numpy()
            except Exception as e:
                print(f"  [WARN] Could not decode diagnostic frame t={t} for video {video_idx}: {e}")
                rec_frame = np.zeros((x_batch.shape[2], x_batch.shape[3], x_batch.shape[1]), dtype=np.float32)

            gt_frame = x_batch[t].detach().cpu().permute(1, 2, 0).numpy()

            axes[0, col].imshow(np.clip(rec_frame, 0.0, 1.0))
            axes[0, col].set_title(f"t={t} | n={len(idx_list)}", fontsize=9)
            axes[0, col].axis('off')

            axes[1, col].imshow(np.clip(gt_frame, 0.0, 1.0))
            axes[1, col].axis('off')

        axes[0, 0].set_ylabel("Reconstruction", fontsize=10)
        axes[1, 0].set_ylabel("Ground Truth", fontsize=10)
        plt.tight_layout(rect=[0, 0, 1, 0.93])

        diagnostic_path = os.path.join(
            output_dir,
            f"latent_encoding_{diagnostic_tag}_{mode}_batch{batch_idx:03d}_vid{video_idx:03d}_t{anchor_t:03d}.png"
        )
        plt.savefig(diagnostic_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  [Latent Diagnostic] Saved: {diagnostic_path}")

    def _save_latent_training_style_block(
        output_dir,
        diagnostic_tag,
        batch_idx,
        video_idx,
        anchor_t,
        frame_indices,
        x_batch,
        mu_plot_batch,
        kp_prior_batch,
        rec_batch,
        mu_tot_batch,
        logvar_tot_batch,
        obj_on_batch,
        mu_scale_batch,
        alpha_masks_batch,
        dec_objects_batch,
        bg_batch,
    ):
        # Match train_ddlp.py panel grid style (saved via torchvision.utils.save_image).
        block_max_imgs = min(8, len(frame_indices))
        if block_max_imgs < 1:
            return

        try:
            sel_t = frame_indices[:block_max_imgs]
            idx_tensor = torch.as_tensor(sel_t, dtype=torch.long, device=x_batch.device)

            x_sel = x_batch.index_select(0, idx_tensor)
            mu_plot_sel = mu_plot_batch.index_select(0, idx_tensor)
            kp_prior_sel = kp_prior_batch.index_select(0, idx_tensor)
            rec_sel = rec_batch.index_select(0, idx_tensor)
            mu_tot_sel = mu_tot_batch.index_select(0, idx_tensor)
            logvar_tot_sel = logvar_tot_batch.index_select(0, idx_tensor)
            obj_on_sel = obj_on_batch.index_select(0, idx_tensor)
            mu_scale_sel = mu_scale_batch.index_select(0, idx_tensor)
            alpha_masks_sel = alpha_masks_batch.index_select(0, idx_tensor)
            dec_objects_sel = dec_objects_batch.index_select(0, idx_tensor)
            bg_sel = bg_batch.index_select(0, idx_tensor)

            img_with_kp = plot_keypoints_on_image_batch(
                mu_plot_sel,
                x_sel,
                radius=3,
                thickness=1,
                max_imgs=block_max_imgs,
                kp_range=kp_range,
            )
            img_with_kp_p = plot_keypoints_on_image_batch(
                kp_prior_sel,
                x_sel,
                radius=3,
                thickness=1,
                max_imgs=block_max_imgs,
                kp_range=kp_range,
            )

            with torch.no_grad():
                logvar_sum = logvar_tot_sel.sum(-1) * obj_on_sel
                bb_scores = -1 * logvar_sum
                k_top = max(1, min(topk, int(mu_tot_sel.shape[1])))
                logvar_topk = torch.topk(logvar_sum, k=k_top, dim=-1, largest=False)
                topk_indices = logvar_topk[1]
                batch_indices = torch.arange(mu_tot_sel.shape[0], device=mu_tot_sel.device).view(-1, 1)
                topk_kp = mu_tot_sel[batch_indices, topk_indices]

            img_with_masks_nms, _ = plot_bb_on_image_batch_from_z_scale_nms(
                mu_plot_sel,
                mu_scale_sel,
                x_sel,
                scores=bb_scores,
                iou_thresh=iou_thresh,
                thickness=1,
                max_imgs=block_max_imgs,
                hard_thresh=None,
            )
            alpha_masks_binary = torch.where(alpha_masks_sel < 0.05, 0.0, 1.0)
            img_with_masks_alpha_nms, _ = plot_bb_on_image_batch_from_masks_nms(
                alpha_masks_binary,
                x_sel,
                scores=bb_scores,
                iou_thresh=iou_thresh,
                thickness=1,
                max_imgs=block_max_imgs,
                hard_thresh=None,
            )
            img_with_kp_topk = plot_keypoints_on_image_batch(
                topk_kp.clamp(min=kp_range[0], max=kp_range[1]),
                x_sel,
                radius=3,
                thickness=1,
                max_imgs=block_max_imgs,
                kp_range=kp_range,
            )

            block_path = os.path.join(
                output_dir,
                f"latent_encoding_{diagnostic_tag}_{mode}_batch{batch_idx:03d}_vid{video_idx:03d}_t{anchor_t:03d}_train_style.png",
            )
            vutils.save_image(
                torch.cat(
                    [
                        x_sel[:block_max_imgs, -3:],
                        img_with_kp[:block_max_imgs, -3:].to(x_sel.device),
                        rec_sel[:block_max_imgs, -3:],
                        img_with_kp_p[:block_max_imgs, -3:].to(x_sel.device),
                        img_with_kp_topk[:block_max_imgs, -3:].to(x_sel.device),
                        dec_objects_sel[:block_max_imgs, -3:],
                        img_with_masks_nms[:block_max_imgs, -3:].to(x_sel.device),
                        img_with_masks_alpha_nms[:block_max_imgs, -3:].to(x_sel.device),
                        bg_sel[:block_max_imgs, -3:],
                    ],
                    dim=0,
                ).data.cpu(),
                block_path,
                nrow=block_max_imgs,
                pad_value=1,
            )
            print(f"  [Latent Diagnostic] Saved train-style block: {block_path}")
        except Exception as e:
            print(
                f"  [WARN] Could not save train-style diagnostic block for video {video_idx} "
                f"(batch={batch_idx}, t={anchor_t}): {e}"
            )
    
    # iterate over all batches in the dataloader
    num_batches_processed = 0
    pbar = tqdm(dataloader, desc=f"Evaluating {mode} set")
    for batch_idx, batch in enumerate(pbar):

        # stop after max_batches only in preview mode (extract_coordinates=False)
        if effective_max_batches is not None and batch_idx >= effective_max_batches:
            print(f"Reached max_batches={effective_max_batches}, stopping evaluation.")
            break
        num_batches_processed += 1
        
        # Unpack batch
        x = batch[0].to(device)  # Images: [B, T, C, H, W]
        
        # Forward pass with temporal tracking
        with torch.no_grad():
            model_output = _encode_sequence_model(model, x, config, label="prob_encoder" if prob_encoder_route != 'none' else "ddlp")
            frozen_model_output = None
            if prob_encoder_route in {'c1', 'c1-dyn-attrs'}:
                frozen_model_output = _encode_sequence_model(
                    prob_encoder_frozen_model,
                    x,
                    config,
                    label="frozen_ddlp",
                )
        
            # Now model_output has consistent shape [B, T, n_kp, ...] for both paths
            T = model_output['z_base'].shape[1]  # Get T from reshaped output
            
            # output for logging and plotting
            z_base = model_output['z_base']  # [bs, T, n_kp, 2]
            z = model_output['z']  # [bs, T, n_kp, 2]
            mu_offset = model_output['mu_offset']  # [bs, T, n_kp, 2]
            logvar_offset = model_output['logvar_offset']  # [bs, T, n_kp, 2]
            mu_depth = model_output['mu_depth']  # [bs, T, n_kp, 1]
            z_depth = model_output['z_depth']  # [bs, T, n_kp, 1]
            mu_scale = model_output['mu_scale']  # [bs, T, n_kp, 2]
            z_scale = model_output['z_scale']  # [bs, T, n_kp, 2]
            kp_p = model_output['kp_p']  # [bs, T, n_kp, 2]
            obj_on = model_output['obj_on']  # [bs, T, n_kp]
            z_bg = model_output['z_bg']  # [bs, T, bg_dim]
            alpha_masks = model_output['alpha_masks']  # [bs, T, n_kp, 1, h, w]
            dec_objects_original = model_output['dec_objects_original']  # [bs, T, n_kp, 4, h_patch, w_patch]
            dec_objects_batch = model_output['dec_objects']  # [bs, T, C, H, W]
            bg_batch = model_output['bg']  # [bs, T, C, H, W]
            rec_batch = model_output['rec']  # [bs, T, C, H, W]

            if prob_encoder_route == 'none':
                mu_tot_nominal = z_base + mu_offset
                prob_encoder_valid_mask = None
            else:
                mu_tot_nominal, prob_encoder_valid_mask = _build_prob_encoder_route_mean(
                    route=prob_encoder_route,
                    train_out=model_output,
                    frozen_out=frozen_model_output,
                    latent_recenter_source=latent_recenter_source,
                    latent_recenter_eps=latent_recenter_eps,
                    kp_range=kp_range,
                )
            mu_tot_recentered = None
            recenter_valid_mask = None
            if compute_recentered_latents:
                if latent_recenter_source == 'patch_alpha':
                    mu_tot_recentered, recenter_valid_mask = _recenter_positions_patch_alpha(
                        mu_tot=mu_tot_nominal,
                        mu_scale=mu_scale,
                        dec_objects_original=dec_objects_original,
                        eps=latent_recenter_eps,
                    )
                else:
                    mu_tot_recentered, recenter_valid_mask = _recenter_positions_global_alpha(
                        mu_tot=mu_tot_nominal,
                        alpha_masks=alpha_masks,
                        eps=latent_recenter_eps,
                    )
                mu_tot_recentered = mu_tot_recentered.clamp(min=kp_range[0], max=kp_range[1])
            if recenter_valid_mask is None and prob_encoder_valid_mask is not None:
                recenter_valid_mask = prob_encoder_valid_mask

            mu_tot = mu_tot_nominal
            logvar_tot = logvar_offset
            mu_plot_nominal = mu_tot_nominal.clamp(min=kp_range[0], max=kp_range[1])
            mu_plot_recentered = None if mu_tot_recentered is None else mu_tot_recentered.clamp(min=kp_range[0], max=kp_range[1])
            logvar_sum = logvar_tot.sum(-1) * obj_on  # [bs, n_kp]
            bb_scores = -1 * logvar_sum
            hard_threshold = None

            if extraction_method == 'latent' and latent_recenter_nms_source == 'recentered':
                if mu_plot_recentered is None:
                    raise RuntimeError(
                        "Requested latent_recenter_nms_source='recentered' but recentered positions were not computed."
                    )
                kp_batch = mu_plot_recentered
            else:
                kp_batch = mu_plot_nominal
            scale_batch = mu_scale

            # reconstruction MSE per frame. We only append entries for kept trajectories.
            frame_mse_batch = None
            if evaluate_latent_alignment:
                frame_mse_batch = ((rec_batch - x) ** 2).mean(dim=(2, 3, 4)).detach().cpu().numpy()  # [B, T]

            # initialize helper to store coordinates for this batch 
            # for later trajectory plotting after processing all batches
            batch_coordinates = []
            batch_coordinates_recentered = []
            batch_gt_coordinates = []
            batch_gt_raw_physical_coordinates = []
            batch_latent_scale_raw = []
            batch_latent_depth_raw = []
            batch_latent_tau_raw = []
            batch_latent_feat_raw = []
            
            # ========================================================================
            # EXTRACTION METHOD BRANCHING: bbox vs latent
            # ========================================================================
            
            if extraction_method == 'latent':
                # =====================================================================
                # LATENT EXTRACTION: Extract continuous positions from latent space
                # =====================================================================
                print(f"  [Latent extraction] Processing {x.shape[0]} videos...")
                
                for video_idx in range(x.shape[0]):
                    trajectory_id = total_trajectories_seen
                    total_trajectories_seen += 1
                    gt_n_particles = int(batch[1][video_idx].shape[1])

                    # Run NMS at each timestep to detect failure time (count < GT particles)
                    nms_indices_per_t = []
                    nms_counts_per_t = []
                    for t in range(T):
                        _, nms_indices_t = plot_bb_on_image_batch_from_z_scale_nms(
                            kp_batch[video_idx:video_idx+1, t],
                            scale_batch[video_idx:video_idx+1, t],
                            x[video_idx:video_idx+1, t],
                            scores=bb_scores[video_idx:video_idx+1, t],
                            iou_thresh=iou_thresh,
                            hard_thresh=hard_threshold
                        )
                        if len(nms_indices_t) == 0:
                            idx_list = []
                        else:
                            idx_raw = nms_indices_t[0]
                            if torch.is_tensor(idx_raw):
                                idx_list = idx_raw.detach().cpu().tolist()
                            else:
                                idx_list = list(idx_raw)
                        nms_indices_per_t.append(idx_list)
                        nms_counts_per_t.append(len(idx_list))
                    failure_times = [t for t, c in enumerate(nms_counts_per_t) if c < gt_n_particles]
                    failure_time = failure_times[0] if len(failure_times) > 0 else None

                    # Diagnostic visualizations for failure and non-failure cases (latent alignment mode only)
                    if (
                        evaluate_latent_alignment
                        and latent_failure_dir is not None
                        and failure_time is not None
                        and num_failed_videos_plotted < max_failed_videos_to_plot
                    ):
                        frame_indices = _get_diagnostic_frame_indices(T, failure_time, max_frames_per_failure_plot)
                        _save_latent_encoding_diagnostic(
                            output_dir=latent_failure_dir,
                            diagnostic_tag="failure",
                            title=(
                                f"Latent Encoding Failure | video={video_idx}, first_failure_t={failure_time}, "
                                f"detected<{gt_n_particles}"
                            ),
                            batch_idx=batch_idx,
                            video_idx=video_idx,
                            anchor_t=failure_time,
                            frame_indices=frame_indices,
                            nms_indices_per_t=nms_indices_per_t,
                            x_batch=x[video_idx],
                            z_batch=z[video_idx],
                            z_features_batch=model_output['z_features'][video_idx],
                            z_bg_batch=z_bg[video_idx],
                            obj_on_batch=obj_on[video_idx],
                            z_depth_batch=z_depth[video_idx],
                            z_scale_batch=z_scale[video_idx],
                        )
                        _save_latent_training_style_block(
                            output_dir=latent_failure_dir,
                            diagnostic_tag="failure",
                            batch_idx=batch_idx,
                            video_idx=video_idx,
                            anchor_t=failure_time,
                            frame_indices=frame_indices,
                            x_batch=x[video_idx],
                            mu_plot_batch=kp_batch[video_idx],
                            kp_prior_batch=kp_p[video_idx],
                            rec_batch=rec_batch[video_idx],
                            mu_tot_batch=mu_tot[video_idx],
                            logvar_tot_batch=logvar_tot[video_idx],
                            obj_on_batch=obj_on[video_idx],
                            mu_scale_batch=mu_scale[video_idx],
                            alpha_masks_batch=alpha_masks[video_idx],
                            dec_objects_batch=dec_objects_batch[video_idx],
                            bg_batch=bg_batch[video_idx],
                        )
                        num_failed_videos_plotted += 1
                    elif (
                        evaluate_latent_alignment
                        and latent_success_dir is not None
                        and failure_time is None
                        and num_success_videos_plotted < max_success_videos_to_plot
                    ):
                        success_anchor_t = T // 2
                        frame_indices = _get_diagnostic_frame_indices(T, success_anchor_t, max_frames_per_failure_plot)
                        _save_latent_encoding_diagnostic(
                            output_dir=latent_success_dir,
                            diagnostic_tag="non_failure",
                            title=(
                                f"Latent Encoding Non-Failure | video={video_idx}, center_t={success_anchor_t}, "
                                f"detected>={gt_n_particles} for all t"
                            ),
                            batch_idx=batch_idx,
                            video_idx=video_idx,
                            anchor_t=success_anchor_t,
                            frame_indices=frame_indices,
                            nms_indices_per_t=nms_indices_per_t,
                            x_batch=x[video_idx],
                            z_batch=z[video_idx],
                            z_features_batch=model_output['z_features'][video_idx],
                            z_bg_batch=z_bg[video_idx],
                            obj_on_batch=obj_on[video_idx],
                            z_depth_batch=z_depth[video_idx],
                            z_scale_batch=z_scale[video_idx],
                        )
                        num_success_videos_plotted += 1

                    # Strict filtering across all timesteps:
                    # if ANY timestep has fewer detected particles than GT, drop trajectory.
                    if failure_time is not None:
                        print(
                            f"  [FILTER] Video {video_idx} (trajectory_id={trajectory_id}) skipped: "
                            f"detected<{gt_n_particles} at t={failure_time}."
                        )
                        _register_filtered_trajectory(
                            trajectory_id=trajectory_id,
                            batch_idx=batch_idx,
                            video_idx=video_idx,
                            reason='insufficient_particles_any_timestep',
                            gt_n_particles=gt_n_particles,
                            detected_counts=nms_counts_per_t,
                            first_faulty_t=failure_time,
                        )
                        continue

                    filtered_indices = nms_indices_per_t[0]  # [K] selected particle indices from first frame
                    if len(filtered_indices) != gt_n_particles:
                        print(
                            f"  [FILTER] Video {video_idx} (trajectory_id={trajectory_id}) skipped: "
                            f"t=0 selected {len(filtered_indices)} particles, GT has {gt_n_particles}."
                        )
                        _register_filtered_trajectory(
                            trajectory_id=trajectory_id,
                            batch_idx=batch_idx,
                            video_idx=video_idx,
                            reason='t0_count_mismatch_gt',
                            gt_n_particles=gt_n_particles,
                            detected_counts=nms_counts_per_t,
                            first_faulty_t=0,
                        )
                        continue
                    
                    # Extract continuous latent positions for filtered particles across all timesteps
                    # Shape: [T, K, 2] in kp_range
                    latent_positions_nominal = mu_tot_nominal[video_idx, :, filtered_indices, :].cpu()
                    latent_positions_recentered = None
                    if mu_tot_recentered is not None:
                        latent_positions_recentered = mu_tot_recentered[video_idx, :, filtered_indices, :].cpu()

                    if latent_position_variant == 'recentered':
                        if latent_positions_recentered is None:
                            raise RuntimeError(
                                "latent_position_variant='recentered' but recentered positions are unavailable."
                            )
                        latent_positions = latent_positions_recentered
                    else:
                        # 'nominal' or 'both' use nominal as the primary extraction payload.
                        latent_positions = latent_positions_nominal

                    # Check if number of filtered particles matches ground truth
                    if latent_positions.shape[1] != batch[1][video_idx].shape[1]:
                        print(f"  [WARN] Video {video_idx}: Filtered {latent_positions.shape[1]} particles, "
                              f"but ground truth has {batch[1][video_idx].shape[1]} objects. Skipping...")
                        _register_filtered_trajectory(
                            trajectory_id=trajectory_id,
                            batch_idx=batch_idx,
                            video_idx=video_idx,
                            reason='post_extract_count_mismatch_gt',
                            gt_n_particles=gt_n_particles,
                            detected_counts=nms_counts_per_t,
                            first_faulty_t=None,
                        )
                        continue

                    if collect_nonlinear_probe_inputs:
                        latent_scale = mu_scale[video_idx, :, filtered_indices, :].cpu().numpy()
                        latent_depth = z_depth[video_idx, :, filtered_indices, :].cpu().numpy()
                        latent_tau = obj_on[video_idx, :, filtered_indices].unsqueeze(-1).cpu().numpy()
                        latent_feat = model_output['z_features'][video_idx, :, filtered_indices, :].cpu().numpy()
                        batch_latent_scale_raw.append(latent_scale)
                        batch_latent_depth_raw.append(latent_depth)
                        batch_latent_tau_raw.append(latent_tau)
                        batch_latent_feat_raw.append(latent_feat)

                    batch_coordinates.append(latent_positions)  # [T, K, 2] in kp_range
                    if latent_position_variant == 'both':
                        if latent_positions_recentered is None:
                            raise RuntimeError(
                                "latent_position_variant='both' requires recentered positions, but they are unavailable."
                            )
                        batch_coordinates_recentered.append(latent_positions_recentered)

                    if mu_tot_recentered is not None:
                        shift_sel = torch.norm(
                            (mu_tot_recentered - mu_tot_nominal)[video_idx, :, filtered_indices, :], dim=-1
                        )
                        recenter_shift_l2_means.append(float(shift_sel.mean().item()))
                        if recenter_valid_mask is not None:
                            valid_sel = recenter_valid_mask[video_idx, :, filtered_indices].float()
                            recenter_valid_fractions.append(float(valid_sel.mean().item()))

                    gt_xy_pixels = batch[1][video_idx].cpu().numpy()  # [T, n_objects, 2] in dataset [x,y] pixels
                    gt_latent_yx = _convert_gt_xy_pixels_to_latent_yx(
                        gt_xy_pixels,
                        image_height=int(x.shape[-2]),
                        image_width=int(x.shape[-1]),
                        kp_range=kp_range,
                    )
                    batch_gt_coordinates.append(gt_latent_yx)  # [T, n_objects, 2] in DDLP latent [y,x]

                    if collect_raw_physical_targets:
                        dataset_item_idx = batch_idx * batch_size + video_idx
                        gt_raw_physical_yx, raw_reason = _load_twobody_raw_physical_yx_sequence(
                            raw_physical_npz_root_resolved,
                            dataset,
                            dataset_item_idx,
                            expected_T=gt_latent_yx.shape[0],
                            cache=raw_physical_cache,
                        )
                        if gt_raw_physical_yx is None:
                            collect_raw_physical_targets = False
                            gt_raw_physical_coordinates = []
                            batch_gt_raw_physical_coordinates = []
                            raw_physical_unavailable_reason = raw_reason
                            print(
                                "[WARN] Raw physical latent-alignment metrics disabled: "
                                f"{raw_physical_unavailable_reason}"
                            )
                        elif gt_raw_physical_yx.shape != gt_latent_yx.shape:
                            collect_raw_physical_targets = False
                            gt_raw_physical_coordinates = []
                            batch_gt_raw_physical_coordinates = []
                            raw_physical_unavailable_reason = (
                                f"raw physical shape {gt_raw_physical_yx.shape} does not match "
                                f"image-space GT shape {gt_latent_yx.shape}"
                            )
                            print(
                                "[WARN] Raw physical latent-alignment metrics disabled: "
                                f"{raw_physical_unavailable_reason}"
                            )
                        else:
                            batch_gt_raw_physical_coordinates.append(gt_raw_physical_yx)

                    kept_trajectories += 1
                    if evaluate_latent_alignment and frame_mse_batch is not None:
                        all_frame_mse_per_video.append(frame_mse_batch[video_idx])

                    print(f"  [Latent] Video {video_idx}: Extracted {latent_positions.shape[1]} particles, "
                          f"coords in kp_range [{kp_range[0]}, {kp_range[1]}]")
            
            else:  # extraction_method == 'bbox'
                # =====================================================================
                # BBOX EXTRACTION: Extract from bounding box centers (current method)
                # =====================================================================
                
                # plot bbs and collect coordinates for each video in the batch
                for video_idx in range(x.shape[0]):
                    trajectory_id = total_trajectories_seen
                    total_trajectories_seen += 1
                    gt_n_particles = int(batch[1][video_idx].shape[1])
                    
                    print(f"Processing video {video_idx} ...")
                    
                    # initialize helpers to store frames with bounding boxes 
                    # and alpha masks for this video for visualization
                    frames = []
                    frames_alpha = []
                    coordinates = []
                    detected_counts_per_t = []
                    first_faulty_t = None

                    # iterate over timesteps for this video and plot bounding boxes with NMS and alpha masks with NMS, 
                    # and collect coordinates for this video for later trajectory plotting after processing all batches
                    for t in range(T):

                        img_with_masks_nms, _ = plot_bb_on_image_batch_from_z_scale_nms(
                            kp_batch[video_idx, t:t+1], 
                            scale_batch[video_idx, t:t+1], 
                            x[video_idx, t:t+1],
                            scores=bb_scores[video_idx, t:t+1],
                            iou_thresh=iou_thresh,
                            thickness=1,max_imgs=1,
                            hard_thresh=hard_threshold
                        )
                        frame_img = img_with_masks_nms[0].permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
                        frame_img = (frame_img * 255).astype(np.uint8)
                        frames.append(frame_img)

                        # Threshold alpha masks for THIS timestep only (don't modify original)
                        alpha_masks_t = torch.where(alpha_masks[video_idx, t:t+1] < 0.05, 0.0, 1.0)
                        img_with_masks_alpha_nms, _, centres = plot_bb_on_image_batch_from_masks_nms(
                            alpha_masks_t, 
                            x[video_idx, t:t+1], 
                            scores=bb_scores[video_idx, t:t+1],
                            iou_thresh=iou_thresh, thickness=1,
                            max_imgs=1,
                            hard_thresh=hard_threshold, 
                            return_centres=True, 
                            debug_info=False
                        )
                        frame_alpha = img_with_masks_alpha_nms[0].permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
                        frame_alpha = (frame_alpha * 255).astype(np.uint8)
                        frames_alpha.append(frame_alpha)

                        # IMPORTANT: centres from get_bb_from_masks uses (y,x) convention
                        # Swap to (x,y) convention for GHNN compatibility
                        # centres shape: [1, N, 2] where [:, :, 0] is y and [:, :, 1] is x
                        centres_xy = centres[:, :, [1, 0]]  # Swap to (x, y)
                        n_detected = int(centres_xy.shape[1])
                        detected_counts_per_t.append(n_detected)
                        if first_faulty_t is None and n_detected < gt_n_particles:
                            first_faulty_t = t
                        coordinates.append(centres_xy)
                    
                    # Strict filtering across all timesteps:
                    # if ANY timestep has fewer detected particles than GT, drop trajectory.
                    if first_faulty_t is not None:
                        print(
                            f"[FILTER] Video {video_idx} (trajectory_id={trajectory_id}) skipped: "
                            f"detected<{gt_n_particles} at t={first_faulty_t}."
                        )
                        _register_filtered_trajectory(
                            trajectory_id=trajectory_id,
                            batch_idx=batch_idx,
                            video_idx=video_idx,
                            reason='insufficient_particles_any_timestep',
                            gt_n_particles=gt_n_particles,
                            detected_counts=detected_counts_per_t,
                            first_faulty_t=first_faulty_t,
                        )
                        continue

                    # Ensure fixed dimensionality over time for downstream stacking.
                    if any(c != detected_counts_per_t[0] for c in detected_counts_per_t):
                        print(
                            f"[FILTER] Video {video_idx} (trajectory_id={trajectory_id}) skipped: "
                            "detected particle count varies across timesteps."
                        )
                        _register_filtered_trajectory(
                            trajectory_id=trajectory_id,
                            batch_idx=batch_idx,
                            video_idx=video_idx,
                            reason='inconsistent_detected_count_across_timesteps',
                            gt_n_particles=gt_n_particles,
                            detected_counts=detected_counts_per_t,
                            first_faulty_t=None,
                        )
                        continue

                    # After processing all timesteps for this video, concatenate coordinates 
                    # and store for trajectory plotting and latent alignment evaluation and
                    # check whehter the extracted coordinates have the same number of keypoints 
                    # as the ground-truth positions for this video
                    coordinates = torch.cat(coordinates, dim=0)  # [T, n_kp, 2]
                    if coordinates.shape[1] != batch[1][video_idx].shape[1]:
                        print(f"[WARN] Number of predicted keypoints ({coordinates.shape[1]}) does not match number of ground-truth objects ({batch[1][video_idx].shape[1]}). This may affect trajectory visualization and latent alignment evaluation.")
                        print(f"Video {video_idx} will NOT be included in trajectory visualization or latent alignment evaluation for this batch.")
                        _register_filtered_trajectory(
                            trajectory_id=trajectory_id,
                            batch_idx=batch_idx,
                            video_idx=video_idx,
                            reason='count_mismatch_gt_after_concat',
                            gt_n_particles=gt_n_particles,
                            detected_counts=detected_counts_per_t,
                            first_faulty_t=None,
                        )
                        continue  # Skip this video for trajectory visualization and latent alignment evaluation
                    batch_coordinates.append(coordinates) 
                    batch_gt_coordinates.append(batch[1][video_idx].cpu().numpy())  # [T, n_objects, 2]
                    kept_trajectories += 1
                    if evaluate_latent_alignment and frame_mse_batch is not None:
                        all_frame_mse_per_video.append(frame_mse_batch[video_idx])
                    
                    # plotting bounding boxes on video frames for each video in the batch
                    if visualize_trajectories:
                        save_path = os.path.join(
                            root_out,
                            "masks_overlay_videos",
                            f'mask_overlay_{mode}_batch{batch_idx:03d}_vid{video_idx:03d}.gif'
                        )
                        imageio.mimsave(save_path, frames, duration=100, loop=0)

                        save_path_alpha = os.path.join(
                            root_out, 
                            "masks_overlay_videos_alpha",
                            f'mask_overlay_alpha_{mode}_batch{batch_idx:03d}_vid{video_idx:03d}.gif'
                        )
                        imageio.mimsave(save_path_alpha, frames_alpha, duration=100, loop=0)

            # first, define batch of coordinates as list of np arrays for re-ordering function, 
            # and also stack into tensor for re-ordering function.
            if len(batch_coordinates) == 0:
                print(
                    f"[WARN] No valid trajectories extracted in batch {batch_idx}. "
                    "Skipping batch."
                )
                continue
            batch_coordinates_np = [
                batch_coordinates[i].cpu().numpy() if isinstance(batch_coordinates[i], torch.Tensor) 
                else batch_coordinates[i] for i in range(len(batch_coordinates))
            ] # list np array

            # then reorder the batch of coordinates using the defined re-ordering function (e.g., smallest_consecutive_distance)
            reordered_batch_coordinates_result = reorder_predictions(
                torch.stack([torch.from_numpy(c) if isinstance(c, np.ndarray) else c 
                            for c in batch_coordinates], dim=0), 
                method=reorder_method, 
                print_debug=False,
                return_permutations=collect_nonlinear_probe_inputs,
            )
            if collect_nonlinear_probe_inputs:
                reordered_batch_coordinates, reordered_batch_permutations = reordered_batch_coordinates_result
            else:
                reordered_batch_coordinates = reordered_batch_coordinates_result
                reordered_batch_permutations = None

            reordered_batch_coordinates_recentered = None
            reordered_batch_permutations_recentered = None
            if extraction_method == 'latent' and latent_position_variant == 'both':
                reordered_batch_coordinates_recentered_result = reorder_predictions(
                    torch.stack([torch.from_numpy(c) if isinstance(c, np.ndarray) else c
                                for c in batch_coordinates_recentered], dim=0),
                    method=reorder_method,
                    print_debug=False,
                    return_permutations=collect_nonlinear_probe_inputs,
                )
                if collect_nonlinear_probe_inputs:
                    reordered_batch_coordinates_recentered, reordered_batch_permutations_recentered = reordered_batch_coordinates_recentered_result
                else:
                    reordered_batch_coordinates_recentered = reordered_batch_coordinates_recentered_result
            
            # ========================================================================
            # SMOOTHING: Only apply for bbox extraction
            # ========================================================================
            if extraction_method == 'bbox':
                # Apply Savitzky-Golay filter for bbox extraction
                smoothed_batch_coordinates = savgol_filter(
                    np.array(reordered_batch_coordinates), 
                    window_length=sg_window_length, 
                    polyorder=sg_polyorder, 
                    axis=1
                )
                print(f"  [Bbox] Applied Savitzky-Golay smoothing (window={sg_window_length}, poly={sg_polyorder})")
            else:  
                # No smoothing for latent extraction - preserve model dynamics
                smoothed_batch_coordinates = np.array(reordered_batch_coordinates)
                print(f"  [Latent] No smoothing applied - preserving continuous latent dynamics")

            # store the reordered coordinates and smoothed reordered coordinates for this batch for later trajectory plotting after processing all batches
            if extract_coordinates:
                # reordered_batch_coordinates is a list of [T, N, 2] numpy arrays, stack to [B, T, N, 2]
                batch_reordered_coordinates = np.stack(reordered_batch_coordinates, axis=0)
                all_coordinates.append(batch_reordered_coordinates)  # [B, T, n_kp, 2]
                # Keep a separate "smoothed" container only for bbox extraction.
                # For latent extraction, smoothed == reordered by design.
                if extraction_method == 'bbox':
                    all_coordinates_smoothed.append(smoothed_batch_coordinates)  # [B, T, n_kp, 2]
                elif extraction_method == 'latent' and latent_position_variant == 'both':
                    batch_reordered_coordinates_recentered = np.stack(reordered_batch_coordinates_recentered, axis=0)
                    all_coordinates_recentered.append(batch_reordered_coordinates_recentered)

            if collect_nonlinear_probe_inputs:
                if reordered_batch_permutations is None:
                    raise RuntimeError(
                        "collect_nonlinear_probe_inputs=True expected reordered_batch_permutations, found None."
                    )
                nonlinear_probe_nominal["p"].append(np.stack(reordered_batch_coordinates, axis=0))
                nonlinear_probe_nominal["s"].append(
                    np.stack(
                        [
                            _apply_temporal_permutation(batch_latent_scale_raw[i], reordered_batch_permutations[i])
                            for i in range(len(batch_latent_scale_raw))
                        ],
                        axis=0,
                    )
                )
                nonlinear_probe_nominal["d"].append(
                    np.stack(
                        [
                            _apply_temporal_permutation(batch_latent_depth_raw[i], reordered_batch_permutations[i])
                            for i in range(len(batch_latent_depth_raw))
                        ],
                        axis=0,
                    )
                )
                nonlinear_probe_nominal["tau"].append(
                    np.stack(
                        [
                            _apply_temporal_permutation(batch_latent_tau_raw[i], reordered_batch_permutations[i])
                            for i in range(len(batch_latent_tau_raw))
                        ],
                        axis=0,
                    )
                )
                nonlinear_probe_nominal["feat"].append(
                    np.stack(
                        [
                            _apply_temporal_permutation(batch_latent_feat_raw[i], reordered_batch_permutations[i])
                            for i in range(len(batch_latent_feat_raw))
                        ],
                        axis=0,
                    )
                )

                if extraction_method == 'latent' and latent_position_variant == 'both':
                    if reordered_batch_coordinates_recentered is None or reordered_batch_permutations_recentered is None:
                        raise RuntimeError(
                            "Expected recentered trajectories and permutations for nonlinear probe collection."
                        )
                    nonlinear_probe_recentered["p"].append(np.stack(reordered_batch_coordinates_recentered, axis=0))
                    nonlinear_probe_recentered["s"].append(
                        np.stack(
                            [
                                _apply_temporal_permutation(batch_latent_scale_raw[i], reordered_batch_permutations_recentered[i])
                                for i in range(len(batch_latent_scale_raw))
                            ],
                            axis=0,
                        )
                    )
                    nonlinear_probe_recentered["d"].append(
                        np.stack(
                            [
                                _apply_temporal_permutation(batch_latent_depth_raw[i], reordered_batch_permutations_recentered[i])
                                for i in range(len(batch_latent_depth_raw))
                            ],
                            axis=0,
                        )
                    )
                    nonlinear_probe_recentered["tau"].append(
                        np.stack(
                            [
                                _apply_temporal_permutation(batch_latent_tau_raw[i], reordered_batch_permutations_recentered[i])
                                for i in range(len(batch_latent_tau_raw))
                            ],
                            axis=0,
                        )
                    )
                    nonlinear_probe_recentered["feat"].append(
                        np.stack(
                            [
                                _apply_temporal_permutation(batch_latent_feat_raw[i], reordered_batch_permutations_recentered[i])
                                for i in range(len(batch_latent_feat_raw))
                            ],
                            axis=0,
                        )
                    )

            # if evaluating latent alignment, also store the ground-truth coordinates for this batch for later comparison with predicted coordinates in latent space
            if needs_alignment_targets:
                gt_coordinates.append(batch_gt_coordinates)  # list of [B, T, n_kp, 2] np arrays for each batch    
                if collect_raw_physical_targets:
                    if len(batch_gt_raw_physical_coordinates) == len(batch_gt_coordinates):
                        gt_raw_physical_coordinates.append(batch_gt_raw_physical_coordinates)
                    else:
                        collect_raw_physical_targets = False
                        gt_raw_physical_coordinates = []
                        raw_physical_unavailable_reason = (
                            "raw physical target count did not match kept image-space target count "
                            f"for batch {batch_idx}"
                        )
                        print(
                            "[WARN] Raw physical latent-alignment metrics disabled: "
                            f"{raw_physical_unavailable_reason}"
                        )

            # plot trajectories of predicted coordinates for this batch of videos
            # Skip smoothing comparison plots for latent extraction
            if visualize_trajectories:
                if extraction_method == 'latent' and latent_position_variant == 'both':
                    if reordered_batch_coordinates_recentered is None:
                        raise RuntimeError(
                            "Expected reordered recentered trajectories for latent_position_variant='both'."
                        )
                    # Save only the ordered nominal-vs-recentered comparison to avoid redundant outputs.
                    comparison_coors_path_out = os.path.join(
                        root_out,
                        f"predicted_coordinates_comparison_{extraction_method}"
                    )
                    os.makedirs(comparison_coors_path_out, exist_ok=True)
                    create_trajectory_videos(
                        [
                            {
                                'pred_positions': reordered_batch_coordinates[i],
                                'pred_ordered_positions': reordered_batch_coordinates_recentered[i],
                            }
                            for i in range(len(reordered_batch_coordinates))
                        ],
                        save_dir=comparison_coors_path_out,
                        mode=f"{mode}_nominal_vs_recentered_{extraction_method}",
                        legend_label_first='Nominal',
                        legend_label_second='Recentered',
                        title_override_template=(
                            "Nominal vs Recentered (both ordered)\n"
                            "Frame {t}/{T} (Circle=Recentered, X=Nominal)"
                        ),
                    )
                else:
                    ###### UN-ORDERED PREDICTED TRAJECTORIES ######
                    ### plot and save predicted trajectories for this batch of videos (without re-ordering)
                    coor_path_out = os.path.join(
                        root_out,
                        f"predicted_coordinates_{extraction_method}"
                    )
                    os.makedirs(coor_path_out, exist_ok=True)
                    create_pred_only_trajectory_videos(
                        batch_coordinates_np, 
                        save_dir=coor_path_out, 
                        mode=f"{mode}_{extraction_method}"
                    )

                    ###### **Re-ORDERED** PREDICTED TRAJECTORIES ######
                    ### plot and save predicted trajectories for this batch of videos (with re-ordering) and comparison videos
                    ordered_coor_path_out = os.path.join(
                        root_out,
                        f"predicted_coordinates_ordered_{extraction_method}"
                    )
                    os.makedirs(ordered_coor_path_out, exist_ok=True)
                    create_pred_only_trajectory_videos(
                        reordered_batch_coordinates, 
                        save_dir=ordered_coor_path_out, 
                        mode=f"{mode}_reordered_{extraction_method}"
                    )

                    ###### UN-ORDERED vs **Re-ORDERED** TRAJECTORIES ######
                    ### plot comparison videos showing predicted vs predicted ordered trajectories for this batch of videos
                    comparison_coors_path_out = os.path.join(
                        root_out,
                        f"predicted_coordinates_comparison_{extraction_method}"
                    )
                    os.makedirs(comparison_coors_path_out, exist_ok=True)
                    create_trajectory_videos(
                        [{'pred_positions': batch_coordinates_np[i], 'pred_ordered_positions': reordered_batch_coordinates[i]} 
                           for i in range(len(batch_coordinates_np))
                        ], 
                        save_dir=comparison_coors_path_out, 
                        mode=f"{mode}_comparison_{extraction_method}"
                    )

                    ###### **SMOOTHED Re-ORDERED** TRAJECTORIES ######
                    ### Only plot smoothing comparison for bbox extraction (latent has no smoothing)
                    if extraction_method == 'bbox':
                        smoothed_comparison_coors_path_out = os.path.join(
                            root_out,
                            "predicted_coordinates_smoothed_comparison_bbox"
                        )
                        os.makedirs(smoothed_comparison_coors_path_out, exist_ok=True)
                        create_trajectory_videos(
                            [{'pred_ordered_positions': reordered_batch_coordinates[i], 
                              'smoothed_pred_ordered_positions': smoothed_batch_coordinates[i]} for i in range(len(batch_coordinates_np))], 
                            save_dir=smoothed_comparison_coors_path_out, 
                            mode=f"{mode}_smoothed_comparison", 
                            keys_to_plot=['pred_ordered_positions', 'smoothed_pred_ordered_positions'], 
                            window_length=sg_window_length
                        )

    print(
        f"\nFiltering summary for mode={mode}, extraction={extraction_method}: "
        f"seen={total_trajectories_seen}, kept={kept_trajectories}, "
        f"filtered={len(filtered_trajectory_ids)}"
    )
    if len(recenter_shift_l2_means) > 0:
        print(
            f"Recentering summary: mean(||p_can - p||)={np.mean(recenter_shift_l2_means):.6f} ± {np.std(recenter_shift_l2_means):.6f}, "
            f"valid-centroid-fraction={np.mean(recenter_valid_fractions):.4f}"
        )
    _update_filtering_report_json(
        report_path=filtering_report_path,
        mode=mode,
        extraction_method=extraction_method,
        total_seen=total_trajectories_seen,
        kept=kept_trajectories,
        filtered_ids=filtered_trajectory_ids,
        filtered_details=filtered_trajectory_details,
    )

    # After processing all batches, concatenate all_coordinates 
    all_coordinates_recentered_agg = None
    nonlinear_probe_payload = None
    if extract_coordinates:
        if len(all_coordinates) == 0:
            raise ValueError(
                f"No valid trajectories extracted for mode={mode}. "
                "All trajectories were filtered out."
            )
        all_coordinates = np.concatenate(all_coordinates, axis=0)  # [total_videos, T, n_kp, 2]
        if extraction_method == 'bbox':
            all_coordinates_smoothed = np.concatenate(all_coordinates_smoothed, axis=0)  # [total_videos, T, n_kp, 2]
        else:
            # Latent extraction has no smoothing; reuse same array to avoid duplication.
            all_coordinates_smoothed = all_coordinates
            if latent_position_variant == 'both':
                if len(all_coordinates_recentered) == 0:
                    raise ValueError(
                        "latent_position_variant='both' requested but no recentered trajectories were collected."
                    )
                all_coordinates_recentered_agg = np.concatenate(all_coordinates_recentered, axis=0)

        if collect_nonlinear_probe_inputs:
            if len(nonlinear_probe_nominal["p"]) == 0:
                raise ValueError(
                    f"No nonlinear probe payload could be collected for mode={mode}. "
                    "All trajectories were filtered out."
                )
            nonlinear_probe_payload = {
                "mode": mode,
                "matching": "index_to_index",
                "nominal": {
                    "p": np.concatenate(nonlinear_probe_nominal["p"], axis=0),
                    "s": np.concatenate(nonlinear_probe_nominal["s"], axis=0),
                    "d": np.concatenate(nonlinear_probe_nominal["d"], axis=0),
                    "tau": np.concatenate(nonlinear_probe_nominal["tau"], axis=0),
                    "feat": np.concatenate(nonlinear_probe_nominal["feat"], axis=0),
                },
            }
            if latent_position_variant == 'both':
                if len(nonlinear_probe_recentered["p"]) == 0:
                    raise ValueError(
                        "latent_position_variant='both' requested nonlinear probe collection but "
                        "no recentered payload was aggregated."
                    )
                nonlinear_probe_payload["recentered"] = {
                    "p": np.concatenate(nonlinear_probe_recentered["p"], axis=0),
                    "s": np.concatenate(nonlinear_probe_recentered["s"], axis=0),
                    "d": np.concatenate(nonlinear_probe_recentered["d"], axis=0),
                    "tau": np.concatenate(nonlinear_probe_recentered["tau"], axis=0),
                    "feat": np.concatenate(nonlinear_probe_recentered["feat"], axis=0),
                }

        if returns:
            if extraction_method == 'latent' and latent_position_variant == 'both':
                print(
                    "[WARN] returns=True with latent_position_variant='both' returns the primary nominal trajectories."
                )
            return all_coordinates_smoothed  # Return [V, T, N, 2] numpy array

        print(f"\nExtracted predicted coordinates for {all_coordinates.shape[0]} videos with shape: {all_coordinates.shape}")
    
    gt_coordinates_ = None
    gt_raw_physical_coordinates_ = None
    raw_physical_gt_metadata = None
    if needs_alignment_targets:
        if len(gt_coordinates) == 0:
            raise ValueError(
                f"No valid ground-truth coordinate trajectories available for alignment "
                f"(mode={mode}). All trajectories were filtered out."
            )
        gt_coordinates_ = np.concatenate(gt_coordinates, axis=0)  # [total_videos, T, n_kp, 2]
        if collect_raw_physical_targets and len(gt_raw_physical_coordinates) > 0:
            gt_raw_physical_coordinates_ = np.concatenate(gt_raw_physical_coordinates, axis=0)
            if gt_raw_physical_coordinates_.shape != gt_coordinates_.shape:
                print(
                    "[WARN] Raw physical latent-alignment metrics disabled: "
                    f"raw shape {gt_raw_physical_coordinates_.shape} does not match "
                    f"image-space GT shape {gt_coordinates_.shape}."
                )
                gt_raw_physical_coordinates_ = None
            else:
                raw_physical_gt_metadata = {
                    "source": "synchronized_twobody_npz",
                    "npz_root": raw_physical_npz_root_resolved,
                    "coordinate_order": "yx",
                    "coordinate_units": "raw_physical",
                    "num_videos": int(gt_raw_physical_coordinates_.shape[0]),
                    "T": int(gt_raw_physical_coordinates_.shape[1]),
                    "N": int(gt_raw_physical_coordinates_.shape[2]),
                }
        elif raw_physical_unavailable_reason is not None:
            raw_physical_gt_metadata = {
                "source": "unavailable",
                "reason": raw_physical_unavailable_reason,
            }

    if return_nonlinear_probe_payload:
        if nonlinear_probe_payload is None or gt_coordinates_ is None:
            raise RuntimeError(
                "return_nonlinear_probe_payload=True expected nonlinear_probe_payload and gt_coordinates_."
            )
        nonlinear_probe_payload["gt"] = gt_coordinates_
        nonlinear_probe_payload["nominal"]["gt"] = gt_coordinates_
        if "recentered" in nonlinear_probe_payload:
            nonlinear_probe_payload["recentered"]["gt"] = gt_coordinates_
        nonlinear_probe_payload["num_batches_processed"] = int(num_batches_processed)
        nonlinear_probe_payload["num_videos_kept"] = int(gt_coordinates_.shape[0])
        nonlinear_probe_payload["num_videos_seen"] = int(total_trajectories_seen)
        return nonlinear_probe_payload

    if evaluate_latent_alignment:
        skip_mse_plot = latent_position_variant == 'both'
        # Save frame-wise reconstruction MSE plot (mean ± std across videos) for this split
        if skip_mse_plot:
            print("Skipping reconstruction MSE plot for latent_position_variant='both' (comparison mode).")
        elif latent_eval_save_dir is not None and len(all_frame_mse_per_video) > 0:
            frame_mse_arr = np.stack(all_frame_mse_per_video, axis=0)  # [V, T]
            mse_mean = frame_mse_arr.mean(axis=0)
            mse_std = frame_mse_arr.std(axis=0)
            ts = np.arange(frame_mse_arr.shape[1])

            plt.figure(figsize=(10, 5))
            plt.plot(ts, mse_mean, color='tab:blue', linewidth=2, label='Mean frame MSE')
            plt.fill_between(ts, mse_mean - mse_std, mse_mean + mse_std, color='tab:blue', alpha=0.2, label='±1 std')
            plt.xlabel('Timestep')
            plt.ylabel('Reconstruction MSE')
            plt.title(f'Reconstruction MSE over Time ({mode}, {extraction_method})')
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            mse_plot_path = os.path.join(
                latent_eval_save_dir,
                f"reconstruction_mse_over_time_{extraction_method}_{mode}.png"
            )
            plt.savefig(mse_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved reconstruction MSE plot to: {mse_plot_path}")

        save_dir_metrics = latent_eval_save_dir if latent_eval_save_dir is not None else save_dir

        if extraction_method == 'latent':
            metrics_common = dict(
                mode=mode,
                extraction_method=extraction_method,
                use_hungarian_for_correlation=use_hungarian_for_correlation,
                reorder_method=reorder_method,
            )

            if latent_position_variant == 'both':
                if all_coordinates_recentered_agg is None:
                    raise RuntimeError(
                        "Expected all_coordinates_recentered_agg for latent_position_variant='both', found None."
                    )

                nominal_metrics = evaluate_latent_alignment_metrics(
                    all_coordinates_smoothed,
                    gt_coordinates_,
                    save_dir=None,
                    raw_physical_gt_coordinates=gt_raw_physical_coordinates_,
                    raw_physical_gt_metadata=raw_physical_gt_metadata,
                    metrics_filename_extra_parts=['variant_nominal'],
                    extra_metadata={
                        'latent_position_variant': 'nominal',
                        'latent_recenter_source': latent_recenter_source,
                        'latent_recenter_nms_source': latent_recenter_nms_source,
                        'latent_recenter_eps': latent_recenter_eps,
                    },
                    **metrics_common,
                )
                recentered_metrics = evaluate_latent_alignment_metrics(
                    all_coordinates_recentered_agg,
                    gt_coordinates_,
                    save_dir=None,
                    raw_physical_gt_coordinates=gt_raw_physical_coordinates_,
                    raw_physical_gt_metadata=raw_physical_gt_metadata,
                    metrics_filename_extra_parts=[
                        'variant_recentered',
                        f'source_{latent_recenter_source}',
                        f'nms_{latent_recenter_nms_source}',
                    ],
                    extra_metadata={
                        'latent_position_variant': 'recentered',
                        'latent_recenter_source': latent_recenter_source,
                        'latent_recenter_nms_source': latent_recenter_nms_source,
                        'latent_recenter_eps': latent_recenter_eps,
                    },
                    **metrics_common,
                )

                if save_dir_metrics is not None:
                    os.makedirs(save_dir_metrics, exist_ok=True)
                    metrics_filename = (
                        f"latent_alignment_metrics_{extraction_method}_{mode}_variant_both"
                        f"_source_{latent_recenter_source}_nms_{latent_recenter_nms_source}"
                        f"_reorder_{reorder_method}_hungarian_{'on' if use_hungarian_for_correlation else 'off'}.json"
                    )
                    metrics_path = os.path.join(save_dir_metrics, metrics_filename)
                    combined_metrics = {
                        'comparison_type': 'nominal_vs_recentered',
                        'mode': mode,
                        'extraction_method': extraction_method,
                        'reorder_method': reorder_method,
                        'use_hungarian_for_correlation': bool(use_hungarian_for_correlation),
                        'latent_recenter_source': latent_recenter_source,
                        'latent_recenter_nms_source': latent_recenter_nms_source,
                        'latent_recenter_eps': float(latent_recenter_eps),
                        'metric_definitions': {
                            'velocity': 'first temporal difference q[t+1] - q[t]',
                            'discrete_acceleration': (
                                'second temporal difference q[t+2] - 2*q[t+1] + q[t]; '
                                'the common dt^-2 factor is omitted'
                            ),
                            'smoothness_ratio_acc_rms': (
                                'RMS discrete acceleration of the aligned prediction divided by '
                                'RMS discrete acceleration of the target'
                            ),
                        },
                        'nominal': nominal_metrics,
                        'recentered': recentered_metrics,
                    }
                    with open(metrics_path, 'w') as f:
                        json.dump(combined_metrics, f, indent=2)
                    print(f"Saved combined latent alignment metrics to: {metrics_path}")

            else:
                metrics_common_with_save = dict(metrics_common)
                metrics_common_with_save['save_dir'] = save_dir_metrics

                if latent_position_variant == 'nominal':
                    evaluate_latent_alignment_metrics(
                        all_coordinates_smoothed,
                        gt_coordinates_,
                        raw_physical_gt_coordinates=gt_raw_physical_coordinates_,
                        raw_physical_gt_metadata=raw_physical_gt_metadata,
                        extra_metadata={
                            'latent_position_variant': 'nominal',
                            'latent_recenter_source': latent_recenter_source,
                            'latent_recenter_nms_source': latent_recenter_nms_source,
                            'latent_recenter_eps': latent_recenter_eps,
                        },
                        **metrics_common_with_save,
                    )
                elif latent_position_variant == 'recentered':
                    evaluate_latent_alignment_metrics(
                        all_coordinates_smoothed,
                        gt_coordinates_,
                        raw_physical_gt_coordinates=gt_raw_physical_coordinates_,
                        raw_physical_gt_metadata=raw_physical_gt_metadata,
                        metrics_filename_extra_parts=[
                            'variant_recentered',
                            f'source_{latent_recenter_source}',
                            f'nms_{latent_recenter_nms_source}',
                        ],
                        extra_metadata={
                            'latent_position_variant': 'recentered',
                            'latent_recenter_source': latent_recenter_source,
                            'latent_recenter_nms_source': latent_recenter_nms_source,
                            'latent_recenter_eps': latent_recenter_eps,
                        },
                        **metrics_common_with_save,
                    )

            print("Skipping non-smoothed latent-alignment pass: latent extraction has no smoothing (identical trajectories).")

        else:
            pred_coordinates = all_coordinates_smoothed  # expected shape: [V, T, N, 2]
            evaluate_latent_alignment_metrics(
                pred_coordinates, 
                gt_coordinates_, 
                save_dir=save_dir_metrics,
                mode=mode,
                extraction_method=extraction_method,
                use_hungarian_for_correlation=use_hungarian_for_correlation,
                reorder_method=reorder_method,
                raw_physical_gt_coordinates=gt_raw_physical_coordinates_,
                raw_physical_gt_metadata=raw_physical_gt_metadata,
            )

            pred_coordinates = all_coordinates
            evaluate_latent_alignment_metrics(
                pred_coordinates, 
                gt_coordinates_, 
                save_dir=save_dir_metrics,
                mode=mode,
                name_out_str='non_smoothed',
                extraction_method=extraction_method,
                use_hungarian_for_correlation=use_hungarian_for_correlation,
                reorder_method=reorder_method,
                raw_physical_gt_coordinates=gt_raw_physical_coordinates_,
                raw_physical_gt_metadata=raw_physical_gt_metadata,
            )

    if evaluate_noisy_gt_reference:
        evaluate_noisy_gt_reference_metrics(
            gt_coordinates=gt_coordinates_,
            save_dir=latent_eval_save_dir if latent_eval_save_dir is not None else save_dir,
            mode=mode,
            extraction_method=extraction_method,
            noise_mode=noisy_gt_noise_mode,
            noise_alphas=noisy_gt_noise_alphas,
            noise_seed=noisy_gt_noise_seed,
        )
#####################################################################################

#####################################################################################
def evaluate_latent_alignment_nonlinear_joint(
    model,
    config,
    device=torch.device('cpu'),
    batch_size=32,
    max_batches=None,
    save_dir=None,
    latent_eval_save_dir=None,
    use_hungarian_for_correlation=False,
    reorder_method='smallest_consecutive_distance',
    extraction_method='latent',
    latent_position_variant='both',
    latent_recenter_source='patch_alpha',
    latent_recenter_nms_source='nominal',
    latent_recenter_eps=1e-6,
    train_seq_len=60,
    valid_seq_len=360,
    probe_hidden_dim=64,
    probe_num_hidden_layers=2,
    probe_epochs=25,
    probe_batch_size=4096,
    probe_lr=1e-3,
    probe_weight_decay=1e-6,
    probe_seed=0,
):
    if extraction_method != 'latent':
        raise ValueError("Nonlinear latent-alignment probe requires extraction_method='latent'.")
    if latent_position_variant != 'both':
        raise ValueError("Nonlinear latent-alignment probe requires latent_position_variant='both'.")
    if reorder_method != 'smallest_consecutive_distance':
        raise ValueError(
            "Nonlinear latent-alignment probe requires reorder_method='smallest_consecutive_distance'."
        )

    if latent_eval_save_dir is None:
        latent_eval_save_dir = save_dir
    if latent_eval_save_dir is None:
        raise ValueError("A save directory is required for nonlinear latent-alignment evaluation.")
    os.makedirs(latent_eval_save_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("NONLINEAR LATENT ALIGNMENT EVALUATION")
    print("=" * 80)
    print("Fitting MLP probes on TRAIN and evaluating on TRAIN + VALID")
    print(f"Train extraction seq_len: {int(train_seq_len)}")
    print(f"Valid extraction seq_len before truncation: {int(valid_seq_len)}")
    print(f"Valid evaluation seq_len after truncation: {int(train_seq_len)}")
    print(f"Supervised Hungarian matching: {'ON' if use_hungarian_for_correlation else 'OFF'}")
    print("=" * 80)

    common_kwargs = dict(
        model=model,
        config=config,
        device=device,
        batch_size=batch_size,
        max_batches=max_batches,
        save_dir=None,
        latent_eval_save_dir=None,
        visualize_trajectories=False,
        extract_coordinates=True,
        returns=False,
        evaluate_latent_alignment=False,
        evaluate_noisy_gt_reference=False,
        use_hungarian_for_correlation=use_hungarian_for_correlation,
        reorder_method=reorder_method,
        sg_window_length=15,
        sg_polyorder=2,
        extraction_method=extraction_method,
        filtering_report_path=None,
        latent_position_variant=latent_position_variant,
        latent_recenter_source=latent_recenter_source,
        latent_recenter_nms_source=latent_recenter_nms_source,
        latent_recenter_eps=latent_recenter_eps,
        collect_nonlinear_probe_inputs=True,
        return_nonlinear_probe_payload=True,
    )

    train_payload = video_to_trajectory(
        mode='train',
        eval_seq_len=int(train_seq_len),
        **common_kwargs,
    )
    valid_payload = video_to_trajectory(
        mode='valid',
        eval_seq_len=int(valid_seq_len),
        **common_kwargs,
    )

    train_num_objects = int(train_payload["gt"].shape[2])
    valid_num_objects = int(valid_payload["gt"].shape[2])
    if train_num_objects != valid_num_objects:
        raise ValueError(
            f"Train/valid object-count mismatch for nonlinear probe evaluation: "
            f"train N={train_num_objects}, valid N={valid_num_objects}."
        )
    if train_num_objects != 2:
        raise ValueError(
            "Nonlinear latent-alignment probe currently supports exactly 2 objects because "
            "it relies on reorder_method='smallest_consecutive_distance'. "
            f"Got N={train_num_objects}."
        )

    train_eval_seq_len = int(train_payload["gt"].shape[1])
    valid_source_seq_len = int(valid_payload["gt"].shape[1])
    if valid_source_seq_len < train_eval_seq_len:
        raise ValueError(
            f"Valid split length ({valid_source_seq_len}) is shorter than train probe length "
            f"({train_eval_seq_len}); cannot truncate valid to match train."
        )
    valid_payload = _truncate_nonlinear_probe_split_payload(valid_payload, train_eval_seq_len)

    matching_mode = "hungarian" if use_hungarian_for_correlation else "index_to_index"
    if use_hungarian_for_correlation:
        train_payload["nominal"] = _apply_supervised_matching_to_variant_payload(train_payload["nominal"])
        train_payload["recentered"] = _apply_supervised_matching_to_variant_payload(train_payload["recentered"])
        valid_payload["nominal"] = _apply_supervised_matching_to_variant_payload(valid_payload["nominal"])
        valid_payload["recentered"] = _apply_supervised_matching_to_variant_payload(valid_payload["recentered"])
        train_payload["matching"] = matching_mode
        valid_payload["matching"] = matching_mode

    probe_specs = {
        "p_only": ["p"],
        "p_s": ["p", "s"],
        "p_s_d_t": ["p", "s", "d", "tau"],
        "p_s_d_t_feat": ["p", "s", "d", "tau", "feat"],
    }
    results = {
        "comparison_type": "nominal_vs_recentered_nonlinear_probe",
        "extraction_method": extraction_method,
        "reorder_method": reorder_method,
        "use_hungarian_for_correlation": bool(use_hungarian_for_correlation),
        "matching": matching_mode,
        "latent_recenter_source": latent_recenter_source,
        "latent_recenter_nms_source": latent_recenter_nms_source,
        "latent_recenter_eps": float(latent_recenter_eps),
        "train_mode": {
            "mode": "train",
            "seq_len": train_eval_seq_len,
            "num_videos": int(train_payload["gt"].shape[0]),
        },
        "valid_mode": {
            "mode": "valid",
            "source_seq_len": valid_source_seq_len,
            "eval_seq_len": train_eval_seq_len,
            "num_videos": int(valid_payload["gt"].shape[0]),
        },
        "probe_config": {
            "hidden_dim": int(probe_hidden_dim),
            "num_hidden_layers": int(probe_num_hidden_layers),
            "epochs": int(probe_epochs),
            "batch_size": int(probe_batch_size),
            "lr": float(probe_lr),
            "weight_decay": float(probe_weight_decay),
            "seed": int(probe_seed),
        },
        "nominal": {},
        "recentered": {},
    }

    for variant_idx, variant_name in enumerate(["nominal", "recentered"]):
        print("\n" + "-" * 80)
        print(f"Training nonlinear probes for variant='{variant_name}'")
        print("-" * 80)
        train_variant = train_payload[variant_name]
        valid_variant = valid_payload[variant_name]
        variant_results = {}
        for probe_idx, (probe_name, feature_names) in enumerate(probe_specs.items()):
            train_features = _build_nonlinear_probe_feature_tensor(train_variant, probe_name)
            train_inputs_flat = train_features.reshape(-1, train_features.shape[-1])
            train_targets_flat = np.asarray(train_variant["gt"], dtype=np.float32).reshape(-1, 2)
            fit_seed = int(probe_seed) + 100 * variant_idx + probe_idx
            print(
                f"  [Probe] {probe_name} | features={feature_names} | "
                f"train_samples={train_inputs_flat.shape[0]} | input_dim={train_inputs_flat.shape[1]}"
            )
            probe_bundle = _fit_nonlinear_probe(
                train_inputs=train_inputs_flat,
                train_targets=train_targets_flat,
                device=device,
                hidden_dim=probe_hidden_dim,
                num_hidden_layers=probe_num_hidden_layers,
                epochs=probe_epochs,
                batch_size=probe_batch_size,
                lr=probe_lr,
                weight_decay=probe_weight_decay,
                seed=fit_seed,
            )
            train_metrics = _evaluate_nonlinear_probe_on_split(probe_bundle, train_variant, probe_name)
            valid_metrics = _evaluate_nonlinear_probe_on_split(probe_bundle, valid_variant, probe_name)

            variant_results[probe_name] = {
                "feature_names": feature_names,
                "input_dim": int(train_inputs_flat.shape[1]),
                "num_parameters": int(probe_bundle["num_parameters"]),
                "training": {
                    "history": probe_bundle["training_history"],
                    "normalization": {
                        "input_mean": [float(v) for v in probe_bundle["x_mean"].detach().cpu().numpy().tolist()],
                        "input_std": [float(v) for v in probe_bundle["x_std"].detach().cpu().numpy().tolist()],
                        "target_mean": [float(v) for v in probe_bundle["y_mean"].detach().cpu().numpy().tolist()],
                        "target_std": [float(v) for v in probe_bundle["y_std"].detach().cpu().numpy().tolist()],
                    },
                    "fit_config": probe_bundle["config"],
                },
                "train": train_metrics,
                "valid": valid_metrics,
            }
        results[variant_name] = variant_results

    metrics_filename = (
        f"latent_alignment_metrics_nonlinear_{extraction_method}_train_valid_variant_both"
        f"_source_{latent_recenter_source}_nms_{latent_recenter_nms_source}"
        f"_reorder_{reorder_method}_hungarian_{'on' if use_hungarian_for_correlation else 'off'}.json"
    )
    metrics_path = os.path.join(latent_eval_save_dir, metrics_filename)
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved nonlinear latent-alignment metrics to: {metrics_path}")
    return results
#####################################################################################

#####################################################################################
def save_trajectories_to_GHNN_format(
    model,
    config,
    device,
    output_dir=None,
    dataset_name='ddlp_extracted',
    train_seq_len=60,
    eval_seq_len=360,
    batch_size=32,
    step_size=1.,
    dt=1.,
    masses=None,
    validation_share=0.025,
    test_share=0.025,
    seed=0,
    sg_window_length=15,
    sg_polyorder=2,
    smooth_momentum=False,
    extraction_method='bbox',
    filtering_report_path=None,
    latent_position_variant='nominal',
    latent_recenter_source='patch_alpha',
    latent_recenter_nms_source='nominal',
    latent_recenter_eps=1e-6,
    prob_encoder_route='none',
    prob_encoder_frozen_model=None,
    write_prob_encoder_alignment_metrics=False,
    alignment_split='test',
    latent_eval_save_dir=None,
    use_hungarian_for_correlation=False,
):
    """
    Extract trajectories from DDLP model for all splits and save in GHNN format.
    
    This function orchestrates the complete pipeline:
    1. Calls video_to_trajectory for train/valid/test splits to extract, filter,
       reorder, and smooth trajectories
    2. Calls convert_trajectories_to_GHNN_format to save in GHNN HDF5 format
    
    Args:
        model: Trained DDLP model
        config: Configuration dictionary
        device: PyTorch device
        output_dir: Directory to save GHNN-formatted data (if None, uses config['root']/ghnn_converted)
        dataset_name: Base name for output files
        train_seq_len: Sequence length to extract for training split
        eval_seq_len: Sequence length to extract for validation and test splits
        batch_size: Batch size for model inference
        step_size: Time step size for GHNN training data
        dt: Simulation time step
        masses: Particle masses (if None, uses [1.0, 1.0, ...])
        validation_share: Fraction for validation split
        test_share: Fraction for test split
        seed: Random seed for splits
        sg_window_length: Window length for Savitzky-Golay smoothing (must be odd, default: 15)
        sg_polyorder: Polynomial order for Savitzky-Golay smoothing (default: 2)
        smooth_momentum: Whether to apply Savitzky-Golay smoothing to momentum (default: False)
        extraction_method: Method for trajectory extraction ('bbox' or 'latent')
        filtering_report_path: Path to filtering report JSON shared across split extraction.
        prob_encoder_route: Optional probabilistic encoder route identifier
            in {'none', 'c1', 'c1-dyn-attrs'}.
        prob_encoder_frozen_model: Frozen base DDLP model required when
            prob_encoder_route is not 'none'.
        write_prob_encoder_alignment_metrics: Whether to run an alignment-metric pass
            after HDF5 export.
        alignment_split: Split used by the optional alignment-metric pass.
        latent_eval_save_dir: Directory for optional alignment metrics.
        use_hungarian_for_correlation: Whether to apply supervised Hungarian
            matching in the optional alignment-metric pass.
    
    Returns:
        Dictionary with paths to generated files and statistics
    """
    prob_encoder_route = _normalize_prob_encoder_route(prob_encoder_route)

    print("\n" + "="*80)
    print("EXTRACTING DDLP TRAJECTORIES AND CONVERTING TO GHNN FORMAT")
    print("="*80)
    print(f"Extraction method: {extraction_method.upper()}")
    if extraction_method == 'bbox':
        print(f"Savitzky-Golay filter: window_length={sg_window_length}, polyorder={sg_polyorder}")
    else:
        print(f"No position smoothing (latent extraction preserves model dynamics)")
        print(f"Latent position variant: {latent_position_variant}")
        print(f"Recentering source: {latent_recenter_source}")
        print(f"Recentering NMS source: {latent_recenter_nms_source}")
    print(f"Momentum smoothing: {'ENABLED' if smooth_momentum else 'DISABLED'}")
    print("="*80)
    
    if output_dir is None:
        output_dir = os.path.join(config['root'], 'ghnn_converted')
    
    # Extract trajectories from all splits using video_to_trajectory
    trajectories_dict = {}
    
    for mode in ['train', 'valid', 'test']:
        print(f"\n{'='*80}")
        print(f"Processing {mode.upper()} split...")
        print(f"{'='*80}")
        
        # Use different sequence lengths for train vs valid/test
        seq_len = train_seq_len if mode == 'train' else eval_seq_len
        print(f"Using sequence length: {seq_len}")
        
        try:
            # Use video_to_trajectory with returns=True to get smoothed, reordered trajectories
            # video_to_trajectory already handles:
            # - Extraction from alpha masks with NMS filtering (bbox) OR latent positions (latent)
            # - Intra-Sequence Reordering via smallest_consecutive_distance
            # - Savitzky-Golay smoothing (only for bbox method)
            trajectories = video_to_trajectory(
                model=model,
                config=config,
                device=device,
                mode=mode,
                batch_size=batch_size,
                max_batches=None,  # Process all batches
                eval_seq_len=seq_len,
                save_dir=None,  # Don't save visualizations
                visualize_trajectories=False,
                extract_coordinates=True,
                returns=True,  # Return the extracted coordinates
                evaluate_latent_alignment=False,
                sg_window_length=sg_window_length,
                sg_polyorder=sg_polyorder,
                extraction_method=extraction_method,
                filtering_report_path=filtering_report_path,
                latent_position_variant=latent_position_variant,
                latent_recenter_source=latent_recenter_source,
                latent_recenter_nms_source=latent_recenter_nms_source,
                latent_recenter_eps=latent_recenter_eps,
                prob_encoder_route=prob_encoder_route,
                prob_encoder_frozen_model=prob_encoder_frozen_model,
            )
            
            if trajectories is not None and trajectories.shape[0] > 0:
                trajectories_dict[mode] = trajectories
                print(f"Extracted {trajectories.shape[0]} trajectories from {mode} split")
            else:
                print(f"No trajectories extracted from {mode} split")
                
        except Exception as e:
            print(f"Error processing {mode} split: {e}")
            import traceback
            traceback.print_exc()
            print(f"Skipping {mode} split...")
            continue
    
    if not trajectories_dict:
        raise ValueError("No trajectories extracted from any split!")
    
    # Convert to GHNN format
    print(f"\n{'='*80}")
    print("Converting to GHNN format...")
    print(f"{'='*80}")
    
    result = convert_trajectories_to_GHNN_format(
        trajectories_dict=trajectories_dict,
        output_dir=output_dir,
        dataset_name=dataset_name,
        step_size=step_size,
        dt=dt,
        masses=masses,
        validation_share=validation_share,
        test_share=test_share,
        seed=seed,
        sg_window_length=sg_window_length,
        sg_polyorder=sg_polyorder,
        smooth_momentum=smooth_momentum,
        train_seq_len=train_seq_len,
        eval_seq_len=eval_seq_len,
    )

    if write_prob_encoder_alignment_metrics:
        if prob_encoder_route == 'none':
            print("[WARN] Requested probabilistic encoder alignment metrics without a probabilistic encoder route.")
        else:
            alignment_mode = alignment_split
            alignment_seq_len = train_seq_len if alignment_mode == 'train' else eval_seq_len
            alignment_save_dir = None
            if latent_eval_save_dir is not None:
                alignment_save_dir = os.path.join(
                    latent_eval_save_dir,
                    _prob_encoder_output_dirname(prob_encoder_route),
                )
                os.makedirs(alignment_save_dir, exist_ok=True)
            print("\n" + "="*80)
            print(f"RUNNING PROBABILISTIC ENCODER ALIGNMENT METRICS ({alignment_mode.upper()})")
            print("="*80)
            video_to_trajectory(
                model=model,
                config=config,
                device=device,
                mode=alignment_mode,
                batch_size=batch_size,
                max_batches=None,
                eval_seq_len=alignment_seq_len,
                save_dir=None,
                latent_eval_save_dir=alignment_save_dir,
                visualize_trajectories=False,
                extract_coordinates=True,
                returns=False,
                evaluate_latent_alignment=True,
                use_hungarian_for_correlation=use_hungarian_for_correlation,
                reorder_method='smallest_consecutive_distance',
                sg_window_length=sg_window_length,
                sg_polyorder=sg_polyorder,
                extraction_method=extraction_method,
                filtering_report_path=None,
                latent_position_variant=latent_position_variant,
                latent_recenter_source=latent_recenter_source,
                latent_recenter_nms_source=latent_recenter_nms_source,
                latent_recenter_eps=latent_recenter_eps,
                prob_encoder_route=prob_encoder_route,
                prob_encoder_frozen_model=prob_encoder_frozen_model,
            )
    
    print("\n Conversion completed successfully!")
    print(f"Output files:")
    print(f"  - {result['all_runs_path']}")
    print(f"  - {result['training_path']}")
    
    return result

def convert_trajectories_to_GHNN_format(
    trajectories_dict,
    output_dir,
    dataset_name='ddlp_extracted',
    step_size=1.0,
    dt=1.0,
    masses=None,
    validation_share=0.025,
    test_share=0.025,
    seed=0,
    sg_window_length=15,
    sg_polyorder=2,
    smooth_momentum=False,
    train_seq_len=None,
    eval_seq_len=None,
):
    """
    Convert extracted and processed trajectories to GHNN-compatible HDF5 format.
    
    This function takes trajectories that have already been extracted, filtered,
    reordered, and smoothed by video_to_trajectory, and saves them in the exact
    format expected by GHNN models.
    
    Args:
        trajectories_dict: Dictionary with keys 'train', 'valid', 'test' containing
                          numpy arrays of shape [V, T, N, 2] for each split
        output_dir: Directory to save GHNN-formatted HDF5 files
        dataset_name: Base name for output files
        step_size: Time step size for GHNN training data (for features/labels pairs)
        dt: Simulation time step (for time column in trajectories)
        masses: List/array of particle masses [m_A, m_B, ...] (if None, uses [1.0, 1.0, ...])
        validation_share: Fraction of runs for validation split
        test_share: Fraction of runs for test split
        seed: Random seed for train/val/test split
        sg_window_length: Window length for Savitzky-Golay smoothing (must be odd)
        sg_polyorder: Polynomial order for Savitzky-Golay smoothing
        smooth_momentum: Whether to apply Savitzky-Golay smoothing to momentum after finite-difference computation
        train_seq_len: Expected trajectory length for train split (for validation)
        eval_seq_len: Expected trajectory length for valid/test splits (for validation)
    
    Returns:
        Dictionary with paths to generated files and statistics
    """
    
    print("\n" + "="*80)
    print("CONVERTING TRAJECTORIES TO GHNN FORMAT")
    print("="*80)
    
    if smooth_momentum:
        print(f"⚠️  MOMENTUM SMOOTHING ENABLED: window_length={sg_window_length}, polyorder={sg_polyorder}")
        print(f"   Note: This further reduces symplecticity but may improve numerical stability")
        print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output file paths
    all_runs_path = os.path.join(output_dir, f'{dataset_name}_all_runs.h5.1')
    training_path = os.path.join(output_dir, f'{dataset_name}_training.h5.1')
    
    # Concatenate all trajectories from all splits while tracking source split IDs.
    all_trajectories_list = []
    split_run_ids = {'train': [], 'valid': [], 'test': []}
    for mode in ['train', 'valid', 'test']:
        if mode in trajectories_dict and trajectories_dict[mode] is not None:
            traj = trajectories_dict[mode]
            print(f"{mode.capitalize()} split: {traj.shape[0]} videos, shape {traj.shape}")
            if traj.shape[0] == 0:
                raise ValueError(f"{mode} split has zero trajectories. Cannot create GHNN split outputs.")
            # Convert to list of individual trajectories: [V, T, N, 2] -> list of [T, N, 2]
            for v in range(traj.shape[0]):
                run_id = len(all_trajectories_list)
                all_trajectories_list.append(traj[v])  # [T, N, 2]
                split_run_ids[mode].append(run_id)
    
    if len(all_trajectories_list) == 0:
        raise ValueError("No trajectories provided in trajectories_dict!")

    # Require all canonical source splits for deterministic GHNN partitioning.
    missing_splits = [m for m in ['train', 'valid', 'test'] if len(split_run_ids[m]) == 0]
    if missing_splits:
        raise ValueError(
            f"Missing source trajectories for split(s): {missing_splits}. "
            "Cannot produce fixed-length /features, /val_features, /test_features."
        )
    
    num_runs = len(all_trajectories_list)
    print(f"\nTotal runs to process: {num_runs}")
    
    # Determine dimensions and bodies from first trajectory
    sample_traj = all_trajectories_list[0]
    T_sample, n_objects, _ = sample_traj.shape
    bodies = [chr(65 + i) for i in range(n_objects)]  # ['A', 'B', 'C', ...]
    dims = ['x', 'y']
    
    print(f"Detected {n_objects} objects: {bodies}")
    print(f"Sample trajectory length: {T_sample} timesteps")
    
    # Default masses if not provided
    if masses is None:
        masses = np.ones(n_objects)
    else:
        masses = np.array(masses)
        if len(masses) != n_objects:
            print(f"Warning: masses length ({len(masses)}) != n_objects ({n_objects}), using default [1.0, ...]")
            masses = np.ones(n_objects)
    
    # =========================================================================
    # Step 1: Create all_runs.h5.1 with individual runs
    # =========================================================================
    print(f"\nCreating {all_runs_path}...")
    
    kwargs = {'complib': 'zlib', 'complevel': 1}
    
    for run_idx, traj in enumerate(tqdm(all_trajectories_list, desc="Writing individual runs")):
        # traj shape: [T, n_objects, 2] - positions in normalized coordinates
        T = traj.shape[0]
        
        # Create time column
        time_col = np.arange(T) * dt
        
        # Prepare positions: q_A_x, q_A_y, q_B_x, q_B_y, ...
        q_cols = {}
        for obj_idx, body in enumerate(bodies):
            q_cols[f'q_{body}_x'] = traj[:, obj_idx, 0]
            q_cols[f'q_{body}_y'] = traj[:, obj_idx, 1]
        
        # Estimate momenta from positions (p = m * v, where v ≈ dq/dt)
        # Use centered finite differences: v[i] = (q[i+1] - q[i-1]) / (2*dt)
        # For boundaries: forward/backward differences
        p_cols = {}
        for obj_idx, body in enumerate(bodies):
            qx = traj[:, obj_idx, 0]
            qy = traj[:, obj_idx, 1]
            
            # Initialize velocity arrays
            vx = np.zeros_like(qx)
            vy = np.zeros_like(qy)
            
            # Forward difference at t=0: v[0] = (q[1] - q[0]) / dt
            vx[0] = (qx[1] - qx[0]) / dt
            vy[0] = (qy[1] - qy[0]) / dt
            
            # Centered difference for interior points: v[i] = (q[i+1] - q[i-1]) / (2*dt)
            vx[1:-1] = (qx[2:] - qx[:-2]) / (2.0 * dt)
            vy[1:-1] = (qy[2:] - qy[:-2]) / (2.0 * dt)
            
            # Backward difference at t=T-1: v[-1] = (q[-1] - q[-2]) / dt
            vx[-1] = (qx[-1] - qx[-2]) / dt
            vy[-1] = (qy[-1] - qy[-2]) / dt
            
            p_cols[f'p_{body}_x'] = masses[obj_idx] * vx
            p_cols[f'p_{body}_y'] = masses[obj_idx] * vy
        
        # Optionally smooth momentum to reduce high-frequency noise
        if smooth_momentum:
            for obj_idx, body in enumerate(bodies):
                # Apply Savitzky-Golay filter to momentum components
                p_x_raw = p_cols[f'p_{body}_x']
                p_y_raw = p_cols[f'p_{body}_y']
                
                # Smooth momentum (only if trajectory is long enough)
                if len(p_x_raw) >= sg_window_length:
                    p_cols[f'p_{body}_x'] = savgol_filter(p_x_raw, window_length=sg_window_length, polyorder=sg_polyorder)
                    p_cols[f'p_{body}_y'] = savgol_filter(p_y_raw, window_length=sg_window_length, polyorder=sg_polyorder)
        
        # Create DataFrame for this run
        run_data = {}
        run_data.update(q_cols)
        run_data.update(p_cols)
        run_data['time'] = time_col
        
        # Column order: q_A_x, q_A_y, q_B_x, q_B_y, p_A_x, p_A_y, p_B_x, p_B_y, time
        columns = [f'q_{body}_{dim}' for body in bodies for dim in dims] + \
                  [f'p_{body}_{dim}' for body in bodies for dim in dims] + \
                  ['time']
        
        df_run = pd.DataFrame(run_data, columns=columns)
        df_run.to_hdf(all_runs_path, key=f'/run{run_idx}', format='fixed', **kwargs)
    
    # Combine all runs into /all_runs
    print("\nCombining runs into /all_runs...")
    runs_dict = {}
    for i in tqdm(range(num_runs), desc="Loading runs for combination"):
        runs_dict[i] = pd.read_hdf(all_runs_path, f'/run{i}').rename_axis("timestep")
    
    all_runs_df = pd.concat(runs_dict, names=['run'])
    all_runs_df.to_hdf(all_runs_path, key='/all_runs', format='fixed', **kwargs)
    
    # Store constants
    constants = pd.Series({
        'bodies': bodies,
        'dimensions': dims,
        'masses': [masses.tolist()],  # Store as list
        'dt': dt,  # Simulation timestep (for individual run trajectories)
        'step_size': dt,  # Keep for backward compatibility
        'num_runs': num_runs,
        'validation_share': validation_share,
        'test_share': test_share,
        'seed': seed,
        'source': 'DDLP_model',
        'scale': 'normalized',  # DDLP trajectories are in normalized coordinates
    })
    constants.to_hdf(all_runs_path, key='/constants', format='fixed', **kwargs)
    
    print(f"✅ Saved {all_runs_path}")
    
    # =========================================================================
    # Step 2: Create training.h5.1 with features/labels splits
    # =========================================================================
    print(f"\nCreating {training_path}...")
    
    # Read the all_runs data
    data = pd.read_hdf(all_runs_path, '/all_runs')
    constants_read = pd.read_hdf(all_runs_path, '/constants')
    
    # Define feature and label names (exclude 'time')
    feature_names = [col for col in data.columns if col != 'time']
    label_names = feature_names.copy()
    
    # Prepare for creating (features, labels) pairs
    train_features = []
    train_labels = []
    
    print("Creating features/labels pairs...")
    
    # Validate step_size matches dt (no synthetic data creation)
    if abs(step_size - dt) > 1e-6:
        raise ValueError(
            f"step_size ({step_size}) must equal dt ({dt}) to avoid creating synthetic data points!\n"
            f"Interpolation would generate data NOT extracted from videos.\n"
            f"Use only extracted datapoints: set --ghnn_step_size = --ghnn_dt"
        )
    
    for i in tqdm(range(num_runs), desc="Processing runs for training data"):
        run = data.loc[i].reset_index(drop=True)  # Reset index to ensure proper ordering
        
        # Ensure time column is monotonically increasing
        if not np.all(np.diff(run['time'].values) >= 0):
            print(f"Warning: Run {i} has non-monotonic time values, skipping")
            continue
        
        # Use extracted data directly (no interpolation)
        # Create (features, labels) pairs: features[t] -> labels[t+dt]
        T = len(run)
        if T > 1:
            run_index = np.array([[i]] * (T-1))
            features_data = run[feature_names].values[:-1]  # All but last timestep
            labels_data = run[label_names].values[1:]       # All but first timestep
            train_features.append(np.concatenate((run_index, features_data), axis=1))
            train_labels.append(labels_data)
    
    # Create DataFrames
    train_features = pd.DataFrame(
        np.concatenate(train_features, axis=0),
        columns=['run'] + feature_names
    ).astype({'run': int})
    train_labels = pd.DataFrame(
        np.concatenate(train_labels, axis=0),
        columns=label_names
    )
    
    # Split into train/validation/test by source extraction split (NOT random).
    train_run_ids = np.array(split_run_ids['train'], dtype=int)
    val_run_ids = np.array(split_run_ids['valid'], dtype=int)
    test_run_ids = np.array(split_run_ids['test'], dtype=int)

    # Validate per-source run lengths are uniform and match expected sequence lengths.
    def _unique_run_lengths(run_ids):
        lengths = []
        for rid in run_ids:
            lengths.append(len(data.loc[int(rid)]))
        return sorted(set(lengths))

    train_lengths = _unique_run_lengths(train_run_ids)
    val_lengths = _unique_run_lengths(val_run_ids)
    test_lengths = _unique_run_lengths(test_run_ids)

    if len(train_lengths) != 1:
        raise ValueError(f"Train trajectories have mixed lengths: {train_lengths}")
    if len(val_lengths) != 1:
        raise ValueError(f"Valid trajectories have mixed lengths: {val_lengths}")
    if len(test_lengths) != 1:
        raise ValueError(f"Test trajectories have mixed lengths: {test_lengths}")

    if train_seq_len is not None and train_lengths[0] != int(train_seq_len):
        raise ValueError(
            f"Train trajectories length {train_lengths[0]} does not match expected train_seq_len={train_seq_len}"
        )
    if eval_seq_len is not None:
        if val_lengths[0] != int(eval_seq_len):
            raise ValueError(
                f"Valid trajectories length {val_lengths[0]} does not match expected eval_seq_len={eval_seq_len}"
            )
        if test_lengths[0] != int(eval_seq_len):
            raise ValueError(
                f"Test trajectories length {test_lengths[0]} does not match expected eval_seq_len={eval_seq_len}"
            )

    validation_runs = np.isin(train_features['run'], val_run_ids)
    test_runs = np.isin(train_features['run'], test_run_ids)
    train_runs = np.isin(train_features['run'], train_run_ids)

    n_train = int(len(train_run_ids))
    n_val = int(len(val_run_ids))
    n_test = int(len(test_run_ids))
    
    validation_features = train_features[validation_runs]
    validation_labels = train_labels[validation_runs]
    test_features = train_features[test_runs]
    test_labels = train_labels[test_runs]
    train_features_final = train_features[train_runs]
    train_labels_final = train_labels[train_runs]
    
    # Save to HDF5
    train_features_final.to_hdf(training_path, key='/features', format='fixed', **kwargs)
    train_labels_final.to_hdf(training_path, key='/labels', format='fixed', **kwargs)
    validation_features.to_hdf(training_path, key='/val_features', format='fixed', **kwargs)
    validation_labels.to_hdf(training_path, key='/val_labels', format='fixed', **kwargs)
    test_features.to_hdf(training_path, key='/test_features', format='fixed', **kwargs)
    test_labels.to_hdf(training_path, key='/test_labels', format='fixed', **kwargs)
    
    # Update constants with step_size for training
    constants_train = constants_read.copy()
    constants_train['step_size'] = step_size
    constants_train.to_hdf(training_path, key='/constants', format='fixed', **kwargs)
    
    print(f"✅ Saved {training_path}")
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print("CONVERSION SUMMARY")
    print(f"{'='*80}")
    print(f"Total runs: {num_runs}")
    print("Split strategy: source_split (train->/features, valid->/val_features, test->/test_features)")
    print(f"Training runs: {n_train} ({100*n_train/num_runs:.1f}%)")
    print(f"Validation runs: {n_val} ({100*n_val/num_runs:.1f}%)")
    print(f"Test runs: {n_test} ({100*n_test/num_runs:.1f}%)")
    print(f"\nTraining features shape: {train_features_final.shape}")
    print(f"Training labels shape: {train_labels_final.shape}")
    print(f"Validation features shape: {validation_features.shape}")
    print(f"Test features shape: {test_features.shape}")
    print(f"\nOutput files:")
    print(f"  - {all_runs_path}")
    print(f"  - {training_path}")
    print(f"{'='*80}\n")
    
    return {
        'all_runs_path': all_runs_path,
        'training_path': training_path,
        'num_runs': num_runs,
        'num_objects': n_objects,
        'train_size': n_train,
        'val_size': n_val,
        'test_size': n_test,
    }
#####################################################################################

#####################################################################################
def main():
    parser = argparse.ArgumentParser(description='Generate visualizations for DDLP model')

    ## Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint directory')
    parser.add_argument('--checkpoint_name', type=str, default=None,
                        help='Specific checkpoint file name (overrides default)')
    parser.add_argument('--mode', type=str, default='valid', choices=['train', 'valid', 'test'],
                       help='Dataset split to evaluate')
    parser.add_argument('--batch_size', type=int, default=20,
                       help='Batch size for evaluation (forced to 1 for full-length videos to avoid memory errors)')
    parser.add_argument('--max_batches', type=str, default='1',
                       help='Maximum number of batches in preview mode (extract_coordinates=0). Use an integer, or "all"/"none".')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use (e.g., cuda:0, cpu)')
    parser.add_argument('--eval_seq_len', type=int, default=None,
                       help='Number of timesteps to encode from sequence (default: use timestep_horizon from config)')
    
    ## Execution options
    parser.add_argument('--visualize_trajectories', type=int, default=0,
                        help='Whether to visualize predicted trajectories (0 or 1)')
    parser.add_argument('--extract_coordinates', type=int, default=0, 
                        help='Whether to extract predicted coordinates for further analysis (0 or 1)')
    parser.add_argument('--evaluate_latent_alignment', type=int, default=0,
                        help='Whether to evaluate latent space alignment with ground truth coordinates (0 or 1)')
    parser.add_argument('--evaluate_latent_alignment_nonlinear', type=int, default=0,
                        help='Whether to fit nonlinear MLP probes on train and evaluate on train+valid (0 or 1)')
    parser.add_argument('--evaluate_noisy_gt_reference', type=int, default=0,
                        help='Whether to compute clean-GT vs noisy-GT latent-alignment reference metrics (0 or 1)')
    parser.add_argument('--noisy_gt_noise_mode', type=str, default='relative_step',
                        choices=['relative_step'],
                        help="Noise mode for noisy-GT reference evaluation (currently: 'relative_step')")
    parser.add_argument('--noisy_gt_noise_alphas', type=str, default='0.05,0.1,0.3,0.5,0.7',
                        help='Comma-separated noise alpha values for noisy-GT reference metrics')
    parser.add_argument('--noisy_gt_noise_seed', type=int, default=0,
                        help='Noise seed for noisy-GT reference metrics')
    parser.add_argument('--use_hungarian_for_correlation', type=int, default=0,
                        help='Whether to use Hungarian matching for latent-alignment correlation metrics (0 or 1)')
    parser.add_argument('--reorder_method', type=str, default='smallest_consecutive_distance',
                        choices=['smallest_consecutive_distance', 'hungarian'],
                        help="Method used for unsupervised temporal reordering of predicted trajectories")
    
    ### GHNN conversion specific arguments
    parser.add_argument('--convert_to_ghnn', type=int, default=0,
                        help='Whether to convert trajectories to GHNN format (0 or 1)')
    parser.add_argument('--ghnn_output_dir', type=str, default=None,
                        help='Output directory for GHNN-formatted data (default: auto-generated)')
    parser.add_argument('--ghnn_dataset_name', type=str, default='ddlp_extracted',
                        help='Name for GHNN dataset (default: ddlp_extracted)')
    parser.add_argument('--ghnn_train_seq_len', type=int, default=60,
                        help='Sequence length for training split (default: same as eval_seq_len)')
    parser.add_argument('--ghnn_step_size', type=float, default=1.,
                        help='Time step size for GHNN training data (default: 0.01)')
    parser.add_argument('--ghnn_dt', type=float, default=1.,
                        help='Simulation time step (default: 0.01)')
    
    ### Savitzky-Golay filter parameters
    parser.add_argument('--sg_window_length', type=int, default=15,
                        help='Window length for Savitzky-Golay smoothing (must be odd, default: 15)')
    parser.add_argument('--sg_polyorder', type=int, default=2,
                        help='Polynomial order for Savitzky-Golay smoothing (default: 2, must be < window_length)')
    parser.add_argument('--smooth_momentum', type=int, default=0,
                        help='Whether to apply Savitzky-Golay smoothing to momentum after finite-difference computation (0 or 1, default: 0)')
    
    ### Extraction method
    parser.add_argument('--extraction_method', type=str, default='bbox', choices=['bbox', 'latent'],
                        help='Trajectory extraction method: "bbox" (bounding box centers, with smoothing) or "latent" (continuous latent positions, no smoothing)')
    parser.add_argument('--latent_position_variant', type=str, default='nominal',
                        choices=['nominal', 'recentered', 'both'],
                        help="Latent extraction output variant: 'nominal', 'recentered', or 'both' for paired comparison")
    parser.add_argument('--latent_recenter_source', type=str, default='patch_alpha',
                        choices=['patch_alpha', 'global_alpha'],
                        help="Recentering centroid source: local decoded patch alpha or global alpha masks")
    parser.add_argument('--latent_recenter_nms_source', type=str, default='nominal',
                        choices=['nominal', 'recentered'],
                        help="Which latent coordinates drive NMS/filtering in latent extraction")
    parser.add_argument('--latent_recenter_eps', type=float, default=1e-6,
                        help='Numerical epsilon for recentering centroid computations')
    parser.add_argument('--raw_physical_npz_root', type=str, default='auto',
                        help=(
                            "Synchronized raw-physics NPZ root for additional latent-alignment metrics. "
                            "Use 'auto' to infer from the DDLP HDF5 root, or 'none' to disable."
                        ))
    parser.add_argument('--prob_encoder_checkpoint', type=str, default=None,
                        help='Optional probabilistic encoder checkpoint (C1 family) to use as the latent position source')
    parser.add_argument('--prob_encoder_route', type=str, default='none',
                        choices=['none', 'c1', 'c1-dyn-attrs', 'c1_dyn_attrs'],
                        help="Probabilistic encoder route label: 'c1', 'c1-dyn-attrs', 'c1_dyn_attrs', or 'none'")
    parser.add_argument('--prob_encoder_output_tag', type=str, default=None,
                        help='Optional output-directory tag for probabilistic-encoder evaluations (e.g. c1_dyn_full).')
    parser.add_argument('--write_prob_encoder_alignment_metrics', type=int, default=0,
                        help='When converting to GHNN, also write alignment metrics for the probabilistic encoder source')
    parser.add_argument('--alignment_split', type=str, default='test',
                        choices=['train', 'valid', 'test'],
                        help='Split for optional probabilistic-encoder alignment metrics')
    parser.add_argument('--nonlinear_train_seq_len', type=int, default=60,
                        help='Sequence length to extract for train split in nonlinear probe evaluation')
    parser.add_argument('--nonlinear_valid_seq_len', type=int, default=360,
                        help='Sequence length to extract for valid split before truncation in nonlinear probe evaluation')
    parser.add_argument('--nonlinear_probe_hidden_dim', type=int, default=64,
                        help='Hidden width for nonlinear MLP probes')
    parser.add_argument('--nonlinear_probe_num_hidden_layers', type=int, default=2,
                        help='Number of hidden layers for nonlinear MLP probes')
    parser.add_argument('--nonlinear_probe_epochs', type=int, default=25,
                        help='Number of epochs for nonlinear MLP probe fitting')
    parser.add_argument('--nonlinear_probe_batch_size', type=int, default=4096,
                        help='Batch size for nonlinear MLP probe fitting')
    parser.add_argument('--nonlinear_probe_lr', type=float, default=1e-3,
                        help='Learning rate for nonlinear MLP probe fitting')
    parser.add_argument('--nonlinear_probe_weight_decay', type=float, default=1e-6,
                        help='Weight decay for nonlinear MLP probe fitting')
    parser.add_argument('--nonlinear_probe_seed', type=int, default=0,
                        help='Random seed for nonlinear MLP probe fitting')

    args = parser.parse_args()
    args.prob_encoder_route = _normalize_prob_encoder_route(args.prob_encoder_route)

    # Parse max_batches from string form to int or None
    max_batches_raw = str(args.max_batches).strip().lower()
    if max_batches_raw in ('all', 'none'):
        args.max_batches = None
    else:
        try:
            args.max_batches = int(max_batches_raw)
        except ValueError as e:
            raise ValueError(
                f"Invalid --max_batches value '{args.max_batches}'. Use a non-negative integer, 'all', or 'none'."
            ) from e
        if args.max_batches < 0:
            raise ValueError(f"--max_batches must be >= 0, got {args.max_batches}")
    
    # Validate Savitzky-Golay parameters
    if args.sg_window_length % 2 == 0:
        raise ValueError(f"sg_window_length must be odd, got {args.sg_window_length}")
    if args.sg_polyorder >= args.sg_window_length:
        raise ValueError(f"sg_polyorder ({args.sg_polyorder}) must be less than sg_window_length ({args.sg_window_length})")

    # Parse noisy GT alphas
    noisy_gt_alpha_tokens = [tok.strip() for tok in str(args.noisy_gt_noise_alphas).split(',') if tok.strip() != ""]
    parsed_noisy_gt_alphas = []
    for tok in noisy_gt_alpha_tokens:
        try:
            alpha = float(tok)
        except ValueError as e:
            raise ValueError(
                f"Invalid value '{tok}' in --noisy_gt_noise_alphas. "
                "Use a comma-separated list of non-negative floats."
            ) from e
        if alpha < 0.0:
            raise ValueError(f"Noise alpha must be non-negative, got {alpha}")
        parsed_noisy_gt_alphas.append(alpha)
    if bool(args.evaluate_noisy_gt_reference) and len(parsed_noisy_gt_alphas) == 0:
        raise ValueError("evaluate_noisy_gt_reference=1 requires at least one value in --noisy_gt_noise_alphas.")
    args.noisy_gt_noise_alphas = parsed_noisy_gt_alphas

    if bool(args.evaluate_latent_alignment) and args.reorder_method != 'smallest_consecutive_distance':
        raise ValueError(
            "For evaluate_latent_alignment=1, reorder_method must be 'smallest_consecutive_distance'. "
            "Use --use_hungarian_for_correlation=1 for supervised global-ID matching."
        )
    if bool(args.evaluate_latent_alignment_nonlinear) and args.reorder_method != 'smallest_consecutive_distance':
        raise ValueError(
            "For evaluate_latent_alignment_nonlinear=1, reorder_method must be 'smallest_consecutive_distance'."
        )
    if bool(args.evaluate_latent_alignment_nonlinear) and bool(args.evaluate_latent_alignment):
        raise ValueError(
            "evaluate_latent_alignment_nonlinear=1 cannot be combined with evaluate_latent_alignment=1. "
            "Choose one evaluation mode per run."
        )
    if bool(args.evaluate_latent_alignment_nonlinear) and bool(args.evaluate_noisy_gt_reference):
        raise ValueError(
            "evaluate_latent_alignment_nonlinear=1 cannot be combined with evaluate_noisy_gt_reference=1."
        )
    if bool(args.evaluate_latent_alignment_nonlinear) and bool(args.convert_to_ghnn):
        raise ValueError(
            "evaluate_latent_alignment_nonlinear=1 cannot be combined with convert_to_ghnn=1."
        )
    if bool(args.evaluate_latent_alignment_nonlinear) and args.extraction_method != 'latent':
        raise ValueError(
            "evaluate_latent_alignment_nonlinear=1 requires --extraction_method latent."
        )
    if bool(args.evaluate_latent_alignment_nonlinear) and args.latent_position_variant != 'both':
        raise ValueError(
            "evaluate_latent_alignment_nonlinear=1 requires --latent_position_variant both."
        )
    if bool(args.evaluate_latent_alignment_nonlinear) and not bool(args.extract_coordinates):
        print(
            "[INFO] evaluate_latent_alignment_nonlinear=1 ignores --extract_coordinates "
            "and forces coordinate extraction internally."
        )
    if bool(args.evaluate_latent_alignment_nonlinear) and int(args.nonlinear_probe_epochs) < 1:
        raise ValueError(
            f"evaluate_latent_alignment_nonlinear=1 requires --nonlinear_probe_epochs >= 1, "
            f"got {args.nonlinear_probe_epochs}."
        )
    if (
        args.latent_position_variant == 'both'
        and bool(args.visualize_trajectories)
        and args.extraction_method != 'latent'
    ):
        raise ValueError(
            "latent_position_variant='both' with visualize_trajectories=1 requires --extraction_method latent."
        )
    if args.latent_position_variant == 'both' and bool(args.convert_to_ghnn):
        raise ValueError(
            "latent_position_variant='both' is not supported with --convert_to_ghnn=1. "
            "Use 'nominal' or 'recentered'."
        )
    if args.prob_encoder_route != 'none':
        if args.prob_encoder_checkpoint is None:
            raise ValueError("--prob_encoder_checkpoint is required when --prob_encoder_route is not 'none'.")
        if args.extraction_method != 'latent':
            raise ValueError("--prob_encoder_route requires --extraction_method latent.")
        if args.latent_position_variant != 'nominal':
            raise ValueError(
                "Probabilistic encoder export emits the selected C1-family mean as the nominal latent stream. "
                "Use --latent_position_variant nominal."
            )
    if args.prob_encoder_route == 'none' and args.prob_encoder_checkpoint is not None:
        raise ValueError("--prob_encoder_checkpoint was provided but --prob_encoder_route is 'none'.")
    
    # Load configuration
    config_path = os.path.join(args.checkpoint, 'hparams.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # print loaded configuration for verification
    print("\n" + "="*80)
    print("LOADED CONFIGURATION")
    print(f"Loaded config from {config_path}")
    print(f"Dataset: {config['ds']}, Root: {config['root']}")
    print(f"Training sequence length: {args.ghnn_train_seq_len}")
    print(f"Eval sequence length (raw arg): {args.eval_seq_len if args.eval_seq_len is not None else 'auto-by-mode'}")
    print("\n" + "="*80)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
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
        filtering_heuristic=config.get('filtering_heuristic', 'variance')
    ).to(device)
    
    # Load checkpoint
    checkpoint_stem = 'twobody_ddlp_minimal_off_cnt'
    if 'BIG' in args.checkpoint.upper():
        checkpoint_stem = f"{checkpoint_stem}_BIG"

    if args.checkpoint_name is None:
        checkpoint_path = os.path.join(args.checkpoint, 'saves', f"{checkpoint_stem}.pth")
    else:
        if args.checkpoint_name.lower().find('best') != -1:
            checkpoint_path = os.path.join(
                args.checkpoint, 'saves', f"{checkpoint_stem}_best_lpips.pth"
            )
        else:
            checkpoint_path = os.path.join(args.checkpoint, 'saves', args.checkpoint_name)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            elif 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Assume the checkpoint is the state dict itself
                model.load_state_dict(checkpoint)
        else:
            # Checkpoint is directly the state dict
            model.load_state_dict(checkpoint)
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        if isinstance(checkpoint, dict):
            print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    else:
        raise FileNotFoundError(f"Warning: No checkpoint found at {checkpoint_path}")

    prob_encoder_frozen_model = None
    if args.prob_encoder_route != 'none':
        if args.prob_encoder_route in {'c1', 'c1-dyn-attrs'}:
            prob_encoder_frozen_model = deepcopy(model).to(device)
            prob_encoder_frozen_model.eval()
            for p in prob_encoder_frozen_model.parameters():
                p.requires_grad = False

        prob_payload = _load_checkpoint_into_model(
            model,
            args.prob_encoder_checkpoint,
            device=device,
            strict=True,
        )
        model.eval()
        print(
            f"Loaded probabilistic encoder ({args.prob_encoder_route}) "
            f"from {args.prob_encoder_checkpoint}"
        )
        if isinstance(prob_payload, dict):
            print(f"Probabilistic encoder epoch: {prob_payload.get('epoch', 'unknown')}")
        
    # Set unified result directories under checkpoint/extraction_evaluation/{checkpoint_name}
    checkpoint_name_tag = args.checkpoint_name if args.checkpoint_name is not None else "default"
    extraction_eval_root = os.path.join(args.checkpoint, "extraction_evaluation", checkpoint_name_tag)

    if bool(args.evaluate_latent_alignment_nonlinear):
        visualizations_dirname = "visualizations_comparison"
        latent_eval_dirname = "latent_alignment_eval_comparison_nonlinear"
    elif args.latent_position_variant == 'recentered':
        visualizations_dirname = "visualizations_recentred"
        latent_eval_dirname = "latent_alignment_eval_recentred"
    elif args.latent_position_variant == 'both':
        visualizations_dirname = "visualizations_comparison"
        latent_eval_dirname = "latent_alignment_eval_comparison"
    else:
        visualizations_dirname = "visualizations"
        latent_eval_dirname = "latent_alignment_eval"

    save_dir = os.path.join(extraction_eval_root, visualizations_dirname)
    latent_eval_save_dir = os.path.join(extraction_eval_root, latent_eval_dirname)

    # Only materialize output directories that this run can actually populate.
    needs_visualization_dir = bool(args.visualize_trajectories)
    needs_latent_eval_dir = bool(
        args.evaluate_latent_alignment
        or args.evaluate_latent_alignment_nonlinear
        or args.evaluate_noisy_gt_reference
        or args.write_prob_encoder_alignment_metrics
    )
    active_save_dir = save_dir if needs_visualization_dir else None
    active_latent_eval_save_dir = latent_eval_save_dir if needs_latent_eval_dir else None
    direct_prob_encoder_alignment_run = bool(
        args.evaluate_latent_alignment
        and not args.convert_to_ghnn
        and not args.evaluate_latent_alignment_nonlinear
        and args.prob_encoder_route != 'none'
    )
    if direct_prob_encoder_alignment_run and active_latent_eval_save_dir is not None:
        active_latent_eval_save_dir = os.path.join(
            active_latent_eval_save_dir,
            _prob_encoder_output_dirname(
                args.prob_encoder_route,
                output_tag=args.prob_encoder_output_tag,
            ),
        )

    filtering_report_path = None if (bool(args.evaluate_latent_alignment) or bool(args.evaluate_latent_alignment_nonlinear)) else os.path.join(extraction_eval_root, "filtering_report.json")
    if active_save_dir is not None:
        os.makedirs(active_save_dir, exist_ok=True)
    if active_latent_eval_save_dir is not None:
        os.makedirs(active_latent_eval_save_dir, exist_ok=True)
    if filtering_report_path is not None and os.path.exists(filtering_report_path):
        os.remove(filtering_report_path)
    print(f"Results root: {extraction_eval_root}")
    if active_save_dir is not None:
        print(f"Visualization outputs: {active_save_dir}")
    else:
        print("Visualization outputs: disabled for this run")
    if active_latent_eval_save_dir is not None:
        print(f"Latent-eval outputs: {active_latent_eval_save_dir}")
    else:
        print("Latent-eval outputs: disabled for this run")
    if filtering_report_path is not None:
        print(f"Filtering report: {filtering_report_path}")
    else:
        print("Filtering report: disabled for latent-alignment evaluation")

    # Evaluate or convert to GHNN format
    if bool(args.evaluate_latent_alignment_nonlinear):
        print("\n" + "="*80)
        print("RUNNING NONLINEAR LATENT ALIGNMENT EVALUATION")
        print("="*80)
        evaluate_latent_alignment_nonlinear_joint(
            model=model,
            config=config,
            device=device,
            batch_size=args.batch_size,
            max_batches=args.max_batches,
            save_dir=active_save_dir,
            latent_eval_save_dir=active_latent_eval_save_dir,
            use_hungarian_for_correlation=bool(args.use_hungarian_for_correlation),
            reorder_method=args.reorder_method,
            extraction_method=args.extraction_method,
            latent_position_variant=args.latent_position_variant,
            latent_recenter_source=args.latent_recenter_source,
            latent_recenter_nms_source=args.latent_recenter_nms_source,
            latent_recenter_eps=args.latent_recenter_eps,
            train_seq_len=args.nonlinear_train_seq_len,
            valid_seq_len=args.nonlinear_valid_seq_len,
            probe_hidden_dim=args.nonlinear_probe_hidden_dim,
            probe_num_hidden_layers=args.nonlinear_probe_num_hidden_layers,
            probe_epochs=args.nonlinear_probe_epochs,
            probe_batch_size=args.nonlinear_probe_batch_size,
            probe_lr=args.nonlinear_probe_lr,
            probe_weight_decay=args.nonlinear_probe_weight_decay,
            probe_seed=args.nonlinear_probe_seed,
        )
        print("\n✅ Nonlinear latent alignment evaluation complete!")
    elif args.convert_to_ghnn:
        print("\n" + "="*80)
        print("CONVERTING TO GHNN FORMAT")
        print("="*80)
        
        # Determine sequence lengths
        eval_seq_len = args.eval_seq_len if args.eval_seq_len else config['animation_horizon']
        train_seq_len = args.ghnn_train_seq_len if args.ghnn_train_seq_len else 60

        ghnn_output_dir = args.ghnn_output_dir
        ghnn_dataset_name = args.ghnn_dataset_name
        if ghnn_output_dir is None:
            # Store all extracted trajectory datasets under a single checkpoint-scoped folder.
            # Distinguish nominal/recentered variants via dataset_name suffixes.
            ghnn_output_dir = os.path.join(extraction_eval_root, "extracted_datasets")
        if args.latent_position_variant == 'recentered' and 'recentered' not in ghnn_dataset_name:
            ghnn_dataset_name = f"{ghnn_dataset_name}_recentered"

        print(f"GHNN output dir: {ghnn_output_dir}")
        print(f"GHNN dataset name: {ghnn_dataset_name}")
        
        _ = save_trajectories_to_GHNN_format(
            model=model,
            config=config,
            device=device,
            output_dir=ghnn_output_dir,
            dataset_name=ghnn_dataset_name,
            train_seq_len=train_seq_len,
            eval_seq_len=eval_seq_len,
            batch_size=args.batch_size,
            step_size=args.ghnn_step_size,
            dt=args.ghnn_dt,
            masses=None,  # Will use default [1.0, 1.0, ...]
            validation_share=0.025,
            test_share=0.025,
            seed=0,
            sg_window_length=args.sg_window_length,
            sg_polyorder=args.sg_polyorder,
            smooth_momentum=bool(args.smooth_momentum),
            extraction_method=args.extraction_method,
            filtering_report_path=filtering_report_path,
            latent_position_variant=args.latent_position_variant,
            latent_recenter_source=args.latent_recenter_source,
            latent_recenter_nms_source=args.latent_recenter_nms_source,
            latent_recenter_eps=args.latent_recenter_eps,
            prob_encoder_route=args.prob_encoder_route,
            prob_encoder_frozen_model=prob_encoder_frozen_model,
            write_prob_encoder_alignment_metrics=bool(args.write_prob_encoder_alignment_metrics),
            alignment_split=args.alignment_split,
            latent_eval_save_dir=active_latent_eval_save_dir,
            use_hungarian_for_correlation=bool(args.use_hungarian_for_correlation),
        )
        
        print("\n✅ GHNN conversion completed successfully!")    
    else:
        eval_seq_len = args.eval_seq_len if args.eval_seq_len is not None else (60 if args.mode == 'train' else 360)
        print(f"\nCollecting video data from {args.mode} set...")
        print(f"Using eval_seq_len={eval_seq_len} for mode={args.mode}")
        video_to_trajectory(
            model, config, device=device, mode=args.mode,
            batch_size=args.batch_size, max_batches=args.max_batches,
            eval_seq_len=eval_seq_len, save_dir=active_save_dir,
            latent_eval_save_dir=active_latent_eval_save_dir,
            visualize_trajectories=bool(args.visualize_trajectories),
            extract_coordinates=bool(args.extract_coordinates), 
            evaluate_latent_alignment=bool(args.evaluate_latent_alignment),
            evaluate_noisy_gt_reference=bool(args.evaluate_noisy_gt_reference),
            noisy_gt_noise_mode=args.noisy_gt_noise_mode,
            noisy_gt_noise_alphas=args.noisy_gt_noise_alphas,
            noisy_gt_noise_seed=args.noisy_gt_noise_seed,
            use_hungarian_for_correlation=bool(args.use_hungarian_for_correlation),
            reorder_method=args.reorder_method,
            sg_window_length=args.sg_window_length,
            sg_polyorder=args.sg_polyorder,
            extraction_method=args.extraction_method,
            filtering_report_path=filtering_report_path,
            latent_position_variant=args.latent_position_variant,
            latent_recenter_source=args.latent_recenter_source,
            latent_recenter_nms_source=args.latent_recenter_nms_source,
            latent_recenter_eps=args.latent_recenter_eps,
            prob_encoder_route=args.prob_encoder_route,
            prob_encoder_frozen_model=prob_encoder_frozen_model,
            raw_physical_npz_root=args.raw_physical_npz_root,
        )

        print("\n✅ Visualization generation complete!")
#####################################################################################

if __name__ == '__main__':
    main()
