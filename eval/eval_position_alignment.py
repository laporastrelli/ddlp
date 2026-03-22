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
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

# Add parent directory to path for local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# torch
import torch
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import r2_score

# datasets
from datasets.get_dataset import get_video_dataset

# models
from models import ObjectDynamicsDLP

import sys


def normalize_positions_to_model_range(positions, image_size=64, kp_range=(-1, 1)):
    """
    Convert ground-truth positions from pixel coordinates to model's keypoint range.
    
    Args:
        positions: (T, N, 2) array in pixel coordinates [0, image_size]
        image_size: Image dimension (assumes square images)
        kp_range: Model's keypoint coordinate range (default [-1, 1])
    
    Returns:
        normalized_pos: (T, N, 2) array in kp_range coordinates
    """
    # Convert from [0, image_size] to [0, 1]
    normalized = positions / image_size
    
    # Convert from [0, 1] to kp_range
    kp_min, kp_max = kp_range
    normalized = normalized * (kp_max - kp_min) + kp_min
    
    return normalized


def evaluate_position_alignment(
    model, config, device=torch.device('cpu'), 
    mode='valid', batch_size=32, max_batches=None,
    use_obj_on_threshold=0.5, num_particles=None,
    eval_seq_len=None, selection='feature_diff'
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
        max_batches: Maximum number of batches to evaluate (None = all)
        use_obj_on_threshold: Threshold for filtering particles by obj_on score (ignored if num_particles is set)
        num_particles: If set, select this many top-k particles by lowest variance (DDLP standard approach)
        eval_seq_len: Number of timesteps to encode from sequence (None = use full sequence from dataset)
        selection: Method for selecting particles ('obj_on_threshold' or other criteria)
    Returns:
        results: Dictionary containing video data for visualization
    """
    model.eval()
    
    # Load dataset - mode selects which split (train/valid/test) to use
    ds = config['ds']
    image_size = config['image_size']
    root = config['root']
    kp_range = model.kp_range
    
    print(f"\nEvaluation mode: FULL MODEL (with temporal tracking)")
    
    # Determine sequence length for evaluation
    if eval_seq_len is None:
        eval_seq_len = config['timestep_horizon']  # Use training horizon by default
    
    # VIDEO dataset for temporal tracking evaluation
    print(f"Using video dataset with temporal tracking (seq_len={eval_seq_len})")
    dataset = get_video_dataset(ds, root, seq_len=eval_seq_len, 
                               mode=mode, image_size=image_size)
    dataloader = DataLoader(
        dataset, shuffle=False, batch_size=batch_size, 
        num_workers=4, drop_last=False
    )
    
    # Store video data for visualization
    video_data_for_vis = []
    
    pbar = tqdm(dataloader, desc=f"Evaluating {mode} set")
    for batch_idx, batch in enumerate(pbar):
        if max_batches is not None and batch_idx >= max_batches:
            break
        
        # Unpack batch
        x = batch[0].to(device)  # Images: [B, T, C, H, W]
        gt_positions = batch[1]  # Ground-truth positions
        
        # Forward pass with temporal tracking
        with torch.no_grad():
            B = x.shape[0]
            T = x.shape[1]
            timestep_horizon = config['timestep_horizon']
            
            # Check if autoregressive encoding is needed
            if T > timestep_horizon:
                # AUTOREGRESSIVE ENCODING: Process in chunks of timestep_horizon
                print(f"Autoregressive encoding: {T} frames in chunks of {timestep_horizon}")
                
                all_outputs = []
                for chunk_start in range(0, T, timestep_horizon):
                    chunk_end = min(chunk_start + timestep_horizon, T)
                    x_chunk = x[:, chunk_start:chunk_end]  # (B, chunk_len, C, H, W)
                    
                    # Use previous chunk's last frame as conditioning if available
                    if chunk_start == 0:
                        x_prior_chunk = x_chunk
                    else:
                        # Use last frame from previous chunk as prior
                        x_prior_chunk = torch.cat([x[:, chunk_start-1:chunk_start], x_chunk[:, :-1]], dim=1)
                    
                    # Ensure tensors are contiguous before passing to model (torch.cat can create non-contiguous tensors)
                    chunk_output = model(x_chunk.contiguous(), x_prior=x_prior_chunk.contiguous(), 
                                       deterministic=True, forward_dyn=True)
                    all_outputs.append(chunk_output)
                
                # Concatenate outputs from all chunks
                model_output = {
                    'z': torch.cat([out['z'] for out in all_outputs], dim=0),
                    'mu_offset': torch.cat([out['mu_offset'] for out in all_outputs], dim=0),
                    'z_base': torch.cat([out['z_base'] for out in all_outputs], dim=0),
                    'logvar_offset': torch.cat([out['logvar_offset'] for out in all_outputs], dim=0),
                    'obj_on': torch.cat([out['obj_on'] for out in all_outputs], dim=0), 
                    'z_features': torch.cat([out['z_features'] for out in all_outputs], dim=0),
                }
            else:
                # STANDARD ENCODING: Single forward pass for sequences <= timestep_horizon
                model_output = model(x, x_prior=x, deterministic=True, forward_dyn=True)
            
            # Extract inferred positions from model output
            # Model outputs are flattened [B*T, N_kp, ...]
            # Reshape to [B, T, N_kp, ...] to evaluate ALL timesteps
            B = x.shape[0]
            T = x.shape[1]
            
            # Reshape from [B*T, N_kp, ...] to [B, T, N_kp, ...] -> flatten to [B*T, N_kp, ...]
            mu_offset = model_output['mu_offset'].view(B, T, -1, 2)
            z_base = model_output['z_base'].view(B, T, -1, 2)
            logvar_offset = model_output['logvar_offset'].view(B, T, -1, 2)
            obj_on = model_output['obj_on'].view(B, T, -1)
            z_features = model_output['z_features'].view(B, T, -1, model_output['z_features'].shape[-1])
            
            # Total position: base + offset
            pred_positions = (z_base + mu_offset).cpu().numpy()  # (B, T, N_kp, 2)
            logvar = logvar_offset.cpu().numpy()  # (B, T, N_kp, 2)
            obj_on = obj_on.cpu().numpy()  # (B, T, N_kp)
            z_features_np = z_features.cpu().numpy()  # (B, T, N_kp, learned_feature_dim)
        
        # Normalize ground-truth positions to model's coordinate range
        # GT has shape (B, T, N_objects, 2) -> flatten to (B*T, N_objects, 2)
        B = x.shape[0]
        T = x.shape[1]
        gt_positions_np = gt_positions.view(B, T, gt_positions.shape[-2], 2).cpu().numpy()  # (B, T, N_objects, 2)
        gt_positions_normalized = normalize_positions_to_model_range(
            gt_positions_np, image_size=image_size, kp_range=kp_range
        )
        
        # IDENTITY CHANGE DETECTION (before Hungarian assignment)
        # Analyze identity consistency for each video in the batch
        for video_idx in range(B):

            # Extract this video's predictions and GT across all timesteps
            pred_video = pred_positions[video_idx]  # (T, N_kp, 2)
            gt_video   = gt_positions_normalized[video_idx]  # (T, N_gt, 2)

            if selection == 'obj_on_threshold':
                # Select active particles using the same method as later
                pred_video_active = []
                for t in range(T):
                    if num_particles is not None:
                        logvar_sum = np.sum(logvar[video_idx, t], axis=-1) * obj_on[video_idx, t]
                        topk_indices = np.argsort(logvar_sum)[:num_particles]
                        pred_video_active.append(pred_video[t, topk_indices])
                    else:
                        active_mask = obj_on[video_idx, t] > use_obj_on_threshold
                        pred_video_active.append(pred_video[t, active_mask])
                pred_video_active = np.array(pred_video_active)  # (T, N_active, 2)
                
                # Extract z_features for this video
                z_features_video = z_features_np[video_idx]  # (T, N_kp, learned_feature_dim)
                
                # Select active particles' features
                z_features_video_active = []
                for t in range(T):
                    if num_particles is not None:
                        logvar_sum = np.sum(logvar[video_idx, t], axis=-1) * obj_on[video_idx, t]
                        topk_indices = np.argsort(logvar_sum)[:num_particles]
                        z_features_video_active.append(z_features_video[t, topk_indices])
                    else:
                        active_mask = obj_on[video_idx, t] > use_obj_on_threshold
                        z_features_video_active.append(z_features_video[t, active_mask])
                z_features_video_active = np.array(z_features_video_active)  # (T, N_active, learned_feature_dim)

            else:
                # Use particles that belong with highest difference in learned features across time
                z_features_video = z_features_np[video_idx]
                pred_video_active = []
                z_features_video_active = []
                dist_matrix = np.zeros((T, pred_video.shape[1], pred_video.shape[1]))
                for t in range(T):
                    for i in range(pred_video.shape[1]):
                        for j in range(pred_video.shape[1]):
                            dist_matrix[t, i, j] = np.linalg.norm(z_features_video[t, i] - z_features_video[t, j])
                    # get highest entry in learned dist_matrix for each time step
                    largest_dist_indices = np.unravel_index(np.argmax(dist_matrix[t]), dist_matrix[t].shape)
                    pred_video_active.append(pred_video[t, largest_dist_indices])
                    z_features_video_active.append(z_features_video[t, largest_dist_indices])
                pred_video_active = np.array(pred_video_active)  # (T, N_active, 2)
                z_features_video_active = np.array(z_features_video_active)  # (T, N_active, learned_feature_dim)
            
            # Store video data for visualization
            video_data_for_vis.append({
                'images': x[video_idx].cpu().numpy(),  # (T, C, H, W)
                'pred_positions': pred_video_active,  # (T, N_active, 2)
                'gt_positions': gt_video,  # (T, N_gt, 2)
                'z_features': z_features_video_active,  # (T, N_active, learned_feature_dim)
                'batch_idx': batch_idx,
                'video_idx': video_idx
            })
    
    # Collect results
    results = {
        'video_data': video_data_for_vis
    }
    
    return results

########################################################################################################
def detect_particle_reordering_from_features(z_features_sequence):
    """
    Detect particle re-ordering events across frames using learned features.
    
    For 2-particle case:
    - Frame 0 establishes reference ordering [0, 1]
    - Each subsequent frame: find nearest match to reference based on feature similarity
    - Track when assignment changes (particle swap events)
    
    Args:
        z_features_sequence: (T, N_particles, learned_feature_dim) learned features across time
    
    Returns:
        Dictionary with:
        - reordering_frames: list of frame indices where re-ordering occurred
        - particle_assignments: (T, N_particles) array showing which reference each particle matches
        - num_reorderings: total number of re-ordering events
    """
    T, N_particles, feature_dim = z_features_sequence.shape
    
    # Initialize reference features from first frame
    reference_features = z_features_sequence[0].copy()  # (N_particles, feature_dim)
    
    # Track assignments over time
    particle_assignments = np.zeros((T, N_particles), dtype=int)
    particle_assignments[0] = np.arange(N_particles)  # Initial ordering: [0, 1, ..., N-1]
        
    # Detect re-orderings
    reordering_frames = []
    
    for t in range(1, T):
        current_features = z_features_sequence[t]  # (N_particles, feature_dim)
        
        # Compute distance matrix: current particles vs reference
        dist_matrix = np.zeros((N_particles, N_particles))
        for i in range(N_particles):
            for j in range(N_particles):
                # Euclidean distance in feature space
                dist_matrix[i, j] = np.linalg.norm(current_features[i] - reference_features[j])
        
        # Find nearest reference for each current particle
        new_assignment = np.argmin(dist_matrix, axis=1)  # (N_particles,)
        particle_assignments[t] = new_assignment
        
        # Check if ordering changed from previous frame
        if not np.array_equal(new_assignment, particle_assignments[t-1]):
            reordering_frames.append(t)
    
    return {
        'reordering_frames': reordering_frames,
        'particle_assignments': particle_assignments,
        'num_reorderings': len(reordering_frames)
    }


def create_particle_distance_plots(video_data_list, save_dir='./', mode='valid'):
    """
    Generate plots showing inter-particle distance over time with re-ordering markers.
    
    For each video, plots:
    - Euclidean distance between particle positions (z_base + mu_offset)
    - Vertical lines marking detected re-ordering events from learned features
    
    Args:
        video_data_list: List of video data dictionaries from evaluation
        save_dir: Directory to save plots
        mode: Dataset split name for filename
    """
    num_videos = len(video_data_list)
    print(f"\n📊 Generating {num_videos} particle distance plots with re-ordering detection...")
    
    for video_idx, video_data in enumerate(video_data_list):
        try:
            pred_positions = video_data['pred_positions']  # (T, N_particles, 2)
            z_features = video_data['z_features']  # (T, N_particles, learned_feature_dim)
            
            T, N_particles = pred_positions.shape[0], pred_positions.shape[1]
            
            if N_particles != 2:
                print(f"  Skipping video {video_idx}: Expected 2 particles, got {N_particles}")
                continue
            
            # Compute inter-particle distance over time
            particle_distances = np.zeros(T)
            for t in range(T):
                # Euclidean distance between particle 0 and particle 1
                particle_distances[t] = np.linalg.norm(pred_positions[t, 0] - pred_positions[t, 1])
            
            # Detect re-ordering events
            reordering_info = detect_particle_reordering_from_features(z_features)
            reordering_frames = reordering_info['reordering_frames']
            num_reorderings = reordering_info['num_reorderings']
            
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot distance time series
            time_steps = np.arange(T)
            ax.plot(time_steps, particle_distances, 'b-', linewidth=2, label='Inter-particle distance')
            
            # Mark re-ordering events with vertical lines
            for reorder_frame in reordering_frames:
                ax.axvline(x=reorder_frame, color='r', linestyle='--', linewidth=1.5, alpha=0.7)
            
            # Add legend entry for re-ordering markers
            if num_reorderings > 0:
                ax.plot([], [], 'r--', linewidth=1.5, label=f'Re-ordering events (n={num_reorderings})')
            
            # Formatting
            ax.set_xlabel('Frame', fontsize=13, fontweight='bold')
            ax.set_ylabel('Euclidean Distance', fontsize=13, fontweight='bold')
            ax.set_title(f'Inter-Particle Distance Over Time\n(Video {video_idx}, {num_reorderings} re-orderings detected)', 
                        fontsize=14, fontweight='bold', pad=15)
            ax.grid(alpha=0.3, linewidth=0.8)
            ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
            
            # Save plot
            save_path = os.path.join(save_dir, f'particle_distance_{mode}_{video_idx:03d}.png')
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            if (video_idx + 1) % 5 == 0:
                print(f"  Generated {video_idx + 1}/{num_videos} distance plots")
        
        except Exception as e:
            print(f"  Error generating distance plot for video {video_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"✅ Completed particle distance plot generation")


def create_trajectory_videos(video_data_list, kp_range, save_dir='./', mode='valid'):
    """
    Generate videos showing low-dimensional trajectories from evaluated video data.
    
    Args:
        video_data_list: List of video data dictionaries from evaluation
        kp_range: Model's keypoint coordinate range
        save_dir: Directory to save videos
        mode: Dataset split name for filename
    """
    num_videos = len(video_data_list)
    print(f"\n🎬 Generating {num_videos} trajectory visualization videos...")
    print(f"   [DEBUG] Received {num_videos} video data entries")
    
    for video_idx, video_data in enumerate(video_data_list):
        try:
            pred_positions_active = video_data['pred_positions']  # (T, N_active, 2)
            gt_positions_normalized = video_data['gt_positions']  # (T, N_gt, 2)
            
            print(f"   [DEBUG] Video {video_idx}: pred shape={pred_positions_active.shape}, gt shape={gt_positions_normalized.shape}")
            print(f"   [DEBUG] Video {video_idx}: pred range=[{pred_positions_active.min():.4f}, {pred_positions_active.max():.4f}]")
            print(f"   [DEBUG] Video {video_idx}: gt range=[{gt_positions_normalized.min():.4f}, {gt_positions_normalized.max():.4f}]")
            
            # Create trajectory visualization
            save_path = os.path.join(save_dir, f'trajectory_{mode}_{video_idx:03d}.gif')
            _create_trajectory_video(
                pred_positions_active, gt_positions_normalized,
                save_path, kp_range
            )
            
            if (video_idx + 1) % 5 == 0:
                print(f"  Generated {video_idx + 1}/{num_videos} trajectory videos")
        
        except Exception as e:
            print(f"  Error generating trajectory video {video_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"✅ Completed trajectory video generation")


def _create_trajectory_video(pred_positions, gt_positions, save_path):
    """
    Create a single trajectory visualization video showing predicted vs GT trajectories.
    
    Args:
        pred_positions: (T, N_pred, 2) predicted positions in model range
        gt_positions: (T, N_gt, 2) ground-truth positions in model range
        save_path: Path to save the GIF
    """
    T = pred_positions.shape[0]
    N_pred = pred_positions.shape[1]
    N_gt = gt_positions.shape[1]
    
    print(f"   [DEBUG] Creating trajectory video: T={T}, N_pred={N_pred}, N_gt={N_gt}")
    
    # Debug: Print statistics for each particle
    for i in range(N_pred):
        pred_particle = pred_positions[:, i, :]
        print(f"   [DEBUG] Pred particle {i}: mean={pred_particle.mean(axis=0)}, std={pred_particle.std(axis=0)}, range=({pred_particle.min():.4f}, {pred_particle.max():.4f})")
    
    for i in range(N_gt):
        gt_particle = gt_positions[:, i, :]
        print(f"   [DEBUG] GT particle {i}: mean={gt_particle.mean(axis=0)}, std={gt_particle.std(axis=0)}, range=({gt_particle.min():.4f}, {gt_particle.max():.4f})")
    
    # Compute global ranges for consistent scaling with equal aspect ratio
    all_pos = np.concatenate([pred_positions.reshape(-1, 2), gt_positions.reshape(-1, 2)], axis=0)
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
    
    print(f"   [DEBUG] Plot ranges: x={x_lim}, y={y_lim}, max_range={max_range:.4f}")
    
    # Prepare figure - single panel showing only trajectories
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
    colors = plt.cm.Set1(np.linspace(0, 1, max(N_pred, N_gt)))
    
    for t in range(T):
        ax.clear()
        
        # Re-apply fixed settings after clear
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_xlabel('X Position', fontsize=13, fontweight='bold')
        ax.set_ylabel('Y Position', fontsize=13, fontweight='bold')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(alpha=0.4, linewidth=0.8)
        
        # Plot ground-truth trajectories
        for obj_idx in range(N_gt):
            gt_traj = gt_positions[:t+1, obj_idx, :]  # (t+1, 2)
            ax.plot(gt_traj[:, 0], gt_traj[:, 1], 
                   '-', color=colors[obj_idx], alpha=0.7, linewidth=3,
                   label=f'GT {obj_idx+1}')
            # Current position
            ax.scatter(gt_traj[-1, 0], gt_traj[-1, 1], 
                      c=[colors[obj_idx]], s=350, marker='o', alpha=0.9,
                      edgecolors='black', linewidths=3, zorder=10)
        
        # Plot predicted trajectories
        for kp_idx in range(N_pred):
            pred_traj = pred_positions[:t+1, kp_idx, :]  # (t+1, 2)
            if t > 0:
                ax.plot(pred_traj[:, 0], pred_traj[:, 1],
                       '--', color=colors[kp_idx], alpha=0.7, linewidth=3,
                       label=f'Pred {kp_idx+1}')
            # Current position
            ax.scatter(pred_traj[-1, 0], pred_traj[-1, 1],
                      c=[colors[kp_idx]], s=350, marker='x', alpha=0.9,
                      linewidths=5, zorder=10)
        
        ax.set_title(f'Trajectories: Frame {t+1}/{T}\n(Circle=GT, X=Pred)', 
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


def create_pred_only_trajectory_videos(video_data_list, kp_range, save_dir='./', mode='valid'):
    """
    Generate videos showing ONLY predicted trajectories in their original scale.
    No ground-truth is displayed, and scaling is based only on predicted positions.
    
    Args:
        video_data_list: List of video data dictionaries from evaluation
        kp_range: Model's keypoint coordinate range
        save_dir: Directory to save videos
        mode: Dataset split name for filename
    """
    num_videos = len(video_data_list)
    print(f"\n🎯 Generating {num_videos} prediction-only trajectory videos...")
    print(f"   [DEBUG] Received {num_videos} video data entries")
    
    for video_idx, video_data in enumerate(video_data_list):
        try:
            pred_positions_active = video_data['pred_positions']  # (T, N_active, 2)
            
            print(f"   [DEBUG] Video {video_idx}: pred shape={pred_positions_active.shape}")
            print(f"   [DEBUG] Video {video_idx}: pred range=[{pred_positions_active.min():.4f}, {pred_positions_active.max():.4f}]")
            
            # Create prediction-only trajectory visualization
            save_path = os.path.join(save_dir, f'pred_only_trajectory_{mode}_{video_idx:03d}.gif')
            _create_pred_only_trajectory_video(
                pred_positions_active,
                save_path, kp_range
            )
            
            if (video_idx + 1) % 5 == 0:
                print(f"  Generated {video_idx + 1}/{num_videos} prediction-only trajectory videos")
        
        except Exception as e:
            print(f"  Error generating prediction-only trajectory video {video_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"✅ Completed prediction-only trajectory video generation")


def _create_pred_only_trajectory_video(pred_positions, save_path, kp_range):
    """
    Create a single trajectory visualization video showing ONLY predicted trajectories.
    
    Args:
        pred_positions: (T, N_pred, 2) predicted positions in model range
        save_path: Path to save the GIF
        kp_range: Model's keypoint coordinate range
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


def create_mask_overlay_videos(video_data_list, model, config, device, 
                               iou_thresh=0.5, save_dir='./', mode='valid'):
    """
    Generate videos with bounding boxes on original frames from evaluated videos.
    Uses position+scale based bounding boxes with NMS filtering.
    
    Args:
        video_data_list: List of video data dictionaries from evaluation
        model: Trained DDLP model
        config: Configuration dictionary
        device: torch device
        iou_thresh: IoU threshold for NMS
        save_dir: Directory to save videos
        mode: Dataset split name for filename
    """
    from utils.util_func import plot_bb_on_image_batch_from_z_scale_nms
    
    model.eval()
    num_videos = len(video_data_list)
    
    print(f"\n🎭 Generating {num_videos} mask overlay videos...")
    print(f"   [DEBUG] Received {num_videos} video data entries")

    low_score_t_idxs_list = []
    
    for video_idx, video_data in enumerate(video_data_list):
        try:
            images = video_data['images']  # (T, C, H, W) numpy array
            T = images.shape[0]
            
            # Convert to torch and add batch dimension
            x = torch.from_numpy(images).to(device)  # (T, C, H, W)
            
            print(f"   [DEBUG] Video {video_idx}: images shape={images.shape}")
            
            # Forward pass through model (temporal tracking)
            with torch.no_grad():
                x_input = x.unsqueeze(0)  # (1, T, C, H, W)
                timestep_horizon = config['timestep_horizon']
                
                # Check if autoregressive encoding is needed (same as evaluation)
                if T > timestep_horizon:
                    all_outputs = []
                    for chunk_start in range(0, T, timestep_horizon):
                        chunk_end = min(chunk_start + timestep_horizon, T)
                        x_chunk = x_input[:, chunk_start:chunk_end]
                        
                        if chunk_start == 0:
                            x_prior_chunk = x_chunk
                        else:
                            x_prior_chunk = torch.cat([x_input[:, chunk_start-1:chunk_start], x_chunk[:, :-1]], dim=1)
                        
                        chunk_output = model(x_chunk.contiguous(), x_prior=x_prior_chunk.contiguous(),
                                           deterministic=True, forward_dyn=True)
                        all_outputs.append(chunk_output)
                    
                    # Concatenate outputs from all chunks
                    model_output = {
                        'mu': torch.cat([out['mu'] for out in all_outputs], dim=0),
                        'mu_offset': torch.cat([out['mu_offset'] for out in all_outputs], dim=0),
                        'z_base': torch.cat([out['z_base'] for out in all_outputs], dim=0),
                        'logvar_offset': torch.cat([out['logvar_offset'] for out in all_outputs], dim=0),
                        'obj_on': torch.cat([out['obj_on'] for out in all_outputs], dim=0),
                        'alpha_masks': torch.cat([out['alpha_masks'] for out in all_outputs], dim=0)
                    }
                else:
                    model_output = model(x_input, x_prior=x_input, deterministic=True, forward_dyn=True)
                
                # Extract latents
                mu_offset = model_output['mu_offset']  # (T, N_kp, 2)
                z_base = model_output['z_base']  # (T, N_kp, 2)
                logvar_offset = model_output['logvar_offset']  # (T, N_kp, 2)
                obj_on = model_output['obj_on']  # (T, N_kp)
                alpha_masks = model_output['alpha_masks']  # (T, N_kp, 1, H, W)
                
                # Compute keypoint positions (same as evaluation)
                kp_positions = z_base + mu_offset  # (T, N_kp, 2)
                
                # Extract scales from masks
                from utils.util_func import get_bb_from_masks
                z_scale_list = []
                for t in range(T):
                    bb_info = get_bb_from_masks(alpha_masks[t], width=images.shape[-1], height=images.shape[-2])
                    z_scale_list.append(bb_info['scales'])  # (N_kp, 2)
                z_scale = torch.stack(z_scale_list, dim=0)  # (T, N_kp, 2)
                
                # Compute bb scores (negative logvar sum, same as training)
                logvar_sum = logvar_offset.sum(-1) * obj_on  # (T, N_kp)
                bb_scores = -1 * logvar_sum
            
            # Create video frames with bounding boxes from positions and scales
            frames = []
            low_score_t_idxs = []
            for t in range(T):
                x_frame = x[t:t+1]  # (1, C, H, W)
                kp_frame = kp_positions[t:t+1]  # (1, N_kp, 2)
                z_scale_frame = z_scale[t:t+1]  # (1, N_kp, 2)
                bb_scores_frame = bb_scores[t:t+1]  # (1, N_kp)

                tmp = len((bb_scores_frame < 2.0).nonzero(as_tuple=True)[1].tolist())
                if tmp > 0:
                    low_score_t_idxs.append(t)
                
                print(f"   [DEBUG] Video {video_idx}, Frame {t}: z_scale range=[{z_scale_frame.min().item():.4f}, {z_scale_frame.max().item():.4f}]")
                if len((z_scale_frame < 0.05).nonzero(as_tuple=True)[1].tolist()) > 0:
                    print(f"   [DEBUG] Video {video_idx}, Frame {t}: Detected very small scales (z_scale < 0.05) for particles: {(z_scale_frame < 0.05).nonzero(as_tuple=True)[1].tolist()}")

                # Apply NMS and overlay bounding boxes using position+scale
                img_with_bb, _ = plot_bb_on_image_batch_from_z_scale_nms(
                    kp_frame, z_scale_frame, x_frame, scores=bb_scores_frame,
                    iou_thresh=iou_thresh, thickness=1, max_imgs=1,
                    hard_thresh=None, scale_normalized=True
                )
                
                # Convert to numpy for video creation
                frame_img = img_with_bb[0].permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
                frame_img = (frame_img * 255).astype(np.uint8)

                # in all frames that are in low_score_t_idxs, add a red border to the image to indicate low confidence
                # using numpy slicing to add a red border of 3 pixels around the image
                if t in low_score_t_idxs:
                    frame_img[:3, :, 0] = 255  # Top border
                    frame_img[-3:, :, 0] = 255  # Bottom border
                    frame_img[:, :3, 0] = 255  # Left border
                    frame_img[:, -3:, 0] = 255  # Right border

                frames.append(frame_img)
            
            # Save as GIF (duration in ms: 100ms = 10fps)
            save_path = os.path.join(save_dir, f'mask_overlay_{mode}_{video_idx:03d}.gif')
            imageio.mimsave(save_path, frames, duration=100, loop=0)
            
            if (video_idx + 1) % 5 == 0:
                print(f"  Generated {video_idx + 1}/{num_videos} mask overlay videos")
            
            if len(low_score_t_idxs) > 0:
                print(f"   [DEBUG] Video {video_idx}: Low confidence frames (bb_score < 2.0) at timesteps: {low_score_t_idxs}")
                low_score_t_idxs_list.append((video_idx, low_score_t_idxs))
        
        except Exception as e:
            print(f"  Error generating mask overlay video {video_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"✅ Completed mask overlay video generation")
    if len(low_score_t_idxs_list) > 0:
        print(f"   [DEBUG] Videos with low confidence frames: {low_score_t_idxs_list}")
    else:
        print(f"   [DEBUG] *** No low confidence frames detected across all videos ***")
########################################################################################################

def main():
    parser = argparse.ArgumentParser(description='Generate visualizations for DDLP model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint directory')
    parser.add_argument('--checkpoint_name', type=str, default=None,
                        help='Specific checkpoint file name (overrides default)')
    parser.add_argument('--mode', type=str, default='valid', choices=['train', 'valid', 'test'],
                       help='Dataset split to evaluate')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for evaluation (forced to 1 for full-length videos to avoid memory errors)')
    parser.add_argument('--max_batches', type=int, default=None,
                       help='Maximum number of batches to evaluate (None = all)')
    parser.add_argument('--obj_on_threshold', type=float, default=0.5,
                       help='Threshold for filtering particles by obj_on score (ignored if --num_particles is set)')
    parser.add_argument('--num_particles', type=int, default=None,
                       help='Number of particles to select by lowest variance (DDLP standard approach). If not set, uses obj_on_threshold.')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use (e.g., cuda:0, cpu)')
    parser.add_argument('--num_videos', type=int, default=3,
                       help='Number of trajectory videos to generate')
    parser.add_argument('--use_hungarian', action='store_true',
                       help='Use Hungarian algorithm for optimal particle-object assignment (default: False)')
    parser.add_argument('--eval_seq_len', type=int, default=None,
                       help='Number of timesteps to encode from sequence (default: use timestep_horizon from config)')
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = os.path.join(args.checkpoint, 'hparams.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Loaded config from {config_path}")
    print(f"Dataset: {config['ds']}, Root: {config['root']}")
    
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
    if args.checkpoint_name is None:
        checkpoint_path = os.path.join(args.checkpoint, 'saves', 'twobody_ddlp_minimal_off_cnt.pth')
    elif args.checkpoint_name.lower().find('best') != -1:
        checkpoint_path = os.path.join(args.checkpoint, 'saves', 'twobody_ddlp_minimal_off_cnt_best_lpips.pth')
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
        
    
    # Evaluate
    print(f"\nCollecting video data from {args.mode} set...")
    results = evaluate_position_alignment(
        model, config, device=device, mode=args.mode,
        batch_size=args.batch_size, max_batches=args.max_batches,
        use_obj_on_threshold=args.obj_on_threshold,
        num_particles=args.num_particles,
        use_hungarian=args.use_hungarian,
        eval_seq_len=args.eval_seq_len
    )
    
    # Generate visualization videos if requested
    if 'video_data' not in results:
        print("\n⚠️  ERROR: No video data available for visualization")
        return
    
    video_data_list = results['video_data']
    num_evaluated_videos = len(video_data_list)
    
    print(f"\n📊 Video generation summary:")
    print(f"   Videos evaluated: {num_evaluated_videos}")
    
    if num_evaluated_videos < 20:
        raise ValueError(
            f"\n⚠️  ERROR: Insufficient videos for visualization\n"
            f"   Required: 20 videos minimum\n"
            f"   Evaluated: {num_evaluated_videos} videos\n"
            f"   Solution: Increase batch_size or remove max_batches limit\n"
            f"   Current settings: batch_size={args.batch_size}, max_batches={args.max_batches}"
        )
    
    # Use latent_alignment directory instead of figures
    if checkpoint_path.find('best_lpips') != -1:
        video_dir = os.path.join(args.checkpoint, 'latent_alignment_best_lpips')
    else:
        video_dir = os.path.join(args.checkpoint, 'latent_alignment')
    os.makedirs(video_dir, exist_ok=True)
    print(f"   Saving videos to: {video_dir}")
    
    # Take first 20 videos for visualization
    video_data_to_visualize = video_data_list[:20]
    
    # Generate trajectory videos (first set of k=20 videos)
    create_trajectory_videos(
        video_data_to_visualize,
        kp_range=model.kp_range,
        save_dir=video_dir,
        mode=args.mode
    )
    
    # Generate mask overlay videos (second set of k=20 videos)
    create_mask_overlay_videos(
        video_data_to_visualize,
        model=model,
        config=config,
        device=device,
        iou_thresh=config.get('iou_thresh', 0.5),
        save_dir=video_dir,
        mode=args.mode
    )
    
    # Generate prediction-only trajectory videos (third set of k=20 videos)
    create_pred_only_trajectory_videos(
        video_data_to_visualize,
        kp_range=model.kp_range,
        save_dir=video_dir,
        mode=args.mode
    )
    
    # Generate particle distance plots with re-ordering detection (fourth set of k=20 plots)
    create_particle_distance_plots(
        video_data_to_visualize,
        save_dir=video_dir,
        mode=args.mode
    )
    
    print("\n✅ Visualization generation complete!")


if __name__ == '__main__':
    main()
