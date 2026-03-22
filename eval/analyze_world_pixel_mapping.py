#!/usr/bin/env python3
"""
Analyze world-to-pixel mapping for the two-body video dataset.

This script:
1. Loads the actual physics trajectory data
2. Computes global bounds exactly as the rendering script does
3. Analyzes the world-to-pixel mapping with actual rendering parameters
4. Reports mean displacement in world units
5. Calculates minimum visible displacement in pixel space
"""

import os
import numpy as np
import pandas as pd
import h5py
from typing import Tuple, Dict

# Dataset path
DATA_PATH = "/data2/users/lr4617/data_twobody_tries/data_try_elliptic/physics_trajectories/GHNN/Data_Circular_2Body_T_720_radius_1_nu_0.05"
FEATURES_H5 = "circ_2body_training.h5.1"

# Rendering parameters (from create_video_dataset_ddlp_df.py)
H = W = 64
SUPERSAMPLING = 4


def compute_global_bounds(splits: Dict[str, pd.DataFrame]) -> Tuple[float, float, float, float]:
    """Compute global world coordinate bounds across all splits."""
    def _collect_minmax(df):
        x_cols = [c for c in df.columns if c.startswith('q_') and c.endswith('_x')]
        y_cols = [c for c in df.columns if c.startswith('q_') and c.endswith('_y')]
        return (
            float(df[x_cols].min().min()),
            float(df[x_cols].max().max()),
            float(df[y_cols].min().min()),
            float(df[y_cols].max().max())
        )

    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    for df in splits.values():
        xmin, xmax, ymin, ymax = _collect_minmax(df)
        xmins.append(xmin)
        xmaxs.append(xmax)
        ymins.append(ymin)
        ymaxs.append(ymax)
    
    return (min(xmins), max(xmaxs), min(ymins), max(ymaxs))


def world_to_pixel_scale(
    global_bounds: Tuple[float, float, float, float],
    fixed_radius_px: int,
    H: int = 64,
    W: int = 64,
) -> Tuple[float, float]:
    """
    Calculate world-to-pixel scale factors.
    
    Returns:
        (scale_x, scale_y): World units per pixel in x and y directions
    """
    xmin, xmax, ymin, ymax = map(float, global_bounds)
    dx = xmax - xmin
    dy = ymax - ymin
    
    margin = fixed_radius_px
    inner_w = max(1, W - 1 - 2 * margin)
    inner_h = max(1, H - 1 - 2 * margin)
    
    # Scale factors: world_units per pixel
    scale_x = dx / inner_w
    scale_y = dy / inner_h
    
    return scale_x, scale_y


def analyze_trajectories(
    store_path: str,
    global_bounds: Tuple[float, float, float, float],
    fixed_radius_px: int = 5,
) -> Dict:
    """Analyze trajectory displacements and world-to-pixel mapping."""
    
    # Load features split
    print("\n" + "="*80)
    print("ANALYZING WORLD-TO-PIXEL MAPPING")
    print("="*80)
    
    print(f"\n[1] Loading dataset from: {store_path}")
    df = pd.read_hdf(store_path, key="/features")
    print(f"    Total samples: {len(df)}")
    
    # Get coordinate columns
    q_cols = [c for c in df.columns if c.startswith('q_')]
    print(f"    Coordinate columns: {len(q_cols)}")
    
    # Analyze world space
    print(f"\n[2] World space analysis:")
    xmin, xmax, ymin, ymax = global_bounds
    print(f"    X range: [{xmin:.6f}, {xmax:.6f}] = {xmax - xmin:.6f} units")
    print(f"    Y range: [{ymin:.6f}, {ymax:.6f}] = {ymax - ymin:.6f} units")
    
    # Compute displacements in world space
    print(f"\n[3] Computing world-space displacements:")
    all_displacements = []
    
    # Sample first 100 trajectories for analysis
    n_samples = 0
    max_samples = 100
    
    for run_id, df_run in df.groupby("run"):
        if n_samples >= max_samples:
            break
        
        positions = df_run[q_cols].values  # (T, 4) for 2-body system
        
        # Compute consecutive displacements
        dx = np.diff(positions[:, 0::2], axis=0)  # X displacements
        dy = np.diff(positions[:, 1::2], axis=0)  # Y displacements
        
        # Euclidean distance for each body
        for body_idx in range(2):
            disp = np.sqrt(dx[:, body_idx]**2 + dy[:, body_idx]**2)
            all_displacements.extend(disp)
        
        n_samples += 1
    
    displacements = np.array(all_displacements)
    mean_disp_world = np.mean(displacements)
    median_disp_world = np.median(displacements)
    min_disp_world = np.min(displacements[displacements > 0])  # Exclude zeros
    max_disp_world = np.max(displacements)
    
    print(f"    Analyzed {n_samples} trajectories")
    print(f"    Mean displacement:   {mean_disp_world:.6f} world units")
    print(f"    Median displacement: {median_disp_world:.6f} world units")
    print(f"    Min displacement:    {min_disp_world:.6f} world units")
    print(f"    Max displacement:    {max_disp_world:.6f} world units")
    
    # Analyze pixel space mapping with different radius assumptions
    print(f"\n[4] World-to-pixel mapping analysis:")
    print(f"    Image resolution: {W}×{H} pixels")
    print(f"    Supersampling factor: {SUPERSAMPLING}×")
    print(f"    High-res resolution: {W*SUPERSAMPLING}×{H*SUPERSAMPLING} pixels")
    
    # Test multiple radius values commonly used
    radius_tests = [3, 4, 5, 6, 7, 8]
    
    print(f"\n[5] Minimum visible displacement in pixel space:")
    print("    (Assuming 1 pixel displacement is minimum visible change)")
    print()
    print("    Radius | Inner W/H | Scale (world/px) | Min Visible (world) | Mean Disp (px)")
    print("    " + "-"*80)
    
    best_results = None
    
    for r_px in radius_tests:
        scale_x, scale_y = world_to_pixel_scale(global_bounds, r_px, H, W)
        scale_avg = (scale_x + scale_y) / 2
        
        # Minimum visible displacement in world units (1 pixel at base resolution)
        min_visible_world_base = scale_avg * 1.0
        
        # Minimum visible at supersampled resolution
        min_visible_world_hires = scale_avg / SUPERSAMPLING
        
        # Mean displacement in pixel space
        mean_disp_px_base = mean_disp_world / scale_avg
        mean_disp_px_hires = mean_disp_px_base * SUPERSAMPLING
        
        margin = r_px
        inner_dim = max(1, W - 1 - 2 * margin)
        
        print(f"    {r_px:6d} | {inner_dim:9d} | {scale_avg:16.6f} | {min_visible_world_base:19.6f} | {mean_disp_px_base:14.6f}")
        
        # Store results for r=5 (commonly used)
        if r_px == 5:
            best_results = {
                'radius_px': r_px,
                'inner_dim': inner_dim,
                'scale_x': scale_x,
                'scale_y': scale_y,
                'scale_avg': scale_avg,
                'min_visible_world_base': min_visible_world_base,
                'min_visible_world_hires': min_visible_world_hires,
                'mean_disp_world': mean_disp_world,
                'mean_disp_px_base': mean_disp_px_base,
                'mean_disp_px_hires': mean_disp_px_hires,
            }
    
    # Detailed report for r=5 (commonly used value)
    if best_results:
        print(f"\n[6] Detailed analysis for radius = {best_results['radius_px']} pixels:")
        print(f"    Inner image dimensions: {best_results['inner_dim']}×{best_results['inner_dim']} pixels")
        print(f"    World-to-pixel scale:")
        print(f"        X: {best_results['scale_x']:.8f} world units/pixel")
        print(f"        Y: {best_results['scale_y']:.8f} world units/pixel")
        print(f"        Average: {best_results['scale_avg']:.8f} world units/pixel")
        print()
        print(f"    Mean displacement in simulation world:")
        print(f"        {best_results['mean_disp_world']:.8f} world units/timestep")
        print()
        print(f"    Minimum visible displacement (1 pixel change):")
        print(f"        Base resolution (64×64):  {best_results['min_visible_world_base']:.8f} world units")
        print(f"        High-res (4× sampling):   {best_results['min_visible_world_hires']:.8f} world units")
        print()
        print(f"    Mean displacement in pixel space:")
        print(f"        Base resolution:  {best_results['mean_disp_px_base']:.8f} pixels/timestep")
        print(f"        High-res (4× ss): {best_results['mean_disp_px_hires']:.8f} pixels/timestep")
        print()
        print(f"    Ratio (mean displacement / minimum visible):")
        print(f"        Base resolution:  {best_results['mean_disp_world'] / best_results['min_visible_world_base']:.4f}×")
        print(f"        High-res (4× ss): {best_results['mean_disp_world'] / best_results['min_visible_world_hires']:.4f}×")
        print()
        
        if best_results['mean_disp_world'] < best_results['min_visible_world_base']:
            print(f"    ⚠️  Mean displacement ({best_results['mean_disp_world']:.6f}) < Min visible base ({best_results['min_visible_world_base']:.6f})")
            print(f"    At base resolution, typical motion is SUB-PIXEL")
            print()
        
        if best_results['mean_disp_world'] < best_results['min_visible_world_hires']:
            print(f"    ⚠️  Mean displacement ({best_results['mean_disp_world']:.6f}) < Min visible hires ({best_results['min_visible_world_hires']:.6f})")
            print(f"    Even at 4× supersampling, typical motion is SUB-PIXEL")
            print()
        else:
            print(f"    ✓ Mean displacement ({best_results['mean_disp_world']:.6f}) > Min visible hires ({best_results['min_visible_world_hires']:.6f})")
            print(f"    At 4× supersampling, typical motion IS VISIBLE")
            print()
        
        # Anti-aliasing explanation
        print(f"[7] Anti-aliasing mechanism:")
        print(f"    The rendering uses float coordinates + 4× supersampling")
        print(f"    Sub-pixel displacements create INTENSITY GRADIENTS via averaging")
        print()
        print(f"    Example: Ball moves 0.02 pixels at base resolution")
        print(f"    → At 4× supersampling: 0.08 hi-res pixels")
        print(f"    → After downsampling: Smooth intensity shift (not discrete jump)")
        print(f"    → THIS IS WHY videos show smooth motion despite sub-pixel displacements!")
        print()
    
    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    return best_results


def main():
    """Main analysis pipeline."""
    store_path = os.path.join(DATA_PATH, FEATURES_H5)
    
    # Load splits to compute global bounds
    print("\nLoading splits for global bounds computation...")
    splits = {}
    for key in ("/features", "/val_features", "/test_features"):
        try:
            splits[key] = pd.read_hdf(store_path, key=key)
            print(f"  ✓ {key}: {len(splits[key])} samples")
        except (KeyError, FileNotFoundError, OSError):
            print(f"  ✗ {key}: not found")
    
    if not splits:
        print("ERROR: No splits loaded!")
        return
    
    # Compute global bounds
    print("\nComputing global bounds...")
    global_bounds = compute_global_bounds(splits)
    print(f"  Global bounds: x=[{global_bounds[0]:.6f}, {global_bounds[1]:.6f}], y=[{global_bounds[2]:.6f}, {global_bounds[3]:.6f}]")
    
    # Analyze trajectories
    results = analyze_trajectories(store_path, global_bounds, fixed_radius_px=5)
    
    # Summary
    if results:
        print("\n" + "="*80)
        print("SUMMARY FOR USER'S QUESTION")
        print("="*80)
        print()
        print(f"Mean displacement in simulation world:")
        print(f"    {results['mean_disp_world']:.8f} world units/timestep")
        print()
        print(f"Minimum visible displacement in pixel space:")
        print(f"    Base resolution (64×64):     {results['min_visible_world_base']:.8f} world units")
        print(f"    High-res (256×256, 4× ss):   {results['min_visible_world_hires']:.8f} world units")
        print()
        print(f"Mean displacement in pixel space:")
        print(f"    Base resolution:   {results['mean_disp_px_base']:.6f} pixels/timestep")
        print(f"    High-res (4× ss):  {results['mean_disp_px_hires']:.6f} pixels/timestep")
        print()
        print("Key insight:")
        if results['mean_disp_world'] < results['min_visible_world_hires']:
            print("    Even with 4× supersampling, typical motion is sub-pixel in magnitude.")
            print("    However, float coordinates + anti-aliasing preserve this as intensity changes.")
            print("    This is why videos show SMOOTH MOTION despite sub-pixel displacements!")
        else:
            print("    At 4× supersampling, typical motion exceeds 1 pixel and is clearly visible.")
        print()
        print("="*80)


if __name__ == "__main__":
    main()
