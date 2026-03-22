#!/usr/bin/env python
"""
Pipeline for evaluating energy drift as a function of smoothing parameters.

This script:
1. Generates extracted trajectories with different Savitzky-Golay configurations
2. Computes energy drift for each configuration
3. Compares against ground-truth baseline
4. Produces a comprehensive summary table

Configurations tested:
- (window, polyorder): (5,2), (15,2), (15,3), (31,2), (31,3)
- Each with and without momentum smoothing (10 total)
"""

import os
import sys
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import json
from datetime import datetime


# Configuration
BASE_OUTPUT_DIR = "/data2/users/lr4617/data_twobody_tries/data_try_ddlp_extracted/smoothing_tuning"
EVAL_SCRIPT = "/data2/users/lr4617/ddlp/eval/eval_bounding_boxes.py"

# Ground-truth data for comparison
GROUNDTRUTH_DATA = "/data2/users/lr4617/data_twobody_tries/data_try_elliptic/physics_trajectories/GHNN/Data_Circular_2Body_T_720_radius_1_nu_0.05/circ_2body_training.h5.1"

# DDLP checkpoint (update this path as needed)
DDLP_CHECKPOINT = "/data2/users/lr4617/ddlp/outputs/290126_164237_twobody_ddlp_minimal_off_cnt/"
CHECKPOINT_NAME = "best"


def compute_energy_drift(data_path: str, num_samples: int = 10) -> Tuple[float, float, List[float]]:
    """
    Compute energy drift statistics for a dataset.
    
    Returns:
        avg_drift: Average maximum energy drift across samples
        std_drift: Standard deviation of energy drifts
        all_drifts: List of maximum drifts for each trajectory
    """
    print(f"  Computing energy drift for: {os.path.basename(data_path)}")
    
    # Load data
    all_runs_path = data_path.replace('_training.h5.1', '_all_runs.h5.1')
    if not os.path.exists(all_runs_path):
        print(f"    ⚠️  Warning: {all_runs_path} not found")
        return np.nan, np.nan, []
    
    all_runs = pd.read_hdf(all_runs_path, '/all_runs')
    
    try:
        constants = pd.read_hdf(all_runs_path, '/constants')
        bodies = constants.get('bodies', ['A', 'B'])
        masses = np.array(constants.get('masses', [[1.0, 1.0]])[0])
    except (ValueError, KeyError) as e:
        bodies = ['A', 'B']
        masses = np.array([1.0, 1.0])
    
    # Sample trajectories
    run_ids = all_runs.index.get_level_values('run').unique()
    sample_size = min(num_samples, len(run_ids))
    sampled_ids = np.random.choice(run_ids, sample_size, replace=False)
    
    energy_drifts = []
    
    for run_id in sampled_ids:
        run = all_runs.loc[run_id]
        
        # Compute kinetic energy
        KE = np.zeros(len(run))
        for i, body in enumerate(bodies):
            p_x = run[f'p_{body}_x'].values
            p_y = run[f'p_{body}_y'].values
            KE += 0.5 * (p_x**2 + p_y**2) / masses[i]
        
        # Compute potential energy (2-body gravitational)
        if len(bodies) == 2:
            q1_x = run[f'q_{bodies[0]}_x'].values
            q1_y = run[f'q_{bodies[0]}_y'].values
            q2_x = run[f'q_{bodies[1]}_x'].values
            q2_y = run[f'q_{bodies[1]}_y'].values
            
            r = np.sqrt((q2_x - q1_x)**2 + (q2_y - q1_y)**2)
            G = 1.0
            PE = -G * masses[0] * masses[1] / (r + 1e-10)
        else:
            PE = np.zeros(len(run))
        
        # Total energy and drift
        E = KE + PE
        E_initial = E[0]
        E_drift = np.abs(E - E_initial) / (np.abs(E_initial) + 1e-10)
        
        max_drift = E_drift.max()
        energy_drifts.append(max_drift)
    
    avg_drift = np.mean(energy_drifts)
    std_drift = np.std(energy_drifts)
    
    print(f"    → Avg drift: {avg_drift:.2%} ± {std_drift:.2%}")
    
    return avg_drift, std_drift, energy_drifts


def generate_trajectories(window_length: int, polyorder: int, smooth_momentum: bool,
                         output_name: str, checkpoint: str, checkpoint_name: str) -> str:
    """
    Generate trajectories with specified smoothing parameters.
    
    Returns:
        Path to generated training.h5.1 file
    """
    # The eval script creates files directly in BASE_OUTPUT_DIR with dataset_name prefix
    training_file = os.path.join(BASE_OUTPUT_DIR, f"{output_name}_training.h5.1")
    
    # Skip if already exists
    if os.path.exists(training_file):
        print(f"  ✓ Already exists: {output_name}")
        return training_file
    
    print(f"  Generating: {output_name}")
    print(f"    window_length={window_length}, polyorder={polyorder}, smooth_momentum={int(smooth_momentum)}")
    
    # Build command
    cmd = [
        "python", EVAL_SCRIPT,
        "--checkpoint", checkpoint,
        "--checkpoint_name", checkpoint_name,
        "--convert_to_ghnn", "1",
        "--ghnn_output_dir", BASE_OUTPUT_DIR,
        "--ghnn_dataset_name", output_name,
        "--ghnn_train_seq_len", "60",
        "--eval_seq_len", "360",
        "--ghnn_step_size", "1.0",
        "--ghnn_dt", "1.0",
        "--sg_window_length", str(window_length),
        "--sg_polyorder", str(polyorder),
        "--smooth_momentum", str(int(smooth_momentum))
    ]
    
    # Run extraction
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"    ✓ Generated successfully")
        return training_file
    except subprocess.CalledProcessError as e:
        print(f"    ✗ Error generating trajectories:")
        print(f"      {e.stderr}")
        return None


def run_pipeline():
    """Main pipeline execution."""
    print("="*80)
    print("SMOOTHING PARAMETER TUNING PIPELINE")
    print("="*80)
    print(f"Output directory: {BASE_OUTPUT_DIR}")
    print(f"DDLP checkpoint:  {DDLP_CHECKPOINT}")
    print(f"Ground-truth:     {GROUNDTRUTH_DATA}")
    print("="*80)
    
    # Create output directory
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    
    # Define configurations
    configs = [
        # (window_length, polyorder, smooth_momentum)
        (5, 2, False),
        (5, 2, True),
        (15, 2, False),
        (15, 2, True),
        (15, 3, False),
        (15, 3, True),
        (31, 2, False),
        (31, 2, True),
        (31, 3, False),
        (31, 3, True),
    ]
    
    print(f"\nConfigurations to test: {len(configs)}")
    for i, (w, p, m) in enumerate(configs, 1):
        mom_str = "with_mom" if m else "no_mom"
        print(f"  {i}. window={w:2d}, polyorder={p}, momentum_smoothing={mom_str}")
    
    # Step 1: Compute ground-truth baseline
    print("\n" + "="*80)
    print("STEP 1: Computing Ground-Truth Baseline")
    print("="*80)
    
    if os.path.exists(GROUNDTRUTH_DATA):
        gt_avg, gt_std, gt_drifts = compute_energy_drift(GROUNDTRUTH_DATA, num_samples=20)
        print(f"\n✓ Ground-truth energy drift: {gt_avg:.4%} ± {gt_std:.4%}")
    else:
        print(f"⚠️  Ground-truth data not found: {GROUNDTRUTH_DATA}")
        gt_avg, gt_std = np.nan, np.nan
    
    # Step 2: Generate trajectories for all configurations
    print("\n" + "="*80)
    print("STEP 2: Generating Trajectories")
    print("="*80)
    
    results = []
    
    for i, (window_length, polyorder, smooth_momentum) in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Configuration: w={window_length}, p={polyorder}, smooth_mom={smooth_momentum}")
        
        # Generate dataset name
        mom_suffix = "_mom" if smooth_momentum else ""
        dataset_name = f"w{window_length}_p{polyorder}{mom_suffix}"
        
        # Generate trajectories
        training_file = generate_trajectories(
            window_length=window_length,
            polyorder=polyorder,
            smooth_momentum=smooth_momentum,
            output_name=dataset_name,
            checkpoint=DDLP_CHECKPOINT,
            checkpoint_name=CHECKPOINT_NAME
        )
        
        if training_file is None or not os.path.exists(training_file):
            print(f"  ✗ Failed to generate {dataset_name}")
            results.append({
                'config': dataset_name,
                'window_length': window_length,
                'polyorder': polyorder,
                'smooth_momentum': smooth_momentum,
                'avg_drift': np.nan,
                'std_drift': np.nan,
                'status': 'failed'
            })
            continue
        
        # Compute energy drift
        avg_drift, std_drift, drifts = compute_energy_drift(training_file, num_samples=20)
        
        results.append({
            'config': dataset_name,
            'window_length': window_length,
            'polyorder': polyorder,
            'smooth_momentum': smooth_momentum,
            'avg_drift': avg_drift,
            'std_drift': std_drift,
            'status': 'success',
            'training_file': training_file
        })
    
    # Step 3: Create summary table
    print("\n" + "="*80)
    print("STEP 3: Summary Results")
    print("="*80)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Add ground-truth row
    gt_row = pd.DataFrame([{
        'config': 'ground_truth',
        'window_length': np.nan,
        'polyorder': np.nan,
        'smooth_momentum': np.nan,
        'avg_drift': gt_avg,
        'std_drift': gt_std,
        'status': 'baseline',
        'training_file': GROUNDTRUTH_DATA
    }])
    
    df_full = pd.concat([gt_row, df], ignore_index=True)
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(BASE_OUTPUT_DIR, f"smoothing_tuning_results_{timestamp}.csv")
    df_full.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to: {csv_path}")
    
    # Print table
    print("\n" + "="*80)
    print("ENERGY DRIFT COMPARISON TABLE")
    print("="*80)
    print(f"\n{'Config':<20s} {'Window':<8s} {'Poly':<6s} {'MomSmooth':<12s} {'Avg Drift':<15s} {'Std Drift':<15s} {'Status':<10s}")
    print("-" * 100)
    
    for _, row in df_full.iterrows():
        config = row['config']
        window = f"{int(row['window_length'])}" if pd.notna(row['window_length']) else "N/A"
        poly = f"{int(row['polyorder'])}" if pd.notna(row['polyorder']) else "N/A"
        mom = "Yes" if row['smooth_momentum'] else ("No" if pd.notna(row['smooth_momentum']) else "N/A")
        avg = f"{row['avg_drift']:.4%}" if pd.notna(row['avg_drift']) else "N/A"
        std = f"± {row['std_drift']:.4%}" if pd.notna(row['std_drift']) else "N/A"
        status = row['status']
        
        print(f"{config:<20s} {window:<8s} {poly:<6s} {mom:<12s} {avg:<15s} {std:<15s} {status:<10s}")
    
    # Step 4: Create visualization
    print("\n" + "="*80)
    print("STEP 4: Creating Visualizations")
    print("="*80)
    
    create_comparison_plots(df_full, BASE_OUTPUT_DIR)
    
    # Step 5: Analysis and recommendations
    print("\n" + "="*80)
    print("STEP 5: Analysis and Recommendations")
    print("="*80)
    
    analyze_results(df_full, gt_avg)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"All results saved to: {BASE_OUTPUT_DIR}")
    print(f"Summary CSV: {csv_path}")


def create_comparison_plots(df: pd.DataFrame, output_dir: str):
    """Create visualization comparing energy drifts."""
    
    # Filter out ground-truth and failed runs
    df_extracted = df[df['status'] == 'success'].copy()
    
    if len(df_extracted) == 0:
        print("  ⚠️  No successful runs to plot")
        return
    
    # Plot 1: Bar chart comparing all configurations
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Subplot 1: Without momentum smoothing
    ax = axes[0]
    df_no_mom = df_extracted[df_extracted['smooth_momentum'] == False].sort_values('avg_drift')
    if len(df_no_mom) > 0:
        x = np.arange(len(df_no_mom))
        ax.bar(x, df_no_mom['avg_drift'].values * 100, yerr=df_no_mom['std_drift'].values * 100,
               capsize=5, alpha=0.7, color='steelblue')
        ax.set_xticks(x)
        ax.set_xticklabels(df_no_mom['config'].values, rotation=45, ha='right')
        ax.set_ylabel('Energy Drift (%)')
        ax.set_title('Without Momentum Smoothing')
        ax.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='10% threshold')
        
        # Add ground-truth baseline
        gt_row = df[df['config'] == 'ground_truth']
        if len(gt_row) > 0 and pd.notna(gt_row['avg_drift'].values[0]):
            gt_drift = gt_row['avg_drift'].values[0] * 100
            ax.axhline(y=gt_drift, color='green', linestyle='--', alpha=0.5, label='Ground-truth')
        
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    # Subplot 2: With momentum smoothing
    ax = axes[1]
    df_with_mom = df_extracted[df_extracted['smooth_momentum'] == True].sort_values('avg_drift')
    if len(df_with_mom) > 0:
        x = np.arange(len(df_with_mom))
        ax.bar(x, df_with_mom['avg_drift'].values * 100, yerr=df_with_mom['std_drift'].values * 100,
               capsize=5, alpha=0.7, color='coral')
        ax.set_xticks(x)
        ax.set_xticklabels(df_with_mom['config'].values, rotation=45, ha='right')
        ax.set_ylabel('Energy Drift (%)')
        ax.set_title('With Momentum Smoothing')
        ax.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='10% threshold')
        
        # Add ground-truth baseline
        gt_row = df[df['config'] == 'ground_truth']
        if len(gt_row) > 0 and pd.notna(gt_row['avg_drift'].values[0]):
            gt_drift = gt_row['avg_drift'].values[0] * 100
            ax.axhline(y=gt_drift, color='green', linestyle='--', alpha=0.5, label='Ground-truth')
        
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'energy_drift_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {plot_path}")
    
    # Plot 2: Momentum smoothing effect
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Group by (window_length, polyorder) and compare with/without momentum smoothing
    for (w, p), group in df_extracted.groupby(['window_length', 'polyorder']):
        no_mom = group[group['smooth_momentum'] == False]
        with_mom = group[group['smooth_momentum'] == True]
        
        if len(no_mom) > 0 and len(with_mom) > 0:
            label = f"w={int(w)}, p={int(p)}"
            x_pos = [0, 1]
            y_vals = [no_mom['avg_drift'].values[0] * 100, with_mom['avg_drift'].values[0] * 100]
            ax.plot(x_pos, y_vals, marker='o', label=label, linewidth=2, markersize=8)
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['No Momentum\nSmoothing', 'With Momentum\nSmoothing'])
    ax.set_ylabel('Energy Drift (%)')
    ax.set_title('Effect of Momentum Smoothing')
    ax.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='10% threshold')
    
    # Add ground-truth baseline
    gt_row = df[df['config'] == 'ground_truth']
    if len(gt_row) > 0 and pd.notna(gt_row['avg_drift'].values[0]):
        gt_drift = gt_row['avg_drift'].values[0] * 100
        ax.axhline(y=gt_drift, color='green', linestyle='--', alpha=0.5, label='Ground-truth')
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'momentum_smoothing_effect.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {plot_path}")


def analyze_results(df: pd.DataFrame, gt_drift: float):
    """Analyze results and provide recommendations."""
    
    df_extracted = df[df['status'] == 'success'].copy()
    
    if len(df_extracted) == 0:
        print("  ⚠️  No successful runs to analyze")
        return
    
    # Find best configuration
    best_config = df_extracted.loc[df_extracted['avg_drift'].idxmin()]
    
    print(f"\n✓ Best Configuration:")
    print(f"  Config:           {best_config['config']}")
    print(f"  Window length:    {int(best_config['window_length'])}")
    print(f"  Polynomial order: {int(best_config['polyorder'])}")
    print(f"  Momentum smooth:  {'Yes' if best_config['smooth_momentum'] else 'No'}")
    print(f"  Energy drift:     {best_config['avg_drift']:.4%} ± {best_config['std_drift']:.4%}")
    
    # Compare to ground-truth
    if pd.notna(gt_drift):
        improvement = (df_extracted['avg_drift'].max() - best_config['avg_drift']) / df_extracted['avg_drift'].max()
        diff_from_gt = best_config['avg_drift'] - gt_drift
        
        print(f"\n✓ Comparison:")
        print(f"  Ground-truth drift:       {gt_drift:.4%}")
        print(f"  Best extracted drift:     {best_config['avg_drift']:.4%}")
        print(f"  Difference:               {diff_from_gt:.4%} ({diff_from_gt/gt_drift:.1f}× worse)")
        print(f"  Improvement over worst:   {improvement:.1%}")
    
    # Momentum smoothing analysis
    print(f"\n✓ Momentum Smoothing Effect:")
    for (w, p), group in df_extracted.groupby(['window_length', 'polyorder']):
        no_mom = group[group['smooth_momentum'] == False]
        with_mom = group[group['smooth_momentum'] == True]
        
        if len(no_mom) > 0 and len(with_mom) > 0:
            drift_no = no_mom['avg_drift'].values[0]
            drift_with = with_mom['avg_drift'].values[0]
            change = ((drift_with - drift_no) / drift_no) * 100
            
            symbol = "↓" if change < 0 else "↑"
            print(f"  w={int(w)}, p={int(p)}: {drift_no:.2%} → {drift_with:.2%} ({symbol} {abs(change):.1f}%)")
    
    # Recommendations
    print(f"\n✓ Recommendations:")
    if best_config['avg_drift'] < 0.10:
        print(f"  ✅ Best config ({best_config['avg_drift']:.1%} drift) is suitable for GHNN")
        print(f"     → Can use symplectic architecture")
    elif best_config['avg_drift'] < 0.30:
        print(f"  ⚠️  Best config ({best_config['avg_drift']:.1%} drift) exceeds 10% GHNN threshold")
        print(f"     → Use MLP architecture instead")
        print(f"     → MLP can handle moderate non-symplectic behavior")
    else:
        print(f"  ❌ Best config ({best_config['avg_drift']:.1%} drift) has high energy drift")
        print(f"     → Visual tracking pipeline fundamentally non-symplectic")
        print(f"     → Consider: (1) MLP with regularization, (2) Physics-informed losses")


def main():
    """Entry point."""
    
    # Check if eval script exists
    if not os.path.exists(EVAL_SCRIPT):
        print(f"❌ Error: Evaluation script not found: {EVAL_SCRIPT}")
        return 1
    
    # Check if checkpoint exists
    if not os.path.exists(DDLP_CHECKPOINT):
        print(f"❌ Error: DDLP checkpoint not found: {DDLP_CHECKPOINT}")
        print(f"   Please update DDLP_CHECKPOINT in the script")
        return 1
    
    # Run pipeline
    run_pipeline()
    
    return 0


if __name__ == '__main__':
    exit(main())
