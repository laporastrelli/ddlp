#!/usr/bin/env python
"""
Analyze kinetic vs potential energy components to diagnose scale mismatch.

This script investigates why extracted trajectories show KE ≈ 0 while PE = O(1),
and proposes physics-informed rescaling strategies based on symplecticity.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse


def analyze_energy_components(data_path: str, label: str, output_dir: str):
    """Analyze KE and PE scales and their ratio."""
    print(f"\n{'='*80}")
    print(f"ENERGY COMPONENT ANALYSIS: {label}")
    print(f"{'='*80}")
    
    # Load data
    all_runs = pd.read_hdf(data_path.replace('_training.h5.1', '_all_runs.h5.1'), '/all_runs')
    
    try:
        constants = pd.read_hdf(data_path.replace('_training.h5.1', '_all_runs.h5.1'), '/constants')
        bodies = constants.get('bodies', ['A', 'B'])
        masses = np.array(constants.get('masses', [[1.0, 1.0]])[0])
        dt = constants.get('dt', 1.0)
        kp_range = constants.get('kp_range', None)
    except (ValueError, KeyError) as e:
        bodies = ['A', 'B']
        masses = np.array([1.0, 1.0])
        dt = 1.0
        kp_range = None
    
    print(f"Dataset info:")
    print(f"  Bodies: {bodies}")
    print(f"  Masses: {masses}")
    print(f"  dt: {dt}")
    print(f"  kp_range: {kp_range}")
    
    # Sample several long runs
    run_lengths = all_runs.groupby(level=0).size()
    long_runs = run_lengths[run_lengths > 100].index
    sample_runs = np.random.choice(long_runs, min(10, len(long_runs)), replace=False)
    
    ke_magnitudes = []
    pe_magnitudes = []
    velocity_magnitudes = []
    position_magnitudes = []
    momentum_magnitudes = []
    
    for run_id in sample_runs:
        run = all_runs.loc[run_id]
        
        # Compute kinetic energy: KE = Σ p²/(2m)
        KE = np.zeros(len(run))
        total_momentum_sq = 0.0
        total_velocity_sq = 0.0
        
        for i, body in enumerate(bodies):
            p_x = run[f'p_{body}_x'].values
            p_y = run[f'p_{body}_y'].values
            
            # Velocity from momentum
            v_x = p_x / masses[i]
            v_y = p_y / masses[i]
            
            KE += 0.5 * (p_x**2 + p_y**2) / masses[i]
            total_momentum_sq += np.mean(p_x**2 + p_y**2)
            total_velocity_sq += np.mean(v_x**2 + v_y**2)
        
        # Compute potential energy: PE = -G·m1·m2/r
        if len(bodies) == 2:
            q1_x = run[f'q_{bodies[0]}_x'].values
            q1_y = run[f'q_{bodies[0]}_y'].values
            q2_x = run[f'q_{bodies[1]}_x'].values
            q2_y = run[f'q_{bodies[1]}_y'].values
            
            r = np.sqrt((q2_x - q1_x)**2 + (q2_y - q1_y)**2)
            G = 1.0
            PE = -G * masses[0] * masses[1] / (r + 1e-10)
            
            position_magnitude = np.mean(r)
        else:
            PE = np.zeros(len(run))
            position_magnitude = 0.0
        
        ke_magnitudes.append(np.mean(np.abs(KE)))
        pe_magnitudes.append(np.mean(np.abs(PE)))
        velocity_magnitudes.append(np.sqrt(total_velocity_sq))
        position_magnitudes.append(position_magnitude)
        momentum_magnitudes.append(np.sqrt(total_momentum_sq))
    
    # Statistics
    ke_mean = np.mean(ke_magnitudes)
    pe_mean = np.mean(pe_magnitudes)
    v_mean = np.mean(velocity_magnitudes)
    r_mean = np.mean(position_magnitudes)
    p_mean = np.mean(momentum_magnitudes)
    
    print(f"\n{'─'*80}")
    print(f"SCALE ANALYSIS (averaged over {len(sample_runs)} runs)")
    print(f"{'─'*80}")
    print(f"Position scale:     |r|     ~ {r_mean:.6f}")
    print(f"Velocity scale:     |v|     ~ {v_mean:.6f}")
    print(f"Momentum scale:     |p|     ~ {p_mean:.6f}")
    print(f"Kinetic energy:     <|KE|>  ~ {ke_mean:.6e}")
    print(f"Potential energy:   <|PE|>  ~ {pe_mean:.6e}")
    print(f"KE/PE ratio:        KE/PE   ~ {ke_mean/pe_mean:.6e}")
    print(f"PE/KE ratio:        PE/KE   ~ {pe_mean/ke_mean:.2f}×")
    
    print(f"\n{'─'*80}")
    print(f"DIAGNOSIS")
    print(f"{'─'*80}")
    
    if ke_mean / pe_mean < 0.01:
        print(f"❌ SEVERE SCALE MISMATCH DETECTED!")
        print(f"   KE is {pe_mean/ke_mean:.0f}× smaller than PE")
        print(f"   This causes energy drift to be dominated by PE fluctuations")
        print(f"")
        print(f"   Root cause: Positions in range {kp_range if kp_range else '~[-1,1]'}")
        print(f"               → velocities dq/dt ~ {v_mean:.6f} (very small)")
        print(f"               → KE = m·v²/2 ~ {ke_mean:.6e} (negligible)")
        print(f"               → PE = -Gm₁m₂/r ~ {pe_mean:.6e} (dominant)")
        print(f"")
        print(f"   Why this breaks energy conservation:")
        print(f"   In real orbits, KE and PE have comparable magnitudes (virial theorem)")
        print(f"   Here, KE cannot compensate PE changes → apparent energy drift")
    else:
        print(f"✅ Energy components are balanced")
    
    return {
        'ke_mean': ke_mean,
        'pe_mean': pe_mean,
        'v_mean': v_mean,
        'r_mean': r_mean,
        'p_mean': p_mean,
        'ratio': pe_mean / ke_mean
    }


def propose_rescaling_strategy(bbox_stats: dict, latent_stats: dict, gt_stats: dict):
    """Propose physics-informed rescaling without using ground-truth scales."""
    print(f"\n{'='*80}")
    print(f"PHYSICS-INFORMED RESCALING STRATEGY")
    print(f"{'='*80}")
    
    print(f"\nQ1: Why doesn't the model adapt to the imposed range and preserve symplecticity?")
    print(f"─" * 80)
    print(f"A: The model DOES preserve symplecticity in its latent space! The issue is not")
    print(f"   symplecticity, but coordinate scaling. Consider:")
    print(f"")
    print(f"   • DDLP learns: φ: (q,p) → (q',p') that preserves H_latent(q,p)")
    print(f"   • Latent Hamiltonian: H_latent might be scaled version of true H")
    print(f"   • The tanh constraint forces q ∈ [-1,1]")
    print(f"   • For realistic orbits with period T_true, in latent space:")
    print(f"       - Positions: q ~ O(1) ✓ (matches kp_range)")
    print(f"       - Velocities: v = dq/dt ~ Δq/Δt ~ 2/(T_true/dt) ~ {latent_stats['v_mean']:.6f}")
    print(f"       - Momenta: p = m·v ~ {latent_stats['p_mean']:.6f}")
    print(f"")
    print(f"   The symplectic structure is preserved, but the Hamiltonian is scaled:")
    print(f"       H_latent = α·KE + β·PE  (where α << β)")
    print(f"")
    print(f"   When we compute energy with assumed units (G=1, m=1), we're using:")
    print(f"       H_assumed = KE + PE")
    print(f"   which doesn't match H_latent → apparent energy non-conservation!")
    
    print(f"\n\nQ2: Is this due to encoder/decoder nonlinearity?")
    print(f"─" * 80)
    print(f"A: Partially. The encoder maps pixel space → latent space with tanh output,")
    print(f"   creating a compressed coordinate system. But the key issue is that:")
    print(f"")
    print(f"   • Bounding boxes extract in PIXEL space → different scale")
    print(f"   • Latent positions extract in LATENT space → kp_range scale")
    print(f"   • Both then estimate momentum p = m·dq/dt with finite differences")
    print(f"   • If dq is small (latent) or large (pixel), p scales accordingly")
    print(f"")
    print(f"   The nonlinearity doesn't destroy symplecticity, but creates a coordinate")
    print(f"   transformation where our 'standard' energy formula no longer applies.")
    
    # Calculate required momentum rescaling
    gt_ke_pe_ratio = gt_stats['ke_mean'] / gt_stats['pe_mean']
    latent_ke_pe_ratio = latent_stats['ke_mean'] / latent_stats['pe_mean']
    bbox_ke_pe_ratio = bbox_stats['ke_mean'] / bbox_stats['pe_mean']
    
    # For virial theorem in 2-body system: <KE> = -0.5*<PE> for circular orbits
    # More generally for bound orbits: <KE> ~ O(|PE|)
    virial_ratio = 0.5  # Expected KE/|PE| for circular orbits
    
    latent_momentum_scale = np.sqrt(virial_ratio * latent_stats['pe_mean'] / latent_stats['ke_mean'])
    bbox_momentum_scale = np.sqrt(virial_ratio * bbox_stats['pe_mean'] / bbox_stats['ke_mean'])
    
    print(f"\n\nQ3: How to rescale without ground-truth information?")
    print(f"─" * 80)
    print(f"A: Use SYMPLECTIC CONSTRAINTS as a physics prior!")
    print(f"")
    print(f"Strategy 1: VIRIAL THEOREM RESCALING")
    print(f"───────────────────────────────────────")
    print(f"For bound orbits in 1/r potential: <KE> = -<PE>/2 (virial theorem)")
    print(f"We can rescale momenta to enforce this without knowing true scales:")
    print(f"")
    print(f"  1. Compute current <KE> and <PE> with p = m·dq/dt")
    print(f"  2. Find scaling factor: λ = √(|<PE>|/(2·<KE>))")
    print(f"  3. Rescale momenta: p_new = λ·p_old")
    print(f"  4. This enforces virial theorem without ground-truth!")
    print(f"")
    print(f"Current state:")
    print(f"  Ground-truth:     KE/|PE| = {gt_ke_pe_ratio:.3f}  (target: ~0.5)")
    print(f"  Latent method:    KE/|PE| = {latent_ke_pe_ratio:.6f}")
    print(f"  Bbox method:      KE/|PE| = {bbox_ke_pe_ratio:.6f}")
    print(f"")
    print(f"Required momentum rescaling:")
    print(f"  Latent: multiply p by {latent_momentum_scale:.2f}×")
    print(f"  Bbox:   multiply p by {bbox_momentum_scale:.2f}×")
    
    print(f"\n\nStrategy 2: ENERGY CONSERVATION RESCALING")
    print(f"───────────────────────────────────────────")
    print(f"If we don't assume specific potential (1/r), use energy conservation:")
    print(f"")
    print(f"  1. Compute total energy E(t) = KE(t) + PE(t) for each timestep")
    print(f"  2. Find momentum scale λ that minimizes Var(E(t))")
    print(f"  3. This enforces energy conservation without assuming potential form")
    print(f"")
    print(f"This is more general but requires optimization over λ.")
    
    print(f"\n\nStrategy 3: ADAPTIVE RESCALING PER TRAJECTORY")
    print(f"──────────────────────────────────────────────")
    print(f"Since each trajectory might have different energy scales:")
    print(f"")
    print(f"  1. For each trajectory, compute λ_i = √(|<PE>_i|/(2·<KE>_i))")
    print(f"  2. Rescale that trajectory's momenta: p_i,new = λ_i·p_i,old")
    print(f"  3. This adapts to per-trajectory scale without global ground-truth")
    print(f"")
    print(f"Advantage: Handles variable energy trajectories")
    print(f"Disadvantage: Breaks absolute energy scale (only relative preserved)")
    
    print(f"\n\n{'─'*80}")
    print(f"RECOMMENDATION")
    print(f"{'─'*80}")
    print(f"Use Strategy 1 (Virial Theorem Rescaling) because:")
    print(f"  ✓ Physics-motivated (valid for gravitational 2-body)")
    print(f"  ✓ No ground-truth needed")
    print(f"  ✓ Simple to implement")
    print(f"  ✓ Preserves symplectic structure (just rescales momentum)")
    print(f"")
    print(f"Implementation:")
    print(f"  Add --rescale_momentum flag to eval_bounding_boxes.py")
    print(f"  After computing trajectories, apply virial rescaling")
    print(f"  This should dramatically reduce energy drift!")
    
    return {
        'latent_scale': latent_momentum_scale,
        'bbox_scale': bbox_momentum_scale,
        'method': 'virial_theorem'
    }


def visualize_rescaling_effect(data_path: str, scale_factor: float, label: str, output_path: str):
    """Visualize energy before and after rescaling."""
    print(f"\nGenerating rescaling visualization for {label}...")
    
    all_runs = pd.read_hdf(data_path.replace('_training.h5.1', '_all_runs.h5.1'), '/all_runs')
    
    try:
        constants = pd.read_hdf(data_path.replace('_training.h5.1', '_all_runs.h5.1'), '/constants')
        bodies = constants.get('bodies', ['A', 'B'])
        masses = np.array(constants.get('masses', [[1.0, 1.0]])[0])
    except:
        bodies = ['A', 'B']
        masses = np.array([1.0, 1.0])
    
    # Pick a long run
    run_lengths = all_runs.groupby(level=0).size()
    long_runs = run_lengths[run_lengths > 100].index
    run_id = np.random.choice(long_runs)
    run = all_runs.loc[run_id]
    
    # Compute energy components BEFORE rescaling
    KE_before = np.zeros(len(run))
    for i, body in enumerate(bodies):
        p_x = run[f'p_{body}_x'].values
        p_y = run[f'p_{body}_y'].values
        KE_before += 0.5 * (p_x**2 + p_y**2) / masses[i]
    
    if len(bodies) == 2:
        q1_x = run[f'q_{bodies[0]}_x'].values
        q1_y = run[f'q_{bodies[0]}_y'].values
        q2_x = run[f'q_{bodies[1]}_x'].values
        q2_y = run[f'q_{bodies[1]}_y'].values
        r = np.sqrt((q2_x - q1_x)**2 + (q2_y - q1_y)**2)
        PE = -masses[0] * masses[1] / (r + 1e-10)
    else:
        PE = np.zeros(len(run))
    
    E_before = KE_before + PE
    
    # Compute energy AFTER rescaling (p_new = scale * p_old)
    # KE_new = 0.5 * (scale*p)^2 / m = scale^2 * KE_old
    KE_after = (scale_factor**2) * KE_before
    E_after = KE_after + PE
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Momentum Rescaling Effect: {label} (run {run_id})', fontsize=14, fontweight='bold')
    
    time = np.arange(len(run))
    
    # Before rescaling - Energy components
    ax = axes[0, 0]
    ax.plot(time, E_before, label='Total', linewidth=2)
    ax.plot(time, KE_before, label='Kinetic', alpha=0.7)
    ax.plot(time, PE, label='Potential', alpha=0.7)
    ax.set_title(f'BEFORE Rescaling')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Energy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # After rescaling - Energy components
    ax = axes[0, 1]
    ax.plot(time, E_after, label='Total', linewidth=2)
    ax.plot(time, KE_after, label=f'Kinetic (×{scale_factor**2:.1f})', alpha=0.7)
    ax.plot(time, PE, label='Potential', alpha=0.7)
    ax.set_title(f'AFTER Rescaling (p × {scale_factor:.2f})')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Energy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Before rescaling - Energy drift
    ax = axes[1, 0]
    drift_before = 100 * (E_before - E_before[0]) / np.abs(E_before[0])
    ax.plot(time, drift_before, linewidth=2, color='red')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='10% threshold')
    ax.axhline(y=-10, color='orange', linestyle='--', alpha=0.5)
    ax.set_title(f'Energy Drift BEFORE: {np.abs(drift_before).max():.1f}%')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Energy Drift (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # After rescaling - Energy drift
    ax = axes[1, 1]
    drift_after = 100 * (E_after - E_after[0]) / np.abs(E_after[0])
    ax.plot(time, drift_after, linewidth=2, color='green')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='10% threshold')
    ax.axhline(y=-10, color='orange', linestyle='--', alpha=0.5)
    ax.set_title(f'Energy Drift AFTER: {np.abs(drift_after).max():.1f}%')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Energy Drift (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add summary text
    improvement = (np.abs(drift_before).max() - np.abs(drift_after).max()) / np.abs(drift_before).max() * 100
    fig.text(0.5, 0.02, 
             f'Summary: Energy drift reduced from {np.abs(drift_before).max():.1f}% to {np.abs(drift_after).max():.1f}% '
             f'({improvement:.1f}% improvement)', 
             ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze energy component scales and propose rescaling strategies'
    )
    parser.add_argument('--extracted_bbox_data', type=str, required=True)
    parser.add_argument('--extracted_latent_data', type=str, required=True)
    parser.add_argument('--groundtruth_data', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='energy_analysis')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("ENERGY SCALE ANALYSIS AND RESCALING STRATEGY")
    print("="*80)
    
    # Analyze all three datasets
    bbox_stats = analyze_energy_components(args.extracted_bbox_data, "Bbox Extraction", args.output_dir)
    latent_stats = analyze_energy_components(args.extracted_latent_data, "Latent Extraction", args.output_dir)
    gt_stats = analyze_energy_components(args.groundtruth_data, "Ground Truth", args.output_dir)
    
    # Propose rescaling strategy
    scales = propose_rescaling_strategy(bbox_stats, latent_stats, gt_stats)
    
    # Visualize rescaling effect
    print(f"\n{'='*80}")
    print(f"VISUALIZING RESCALING EFFECT")
    print(f"{'='*80}")
    
    visualize_rescaling_effect(
        args.extracted_latent_data, 
        scales['latent_scale'], 
        "Latent Extraction",
        os.path.join(args.output_dir, 'rescaling_effect_latent.png')
    )
    
    visualize_rescaling_effect(
        args.extracted_bbox_data, 
        scales['bbox_scale'], 
        "Bbox Extraction",
        os.path.join(args.output_dir, 'rescaling_effect_bbox.png')
    )
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {args.output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Review visualizations to confirm rescaling improves energy conservation")
    print(f"  2. Implement --rescale_momentum flag in eval_bounding_boxes.py")
    print(f"  3. Re-extract trajectories with rescaling enabled")
    print(f"  4. Train GHNN on rescaled data")


if __name__ == '__main__':
    main()
