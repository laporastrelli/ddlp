#!/usr/bin/env python
"""
Test if the "scaled Hamiltonian" hypothesis holds.

If H_latent = α·KE + β·PE with global α, β, then:
- The ratio KE/PE should be constant across all trajectories
- The rescaling factor λ = √(|PE|/(2·KE)) should be the same for all trajectories

This script tests these assumptions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os


def test_rescaling_hypothesis(data_path: str, label: str, output_dir: str):
    """Test if rescaling parameters are trajectory-independent."""
    print(f"\n{'='*80}")
    print(f"TESTING RESCALING HYPOTHESIS: {label}")
    print(f"{'='*80}")
    print(f"\nHypothesis: If latent dynamics follow H_latent = α·KE + β·PE,")
    print(f"            then KE/PE ratio should be constant across trajectories")
    
    # Load data
    all_runs = pd.read_hdf(data_path.replace('_training.h5.1', '_all_runs.h5.1'), '/all_runs')
    
    try:
        constants = pd.read_hdf(data_path.replace('_training.h5.1', '_all_runs.h5.1'), '/constants')
        bodies = constants.get('bodies', ['A', 'B'])
        masses = np.array(constants.get('masses', [[1.0, 1.0]])[0])
    except:
        bodies = ['A', 'B']
        masses = np.array([1.0, 1.0])
    
    # Sample many long trajectories
    run_lengths = all_runs.groupby(level=0).size()
    long_runs = run_lengths[run_lengths > 100].index
    
    sample_size = min(50, len(long_runs))
    sample_runs = np.random.choice(long_runs, sample_size, replace=False)
    
    print(f"\nAnalyzing {sample_size} trajectories with >100 timesteps")
    
    # For each trajectory, compute:
    # 1. Time-averaged KE/PE ratio
    # 2. Required rescaling factor λ
    # 3. Initial energy
    # 4. Energy drift
    
    ke_pe_ratios = []
    rescaling_factors = []
    initial_energies = []
    energy_drifts = []
    trajectory_ids = []
    
    for run_id in sample_runs:
        run = all_runs.loc[run_id]
        
        # Compute KE
        KE = np.zeros(len(run))
        for i, body in enumerate(bodies):
            p_x = run[f'p_{body}_x'].values
            p_y = run[f'p_{body}_y'].values
            KE += 0.5 * (p_x**2 + p_y**2) / masses[i]
        
        # Compute PE
        if len(bodies) == 2:
            q1_x = run[f'q_{bodies[0]}_x'].values
            q1_y = run[f'q_{bodies[0]}_y'].values
            q2_x = run[f'q_{bodies[1]}_x'].values
            q2_y = run[f'q_{bodies[1]}_y'].values
            r = np.sqrt((q2_x - q1_x)**2 + (q2_y - q1_y)**2)
            PE = -masses[0] * masses[1] / (r + 1e-10)
        else:
            continue  # Skip if not 2-body
        
        # Time-averaged quantities
        avg_KE = np.mean(KE)
        avg_PE = np.mean(np.abs(PE))
        
        if avg_KE > 1e-10 and avg_PE > 1e-10:
            ke_pe_ratio = avg_KE / avg_PE
            rescaling_factor = np.sqrt(0.5 * avg_PE / avg_KE)  # To achieve KE/PE = 0.5
            
            ke_pe_ratios.append(ke_pe_ratio)
            rescaling_factors.append(rescaling_factor)
            
            # Energy drift
            E = KE + PE
            drift = np.abs(E - E[0]).max() / (np.abs(E[0]) + 1e-10)
            energy_drifts.append(drift)
            initial_energies.append(E[0])
            trajectory_ids.append(run_id)
    
    # Statistical analysis
    ke_pe_ratios = np.array(ke_pe_ratios)
    rescaling_factors = np.array(rescaling_factors)
    energy_drifts = np.array(energy_drifts)
    initial_energies = np.array(initial_energies)
    
    print(f"\n{'─'*80}")
    print(f"STATISTICAL RESULTS")
    print(f"{'─'*80}")
    print(f"\nKE/PE Ratio (time-averaged):")
    print(f"  Mean:   {ke_pe_ratios.mean():.6e}")
    print(f"  Std:    {ke_pe_ratios.std():.6e}")
    print(f"  CV:     {ke_pe_ratios.std()/ke_pe_ratios.mean():.2%} (coefficient of variation)")
    print(f"  Range:  [{ke_pe_ratios.min():.6e}, {ke_pe_ratios.max():.6e}]")
    
    print(f"\nRescaling Factor λ:")
    print(f"  Mean:   {rescaling_factors.mean():.2f}×")
    print(f"  Std:    {rescaling_factors.std():.2f}×")
    print(f"  CV:     {rescaling_factors.std()/rescaling_factors.mean():.2%}")
    print(f"  Range:  [{rescaling_factors.min():.2f}×, {rescaling_factors.max():.2f}×]")
    
    print(f"\nEnergy Drift:")
    print(f"  Mean:   {energy_drifts.mean():.2%}")
    print(f"  Std:    {energy_drifts.std():.2%}")
    print(f"  Range:  [{energy_drifts.min():.2%}, {energy_drifts.max():.2%}]")
    
    print(f"\n{'─'*80}")
    print(f"HYPOTHESIS TEST")
    print(f"{'─'*80}")
    
    # Test if CV is small (< 10% would suggest global parameters)
    cv_ke_pe = ke_pe_ratios.std() / ke_pe_ratios.mean()
    cv_rescaling = rescaling_factors.std() / rescaling_factors.mean()
    
    if cv_ke_pe < 0.10:
        print(f"✅ HYPOTHESIS SUPPORTED: KE/PE ratio is consistent across trajectories")
        print(f"   CV = {cv_ke_pe:.1%} < 10% suggests global α, β parameters exist")
        print(f"   This supports the scaled Hamiltonian model: H_latent = α·KE + β·PE")
    elif cv_ke_pe < 0.30:
        print(f"⚠️  HYPOTHESIS PARTIALLY SUPPORTED: KE/PE ratio has moderate variation")
        print(f"   CV = {cv_ke_pe:.1%} suggests α, β may vary somewhat across trajectories")
        print(f"   A simple scaled Hamiltonian is an approximation")
    else:
        print(f"❌ HYPOTHESIS REJECTED: KE/PE ratio varies significantly across trajectories")
        print(f"   CV = {cv_ke_pe:.1%} > 30% indicates no global α, β parameters")
        print(f"   The scaled Hamiltonian model H_latent = α·KE + β·PE does not hold")
        print(f"   Relationship between KE and PE is trajectory-dependent")
    
    # Visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Rescaling Hypothesis Test: {label}', fontsize=16, fontweight='bold')
    
    # 1. KE/PE ratio distribution
    ax = axes[0, 0]
    ax.hist(ke_pe_ratios, bins=30, alpha=0.7, edgecolor='black')
    ax.axvline(ke_pe_ratios.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={ke_pe_ratios.mean():.2e}')
    ax.axvline(0.5, color='green', linestyle='--', linewidth=2, label='Virial (0.5)')
    ax.set_xlabel('KE/PE Ratio (time-averaged)')
    ax.set_ylabel('Count')
    ax.set_title(f'KE/PE Distribution (CV={cv_ke_pe:.1%})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Rescaling factor distribution
    ax = axes[0, 1]
    ax.hist(rescaling_factors, bins=30, alpha=0.7, edgecolor='black', color='orange')
    ax.axvline(rescaling_factors.mean(), color='red', linestyle='--', linewidth=2, 
               label=f'Mean={rescaling_factors.mean():.1f}×')
    ax.set_xlabel('Required Rescaling Factor λ')
    ax.set_ylabel('Count')
    ax.set_title(f'Rescaling Factor Distribution (CV={cv_rescaling:.1%})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Energy drift distribution
    ax = axes[0, 2]
    ax.hist(energy_drifts * 100, bins=30, alpha=0.7, edgecolor='black', color='green')
    ax.axvline(energy_drifts.mean() * 100, color='red', linestyle='--', linewidth=2,
               label=f'Mean={energy_drifts.mean():.1%}')
    ax.set_xlabel('Energy Drift (%)')
    ax.set_ylabel('Count')
    ax.set_title('Energy Drift Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. KE/PE vs trajectory ID (check for systematic trends)
    ax = axes[1, 0]
    ax.scatter(trajectory_ids, ke_pe_ratios, alpha=0.5)
    ax.axhline(ke_pe_ratios.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    ax.set_xlabel('Trajectory ID')
    ax.set_ylabel('KE/PE Ratio')
    ax.set_title('KE/PE vs Trajectory ID')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Rescaling factor vs initial energy (check for energy-dependent rescaling)
    ax = axes[1, 1]
    scatter = ax.scatter(initial_energies, rescaling_factors, c=energy_drifts, 
                        cmap='viridis', alpha=0.6)
    ax.set_xlabel('Initial Energy E(0)')
    ax.set_ylabel('Rescaling Factor λ')
    ax.set_title('Rescaling Factor vs Initial Energy')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Energy Drift')
    
    # 6. Energy drift vs KE/PE ratio
    ax = axes[1, 2]
    ax.scatter(ke_pe_ratios, energy_drifts * 100, alpha=0.5)
    ax.set_xlabel('KE/PE Ratio')
    ax.set_ylabel('Energy Drift (%)')
    ax.set_title('Energy Drift vs KE/PE Ratio')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'rescaling_hypothesis_{label.replace(" ", "_")}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: {output_path}")
    
    return {
        'ke_pe_mean': ke_pe_ratios.mean(),
        'ke_pe_std': ke_pe_ratios.std(),
        'ke_pe_cv': cv_ke_pe,
        'rescaling_mean': rescaling_factors.mean(),
        'rescaling_std': rescaling_factors.std(),
        'rescaling_cv': cv_rescaling,
        'hypothesis_supported': cv_ke_pe < 0.10
    }


def main():
    parser = argparse.ArgumentParser(
        description='Test if rescaling parameters α, β are trajectory-independent'
    )
    parser.add_argument('--extracted_bbox_data', type=str, required=True)
    parser.add_argument('--extracted_latent_data', type=str, required=True)
    parser.add_argument('--groundtruth_data', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='rescaling_hypothesis_test')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("TESTING RESCALING HYPOTHESIS")
    print("="*80)
    print("\nQuestion: Are α, β in H_latent = α·KE + β·PE global constants?")
    print("Method: Check if KE/PE ratio is consistent across trajectories")
    print("="*80)
    
    # Test all three datasets
    bbox_results = test_rescaling_hypothesis(args.extracted_bbox_data, "Bbox Extraction", args.output_dir)
    latent_results = test_rescaling_hypothesis(args.extracted_latent_data, "Latent Extraction", args.output_dir)
    gt_results = test_rescaling_hypothesis(args.groundtruth_data, "Ground Truth", args.output_dir)
    
    # Summary
    print(f"\n{'='*80}")
    print(f"COMPARATIVE SUMMARY")
    print(f"{'='*80}")
    
    for label, results in [("Bbox", bbox_results), ("Latent", latent_results), ("Ground Truth", gt_results)]:
        print(f"\n{label}:")
        print(f"  KE/PE CV: {results['ke_pe_cv']:.1%}")
        print(f"  Rescaling λ: {results['rescaling_mean']:.2f} ± {results['rescaling_std']:.2f}×")
        print(f"  Hypothesis: {'✅ Supported' if results['hypothesis_supported'] else '❌ Not supported'}")
    
    print(f"\n{'='*80}")
    print(f"CONCLUSIONS")
    print(f"{'='*80}")
    
    if bbox_results['hypothesis_supported'] and latent_results['hypothesis_supported']:
        print(f"\n✅ Both extraction methods show consistent KE/PE ratios across trajectories")
        print(f"   This supports using global rescaling with virial theorem")
        print(f"   Recommended: Apply λ_bbox = {bbox_results['rescaling_mean']:.1f}× and λ_latent = {latent_results['rescaling_mean']:.1f}×")
    else:
        print(f"\n⚠️  KE/PE ratio varies significantly across trajectories")
        print(f"   Global rescaling (Strategy 1) may not be optimal")
        print(f"   Consider:")
        print(f"   - Per-trajectory rescaling (Strategy 3)")
        print(f"   - Energy conservation minimization (Strategy 2)")
        print(f"   - The 'scaled Hamiltonian' assumption may be too simplistic")
    
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
