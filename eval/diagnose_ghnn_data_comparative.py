#!/usr/bin/env python
"""
Enhanced diagnostic script to analyze GHNN-formatted data with comparison capabilities.

This script can:
1. Analyze extracted trajectories (from DDLP visual tracking)
2. Analyze ground-truth trajectories (from physics simulation)
3. Compare both datasets side-by-side

Helps identify whether training failure is due to:
- Data format errors
- Physical inconsistencies (non-symplectic behavior)
- Numerical issues (NaN, Inf, extreme values)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import json
from tqdm import tqdm
from typing import Optional, Dict, Tuple

# OK
def load_trajectory_data(data_path: str, max_rows: Optional[int] = None) -> pd.DataFrame:
    """Load trajectory data from /test_features in training.h5.1.
    
    Args:
        data_path: Path to training.h5.1 file
        max_rows: Maximum number of rows to load, or None to load all
        
    Returns:
        DataFrame with trajectory data indexed by run
    """
    if max_rows is None:
        test_features = pd.read_hdf(data_path, '/test_features')
    else:
        test_features = pd.read_hdf(data_path, '/test_features', start=0, stop=max_rows)
    # Convert to run-indexed format
    if 'run' in test_features.columns:
        test_features = test_features.set_index('run', append=True)
        test_features = test_features.swaplevel(0, 1)
    return test_features

# OK
def load_constants(data_path: str) -> pd.Series:
    """Load constants from training.h5.1.
    
    Args:
        data_path: Path to training.h5.1 file
        
    Returns:
        Series with constants (bodies, masses, dt, etc.)
    """
    return pd.read_hdf(data_path, '/constants')


# OK
def check_data_integrity(data_path: str, label: str = "Dataset") -> Tuple[bool, Dict]:
    """Check for NaN, Inf, and extreme values."""
    print("\n" + "="*80)
    print(f"1. DATA INTEGRITY CHECK - {label}")
    print("="*80)
    
    # Load a subset of training data (100k rows) for faster integrity check
    # This is sufficient to detect NaN/Inf issues and check value ranges
    print(f"Loading subset of data for integrity check (first 100k rows)...")
    features = pd.read_hdf(data_path, '/features', start=0, stop=100000)
    labels = pd.read_hdf(data_path, '/labels', start=0, stop=100000)
    
    # Try to read constants, handle pickle protocol issues
    try:
        constants = pd.read_hdf(data_path, '/constants')
    except ValueError as e:
        print(f"Warning: Could not read constants due to pickle protocol: {e}")
        print("Using default values for analysis...")
        constants = pd.Series({'step_size': 1.0, 'dt': 1.0})
    
    print(f"\nConstants:")
    for key, val in constants.items():
        if key == 'masses' and isinstance(val, list) and len(val) > 3:
            # Print only first element if it's a long repeated list
            print(f"  {key}: {val[0]} (showing first of {len(val)} entries)")
        else:
            print(f"  {key}: {val}")
    
    # Check for NaN/Inf
    print(f"\nFeatures shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    
    nan_features = features.isna().sum().sum()
    nan_labels = labels.isna().sum().sum()
    inf_features = np.isinf(features.select_dtypes(include=[np.number]).values).sum()
    inf_labels = np.isinf(labels.select_dtypes(include=[np.number]).values).sum()
    
    print(f"\nNaN count - Features: {nan_features}, Labels: {nan_labels}")
    print(f"Inf count - Features: {inf_features}, Labels: {inf_labels}")
    
    integrity_ok = True
    if nan_features > 0 or nan_labels > 0 or inf_features > 0 or inf_labels > 0:
        print("❌ ERROR: Data contains NaN or Inf values!")
        integrity_ok = False
    else:
        print("✅ No NaN or Inf values found")
    
    # Check value ranges
    print(f"\n--- Value Ranges ---")
    stats = {}
    for col in features.columns:
        if col == 'run':
            continue
        vals = features[col].values
        stats[col] = {
            'min': vals.min(),
            'max': vals.max(),
            'mean': vals.mean(),
            'std': vals.std()
        }
        print(f"{col:12s}: [{vals.min():+.6f}, {vals.max():+.6f}], mean={vals.mean():+.6f}, std={vals.std():.6f}")
    
    return integrity_ok, stats

# OK
def check_energy_conservation(data_path: str, num_samples: int = 5, label: str = "Dataset") -> Tuple[float, list]:
    """Check if energy is approximately conserved (symplectic check).
    
    Args:
        data_path: Path to dataset
        num_samples: Number of trajectories to sample
        label: Dataset label for printing
        test_finite_difference_error: If True, recompute momentum using finite differences
                                     from positions to isolate FD discretization error
    """
    
    print("\n" + "="*80)
    print(f"3. ENERGY CONSERVATION CHECK (Symplectic Test) - {label}")
    print("="*80)
    
    # Load constants first to avoid loading full dataset
    try:
        constants = load_constants(data_path)
        bodies = constants.get('bodies', ['A', 'B'])
        masses = np.array(constants.get('masses', [[1.0, 1.0]])[0])
    except (ValueError, KeyError) as e:
        bodies = ['A', 'B']
        masses = np.array([1.0, 1.0])
    
    # print information about the system being analyzed
    print(f"\n=== 2-Body Hamiltonian ===")
    print(f"H(q,p) = Σ_i |p_i|²/(2m_i) - G·m₁·m₂/|q₁-q₂|")
    print(f"Kinetic Energy:  T = Σ_i (p_i,x² + p_i,y²)/(2m_i)")
    print(f"Potential Energy: V = -G·m₁·m₂/r₁₂")
    print(f"Total Energy:     E = T + V")
    print(f"")
    print(f"Parameters: G=1.0, m₁={masses[0]}, m₂={masses[1]}\n")
    
    # Load data
    print(f"Loading subset of data for analysis (first 100k rows)...")
    runs_df = load_trajectory_data(data_path)
    run_ids_available = runs_df.index.get_level_values('run').unique()
    
    # collect run IDs of long trajectories (length > 100)
    run_ids = np.random.choice(
        run_ids_available, 
        min(num_samples, len(run_ids_available)), 
        replace=False
    )
    
    energy_drifts = []    
    for run_id in run_ids:

        # get run data
        run = runs_df.loc[run_id]
        
        # Compute kinetic energy
        KE = np.zeros(len(run))
        for i, body in enumerate(bodies):
            p_x = run[f'p_{body}_x'].values
            p_y = run[f'p_{body}_y'].values
            KE += 0.5 * (p_x**2 + p_y**2) / masses[i]
                
        # Compute potential energy (assuming gravitational): PE = -G * m1*m2 / r
        if len(bodies) == 2:
            q1_x = run[f'q_{bodies[0]}_x'].values
            q1_y = run[f'q_{bodies[0]}_y'].values
            q2_x = run[f'q_{bodies[1]}_x'].values
            q2_y = run[f'q_{bodies[1]}_y'].values
            
            r = np.sqrt((q2_x - q1_x)**2 + (q2_y - q1_y)**2)
            G = 1.0  # Assuming normalized units
            PE = -G * masses[0] * masses[1] / (r + 1e-10)  # Add epsilon to avoid division by zero
        else:
            raise ValueError("Energy conservation check currently only supports 2-body systems.")
        
        # Compute total energy with stored momentum
        E = KE + PE
        E_initial = E[0]
        E_drift = np.abs(E - E_initial) / (np.abs(E_initial) + 1e-10)
        
        max_drift = E_drift.max()
        mean_drift = E_drift.mean()
        energy_drifts.append(max_drift)
        
        # Compute total energy with finite-difference momentum (if testing)
        print(f"Run {run_id}: max_drift={max_drift:.2%}, mean_drift={mean_drift:.2%}")
    
    avg_drift = np.mean(energy_drifts)
    print(f"\nAverage maximum energy drift: {avg_drift:.2%}")
    
    if avg_drift > 0.5:  # 50% drift
        print("⚠️  WARNING: Large energy drift - trajectories may not be physically consistent")
    elif avg_drift > 0.1:  # 10% drift
        print("⚠️  CAUTION: Moderate energy drift - this might affect GHNN training")
    else:
        print("✅ Energy drift is reasonable (<10% - suitable for GHNN)")
    
    return avg_drift, energy_drifts


def compute_optimal_scale_factor(data_path: str, num_samples: int = 10, s_range: tuple = (0.001, 5.0), n_points: int = 50) -> float:
    """Compute optimal scale factor s using virial theorem with fixed Δt=1.
    
    For isotropic scaling q_lat = s*q_true:
    - Rescales positions: q_rescaled = q_lat / s
    - Computes momenta with Δt=1: p = m * dq_rescaled/dt
    - Both K and V are affected by s:
      * K ∝ (s/Δt)² with Δt=1 fixed → K ∝ s²
      * V ∝ 1/s for gravity
    
    Returns s that makes <KE> ≈ 0.5 * <|PE|> (virial theorem for 2-body gravity).
    """
    # Load only a manageable subset (100k rows) for faster performance
    runs_df = load_trajectory_data(data_path, max_rows=100000)
    
    try:
        constants = load_constants(data_path)
        bodies = constants.get('bodies', ['A', 'B'])
        masses = np.array(constants.get('masses', [[1.0, 1.0]])[0])
    except:
        bodies = ['A', 'B']
        masses = np.array([1.0, 1.0])
    
    # Sample trajectories
    run_lengths = runs_df.groupby(level=0).size()
    long_runs = run_lengths[run_lengths > 100].index
    sample_size = min(num_samples, len(long_runs))
    sample_runs = np.random.choice(long_runs, sample_size, replace=False)
    
    # Scan scale factors
    s_values = np.logspace(np.log10(s_range[0]), np.log10(s_range[1]), n_points)
    virial_errors = []
    
    for s in s_values:
        ke_values = []
        pe_values = []
        
        for run_id in sample_runs:
            run = runs_df.loc[run_id]
            
            # Rescale positions
            positions_rescaled = {}
            for body in bodies:
                positions_rescaled[body] = {
                    'x': run[f'q_{body}_x'].values / s,
                    'y': run[f'q_{body}_y'].values / s
                }
            
            # Compute KE with Δt=1 using finite differences on rescaled positions
            KE = 0.0
            dt = 1.0  # Fixed gauge
            for i, body in enumerate(bodies):
                q_x = positions_rescaled[body]['x']
                q_y = positions_rescaled[body]['y']
                
                # Centered differences for velocity
                v_x = np.zeros_like(q_x)
                v_y = np.zeros_like(q_y)
                v_x[0] = (q_x[1] - q_x[0]) / dt
                v_y[0] = (q_y[1] - q_y[0]) / dt
                v_x[1:-1] = (q_x[2:] - q_x[:-2]) / (2.0 * dt)
                v_y[1:-1] = (q_y[2:] - q_y[:-2]) / (2.0 * dt)
                v_x[-1] = (q_x[-1] - q_x[-2]) / dt
                v_y[-1] = (q_y[-1] - q_y[-2]) / dt
                
                p_x = masses[i] * v_x
                p_y = masses[i] * v_y
                KE += np.mean(0.5 * (p_x**2 + p_y**2) / masses[i])
            
            # Compute PE from rescaled positions (V ∝ 1/s for gravity)
            if len(bodies) == 2:
                q1_x = positions_rescaled[bodies[0]]['x']
                q1_y = positions_rescaled[bodies[0]]['y']
                q2_x = positions_rescaled[bodies[1]]['x']
                q2_y = positions_rescaled[bodies[1]]['y']
                r = np.sqrt((q2_x - q1_x)**2 + (q2_y - q1_y)**2)
                PE_vals = -masses[0] * masses[1] / (r + 1e-10)
                PE = np.mean(np.abs(PE_vals))
            else:
                continue
            
            ke_values.append(KE)
            pe_values.append(PE)
        
        avg_KE = np.mean(ke_values)
        avg_PE = np.mean(pe_values)
        
        # Virial theorem error: KE should equal 0.5*PE for 2-body gravity
        virial_error = abs(avg_KE - 0.5 * avg_PE) / (0.5 * avg_PE + 1e-10)
        virial_errors.append(virial_error)
    
    # Find s with minimum virial error
    optimal_idx = np.argmin(virial_errors)
    s_opt = s_values[optimal_idx]
    
    return s_opt


def compute_optimal_scale_by_drift(data_path: str, num_samples: int = 10, s_range: tuple = (0.001, 5.0), n_points: int = 50) -> float:
    """Compute optimal scale factor s by minimizing energy drift with Δt=1 fixed.
    
    Rescales positions q_rescaled = q_lat/s, computes momenta with Δt=1,
    and picks s that minimizes energy drift.
    
    Args:
        data_path: Path to HDF5 data file
        num_samples: Number of trajectories to sample
        s_range: Range of scale factors to scan (min, max)
        n_points: Number of points to scan
    
    Returns:
        s_opt: Optimal scale factor that minimizes energy drift
    """
    try:
        # Load only a manageable subset (100k rows) for faster performance
        runs_df = load_trajectory_data(data_path, max_rows=100000)
        constants = load_constants(data_path)
    except Exception as e:
        print(f"    Warning: Could not load data for dt optimization: {e}")
        return 1.0
    
    bodies = constants.get('bodies', ['A', 'B'])
    masses = np.array(constants.get('masses', [[1.0, 1.0]])[0])
    
    # Sample long trajectories
    run_lengths = runs_df.groupby(level=0).size()
    long_runs = run_lengths[run_lengths > 100].index.tolist()
    
    if len(long_runs) < num_samples:
        sampled_runs = long_runs
    else:
        sampled_runs = sorted(np.random.choice(long_runs, num_samples, replace=False))
    
    # Scan scale factors
    s_values = np.logspace(np.log10(s_range[0]), np.log10(s_range[1]), n_points)
    drift_scores = []
    
    dt = 1.0  # Fixed gauge
    for s in s_values:
        max_drifts = []
        for run_id in sampled_runs:
            run = runs_df.loc[run_id]
            
            # Compute energy with rescaled positions and Δt=1
            KE = np.zeros(len(run))
            for i, body in enumerate(bodies):
                q_x = run[f'q_{body}_x'].values / s  # Rescale positions
                q_y = run[f'q_{body}_y'].values / s
                v_x = np.zeros_like(q_x)
                v_y = np.zeros_like(q_y)
                v_x[0] = (q_x[1] - q_x[0]) / dt
                v_y[0] = (q_y[1] - q_y[0]) / dt
                v_x[1:-1] = (q_x[2:] - q_x[:-2]) / (2.0 * dt)
                v_y[1:-1] = (q_y[2:] - q_y[:-2]) / (2.0 * dt)
                v_x[-1] = (q_x[-1] - q_x[-2]) / dt
                v_y[-1] = (q_y[-1] - q_y[-2]) / dt
                p_x = masses[i] * v_x
                p_y = masses[i] * v_y
                KE += 0.5 * (p_x**2 + p_y**2) / masses[i]
            
            # Compute PE from rescaled positions
            if len(bodies) == 2:
                q1_x = run[f'q_{bodies[0]}_x'].values / s
                q1_y = run[f'q_{bodies[0]}_y'].values / s
                q2_x = run[f'q_{bodies[1]}_x'].values / s
                q2_y = run[f'q_{bodies[1]}_y'].values / s
                r = np.sqrt((q2_x - q1_x)**2 + (q2_y - q1_y)**2)
                PE = -masses[0] * masses[1] / (r + 1e-10)
            else:
                PE = np.zeros(len(run))
            
            E = KE + PE
            drift = np.abs((E - E[0]) / np.abs(E[0]))
            max_drifts.append(drift.max())
        
        drift_scores.append(np.mean(max_drifts))
    
    # Find s with minimum drift
    optimal_idx = np.argmin(drift_scores)
    s_opt = s_values[optimal_idx]
    
    return s_opt


def compute_optimal_scale_by_variance(data_path: str, num_samples: int = 10, s_range: tuple = (0.001, 5.0), n_points: int = 50) -> float:
    """Compute optimal scale factor s by minimizing energy variance with Δt=1 fixed.
    
    Rescales positions q_rescaled = q_lat/s, computes momenta with Δt=1,
    and picks s that minimizes energy variance.
    
    Args:
        data_path: Path to HDF5 data file
        num_samples: Number of trajectories to sample
        s_range: Range of scale factors to scan (min, max)
        n_points: Number of points to scan
    
    Returns:
        s_opt: Optimal scale factor that minimizes energy variance
    """
    try:
        # Load only a manageable subset (100k rows) for faster performance
        runs_df = load_trajectory_data(data_path, max_rows=100000)
        constants = load_constants(data_path)
    except Exception as e:
        print(f"    Warning: Could not load data for scale optimization: {e}")
        return 1.0
    
    bodies = constants.get('bodies', ['A', 'B'])
    masses = np.array(constants.get('masses', [[1.0, 1.0]])[0])
    
    # Sample long trajectories
    run_lengths = runs_df.groupby(level=0).size()
    long_runs = run_lengths[run_lengths > 100].index.tolist()
    
    if len(long_runs) < num_samples:
        sampled_runs = long_runs
    else:
        sampled_runs = sorted(np.random.choice(long_runs, num_samples, replace=False))
    
    # Scan scale factors
    s_values = np.logspace(np.log10(s_range[0]), np.log10(s_range[1]), n_points)
    variance_scores = []
    
    dt = 1.0  # Fixed gauge
    for s in s_values:
        variances = []
        for run_id in sampled_runs:
            run = runs_df.loc[run_id]
            
            # Compute energy with rescaled positions and Δt=1
            KE = np.zeros(len(run))
            for i, body in enumerate(bodies):
                q_x = run[f'q_{body}_x'].values / s  # Rescale positions
                q_y = run[f'q_{body}_y'].values / s
                v_x = np.zeros_like(q_x)
                v_y = np.zeros_like(q_y)
                v_x[0] = (q_x[1] - q_x[0]) / dt
                v_y[0] = (q_y[1] - q_y[0]) / dt
                v_x[1:-1] = (q_x[2:] - q_x[:-2]) / (2.0 * dt)
                v_y[1:-1] = (q_y[2:] - q_y[:-2]) / (2.0 * dt)
                v_x[-1] = (q_x[-1] - q_x[-2]) / dt
                v_y[-1] = (q_y[-1] - q_y[-2]) / dt
                p_x = masses[i] * v_x
                p_y = masses[i] * v_y
                KE += 0.5 * (p_x**2 + p_y**2) / masses[i]
            
            # Compute PE from rescaled positions
            if len(bodies) == 2:
                q1_x = run[f'q_{bodies[0]}_x'].values / s
                q1_y = run[f'q_{bodies[0]}_y'].values / s
                q2_x = run[f'q_{bodies[1]}_x'].values / s
                q2_y = run[f'q_{bodies[1]}_y'].values / s
                r = np.sqrt((q2_x - q1_x)**2 + (q2_y - q1_y)**2)
                PE = -masses[0] * masses[1] / (r + 1e-10)
            else:
                PE = np.zeros(len(run))
            
            E = KE + PE
            variances.append(np.var(E))
        
        variance_scores.append(np.mean(variances))
    
    # Find s with minimum variance
    optimal_idx = np.argmin(variance_scores)
    s_opt = s_values[optimal_idx]
    
    return s_opt


def compute_optimal_scale_per_trajectory(data_path: str, num_samples: int = 10, s_range: tuple = (0.001, 5.0), n_points: int = 30) -> dict:
    """Compute optimal scale factor s for each trajectory individually with Δt=1 fixed.
    
    Returns dictionary with per-trajectory results for statistical analysis.
    
    Args:
        data_path: Path to HDF5 data file
        num_samples: Number of trajectories to sample
        s_range: Range of scale factors to scan (min, max)
        n_points: Number of points to scan (fewer for per-traj to save time)
    
    Returns:
        dict with 's_values': list of optimal s per trajectory,
                    'run_ids': list of run IDs
    """
    try:
        # Load only a manageable subset (100k rows) for faster performance
        runs_df = load_trajectory_data(data_path, max_rows=100000)
        constants = load_constants(data_path)
    except Exception as e:
        print(f"    Warning: Could not load data for scale optimization: {e}")
        return {'s_values': [1.0], 'run_ids': [0]}
    
    bodies = constants.get('bodies', ['A', 'B'])
    masses = np.array(constants.get('masses', [[1.0, 1.0]])[0])
    
    # Sample long trajectories
    run_lengths = runs_df.groupby(level=0).size()
    long_runs = run_lengths[run_lengths > 100].index.tolist()
    
    if len(long_runs) < num_samples:
        sampled_runs = long_runs
    else:
        sampled_runs = sorted(np.random.choice(long_runs, num_samples, replace=False))
    
    s_values = np.logspace(np.log10(s_range[0]), np.log10(s_range[1]), n_points)
    optimal_scales = []
    
    dt = 1.0  # Fixed gauge
    for run_id in sampled_runs:
        run = runs_df.loc[run_id]
        drift_scores = []
        
        for s in s_values:
            # Compute energy with rescaled positions and Δt=1
            KE = np.zeros(len(run))
            for i, body in enumerate(bodies):
                q_x = run[f'q_{body}_x'].values / s  # Rescale positions
                q_y = run[f'q_{body}_y'].values / s
                v_x = np.zeros_like(q_x)
                v_y = np.zeros_like(q_y)
                v_x[0] = (q_x[1] - q_x[0]) / dt
                v_y[0] = (q_y[1] - q_y[0]) / dt
                v_x[1:-1] = (q_x[2:] - q_x[:-2]) / (2.0 * dt)
                v_y[1:-1] = (q_y[2:] - q_y[:-2]) / (2.0 * dt)
                v_x[-1] = (q_x[-1] - q_x[-2]) / dt
                v_y[-1] = (q_y[-1] - q_y[-2]) / dt
                p_x = masses[i] * v_x
                p_y = masses[i] * v_y
                KE += 0.5 * (p_x**2 + p_y**2) / masses[i]
            
            # Compute PE from rescaled positions
            if len(bodies) == 2:
                q1_x = run[f'q_{bodies[0]}_x'].values / s
                q1_y = run[f'q_{bodies[0]}_y'].values / s
                q2_x = run[f'q_{bodies[1]}_x'].values / s
                q2_y = run[f'q_{bodies[1]}_y'].values / s
                r = np.sqrt((q2_x - q1_x)**2 + (q2_y - q1_y)**2)
                PE = -masses[0] * masses[1] / (r + 1e-10)
            else:
                PE = np.zeros(len(run))
            
            E = KE + PE
            drift = np.abs((E - E[0]) / np.abs(E[0]))
            drift_scores.append(drift.max())
        
        # Find optimal s for this trajectory
        optimal_idx = np.argmin(drift_scores)
        optimal_scales.append(s_values[optimal_idx])
    
    return {'s_values': optimal_scales, 'run_ids': sampled_runs}


def compute_energy_drift_with_dt_methods(run, positions, dt_methods, masses, bodies, dt_base, is_gt=False):
    """Helper function to compute energy drifts for all dt methods.
    
    Returns dict with drift arrays for each method and per-traj statistics.
    """
    # Compute PE (same for all)
    if len(bodies) == 2:
        q1_x = positions[bodies[0]]['x']
        q1_y = positions[bodies[0]]['y']
        q2_x = positions[bodies[1]]['x']
        q2_y = positions[bodies[1]]['y']
        r = np.sqrt((q2_x - q1_x)**2 + (q2_y - q1_y)**2)
        PE = -masses[0] * masses[1] / (r + 1e-10)
    else:
        raise ValueError("Energy drift computation currently only supports 2-body systems.")
    
    results = {}
    
    # Baseline: stored momentum (dt=1 or true p for GT)
    KE_base = np.zeros(len(run))
    for i, body in enumerate(bodies):
        p_x = run[f'p_{body}_x'].values
        p_y = run[f'p_{body}_y'].values
        KE_base += 0.5 * (p_x**2 + p_y**2) / masses[i]
    E_base = KE_base + PE
    results['baseline'] = 100 * (E_base - E_base[0]) / np.abs(E_base[0])
    
    if is_gt:
        # For GT: Finite difference methods with various dt values
        # Use dt_base if dt_methods is None
        if dt_methods is not None:
            dt_drift = dt_methods.get('drift', dt_base)
            dt_variance = dt_methods.get('variance', dt_base)
        else:
            dt_drift = dt_base
            dt_variance = dt_base
        
        for method_name, dt_val in [('dt=1', dt_base), ('drift', dt_drift), ('variance', dt_variance)]:
            KE = np.zeros(len(run))
            for i, body in enumerate(bodies):
                q_x = positions[body]['x']
                q_y = positions[body]['y']
                v_x = np.zeros_like(q_x)
                v_y = np.zeros_like(q_y)
                v_x[0] = (q_x[1] - q_x[0]) / dt_val
                v_y[0] = (q_y[1] - q_y[0]) / dt_val
                v_x[1:-1] = (q_x[2:] - q_x[:-2]) / (2.0 * dt_val)
                v_y[1:-1] = (q_y[2:] - q_y[:-2]) / (2.0 * dt_val)
                v_x[-1] = (q_x[-1] - q_x[-2]) / dt_val
                v_y[-1] = (q_y[-1] - q_y[-2]) / dt_val
                p_x_fd = masses[i] * v_x
                p_y_fd = masses[i] * v_y
                KE += 0.5 * (p_x_fd**2 + p_y_fd**2) / masses[i]
            E = KE + PE
            results[method_name] = 100 * (E - E[0]) / np.abs(E[0])
    else:
        # For extracted: Position rescaling methods (q' = q/s, p' = m*dq'/dt)
        # Under A=sI: KE scales as 1/s², PE scales as s (for gravity V∝-1/r)
        # s_opt is stored as the dt_methods values (virial, drift, variance)
        # Use dt_base if dt_methods is None
        if dt_methods is not None:
            dt_virial = dt_methods.get('virial', dt_base)
            dt_drift = dt_methods.get('drift', dt_base)
            dt_variance = dt_methods.get('variance', dt_base)
        else:
            dt_virial = dt_base
            dt_drift = dt_base
            dt_variance = dt_base
        
        for method_name, s_val in [('virial', dt_virial), ('drift', dt_drift), ('variance', dt_variance)]:
            # Rescaling q' = q/s gives:
            #   p' = m * d(q/s)/dt = p_stored / s  →  KE' = KE_stored / s²
            #   r' = r/s  →  PE' = -m1*m2/r' = s * PE_stored
            scale = dt_base / s_val  # = 1/s
            KE = np.zeros(len(run))
            for i, body in enumerate(bodies):
                p_x = run[f'p_{body}_x'].values * scale
                p_y = run[f'p_{body}_y'].values * scale
                KE += 0.5 * (p_x**2 + p_y**2) / masses[i]
            PE_rescaled = PE * s_val  # V(q/s) = s * V(q) for gravity
            E = KE + PE_rescaled
            results[method_name] = 100 * (E - E[0]) / np.abs(E[0])
    
    return results, PE


def visualize_sample_trajectories(data_path: str, output_dir: str, num_samples: int = 3, label: str = "Dataset", dt_methods: dict = None, enable_per_traj: bool = False):
    """Visualize sample trajectories with energy plots including dt variants.
    
    Args:
        data_path: Path to dataset
        output_dir: Output directory
        num_samples: Number of trajectories to visualize
        label: Dataset label
        dt_methods: Dict with dt optimization results (virial, drift, variance, per_traj)
    """
    print("\n" + "="*80)
    print(f"5. TRAJECTORY VISUALIZATION - {label}")
    print("="*80)
    
    # Load only a manageable subset (100k rows) for faster performance
    print(f"Loading subset of data for visualization (first 100k rows)...")
    runs_df = load_trajectory_data(data_path, max_rows=100000)
    
    try:
        constants = load_constants(data_path)
        bodies = constants.get('bodies', ['A', 'B'])
        masses = np.array(constants.get('masses', [[1.0, 1.0]])[0])
        dt_base = constants.get('dt', 1.0)
    except (ValueError, KeyError) as e:
        bodies = ['A', 'B']
        masses = np.array([1.0, 1.0])
        dt_base = 1.0
    
    # Compute dt methods if not provided
    if dt_methods is None:
        print("Computing optimal scale factor using all methods (with Δt=1 fixed)...")
        dt_methods = {
            'virial': compute_optimal_scale_factor(data_path, num_samples=10),
            'drift': compute_optimal_scale_by_drift(data_path, num_samples=10),
            'variance': compute_optimal_scale_by_variance(data_path, num_samples=10)
        }
        if enable_per_traj:
            dt_methods['per_traj'] = compute_optimal_scale_per_trajectory(data_path, num_samples=10)
        print(f"  s_opt (virial):   {dt_methods['virial']:.6f}")
        print(f"  s_opt (drift):    {dt_methods['drift']:.6f}")
        print(f"  s_opt (variance): {dt_methods['variance']:.6f}")
    
    bodies = bodies if isinstance(bodies, list) else ['A', 'B']
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all unique run IDs and their lengths
    all_run_ids = runs_df.index.get_level_values('run').unique()
    run_lengths = runs_df.groupby(level=0).size()
    
    # Prefer longer trajectories for visualization (filter for length > 100)
    long_runs = run_lengths[run_lengths > 100].index.values
    
    if len(long_runs) >= num_samples:
        print(f"Found {len(long_runs)} runs with >100 timesteps, sampling from these for better visualization")
        run_ids = np.random.choice(long_runs, num_samples, replace=False)
    else:
        print(f"Only {len(long_runs)} long runs available, sampling from all {len(all_run_ids)} runs")
        run_ids = np.random.choice(all_run_ids, min(num_samples, len(all_run_ids)), replace=False)
    
    for run_id in run_ids:
        run = runs_df.loc[run_id]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Trajectory Analysis: Run {run_id} - {label}', fontsize=14, fontweight='bold', y=0.995)
        
        # Position trajectory
        ax = axes[0, 0]
        for body in bodies:
            q_x = run[f'q_{body}_x'].values
            q_y = run[f'q_{body}_y'].values
            ax.plot(q_x, q_y, label=f'Body {body}', alpha=0.7)
        ax.set_xlabel('x position')
        ax.set_ylabel('y position')
        ax.set_title('Trajectory')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Positions over time
        ax = axes[0, 1]
        for body in bodies:
            q_x = run[f'q_{body}_x'].values
            q_y = run[f'q_{body}_y'].values
            time = np.arange(len(q_x))
            ax.plot(time, q_x, label=f'{body}_x', alpha=0.7)
            ax.plot(time, q_y, label=f'{body}_y', alpha=0.7, linestyle='--')
        ax.set_xlabel('Time step')
        ax.set_ylabel('Position')
        ax.set_title('Positions vs Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Momenta over time
        ax = axes[1, 0]
        for body in bodies:
            p_x = run[f'p_{body}_x'].values
            p_y = run[f'p_{body}_y'].values
            time = np.arange(len(p_x))
            ax.plot(time, p_x, label=f'{body}_px', alpha=0.7)
            ax.plot(time, p_y, label=f'{body}_py', alpha=0.7, linestyle='--')
        ax.set_xlabel('Time step')
        ax.set_ylabel('Momentum')
        ax.set_title('Momenta vs Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Energy drift over time (with all dt optimization methods)
        ax = axes[1, 1]
        
        # Get positions for FD computation
        positions = {}
        for body in bodies:
            positions[body] = {
                'x': run[f'q_{body}_x'].values,
                'y': run[f'q_{body}_y'].values
            }
        
        # Compute PE (same for all methods)
        if len(bodies) == 2:
            q1_x = positions[bodies[0]]['x']
            q1_y = positions[bodies[0]]['y']
            q2_x = positions[bodies[1]]['x']
            q2_y = positions[bodies[1]]['y']
            r = np.sqrt((q2_x - q1_x)**2 + (q2_y - q1_y)**2)
            PE = -masses[0] * masses[1] / (r + 1e-10)
        else:
            PE = np.zeros(len(run))
        
        time = np.arange(len(run))
        is_gt = 'ground' in label.lower() or 'gt' in label.lower()
        
        if is_gt:
            # ===== GROUND TRUTH: Show only K, V, E, and Drift (no optimization) =====
            KE = np.zeros(len(run))
            for i, body in enumerate(bodies):
                p_x = run[f'p_{body}_x'].values
                p_y = run[f'p_{body}_y'].values
                KE += 0.5 * (p_x**2 + p_y**2) / masses[i]
            E_total = KE + PE
            drift = 100 * (E_total - E_total[0]) / np.abs(E_total[0])
            
            # Plot K, V, E on secondary axis
            ax2 = ax.twinx()
            ax2.plot(time, KE, label='Kinetic Energy', linestyle='-', alpha=0.6, color='blue')
            ax2.plot(time, PE, label='Potential Energy', linestyle='-', alpha=0.6, color='orange')
            ax2.plot(time, E_total, label='Total Energy', linestyle='-', alpha=0.8, color='green', linewidth=2)
            ax2.set_ylabel('Energy', color='green')
            ax2.tick_params(axis='y', labelcolor='green')
            ax2.legend(loc='upper right')
            
            # Plot drift on primary axis
            ax.plot(time, drift, label=f'GT Drift = {np.abs(drift).max():.2f}%', 
                   linestyle='solid', linewidth=2.0, alpha=0.9, color='red')
            ax.set_ylabel('Energy Drift (%)', color='red')
            ax.tick_params(axis='y', labelcolor='red')
        else:
            # ===== EXTRACTED (bbox/latent): Show baseline + best optimized =====
            # Method 1: Baseline (s=1, dt=1)
            KE_base = np.zeros(len(run))
            for i, body in enumerate(bodies):
                p_x = run[f'p_{body}_x'].values
                p_y = run[f'p_{body}_y'].values
                KE_base += 0.5 * (p_x**2 + p_y**2) / masses[i]
            E_base = KE_base + PE
            drift_base = 100 * (E_base - E_base[0]) / np.abs(E_base[0])
            ax.plot(time, drift_base, label=f'Baseline (s=1, \u0394t=1): {np.abs(drift_base).max():.2f}%', 
                   linestyle='solid', linewidth=2.0, alpha=0.8, color='gray')
            
            # Compute drift for all optimization methods to find best
            method_drifts = {'baseline': np.abs(drift_base).max()}
            
            # Virial method
            s_virial = dt_methods.get('virial', 1.0)
            KE_virial = np.zeros(len(run))
            for i, body in enumerate(bodies):
                q_x = positions[body]['x'] / s_virial
                q_y = positions[body]['y'] / s_virial
                v_x = np.zeros_like(q_x)
                v_y = np.zeros_like(q_y)
                dt = 1.0
                v_x[0] = (q_x[1] - q_x[0]) / dt
                v_y[0] = (q_y[1] - q_y[0]) / dt
                v_x[1:-1] = (q_x[2:] - q_x[:-2]) / (2.0 * dt)
                v_y[1:-1] = (q_y[2:] - q_y[:-2]) / (2.0 * dt)
                v_x[-1] = (q_x[-1] - q_x[-2]) / dt
                v_y[-1] = (q_y[-1] - q_y[-2]) / dt
                p_x_fd = masses[i] * v_x
                p_y_fd = masses[i] * v_y
                KE_virial += 0.5 * (p_x_fd**2 + p_y_fd**2) / masses[i]
            # PE with rescaled positions
            if len(bodies) == 2:
                q1_x_s = positions[bodies[0]]['x'] / s_virial
                q1_y_s = positions[bodies[0]]['y'] / s_virial
                q2_x_s = positions[bodies[1]]['x'] / s_virial
                q2_y_s = positions[bodies[1]]['y'] / s_virial
                r_s = np.sqrt((q2_x_s - q1_x_s)**2 + (q2_y_s - q1_y_s)**2)
                PE_virial = -masses[0] * masses[1] / (r_s + 1e-10)
            else:
                PE_virial = PE
            E_virial = KE_virial + PE_virial
            drift_virial = 100 * (E_virial - E_virial[0]) / np.abs(E_virial[0])
            method_drifts['virial'] = np.abs(drift_virial).max()
            
            # Drift method
            s_drift = dt_methods.get('drift', 1.0)
            KE_drift = np.zeros(len(run))
            for i, body in enumerate(bodies):
                q_x = positions[body]['x'] / s_drift
                q_y = positions[body]['y'] / s_drift
                v_x = np.zeros_like(q_x)
                v_y = np.zeros_like(q_y)
                dt = 1.0
                v_x[0] = (q_x[1] - q_x[0]) / dt
                v_y[0] = (q_y[1] - q_y[0]) / dt
                v_x[1:-1] = (q_x[2:] - q_x[:-2]) / (2.0 * dt)
                v_y[1:-1] = (q_y[2:] - q_y[:-2]) / (2.0 * dt)
                v_x[-1] = (q_x[-1] - q_x[-2]) / dt
                v_y[-1] = (q_y[-1] - q_y[-2]) / dt
                p_x_fd = masses[i] * v_x
                p_y_fd = masses[i] * v_y
                KE_drift += 0.5 * (p_x_fd**2 + p_y_fd**2) / masses[i]
            if len(bodies) == 2:
                q1_x_s = positions[bodies[0]]['x'] / s_drift
                q1_y_s = positions[bodies[0]]['y'] / s_drift
                q2_x_s = positions[bodies[1]]['x'] / s_drift
                q2_y_s = positions[bodies[1]]['y'] / s_drift
                r_s = np.sqrt((q2_x_s - q1_x_s)**2 + (q2_y_s - q1_y_s)**2)
                PE_drift = -masses[0] * masses[1] / (r_s + 1e-10)
            else:
                PE_drift = PE
            E_drift = KE_drift + PE_drift
            drift_drift_val = 100 * (E_drift - E_drift[0]) / np.abs(E_drift[0])
            method_drifts['drift'] = np.abs(drift_drift_val).max()
            
            # Variance method
            s_variance = dt_methods.get('variance', 1.0)
            KE_variance = np.zeros(len(run))
            for i, body in enumerate(bodies):
                q_x = positions[body]['x'] / s_variance
                q_y = positions[body]['y'] / s_variance
                v_x = np.zeros_like(q_x)
                v_y = np.zeros_like(q_y)
                dt = 1.0
                v_x[0] = (q_x[1] - q_x[0]) / dt
                v_y[0] = (q_y[1] - q_y[0]) / dt
                v_x[1:-1] = (q_x[2:] - q_x[:-2]) / (2.0 * dt)
                v_y[1:-1] = (q_y[2:] - q_y[:-2]) / (2.0 * dt)
                v_x[-1] = (q_x[-1] - q_x[-2]) / dt
                v_y[-1] = (q_y[-1] - q_y[-2]) / dt
                p_x_fd = masses[i] * v_x
                p_y_fd = masses[i] * v_y
                KE_variance += 0.5 * (p_x_fd**2 + p_y_fd**2) / masses[i]
            if len(bodies) == 2:
                q1_x_s = positions[body]['x'] / s_variance
                q1_y_s = positions[bodies[0]]['y'] / s_variance
                q2_x_s = positions[bodies[1]]['x'] / s_variance
                q2_y_s = positions[bodies[1]]['y'] / s_variance
                r_s = np.sqrt((q2_x_s - q1_x_s)**2 + (q2_y_s - q1_y_s)**2)
                PE_variance = -masses[0] * masses[1] / (r_s + 1e-10)
            else:
                PE_variance = PE
            E_variance = KE_variance + PE_variance
            drift_variance_val = 100 * (E_variance - E_variance[0]) / np.abs(E_variance[0])
            method_drifts['variance'] = np.abs(drift_variance_val).max()
            
            # Find best method
            best_method = min(method_drifts, key=method_drifts.get)
            best_drift = method_drifts[best_method]
            
            # Plot best method
            if best_method == 'virial':
                best_s = s_virial
                best_drift_curve = drift_virial
            elif best_method == 'drift':
                best_s = s_drift
                best_drift_curve = drift_drift_val
            elif best_method == 'variance':
                best_s = s_variance
                best_drift_curve = drift_variance_val
            else:
                best_s = 1.0
                best_drift_curve = drift_base
            
            ax.plot(time, best_drift_curve, 
                   label=f'Best ({best_method}, s={best_s:.3f}): {best_drift:.2f}%', 
                   linestyle='dashed', linewidth=2.0, alpha=0.9, color='green')
            ax.set_ylabel('Energy Drift (%)')
        
        ax.axhline(0, color='k', linestyle='-', linewidth=0.5)
        ax.axhline(10, color='r', linestyle='--', alpha=0.3, linewidth=1)
        ax.axhline(-10, color='r', linestyle='--', alpha=0.3, linewidth=1)
        ax.set_xlabel('Time step')
        ax.set_ylabel('Energy Drift (%)')
        ax.set_title('Energy Drift: dt=1, Virial, and Per-Trajectory')
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.985])
        output_path = os.path.join(output_dir, f'trajectory_{label.replace(" ", "_")}_run_{run_id}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path}")


def create_3way_comparison_plots(bbox_data_path: str, latent_data_path: str, gt_data_path: str, output_dir: str, dt_methods_bbox: dict = None, dt_methods_latent: dict = None, dt_methods_gt: dict = None, enable_per_traj: bool = False):
    """Create 3-way comparison plots: bbox vs latent vs ground-truth with dt variants.
    
    Args:
        bbox_data_path: Path to bbox extracted data
        latent_data_path: Path to latent extracted data
        gt_data_path: Path to ground truth data
        output_dir: Output directory
        dt_methods_bbox: Dict with dt optimization results for bbox
        dt_methods_latent: Dict with dt optimization results for latent
        dt_methods_gt: Dict with dt optimization results for GT
    """
    print("\nCreating 3-way comparison visualization...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all three datasets
    try:
        bbox_runs = load_trajectory_data(bbox_data_path, max_rows=None)
        bbox_constants = load_constants(bbox_data_path)
    except Exception as e:
        print(f"  ⚠️  Could not load bbox data: {e}")
        return
    
    try:
        latent_runs = load_trajectory_data(latent_data_path, max_rows=None)
        latent_constants = load_constants(latent_data_path)
    except Exception as e:
        print(f"  ⚠️  Could not load latent data: {e}")
        return
    
    try:
        gt_runs = load_trajectory_data(gt_data_path, max_rows=None)
        gt_constants = load_constants(gt_data_path)
    except Exception as e:
        print(f"  ⚠️  Could not load ground-truth data: {e}")
        return
    
    bodies = bbox_constants.get('bodies', ['A', 'B'])
    masses = np.array(bbox_constants.get('masses', [[1.0, 1.0]])[0])
    dt_base = bbox_constants.get('dt', 1.0)
    
    # Compute optimal dt methods if not provided
    if dt_methods_bbox is None:
        print("  Computing optimal dt methods for bbox...")
        dt_methods_bbox = {
            'virial': compute_optimal_scale_factor(bbox_data_path, num_samples=10),
            'drift': compute_optimal_scale_by_drift(bbox_data_path, num_samples=10),
            'variance': compute_optimal_scale_by_variance(bbox_data_path, num_samples=10)
        }
        if enable_per_traj:
            dt_methods_bbox['per_traj'] = compute_optimal_scale_per_trajectory(bbox_data_path, num_samples=10)
    if dt_methods_latent is None:
        print("  Computing optimal scale methods for latent...")
        dt_methods_latent = {
            'virial': compute_optimal_scale_factor(latent_data_path, num_samples=10),
            'drift': compute_optimal_scale_by_drift(latent_data_path, num_samples=10),
            'variance': compute_optimal_scale_by_variance(latent_data_path, num_samples=10)
        }
        if enable_per_traj:
            dt_methods_latent['per_traj'] = compute_optimal_scale_per_trajectory(latent_data_path, num_samples=10)
    # GT: No optimization needed - use None to signal different plotting
    if dt_methods_gt is None:
        dt_methods_gt = None  # Signal that GT doesn't use optimization
    
    print(f"  Bbox s: virial={dt_methods_bbox['virial']:.4f}, variance={dt_methods_bbox['variance']:.4f}")
    print(f"  Latent s: virial={dt_methods_latent['virial']:.4f}, variance={dt_methods_latent['variance']:.4f}")
    print(f"  GT: No optimization (using true momenta)")
    
    # Pick common run IDs - prefer longer trajectories
    bbox_ids = bbox_runs.index.get_level_values('run').unique()
    latent_ids = latent_runs.index.get_level_values('run').unique()
    gt_ids = gt_runs.index.get_level_values('run').unique()
    
    # Get run lengths and filter for longer trajectories (>100 timesteps)
    bbox_lengths = bbox_runs.groupby(level=0).size()
    latent_lengths = latent_runs.groupby(level=0).size()
    gt_lengths = gt_runs.groupby(level=0).size()
    
    long_bbox = sorted(bbox_lengths[bbox_lengths > 100].index)
    long_latent = sorted(latent_lengths[latent_lengths > 100].index)
    long_gt = sorted(gt_lengths[gt_lengths > 100].index)
    
    # Find common long runs across all three datasets
    common_long_runs = sorted(set(long_bbox) & set(long_latent) & set(long_gt))
    
    if len(common_long_runs) >= 3:
        print(f"  Found {len(common_long_runs)} common runs with >100 timesteps")
        # Use tuples with same run ID for all three
        run_triples = [(rid, rid, rid) for rid in common_long_runs[:3]]
    elif len(long_bbox) >= 3 and len(long_latent) >= 3 and len(long_gt) >= 3:
        print(f"  No common run IDs with long trajectories across all three datasets")
        print(f"  Using independent long runs: {len(long_bbox)} bbox, {len(long_latent)} latent, {len(long_gt)} groundtruth")
        # Pair up long runs independently from each dataset
        run_triples = list(zip(long_bbox[:3], long_latent[:3], long_gt[:3]))
    else:
        print(f"  ⚠️  Insufficient long trajectories in one or more datasets")
        print(f"     Bbox long runs: {len(long_bbox)}, Latent long runs: {len(long_latent)}, GT long runs: {len(long_gt)}")
        return
    
    for bbox_run_id, latent_run_id, gt_run_id in run_triples:
        bbox_run = bbox_runs.loc[bbox_run_id]
        latent_run = latent_runs.loc[latent_run_id]
        # Load GT directly from test features (no downsampling)
        gt_run = gt_runs.loc[gt_run_id]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        # Create title showing run IDs
        if bbox_run_id == latent_run_id == gt_run_id:
            title_suffix = f'Run {bbox_run_id}'
        else:
            title_suffix = f'Runs: Bbox={bbox_run_id}, Latent={latent_run_id}, GT={gt_run_id}'
        fig.suptitle('Trajectory Extraction Alignment Comparison: Bbox, Latent and GT', fontsize=16, fontweight='bold', y=0.998)
        
        # === ROW 1: TRAJECTORIES ===
        for idx, (run, title) in enumerate([(bbox_run, 'Bbox Extraction'), 
                                             (latent_run, 'Latent Extraction'), 
                                             (gt_run, 'Ground Truth')]):
            ax = axes[0, idx]
            for body in bodies:
                ax.plot(run[f'q_{body}_x'], run[f'q_{body}_y'], 'o-', label=body, markersize=2, alpha=0.7)
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('x position')
            ax.set_ylabel('y position')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
        
        # === ROW 2: ENERGY DRIFT WITH DT VARIANTS ===
        for idx, (run, title, dt_methods, is_gt) in enumerate([
            (bbox_run, 'Bbox', dt_methods_bbox, False),
            (latent_run, 'Latent', dt_methods_latent, False),
            (gt_run, 'Ground Truth', dt_methods_gt, True)
        ]):
            ax = axes[1, idx]
            
            # Get positions for FD computation
            positions = {}
            for body in bodies:
                positions[body] = {
                    'x': run[f'q_{body}_x'].values,
                    'y': run[f'q_{body}_y'].values
                }
            
            # Compute PE (same for all)
            if len(bodies) == 2:
                q1_x = positions[bodies[0]]['x']
                q1_y = positions[bodies[0]]['y']
                q2_x = positions[bodies[1]]['x']
                q2_y = positions[bodies[1]]['y']
                r = np.sqrt((q2_x - q1_x)**2 + (q2_y - q1_y)**2)
                PE = -masses[0] * masses[1] / (r + 1e-10)
            else:
                PE = np.zeros(len(run))
            
            time = np.arange(len(run))
            
            # Compute energy drifts using helper function
            drift_results, PE = compute_energy_drift_with_dt_methods(
                run, positions, dt_methods, masses, bodies, dt_base, is_gt
            )
            
            # Plot baseline (dt=1)
            drift_base = drift_results['baseline']
            base_label = 'True p' if is_gt else 'dt=1'
            ax.plot(time, drift_base, label=f'{base_label} ({np.abs(drift_base).max():.1f}%)', 
                   linestyle='solid', linewidth=2.0, alpha=0.9)
            
            # Plot method-specific drifts - FILTERED to show only dt=1, Virial, and per-trajectory
            if is_gt:
                # GT: FD methods - show only dt=1 (baseline, already plotted) and virial
                if 'virial' in drift_results and dt_methods is not None:
                    drift = drift_results['virial']
                    dt_val = dt_methods.get('virial', dt_base)
                    ax.plot(time, drift, label=f'FD Virial={dt_val:.4f} ({np.abs(drift).max():.1f}%)', 
                           linestyle='dashed', linewidth=1.5, alpha=0.7)
            else:
                # Extracted: Rescaling methods - show only virial
                if 'virial' in drift_results and dt_methods is not None:
                    drift = drift_results['virial']
                    dt_val = dt_methods.get('virial', dt_base)
                    ax.plot(time, drift, label=f'Virial dt={dt_val:.4f} ({np.abs(drift).max():.1f}%)', 
                           linestyle='dashed', linewidth=1.5, alpha=0.7)
            
            # Per-trajectory optimization: Show mean ± std across all per-traj runs
            per_traj_info = dt_methods.get('per_traj', {}) if dt_methods is not None else {}
            if per_traj_info and len(per_traj_info.get('run_ids', [])) > 0:
                # Compute drift for each trajectory in per_traj with its optimal dt
                all_per_traj_drifts = []
                min_length = len(run)  # Use current run length as reference
                
                # Load runs from dataset to compute per-traj drifts
                try:
                    if is_gt:
                        per_traj_df = load_trajectory_data(gt_data_path, max_rows=None)
                    elif 'bbox' in title.lower():
                        per_traj_df = load_trajectory_data(bbox_data_path, max_rows=None)
                    else:
                        per_traj_df = load_trajectory_data(latent_data_path, max_rows=None)
                    
                    for traj_idx, traj_run_id in enumerate(per_traj_info['run_ids']):
                        traj_run = per_traj_df.loc[traj_run_id]
                        traj_dt = per_traj_info['dt_values'][traj_idx]
                        
                        # Compute positions
                        traj_pos = {}
                        for body in bodies:
                            traj_pos[body] = {
                                'x': traj_run[f'q_{body}_x'].values,
                                'y': traj_run[f'q_{body}_y'].values
                            }
                        
                        # Compute drift with per-traj dt
                        traj_drift_results, _ = compute_energy_drift_with_dt_methods(
                            traj_run, traj_pos, {'variance': traj_dt}, masses, bodies, dt_base, is_gt
                        )
                        traj_drift = traj_drift_results['variance']
                        
                        # Truncate or pad to match reference length
                        if len(traj_drift) >= min_length:
                            all_per_traj_drifts.append(traj_drift[:min_length])
                        else:
                            # Pad with last value
                            padded = np.pad(traj_drift, (0, min_length - len(traj_drift)), mode='edge')
                            all_per_traj_drifts.append(padded)
                    
                    # Compute mean and std
                    if len(all_per_traj_drifts) > 0:
                        all_per_traj_drifts = np.array(all_per_traj_drifts)
                        mean_drift = np.mean(all_per_traj_drifts, axis=0)
                        std_drift = np.std(all_per_traj_drifts, axis=0)
                        
                        time_ref = np.arange(min_length)
                        ax.plot(time_ref, mean_drift, label=f'Per-traj (μ={np.abs(mean_drift).max():.1f}%)', 
                               linestyle=(0, (3, 1, 1, 1)), linewidth=1.8, alpha=0.8, color='purple')
                        ax.fill_between(time_ref, mean_drift - std_drift, mean_drift + std_drift, 
                                      alpha=0.2, color='purple', label='Per-traj ±σ')
                except Exception as e:
                    print(f"    Warning: Could not compute per-traj statistics: {e}")
            
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
            ax.axhline(y=10, color='r', linestyle='--', alpha=0.3, linewidth=0.8)
            ax.axhline(y=-10, color='r', linestyle='--', alpha=0.3, linewidth=0.8)
            ax.set_title(f'{title} Energy Drift', fontweight='bold')
            ax.set_xlabel('Time step')
            ax.set_ylabel('Energy Drift (%)')
            ax.legend(fontsize=6, loc='best', ncol=1)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave more space for title
        # Create filename based on run IDs
        if bbox_run_id == latent_run_id == gt_run_id:
            filename = f'3way_comparison_run_{bbox_run_id}.png'
        else:
            filename = f'3way_comparison_bbox{bbox_run_id}_latent{latent_run_id}_gt{gt_run_id}.png'
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {output_path}")


def create_comparison_plots(extracted_data_path: str, groundtruth_data_path: str, output_dir: str, label1: str = "Extracted", label2: str = "Ground-Truth", dt_eff1: float = None, dt_eff2: float = None):
    """Create side-by-side comparison plots with dt variants.
    
    Args:
        extracted_data_path: Path to first dataset
        groundtruth_data_path: Path to second dataset
        output_dir: Output directory
        label1: Label for first dataset
        label2: Label for second dataset
        dt_eff1: Optimal dt for dataset 1 (computed if None)
        dt_eff2: Optimal dt for dataset 2 (computed if None)
    """
    print("\n" + "="*80)
    print("COMPARISON VISUALIZATION")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load both datasets
    ext_runs = load_trajectory_data(extracted_data_path, max_rows=None)
    gt_runs = load_trajectory_data(groundtruth_data_path, max_rows=None)
    
    try:
        ext_constants = load_constants(extracted_data_path)
        bodies = ext_constants.get('bodies', ['A', 'B'])
        masses = np.array(ext_constants.get('masses', [[1.0, 1.0]])[0])
    except:
        bodies = ['A', 'B']
        masses = np.array([1.0, 1.0])
    
    # Get run lengths and filter for longer trajectories
    ext_lengths = ext_runs.groupby(level=0).size()
    gt_lengths = gt_runs.groupby(level=0).size()
    
    long_ext = sorted(ext_lengths[ext_lengths > 100].index)
    long_gt = sorted(gt_lengths[gt_lengths > 100].index)
    
    # Try to find common long runs first
    common_long = sorted(set(long_ext) & set(long_gt))
    
    if len(common_long) >= 3:
        print(f"  Found {len(common_long)} common runs with >100 timesteps")
        # Use paired run IDs (same ID in both datasets)
        pairs = [(rid, rid) for rid in common_long[:3]]
    elif len(long_ext) >= 3 and len(long_gt) >= 3:
        print(f"  No common run IDs with long trajectories")
        print(f"  Using independent long runs: {len(long_ext)} from {label1}, {len(long_gt)} from {label2}")
        # Pair up long runs from each dataset independently
        pairs = list(zip(long_ext[:3], long_gt[:3]))
    else:
        print(f"⚠️  Insufficient long trajectories in one or both datasets")
        print(f"   {label1} long runs: {len(long_ext)}, {label2} long runs: {len(long_gt)}")
        return
    
    for ext_run_id, gt_run_id in pairs:
        ext_run = ext_runs.loc[ext_run_id]
        # Load GT directly from test features (no downsampling)
        gt_run = gt_runs.loc[gt_run_id]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        title_suffix = f'Run {ext_run_id} vs {gt_run_id}' if ext_run_id != gt_run_id else f'Run {ext_run_id}'
        fig.suptitle(f'Comparison: {label1} vs {label2} ({title_suffix})', fontsize=14, fontweight='bold', y=0.995)
        
        # === EXTRACTED (left column) ===
        # Trajectory
        ax = axes[0, 0]
        for body in bodies:
            q_x = ext_run[f'q_{body}_x'].values
            q_y = ext_run[f'q_{body}_y'].values
            ax.plot(q_x, q_y, label=f'Body {body}', alpha=0.7)
        ax.set_title(f'{label1} - Trajectory')
        ax.set_xlabel('x position')
        ax.set_ylabel('y position')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Energy
        ax = axes[1, 0]
        KE = np.zeros(len(ext_run))
        for i, body in enumerate(bodies):
            p_x = ext_run[f'p_{body}_x'].values
            p_y = ext_run[f'p_{body}_y'].values
            KE += 0.5 * (p_x**2 + p_y**2) / masses[i]
        if len(bodies) == 2:
            q1_x = ext_run[f'q_{bodies[0]}_x'].values
            q1_y = ext_run[f'q_{bodies[0]}_y'].values
            q2_x = ext_run[f'q_{bodies[1]}_x'].values
            q2_y = ext_run[f'q_{bodies[1]}_y'].values
            r = np.sqrt((q2_x - q1_x)**2 + (q2_y - q1_y)**2)
            PE = -1.0 * masses[0] * masses[1] / (r + 1e-10)
        else:
            PE = np.zeros(len(ext_run))
        E_ext = KE + PE
        drift_ext = 100 * (E_ext - E_ext[0]) / np.abs(E_ext[0])
        
        time = np.arange(len(E_ext))
        ax.plot(time, E_ext, label='Total', linewidth=2)
        ax.plot(time, KE, label='Kinetic', alpha=0.7)
        ax.plot(time, PE, label='Potential', alpha=0.7)
        ax.set_title(f'{label1} - Energy (drift: {np.abs(drift_ext).max():.1f}%)')
        ax.set_xlabel('Time step')
        ax.set_ylabel('Energy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # === GROUND-TRUTH (middle column) ===
        # Trajectory
        ax = axes[0, 1]
        for body in bodies:
            q_x = gt_run[f'q_{body}_x'].values
            q_y = gt_run[f'q_{body}_y'].values
            ax.plot(q_x, q_y, label=f'Body {body}', alpha=0.7)
        ax.set_title(f'{label2} - Trajectory')
        ax.set_xlabel('x position')
        ax.set_ylabel('y position')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Energy
        ax = axes[1, 1]
        KE = np.zeros(len(gt_run))
        for i, body in enumerate(bodies):
            p_x = gt_run[f'p_{body}_x'].values
            p_y = gt_run[f'p_{body}_y'].values
            KE += 0.5 * (p_x**2 + p_y**2) / masses[i]
        if len(bodies) == 2:
            q1_x = gt_run[f'q_{bodies[0]}_x'].values
            q1_y = gt_run[f'q_{bodies[0]}_y'].values
            q2_x = gt_run[f'q_{bodies[1]}_x'].values
            q2_y = gt_run[f'q_{bodies[1]}_y'].values
            r = np.sqrt((q2_x - q1_x)**2 + (q2_y - q1_y)**2)
            PE = -1.0 * masses[0] * masses[1] / (r + 1e-10)
        else:
            PE = np.zeros(len(gt_run))
        E_gt = KE + PE
        drift_gt = 100 * (E_gt - E_gt[0]) / np.abs(E_gt[0])
        
        time = np.arange(len(E_gt))
        ax.plot(time, E_gt, label='Total', linewidth=2)
        ax.plot(time, KE, label='Kinetic', alpha=0.7)
        ax.plot(time, PE, label='Potential', alpha=0.7)
        ax.set_title(f'{label2} - Energy (drift: {np.abs(drift_gt).max():.1f}%)')
        ax.set_xlabel('Time step')
        ax.set_ylabel('Energy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # === DIFFERENCE (right column) ===
        # Trajectory overlay
        ax = axes[0, 2]
        for body in bodies:
            ext_x = ext_run[f'q_{body}_x'].values
            ext_y = ext_run[f'q_{body}_y'].values
            gt_x = gt_run[f'q_{body}_x'].values
            gt_y = gt_run[f'q_{body}_y'].values
            ax.plot(ext_x, ext_y, label=f'{label1[:3]}-{body}', alpha=0.5, linestyle='--')
            ax.plot(gt_x, gt_y, label=f'{label2[:3]}-{body}', alpha=0.5)
        ax.set_title('Overlay Comparison')
        ax.set_xlabel('x position')
        ax.set_ylabel('y position')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Energy drift comparison
        ax = axes[1, 2]
        # Align lengths
        min_len = min(len(drift_ext), len(drift_gt))
        time = np.arange(min_len)
        ax.plot(time, drift_ext[:min_len], label=label1, linewidth=2, alpha=0.8)
        ax.plot(time, drift_gt[:min_len], label=label2, linewidth=2, alpha=0.8)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.axhline(y=10, color='r', linestyle='--', alpha=0.3, label='10% threshold')
        ax.axhline(y=-10, color='r', linestyle='--', alpha=0.3)
        ax.set_title('Energy Drift Comparison')
        ax.set_xlabel('Time step')
        ax.set_ylabel('Energy Drift (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.985])
        filename = f'comparison_run_{ext_run_id}_vs_{gt_run_id}.png' if ext_run_id != gt_run_id else f'comparison_run_{ext_run_id}.png'
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced GHNN data diagnostics with 3-way comparison (bbox, latent, ground-truth)'
    )
    parser.add_argument(
        '--extracted_bbox_data', type=str, default=None,
        help='Path to bbox-extracted training.h5.1 file (from DDLP bbox method)'
    )
    parser.add_argument(
        '--extracted_latent_data', type=str, default=None,
        help='Path to latent-extracted training.h5.1 file (from DDLP latent method)'
    )
    parser.add_argument(
        '--groundtruth_data', type=str, default=None,
        help='Path to ground-truth training.h5.1 file (from physics simulation)'
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Output directory for visualizations (default: diagnostics_3way_comparative/)'
    )
    parser.add_argument(
        '--enable_per_traj_optimization', action='store_true', default=False,
        help='Enable expensive per-trajectory scale optimization (default: False)'
    )
    
    args = parser.parse_args()
    
    # Collect all provided datasets
    datasets = {}
    if args.extracted_bbox_data and os.path.exists(args.extracted_bbox_data):
        datasets['bbox'] = args.extracted_bbox_data
    if args.extracted_latent_data and os.path.exists(args.extracted_latent_data):
        datasets['latent'] = args.extracted_latent_data
    if args.groundtruth_data and os.path.exists(args.groundtruth_data):
        datasets['groundtruth'] = args.groundtruth_data
    
    # Validate we have at least one dataset
    if len(datasets) == 0:
        print("❌ ERROR: No valid data files provided!")
        print("Please provide at least one of:")
        print("  --extracted_bbox_data")
        print("  --extracted_latent_data")
        print("  --groundtruth_data")
        return 1
    
    # Set default groundtruth if not provided but try standard location
    if 'groundtruth' not in datasets:
        default_gt = "/data2/users/lr4617/data_twobody_tries/data_try_elliptic/physics_trajectories/GHNN/Data_Circular_2Body_T_720_radius_1_nu_0.05/circ_2body_training.h5.1"
        if os.path.exists(default_gt):
            datasets['groundtruth'] = default_gt
            print(f"Using default ground-truth data: {default_gt}")
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = '/data2/users/lr4617/data_twobody_tries/extraction_analysis/diagnostics_3way_comparative'
    
    # print summary of what we will analyze
    print("\n" + "="*80)
    print("ENHANCED GHNN DATA DIAGNOSTICS WITH 3-WAY COMPARISON")
    print("="*80)
    print(f"Datasets to analyze:")
    for label, path in datasets.items():
        print(f"  {label.upper():<12s}: {path}")
    print(f"Output directory:  {args.output_dir}")
    print("="*80)
    
    # === ANALYZE ALL DATASETS ===
    results = {}
    dt_eff_values = {}  # Store optimal dt for each dataset
    
    for label, data_path in datasets.items():
        
        # Reset per-iteration variables to avoid stale values from previous datasets
        avg_optimized_drift = None
        improvement = None
        
        print("\n" + "#"*80)
        print(f"# ANALYZING {label.upper()} DATASET")
        print("#"*80)
        
        # check data integrity and basic stats
        integrity_ok, stats = check_data_integrity(data_path, label.capitalize())
        
        if integrity_ok:
            # Check energy conservation with stored momentum
            avg_drift, _ = check_energy_conservation(
                data_path, num_samples=5, label=label.capitalize(), 
            )
            
            # Compute optimal scale factor only for extracted data (not for ground truth)
            if label != 'groundtruth':
                print(f"\nComputing optimal scale factor for {label}...")
                print(f"  Exploring s ∈ [0.001, 5.0] with 50 points (logspace)")
                dt_eff_virial = compute_optimal_scale_factor(data_path, num_samples=10)
                dt_eff_drift = compute_optimal_scale_by_drift(data_path, num_samples=10)
                dt_eff_variance = compute_optimal_scale_by_variance(data_path, num_samples=10)
                
                # store grid-search optimized dt values for this dataset
                dt_eff_values[label] = {
                    'virial': dt_eff_virial,
                    'drift': dt_eff_drift,
                    'variance': dt_eff_variance
                }
                
                # Optionally compute per-trajectory optimized dt values (EXPENSIVE)
                if args.enable_per_traj_optimization:
                    dt_per_traj = compute_optimal_scale_per_trajectory(data_path, num_samples=10)
                    dt_eff_values[label]['per_traj'] = dt_per_traj
                    print(f"  s_per_traj: mean={np.mean(dt_per_traj['s_values']):.6f}, std={np.std(dt_per_traj['s_values']):.6f}")
                else:
                    dt_eff_values[label]['per_traj'] = None
                
                # print optimized dt values for this dataset
                print(f"  s_opt (virial):   {dt_eff_virial:.6f}")
                print(f"  s_opt (drift):    {dt_eff_drift:.6f}")
                print(f"  s_opt (variance): {dt_eff_variance:.6f}")
                
                # Re-compute energy drift with optimized parameters on the SAME 5 trajectories
                # to get accurate before/after comparison
                print(f"\nComputing optimized energy drift for comparison...")
                runs_opt = load_trajectory_data(data_path, max_rows=100000)
                run_ids_opt_available = runs_opt.index.get_level_values('run').unique()
                
                # Use same 5 trajectories as check_energy_conservation (re-sample with same logic)
                run_ids_opt = np.random.choice(run_ids_opt_available, min(5, len(run_ids_opt_available)), replace=False)
                
                optimized_drifts = []
                for run_id in run_ids_opt:
                    run = runs_opt.loc[run_id]
                    positions = {}
                    for body in ['A', 'B']:
                        positions[body] = {
                            'x': run[f'q_{body}_x'].values,
                            'y': run[f'q_{body}_y'].values
                        }
                    
                    drift_results, _ = compute_energy_drift_with_dt_methods(
                        run, positions, dt_eff_values[label], np.array([1.0, 1.0]), ['A', 'B'], 1.0, is_gt=False
                    )
                    
                    # Find best method drift
                    best_drift = min([np.abs(drift_results[m]).max() for m in drift_results if m != 'baseline'])
                    optimized_drifts.append(best_drift)
                
                avg_optimized_drift = np.mean(optimized_drifts)
                improvement = 100 * (avg_drift - avg_optimized_drift) / avg_drift
                
                print(f"  Baseline drift: {avg_drift:.2%}")
                print(f"  Optimized drift: {avg_optimized_drift:.2%}")
                print(f"  Improvement: {improvement:.1f}%")
            else:
                # For ground truth, no optimization needed
                print(f"\nSkipping scale optimization for ground truth (using true momenta)")
                dt_eff_values[label] = None
            
            label_output = os.path.join(args.output_dir, 'single_method_plots', label)
            visualize_sample_trajectories(data_path, label_output, num_samples=3, label=label.capitalize(), dt_methods=dt_eff_values[label], enable_per_traj=args.enable_per_traj_optimization)
            
            results[label] = {
                'integrity_ok': integrity_ok,
                'avg_energy_drift': avg_drift,
                'avg_energy_drift_optimized': avg_optimized_drift,
                'drift_improvement_percent': improvement,
                'stats': stats
            }
        else:
            results[label] = {
                'integrity_ok': False,
                'error': 'Data integrity check failed'
            }
    
    # === CREATE COMPARISON VISUALIZATIONS ===
    if len(datasets) >= 2:
        print("\n" + "="*80)
        print("CREATING COMPARISON VISUALIZATIONS")
        print("="*80)
        
        # Create comparison plots (works with 2 or 3 datasets)
        print(f"\n  Creating comparison visualization ({len(datasets)} datasets)...")
        try:
            threeway_output = os.path.join(args.output_dir, 'comparison_plots', 'multi_comparison')
            create_3way_comparison_plots(
                datasets.get('bbox'), 
                datasets.get('latent'), 
                datasets.get('groundtruth'),
                threeway_output,
                dt_methods_bbox=dt_eff_values.get('bbox'),
                dt_methods_latent=dt_eff_values.get('latent'),
                dt_methods_gt=dt_eff_values.get('groundtruth'),
                enable_per_traj=args.enable_per_traj_optimization
            )
        except Exception as e:
            print(f"    ⚠️  Error: {e}")
            import traceback
            traceback.print_exc()
    
    # === SAVE ENERGY DRIFT RESULTS TO JSON ===
    print("\n" + "="*80)
    print("SAVING ENERGY DRIFT RESULTS")
    print("="*80)
    
    # Collect all energy drift results
    energy_drift_results = {}
    for label in datasets.keys():
        data_path = datasets.get(label)
        dt_methods = dt_eff_values.get(label)
        
        if data_path is None:
            continue
        
        # Skip if dt_methods is None (ground truth doesn't need optimization)
        if dt_methods is None:
            print(f"  Skipping energy drift collection for {label} (no dt optimization performed)")
            energy_drift_results[label] = {'note': 'Ground truth - no dt optimization needed'}
            continue
            
        try:
            runs_df = load_trajectory_data(data_path, max_rows=None)  # Load all data for final comparison
            try:
                constants = load_constants(data_path)
                bodies = constants.get('bodies', ['A', 'B'])
                masses = np.array(constants.get('masses', [[1.0, 1.0]])[0])
                dt_base = constants.get('dt', 1.0)
            except (ValueError, KeyError):
                bodies = ['A', 'B']
                masses = np.array([1.0, 1.0])
                dt_base = 1.0
            
            run_ids = runs_df.index.get_level_values('run').unique()[:10]  # Sample 10 trajectories
            is_gt = (label == 'groundtruth')
            
            energy_drift_results[label] = {
                'dt_base': dt_base,
                'masses': masses.tolist(),
                'methods': {},
                'trajectories': {}
            }
            
            # Store dt values for each method
            energy_drift_results[label]['methods']['dt=1'] = dt_base
            energy_drift_results[label]['methods']['virial'] = dt_methods.get('virial', dt_base)
            energy_drift_results[label]['methods']['drift'] = dt_methods.get('drift', dt_base)
            energy_drift_results[label]['methods']['variance'] = dt_methods.get('variance', dt_base)
            
            # Compute energy drift for each trajectory with each method
            all_baseline_drifts = []
            all_best_method_drifts = []
            
            for run_id in run_ids:
                run = runs_df.loc[run_id]
                
                # Get positions
                positions = {}
                for body in bodies:
                    positions[body] = {
                        'x': run[f'q_{body}_x'].values,
                        'y': run[f'q_{body}_y'].values
                    }
                
                # Compute drift for all methods
                drift_results, _ = compute_energy_drift_with_dt_methods(
                    run, positions, dt_methods, masses, bodies, dt_base, is_gt
                )
                
                # Store results
                traj_results = {}
                baseline_drift = None
                best_drift = None
                
                for method_name, drift_array in drift_results.items():
                    max_abs_drift = float(np.abs(drift_array).max())
                    mean_abs_drift = float(np.abs(drift_array).mean())
                    traj_results[method_name] = {
                        'max_absolute_drift_percent': max_abs_drift,
                        'mean_absolute_drift_percent': mean_abs_drift
                    }
                    
                    # Track baseline and best method for improvement calculation
                    if method_name == 'baseline':
                        baseline_drift = max_abs_drift
                    elif best_drift is None or max_abs_drift < best_drift:
                        best_drift = max_abs_drift
                
                # Store per-trajectory improvement
                if baseline_drift is not None and best_drift is not None:
                    all_baseline_drifts.append(baseline_drift)
                    all_best_method_drifts.append(best_drift)
                
                energy_drift_results[label]['trajectories'][str(run_id)] = traj_results
            
            # Calculate overall improvement statistics
            if all_baseline_drifts and all_best_method_drifts:
                avg_baseline = np.mean(all_baseline_drifts)
                avg_best = np.mean(all_best_method_drifts)
                improvement_percent = 100 * (avg_baseline - avg_best) / avg_baseline
                
                energy_drift_results[label]['summary'] = {
                    'average_baseline_drift_percent': float(avg_baseline),
                    'average_optimized_drift_percent': float(avg_best),
                    'improvement_percent': float(improvement_percent),
                    'note': f'Optimization reduced drift by {improvement_percent:.1f}%'
                }
            
            # Add per-trajectory dt information if available
            per_traj_info = dt_methods.get('per_traj', {})
            if per_traj_info and 'run_ids' in per_traj_info:
                energy_drift_results[label]['per_trajectory_dts'] = {
                    str(rid): float(dt_val) 
                    for rid, dt_val in zip(per_traj_info['run_ids'], per_traj_info['dt_values'])
                }
                
        except Exception as e:
            print(f"  ⚠️  Error collecting energy drift for {label}: {e}")
            energy_drift_results[label] = {'error': str(e)}
    
    # Save to JSON
    json_output_path = os.path.join(args.output_dir, 'energy_drift_results.json')
    with open(json_output_path, 'w') as f:
        json.dump(energy_drift_results, f, indent=2)
    
    print(f"Energy drift results saved to: {json_output_path}")
    
    # === SUMMARY AND RECOMMENDATIONS ===
    print("\n" + "="*80)
    print("SUMMARY AND RECOMMENDATIONS")
    print("="*80)
    
    for label, result in results.items():
        if result['integrity_ok']:
            dt_methods = dt_eff_values.get(label, None)
            print(f"\n{label.upper()}:")
            
            # Show drift with optimization info if available
            if result['avg_energy_drift_optimized'] is not None:
                print(f"  Energy drift (baseline):  {result['avg_energy_drift']:.2%}")
                print(f"  Energy drift (optimized): {result['avg_energy_drift_optimized']:.2%}")
                print(f"  Improvement:              {result['drift_improvement_percent']:.1f}%")
            else:
                print(f"  Energy drift:     {result['avg_energy_drift']:.2%}")
            
            print(f"  Pos/Mom ratio:    {result['pos_mom_ratio']:.2f}")
            if dt_methods is not None:
                print(f"  Optimal dt methods:")
                print(f"    - Virial:   {dt_methods['virial']:.6f}")
                print(f"    - Drift:    {dt_methods['drift']:.6f}")
                print(f"    - Variance: {dt_methods['variance']:.6f}")
                if 'per_traj' in dt_methods and dt_methods['per_traj'] is not None:
                    per_traj_dts = dt_methods['per_traj']['dt_values']
                    print(f"    - Per-traj: mean={np.mean(per_traj_dts):.6f}, std={np.std(per_traj_dts):.6f}")
            
            # Use optimized drift if available for recommendations
            drift_to_use = result['avg_energy_drift_optimized'] if result['avg_energy_drift_optimized'] is not None else result['avg_energy_drift']
            
            if drift_to_use < 0.10:
                print(f"  → ✅ Suitable for GHNN (symplectic architecture)")
            elif drift_to_use < 0.30:
                print(f"  → ⚠️  Moderate drift, use MLP architecture")
            else:
                print(f"  → ❌ High drift, consider physics-informed losses")
        else:
            print(f"\n{label.upper()}: ❌ Data integrity check failed")
    
    # Compare extraction methods if both present
    if 'bbox' in results and 'latent' in results:
        if results['bbox']['integrity_ok'] and results['latent']['integrity_ok']:
            bbox_drift = results['bbox']['avg_energy_drift']
            latent_drift = results['latent']['avg_energy_drift']
            improvement = (bbox_drift - latent_drift) / bbox_drift if bbox_drift > 0 else 0
            
            print(f"\n" + "="*80)
            print("EXTRACTION METHOD COMPARISON")
            print("="*80)
            print(f"Bbox extraction:   {bbox_drift:.2%} energy drift")
            print(f"Latent extraction: {latent_drift:.2%} energy drift")
            print(f"Improvement:       {improvement:.1%} {'better' if improvement > 0 else 'worse'}")
            
            if latent_drift < bbox_drift and latent_drift < 0.10:
                print(f"\n✅ Latent extraction shows {improvement:.0%} improvement!")
                print(f"   Recommendation: Use latent extraction for GHNN training")
            elif latent_drift < 0.10:
                print(f"\n✅ Both methods suitable, latent preserves model dynamics")
            else:
                print(f"\n⚠️  Neither method achieves <10% drift for GHNN")
    
    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)
    print(f"All visualizations saved to: {args.output_dir}")
    
    return 0


if __name__ == '__main__':
    exit(main())
