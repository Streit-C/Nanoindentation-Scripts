# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 11:20:46 2025

@author: streit
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from pathlib import Path

sections = ['I','II','III','IV']
columns = ['A','B','C','D']

def load_and_validate(file_path):
    """Load and validate a nanoindentation CSV file."""
    try:
        df = pd.read_csv(file_path, usecols=['DEPTH', 'HARDNESS', 'MODULUS'])
        df.columns = [col.upper().strip() for col in df.columns]
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df[
            (df['DEPTH'].notna()) &
            (df['HARDNESS'] > 0) &
            (df['MODULUS'] > 0) &
            (df['DEPTH'] > 0)
        ]
        return df.sort_values('DEPTH').reset_index(drop=True) if not df.empty else None
    except Exception as e:
        print(f"  [ERROR] Skipping {os.path.basename(file_path)}: {str(e)}")
        return None

def write_empty_indents_log(empty_indents, log_path):
    """Write a log of all empty or invalid indents to a text file."""
    with open(log_path, "w") as f:
        f.write("Empty or invalid indents detected during processing:\n\n")
        if not empty_indents:
            f.write("None. All indents contained valid data.\n")
        else:
            for path in empty_indents:
                f.write(path + "\n")
    print(f"\nEmpty indent log written to: {log_path}")

def remove_outliers(data_arrays, threshold=3.0):
    """
    Remove outlier curves using z-score thresholding across depth points.
    
    Args:
        data_arrays (list): List of numpy arrays containing individual curves
        threshold (float): Z-score threshold for outlier removal (default: 3.0)
    
    Returns:
        tuple: (filtered_data, outlier_mask) where:
            filtered_data - list of non-outlier curves
            outlier_mask - boolean mask indicating removed outliers
    """
    data_stack = np.array(data_arrays)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_curve = np.nanmean(data_stack, axis=0)
        std_curve = np.nanstd(data_stack, axis=0, ddof=1)  # Use ddof=1 for sample std

    # Handle all-NaN slices
    valid_mask = ~np.isnan(mean_curve) & ~np.isnan(std_curve)
    z_scores = np.zeros_like(data_stack)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        z_scores[:, valid_mask] = np.abs(
            (data_stack[:, valid_mask] - mean_curve[valid_mask]) 
            / std_curve[valid_mask]
        )

    outlier_mask = np.any(z_scores > threshold, axis=1)
    filtered_data = [arr for arr, is_outlier in zip(data_arrays, outlier_mask) 
                    if not is_outlier]
    
    return filtered_data, outlier_mask

def detect_yield(strain, stress, elastic_end=0.02, window=51, polyorder=3):
    """
    Detect yield as the point of maximum curvature (second derivative)
    after the initial elastic region.
    """
    
    from scipy.signal import savgol_filter
    
    try:
        # Smooth the stress-strain curve to reduce noise
        stress_smooth = savgol_filter(stress, window, polyorder)
        # Compute second derivative (curvature)
        d2y = np.gradient(np.gradient(stress_smooth, strain), strain)
        # Only consider points after the elastic region
        mask = strain > elastic_end
        if not np.any(mask):
            return np.nan, np.nan
        idx = np.argmax(np.abs(d2y[mask]))
        idx_global = np.where(mask)[0][idx]
        return strain[idx_global], stress[idx_global]
    except Exception as e:
        print(f"Max curvature yield detection failed: {str(e)}")
        return np.nan, np.nan

def detect_inflection(strain, stress, yield_strain,
                      window=51, polyorder=3, 
                      yield_buffer=1.5,
                      plateau_detection_window=0.03,  # 3% strain window for plateau detection
                      min_plateau_slope=0.2,          # Stricter: must be flatter
                      min_plateau_length=0.02,        # Minimum plateau length
                      min_hardening_slope_ratio=1.5,  # Slope must increase by 50%
                      validation_window=0.025):       # 2.5% strain for post-inflection validation
    """
    S-curve inflection detection with balanced false positive/negative rate.
    Returns np.nan for monotonic curves with no plateau or late hardening.
    """
    from scipy.signal import savgol_filter
    import numpy as np

    if len(stress) < 100 or yield_strain is None or yield_strain <= 0:
        return np.nan

    stress_smoothed = savgol_filter(stress, window, polyorder)
    dy = np.gradient(stress_smoothed, strain)

    # Search region: post-yield, not too close to end
    search_start = max(yield_strain * yield_buffer, np.percentile(strain, 30))
    search_end = np.percentile(strain, 90)
    search_mask = (strain >= search_start) & (strain <= search_end)
    if not np.any(search_mask):
        return np.nan

    search_indices = np.where(search_mask)[0]

    # Find plateau regions
    plateau_candidates = []
    stress_range = np.ptp(stress_smoothed)
    strain_range = np.ptp(strain)
    for i in range(len(search_indices) - 10):
        idx = search_indices[i]
        window_end = strain[idx] + plateau_detection_window
        window_mask = (strain >= strain[idx]) & (strain <= window_end)
        if not np.any(window_mask):
            continue
        window_slopes = dy[window_mask]
        avg_slope = np.mean(window_slopes)
        slope_std = np.std(window_slopes)
        normalized_slope = abs(avg_slope) / (stress_range / strain_range)
        # Require a flat region (plateau) with low slope and low variability
        if normalized_slope < min_plateau_slope and slope_std < 0.5 * abs(avg_slope) and len(window_slopes) > int(min_plateau_length / (strain[1] - strain[0])):
            plateau_candidates.append((strain[idx], avg_slope))

    if not plateau_candidates:
        return np.nan  # No plateau, no S-curve inflection

    # Look for significant slope increase after the plateau
    for plateau_strain, plateau_slope in plateau_candidates:
        post_plateau_mask = strain > plateau_strain + min_plateau_length
        post_indices = np.where(post_plateau_mask)[0]
        for idx_curr in post_indices:
            # Pre and post windows for slope comparison
            pre_window = slice(max(0, idx_curr-10), idx_curr)
            post_window = slice(idx_curr, min(len(dy), idx_curr+15))
            if len(dy[pre_window]) < 5 or len(dy[post_window]) < 5:
                continue
            pre_slope = np.mean(dy[pre_window])
            post_slope = np.mean(dy[post_window])
            # Require significant and sustained slope increase
            if (post_slope > pre_slope * min_hardening_slope_ratio and
                post_slope > plateau_slope * min_hardening_slope_ratio):
                # Validate sustained hardening
                validation_mask = ((strain >= strain[idx_curr]) & 
                                   (strain <= strain[idx_curr] + validation_window))
                if np.any(validation_mask):
                    validation_slopes = dy[validation_mask]
                    if np.mean(validation_slopes) > pre_slope * 1.2:
                        return strain[idx_curr]
    return np.nan

def process_nanoindentation_data(root_dir, output_dir="Processed_Data", log_empty_path="empty_indents_log.txt", 
                                 remove_outliers_flag=False, outlier_threshold=3.0):
    """
    Process all nanoindentation data with optional outlier removal.
    
    Args:
        root_dir (str): Root directory containing data files
        output_dir (str): Output directory for processed data (default: 'Processed_Data')
        log_empty_path (str): Path for logging empty indents (default: 'empty_indents_log.txt')
        remove_outliers_flag (bool): Enable outlier removal (default: False)
        outlier_threshold (float): Z-score threshold for outlier detection (default: 3.0)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    empty_indents = []

    # First pass: determine global depth range
    print("Scanning files to determine depth range...")
    all_depths = []
    for root, _, files in os.walk(root_dir):
        for fname in files:
            if fname.upper().endswith('.CSV'):
                df = load_and_validate(os.path.join(root, fname))
                if df is not None:
                    all_depths.extend(df['DEPTH'].values)
    if not all_depths:
        print("No valid depth data found in any files.")
        write_empty_indents_log(empty_indents, log_empty_path)
        return

    global_min, global_max = np.min(all_depths), np.max(all_depths)
    common_depth = np.linspace(global_min, global_max, 2000)
    print(f"Global depth range: {global_min:.2f}-{global_max:.2f} nm")

    # Second pass: process and save data
    for film in ['I', 'II', 'III', 'IV']:
        film_path = os.path.join(root_dir, film)
        if not os.path.exists(film_path):
            continue

        print(f"\n{'='*40}\nPROCESSING FILM {film}\n{'='*40}")
        for column in ['A', 'B', 'C', 'D']:
            col_path = os.path.join(film_path, column)
            if not os.path.exists(col_path):
                continue

            for row in range(1, 30):
                row_path = os.path.join(col_path, str(row))
                if not os.path.exists(row_path):
                    continue

                print(f"|-- Processing {film}_{column}{row} ", end='')
                hardness_arrays = []
                modulus_arrays = []
                valid_count = 0

                # Collect all curves for current location
                for fname in os.listdir(row_path):
                    if fname.upper().endswith('.CSV') and fname.startswith(f'RT_{film}_{column}{row}_'):
                        file_path = os.path.join(row_path, fname)
                        df = load_and_validate(file_path)
                        if df is not None:
                            h_interp = np.interp(
                                common_depth, df['DEPTH'], df['HARDNESS'],
                                left=np.nan, right=np.nan
                            )
                            m_interp = np.interp(
                                common_depth, df['DEPTH'], df['MODULUS'],
                                left=np.nan, right=np.nan
                            )
                            if np.isfinite(h_interp).any() and np.isfinite(m_interp).any():
                                hardness_arrays.append(h_interp)
                                modulus_arrays.append(m_interp)
                                valid_count += 1
                            else:
                                empty_indents.append(file_path)
                        else:
                            empty_indents.append(file_path)

                # Outlier removal step
                if remove_outliers_flag and valid_count > 0:
                    hardness_arrays, h_outliers = remove_outliers(hardness_arrays, outlier_threshold)
                    modulus_arrays, m_outliers = remove_outliers(modulus_arrays, outlier_threshold)
                    
                    # Update valid count after outlier removal
                    valid_count = len(hardness_arrays)
                    print(f"[Outliers removed: {sum(h_outliers)} hardness, {sum(m_outliers)} modulus] ", end='')

                # Averaging and processing
                if valid_count > 0:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        hardness_stack = np.array(hardness_arrays)
                        modulus_stack = np.array(modulus_arrays)
                        
                        # Check for valid data after outlier removal
                        if len(hardness_arrays) == 0 or len(modulus_arrays) == 0:
                            print(" [SKIPPED] All data removed as outliers")
                            continue
        
                        min_valid = max(2, int(0.75 * len(hardness_arrays)))
        
                        valid_hardness = np.sum(~np.isnan(hardness_stack), axis=0)
                        valid_modulus = np.sum(~np.isnan(modulus_stack), axis=0)
        
                        hardness_avg = np.where(valid_hardness >= min_valid, 
                                       np.nanmean(hardness_stack, axis=0), np.nan)
                        modulus_avg = np.where(valid_modulus >= min_valid, 
                                      np.nanmean(modulus_stack, axis=0), np.nan)
                        std_hardness = np.where(valid_hardness >= min_valid, 
                                       np.nanstd(hardness_stack, axis=0), np.nan)
                        std_modulus = np.where(valid_modulus >= min_valid, 
                                      np.nanstd(modulus_stack, axis=0), np.nan)
        
                    print(f" [SAVED] {valid_count} indents")
                else:
                    print(" [SKIPPED] No valid data")
                    hardness_avg = np.full_like(common_depth, np.nan)
                    modulus_avg = np.full_like(common_depth, np.nan)
                    std_hardness = np.full_like(common_depth, np.nan)
                    std_modulus = np.full_like(common_depth, np.nan)

                # Save results
                result_df = pd.DataFrame({
                    'DEPTH': common_depth,
                    'HARDNESS_AVG': hardness_avg,
                    'MODULUS_AVG': modulus_avg,
                    'STD_HARDNESS': std_hardness,
                    'STD_MODULUS': std_modulus
                }).dropna(subset=['HARDNESS_AVG', 'MODULUS_AVG'], how='all')

                output_file = os.path.join(output_dir, f"{film}_{column}{int(row):02d}_averaged.csv")
                result_df.to_csv(output_file, index=False)

    write_empty_indents_log(empty_indents, log_empty_path)
    
def calculate_film_averages(processed_data_dir, output_file="film_mechanical_properties.csv", depth_range=(500, 1500)):
    """
    Calculate average hardness and modulus values for each film from processed CSV files.
    Each CSV in the processed folder is treated as a separate film.
    
    Parameters:
    -----------
    processed_data_dir : str
        Directory containing processed nanoindentation CSV files
    output_file : str
        Path to save the output CSV file with film averages
    depth_range : tuple
        Depth range (min, max) in nm to use for averaging (stable region)
    """
    import os
    import pandas as pd
    import numpy as np
    
    # Initialize results dictionary
    results = {
        'Film': [],
        'Hardness_Avg (GPa)': [],
        'Hardness_Std (GPa)': [],
        'Modulus_Avg (GPa)': [],
        'Modulus_Std (GPa)': [],
        'Depth_Min (nm)': [],
        'Depth_Max (nm)': []
    }
    
    csv_files = [f for f in os.listdir(processed_data_dir) 
                if f.endswith('_averaged.csv')]
    
    # Sort files for consistent ordering
    csv_files.sort()
    
    for filename in csv_files:
        film_name = filename.replace('_averaged.csv', '')
        file_path = os.path.join(processed_data_dir, filename)
        
        try:
            df = pd.read_csv(file_path)
            
            # Filter to the stable region
            mask = (df['DEPTH'] >= depth_range[0]) & (df['DEPTH'] <= depth_range[1])
            stable_region = df[mask]
            
            if len(stable_region) > 10:  # Ensure enough points for averaging
                # Calculate averages in stable region
                h_avg = np.nanmean(stable_region['HARDNESS_AVG'])
                h_std = np.nanstd(stable_region['HARDNESS_AVG'])
                m_avg = np.nanmean(stable_region['MODULUS_AVG'])
                m_std = np.nanstd(stable_region['MODULUS_AVG'])
                
                # Store results
                results['Film'].append(film_name)
                results['Hardness_Avg (GPa)'].append(h_avg)
                results['Hardness_Std (GPa)'].append(h_std)
                results['Modulus_Avg (GPa)'].append(m_avg)
                results['Modulus_Std (GPa)'].append(m_std)
                results['Depth_Min (nm)'].append(depth_range[0])
                results['Depth_Max (nm)'].append(depth_range[1])
                
                print(f"Film {film_name}: H={h_avg:.2f}±{h_std:.2f} GPa, M={m_avg:.2f}±{m_std:.2f} GPa")
            else:
                print(f"Skipping {film_name}: insufficient data points in stable region")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    # Save results to CSV
    if results['Film']:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        return results_df
    else:
        print("No valid results to save")
        return None

def load_and_validate_stress_strain(file_path):
    """Load and validate a nanoindentation CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.columns = [col.upper().strip() for col in df.columns]

        if not {'STRAIN', 'STRESS'}.issubset(df.columns):
            raise ValueError("Missing required columns: 'Strain' and/or 'Stress'")

        df = df[['STRAIN', 'STRESS']].apply(pd.to_numeric, errors='coerce')
        df = df[
            (df['STRAIN'].notna()) &
            (df['STRESS'].notna()) &
            (df['STRESS'] > 0) &
            (df['STRAIN'] > 0)
        ]
        return df.sort_values('STRAIN').reset_index(drop=True) if not df.empty else None
    except Exception as e:
        print(f"  [ERROR] Skipping {os.path.basename(file_path)}: {str(e)}")
        return None

def process_stress_strain_data(root_dir, output_dir="Processed_Data", log_empty_path="empty_indents_log.txt", 
                               remove_outliers_flag=False, outlier_threshold=3.0, max_strain=None):
    """
    Process all nanoindentation data with optional outlier removal and strain cutoff.

    Args:
        root_dir (str): Root directory containing data files
        output_dir (str): Output directory for processed data (default: 'Processed_Data')
        log_empty_path (str): Path for logging empty indents (default: 'empty_indents_log.txt')
        remove_outliers_flag (bool): Enable outlier removal (default: False)
        outlier_threshold (float): Z-score threshold for outlier detection (default: 3.0)
        max_strain (float, optional): Maximum allowed strain. Data above this will be ignored. (default: None)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    empty_indents = []

    # First pass: determine global strain range (do NOT trim by max_strain here)
    print("Scanning files to determine strain range...")
    all_strains = []
    for root, _, files in os.walk(root_dir):
        for fname in files:
            if fname.upper().endswith('.CSV'):
                file_path = os.path.join(root, fname)
                df = load_and_validate_stress_strain(file_path)
                if df is not None and not df.empty:
                    # Zero this curve
                    initial_strain = df['STRAIN'].iloc[0]
                    initial_stress = df['STRESS'].iloc[0]
                    df['STRAIN'] = df['STRAIN'] - initial_strain
                    df['STRESS'] = df['STRESS'] - initial_stress
                    all_strains.extend(df['STRAIN'].values)
                else:
                    empty_indents.append(file_path)

    if not all_strains:
        print("No valid strain data found in any files.")
        write_empty_indents_log(empty_indents, log_empty_path)
        return

    global_min = 0  # All curves start at 0 after zeroing
    global_max = np.max(all_strains)
    common_strain = np.linspace(global_min, global_max, 8000)
    print(f"Global strain range: 0.00-{global_max:.2f}%")

    # Second pass: process and save data
    for film in sections:
        film_path = os.path.join(root_dir, film)
        if not os.path.exists(film_path):
            continue

        print(f"\n{'='*40}\nPROCESSING FILM {film}\n{'='*40}")
        for column in columns:
            col_path = os.path.join(film_path, column)
            if not os.path.exists(col_path):
                continue

            for row in range(1, 30):
                row_path = os.path.join(col_path, str(row))
                if not os.path.exists(row_path):
                    continue

                print(f"|-- Processing {film}_{column}{row} ", end='')
                stress_arrays = []
                valid_count = 0

                # Collect all curves for current location
                for fname in os.listdir(row_path):
                    if fname.upper().endswith('.CSV') and fname.startswith(f'RT_{film}_{column}{row}_'):
                        file_path = os.path.join(row_path, fname)
                        df = load_and_validate_stress_strain(file_path)
                        if df is not None and not df.empty:
                            # Zero this curve
                            initial_strain = df['STRAIN'].iloc[0]
                            initial_stress = df['STRESS'].iloc[0]
                            df['STRAIN'] = df['STRAIN'] - initial_strain
                            df['STRESS'] = df['STRESS'] - initial_stress
                            # Do NOT trim by max_strain here
                            if df.empty:
                                empty_indents.append(file_path)
                                continue
                            s_interp = np.interp(
                                common_strain, df['STRAIN'], df['STRESS'],
                                left=np.nan, right=np.nan
                            )
                            if np.isfinite(s_interp).any():
                                stress_arrays.append(s_interp)
                                valid_count += 1
                            else:
                                empty_indents.append(file_path)
                        else:
                            empty_indents.append(file_path)

                # Outlier removal step
                if remove_outliers_flag and valid_count > 0:
                    stress_arrays, s_outliers = remove_outliers(stress_arrays, outlier_threshold)
                    valid_count = len(stress_arrays)
                    print(f"[Outliers removed: {sum(s_outliers)} stress] ", end='')

                # Averaging and processing
                if valid_count > 0:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        stress_stack = np.array(stress_arrays)
                        if len(stress_arrays) == 0:
                            print(" [SKIPPED] All data removed as outliers")
                            continue

                        min_valid = max(2, int(0.75 * len(stress_arrays)))
                        valid_stress = np.sum(~np.isnan(stress_stack), axis=0)
                        stress_avg = np.where(valid_stress >= min_valid, 
                                       np.nanmean(stress_stack, axis=0), np.nan)
                        std_stress = np.where(valid_stress >= min_valid, 
                                       np.nanstd(stress_stack, axis=0), np.nan)

                    print(f" [SAVED] {valid_count} indents")
                else:
                    print(" [SKIPPED] No valid data")
                    stress_avg = np.full_like(common_strain, np.nan)
                    std_stress = np.full_like(common_strain, np.nan)

                # --- Chop off fluctuating high-strain end (last 5% of valid data) ---
                chop_fraction = 0.05

                valid_indices = np.where(~np.isnan(stress_avg))[0]
                if len(valid_indices) == 0:
                    common_strain_chopped = np.array([])
                    stress_avg_chopped = np.array([])
                    std_stress_chopped = np.array([])
                else:
                    last_valid_idx = valid_indices[-1]
                    chop_idx = int(last_valid_idx * (1 - chop_fraction))
                    if chop_idx < 1:
                        chop_idx = 1

                    common_strain_chopped = common_strain[:chop_idx]
                    stress_avg_chopped = stress_avg[:chop_idx]
                    std_stress_chopped = std_stress[:chop_idx]

                # --- Now trim by max_strain (AFTER chopping) ---
                if max_strain is not None:
                    mask = common_strain_chopped <= max_strain
                    common_strain_chopped = common_strain_chopped[mask]
                    stress_avg_chopped = stress_avg_chopped[mask]
                    std_stress_chopped = std_stress_chopped[mask]

                # Save results
                result_df = pd.DataFrame({
                    'STRAIN': common_strain_chopped,
                    'STRESS_AVG': stress_avg_chopped,
                    'STD_STRESS': std_stress_chopped,
                }).dropna(subset=['STRESS_AVG'], how='all')

                output_file = os.path.join(output_dir, f"{film}_{column}{int(row):02d}_averaged.csv")
                result_df.to_csv(output_file, index=False)

    write_empty_indents_log(empty_indents, log_empty_path)
    
def generate_elbow_plot(processed_data_dir, k_range=(1, 10), standardize=True, 
                        save_path=None, show_plot=True):
    """
    Generate elbow plot for K-means clustering of stress-strain curves.
    
    Parameters:
    -----------
    processed_data_dir : str
        Directory containing processed CSV files
    k_range : tuple
        Range of K values to test (min, max)
    standardize : bool
        Whether to standardize features before clustering
    save_path : str
        Optional path to save the elbow plot
    show_plot : bool
        Whether to display the plot interactively
    
    Returns:
    --------
    dict with WCSS values and matplotlib figure object
    """

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    # Collect and align curves (same as clustering function)
    all_curves = []
    csv_files = sorted([f for f in os.listdir(processed_data_dir) 
                       if f.endswith('_averaged.csv')])
    
    for filename in csv_files:
        file_path = os.path.join(processed_data_dir, filename)
        try:
            df = pd.read_csv(file_path)
            strain = df['STRAIN'].values
            stress = df['STRESS_AVG'].values
            all_curves.append(np.column_stack((strain, stress)))
        except Exception as e:
            print(f"Skipping {filename}: {e}")
            continue

    # Create aligned dataset
    common_strain = np.linspace(0, 0.1, 100)
    X = np.array([np.interp(common_strain, c[:,0], c[:,1]) for c in all_curves])
    
    if standardize:
        X = StandardScaler().fit_transform(X)

    # Calculate WCSS for each K
    wcss = []
    k_values = range(k_range[0], k_range[1]+1)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    # Create elbow plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_values, wcss, 'bo-', markersize=8)
    ax.set_title('Elbow Method for Optimal K')
    ax.set_xlabel('Number of Clusters (K)')
    ax.set_ylabel('Within-Cluster Sum of Squares (WCSS)')
    ax.set_xticks(k_values)
    ax.grid(True, alpha=0.3)

    # Automatic elbow detection (simple gradient method)
    gradients = np.diff(wcss) / np.diff(k_values)
    elbow_k = np.argmin(gradients) + k_range[0] + 1
    ax.axvline(elbow_k, color='r', linestyle='--', 
              label=f'Suggested K: {elbow_k}')
    ax.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()

    return {'k_values': list(k_values), 'wcss': wcss, 'figure': fig}

def kmeans_clustering(processed_data_dir, n_clusters=3, output_file="film_clusters.csv"):
    """
    Perform k-means clustering on stress-strain curves from nanoindentation data,
    and save the cluster assignments as a CSV file.

    Parameters:
    -----------
    processed_data_dir : str
        Directory containing processed CSV files with stress-strain data
    n_clusters : int
        Number of clusters to use for k-means (default=3)
    output_file : str
        Path to save the output CSV file with cluster assignments

    Returns:
    --------
    pd.DataFrame with Film names and their assigned cluster labels
    """

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    # Collect all stress-strain curves
    all_curves = []
    film_names = []
    csv_files = [f for f in os.listdir(processed_data_dir) 
                if f.endswith('_averaged.csv')]
    csv_files.sort()

    for filename in csv_files:
        film_name = filename.replace('_averaged.csv', '')
        file_path = os.path.join(processed_data_dir, filename)
        try:
            df = pd.read_csv(file_path)
            # Extract relevant columns
            strain = df['STRAIN'].values
            stress = df['STRESS_AVG'].values
            # Store curve and film name
            all_curves.append(np.column_stack((strain, stress)))
            film_names.append(film_name)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Create common strain grid for alignment
    common_strain = np.linspace(0, 0.1, 100)  # 0-10% strain, 100 points
    interpolated_stress = []
    
    # Interpolate all curves to common strain grid
    for curve in all_curves:
        interp_stress = np.interp(common_strain, curve[:, 0], curve[:, 1])
        interpolated_stress.append(interp_stress)

    # Prepare data matrix for clustering
    X = np.array(interpolated_stress)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Create DataFrame and save to CSV
    df_clusters = pd.DataFrame({'Film': film_names, 'Cluster': cluster_labels})
    df_clusters.to_csv(output_file, index=False)
    print(f"Cluster assignments saved to {output_file}")

    return df_clusters

def hmm_clustering_with_elbow(processed_data_dir, use_optimal_k=True, n_clusters=3,
    k_range=(2, 8), n_states=5, output_file="hmm_clusters.csv", elbow_plot_file="hmm_elbow.png", random_state=42
):
    """
    HMM-based clustering with elbow plot and toggle for optimal/manual K.

    Parameters
    ----------
    processed_data_dir : str
        Directory containing processed CSV files.
    use_optimal_k : bool
        If True, use elbow method to select optimal K. If False, use n_clusters.
    n_clusters : int
        Number of clusters to use if use_optimal_k is False.
    k_range : tuple
        Range of K values to test for elbow (min, max).
    n_states : int
        Number of HMM states for each sequence.
    output_file : str
        Path to save the cluster assignments.
    elbow_plot_file : str
        Path to save the elbow plot.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame with Film names and cluster assignments.
    """

    from hmmlearn import hmm
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    # Step 1: Fit HMMs to each curve and extract parameters
    film_names = []
    hmm_params = []
    csv_files = sorted([f for f in os.listdir(processed_data_dir) if f.endswith('_averaged.csv')])

    for filename in csv_files:
        film_name = filename.replace('_averaged.csv', '')
        file_path = os.path.join(processed_data_dir, filename)
        try:
            df = pd.read_csv(file_path)
            strain = df['STRAIN'].values
            stress = df['STRESS_AVG'].values

            # Normalize stress for discretization
            stress_norm = (stress - np.min(stress)) / (np.max(stress) - np.min(stress) + 1e-10)
            n_bins = 20
            bins = np.linspace(0, 1, n_bins + 1)
            discretized = np.digitize(stress_norm, bins) - 1

            # Fit HMM
            model = hmm.MultinomialHMM(n_components=n_states, random_state=random_state)
            model.fit(discretized.reshape(-1, 1))

            # Extract parameters as feature vector
            trans_features = model.transmat_.flatten()
            emis_features = model.emissionprob_.flatten()
            features = np.concatenate([trans_features, emis_features])

            film_names.append(film_name)
            hmm_params.append(features)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    X = np.array(hmm_params)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 2: Elbow method to determine optimal K (if requested)
    wcss = []
    k_values = range(k_range[0], min(k_range[1]+1, len(X)))
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)

    # Plot elbow
    plt.figure(figsize=(8, 5))
    plt.plot(list(k_values), wcss, 'bo-', markersize=8)
    plt.title('Elbow Method for HMM-based Clustering')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Within-Cluster Sum of Squares')
    plt.xticks(list(k_values))
    plt.grid(True, alpha=0.3)
    plt.savefig(elbow_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Elbow plot saved to {elbow_plot_file}")

    # Step 3: Select K
    if use_optimal_k:
        gradients = np.diff(wcss)
        optimal_k = np.argmin(gradients) + k_range[0] + 1
        print(f"Auto-selected optimal K={optimal_k}")
        final_k = optimal_k
    else:
        print(f"Manual selection: K={n_clusters}")
        final_k = n_clusters

    # Step 4: Final clustering
    kmeans = KMeans(n_clusters=final_k, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Step 5: Save results
    df_clusters = pd.DataFrame({
        'Film': film_names,
        'Cluster': cluster_labels
    })
    df_clusters.to_csv(output_file, index=False)
    print(f"Cluster assignments saved to {output_file}")

    return df_clusters

def feature_kmeans(processed_data_dir, n_clusters=3, k_range=(1, 10),
    output_file="feature_clusters.csv", export_features_csv=None, show_elbow=True, plot_representatives=True):
    """
    Feature-based K-means clustering with elbow plot and manual K selection.

    Parameters
    ----------
    processed_data_dir : str
        Directory containing processed CSV files.
    n_clusters : int
        Number of clusters to use for KMeans.
    k_range : tuple
        Range of K values for elbow plot (min, max).
    output_file : str
        Path to save cluster assignments.
    export_features_csv : str (optional)
        Path to save raw features as CSV.
    show_elbow : bool
        Whether to display the elbow plot.
    """

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.impute import SimpleImputer
    from display import plot_cluster_representatives_center

    # --- Feature extraction ---
    features_dict = {
        'film_name': [],
        'elastic_slope': [],
        'yield_strain': [],
        'yield_stress': [],
        'inflection_strain': [],
        'hardening_exponent': [],
        'max_stress': [],
        'curve_length': []
    }

    csv_files = sorted([f for f in os.listdir(processed_data_dir) 
                        if f.endswith('_averaged.csv')])

    for filename in csv_files:
        film_name = filename.replace('_averaged.csv', '')
        file_path = os.path.join(processed_data_dir, filename)
        try:
            df = pd.read_csv(file_path)
            strain = df['STRAIN'].values
            stress = df['STRESS_AVG'].values

            # Elastic slope
            elastic_pts = min(100, len(strain))
            if elastic_pts > 1 and np.var(strain[:elastic_pts]) > 1e-10:
                elastic_slope = np.polyfit(strain[:elastic_pts], stress[:elastic_pts], 1)[0]
            else:
                elastic_slope = 0.0

            # Yield point detection
            yield_strain, yield_stress = detect_yield(strain, stress)
            
            # Inflection detection (pass yield_strain directly)
            inflection_strain = detect_inflection(strain, stress, yield_strain, window=71)

            # Hardening exponent
            hardening_exp = 0.0
            plastic_mask = (strain > yield_strain) & (stress > yield_stress)
            if np.sum(plastic_mask) > 5:
                strain_plastic = strain[plastic_mask] - yield_strain
                stress_plastic = stress[plastic_mask] - yield_stress
                valid = (strain_plastic > 1e-6) & (stress_plastic > 1e-6)
                if np.sum(valid) > 2:
                    log_strain = np.log(strain_plastic[valid])
                    log_stress = np.log(stress_plastic[valid])
                    if np.var(log_strain) > 1e-10:
                        hardening_exp = np.polyfit(log_strain, log_stress, 1)[0]

            features_dict['film_name'].append(film_name)
            features_dict['elastic_slope'].append(float(elastic_slope))
            features_dict['yield_strain'].append(float(yield_strain))
            features_dict['yield_stress'].append(float(yield_stress))
            features_dict['inflection_strain'].append(float(inflection_strain))
            features_dict['hardening_exponent'].append(float(hardening_exp))
            features_dict['max_stress'].append(float(np.nanmax(stress)))
            features_dict['curve_length'].append(float(strain[-1] if len(strain) > 0 else 0))

        except Exception as e:
            print(f"Skipped {filename} due to error: {str(e)}")
            continue

    features_df = pd.DataFrame(features_dict)
    if export_features_csv:
        features_df.to_csv(export_features_csv, index=False)
        print(f"Raw features saved to {export_features_csv}")

    # --- Clustering and elbow plot ---
    X = features_df.drop('film_name', axis=1).values
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    X_scaled = StandardScaler().fit_transform(X_imputed)

    wcss = []
    k_values = range(k_range[0], k_range[1]+1)
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)

    if show_elbow:
        plt.figure(figsize=(8, 5))
        plt.plot(k_values, wcss, 'bo-', markersize=8)
        plt.axvline(n_clusters, color='r', linestyle='--', label=f'Selected K: {n_clusters}')
        plt.title('Elbow Method for Feature-based Clustering')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
        plt.xticks(k_values)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    # --- Final clustering ---
    kmeans_final = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    features_df['cluster'] = kmeans_final.fit_predict(X_scaled)
    features_df.to_csv(output_file, index=False)
    print(f"Feature-based clustering saved to {output_file}")
    
    if plot_representatives:
        plot_cluster_representatives_center(processed_data_dir, features_df, X_scaled, n_clusters, figsize=(10,14))

    return features_df

