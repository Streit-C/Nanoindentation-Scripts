# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 11:20:46 2025

@author: streit
"""

import os
import pandas as pd
import numpy as np
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

def process_nanoindentation_data(root_dir, output_dir="Processed_Data", log_empty_path="empty_indents_log.txt", remove_outliers_flag=False, outlier_threshold=3.0):
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
    
def calculate_film_averages(processed_data_dir, output_file="film_mechanical_properties.csv", 
                           depth_range=(500, 1500)):
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

def process_stress_strain_data(root_dir, output_dir="Processed_Data", log_empty_path="empty_indents_log.txt", remove_outliers_flag=False, outlier_threshold=3.0):
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

    # First pass: determine global strain range
    print("Scanning files to determine strain range...")
    all_strains = []
    for root, _, files in os.walk(root_dir):
        for fname in files:
            if fname.upper().endswith('.CSV'):
                df = load_and_validate_stress_strain(os.path.join(root, fname))
                if df is not None:
                    all_strains.extend(df['STRAIN'].values)
    if not all_strains:
        print("No valid strain data found in any files.")
        write_empty_indents_log(empty_indents, log_empty_path)
        return

    global_min, global_max = np.min(all_strains), np.max(all_strains)
    common_strain = np.linspace(global_min, global_max, 8000)
    print(f"Global strain range: {global_min:.2f}-{global_max:.2f}%")

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
                        if df is not None:
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
                    
                    # Update valid count after outlier removal
                    valid_count = len(stress_arrays)
                    print(f"[Outliers removed: {sum(s_outliers)} stress] ", end='')

                # Averaging and processing
                if valid_count > 0:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        stress_stack = np.array(stress_arrays)
                        
                        # Check for valid data after outlier removal
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

                # Save results
                result_df = pd.DataFrame({
                    'STRAIN': common_strain,
                    'STRESS_AVG': stress_avg,
                    'STD_STRESS': std_stress,
                }).dropna(subset=['STRESS_AVG'], how='all')

                output_file = os.path.join(output_dir, f"{film}_{column}{int(row):02d}_averaged.csv")
                result_df.to_csv(output_file, index=False)

    write_empty_indents_log(empty_indents, log_empty_path)
