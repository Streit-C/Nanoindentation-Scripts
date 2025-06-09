# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 12:16:44 2025

@author: streit
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
    
def plot_all_curves(processed_data_dir, plot_hardness=True, plot_modulus=True):
    """
    Plot all processed curves (HARDNESS_AVG and/or MODULUS_AVG) from every *_averaged.csv
    in the processed data directory and its subdirectories.
    """
    plt.figure(figsize=(12, 7))
    for root, _, files in os.walk(processed_data_dir):
        for fname in files:
            if fname.endswith('_averaged.csv'):
                fpath = os.path.join(root, fname)
                df = pd.read_csv(fpath)
                label = os.path.splitext(fname)[0].replace('_averaged', '')
                if plot_hardness and 'HARDNESS_AVG' in df.columns:
                    plt.plot(df['DEPTH'], df['HARDNESS_AVG'], label=f'{label} Hardness')
                if plot_modulus and 'MODULUS_AVG' in df.columns:
                    plt.plot(df['DEPTH'], df['MODULUS_AVG'], label=f'{label} Modulus')
    plt.xlabel('Depth (nm)')
    plt.ylabel('Mechanical Property')
    title = 'All Averaged Nanoindentation Curves'
    if plot_hardness and not plot_modulus:
        title += ' (Hardness Only)'
    elif plot_modulus and not plot_hardness:
        title += ' (Modulus Only)'
    plt.title(title)
    plt.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.show()
    
def plot_all_curves_stress_strain(processed_data_dir):
    """
    Plot all processed curves (STRAIN_AVG) from every *_averaged.csv
    in the processed data directory and its subdirectories.
    """
    plt.figure(figsize=(12, 7))
    for root, _, files in os.walk(processed_data_dir):
        for fname in files:
            if fname.endswith('_averaged.csv'):
                fpath = os.path.join(root, fname)
                df = pd.read_csv(fpath)
                label = os.path.splitext(fname)[0].replace('_averaged', '')
                if 'STRESS_AVG' in df.columns:
                    plt.plot(df['STRAIN'], df['STRESS_AVG'], label=f'{label} Stress')
    plt.xlabel('Strain (%)')
    plt.ylabel('Stress (MPa)')
    title = 'All Averaged Nanoindentation Curves'
    plt.title(title)
    plt.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.show()
    
def plot_all_single_curves(root_dir, film, column, row, 
                                plot_hardness=True, plot_modulus=True, 
                                alpha=0.8, colormap='tab10', figsize=(10, 6)):
    """
    Plot all individual nanoindentation curves with correct scaling and dual y-axis support.

    Args:
        root_dir (str): Root directory containing the data files
        film (str): Film identifier (e.g., 'I', 'II', 'III')
        column (str): Column identifier (e.g., 'A', 'B', 'C')
        row (int or str): Row number
        plot_hardness (bool): Whether to plot hardness curves. (default: True)
        plot_modulus (bool): Whether to plot modulus curves. (default: True)
        alpha (float): Transparency of curves (default: 0.8)
        colormap (str): Matplotlib colormap name (default: 'viridis')
        figsize (tuple): Figure size (width, height) in inches
    """
    row = str(int(row))
    location_dir = os.path.join(root_dir, film, column, row)
    if not os.path.exists(location_dir):
        print(f"Directory not found: {location_dir}")
        return

    files = [f for f in os.listdir(location_dir) 
             if f.upper().endswith('.CSV') and f.startswith(f'RT_{film}_{column}{row}_')]
    if not files:
        print(f"No CSV files found in {location_dir}")
        return

    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx() if (plot_hardness and plot_modulus) else None

    colors = plt.get_cmap(colormap)(np.linspace(0, 1, len(files)))
    all_hardness_data = []
    all_modulus_data = []

    for idx, fname in enumerate(sorted(files)):
        fpath = os.path.join(location_dir, fname)
        try:
            df = pd.read_csv(fpath)
            for col in ['DEPTH', 'HARDNESS', 'MODULUS']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            if not all(col in df.columns for col in ['DEPTH', 'HARDNESS', 'MODULUS']):
                print(f"Missing required columns in {fname}")
                continue
            df = df.dropna(subset=['DEPTH', 'HARDNESS', 'MODULUS'])
            if len(df) == 0:
                print(f"No valid data in {fname}")
                continue
            depth = df['DEPTH'].values
            hardness = df['HARDNESS'].values
            modulus = df['MODULUS'].values
            if len(hardness) > 0:
                all_hardness_data.extend(hardness)
            if len(modulus) > 0:
                all_modulus_data.extend(modulus)
            label = f"Indent {idx+1}"
            if plot_hardness:
                ax1.plot(depth, hardness, color=colors[idx], alpha=alpha,
                         linestyle='-', label=f"{label} Hardness")
            if plot_modulus:
                target_ax = ax2 if ax2 else ax1
                target_ax.plot(depth, modulus, color=colors[idx], alpha=alpha,
                               linestyle='--', label=f"{label} Modulus")
        except Exception as e:
            print(f"Error processing {fname}: {str(e)}")
            continue

    if plot_hardness and len(all_hardness_data) > 0:
        h_min = min(all_hardness_data)
        h_max = max(all_hardness_data)
        h_margin = max(0.1 * (h_max - h_min), 1.0)
        ax1.set_ylim(h_min - h_margin, h_max + h_margin)
        ax1.set_ylabel('Hardness (GPa)', color='#1f77b4', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='#1f77b4')
    if plot_modulus and ax2 and len(all_modulus_data) > 0:
        m_min = min(all_modulus_data)
        m_max = max(all_modulus_data)
        m_margin = max(0.1 * (m_max - m_min), 10.0)
        ax2.set_ylim(m_min - m_margin, m_max + m_margin)
        ax2.set_ylabel('Modulus (GPa)', color='#ff7f0e', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='#ff7f0e')
    ax1.set_xlabel('Depth (nm)', fontsize=12)

    title = f"Raw Curves: {film}_{column}{row}"
    if plot_hardness and plot_modulus:
        title += " (Dual Axis)"
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels() if ax2 else ([], [])
    handles = handles1 + handles2
    labels = labels1 + labels2
    if handles:
        fig.legend(handles, labels, bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.show()
    
def plot_all_single_stress_strain_curves(root_dir, film, column, row, 
                                 alpha=0.8, colormap='tab10', figsize=(10, 6)):
    """
    Plot all individual stress-strain curves before averaging.
    
    Args:
        root_dir (str): Root directory containing the data files
        film (str): Film identifier (e.g., 'I', 'II', 'III')
        column (str): Column identifier (e.g., 'A', 'B', 'C')
        row (int or str): Row number
        alpha (float): Transparency of curves (default: 0.8)
        colormap (str): Matplotlib colormap name (default: 'viridis')
        figsize (tuple): Figure size (width, height) in inches
    """
    row = str(int(row))  # Format row as two digits
    location_dir = os.path.join(root_dir, film, column, row)
    print(f"Searching in: {location_dir}")
    
    if not os.path.exists(location_dir):
        print(f"Directory not found: {location_dir}")
        return

    # Find all individual curve files (not averaged)
    files = [f for f in os.listdir(location_dir) 
             if f.upper().endswith('.CSV') and f.startswith(f'RT_{film}_{column}{row}_')]
    
    if not files:
        print(f"No CSV files found in {location_dir}")
        return
    
    print(f"Found {len(files)} files to plot")
    
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.get_cmap(colormap)(np.linspace(0, 1, len(files)))
    all_stress_data = []
    
    for idx, fname in enumerate(sorted(files)):
        fpath = os.path.join(location_dir, fname)
        try:
            df = pd.read_csv(fpath)
            
            # Look for strain/stress columns with flexible naming
            strain_col = next((col for col in df.columns if 'STRAIN' in col.upper()), None)
            stress_col = next((col for col in df.columns if 'STRESS' in col.upper()), None)
            
            if not strain_col or not stress_col:
                print(f"Missing strain/stress columns in {fname}")
                continue
            
            # Convert to numeric and clean data
            df[strain_col] = pd.to_numeric(df[strain_col], errors='coerce')
            df[stress_col] = pd.to_numeric(df[stress_col], errors='coerce')
            df = df.dropna(subset=[strain_col, stress_col])
            
            if len(df) == 0:
                print(f"No valid data in {fname}")
                continue
            
            strain = df[strain_col].values
            stress = df[stress_col].values * 0.001  # Convert to GPa if in MPa
            
            if len(stress) > 0:
                all_stress_data.extend(stress)
            
            label = f"Test {idx+1}"
            ax.plot(strain, stress, color=colors[idx], alpha=alpha,
                   linestyle='-', label=label)
            
        except Exception as e:
            print(f"Error processing {fname}: {str(e)}")
            continue
    
    if all_stress_data:
        s_min = min(all_stress_data)
        s_max = max(all_stress_data)
        s_margin = 0.1 * (s_max - s_min)
        ax.set_ylim(s_min - s_margin, s_max + s_margin)
    
    # Set up axes and labels
    ax.set_xlabel('Strain (%)', fontsize=12)
    ax.set_ylabel('Stress (GPa)', fontsize=12)
    ax.set_title(f"Individual Stress-Strain Curves: {film}_{column}{row}", fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add legend if we have curves
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, bbox_to_anchor=(1.15, 1), 
                  loc='upper left', fontsize=9)
    
    plt.tight_layout()
    plt.show()

def plot_single_curve(processed_data_dir, film, column, row, plot_hardness=True, plot_modulus=True, plot_stress=False):
    """
    Plot a single processed curve (choose hardness, modulus, stress, or any combination).
    
    Args:
        root_dir (str): Root directory containing the data files
        film (str): Film identifier (e.g., 'I', 'II', 'III')
        column (str): Column identifier (e.g., 'A', 'B', 'C')
        row (int or str): Row number
        plot_hardness (bool): Whether to plot hardness curve. (default: True)
        plot_modulus (bool): Whether to plot modulus curve. (default: True)
        plot_stress (bool): Whether to plot stress-strain curve. (default: False)
    """
    formatted_row = f"{int(row):02d}"
    fname = f"{film}_{column}{formatted_row}_averaged.csv"
    
    for root, _, files in os.walk(processed_data_dir):
        if fname in files:
            fpath = os.path.join(root, fname)
            df = pd.read_csv(fpath)
            
            # --- Stress-Strain Plot ---
            if plot_stress and 'STRAIN' in df.columns and 'STRESS_AVG' in df.columns:
                plt.figure(figsize=(10, 6))
                plt.plot(df['STRAIN'], df['STRESS_AVG'], color='tab:red', label='Stress Avg')
                if 'STD_STRESS' in df.columns:
                    plt.fill_between(
                        df['STRAIN'],
                        df['STRESS_AVG'] - df['STD_STRESS'],
                        df['STRESS_AVG'] + df['STD_STRESS'],
                        color='tab:red', alpha=0.18, label='Stress Std'
                    )
                plt.xlabel('Strain (%)')
                plt.ylabel('Stress (MPa)')
                plt.title(f"Averaged Stress-Strain Curve: {film}_{column}{formatted_row}")
                plt.legend()
                plt.grid(True)
                plt.xlim(left=0)
                plt.ylim(bottom=0)
                plt.tight_layout()
                plt.show()
                # Return here if you want to plot only stress-strain when plot_stress is True
                # return

            # --- Hardness/Modulus Plot ---
            if plot_hardness or plot_modulus:
                plt.figure(figsize=(10, 6))
                if plot_hardness and plot_modulus:
                    ax1 = plt.gca()
                    ax2 = ax1.twinx()
                    plotted_hardness = False
                    plotted_modulus = False

                    if 'HARDNESS_AVG' in df.columns:
                        ax1.plot(df['DEPTH'], df['HARDNESS_AVG'], label='Hardness Avg', color='tab:blue')
                        if 'STD_HARDNESS' in df.columns:
                            ax1.fill_between(
                                df['DEPTH'],
                                df['HARDNESS_AVG'] - df['STD_HARDNESS'],
                                df['HARDNESS_AVG'] + df['STD_HARDNESS'],
                                color='tab:blue', alpha=0.15, label='Hardness Std'
                            )
                        plotted_hardness = True

                    if 'MODULUS_AVG' in df.columns:
                        ax2.plot(df['DEPTH'], df['MODULUS_AVG'], label='Modulus Avg', color='tab:orange')
                        if 'STD_MODULUS' in df.columns:
                            ax2.fill_between(
                                df['DEPTH'],
                                df['MODULUS_AVG'] - df['STD_MODULUS'],
                                df['MODULUS_AVG'] + df['STD_MODULUS'],
                                color='tab:orange', alpha=0.15, label='Modulus Std'
                            )
                        plotted_modulus = True

                    ax1.set_xlabel('Depth (nm)')
                    ax1.set_ylabel('Hardness (GPa)', color='tab:blue')
                    ax2.set_ylabel('Modulus (GPa)', color='tab:orange')
                    ax1.tick_params(axis='y', labelcolor='tab:blue')
                    ax2.tick_params(axis='y', labelcolor='tab:orange')

                    # Combine legends
                    lines, labels = [], []
                    if plotted_hardness:
                        l1, lab1 = ax1.get_legend_handles_labels()
                        lines += l1
                        labels += lab1
                    if plotted_modulus:
                        l2, lab2 = ax2.get_legend_handles_labels()
                        lines += l2
                        labels += lab2
                    if lines:
                        ax1.legend(lines, labels, loc='upper left')

                else:
                    if plot_hardness and 'HARDNESS_AVG' in df.columns:
                        plt.plot(df['DEPTH'], df['HARDNESS_AVG'], label='Hardness Avg', color='tab:blue')
                        if 'STD_HARDNESS' in df.columns:
                            plt.fill_between(
                                df['DEPTH'],
                                df['HARDNESS_AVG'] - df['STD_HARDNESS'],
                                df['HARDNESS_AVG'] + df['STD_HARDNESS'],
                                color='tab:blue', alpha=0.15, label='Hardness Std'
                            )

                    if plot_modulus and 'MODULUS_AVG' in df.columns:
                        plt.plot(df['DEPTH'], df['MODULUS_AVG'], label='Modulus Avg', color='tab:orange')
                        if 'STD_MODULUS' in df.columns:
                            plt.fill_between(
                                df['DEPTH'],
                                df['MODULUS_AVG'] - df['STD_MODULUS'],
                                df['MODULUS_AVG'] + df['STD_MODULUS'],
                                color='tab:orange', alpha=0.15, label='Modulus Std'
                            )

                    plt.xlabel('Depth (nm)')
                    plt.ylabel('Value (GPa)')
                    plt.legend()

                title = f"Averaged Nanoindentation Curve: {film}_{column}{formatted_row}"
                if plot_hardness and not plot_modulus:
                    title += " (Hardness Only)"
                elif plot_modulus and not plot_hardness:
                    title += " (Modulus Only)"
                plt.title(title)
                plt.grid(True)
                plt.tight_layout()
                plt.show()
            return

    print(f"File not found: {fname} in {processed_data_dir}")

    
def plot_wafer_stress_strain(processed_data_dir, positions, scale=0.35, region_min=0.3, region_max=0.7):
    """
    Plot all stress-strain curves at their respective wafer positions,
    compressing positions to reduce whitespace and increasing curve size.
    Args:
        processed_data_dir (str): Directory containing processed CSV files
        positions_csv (str): Path to CSV with 'Film', 'X', 'Y' columns
        scale (float): Size scale of each curve
        region_min (float): Minimum normalized position for compression
        region_max (float): Maximum normalized position for compression
        plot_stress (bool): Whether to plot stress-strain curves
    """
    positions_df = pd.read_csv(positions)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('off')

    # Normalize wafer positions
    x_norm = (positions_df['X'] - positions_df['X'].min()) / (positions_df['X'].max() - positions_df['X'].min())
    y_norm = (positions_df['Y'] - positions_df['Y'].min()) / (positions_df['Y'].max() - positions_df['Y'].min())

    # Compress wafer positions to central region
    x_compressed = x_norm * (region_max - region_min) + region_min
    y_compressed = y_norm * (region_max - region_min) + region_min

    # Fix axes limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    for idx, row in positions_df.iterrows():
        film = str(row['Film'])
        x_pos = x_compressed.iloc[idx]
        y_pos = y_compressed.iloc[idx]

        # Find matching data file
        found_file = None
        for root, _, files in os.walk(processed_data_dir):
            for file in files:
                if file.startswith(film) and file.endswith('_averaged.csv'):
                    found_file = os.path.join(root, file)
                    break
            if found_file:
                break

        if not found_file:
            continue

        df = pd.read_csv(found_file)
        if 'STRAIN' in df.columns and 'STRESS_AVG' in df.columns:
            strain = df['STRAIN']
            stress = df['STRESS_AVG']
            # Normalize curve data
            strain_norm = (strain - strain.min()) / (strain.max() - strain.min()) - 0.5
            stress_norm = (stress - stress.min()) / (stress.max() - stress.min()) - 0.5
            # Center curve at wafer position
            x_curve = x_pos + strain_norm * scale
            y_curve = y_pos + stress_norm * scale
            ax.plot(x_curve, y_curve, color='tab:red', linewidth=1.5)
            if 'STD_STRESS' in df.columns:
                std = df['STD_STRESS']
                std_norm = std / (stress.max() - stress.min()) * scale
                ax.fill_between(x_curve, y_curve - std_norm, y_curve + std_norm, color='tab:red', alpha=0.18)

    plt.show()
    
def plot_wafer_stress_strain_interactive(processed_data_dir, positions, scale=0.35, region_min=0.3, region_max=0.7, use_colormap=True, cmap_name='Viridis'):
    print("Indexing available data files...")
    # Build a lookup dict: {film_name: file_path}
    film_to_file = {}
    for root, _, files in os.walk(processed_data_dir):
        for file in files:
            if file.endswith('_averaged.csv'):
                film_name = file.split('_averaged.csv')[0]
                film_to_file[film_name] = os.path.join(root, file)
    print(f"Indexed {len(film_to_file)} data files.")

    print("Reading wafer positions CSV...")
    positions_df = pd.read_csv(positions)
    print(f"Found {len(positions_df)} positions.")

    # Normalize and compress wafer positions
    x_norm = (positions_df['X'] - positions_df['X'].min()) / (positions_df['X'].max() - positions_df['X'].min())
    y_norm = (positions_df['Y'] - positions_df['Y'].min()) / (positions_df['Y'].max() - positions_df['Y'].min())
    x_compressed = x_norm * (region_max - region_min) + region_min
    y_compressed = y_norm * (region_max - region_min) + region_min

    # Read all data files once and cache
    print("Reading all data files...")
    film_to_df = {}
    all_stress = []
    for idx, row in positions_df.iterrows():
        film = str(row['Film'])
        file_path = film_to_file.get(film)
        if not file_path or not os.path.exists(file_path):
            print(f"  [!] File not found for film {film}, skipping.")
            continue
        df = pd.read_csv(file_path)
        film_to_df[film] = df
        if 'STRESS_AVG' in df.columns:
            all_stress.extend(df['STRESS_AVG'].dropna().values)
        else:
            print(f"  [!] 'STRESS_AVG' column missing in {file_path}, skipping.")

    if all_stress:
        global_stress_min = min(all_stress)
        global_stress_max = max(all_stress)
        print(f"Global stress range: {global_stress_min:.3f} to {global_stress_max:.3f}")
    else:
        global_stress_min = 0
        global_stress_max = 1
        print("No stress data found. Using default range 0 to 1.")

    fig = go.Figure()
    print("Plotting curves...")
    for idx, row in positions_df.iterrows():
        film = str(row['Film'])
        x_pos = x_compressed.iloc[idx]
        y_pos = y_compressed.iloc[idx]
        df = film_to_df.get(film)
        if df is None or 'STRAIN' not in df.columns or 'STRESS_AVG' not in df.columns:
            print(f"  [!] Data missing for film {film}, skipping.")
            continue

        strain = df['STRAIN'].values
        stress = df['STRESS_AVG'].values
        strain_norm = (strain - strain.min()) / (strain.ptp()) - 0.5
        stress_norm = (stress - stress.min()) / (stress.ptp()) - 0.5
        x_curve = x_pos + strain_norm * scale
        y_curve = y_pos + stress_norm * scale

        if use_colormap:
            # Plot as colored markers (fast), not line gradient
            #norm_stress = (stress - global_stress_min) / (global_stress_max - global_stress_min)
            fig.add_trace(go.Scatter(
                x=x_curve, y=y_curve, mode='lines+markers',
                marker=dict(color=stress, colorscale=cmap_name, size=6, colorbar=dict(title="Stress (MPa)") if idx==0 else None,
                            cmin=global_stress_min, cmax=global_stress_max),
                line=dict(color='rgba(150,150,150,0.3)', width=1),
                name=film,
                showlegend=False
            ))
            print(f"  [+] Plotted film {film} with colormap.")
        else:
            fig.add_trace(go.Scatter(
                x=x_curve, y=y_curve, mode='lines',
                line=dict(color='red', width=2),
                name=film,
                showlegend=False
            ))
            print(f"  [+] Plotted film {film} in red.")

    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, visible=False, range=[0,1]),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, range=[0,1]),
        plot_bgcolor='white',
        width=800, height=800,
        title="Wafer Stress-Strain Curves" + (" (Colormap)" if use_colormap else " (Red)")
    )

    print("Displaying interactive plot...")
    fig.show()
    
def plot_cluster_representatives(processed_data_dir, cluster_file, output_file=None, figsize=(10, 8)):
    """
    Create stacked plot of representative curves with separate y-axes for each cluster.
    
    Parameters:
    -----------
    processed_data_dir : str
        Directory containing processed CSV files
    cluster_file : str
        Path to CSV file with 'film_name' and 'cluster' columns
    output_file : str (optional)
        Path to save the plot image
    figsize : tuple
        Figure size (width, height) in inches
    """

    # Load cluster assignments
    df = pd.read_csv(cluster_file)
    if not {'film_name', 'cluster'}.issubset(df.columns):
        raise ValueError("CSV must contain 'film_name' and 'cluster' columns.")

    clusters = sorted(df['cluster'].unique())
    n_clusters = len(clusters)

    # Create stacked subplots with shared x-axis
    fig, axes = plt.subplots(n_clusters, 1, sharex=True, figsize=figsize)
    if n_clusters == 1:
        axes = [axes]  # Ensure axes is always a list

    plt.suptitle('Stacked Representative Stress-Strain Curves', y=0.98, fontsize=14)
    colors = plt.cm.viridis_r(np.linspace(0, 1, n_clusters))

    for idx, cluster_label in enumerate(clusters):
        ax = axes[idx]
        
        # Select representative curve
        cluster_df = df[df['cluster'] == cluster_label]
        representative = cluster_df.sample(1, random_state=42).iloc[0]
        film_name = representative['film_name']
        
        # Load and plot curve
        file_path = os.path.join(processed_data_dir, f"{film_name}_averaged.csv")
        try:
            curve_data = pd.read_csv(file_path)
            ax.plot(curve_data['STRAIN'], curve_data['STRESS_AVG'], 
                   color=colors[idx],
                   label=f'Cluster {cluster_label} ({film_name})')
            
            # Format individual axes
            ax.set_ylabel('Stress (MPa)', fontsize=10)
            ax.legend(loc='upper right', frameon=False)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(left=0)    # Start x-axis at 0
            ax.set_ylim(bottom=0)  # Start y-axis at 0
            
            # Remove top and right spines
            ax.spines[['right', 'top']].set_visible(False)
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

    # Common x-axis label
    axes[-1].set_xlabel('Strain', fontsize=12)
    
    # Adjust spacing
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    plt.show()
    
def plot_cluster_representatives_center(processed_data_dir, features_df, X_scaled, n_clusters, output_file=None, figsize=(10, 8)):
    """
    Plot the representative (center-most) curve for each cluster,
    with shared x-axis set to the largest strain value among all representatives.
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import pairwise_distances_argmin_min

    clusters = sorted(features_df['cluster'].unique())
    centers = []
    for k in clusters:
        cluster_indices = np.where(features_df['cluster'] == k)[0]
        cluster_points = X_scaled[cluster_indices]
        centroid = cluster_points.mean(axis=0)
        centers.append(centroid)
    centers = np.vstack(centers)

    # Find representatives
    representative_indices, _ = pairwise_distances_argmin_min(centers, X_scaled)
    representative_df = features_df.iloc[representative_indices]

    # Find max strain across all representatives
    max_strain = 0
    curve_data_list = []
    for row in representative_df.itertuples():
        film_name = row.film_name
        file_path = os.path.join(processed_data_dir, f"{film_name}_averaged.csv")
        try:
            curve_data = pd.read_csv(file_path)
            curve_data_list.append((curve_data, row.cluster, film_name))
            max_strain = max(max_strain, curve_data['STRAIN'].max())
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

    # Plot
    fig, axes = plt.subplots(n_clusters, 1, sharex=True, figsize=figsize)
    if n_clusters == 1:
        axes = [axes]
    plt.suptitle('Cluster Center Representative Stress-Strain Curves', y=0.98, fontsize=14)
    colors = plt.cm.viridis_r(np.linspace(0, 1, n_clusters))

    for idx, (ax, (curve_data, cluster_label, film_name)) in enumerate(zip(axes, curve_data_list)):
        ax.plot(curve_data['STRAIN'], curve_data['STRESS_AVG'],
                color=colors[idx],
                label=f'Cluster {cluster_label} ({film_name})')
        ax.set_ylabel('Stress (MPa)', fontsize=10)
        ax.legend(loc='upper right', frameon=False)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0, right=max_strain)
        ax.set_ylim(bottom=0)
        ax.spines[['right', 'top']].set_visible(False)

    axes[-1].set_xlabel('Strain', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    plt.show()
