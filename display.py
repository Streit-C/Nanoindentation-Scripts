# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 12:16:44 2025

@author: strei
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
    
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

def plot_single_curve(processed_data_dir, film, column, row, plot_hardness=True, plot_modulus=True, plot_stress=False):
    """
    Plot a single processed curve (choose hardness, modulus, stress, or any combination).
    Example: plot_single_curve('Processed_Data', 'III', 'A', 2, plot_hardness=True, plot_modulus=False, plot_stress=True)
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
                plt.ylabel('Stress (GPa)')
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
