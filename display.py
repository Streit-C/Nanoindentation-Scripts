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

def plot_single_curve(processed_data_dir, film, column, row, plot_hardness=True, plot_modulus=True):
    """
    Plot a single processed curve (choose hardness, modulus, or both).
    Example: plot_single_curve('Processed_Data', 'III', 'A', 2, plot_hardness=True, plot_modulus=False)
    """
    fname = f"{film}_{column}{row}_averaged.csv"
    for root, _, files in os.walk(processed_data_dir):
        if fname in files:
            fpath = os.path.join(root, fname)
            df = pd.read_csv(fpath)
            plt.figure(figsize=(10, 6))
            if plot_hardness and 'HARDNESS_AVG' in df.columns:
                plt.plot(df['DEPTH'], df['HARDNESS_AVG'], label='Hardness Avg', color='tab:blue')
                if 'STD_HARDNESS' in df.columns:
                    plt.fill_between(df['DEPTH'],
                                     df['HARDNESS_AVG']-df['STD_HARDNESS'],
                                     df['HARDNESS_AVG']+df['STD_HARDNESS'],
                                     color='tab:blue', alpha=0.15, label='Hardness Std')
            if plot_modulus and 'MODULUS_AVG' in df.columns:
                plt.plot(df['DEPTH'], df['MODULUS_AVG'], label='Modulus Avg', color='tab:orange')
                if 'STD_MODULUS' in df.columns:
                    plt.fill_between(df['DEPTH'],
                                     df['MODULUS_AVG']-df['STD_MODULUS'],
                                     df['MODULUS_AVG']+df['STD_MODULUS'],
                                     color='tab:orange', alpha=0.15, label='Modulus Std')
            plt.xlabel('Depth (nm)')
            plt.ylabel('Value (GPa)')
            title = f"Averaged Nanoindentation Curve: {film}_{column}{row}"
            if plot_hardness and not plot_modulus:
                title += " (Hardness Only)"
            elif plot_modulus and not plot_hardness:
                title += " (Modulus Only)"
            plt.title(title)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            return
        
    print(f"File not found: {fname} in {processed_data_dir}")
