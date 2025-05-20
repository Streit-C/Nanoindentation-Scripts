# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 12:18:03 2025

@author: strei
"""

import os
from pathlib import Path
from processing import process_nanoindentation_data, calculate_film_averages, process_stress_strain_data
from display import plot_single_curve, plot_all_single_curves, plot_all_curves, plot_all_curves_stress_strain

def main():
    root_directory = r"D:\Data\Combi Ceramics\MoTaNbCrZr Combi\RT Wafer As-deposited"
    processed_directory = os.path.abspath("Processed_Data Stress_Strain")
    
    process_nanoindentation_data(root_directory, remove_outliers_flag=True)
    
    if not Path(processed_directory).exists():
        print(f"Error: Directory not found - {processed_directory}")
        return
    
    # Plot all curves
    plot_all_curves(processed_directory, plot_hardness=True, plot_modulus=True)
    
    # Plot a single curve (example: film III, column A, row 2)
    plot_single_curve(processed_directory, 'II', 'B', 8 , plot_hardness=True, plot_modulus=True)
    
    plot_all_single_curves(root_directory, 'III', 'B', 7, plot_hardness=True, plot_modulus=True)
    
    calculate_film_averages(processed_directory, output_file="film_mechanical_properties.csv", depth_range=(200, 500))
    
    process_stress_strain_data(root_directory, remove_outliers_flag=True, outlier_threshold=3.5, max_strain=18.0)
    
    plot_all_curves_stress_strain(processed_directory)
    
    plot_single_curve(processed_directory, 'IV', 'A', 7 , plot_hardness=False, plot_modulus=False, plot_stress=True)

if __name__ == "__main__":
    main()
