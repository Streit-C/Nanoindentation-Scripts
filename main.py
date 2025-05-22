# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 12:18:03 2025

@author: strei
"""

import os
import plotly.io as pio
from pathlib import Path
from processing import process_nanoindentation_data, calculate_film_averages, process_stress_strain_data, generate_elbow_plot, kmeans_clustering, feature_kmeans
from display import plot_single_curve, plot_all_single_curves, plot_all_single_stress_strain_curves, plot_all_curves, plot_all_curves_stress_strain, plot_wafer_stress_strain, plot_wafer_stress_strain_interactive, plot_cluster_representatives

def main():
    pio.renderers.default = 'browser'  #Necessary command for generating images in browser using plotly
    
    root_directory = r"D:\Data\Combi Ceramics\MoTaNbCrZr Combi\RT Wafer As-deposited"
    processed_directory = os.path.abspath("Processed_Data_Stress_Strain")
    positions = "positions.csv"
    cluster_file = "feature_clusters.csv"
    
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
    
    #Stress-strain curves
    process_stress_strain_data(root_directory, remove_outliers_flag=True, outlier_threshold=3.5, max_strain=18.0)
    
    plot_all_curves_stress_strain(processed_directory)
    
    plot_single_curve(processed_directory, 'IV', 'A', 7 , plot_hardness=False, plot_modulus=False, plot_stress=True)
    plot_all_single_stress_strain_curves(root_directory, 'IV', 'A', 9, colormap='viridis')
    
    plot_wafer_stress_strain(processed_directory, positions, scale=0.072, region_min=0.05, region_max=0.95)
    plot_wafer_stress_strain_interactive(processed_directory, positions, scale=0.065, region_min=0.05, region_max=0.95, use_colormap=True)
    
    #K-means cluster analysis
    generate_elbow_plot(processed_directory)
    kmeans_clustering(processed_directory, n_clusters=4)
    
    feature_kmeans(processed_directory, n_clusters=4, k_range=(1, 10), output_file="feature_clusters.csv", export_features_csv=None, show_elbow=True)
    plot_cluster_representatives(processed_directory, cluster_file, figsize=(10, 14))

if __name__ == "__main__":
    main()
