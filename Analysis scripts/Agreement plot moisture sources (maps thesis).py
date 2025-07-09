import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import griddata
import os
from UTrack_functions import find_land_mask  # Assuming this function is correctly defined
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.util import add_cyclic_point
import pandas as pd

# Collect results here (place at top of script before the loop if needed)
tmrr_records = []

def compute_gridcell_areas(lat, lon, R=6371000):
    lat_rad = np.radians(lat)
    dlat = np.radians(np.abs(np.gradient(lat)))
    dlon = np.radians(np.abs(np.gradient(lon)))

    area = np.zeros((len(lat), len(lon)))

    for i in range(len(lat)):
        area[i, :] = R**2 * dlat[i] * dlon * np.cos(lat_rad[i])

    return area  # in m²

# Configuration
models = ['EC-Earth3', 'MPI-ESM1-2-HR', 'NorESM2-MM', 'TaiESM1']
scenarios = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
target_periods = [(2050, 2059), (2090, 2099)]
clusterstypes = ['drying', 'wetting']
confidence_level_percentile = 95 # Percentage of total moisture flux to consider as 'top source'

# 1. Define Common Low-Resolution Grid
file_noresm_example = f'Cumulative moisture sources results\Cumulative_moisture_sources_drying_NorESM2-MM_{scenarios[0]}_{target_periods[0][0]}-{target_periods[0][1]}.nc'
# Ensure __file__ is defined, e.g. by running as a script. If in Jupyter, use os.getcwd() or specify path.

script_dir = os.path.dirname(__file__)
path_noresm_example = os.path.join(script_dir, file_noresm_example)


ds_noresm_example = xr.open_dataset(path_noresm_example)
common_lon = ds_noresm_example.lon.values
common_lat = ds_noresm_example.lat.values
common_lon_grid, common_lat_grid = np.meshgrid(common_lon, common_lat)


for scenario in scenarios:
    # Set up the figure and axes for plotting
    fig, axes = plt.subplots(len(target_periods), len(clusterstypes), figsize=(15, 10), # Adjusted figsize slightly
                             sharex=True, sharey=True, subplot_kw={'projection': ccrs.PlateCarree()})
    subplot_labels = ['a)', 'b)', 'c)', 'd)']
    label_idx = 0
    
    if len(target_periods) == 1 or len(clusterstypes) == 1:
        axes = np.atleast_2d(axes)
    
    # Loop through each target period
    for i, (start_year, end_year) in enumerate(target_periods):
        # Loop through each cluster type
        for j, clusterstype in enumerate(clusterstypes):
            agreement_grid = np.zeros_like(common_lon_grid, dtype=int)
    
            # Loop through each Earth System Model
            for model in models:
                try:
                    # --- Load land mask (native grid) ---
                    # Ensure find_land_mask returns a mask compatible with ds['footprint'] dimensions
                    current_land_mask_values_native = find_land_mask(rf"F:\Forcing_data_{model}", model, scenario)
    
                    # --- Load cumulative moisture sources (native grid) ---
                    file = f'Cumulative moisture sources results\Cumulative_moisture_sources_{clusterstype}_{model}_{scenario}_{start_year}-{end_year}.nc'
                    path = os.path.join(script_dir, file)
                    
                    ds = xr.open_dataset(path)
                    output_map = ds['footprint'].values # Native grid moisture source footprint
                    lat_native = ds['lat'].values
                    lon_native = ds['lon'].values
                    
                    # Compute gridcell area
                    gridcell_area_native = compute_gridcell_areas(lat_native, lon_native)  # m²
                    
                    # make the first and last column equal (average for each row) as the plotting logic was faulty causing a line at the prime meridian
                    # with the adjusted updated plotting logic of get_nearest_index this is unnecessary
                    for row in range(agreement_grid.shape[0]):
                        row_avg = (output_map[row,-1] + output_map[row,0])/2
                        output_map[row,-1] = row_avg
                        output_map[row,0] = row_avg
    
                    # --- Continue with existing logic for plotting model agreement ---
                    # Apply land mask for top percentage calculation (original logic, if needed for 'top_mask')
                    # The line below means masked_output_map is currently same as output_map
                    masked_output_map_for_threshold = output_map # np.where(current_land_mask_values_native == 1, output_map, np.nan)
    
                    flat_values = masked_output_map_for_threshold#[~np.isnan(masked_output_map_for_threshold) & (masked_output_map_for_threshold > 0)]
                    
                    threshold_value = 0 
                    if flat_values.size > 0:
                        # Calculate the threshold based on the confidence_level_percentile
                        # For "top 95% confidence" you might want the 95th percentile, or even higher for very strong sources.
                        # Using np.percentile is robust to non-normal distributions.
                        threshold_value = np.percentile(flat_values, confidence_level_percentile)
                        print(f"Model: {model}, Cluster: {clusterstype}, Period: {start_year}-{end_year}, Calculated Threshold (at {confidence_level_percentile}th percentile): {threshold_value:.2e}")
                    else:
                        print(f"Warning: No valid (non-NaN, positive) moisture flux values found for {model} {clusterstype} {start_year}-{end_year}. Threshold set to 0.")
    
    
                    top_mask_native = np.where(output_map >= threshold_value, 1, 0)
                    
                    # Optional normalization (e.g., for TMRR-style fraction)
                    tmr_weighted_land = np.nansum(gridcell_area_native * output_map * current_land_mask_values_native)
                    tmr_weighted_total = np.nansum(gridcell_area_native * output_map)
                    
                    # Avoid division by zero
                    TMRR_cumulative = tmr_weighted_land / tmr_weighted_total if tmr_weighted_total > 0 else np.nan
                    
                    tmrr_records.append({
                        'Model': model,
                        'Scenario': scenario,
                        'Period': f'{start_year}-{end_year}',
                        'Cluster': clusterstype,
                        'TMRR_cumulative': TMRR_cumulative
                    })
                    
                    # You can print or store `tmrr_fraction` for later use if needed
                    print(f"Model: {model}, Period: {start_year}-{end_year}, {clusterstype}, Area-weighted TMRR proxy: {TMRR_cumulative:.2e}")
                    
                    
                    # Interpolate the top_mask_native to the common grid
                    lon2d_native, lat2d_native = np.meshgrid(lon_native, lat_native)
                    points_native = np.column_stack((lon2d_native.flatten(), lat2d_native.flatten()))
                    values_native = top_mask_native.flatten().astype(float)
                    
                    interpolated_mask_common = griddata(points_native, values_native, (common_lon_grid, common_lat_grid), method='nearest')
                    agreement_grid += np.nan_to_num(interpolated_mask_common, nan=0).astype(int)
    
                except FileNotFoundError:
                    print(f'File not found for {model} {clusterstype} {start_year}-{end_year}: {file}. Skipping...')
    
            
            # --- Plotting the agreement grid for the current subplot ---
            ax = axes[i, j]
           
            if clusterstype == 'drying':
                cmap = plt.cm.Reds
            elif clusterstype == 'wetting':
                cmap = plt.cm.Blues
            
            norm = mcolors.Normalize(vmin=0, vmax=len(models))
            
            # Convert agreement_grid to float type to allow NaN values
            agreement_grid = agreement_grid.astype(float)

            # Now, set all zero values to NaN to prevent them from becoming orange or slightly blue on the colorscale
            agreement_grid[agreement_grid == 0] = np.nan
 
            im = ax.pcolormesh(common_lon, common_lat, agreement_grid,
                               cmap=cmap, norm=norm, shading='auto',
                               transform=ccrs.PlateCarree())
            
            period_label = 'Mid-Century' if (start_year, end_year) == target_periods[0] else 'End-Century'
            ax.set_title(f'{clusterstype.capitalize()} ({period_label})', fontsize=16)
            ax.text(-0.05, 1.05, subplot_labels[label_idx], transform=ax.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
            label_idx += 1
            
            ax.set_extent([common_lon.min(), common_lon.max(), common_lat.min(), common_lat.max()], crs=ccrs.PlateCarree())
            ax.coastlines(linewidth=0.5, color='gray')
            ax.add_feature(cfeature.LAND, edgecolor='k', linewidth=0.5, facecolor='white', zorder=0)
            ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5, edgecolor='gray')
    
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            gl.xformatter = LongitudeFormatter()
            gl.yformatter = LatitudeFormatter()
            gl.xlabel_style = {'size': 12, 'color': 'black'} # Adjusted back from 12 to 10 for potentially better fit
            gl.ylabel_style = {'size': 12, 'color': 'black'}
    
    
    fig.suptitle(f'Model Agreement on {confidence_level_percentile}th percentile Terrestrial Moisture Sources (Scenario: {scenario.upper()})', fontsize=18, y=0.98)
    
    # Apply tight_layout with your desired rectangle for overall positioning.
    # The w_pad (and h_pad) are ignored by tight_layout when rect is specified.
    plt.tight_layout(rect=[0, 0.16, 1, 0.95])
    
    # Now, specifically adjust the vertical spacing (hspace) between subplots.
    # Experiment with the hspace value (e.g., 0.3, 0.4, 0.5) to get desired spacing.
    # hspace is the height of the padding between subplots, as a fraction of the average axes height.
    fig.subplots_adjust(hspace=0.2)  # INCREASE THIS VALUE TO ADD MORE VERTICAL SPACE
    
    # Drying Colorbar
    cbar_ax_drying = fig.add_axes([0.25, 0.1, 0.5, 0.04]) # [left, bottom, width, height]
    cmap_drying = plt.cm.Reds
    norm_drying = mcolors.Normalize(vmin=0, vmax=len(models))
    sm_drying = plt.cm.ScalarMappable(cmap=cmap_drying, norm=norm_drying)
    sm_drying.set_array([])
    cbar_drying = fig.colorbar(sm_drying, cax=cbar_ax_drying, orientation='horizontal', ticks=[])

    # Wetting Colorbar
    cbar_ax_wetting = fig.add_axes([0.25, 0.1, 0.5, 0.02]) # [left, bottom, width, height]
    cmap_wetting = plt.cm.Blues
    norm_wetting = mcolors.Normalize(vmin=0, vmax=len(models))
    sm_wetting = plt.cm.ScalarMappable(cmap=cmap_wetting, norm=norm_wetting)
    sm_wetting.set_array([])
    cbar_wetting = fig.colorbar(sm_wetting, cax=cbar_ax_wetting, orientation='horizontal', ticks=np.arange(len(models) + 1))
    cbar_wetting.ax.tick_params(labelsize=10)
    cbar_wetting.set_label('Number of Models Agreeing', fontsize=15)
    cbar_wetting.set_ticks(np.arange(1, len(models) + 1))
    cbar_wetting.set_ticklabels([str(i) for i in range(1, len(models) + 1)])
    
    plt.show()
    plt.savefig(f'Agreement on {confidence_level_percentile}th percentile of most important moisture sources {scenario}.png')
    
    
    
# Convert records to DataFrame
tmrr_df = pd.DataFrame(tmrr_records)

# Pivot so "drying" and "wetting" become columns
tmrr_pivot = tmrr_df.pivot_table(
    index=['Scenario', 'Period', 'Model'],
    columns='Cluster',
    values='TMRR_cumulative'
).reset_index()

# Optional: Rename columns to be clearer
tmrr_pivot.columns.name = None
tmrr_pivot = tmrr_pivot.rename(columns={'drying': 'TMRR_drying', 'wetting': 'TMRR_wetting'})

# Save to Excel
tmrr_pivot.to_excel('Area_Weighted_TMRR_Global_Pivoted.xlsx', index=False)

print("Saved pivoted global cumulative TMRR values to Excel.")