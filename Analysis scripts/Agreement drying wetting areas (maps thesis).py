import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
from scipy.interpolate import griddata
from UTrack_functions import find_land_mask
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# Configuration
models = ['EC-Earth3', 'MPI-ESM1-2-HR', 'NorESM2-MM', 'TaiESM1']
scenarios = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
target_periods = [(2050, 2059), (2090, 2099)]
dw_types = ['drying', 'wetting']
agreement_min = 2

# Define common grid from a reference file
low_resolution_land_mask = find_land_mask(r"F:\Forcing_data_NorESM2-MM", "NorESM2-MM", "ssp126")
file_noresm = rf'Drying wetting results/Significantly_drying_wetting_map_NorESM2-MM_ssp126_{target_periods[0][0]}-{target_periods[0][1]}.nc'
path_noresm = os.path.join(os.path.dirname(__file__), file_noresm)
ds_noresm = xr.open_dataset(path_noresm)

common_lon = ds_noresm.lon.values
common_lat = ds_noresm.lat.values
common_lon_grid, common_lat_grid = np.meshgrid(common_lon, common_lat)

for scenario in scenarios:
    fig, axes = plt.subplots(len(target_periods), len(dw_types), figsize=(15, 10), sharex=True, sharey=True, subplot_kw={'projection': ccrs.PlateCarree()})
    subplot_labels = ['a)', 'b)', 'c)', 'd)']
    label_idx = 0

    for i, (start_year, end_year) in enumerate(target_periods):
        for j, dw_type in enumerate(dw_types):
            ax = axes[i, j]
            model_masks = []

            for model in models:
                try:
                    file = rf'Drying wetting results/Significantly_drying_wetting_map_{model}_{scenario}_{start_year}-{end_year}.nc'
                    path = os.path.join(os.path.dirname(__file__), file)
                    ds = xr.open_dataset(path)
                    cluster_map = ds['clustermap']
                    
                    clusters_file = rf'Drying wetting results/Drying_wetting_cluster_info_{model}_{scenario}_{start_year}-{end_year}.npy'
                    clusters_path = os.path.join(os.path.dirname(__file__), clusters_file)
                    clusters_list = np.load(clusters_path,allow_pickle=True).tolist()
                    
                    for cluster in clusters_list:
                        cluster_size_gridcells = len(cluster['coords'])
                        cluster_ID = cluster['id']
                        
                        if model == 'EC-Earth3' and cluster_size_gridcells < 119:
                            print(f"      FILTERED: Skipping Cluster ID {cluster_ID} for model {model} (rule: < 118).")
                            cluster_map = np.where(cluster_map==cluster_ID, 0, cluster_map)
                        
                        if model == 'MPI-ESM1-2-HR' and cluster_size_gridcells < 67:
                            print(f"      FILTERED: Skipping Cluster ID {cluster_ID} for model {model} (rule: < 66).")
                            cluster_map = np.where(cluster_map==cluster_ID, 0, cluster_map)
                    
                    clusters_mask = np.where(cluster_map>0,1,0)

                    #mask the mean_pr_change by significance mask and the clusters mask so it shows all the used areas only
                    pr_change_map = ds['mean_pr_change'].values * ds['mean_significance_mask'].values * clusters_mask

                    if dw_type == 'drying':
                        output_map = np.where(pr_change_map < 0, 1, 0)
                    else:
                        output_map = np.where(pr_change_map > 0, 1, 0)

                    lat = ds['lat'].values
                    lon = ds['lon'].values
                    lon2d, lat2d = np.meshgrid(lon, lat)
                    points = np.column_stack((lon2d.flatten(), lat2d.flatten()))
                    values = output_map.flatten().astype(float)
                    interpolated_mask = griddata(points, values, (common_lon_grid, common_lat_grid), method='nearest')

                    interpolated_mask = interpolated_mask * low_resolution_land_mask

                    model_masks.append(np.nan_to_num(interpolated_mask, nan=0).astype(int))

                except Exception as e:
                    print(f'Error loading {model} for {dw_type} {start_year}-{end_year}: {e}')
                    model_masks.append(np.zeros_like(common_lon_grid))

            model_masks_array = np.stack(model_masks, axis=0)
            agreement_grid = np.sum(model_masks_array, axis=0)
            
            # Convert agreement_grid to float type to allow NaN values
            agreement_grid = agreement_grid.astype(float)

            # Now, set ocean values to NaN
            agreement_grid[agreement_grid == 0] = np.nan

            # Add cyclic point along the longitude axis
            agreement_grid_cyclic, lon_cyclic = add_cyclic_point(agreement_grid, coord=common_lon, axis=1)

            if dw_type == 'drying':
                cmap_used = plt.cm.Reds
            elif dw_type == 'wetting':
                cmap_used = plt.cm.Blues

            norm_used = mcolors.Normalize(vmin=0, vmax=len(models))

            # Plot the cyclic version of the map
            im = ax.pcolormesh(lon_cyclic, common_lat, agreement_grid_cyclic, cmap=cmap_used, norm=norm_used, shading='auto',
                              transform=ccrs.PlateCarree())

            period_label = 'Mid-Century' if (start_year, end_year) == target_periods[0] else 'End-Century'
            ax.set_title(f'{dw_type.capitalize()} ({period_label})', fontsize=16)
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
            gl.xlabel_style = {'size': 12, 'color': 'black'}
            gl.ylabel_style = {'size': 12, 'color': 'black'}

    # Title and layout
    fig.suptitle(f'Model Agreement — Scenario: {scenario.upper()}', fontsize=20)
    plt.tight_layout(rect=[0, 0.16, 1, 0.95]) # Adjusted rect to make space for the colorbars
    fig.subplots_adjust(hspace=0.2)
    
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
    plt.savefig(f'Model agreement {scenario} drying wetting.png')