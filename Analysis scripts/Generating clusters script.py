import os
import numpy as np
import xarray as xr
import statsmodels.api as sm
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
from scipy.ndimage import label
import matplotlib.colors as mcolors
import plotly.graph_objects as go
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Patch
import cftime
from datetime import datetime
import pandas as pd

def parse_time_range(filename):
    """Extracts the time range from the filename."""
    parts = filename.split('_')
    if len(parts) > 1:
        date_range = parts[-1].replace('.nc', '')
        try:
            start_year = int(date_range[:4])
            end_year = int(date_range[9:13])
            return start_year, end_year
        except ValueError:
            return None, None
    return None, None

def filter_files_by_scenario_and_variable(forcing_directory, scenario, variable):
    """Gets all files for a specific scenario."""
    files = []
    for file in os.listdir(forcing_directory):
        if scenario in file and file.endswith('.nc') and file.startswith(f'{variable}_'):
            filepath = os.path.join(forcing_directory, file)
            files.append(filepath)
    return files

def load_precipitation_data(file):
    """Loads precipitation data from a .nc file."""
    ds = xr.open_dataset(file)
    pr = ds['pr']    #'pr' is the precipitation variable in [kg m-2 s-1] or better known as mm/s
    pr = pr*60*60*24*28  #convert to mm/4 weeks
    return pr

def retrieve_precipitation_data(files, directory, start_year, end_year):
    """Calculates the average precipitation over a given time range."""
    precipitation_data = []

    for file in files:
        file_start, file_end = parse_time_range(file)
        if file_start is None or file_end is None:
            continue

        # Check if the file's time range overlaps with the target range
        if file_end >= start_year and file_start <= end_year:
            file_path = os.path.join(directory, file)
            data = load_precipitation_data(file_path)

            # Select the time range within the file
            data = data.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
            precipitation_data.append(data)

    # Combine and calculate the average precipitation
    if precipitation_data:
        combined_data = xr.concat(precipitation_data, dim='time')
        mean_data = combined_data.mean(dim='time')
        median_data = combined_data.median(dim='time')
        return combined_data, mean_data, median_data
    return None

def compute_p_values_per_gridcell(baseline, target):
    """Performs a statistical test for each grid cell to determine differences."""
    mean_p_values = np.empty(baseline.shape[1:])
    median_p_values = np.empty(baseline.shape[1:])
    
    # Iterate over each grid cell
    for i in range(baseline.shape[1]):
        for j in range(baseline.shape[2]):
            baseline_data = baseline[:, i, j]
            target_data = target[:, i, j]

            # Remove NaN values for testing
            baseline_data = baseline_data[~np.isnan(baseline_data)]
            target_data = target_data[~np.isnan(target_data)]

            if len(baseline_data) > 1 and len(target_data) > 1:
                _, mean_p_value, _ = sm.stats.ttest_ind(baseline_data, target_data, usevar='unequal')         # this is the method I used before with the simple independent t-test for mean comparison
                _, median_p_value = mannwhitneyu(baseline_data, target_data, alternative='two-sided')   #use the mannwhitneyu test as I am using median values for the significant difference
                mean_p_values[i, j] = mean_p_value
                median_p_values[i, j] = median_p_value
            else:
                mean_p_values[i, j] = np.nan  # Not enough data
                median_p_values[i, j] = np.nan  # Not enough data

    return mean_p_values, median_p_values

def significant_change_in_treefrac(scenario, baseline_period, target_period):
    # === Configuration ===
    CLM5_veg_dir = r"F:\CLM5 vegetation projections SSPs"
    baseline_slice = slice(cftime.DatetimeNoLeap(baseline_period[0], 1, 1), cftime.DatetimeNoLeap(baseline_period[1]+1, 1, 1))
    target_slice = slice(cftime.DatetimeNoLeap(target_period[0], 1, 1), cftime.DatetimeNoLeap(target_period[1]+1, 1, 1))
    
    # === Utility to load and concatenate ===
    def load_concat_files(scenario):
        files = sorted([f for f in os.listdir(CLM5_veg_dir) if f.startswith('treeFrac') and scenario in f and 'r1i1p1f1' in f])
        datasets = [xr.open_dataset(os.path.join(CLM5_veg_dir, f)) for f in files]
        ds_merged = xr.concat(datasets, dim='time')
        return ds_merged
        
    #version that loads all ensembles and take the mean of them 
    # def load_concat_files(scenario):  
    #     # === Step 1: Find all files for the given scenario and variable ===
    #     files = sorted([
    #         f for f in os.listdir(CLM5_veg_dir)
    #         if f.startswith('treeFrac') and scenario in f
    #     ])

    #     # === Step 2: Organize files by ensemble member (e.g., r1i1p1f1) ===
    #     ensemble_files = {}
    #     for f in files:
    #         # Extract ensemble ID from filename (3rd element when split by '_')
    #         ensemble_id = f.split('_')[2]
    #         if ensemble_id not in ensemble_files:
    #             ensemble_files[ensemble_id] = []
    #         ensemble_files[ensemble_id].append(f)

    #     # === Step 3: Load, combine and clean up each ensemble ===
    #     all_ensembles = []
    #     for ensemble_id, file_list in ensemble_files.items():
    #         # Sort files to ensure chronological order
    #         file_list = sorted(file_list)

    #         # Load all parts for this ensemble
    #         datasets = [
    #             xr.open_dataset(os.path.join(CLM5_veg_dir, f))
    #             for f in file_list
    #         ]

    #         # Combine them along the time dimension
    #         combined = xr.concat(datasets, dim='time')

    #         # Drop any duplicate time values (common when files overlap in time)
    #         unique_times, index = np.unique(combined['time'], return_index=True)
    #         combined = combined.isel(time=index)

    #         # Add an 'ensemble' dimension so we can later combine all ensembles
    #         combined = combined.expand_dims({'ensemble': [ensemble_id]})

    #         all_ensembles.append(combined)

    #     # === Step 4: Combine all ensembles and take the mean ===
    #     if all_ensembles:
    #         merged = xr.concat(all_ensembles, dim='ensemble')
    #         return merged.mean(dim='ensemble')
    #     else:
    #         raise ValueError(f"No data found for scenario '{scenario}'")
    
    try:
        ds = load_concat_files('ssp245')
        da = ds['treeFrac']
        baseline_data = da.sel(time=baseline_slice)
        base_mean = baseline_data.mean(dim='time')

        ds_target = load_concat_files(scenario)
        db = ds_target['treeFrac']
        target_data = db.sel(time=target_slice)
        target_mean = target_data.mean(dim='time')

        diff = target_mean - base_mean

        p_values, _ = compute_p_values_per_gridcell(baseline_data.values, target_data.values)
        mean_significance_mask = p_values < 0.05  # Threshold for significance
        significant_change_map = mean_significance_mask * diff
        
        return base_mean, target_mean, significant_change_map
            
    except Exception as e:
        print(f"Failed to load treefrac for {scenario} in baseline period {baseline_period} or target period {target_period}: {e}")
        

def significant_change_in_temp(model, scenario, baseline_period, target_period):
    
    # === Utility to load data for a time period from multiple files ===        
    def load_temp_data_for_period(scenario, model, period):
        """
        Loads temperature data for a given time period from multiple files,
        searching for files based on the 'tas' prefix, scenario, 'r1i1p1f1',
        and the year range within the filename.
    
        Args:
            scenario (str): The climate scenario (e.g., 'ssp126').
            model (str): The climate model (e.g., 'NorESM2-MM').
            period (tuple): A tuple containing the start and end year of the period (e.g., (2050, 2059)).
    
        Returns:
            xr.DataArray or None: An xarray DataArray containing the combined
                                 temperature data for the specified period, or None
                                 if no data is found.
        """
        folder = fr"F:\Forcing_data_{model}"
        if not os.path.isdir(folder):
            print(f"Warning: Folder not found: {folder}")
            return None
    
        all_data = []
        start_year, end_year = period
        start_date = datetime(start_year, 1, 1)
        end_date = datetime(end_year + 1, 1, 1)
        cftime_start_date = cftime.DatetimeNoLeap(start_year, 1, 1)
        cftime_end_date = cftime.DatetimeNoLeap(end_year + 1, 1, 1)
    
        for filename in os.listdir(folder):
            if filename.startswith('tas_day') and scenario in filename and 'r1i1p1f1' in filename and filename.endswith('.nc'):
                try:
                    file_start_str, file_end_str = filename.split('_')[-1].replace('.nc', '').split('-')
                    file_start_year = int(file_start_str[:4])
                    file_end_year = int(file_end_str[:4])
    
                    # Check if the file's time range overlaps with the desired period
                    if file_end_year >= start_year and file_start_year <= end_year:
                        file_path = os.path.join(folder, filename)
                        try:
                            with xr.open_dataset(file_path) as ds:
                                try:
                                    # Try selecting with datetime
                                    data = ds['tas'].sel(time=slice(start_date, end_date))
                                    if data.size > 0:
                                        all_data.append(data)
                                except TypeError:
                                    try:
                                        # Try selecting with cftime.DatetimeNoLeap
                                        data = ds['tas'].sel(time=slice(cftime_start_date, cftime_end_date))
                                        if data.size > 0:
                                            all_data.append(data)
                                    except Exception as e:
                                        print(f"Error selecting time in {filename}: {e}")
                        except Exception as e:
                            print(f"Error opening dataset {filename}: {e}")
    
                except ValueError:
                    print(f"Warning: Could not parse year range from filename: {filename}")
    
        if all_data:
            combined_data = xr.concat(all_data, dim='time')
            return combined_data
        else:
            print(f"Warning: No temperature data found for model: {model}, scenario: {scenario}, period: {period}")
            return None

    try:
        # Load baseline data
        baseline_data = load_temp_data_for_period('ssp245', model, baseline_period)
        if baseline_data is None:
            print(f"Failed to load baseline data for {model}, {scenario}")
            return None, None, None
        base_mean = baseline_data.mean(dim='time')

        # Load target data
        target_data = load_temp_data_for_period(scenario, model, target_period)
        if target_data is None:
            print(f"Failed to load target data for {model}, {scenario}")
            return None, None, None
        target_mean = target_data.mean(dim='time')

        diff = target_mean - base_mean

        p_values, _ = compute_p_values_per_gridcell(baseline_data.values, target_data.values)
        mean_significance_mask = p_values < 0.05  # Threshold for significance
        significant_change_map = mean_significance_mask * diff

        return base_mean, target_mean, significant_change_map

    except Exception as e:
        print(f"Failed to process tas for {scenario} in baseline period {baseline_period} or target period {target_period}: {e}")
        return None, None, None
    
    
def calculate_drying_wetting_to_nc(model, forcing_directory, baseline_scenario, baseline_period, target_scenario, target_period):
    """Computes precipitation change, checks significance, and saves as NetCDF"""
    
    # Load baseline and target precipitation data
    baseline_files = filter_files_by_scenario_and_variable(forcing_directory, baseline_scenario, variable='pr')
    baseline, mean_baseline, median_baseline = retrieve_precipitation_data(baseline_files, forcing_directory, *baseline_period)

    target_files = filter_files_by_scenario_and_variable(forcing_directory, target_scenario, variable='pr')
    target, mean_target, median_target = retrieve_precipitation_data(target_files, forcing_directory, *target_period)

    # Compute the precipitation change in absolute terms
    mean_pr_change = mean_target - mean_baseline
    median_pr_change = median_target - median_baseline

    # Compute p-values for significance
    mean_p_values, median_p_values = compute_p_values_per_gridcell(baseline.values, target.values)
    mean_significance_mask = mean_p_values < 0.05  # Threshold for significance
    median_significance_mask = median_p_values < 0.05 
    
    # Convert to xarray DataArray abs
    results = {'mean_baseline_precipitation': xr.DataArray(
        mean_baseline,
        dims=mean_baseline.dims, 
        coords={mean_baseline.dims[0]: mean_baseline.coords[mean_baseline.dims[0]].values,
                mean_baseline.dims[1]: mean_baseline.coords[mean_baseline.dims[1]].values},
        name=f"Mean precipitation {target_period[0]}-{target_period[1]}",
        attrs={
            "description": f"Mean precipitation in baseline years ({baseline_period[0]}-{baseline_period[1]}).",
            "model": model,
            "scenario": target_scenario,
            "period": f"{baseline_period[0]}-{baseline_period[1]}",
            "units": "mm/4 weeks",
        }
    ),
        'median_baseline_precipitation': xr.DataArray(
            median_baseline,
            dims=mean_baseline.dims, 
            coords={median_baseline.dims[0]: median_baseline.coords[median_baseline.dims[0]].values,
                    median_baseline.dims[1]: median_baseline.coords[median_baseline.dims[1]].values},
            name=f"Median precipitation {target_period[0]}-{target_period[1]}",
            attrs={
                "description": f"Median precipitation in baseline years ({baseline_period[0]}-{baseline_period[1]}).",
                "model": model,
                "scenario": target_scenario,
                "period": f"{baseline_period[0]}-{baseline_period[1]}",
                "units": "mm/4 weeks",
            }
    ),
        'mean_target_precipitation': xr.DataArray(
        mean_target,
        dims=mean_baseline.dims, 
        coords={mean_baseline.dims[0]: mean_baseline.coords[mean_baseline.dims[0]].values,
                mean_baseline.dims[1]: mean_baseline.coords[mean_baseline.dims[1]].values},
        name=f"Mean precipitation {target_period[0]}-{target_period[1]}",
        attrs={
            "description": f"Mean precipitation in target years ({target_period[0]}-{target_period[1]}).",
            "model": model,
            "scenario": target_scenario,
            "period": f"{target_period[0]}-{target_period[1]}.",
            "units": "mm/4 weeks",
        }
    ),
        'median_target_precipitation': xr.DataArray(
        median_target,
        dims=median_baseline.dims, 
        coords={median_baseline.dims[0]: median_baseline.coords[median_baseline.dims[0]].values,
                median_baseline.dims[1]: median_baseline.coords[median_baseline.dims[1]].values},
        name=f"Median precipitation {target_period[0]}-{target_period[1]}",
        attrs={
            "description": f"Median precipitation in target years ({target_period[0]}-{target_period[1]}).",
            "model": model,
            "scenario": target_scenario,
            "period": f"{target_period[0]}-{target_period[1]}.",
            "units": "mm/4 weeks",
        }
    ),
        'mean_pr_change': xr.DataArray(
            mean_pr_change,
            dims=mean_baseline.dims, 
            coords={mean_baseline.dims[0]: mean_baseline.coords[mean_baseline.dims[0]].values,
                    mean_baseline.dims[1]: mean_baseline.coords[mean_baseline.dims[1]].values},
            name="Mean change in precipitation",
            attrs={
                "description": f"Mean precipitation change in target years ({target_period[0]}-{target_period[1]}) compared to baseline years ({baseline_period[0]}-{baseline_period[1]})",
                "model": model,
                "target_scenario": target_scenario,
                "baseline_scenario": baseline_scenario,
                "baseline_period": f"{baseline_period[0]}-{baseline_period[1]}",
                "target_period": f"{target_period[0]}-{target_period[1]}.",
                "units": "mm/4 weeks",
            }
    ),
        'median_pr_change': xr.DataArray(
            median_pr_change,
            dims=median_baseline.dims, 
            coords={median_baseline.dims[0]: median_baseline.coords[median_baseline.dims[0]].values,
                    median_baseline.dims[1]: median_baseline.coords[median_baseline.dims[1]].values},
            name="Median change in precipitation",
            attrs={
                "description": f"Median precipitation change in target years ({target_period[0]}-{target_period[1]}) compared to baseline years ({baseline_period[0]}-{baseline_period[1]})",
                "model": model,
                "target_scenario": target_scenario,
                "baseline_scenario": baseline_scenario,
                "baseline_period": f"{baseline_period[0]}-{baseline_period[1]}",
                "target_period": f"{target_period[0]}-{target_period[1]}.",
                "units": "mm/4 weeks",
            }
    ),
        'mean_significance_mask': xr.DataArray(
            mean_significance_mask,
            dims=mean_baseline.dims, 
            coords={mean_baseline.dims[0]: mean_baseline.coords[mean_baseline.dims[0]].values,
                    mean_baseline.dims[1]: mean_baseline.coords[mean_baseline.dims[1]].values},
            name="Mean significant difference mask",
            attrs={
                "description": f"Significant difference mask in target years ({target_period[0]}-{target_period[1]}) compared to baseline years ({baseline_period[0]}-{baseline_period[1]})",
                "model": model,
                "target_scenario": target_scenario,
                "baseline_scenario": baseline_scenario,
                "baseline_period": f"{baseline_period[0]}-{baseline_period[1]}",
                "target_period": f"{target_period[0]}-{target_period[1]}.",
                "Statistical test": "Independent T-test",
                "units": "Unitless",
    }),
        'median_significance_mask': xr.DataArray(
            median_significance_mask,
            dims=mean_baseline.dims, 
            coords={mean_baseline.dims[0]: mean_baseline.coords[mean_baseline.dims[0]].values,
                    mean_baseline.dims[1]: mean_baseline.coords[mean_baseline.dims[1]].values},
            name="Median significant difference mask",
            attrs={
                "description": f"Significant difference mask in target years ({target_period[0]}-{target_period[1]}) compared to baseline years ({baseline_period[0]}-{baseline_period[1]})",
                "model": model,
                "target_scenario": target_scenario,
                "baseline_scenario": baseline_scenario,
                "baseline_period": f"{baseline_period[0]}-{baseline_period[1]}",
                "target_period": f"{target_period[0]}-{target_period[1]}.",
                "Statistical test": "Mann Whitney U test",
                "units": "Unitless",
    })
    }
    
    return results  # Return results for further plotting

def calculate_cluster_area(cluster_indices, lat, lon):
    """Calculates the approximate area of a cluster in square kilometers
    using spherical geometry, adapted for non-uniform lat/lon spacing.

    Args:
        cluster_indices (np.ndarray): Array of (lat_index, lon_index) for the cluster.
        lat (np.ndarray): 1D array of latitude values (in degrees).
        lon (np.ndarray): 1D array of longitude values (in degrees).

    Returns:
        float: Approximate area of the cluster in square kilometers.
    """
    # Earth's radius in meters
    R = 6371000
    total_area = 0.0

    if lat is None or lon is None:
        return np.nan

    lats = np.array(lat)
    lons = np.array(lon)

    # Create latitude bounds
    lat_bounds = np.zeros(len(lats) + 1)
    lat_bounds[1:-1] = 0.5 * (lats[:-1] + lats[1:])
    lat_bounds[0] = lats[0] - 0.5 * (lats[1] - lats[0])
    lat_bounds[-1] = lats[-1] + 0.5 * (lats[-1] - lats[-2])
    lat_bounds = np.radians(lat_bounds)  # Convert to radians

    # Create longitude bounds
    lon_bounds = np.zeros(len(lons) + 1)
    lon_bounds[1:-1] = 0.5 * (lons[:-1] + lons[1:])
    lon_bounds[0] = lons[0] - 0.5 * (lons[1] - lons[0])
    lon_bounds[-1] = lons[-1] + 0.5 * (lons[-1] - lons[-2])
    lon_bounds = np.radians(lon_bounds)  # Convert to radians

    # Compute area for each grid cell in the cluster
    for lat_idx, lon_idx in cluster_indices:
        phi1 = lat_bounds[lat_idx]
        phi2 = lat_bounds[lat_idx + 1]
        lambda1 = lon_bounds[lon_idx]
        lambda2 = lon_bounds[lon_idx + 1]

        # Ensure lambda_diff is positive
        lambda_diff = np.abs(lambda2 - lambda1)
        sin_diff = np.abs(np.sin(phi2) - np.sin(phi1))
        cell_area = R**2 * lambda_diff * sin_diff
        total_area += cell_area

    return total_area / 1e6  # Convert m^2 to km^2

def find_nearest_idx(array, value, is_lon=False):
    if is_lon:
        array_wrapped = np.mod(array, 360)
        value_wrapped = value % 360
        return np.abs(array_wrapped - value_wrapped).argmin()
    else:
        return np.abs(array - value).argmin()

# === UPDATED weighted_cluster_mean ===
def weighted_cluster_mean(values, coords, area_grid, lat_ref, lon_ref):
    total_weighted = 0
    total_area = 0
    for lat_val, lon_val in coords:
        i = find_nearest_idx(lat_ref, lat_val)
        j = find_nearest_idx(lon_ref, lon_val, is_lon=True)
        value = values[i, j]
        if not np.isnan(value):
            area = area_grid[i, j]
            total_weighted += value * area
            total_area += area
    return total_weighted / total_area if total_area > 0 else np.nan

# === compute_gridcell_areas remains unchanged ===
def compute_gridcell_areas(lat, lon, R=6371000):
    lat_rad = np.radians(lat)
    dlat = np.radians(np.abs(np.gradient(lat)))
    dlon = np.radians(np.abs(np.gradient(lon)))

    area = np.zeros((len(lat), len(lon)))

    for i in range(len(lat)):
        area[i, :] = R**2 * dlat[i] * dlon * np.cos(lat_rad[i])

    return area  # in m²

# Plotting function
def plot_precipitation_change(results, model, scenario, land_mask, target_period, figure_nr, clusters, cluster_map):
    """Plots precipitation change (absolute and %) and cluster IDs using Cartopy, with color-coded cluster signs."""

    def plot_map_cartopy(data_array, title, colorbar_label, output_filename, land_mask=None, cmap='seismic_r'):
        """Plot xarray map with Cartopy and masking."""
        plt.figure(figsize=(10, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_global()

        if land_mask is not None:
            data_array = data_array.where(land_mask > 0)

        vmax = np.nanmax(np.abs(data_array))
        vmin = -vmax

        im = data_array.plot.imshow(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            add_colorbar=False
        )

        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.gridlines(draw_labels=True, linewidth=0.3, linestyle='--', alpha=0.5)

        plt.title(title)
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)
        cbar.set_label(colorbar_label)

        plt.savefig(output_filename, dpi=800, bbox_inches='tight')
        print(f"Saved plot: {output_filename}")
        plt.close()

    # Plot absolute precipitation change
    absolute_change = results['mean_pr_change']
    plot_map_cartopy(
        absolute_change,
        f"{model} {scenario} {target_period[0]}–{target_period[1]}",
        "Mean significant change (mm/4 weeks)",
        f"Drying wetting results/Significant_absolute_Precip_Change_{model}_{scenario}_{target_period[0]}-{target_period[1]}.png",
        land_mask=land_mask
    )

    # Plot percentual precipitation change
    percentual_change = results['mean_pr_change'] / results['mean_baseline_precipitation']
    plot_map_cartopy(
        percentual_change,
        f"{model} {scenario} {target_period[0]}–{target_period[1]}",
        "Mean significant change (%)",
        f"Drying wetting results/Significant_percentual_Precip_Change_{model}_{scenario}_{target_period[0]}-{target_period[1]}.png",
        land_mask=land_mask
    )

    # Categorize clusters
    drying_clusters = [cl for cl in clusters if cl.get('type', '').lower() == 'drying']
    wetting_clusters = [cl for cl in clusters if cl.get('type', '').lower() == 'wetting']
    
    def zigzag_indices(n):
        indices = []
        left, right = 0, n - 1
        while left <= right:
            if left == right:
                indices.append(left)
            else:
                indices.extend([left, right])
            left += 1
            right -= 1
        return indices
    
    # Get shades
    reds_raw = plt.get_cmap('Reds')(np.linspace(0.3, 0.9, len(drying_clusters)))
    blues_raw = plt.get_cmap('Blues')(np.linspace(0.3, 0.9, len(wetting_clusters)))
    
    # Zigzag reorder
    zigzag_r = zigzag_indices(len(reds_raw))
    zigzag_b = zigzag_indices(len(blues_raw))
    reds = reds_raw[zigzag_r]
    blues = blues_raw[zigzag_b]

    # Create full colormap
    max_id = int(np.max([cl['id'] for cl in clusters])) + 1
    cluster_colors = [(0, 0, 0, 0)] * max_id  # Start with all transparent

    for i, cl in enumerate(drying_clusters):
        cluster_colors[cl['id']] = reds[i]
    for i, cl in enumerate(wetting_clusters):
        cluster_colors[cl['id']] = blues[i]

    cmap = mcolors.ListedColormap(cluster_colors)
    bounds = np.arange(len(cluster_colors) + 1) - 0.5
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Plot cluster map
    plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()

    im = cluster_map.plot.imshow(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        add_colorbar=False
    )

    # Add cluster labels in purple
    for cluster in clusters:
        cid = cluster["id"]
        coords = cluster["indices"]
        if coords.size > 0:
            topmost = coords[np.argmin(coords[:, 0])]
            lat = cluster_map.lat.values[int(topmost[0])]
            lon = cluster_map.lon.values[int(topmost[1])]
            ax.text(
                lon, lat, str(cid),
                color='#8300cf', fontsize=6, weight='bold',
                ha='center', va='bottom',
                transform=ccrs.PlateCarree()
            )

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.gridlines(draw_labels=True, linewidth=0.3, linestyle='--', alpha=0.5)

    # Create legend
    legend_elements = [
        Patch(facecolor=reds[len(reds)//2], edgecolor='k', label='Drying Cluster'),
        Patch(facecolor=blues[len(blues)//2], edgecolor='k', label='Wetting Cluster')
    ]
    ax.legend(handles=legend_elements, loc='lower left')

    plt.title(f"Cluster Map with IDs for {model} {scenario} {target_period[0]}–{target_period[1]}")
    output_png_clusters = f"Drying wetting results/Cluster_Map_with_IDs_{model}_{scenario}_{target_period[0]}-{target_period[1]}.png"
    plt.savefig(output_png_clusters, dpi=1200, bbox_inches='tight')
    print(f"Saved plot: {output_png_clusters}")
    plt.close()

def find_land_mask(forcing_directory, model, scenario):
    #select the right land mask for the model
    land_mask = None
    for file in os.listdir(forcing_directory):
        if model in file and scenario in file and 'sftlf' in file:
            try:
                land_mask = xr.open_dataset(os.path.join(forcing_directory, file))['sftlf']
                return land_mask/100  #land mask is given in percentage so convert to fraction by dividing by 100
                break  #stop loop as the right mask was found
            except:
                print('Failed to open land_mask file')
    if not land_mask:
        raise('Failed to find suitable land_mask file for model and scenario')


# Function to find and filter clusters
def find_clusters_drying_wetting(results, min_size, land_mask, baseline_period, target_period, scenario, model):
    global current_cluster_id
    
    """
    Identifies clusters of drying (negative values) and wetting (positive values) regions in a grid.
    Assigns each cluster a unique number chronologically, considering wraparound in the horizontal direction.

    Parameters:
    - results: xarray dataset of the results obtained by calculating the significant changes in precipitation
    - min_size: Minimum size of the clusters to consider.
    - mask indicating where land (1=land, 0=no land)

    Returns:
    - cluster_map: 2D array where each cluster has a unique ID (drying and wetting combined).
    - clusters: List of dictionaries with details for each cluster.
    """

    mean_pr_change_array = land_mask.values * results['mean_pr_change'].values * results['mean_significance_mask'].values #open the mean_pr_change over only signifant changing places and convert to only terrestrial areas
    median_pr_change_array = land_mask.values * results['median_pr_change'].values * results['median_significance_mask']
    
    mean_baseline_pr_array = results['mean_baseline_precipitation'].values
    mean_target_pr_array = results['mean_target_precipitation'].values
    median_baseline_pr_array = results['median_baseline_precipitation'].values
    median_target_pr_array = results['median_target_precipitation'].values
    

    # Create binary masks for drying and wetting regions
    drying_mask = mean_pr_change_array < 0  # Negative values indicate drying
    wetting_mask = mean_pr_change_array > 0  # Positive values indicate wetting

    # Label connected components for drying and wetting regions
    structure = np.array([[1, 1, 1], # means that it takes all directions as connected also diagonal. If structure = np.array([[0,1,0],[1,1,1], [0,1,0]]) it would only take right,left,upper or lower neighnours
                                [1, 1, 1],
                                [1, 1, 1]])
    labeled_drying_grid, num_drying_features = label(drying_mask, structure=structure)
    labeled_wetting_grid, num_wetting_features = label(wetting_mask, structure=structure)

    # Handle wraparound by merging clusters at the left and right edges and make all clusters that touch the upper axis or lower axis into the same cluster
    def merge_wraparound(labeled_grid, num_features):
        """ Ensures clusters are connected across left and right edges. """
        label_map = {}  # Stores equivalences
        for row in range(median_pr_change_array.shape[0]):
            left_label = labeled_grid[row, 0]
            right_label = labeled_grid[row, -1]
            if left_label > 0 and right_label > 0 and left_label != right_label:
                if left_label in label_map:
                    label_map[right_label] = label_map[left_label]
                else:
                    label_map[left_label] = right_label

        # Apply relabeling
        for old_label, new_label in label_map.items():
            labeled_grid[labeled_grid == old_label] = new_label


        # Merge clusters at the top row
        labels_top = set()
        for col in range(median_pr_change_array.shape[1]):
            top_label = labeled_grid[0, col]
            if top_label != 0:
                labels_top.add(top_label)

        if len(labels_top) > 1:
            labels_top = list(labels_top)
            for old_label in labels_top[1:]:  # Merge all into the first label
                labeled_grid[labeled_grid == old_label] = labels_top[0]

        # Merge clusters at the bottom row
        labels_bottom = set()
        for col in range(median_pr_change_array.shape[1]):
            bottom_label = labeled_grid[-1, col]
            if bottom_label != 0:
                labels_bottom.add(bottom_label)

        if len(labels_bottom) > 1:
            labels_bottom = list(labels_bottom)
            for old_label in labels_bottom[1:]:  # Merge all into the first label
                labeled_grid[labeled_grid == old_label] = labels_bottom[0]

        return labeled_grid

    labeled_drying_grid = merge_wraparound(labeled_drying_grid, num_drying_features)
    labeled_wetting_grid = merge_wraparound(labeled_wetting_grid, num_wetting_features)

    # Initialize a new grid for uniquely numbered clusters
    cluster_map = np.zeros_like(median_pr_change_array, dtype=int)

    # Initialize a list to store cluster details
    clusters = []

    # Get latitude and longitude arrays from the results
    lat = results['mean_baseline_precipitation'].lat.values
    lon = results['mean_baseline_precipitation'].lon.values
    
    area_grid = compute_gridcell_areas(lat, lon)
    
    # Convert the cluster_map to an xarray.DataArray
    cluster_map = xr.DataArray(cluster_map, dims=["lat", "lon"], coords={"lat": lat, "lon": lon})

    # Define cluster types for iteration
    cluster_types = {
        "drying": labeled_drying_grid,
        "wetting": labeled_wetting_grid
    }
    
    # Calculate the significant mean change of treefrac according to the scenario
    baseline_treefrac, target_treefrac, treefrac_change_map = significant_change_in_treefrac(scenario, baseline_period, target_period)

    #calculate the significant mean change in temperature compared to the baseline for the cluster
    base_temp_mean, target_temp_mean, significant_temp_change_map = significant_change_in_temp(model, scenario, baseline_period, target_period)


    # Process both drying and wetting clusters
    for cluster_type, labeled_grid in cluster_types.items():
        for cluster_id in range(1, np.max(labeled_grid) + 1):
            cluster_indices = np.argwhere(labeled_grid == cluster_id)

            if len(cluster_indices) >= min_size:
                cluster_mask = np.zeros_like(median_pr_change_array, dtype=int)

                for x, y in cluster_indices:
                    cluster_map[x, y] = current_cluster_id
                    cluster_mask[x, y] = 1

                # Calculate approximate center of mass
                center_idx_lat = (np.max(cluster_indices[:, 0]) + np.min(cluster_indices[:, 0])) // 2
                center_idx_lon = (np.max(cluster_indices[:, 1]) + np.min(cluster_indices[:, 1])) // 2
                
                nlat = cluster_map.shape[0]
                nlon = cluster_map.shape[1]
                
                center_lat = -90 + (center_idx_lat/(nlat)) * 180
                center_lon = (center_idx_lon/(nlon)) * 360

                # Calculate the approximate area of the cluster
                cluster_area = calculate_cluster_area(cluster_indices, lat, lon)
                
                #calculate the actual coords from the indices
                cluster_coords = [(float(lat[x]), float(lon[y])) for x, y in cluster_indices]
                
                
                #calculate the treefrac change on average over the coords
                # Assume this comes from the treefrac result
                if isinstance(treefrac_change_map, xr.DataArray):
                    treefrac_lats = treefrac_change_map['lat'].values
                    treefrac_lons = treefrac_change_map['lon'].values
                    treefrac_change_data = treefrac_change_map.values
                    treefrac_baseline_data = baseline_treefrac.values
                    treefrac_target_data = target_treefrac.values
                else:
                    raise ValueError("Expected significant_change_in_treefrac to return an xarray.DataArray")
                
                # Calculate mean treefrac values if values exist
                mean_treefrac_change = weighted_cluster_mean(treefrac_change_data, cluster_coords, area_grid, treefrac_lats, treefrac_lons)
                mean_treefrac_baseline = weighted_cluster_mean(treefrac_baseline_data, cluster_coords, area_grid, treefrac_lats, treefrac_lons)
                mean_treefrac_target = weighted_cluster_mean(treefrac_target_data, cluster_coords, area_grid, treefrac_lats, treefrac_lons)
                
                mean_sig_temp_change = weighted_cluster_mean(significant_temp_change_map.values, cluster_coords, area_grid, lat, lon)
                mean_temp_baseline = weighted_cluster_mean(base_temp_mean.values, cluster_coords, area_grid, lat, lon)
                mean_temp_target = weighted_cluster_mean(target_temp_mean.values, cluster_coords, area_grid, lat, lon)
                
                mean_pr_change = weighted_cluster_mean(mean_pr_change_array, cluster_coords, area_grid, lat, lon)
                mean_baseline_pr = weighted_cluster_mean(mean_baseline_pr_array, cluster_coords, area_grid, lat, lon)
                mean_target_pr = weighted_cluster_mean(mean_target_pr_array, cluster_coords, area_grid, lat, lon)


                clusters.append({
                    "model": model,
                    "scenario": scenario,
                    "baseline_period": baseline_period,
                    "target_period": target_period,
                    "id": current_cluster_id,
                    "type": cluster_type,
                    "indices": cluster_indices,
                    "coords": cluster_coords,
                    "mask_array": cluster_mask,
                    "mean_significant_pr_change": mean_pr_change, #calculate the average change in precipitation in mm per gridcell in the cluster
                    "mean_baseline_precipitation": mean_baseline_pr, #calculate the average baseline precipitation rate (mm/4weeks) per gridcell in the cluster
                    "mean_target_precipitation": mean_target_pr,
                    "center_lat": float(center_lat),
                    "center_lon": float(center_lon),
                    "area_km2": float(cluster_area),
                    "number_of_gridcells": len(cluster_indices),
                    "mean_temp_baseline (K)": mean_temp_baseline,
                    "mean_temp_target (K)": mean_temp_target,
                    "mean_significant_temp_change (K)": mean_sig_temp_change,
                    "mean_baseline_treefrac (%)": mean_treefrac_baseline,
                    "mean_target_treefrac (%)": mean_treefrac_target,
                    "mean_significant_treefrac_change (%)": mean_treefrac_change
                })
                current_cluster_id += 1


    return clusters, cluster_map

# Main loop
models = ['EC-Earth3', 'MPI-ESM1-2-HR', 'NorESM2-MM', 'TaiESM1']
scenarios = ['ssp126', 'ssp245', 'ssp370', 'ssp585']

baseline_period = (2015, 2024)
baseline_scenario = 'ssp245'
target_periods = [(2050, 2059), (2090, 2099)]

#%% main loop for producing results for all models and scenarios regarding drying or wetting regions
figure_nr = 1

# Initialize cluster ID counter
current_cluster_id = 1

all_clusters = []

for model in models:
    forcing_directory = f"F:\\Forcing_data_{model}"
    for scenario in scenarios:
        for target_period in target_periods:
            # Compute precipitation changes
            # results = calculate_drying_wetting_to_nc(model, forcing_directory, baseline_scenario, baseline_period, scenario, target_period)
            
            # If results are already generated once, use the line below to load the data instead of regenerating as this costs much time
            results = xr.open_dataset(fr"C:\Users\chiel\OneDrive - Universiteit Utrecht\Master year 2 (2024-2025)\Thesis\Code files\Drying wetting results (1)\Significantly_drying_wetting_map_{model}_{scenario}_{target_period[0]}-{target_period[1]}.nc")
            
            # Load the land mask
            land_mask = find_land_mask(forcing_directory, model, scenario)
            
            # Drying and wetting clusters in a txt file as by copying their coordinates (latidx and lonidx)
            clusters, cluster_map = find_clusters_drying_wetting(results, 50, land_mask, baseline_period, target_period, scenario, model)    #the number indicates the minimal size of a cluster
            
            #add all dictionaries of each cluster to one final clusters list
            all_clusters.extend(clusters)
            
            results['clustermap'] = xr.DataArray(
                                cluster_map,
                                dims=results['mean_baseline_precipitation'].dims, 
                                coords={results['mean_baseline_precipitation'].dims[0]: results['mean_baseline_precipitation'].coords[results['mean_baseline_precipitation'].dims[0]].values,
                                        results['mean_baseline_precipitation'].dims[1]: results['mean_baseline_precipitation'].coords[results['mean_baseline_precipitation'].dims[1]].values},
                                name="Clusters of signifcantly changing precipitation",
                                attrs={
                                    "description": "Clusters of signifcantly changing precipitation in target years compared to baseline years.",
                                    "model": model,
                                    "scenario": scenario,
                                    "baseline_period": f"{baseline_period[0]}-{baseline_period[1]}",
                                    "target_period": f"{target_period[0]}-{target_period[1]}",
                                    "units": "Indexed clusters"
                                }
                            )
            
            os.makedirs(os.path.join(os.path.dirname(__file__),"Drying wetting results"), exist_ok=True)  
            
            # Save result as NetCDF
            output_filename = f"Drying wetting results\Significantly_drying_wetting_map_{model}_{scenario}_{target_period[0]}-{target_period[1]}.nc"
            xr.Dataset(results).to_netcdf(output_filename)
            print(f"Saved: {output_filename}")
            
            #save the masks of each clusters separately in a list of dictionaries to easily acces them later
            clusters_fn = rf"Drying wetting results\Drying_wetting_cluster_info_{model}_{scenario}_{target_period[0]}-{target_period[1]}.npy"
            np.save(clusters_fn, clusters)
            
            # Plot and save
            plot_precipitation_change(results, model, scenario, land_mask, target_period, figure_nr, clusters, cluster_map)
            
            figure_nr += 1
            
#Save the cluster properties as an excel file for easy analysis
# Convert to DataFrame
df = pd.DataFrame(all_clusters)
df = df.drop(['indices', 'coords', 'mask_array'], axis=1) #remove the keys that have really large datasets which are not suitable for excel
# Save to excel
df.to_excel("cluster_data.xlsx")
