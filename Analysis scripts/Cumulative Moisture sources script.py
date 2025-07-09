import xarray as xr
import os
import pandas as pd
import numpy as np
#%%
# Main loop
models = ['EC-Earth3', 'MPI-ESM1-2-HR', 'NorESM2-MM', 'TaiESM1']
scenarios = ['ssp126', 'ssp245', 'ssp370', 'ssp585']

baseline_period = (2015, 2024)
baseline_scenario = 'ssp245'
target_periods = [(2050, 2059), (2090, 2099)]

counter = 0
for model in models:
    folder = os.path.join(os.path.dirname(__file__), f'{model}-output')
    for scenario in scenarios:
        for clusterstype in ['drying', 'wetting']:
            for start_year, end_year in target_periods:
                final_footprint = None
                for filename in os.listdir(folder):
                    try:
                        sim_start_year = int(filename.split('_')[2][-4:])
                        sim_end_year = int(filename.split('_')[3][-4:])
                        cluster_id_file = int(filename.split('_')[-3].split('-')[1])
                    except:
                        continue
                    
                    if 'Cluster' in filename and model in filename and scenario in filename and clusterstype in filename and ((end_year+5) >= sim_start_year >= (start_year-5) or (end_year+5) >= sim_end_year >= (start_year-5)):
                        filepath = os.path.join(folder, filename)
                        try:
                            ds = xr.open_dataset(filepath)
                            footprint = ds['footprint']
                            size_mask = np.sum(ds['simulation mask'])
                            
                            # --- ADD THIS NEW FILTERING BLOCK ---
                            if model == 'EC-Earth3' and size_mask < 119:
                                print(f"      FILTERED: Skipping Cluster ID {cluster_id_file} for model {model} (rule: < 118).")
                                continue # Skip to the next cluster
                            
                            if model == 'MPI-ESM1-2-HR' and size_mask < 67:
                                print(f"      FILTERED: Skipping Cluster ID {cluster_id_file} for model {model} (rule: < 66).")
                                continue # Skip to the next cluster
                            
                            if final_footprint is None:
                                final_footprint = footprint
                            else:
                                final_footprint += footprint
                            counter+=1
                        except Exception as e:
                            print(f"Error processing {filepath}: {e}")
                            continue  # Skip to the next file if there's an error
        
                if final_footprint is not None:
                    try:
                        final_footprint.to_netcdf(f'Cumulative_moisture_sources_{clusterstype}_{model}_{scenario}_{start_year}-{end_year}.nc')
                        print(f"Successfully created: Cumulative_moisture_sources_{clusterstype}_{model}_{scenario}_{start_year}-{end_year}.nc")
                    except Exception as e:
                        print(f"Error saving Cumulative_moisture_sources_{clusterstype}_{model}_{scenario}_{start_year}-{end_year}.nc: {e}")
                else:
                    print(f"No footprint files found for model {model} and scenario {scenario}.")

print(f"Script finished. {counter} files processed!")