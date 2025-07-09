# Imports
from netCDF4 import Dataset
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta  # Add this import
from UTrack_combinedESM_functions import *
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import xarray as xr
import csv
import statistics

#Models, scenarios and target years to loop over
models = ['TaiESM1', 'EC-Earth3', 'MPI-ESM1-2-HR', 'NorESM2-MM']
scenarios = ['ssp126', 'ssp245', 'ssp370', 'ssp585']

baseline_period = (2015, 2024)
baseline_scenario = 'ssp245'
target_periods = [(2050, 2059), (2090, 2099)]

# SIMULATION SETTINGS
parcels_mm = 100                     # Number of parcels to be released per mm evaporation/precipitation
scenario = 'ssp245'                 # SSP of the simulation
dt_hours = 4                              # Set timestep length in hours
forward_tracking = False             # implement forward or backward tracking
dynamic_plotting = False            # dynamically plot the parcels and allocation
tracking_days = 30                  # days that parcels are tracked
kill_threshold = 0.01               # Set minimum amount of moisture present in a parcel

#%% count total number of masks to perform a simulation for, which is the max number of sims
number_of_possible_sims = 0
for model in models:
    # Directory where data for modelchoice is stored
    forcing_directory = rf"{find_forcing_files_drive_letter()}:\Forcing_data_{model}"
    for scenario in scenarios:
        for start_year, end_year in target_periods:
            #load the data about the drying and wetting clusters globally
            drying_wetting_clusters_list = np.load(fr"{os.path.dirname(__file__)}\Drying wetting results\Drying_wetting_cluster_info_{model}_{scenario}_{start_year}_{end_year}.npy"
                           , allow_pickle=True).tolist() #example mask from the cluster files I saved
            #count number of clusters in list
            number_of_clusters = len(drying_wetting_clusters_list)
            for cluster in drying_wetting_clusters_list:
                for year in range(start_year, end_year + 1):
                    for month in range(1, 13):
                        number_of_possible_sims += 1

#%%
start_simulation = datetime.now()
simulation_times = []  # Store simulation times
simulation_nr = 0

#LOOP OVER THE DIFFERENT MASKS FOR DIFFERENT MODELS, SCENARIOS AND TIMEFRAMES

for targ_period in target_periods:
    start_year_target, end_year_target = targ_period
    start_year_baseline, end_year_baseline = baseline_period
    for year in list(range(start_year_target, end_year_target + 1))  + list(range(start_year_baseline, end_year_baseline + 1)): #run for the months in the target years as well as for the comparable baseline months in the years
        for month in range(1, 13):
            if month == 1 and year == 2015:  #this run is impossible as the data before 2015 is not available and the runs need to keep tracking until 30 days before the release end date (1-1-2015)
                continue
            for model in models:
                # Directory where data for modelchoice is stored
                forcing_directory = rf"{find_forcing_files_drive_letter()}:\Forcing_data_{model}"
                for scenario in scenarios:
                    #load the data about the drying and wetting clusters globally
                    drying_wetting_clusters_list = np.load(fr"{os.path.dirname(__file__)}\Drying wetting results\Drying_wetting_cluster_info_{model}_{scenario}_{start_year_target}_{end_year_target}.npy"
                                   , allow_pickle=True).tolist() #example mask from the cluster files I saved
                    for cluster in drying_wetting_clusters_list:

                        release_end_date = datetime(year, month, 1)
                        start_date = release_end_date + relativedelta(months=1) #reversed due to backward tracking
                        
                        #details about specific clusters
                        baseline_period = cluster['baseline_period']
                        target_period = cluster['target_period']
                        cluster_id = cluster['id']
                        cluster_area_km2 = cluster['area_km2']
                        center_lat = cluster['center_lat']
                        center_lon = cluster['center_lon']
                        mask = cluster['mask_array']
                        mean_prec_change = cluster['mean_significant_pr_change']
                        median_prec_change = cluster['median_precipitation_change']
                        drying_or_wetting_sim = cluster['type']
                        size_cluster_in_gridcells = cluster['coords'].shape[0]
                        mean_baseline_pr = cluster['mean_baseline_precipitation']
                        mean_target_pr = cluster['mean_target_precipitation']
                        median_baseline_pr = cluster['median_baseline_precipitation']
                        median_target_pr = cluster['median_target_precipitation']
                        
                        if start_date.year < 2040: 
                            century_indication_cluster = "Start of the century"
                        if 2040 < start_date.year < 2070:
                            century_indication_cluster = "Middle of the century"
                        if start_date.year > 2070: 
                            century_indication_cluster = "End of the century"
                        
                            
                        if start_year_baseline <= start_date.year <= end_year_baseline:
                            simulation_type = 'Baseline'
                        else:
                            simulation_type = 'Target'
                        
                        #SIMULATION LOGIC FROM HERE
                        
                        # Convert to datetime object that are easily worked with
                        tracking_time = timedelta(days=tracking_days) 
                        dt = timedelta(hours=dt_hours)
                        
                        #set release_end_date according to tracking time
                        if forward_tracking:
                            end_date = release_end_date + tracking_time
                        else:
                            end_date = release_end_date - tracking_time
                        
                        #check if the dates are set and if all files are available for the specific simulation
                        if forward_tracking and start_date > release_end_date:
                            raise ValueError('For forward tracking, the release_end_date MUST be later than the start_date.')
                        if not forward_tracking and start_date < release_end_date:
                            raise ValueError('For backward tracking, the release_end_date must NOT be later than the start_date.')
                        if abs(end_date - start_date) < tracking_time:
                            raise ValueError('The difference in days between the start and end date cannot be smaller than the tracking time')
                        
                        #check if all years of climate data are available for the selected model and scenario
                        # check_file_availability(forcing_directory, start_date.year, end_date.year, model, scenario)
                        
                        # INITIALISATION
                        try:
                            # Create empty meteo objects for holding forcing data
                            m = Meteo()
                            # # Load forcing data for the current day
                            m.read_from_file(start_date.year, start_date.month, start_date.day, scenario, forcing_directory, model)
                            #calculate the degreelengths initially before starting the simulation loop so it can be accessed through the meteo object throughout the simulation without the need to recalculate every loop
                            m.compute_degree_lengths()
                        except:
                            simulation_nr +=1
                            continue  #skip if the files are not available for the particular simulation
                        
                        # SIMULATION
                        print(f'Model: {model} Scenario: {scenario} From {start_date.day}-{start_date.month}-{start_date.year} till {end_date.day}-{end_date.month}-{end_date.year}')#', Cells to go: {str(to_go)}.')
                        
                        # Create output array with the dimensions of the forcing data
                        output_array = np.zeros([len(m.lats), len(m.lons)])
                        
                        parcel_data_columns =  ['latitude', 'longitude', 'level', 'moisture_present', 'original_moisture', 'time', 'start_time', 'startlatidx', 
                                               'startlonidx', 'current_latidx', 'current_lonidx']
                        
                        # Define indexes for parcel columns in parcel_data array
                        LAT_IDX = parcel_data_columns.index('latitude')
                        LON_IDX = parcel_data_columns.index('longitude')
                        LEVEL_IDX = parcel_data_columns.index('level')
                        MOISTURE_IDX = parcel_data_columns.index('moisture_present')
                        ORIGINAL_MOISTURE_IDX = parcel_data_columns.index('original_moisture')
                        TIME_IDX = parcel_data_columns.index('time')
                        START_TIME_IDX = parcel_data_columns.index('start_time')
                        START_LATIDX = parcel_data_columns.index('startlatidx')
                        START_LONIDX = parcel_data_columns.index('startlonidx')
                        CURRENT_LATIDX = parcel_data_columns.index('current_latidx')
                        CURRENT_LONIDX = parcel_data_columns.index('current_lonidx')
                        
                        if forward_tracking: track_key_word = 'Forward'
                        else: track_key_word = 'Backward'
                        
                        output_directory = os.path.join(os.path.dirname(__file__), f"{model}-output")
                        output_fn = os.path.join(output_directory, 
                        f"Utrack-{track_key_word}-{model}_{scenario}_{start_date.day}-{start_date.month}-{start_date.year}_{release_end_date.day}-{release_end_date.month}-{release_end_date.year}_{drying_or_wetting_sim}_Cluster-{cluster_id}_Baseline({baseline_period[0]}-{baseline_period[1]})_Target({target_period[0]}-{target_period[1]}).nc")
                        # Check if the file exists before running the simulation
                        if os.path.exists(output_fn):
                            simulation_nr +=1
                            print(f"Simulation {simulation_nr}/{number_of_possible_sims} already completed. Skipping...")
                            continue
                    
                        parcels_data = np.empty((0, len(parcel_data_columns)))
                        killed_expired_parcels_data = np.empty((0, 12))
                        
                        # Set the model timer
                        current_date = start_date
                        
                        # Define tracking variables
                        parcels_in, killed, expired, total_released, max_parcel_count  = 0, 0, 0, 0, 0
                        
                        # print(
                        #     "Date   Time    Parcels in system   Parcels added   Parcels destroyed  Simulation time of time step (s)  Fraction of moisture allocated  Estimated time left")
                        
                        # SETTINGS REGARDING DYNAMIC VISUALISATION
                        
                        if dynamic_plotting:
                            #open land mask to use for the map that gets plotted in the loop
                            land_mask = find_land_mask(forcing_directory, model, scenario)
                            # Create a linear segmented colormap from white to blue in 100 steps
                            cmap = mcolors.LinearSegmentedColormap.from_list("white_to_blue", [(1, 1, 1), (0, 0, 1)], N=100)
                            # Create the plot outside the loop
                            fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size if needed
                            # Create a land mask contour
                            ax.contour(land_mask, levels=[0.5], colors='k', linewidths=1)
                            
                            img = plt.imshow(output_array, cmap=cmap, interpolation='nearest', origin='lower', vmin=0)  # Initial plot of the output array
                            fig.colorbar(img, ax=ax, label="mm moisture allocated")  # Optionally add a colorbar for better visualization
                        
                            # Create a scatter plot for parcels outside the loop (initial empty scatter)
                            parcel_scatter = ax.scatter([], [], color='red', marker='o', s=0.5, label="Parcels")
                        
                            # Enable interactive mode
                            plt.ion()  # Turn on interactive mode for real-time updates
                            singlepoint = False
                            if singlepoint:
                                # Highlight the singlepoint with a larger, brighter color and higher zorder
                                singlepoint_scatter = ax.scatter(lonidx_singlepoint, latidx_singlepoint, 
                                                             color='green', marker='o', s=20, zorder=3, alpha=0.7, label="Single Point")  # Increased size and bright color
                                ax.legend(loc='upper right')  # Show the legend in the top-right corner
                            
                            else:
                                # Create mask as transparent region to show the region of interest
                                ax.imshow(np.where(mask == 1, 1, np.nan), cmap="Greys_r", alpha=0.5, origin='lower', zorder=2)
                        
                        
                        start_full_simulation = datetime.now()
                        timing_prev = datetime.now()
                        # Repeat for each time step
                        while True:
                        
                            # Check stopping conditions
                            if forward_tracking and current_date > end_date:
                                break
                            if not forward_tracking and current_date < end_date:
                                break
                        
                            # Handle 'noleap' calendar (skip Feb 29)
                            if m.calendar == 'noleap' and current_date.month == 2 and current_date.day == 29:
                                current_date += dt if forward_tracking else -dt
                                continue
                            
                            # Load forcing data
                            if current_date.hour == 0:
                                m.read_from_file(current_date.year, current_date.month, current_date.day, scenario, forcing_directory, model)
                        
                            # Release moisture parcels
                            if (forward_tracking and current_date <= release_end_date) or (not forward_tracking and current_date >= release_end_date):
                                new_parcels = create_parcels_mask(mask, current_date, dt, m, parcels_mm, forward_tracking, parcel_data_columns)
                                num_new_parcels = new_parcels.shape[0]
                                if num_new_parcels > 0: #only add parcels if there are any
                                    parcels_data = np.concatenate((parcels_data, new_parcels))
                                    total_released += np.sum(new_parcels[:, ORIGINAL_MOISTURE_IDX])     # Sum original moisture
                        
                        
                            # Progress moisture parcels
                            if parcels_data.size > 0: #only update parcels if there are any
                                output_array = update_parcels(parcels_data, dt.total_seconds(), m.ua, m.va, m.wap, m.pr, m.evspsbl, m.prw, m.hus, m.levels, m.lats, m.lons, m.degreelength_lat, m.degreelength_lon, forward_tracking, output_array)
                        
                        
                            # Destroy moisture parcels
                            
                            # Compute which parcels should be labeled as killed or expired
                            present_fraction = parcels_data[:, MOISTURE_IDX] / parcels_data[:, ORIGINAL_MOISTURE_IDX]  # Column 3 = moisture present, Column 4 = original moisture
                            kill_mask = present_fraction < kill_threshold  # Parcels that should be killed
                            expire_mask = np.abs(parcels_data[:, TIME_IDX] - parcels_data[:, START_TIME_IDX]) > tracking_time.total_seconds()  # Tracking time exceeded
                            
                            # Combine masks
                            combined_mask = np.logical_or(kill_mask, expire_mask)
                            
                            ##POTENTIALLY ADD CODE TO SAVE ALL PARCELS THAT HAVE BEEN ALIVE IN THE SIMULATION 
                            
                            # Filter parcels_data based on combined mask
                            parcels_data = parcels_data[np.logical_not(combined_mask)]
                            
                            killed = np.sum(kill_mask)
                            expired = np.sum(expire_mask)
                            parcels_destroyed = killed + expired
                            parcels_in = parcels_data.shape[0]
                        
                        
                            # Update visualization (IF REMOVED IT CAN REDUCE SIMULATION TIMES SIGNIFCANTLY)
                            if dynamic_plotting:
                                parcel_latidxs = parcels_data[:, CURRENT_LATIDX]
                                parcel_lonidxs = parcels_data[:, CURRENT_LONIDX]
                                
                                img.set_data(output_array)
                                img.set_clim(vmin=0, vmax=np.max(output_array))
                                parcel_scatter.set_offsets(np.c_[parcel_lonidxs, parcel_latidxs])
                                ax.set_title(f"Output Map at {current_date.strftime('%Y-%m-%d %H:%M')}\nParcels in system: {len(parcels_data)}; Allocated: {round((np.sum(output_array) / total_released)*100, 2) if total_released else 0}%")
                                
                                fig.canvas.draw()
                                fig.canvas.flush_events()
                                plt.pause(0.001)
                        
                            
                            # Print simulation status
                            # if current_date.hour == 0:    
                            #     print(current_date, parcels_in, parcels_destroyed, (datetime.now() - timing_prev).total_seconds(), 
                            #           round(np.sum(output_array) / total_released, 2) if total_released else 0)
                            
                            if parcels_in > max_parcel_count:
                                max_parcel_count = parcels_in
                                
                            
                            timing_prev = datetime.now()
                        
                            # Progress time forward or backward
                            current_date += dt if forward_tracking else -dt
                            
                        end_full_simulation = datetime.now()
                        simulation_nr += 1
                        simulation_time = (end_full_simulation - start_full_simulation).total_seconds()
                        simulation_times.append(simulation_time)
                        
                        # Calculate average runtime/simulation
                        average_time = statistics.median(simulation_times) # use all simulations if there are less simulations than the window size
                        
                        seconds_left = average_time * (number_of_possible_sims - simulation_nr)
                        
                        print(f"Finished simulation {simulation_nr}/{number_of_possible_sims} in {simulation_time} seconds")
                        print(f"Estimated time left: {timedelta(seconds=seconds_left)}")
                        
                        if dynamic_plotting:
                            plt.ioff()
                            plt.show() #show plot at the end
                            
                        # Construct the output filename
                        create_nc_file(output_fn, output_directory, forward_tracking, model, scenario, start_date, release_end_date, 
                                       mask, m, output_array, total_released, parcels_mm, tracking_time, baseline_period, target_period, 
                                       cluster_id, drying_or_wetting_sim)
                        
                        #calculate the fraction of the initial moisture that has been allocated before the footprint is rescaled to 1
                        fraction_allocated = (np.sum(output_array) / total_released) if total_released > 0 else 0
                        
                        land_mask = find_land_mask(forcing_directory, model, scenario)
                        
                        # Compute moisture that ends up on land
                        land_moisture = np.sum(output_array * land_mask)
                        
                        # Compute percentage of tracked moisture that reaches land
                        land_moisture_percentage = (land_moisture / np.sum(output_array)) * 100 if total_released > 0 else 0   
                        
                        #calculate the amount of moisture ending up in the mask itself
                        cluster_moisture = np.sum(output_array[mask == 1])
                        
                        cluster_moisture_percentage = (cluster_moisture / np.sum(output_array)) * 100 if total_released > 0 else 0
                            
                        csv_file = os.path.join(os.path.dirname(__file__), "simulation_results.csv")
                        
                        # Check if file exists to write headers if necessary
                        write_header = not os.path.exists(csv_file)
                        
                        with open(csv_file, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            
                            if write_header:
                                writer.writerow(["Creation datetime", "Model", "Scenario", "Parcels/mm released", "Tracking time (days)", "Cluster_ID", "Size_Cluster_Gridcells", "cluster_area_km2", "Center_lat(°N)", "Center_lon(°E)", "Drying/Wetting", "Sim_Start_Date", 
                                                 "Sim_Release_End_Date", "Sim_End_date", "start_baseline_years_cluster", "end_baseline_years_cluster", "Simulationtype", "start_target_years_cluster", 
                                                 "end_target_years_cluster", "Century_indication", "Total_Released (mm)", "Max_Parcels", "Fraction_Allocated",
                                                 "Mean_Prec_Change (mm/gridcell/4 weeks)", "mean_baseline_precipitation (mm/gridcell/4 weeks)", "mean_target_precipitation (mm/gridcell/4 weeks)", 
                                                 "median_Prec_Change (mm/gridcell/4 weeks)", "median_baseline_precipitation (mm/gridcell/4 weeks)", "median_target_precipitation (mm/gridcell/4 weeks)", 
                                                 "TMR (mm)", "TMRR (%)", "CMR (mm)", "CMRR (%)"])
                            
                            writer.writerow([end_full_simulation, model, scenario, parcels_mm, tracking_days, cluster_id, size_cluster_in_gridcells, cluster_area_km2, center_lat, center_lon, drying_or_wetting_sim, 
                                             start_date.strftime('%Y-%m-%d'), release_end_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), 
                                             start_year_baseline, end_year_baseline, simulation_type, start_year_target, end_year_target, century_indication_cluster,
                                             total_released, max_parcel_count, fraction_allocated, mean_prec_change, mean_baseline_pr, 
                                             mean_target_pr, median_prec_change, median_baseline_pr, median_target_pr,  
                                             land_moisture, land_moisture_percentage, cluster_moisture, cluster_moisture_percentage])
