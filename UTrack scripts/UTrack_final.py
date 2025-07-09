# Imports
from netCDF4 import Dataset
from datetime import datetime, timedelta
from UTrack_functions_final import *
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import xarray as xr

#Models and scenarios that can be used when using ESMs (not ERA5)
models = ['EC-Earth3', 'MPI-ESM1-2-HR', 'NorESM2-MM', 'TaiESM1', 'ERA5']
scenarios = ['ssp126', 'ssp245', 'ssp370', 'ssp585']

###############################################################################
#############                SIMULATION SETTINGS                  #############
###############################################################################

# SIMULATION SETTINGS
model = 'EC-Earth3'   #this is case sensitive and the following models are currently possible ['EC-Earth3', 'MPI-ESM1-2-HR', 'NorESM2-MM', 'TaiESM1', 'ERA5']
start_date = '1-10-2015'
release_end_date = '15-10-2015'
parcels_mm = 100                     # Number of parcels to be released per mm evaporation/precipitation
scenario = 'ssp245'                 # SSP of the simulation only applicable to ESM's not to ERA5 data
delta_t = 0.25                             # Set timestep length in hours
forward_tracking = True             # implement forward or backward tracking
dynamic_plotting = True            # dynamically plot the parcels and allocation
surface_area_weighting = False       # Do you want the simulation to consider the actual gridcell sizes for distributing the parcels over the mask
tracking_days = 15                 # days that parcels are tracked
kill_threshold = 0.01               # Set minimum amount of moisture present in a parcel
singlepoint = True  # Run the model for a single coordinate defined below
single_point_coords = (-20.891993,-50.911292)  # If singlepoint is set to True, define the latitude and longitude of the source cell here

# Directory where data for modelchoice is stored
forcing_directory = f"{find_forcing_files_drive_letter()}:\Forcing_data_{model}"




#if not singlepoint insert a mask here
#insert the mask from where you want parcels of the simulation to start (backward or forward tracking)
#mask = np.load(fr"{os.path.dirname(__file__)}\Drying wetting results\Drying_wetting_cluster_info_EC-Earth3_ssp585_2090-2099.npy"
#               , allow_pickle=True).tolist()[1]['mask_array'] #example mask from the cluster files I saved

###############################################################################
#############      TIME CONVERSION FORWARD/BACKWARD CONFIG        #############
###############################################################################

# Convert to datetime object that are easily worked with
start_date = datetime.strptime(start_date, "%d-%m-%Y")
release_end_date = datetime.strptime(release_end_date, "%d-%m-%Y")
tracking_time = timedelta(days=tracking_days) 
dt = timedelta(hours=delta_t)

#set release_end_date according to tracking time and set the track key word
if forward_tracking:
    end_date = release_end_date + tracking_time
    track_key_word = 'Forward'
else:
    end_date = release_end_date - tracking_time
    track_key_word = 'Backward'

###############################################################################
#############                      VALIDATION                     #############
###############################################################################


#check if the dates are set and if all files are available for the specific simulation
if forward_tracking and start_date > release_end_date:
    raise ValueError('For forward tracking the release_end_date must be later than the start_date.')
if not forward_tracking and start_date < release_end_date:
    raise ValueError('For backward tracking the release_end_date must NOT be later than the start_date.')
if abs(end_date - start_date) < tracking_time:
    raise ValueError('The difference in days between the start and end date cannot be smaller than the tracking time')

#check if all years of climate data are available for the selected model and scenario
check_file_availability(forcing_directory, start_date, end_date, model, scenario)

#Set the right output directory and check whether the simulation has already been executed before (file exists)
output_directory = os.path.join(os.path.dirname(__file__), f"{model}-output")
output_fn = os.path.join(output_directory, 
f"Utrack-{track_key_word}-{model}_{scenario}_{start_date.day}-{start_date.month}-{start_date.year}_{release_end_date.day}-{release_end_date.month}-{release_end_date.year}.nc")
# Check if the file exists before running the simulation
if os.path.exists(output_fn):
    raise FileExistsError(f"Simulation already completed. Output file '{output_fn}' exists.")


###############################################################################
#############                 INITIALISATION                      #############
###############################################################################

# Create empty meteo objects for holding forcing data
m = Meteo()
# # Load forcing data for the current day
m.read_from_file(start_date, scenario, forcing_directory, model)
#calculate the degreelengths initially before starting the simulation loop so it can be accessed through the meteo object throughout the simulation without the need to recalculate every loop
m.compute_degree_lengths()
m.compute_grid_cell_areas()

if singlepoint:
    mask = np.zeros(m.pr.shape)
    latidx_singlepoint, lonidx_singlepoint = get_nearest_index(single_point_coords[0], single_point_coords[1], m.lats, m.lons)
    mask[latidx_singlepoint, lonidx_singlepoint] = 1

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

parcels_data = np.empty((0, len(parcel_data_columns)),dtype=np.float32)

# Set the model timer
current_date = start_date

# Define tracking variables
parcels_in, killed, expired, total_released = 0, 0, 0, 0

# SIMULATION
print(f'Model: {model} Scenario: {scenario} From {start_date.day}-{start_date.month}-{start_date.year} till {end_date.day}-{end_date.month}-{end_date.year}')
print("Date   Time    Parcels in system   Parcels added   Parcels destroyed  Simulation time of time step (s)  Fraction of moisture allocated")

###############################################################################
#############         DYNAMIC VISUALISATION CONFIGURATION         #############
###############################################################################

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
    
    if singlepoint:
        # Highlight the singlepoint with a larger, brighter color and higher zorder
        singlepoint_scatter = ax.scatter(lonidx_singlepoint, latidx_singlepoint, 
                                     color='green', marker='o', s=20, zorder=3, alpha=0.7, label="Single Point")  # Increased size and bright color
        ax.legend(loc='upper right')  # Show the legend in the top-right corner
    
    else:
        # Create mask as transparent region to show the region of interest
        ax.imshow(np.where(mask == 1, 1, np.nan), cmap="Greys_r", alpha=0.5, origin='lower', zorder=2)


###############################################################################
#############                 SIMULATION LOOP                     #############
###############################################################################

start_full_simulation = datetime.now()
timing_prev = datetime.now()
# Repeat for each time step
while True:
    # Check stopping conditions, specific for forward and backward tracking
    if forward_tracking and current_date > end_date:
        break
    if not forward_tracking and current_date < end_date:
        break

    # Handle 'noleap' calendar (skip Feb 29)
    if m.calendar == 'noleap' and current_date.month == 2 and current_date.day == 29:
        current_date += dt if forward_tracking else -dt
        continue
    
    # Load forcing data interval of loading is different for ERA5 (hourly) and ESM (daily in this case) data
    if model != 'ERA5' and current_date.hour == 0:  #check if the model is an ESM and if this is the case and it is also exactly 00:00 hours reload the data
        m.read_from_file(current_date, scenario, forcing_directory, model)
    if model == 'ERA5' and current_date.minute == 0:  #if it is about ERA5 and the current date time is exactly at the hour refresh the data
        m.read_from_file(current_date, scenario, forcing_directory, model)

    # Release moisture parcels
    if (forward_tracking and current_date <= release_end_date) or (not forward_tracking and current_date >= release_end_date):
        new_parcels = create_parcels_mask(mask, current_date, dt, m, parcels_mm, forward_tracking, parcel_data_columns, surface_area_weighting)
        num_new_parcels = new_parcels.shape[0]
        if num_new_parcels > 0: #only add parcels if there are any
            parcels_data = np.concatenate((parcels_data, new_parcels))
            total_released += np.sum(new_parcels[:, ORIGINAL_MOISTURE_IDX])     # Sum original moisture


    # Progress moisture parcels
    if parcels_data.size > 0: #only update parcels if there are any
        output_array = update_parcels(parcels_data, dt.total_seconds(), m.ua, m.va, m.wap, m.pr, m.evspsbl, m.prw, m.hus, m.levels, m.lats, m.lons, m.degreelength_lat, m.degreelength_lon, forward_tracking, output_array)
    
    # Compute which parcels should be labeled as killed or expired
    present_fraction = parcels_data[:, MOISTURE_IDX] / parcels_data[:, ORIGINAL_MOISTURE_IDX]  # Column 3 = moisture present, Column 4 = original moisture
    kill_mask = present_fraction < kill_threshold  # Parcels that should be killed
    expire_mask = (parcels_data[:, TIME_IDX] - parcels_data[:, START_TIME_IDX]) > tracking_time.total_seconds()  # Tracking time exceeded
    
    # Combine masks
    combined_mask = np.logical_or(kill_mask, expire_mask)
    
    ##POTENTIALLY ADD CODE TO SAVE ALL PARCELS THAT HAVE BEEN ALIVE IN THE SIMULATION 
    
    # Filter parcels_data based on combined mask
    parcels_data = parcels_data[np.logical_not(combined_mask)]
    
    killed = np.sum(kill_mask)
    expired = np.sum(expire_mask)
    parcels_destroyed = killed + expired
    parcels_in = parcels_data.shape[0]


    # Update visualization (IF TURNED OFF IT CAN REDUCE SIMULATION TIMES SIGNIFCANTLY)
    if dynamic_plotting:
        parcel_latidxs = parcels_data[:, CURRENT_LATIDX].astype(int)
        parcel_lonidxs = parcels_data[:, CURRENT_LONIDX].astype(int)
        
        img.set_data(output_array)
        img.set_clim(vmin=0, vmax=np.max(output_array))
        parcel_scatter.set_offsets(np.c_[parcel_lonidxs, parcel_latidxs])
        ax.set_title(f"Output Map at {current_date.strftime('%Y-%m-%d %H:%M')}\nParcels in system: {len(parcels_data)}; Allocated: {round((np.sum(output_array) / total_released)*100, 2) if total_released else 0}%")
        
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)
    
    # Print simulation status
    # if current_date.hour == 0:    #only print for each day
    print(current_date, parcels_in, parcels_destroyed, (datetime.now() - timing_prev).total_seconds(), 
              round(np.sum(output_array) / total_released, 2) if total_released else 0)

    timing_prev = datetime.now()

    # Progress time forward or backward
    current_date += dt if forward_tracking else -dt
    
end_full_simulation = datetime.now()

if dynamic_plotting:
    plt.ioff()
    plt.show() #show plot at the end

# Save output file
# create_nc_file(output_fn, output_directory, forward_tracking, model, scenario, start_date, release_end_date, m, output_array, total_released)

print(f'Finished in {int((end_full_simulation-start_full_simulation).total_seconds()/3600)} hours, {int(((end_full_simulation-start_full_simulation).total_seconds()%3600)/60)} minutes and {round(((end_full_simulation-start_full_simulation).total_seconds()%3600)%60,2)} seconds')