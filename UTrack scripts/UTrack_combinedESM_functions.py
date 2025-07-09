from netCDF4 import Dataset, date2index
import random
import numpy as np
import datetime as dt
import os
import xarray as xr
from numba import njit, prange
import glob

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

def find_forcing_files_drive_letter():
    #below shows the possible driveletters starting from D as A, B and C are usually not used for the external drive
    driveletters = ['D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    
    for driveletter in driveletters:
        drive_path = f"{driveletter}:\\"
        if not os.path.exists(drive_path):
            continue  # Skip non-existent drives
        
        try:
            for folder in os.listdir(drive_path):
                full_path = os.path.join(drive_path, folder)
                if os.path.isdir(full_path) and folder.startswith("Forcing_data_"):
                    return driveletter  # Return the first matching drive letter
        except Exception as e:
            print(f"Error accessing {drive_path}: {e}")  # Log error for debugging

    return None  # Explicitly return None if no drive is found
                    


def estimate_runtime_left(runtimes, window_size, start_date, current_date, release_end_date, end_date, tracking_time, delta_t):
    time_estimate = 0
    # Compute moving average of the last `window_size` slopes
    slopes = [(runtimes[i] - runtimes[i - 1]) for i in range(-window_size, 0)]
    avg_slope = sum(slopes) / window_size  # Moving average of the slope
    
    # Calculate the remaining time until each relevant date
    time_till_parcels_killed = (start_date + tracking_time) - current_date
    time_till_release_end_date = release_end_date - current_date
    time_till_end_date = end_date - current_date
    
    # Convert time to steps based on delta_t
    steps_till_parcels_killed = time_till_parcels_killed.total_seconds() // delta_t.total_seconds()
    steps_till_release_end_date = time_till_release_end_date.total_seconds() // delta_t.total_seconds()
    steps_till_end_date = time_till_end_date.total_seconds() // delta_t.total_seconds()

    # Growth phase (before parcels start getting killed)
    if steps_till_parcels_killed > -window_size:
        for step in range(int(steps_till_parcels_killed) + 1):
            time_estimate += max(0, runtimes[-1] + step * avg_slope)

        # Maximum runtime phase
        max_runtime_estimate = max(0, steps_till_parcels_killed * max(avg_slope, 0))  # Prevent negative estimates
        total_runtime_constant_phase = (steps_till_release_end_date - steps_till_parcels_killed) * max_runtime_estimate

        time_estimate += total_runtime_constant_phase

        # Lag phase (decreasing slope when parcels are no longer generated)
        time_estimate += max_runtime_estimate * tracking_time.days * 0.5  # Assuming simple linear decrease
        
        # Convert seconds to hours, minutes, and seconds
        hours = time_estimate // 3600
        minutes = (time_estimate % 3600) // 60
        seconds = time_estimate % 60
        time_estimate = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
        
        return time_estimate

    # Second case: after parcels are killed but before the release end date
    elif steps_till_parcels_killed <= -window_size and steps_till_release_end_date > -window_size:
        for step in range(int(steps_till_release_end_date) + 1):
            time_estimate += max(0, runtimes[-1] + step * avg_slope)

        max_runtime_estimate = runtimes[-1] + steps_till_release_end_date * avg_slope

        # Lag phase (decreasing slope when parcels are no longer generated)
        time_estimate += max_runtime_estimate * tracking_time.days * 0.5  # Assuming simple linear decrease
        
        # Convert seconds to hours, minutes, and seconds
        hours = time_estimate // 3600
        minutes = (time_estimate % 3600) // 60
        seconds = time_estimate % 60
        time_estimate = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
        
        return time_estimate

    # Third case: after the release end date
    elif steps_till_release_end_date <= -window_size:
        for step in range(int(steps_till_end_date) + 1):
            time_estimate += max(0, runtimes[-1] + step * avg_slope)  # Ensure no negative runtime

        # Convert seconds to hours, minutes, and seconds
        hours = time_estimate // 3600
        minutes = (time_estimate % 3600) // 60
        seconds = time_estimate % 60
        time_estimate = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
        
        return time_estimate

            
def create_nc_file(outputfn, output_directory, forward_tracking, model_choice, scenario, start_date, release_end_date, mask, m, 
                   output_array, total_released, parcels_per_mm, tracking_time, baseline_period, target_period, 
                   cluster_id=None, drying_or_wetting_sim=None):
    if forward_tracking:
        track_key_word = 'Forward'
        footprint_type = 'evaporation'
    else:
        track_key_word = 'Backward'
        footprint_type = 'precipitation'
    
    # check if the output folder exists and make it if it doesnt
    os.makedirs(output_directory, exist_ok=True)
    
    with Dataset(outputfn, mode="w", format="NETCDF4_CLASSIC") as output:
        output.title = f"{footprint_type} footprints"
        output.model = model_choice
        output.scenario = scenario
        output.simulation_period = f"{start_date.strftime('%Y-%m-%d')} to {release_end_date.strftime('%Y-%m-%d')}"
        output.creation_date = dt.datetime.now().isoformat()
        output.parcels_per_mm = parcels_per_mm
        output.tracking_time = str(tracking_time)
        output.tracking_direction = track_key_word
    
        # Store cluster-related metadata only if clusters are used
        if cluster_id:  
            output.cluster_id = cluster_id
            output.drying_or_wetting = drying_or_wetting_sim
            
            # Store baseline and target precipitation data in global attributes
            output.baseline_period = f"{baseline_period[0]}-{baseline_period[1]}"
            output.target_period = f"{target_period[0]}-{target_period[1]}"
            
            
        # Create dimensions
        output.createDimension("lat", len(m.lats))
        output.createDimension("lon", len(m.lons))
        output.createDimension("frac", 1)
        
        # Create variables
        lat = output.createVariable("lat", np.float32, ("lat",))
        lat.units = "degrees_north"
        lat.long_name = "latitude"
        lat[:] = m.lats

        lon = output.createVariable("lon", np.float32, ("lon",))
        lon.units = "degrees_east"
        lon.long_name = "longitude"
        lon[:] = m.lons

        footprint = output.createVariable("footprint", np.float64, ("lat", "lon"), zlib=True)
        footprint.units = "mm"
        footprint.long_name = f"{footprint_type} footprint"

        fraction_allocated = output.createVariable("fraction_allocated", np.float32, ("frac",))
        fraction_allocated.units = "fraction"
        fraction_allocated.long_name = "Fraction of total released moisture allocated to precipitation"
        
        mask = output.createVariable("simulation mask", np.float64, ("lat", "lon"), zlib=True)
        mask.long_name = "Mask used as starting region for running the moisture tracking model"
        mask.units = "No Unit"
        
    with Dataset(outputfn, "r+") as output:
        output["footprint"][:, :] = output_array
        output["fraction_allocated"][:] = np.sum(output_array) / total_released if total_released > 0 else 0
        
    return outputfn

#determine if all years are available for a specific scenario and model
def check_file_availability(directory, start_year, end_year, model, scenario):     
    try:
        variables = ['pr', 'prw', 'tas', 'hfls', 'wap', 'ua', 'va', 'hus', 'sftlf']    #possibly later add the sftlf variable but watch out as this causes errors because it has not timestamp
    
        # Create a set of all required years
        required_years = set(range(start_year, end_year + 1))
        
        # Dictionary to track available years for each variable
        available_years = {var: set() for var in variables}
        
        # Flag to track if sftlf file is found
        sftlf_found = False
        
        # Iterate through the files in the directory
        for filename in os.listdir(directory):
            # Only process files that match the desired model and scenario format
            if model in filename and scenario in filename and filename.endswith('.nc'):
                try:
                    if 'sftlf' in filename:   #sftlf variable is only available for all year and has not date definition so here we add all years to the set to prevent popping up as missing files
                        available_years['sftlf'] = required_years
                        sftlf_found = True
                    else:
                        # Extract the year range from the filename
                        date_range = filename.split('_')[-1].replace('.nc', '')
                        file_start_year = int(date_range[:4])
                        file_end_year = int(date_range[-8:-4])
                    
                    # Iterate over the variables and check if they are in the current file
                    for variable in variables:
                        if variable in filename:
                            # Add all years from the file's range to the available years for the specific variable
                            available_years[variable].update(range(file_start_year, file_end_year + 1))
                except ValueError:
                    continue
        
        # Handle the case where sftlf file is not found
        if not sftlf_found:
            raise Exception(f"Error: No suitable sftlf file found for model {model} and scenario {scenario}.")
        
        # Now check for missing years for each variable
        missing_years = {}
        for variable in variables:
            missing_years[variable] = required_years - available_years[variable]
        
        # Output missing years for each variable
        for variable, missing in missing_years.items():
            if missing:
                raise FileNotFoundError(f"Missing files for {variable}: {sorted(missing)}")
                # Return the missing years for each variable
                return missing_years
    
    except FileNotFoundError:
        raise FileNotFoundError('Forcing data directory not found, check if the forcing directory is defined properly (right drive letter?) or if the external data storage is connected properly (or at all)')
            

def find_land_mask(forcing_directory, model, scenario):
    #select the right land mask for the model
    land_mask = None
    for file in os.listdir(forcing_directory):
        if model in file and scenario in file and 'sftlf' in file:
            try:
                land_mask = xr.open_dataset(os.path.join(forcing_directory, file))['sftlf'].values
                return land_mask / 100  #land mask is given in percentage and should be a mask from 0-1 so divide by 100
                break  #stop loop as the right mask was found
            except:
                print('Failed to open land_mask file')
    if not land_mask:
        raise('Failed to find suitable land_mask file for model and scenario')

# Get index based on vertical position
@njit
def get_level_index(level):
    if level > 92500: return 0
    if level > 77500: return 1
    if level > 60000: return 2
    if level > 37500: return 3
    if level > 17500: return 4
    if level > 7500: return 5
    if level > 3000: return 6
    return 7

@njit
def get_pos_index(lat, lon, lats, lons):
    """
    Get the indices of the lats and lons arrays that are closest to the given lat and lon.
    """
    #make sure the lat and lons are within bounds of the indexes
 
    if lat > 90: 
        lat = 90
        lon += 180
    if lat < -90: 
        lat = -90
        lon += 180
    if lon >= 360 or lon < 0:
        lon = lon % 360    
        
    lat_idx = (np.abs(lats - lat)).argmin()
    lon_idx = (np.abs(lons - lon)).argmin()

    return lat_idx, lon_idx


#vectorized version of the create parcel mask function, is a bit faster especially at more parcels to create
def create_parcels_mask(mask, parcel_start_time, delta_t, m, parcels_per_mm, forward_tracking, parcels_data_columns):
    """Parcel creation with NumPy's random.choice and unravel_index outside Numba."""
    if forward_tracking:
        m.evspsbl = np.where(m.evspsbl < 0, 0, m.evspsbl)
        total_evap_prec = np.sum(mask * m.evspsbl) * delta_t.total_seconds()
        average_evap_prec = total_evap_prec / np.sum(mask > 0) if np.sum(mask > 0) > 0 else 0
        norm_mask = (m.evspsbl * mask) / total_evap_prec if total_evap_prec > 0 else np.zeros_like(mask)
    else:
        m.pr = np.where(m.pr < 0, 0, m.pr)
        total_evap_prec = np.sum(mask * m.pr) * delta_t.total_seconds()
        average_evap_prec = total_evap_prec / np.sum(mask > 0) if np.sum(mask > 0) > 0 else 0
        norm_mask = (m.pr * mask) / total_evap_prec if total_evap_prec > 0 else np.zeros_like(mask)

    norm_mask = norm_mask.astype(np.float64)

    if np.sum(norm_mask) == 0:
        return np.empty((0, len(parcels_data_columns)))  # No parcels to create

    norm_mask /= np.sum(norm_mask)

    x = np.arange(len(m.lats) * len(m.lons)).reshape((len(m.lats), len(m.lons)))
    num_parcels = int(average_evap_prec * parcels_per_mm)

    if num_parcels == 0:
        return np.empty((0, len(parcels_data_columns)))  # No parcels to create

    xy = np.random.choice(np.ravel(x), num_parcels, p=np.ravel(norm_mask))
    rows, cols = np.unravel_index(xy, x.shape)
    locs = np.column_stack((rows, cols))

    lat_step_size = np.abs(m.lats[1] - m.lats[0])
    lon_step_size = np.abs(m.lons[1] - m.lons[0])

    lats = m.lats[0] + lat_step_size * locs[:, 0]
    lons = m.lons[0] + lon_step_size * locs[:, 1]

    parcel_moisture = total_evap_prec / num_parcels

    n_parcels = len(lats)
    parcels_data = np.zeros((n_parcels, len(parcels_data_columns)))

    # Vectorized part:
    random_lat_offsets = np.random.random(n_parcels) * lat_step_size - lat_step_size / 2
    random_lon_offsets = np.random.random(n_parcels) * lon_step_size - lon_step_size / 2

    final_lats = lats + random_lat_offsets
    final_lons = lons + random_lon_offsets

    lat_indices = np.array([get_pos_index(lat, lon, m.lats, m.lons)[0] for lat, lon in zip(final_lats, final_lons)], dtype=int)
    lon_indices = np.array([get_pos_index(lat, lon, m.lats, m.lons)[1] for lat, lon in zip(final_lats, final_lons)], dtype=int)

    levels = np.array([get_starting_level(lat_index, lon_index, m.hus, m.levels) for lat_index, lon_index in zip(lat_indices, lon_indices)])

    parcels_data[:, LAT_IDX] = final_lats
    parcels_data[:, LON_IDX] = final_lons
    parcels_data[:, LEVEL_IDX] = levels
    parcels_data[:, MOISTURE_IDX] = parcel_moisture
    parcels_data[:, ORIGINAL_MOISTURE_IDX] = parcel_moisture
    parcels_data[:, TIME_IDX] = parcel_start_time.timestamp()
    parcels_data[:, START_TIME_IDX] = parcel_start_time.timestamp()
    parcels_data[:, START_LATIDX] = lat_indices
    parcels_data[:, START_LONIDX] = lon_indices
    parcels_data[:, CURRENT_LATIDX] = lat_indices
    parcels_data[:, CURRENT_LONIDX] = lon_indices

    return parcels_data


@njit
def convert_hfls_to_evap(hfls, tas):
    # Constants
    L_ref = 2502.2  # Latent heat of vaporization at 0°C in kJ/kg
    L_slope = -2.4337  # Change in L per °C in kJ/kg
    
    tas_C = tas-273.15         #convert to Celsius for the formula

    # Calculate latent heat of vaporization (L) based on T
    L = (L_slope * tas_C + L_ref) * 1000 #convert L to J/kg

    # Calculate evaporation from latent heat flux
    evap_from_hfls = hfls / L  # Result in kg m-2 s-1 just like evspsbl
    return evap_from_hfls
    

# Parcel class
# Fast humidity-based level selection
@njit
def get_starting_level(latidx, lonidx, hus, levels):
    H_sum = np.sum(hus[:, latidx, lonidx])  # Sum total humidity
    if H_sum == 0:
        return 950  # Default level if no humidity

    frac = np.random.random() * H_sum  # Random fraction of total humidity
    count = 0

    for i in range(len(levels)):
        if hus[i, latidx, lonidx] > 0:
            count += hus[i, latidx, lonidx]
            if count > frac:
                return levels[i]
    return 950  # Fallback

@njit(parallel=True)
def update_parcels(parcels_data, delta_t_sec, m_ua, m_va, m_wap, m_pr, m_evspsbl, m_prw, m_hus, m_levels, m_lats, m_lons, m_degreelength_lat, m_degreelength_lon, forward_tracking, output_array):
    n_parcels = parcels_data.shape[0]

    for i in prange(n_parcels):
        lat = parcels_data[i, 0]
        lon = parcels_data[i, 1]
        level = parcels_data[i, 2]
        moisture_present = parcels_data[i, 3]
        time = parcels_data[i, 5]

        levidx = get_level_index(level)
        latidx, lonidx = get_pos_index(lat, lon, m_lats, m_lons)

        w = m_wap[levidx, latidx, lonidx]
        while np.abs(w) > 1000:
            level -= 50
            levidx = get_level_index(level)
            w = m_wap[levidx, latidx, lonidx]

        u = m_ua[levidx, latidx, lonidx]
        v = m_va[levidx, latidx, lonidx]
        
        if not forward_tracking:
            u, v, w = -u, -v, -w

        lon += (delta_t_sec * u / m_degreelength_lon[latidx])
        lat += (delta_t_sec * v / m_degreelength_lat[latidx])
        level += (delta_t_sec * w)

        # Ensure values stay within valid range of levels
        if level < 50: level = 50
        if level > 1000: level = 1000
        
        if np.random.random() * 24 < (delta_t_sec / 3600):
            latidx, lonidx = get_pos_index(lat, lon, m_lats, m_lons)
            level = get_starting_level(latidx, lonidx, m_hus, m_levels)

        P = m_pr[latidx, lonidx] * delta_t_sec
        E = m_evspsbl[latidx, lonidx] * delta_t_sec
        PW = m_prw[latidx, lonidx]
        
        if forward_tracking:
            fraction_allocated = P / PW if PW > 0 and P > 0 else 0  # double check if there is any P and PW otherwise no allocation because without precipitation no allocation is possible and without precipitable water no precipitation is possible either
        else:
            fraction_allocated = E / PW if PW > 0 and E > 0 else 0

        allocated = fraction_allocated * moisture_present
        moisture_present -= allocated
        
        outlatidx, outlonidx = get_pos_index(lat, lon, m_lats, m_lons)
        output_array[outlatidx, outlonidx] += allocated

        parcels_data[i, 0] = lat
        parcels_data[i, 1] = lon
        parcels_data[i, 2] = level
        parcels_data[i, 3] = moisture_present
        parcels_data[i, 5] = time + delta_t_sec
        parcels_data[i, 8] = latidx
        parcels_data[i, 9] = lonidx

    return output_array


# Forcing data class
class Meteo:
    def __init__(self):
        self.time = None
        self.levels = None
        self.lats = None
        self.lons = None
        self.ua = None
        self.va = None
        self.evspsbl = None
        self.pr = None
        self.hus = None
        self.prw = None
        self.wap = None
        self.calendar = None
        self.forcing_data = {}  # Store open datasets
        self.current_yearstr = None #store the current year string.

    def read_from_file(self, year, month, day, scenario, forcing_directory, model_choice):
        """Reads forcing data for the given date, efficiently managing file openings."""
        # Determine the yearstr for the model automatically by scanning through the filenames
        for filename in os.listdir(forcing_directory):
            parts = filename.split("_")
            date_range = parts[-1].replace(".nc", "")
            try:
                start_year = int(date_range[:4])
                end_year = int(date_range[-8:-4])
            except ValueError:
                continue  # Skip the file if it's a different file that doesn't have the same format
            if start_year <= year <= end_year:
                yearstr = date_range
                break

        if self.current_yearstr != yearstr: #Only open new files if the year string changed.
            self.close_datasets() #close the old datasets before opening new ones.
            self.current_yearstr = yearstr;

            variables = ['pr', 'prw', 'tas', 'hfls', 'wap', 'ua', 'va', 'hus']

            for variable in variables:
                frequency = 'Eday' if variable in ('prw', 'evspsbl') else 'day'
                pattern = os.path.join(forcing_directory, f'{variable}_{frequency}_{model_choice}_{scenario}_r1i1p1f1*{yearstr}.nc')
                files = glob.glob(pattern)

                if files:
                    forcing_file = files[0]
                    if variable not in self.forcing_data:
                        self.forcing_data[variable] = {}
                    self.forcing_data[variable]['fn'] = forcing_file
                    self.forcing_data[variable]['ds'] = Dataset(forcing_file)

            self.time = self.forcing_data['ua']['ds'].variables['time']
            self.calendar = self.time.calendar

        time_idx = date2index(dt.datetime(year, month, day), self.time, calendar=self.calendar, select='nearest')

        # convert latent heatflux (hfls) to evaporation (evspsbl)
        hfls_ds = self.forcing_data['hfls']['ds']
        tas_ds = self.forcing_data['tas']['ds']
        hfls = hfls_ds['hfls'][time_idx].data
        tas = tas_ds['tas'][time_idx].data
        self.evspsbl = convert_hfls_to_evap(hfls, tas)

        lats = self.forcing_data['ua']['ds'].variables['lat'][:].data
        lons = self.forcing_data['ua']['ds'].variables['lon'][:].data

        self.levels = self.forcing_data['ua']['ds'].variables['plev'][:].data   #unit is Pa
        self.ua = self.forcing_data['ua']['ds'].variables['ua'][time_idx].data  #unit is m/s
        self.va = self.forcing_data['va']['ds'].variables['va'][time_idx].data  #unit is m/s
        self.pr = self.forcing_data['pr']['ds'].variables['pr'][time_idx].data  #unit is [kg m-2 s-1] = mm/s
        self.hus = self.forcing_data['hus']['ds'].variables['hus'][time_idx].data #unitless (0-1)
        self.prw = self.forcing_data['prw']['ds'].variables['prw'][time_idx].data #unit is [kg m-2] = mm
        self.wap = self.forcing_data['wap']['ds'].variables['wap'][time_idx].data #unit is Pa/s

    def close_datasets(self):
        """Closes all open datasets."""
        for variable, data in self.forcing_data.items():
            if 'ds' in data and data['ds'] is not None:
                try:
                    data['ds'].close()
                except Exception as e:
                    print(f"Error closing dataset for {variable}: {e}")
        self.forcing_data = {} #reset the forcing data dictionary.
        
    
    # Pre-compute degree lengths based on latitude and store them in the model data only once for each model (m)
    def compute_degree_lengths(self):
        degreelength_lat = np.empty_like(self.lats)
        degreelength_lon = np.empty_like(self.lats)
    
        for i, lat in enumerate(self.lats):
            curlatrad = lat * 2 * np.pi / 360
            degreelength_lat[i] = (
                111132.92 
                + (-559.82 * np.cos(2 * curlatrad)) 
                + 1.175 * np.cos(4 * curlatrad) 
                - 0.0023 * np.cos(6 * curlatrad)
            )
            degreelength_lon[i] = (
                (111412.84 * np.cos(curlatrad)) 
                - 93.5 * np.cos(3 * curlatrad) 
                + 0.118 * np.cos(5 * curlatrad)
            )
    
        # Store these precomputed values in the model for later access
        self.degreelength_lat = degreelength_lat
        self.degreelength_lon = degreelength_lon
    
    def compute_grid_cell_areas(self):
        # Earth's radius in m
        R = 6371000  #not exactly true (earth is not a perfect sphere) but a pretty good estimation for estimating the surface area of gridcells
        
        if self.lats == None or self.lons == None:
            raise('Lats and/or lons are not defined so surface area calculation is impossible')
        
        # Calculate latitude and longitude step sizes in radians (assuming regular grid)
        lat_bounds = np.radians(np.linspace(self.lats[0] - (self.lats[1]-self.lats[0])/2,
                                            self.lats[-1] + (self.lats[1]-self.lats[0])/2,
                                            len(self.lats)+1))
        
        lon_bounds = np.radians(np.linspace(self.lons[0] - (self.lons[1]-self.lons[0])/2,
                                            self.lons[-1] + (self.lons[1]-self.lons[0])/2,
                                            len(self.lons)+1))
        
        # Initialize area array
        area_grid = np.zeros((len(self.lats), len(self.lons)))
        
        # Compute area for each grid cell
        for i in range(len(self.lats)):
            phi1 = lat_bounds[i]
            phi2 = lat_bounds[i + 1]
            sin_diff = np.abs(np.sin(phi2) - np.sin(phi1))
            for j in range(len(self.lons)):
                lambda_diff = np.abs(lon_bounds[j + 1] - lon_bounds[j])
                area = R**2 * lambda_diff * sin_diff
                area_grid[i, j] = area
                
        self.surface_areas = area_grid