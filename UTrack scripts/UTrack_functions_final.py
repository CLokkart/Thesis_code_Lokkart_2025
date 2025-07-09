from netCDF4 import Dataset, date2index
import random
import numpy as np
from datetime import datetime, timedelta
import os
import xarray as xr
from numba import njit, prange
import glob

parcel_data_columns =  ['latitude', 'longitude', 'level', 'moisture_present', 'original_moisture', 'time', 'start_time', 'startlatidx', 
                       'startlonidx', 'current_latidx', 'current_lonidx']

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
    driveletters = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
    
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

            
def create_nc_file(outputfn, output_directory, forward_tracking, model_choice, scenario, start_date, release_end_date, m, output_array, total_released):
    if forward_tracking:
        track_key_word = 'Forward'
        footprint_type = 'evaporation'
    else:
        track_key_word = 'Backward'
        footprint_type = 'precipitation'
    
    # check if the output folder exists and make it if it doesnt
    os.makedirs(output_directory, exist_ok=True)
    
    with Dataset(outputfn, mode="w", format="NETCDF4_CLASSIC") as output:
        # Add global attributes
        output.title = f"{footprint_type} footprints"
        output.model = model_choice
        output.scenario = scenario
        output.simulation_period = f"{start_date.strftime('%Y-%m-%d')} to {release_end_date.strftime('%Y-%m-%d')}"
        output.creation_date = datetime.now().isoformat()
        
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
        
    with Dataset(outputfn, "r+") as output:
        output_array = output_array / (np.sum(output_array) / total_released)
        output["footprint"][:, :] = output_array
        output["fraction_allocated"][:] = np.sum(output_array) / total_released
        
    return outputfn

#determine if all years are available for a specific scenario and model
def check_file_availability(directory, start_date, end_date, model, scenario):     
    if model == "ERA5":
        variables = ['2d', 'q', 'uv', 'w']
        
        # Create a set of all required days
        required_dates = {(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range((end_date - start_date).days + 1)}
        
        # Dictionary to track available years for each variable
        available_dates = {var: set() for var in variables}
        
        # Flag to track if sftlf file is found
        lsm_found = False
        
        # Iterate through the files in the directory
        for filename in os.listdir(directory):
        
            if 'land_mask_ERA5' in filename:                                 #check whether the land mask is avalable in the directory
                    lsm_found = True
            
            # Only process files that match the desired file type
            if filename.endswith('.nc'):
                try:
                    # Extract the year range from the filename
                    date_string = filename.split('_')[-1].replace('.nc', '')
                    
                    # Iterate over the variables and check if they are in the current file
                    for variable in variables:
                        if variable in filename:
                            # Add all dates from the file's range to the available years for the specific variable
                            available_dates[variable].update([date_string])
                except ValueError:
                    continue
        
        # Handle the case where sftlf file is not found
        if not lsm_found:
            raise Exception(f"Error: No suitable lsm file found for {model}.")
        
        # Now check for missing dates for each variable
        missing_files_details = {}
        missing_dates = {}
        for variable in variables:
            missing_dates[variable] = sorted(list(required_dates - available_dates[variable]))
            if missing_dates[variable]:
                missing_files_details[variable] = missing_dates[variable]

        # Raise the error only after checking all variables
        if missing_files_details:
            error_message = "Not all files available for the requested simulation:\n"
            for var, missing in missing_files_details.items():
                error_message += f"Missing files for {var}: {missing}\n"
            raise FileNotFoundError(error_message.strip())
    else:
        try:
            variables = ['pr', 'prw', 'tas', 'hfls', 'wap', 'ua', 'va', 'hus', 'sftlf']    #possibly later add the sftlf variable but watch out as this causes errors because it has not timestamp
        
            # Create a set of all required years
            required_years = set(range(start_date.year, end_date.year + 1))
            
            # Dictionary to track available years for each variable
            available_years = {var: set() for var in variables}
            
            # Flag to track if sftlf file is found
            sftlf_found = False
            
            # Iterate through the files in the directory
            for filename in os.listdir(directory):
                # Only process files that match the desired model, scenario and format
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
            
            # Now check for missing dates for each variable
            missing_files_details = {}
            missing_dates = {}
            for variable in variables:
                missing_dates[variable] = sorted(list(required_years - available_years[variable]))
                if missing_dates[variable]:
                    missing_files_details[variable] = missing_dates[variable]
    
            # Raise the error only after checking all variables
            if missing_files_details:
                error_message = "Not all files available for the requested simulation:\n"
                for var, missing in missing_files_details.items():
                    error_message += f"Missing files for {var}: {missing}\n"
                raise FileNotFoundError(error_message.strip())
        
        except FileNotFoundError:
            raise FileNotFoundError('Forcing data directory not found, check if the forcing directory is defined properly (right drive letter?) or if the external data storage is connected properly (or at all)')
            

def find_land_mask(forcing_directory, model, scenario):
    """
    Loads the land mask data and standardizes its latitude and longitude
    dimensions to match the common [-90, 90] ascending and [0, 360) ascending convention.
    """
    land_mask_data = None
    original_lats = None
    original_lons = None

    for file in os.listdir(forcing_directory):
        if model == 'ERA5':
            if 'land_mask_ERA5.nc' in file:
                try:
                    ds = xr.open_dataset(os.path.join(forcing_directory, file))
                    land_mask_data = ds['lsm'].values[0, :, :]
                    original_lats = ds['latitude'].values
                    original_lons = ds['longitude'].values
                    ds.close()
                    break
                except Exception as e:
                    print(f'Failed to open ERA5 land mask: {e}')
        else:
            if model in file and scenario in file and 'sftlf' in file:
                try:
                    ds = xr.open_dataset(os.path.join(forcing_directory, file))
                    land_mask_data = ds['sftlf'].values / 100  # Normalize from % to [0,1]
                    original_lats = ds['lat'].values
                    original_lons = ds['lon'].values
                    ds.close()
                    break
                except Exception as e:
                    print(f'Failed to open ESM land mask: {e}')

    if land_mask_data is None:
        raise FileNotFoundError(f"No suitable land mask found for model '{model}' and scenario '{scenario}'.")

    # Standardize longitudes to [0, 360)
    original_lons = np.mod(original_lons, 360)

    # Determine slicing index for latitudes
    if original_lats[0] > original_lats[1]:
        lat_slice = slice(None, None, -1)
    else:
        lat_slice = slice(None)

    # Determine slicing index for longitudes
    if original_lons[0] > original_lons[1]:
        lon_slice = slice(None, None, -1)
    else:
        lon_slice = slice(None)

    # Apply slicing to standardize orientation
    land_mask_data = land_mask_data[lat_slice, :][:, lon_slice]

    return land_mask_data


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
def get_nearest_index(lat, lon, lats, lons):
    """
    Get the indices of the lats and lons arrays that are closest to the given lat and lon.
    Works reliably for any input longitude range.
    """
    # 1. Normalize latitude for pole crossings
    if lat > 90:
        lat = 90 - (lat - 90)
        lon += 180
    elif lat < -90:
        lat = -90 - (lat + 90)
        lon += 180

    # 2. Wrap input longitude to [0, 360]
    lon = lon % 360

    # 4. Find closest latitude
    lat_idx = np.abs(lats - lat).argmin()

    # 5. Find closest longitude (allowing for wrap-around)
    lon_diff = np.abs(lons - lon)
    lon_diff = np.minimum(lon_diff, 360 - lon_diff)
    lon_idx = lon_diff.argmin()

    return lat_idx, lon_idx

#vectorized version of the create parcel mask function, is a bit faster especially at more parcels to create
def create_parcels_mask(mask, parcel_start_time, delta_t, m, parcels_per_mm, forward_tracking, parcels_data_columns, surface_area_weighting):
    """Parcel creation with NumPy's random.choice using grid cell areas."""
    if forward_tracking:
        moisture_rate = np.where(m.evspsbl < 0, 0, m.evspsbl) # mm/s (or kg m-2 s-1)
    else:
        moisture_rate = np.where(m.pr < 0, 0, m.pr) # mm/s (or kg m-2 s-1)
    
    # Calculate total moisture released/precipitated over the *masked area* for this timestep.
    # This `total_evap_prec` variable (sum of (rate * time) over masked cells)
    # has units of `mm` * `number of masked cells`.
    total_evap_prec = np.sum(mask * moisture_rate) * delta_t.total_seconds()
    
    # Calculate the average moisture depth per active masked cell (in mm).
    average_evap_prec = total_evap_prec / np.sum(mask > 0) if np.sum(mask > 0) > 0 else 0

    # If no moisture, return empty array.
    if total_evap_prec == 0:
        return np.empty((0, len(parcels_data_columns)))  # No parcels to create

    # Calculate the total number of parcels to release.
    # This assumes `parcels_per_mm` is "parcels per mm depth per active cell".
    num_parcels = int(average_evap_prec * parcels_per_mm)

    # If no parcels to create after rounding, return empty array.
    if num_parcels == 0:
        return np.empty((0, len(parcels_data_columns)))

    # Calculate the moisture content for each individual parcel.
    # Units will be `(mm * number_of_masked_cells) / parcels`.
    parcel_moisture = total_evap_prec / num_parcels # Units: [mm * N_cells] / [N_parcels]


    if surface_area_weighting:
        # Weigh the probability of selecting a grid cell by its moisture rate and surface area.
        # `weighted_probability` will be proportional to the *volume* of moisture from that cell.
        weighted_probability = (moisture_rate * mask) * m.surface_areas
    else:
        # Assume all grid cells have effectively the same "area" for probability distribution
        # In this case, probability is just proportional to the moisture rate within the mask.
        weighted_probability = (moisture_rate * mask)

    # Normalize the weighted probability
    total_weighted_prob = np.sum(weighted_probability)
    if total_weighted_prob == 0:
        return np.empty((0, len(parcels_data_columns)))  # No valid cells to create parcels

    norm_mask = weighted_probability / total_weighted_prob
    norm_mask = norm_mask.astype(np.float64)

    # Choose grid cell indices based on weighted probability
    flat_indices = np.random.choice(np.arange(norm_mask.size), num_parcels, p=norm_mask.ravel())
    rows, cols = np.unravel_index(flat_indices, norm_mask.shape)

    # Retrieve the center latitudes and longitudes of the chosen grid cells
    lats_centers = m.lats[rows]
    lons_centers = m.lons[cols]

    # Calculate the actual degree width (step size) for each *chosen* cell
    # You need access to `m.lat_bounds` and `m.lon_bounds` (in radians)
    # Ensure these are saved as attributes in your Meteo class after compute_grid_cell_areas() runs.
    lat_bounds_deg = np.degrees(m.lat_bounds) # Convert to degrees for calculating offsets
    lon_bounds_deg = np.degrees(m.lon_bounds) # Convert to degrees for calculating offsets

    # `parcel_delta_lats` and `parcel_delta_lons` will now be arrays,
    # with each element corresponding to the specific cell's lat/lon width.
    parcel_delta_lats = np.abs(lat_bounds_deg[rows + 1] - lat_bounds_deg[rows])
    parcel_delta_lons = np.abs(lon_bounds_deg[cols + 1] - lon_bounds_deg[cols])

    # Now generate random offsets using these individual step sizes
    # This ensures parcels are randomly distributed within their actual cell boundaries.
    random_lat_offsets = (np.random.random(num_parcels) - 0.5) * parcel_delta_lats
    random_lon_offsets = (np.random.random(num_parcels) - 0.5) * parcel_delta_lons

    final_lats = lats_centers + random_lat_offsets
    final_lons = lons_centers + random_lon_offsets

    # Assuming get_pos_index and get_starting_level are defined elsewhere
    lat_indices = np.array([get_nearest_index(lat, lon, m.lats, m.lons)[0] for lat, lon in zip(final_lats, final_lons)], dtype=int)
    lon_indices = np.array([get_nearest_index(lat, lon, m.lats, m.lons)[1] for lat, lon in zip(final_lons, final_lons)], dtype=int) # Corrected variable

    parcels_data = np.zeros((num_parcels, len(parcels_data_columns)))

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
        lat = parcels_data[i, LAT_IDX]
        lon = parcels_data[i, LON_IDX]
        level = parcels_data[i, LEVEL_IDX]
        moisture_present = parcels_data[i, MOISTURE_IDX]
        time = parcels_data[i, TIME_IDX]

        levidx = get_level_index(level)
        latidx, lonidx = get_nearest_index(lat, lon, m_lats, m_lons)

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
        
        # --- Pole Crossing Physics ---
        # Handle pole crossings
        if lat > 90:
            lat = 90 - (lat - 90)
            lon += 180
        if lat < -90:
            lat = -90 - (lat + 90)
            lon += 180
    
        # Wrap input longitude to [-180, 180]
        lon = ((lon + 180) % 360) - 180
    
        # Adjust longitude based on the target array range
        if m_lons.min() >= 0 and lon < 0:
            lon = lon % 360
        
        level += (delta_t_sec * w)
        
        if level < 50: level = 50
        if level > 1000: level = 1000
        
        if np.random.random() * 24 < (delta_t_sec / 3600):
            latidx, lonidx = get_nearest_index(lat, lon, m_lats, m_lons)
            level = get_starting_level(latidx, lonidx, m_hus, m_levels)

        P = m_pr[latidx, lonidx] * delta_t_sec
        E = m_evspsbl[latidx, lonidx] * delta_t_sec
        PW = m_prw[latidx, lonidx]
        
        if forward_tracking:
            fraction_allocated = P / PW if PW > 0 and P > 0 else 0
        else:
            fraction_allocated = E / PW if PW > 0 and E > 0 else 0

        allocated = fraction_allocated * moisture_present
        moisture_present -= allocated
        
        outlatidx, outlonidx = get_nearest_index(lat, lon, m_lats, m_lons)
        output_array[outlatidx, outlonidx] += allocated

        parcels_data[i, LAT_IDX] = lat
        parcels_data[i, LON_IDX] = lon
        parcels_data[i, LEVEL_IDX] = level
        parcels_data[i, MOISTURE_IDX] = moisture_present
        parcels_data[i, TIME_IDX] = time + delta_t_sec
        parcels_data[i, CURRENT_LATIDX] = latidx
        parcels_data[i, CURRENT_LONIDX] = lonidx

    return output_array

def standardize_lat_lon(lats, lons, data_vars):    #    CAN PROBABLY BE DELETED AS THE STANDARDIZATION IS MADE MORE SIMPLE AND WITHIN THE READ_FROM_FILE FUNCTION
    """
    Converts latitudes to [-90, 90] (ascending) and longitudes to [0, 360] (ascending),
    and reorders all associated data variables accordingly.

    Args:
        lats (numpy array): Original latitude values.
        lons (numpy array): Original longitude values.
        data_vars (dict): Dictionary of data variables that need latitude and longitude sorting.

    Returns:
        tuple: (corrected lats, corrected lons, updated data_vars dictionary)
    """
    lats_working = lats.copy()
    lons_working = lons.copy()
    data_vars_processed = data_vars.copy()

    # --- Standardize Latitudes ---
    # First, handle potential 0-180 latitude range (e.g., 0=South Pole, 180=North Pole)
    if np.min(lats_working) >= -1e-6 and np.max(lats_working) <= 180 + 1e-6:
        if (np.abs(np.max(lats_working) - np.min(lats_working)) > 170):
            lats_working -= 90

    # Now, ensure latitudes are in ascending order (-90 to 90 or shifted equivalent)
    if lats_working[0] > lats_working[-1]:
        lats_working = np.flip(lats_working)
        for var_name, data in data_vars_processed.items():
            data_vars_processed[var_name] = np.flip(data, axis=-2)

    # --- Standardize Longitudes ---
    lons_working[lons_working < 0] += 360
    sorted_indices_lon = np.argsort(lons_working)
    lons_sorted = lons_working[sorted_indices_lon]

    for var_name, data in data_vars_processed.items():
        data_vars_processed[var_name] = data[..., sorted_indices_lon]

    return lats_working, lons_sorted, data_vars_processed

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
        self.current_file_datestr = None #store the current datestring.
        self.current_timeidx = None
        self.lat_sort_idx = None
        self.lon_sort_idx = None


    def read_from_file(self, date, scenario, forcing_directory, model_choice):
        year = date.year
        month = date.month
        day = date.day
        
        if model_choice == 'ERA5':  
            date_str = f"{year:04}-{month:02}-{day:02}"
            
            if self.current_file_datestr != date_str:  # Only open new files if the date has changed to prevent unnecessary opening and closing of NC files (more efficient)
                self.close_datasets()  # Close old datasets before opening new ones
                self.current_file_datestr = date_str
    
                variables = ['e', 'tcw', 'tp', 'q', 'u', 'v', 'w']
                for variable in variables:
                    variable_fn = '2d' if variable in ['e', 'tcw', 'tp'] else ('uv' if variable in ['u', 'v'] else variable)
                    pattern = os.path.join(forcing_directory, f'{model_choice}_{variable_fn}_{date_str}.nc')
                    files = glob.glob(pattern)
    
                    if files:
                        forcing_file = files[0]  # Assume only one file matches
                        if variable not in self.forcing_data:
                            self.forcing_data[variable] = {}
                        self.forcing_data[variable]['fn'] = forcing_file
                        self.forcing_data[variable]['ds'] = Dataset(forcing_file)
    
            self.time = self.forcing_data['tp']['ds'].variables['time']
            self.calendar = self.time.calendar
            time_idx = date2index(date, self.time, calendar=self.calendar, select='nearest')
            
            # Load only the constant variables once at the first iteration of the simulation and standardize lat lon to lats [-90,90] and lons [0,360]
            if self.levels is None or self.lats is None or self.lons is None:
                self.levels = self.forcing_data['u']['ds'].variables['level'][:].data * 100
                raw_lats = self.forcing_data['u']['ds'].variables['latitude'][:].data
                raw_lons = np.mod(self.forcing_data['u']['ds'].variables['longitude'][:].data, 360)
                
                # ERA5 latitudes are typically descending (90 to -90), flip to ascending
                if raw_lats[1] < raw_lats[0]:
                    self.lat_sort_idx = slice(None, None, -1)  #slicer that reverses order
                else:
                    self.lat_sort_idx = slice(None)
                
                self.lats = raw_lats[self.lat_sort_idx]
                
                #check if the lon values are ascending or descending
                self.lon_sort_idx = np.argsort(raw_lons)  #slicer that reverses order
                self.lons = raw_lons[self.lon_sort_idx]
                    

            if self.current_timeidx != time_idx:
                self.current_timeidx = time_idx
                self.evspsbl = self.forcing_data['e']['ds'].variables['e'][time_idx].data[self.lat_sort_idx, :][:, self.lon_sort_idx] * -1000 / 3600
                self.pr = self.forcing_data['tp']['ds'].variables['tp'][time_idx].data[self.lat_sort_idx, :][:, self.lon_sort_idx] * 1000 / 3600
                self.prw = self.forcing_data['tcw']['ds'].variables['tcw'][time_idx].data[self.lat_sort_idx, :][:, self.lon_sort_idx]
                self.ua = self.forcing_data['u']['ds'].variables['u'][time_idx].data[:, self.lat_sort_idx, :][:, :, self.lon_sort_idx]
                self.va = self.forcing_data['v']['ds'].variables['v'][time_idx].data[:, self.lat_sort_idx, :][:, :, self.lon_sort_idx]
                self.hus = self.forcing_data['q']['ds'].variables['q'][time_idx].data[:, self.lat_sort_idx, :][:, :, self.lon_sort_idx]
                self.wap = self.forcing_data['w']['ds'].variables['w'][time_idx].data[:, self.lat_sort_idx, :][:, :, self.lon_sort_idx]
            
            
        else:   # Handle non-ERA5 data models
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
    
            if self.current_file_datestr != yearstr:  # Only open new files if the year string changed
                self.close_datasets()  # Close the old datasets before opening new ones
                self.current_file_datestr = yearstr
    
                variables = ['pr', 'prw', 'tas', 'hfls', 'wap', 'ua', 'va', 'hus']
    
                for variable in variables:
                    frequency = 'Eday' if variable == 'prw' else 'day'
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
            time_idx = date2index(date, self.time, calendar=self.calendar, select='nearest')
    
            if self.levels is None or self.lats is None or self.lons is None:
                self.levels = self.forcing_data['ua']['ds'].variables['plev'][:].data  # Pa
                raw_lats = self.forcing_data['ua']['ds'].variables['lat'][:].data
                raw_lons = np.mod(self.forcing_data['ua']['ds'].variables['lon'][:].data, 360)
    
                if raw_lats[1] < raw_lats[0]:
                    self.lat_sort_idx = slice(None, None, -1)  #determine slicer that reverses order
                else:
                    self.lat_sort_idx = slice(None) #determine slicer that doesnt change the order
                
                self.lats = raw_lats[self.lat_sort_idx]
                
                #check if the lon values are ascending or descending
                self.lon_sort_idx = np.argsort(raw_lons)  #slicer that reverses order
                self.lons = raw_lons[self.lon_sort_idx]
                
            if self.current_timeidx != time_idx:
                self.current_timeidx = time_idx
        
                hfls = self.forcing_data['hfls']['ds']['hfls'][time_idx].data
                tas = self.forcing_data['tas']['ds']['tas'][time_idx].data
                evspsbl_data = convert_hfls_to_evap(hfls, tas)
        
                self.evspsbl = evspsbl_data[self.lat_sort_idx, :][:, self.lon_sort_idx]
                self.pr = self.forcing_data['pr']['ds']['pr'][time_idx].data[self.lat_sort_idx, :][:, self.lon_sort_idx]
                self.prw = self.forcing_data['prw']['ds']['prw'][time_idx].data[self.lat_sort_idx, :][:, self.lon_sort_idx]
                self.ua = self.forcing_data['ua']['ds']['ua'][time_idx].data[:, self.lat_sort_idx, :][:, :, self.lon_sort_idx]
                self.va = self.forcing_data['va']['ds']['va'][time_idx].data[:, self.lat_sort_idx, :][:, :, self.lon_sort_idx]
                self.hus = self.forcing_data['hus']['ds']['hus'][time_idx].data[:, self.lat_sort_idx, :][:, :, self.lon_sort_idx]
                self.wap = self.forcing_data['wap']['ds']['wap'][time_idx].data[:, self.lat_sort_idx, :][:, :, self.lon_sort_idx]
    
    
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
        """
        Computes the surface area of each grid cell using spherical geometry,
        adapted for grids with non-uniform lat/lon spacing.
        """
        # Earth's radius in meters
        R = 6371000
    
        if self.lats is None or self.lons is None:
            raise ValueError('Lats and/or lons are not defined so surface area calculation is impossible')
    
        lats = np.array(self.lats)
        lons = np.array(self.lons)
    
        # Create latitude bounds
        lat_bounds = np.zeros(len(lats) + 1)
        lat_bounds[1:-1] = 0.5 * (lats[:-1] + lats[1:])
        lat_bounds[0] = lats[0] - 0.5 * (lats[1] - lats[0])
        lat_bounds[-1] = lats[-1] + 0.5 * (lats[-1] - lats[-2])
        lat_bounds = np.radians(lat_bounds)
    
        # Create longitude bounds
        lon_bounds = np.zeros(len(lons) + 1)
        lon_bounds[1:-1] = 0.5 * (lons[:-1] + lons[1:])
        lon_bounds[0] = lons[0] - 0.5 * (lons[1] - lons[0])
        lon_bounds[-1] = lons[-1] + 0.5 * (lons[-1] - lons[-2])
        lon_bounds = np.radians(lon_bounds)
    
        # Initialize area array
        area_grid = np.zeros((len(lats), len(lons)))
    
        # Compute area for each grid cell
        for i in range(len(lats)):
            phi1 = lat_bounds[i]
            phi2 = lat_bounds[i + 1]
            sin_diff = np.abs(np.sin(phi2) - np.sin(phi1))
            for j in range(len(lons)):
                lambda_diff = np.abs(lon_bounds[j + 1] - lon_bounds[j])
                area = R**2 * lambda_diff * sin_diff
                area_grid[i, j] = area
    
        self.surface_areas = area_grid
        self.lat_bounds = lat_bounds
        self.lon_bounds = lon_bounds