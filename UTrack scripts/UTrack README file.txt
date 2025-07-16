General note about scripts in this folder:


UTrack Moisture Tracking Model
==============================

Description:
------------
This script simulates atmospheric moisture tracking using the UTrack model.
It allows both forward (evaporation → precipitation) and backward 
(precipitation → source) tracking of moisture parcels using either 
ERA5 reanalysis data or Earth System Model (ESM) outputs.

Major Features:
---------------
- Support for multiple climate models (ERA5 and CMIP6 ESMs).
- Forward and backward moisture parcel tracking.
- Dynamic visualization of parcel movement and allocation (optional).
- Optional weighting by grid cell surface area.
- Capability to track a single coordinate or a custom spatial mask.
- NetCDF export of results.

Supported Models and Scenarios:
-------------------------------
Models:     EC-Earth3, MPI-ESM1-2-HR, NorESM2-MM, TaiESM1, ERA5  
Scenarios:  ssp126, ssp245, ssp370, ssp585 (not used for ERA5)

Important Usage Notes:
----------------------
1. All input files must share the same latitude/longitude format.
   Ensure consistent land-mask and grid alignment.

2. ERA5 Variable Names and File Naming:
   ------------------------------------
   The ERA5 data should follow the Tuinenburg & Staal (2020) file format.
   Naming conventions and variable names are:

   Filenames (per day):
     ERA5_2d_YYYY-MM-DD.nc       → Contains: "e", "tp", "tcw"
     ERA5_q_YYYY-MM-DD.nc        → Contains: "q" (specific humidity, 3D)
     ERA5_uv_YYYY-MM-DD.nc       → Contains: "u", "v" (3D wind components)
     ERA5_w_YYYY-MM-DD.nc        → Contains: "w" (vertical velocity)

   Variable Names:
     evaporation                 = "e"        (kg m-2 s-1)
     precipitation               = "tp"       (kg m-2 s-1)
     total column water          = "tcw"      (kg m-2)
     specific humidity           = "q"        (kg kg-1)
     zonal wind                  = "u"        (m s-1)
     meridional wind             = "v"        (m s-1)
     vertical velocity           = "w"        (Pa s-1)

3. ESM (Earth System Models) Variable Names:
   -----------------------------------------
   ESMs use standard CMIP6 naming conventions. Expected variables:

     precipitation               = "pr"       (kg m-2 s-1)
     total column water          = "prw"      (kg m-2)
     surface air temperature     = "tas"      (K)
     surface latent heat flux    = "hfls"     (W m-2)
     zonal wind                  = "ua"       (m s-1)
     meridional wind             = "va"       (m s-1)
     vertical velocity           = "wap"      (Pa s-1)
     specific humidity           = "hus"      (kg kg-1)
     land surface fraction       = "sftlf"    (%)

   Notes:
   - "hfls" and "tas" are combined to derive evaporation internally.
   - Files must span full years and follow standard CMIP filename conventions.

4. Coordinate Standards:
   ----------------------
   - Longitudes must be standardized to [0, 360) range.
   - Latitudes must be sorted in ascending order (-90 to 90).
   - Grids from different sources must match in shape and resolution.

5. Input data must be complete for the simulation period.
   Use the `check_file_availability` function to validate availability.

Configuration Parameters (in the main script):
----------------------------------------------
model                = 'EC-Earth3'        # model name (case sensitive)
scenario             = 'ssp245'           # climate scenario (ignored for ERA5)
start_date           = '1-10-2015'        # start of release period (DD-MM-YYYY)
release_end_date     = '15-10-2015'       # end of release period (DD-MM-YYYY)
parcels_mm           = 100                # parcels per mm of moisture
delta_t              = 0.25               # time step in hours
forward_tracking     = True               # True = forward; False = backward
tracking_days        = 15                 # parcel lifetime in days
kill_threshold       = 0.01               # min moisture before parcel is removed
dynamic_plotting     = True               # enable/disable live plotting
surface_area_weighting = False            # enable surface area weighting
singlepoint          = True               # simulate from a single point
single_point_coords  = (-20.891993, -50.911292)  # (lat, lon)

Custom Mask Mode (alternative to single point):
-----------------------------------------------
Instead of a single point, define a 2D spatial mask:
Example:
mask = np.load("your_mask_file.npy")  # Must be aligned with model grid

Output:
-------
Output is saved to a NetCDF file in the format:
  ./<model>-output/Utrack-<Forward|Backward>-<model>_<scenario>_<dates>.nc

The output includes:
- footprint (2D mm): spatial distribution of allocated moisture
- fraction_allocated: fraction of total released moisture that was allocated

Execution Notes:
----------------
- Real-time plotting can slow down simulations significantly. Disable if not needed.
- If a simulation with the same configuration was already completed,
  the script will stop and not overwrite the output file.

Error Handling:
---------------
- The script checks that all necessary forcing data is available before starting.
- It raises errors for:
   - Missing input files
   - Invalid time configurations
   - Previously completed simulations (duplicate output)

Citation:
---------
If using the ERA5 setup, please cite:
Tuinenburg, O. A., & Staal, A. (2020).
"Tracking the global flows of atmospheric moisture and associated uncertainties."
Nature Communications, 11(1), 1–10.

Final Remarks:
--------------
This code is optimized for flexible and large-scale moisture tracking experiments.
Ensure all forcing data is properly structured and accessible (e.g., external drives connected).

If needed, adjust function paths, plotting options, or data directories to match your environment.
