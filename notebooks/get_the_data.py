import numpy as np
import xarray as xr
import xesmf as xe
from tqdm.autonotebook import tqdm  # Fancy progress bars for our loops!
import intake
# util.py is in the local directory
# it contains code that is common across project notebooks
# or routines that are too extensive and might otherwise clutter
# the notebook design
import util 


# Regridding function
def regrid_to_common(ds, ds_out):
    """
    Regrid from rectilinear grid to common grid
    """
    regridder = xe.Regridder(ds, ds_out, 'bilinear',periodic=True, reuse_weights=True)
    return regridder(ds)

def get_the_data():
    '''Return a dict of xr.Dataset objects.
    Keys to the dict are MIP generations.
    Each object is a dataset of dimension lat/lon/ensemble
    Representing the data variables for every model in that MIP generation.'''
    
    col_dict = {}
    if util.is_ncar_host():
        col = intake.open_esm_datastore("../catalogs/glade-cmip6.json")
    else:
        col = intake.open_esm_datastore("../catalogs/pangeo-cmip6.json")
    col_dict["CMIP6"] = col

    col = intake.open_esm_datastore("../catalogs/adhoc-ipcc-ar.json")
    col_dict["pre-CMIP6"] = col

    mip_ids = ['FAR', 'SAR', 'TAR', 'CMIP6']
    mip_catalog_dict = {}
    for mip_id in mip_ids:
        if mip_id == 'CMIP6':
            mip_catalog_dict[mip_id] = "CMIP6"
        else:
            mip_catalog_dict[mip_id] = "pre-CMIP6"

    # Define the common target grid axes
    dlon, dlat = 1., 1.
    ds_out = xr.Dataset({'lat': (['lat'], np.arange(-90.+dlat/2., 90., dlat)),
                         'lon': (['lon'], np.arange(0.+dlon/2., 360., dlon)),})
    Rearth = 6.378E6   # radius of Earth in meters
    # a DataArray that gives grid cell areas on the lat/lon grid (in units of m^2)
    area = (np.deg2rad(dlat)*Rearth) * (np.deg2rad(dlon)*Rearth*np.cos(np.deg2rad(ds_out.lat))) * xr.ones_like(ds_out.lon)

    #varnames = ['tas','psl','pr','uas','vas']
    varnames = ['tas', 'pr']
    time_slice = slice('1981', '2010') # date range consistent with NCEP reanalysis long-term-mean

    # For converting units for precip out
    cm_to_m = 1.e-2
    rho_water = 1.e3
    day_in_s = (24.*60.*60.)

    ds_dict = {}

    for mip_id in tqdm(mip_ids):
        ds_dict[mip_id] = {}
        for varname in varnames:

            col = col_dict[mip_catalog_dict[mip_id]]
            cat = col.search(experiment_id='historical', 
                             table_id='Amon', 
                             variable_id=varname,
                             member_id='r1i1p1f1'  # choose first ensemble member only (for now)
                            )

            dset_dict = cat.to_dataset_dict(zarr_kwargs={'consolidated': True, 'decode_times': True})

            ds_dict[mip_id][varname] = {}
            for key, ds in dset_dict.items():
                if (mip_catalog_dict[mip_id] == 'pre-CMIP6') and (mip_id != key.split(".")[-1]): continue

                # rename spatial dimensions if necessary
                if ('longitude' in ds.dims) and ('latitude' in ds.dims):
                    ds = ds.rename({'longitude':'lon', 'latitude': 'lat'})
                ds = xr.decode_cf(ds) # Need this temporarily because setting 'decode_times': True appears broken
                ds = ds.squeeze() # get rid of member_id (for now)

                # take long-term mean
                timeave = ds.sel(time=time_slice).mean(dim='time')

                # modify pre-CMIP6 chunks
                if mip_catalog_dict[mip_id] == 'pre-CMIP6':
                    timeave = timeave.chunk({'lat':timeave['lat'].size, 'lon':timeave['lon'].size})

                # regrid to common grid
                ds_new = regrid_to_common(timeave[varname], ds_out)

                # Add metadata and apply various corrections
                if mip_catalog_dict[mip_id] == 'CMIP6':
                    # Correct MCM-UA precipitation due to broken units (Ron Stouffer, personal communication)
                    if ('MCM-UA' in ds.attrs['parent_source_id']) and (varname == 'pr'):
                        # convert from cm/day to kg/m^2/s
                        ds_new *= (cm_to_m * rho_water / day_in_s)

                    # TEMPORARY FIX: Correct BCC-ESM1 and CanESM5 which inexplicably have latitude flipped
                    if ("BCC-ESM1" in key) or ("CanESM5" in key):
                        ds_new['lat'].values = ds_new['lat'].values[::-1]

                    ds_new.attrs['name'] = ds.attrs['source_id']

                else:
                    # Maybe chance this at pre-processing stage?
                    ds_new.attrs['name'] = ds.attrs['institution']

                # drop redundant variables (like "height: 2m")
                for coord in ds_new.coords:
                    if coord not in ['lat','lon']:
                        ds_new = ds_new.drop(coord)

                # Add ensemble as new dimension
                ds_new = ds_new.expand_dims({'ensemble': np.array([ds_new.attrs['name']])}, 0)

                # Add var as new dimension
                #ds_new = ds_new.expand_dims({'var': np.array([varname])}, 0)

                # We should keep the metadata!!!
                ds_new.attrs['mip_id'] = mip_id

                ds_dict[mip_id][varname][key] = ds_new  # add this to the dictionary

    # Create a single dictionary whose keys are the MIP id
    #  Each item in the dict will be a single xr.Dataset combining all data from each MIP generation
    ens_dict = {}
    for mip_id in mip_ids:
        mipdataset = xr.Dataset()
        for varname in varnames:
            vardataarray = xr.concat([ds for name, ds in ds_dict[mip_id][varname].items()], dim='ensemble')
            mipdataset[varname] = vardataarray
        ens_dict[mip_id] = mipdataset

    return(ens_dict)
