"""This is a general purpose module containing routines
(a) that are used in multiple notebooks; or 
(b) that are complicated and would thus otherwise clutter notebook design.
"""

import re
import socket
import numpy as np
import xarray as xr
import xesmf as xe
import pandas as pd
import os, sys
from tqdm.autonotebook import tqdm  # Fancy progress bars for our loops!

def is_ncar_host():
    """Determine if host is an NCAR machine."""
    hostname = socket.getfqdn()
    
    return any([re.compile(ncar_host).search(hostname) 
                for ncar_host in ['cheyenne', 'casper', 'hobart']])




# Define the common target grid axes
dlon, dlat = 1., 1.
ds_out = xr.Dataset({'lat': (['lat'], np.arange(-90.+dlat/2., 90., dlat)),
                     'lon': (['lon'], np.arange(0.+dlon/2., 360., dlon)),})

# Regridding function
def regrid_to_common(ds, ds_out=ds_out):
    """
    Regrid from rectilinear grid to common grid
    """
    regridder = xe.Regridder(ds, ds_out, 'bilinear', periodic=True, reuse_weights=True)
    return regridder(ds)

def calc_area(lat, lon, coarsen_size=1., dlat=1., dlon=1.):
    Rearth = 6.378E6   # radius of Earth in meters
    if coarsen_size != 1.:
        return (
        (np.deg2rad(dlat)*Rearth) * (np.deg2rad(dlon)*Rearth*np.cos(np.deg2rad(lat))) * xr.ones_like(lon)
        ).coarsen({'lat': coarsen_size, 'lon': coarsen_size}, boundary='exact').mean()
    else:
        return (np.deg2rad(dlat)*Rearth) * (np.deg2rad(dlon)*Rearth*np.cos(np.deg2rad(lat))) * xr.ones_like(lon)

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        
def vec_dt_replace(series, year=None, month=None, day=None):
    return pd.to_datetime(
        {'year': series.dt.year if year is None else year,
         'month': series.dt.month if month is None else month,
         'day': series.dt.day if day is None else day})

def add_ens_mean(ens_dict):
    for mip_id, ens in ens_dict.items():
        ensmean = ens.mean(dim='ensemble', skipna=True)
        ensmean = ensmean.assign_coords({
            'source_id': 'All',
            'member_id': 'All'
        })
        ens = xr.concat([ensmean.expand_dims({'ensemble': np.array(['ens-mean'])}), ens], dim='ensemble')
        ens.attrs['name'] = mip_id
        ens_dict[mip_id] = ens
    return ens_dict

def dict_func(d, func, on_self=False, *args, **kwargs):
    new_d = {}
    for key, item in tqdm(d.items()):
        if on_self:
            new_d[key] = func(self=d[key], *args, **kwargs)
        else:
            new_d[key] = func(d[key], *args, **kwargs)
        
    return new_d


def _compute_slope(x, y):
    """
    Private function to compute slopes at each grid cell using
    polyfit. 
    """
    idx = np.logical_and(~np.isnan(x), ~np.isnan(y))
    if np.sum(idx) > len(idx)/2.0:
        xtmp = x[idx]
        ytmp = y[idx]
        slope = np.polyfit(xtmp, ytmp, 1)[0] # return only the slope
    else:
        slope = np.nan
    return slope


def compute_slope(da, dim='time'):
    """
    Computes linear slope (m) at each grid cell.
    
    Args:
      da: xarray DataArray to compute slopes for
      
    Returns:
      xarray DataArray with slopes computed at each grid cell.
    """
    # convert to days.
    if dim=='time':
        x = ((da['time']-np.datetime64('1990'))/np.timedelta64(1,'D'))
    elif dim=='year':
        x = da['year']
    
    slopes = xr.apply_ufunc(_compute_slope,
                            x,
                            da,
                            vectorize=True,
                            dask='parallelized', 
                            input_core_dims=[[dim], [dim]],
                            output_dtypes=[float],
                            )
    return slopes


def pseudo_enso(tas, index="3.4"):
    """
    Computes pseudo ENSO variability index from anomalous near-surface air temperature
    
    Args:
    
    Returns:
    
    Method:
    From UCAR website (https://climatedataguide.ucar.edu/
    climate-data/nino-sst-indices-nino-12-3-34-4-oni-and-tni)

    Nino 3.4 Index computation:
    (a) Compute area averaged total SST from Niño 3.4 region (5N-5S, 190E-240E);
    (b) Compute monthly climatology (e.g., 1950-1979) for area averaged total SST from Niño X region,
    and subtract climatology from area averaged total SST time series to obtain anomalies; 
    (c) Smooth the anomalies with a 5-month running mean; (d) Normalize the smoothed values by its
    standard deviation over the climatological period.
    """ 
    
    if index == "3.4":
        tas_34 = tas.sel(lat=slice(-5.,5.), lon=slice(190,240)).mean(dim=['lat', 'lon'])
        tas_34_smooth = tas_34.rolling(time=5, center=True).mean()
        pseudo_enso_index = tas_34_smooth / tas_34_smooth.sel(time=slice('1961','1990')).std(dim='time')
        
    return pseudo_enso_index

def compute_derived_variables(ds):
    """Computes DataArrays for monthly climatology, monthly anomalies (with respect to climatology), and annual-mean anomalies for a variable.
    
    NOTE: Make sure that your DataArray is appropriately chunked (~100MB chunks)
    Args:
      da: xarray DataArray for which to compute derived variables.
      
    Returns:
      anom (monthly anomalies), clim (monthly climatology), ann (annual anomalies)
      
      These will be returned as dask objects for the user to compute.
    """
    
    gb = ds.groupby("time.month")
    
    clim = gb.mean(dim='time', skipna=True).compute()
    anom = (gb - clim).compute()
    ann = anom.groupby('time.year').mean('time', skipna=True).compute()
    
    return clim, anom, ann