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