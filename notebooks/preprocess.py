import intake
import os, sys
import util
import qc
import pandas as pd
from tqdm.autonotebook import tqdm  # Fancy progress bars for our loops!
import numpy as np
import xarray as xr
import xesmf as xe
import pandas as pd

all_mip_ids = ['far', 'sar', 'tar', 'cmip3', 'cmip5', 'cmip6']

def get_ipcc_collection(mip_ids = all_mip_ids):
    col_dict = {}
    for col_name in mip_ids:
        if 'ar' in col_name:
            json_name = f"https://storage.googleapis.com/ipcc-{col_name}/pangeo-{col_name}.json"
        else:
            json_name = f"https://storage.googleapis.com/cmip6/pangeo-{col_name}.json"
        col = intake.open_esm_datastore(json_name)
        col_dict[col_name] = col
    return col_dict

def get_ipcc_dataset(mip_ids = all_mip_ids):
    col_dict = get_ipcc_collection(mip_ids)
    return

def load_ensembles(varnames, mip_ids=all_mip_ids, timeslice=None):
    col_dict = get_ipcc_collection(mip_ids=mip_ids)
    ds_dict = load_col_as_dict(col_dict, varnames, timeslice=timeslice)
    ens_dict = model_dict_to_ensemble_dict(ds_dict, varnames)
    return ens_dict


def model_dict_to_ensemble_dict(ds_dict, varnames):
    ens_dict = {}
    for mip_id, mip_ds in ds_dict.items():
        mipdataset = xr.Dataset()
        for varname, var_ds in mip_ds.items():
            vardataarray = xr.concat(
                [ds for name, ds in var_ds.items()], dim='ensemble'
            )
            mipdataset[varname] = vardataarray

        ens_dict[mip_id] = mipdataset

    return ens_dict

def load_col_as_dict(col_dict, varnames, timeslice=None, coarsen_size=2):
    ds_dict = {}
    for mip_id, col in tqdm(col_dict.items()):
        ds_dict[mip_id] = {}
        for varname in varnames:
            print("Loaded: variable_id `", varname, "` from activity_id `",mip_id,"`")
            cat = col.search(
                experiment_id='historical',
                variable_id=varname,
                table_id='Amon'
            )

            if cat.df.size == 0: continue

            with util.HiddenPrints():
                dset_dict = cat.to_dataset_dict(
                    aggregate=False,
                    zarr_kwargs={'consolidated': True, 'decode_times': False}
                )

            ds_dict[mip_id][varname] = {}
            for key, ds in dset_dict.items():
                # rename spatial dimensions if necessary
                if ('longitude' in ds.dims) and ('latitude' in ds.dims):
                    ds = ds.rename({'longitude':'lon', 'latitude': 'lat'})

                # Need this temporarily because setting 'decode_times': True is broken
                ds = xr.decode_cf(ds)
                ds['time'] = ds['time'].astype('<M8[ns]')
                ds['time'].values = np.array(
                    pd.to_datetime(
                        util.vec_dt_replace(pd.Series(ds['time'].values), day=1.)
                    )
                )

                repeats = len(ds['time']) - len(np.unique(ds['time']))
                if repeats != 0:
                    print(f"Skip {key} before datetime conflict.")
                    continue

                ds = ds.squeeze() # get rid of member_id (for now)

                chunks = {'lat':ds['lat'].size, 'lon':ds['lon'].size, 'time':30}
                ds = ds.chunk(chunks)

                if timeslice is not None:
                    try:
                        ds = ds.sel(time=timeslice)
                    except:
                        print(f"Skip {key} due to timesclicing error.")

                with util.HiddenPrints():
                    try:
                        ds_new = util.regrid_to_common(ds[varname])
                    except:
                        print(f"Skip {key} due to regridding conflict.")
                        continue

                ds_new.attrs.update(ds.attrs)
                ds_new = qc.quality_control(ds_new, varname, key, mip_id)

                ds_new.attrs['name'] = "-".join(key.split(".")[1:3])

                for coord in ds_new.coords:
                    if coord not in ['lat', 'lon', 'time']:
                        ds_new = ds_new.drop(coord)

                member_id = key.split(".")[4]
                ds_new = ds_new.expand_dims(
                    {'ensemble': np.array([ds_new.attrs['name'] + "-" + member_id])}, 0
                )

                ds_new = ds_new.assign_coords({
                    'member_id': member_id,
                    'source_id': key.split(".")[2],
                    'mip_id': key.split(".")[0]
                })

                ds_new.attrs['mip_id'] = mip_id

                coarsen_dict = {'lat': coarsen_size, 'lon': coarsen_size}
                ds_new = ds_new.coarsen(coarsen_dict, boundary='exact').mean()

                ds_dict[mip_id][varname][key] = ds_new  # add this to the dictionary
    return ds_dict

def load_era(path, timeslice=None, coarsen_size=2):
    era_native = xr.open_dataset(path, chunks={'time': -1})

    era_native = era_native.sel(time=timeslice)
    era_native = era_native.rename({'msl': 'psl', 't2m': 'tas', 'tp':'pr', 'latitude':'lat', 'longitude':'lon'})

    # convert from "m of water per day" to "kg m^-2 s^-1"
    # See https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation for details
    era_native['pr'] *= 1000./(24.*60.*60.)

    with util.HiddenPrints():
        era = util.regrid_to_common(era_native)
    era.attrs.update(era_native.attrs)

    era = era.coarsen({'lat':coarsen_size, 'lon': coarsen_size}, boundary='exact').mean()

    return era

