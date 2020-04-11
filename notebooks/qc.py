# For converting units for precip output
cm_to_m = 1.e-2
rho_water = 1.e3
day_in_s = (24.*60.*60.)

# reverse lat list
reverse_list = [
    'SAR.MPIfM.MPIfM-01.historical.Amon.gn',
    'SAR.HCCPR-HCCPR-01.historical.Amon.gn',
    'TAR.MPIfM.MPIfM.historical.Amon.gn',
    'CMIP3.IPSL.ipsl_cm4.historical.Amon.gn',
    'CMIP3.MOHC.ukmo_hadcm3.historical.Amon.gn',
    'CMIP5.MOHC.HadCM3.historical.Amon.gn'
]

def quality_control(ds, varname, key, mip_id):
    # Add metadata and apply various corrections
    if mip_id == 'cmip6':
        # Correct MCM-UA precipitation due to broken units (Ron Stouffer, personal communication)
        if ('MCM-UA-1-0' in key.split(".")[2]) and (varname == 'pr'):
            # convert from cm/day to kg/m^2/s
            ds *= (cm_to_m * rho_water / day_in_s)

        if ('THU-CIESM' in ds.attrs['parent_source_id']) and (varname == 'pr'):
            ds *= np.nan

    if key in reverse_list:
        # TEMPORARY FIX: Correct models which inexplicably have latitude flipped
        ds['lat'].values = ds['lat'].values[::-1]
        
    return ds