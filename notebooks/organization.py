import intake

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