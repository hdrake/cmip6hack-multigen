# Instructions to scrape ERA5 data
Last run took about ~5 minutes to download 4.7 Gb of reanalysis data

## Install cdsapi via anaconda
conda install -c conda-forge cdsapi

## Follow instructions at https://cds.climate.copernicus.eu/api-how-to to install the CDS API key at ~/.cdsapirc and then run:
python get_ERA5_data_2d.py

