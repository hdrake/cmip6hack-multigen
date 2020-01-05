# CMIP6 Hackathon
# Project: Multi-Generational Climate Model Inter-Comparison

See below for project description (modified from <a href="https://discourse.pangeo.io/t/how-has-the-performance-of-climate-models-changed-over-30-years-of-model-development/113">discourse</a>).

## Setting up on the cloud

Go to <a href="https://ocean.pangeo.io">ocean.pangeo.io</a> and login with Globus using your Orcid ID (full instructions <a href="https://discourse.pangeo.io/t/using-ocean-pangeo-io-for-the-cmip6-hackathon/291">here</a>).

Next, clone the repository.

1. Open a JupyterLab session on the system you plan to use.
2. Open a terminal in the JupyterLab environment.
3. Clone the project: `git clone https://github.com/hdrake/cmip6hack-multigen.git`
4. Navigate into the project folder with: `cd cmip6hack-multigen`
5. Activate the `cmip6hack-multigen` environment by running `source spinup_env.sh`.
6. Analyze the data interactively in the iPython jupyter notebooks in `/notebooks/` (do not forget to activate the `cmip6hack-multigen` kernel when you open a notebook!

## Project proposal

### Scientific Motivation

While the first ocean-atmosphere coupled general circulation models date back to 1969, transient simulations attempting to reproduce the historical record and project future climate changes were not available until the late 1980s. These first-generation GCMs included only atmospheric and oceanic components, were run at nominal resolutions of 3º-10º, and omitted many important sub-grid scale processes whose parameterizations are now common-place. Since then, climate model development has continued, with pushes towards higher resolution, the inclusion of other components of the Earth System, and a flurry of new and/or improved parameterizations. While several studies have documented the improvements in model skill reaped by these model developments (e.g. <a href="https://journals.ametsoc.org/doi/abs/10.1175/BAMS-89-3-303">Reichler et al. 2008</a>, <a href="https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/grl.50256">Knutti et al. 2013</a>, and <a href="https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2007JD008972">Glecker et al. 2008</a>), there has been no comprehensive study of climate model skill that spans all the way from the first-generation models of the late 1980s to the state-of-the-art CMIP6 ensemble.

Below: change in climate model performance across CMIP1, CMIP2, and CMIP3 from <a href="https://journals.ametsoc.org/doi/abs/10.1175/BAMS-89-3-303">Reichler et al. (2008)</a>.

![change in climate model performance across CMIP1, CMIP2, and CMIP3 from <a href="https://journals.ametsoc.org/doi/abs/10.1175/BAMS-89-3-303">Reichler et al. (2008)</a>](https://github.com/hdrake/cmip6hack-multigen/blob/master/references/Reichler2008_Figure_1.png)

### Overall Project Goal

Compute variable-specific and general model performance metrics (e.g. normalized area-weighted root-mean square; pattern correlations; area-weighted absolute bias; etc) across several model generations, including CMIP6.

### Proposed Hacking Methods

We will use `xskillscore` to calculate global skill metrics such as the root-mean-square error (`xskillscore.rmse`) and mean absolute error (`xskillscore.mae`) and compare across different variables, different models (within a model generation), and different model generations.

Time permitting, we will also compute regional skill metrics (and perhaps interesting cross-correlations) using `regionalmask` to delineate regions.

### Data Needs

Monthly-mean values of a number of common variables for CMIP6 historical (e.g. 1800–2014) simulations, ~1000 years of preindustrial control simulations, and 1% per year CO2 runs.

Variables of interest (based on model skill metric in Reichler 2008):

    sea level pressure (psl)
    air temperature (ta)
    2-m air temperature (tas)
    zonal and meridional wind (ua, va, uas, vas)
    precipitation (pr)
    specific and/or relative humidity (hus, hur)
    snow fraction (?)
    sea ice fraction (sic or siconc)

We currently are pulling CMIP6 from the <a href="https://pangeo-data.github.io/pangeo-datastore/">google cloud storage</a>, pre-CMIP model output from a restricted google cloud storage bucket (to be made publically available upon preprint publication), and ERA5 reanalysis data (as our 'observational' reference) from the <a href="https://cds.climate.copernicus.eu/#!/home">Copernicus Climate Data Store</a>.

### Software Tools

We will build off of (and hopefully contribute to) existing packages such as <a href="https://github.com/JiaweiZhuang/xESMF">`xESMF`</a> and <a href="https://github.com/raybellwaves/xskillscore">`xskillscore`</a> that leverage <a href="https://github.com/pydata/xarray">`xarray`</a> and already feature tools for handling model ensembles and computing model performance metrics. We will also make use of packages containing useful metadata, such as the <a href="https://github.com/mathause/regionmask">`regionalmask`</a> library of regional masks.

### How to make our work citable (once there is enough for a public release)

[Zenodo](https://about.zenodo.org/) is a data archiving tool that can help make your project citable by assigning a DOI to the project's GitHub repository.

Follow the guidelines <a href="https://guides.github.com/activities/citable-code">here</a>.
