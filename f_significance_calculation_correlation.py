import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd
import xarray as xr
import os
import iris
import iris.analysis.cartography
import iris.plot as iplt
import iris.coord_categorisation
import iris.analysis.stats

from f_statistics import *
from f_preprocessing import *

from matplotlib.colors import LinearSegmentedColormap
colors =['lightgrey','lightgrey','lightgrey']
cmap_grey = LinearSegmentedColormap.from_list('cm',colors, N=3)

# function to test significance of correlation
"""
d_mod1: model output 1
d_mod2: model output 2
d_refd: reference data
reps: how many sample datasets
output: xarray with quantile values
"""

# ALL MONTHS INTERANNUAL EVAPORATION CORRELATION   
def calculate_correlation(fol,ref_data,tp,start_year, end_year,var,name1,name2,dt):
    ref_file = ref_selection(fol, ref_data, tp, start_year=start_year, end_year=end_year)

    f_mod1 = f'{fol}/data/htessel_output/{name1}'
    f_mod2 = f'{fol}/data/htessel_output/{name2}'

    ref = xr.open_dataset(f'{ref_file[0]}')

    mod1 = xr.open_dataset(f'{f_mod1}/o_efl_{start_year}_{end_year}_ia_anomalies.nc')
    mod2 = xr.open_dataset(f'{f_mod2}/o_efl_{start_year}_{end_year}_ia_anomalies.nc')

    # preprocess - select variable and change units
    d_ref = pp_ref(ref,ref_data)[0]
    d_mod1 = pp_mod(mod1)    
    d_mod2 = pp_mod(mod2)  

    # detrend
    if (dt=='dt_yes'):
        d_refd = detrend_permonth(da=d_ref, dim='time', deg=1)
        d_mod1d = detrend_permonth(da=d_mod1, dim='time', deg=1)
        d_mod2d = detrend_permonth(da=d_mod2, dim='time', deg=1)
    else:
        d_refd=d_ref
        d_mod1d=d_mod1
        d_mod2d=d_mod2
        
    r1=LE_corr(d_refd, d_mod1d)
    r2=LE_corr(d_refd, d_mod2d)
    r = r2-r1
    return r,r1,r2


# SEASONS INTERNANNUAL CORRELATION EVAPORATION
def calculate_correlation_seasons(fol,ref_data,tp,start_year, end_year,var,name1,name2,season,dt):
    ref_file = ref_selection(fol, ref_data, tp, start_year=start_year, end_year=end_year)

    f_mod1 = f'{fol}/data/htessel_output/{name1}'
    f_mod2 = f'{fol}/data/htessel_output/{name2}'

    ref = xr.open_dataset(f'{ref_file[0]}')

    mod1 = xr.open_dataset(f'{f_mod1}/o_efl_{start_year}_{end_year}_{season}_anomalies.nc')
    mod2 = xr.open_dataset(f'{f_mod2}/o_efl_{start_year}_{end_year}_{season}_anomalies.nc')

    # preprocess - select variable and change units
    d_ref = pp_ref(ref,ref_data)[0]
    d_mod1 = pp_mod(mod1)    
    d_mod2 = pp_mod(mod2)  

    # detrend
    if (dt=='dt_yes'):
        d_refd = detrend_allmonths(da=d_ref, dim='time', deg=1)
        d_mod1d = detrend_allmonths(da=d_mod1, dim='time', deg=1)
        d_mod2d = detrend_allmonths(da=d_mod2, dim='time', deg=1)
    else:
        d_refd=d_ref
        d_mod1d=d_mod1
        d_mod2d=d_mod2

    r1=LE_corr(d_refd, d_mod1d)
    r2=LE_corr(d_refd, d_mod2d)
    r = r2-r1
    
    return r,r1,r2

    
def calculate_correlation_SM_esa(fol,tp,start_year, end_year,name1,name2,layer,dt,th):
    smfol=f'{fol}/data/ref_data'
    if (layer=='layer1'):
        somo=xr.open_dataset(f'{smfol}/ESACCI-SOILMOISTURE-1999-2019-ia_anomalies_{th}mask.nc')
        somo=somo.sm

    # model
    f_mod1 = f'{fol}/data/htessel_output/{name1}'
    mod1 = xr.open_dataset(f'{f_mod1}/o_soil_{start_year}_{end_year}_ia_anomalies.nc')
    f_mod2 = f'{fol}/data/htessel_output/{name2}'
    mod2 = xr.open_dataset(f'{f_mod2}/o_soil_{start_year}_{end_year}_ia_anomalies.nc')    
    if (layer=='layer1'):
        sm1 = mod1.SWVL1 #* 0.07 * 1e3 # 7cm
        sm2 = mod2.SWVL1 #* 0.07 * 1e3 # 7cm

    # detrend
    if (dt=='dt_yes'):
        sm1_dt = detrend_permonth(sm1, dim='time', deg=1)
        sm2_dt = detrend_permonth(sm2, dim='time', deg=1)
        somo_dt = detrend_permonth(somo, dim='time', deg=1)
        d_mod1=sm1_dt
        d_mod2=sm2_dt
        d_refd=somo_dt
    else:
        sm1_dt=sm1
        sm2_dt=sm2
        somo_dt=somo
        
    r1 = xr.corr(sm1_dt, somo_dt, dim=['time'])
    r2 = xr.corr(sm2_dt, somo_dt, dim=['time'])
    r = r2-r1
    return r,r1,r2



    
def calculate_correlation_SM_esa_seasons(fol,tp,start_year, end_year,name1,name2,layer,season,dt,th):
    smfol=f'{fol}/data/ref_data'
    if (layer=='layer1'):
        somo=xr.open_dataset(f'{smfol}/ESACCI-SOILMOISTURE-1999-2019_{season}_anomalies_{th}mask.nc')
        somo=somo.sm

    # detrend
    somo_dt = detrend_allmonths(somo, dim='time', deg=1)

    # model
    f_mod1 = f'{fol}/data/htessel_output/{name1}'
    mod1 = xr.open_dataset(f'{f_mod1}/o_soil_{start_year}_{end_year}_{season}_anomalies.nc')
    f_mod2 = f'{fol}/data/htessel_output/{name2}'
    mod2 = xr.open_dataset(f'{f_mod2}/o_soil_{start_year}_{end_year}_{season}_anomalies.nc')    
    if (layer=='layer1'):
        sm1 = mod1.SWVL1 #* 0.07 * 1e3 # 7cm
        sm2 = mod2.SWVL1 #* 0.07 * 1e3 # 7cm
    elif (layer=='layer2'):
        sm1 = mod1.SWVL2 #* 0.21 * 1e3 # 21cm
        sm2 = mod2.SWVL2 #* 0.21 * 1e3 # 21cm   
    elif (layer=='layer3'):
        sm1 = mod1.SWVL3 #* 0.72 * 1e3 # 72 cm
        sm2 = mod2.SWVL3 #* 0.72 * 1e3 # 72 cm

    # detrend
    if (dt=='dt_yes'):
        sm1_dt = detrend_allmonths(sm1, dim='time', deg=1)
        sm2_dt = detrend_allmonths(sm2, dim='time', deg=1)
        somo_dt = detrend_allmonths(somo, dim='time', deg=1)
        d_mod1=sm1_dt
        d_mod2=sm2_dt
        d_refd=somo_dt
    else:
        sm1_dt=sm1
        sm2_dt=sm2
        somo_dt=somo

    r1 = xr.corr(sm1_dt, somo_dt, dim=['time'])
    r2 = xr.corr(sm2_dt, somo_dt, dim=['time'])
    r = r2-r1
    return r,r1,r2
