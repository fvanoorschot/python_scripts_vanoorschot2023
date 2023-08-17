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
from f_figures import *
from f_run_evaluation import *
from f_vegstats import *

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
def sig_test_rmse(d_mod1, d_mod2,d_refd, reps):
    n = d_mod1.time.values.size #size of timeseries
    art_diff_col = [] #list of artificial r-difference

    # interate over reps
    for j in range(reps):
        rd = np.random.randint(2, size=n) # n random choices of 0 or 1
        da1 = [] # artificial data 1
        for k in range(n): # for each timestep select either mod1 or mod2 depending on random number 0 or 1
            if rd[k]==0:
                r = d_mod1[k] 
            elif rd[k]==1:
                r = d_mod2[k]
            da1.append(r) # append values to da1 list
        art_d1 = xr.concat(da1, dim='time').assign_coords(d_mod1.coords) # convert da1 list to xarray
        art_r1=LE_rmse(d_refd, art_d1) # calculate correlation of artificial dataset 1

        rd = np.random.randint(2, size=n)
        da2 = []
        for k in range(n):
            if rd[k]==0:
                r = d_mod1[k]
            elif rd[k]==1:
                r = d_mod2[k]
            da2.append(r)
        art_d2 = xr.concat(da2, dim='time').assign_coords(d_mod1.coords)
        art_r2=LE_rmse(d_refd, art_d2) # calculate correlation of artificial dataset 2

        art_diff = art_r1 - art_r2 # calculate correlation difference of artificial dataset 1 and 2
        art_diff_col.append(art_diff) # append 

    art_diff_cols = xr.concat(art_diff_col, dim='reps') # concatenate art_diff_col to xarray with all results of reps together
    q = art_diff_cols.quantile([0.025,0.05,0.10,0.90,0.95,0.975], dim="reps") #calculate quantiles of r difference
    
    return q

# ALL MONTHS RMSE E

def run_significance_rmse_e(exp_name1, exp_name2, tp, start_year, end_year, ref_data, fol, reps,dt):
    ref_file = ref_selection(fol, ref_data, tp, start_year=start_year, end_year=end_year)

    f_mod1 = f'{fol}/htessel_output/{exp_name1}'
    f_mod2 = f'{fol}/htessel_output/{exp_name2}'
    out_fol=f'{fol}/scripts/HTESSEL/figures/diff/{exp_name1}_{exp_name2}'

    ref = xr.open_dataset(f'{ref_file[0]}')
    
    mod1 = xr.open_dataset(f'{f_mod1}/monthly/o_efl_{start_year}_{end_year}_monthly.nc')
    mod2 = xr.open_dataset(f'{f_mod2}/monthly/o_efl_{start_year}_{end_year}_monthly.nc')

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
        
    # calculate quantiles of correlation difference
    q = sig_test_rmse(d_mod1d, d_mod2d, d_refd, reps)
    tp2 = tp.replace(" ", "_")
    q.to_netcdf(f'{fol}/scripts/HTESSEL/sig_files/q_levels/quantiles_reps{reps}_{exp_name1}_{exp_name2}_{tp2}_{ref_data}_{start_year}_{end_year}_rmse_2_{dt}_allmonths.nc')
    q2 = xr.open_dataset(f'{fol}/scripts/HTESSEL/sig_files/q_levels/quantiles_reps{reps}_{exp_name1}_{exp_name2}_{tp2}_{ref_data}_{start_year}_{end_year}_rmse_2_{dt}_allmonths.nc')
    q2 = q2.rename({'__xarray_dataarray_variable__':'q_value'})
    # save quantile xarray to file
    q2.to_netcdf(f'{fol}/scripts/HTESSEL/sig_files/q_levels/quantiles_reps{reps}_{exp_name1}_{exp_name2}_{tp2}_{ref_data}_{start_year}_{end_year}_rmse_{dt}_allmonths.nc')
    os.remove(f'{fol}/scripts/HTESSEL/sig_files/q_levels/quantiles_reps{reps}_{exp_name1}_{exp_name2}_{tp2}_{ref_data}_{start_year}_{end_year}_rmse_2_{dt}_allmonths.nc')


def calculate_correlation_rmse_e(fol,ref_data,tp,start_year, end_year,var,exp_name1,exp_name2,dt):
    ref_file = ref_selection(fol, ref_data, tp, start_year=start_year, end_year=end_year)

    f_mod1 = f'{fol}/htessel_output/{exp_name1}'
    f_mod2 = f'{fol}/htessel_output/{exp_name2}'
    out_fol=f'{fol}/scripts/HTESSEL/figures/diff/{exp_name1}_{exp_name2}'

    ref = xr.open_dataset(f'{ref_file[0]}')

    mod1 = xr.open_dataset(f'{f_mod1}/monthly/o_efl_{start_year}_{end_year}_monthly.nc')
    mod2 = xr.open_dataset(f'{f_mod2}/monthly/o_efl_{start_year}_{end_year}_monthly.nc')

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

    r1=LE_rmse(d_refd, d_mod1d)
    r2=LE_rmse(d_refd, d_mod2d)
    r = r1-r2
    
    return r,r1,r2

# SEASONS RMSE E

def run_significance_seasons_rmse_e(exp_name1, exp_name2, tp, start_year, end_year, ref_data, fol, reps,season,dt):
    ref_file = ref_selection(fol, ref_data, tp, start_year=start_year, end_year=end_year)

    f_mod1 = f'{fol}/htessel_output/{exp_name1}'
    f_mod2 = f'{fol}/htessel_output/{exp_name2}'
    out_fol=f'{fol}/scripts/HTESSEL/figures/diff/{exp_name1}_{exp_name2}'

    ref = xr.open_dataset(f'{ref_file[0]}')
    
    mod1 = xr.open_dataset(f'{f_mod1}/seasonal/o_efl_{start_year}_{end_year}_{season}.nc')
    mod2 = xr.open_dataset(f'{f_mod2}/seasonal/o_efl_{start_year}_{end_year}_{season}.nc')

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
        
    # calculate quantiles of correlation difference
    q = sig_test_rmse(d_mod1d, d_mod2d, d_refd, reps)
    tp2 = tp.replace(" ", "_")
    q.to_netcdf(f'{fol}/scripts/HTESSEL/sig_files/q_levels/quantiles_reps{reps}_{exp_name1}_{exp_name2}_{tp2}_{season}_{ref_data}_{start_year}_{end_year}_rmse_2_{dt}.nc')
    q2 = xr.open_dataset(f'{fol}/scripts/HTESSEL/sig_files/q_levels/quantiles_reps{reps}_{exp_name1}_{exp_name2}_{tp2}_{season}_{ref_data}_{start_year}_{end_year}_rmse_2_{dt}.nc')
    q2 = q2.rename({'__xarray_dataarray_variable__':'q_value'})
    # save quantile xarray to file
    q2.to_netcdf(f'{fol}/scripts/HTESSEL/sig_files/q_levels/quantiles_reps{reps}_{exp_name1}_{exp_name2}_{tp2}_{season}_{ref_data}_{start_year}_{end_year}_rmse_{dt}.nc')
    os.remove(f'{fol}/scripts/HTESSEL/sig_files/q_levels/quantiles_reps{reps}_{exp_name1}_{exp_name2}_{tp2}_{season}_{ref_data}_{start_year}_{end_year}_rmse_2_{dt}.nc')


def calculate_correlation_seasons_rmse_e(fol,ref_data,tp,start_year, end_year,var,exp_name1,exp_name2,season,dt):
    ref_file = ref_selection(fol, ref_data, tp, start_year=start_year, end_year=end_year)

    f_mod1 = f'{fol}/htessel_output/{exp_name1}'
    f_mod2 = f'{fol}/htessel_output/{exp_name2}'
    out_fol=f'{fol}/scripts/HTESSEL/figures/diff/{exp_name1}_{exp_name2}'

    ref = xr.open_dataset(f'{ref_file[0]}')

    mod1 = xr.open_dataset(f'{f_mod1}/seasonal/o_efl_{start_year}_{end_year}_{season}.nc')
    mod2 = xr.open_dataset(f'{f_mod2}/seasonal/o_efl_{start_year}_{end_year}_{season}.nc')

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

    r1=LE_rmse(d_refd, d_mod1d)
    r2=LE_rmse(d_refd, d_mod2d)
    r = r1-r2
    
    return r,r1,r2


# ALL MONTHS INTERANNUAL std SOIL MOISTURE ESA CCI CORRELATION
def run_significance_SM_esa_rmse(exp_name1, exp_name2, tp, start_year, end_year, fol, reps,layer,dt,th):
   
    # ESACCI anomalies
    smfol=f'{fol}/DATA/esacci_soilmoisture/processed'
    if (layer=='layer1'):
        somo=xr.open_dataset(f'{smfol}/ESACCI-SOILMOISTURE-1993-2019-stdia_anomalies.nc')
        somo=somo.sm

    # model
    f_mod1 = f'{fol}/htessel_output/{exp_name1}'
    mod1 = xr.open_dataset(f'{f_mod1}/ia_anomalies/o_soil_{start_year}_{end_year}_stdia_anomalies.nc')
    f_mod2 = f'{fol}/htessel_output/{exp_name2}'
    mod2 = xr.open_dataset(f'{f_mod2}/ia_anomalies/o_soil_{start_year}_{end_year}_stdia_anomalies.nc')    
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
        d_mod1=sm1
        d_mod2=sm2
        d_refd=somo
    # calculate quantiles of correlation difference
    q = sig_test_rmse(d_mod1, d_mod2, d_refd, reps)
    tp2 = tp.replace(" ", "_")
    q.to_netcdf(f'{fol}/scripts/HTESSEL/sig_files/q_levels/quantiles_reps{reps}_{exp_name1}_{exp_name2}_{tp2}_esacci_{start_year}_{end_year}_rmse_2_{dt}_{th}.nc')
    q2 = xr.open_dataset(f'{fol}/scripts/HTESSEL/sig_files/q_levels/quantiles_reps{reps}_{exp_name1}_{exp_name2}_{tp2}_esacci_{start_year}_{end_year}_rmse_2_{dt}_{th}.nc')
    q2 = q2.rename({'__xarray_dataarray_variable__':'q_value'})
    # save quantile xarray to file
    q2.to_netcdf(f'{fol}/scripts/HTESSEL/sig_files/q_levels/quantiles_reps{reps}_{exp_name1}_{exp_name2}_{tp2}_esacci_{start_year}_{end_year}_rmse_{dt}_{th}.nc')
    os.remove(f'{fol}/scripts/HTESSEL/sig_files/q_levels/quantiles_reps{reps}_{exp_name1}_{exp_name2}_{tp2}_esacci_{start_year}_{end_year}_rmse_2_{dt}_{th}.nc')   

    
def calculate_rmse_SM_esa(fol,tp,start_year, end_year,exp_name1,exp_name2,layer,dt,th):
    smfol=f'{fol}/DATA/esacci_soilmoisture/processed'
    if (layer=='layer1'):
        somo=xr.open_dataset(f'{smfol}/ESACCI-SOILMOISTURE-1993-2019-stdia_anomalies.nc')
        somo=somo.sm

    # model
    f_mod1 = f'{fol}/htessel_output/{exp_name1}'
    mod1 = xr.open_dataset(f'{f_mod1}/ia_anomalies/o_soil_{start_year}_{end_year}_stdia_anomalies.nc')
    f_mod2 = f'{fol}/htessel_output/{exp_name2}'
    mod2 = xr.open_dataset(f'{f_mod2}/ia_anomalies/o_soil_{start_year}_{end_year}_stdia_anomalies.nc')    
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
        
    r1 = LE_rmse(sm1_dt, somo_dt)
    r2 = LE_rmse(sm1_dt, somo_dt)
    r = r1-r2
    return r,r1,r2

    # SEASONAL INTERANNUAL SOIL MOISTURE CORRELATION
def run_significance_SM_esa_seasons_rmse(exp_name1, exp_name2, tp, start_year, end_year, fol, reps,layer,season,dt,th):
   
    # ESACCI anomalies
    smfol=f'{fol}/DATA/esacci_soilmoisture/processed'
    if (layer=='layer1'):
        somo=xr.open_dataset(f'{smfol}/ESACCI-SOILMOISTURE-1999-2019_{season}_stdanomalies.nc')
        somo=somo.sm

    # model
    f_mod1 = f'{fol}/htessel_output/{exp_name1}'
    mod1 = xr.open_dataset(f'{f_mod1}/seasonal/o_soil_{start_year}_{end_year}_{season}_stdanomalies.nc')
    f_mod2 = f'{fol}/htessel_output/{exp_name2}'
    mod2 = xr.open_dataset(f'{f_mod2}/seasonal/o_soil_{start_year}_{end_year}_{season}_stdanomalies.nc')    
    if (layer=='layer1'):
        sm1 = mod1.SWVL1 #* 0.07 * 1e3 # 7cm
        sm2 = mod2.SWVL1 #* 0.07 * 1e3 # 7cm

    # detrend
    if (dt=='dt_yes'):
        sm1_dt = detrend_allmonths(sm1, dim='time', deg=1)
        sm2_dt = detrend_allmonths(sm2, dim='time', deg=1)
        somo_dt = detrend_allmonths(somo, dim='time', deg=1)
        d_mod1=sm1_dt
        d_mod2=sm2_dt
        d_refd=somo_dt
    else:
        d_mod1=sm1
        d_mod2=sm2
        d_refd=somo
    
    # calculate quantiles of correlation difference
    q = sig_test_rmse(d_mod1, d_mod2, d_refd, reps)
    tp2 = tp.replace(" ", "_")
    q.to_netcdf(f'{fol}/scripts/HTESSEL/sig_files/q_levels/quantiles_reps{reps}_{exp_name1}_{exp_name2}_{tp2}_esacci_{season}_{start_year}_{end_year}_rmse_2_{dt}_{th}.nc')
    q2 = xr.open_dataset(f'{fol}/scripts/HTESSEL/sig_files/q_levels/quantiles_reps{reps}_{exp_name1}_{exp_name2}_{tp2}_esacci_{season}_{start_year}_{end_year}_rmse_2_{dt}_{th}.nc')
    q2 = q2.rename({'__xarray_dataarray_variable__':'q_value'})
    # save quantile xarray to file
    q2.to_netcdf(f'{fol}/scripts/HTESSEL/sig_files/q_levels/quantiles_reps{reps}_{exp_name1}_{exp_name2}_{tp2}_esacci_{season}_{start_year}_{end_year}_rmse_{dt}_{th}.nc')
    os.remove(f'{fol}/scripts/HTESSEL/sig_files/q_levels/quantiles_reps{reps}_{exp_name1}_{exp_name2}_{tp2}_esacci_{season}_{start_year}_{end_year}_rmse_2_{dt}_{th}.nc')   

    
def calculate_rmse_SM_esa_seasons(fol,tp,start_year, end_year,exp_name1,exp_name2,layer,season,dt,th):
    smfol=f'{fol}/DATA/esacci_soilmoisture/processed'
    if (layer=='layer1'):
        somo=xr.open_dataset(f'{smfol}/ESACCI-SOILMOISTURE-1993-2019_{season}_stdanomalies.nc')
        somo=somo.sm

    # detrend
    somo_dt = detrend_allmonths(somo, dim='time', deg=1)

    # model
    f_mod1 = f'{fol}/htessel_output/{exp_name1}'
    mod1 = xr.open_dataset(f'{f_mod1}/seasonal/o_soil_{start_year}_{end_year}_{season}_stdanomalies.nc')
    f_mod2 = f'{fol}/htessel_output/{exp_name2}'
    mod2 = xr.open_dataset(f'{f_mod2}/seasonal/o_soil_{start_year}_{end_year}_{season}_stdanomalies.nc')    
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

    r1 = LE_rmse(sm1_dt, somo_dt)
    r2 = LE_rmse(sm1_dt, somo_dt)
    r = r1-r2
    return r,r1,r2