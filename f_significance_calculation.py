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
def sig_test_corr(d_mod1, d_mod2,d_refd, reps):
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
        art_r1=LE_corr(d_refd, art_d1) # calculate correlation of artificial dataset 1

        rd = np.random.randint(2, size=n)
        da2 = []
        for k in range(n):
            if rd[k]==0:
                r = d_mod1[k]
            elif rd[k]==1:
                r = d_mod2[k]
            da2.append(r)
        art_d2 = xr.concat(da2, dim='time').assign_coords(d_mod1.coords)
        art_r2=LE_corr(d_refd, art_d2) # calculate correlation of artificial dataset 2

        art_diff = art_r1 - art_r2 # calculate correlation difference of artificial dataset 1 and 2
        art_diff_col.append(art_diff) # append 

    art_diff_cols = xr.concat(art_diff_col, dim='reps') # concatenate art_diff_col to xarray with all results of reps together
    q = art_diff_cols.quantile([0.025,0.05,0.10,0.90,0.95,0.975], dim="reps") #calculate quantiles of r difference
    
    return q

def sig_test_bias(d_mod1, d_mod2,d_refd, reps):
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
        art_r1=LE_pbias(d_refd, art_d1) # calculate correlation of artificial dataset 1

        rd = np.random.randint(2, size=n)
        da2 = []
        for k in range(n):
            if rd[k]==0:
                r = d_mod1[k]
            elif rd[k]==1:
                r = d_mod2[k]
            da2.append(r)
        art_d2 = xr.concat(da2, dim='time').assign_coords(d_mod1.coords)
        art_r2=LE_pbias(d_refd, art_d2) # calculate correlation of artificial dataset 2

        art_diff = abs(art_r1) - abs(art_r2) # calculate correlation difference of artificial dataset 1 and 2
        art_diff_col.append(art_diff) # append 

    art_diff_cols = xr.concat(art_diff_col, dim='reps') # concatenate art_diff_col to xarray with all results of reps together
    q = art_diff_cols.quantile([0.025,0.05,0.10,0.90,0.95,0.975], dim="reps") #calculate quantiles of r difference
    
    return q


# ALL MONTHS INTERANNUAL EVAPORATION CORRELATION
def run_significance(exp_name1, exp_name2, tp, start_year, end_year, ref_data, fol, reps,dt):
    ref_file = ref_selection(fol, ref_data, tp, start_year=start_year, end_year=end_year)
    
    f_mod1 = f'{fol}/htessel_output/{exp_name1}'
    f_mod2 = f'{fol}/htessel_output/{exp_name2}'

    ref = xr.open_dataset(f'{ref_file[0]}')
    
    mod1 = xr.open_dataset(f'{f_mod1}/ia_anomalies/o_efl_{start_year}_{end_year}_ia_anomalies.nc')
    mod2 = xr.open_dataset(f'{f_mod2}/ia_anomalies/o_efl_{start_year}_{end_year}_ia_anomalies.nc')

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
        
    # calculate quantiles of correlation difference
    q = sig_test_corr(d_mod1d, d_mod2d, d_refd, reps)
    tp2 = tp.replace(" ", "_")
    q.to_netcdf(f'{fol}/scripts/HTESSEL/sig_files/q_levels/quantiles_reps{reps}_{exp_name1}_{exp_name2}_{tp2}_{ref_data}_{start_year}_{end_year}_corr_2_{dt}.nc')
    q2 = xr.open_dataset(f'{fol}/scripts/HTESSEL/sig_files/q_levels/quantiles_reps{reps}_{exp_name1}_{exp_name2}_{tp2}_{ref_data}_{start_year}_{end_year}_corr_2_{dt}.nc')
    q2 = q2.rename({'__xarray_dataarray_variable__':'q_value'})
    # save quantile xarray to file
    q2.to_netcdf(f'{fol}/scripts/HTESSEL/sig_files/q_levels/quantiles_reps{reps}_{exp_name1}_{exp_name2}_{tp2}_{ref_data}_{start_year}_{end_year}_corr_{dt}.nc')
    os.remove(f'{fol}/scripts/HTESSEL/sig_files/q_levels/quantiles_reps{reps}_{exp_name1}_{exp_name2}_{tp2}_{ref_data}_{start_year}_{end_year}_corr_2_{dt}.nc')   

    
def calculate_correlation(fol,ref_data,tp,start_year, end_year,var,exp_name1,exp_name2,dt):
    ref_file = ref_selection(fol, ref_data, tp, start_year=start_year, end_year=end_year)

    f_mod1 = f'{fol}/htessel_output/{exp_name1}'
    f_mod2 = f'{fol}/htessel_output/{exp_name2}'

    ref = xr.open_dataset(f'{ref_file[0]}')

    mod1 = xr.open_dataset(f'{f_mod1}/ia_anomalies/o_efl_{start_year}_{end_year}_ia_anomalies.nc')
    mod2 = xr.open_dataset(f'{f_mod2}/ia_anomalies/o_efl_{start_year}_{end_year}_ia_anomalies.nc')

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

def plot_correlation_significance(q,r,exp_name1,exp_name2,tp,ref_data,start_year,end_year,reps,dt):
# load quantiles for significance test
# tp2 = tp.replace(" ", "_")
# q = xr.open_dataset(f'{fol}/scripts/HTESSEL/sig_files/quantiles_reps{reps}_{exp_name1}_{exp_name2}_{tp2}_{ref_data}_{start_year}_{end_year}_corr_{dt}.nc')

    # PLOT RESULTS WITH SIGNIFICANCE
    lvls = np.linspace(-0.05,0.05,11)
    cm = plt.cm.get_cmap('PiYG')   

    # r - all values
    # count total points
    rc = r.stack(dim=['lat','lon']).values
    rc = rc[~np.isnan(rc)]
    rc = len(rc)

    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    cf = r.plot(ax=ax, cmap=cm, add_colorbar=False, vmin=0, vmax=5, levels=lvls)
    cbar = plt.colorbar(cf, orientation='horizontal',extend='both',label=f'r {exp_name2}-{exp_name1} (-)', pad=0.05, ticks=lvls)
    ax.set_title(f' {exp_name2}-{exp_name1} - r - {tp} E - {start_year}-{end_year} - no sig - {rc}points {dt}')
    ax.set_ylim(-60,90)
    ax.set_xlim(-180,180)
    fig.savefig(f'/home/vanoorschot/work/fransje/scripts/HTESSEL/figures/paper_figures/{exp_name1}_{exp_name2}/sig_tests/figures/evaporation/corrdiff_E_{exp_name1}_{exp_name2}_ia_anomalies_allmonths_{start_year}_{end_year}_dolce_nosig_{dt}.jpg',dpi=300,bbox_inches='tight')

    # r - significant 10%
    # count total points
    rs = r.where((r<q.q_value[4]) & (r>q.q_value[1])) #non significant points
    rc10 = rs.stack(dim=['lat','lon']).values
    rc10 = rc10[~np.isnan(rc10)]
    rc10 = len(rc10)
    rc10p = np.round((rc10/rc * 100),2)

    lvls = np.linspace(-0.05,0.05,11)
    lvls2 = np.linspace(-0.1,0.1,2)
    cm = plt.cm.get_cmap('PiYG')  
    cm2 = plt.cm.get_cmap('Pastel2_r')

    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    cf = r.plot(ax=ax, cmap=cm, add_colorbar=False, vmin=0, vmax=5, levels=lvls)#plot all points
    cf2 = rs.plot(ax=ax,cmap=cm2,add_colorbar=False,levels=lvls2)#plot in grey all non-significant points
    cbar = plt.colorbar(cf, orientation='horizontal',extend='both',label=f'r {exp_name2}-{exp_name1} (-)', pad=0.05, ticks=lvls)
    ax.set_title(f' {exp_name2}-{exp_name1} - r - {tp} E - {start_year}-{end_year} - sig 10%, reps{reps} - {rc10}/{rc}={rc10p} % points sig')
    ax.set_ylim(-60,90)
    ax.set_xlim(-180,180)
    fig.savefig(f'/home/vanoorschot/work/fransje/scripts/HTESSEL/figures/paper_figures/{exp_name1}_{exp_name2}/sig_tests/figures/evaporation/corrdiff_E_{exp_name1}_{exp_name2}_{reps}_ia_anomalies_allmonths_{start_year}_{end_year}_dolce_sig10_{dt}.jpg',dpi=300,bbox_inches='tight')

    
# SEASONS INTERNANNUAL CORRELATION EVAPORATION

def run_significance_seasons(exp_name1, exp_name2, tp, start_year, end_year, ref_data, fol, reps,season,dt):
    ref_file = ref_selection(fol, ref_data, tp, start_year=start_year, end_year=end_year)

    f_mod1 = f'{fol}/htessel_output/{exp_name1}'
    f_mod2 = f'{fol}/htessel_output/{exp_name2}'
    out_fol=f'{fol}/scripts/HTESSEL/figures/diff/{exp_name1}_{exp_name2}'

    ref = xr.open_dataset(f'{ref_file[0]}')
    
    mod1 = xr.open_dataset(f'{f_mod1}/seasonal/o_efl_{start_year}_{end_year}_{season}_anomalies.nc')
    mod2 = xr.open_dataset(f'{f_mod2}/seasonal/o_efl_{start_year}_{end_year}_{season}_anomalies.nc')

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
    q = sig_test_corr(d_mod1d, d_mod2d, d_refd, reps)
    tp2 = tp.replace(" ", "_")
    q.to_netcdf(f'{fol}/scripts/HTESSEL/sig_files/q_levels/quantiles_reps{reps}_{exp_name1}_{exp_name2}_{tp2}_{season}_{ref_data}_{start_year}_{end_year}_corr_2_{dt}.nc')
    q2 = xr.open_dataset(f'{fol}/scripts/HTESSEL/sig_files/q_levels/quantiles_reps{reps}_{exp_name1}_{exp_name2}_{tp2}_{season}_{ref_data}_{start_year}_{end_year}_corr_2_{dt}.nc')
    q2 = q2.rename({'__xarray_dataarray_variable__':'q_value'})
    # save quantile xarray to file
    q2.to_netcdf(f'{fol}/scripts/HTESSEL/sig_files/q_levels/quantiles_reps{reps}_{exp_name1}_{exp_name2}_{tp2}_{season}_{ref_data}_{start_year}_{end_year}_corr_{dt}.nc')
    os.remove(f'{fol}/scripts/HTESSEL/sig_files/q_levels/quantiles_reps{reps}_{exp_name1}_{exp_name2}_{tp2}_{season}_{ref_data}_{start_year}_{end_year}_corr_2_{dt}.nc')


def calculate_correlation_seasons(fol,ref_data,tp,start_year, end_year,var,exp_name1,exp_name2,season,dt):
    ref_file = ref_selection(fol, ref_data, tp, start_year=start_year, end_year=end_year)

    f_mod1 = f'{fol}/htessel_output/{exp_name1}'
    f_mod2 = f'{fol}/htessel_output/{exp_name2}'
    out_fol=f'{fol}/scripts/HTESSEL/figures/diff/{exp_name1}_{exp_name2}'

    ref = xr.open_dataset(f'{ref_file[0]}')

    mod1 = xr.open_dataset(f'{f_mod1}/seasonal/o_efl_{start_year}_{end_year}_{season}_anomalies.nc')
    mod2 = xr.open_dataset(f'{f_mod2}/seasonal/o_efl_{start_year}_{end_year}_{season}_anomalies.nc')

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

    
def plot_correlation_significance_seasons(q,r,exp_name1,exp_name2,tp,ref_data,start_year,end_year,season,reps,dt):
# load quantiles for significance test
# tp2 = tp.replace(" ", "_")
# q = xr.open_dataset(f'{fol}/scripts/HTESSEL/sig_files/quantiles_reps{reps}_{exp_name1}_{exp_name2}_{tp2}_{ref_data}_{start_year}_{end_year}_corr_{dt}.nc')

    # PLOT RESULTS WITH SIGNIFICANCE
    lvls = np.linspace(-0.10,0.10,11)
    cm = plt.cm.get_cmap('PiYG')   

    # r - all values
    # count total points
    rc = r.stack(dim=['lat','lon']).values
    rc = rc[~np.isnan(rc)]
    rc = len(rc)

    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    cf = r.plot(ax=ax, cmap=cm, add_colorbar=False, vmin=0, vmax=5, levels=lvls)
    cbar = plt.colorbar(cf, orientation='horizontal',extend='both',label=f'r {exp_name2}-{exp_name1} (-)', pad=0.05, ticks=lvls)
    ax.set_title(f' {exp_name2}-{exp_name1} - r - {tp} E - {start_year}-{end_year} - no sig - {rc}points')
    ax.set_ylim(-60,90)
    ax.set_xlim(-180,180)
    fig.savefig(f'/home/vanoorschot/work/fransje/scripts/HTESSEL/figures/paper_figures/{exp_name1}_{exp_name2}/sig_tests/figures/evaporation/corrdiff_E_{exp_name1}_{exp_name2}_ia_anomalies_{start_year}_{end_year}_{season}_dolce_nosig_{dt}.jpg',dpi=300,bbox_inches='tight')

    # r - significant 10%
    # count total points
    rs = r.where((r>q.q_value[4]) | (r<q.q_value[1])) #5% and 95%
    rc10 = rs.stack(dim=['lat','lon']).values
    rc10 = rc10[~np.isnan(rc10)]
    rc10 = len(rc10)
    rc10p = np.round((rc10/rc * 100),2)
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    cf = rs.plot(ax=ax, cmap=cm, add_colorbar=False, vmin=0, vmax=5, levels=lvls)
    cbar = plt.colorbar(cf, orientation='horizontal',extend='both',label=f'r {exp_name2}-{exp_name1} (-)', pad=0.05, ticks=lvls)
    ax.set_title(f' {exp_name2}-{exp_name1} - r - {tp} E - {start_year}-{end_year} {season}- sig 10%, reps{reps} - {rc10}/{rc}={rc10p} % points sig')
    ax.set_ylim(-60,90)
    ax.set_xlim(-180,180)
    fig.savefig(f'/home/vanoorschot/work/fransje/scripts/HTESSEL/figures/paper_figures/{exp_name1}_{exp_name2}/sig_tests/figures/evaporation/corrdiff_E_{exp_name1}_{exp_name2}_{reps}_ia_anomalies_{start_year}_{end_year}_{season}_dolce_sig10_{dt}.jpg',dpi=300,bbox_inches='tight')
    
    
# ALL MONTHS INTERANNUAL SOIL MOISTURE ESA CCI CORRELATION
def run_significance_SM_esa(exp_name1, exp_name2, tp, start_year, end_year, fol, reps,layer,dt,th):
   
    # ESACCI anomalies
    smfol=f'{fol}/DATA/esacci_soilmoisture/processed'
    if (layer=='layer1'):
        somo=xr.open_dataset(f'{smfol}/ESACCI-SOILMOISTURE-1999-2019-ia_anomalies_{th}mask.nc')
        somo=somo.sm

    # model
    f_mod1 = f'{fol}/htessel_output/{exp_name1}'
    mod1 = xr.open_dataset(f'{f_mod1}/ia_anomalies/o_soil_{start_year}_{end_year}_ia_anomalies.nc')
    f_mod2 = f'{fol}/htessel_output/{exp_name2}'
    mod2 = xr.open_dataset(f'{f_mod2}/ia_anomalies/o_soil_{start_year}_{end_year}_ia_anomalies.nc')    
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
    q = sig_test_corr(d_mod1, d_mod2, d_refd, reps)
    tp2 = tp.replace(" ", "_")
    q.to_netcdf(f'{fol}/scripts/HTESSEL/sig_files/q_levels/quantiles_reps{reps}_{exp_name1}_{exp_name2}_{tp2}_esacci_{start_year}_{end_year}_corr_2_{dt}_{th}.nc')
    q2 = xr.open_dataset(f'{fol}/scripts/HTESSEL/sig_files/q_levels/quantiles_reps{reps}_{exp_name1}_{exp_name2}_{tp2}_esacci_{start_year}_{end_year}_corr_2_{dt}_{th}.nc')
    q2 = q2.rename({'__xarray_dataarray_variable__':'q_value'})
    # save quantile xarray to file
    q2.to_netcdf(f'{fol}/scripts/HTESSEL/sig_files/q_levels/quantiles_reps{reps}_{exp_name1}_{exp_name2}_{tp2}_esacci_{start_year}_{end_year}_corr_{dt}_{th}.nc')
    os.remove(f'{fol}/scripts/HTESSEL/sig_files/q_levels/quantiles_reps{reps}_{exp_name1}_{exp_name2}_{tp2}_esacci_{start_year}_{end_year}_corr_2_{dt}_{th}.nc')   

    
def calculate_correlation_SM_esa(fol,tp,start_year, end_year,exp_name1,exp_name2,layer,dt,th):
    smfol=f'{fol}/DATA/esacci_soilmoisture/processed'
    if (layer=='layer1'):
        somo=xr.open_dataset(f'{smfol}/ESACCI-SOILMOISTURE-1999-2019-ia_anomalies_{th}mask.nc')
        somo=somo.sm

    # model
    f_mod1 = f'{fol}/htessel_output/{exp_name1}'
    mod1 = xr.open_dataset(f'{f_mod1}/ia_anomalies/o_soil_{start_year}_{end_year}_ia_anomalies.nc')
    f_mod2 = f'{fol}/htessel_output/{exp_name2}'
    mod2 = xr.open_dataset(f'{f_mod2}/ia_anomalies/o_soil_{start_year}_{end_year}_ia_anomalies.nc')    
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


def plot_correlation_significance_SM_esa(q,r,exp_name1,exp_name2,tp,start_year,end_year,reps,layer,dt):
# load quantiles for significance test
# tp2 = tp.replace(" ", "_")
# q = xr.open_dataset(f'{fol}/scripts/HTESSEL/sig_files/quantiles_reps{reps}_{exp_name1}_{exp_name2}_{tp2}_{ref_data}_{start_year}_{end_year}_corr_{dt}.nc')

    # PLOT RESULTS WITH SIGNIFICANCE
    lvls = np.linspace(-0.05,0.05,11)
    cm = plt.cm.get_cmap('PiYG')   

    # r - all values
    # count total points
    rc = r.stack(dim=['lat','lon']).values
    rc = rc[~np.isnan(rc)]
    rc = len(rc)

    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    cf = r.plot(ax=ax, cmap=cm, add_colorbar=False, vmin=0, vmax=5, levels=lvls)
    cbar = plt.colorbar(cf, orientation='horizontal',extend='both',label=f'r {exp_name2}-{exp_name1} (-)', pad=0.05, ticks=lvls)
    ax.set_title(f' {exp_name2}-{exp_name1} - r - {tp} - SM {layer} - esacci- {start_year}-{end_year} - no sig - {rc}points {dt}')
    ax.set_ylim(-60,90)
    ax.set_xlim(-180,180)
    fig.savefig(f'/home/vanoorschot/work/fransje/scripts/HTESSEL/figures/paper_figures/{exp_name1}_{exp_name2}/sig_tests/figures/soilmoisture/corrdiff_SM{layer}_{exp_name1}_{exp_name2}_ia_anomalies_allmonths_{start_year}_{end_year}_esacci_nosig_{dt}.jpg',dpi=300,bbox_inches='tight')
    
    # r - significant 10%
    rs = r.where((r<q.q_value[4]) & (r>q.q_value[1])) #non significant points
    rc10 = rs.stack(dim=['lat','lon']).values
    rc10 = rc10[~np.isnan(rc10)]
    rc10 = len(rc10)
    rc10p = np.round((rc10/rc * 100),2)

    lvls = np.linspace(-0.05,0.05,11)
    lvls2 = np.linspace(-0.001,0.001,2)
    cm = plt.cm.get_cmap('PiYG')  
    # cm2 = plt.cm.get_cmap('Pastel2')

    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    cf = r.plot(ax=ax, cmap=cm, add_colorbar=False, vmin=0, vmax=5, levels=lvls)#plot all points
    cf2 = rs.plot(ax=ax,cmap=cmap_grey,add_colorbar=False,levels=lvls2)#plot in grey all non-significant points
    cbar = plt.colorbar(cf, orientation='horizontal',extend='both',label=f'r {exp_name2}-{exp_name1} (-)', pad=0.05, ticks=lvls)
    ax.set_title(f' {exp_name2}-{exp_name1} - r - {tp} - SM {layer} - esacci - {start_year}-{end_year} - sig 10%, reps{reps} - {rc10}/{rc}={rc10p} % points sig {dt}')
    ax.set_ylim(-60,90)
    ax.set_xlim(-180,180)
    fig.savefig(f'/home/vanoorschot/work/fransje/scripts/HTESSEL/figures/paper_figures/{exp_name1}_{exp_name2}/sig_tests/figures/soilmoisture/corrdiff_SM{layer}_{exp_name1}_{exp_name2}_{reps}_ia_anomalies_allmonths_{start_year}_{end_year}_esacci_sig10_{dt}.jpg',dpi=300,bbox_inches='tight')


    
    # SEASONAL INTERANNUAL SOIL MOISTURE CORRELATION
def run_significance_SM_esa_seasons(exp_name1, exp_name2, tp, start_year, end_year, fol, reps,layer,season,dt,th):
   
    # ESACCI anomalies
    smfol=f'{fol}/DATA/esacci_soilmoisture/processed'
    if (layer=='layer1'):
        somo=xr.open_dataset(f'{smfol}/ESACCI-SOILMOISTURE-1999-2019_{season}_anomalies_{th}mask.nc')
        somo=somo.sm

    # model
    f_mod1 = f'{fol}/htessel_output/{exp_name1}'
    mod1 = xr.open_dataset(f'{f_mod1}/seasonal/o_soil_{start_year}_{end_year}_{season}_anomalies.nc')
    f_mod2 = f'{fol}/htessel_output/{exp_name2}'
    mod2 = xr.open_dataset(f'{f_mod2}/seasonal/o_soil_{start_year}_{end_year}_{season}_anomalies.nc')    
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
    q = sig_test_corr(d_mod1, d_mod2, d_refd, reps)
    tp2 = tp.replace(" ", "_")
    q.to_netcdf(f'{fol}/scripts/HTESSEL/sig_files/q_levels/quantiles_reps{reps}_{exp_name1}_{exp_name2}_{tp2}_esacci_{season}_{start_year}_{end_year}_corr_2_{dt}_{th}.nc')
    q2 = xr.open_dataset(f'{fol}/scripts/HTESSEL/sig_files/q_levels/quantiles_reps{reps}_{exp_name1}_{exp_name2}_{tp2}_esacci_{season}_{start_year}_{end_year}_corr_2_{dt}_{th}.nc')
    q2 = q2.rename({'__xarray_dataarray_variable__':'q_value'})
    # save quantile xarray to file
    q2.to_netcdf(f'{fol}/scripts/HTESSEL/sig_files/q_levels/quantiles_reps{reps}_{exp_name1}_{exp_name2}_{tp2}_esacci_{season}_{start_year}_{end_year}_corr_{dt}_{th}.nc')
    os.remove(f'{fol}/scripts/HTESSEL/sig_files/q_levels/quantiles_reps{reps}_{exp_name1}_{exp_name2}_{tp2}_esacci_{season}_{start_year}_{end_year}_corr_2_{dt}_{th}.nc')   

    
def calculate_correlation_SM_esa_seasons(fol,tp,start_year, end_year,exp_name1,exp_name2,layer,season,dt,th):
    smfol=f'{fol}/DATA/esacci_soilmoisture/processed'
    if (layer=='layer1'):
        somo=xr.open_dataset(f'{smfol}/ESACCI-SOILMOISTURE-1999-2019_{season}_anomalies_{th}mask.nc')
        somo=somo.sm

    # detrend
    somo_dt = detrend_allmonths(somo, dim='time', deg=1)

    # model
    f_mod1 = f'{fol}/htessel_output/{exp_name1}'
    mod1 = xr.open_dataset(f'{f_mod1}/seasonal/o_soil_{start_year}_{end_year}_{season}_anomalies.nc')
    f_mod2 = f'{fol}/htessel_output/{exp_name2}'
    mod2 = xr.open_dataset(f'{f_mod2}/seasonal/o_soil_{start_year}_{end_year}_{season}_anomalies.nc')    
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

def plot_correlation_significance_SM_esa_seasons(q,r,exp_name1,exp_name2,tp,start_year,end_year,reps,layer,season,dt):
# load quantiles for significance test
# tp2 = tp.replace(" ", "_")
# q = xr.open_dataset(f'{fol}/scripts/HTESSEL/sig_files/quantiles_reps{reps}_{exp_name1}_{exp_name2}_{tp2}_{ref_data}_{start_year}_{end_year}_corr_{dt}.nc')

    # PLOT RESULTS WITH SIGNIFICANCE
    lvls = np.linspace(-0.05,0.05,11)
    cm = plt.cm.get_cmap('PiYG')   

    # r - all values
    # count total points
    rc = r.stack(dim=['lat','lon']).values
    rc = rc[~np.isnan(rc)]
    rc = len(rc)

    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    cf = r.plot(ax=ax, cmap=cm, add_colorbar=False, vmin=0, vmax=5, levels=lvls)
    cbar = plt.colorbar(cf, orientation='horizontal',extend='both',label=f'r {exp_name2}-{exp_name1} (-)', pad=0.05, ticks=lvls)
    ax.set_title(f' {exp_name2}-{exp_name1} - r - {tp} - SM {layer} - esacci {season} - {start_year}-{end_year} - no sig - {rc}points {dt}')
    ax.set_ylim(-60,90)
    ax.set_xlim(-180,180)
    fig.savefig(f'/home/vanoorschot/work/fransje/scripts/HTESSEL/figures/paper_figures/{exp_name1}_{exp_name2}/sig_tests/figures/soilmoisture/corrdiff_SM{layer}_{exp_name1}_{exp_name2}_ia_anomalies_{season}_{start_year}_{end_year}_esacci_nosig_{dt}.jpg',dpi=300,bbox_inches='tight')

    # r - significant 10%
    # count total points
    rs = r.where((r>q.q_value[4]) | (r<q.q_value[1])) #5% and 95%
    rc10 = rs.stack(dim=['lat','lon']).values
    rc10 = rc10[~np.isnan(rc10)]
    rc10 = len(rc10)
    rc10p = np.round((rc10/rc * 100),2)
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    cf = rs.plot(ax=ax, cmap=cm, add_colorbar=False, vmin=0, vmax=5, levels=lvls)
    cbar = plt.colorbar(cf, orientation='horizontal',extend='both',label=f'r {exp_name2}-{exp_name1} (-)', pad=0.05, ticks=lvls)
    ax.set_title(f' {exp_name2}-{exp_name1} - r - {tp} - SM {layer} - esacci {season}- {start_year}-{end_year} - sig 10%, reps{reps} - {rc10}/{rc}={rc10p} % points sig {dt}')
    ax.set_ylim(-60,90)
    ax.set_xlim(-180,180)
    fig.savefig(f'/home/vanoorschot/work/fransje/scripts/HTESSEL/figures/paper_figures/{exp_name1}_{exp_name2}/sig_tests/figures/soilmoisture/corrdiff_SM{layer}_{exp_name1}_{exp_name2}_{reps}_ia_anomalies_{season}_{start_year}_{end_year}_esacci_sig10_{dt}.jpg',dpi=300,bbox_inches='tight')

