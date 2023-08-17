import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd
import xarray as xr

# reference data selection
def ref_selection(fol, ref_data, tp, start_year, end_year):
    fol_ref = f'{fol}/data/ref_data'
    tp2 = tp.replace(" ", "_")
    file_ref = f'{fol_ref}/E_{ref_data}_monthly_{start_year}_{end_year}_{tp2}.nc'
    file_ref_f = f'{fol_ref}/E_{ref_data}_monthly_{start_year}_{end_year}_{tp2}_fluxmask.nc'
    return file_ref, file_ref_f

## preprocess datasets
def pp_ref(ref,ref_data):
    if (ref_data == 'FLUXCOM_RS') or (ref_data == 'FLUXCOM_RSMETEO_GSWP3'):  
        d_ref = ref.LE *(1/2.45)
    if (ref_data == 'CLASS') or (ref_data == 'DOLCE_v3') or (ref_data == 'DOLCE_v2-1'):
        d_ref = ref.hfls/28.94 # from w/m2 to mm/day
        d_ref_u = ref.hfls_sd/28.94
    if (ref_data == 'GLEAM_v3.5a') or (ref_data == 'GLEAM_v3.5b'):
        d_ref = ref.E
    if (ref_data == 'ensemble_FLUXCOM_RS_BESS_PML'):
        d_ref = ref.Evaporation
    return(d_ref,d_ref_u)
    
def pp_mod(mod):
    d_mod = mod.SLHF * -1 * 10 **-6*(1/2.45) #change units
    # d_mod = mod_pos(d_mod)
    return(d_mod)

def detrend_allmonths(da, dim, deg=1):
    # detrend along a single dimension
    p = da.polyfit(dim=dim, deg=deg)
    slope = p.polyfit_coefficients[0]
    intc = p.polyfit_coefficients[1]
    # intc_z = intc.where(((intc<0)&(intc>0)) | intc.isnull(), other=0) # we only want to subtract the slope and not the intercept, keep the mean so set intercept values to zero
    f = xr.concat([slope,intc], dim='degree')
    fit = xr.polyval(da[dim], f) 
    return da - fit

def detrend_permonth(da, dim, deg=1):
    # remove the trend per month separately
    month_idxs=da.groupby('time.month').groups
    l = []
    for k in range(12):
        m_idxs=month_idxs[k+1]
        da_m=da.isel(time=m_idxs)
        p = da_m.polyfit(dim=dim, deg=deg)
        slope = p.polyfit_coefficients[0]
        intc = p.polyfit_coefficients[1]
        # intc_z = intc.where(((intc<0)&(intc>0)) | intc.isnull(), other=0) # we only want to subtract the slope and not the intercept, keep the mean so set intercept values to zero
        f = xr.concat([slope,intc], dim='degree')
        fit_m = xr.polyval(da_m[dim], f)
        d_m = da_m - fit_m
        l.append(d_m)

    c = xr.concat(l, dim='time')
    c = c.sortby('time')
    return c

def mod_pos(d_mod):
    d_mod = d_mod.where((d_mod>0) | d_mod.isnull(), other=0)
    return d_mod