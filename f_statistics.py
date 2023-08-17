import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd
import xarray as xr
import iris
import iris.analysis.cartography
import iris.plot as iplt
import iris.coord_categorisation
import iris.analysis.stats

   
## STATISTICS
def LE_mean_ref(d_ref):
    m = d_ref.mean(dim='time')
    return(m)

def LE_mean_mod(d_mod):
    m = d_mod.mean(dim='time')
    return(m)

def LE_corr(d_ref, d_mod):
    r = xr.corr(d_ref,d_mod, dim=['time'])
    return(r)

def LE_corr_sp(d_ref, d_mod):
    r = scipy.stats.spearmanr(d_ref,d_mod)
    return(r)


def LE_rmse(d_ref, d_mod):
    er2 = (d_ref - d_mod)**2
    rmse = np.sqrt(er2.mean(dim=['time']))  
    return(rmse)

def LE_nrmse(d_ref, d_mod): #only for full field
    er2 = (d_ref - d_mod)**2
    nrmse = np.sqrt(er2.mean(dim=['time']))/ d_ref.mean(dim=['time'])  
    return(nrmse)

def LE_pbias(d_ref, d_mod): #only for full field
    pbias = ((d_mod.mean(dim=['time']) - d_ref.mean(dim=['time']))/d_ref.mean(dim=['time']))
    return(pbias)

def LE_var(d_ref, d_mod): #only for full field
    ref_std = d_ref.std(dim=['time'])
    mod_std = d_mod.std(dim=['time'])
    a = mod_std/ref_std
    v = (1-a)#**2
    return(v)

def LE_global_mean(cube_ref, cube_mod):
    # calculate area weighted mean
    cube_ref.coord('latitude').guess_bounds()
    cube_ref.coord('longitude').guess_bounds()
    cube_mod.coord('latitude').guess_bounds()
    cube_mod.coord('longitude').guess_bounds()
    grid_areas = iris.analysis.cartography.area_weights(cube_ref)
    cube_ref_m = cube_ref.collapsed(['longitude','latitude'], iris.analysis.MEAN, weights=grid_areas)
    cube_mod_m = cube_mod.collapsed(['longitude','latitude'], iris.analysis.MEAN, weights=grid_areas)
    
    # convert to xarray
    ref_m = xr.DataArray.from_iris(cube_ref_m)
    mod_m = xr.DataArray.from_iris(cube_mod_m)
    return(ref_m,mod_m)

def LE_global_mean_mod(cube_mod):
    # calculate area weighted mean
    cube_mod.coord('latitude').guess_bounds()
    cube_mod.coord('longitude').guess_bounds()
    grid_areas = iris.analysis.cartography.area_weights(cube_mod)
    cube_mod_m = cube_mod.collapsed(['longitude','latitude'], iris.analysis.MEAN, weights=grid_areas)
    
    # convert to xarray
    mod_m = xr.DataArray.from_iris(cube_mod_m)
    return(mod_m)