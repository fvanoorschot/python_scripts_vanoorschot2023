import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import leastsq
import scipy.stats as spst
from scipy.stats import gaussian_kde
from matplotlib import cm
from pathos.threading import ThreadPool as Pool

def veg_types(vt):
    if (vt==1):
        l = 'TVL1 Crops'
    if (vt==2):
        l = 'TVL2 Short grass'
    if (vt==3):
        l = 'TVH3 Evergreen needleleaf trees'
    if (vt==4):
        l = 'TVH4 Deciduous needleleaf trees'
    if (vt==5):
        l = 'TVH5 Deciduous broadleaf trees'
    if (vt==6):
        l = 'TVH6 Evergreen broadleaf trees'
    if (vt==9):
        l = 'TVL9 Tundra'
    if (vt==13):
        l = 'TVL13 Bogs and marshes'
    if (vt==16):
        l = 'TVL16 Evergreen shrubs'
    if (vt==17):
        l = 'TVL17 Deciduous shrubs'
    return l

# function to do the fitting and plot results
def fitting_run(vt,th,year_start,year_end):
    fol='/home/vanoorschot/work/fransje/scripts/LAI_FCOVER/fittings/fitting_1km/final'
    lai_array = np.load(f'{fol}/output/all_years_arrays_2000/x_{year_start}_{year_end}_{vt}_{th}.npy')
    fc_array = np.load(f'{fol}/output/all_years_arrays_2000/y_{year_start}_{year_end}_{vt}_{th}.npy')
    
    a = fitting(lai_array,fc_array)
    x = lai_array
    y = fc_array
    
    x_fitted = a[2]
    y_fitted = a[3]
    pval = a[4]
    par = a[5]
    rmse = a[6]
    np.save(f'{fol}/output/fitting_2000/x_fitted_{year_start}_{year_end}_{vt}_{th}_k.npy',x_fitted)
    np.save(f'{fol}/output/fitting_2000/y_fitted_{year_start}_{year_end}_{vt}_{th}_k.npy',y_fitted)
    np.save(f'{fol}/output/fitting_2000/pval_{year_start}_{year_end}_{vt}_{th}_k.npy',pval)
    np.save(f'{fol}/output/fitting_2000/par_{year_start}_{year_end}_{vt}_{th}_k.npy',par)
    np.save(f'{fol}/output/fitting_2000/rmse_{year_start}_{year_end}_{vt}_{th}_k.npy',rmse)
    return x,y,x_fitted,y_fitted,pval,par

# fucntion to make table of fitting results
def make_fitting_table(vt_l, th_l,year_start,year_end):
    fol='/home/vanoorschot/work/fransje/scripts/LAI_FCOVER/fittings/fitting_1km/final'
    df = pd.DataFrame(index=['high/low','#points','threshold','k','p-val','rmse'])
    
    for i in range(len(vt_l)):
        vt = vt_l[i]
        th = th_l[i]
        x= np.load(f'{fol}/output/all_years_arrays_2000/x_{year_start}_{year_end}_{vt}_{th}.npy')
        y= np.load(f'{fol}/output/all_years_arrays_2000/y_{year_start}_{year_end}_{vt}_{th}.npy') 
        x_fitted=np.load(f'{fol}/output/fitting_2000/x_fitted_{year_start}_{year_end}_{vt}_{th}_k.npy')
        y_fitted=np.load(f'{fol}/output/fitting_2000/y_fitted_{year_start}_{year_end}_{vt}_{th}_k.npy')
        pval= np.load(f'{fol}/output/fitting_2000/pval_{year_start}_{year_end}_{vt}_{th}_k.npy')
        par= np.load(f'{fol}/output/fitting_2000/par_{year_start}_{year_end}_{vt}_{th}_k.npy')
        rmse= np.load(f'{fol}/output/fitting_2000/rmse_{year_start}_{year_end}_{vt}_{th}_k.npy')

        if (vt==3) or (vt==4) or (vt==5) or (vt==6):
            t = 'high'
        else:
            t = 'low'
        df.loc['high/low',vt] = t
        df.loc['#points',vt] = len(x)
        df.loc['threshold',vt] = th
        df.loc['k',vt] = np.round(par[0],3)
        df.loc['p-val',vt] = pval[0]
        df.loc['rmse',vt] = np.round(rmse,3)
    
    return df

# function to plot all fitting results in one figure
def plot_all_fits(vt_l, th_l, year_start,year_end):
    cl = cm.cool(np.linspace(0, 1, 6))
    ch = cm.summer(np.linspace(0, 1, 4))
    c = np.concatenate([cl,ch])
        
    fol='/home/vanoorschot/work/fransje/scripts/LAI_FCOVER/fittings/fitting_1km/final'
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    x = np.linspace(0,8,100)
    y = 1 - np.exp(-0.5*x)
    ax.plot(x,y,'k:', label='k=0.5')

    for i in range(len(vt_l)):
        l = veg_types(vt_l[i])
        vt = vt_l[i]
        th = th_l[i]
        x_fitted=np.load(f'{fol}/output/fitting_2000/x_fitted_{year_start}_{year_end}_{vt}_{th}_k.npy')
        y_fitted=np.load(f'{fol}/output/fitting_2000/y_fitted_{year_start}_{year_end}_{vt}_{th}_k.npy')
        par= np.load(f'{fol}/output/fitting_2000/par_{year_start}_{year_end}_{vt}_{th}_k.npy')

        if (vt==3) or (vt==4) or (vt==5) or (vt==6):
            ax.plot(x_fitted,y_fitted,color=c[i],linestyle='--',label=f'{l},k={np.round(par[0],3)}')
        else:
            ax.plot(x_fitted,y_fitted,color=c[i],linestyle='-',label=f'{l},k={np.round(par[0],3)}')
    ax.legend(fontsize=10)
    ax.set_xlim(0,7)
    ax.set_ylim(0,1)
    ax.tick_params(labelsize=15)
    ax.set_xlabel('LAI [-]',size=15)
    ax.set_ylabel('FCOVER [-]',size=15)
    # ax.set_title(f'Fitting results {year_start}', fontsize=15)
    ax.set_title(f'Fitting results {year_start} {year_end}', fontsize=15)
    fig.savefig(f'{fol}/output/figures_2000/fit_allvegtypes_{year_start}_{year_end}.jpg',dpi=300,bbox_inches='tight')


#%% function to be fitted
def lb_fit(x,k):
    cv =  (1 - np.exp(-k * x))
    return(cv)

def lambert_beer(k,LAI):
    cv =  (1 - np.exp(-k * LAI))
    return(cv)



#%% try regression using nls class
class NLS:
    ''' This provides a wrapper for scipy.optimize.leastsq to get the relevant output for nonlinear least squares. Although scipy provides curve_fit for that reason, curve_fit only returns parameter estimates and covariances. This wrapper returns numerous statistics and diagnostics'''

    import numpy as np
    import scipy.stats as spst
    from scipy.optimize import leastsq

    def __init__(self, func, p0, xdata, ydata):
        # Check the data
        if len(xdata) != len(ydata):
            msg = 'The number of observations does not match the number of rows for the predictors'
            raise ValueError(msg)

        # Check parameter estimates
        if type(p0) != dict:
            msg = "Initial parameter estimates (p0) must be a dictionry of form p0={'a':1, 'b':2, etc}"
            raise ValueError(msg)

        self.func = func
        # self.inits = p0.values()
        self.inits = list(p0.values()) #changed to get list
        self.xdata = xdata
        self.ydata = ydata
        self.nobs = len( ydata )
        self.nparm= len( self.inits )

        # self.parmNames = p0.keys()
        self.parmNames = list(p0.keys()) #changed to get list

        for i in range( len(self.parmNames) ):
            if len(self.parmNames[i]) > 5:
                self.parmNames[i] = self.parmNames[i][0:4]

        # Run the model
        self.mod1 = leastsq(self.func, self.inits, args = (self.xdata, self.ydata), full_output=1)

        # Get the parameters
        self.parmEsts = np.round( self.mod1[0], 4 ) #optimal parameter estimates for a and b

        # Get the Error variance and standard deviation
        self.RSS = np.sum( self.mod1[2]['fvec']**2 ) #take the sum of the squared residuals (Residual Sum of Squares)
        self.df = self.nobs - self.nparm #the size of observations(nobs) minus the amount of parameters
        self.MSE = self.RSS / self.df #Mean Squared Error
        self.RMSE = np.sqrt( self.MSE ) #Root Mean Squared Error

        # Get the covariance matrix
        #cov_x is the inverse of the Hessian
        #to obtain the covariance matrix of the parameters x cov_x must be multiplied by the variance of the residuals
        #mod1[1] is the covariance-variance matrix with main diagonal variances of par a and b
        #when you multiply this with MSE you get cov????
        self.cov = self.MSE * self.mod1[1] 

        # Get parameter standard errors
        self.parmSE = np.diag( np.sqrt( self.cov ) ) #sqrt of cov diagonals (variances) is standard deviation of parameter estimates

        # Calculate the t-values
        self.tvals = self.parmEsts/self.parmSE #tvalue is coefficient divided by standard error (standardize)

        # Get p-values
        self.pvals = (1 - spst.t.cdf( np.abs(self.tvals), self.df))*2 #why calculated like this???

        # Get biased variance (MLE) and calculate log-likehood
        # likelihood is a measure of the goodness of fit? how calculated?
        self.s2b = self.RSS / self.nobs
        self.logLik = -self.nobs/2 * np.log(2*np.pi) - self.nobs/2 * np.log(self.s2b) - 1/(2*self.s2b) * self.RSS

        del(self.mod1)
        del(self.s2b)
        del(self.inits)

    # Get AIC. Add 1 to the df to account for estimation of standard error
    # Akaike Information Criterion: how well does model fit data it was generated from?
    def AIC(self, k=2):
        return -2*self.logLik + k*(self.nparm + 1)

    del(np)
    del(leastsq)

    # Print the summary
    def summary(self):
        print ()
        print ('Non-linear least squares')
        # print ('Model: ' + self.func.func_name)
        print ('Parameters:')
        print (" Estimate Std. Error t-value P(>|t|)")
        for i in range( len(self.parmNames) ):
                print ("% -5s % 5.4f % 5.4f % 5.4f % 5.4f" % tuple( [self.parmNames[i], self.parmEsts[i], self.parmSE[i], self.tvals[i], self.pvals[i]] ))
        print ()
        print ('Residual Standard Error: % 5.4f' % self.RMSE)
        print ('Df: %i' % self.df)



def fitting(lai_array,fc_array):
        # prepare data for fitting
    params = [0.5] # initial guess alpha=0.5, k=0.5   
    lai = lai_array
    yObs = fc_array
        
        # First, define the likelihood null model
    def nullMod(params, lai, yObs):   
        a = params[0]
        yHat = (1-np.exp(-a*lai))
        err = yObs - yHat
        return(err)
        
    p0 = {'a':0.5}
        
    x = lai_array
    y = fc_array
    
    # run nls function
    tMod = NLS(nullMod, p0, x, y) #NLS(func,p0,xdata,ydata)
        
    # get parm estimates
    par = tMod.parmEsts
    k_opt = par[0]
        
    # get pvalues of parm estimates
    pval = tMod.pvals
    k_pval = pval[0]

    # get rmse
    rmse = tMod.RMSE

    # calculate curve with fitted parameters
    lai_x = np.arange(0,7.2,0.2)
    cv_opt = np.zeros(len(lai_x))
    for j in range(len(lai_x)):
        cv_opt[j] = lambert_beer(k_opt,lai_x[j])
    
    return(x,y,lai_x,cv_opt,pval,par,rmse)

def plotting(vt,th,x,y,x_fitted,y_fitted,pval,par,year_start,year_end):
    fol='/home/vanoorschot/work/fransje/scripts/LAI_FCOVER/fittings/fitting_1km/final'
    xdata = x
    ydata = y
    xline = x_fitted
    yline = y_fitted
    k_pval = pval[0]
    k_opt = par[0]
    
    l = veg_types(vt)

    ar = np.arange(0,len(xdata),1)
    idx = np.random.choice(ar,5000, replace=False)
    xd = xdata[idx]
    yd = ydata[idx]
    # xd = xdata
    # yd = ydata

    # calculate point density
    xy = np.vstack([xd,yd])
    z = gaussian_kde(xy)(xy)

    # plot observations and fitted line
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.scatter(xd,yd,c=z,s=10)
    ax.plot(xline,yline,'-r',label='k='+str(np.round(k_opt,3)))
    ax.legend(fontsize=15)
    ax.set_xlim(0,7)
    ax.set_ylim(0,1)
    ax.tick_params(labelsize=15)
    ax.set_xlabel('LAI [-]',size=15)
    ax.set_ylabel('FCOVER [-]',size=15)
    ax.set_title(f'{l}', fontsize=17)
    # fig.savefig(f'{fol}/output/figures_2000/fit_{year_start}_{year_end}_{vt}_{th}_k.jpg',bbox_inches='tight')
    fig.savefig(f'/home/vanoorschot/work/fransje/scripts/HTESSEL/figures/paper_figures/fitting/fit_{year_start}_{year_end}_{vt}_{th}_k.jpg',dpi=300,bbox_inches='tight')
                 
                 

    