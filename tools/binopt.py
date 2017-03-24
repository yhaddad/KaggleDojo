import numpy as np
import pandas as pd
np.set_printoptions(precision=4)

from scipy import optimize
from scipy.integrate import quad
import matplotlib.pyplot as plt

import kde

class binner_base(object):
    """ Abstract class for classification based binning
    """
    def __init__(self, nbins, range):
        self.nbins = nbins
        self.range = range
        raise NotImplementedError('Method or function has not been implemented yet')
    def fit(self, X, y, sample_weights=None):
        raise NotImplementedError('Method or function has not been implemented yet')
        return self
    
class costum_bounds(object):
    def __init__(self, xmax=[1.1,1.1], xmin=[-1.1,-1.1] ):
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)
    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin
    
class zbinner(binner_base):
    def __init__(self,nbins, range, drop_last_bin=True, 
                 fix_upper=True, fix_lower=False, use_kde_density = True):
        self.nbins = nbins
        self.range = range
        self.drop_last_bin = drop_last_bin
        self.X = None
        self.y = None
        self.pdf_s  = None
        self.pdf_b  = None
        self.result = None
        self.fix_upper = fix_upper
        self.fix_lower = fix_lower
        self.breg = 0
        self.use_kde_density = use_kde_density
        self.sample_weights = None
        self.scan = {"bounds": [],"cost": []}
        
    def _fom_(self,s,b,breg=10):
        c = np.zeros(s.shape[0])
        c[(s == 0) & (b==0)] = 0
        c[(s+b)!=0] = s[(s+b)!=0] / np.sqrt((s+b+breg)[(s+b)!=0])
        return c
    def binned_score(self,x):
        x = np.sort(x)
        nb_,_  = np.histogram(self.X[self.y==0],bins=x, range=self.range,
                              weights=self.sample_weights[self.y==0])
        ns_,_  = np.histogram(self.X[self.y==1],bins=x, range=self.range,
                              weights=self.sample_weights[self.y==1])
        if nb_.shape !=  ns_.shape :    
            return 0
        else:
            return self._fom_(ns_, nb_)
    def binned_score_density(self,x):
        x = np.sort(x)
        ns_  = self.sample_weights[self.y==1].sum()
        ns_ *= np.array([self.pdf_s.integrate_box_1d(x[i], x[i+1]) for i in range(x.shape[0]-1)])
        nb_  = self.sample_weights[self.y==0].sum()
        nb_ *= np.array([self.pdf_b.integrate_box_1d(x[i], x[i+1]) for i in range(x.shape[0]-1)])
        if nb_.shape !=  ns_.shape :    
            return 0
        else:
            return self._fom_(ns_, nb_)
    def cost_fun(self,x):
        z  = None
        if self.use_kde_density:
            z  = self.binned_score_density(x)
        else:
            z  = self.binned_score(x)
        self.scan['bounds'].append(np.sort(x))
        self.scan['cost'  ].append(-np.sqrt((z**2).sum()))
        return -np.sqrt((z**2).sum())

    def fit(self, X, y, sample_weights=None):
        self.X = X
        self.y = y 
        if sample_weights is not None:
            self.sample_weights = sample_weights
        else:
            self.sample_weights = np.ones(X.shape[0])
        
        x_init = np.linspace (self.range[0],self.range[1],self.nbins+1)
        np.random.seed(555)
        _bounds_   = np.array([self.range for i in range(self.nbins + 1)])
        
        def print_fun(x, f, accepted):
            print("at minimum %.4f accepted %d" % (f, int(accepted)))

        if self.use_kde_density:
            self.pdf_s = kde.gaussian_kde(self.X[self.y==1], weights=self.sample_weights[self.y==1])
            self.pdf_b = kde.gaussian_kde(self.X[self.y==0], weights=self.sample_weights[self.y==0])
            self.result = optimize.minimize(self.cost_fun,x_init,
                                            bounds=_bounds_,
                                            method='Nelder-Mead')
        else:
            min_args = {"method": "BFGS"}
            _bounds_ = costum_bounds(bound_max, bound_min)
            self.result = optimize.basinhopping(self.cost_fun, x_init, 
                                                minimizer_kwargs=min_args,
                                                accept_test=_bounds_,
                                                #callback=print_fun,
                                                niter=100)
            
        return self.result
    def optimisation_monitoring_(self,fig=None):
        if fig is None :
            fig = plt.figure(figsize=(10,2))
            
        plt.subplots_adjust(hspace=0.001)
        ax1 = plt.subplot(211)
        for ix in range(self.result.x.shape[0]):
            ax1.plot(range(np.array(self.scan['bounds']).shape[0]),
                     np.array(self.scan['bounds'])[:,ix] )
            
        ax1.set_ylabel('bin boundaries')
        ax2 = plt.subplot(212)
        ax2.plot(range(np.array(self.scan['bounds']).shape[0]), np.array(self.scan['cost']))
        ax2.set_ylabel('cost function')
        ax2.set_xlabel('optimisation steps')
        xticklabels = ax1.get_xticklabels()
        plt.setp(xticklabels, visible=False)
        return fig

    def parameter_scan_2d(self, fig=None):
        if self.nbins <= 3 : return None
        tx = np.arange(self.range[0], self.range[1], 0.01)
        ty = np.arange(self.range[0], self.range[1], 0.01)
        
        xx,yy = np.meshgrid(tx,ty)
        for i in range(1,self.nbins):
            for j in range(1,self.nbins):
                if i >= j : continue
                print 'parameter scan : ', i, j
                fig = plt.figure(figsize=(4,4))
                def _fun_(x,y):
                    _param_ = [self.result.x[k] for k in range(self.nbins+1)]
                    _param_[i] = x
                    _param_[j] = y
                    _param_[ 0] = self.range[0]
                    _param_[-1] = self.range[1]
                    return self.cost_fun(np.array(_param_))
                vec_fun_ = np.vectorize(_fun_)
                zz     = vec_fun_(xx,yy)
                levels = np.linspace(zz.min(),0.98*zz.min(),5)
                plt.contour(xx, yy, zz,levels, fontsize=9, inline=1)
                plt.plot(self.result.x[i], self.result.x[j], 'ro', label = 'best fit')
                plt.xlabel('$x_{%i}$'%i)
                plt.ylabel('$x_{%i}$'%j)
                plt.legend(loc='best')
                
                plt.savefig('parameter_scan_%i_%i.png' % (i,j) )
                plt.savefig('parameter_scan_%i_%i.pdf' % (i,j) )
        return fig

    def covariance_matrix(self):
        pass
