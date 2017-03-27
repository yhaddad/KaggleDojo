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

class costum_steps(object):
    def __init__(self, stepsize=0.01):
        self.stepsize = stepsize
    def __call__(self, x):
        s = self.stepsize
        x += np.random.norm(-s, s, x.shape)
        return 1.0/(1+np.exp(-x))

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
        nb_,_  = np.histogram(self.X[self.y==0],bins=x, range=self.range,
                              weights=self.sample_weights[self.y==0])
        ns_,_  = np.histogram(self.X[self.y==1],bins=x, range=self.range,
                              weights=self.sample_weights[self.y==1])
        if nb_.shape !=  ns_.shape :
            return 0
        else:
            return self._fom_(ns_, nb_)
    def binned_score_density(self,x):
        ns_  = self.sample_weights[self.y==1].sum()
        ns_ *= np.array([self.pdf_s.integrate_box_1d(x[i], x[i+1]) for i in range(x.shape[0]-1)])
        nb_  = self.sample_weights[self.y==0].sum()
        nb_ *= np.array([self.pdf_b.integrate_box_1d(x[i], x[i+1]) for i in range(x.shape[0]-1)])
        if nb_.shape !=  ns_.shape :
            return 0
        else:
            return self._fom_(ns_, nb_)
    def cost_fun(self,x, lower_bound=None, upper_bound=None ):
        z  = None
        x = np.sort(x)
        if upper_bound is not None:
            x[-1] = upper_bound
        if lower_bound is not None:
            x[ 0] = lower_bound
        if self.use_kde_density:
            z  = self.binned_score_density(x)
        else:
            z  = self.binned_score(x)

        self.scan['bounds'].append(np.sort(x))
        self.scan['cost'  ].append(z)

        if self.drop_last_bin :
            return -np.sqrt((z[1:]**2).sum())
        else :
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
        print _bounds_
        print x_init
        def print_fun(x, f, accepted):
            print("at minimum %.4f accepted %d" % (f, int(accepted)))

        if self.use_kde_density:
            self.pdf_s = kde.gaussian_kde(self.X[self.y==1], weights=self.sample_weights[self.y==1])
            self.pdf_b = kde.gaussian_kde(self.X[self.y==0], weights=self.sample_weights[self.y==0])
            self.result = optimize.minimize(self.cost_fun,x_init,
                                            args  = (min(self.range),max(self.range)),
                                            bounds=_bounds_,
                                            method='Nelder-Mead')
            self.result.x = np.sort(self.result.x)
        else:
            # min_args = {"method": "BFGS"}
            min_args = {"method": "BFGS", "args": (self.range[1],)}
            bound_max   = np.array([ max(self.range) for i in range(self.nbins + 1)])
            bound_min   = np.array([ min(self.range) for i in range(self.nbins + 1)])
            _bounds_ = costum_bounds(bound_max, bound_min)
            self.result = optimize.basinhopping(self.cost_fun, x_init,
                                                args  = (min(self.range),max(self.range)),
                                                minimizer_kwargs=min_args,
                                                accept_test=_bounds_,
                                                # take_step=costum_steps,
                                                #callback=print_fun,
                                                niter=500)
            self.result.x = np.sort(self.result.x)
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
        ax2.plot(range(np.array(self.scan['bounds']).shape[0]),np.sort(-np.array(self.scan['cost'])) )
        ax2.set_ylabel('cost function')
        ax2.set_xlabel('optimisation steps')
        xticklabels = ax1.get_xticklabels()
        plt.setp(xticklabels, visible=False)
        #return fig

    def parameter_scan_2d(self, label='parameter_scan'):
        if self.nbins <= 2 : return None
        tx = np.arange(self.range[0], self.range[1], 0.01)
        ty = np.arange(self.range[0], self.range[1], 0.01)

        xx,yy = np.meshgrid(tx,ty)
        fig = plt.figure(figsize=(self.nbins*2,self.nbins*2))
        plt.subplots_adjust(hspace=0.1)
        plt.subplots_adjust(wspace=0.1)
        xticklabels = []
        yticklabels = []

        for i in range(1,self.nbins):
            for j in range(1,self.nbins):
                if i < j : continue
                ax_ = plt.subplot2grid((self.nbins-1,self.nbins-1),(i-1,j-1))
                print 'parameter scan : ', i, j
                if i != j :
                    # def _fun_(x,y):
                    #     _param_ = [self.result.x[k] for k in range(self.nbins+1)]
                    #     _param_[i] = x
                    #     _param_[j] = y
                    #     _param_[ 0] = self.range[0]
                    #     _param_[-1] = self.range[1]
                    #     return self.cost_fun(np.array(_param_))
                    # vec_fun_ = np.vectorize(_fun_)
                    # zz     = vec_fun_(xx,yy)
                    # levels = np.linspace(zz.min(),0.95*zz.min(),5)
                    # plt.contourf   (xx, yy, zz, np.linspace(zz.min(),0.85*zz.min(),20),
                    #                 cmap=plt.cm.Spectral_r)
                    # C = ax_.contour(xx, yy, zz, levels, linewidth=0.1,colors='black')
                    # ax_.clabel(C, inline=1, fontsize=5)

                    ax_.plot(self.result.x[j], self.result.x[i], 'ro', label = 'best fit')
                    ax_.set_xlim(self.range)
                    ax_.set_ylim(self.range)
                    if j == self.nbins-1:
                        ax_.set_xlabel('$x_{%i}$'%j)
                        ax_.set_xticks([])
                        # plt.setp(ax_.get_xticklabels(), visible=False)
                    # else : ax_.set_xticks([])
                    if i == 1:
                        ax_.set_ylabel('$x_{%i}$'%j)
                        # ax.set_xticks([])
                        # plt.setp(ax_.get_yticklabels(), visible=False)
                    # else : ax_.set_yticks([])
                else:
                    # def _fun_1d(x):
                    #     _param_ = [self.result.x[k] for k in range(self.nbins+1)]
                    #     _param_[i] = x
                    #     _param_[ 0] = self.range[0]
                    #     _param_[-1] = self.range[1]
                    #     return self.cost_fun(np.array(_param_))
                    # vec_fun_1d = np.vectorize(_fun_1d)
                    # z1d = vec_fun_1d(tx)
                    # ax_.plot(tx, z1d, 'r-', lw=1.5)
                    # ax_.set_ylim([z1d.min(),0.90*z1d.min()])
                    ax_.set_xlim(self.range)
                    ax_.axvline(x=self.result.x[i])
                    ax_.set_xlabel('$x_{%i}$'%i)
                    if i != 1 : ax_.yaxis.tick_right()
        #return fig
    def covariance_matrix(self):
        pass
