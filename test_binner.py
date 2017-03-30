import numpy as np
import pandas as pd
np.set_printoptions(precision=4)

import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML

from tools import kde

s_raw = np.random.beta(8,3, 8000)
b_raw = np.random.beta(3,3, 8000)

weight =np.concatenate((np.random.binomial(10,0.5, 8000)/100.0,np.random.binomial(10,0.5, 8000)/1.0))
y =np.concatenate((np.ones(s_raw.shape[0]),np.zeros(b_raw.shape[0])))

data = np.concatenate((s_raw,b_raw))

fig = plt.figure(figsize=(5,5))

pdf_s = kde.gaussian_kde(data[y==1])
pdf_b = kde.gaussian_kde(data[y==0])

x  = np.linspace(0,1, 200)
pss = pdf_s(x)
pbb = pdf_b(x)

bb,_,_ = plt.hist(data[y==0], bins=100, range=[0,1],weights=weight[y==0],
                  histtype='stepfilled', alpha=0.5, normed=True)
ss,_,_ = plt.hist(data[y==1], bins=100, range=[0,1],weights=weight[y==1],
                  histtype='stepfilled', alpha=0.5, normed=True)

plt.plot(x, pss, label='kde-s')
plt.plot(x, pbb, label='kde-b')

plt.plot(x, kde.gaussian_kde(data[y==1], weights=weight[y==1])(x), label='kde-s')
plt.plot(x, kde.gaussian_kde(data[y==0], weights=weight[y==0])(x), label='kde-b')

plt.savefig('input_data.pdf')
plt.savefig('input_data.png')
print " ----------------------------- "

from tools import binopt as bo

ncat    = 6
binner  = bo.zbinner(ncat,[0,1], use_kde_density = False)
results = binner.fit(data, y, sample_weights = weight)

print results

fig = plt.figure(figsize=(5,4))
plt.hist(data[y==0], bins=100, range=[0,1],weights=weight[y==0], histtype='stepfilled', alpha=0.5,normed=True)
plt.hist(data[y==1], bins=100, range=[0,1],weights=weight[y==1], histtype='stepfilled', alpha=0.5,normed=True)
for b in np.sort(results.x):
    plt.axvline(x=b, linewidth=2.0, color='red',lw=0.8)
for b in np.sort(np.linspace (0,1,ncat+1)):
    plt.axvline(x=b, linewidth=2.0, color='blue',lw=0.8, ls='--')

plt.savefig('binning_results.pdf')
plt.savefig('binning_results.png')


fig = plt.figure(figsize=(10,4))
binner.optimisation_monitoring_(fig).savefig('test.pdf')

binner.parameter_scan_2d(label='parameter_scan_SA')

print " ----------------------------- "
