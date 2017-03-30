import numpy as np
import pandas as pd
np.set_printoptions(precision=3)

import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML

# plt.style.use('muted')


# In[2]:

s_raw = np.random.beta(8,3, 8000)
b_raw = np.random.beta(3,3, 8000)

weight =np.concatenate((np.random.binomial(10,0.5, 8000)/100.0,np.random.binomial(10,0.5, 8000)/1.0))
y =np.concatenate((np.ones(s_raw.shape[0]),np.zeros(b_raw.shape[0])))

data = np.concatenate((s_raw,b_raw))


# In[3]:

from tools import kde
from scipy.integrate import quad
from tools import binopt as bo

ncat    = 7
binner  = bo.zbinner(ncat,[0,1],use_kde_density = True, drop_last_bin=True)
results = binner.fit(data, y, sample_weights = weight)

print "ncat  : ", ncat
print results

fig = plt.figure(figsize=(10,4))
binner.optimisation_monitoring_(fig).savefig('test.pdf')

binner.parameter_scan_2d(label='parameter_scan')
plt.show()
