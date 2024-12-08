"""
===============================================
Using built-in models
===============================================

We illustrate the basic use of ``mdreg`` for the use case of fitting the 
longitudinal MRI relaxation time T1 from a Look-Locker MRI 
sequence. 
"""

#%%
# Import packages and data
# ------------------------
# Let's start by importing the packages needed in this tutorial. 

import numpy as np
import mdreg

#%%
# ``mdreg`` includes a number of test data sets for demonstration purposes. 
# Let's fetch the MOLLI example and use ``mdreg``'s built-in plotting tools to 
# visualise the motion:

# fetch the data
data = mdreg.fetch('MOLLI')

# We will consider the slice z=0 of the data array:
array = data['array'][:,:,0,:]

# Use the built-in animation function of mdreg to visualise the motion:
mdreg.animation(array, vmin=0, vmax=1e4, show=True)

#%%
# Using built-in models
# =====================
#
# The breathing motion is clearly visible in this slice and we can use 
# ``mdreg`` to remove it. As a starting point, we could try ``mdreg`` with 
# default settings.

# Perform model-driven coregistration with default settings
coreg, defo, fit, pars = mdreg.fit(array)

# And visualise the results
mdreg.plot_series(array, fit, coreg, vmin=0, vmax=1e4, show=True) 

# %%
# The default model is a constant, so the model fit (left) does not show any 
# changes. The coregistered image (right) clearly shows the deformations, but 
# they do not have the desired effect of removing the motion. This is not 
# unexpected, because a constant model does not provide a good approximation 
# to the changes in image contrast. We need a dedicated model for this 
# sequence.

#%%
# Changing the signal model
# -------------------------
# he appropriate model for a MOLLI sequence is 
# `~mdreg.abs_exp_recovery_2p` and is included in ``mdreg``'s model library. 
# We just 
# need to tell ``mdreg`` which fitting function to use (*func*), and 
# provide the keyword arguments required by the model - in this example 
# the inversion times TI in units of sec. We define the signal model up front 
# so it can be used again later in this script:

molli = {

    # The function to fit the data
    'func': mdreg.abs_exp_recovery_2p,

    # The keyword arguments required by the function
    'TI': np.array(data['TI'])/1000,
}


#%%
#Now we can run ``mdreg`` with this model and check the result again:

# Perform model-driven coregistration
coreg, defo, fit, pars = mdreg.fit(array, fit_image=molli)

# # Visualise the results
mdreg.plot_series(array, fit, coreg, vmin=0, vmax=1e4, show=True)

#%%
# This now shows essentially the desired result. The model fit (left) and the 
# deformed image (right) are now both very similar in image contrast to the 
# original data (middle), but with motion removed. 

# sphinx_gallery_start_ignore

# Choose the last image as a thumbnail for the gallery
# sphinx_gallery_thumbnail_number = -1

# sphinx_gallery_end_ignore