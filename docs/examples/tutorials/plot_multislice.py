"""
===============================================
Multi-slice 2D registration
===============================================

This examples shows how mdreg can be used to perform 2D motion correction 
slice-by-slice on a 4D array. Typical application examples are motion 
correction of multi-slice MRI series such as Look-Locker T1 mapping as in 
this example.
"""

#%%
# Import packages and data
# ----------------------------
# Let's start by importing the packages needed in this tutorial. 

import numpy as np
import mdreg

#%%
# Fetch the multi-slice renal MOLLI dataset:

data = mdreg.fetch('MOLLI')

# Get the relevant variables
array = data['array']  
TI = np.array(data['TI'])/1000

# Use the built-in animation function of mdreg to visualise the motion:
mdreg.animation(array, vmin=0, vmax=1e4, show=True)


#%%
# Perform slice-by-slice motion correction
# ----------------------------------------
# The signal model for the MOLLI is the following:

molli = {
    'func': mdreg.abs_exp_recovery_2p, 
    'TI': TI,
}

#%%
# Slice-by-slice analysis works the same way as single-slice analysis. We 
# just have to remember to set the keyword argument force_2d to True so 
# `~mdreg.fit` knows that we want 2D motion correction. This overrules the 
# default behaviour of fitting 3D data with 3D motion correction:

coreg, defo, fit, pars = mdreg.fit(
    array, fit_image=molli, verbose=2, 
    force_2d=True, xtol=1e-3, maxit=2,
)

#%%
# Visualise the results

anim = mdreg.animation(
    coreg, title='Motion corrected', 
    interval=500, vmin=0, vmax=1e4, show=True,
)

#%%
# Different options per slice
# ---------------------------
# It is not uncommon that each slice in a multislice sequence has different 
# imaging parameters. For instance in a MOLLI sequence such as illustrated 
# in this example, it may well be that each slice has its own set of TI
# values. 
# 
# The situation can be accomodated in `~mdreg.fit` by assigning a 
# list of dictionaries to the fit_image argument - one for each slice. 
# We illustrate that here assuming that the TI's for each slice are offset by 
# 100 ms relative to the previous slice. So now the signal model is a list of 
# dictionaries:

molli = [
    {
        'func': mdreg.abs_exp_recovery_2p, 
        'TI': TI + 0.1 * z, 
    }
for z in range(array.shape[2])]


#%%
# Other than that the call to mdreg is the same as before:

coreg, defo, fit, pars = mdreg.fit(
    array, fit_image=molli, verbose=2, 
    force_2d=True, xtol=1e-3, maxit=2,
)

# In this case the result is poor because the real TI's are in fact the same 
# for each slice, so we have provided a biased model:

anim = mdreg.animation(
    coreg, title='Motion corrected with biased model', 
    interval=500, vmin=0, vmax=1e4, show=True,
)

#%%
# Slice-by-slice with custom models
# ---------------------------------
# If the signal model fit is defined with a custom model via the 
# *fit_pixel* argument, the slice-by-slice operation works the same: 
# set the force_2d keyword to True, and supply the *fit_pixel* 
# argument as a list of dictionaries in the event that each slice needs 
# different parameter settings.

# sphinx_gallery_start_ignore

# Choose the last image as a thumbnail for the gallery
# sphinx_gallery_thumbnail_number = -1

# sphinx_gallery_end_ignore