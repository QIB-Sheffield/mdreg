"""
===============================================
Getting started
===============================================

Model-driven registration is a method to remove motion from a series of 2D or 
3D images. It applies specifically to situations where a model exists that can 
describe the changes in signal intensity through the series. 

The default engine for coregistration in ``mdreg`` is the package 
``itk-elastix``, but coregistration models from other packages are integrated 
as well (``scikit-image``, ``dipy``). 

For modelling, a number of generic models are built-in, but ``mdreg`` includes 
an interface for integrating custom-built models, or models from any other 
package. 

This guide illustrates these different types of usage with an example use case, 
that of fitting the longitudinal MRI relaxation time T1 in the abdomen from a 
Look-Locker MRI sequence. 

"""

#%%
# Import packages and data
# ----------------------------
# Let's start by importing the packages needed in this tutorial. 
#

import numpy as np
import mdreg

#%%
# Fetch the data
#`mdreg` includes a number of test data sets for demonstration purposes. Let's 
# fetch the MOLLI example and use `mdreg`'s built-in plotting tools to 
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
# The breathing motion is clearly visible in this slice. Let's use ``mdreg`` to 
# remove it. As a starting point, we try ``mdreg`` with default settings.

# Perform model-driven coregistration with default settings
coreg, defo, fit, pars = mdreg.fit(array, verbose=0)

# # Visualise the results
mdreg.plot_series(array, fit, coreg, vmin=0, vmax=1e4, show=True) 

#%%
# Changing the signal model
# -------------------------
# The default model is a constant, so the model fit (left) does not show any 
# changes. The coregistered image (right) clearly shows the deformations, but 
# they do not have the desired effect. This is not unexpected, because a
# constant model does not provide a good approximation to the changes in image 
# contrast.
#
# In this case we are lucky -- the appropriate model for a MOLLI sequence is 
# `abs_exp_recovery_2p` and is included in ``mdreg``'s model library. We just 
# need to tell ``mdreg`` which fitting function to use (``'func'``), and 
# provide the keyword arguments required by the model. For this model these are 
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
coreg, defo, fit, pars = mdreg.fit(array, fit_image=molli, verbose=0)

# # Visualise the results
mdreg.plot_series(array, fit, coreg, vmin=0, vmax=1e4, show=True)

#%%
# This now shows essentially the desired result. The model fit (left) and the 
# deformed image (right) are now both very similar in image contrast to the 
# original data (middle), but with motion removed. 

#%%
# Using custom-built models
# --------------------------
#
# In some cases the required model may not be available in the ``mdreg`` 
# library, in which case it needs to be custom built. 
#
# We illustrate this idea using the T1-MOLLI model from the ``ukat`` library. 
# All that is needed in order to use it inside ``mdreg`` is to wrap it into a 
# function that takes the image array as argument, and returns the fit to the 
# model and the fitted model parameters. 

from ukat.mapping.t1 import T1

def ukat_t1_model(array, TI=None, **kwargs):
    map = T1(np.abs(array), TI, np.eye(4), parameters=2, multithread=False)
    return map.get_fit_signal(), (map.t1_map, map.m0_map)

#%%
#
# We can now use this custom model in the same way as built-in models when we 
# run ``mdreg``:

# Define the fit function and its arguments
ukat_model = {
    'func': ukat_t1_model,
    'TI': np.array(data['TI']),
}

# Perform model-driven coregistration
coreg, defo, fit, pars = mdreg.fit(array, fit_image=ukat_model)

# Visualise the results
mdreg.plot_series(array, fit, coreg, vmin=0, vmax=1e4, show=True)

#%%
# As expected, the result is the same as before using the built-in model 
# `abs_exp_recovery_2p`


#%%
# Pixel-by-pixel fitting
#----------------------
#
# In cases where the model is not available in any existing package, or the 
# user is not prepared to import an existing package, the fit function must be 
# written from scratch. In general, ``mdreg`` only requires that it has the 
# same interface as the `ukat_t1_model` defined above: one argument (the image 
# array), any number of keyword arguments, and two return values (the model fit 
# and the fit parameters).
#
# `mdreg` offers a convenient shortcut for the common scenario where a 1D 
# signal model is applied for each pixel independently (*pixel-by-pixel 
# fitting*). All that is needed is to define the 1D model explicitly. 
#

def my_pixel_model(xdata, S0, T1, **kwargs):
    return np.abs(S0 * (1 - 2 * np.exp(-xdata/T1)))

#%%
# Optionally, one may also provide a function that derives initial values from 
# the data and any constant initial values provided by the user.
#

def my_pixel_model_init(xdata, ydata, p0):
    S0 = np.amax(np.abs(ydata))
    return [S0*p0[0], p0[1]]

#%%
#
# With these definitions in hand, a pixel model fit can be defined as a 
# dictionary specifying the model, its parameters (xdata), and any optional 
# arguments.


my_pixel_fit = {

    # The custom-built single pixel model
    'model': my_pixel_model,

    # xdata for the single-pixel model
    'xdata': np.array(data['TI'])/1000,

    # Optional: custom-built initialization function
    'func_init': my_pixel_model_init,

    # Optional: initial values for the free parameters
    'p0': [1,1.3], 

    # Optional: bounds for the free model parameters
    'bounds': (0, np.inf),

    # Optional: any keyword arguments accepted by scipy's curve_fit
    'xtol': 1e-3,
}   

#%%
#And this can be provided directly to `mdreg.fit` via the keyword argument 
# ``fit_pixel`` - instructing ``mdreg`` to perform pixel-based fitting using 
# the parameters defined in ``my_pixel_fit``.

# Perform model-driven coregistration with a custom pixel model
coreg, defo, fit, pars = mdreg.fit(array, fit_pixel=my_pixel_fit, verbose=0)

# Visualise the results
mdreg.plot_series(array, fit, coreg, vmin=0, vmax=1e4, show=True)

#%%
#
#As expected, the result is the same as before using the built-in model 
# `abs_exp_recovery_2p` and the ukat implementation `ukat_t1_model`. **TODO: 
# currently NOT the case - fix bug in ukat solution**

#%%
#Customizing the coregistration
#==============================
#
#In the above examples we have not provided any detail on the coregistration 
# itself, which means that the default in ``mdreg`` has been applied. This is 
# the standard b-spline coregistration of elastix, but modified to use a 
# least-squares metric rather than mutual information. The detailed default 
# parameter settings can be found in the function `mdreg.elastix.params`.
#
#We can try to improve the result further by customizing the coregistration 
# model rather than using the default. This can be done either by modifying the
# ``elastix`` parameters, or by using another coregistration package supported 
# by ``mdreg`` (currently only ``skimage`` available).
#
# Customizing elastix coregistration
#----------------------------------
#
# To illustrate customizing the ``elastix`` parameters, we perform ``mdreg`` 
# with a more fine-grained deformation field. The default coregistration uses 
# a grid spacing of 5cm, which is relatively coarse, so we will try a finer 
# deformation of 5mm. In order to do that, we need to provide the actual pixel 
# spacing of the data, and modify default elastix parameters.

deform5mm = {

    # Pixel spacing in the images
    'spacing': data['pixel_spacing'],

    # Default elastix parameters with custom grid spacing
    'params': mdreg.elastix.params(FinalGridSpacingInPhysicalUnits= "5.0"),
}

#%%
# We run ``mdreg`` again with the correct signal model, but now using the 5mm 
# coregistration:

# Perform model-driven coregistration
coreg, defo, fit, pars = mdreg.fit(array, fit_image=molli, fit_coreg=deform5mm)

# Visualise the results
mdreg.plot_series(array, fit, coreg, vmin=0, vmax=1e4, show=True)

#%%
# The effect of the finer deformations is apparent, but it has not 
# fundamentally improved the result. In fact, it has created smaller unphysical
# deformations that have blurred out some of the anatomical features. An 
# example is the small cyst in the upper pole of the right kidney, which is 
# clearly visible in the data but can no longer be seen in the model fit. The 
# example illustrates that the grid spacing is a critical parameter and should 
# be chosen to reflect the scale of the expected deformations. 
#
# Any coregistration method available in elastix can be used in the same way by
# providing a custom set of elstix parameters.

#%%
#Coregistration with ``skimage``
#-------------------------------
#
#While `itk-elastix` is the default package for coregistration, ``mdreg`` also 
# has an option to use coregistration modules from the package `scikit-image`. 
#
#For this illustration we run skimage coregistration with default parameters, 
# except for the attachment which is increased to 30 (default=15) to allow for 
# finer deformations.
#

attach30 = {

    # The package needs to be defined if it is not elastix
    'package': 'skimage',

    # Use default parameters except for the attachment
    'params': mdreg.skimage.params(attachment=30)
}

#%%
#
# Run ``mdreg`` again with the correct signal model, but now using the 
# customized `skimage` coregistration:


# Perform model-driven coregistration
coreg, defo, fit, pars = mdreg.fit(array, fit_image=molli, fit_coreg=attach30)

# Visualise the results
mdreg.plot_series(array, fit, coreg, vmin=0, vmax=1e4, show=True)

#%%
# This result shows good coregistration results, in this case better preserving 
# fine grain features such as kidney cysts in comparison to the default elastix 
# implementation.

# sphinx_gallery_start_ignore

# Choose the last image as a thumbnail for the gallery
# sphinx_gallery_thumbnail_number = -1

# sphinx_gallery_end_ignore