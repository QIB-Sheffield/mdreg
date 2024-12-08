"""
===============================================
Customizing the coregistration
===============================================

By default, ``dcmri`` uses free-form deformation implemented in the package 
`itk.elastix`, which default settings for all configuration parameters. 

This example shows how thse defaults can be modified, and how coregistration 
can be done by another package, `skimage`.

"""

#%%
# Import packages and data
# ----------------------------
# Import packages 
import numpy as np
import mdreg

#%%
# Load test data
data = mdreg.fetch('MOLLI')
array = data['array'][:,:,0,:]

# Throughout this example we use the same signal model:
molli = {
    'func': mdreg.abs_exp_recovery_2p,
    'TI': np.array(data['TI'])/1000,
}

# Visualise the data
mdreg.animation(array, vmin=0, vmax=1e4, show=True)

#%%
# Customizing the ``elastix`` coregistration
# ------------------------------------------
#
# By default `~mdreg.fit` performs image coregistration using the package 
# ``elastix`` with bspline deformations, and default settings for all 
# parameters. If the result is suboptimal, one way to improve is to customize 
# the coregistration model in elastix.
#
# A critical parameter in elastix bspline deformation is the grid spacing, 
# which determines the level of detail in the deformation field. The default 
# coregistration uses a grid spacing of 16mm. Let's try what happens if we 
# allow finer deformations of 5mm. In order to do that, we need to 
# provide the actual pixel spacing of the data, and modify the default elastix 
# parameter:

deform5mm = {

    # Pixel spacing in the images (in mm)
    'spacing': data['pixel_spacing'],

    # Provide custom grid spacing (in mm)
    'FinalGridSpacingInPhysicalUnits': 5.0,
}

#%%
# We run ``mdreg`` using the 5mm coregistration:

coreg, defo, fit, pars = mdreg.fit(
    array, fit_image=molli, fit_coreg=deform5mm)

# Visualise the results
mdreg.plot_series(array, fit, coreg, vmin=0, vmax=1e4, show=True)


#%%
# Compared to results in other examples using the default grid spacing of 16mm, 
# this shows smaller deformations in the upper pole of the right kidney that 
# are not present in the original moving images. The example 
# illustrates that the grid spacing should be chosen carefully to reflect the 
# scale of the expected deformations. 
#
# Any coregistration method available in elastix can be applied in the same 
# way by providing a custom set of elastix parameters.

# %%
# Coregistration with ``skimage``
# -------------------------------
#
# While ``elastix`` is the default package for coregistration, ``mdreg`` also 
# has an option to use the optical flow method 
# :func:`~skimage.optical_flow_tvl1` from the package ``skimage``. 
#
# For this illustration we run ``skimage`` coregistration with default parameters, 
# except for the attachment which is increased to 30 (default=15) to allow for 
# finer deformations:

coreg_skimg = {

    # The package needs to be defined if it is not elastix
    'package': 'skimage',

    # Provide a custom attachment value
    'attachment': 30,
}

#%%
# Run ``mdreg`` again, now using the ``skimage`` coregistration instead of 
# elastix:

# Perform model-driven coregistration
coreg, defo, fit, pars = mdreg.fit(
    array, fit_image=molli, fit_coreg=coreg_skimg)

# Visualise the results
mdreg.plot_series(array, fit, coreg, vmin=0, vmax=1e4, show=True)

#%%
# This result shows good coregistration results, nicely preserving even
# fine grained features such as the small kidney cysts.

# sphinx_gallery_start_ignore

# Choose the last image as a thumbnail for the gallery
# sphinx_gallery_thumbnail_number = -1

# sphinx_gallery_end_ignore