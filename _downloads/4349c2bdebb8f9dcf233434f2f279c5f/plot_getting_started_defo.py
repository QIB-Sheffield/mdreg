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

# Show the motion:
mdreg.animation(array, vmin=0, vmax=1e4, show=True)

#%%
# Customizing the coregistration
# ------------------------------
#
# In the above examples we have not provided any detail on the coregistration 
# itself, which means that the default in ``mdreg`` has been applied. The 
# detailed default parameter settings can be found in the function 
# ``mdreg.elastix.params``.
#
# We can try to improve the result further by customizing the coregistration 
# model rather than using the default. This can be done either by modifying the
# ``elastix`` parameters, or by using another coregistration package supported 
# by ``mdreg`` (currently only ``skimage`` available).
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

molli = {
    'func': mdreg.abs_exp_recovery_2p,
    'TI': np.array(data['TI'])/1000,
}

# Perform model-driven coregistration
coreg, defo, fit, pars = mdreg.fit(
    array, 
    fit_image = molli, 
    fit_coreg = deform5mm, 
    verbose = 0,
)

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

# %%
# Coregistration with ``skimage``
# -------------------------------
#
# While ``skimage`` is the default package for coregistration, ``mdreg`` also 
# has an option to use coregistration modules from the package ``scikit-image``. 
#
# For this illustration we run skimage coregistration with default parameters, 
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
# customized ``skimage`` coregistration:


# Perform model-driven coregistration
coreg, defo, fit, pars = mdreg.fit(
    array, 
    fit_image = molli, 
    fit_coreg = attach30, 
)

# Visualise the results
mdreg.plot_series(array, fit, coreg, vmin=0, vmax=1e4, show=True)

#%%
# This result shows good coregistration results, in this case better preserving 
# fine grained features such as kidney cysts in comparison to the default 
# elastix implementation.

# sphinx_gallery_start_ignore

# Choose the last image as a thumbnail for the gallery
# sphinx_gallery_thumbnail_number = -1

# sphinx_gallery_end_ignore