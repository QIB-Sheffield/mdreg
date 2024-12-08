"""
===============================================
3D Variable Flip Angle (Linear)
===============================================

This example illustrates motion correction of a 3D time series with 
variable flip angles (VFA). The motion correction is performed with 3D 
coregistration and using a linear signal model fit.

"""

#%% 
# Import packages and load data
# -----------------------------

import numpy as np
import mdreg

# Example data included in mdreg
data = mdreg.fetch('VFA')

# Variables used in this examples
array = data['array']       # 4D signal data (x, y, z, FA)
FA = data['FA']             # The FA values in degrees
spacing = data['spacing']   # (x,y,z) voxel size in mm.

#%%
# Signal model
# ------------
# The signal data are acquired using a spoiled gradient-echo 
# sequence in the steady-state, with different flip angles:
#
# :math:`S(\phi)=S_0\sin{\phi} \frac{1-e^{-T_R/T_1}}{1-\cos{\phi}\,e^{-T_R/T_1}}`
#
# Here :math:`S(\phi)` is the signal at flip angle :math:`\phi`, 
# :math:`S_0` a scaling factor, :math:`T_R` the repetition time and 
# :math:`T_1` the longitudinal relaxation time. The equation can be rewritten 
# in a linear form: 
# 
# :math:`Y(\phi) = AX(\phi)+B` 
# 
# with the variables defined as:
#
# :math:`X=S(\phi)/\sin{\phi};~~~~Y=S(\phi)\cos{\phi} / \sin{\phi}`
#
# and the constants:
#
# :math:`E=e^{-T_R/T_1};~~~~A=\frac{1}{E};~~~~B=-S_0\frac{1-E}{E}~`
#
# Plotting :math:`Y(\phi)` against :math:`X(\phi)` produces a straight line 
# with slope :math:`A` and intercept :math:`B`. After solving for :math:`A, B` 
# these constants can then be used reconstruct the signal:
#
# :math:`S(\phi)=\frac{B\sin{\phi}}{\cos{\phi}-A}`

#%%
# Perform motion correction
# -------------------------
# The signal model above is included in `mdreg` as the function 
# `mdreg.spgr_vfa_lin`, which require the flip angle (FA) values in degrees as 
# input:

vfa_fit = {
    'func': mdreg.spgr_vfa_lin,     # VFA signal model
    'FA': FA,                       # Flip angle in degress  
}

#%%
# For this example we will use a relatively coarse deformation field with 
# grid spacing 50mm:

coreg_params = {
    'spacing': spacing,
    'FinalGridSpacingInPhysicalUnits': 50.0,
}

#%% 
# We can now perform the motion correction:

coreg, defo, fit, pars = mdreg.fit(
    array,                          # Signal data to correct
    fit_image = vfa_fit,            # Signal model
    fit_coreg = coreg_params,       # Coregistration model
    maxit = 5,                      # Maximum number of iteration
)

#%% 
# Visualize the results
# ---------------------
# We visualise the original data and results of the computation using the 
# builtin `mdreg.animation` function. Since we want to call this 3 times, 
# we define the settings up front:

plot_settings = {
    'interval' : 500,                   # Time between animation frames in ms
    'vmin' : 0,                         # Minimum value of the colorbar
    'vmax' : np.percentile(array,99),   # Maximum value of the colorbar
    'show' : True,                      # Display the animation on screen
}

#%% 
# Now we can plot the data, coregistered images and model fits separately:

#%%
anim = mdreg.animation(array, title='Original data', **plot_settings)

#%%
anim = mdreg.animation(coreg, title='Motion corrected', **plot_settings)

#%%
anim = mdreg.animation(fit, title='Model fit', **plot_settings)

#%% 
# It's also instructive to show the deformation field and check whether 
# deformations are consistent with the effect of breathing motion. Since the 
# deformation field is a vector we show here its norm:

#%%

# Get the norm of the deformation field and adjust the plot settings
defo = mdreg.defo_norm(defo)
plot_settings['vmax'] = np.percentile(defo, 99)

# Display the norm of the deformation field
anim = mdreg.animation(defo, title='Deformation field', **plot_settings)

# sphinx_gallery_start_ignore

# Choose the last image as a thumbnail for the gallery
# sphinx_gallery_thumbnail_number = -1

# sphinx_gallery_end_ignore