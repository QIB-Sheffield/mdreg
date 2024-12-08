***************
Getting started
***************

What is model-driven registration?
----------------------------------

Model-driven registration is a method to remove motion from a series of 2D or 
3D images. It applies specifically to situations where a model exists that can 
describe the changes in signal intensity through the series. 

Background
----------

Many applications in medical imaging involve time series of 2D images or 
3D volumes that are corrupted by subject motion. Examples are T1- or T2- 
mapping in MRI, diffusion-weighted MRI, or dynamic contrast-enhanced imaging 
in MRI or CT.

Motion correction of such data is challenging because the signal changes 
caused by the motion are superposed on the often drastic changes in intrinsic 
image contrast. However in most cases these intrinsic changes in image 
contrast can be described by a known signal model. Indeed many of these 
applications critically depend on the availability of a model to derive 
parametric maps from the signal data.

Model-driven image registration leverages the existence of a signal model to 
remove the confounding effects of changes in image contrast on the results 
of the motion correction. 

Any model-driven registration method therefore requires two ingredients:

- a *signal model* that describes the changes in signal in the 
  absence of motion.
- a *motion model* that describes the changes in signal caused 
  by motion alone.

``mdreg`` has a library of built-in signal models for common scenarios, 
also includes a simple mechanism for integrating custom-build signal models. 
If a model is not specified, ``mdreg`` will assume there are no intrinsic 
changes in intensity and will use a constant signal model.

For motion modelling, ``mdreg`` offers a unified interface to coregistration 
methods from different packages - including ``itk-elastix``, ``scikit-image``
and ``dipy``. If a coregistration method is not specified, ``mdreg`` will 
use *bspline* coregistration fom the package *elastix*

How to use mdreg?
-----------------

The *getting started* section in :ref:`tutorials <tutorials>` illustrates 
different types of usage, and is a good place to start if you have not 
used ``mdreg`` before.

[.. more coming soon..]

