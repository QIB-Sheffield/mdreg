***************
Getting started
***************

Model-driven registration is a method to remove motion from a series of 2D or 
3D images. It applies specifically to situations where a model exists that can 
describe the changes in signal intensity through the series. 

A number of generic models are built-in, but ``mdreg`` includes 
an interface for integrating custom-built models, or models from any other 
package. The default engine for coregistration in ``mdreg`` is the package 
``itk-elastix``, but coregistration models from other packages are integrated 
as well (``scikit-image``, ``dipy``). 

The *getting started* section in :ref:`tutorials <tutorials>` illustrates 
these different types of usage, and is a good place to start if you have not 
used ``mdreg`` before.