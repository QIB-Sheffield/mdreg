.. _coreg_dict:

*****************************
Coregistration options
*****************************

This section provides information about the arguments that can be
passed to `mdreg.fit` to control the coregistration process. These arguments 
are provided as a dictionary referred to as `fit_coreg` in the `mdreg.fit` 
function.

By default, `mdreg` uses the `elastix` package for coregistration, with default
parameters. The options allow the user to control the coregistration package 
used, and the optional parameters which can be controlled within the 
different packages.


Elastix
-------

Elastix is one of the coregistration engines available to perform 
the coregistration components in `mdreg`. For consistent usage  
across different coregistration engines, `mdreg` contains pythonic wrappers 
for the core functionality available in elastix, but the functionality is 
not otherwise modified. 

We refer to the original 
`elastix pages <https://github.com/SuperElastix>`_ 
for more detail on elastix. Please note elastix authors request that the 
following papers are cited if you use the elastix software anywhere:

- S. Klein, M. Staring, K. Murphy, M.A. Viergever, J.P.W. Pluim, 
  "elastix: a toolbox for intensity based medical image registration,
  " IEEE Transactions on Medical Imaging, vol. 29, no. 1, pp. 196 - 205, 
  January 2010. 

- D.P. Shamonin, E.E. Bron, B.P.F. Lelieveldt, M. Smits, S. Klein and M. Staring, 
  "Fast Parallel Image Registration on CPU and GPU for Diagnostic Classification 
  of Alzheimer’s Disease", Frontiers in Neuroinformatics, vol. 7, no. 50, 
  pp. 1-15, January 2014. 

`mdreg` uses the interface `itk-elastix` which is based on SimpleElastix, 
created by Kasper Marstal:

- Kasper Marstal, Floris Berendsen, Marius Staring and Stefan Klein, 
  "SimpleElastix: A user-friendly, multi-lingual library for medical image 
  registration", International Workshop on Biomedical Image Registration 
  (WBIR), Las Vegas, Nevada, USA, 2016.
