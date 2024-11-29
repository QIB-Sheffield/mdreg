.. _coreg_dict:

*****************************
Coregistration Options
*****************************

This section provides detailed information about the arguments that can be
passed to `mdreg.fit` to control the coregistration process. These arguements 
are provided as a dictionary referred to as `fit_coreg` in the `mdreg.fit` 
function.

The options allow the user to control the coregistration package used, and the 
optional parameters which can be controlled within the different packages.

By default, `mdreg` uses the `elastix` package for coregistration, with default
parameters. To use a different package or to control the parameters, the user
will need to construct a dictionary with the desired arguments. Use of bespoke
coregistration parameters is illustrated in the :ref:`examples`.

Below is a table that describes each parameter in the dictionary, including its
type and a brief description of its purpose.

.. _fit-coreg-table:
.. list-table:: **Description of coregistration dictionary arguments**
    :header-rows: 1

    * - Argument
      - Type
      - Description
    * - package
      - `str`
      - The package to use for coregistration. Options are 'elastix' or 'skimage'.
    * - parameters
      - `dict`
      - The parameters to pass to the coregistration package to control specifics of the registration. These are dependent on the package used. Leaving this empty will use default parameters for each package from `mdreg`.
    * - spacing
      - `list`
      - The spacing of voxels in the images in mm [X,Y,Z]. The default is 1.0.
