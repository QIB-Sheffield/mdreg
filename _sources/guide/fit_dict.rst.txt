.. _fit_dict:

*****************************
Fitting Options
*****************************

This section provides detailed information about the arguments required for the 
`fit_image` dictionary in the `mdreg` package. The `fit_image` dictionary is used 
to fit a model to image data, and the arguments specified in the dictionary 
control various aspects of the fitting process. Below is a table that describes 
each parameter in the dictionary, including its type and a brief description of 
its purpose.

.. _fit-image-table:
.. list-table:: **Description of image fitting dictionary arguments**
    :header-rows: 1

    * - Key
      - Type
      - Description
    * - func
      - `func`
      - The function name to be used for model fitting. These can either use a predefined model from `mdreg.models` or a custom function.
    * - various
      - `dict` key pairs
      - Additional required arguments that are passed to the fitting function. Such as known values: flip angle, inversion times, TR, TE etc. Or additional parameters for the fitting function e.g. bounds, p0 ect.



All that is needed in order to create a custom function and use it inside mdreg
is to wrap it into a function that takes the image array as argument, and 
returns the fit to the model and the fitted model parameters. Use the format of
functions in the :ref:`models` module as a guide.
