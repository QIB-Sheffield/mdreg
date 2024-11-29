.. _plot_dict:

*****************************
Plotting Options
*****************************

This section provides detailed information about the arguments required for the
functions within the `mdreg.plot` module. The plotting functions are used to
visualize the results of the motion correction and fitting processes. Below is 
a table that describes each parameter in the dictionary, including its type and
a brief description of its purpose.

Not all arguments are required for each plotting function, and the default 
values are provided in the table below. To use custom values, you should 
create a dictionary with the desired arguments and pass it to the plotting 
functions. See this carried out in the :ref:`examples`.

.. _plot-param-table:
.. list-table:: **Description of image plotting dictionary arguments**
    :widths: 20 20 60
    :header-rows: 1

    * - Arguement
      - Type
      - Description
    * - path
      - str, optional
      - The path to save the animation. The default is None.
    * - filename
      - str, optional
      - The filename of the animation. The default is 'animation'.
    * - vmin
      - float, optional
      - The minimum value for the colormap. The default is None.
    * - vmax
      - float, optional
      - The maximum value for the colormap. The default is None.
    * - slice
      - int, optional
      - The slice to plot. The default is None. This argument does not apply to 2D images, but is used for 3D images. None will plot all slices.
    * - interval
      - int, optional
      - The interval between animation frames. The default is 250ms.
    * - show
      - bool, optional
      - Whether to display the animation. The default is False.