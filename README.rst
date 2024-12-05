mdreg
=====

.. image:: https://img.shields.io/badge/License-Apache_2.0-blue.svg
  :target: https://opensource.org/licenses/Apache-2.0



Model-driven motion correction for medical imaging
--------------------------------------------------

- **Documentation:** https://qib-sheffield.github.io/mdreg/
- **Source code:** https://github.com/QIB-Sheffield/mdreg


*Note:* mdreg is under construction. At this stage, the API may still change 
and features may be deprecated without warning.


Installation
------------

.. code-block:: console

    pip install mdreg


Typical usage
-------------

.. code-block:: python

    import mdreg

    # Get some test data (variable flip angle MRI)
    data = mdreg.fetch('VFA')  

    # Configure the signal model fit
    fit_image = {
        'func': mdreg.spgr_vfa_lin,     # VFA signal model
        'FA': data['FA'],               # Flip angle in degress    
        'progress_bar': True,           # Show a progress bar
    }

    # Configure the coregistration method
    fit_coreg = {
        'package': 'elastix',
        'spacing': data['spacing'],           
    } 

    # Perform the motion correction
    coreg, defo, fit, pars = mdreg.fit(
        data['array'],                # Signal data to correct
        fit_image = fit_image,        # Signal model fit
        fit_coreg = fit_coreg,        # Coregistration
        verbose = 2,                  # Show progress update
    )

    # Visualize the results
    anim = mdreg.animation(
        coreg, 
        title = 'Motion corrected VFA', 
        'interval' : 500,                   # Time between animation frames in ms
        'vmin' : 0,                         # Minimum value of the colorbar
        'vmax' : np.percentile(coreg, 99),  # Maximum value of the colorbar
    )


.. image:: https://qib-sheffield.github.io/mdreg/_images/sphx_glr_plot_3Dexample_vfa_002.gif
  :width: 800


License
-------

Released under the `Apache 2.0 <https://opensource.org/licenses/Apache-2.0>`_  
license::

  Copyright (C) 2023-2024 dcmri developers
