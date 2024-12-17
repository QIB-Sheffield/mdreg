#######################
**mdreg** documentation
#######################

Model-driven image registration for medical imaging

.. note::

   ``mdreg`` is under construction. At this stage, the API may still change and 
   features may be deprecated without warning.


***
Aim
***

TO offer a user-friendly approach to remove subject motion 
from a time series of medical images with changing intrinsic contrast. 


************
Installation
************

.. code-block:: console

    pip install mdreg


*************
Typical usage
*************

Consider a dataset consisting of:

- a 4D array *signal* with a series of free-breathing 3D MRI images of 
  the abdomen with variable flip angles (VFA).
- a 1D array *FA* with the respective flip angles.

The following script removes the motion from the array and shows an animation 
with the results:

.. code-block:: python

    import mdreg

    # Identify a suitable signal model from the library
    vfa = {
        'func': mdreg.spgr_vfa_lin,  
        'FA': FA,
    }
  
    # Remove the motion from the signal data
    coreg, defo, fit, pars = mdreg.fit(signal, vfa) 
    
    # Inspect the result visually
    mdreg.animation(coreg, show=True)


The function :func:`mdreg.fit` returns 4 arrays:

- *coreg* is the signal array with motion removed;
- *defo* is the deformation field;
- *fit* is the array with model fits;
- *pars* is an array with fitted parameters.


********
Features
********

- A simple customizable high-level interface :func:`mdreg.fit` for 2D or 3D 
  motion correction of time series.

- A growing library of :ref:`signal models <models>` for different 
  applications, including T1- or T2 mapping, dynamic contrast-enhanced 
  MRI or CT, no contrast change.
  
- An interface for integrating custom-built models in case a suitable model is 
  not currently available in the library.

- A harmonized interface for different coregistration models (rigid, 
  deformable, optical flow) available in different packages (`elastix`, 
  `skimage`, `dipy`).

***************
Getting started
***************

Have look at the :ref:`user guide <user-guide>` or the list 
of :ref:`examples <examples>`.


******
Citing
******

When you use ``mdreg``, please cite: 

Kanishka Sharma, Fotios Tagkalakis, Irvin Teh, Bashair A Alhummiany, 
David Shelley, Margaret Saysell, Julie Bailey, Kelly Wroe, Cherry Coupland, 
Michael Mansfield, Steven P Sourbron. An open-source, platform independent 
library for model-driven registration in quantitative renal MRI. ISMRM 
workshop on renal MRI, Lisbon/Philadephia, sept 2021.


*******
License
*******

``dcmri`` is distributed under the 
`Apache 2.0 <https://www.apache.org/licenses/LICENSE-2.0>`_ license - a 
permissive, free license that allows users to use, modify, and 
distribute the software without restrictions.


.. toctree::
   :maxdepth: 2
   :hidden:
   
   guide/index
   reference/index
   generated/examples/index
   contribute/index
   releases/index
   about/index

