#######################
**mdreg** documentation
#######################

Model-driven image registration for medical imaging

.. note::

   ``mdreg`` is under construction. At this stage, the API may still change and 
   features may be deprecated without warning.


********
Features
********

`mdreg` aims to offer a user-friendly approach to remove subject motion 
from time series of medical images with changing intrinsic contrast. 

It includes:

- A simple customizable high-level interface `~mdreg.fit` for 2D or 3D motion 
  correction of time series.

- A growing library of :ref:`signal models <models>` for different 
  applications, including T1- or T2 mapping, dynamic contrast-enhanced 
  MRI or CT, no contrast change.
  
- An interface for integrating custom-built models in case a suitable model is 
  not currently available in the library.

- A harmonized interface for different coregistration models (rigid, 
  deformable, optical flow) available in different packages (`elastix`, 
  `skimage`, `dipy`).


*********
Rationale
*********

Many applications in medical imaging involve time series of 2D images or 
3D volumes that are corrupted by subject motion. Examples are T1- or T2- 
mapping in MRI, diffusion-weighted MRI, or dynamic contrast-enhanced imaging 
in MRI or CT.

Motion correction of such data is challenging because the signal changes 
caused by the motion are superposed on the often drastic changes in intrinsic 
image contrast. 

In most cases though, these intrinsic changes in image 
contrast can be described analytically. Indeed many of these applications 
critically depend on the availability of a model to derive parametric maps 
from the signal data.

Model-driven image registration leverages the existence of a signal model to 
remove the confounding effects of changes in image contrast on the results 
of the motion correction. 


***************
Getting started
***************

To get started, have look at the :ref:`user guide <user-guide>` or the list 
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

