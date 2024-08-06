#####
About
#####

.. note::

   `mdreg` is developed in public but is work in progress. As long as this notice is visible, backwards compatibility is not guaranteed and features may be deprecated without warning.

`mdreg` is developed by the `medical imaging group <https://www.sheffield.ac.uk/smph/research/themes/imaging>`_ at the University of Sheffield, UK. Contact: `Steven Sourbron <https://github.com/plaresmedima>`_.

A first python implementation of model-driven registration was developed for use in quantitative renal MRI in the iBEAt study, and compared against group-wise model-free registration implemented in elastix (Tagkalakis F, et al. Model-based motion correction outperforms a model-free method in quantitative renal MRI. Abstract-1383, ISMRM 2021). 

The code base has since been rewritten from scratch. Up to version 0.3.x, development and deployment was funded by the Innovative Medicines Initiative through the `BEAt-DKD project <https://www.beat-dkd.eu/>`_.

The coregistration engines for `mdreg` are `itk-elastix` and `scikit-image`.

## Acknowledgement

The iBEAt study is part of the BEAt-DKD project. The BEAt-DKD project has received funding from the Innovative Medicines Initiative 2 Joint Undertaking under grant agreement No 115974. This Joint Undertaking receives support from the European Union’s Horizon 2020 research and innovation programme and EFPIA with JDRF. For a full list of BEAt-DKD partners, see `www.beat-dkd.eu <https://www.beat-dkd.eu/>`_.

..
    This works but does not format properly.
    .. include:: teams.inc

