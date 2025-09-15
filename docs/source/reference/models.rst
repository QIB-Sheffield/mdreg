.. _models:

*************
Signal models
*************

.. currentmodule:: mdreg

.. _signal-models:

Signal models
-------------

Single-pixel signal forward models.

.. autosummary::
   :toctree: ../generated/api/
   :template: autosummary.rst

   const 
   lin
   quad
   othree
   ofour
   exp_decay
   exp_recovery_2p
   abs_exp_recovery_2p
   exp_recovery_3p
   abs_exp_recovery_3p
   spgr_vfa


Signal models - initializers
----------------------------

Estimate parameter values for single-pixel models from data.

.. autosummary::
   :toctree: ../generated/api/
   :template: autosummary.rst

   exp_decay_init
   exp_recovery_2p_init
   abs_exp_recovery_2p_init
   exp_recovery_3p_init
   abs_exp_recovery_3p_init
   spgr_vfa_init


.. _fit-funcs:

Fitting signal models
---------------------

Functions to fit signal models directly to signal array data.

.. autosummary::
   :toctree: ../generated/api/
   :template: autosummary.rst

   fit_pixels
   fit_constant
   fit_exp_decay
   fit_abs_exp_recovery_2p
   fit_exp_recovery_2p
   fit_abs_exp_recovery_3p
   fit_exp_recovery_3p
   fit_spgr_vfa
   fit_spgr_vfa_lin
   fit_2cm_lin
   fit_pca
   fit_deconvolution