**********************
Variable Descriptions
**********************

This table provides a comprehensive and detailed description of the various 
variable types utilized within the mdregs functions. Each variable type is 
explained with its corresponding dimensions and usage context to aid in 
understanding their roles and applications.

For the purpose of the information within the tables below the 2D description 
relates to the motion correction of 2D image slices that have an additional
dimension of time or similar. These are 3D numpy arrays but the spatial 
dimensions are 2D.

The 3D description relates to the motion of full 3D spatial image volumes that 
have an additional dimension of time or similar. These are 4D numpy arrays with
the spatial dimensions being 3D.

Within the table the array shapes for 2D and 3D spatial motion correction are
noted separately, with a shared description of the variable name. Here, X, Y, Z 
are the spatial dimensions and T is the dimension denoting change e.g. temporal 
dimension, flip angle or similar.  

.. _variable-types-table:
.. list-table:: **Description of main mdreg variable types**
    :header-rows: 1

    * - Variable Name
      - Full Name
      - 2D Motion Correction
      - 3D Motion Correction
      - Description
    * - signal
      - Signal for model fitting 
      - (X, Y, T)
      - (X, Y, Z, T)
      - The current series of images to be corrected.
    * - moving
      - Intermediate image during motion correction
      - (X, Y, T)
      - (X, Y, Z, T)
      - The series of images to be corrected. This is the original signal data, in the first iteration, and the coregistered image for refinement in subsequent iterations.
    * - coreg
      - Coregistered Images
      - (X, Y, T)
      - (X, Y, Z, T)
      - The coregistered images.
    * - defo
      - Deformation Field
      - (X, Y, 2, T)
      - (X, Y, Z, 3, T)
      - The deformation field. The array matches the shape of the input moving array, with an additional dimension showing deformation components.
    * - fit
      - Signal Model Fit
      - (X, Y, T)
      - (X, Y, Z, T)
      - The fitted signal model.
    * - pars
      - Model Parameters
      - (X, Y, N)
      - (X, Y, Z, N)
      - The parameters of the fitted signal model. Array has the same shape as the spatial coordinates of the signal, with a final extra axis length based on how many parameters the model requires (N).
    * - time
      - Timepoints
      - ( T )
      - ( T )
      - Array of time points, with length equal to the number of time points in the signal data.
    * - TI
      - Inversion Times
      - ( T )
      - ( T )
      - Array of inversion times, with length equal to the number of inversion times in the signal data.
    * - FA
      - Flip Angles
      - ( T )
      - ( T )
      - Array of flip angles, with length equal to the number of flip angles in the signal data.