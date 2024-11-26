**********************
Parameter Descriptions
**********************

Table of parameter types:

moving
The series of images to be corrected.
The array can be either 3D or 4D with the following shapes: 
3D: (X, Y, T). 4D: (X, Y, Z, T). Here, X, Y, Z are the spatial 
dimensions and T is the dimension denoting change e.g. temporal 
dimension or flip angle.

coreg : numpy.array
The coregistered images.
The array matches the shape of the input moving array.
defo : numpy.array
The deformation field.
The array matches the shape of the input moving array, with an 
additional dimension showing deformation components. For 2D spatial
images the deformation field has shape (X, Y, 2, T). For 3D spatial 
images the deformation field has shape (X, Y, Z, 3, T).

 fit : numpy.array
The fitted signal model.
The array matches the shape of the input moving array.

pars : dict
The parameters of the fitted signal model.
Array has the same shape as the spatial coordinates of the signal, 
with a final extra axis length based on how many parameters the model 
requires (N). For 2D spatial data: (X, Y, N). For 3D spatial data: 
(X, Y, Z, N).

