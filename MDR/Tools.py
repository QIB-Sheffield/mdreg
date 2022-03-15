"""
TOOLS FOR MODEL DRIVEN REGISTRATION (MDR)
This file contains several functions that help and support the development of a motion correction analysis pipeline

MDR Library
@Kanishka Sharma
@Joao Almeida e Sousa
@Steven Sourbron
2021
"""

import os
import numpy as np
from PIL import Image
import itk
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def read_elastix_model_parameters(Elastix_Parameter_file_PATH, *argv):
    """ Read elastix parameters from given text file.

    Args:
    ----
        Elastix_Parameter_file_PATH (string): Path to text file containing motion correction parameters.

    Returns:
    -------
        elastix_model_parameters (itk-elastix.ParameterObject): ITK-Elastix variable containing the parameters for the registration.

    Example:
    -------
        elastix_model_parameters = read_elastix_model_parameters('BSplines_T2star.txt', ['MaximumNumberOfIterations', 256], ['MaximumStepLength ', 1.0])

    """
    elastix_model_parameters = itk.ParameterObject.New()
    elastix_model_parameters.AddParameterFile(Elastix_Parameter_file_PATH)
    for lstParameters in argv:
        elastix_model_parameters.SetParameter(str(lstParameters[0]), str(lstParameters[1]))
    return elastix_model_parameters
   

def export_results(MDR_output=(), path='', model='', pars=[], xy=()):

    if not os.path.exists(path):
        os.mkdir(path)

    defx = np.squeeze(MDR_output[2][:,:,0,:])
    defy = np.squeeze(MDR_output[2][:,:,1,:])

    export_animation(MDR_output[0], os.path.join(path, model), 'coregistered')
    export_animation(MDR_output[1], os.path.join(path, model), 'modelfit')
    export_animation(defx, os.path.join(path, model), 'deformation_field_x')
    export_animation(defx, os.path.join(path, model), 'deformation_field_x')
    export_animation(np.sqrt(defx**2 + defy**2), os.path.join(path, model), 'deformation_field')
    for i in range(len(pars)):
    #    export_maps(MDR_output[3][i,:], os.path.join(path, model, pars[i]), xy)
        export_imgs(MDR_output[3][i,:], os.path.join(path, model, pars[i]), xy)
    MDR_output[4].to_csv(os.path.join(path, model, 'largest_deformations.csv'))


def export_animation(arr, path, filename):

    if not os.path.exists(path): os.mkdir(path)
    file = os.path.join(path, filename + '.gif')
    fig = plt.figure()
    im = plt.imshow(np.squeeze(arr[:,:,0]), animated=True)
    def updatefig(i):
        im.set_array(np.squeeze(arr[:,:,i]))
    anim = animation.FuncAnimation(fig, updatefig, interval=50, frames=arr.shape[2])
    anim.save(file)
    #plt.show()


def export_images(MDR_individual_output, folder):
    """ Save MDR results to given folder.

    Args:
    ----
        MDR_individual_output (numpy.array): Numpy array representing one of the outputs of MDR.
        folder (string): Path to the folder that will host the results/output of this method.
    """
    if not os.path.exists(os.path.dirname(folder)): os.makedirs(os.path.dirname(folder))
    shape = np.shape(MDR_individual_output)
    for i in range(shape[2]):
        im = Image.fromarray(MDR_individual_output[:,:,i])
        im.save(folder + str(i) + ".tiff")

def export_maps(MDR_individual_output, folder, shape):
    """ Save MDR results to given folder. Fitted Parameters to output folder.

    Args:
    ----
        MDR_individual_output (numpy.array): Numpy array representing one of the outputs of MDR.
        folder (string): Path to the folder that will host the results/output of this method.
        shape (list): Shape of the output array to which MDR_individual_output will be reshaped
    """
    if not os.path.exists(os.path.dirname(folder)): 
        os.makedirs(os.path.dirname(folder))
    array = np.reshape(MDR_individual_output, [shape[0],shape[1]]) 
    Img = Image.fromarray(array)
    Img.save(folder + ".tiff")

def export_imgs(array, folder, shape):

    if not os.path.exists(os.path.dirname(folder)): 
        os.makedirs(os.path.dirname(folder))
    array = np.reshape(array, [shape[0],shape[1]])
    plt.imshow(array)
    #plt.clim(int(minValue), int(maxValue))
    cBar = plt.colorbar()
    cBar.minorticks_on()
    plt.savefig(fname=folder + '.png')
    plt.close()
