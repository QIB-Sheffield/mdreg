__all__ = ['MDReg']


import time, os, copy
import multiprocessing
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import itk
import SimpleITK as sitk

from .models import constant

default_path = os.path.dirname(__file__)

class MDReg:

    def __init__(self):

        # input
        self.array = None
        self.signal_parameters = None
        self.pixel_spacing = 1.0
        self.signal_model = constant
        self.elastix = itk.ParameterObject.New()
        self.elastix.AddParameterFile(os.path.join(default_path, 'BSplines.txt'))

        # mdr optimization
        self.max_iterations = 5
        self.precision = 1.0

        # output
        self.coreg = None
        self.model_fit = None
        self.deformation = None
        self.pars = None
        self.iter = None
        self.export_path = os.path.join(default_path, 'results')
        self.export_unregistered = True

    @property
    def _npdt(self): 
        """
        (nr of pixels, nr of dimensions, nr of time points)
        """
        shape = self.array.shape  
        return np.prod(shape[:-1]), len(shape)-1, shape[-1]

    def set_array(self, array):
        self.array = array
        self.coreg = array
        n = self._npdt
        self.coreg = np.reshape(self.coreg, (n[0],n[2]))

    def read_elastix(self, file):
        self.elastix.AddParameterFile(file)
    
    def set_elastix(self, **kwargs):
        for tag, value in kwargs.items():
            self.elastix.SetParameter(tag, str(value))       

    def fit(self):

        n = self._npdt
        self.coreg = copy.deepcopy(self.array)
        self.coreg = np.reshape(self.coreg, (n[0],n[2]))
        self.deformation = np.zeros(n)
        start = time.time()
        improvement = []
        converged = False
        it = 1
        while not converged: 
            startit = time.time()
            print('Starting MDR iteration ' + str(it))
            self.fit_signal()
            if self.export_unregistered:
                if it == 1: self.export_fit(name='_unregistered')
            deformation = self.fit_deformation()
            improvement.append(_maxnorm(self.deformation-deformation))
            self.deformation = deformation
            converged = improvement[-1] <= self.precision 
            if it == self.max_iterations: converged=True
            calctime = (time.time()-startit)/60
            print('Finished MDR iteration ' + str(it) + ' after ' + str(calctime) + ' min') 
            print('Improvement after MDR iteration ' + str(it) + ': ' + str(improvement[-1]) + ' pixels')  
            it += 1       
        self.fit_signal()
        shape = self.array.shape
        self.coreg = np.reshape(self.coreg, shape)
        nd = len(shape)-1
        self.deformation = np.reshape(self.deformation, shape[:-1]+(nd,)+(shape[-1],))
        self.iter = pd.DataFrame({'Maximum deformation': improvement}) 

        print('Calculation time: ' + str((time.time()-start)/60) + ' min')

    def fit_signal(self):

        start = time.time()
        print('Fitting signal model..')
        fit, pars = self.signal_model.main(self.coreg, self.signal_parameters)
        shape = self.array.shape
        self.model_fit = np.reshape(fit, shape)
        self.pars = np.reshape(pars, shape[:-1] + (pars.shape[-1],))
        print('Finished fitting signal model (' + str((time.time()-start)/60) + ' min)')

    def fit_deformation(self, parallel=True, log=False, mask=None):

        start = time.time()
        print('Performing coregistration..')
        nt = self._npdt[-1]
        deformation = np.empty(self._npdt)
        dict_param = _elastix2dict(self.elastix) # Hack necessary for parallelization
        # If mask isn't same shape as images, then don't use it
        if isinstance(mask, np.ndarray):
            if np.shape(mask) != self.array.shape: mask = None  
        if not parallel:
            for t in tqdm(range(nt), desc='Coregistration progress'): #dynamics
                if mask is not None: 
                    mask_t = mask[...,t]
                else: 
                    mask_t = None
                args = (self.array[...,t], self.model_fit[...,t], dict_param, self.pixel_spacing, log, mask_t)
                self.coreg[:,t], deformation[:,:,t] = _coregister(args)
        else:
            pool = multiprocessing.Pool(processes=os.cpu_count()-1)
            if mask is None:
                args = [(self.array[...,t], self.model_fit[...,t], dict_param, self.pixel_spacing, log, mask) for t in range(nt)] #dynamics
            else:
                args = [(self.array[...,t], self.model_fit[...,t], dict_param, self.pixel_spacing, log, mask[...,t]) for t in range(nt)] #dynamics
            results = list(tqdm(pool.imap(_coregister, args), total=nt, desc='Coregistration progress'))
            for t in range(nt):
                self.coreg[:,t] = results[t][0]
                deformation[:,:,t] = results[t][1]
        print('Finished coregistration (' + str((time.time()-start)/60) +' min)')
        return deformation

    def export(self):

        self.export_data()
        self.export_fit()
        self.export_registered()

    def export_data(self):

        print('Exporting data..')
        path = self.export_path 
        if not os.path.exists(path): os.mkdir(path)
        _export_animation(self.array, path, 'images')

    def export_fit(self, name=''):

        print('Exporting fit..' + name)
        path = self.export_path 
        pars = self.signal_model.pars()
        if not os.path.exists(path): os.mkdir(path)
        lower, upper = self.signal_model.bounds()
        for i in range(len(pars)):
            _export_imgs(self.pars[...,i], path, pars[i] + name, bounds=[lower[i],upper[i]])
        _export_animation(self.model_fit, path, 'modelfit' + name)

    def export_registered(self):

        print('Exporting registration..')
        path = self.export_path 
        if not os.path.exists(path): os.mkdir(path)
        defx = np.squeeze(self.deformation[:,:,0,:])
        defy = np.squeeze(self.deformation[:,:,1,:])
        _export_animation(self.coreg, path, 'coregistered')
        _export_animation(defx, path, 'deformation_field_x')
        _export_animation(defy, path, 'deformation_field_y')
        _export_animation(np.sqrt(defx**2 + defy**2), path, 'deformation_field')
        self.iter.to_csv(os.path.join(path, 'largest_deformations.csv'))


def _export_animation(array, path, filename):

    file = os.path.join(path, filename + '.gif')
    array[np.isnan(array)] = 0
    fig = plt.figure()
    im = plt.imshow(np.squeeze(array[:,:,0]).T, animated=True)
    def updatefig(i):
        im.set_array(np.squeeze(array[:,:,i]).T)
    anim = animation.FuncAnimation(fig, updatefig, interval=50, frames=array.shape[2])
    anim.save(file)
    #plt.show()

def _export_imgs(array, path, filename, bounds=[-np.inf, np.inf]):

    file = os.path.join(path, filename + '.png')
    array[np.isnan(array)] = 0
    array[np.isinf(array)] = 0
    array = np.clip(array, bounds[0], bounds[1])
    plt.imshow(array.T)
    plt.clim(np.amin(array), np.amax(array))
    cBar = plt.colorbar()
    cBar.minorticks_on()
    plt.savefig(fname=file)
    plt.close()

def _maxnorm(d):
    """This function calculates diagnostics from the registration process.

    It takes as input the original deformation field and the new deformation field
    and returns the maximum deformation per pixel (in mm).
    The maximum deformation per pixel is calculated as 
    the euclidean distance of difference between the old and new deformation field. 
    """
    d = d[:,0,:]**2 + d[:,1,:]**2
    return np.nanmax(np.sqrt(d))

def _elastix2dict(elastix_model_parameters):
    """
    Hack to allow parallel processing
    """
    list_dictionaries_parameters = []
    for index in range(elastix_model_parameters.GetNumberOfParameterMaps()):
        parameter_map = elastix_model_parameters.GetParameterMap(index)
        one_parameter_map_dict = {}
        for i in parameter_map:
            one_parameter_map_dict[i] = parameter_map[i]
        list_dictionaries_parameters.append(one_parameter_map_dict)
    return list_dictionaries_parameters


def _dict2elastix(list_dictionaries_parameters):
    """
    Hack to allow parallel processing
    """
    elastix_model_parameters = itk.ParameterObject.New()
    for one_map in list_dictionaries_parameters:
        elastix_model_parameters.AddParameterMap(one_map)
    return elastix_model_parameters



def _coregister(args):
    """
    Coregister two arrays and return coregistered + deformation field 
    """
    target, source, elastix_model_parameters, spacing, log, mask = args
    shape_source = np.shape(source)
    shape_target = np.shape(target)

    source = sitk.GetImageFromArray(source)
    source.SetSpacing(spacing)
    source.__SetPixelAsUInt16__
    source = np.nan_to_num(np.reshape(source, [shape_source[0], shape_source[1]]))
    
    target = sitk.GetImageFromArray(target)
    target.SetSpacing(spacing)
    target.__SetPixelAsUInt16__
    target = np.nan_to_num(np.reshape(target, [shape_target[0], shape_target[1]]))
    
    ## read the source and target images
    elastixImageFilter = itk.ElastixRegistrationMethod.New()
    elastixImageFilter.SetFixedImage(itk.GetImageFromArray(np.array(source, np.float32)))
    elastixImageFilter.SetMovingImage(itk.GetImageFromArray(np.array(target, np.float32)))
    if mask is not None:
        shape_mask = np.shape(mask)
        mask = sitk.GetImageFromArray(mask)
        mask.SetSpacing(spacing)
        mask.__SetPixelAsUInt8__
        mask = np.nan_to_num(np.reshape(mask, [shape_mask[0], shape_mask[1]]))
        elastixImageFilter.SetFixedMask(itk.GetImageFromArray(np.array(mask, np.uint8)))
        elastixImageFilter.SetMovingMask(itk.GetImageFromArray(np.array(mask, np.uint8)))

    ## call the parameter map file specifying the registration parameters
    elastix_model_parameters = _dict2elastix(elastix_model_parameters) # Hack
    elastixImageFilter.SetParameterObject(elastix_model_parameters)

    ## set additional options
    elastixImageFilter.SetNumberOfThreads(os.cpu_count()-1)
    
    # logging; note that nothing is printed in Jupyter Notebooks
    elastixImageFilter.SetLogToFile(log)
    elastixImageFilter.SetLogToConsole(log)
    if log == True:
        print("Parameter Map: ")
        print(elastix_model_parameters)
        output_dir = os.path.join(os.getcwd(), "Elastix Log")
        os.makedirs(output_dir, exist_ok=True)
        #log_filename = "ITK-Elastix.log"
        i = 0
        while os.path.exists(os.path.join(output_dir, f"ITK-Elastix_{i}.log")): i += 1
        log_filename = f"ITK-Elastix_{i}.log"
        elastixImageFilter.SetOutputDirectory(output_dir)
        elastixImageFilter.SetLogFileName(log_filename)

    ## update filter object (required)
    elastixImageFilter.UpdateLargestPossibleRegion()

    ## RUN ELASTIX using ITK-Elastix filters
    coregistered = itk.GetArrayFromImage(elastixImageFilter.GetOutput()).flatten()

    transformixImageFilter = itk.TransformixFilter.New()
    transformixImageFilter.SetTransformParameterObject(elastixImageFilter.GetTransformParameterObject())
    transformixImageFilter.ComputeDeformationFieldOn()
    transformixImageFilter.SetMovingImage(itk.GetImageFromArray(np.array(target, np.float32)))
    deformation_field = itk.GetArrayFromImage(transformixImageFilter.GetOutputDeformationField()).flatten()
    deformation_field = np.reshape(deformation_field, [int(len(deformation_field)/2), 2])

    return coregistered, deformation_field

