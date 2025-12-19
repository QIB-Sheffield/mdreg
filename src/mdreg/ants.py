from typing import Tuple, Union
import os
import numpy as np
import zarr
from tqdm import tqdm
import dask
try:
    import ants
except:
    not_installed = True
else:
    not_installed = False

from mdreg import io


def defaults():
    """Default parameters for ants registration

    Returns:
        dict: Default keyword arguments.
    """
    return {
        'type_of_transform': "SyN",
        'initial_transform': None,
        'outprefix': "",
        'mask': None,
        'moving_mask': None,
        'mask_all_stages': False,
        'grad_step': 0.2,
        'flow_sigma': 3,
        'total_sigma': 0,
        'aff_metric': "mattes",
        'aff_sampling': 32,
        'aff_random_sampling_rate': 0.2,
        'syn_metric': "mattes",
        'syn_sampling': 32,
        'reg_iterations': (40, 20, 0),
        'aff_iterations': (2100, 1200, 1200, 10),
        'aff_shrink_factors': (6, 4, 2, 1),
        'aff_smoothing_sigmas': (3, 2, 1, 0),
        'write_composite_transform': False,
        'random_seed': None,
        'verbose': False,
        'multivariate_extras': None,
        'restrict_transformation': None,
        'smoothing_in_mm': False,
        'singleprecision': True,
        'use_legacy_histogram_matching': False,
    }


def coreg_series(
        moving: Union[np.ndarray, zarr.Array], 
        fixed: Union[np.ndarray, zarr.Array], 
        parallel=True, 
        progress_bar=False, 
        path=None, 
        name='coreg',
        return_transfo=True,
        **kwargs,
    ):
    """
    Coregister two series of 2D images or 3D volumes.

    Parameters
    ----------
    moving : numpy.ndarray | zarr.Array
        The moving image or volume, with dimensions (x,y,t) or (x,y,z,t). 
    fixed : numpy.ndarray
        The fixed target image or volume, in the same dimensions as the 
        moving image. 
    parallel : bool
        Set to True to parallelize the computations. Defaults to True.
    progress_bar : bool
        Show a progress bar during the computation. This keyword is ignored 
        if parallel = True. Defaults to False.
    path : str, optional
        Path on disk where to save the results. If no path is provided, the 
        results are not saved to disk. Defaults to None.
    name : str, optional
        For data that are saved on disk, provide an optional filename. This 
        argument is ignored if no path is provided.
    return_transfo (bool): is True, return the transformations as paths 
        to files on disk. If this is set to False, only the coregistered 
        image is returned and the transformations are deleted on disk.
        Defaults to True.
    kwargs : dict
        Any keyword argument accepted by 
        `ants.registration <https://antspy.readthedocs.io/en/latest/registration.html>`_. 
        Array arguments need to be provided as numpy arrays rather 
        than ants's own data type.

    Returns
    -------
    cor : numpy.ndarray | zarr.Array
        Coregistered series with the same dimensions as the moving image. 
    transfo : list
        List of paths to files containing the transformation parameters.
    """

    if not_installed:
        raise ImportError(
            "Coregistration with ants is optional - please install mdreg as "
            "pip install mdreg[ants] if you want to use these features."
        )
    
    if moving.shape != fixed.shape:
        raise ValueError('Moving and fixed arrays must have the '
                         'same shape.')
    
    coreg = io._copy(moving, path, name)
    transfo = np.empty(moving.shape[-1], dtype=object)

    if parallel:
        tasks = []
        for t in range(moving.shape[-1]): 
            task_t = dask.delayed(_coreg_t)(
                t, moving, fixed, coreg, transfo, **kwargs,
            )
            tasks.append(task_t)
        dask.compute(*tasks)
    else:
        for t in tqdm(
                range(moving.shape[-1]), 
                desc='Coregistering series', 
                disable=not progress_bar, 
            ): 
            _coreg_t(t, moving, fixed, coreg, transfo, **kwargs)

    # Create return values
    transfo = list(transfo)
    if not return_transfo:
        for transfo_t in transfo:
            if isinstance(transfo_t, list):
                [os.remove(t) for t in transfo_t]
            else:
                os.remove(transfo_t)
        return coreg
    else:
        return coreg, transfo
       


def _coreg_t(t, moving, fixed, deformed, transfo, **kwargs):
    deformed[...,t], transfo[t] = coreg(
        moving[...,t], fixed[...,t], **kwargs,
    )

def transform_series(
        moving, 
        transfo, 
        path=None, 
        name='deform',
    ):
    """
    Transforms a series of images using a transformation produced by 
    `mdreg.ants.coreg_series()`.

    Parameters
    ----------
    moving : numpy.ndarray | zarr.Array
        The moving image or volume, with dimensions (x,y,t) or (x,y,z,t).  
    transfo : list
        List of itk.elastixParameterObject with one transform per image 
        in the series.
    path : str, optional
        Path on disk where to save the results. If no path is provided, the 
        results are not saved to disk. Defaults to None.
    name : str, optional
        For data that are saved on disk, provide an optional filename. This 
        argument is ignored if no path is provided.

    Returns
    -------
        numpy.ndarray: The transformed image.
    """

    if not_installed:
        raise ImportError(
            "Coregistration with ants is optional - please install mdreg as "
            "pip install mdreg[ants] if you want to use these features."
        )
    
    deform = io._copy(moving, path, name)
    for t, transfo_t in enumerate(transfo):
        deform[...,t] = transform(moving[...,t], transfo_t)
    return deform


def coreg(
        moving: np.ndarray, 
        fixed: np.ndarray, 
        return_transfo = True,
        return_inverse = False,
        **kwargs,
    ):
    """
    Coregister two 2D images or 3D volumes.

    Args:
        moving (np.ndarray): The moving 2D or 3D image.
        fixed (np.ndarray): The fixed target image with the same shape as the 
            moving image. 
        return_transfo (bool): is True, return the transformations as paths 
            to files on disk. If this is set to False, only the coregistered 
            image is returned and the transformations are deleted on disk.
            Defaults to True.
        kwargs: Any keyword argument accepted by 
          `ants.registration <https://antspy.readthedocs.io/en/latest/registration.html>`_. 
          Note that array arguments need to be provided as numpy arrays rather 
          than ants's own data type.

    Returns:
        coreg : np.ndarray
            The registered moving image.
        transfo : str | list
            path or paths of parameter files encoding the transformations.
            If return_inverse = False, transfo contains the path(s) to the forward
            transformation parameter file only (from moving to coregistered
            image). If return_inverse = True, transfo is a list where each element
            contains the path(s) to the forward and inverse (from coregistered to
            moving image) transformation parameter files, respectively.
    """

    if not_installed:
        raise ImportError(
            "Coregistration with ants is optional - please install mdreg as "
            "pip install mdreg[ants] if you want to use these features."
        )
    
    # Convert NumPy arrays to ANTs images
    fixed_ants = ants.from_numpy(fixed)
    moving_ants = ants.from_numpy(moving)
    if 'mask' in kwargs:
        kwargs['mask'] = ants.from_numpy(kwargs['mask'])
    if 'moving_mask' in kwargs:
        kwargs['moving_mask'] = ants.from_numpy(kwargs['moving_mask'])

    # Perform registration
    registration = ants.registration(fixed_ants, moving_ants, **kwargs)

    # Get the transformed moving image as an array
    coreg_ants = registration['warpedmovout']
    coreg = coreg_ants.numpy().astype(fixed.dtype)

    # Create return values
    if return_inverse == True:
        transfo = [registration['fwdtransforms'], registration['invtransforms']]
    else:
        transfo = registration['fwdtransforms']
    if not return_transfo:
        if isinstance(transfo, list):
            [os.remove(t) for t in transfo]
        else:
            os.remove(transfo)
        return coreg
    else:
        return coreg, transfo


def transform(moving, transfo, interpolator='linear'):
    """
    Transforms an image using a transformation produced by 
    `mdreg.ants.coreg()`.

    Parameters:
        moving : numpy.ndarray
            The input 2D or 3D image array.
        transfo : str | list
            path or paths to parameter files encoding the transformation 
            from moving to coregistered image 
        interpolator : str
            Type of interpolation to use. For options see the 
            `ants documentation <https://antspy.readthedocs.io/en/latest/registration.html>`_ 

    Returns:
        numpy.ndarray: The transformed image.
    """
    if not_installed:
        raise ImportError(
            "Coregistration with ants is optional - please install mdreg as "
            "pip install mdreg[ants] if you want to use these features."
        )
    
    moving_ants = ants.from_numpy(moving)

    # Apply transformation
    warped_image_ants = ants.apply_transforms(
        fixed=moving_ants, 
        moving=moving_ants, 
        transformlist=transfo,
        interpolator=interpolator,
    )

    # Return as numpy array
    return warped_image_ants.numpy().astype(moving.dtype)



# def warp(moving: np.ndarray, deformation_field: np.ndarray) -> np.ndarray:
#     """
#     Warps a moving image using a given deformation field without requiring a 
#     fixed reference volume.

#     Args:
#         moving_array (np.ndarray): The moving image (2D or 3D).
#         deformation_field_array (np.ndarray): The deformation field (vector 
#             displacement field).

#     Returns:
#         np.ndarray: The warped moving image.
#     """

#     # Convert NumPy arrays to ANTs images
#     moving = ants.from_numpy(moving)

#     # Convert deformation field to an ANTs image
#     deformation_field = ants.from_numpy(deformation_field)

#     # Save on disk temporarily
#     tmp = os.path.join(os.getcwd(), '_warp_tmp_defo.nii.gz')
#     if os.path.exists(tmp):
#         raise ValueError(
#             "Temporary file _warp_tmp_defo.nii.gz already exists in the "
#             f"current working directory {os.getcwd()}. Please delete it "
#             "first before calling warp() again.")
#     ants.image_write(deformation_field, tmp)

#     # Apply transformation
#     warped_image = ants.apply_transforms(
#         fixed=moving, 
#         moving=moving, 
#         transformlist=[tmp],
#         interpolator='linear',
#     )

#     # Remove tmp file on disk
#     os.remove(tmp)

#     # Return as numpy array
#     return warped_image.numpy()


