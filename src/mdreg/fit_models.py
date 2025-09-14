import os
from typing import Union, Tuple

from tqdm import tqdm
import numpy as np
import zarr
import dask
import dask.array as da
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA

from mdreg import pixel_models, io


def _func_init(xdata, ydata, p0):
    return p0


def _fit_func(func, func_init, xdata, ydata, p0, bounds, **kwargs):

    p0 = func_init(xdata, ydata, p0)

    for i, p in enumerate(p0):
        if np.isscalar(bounds[0]):
            if p<bounds[0]:
                raise ValueError(f"Initial value {p} for parameter {i} is "
                                 f"below the lower bound {bounds[0]}.")
        else:
            if p<bounds[0][i]:
                raise ValueError(f"Initial value {p} for parameter {i} is "
                                 f"below the lower bound {bounds[0][i]}.")
        if np.isscalar(bounds[1]):
            if p>bounds[1]:
                raise ValueError(f"Initial value {p} for parameter {i} is "
                                 f"above the upper bound {bounds[1]}.")
        else:
            if p>bounds[1][i]:
                raise ValueError(f"Initial value {p} for parameter {i} is "
                                 f"above the upper bound {bounds[1][i]}.")
            
    try:
        p, _ = curve_fit(func, 
            xdata = xdata, 
            ydata = ydata, 
            p0 = p0, 
            bounds = bounds, 
            **kwargs, 
        )
        return p
    except RuntimeError:
        return p0
    

def fit_pixels(ydata,
        model = None,
        p0 = None,
        xdata = None, 
        func_init = _func_init,
        parallel = True,
        progress_bar = True,
        path = None,
        bounds = (-np.inf, +np.inf),  
        memdim = 2,      
        **kwargs, 
    ):

    """
    Fit a model pixel-wise

    Parameters
    ----------
    ydata : numpy.ndarray | zarr.Array
        2D or 3D array of signal intensities
    model : function
        Model function to fit to the data (required). Model functions need to 
        have one argument *xdata* followed by the free parameters of the model. 
        The return value is the signal at each value of *xdata*. 
    p0 : array
        Initial guess for the model parameters (required)
    xdata : array-like
        Independent variables for the model. If this is not provided, an 
        index array is used.
    func_init : function
        Function to initialize the model parameters. It must take *xdata* and 
        *ydata* as arguments, followed by *p0* (see below). It returns 
        estimates for the free model parameters as a tuple.
    parallel : bool
        Option to perform fitting in parallel. Default is True.
    progress_bar : bool
        Option to display a progress bar. This option is ignored when 
        parallel = True. Default is True.
    path : str, optional
        Path on disk where to save the results. If no path is provided, the 
        results are not saved to disk. Defaults to None.
    memdim : int
        Number of array dimensions to be held in memory at any one time. 
        Possible values for memdim range from 0 (pixel-by-pixel processing) 
        to ydata.ndim-1 (process the whole array at once). With memdim=1,
        data are loaded and processed row-by-row, with memdim=2 the are 
        processed slice-by-slice, and so on. Default is 2.
    bounds : tuple
        Bounds for the model parameters, in the format required by 
        scipy.curve_fit. Note the initial value p0 needs to 
        be contained in the bounds.
    **kwargs : Any additional arguments accepted by scipy.curve_fit().

    Returns
    -------
    fit : numpy.ndarray | zarr.Array
        The fitted signal model with the same dimensions as *arr*.
    pars : numpy.ndarray | zarr.Array
        The parameters of the fitted signal model with dimensions (x,y,n) or 
        (x,y,z,n), where n is the number of free parameters of the signal 
        model.
    """
    # Check parameters
    if model is None:
        raise ValueError('model is a required argument')
    if p0 is None:
        raise ValueError('p0 is a required argument')
    
    if xdata is None:
        xdata = np.arange(ydata.shape[-1])

    if isinstance(ydata, zarr.Array):
        fit, par = _fit_pixels_zarr(
            ydata, model, xdata, func_init, parallel, progress_bar, 
            bounds, p0, path, memdim, **kwargs)
    else:
        fit, par = _fit_pixels_numpy(
            ydata, model, xdata, func_init, parallel, progress_bar, 
            bounds, p0, **kwargs)
        if path is not None:
            np.save(os.path.join(path, 'fit'), fit)
            np.save(os.path.join(path, 'pars'), par)

    return fit, par



def _fit_pixels_numpy(
        ydata, model, xdata, func_init, parallel, progress_bar, 
        bounds,  p0, **kwargs):

    shape = ydata.shape
    ydata = ydata.reshape((-1,shape[-1]))
    nx, nt = ydata.shape

    if not parallel:
        p = []
        for x in tqdm(
                range(nx), desc='Fitting pixels', disable=not progress_bar,
            ):
            p_x = _fit_func(
                model, func_init, xdata, ydata[x,:], p0, bounds, **kwargs,
            )
            p.append(p_x)
    else:
        tasks = []
        for x in range(nx):
            task_x = dask.delayed(_fit_func)(
                model, func_init, xdata, ydata[x,:], p0, bounds, **kwargs,
            )
            tasks.append(task_x)
        p = dask.compute(*tasks)

    # Compute output arrays
    n = len(p[0])
    par = np.empty((nx, n)) 
    fit = np.empty((nx, nt))
    for x in range(nx):
        par[x,:] = p[x]
        fit[x,:] = model(xdata, *tuple(p[x]))
    fit = fit.reshape(shape)
    par = par.reshape(shape[:-1]+(n,))

    return fit, par


def _fit_pixels_zarr(
        ydata, model, xdata, func_init, parallel, progress_bar, 
        bounds, p0, path, memdim, **kwargs):
    
    if memdim is None:
        memdim = ydata.ndim-1
    
    # Dimension of data in memory
    if memdim < 0:
        raise ValueError("memdim cannot be negative.")
    if memdim > ydata.ndim-1:
        raise ValueError("memdim cannot be larger than ydata.ndim-1.")

    # Get the shape and number of slice dimensions
    shape = ydata.shape[memdim:-1]
    n = int(np.prod(shape))

    # All indices in slice and time dimensions
    p = tuple([slice(None) for _ in range(memdim)])
    t = (slice(None), )

    for k in tqdm(
            range(n), 
            desc='Fitting zarray', 
            disable=(not progress_bar) or (n==1),
        ):

        # Convert flat index to multi-index
        z = np.unravel_index(k, shape)

        # Load slice z into memory
        ydata_k = ydata[p + z + t]

        # Fit in memory
        fit_k, par_k = _fit_pixels_numpy(
            ydata_k, model, xdata, func_init, parallel, 
            progress_bar=progress_bar and (n==1), 
            bounds=bounds, p0=p0, **kwargs)
        
        # If this is the first slice, create the zarrays
        if k==0:
            fit, par = io._fit_models_init(ydata, path, par_k.shape[-1])

        # Save results for slize z in the zarray
        fit[p + z + t] = fit_k
        par[p + z + t] = par_k

    return fit, par



def fit_deconvolution(signals: np.ndarray, aif=None, tol=0.2, n0=1):
    shape = signals.shape
    signals = signals.reshape(-1, shape[-1])

    # Build signal change
    ca = aif - np.mean(aif[:n0])
    S0 = np.mean(signals[...,:n0], axis=-1)[..., None]
    conc = signals - S0

    # Build matrix
    n = len(ca)
    mat = np.zeros((n,n))
    for i in range(0,n):
        for j in range(0,i+1):
            mat[i,j] = ca[i-j]

    # Invert matrix
    U, s, Vt = np.linalg.svd(mat, full_matrices=False)
    svmin = tol*np.amax(s)
    s_inv = np.array([1/x if x > svmin else 0 for x in s])
    mat_inv = np.dot(Vt.T * s_inv, U.T)

    # Apply matrices
    conc_rec = (mat @ mat_inv) @ conc.T
    signal_rec = conc_rec.T + S0

    return signal_rec.reshape(shape), None



def fit_pca(data_4d, n_components=None):
    """
    Performs Principal Component Analysis (PCA) on a 4D dataset (3D spatial + 1D time).

    The function reshapes the 4D array into a 2D matrix where each row
    represents the time series of a single voxel. PCA is then applied to
    this matrix to identify the principal components of the temporal variations.

    Args:
        data_4d (np.ndarray): The input 4D array with shape (X, Y, Z, T),
                              where T is the time dimension.
        n_components (int, optional): The number of principal components to keep.
                                      If None, all components are kept. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - components (np.ndarray): The principal components (eigen-curves) of the
                                       time series. Shape: (n_components, T).
            - spatial_maps (np.ndarray): The 3D spatial weights (scores) for each
                                         component. Shape: (X, Y, Z, n_components).
            - explained_variance (np.ndarray): The amount of variance explained by
                                               each component.
    """
    # --- 2. Reshape the 4D data to 2D ---
    # The new shape will be (number_of_voxels, time_points)
    # This is the format required by scikit-learn's PCA
    reshaped_data = data_4d.reshape(-1, data_4d.shape[-1])

    # --- 3. Perform PCA ---
    pca = PCA(n_components=n_components)
    
    # fit_transform calculates the principal components and projects the data onto them
    scores = pca.fit_transform(reshaped_data)
    
    # The principal components are the "eigen-curves"
    components = pca.components_
    
    # --- 4. Reshape the scores back to 3D spatial maps ---
    # This gives us a 3D map for each component, showing its spatial distribution
    num_actual_components = components.shape[0]
    spatial_maps = scores.reshape(data_4d.shape[:-1] + (num_actual_components, ))

    pca_fit = _reconstruct_from_pca(spatial_maps, components)

    return pca_fit, spatial_maps


def _reconstruct_from_pca(spatial_maps, components):
    """
    Reconstructs the 4D signal from its PCA components and spatial maps.

    This is the inverse operation of the PCA decomposition. It performs a
    matrix multiplication of the scores (spatial maps) and the components
    to rebuild the time series for each voxel.

    Args:
        spatial_maps (np.ndarray): The 3D spatial weights (scores) for each
                                   component. Shape: (X, Y, Z, n_components).
        components (np.ndarray): The principal components (eigen-curves).
                                 Shape: (n_components, T).

    Returns:
        np.ndarray: The reconstructed 4D data array. Shape: (X, Y, Z, T).
    """
    # Get original dimensions
    t_dim = components.shape[1]

    # Reshape spatial maps from (X, Y, Z, n_components) to (X*Y*Z, n_components)
    scores = spatial_maps.reshape(-1, spatial_maps.shape[-1])

    # Reconstruct the 2D data matrix by matrix multiplication
    # (N_voxels, n_components) @ (n_components, T) -> (N_voxels, T)
    reconstructed_2d = scores @ components

    # Reshape the 2D data back to the original 4D shape
    reconstructed_4d = reconstructed_2d.reshape(spatial_maps.shape[:-1] + (t_dim, ))
    
    return reconstructed_4d



def fit_constant(signal: Union[np.ndarray, zarr.Array]):
    r"""
    Fit to a constant.

    .. math::

        S(\mathbf{r},t) = S_0(\mathbf{r})

    Parameters
    ----------
        signal : numpy.ndarray | zarr.Array 
            3D or 4D array with signal intensities. Dimensions are (x, y, t) 
            or (x, y, z, t).
    
    Returns
    -------
        fit : numpy.ndarray | zarr.Array
            Fit to the signal data, with same dimensions as the signal array.
        pars : numpy.ndarray | zarr.Array
            Fitted model parameter S0. Dimensions are (x,y,1) or (x,y,z,1). 
    """
    if isinstance(signal, zarr.Array):
        signal = da.from_zarr(signal)
    else:
        signal = da.from_array(signal)
    shape = signal.shape
    avr = da.mean(signal, axis=-1) 
    par = da.reshape(avr, shape[:-1] + (1,))
    par.compute()
    fit = da.repeat(par, repeats=shape[-1], axis=-1)
    fit.compute()
    return fit, par






def fit_exp_decay(
        signal,
        time=None,
        parallel=True,
        bounds=([0,0], [np.inf, np.inf]),
        p0=[1,1], 
        **kwargs):
    r"""
    Fit to an exponential decay.

    .. math::

        S(\mathbf{r},t) = S_0(\mathbf{r}) e^{-t/T(\mathbf{r})}

    Parameters
    ----------
        signal : numpy.ndarray | zarr.Array 
            3D or 4D array with signal intensities. Dimensions are (x, y, t) 
            or (x, y, z, t).
        time : numpy.ndarray
            Timepoints of the signal data
        parallel : bool
            If True, use parallel processing. Default is False
        bounds : tuple 
            Bounds for the fit as (lower_bound, upper_bound) where lower_bound 
            and upper_bound are either a scalar or a list of 2 values.
        p0 : list
            Initial values as a 2-element list.
        **kwargs :
            Additional keyword arguments accepted by fit_pixels.
    
    Returns
    -------
        fit : numpy.ndarray | zarr.Array
            Fitted to the signal data, with same dimensions as the signal array.
        pars : numpy.ndarray | zarr.Array
            Fitted model parameters S0 and T. Dimensions are (x,y,2) or 
            (x,y,z,2).

    """
    if time is None:
        raise ValueError('time is a required argument.')
    
    return fit_pixels(signal,
        model = pixel_models.exp_decay, 
        xdata = np.array(time),
        func_init = pixel_models.exp_decay_init,
        parallel = parallel,
        bounds = bounds,
        p0 = p0, 
        **kwargs,
    )



def fit_abs_exp_recovery_2p(
        signal, 
        TI=None,
        parallel=True,
        bounds=([0,0], [np.inf, np.inf]),
        p0=[1,1.3],
        **kwargs,
    ):
    r"""
    2-parameter fit to an absolute exponential-recovery model fit

    .. math::

        S(\mathbf{r},T_I) = \left| S_0(\mathbf{r}) \left( 1 - 2 e^{-T_I/T(\mathbf{r})} \right) \right|

    Parameters
    ----------
        signal : numpy.ndarray | zarr.Array
            3D or 4D array with signal intensities. Dimensions are (x, y, t) 
            or (x, y, z, t).
        TI : numpy.array
            Inversion times
        parallel : bool
            If True, use parallel processing. Default is False.
        bounds : tuple 
            Bounds for the fit as (lower_bound, upper_bound) where lower_bound 
            and upper_bound are either a scalar or a list of 2 values.
        p0 : list
            Initial values as a 2-element list.
        **kwargs :
            Additional keyword arguments accepted by fit_pixels.
    
    Returns
    -------
        fit : numpy.ndarray | zarr.Array
            Fit to the signal data, with same dimensions as the signal array.
        pars : numpy.ndarray | zarr.Array
            Fitted model parameters S0 and T. Dimensions are (x,y,2) or 
            (x,y,z,2).
        
    """
    if TI is None:
        raise ValueError('TI is a required parameter.')

    return fit_pixels(signal, 
        model=pixel_models.abs_exp_recovery_2p, 
        xdata = np.array(TI),
        func_init = pixel_models.abs_exp_recovery_2p_init,
        bounds = bounds,
        p0 = p0, 
        parallel = parallel,
        **kwargs, 
    )


def fit_exp_recovery_2p(signal, 
        TI=None,
        parallel=True,
        bounds=([0,0], [np.inf, np.inf]),
        p0=[1,1.3], 
        **kwargs):
    r"""
    2-parameter fit to an exponential recovery model

    .. math::

        S(\mathbf{r},T_I) = S_0(\mathbf{r}) \left( 1 - 2 e^{-T_I/T(\mathbf{r})} \right)
    
    Parameters
    ----------
        signal : numpy.ndarray | zarr.Array
            3D or 4D array with signal intensities. Dimensions are (x, y, t) 
            or (x, y, z, t).
        TI : numpy.array
            Inversion times
        parallel : bool
            If True, use parallel processing. Default is False.
        bounds : tuple 
            Bounds for the fit as (lower_bound, upper_bound) where lower_bound 
            and upper_bound are either a scalar or a list of 2 values.
        p0 : list
            Initial values as a 2-element list.
        **kwargs :
            Additional keyword arguments accepted by fit_pixels.
    
    Returns
    -------
        fit : numpy.ndarray | zarr.Array
            Fit to the signal data, with same dimensions as the signal array.
        pars : numpy.ndarray | zarr.Array
            Fitted model parameters S0 and T. Dimensions are (x,y,2) or 
            (x,y,z,2).
    """
    
    if TI is None:
        raise ValueError('TI is a required parameter.')
    
    return fit_pixels(signal, 
        model=pixel_models.exp_recovery_2p, 
        xdata=np.array(TI),
        func_init=pixel_models.exp_recovery_2p_init,
        parallel=parallel,
        bounds=bounds,
        p0=p0, 
        **kwargs, 
    )



def fit_abs_exp_recovery_3p(signal, 
        TI=None,
        parallel=True,
        bounds=([0,0,0], [np.inf, np.inf, 2]),
        p0=[1, 1.3, 2], 
        **kwargs):
    r"""
    2-parameter fit to an absolute exponential-recovery model fit.

    .. math::

        S(\mathbf{r},T_I) = \left| S_0(\mathbf{r}) \left( 1 - A(\mathbf{r}) e^{-T_I/T(\mathbf{r})} \right) \right|

    Parameters
    ----------
        signal : numpy.ndarray | zarr.Array
            3D or 4D array with signal intensities. Dimensions are (x, y, t) 
            or (x, y, z, t).
        TI : numpy.array
            Inversion times
        parallel : bool
            If True, use parallel processing. Default is False.
        bounds : tuple 
            Bounds for the fit as (lower_bound, upper_bound) where lower_bound 
            and upper_bound are either a scalar or a list of 3 values.
        p0 : list
            Initial values as a 3-element list.
        **kwargs :
            Additional keyword arguments accepted by fit_pixels.
    
    Returns
    -------
        fit : numpy.ndarray | zarr.Array
            Fit to the signal data, with same dimensions as the signal array.
        pars : numpy.ndarray | zarr.Array
            Fitted model parameters S0, T and A. Dimensions are (x,y,3) or 
            (x,y,z,3).
    """
    
    if TI is None:
        raise ValueError('TI is a required parameter.')
    
    return fit_pixels(signal, 
        model=pixel_models.abs_exp_recovery_3p, 
        xdata = np.array(TI),
        func_init = pixel_models.abs_exp_recovery_3p_init,
        bounds = bounds,
        p0 = p0, 
        parallel = parallel,
        **kwargs, 
    )


def fit_exp_recovery_3p(signal, 
        TI=None,
        parallel=True,
        bounds=([0,0,0],[np.inf, np.inf, 2]),
        p0=[1,1.3,2], 
        **kwargs,
    ):
    r"""
    3-parameter fit to an exponential-recovery model fit.

    .. math::

        S(\mathbf{r},T_I) = S_0(\mathbf{r}) \left( 1 - A(\mathbf{r}) e^{-T_I/T(\mathbf{r})} \right) 

    Parameters
    ----------
        signal : numpy.ndarray | zarr.Array
            3D or 4D array with signal intensities. Dimensions are (x, y, t) 
            or (x, y, z, t).
        TI : numpy.array
            Inversion times
        parallel : bool
            If True, use parallel processing. Default is False.
        bounds : tuple 
            Bounds for the fit as (lower_bound, upper_bound) where lower_bound 
            and upper_bound are either a scalar or a list of 3 values.
        p0 : list
            Initial values as a 3-element list.
        **kwargs :
            Additional keyword arguments accepted by fit_pixels.
    
    Returns
    -------
        fit : numpy.ndarray | zarr.Array
            Fit to the signal data, with same dimensions as the signal array.
        pars : numpy.ndarray | zarr.Array
            Fitted model parameters S0, T and A. Dimensions are (x,y,3) or 
            (x,y,z,3).
    """
    
    if TI is None:
        raise ValueError('TI is a required parameter.')
    
    return fit_pixels(signal, 
        model=pixel_models.exp_recovery_3p, 
        xdata = np.array(TI),
        func_init = pixel_models.exp_recovery_3p_init,
        parallel = parallel,
        bounds = bounds,
        p0 = p0, 
        **kwargs, 
    )


def fit_spgr_vfa(signal, 
        FA=None,
        parallel=True,
        bounds=([0, 0], [np.inf, 1]),
        p0=[1, 0.5], 
        **kwargs,
    ):
    r"""
    Non-linear fit to a variable flip angle model.

    .. math::

        S(\mathbf{r},\alpha) = S_0(\mathbf{r}) \sin(\alpha) \frac{1 - e^{-T_R/T_1(\mathbf{r})}}{1 - \cos(\alpha) e^{-T_R/T_1(\mathbf{r})}}

    Parameters
    ----------
        signal : numpy.ndarray | zarr.Array
            Array with signal intensities for different flip angles (FA). 
            Dimensions are (x,y,FA) or (x,y,z,FA)
        FA : array-like
            Flip angles in degrees (required). This is a 1D array with the same
            length as the last dimension of the signal array. Defaults to None.
        parallel : bool
            If True, use parallel processing. Default is False.
        bounds : tuple 
            Bounds for the fit as (lower_bound, upper_bound) where lower_bound 
            and upper_bound are either a scalar or a list of 3 values.
        p0 : list
            Initial values as a 3-element list.
        **kwargs :
            Additional keyword arguments accepted by fit_pixels.

    Returns
    -------
        fit : numpy.ndarray | zarr.Array
            Fit to the signal data, with same dimensions as the signal array.
        pars : numpy.ndarray | zarr.Array
            Fitted model parameters S0 and E. Dimensions are (x,y,2) or 
            (x,y,z,2). 
            
    """
    
    if FA is None:
        raise ValueError('Flip angle (FA) is a required parameter.')
    
    return fit_pixels(signal, 
        model=pixel_models.spgr_vfa, 
        xdata=FA,
        func_init=pixel_models.spgr_vfa_init,
        parallel=parallel,
        bounds=bounds,
        p0=p0, 
        **kwargs, 
    )


def fit_spgr_vfa_lin(
        signal:np.ndarray, 
        FA=None, 
        path=None,
        memdim=2, 
        parallel=True,
        progress_bar=False,
    ):
    r"""
    Linear fit to a variable flip angle model.

    .. math::

        S(\mathbf{r},\alpha)=S_0(\mathbf{r})\sin{\alpha} \frac{1-e^{-T_R/T_1(\mathbf{r})}}{1-\cos{\alpha}\,e^{-T_R/T_1(\mathbf{r})}}

    Here :math:`S(\alpha)` is the signal at flip angle :math:`\alpha`, 
    :math:`S_0` a scaling factor, :math:`T_R` the repetition time and 
    :math:`T_1` the longitudinal relaxation time. 

    Parameters
    ----------
        signal : numpy.ndarray | zarr.Array
            Array with signal intensities for different flip angles (FA). 
            Dimensions are (x,y,FA) or (x,y,z,FA)
        FA : array
            Flip Angles in degrees (required). This is a 1D array with the same
            length as the last dimension of the signal array. Defaults to None.
        path : str, optional
            Path on disk where to save the results. If no path is provided, the 
            results are not saved to disk. Defaults to None.
        memdim : int
            For zarrays, the number of array dimensions to be held in memory at 
            any one time. This keyword is ignored when the argument is a numpy
            array.
            Possible values for memdim range from 0 (pixel-by-pixel processing) 
            to ydata.ndim-1 (process the whole array at once). With memdim=1,
            data are loaded and processed row-by-row, with memdim=2 the are 
            processed slice-by-slice, and so on. Default is 2.
        parallel : bool
            Option to perform fitting in parallel. This is only available for 
            zarrays when memdim is provided. 
        progress_bar: bool
            Display a progress bar (default = False).
    
    Returns
    -------
        fit : numpy.ndarray | zarr.Array
            Fit to the signal data, with same dimensions as the signal array.
        pars : numpy.ndarray | zarr.Array
            Fitted model parameters S0 and E. Dimensions are (x,y,2) or 
            (x,y,z,2). 

    Notes:

        To derive the linear version of the model, the equation can be rewritten 
        as: 

        .. math::

            Y(\alpha) = AX(\alpha)+B

        with the variables defined as:
        
        .. math::

            X = S(\alpha/\sin{\alpha};~~~~Y=S(\alpha)\cos{\alpha} / \sin{\alpha}

        and the constants:

        .. math::

            E=e^{-T_R/T_1};~~~~A=\frac{1}{E};~~~~B=-S_0\frac{1-E}{E}~

        Plotting :math:`Y(\alpha)` against :math:`X(\alpha)` produces a 
        straight line with slope :math:`A` and intercept :math:`B`. After 
        solving for :math:`A, B` these constants can then be used reconstruct 
        the signal:

        .. math::

            S(\alpha)=\frac{B\sin{\alpha}}{\cos{\alpha}-A}
    
    """
    if isinstance(signal, zarr.Array):
        raise ValueError("fit_spgr_vfa_lin is not yet available for zarrays.")
    if FA is None:
        raise ValueError('Flip angle (FA) is a required parameter.')

    if isinstance(signal, np.ndarray):
        fit, par = _fit_spgr_vfa_lin_compute(signal, FA, progress_bar)
        if path is not None:
            np.save(os.path.join(path, 'fit'), fit)
            np.save(os.path.join(path, 'pars'), par)
        return fit, par
    
    if memdim is None:
        memdim = signal.ndim-1
    
    # Dimension of data in memory
    if memdim < 0:
        raise ValueError("memdim cannot be negative.")
    if memdim > signal.ndim-1:
        raise ValueError("memdim cannot be larger than signal.ndim-1.")

    # Build stores for outputs
    fit, par = io._fit_models_init(signal, path, 2)

    # Get the shape and number of slice dimensions
    shape = signal.shape[memdim:-1]
    n = int(np.prod(shape))

    # All indices in slice dimensions
    p = tuple([slice(None) for _ in range(memdim)])
    
    if parallel:
        pbar = False
        tasks = [
            dask.delayed(_fit_spgr_vfa_lin_slice)(
                k, signal, shape, p, FA, fit, par, pbar,
            )
            for k in range(n)
        ]
        dask.compute(*tasks)
    else:
        for k in tqdm(
                range(n), 
                desc='Fitting vfa', 
                disable=(not progress_bar) or (n==1),
            ):
            pbar = progress_bar and (n==1)
            _fit_spgr_vfa_lin_slice(
                k, signal, shape, p, FA, fit, par, pbar,
            )

    return fit, par


def _fit_spgr_vfa_lin_slice(
        k, signal, shape, p, FA, fit, par, progress_bar):

    # Convert flat index to multi-index
    z = np.unravel_index(k, shape)

    # Load all values for slice z into memory
    t = (slice(None), )
    signal_k = signal[p + z + t]

    # Compute
    fit_k, par_k = _fit_spgr_vfa_lin_compute(signal_k, FA, progress_bar)

    # Save results for slize z in the zarray
    fit[p + z + t] = fit_k
    par[p + z + t] = par_k
    


def _fit_spgr_vfa_lin_compute(signal, FA, progress_bar):
    
    FA = np.deg2rad(FA)

    # Construct FA array in matching shape to signal data
    FA_array = np.ones_like(signal)*FA
    sFA_array = np.sin(FA_array)

    X = signal / sFA_array
    Y = signal * np.cos(FA_array) / sFA_array

    shape = signal.shape
    X = X.reshape(-1, shape[-1])
    Y = Y.reshape(-1, shape[-1])
    signal = signal.reshape(-1, shape[-1])

    pars = np.empty((X.shape[0], 2))
    fit = np.empty(X.shape)

    sFA, cFA = np.sin(FA), np.cos(FA)
    ones = np.ones(shape[-1])

    for i in tqdm(
            range(X.shape[0]), 
            desc='Fitting VFA model', 
            disable=not progress_bar,
        ):
        A = np.vstack([X[i,:], ones])
        pars[i,:] = np.linalg.lstsq(A.T, Y[i,:], rcond=None)[0]
        if 0 in (cFA - pars[i,0]):
            fit[i,:] = signal[i,:]
        else:
            fit[i,:] = pars[i,1] * sFA / (cFA - pars[i,0])
        smax = np.amax(signal[i,:])
        fit[i,:][fit[i,:] > smax] = smax

    fit[fit<0] = 0
    fit[np.isnan(fit)] = 0

    # Convert to T1 and S0
    A = pars[:,0]
    B = pars[:,1]
    S0 = -B/(A-1)
    E = 1/A
    pars[:,0] = S0
    pars[:,1] = E
    
    return fit.reshape(shape), pars.reshape(shape[:-1] + (2,))



def fit_2cm_lin(
        signal: Union[np.ndarray, zarr.Array], 
        aif=None,
        time=None,
        baseline=1,
        path=None,
        memdim=2, 
        parallel=False,
        progress_bar=True,
        input_corr=False,

    ) -> Tuple[Union[np.ndarray, zarr.Array], Union[np.ndarray, zarr.Array]]:
    
    """
    Linearised 2-compartment model fit

    Parameters
    ----------
        signal : numpy.ndarray | zarr.Array 
            Array with signal intensities at 
            different times. Dimensions are (x,y,t) or (x,y,z,t)
        aif : numpy.ndarray
            Arterial input function. 1D array of input artery signal 
            intensities, length equal to the number of time points in the 
            signal data.
        time : numpy.ndarray
            Timepoints of the signal data
        baseline : int
            Baseline. Number of time points to use for the baseline signal. 
            Default is 1.
        path : str, optional
            Path on disk where to save the results. If no path is provided, the 
            results are not saved to disk. Defaults to None.
        memdim : int
            For zarrays, the number of array dimensions to be held in memory at 
            any one time. This keyword is ignored when the argument is a numpy
            array.
            Possible values for memdim range from 0 (pixel-by-pixel processing) 
            to ydata.ndim-1 (process the whole array at once). With memdim=1,
            data are loaded and processed row-by-row, with memdim=2 the are 
            processed slice-by-slice, and so on. Default is 2. 
        parallel : bool
            Option to perform fitting in parallel. This is only available for 
            zarrays when memdim is provided.
        progress_bar : bool
            Set to True to display a progress bar during the computations. This 
            is ignored if parallel=True.

    Returns
    -------
        fit : numpy.ndarray | zarr.Array
            Fit to the signal data with same dimensions as the signal array.
        pars : numpy.ndarray | zarr.Array
            Fitted model parameters Fb, Tb, PS, Te. Dimensions are (x,y,4) or 
            (x,y,z,4). 
    """
    # Check arguments
    if aif is None:
        raise ValueError('aif is a required parameter.')
    if time is None:
        raise ValueError('Time is a required parameter.')
    if parallel:
        if progress_bar:
            raise ValueError(
                "A progress bar cannot be shown when parallel=True. "
                "Set parallel=False or progress_bar=False. "
            )
    
    if isinstance(signal, np.ndarray):
        fit, par = _fit_2cm_lin_compute(
            signal, aif, time, baseline, input_corr, progress_bar,
        )
        if path is not None:
            np.save(os.path.join(path, 'fit'), fit)
            np.save(os.path.join(path, 'pars'), par)
        return fit, par
    
    if memdim is None:
        memdim = signal.ndim-1
    
    # Dimension of data in memory
    if memdim < 0:
        raise ValueError("memdim cannot be negative.")
    if memdim > signal.ndim-1:
        raise ValueError("memdim cannot be larger than signal.ndim-1.")

    # Build stores for outputs
    npar = 5 if input_corr else 4
    fit, par = io._fit_models_init(signal, path, npar)

    # Get the shape and number of slice dimensions
    shape = signal.shape[memdim:-1]
    n = int(np.prod(shape))

    # All indices in slice dimensions
    p = tuple([slice(None) for _ in range(memdim)])
    
    if parallel:
        pbar = False
        tasks = [
            dask.delayed(_fit_2cm_lin_slice)(
                k, signal, shape, p, aif, time, baseline, input_corr, fit, par, pbar,
            )
            for k in range(n)
        ]
        dask.compute(*tasks)
    else:
        for k in tqdm(
                range(n), 
                desc='Fitting 2cm', 
                disable=(not progress_bar) or (n==1),
            ):
            pbar = progress_bar and (n==1)
            _fit_2cm_lin_slice(
                k, signal, shape, p, aif, time, baseline, input_corr, fit, par, pbar,
            )

    return fit, par


def _fit_2cm_lin_slice(k, signal, shape, p, aif, time, baseline, input_corr, fit, par, 
                 progress_bar):

    # Convert flat index to multi-index
    z = np.unravel_index(k, shape)

    # Load all values for slice z into memory
    t = (slice(None), )
    signal_k = signal[p + z + t]

    # Compute
    fit_k, par_k = _fit_2cm_lin_compute(
        signal_k, aif, time, baseline, input_corr, progress_bar,
    )

    # Save results for slize z in the zarray
    fit[p + z + t] = fit_k
    par[p + z + t] = par_k



def _fit_2cm_lin_compute(signal, aif, time, baseline, input_corr, progress_bar):

    npar = 5 if input_corr else 4

    # Reshape to 2D (x,t)
    shape = signal.shape
    signal = signal.reshape((-1,shape[-1]))

    S0 = np.mean(signal[:,:baseline], axis=1)
    ca = aif-np.mean(aif[:baseline])
    
    A = np.empty((signal.shape[1],npar))
    A[:,2], A[:,3] = _ddint(ca, time)
    if input_corr:
        A[:,4] = ca

    fit = np.empty(signal.shape)
    par = np.empty((signal.shape[0], npar))
    for x in tqdm(
            range(signal.shape[0]), 
            desc='Fitting 2-comp model', 
            disable=not progress_bar,
        ):
        c = signal[x,:] - S0[x]
        ctii, cti = _ddint(c, time)
        A[:,0] = -ctii
        A[:,1] = -cti
        p = np.linalg.lstsq(A, c, rcond=None)[0] 
        fit[x,:] = S0[x] + p[0]*A[:,0] + p[1]*A[:,1] + p[2]*A[:,2] + p[3]*A[:,3]
        if input_corr:
            fit[x,:] += p[4]*A[:,4]
        if input_corr:
            par[x,:] = _2cm_lin_params(p[:4]) + [p[4]]
        else:
            par[x,:] = _2cm_lin_params(p)

    # Apply bounds
    smax = np.amax(signal)
    fit[fit<0]=0
    fit[fit>2*smax]=2*smax

    # Return in original shape
    fit = fit.reshape(shape)
    par = par.reshape(shape[:-1] + (npar,))

    return fit, par


def _ddint(c, t):
    ci = cumulative_trapezoid(c, t, initial=0)
    cii = cumulative_trapezoid(ci, t, initial=0)
    return cii, ci


def _2cm_lin_params(X):

    alpha = X[0]
    beta = X[1]
    gamma = X[2]
    Fp = X[3]
    
    if alpha == 0: 
        if beta == 0:
            return [Fp, 0, 0, 0]
        else:
            return [Fp, 1/beta, 0, 0]

    nom = 2*alpha
    det = beta**2 - 4*alpha
    if det < 0 :
        Tp = beta/nom
        Te = Tp
    else:
        root = np.sqrt(det)
        Tp = (beta - root)/nom
        Te = (beta + root)/nom

    if Te == 0:
        PS = 0
    else:   
        if Fp == 0:
            PS = 0
        else:
            T = gamma/(alpha*Fp) 
            PS = Fp*(T-Tp)/Te   
    
    return [Fp, Tp, PS, Te]




