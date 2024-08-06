import sys
import os
import pickle
import multiprocessing
from tqdm import tqdm

import numpy as np
from scipy.optimize import curve_fit


# filepaths need to be identified with importlib_resources
# rather than __file__ as the latter does not work at runtime 
# when the package is installed via pip install
if sys.version_info < (3, 9):
    # importlib.resources either doesn't exist or lacks the files()
    # function, so use the PyPI version:
    import importlib_resources
else:
    # importlib.resources has files(), so use that:
    import importlib.resources as importlib_resources



try: 
    num_workers = int(len(os.sched_getaffinity(0)))
except: 
    num_workers = int(os.cpu_count())



def fetch(dataset:str)->dict:
    f = importlib_resources.files('mdreg.datafiles')
    datafile = str(f.joinpath(dataset + '.pkl'))
    with open(datafile, 'rb') as fp:
        data_dict = pickle.load(fp)
    return data_dict


def _init_output(array:np.ndarray):
    #Initialize outputs
    if array.ndim == 3: #2D
        shape = (array.shape[0], array.shape[1], 2, array.shape[2]) 
    else: #3D
        shape = (array.shape[0], array.shape[1], array.shape[2], 3, array.shape[3])
    deformation = np.zeros(shape)
    coreg = array.copy()

    return coreg, deformation


def _func_init(xdata, ydata, p0):
    return p0


def _fit_func(args):
    func, func_init, xdata, ydata, p0, bounds, kwargs = args
    p0 = func_init(xdata, ydata, p0)
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
        xdata = None,
        func_init = _func_init,
        bounds = (-np.inf, +np.inf),
        p0 = None, 
        parallel = False,
        **kwargs, 
    ):

    shape = np.shape(ydata)
    ydata = ydata.reshape((-1,shape[-1]))
    nx, nt = ydata.shape

    if not parallel:
        p = []
        for x in tqdm(range(nx), desc='Fitting pixels'):
            args_x = (model, func_init, xdata, ydata[x,:], p0, bounds, kwargs)
            p_x = _fit_func(args_x)
            p.append(p_x)
    else:
        args = []
        for x in range(nx):
            args_x = (model, func_init, xdata, ydata[x,:], p0, bounds, kwargs)
            args.append(args_x)
        pool = multiprocessing.Pool(processes=num_workers)
        p = pool.map(_fit_func, args)
        pool.close()
        pool.join()

    n = len(p[0])
    par = np.empty((nx, n)) 
    fit = np.empty((nx, nt))
    for x in range(nx):
        par[x,:] = p[x]
        fit[x,:] = model(xdata, *tuple(p[x]))
  
    return fit.reshape(shape), par.reshape(shape[:-1]+(n,))

