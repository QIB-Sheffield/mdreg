
import numpy as np

from mdreg import utils

    


def constant(ydata):
    shape = np.shape(ydata)
    avr = np.mean(ydata, axis=-1) 
    par = avr.reshape(shape[:-1] + (1,))
    fit = np.repeat(avr[...,np.newaxis], shape[-1], axis=-1)
    return fit, par



def _exp_decay_func(t, S, T):
    return S*np.exp(-t/T)

def _exp_decay_func_init(t, signal, init):
    S0 = np.amax([np.amax(signal),0])
    return [S0*init[0], init[1]]
    
def exp_decay(ydata, 
        time = None,
        bounds = (
            [0,0],
            [np.inf, np.inf], 
        ),
        p0 = [1,1], 
        parallel = False,
        **kwargs):
    
    if time is None:
        raise ValueError('time is a required argument.')
    
    return utils.fit_pixels(ydata, 
        model = _exp_decay_func, 
        xdata = np.array(time),
        func_init = _exp_decay_func_init,
        bounds = bounds,
        p0 = p0, 
        parallel = parallel,
        **kwargs)



def _abs_exp_recovery_2p_func(t, S0, T1):
    return np.abs(S0 * (1 - 2 * np.exp(-t/T1)))

def _abs_exp_recovery_2p_func_init(t, signal, init):
    S0 = np.amax(np.abs(signal))
    return [S0*init[0], init[1]]

def abs_exp_recovery_2p(signal, 
        TI = None,
        bounds = (
            [0,0],
            [np.inf, np.inf], 
        ),
        p0 = [1,1.3], 
        parallel = False,
        **kwargs):
    
    if TI is None:
        raise ValueError('TI is a required parameter.')
    
    return utils.fit_pixels(signal, 
        model=_abs_exp_recovery_2p_func, 
        xdata = np.array(TI),
        func_init = _abs_exp_recovery_2p_func_init,
        bounds = bounds,
        p0 = p0, 
        parallel = parallel,
        **kwargs)



def _exp_recovery_2p_func(t, S0, T1):
    return S0 * (1 - 2 * np.exp(-t/T1))

def _exp_recovery_2p_func_init(t, signal, init):
    S0 = np.amax(np.abs(signal))
    return [S0*init[0], init[1]]

def exp_recovery_2p(signal, 
        TI = None,
        bounds = (
            [0,0],
            [np.inf, np.inf], 
        ),
        p0 = [1,1.3], 
        parallel = False,
        **kwargs):
    
    if TI is None:
        raise ValueError('TI is a required parameter.')
    
    return utils.fit_pixels(signal, 
        model=_exp_recovery_2p_func, 
        xdata = np.array(TI),
        func_init = _exp_recovery_2p_func_init,
        bounds = bounds,
        p0 = p0, 
        parallel = parallel,
        **kwargs)




def _abs_exp_recovery_3p_func(t, S0, T1, eff):
    return np.abs(S0 * (1 - eff * np.exp(-t/T1)))

def _abs_exp_recovery_3p_func_init(t, signal, init):
    S0 = np.amax(np.abs(signal))
    return [S0*init[0], init[1], init[2]]
    
def abs_exp_recovery_3p(signal, 
        TI = None,
        bounds = (
            [0,0,0],
            [np.inf, np.inf, 2], 
        ),
        p0 = [1,1.3,2], 
        parallel = False,
        **kwargs):
    
    if TI is None:
        raise ValueError('TI is a required parameter.')
    
    return utils.fit_pixels(signal, 
        model = _abs_exp_recovery_3p_func, 
        xdata = np.array(TI),
        func_init = _abs_exp_recovery_3p_func_init,
        bounds = bounds,
        p0 = p0, 
        parallel = parallel,
        **kwargs)



def _exp_recovery_3p_func(t, S0, T1, eff):
    return S0 * (1 - eff * np.exp(-t/T1))

def _exp_recovery_3p_func_init(t, signal, init):
    S0 = np.amax(np.abs(signal))
    return [S0*init[0], init[1], init[2]]
    
def exp_recovery_3p(signal, 
        TI = None,
        bounds = (
            [0,0,0],
            [np.inf, np.inf, 2], 
        ),
        p0 = [1,1.3,2], 
        parallel = False,
        **kwargs):
    
    if TI is None:
        raise ValueError('TI is a required parameter.')
    
    return utils.fit_pixels(signal, 
        model = _exp_recovery_3p_func, 
        xdata = np.array(TI),
        func_init = _exp_recovery_3p_func_init,
        bounds = bounds,
        p0 = p0, 
        parallel = parallel,
        **kwargs)
