import os
import numpy as np
import mdreg



def test_constant():

    print('Testing model: constant')

    data = mdreg.fetch('MOLLI')
    fit, pars = mdreg.constant(data['array'][:,:,0,:])
    mdreg.plot.animation(fit, vmin=0, vmax=10000, show=True)


def test_exp_decay():

    print('Testing model: exp_decay')

    data = mdreg.fetch('MOLLI')
    fit, pars = mdreg.exp_decay(data['array'][:,:,0,:], time=np.array(data['TI'])/1000, p0=[1,1.3])
    mdreg.plot.animation(fit, path=os.getcwd(), vmin=0, vmax=10000, show=True)


def test_abs_exp_recovery_2p():

    print('Testing model: abs_exp_recovery_2p')

    data = mdreg.fetch('MOLLI')
    fit, pars = mdreg.abs_exp_recovery_2p(data['array'][:,:,0,:], TI=np.array(data['TI'])/1000)
    mdreg.plot.animation(fit, path=os.getcwd(), vmin=0, vmax=10000, show=True)

    # data['array'] = fit.astype(np.int16)
    # with open(os.path.join(os.getcwd(), 'MOLLIfit.pkl'), 'wb') as fp:
    #     pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == '__main__':

    #test_constant()
    test_exp_decay()
    #test_abs_exp_recovery_2p()

    print('------------------------')
    print('models testing: passed!!')
    print('------------------------')
    