"""
@author: Kanishka Sharma - modified for mdreg from implementation of @Fotios Tagkalakis   
iBEAt study DTI model fit for mdreg   
2021 
"""

import numpy as np

def pars():
    return ['FA', 'ADC']

def bounds():
    lower = [0,0]
    upper = [1.0, 10]
    return lower, upper


def DTI_fitting(im, b, thresh_val, method='linear'):
    """ DTI fit function.  

    Args:
    ----
    im (numpy.ndarray): input image at all time-series (i.e. at each b-value and direction) with shape [x-dim*y-dim, total time-series].  
    b (list): list of b_values  
    thresh_val (numpy.ndarray): B matrix  

    Returns
    -------
    fit (numpy.ndarray): signal model fit at all time-series with shape [x-dim*y-dim, total time-series].    
    DiffusionTensor (M), Bv_new, FA (Fractional Anisotropy), and ADC(mm2/sec*10^-3): fitted parameters 
    """
   
    sz = np.shape(im)
    if len(sz)==2 and sz[1]==1:
        im = np.transpose(im)
        sz = np.shape(im)

    # reshape to a matrix
    im_matrix = np.reshape(im, (-1,sz[-1]))
    
    # get the mask
    mask = np.array(thresh_val, dtype=bool)

    # take only the voxels inside the mask
    mask = np.reshape(mask,(-1, sz[-1]))
    I = im_matrix[mask]
    
    if not np.all(np.isreal(I)):
        # Some voxels are complex. Taking magnitude.'
        I = np.abs(I)
        
    # take the log of the image to linearise the equation
    abs_I = np.abs(I)
    imlog = np.ma.log(abs_I)
    imlog = np.reshape(imlog, (-1, sz[-1]))
    
    # Sort all b matrices in to a vector Bv=[Bxx,2*Bxy,2*Bxz,Byy,2*Byz,Bzz];
    Bv = np.vstack((b[0,0,:],
                    2*b[0,1,:],
                    2*b[0,2,:],
                    b[1,1,:],
                    2*b[1,2,:],
                    b[2,2,:]))
    
    Bv = np.transpose(Bv)
    
    # Add another column to Bv to handle the constant term:
    # Slog = Bv * M + log(S0)
    # becomes:
    # Slog = [Bv, -1] * [M; -log(S0)]
    minus_one_column = -np.ones((np.shape(Bv)[0]))
    Bv_new = np.c_[Bv, minus_one_column]
    assert method=='linear', "Wrong method as argument!!!"
    M = np.linalg.lstsq(Bv_new, -imlog.T, rcond=None)
    M = M[0].T
    
    M[np.isnan(M)]=0
    M[np.isinf(M)]=0
    
    ### Initialize Variables
    npixels = np.shape(M)[0]
    FA_mask = np.empty(npixels) 
    ADC_mask = np.empty(npixels) 

    for i in range(npixels):
        
        # The DiffusionTensor (Remember it is a symetric matrix,
        # thus for instance Dxy == Dyx)
        # DiffusionTensor=[Mi[0] Mi[1] Mi[2]; Mi[1] Mi[3] Mi[4]; Mi[2] Mi[4] Mi[5]]
        DiffusionTensor = np.zeros((3,3))
        DiffusionTensor[0][0] = M[i, 0]
        DiffusionTensor[0][1] = M[i, 1]
        DiffusionTensor[0][2] = M[i, 2]
        DiffusionTensor[1][0] = M[i, 1]
        DiffusionTensor[1][1] = M[i, 3]
        DiffusionTensor[1][2] = M[i, 4]
        DiffusionTensor[2][0] = M[i, 2]
        DiffusionTensor[2][1] = M[i, 4]
        DiffusionTensor[2][2] = M[i, 5]

        # Calculate the eigenvalues and vectors, and sort the 
        # eigenvalues from small to large
        [EigenValues, EigenVectors]=np.linalg.eig(DiffusionTensor)
        if np.sum(EigenValues)!=0:
            EigenValues, EigenVectors = zip(*sorted(zip(EigenValues, EigenVectors)))
        
        # Regulating of the eigen values (negative eigenvalues are
        # due to noise and other non-idealities of MRI)
        EigenValues=np.abs(EigenValues)

        # Apparent Diffuse Coefficient
        ADCv = np.mean(EigenValues)

        # FA definition:
        denominator = np.sqrt(EigenValues[0]**2+EigenValues[1]**2+EigenValues[2]**2)
        if denominator == 0:
            FA_mask[i] = 0
        else:    
            FA_mask[i]=np.sqrt(1.5)*(np.sqrt((EigenValues[0]-ADCv)**2+(EigenValues[1]-ADCv)**2+(EigenValues[2]-ADCv)**2)/denominator)
        ADC_mask[i]=ADCv

    Bv_new_times_M_new = np.moveaxis(np.dot(Bv_new, M.T),0,-1).reshape(sz) 
    fit = np.exp(-Bv_new_times_M_new)
    
    par = np.empty((npixels, 2))
    par[:,0] = FA_mask
    par[:,1] = ADC_mask
    
    return fit, par


def main(images_to_be_fitted, signal_model_parameters):
    """ main function to perform DTI model fit. 

    Args
    ----
    images_to_be_fitted (numpy.ndarray): input image at all time-series (i.e. at each b-value and all directions) with shape [x-dim*y-dim, total time-series].    
    signal_model_parameters (list): list containing 'b-values', 'b-vectors' and 'image_orientation_patient' as list elements.    

    Returns
    -------
    fit (numpy.ndarray): signal model fit at all time-series (i.e. at each b-value and direction) with shape [x-dim*y-dim, total time-series].   
    fitted_parameters (numpy.ndarray): output signal model fit parameters 'FA' and 'ADC' stored in a single nd-array with shape [2, x-dim*y-dim].   
    """
    b_values = signal_model_parameters[0]
    bVec_original = signal_model_parameters[1]
    image_orientation_patient = signal_model_parameters[2]
 
    R1 = image_orientation_patient[0][3:6]
    R1 = [-float(x) for x in R1] 

    R2 = image_orientation_patient[0][0:3]
    R2 = [-float(x) for x in R2]

    R3 = np.cross(R1, R2)

    R = np.vstack((R1, R2, R3))
    bVec = np.dot(R, np.array(bVec_original).T).T
    
    ### Mask
    sz_mask = np.shape(images_to_be_fitted)
    
    mask = np.ones(sz_mask)

    ### b matrix
    b = np.zeros((3, 3, len(b_values)))
    for idx_b in range(len(b_values)):
        b[:, :, idx_b] = np.outer(np.outer(b_values[idx_b], bVec[idx_b,:].T), bVec[idx_b,:])
  
    return DTI_fitting(images_to_be_fitted, b, mask, 'linear')

