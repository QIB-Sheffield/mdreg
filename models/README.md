# Dummy models

**Note:** This is a library of dummy models to illustrate code structure and how MDR is meant to be used. 
Some models themselves are not physically accurate and should not be used for image analysis.
Users are expected to implement their own signal model fits for corresponding MR datasets to use as part of the MDR-Library. 
The supplied models have been tested only on the MR data provided with this library and should only be used as a reference to understand how to apply your own signal model fit to the MDR-Library.

The following models (found in the 'models' folder) have been implemented for the purpose of testing the MDR-Library on the MRI dataset acquired (and supplied with this library) as part of the iBEAt study.

The 'tests' folder within the MDR-Library consists of corresponding python test scripts for each of these models.

**1.** T1.py (WARNING: dummy model only sufficient for MDR - may not be accurate for renal T1 map fit)

T1-MOLLI fitting following: Messroghli DR, Radjenovic A, Kozerke S, Higgins DM, Sivananthan MU, Ridgway JP. Modified Look-Locker inversion recovery (MOLLI) for high-resolution T1 mapping of the heart. Magn Reson Med. 2004 Jul;52(1):141-6. doi: 10.1002/mrm.20110. PMID: 15236377.

Test-script: 'MDR_test_T1.py'

**2.** T2.py (WARNING: dummy model only sufficient for MDR - not accurate for renal T2 map)

Monoexponential T2 decay

Test-script: 'MDR_test_T2.py' 

**3.** T2star.py 

Monoexponential T2* decay

Test-script: 'MDR_test_T2star.py'

**4.** DWI_monoexponential.py

Monoexponential ADC fit

Test-script: 'MDR_test_DWI.py'

**5.** DTI.py

Linearised signal model with return fitted parameters as ADC and FA

Test-script: 'MDR_test_DTI.py'

**6.** constant_model.py

Constant model fit currently defaults for MT sequence (eg: to output MTR)

Test-script: 'MDR_test_MT.py'

**7.** two_compartment_filtration_model_DCE.py

2-compartment filtration model implemented for DCE-MRI

Test-script: 'MDR_test_DCE.py'

# How to implement your own model into the MDR-Library:

Your python script for the model definition is expected to contain the following main function (see below).

`def main(images_to_be_fitted, signal_model_parameters):`

where,  
*images_to_be_fitted (numpy.ndarray)* is the input image at different time-series with shape [x-dim\*y-dim, total time-series].    
and    
*signal_model_parameters (list)* is a list containing independent model-fit parameters as the list elements.   

With the following 2 return parameters for the main function:

*fit (numpy.ndarray):* is the signal model fit at all time-series with shape [x-dim\*y-dim, total time-series].   
and     
*fitted_parameters (numpy.ndarray)* are the output signal model fit parameters stored in a single nd-array with shape [2, x-dim\*y-dim].     
