# MDR-Library
 New open-source, platform independent python based library for model-driven registration in quantitative renal MRI.
 Currently developed for and tested on the iBEAt study dataset. 
 Can be utilised for other quantitative MRI studies.
 Models included: T1, T2, T2*, DTI, DWI, DCE-MRI.
 To be extended for: MT, PC-MRI, ASL.
 
 
 Requirements:
 1. Install SimpleElastix -(Installation guide: https://simpleelastix.readthedocs.io/GettingStarted.html) 
 2. Download test DICOM data (here) and unzip file in the 'test_data' folder.


How to use:
Run 'MDR_test_main.py' in the 'test_MDR' folder.
 
 
The MDR library has been developed to simplify use, reduce workflow overhead, and allow generalisability by restructuring the MDR algorithm
into an open-source, platform-independent, python based library for model based registration in quantitative renal MRI.
The prototype MDR algorithm (Tagkalakis F, et al. Model-based motion correction outperforms a model-free method in quantitative renal MRI. Abstract-1383, ISMRM 2021) 
was initially developed using Elastix for co-registration and validated for renal T1-mapping, DTI and DCE against groupwise model-free registration (GMFR). 

The MDR-Library has restructured and extended the prototype MDR implementation to provide a simple and intuitive application programming interface using SimpleElastix.


Acknowledgement:
The iBEAt study is part of the BEAt-DKD project. The BEAt-DKD project has received funding from the Innovative Medicines Initiative 2 Joint Undertaking under grant agreement No 115974. This Joint Undertaking receives support from the European Union’s Horizon 2020 research and innovation programme and EFPIA with JDRF. For a full list of BEAt-DKD partners, see www.beat-dkd.eu

For queries please email:

kanishka.sharma@sheffield.ac.uk or s.sourbron@sheffield.ac.uk
 
