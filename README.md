# MDR-Library
New open-source, platform independent python based library for model-driven registration in quantitative renal MRI.

Currently developed for and tested on the iBEAt study dataset. Can be utilised for other quantitative MRI studies.

Models included: T1, T2, T2*, DTI, DWI, DCE-MRI. To be extended for: MT, PC-MRI, ASL.
 
 
## Requirements
1. Have [`Python >= 3.6`](https://www.python.org/) installed and type `pip install mdr-library`. If you are a developer, then you can run `pip install -r requirements.txt` in a terminal or in an IDE of your preference. This will install the Python Packages required to run this library, with special focus on [ITK-Elastix](https://github.com/InsightSoftwareConsortium/ITKElastix) which is the one used for the model-driven registration.

2. Download test [DICOM data](https://shorturl.at/rwCUV), unzip file and place it in the 'tests/test_data' folder.


## How to use
Run `MDR_test_main.py` in the `tests` folder. Read the code in the other python files in the `tests` folder to find examples of how to use the `MDR` package.


## Context
The MDR library has been developed to simplify use, reduce workflow overhead, and allow generalisability by restructuring the MDR algorithm into an open-source, platform-independent, python based library for model based registration in quantitative renal MRI.

The prototype MDR algorithm (Tagkalakis F, et al. Model-based motion correction outperforms a model-free method in quantitative renal MRI. Abstract-1383, ISMRM 2021) was initially developed using Elastix for co-registration and validated for renal T1-mapping, DTI and DCE against groupwise model-free registration (GMFR). The prototype MDR algorithm has been restructured and extended into the MDR-Library to provide a simple and intuitive application programming interface using ITK-Elastix.

For more details about the code, please consult the [Reference Manual](https://qib-sheffield.github.io/MDR-Library/)


**Updating the Reference Manual:** 

First, run the command `pdoc --html --force --output-dir "docs" "models" "tests" "MDR"` in the parent folder. Commit the newly generated files to Github so that the [website](https://qib-sheffield.github.io/MDR-Library/) is refreshed.


## Acknowledgement
The iBEAt study is part of the BEAt-DKD project. The BEAt-DKD project has received funding from the Innovative Medicines Initiative 2 Joint Undertaking under grant agreement No 115974. This Joint Undertaking receives support from the European Union’s Horizon 2020 research and innovation programme and EFPIA with JDRF. For a full list of BEAt-DKD partners, see www.beat-dkd.eu.

**For queries please email:** kanishka.sharma@sheffield.ac.uk or s.sourbron@sheffield.ac.uk
