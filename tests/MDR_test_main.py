"""
MODEL DRIVEN REGISTRATION for iBEAt study: quantitative renal MRI
@Kanishka Sharma 2021
@Steven Sourbron 2021
Main script to test Model Driven Registration Library (MDR-Library)
"""

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__))) # Add folder "tests" to PYTHONPATH
sys.path.append(os.path.dirname(sys.path[0])) # Add the parent directory of "tests" to the system

# import the test script for your purpose
# select from currently available test scripts below
# T2:  MDR_test_T2
# T2*: MDR_test_T2star
# DWI: MDR_test_DWI
# DTI: MDR_test_DTI
# DCE: MDR_test_DCE
# example import script for T2* sequence
import MDR_test_T2star

# main function to call MDR
if __name__ == '__main__':
   # example use case for MDR test script using T2star sequence
    MDR_test_T2star.main()
