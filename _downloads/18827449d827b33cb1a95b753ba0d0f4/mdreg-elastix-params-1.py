import mdreg
#
# Adjust the default parameters associated with grid spacing for elastix
# registration.
#
params = mdreg.elastix.params()
print(params['FinalGridSpacingInPhysicalUnits'])
# Expected:
## 50.0
#
# Override the default parameters associated with grid spacing for
# elastix registration.
#
params = mdreg.elastix.params(FinalGridSpacingInPhysicalUnits='5.0')
print(params['FinalGridSpacingInPhysicalUnits'])
# Expected:
## 5.0
