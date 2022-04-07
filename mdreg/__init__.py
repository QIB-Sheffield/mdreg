import os
# Use README.md for description in Github Page
with open(os.path.join(".", 'README.md'), encoding='utf-8') as f:
    introduction = f.read()
__doc__ = introduction

from .main import *
__all__ = ['MDReg', 'models']