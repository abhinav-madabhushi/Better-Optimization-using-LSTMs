#!/usr/bin/env python
# (C)2011 Arpad Buermen
# (C)2022 Jaroslav Fowkes, Lindon Roberts
# Licensed under GNU GPL V3

#
# Do not edit. This is a computer-generated file.
#

import os

import numpy as np
from glob import glob
from setuptools import setup, Extension, find_packages

#
# OS specific
#


define_macros=[('LINUX', None)]
include_dirs=[np.get_include(),os.environ['CUTEST']+'/include/']
objFileList=glob('*.o')
objFileList.append(os.environ['CUTEST']+'/objects/'+os.environ['MYARCH']+'/double/libcutest.a')
libraries=['gfortran']
library_dirs=[]
extra_link_args=[]


#
# End of OS specific
#

# Module
module = Extension(
    str('_pycutestitf'),
    sources=[str('cutestitf.c')],
    include_dirs=include_dirs,
    define_macros=define_macros,
    extra_objects=objFileList,
    libraries=libraries,
    library_dirs=library_dirs,
    extra_link_args=extra_link_args,
)

# Settings
setup(
    name='PyCUTEst automatic test function interface builder',
    version='1.7.2',
    description='Builds a CUTEst test function interface for Python.',
    long_description='Builds a CUTEst test function interface for Python.',
    author='Arpad Buermen, Jaroslav Fowkes, Lindon Roberts',
    author_email='arpadb@fides.fe.uni-lj.si, fowkes@maths.ox.ac.uk, robertsl@maths.ox.ac.uk',
    url='',
    platforms='Linux',
    license='GNU GPL',
    packages=find_packages(),
    ext_modules=[module],
)
