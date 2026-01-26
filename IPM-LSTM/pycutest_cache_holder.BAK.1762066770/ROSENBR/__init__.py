# PyCutest problem interface module initialization file
# (C)2011 Arpad Buermen
# (C)2022 Jaroslav Fowkes, Lindon Roberts
# Licensed under GNU GPL V3

"""Interface module for CUTEst problem ROSENBR with ordering
  efirst=False, lfirst=False, nvfirst=False
sifdecode parameters : 
sifdecode options    : 

Available interface functions (should not be called directly):
setup      -- setup problem and get problem information
dims       -- get problem dimensions
varnames   -- get names of problem's variables
connames   -- get names of problem's constraints
objcons    -- objective and constraints
obj        -- objective and objective gradient
cons       -- constraints and constraints gradients/Jacobian
lagjac     -- gradient of objective/Lagrangian and constraints Jacobian
jprod      -- product of constraints Jacobian with a vector
hess       -- Hessian of objective/Lagrangian
ihess      -- Hessian of objective/constraint
hprod      -- product of Hessian of objective/Lagrangian with a vector
gradhess   -- gradient and Hessian of objective (unconstrained problems) or
               gradient of objective/Lagrangian, Jacobian of constraints and
               Hessian of Lagrangian (constrained problems)
scons      -- constraints and sparse Jacobian of constraints
slagjac    -- gradient of objective/Lagrangian and sparse Jacobian
sphess     -- sparse Hessian of objective/Lagrangian
isphess    -- sparse Hessian of objective/constraint
gradsphess -- gradient and sparse Hessian of objective (unconstrained probl.)
               or gradient of objective/Lagrangian, sparse Jacobian of
               constraints and sparse Hessian of Lagrangian (constrained probl.)
report     -- get usage statistics
terminate  -- clear problem memory
"""

from ._pycutestitf import *
from . import _pycutestitf

def setup():
    """
    Set up the problem and get problem information.

    info=setup()

    info -- dictionary with the summary of test function's properties (see getinfo())
    """
    import os

    # Get the directory where the binary module (and OUTSDIF.d) are found.
    (_directory, _module)=os.path.split(_pycutestitf.__file__)

    # Problem info structure and dimension
    info=None
    n=None
    m=None

    # Constraints and variable ordering
    efirst=False
    lfirst=False
    nvfirst=False

    # Remember current directory and go to module directory where OUTSDIF.d is located
    fromDir=os.getcwd()
    os.chdir(_directory)

    # Get problem dimension
    (n, m)=_pycutestitf.dims()

    # Set up the problem and get basic information
    info=_pycutestitf.setup(efirst, lfirst, nvfirst)

    # Store constraint and variable ordering information
    if m>0:
        info['efirst']=efirst
        info['lfirst']=lfirst
    info['nvfirst']=nvfirst

    # Store sifdecode parameters and options
    info['sifparams']=None
    info['sifoptions']=None

    # Go back to initial directory
    os.chdir(fromDir)

    return info

