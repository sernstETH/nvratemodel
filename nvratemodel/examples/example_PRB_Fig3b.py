import numpy as np
from copy import deepcopy

from os.path import abspath, dirname
from os.path import join as osjoin
import sys
sys.path.insert(0, abspath(osjoin(dirname(__file__), '..'))) # add root directory to import paths

from src.core import MEmodel, HighTmodel, LowTmodel, \
        loadModelDictFromFile, makeModelDict, \
        diamondDebyeEnergy, AbtewCutoffEnergy, PlakhotnikCutoffEnergy, \
        makeTstepsFromDurations, piPulse, LindbladOp_GS_msp1_ypiPulse_EZ
        
from src.simulationRoutines import simulatePopTimeTrace

from GLOBAL import PATH_DATA

def run(path=None):
    """
    Plot Fig. 3 (b) of https://arxiv.org/abs/2304.02521.

    Returns
    -------
    results of savePopTimeTrace()
    """
    path_save = path # should use PATH_DATA
    # path_save = None # do not save the data
    
    modelClass = MEmodel
    # modelClass = LowTmodel
    # modelClass = HighTmodel
    
    basisName='EZ' # 'EZ', 'ZF', 'HF', 'ZB', 'avgEZ'
    # basisName='avgEZ' # for room temperature levels
    
    explainStr=r'$\pi$-pulse applied before the last laser pulse'
    
    modeldict = makeModelDict()

    modeldict['phiE'] = np.radians(0) # unit: RAD
    modeldict['B'] = 0e-3 # unit: T
    modeldict['thetaB'] = np.radians(0) # unit: RAD
    modeldict['phiB'] = np.radians(0) # unit: RAD
    modeldict['T'] = 0 # unit: K
    
    # make modeldicts for laser on and off:
    modeldict_Off = deepcopy(modeldict)
    modeldict_Off['laserpower'] = 0
    modeldict_On = deepcopy(modeldict)

    # start in thermal state:
    state0 = modelClass(**modeldict_Off).steadyState()
    
    # set the time steps:
    ksteps = [
        modelClass(**modeldict_Off),
        modelClass(**modeldict_On),
        modelClass(**modeldict_Off),
        modelClass(**modeldict_On),
        modelClass(**modeldict_Off),
        modelClass(**modeldict_On),
        modelClass(**modeldict_Off),
        ]
    tdurations = [ # unit: s
        0.1e-6,
        0.9e-6,
        2e-6,
        0.9e-6,
        2e-6,
        0.9e-6,
        2e-6,
        ]
    tsteps = makeTstepsFromDurations(tdurations)
    times = np.linspace(0, tsteps[-1], num=3000) # unit: s
    
    # set the pi pulse as a jump operation:
    # Note: jumpOps have to be in EZ basis.
    # Note: jumpOps only work with MEmodel, for classical models one has to 
    # run savePopTimeTrace() twice with the respective start states: once with
    # and once without a pi-pulse piPulse(state0) applied.
    # NOTE: this is only a correct pi-pulse if EZ basis has same states as EIG basis in the GS.
    jumpTimes = [tsteps[-3]-0.2e-6,] # unit: s
    jumpOps = [LindbladOp_GS_msp1_ypiPulse_EZ,]


    axes1, axes2, dic, thispath = simulatePopTimeTrace(
        times, tsteps, ksteps, state0,
        path=path_save, basisName=basisName,
        explainStr=explainStr,
        jumpTimes=jumpTimes, jumpOps=jumpOps)
