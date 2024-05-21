import numpy as np
from copy import deepcopy

from nvratemodel.src.core import MEmodel, HighTmodel, LowTmodel, \
        loadModelDictFromFile, makeModelDict, \
        diamondDebyeEnergy, AbtewCutoffEnergy, PlakhotnikCutoffEnergy
        
from nvratemodel.src.simulationRoutines import simulatePulseVsParam

from nvratemodel.GLOBAL import PATH_DATA

def run(path=None):
    """
    Plot Fig. 7 (b) of https://arxiv.org/abs/2304.02521.

    Returns
    -------
    dic : dictionary of results of savePulseVsParam()
    """
    path_save = path # should use PATH_DATA
    # path_save = None # do not save the data

    # Relevant only for Contrast calculation:
    thistint = 'optSNR'
    # thistint = 400e-9

    modeldict = makeModelDict()

    modeldict['phiE'] = np.radians(0) # unit: RAD
    modeldict['B'] = 0e-3 # unit: T
    modeldict['thetaB'] = np.radians(0) # unit: RAD
    modeldict['phiB'] = np.radians(0) # unit: RAD

    argsList = []
    values = [0, 150, 300]
    for val in values:
        modeldicti = deepcopy(modeldict)
        modeldicti['T'] = val
        argsList.append((MEmodel, deepcopy(modeldicti)))
    
    modeldicti = deepcopy(modeldict)
    modeldicti['T'] = 300
    argsList.append((HighTmodel, deepcopy(modeldicti)))
        
    modeldicti = deepcopy(modeldict)
    modeldicti['T'] = 0
    argsList.append((LowTmodel, deepcopy(modeldicti)))
    
    dic = simulatePulseVsParam(path=path_save, plot=True, stackedPlot=True,
                 argsList = argsList, integrationTime=thistint)
