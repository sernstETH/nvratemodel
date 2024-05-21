import numpy as np
from copy import deepcopy

from nvratemodel.src.core import MEmodel, HighTmodel, LowTmodel, \
        loadModelDictFromFile, makeModelDict, \
        diamondDebyeEnergy, AbtewCutoffEnergy, PlakhotnikCutoffEnergy
        
from nvratemodel.src.simulationRoutines import simulateReadoutVsParam

from nvratemodel.GLOBAL import PATH_DATA

def run(path=None):
    """
    Plot Fig. 4 (h) with aspects of Fig. 10 (a) and Fig. 5 (b)
    of https://arxiv.org/abs/2304.02521.

    Returns
    -------
    dic : dictionary of results of saveReadoutVsParam()
    """
    path_save = path # should use PATH_DATA
    # path_save = None # do not save the data
    
    # yAxis='PL'
    # yAxis='C'
    # yAxis='normSens'
    # yAxis='Sens'
    # yAxis='initFid'
    # yAxis='readFid'
    yAxis='SNR'
    
    # Relevant only for readout yAxis types that have an integration time:
    thistint = 'optSNR'
    # thistint = 400e-9
    
    paramList = np.linspace(0, 300, 100)
    paramName = 'T'
    
    modeldict = makeModelDict()

    modeldict['phiE'] = np.radians(0) # unit: RAD
    modeldict['B'] = 0e-3 # unit: T
    modeldict['thetaB'] = np.radians(0) # unit: RAD
    modeldict['phiB'] = np.radians(0) # unit: RAD

    argsList = []
    values = [diamondDebyeEnergy, AbtewCutoffEnergy, PlakhotnikCutoffEnergy]
    for val in values:
        modeldicti = deepcopy(modeldict)
        modeldicti['phonCutoff'] = val
        argsList.append((MEmodel, deepcopy(modeldicti)))
    
    argsList.append((HighTmodel, deepcopy(modeldict)))
    argsList.append((LowTmodel, deepcopy(modeldict)))
    
    dic = simulateReadoutVsParam(yAxis=yAxis, path=path_save, plot=True,
           xParamName=paramName, x=paramList,
           argsList=argsList, integrationTime=thistint)
