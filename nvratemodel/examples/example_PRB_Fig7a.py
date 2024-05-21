import numpy as np
from copy import deepcopy

from nvratemodel.src.core import MEmodel, HighTmodel, LowTmodel, \
    loadModelDictFromFile, makeModelDict
    
from nvratemodel.src.simulationRoutines import simulateReadoutVsParam

from nvratemodel.GLOBAL import PATH_DATA

def run(path=None):
    """
    Plot Fig. 7 (a) of https://arxiv.org/abs/2304.02521.

    Returns
    -------
    dic : dictionary of results of saveReadoutVsParam()
    """
    path_save = path # should use PATH_DATA
    # path_save = None # do not save the data
    
    yAxis='PL'
    # yAxis='C'
    # yAxis='normSens'
    # yAxis='Sens'
    # yAxis='initFid'
    # yAxis='readFid'
    # yAxis='SNR'
    
    # Relevant only for readout yAxis types that have an integration time:
    # (is simply ignored with yAxis='PL' here)
    thistint = 'optSNR'
    # thistint = 400e-9

    paramList = np.linspace(0, 200e-3, 200)
    paramName = 'B'
    
    modeldict = makeModelDict()

    modeldict['phiE'] = np.radians(0) # unit: RAD

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
    
    
    dic = simulateReadoutVsParam(yAxis=yAxis, path=path_save, plot=True,
           xParamName=paramName, x=paramList,
           argsList=argsList, integrationTime=thistint)
