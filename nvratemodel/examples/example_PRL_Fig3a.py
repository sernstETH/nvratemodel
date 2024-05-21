import numpy as np
from copy import deepcopy

from nvratemodel.src.core import MEmodel, HighTmodel, LowTmodel, \
    loadModelDictFromFile, makeModelDict
    
from nvratemodel.src.simulationRoutines import simulate2DMap

from nvratemodel.GLOBAL import PATH_DATA, kE12OVERkA1

def run(path=None):
    """
    Plot Fig. 3 (a) of https://arxiv.org/abs/2301.05091.

    Returns
    -------
    dic : dictionary of results of save2DMap()
    """
    path_save = path # should use PATH_DATA
    # path_save = None # do not save the data
    
    modelClass = MEmodel
    # modelClass = HighTmodel
    # modelClass = LowTmodel
    
    zAxis='PL'
    # zAxis='C'
    # zAxis='normSens'
    # zAxis='Sens'
    # zAxis='initFid'
    # zAxis='readFid'
    # zAxis='SNR'
    
    # Relevant only for readout yAxis types that have an integration time:
    # (is simply ignored with yAxis='PL' here)
    thistint = 'optSNR'
    # thistint = 400e-9

    paramList1 = np.linspace(0e-3, 200e-3, 50)
    paramName1 = 'B'
    
    paramList2 = np.linspace(0, 100, 50)
    paramName2 = 'T'
    
    modeldict = makeModelDict()

    modeldict['Eperp'] = 31.8e9 # unit: Hz
    modeldict['phiE'] = np.radians(39.9) # unit: RAD
    modeldict['thetaB'] = np.radians(1.7) # unit: RAD
    modeldict['phiB'] = np.radians(256.2) # unit: RAD
    modeldict['kr'] = 55.1e6 # unit: 1/s
    modeldict['kE12'] = 112.4e6 # unit: 1/s
    modeldict['kA1'] = modeldict['kE12']/kE12OVERkA1 # unit: 1/s
    modeldict['kExy'] = 9.1e6 # unit: 1/s
    modeldict['SSRatio'] = 1.36 # unit: 1
    modeldict['SSTauDecay'] = 342e-9 # unit: s
    modeldict['elPhonCoup'] = 197. # unit: 1/us*1/meV^3
    modeldict['background'] = 40.8e6 # unit: cps/W
    modeldict['colExRatio'] = 67.1e-6 # unit: cps*W*s
    modeldict['opticAlign'] = 245.1 # unit: 1/W
    
    # Relevant only for readout yAxis types that have an integration time:
    # (is simply ignored with yAxis='PL' here)
    thistauR = 23e-9 # unit: s
    
    dic = simulate2DMap(zAxis=zAxis, path=path_save, plot=True,
                 xParamName=paramName1, x=paramList1, 
                 yParamName=paramName2, y=paramList2,
                 modelClass = modelClass,
                 integrationTime = thistint, tauR = thistauR,
                 **modeldict)
