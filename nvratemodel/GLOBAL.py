################### Optimization with numba ##################################
NUMBA_OPT = True
# For debugging purposes (better error reports) and for single executions
# it can be useful to use NUMBA_OPT=False.


################### MACHINE SPECIFIC SETTINGS ################################
from os.path import abspath, dirname, expanduser
from os.path import join as osjoin
ROOTDIR = dirname(abspath(__file__))
# PATH_DATA =  ROOTDIR # alternative, set yourself:
# PATH_DATA = abspath('C:/nvratemodel_data')
PATH_DATA = osjoin(expanduser('~'), 'nvratemodel_data')


################### Common parameter settings ################################
import numpy as np

# References to the parameters used below are given in Ernst2023Modeling:
# https://arxiv.org/abs/2304.02521
Dgs = 2.87771e9  # units: Hz, at T=0K.
# units: Hz; see Chen et al., APL 99, 161903 (2011):
d1, d2, d3, d4, d5 = -4.625e3, 1.067e2, -9.325e-1, 1.739e-3, -1.838e-6
Des_para = 1.44e9 # units: Hz
Des_perp = 1.541e9/2 # units: Hz
Les_para = 5.33e9 # units: Hz
Les_perp = 0.154e9 # units: Hz
gl = 0.1 # units: 1

# orbital g factor for both GS and ES:
B_convfactor_GSandES = 2.003*9.2740100783e-24/6.62607015e-34 # SI units: Hz/T; gs*mu_b/h;

diamondDebyeEnergy = 168e-3 # unit: [eV]
PlakhotnikCutoffEnergy = 13.4e-3 # unit: [eV], found in Palkhotnik2015.
AbtewCutoffEnergy = 50e-3 # unit: [eV], found in Abtew2011.

kE12OVERkA1 = 0.52 # based on Goldman2015

# Default values used here are based on NV2 of https://arxiv.org/abs/2301.05091.
# In contrast to Tab 1 in https://arxiv.org/abs/2304.02521 some values 0.0 
# are replaced to avoid special cases.
Eperp_default      = 40e9 # unit: Hz
phiE_default       = np.radians(24.4) # unit: RAD
B_default          = 1e-3 # unit: T
thetaB_default     = np.radians(1.9) # unit: RAD
phiB_default       = np.radians(194.2) # unit: RAD
kr_default         = 55.7e6 # unit: 1/s
kE12_default       = 98.7e6 # unit: 1/s
kA1_default        = kE12_default/kE12OVERkA1 # unit: 1/s
kExy_default       = 8.2e6 # unit: 1/s
exRatio_default    = 1. # unit: 1
SSRatio_default    = 2.26 # unit: 1
SSPhonE_default    = 16.6e-3 # unit: eV
SSTauDecay_default = 320e-9 # unit: s
T_default          = 300. # unit: K
elPhonCoup_default = 176. # unit: 1/us*1/meV^3
phonCutoff_default = diamondDebyeEnergy # unit: eV
laserpower_default = 2.23e-3 # unit: W
background_default = 27.5e6 # unit: cps/W
colExRatio_default = 88.4e-6 # unit: cps*W*s
opticAlign_default = 136.2 # unit: 1/W
darkcounts_default = 0. # unit: cps
piPulseFid_default = 1. # unit: 1
highT_trf_default  = False # unit: bool


spinCoherentOptics = True # see Fuchs, et al., PRL 108, 157602 (2012).
