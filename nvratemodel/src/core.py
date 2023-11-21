import numpy as np
from scipy.linalg import expm
from scipy.integrate import quad
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec # used in plotRoutines.py

from copy import deepcopy
from time import strftime # used in plotRoutines.py
from json import load, dump


# Configure paths for LUT and saving/loading data:
from os.path import abspath, dirname, pardir
from os.path import join as osjoin
import sys
ROOTDIR = abspath(osjoin(dirname(__file__), pardir))
sys.path.insert(0, ROOTDIR) # add root directory to import paths

PATH_LUT = ROOTDIR
from GLOBAL import PATH_DATA # used in plotRoutines.py


# Load routines:
from src.auxRoutines import ensure_dir, getmylinestyle, getmycolor, getIndx, \
    sortarrays, printMatrix  # used in plotRoutines.py


# Configure numba:
import numba as nb
from GLOBAL import NUMBA_OPT
jitOpt = nb.jit(nopython=True) if NUMBA_OPT else lambda x: x
# For debugging purposes (better error reports) and for single executions
# it can be useful to use NUMBA_OPT=False.


# Load globally accessible default NV center parameters:
from GLOBAL import Dgs, Des_para, Des_perp, Les_para, Les_perp, gl, d1, d2, d3, d4, d5, \
    convfactor_GSandES, diamondDebyeEnergy, PlakhotnikCutoffEnergy, \
    AbtewCutoffEnergy, kE12OVERkA1, \
    Eperp_default, phiE_default, B_default, thetaB_default, phiB_default, \
    kr_default, kE12_default, kA1_default, kExy_default, exRatio_default, \
    SSRatio_default, SSPhonE_default, SSTauDecay_default, T_default, \
    elPhonCoup_default, phonCutoff_default, laserpower_default, \
    background_default, colExRatio_default, opticAlign_default, \
    darkcounts_default, piPulseFid_default, highT_trf_default
    

#######################
#%% Matrix operations
#######################

@jitOpt
def inverse(M):
    """Wrapper for np.linalg.inv(M)"""
    return np.linalg.inv(M)

@jitOpt
def conjTransp(M):
    """
    For trafo matrices T_XtoY, this can be used instead of inverse() and is
    a bit faster.
    """
    return M.conjugate().transpose()

@jitOpt # optional argument here does not make numba slower.
def basisTrafo(M_old, T_new_to_old, T_old_to_new=None):
    """
    Return matrix M expressed in the new basis. M_new out, M_old in.
    T_new_to_old=T^-1 is matrix representation of the transformation
    from *new basis to old*.
    I.e. T^-1 contains as columns (ith column is T_new_to_old[:,i])
    the new basis vectors expressed in the old basis.
    In common language, T_old_to_new=T, and M_new = T M_old T^-1.
    
    NOTE: Both arguments should have the same dtype.
    """
    if T_old_to_new is None:
        T_old_to_new = conjTransp(T_new_to_old)
    T_old_to_new = np.ascontiguousarray(T_old_to_new) # the memory order can be 'A' but should be 'C' for other numba functions to work with it.
    return np.dot(T_old_to_new, np.dot(M_old,T_new_to_old))

@jitOpt
def get_T_XtoY(T_XtoZ, T_YtoZ):
    """Given trafo matrices T_XtoZ, T_YtoZ, return the trafo matrix T_XtoY."""
    T_ZtoY = conjTransp(T_YtoZ)
    return np.dot(T_ZtoY, T_XtoZ)

@jitOpt 
def expectationValue(O, ket):
    """
    Return the expectation value of the operator O for the state ket.
    This is <bra=ket^*T|O^hat|ket>.
    Make sure they are written in the same basis.
    """
    ketdagg = conjTransp(ket)
    ket = np.ascontiguousarray(ket) # the memory order can be 'A' when result of eigh() is used, but should be 'C' for numba np.dot functions.
    Oket = np.dot(O, ket)
    return np.dot(ketdagg, Oket)

# np.block is not supported by numba, thus write differently:
@jitOpt
def compositeDiagMatrix(A, B):
    """Put matrices A and B on the diagonal of a matrix M."""
    a1, a2 = A.shape
    b1, b2 = B.shape
    M = np.zeros((a1+b1, a2+b2), dtype=A.dtype)
    M[0:a1, 0:a2] = A
    M[a1:, a2:] = B
    return M

@jitOpt
def eig(M):
    """
    For any square matrix. For Hamiltonian see eigh().
        
    NOTE: When feeding the result into another numba jited function, it might
    be required to transform to contiguous memory order
        
        ev =  np.ascontiguousarray(ev)
        
        evec =  np.ascontiguousarray(evec)
        
    since the np.ascontiguousarray(...) below might be ignored by numba.
    """
    ev, evec = np.linalg.eig(M)
    # the memory order is 'A' but should be 'C' for other numba functions to work with it:
    ev =  np.ascontiguousarray(ev)
    evec =  np.ascontiguousarray(evec)
    return ev, evec

@jitOpt
def eigh(M):
    """
    For Hermitian matrices M.
    Return a numpy array of eigenvalues (ordered low to high)
    and a 2d array containing eigenvectors as columns.
    This means associated to ev[i] is evec[:,i].
    
    NOTE: When feeding the result into another numba jited function, it might
    be required to transform to contiguous memory order
    
        ev =  np.ascontiguousarray(ev)
        
        evec =  np.ascontiguousarray(evec)
        
    since the np.ascontiguousarray(...) below might be ignored by numba.
    """
    ev, evec = np.linalg.eigh(M)
    # the memory order is 'A' but should be 'C' for other numba functions to work with it:
    ev =  np.ascontiguousarray(ev)
    evec =  np.ascontiguousarray(evec)
    return ev, evec


################
#%% States
################

class BasisStateNames(dict):
    """
    basisStateNames['xx'][i] gives the name of ith state of the basis 'xx'.

    The naming of the various bases \
    (see basisStateNames.keys()) relates to the setting in which they are \
    approximately eigenstates of the NV centers Hamiltonian.
    Generally, for each basis, the first 3 states are states of the \
    electronic/optical ground state GS. The last state is always the shelving \
    state SS. All other states in between are of the excited state ES. The \
    ES has 6 levels. But at elevated temperature, these are averaged to 3 levels, \
    which is the purpose of the averaged EZ basis 'avgEZ'. The HighTmodel works \
    with the 'avgEZ' basis, the LowTmodel and MEmodel work with the other bases.
    Another abbreviation used here is ISC for the inter-system crossing transition
    of the rate model from the ES to the SS.

    Available bases and their names:

    - The 'EZ' basis is the most common basis, and used for states of the \
    MEmodel. Its states are eigenstates of the spin S_z and orbit sigma_z \
    operator, which gives the name of the e_z unit vector 'EZ'. They are further \
    approximately eigenstates at high magnetic field along the z-axis and high \
    Eperp along the x-axis, which is indicated by the '0' in the states names.

    - The 'ZF' basis states are approximately eigenstates at zero magnetic field \
    and Eperp. Thus 'ZF' for zero-field.

    - The 'HF' basis states are approximately eigenstates at high magnetic field \
    along the z-axis and in-plane strain Eperp. Thus 'HF' for high-field.

    - The 'ZB' basis states are approximately eigenstates at zero magnetic field \
    and high in-plane strain Eperp. Thus 'ZB' for zero-B-field.

    - The 'avgEZ' basis is the 'EZ' basis after averaging over the orbital \
    sub-space and commonly used at room temperature.

    - The 'EIG' basis name is used for the actual eigenstates of the Hamiltonian \
    under given conditions. They are ordered from low to high energy per GS, ES, SS.

    For more information, see https://arxiv.org/abs/2304.02521.
    """
    pass

basisStateNames = BasisStateNames()
basisStateNames['EIG'] = tuple([rf'$A_{{eig, {idx+1}}}$' for idx in range(3)]+[rf'$E_{{eig, {idx+1}}}$' for idx in range(6)]+[r'SS']) # tuple(['eigSt {}'.format(idx) for idx in range(9)]+[r'SS'])
basisStateNames['EZ'] = (r'$A_{+1}$', r'$A_{0}$', r'$A_{-1}$',
                         r'$E_{x,+1}^0$', r'$E_{x,0}^0$', r'$E_{x,-1}^0$',
                         r'$E_{y,+1}^0$', r'$E_{y,0}^0$', r'$E_{y,-1}^0$',
                         r'SS') # see Doherty2013
basisStateNames['ZF'] = (r'$A_{0}$', r'$A_{-1}$', r'$A_{+1}$',
                         r'$E_{1}$', r'$E_{2}$',
                         r'$E_{y,0}^0$', r'$E_{x,0}^0$',
                         r'$A_{1}$', r'$A_{2}$',
                         r'SS') # see Doherty2011 Tab 1
basisStateNames['HF'] = (r'$A_{+1}$', r'$A_{0}$', r'$A_{-1}$',
                         r'$E_{x,+1}$', r'$E_{x,0}$', r'$E_{x,-1}$',
                         r'$E_{y,+1}$', r'$E_{y,0}$', r'$E_{y,-1}$',
                         r'SS') # higher energy branch is called Ex
basisStateNames['ZB'] = (r'$A_{+1}$', r'$A_{0}$', r'$A_{-1}$',
                         r'$Sx_{x}$', r'$Sy_{x}$', r'$Sz_{x}$',
                         r'$Sx_{y}$', r'$Sy_{y}$', r'$Sz_{y}$',
                         r'SS') # higher energy branch is called x.
                         # Based on https://arxiv.org/abs/2304.02521 Fig. 2 (b) for high Eperp, low B field:
                         # r'$\|{-}\rangle = \|{S_x}\rangle$' and r'$\|{+}\rangle = \|{S_y}\rangle$' and r'$\|{0}\rangle = \|{S_z}\rangle$'
basisStateNames['avgEZ'] = (r'$A_{+1}$', r'$A_{0}$', r'$A_{-1}$',
                         r'$E_{+1}$', r'$E_{0}$', r'$E_{-1}$',
                         r'SS')

def printPop(basisName, Pop_basis, decimals=4):
    """
    Print a population vector with names of the levels.
    Works on return values of NVrateModel.population().
    """
    for i in range(Pop_basis.size):
        name,pop = basisStateNames[basisName][i], Pop_basis[i]
        if pop == 0:
            element = '0'
        else:
            element = f'{pop:.{decimals}f}'
        if i == Pop_basis.size-1: # treat the SS correctly also for avgEZ EIG states.
            name = basisStateNames[basisName][-1]
        print(name, '\t:\t', element)



kB = 8.6173e-5  # Boltzman constant, unit: eV/K
sq2 = 1/np.sqrt(2)
h = 4.135667696e-15 # Plank constant to convert energy in GHz to eV. Units: eV/Hz
hbar = h/(2*np.pi) # Units: eV/(1/s)

S_z_opp = np.array([[  1,  0,  0],
                    [  0,  0,  0],
                    [  0,  0, -1]], dtype=(np.complex128))
S_x_opp      = np.array([[  0,sq2,  0],
                    [sq2,  0,sq2],
                    [  0,sq2,  0]], dtype=(np.complex128))
S_y_opp      = np.array([[     0,-sq2*1j,      0],
                    [sq2*1j,      0,-sq2*1j],
                    [     0, sq2*1j,     0]], dtype=(np.complex128))
sigma_z = np.array([[ 1,  0],
                    [ 0, -1]], dtype=(np.complex128))
sigma_x = np.array([[ 0,  1],
                    [ 1,  0]], dtype=(np.complex128))
sigma_y = np.array([[ 0,-1j],
                    [1j,  0]], dtype=(np.complex128))
Id2     = np.eye(2, dtype=(np.complex128))
Id3     = np.eye(3, dtype=(np.complex128))
Id6     = np.eye(6, dtype=(np.complex128))
Id7     = np.eye(7, dtype=(np.complex128))
Id9     = np.eye(9, dtype=(np.complex128))
Id10    = np.eye(10, dtype=(np.complex128))

def purerho(idx, dim=10):
    """
    Return a MEmodel state rho[idx, idx] = 1. else 0.
    """
    rho = np.zeros((dim,dim), dtype=np.complex128)
    rho[idx, idx] = 1.
    return rho

def vecToDensityMatrix(vec):
    """Create a MEmodel state rho from a classical model state vector of
    populations."""
    rho = np.outer(vec, conjTransp(vec))
    return rho


@jitOpt
def removeSSDim(M):
    """See addSSDim()"""
    return np.ascontiguousarray(M[:-1,:-1])

@jitOpt
def addSSDim_entry(M, entry):
    """
    Adds one column and one row to the matrix M at the end and puts entry
    on the diagonal element.
    Use this to add the shelving state SS to the matrices of the 9 level
    Hamilton of ground state GS and excited state ES.
    """
    M_withSS = np.zeros((M.shape[0]+1, M.shape[1]+1), dtype=M.dtype)
    M_withSS[:-1,:-1] = M # shelving state is the last index.
    M_withSS[-1,-1] = entry
    return M_withSS

@jitOpt
def addSSDim(M):
    """
    Adds one column and one row to the matrix M at the end and puts entry 1
    on the diagonal element.
    Use this to add the shelving state SS to the eigenstate matrices of the 9
    level Hamilton of ground state GS and excited state ES.
    """
    return addSSDim_entry(M, 1.0)


################################
#%% Low temperature Hamiltonian
################################

@jitOpt
def B_Hz2T(B_GHz):
    """
    Input: magnetic field [GHz]
    
    Output: magnetic field [T]
    """
    return B_GHz/(convfactor_GSandES)

@jitOpt
def B_T2Hz(B_T):
    """
    Input: magnetic field [GHz]
    
    Output: magnetic field [T]
    """
    return B_T*(convfactor_GSandES)

@jitOpt
def polar2cart(r, theta, phi):
    """
    angles are in RAD.
    
    Return: (x, y, z)
    """
    return (
         r * np.sin(theta) * np.cos(phi),
         r * np.sin(theta) * np.sin(phi),
         r * np.cos(theta)
    )


@jitOpt
def Hgs_EZ(T):
    Dgs_T =  Dgs + d1*T + d2*T**2 + d3*T**3 + d4*T**4 + d5*T**5
    Hgs_EZ = Dgs_T*(np.dot(S_z_opp, S_z_opp) - 1*(1+1)/3*Id3) # Doherty2013: equ(1)
    return Hgs_EZ

@jitOpt
def Vgs_EZ(Bx, By, Bz, Ex, Ey, Ez): # Doherty2013: equ(2)
    """
    Units of B and E in Hz.
    
    Note: when using the fields in GHz, the Ex,y and Ez have a different
    conversion between V/m and GHz!
    """
    Vgs_EZ= (
        Bz*S_z_opp + Bx*S_x_opp + By*S_y_opp
        + Ez*(np.dot(S_z_opp, S_z_opp) - 1*(1+1)/3*Id3)
        + Ex*(np.dot(S_y_opp, S_y_opp)-np.dot(S_x_opp, S_x_opp))
        + Ey*(np.dot(S_x_opp, S_y_opp)+np.dot(S_y_opp, S_x_opp))
        )
    return Vgs_EZ

Hes_EZ = ( # Doherty2013: equ(3)
    np.kron(Id2, Des_para*(np.dot(S_z_opp, S_z_opp)- 1*(1+1)/3*Id3))
    - Les_para*np.kron(sigma_y, S_z_opp)
    + Des_perp*(
        np.kron(sigma_z, np.dot(S_y_opp, S_y_opp)-np.dot(S_x_opp, S_x_opp))
        - np.kron(sigma_x, np.dot(S_y_opp, S_x_opp)+np.dot(S_x_opp, S_y_opp))
        )
    + Les_perp*(
        np.kron(sigma_z, np.dot(S_x_opp, S_z_opp)+np.dot(S_z_opp, S_x_opp))
        - np.kron(sigma_x, np.dot(S_y_opp, S_z_opp)+np.dot(S_z_opp, S_y_opp))
        )
    )

@jitOpt
def Ves_EZ(Bx, By, Bz, Ex, Ey, Ez): # Doherty2013: equ(4)
    """
    Units of B and E in Hz.
    
    Note: when using the fields in GHz, the Ex,y and Ez have a different
    conversion between V/m and GHz!
    """
    Ves_EZ = (
        Ez*Id6
        + Ex*np.kron(sigma_z, Id3)
        - Ey*np.kron(sigma_x, Id3)
        + gl*Bz/2.003*np.kron(sigma_y, Id3) # /2 since Bz [Hz] needs conversion.
        + Bz*np.kron(Id2, S_z_opp)
        + Bx*np.kron(Id2, S_x_opp)
        + By*np.kron(Id2, S_y_opp)        
        )
    return Ves_EZ

@jitOpt
def get_Hgs_EZ(B, thetaB, phiB, Eperp, T):
    """
    Unit of B: T
    
    Unit of Eperp: Hz
    
    Unit of angles: RAD
    
    Unit of T: K
    
    Unit of returned Hamiltonian: [Hz]
    
    So far Eperp is ignored for the GS!
        
    Note: the same conversion (g-factor) of GHz to T for B fields is used in
    the GS and ES.
    """
    Bx, By, Bz = polar2cart(B_T2Hz(B), thetaB, phiB)
    # strain/el. field Eperp is in-plane. Along NV axis it is ignored.
    # The following two lines can be used to also model the Eperp effect on the GS:
    # strainScalingFactorForGS = XXX # convert the (~GHz) strain of the ES to the (~MHz) strain in the GS.
    # Ex, Ey, Ez = polar2cart(Eperp, np.radians(90), np.radians(0))*strainScalingFactorForGS
    HamiltonianMatrix_gs_EZ = Hgs_EZ(T) + Vgs_EZ(Bx, By, Bz, 0., 0., 0.)
    return HamiltonianMatrix_gs_EZ

@jitOpt
def get_Hes_EZ(B, thetaB, phiB, Eperp, phiE):
    """
    Unit of B: T
    
    Unit of Eperp: Hz
    
    Unit of angles: RAD
    
    Unit of returned Hamiltonian: [Hz]
        
    Note: the same conversion (g-factor) of GHz to T for B fields is used in
    the GS and ES.
    """
    Bx, By, Bz = polar2cart(B_T2Hz(B), thetaB, phiB)
    Ex, Ey, Ez = polar2cart(Eperp, np.radians(90), phiE)
    # The strain Epara in z-direction only shifts all ES levels up/down in energy and can thus be ignored.
    HamiltonianMatrix_es_EZ = Hes_EZ + Ves_EZ(Bx, By, Bz, Ex, Ey, Ez)  
    return HamiltonianMatrix_es_EZ

@jitOpt
def get_H_EZ(B, thetaB, phiB, Eperp, phiE, T, GStoESsplitting=0e9):
    """
    Unit of B: T
    
    Unit of Eperp: Hz
    
    Unit of angles: RAD
    
    Unit of T: K
    
    Unit of returned Hamiltonian: [Hz]
    
    So far Eperp is ignored for the GS!
        
    Note: the same conversion (g-factor) of GHz to T for B fields is used in
    the GS and ES.
    """
    HamiltonianMatrix_gs_EZ = get_Hgs_EZ(B, thetaB, phiB, Eperp, T)
    HamiltonianMatrix_gs_EZ = (HamiltonianMatrix_gs_EZ 
                               - np.diag(np.ones((3), dtype = np.complex128)
                                         )*GStoESsplitting
                               )
    HamiltonianMatrix_es_EZ = get_Hes_EZ(B, thetaB, phiB, Eperp, phiE)
    H_EZ = compositeDiagMatrix(HamiltonianMatrix_gs_EZ, HamiltonianMatrix_es_EZ)
    return H_EZ

@jitOpt
def get_H_EZ_withSS(B, thetaB, phiB, Eperp, phiE, T,
                    GStoESsplitting=0e9,  GStoSSsplitting=0e9):
    """
    Get the NV centers low temperature Hamiltonian of a 10 level model.
    For details see https://arxiv.org/abs/2304.02521.
    
    Unit of B: T
    
    Unit of Eperp: Hz
    
    Unit of angles: RAD
    
    Unit of T: K
    
    Unit of returned Hamiltonian: [Hz]
    
    So far Eperp is ignored for the GS!
        
    Note: the same conversion (g-factor) of GHz to T for B fields is used in
    the GS and ES.
    """
    H_EZ = get_H_EZ(B, thetaB, phiB, Eperp, phiE, T, GStoESsplitting=GStoESsplitting)
    H_EZ_SS = addSSDim_entry(H_EZ, -(GStoESsplitting-GStoSSsplitting) )
    return H_EZ_SS


@jitOpt
def partialTraceOrbit(M):
    """
    Assume NxN matrix originating from a (2x2) x (nxn) composite Hilbert space.
    N=2*n.
    Return the partial trace over the first Hilbert space.
    """
    n = int(M.shape[0]/2)
    X = M[:n,:n]
    Y = M[n:,n:]
    return X+Y

@jitOpt
def partialTraceSpin(M):
    """
    Assume NxN matrix originating from a (2x2) x (nxn) composite Hilbert space.
    N=2*n.
    Return the partial trace over the second Hilbert space.
    """
    n = int(M.shape[0]/2)
    m11 = np.trace(M[:n,:n])
    m22 = np.trace(M[n:,n:])
    m12 = np.trace(M[:n,n:])
    m21 = np.trace(M[n:,:n])
    M = np.array([[m11, m12],[m21, m22]])
    return M    


#################################
#%% Room temperature Hamiltonian
#################################

@jitOpt
def get_avgHes_EZ(B, thetaB, phiB, Eperp, phiE):
    """
    Unit of B: T
    
    Unit of Eperp: Hz
    
    Unit of angles: RAD
    
    Unit of returned Hamiltonian: [Hz]
        
    Note: the same conversion (g-factor) of GHz to T for B fields is used in
    the GS and ES.
    """
    HamiltonianMatrix_es_EZ = get_Hes_EZ(B, thetaB, phiB, Eperp, phiE)
    # trace out the orbital sub-space to be left with the spin Hamiltonian:
    avgHamiltonianMatrix_es_EZ = partialTraceOrbit(HamiltonianMatrix_es_EZ) # correct but not jitable: np.trace(np.reshape(HamiltonianMatrix_es_EZ, (2, 3, 2, 3)), axis1=0, axis2=2)
    avgHes_EZ = 1/2.*avgHamiltonianMatrix_es_EZ # See Doherty2013 p.16 why this gives the room temperature averaged ES Hamiltonian.
    return avgHes_EZ

@jitOpt
def get_avgH_EZ(B, thetaB, phiB, Eperp, phiE, T, GStoESsplitting=0e9):
    """
    Unit of B: T
    
    Unit of Eperp: Hz
    
    Unit of angles: RAD
    
    Unit of T: K
    
    Unit of returned Hamiltonian: [Hz]
    
    So far Eperp is ignored for the GS!
        
    Note: the same conversion (g-factor) of GHz to T for B fields is used in
    the GS and ES.
    """
    HamiltonianMatrix_gs_EZ = get_Hgs_EZ(B, thetaB, phiB, Eperp, T)
    HamiltonianMatrix_gs_EZ = HamiltonianMatrix_gs_EZ - np.diag(np.ones((3), dtype = np.complex128))*GStoESsplitting
    avgHes_EZ = get_avgHes_EZ(B, thetaB, phiB, Eperp, phiE)
    avgH_EZ = compositeDiagMatrix(HamiltonianMatrix_gs_EZ, avgHes_EZ) # See Doherty2013 p.16 why this gives the room temperature averaged ES Hamiltonian.
    return avgH_EZ

@jitOpt
def get_avgH_EZ_withSS(B, thetaB, phiB, Eperp, phiE, T, GStoESsplitting=0e9, GStoSSsplitting=0e9):
    """
    Get the NV centers room temperature Hamiltonian of a 7 level model.
    For details see https://arxiv.org/abs/2304.02521.
    
    Unit of B: T
    
    Unit of Eperp: Hz
    
    Unit of angles: RAD
    
    Unit of T: K
    
    Unit of returned Hamiltonian: [Hz]
    
    So far Eperp is ignored for the GS!
        
    Note: the same conversion (g-factor) of GHz to T for B fields is used in
    the GS and ES.
    """
    HamiltonianMatrix_gs_EZ = get_Hgs_EZ(B, thetaB, phiB, Eperp, T)
    HamiltonianMatrix_gs_EZ = (HamiltonianMatrix_gs_EZ 
                               - np.diag(np.ones((3), dtype = np.complex128)
                                         )*GStoESsplitting
                               )
    avgHes_EZ = get_avgHes_EZ(B, thetaB, phiB, Eperp, phiE)
    avgH_EZ = compositeDiagMatrix(HamiltonianMatrix_gs_EZ, avgHes_EZ)
    avgH_EZ_withSS = addSSDim_entry(avgH_EZ, -(GStoESsplitting-GStoSSsplitting) )
    return avgH_EZ_withSS


##########################################
#%% Room temperature Hamiltonian with TRF
##########################################

@jitOpt
def tempReductionFac(B, thetaB, phiB, Eperp, phiE, T): # see Plakhotnik2014
    """
    Unit of B: T
    
    Unit of Eperp: Hz
    
    Unit of angles: RAD
    
    Unit of returned Hamiltonian: [Hz]
        
    Note: the same conversion (g-factor) of GHz to T for B fields is used in
    the GS and ES.
    """
    Delta = get_orbitalSplitting(B, thetaB, phiB, Eperp, phiE)
    e = np.exp( (h*Delta)/(kB*T) )
    return (e-1)/(e+1)

@jitOpt
def get_avgHesTRF_EZ_approx(B, thetaB, phiB, Eperp, phiE, T): # see Plakhotnik2014 SI
    """
    This is completely correct if B=0. It is a good approximation
    if gl is small (0.1 is small, but might depend on strain Eperp, see Happacher2022Low),
    or B is not large (like Bz<1T with gl=0.1 at 300K).
    
    Unit of B: T
    
    Unit of Eperp: Hz
    
    Unit of angles: RAD
    
    Unit of returned Hamiltonian: [Hz]    
    
    So far Eperp is ignored for the GS!
        
    Note: the same conversion (g-factor) of GHz to T for B fields is used in
    the GS and ES.
    """
    R = tempReductionFac(B, thetaB, phiB, Eperp, phiE, T)
    Bx, By, Bz = polar2cart(B_T2Hz(B), thetaB, phiB)
    Ves_EZ = Bz*S_z_opp + Bx*S_x_opp + By*S_y_opp
    c = np.cos(phiE)
    s = np.sin(phiE)
    avgHes_EZ = (Des_para*(np.dot(S_z_opp, S_z_opp)- 1*(1+1)/3*Id3)
                 - R*Des_perp*(c*(np.dot(S_y_opp, S_y_opp) - np.dot(S_x_opp, S_x_opp))
                               + s*(np.dot(S_y_opp, S_x_opp) + np.dot(S_x_opp, S_y_opp)))
                 - R*Les_perp*(c*(np.dot(S_x_opp, S_z_opp) + np.dot(S_z_opp, S_x_opp))
                               + s*(np.dot(S_y_opp, S_z_opp) + np.dot(S_z_opp, S_y_opp)))
                 + Ves_EZ
                 )
    return avgHes_EZ

@jitOpt
def get_avgHesTRF_EZ(B, thetaB, phiB, Eperp, phiE, T): # see Plakhotnik2014 SI
    """    
    Unit of B: T
    
    Unit of Eperp: Hz
    
    Unit of angles: RAD
    
    Unit of returned Hamiltonian: [Hz]
    
    So far Eperp is ignored for the GS!
    
    Note: the same conversion (g-factor) of GHz to T for B fields is used in
    the GS and ES.
    """
    Delta = get_orbitalSplitting(B, thetaB, phiB, Eperp, phiE)
    if T <= 0.:
        em = 0.
    else:
        em = np.exp( -(h*Delta)/(kB*T) )
    rho_o_Boltzmann_HF = 1/(1+em)*np.array([[em,0.],[0.,1.]], dtype=np.complex128)
        
    HamiltonianMatrix_es_EZ = get_Hes_EZ(B, thetaB, phiB, Eperp, phiE)
    H_o_EZ = 1/3.*partialTraceSpin(HamiltonianMatrix_es_EZ)
    ev_o, evec_o = eigh( H_o_EZ )
    evec_o = evec_o[:,::-1] # higher energy first
    T_o_HFtoEZ = np.ascontiguousarray(evec_o) # the memory order is 'A' but should be 'C' for other numba functions to work with it.
    T_o_EZtoHF = conjTransp(T_o_HFtoEZ)
    rho_o_Boltzmann_EZ = basisTrafo(rho_o_Boltzmann_HF, T_o_EZtoHF, 
                                    T_old_to_new=T_o_HFtoEZ)
    
    sigma_x_tr = np.trace(np.dot(sigma_x, rho_o_Boltzmann_EZ))
    sigma_y_tr = np.trace(np.dot(sigma_y, rho_o_Boltzmann_EZ))
    sigma_z_tr = np.trace(np.dot(sigma_z, rho_o_Boltzmann_EZ))
    Id2_tr = 1. # equal to np.trace(np.dot(Id2, rho_o_Boltzmann_EZ))
    
    Bx, By, Bz = polar2cart(B_T2Hz(B), thetaB, phiB)
    # Ex, Ey, Ez = polar2cart(Eperp, np.radians(90), phiE) # all three Ex/y/z terms only give a constant energy shift and can thus be left out.
    # copy of Ves_EZ with the replacements:
    avgVes_EZ = (Id2_tr*Bz*S_z_opp + Id2_tr*Bx*S_x_opp + Id2_tr*By*S_y_opp
                 # + sigma_y_tr*gl*Bz/2*Id3 # since g=2   # only gives a constant energy shift and can thus be left out. But this term can matter for rho_o_Boltzmann_EZ.
                 # + sigma_z_tr*Ex*Id3
                 # - sigma_x_tr*Ey*Id3
                 # + Id2_tr*Ez*Id3
                 )
    # copy of Hes_EZ with the replacements:
    avgHes_EZ = (Id2_tr*Des_para*(np.dot(S_z_opp, S_z_opp)- 1*(1+1)/3*Id3)
                 - sigma_y_tr*Les_para*S_z_opp
                 + sigma_z_tr*Des_perp*(
                     np.dot(S_y_opp, S_y_opp)-np.dot(S_x_opp, S_x_opp)
                     )
                 - sigma_x_tr*Des_perp*(
                     np.dot(S_y_opp, S_x_opp)+np.dot(S_x_opp, S_y_opp)
                     )
                 + sigma_z_tr*Les_perp*(
                     np.dot(S_x_opp, S_z_opp)+np.dot(S_z_opp, S_x_opp)
                     )
                 - sigma_x_tr*Les_perp*(
                     np.dot(S_y_opp, S_z_opp)+np.dot(S_z_opp, S_y_opp)
                     )
                 ) + avgVes_EZ

    return avgHes_EZ

@jitOpt
def get_avgHTRF_EZ(B, thetaB, phiB, Eperp, phiE, T):
    """
    Unit of B: T
    
    Unit of Eperp: Hz
    
    Unit of angles: RAD
    
    Unit of T: K
    
    Unit of returned Hamiltonian: [Hz]
    
    Note: the same conversion (g-factor) of GHz to T for B fields is used in
    the GS and ES.
    """
    HamiltonianMatrix_gs_EZ = get_Hgs_EZ(B, thetaB, phiB, Eperp, T)
    avgHes_EZ = get_avgHesTRF_EZ(B, thetaB, phiB, Eperp, phiE, T)
    avgH_EZ = compositeDiagMatrix(HamiltonianMatrix_gs_EZ, avgHes_EZ) # See Doherty2013 p.16 why this gives the room temperature averaged ES Hamiltonian.
    return avgH_EZ

@jitOpt
def get_avgHTRF_EZ_withSS(B, thetaB, phiB, Eperp, phiE, T, 
                          GStoESsplitting=0e9, GStoSSsplitting=0e9):
    """
    Get the NV centers room temperature Hamiltonian of a 7 level model while
    considering a remaining population difference of the two low temperature
    orbital branches.
    For details see https://arxiv.org/abs/2304.02521.
    
    Unit of B: T
    
    Unit of Eperp: Hz
    
    Unit of angles: RAD
    
    Unit of T: K
    
    Unit of returned Hamiltonian: [Hz]
    
    Note: the same conversion (g-factor) of GHz to T for B fields is used in
    the GS and ES.
    """
    HamiltonianMatrix_gs_EZ = get_Hgs_EZ(B, thetaB, phiB, Eperp, T)
    HamiltonianMatrix_gs_EZ = (HamiltonianMatrix_gs_EZ 
                               - np.diag(np.ones((3), dtype = np.complex128)
                                         )*GStoESsplitting
                               )
    avgHes_EZ = get_avgHesTRF_EZ(B, thetaB, phiB, Eperp, phiE, T)
    avgH_EZ = compositeDiagMatrix(HamiltonianMatrix_gs_EZ, avgHes_EZ)
    avgH_EZ_withSS = addSSDim_entry(avgH_EZ, -(GStoESsplitting-GStoSSsplitting) )
    return avgH_EZ_withSS


#################
#%% Basis trafos
#################

@jitOpt
def get_T_HFtoEZ_withSS(B, thetaB, phiB, Eperp, phiE):
    """
    Get the trafo matrix for the full system (GS, ES, SS) from the ES
    orbital-eigenstate spin-eigenstate basis 'HF' to the 'EZ' basis.
    GS and SS stays as is. See also get_HFev().

    Unit of B: T
    
    Unit of Eperp: Hz
    
    Unit of angles: RAD
    """
    HamiltonianMatrix_es_EZ = get_Hes_EZ(B, thetaB, phiB, Eperp, phiE)
    # trace out the spin sub-space to be left with the orbital Hamiltonian:
    H_o = 1/3.*partialTraceSpin(HamiltonianMatrix_es_EZ)
    ev_o, evec_o = eigh( H_o )
    evec_o = evec_o[:,::-1] # higher energy first
    evec_o = np.ascontiguousarray(evec_o) # the memory order is 'A' but should be 'C' for other numba functions to work with it.
    evec_es = np.kron(evec_o, Id3) # leave ES spin space as is.
    T_HFtoEZ = compositeDiagMatrix(Id3, evec_es)
    T_HFtoEZ_withSS = addSSDim(T_HFtoEZ)
    return T_HFtoEZ_withSS

@jitOpt
def get_T_HFtoEZ_withSS_andDelta(B, thetaB, phiB, Eperp, phiE):
    """
    Equivalent to, but to save time by not computing the same twice:
    get_T_HFtoEZ_withSS(B, thetaB, phiB, Eperp, phiE)
    get_orbitalSplitting(B, thetaB, phiB, Eperp, phiE)
    
    Unit of B: T
    
    Unit of Eperp: Hz
    
    Unit of angles: RAD
    
    Unit of returned splitting Delta: Hz
    """
    HamiltonianMatrix_es_EZ = get_Hes_EZ(B, thetaB, phiB, Eperp, phiE)
    # trace out the spin sub-space to be left with the orbital Hamiltonian:
    H_o = 1/3.*partialTraceSpin(HamiltonianMatrix_es_EZ)
    ev_o, evec_o = eigh( H_o )
    evec_o = evec_o[:,::-1] # higher energy first
    evec_o = np.ascontiguousarray(evec_o) # the memory order is 'A' but should be 'C' for other numba functions to work with it.
    evec_es = np.kron(evec_o, Id3) # leave ES spin space as is.
    T_HFtoEZ = compositeDiagMatrix(Id3, evec_es)
    T_HFtoEZ_withSS = addSSDim(T_HFtoEZ)  
    Delta = abs(np.diff(ev_o)[0])
    return T_HFtoEZ_withSS, Delta

@jitOpt
def get_HFev(B, thetaB, phiB, Eperp, phiE):
    """
    Get the energy eigenvalues of the ES orbital-eigenstates.
    Order same as HF basis: first higher energy Ex state, then lower energy Ey state.
    Note that this is almost the same as 2*Eperp, but for large B, a deviation
    occurs due to the orbital gl factor.

    Unit of B: T
    
    Unit of Eperp: Hz
    
    Unit of angles: RAD    

    Unit of returned energy eigenvalues same as Hamiltonian: [Hz]
    """
    HamiltonianMatrix_es_EZ = get_Hes_EZ(B, thetaB, phiB, Eperp, phiE)
    # trace out the spin sub-space to be left with the orbital Hamiltonian:
    H_o = 1/3.*partialTraceSpin(HamiltonianMatrix_es_EZ)
    ev_o, evec_o = eigh( H_o )
    return np.ascontiguousarray(ev_o[::-1]) # higher energy first; the memory order is 'A' but should be 'C' for other numba functions to work with it.

@jitOpt
def get_orbitalSplitting(B, thetaB, phiB, Eperp, phiE):
    """
    Get the level splitting between the two orbital branches.
    
    Unit of B: T
    
    Unit of Eperp: Hz
    
    Unit of angles: RAD
    
    Unit of returned splitting Delta: Hz
    """
    return abs(np.diff(get_HFev(B, thetaB, phiB, Eperp, phiE))[0])

@jitOpt
def get_GSandES_EigenValVecs(H_X):
    """
    Given a (3+Nes)x(3+Nes) matrix representation of a Hamiltonian H_X written
    in any basis X which does not mix the GS (first 3 indices) and ES (last Nes indices),
    return the ev, evec as np.linalg.eigh would do, only that the order of the
    GS and ES is not confused. I.e. the first 3 indices i in evec[:, i] and ev[i]
    in will always be the GS, and the second part will always be ES.
    More see eigh() function. Order is low to high energy.
    """
    HamiltonianMatrix_gs_X = H_X[:3,:3]
    HamiltonianMatrix_es_X = H_X[3:,3:]
    ev_gs, evec_gs = eigh(HamiltonianMatrix_gs_X)
    ev_es, evec_es = eigh(HamiltonianMatrix_es_X)
    ev = np.append(ev_gs, ev_es)
    evec = compositeDiagMatrix(evec_gs, evec_es)
    return ev, evec

# Transformation matrix from zero-field 'ZF' basis as used in Goldman2015
# to Doherty2013 'EZ' states basis.
# Constructed based on Doherty2011 Table 1.
# H_ZF = T_ZFtoEZ^-1 . H_EZ . T_ZFtoEZ = basisTrafo(H_EZ, T_ZFtoEZ)
T_ZFtoEZ = np.array(
       [[ 0,    0,    1,     0,    0,    0,    0,    0,    0  ],
        [ 1,    0,    0,     0,    0,    0,    0,    0,    0  ],
        [ 0,    1,    0,     0,    0,    0,    0,    0,    0  ],
        [ 0,    0,    0, -1j/2,  1/2,    0,    0,-1j/2, -1/2  ],
        [ 0,    0,    0,     0,    0,    0,    1,    0,    0  ],
        [ 0,    0,    0, -1j/2, -1/2,    0,    0,-1j/2,  1/2  ],
        [ 0,    0,    0,   1/2, 1j/2,    0,    0, -1/2, 1j/2  ],
        [ 0,    0,    0,     0,    0,    1,    0,    0,    0  ],
        [ 0,    0,    0,  -1/2, 1j/2,    0,    0,  1/2, 1j/2  ]]
       , dtype = np.complex128
       )
T_ZFtoEZ_withSS = addSSDim(T_ZFtoEZ)
T_EZtoZF = conjTransp(T_ZFtoEZ)
T_EZtoZF_withSS = addSSDim(T_EZtoZF)

# Transformation matrix from 'ZB' basis to 'HF' states basis.
T_ZBtoHF = np.array(
       [[ 1,    0,    0,     0,    0,    0,    0,    0,    0  ],
        [ 0,    1,    0,     0,    0,    0,    0,    0,    0  ],
        [ 0,    0,    1,     0,    0,    0,    0,    0,    0  ],
        [ 0,    0,    0,   sq2,  sq2,    0,    0,    0,    0  ],
        [ 0,    0,    0,     0,    0,    1,    0,    0,    0  ],
        [ 0,    0,    0,  -sq2,  sq2,    0,    0,    0,    0  ],
        [ 0,    0,    0,     0,    0,    0,  sq2,  sq2,    0  ],
        [ 0,    0,    0,     0,    0,    0,    0,    0,    1  ],
        [ 0,    0,    0,     0,    0,    0, -sq2,  sq2,    0  ]]
       , dtype = np.complex128
       )
T_ZBtoHF_withSS = addSSDim(T_ZBtoHF)
T_HFtoZB = conjTransp(T_ZBtoHF)
T_HFtoZB_withSS = addSSDim(T_HFtoZB)

@jitOpt
def get_T_ZBtoEZ_withSS(B, thetaB, phiB, Eperp, phiE):
    """
    Get the trafo matrix for the full system (GS, ES, SS) from the 'ZB' basis,
    which is trivially formulated in the 'HF' basis, to the 'EZ' basis.
    GS and SS stays as is. See also get_T_HFtoEZ_withSS().

    Unit of B: T
    
    Unit of Eperp: Hz
    
    Unit of angles: RAD
    """
    T_HFtoEZ_withSS = get_T_HFtoEZ_withSS(B, thetaB, phiB, Eperp, phiE)
    T_ZBtoEZ_withSS = np.dot(T_HFtoEZ_withSS, T_ZBtoHF_withSS)
    return T_ZBtoEZ_withSS

@jitOpt
def get_T_EIGtoX(H_X):
    """
    Return the trafo matrix T_EIGtoX for the basis change from the eigenstates
    basis EIG of the Hamiltonian matrix H_X, written in X basis, to the basis X.
    H_X contains the GS (first 3 dimensions), the SS (last dimension), and
    the ES (rest of dimensions - 6 or in averaged case 3). This order is
    maintained in the EIG basis, while within the 3 groups (GS, ES, SS) the 
    states are ordered by increasing energy.
    """
    # Since this is used for the Hamiltonian matrix, and it is nicer when
    # the GS and ES eigenstates are not mixed up in order,
    # use get_GSandES_EigenValVecs instead of eigh().
    _, T_EIGtoX = get_GSandES_EigenValVecs( removeSSDim(H_X) )
    T_EIGtoX_withSS = addSSDim(T_EIGtoX)
    return T_EIGtoX_withSS


#########################
#%% Rates for all models
#########################

def get_LarmorFrequ(i, j, **modeldict):
    """
    Get Larmor Frequency between the ith and jth eigenstates of the ES.
    Units of Larmor Frequency: Hz (i.e. not as rates usually here have)
    """
    modeldict = makeModelDict(**modeldict) # generate from modeldict kwargs
    B=modeldict['B']
    thetaB=modeldict['thetaB']
    phiB=modeldict['phiB']
    Eperp=modeldict['Eperp']
    phiE=modeldict['phiE']
    H_es_EZ = get_Hes_EZ(B, thetaB, phiB, Eperp, phiE)
    ev_es, _ = eigh( H_es_EZ ) # lowest energy first
    larmorFrequ = np.abs(ev_es[j]-ev_es[i]) # units: Hz as Hamiltonian
    return larmorFrequ

@jitOpt
def getSSRatesFromSSL(tSSL, SSRatio):
    """
    Rate from SS to ms0 is k71.
    Rate from SS to ms+-1 is k72.
    Rate from SS to ms+1 is thus k72/2, just as for ms-1.
    This means shelving state lifetime tSSL = (k71+k72)**-1, units s.
    The ratio k71/k72 = SSRatio.
    """
    k71 = ((1/SSRatio+1)*tSSL)**(-1)
    k72 = ((SSRatio+1)*tSSL)**(-1)
    return k71, k72

@jitOpt
def lifetimeBoseEinsteinFunc_scalar(T, SSTauDecay, SSPhonE):
    """
    Return shelving state lifetime SSL in units s.
    
    T: temperature [K]
    
    SSTauDecay, SSPhonE are found by fitting experimental data or in 
    Robledo, et al. New J. Phys. 13, 025013 (2011).
    """
    if T > 1e-3:
        SSL = SSTauDecay*( 1 - np.exp(-SSPhonE/(kB*T) ) )
    else:
        SSL = SSTauDecay
    return SSL
lifetimeBoseEinsteinFunc = np.vectorize(lifetimeBoseEinsteinFunc_scalar)


khoppLimit = 250e12 # units: Hz, to prevent numerical instability

@jitOpt
def kmix1Full_scalar(T, Delta, elPhonCoup, T0):
    """
    Get the rate of the 1-phonon process for going down (E_x -> E_y in HF basis).
    See Goldman2015 and Ulbricht2016.
    
    Units: T [K], Delta [Hz] is about 2*Eperp, elPhonCoup [/us/meV**3]
    
    return Units: 1/s as all rates
    """
    if Delta == 0.:
        return 0. # since spectral density goes to 0.
    kmix = 4 * elPhonCoup * (1e15*h) * (Delta**2*h) * (Delta*h)
    if T <= T0:
        return kmix
        # due to spontaneous emission kmix is not 0 at T=0K!
    else:
        T -= T0
    # To convert elPhonCoup to Hz/eV**3, we need *1e15.
    kmix *= (1/(np.exp(Delta*h/(kB*T))-1) + 1)
    return kmix if kmix > 1e0 else 0.  # limit the rate since otherwise the rate model becomes numerically unstable.
def kmix1Full(T, Delta, elPhonCoup=elPhonCoup_default, T0=0.):
    return kmix1Full_scalar(T, Delta, elPhonCoup, T0)
kmix1Full = np.vectorize(kmix1Full)

def kmix2TwoEmissions(T, Delta, elPhonCoup=elPhonCoup_default, T0=0., 
                          phononCutoffEnergy=phonCutoff_default):
    """
    Get the rate for the 2-phonon-emission process for going down (E_x -> E_y in HF basis).

    Units: T [K], Delta [Hz] is about 2*Eperp, elPhonCoup [/us/meV**3], phononCutoffEnergy [eV]
    
    return Units: 1/s as all rates
    """
    if T <= T0:
        return 0.
    else:
        T -= T0
    # To convert elPhonCoup to Hz/eV**3, we need *(1e15)**2.
    # The 2-phonon emission process has 1/4th the prefactor of the 2-phonon Raman process.
    kmix = 0.25 * 64/np.pi * elPhonCoup**2 * (hbar*1e15) * (kB**3*1e15) * (T**5*kB**2) 
    
    def DebyeIntegrandTwoEmissions(x, T, Delta):
        x_perp = h*Delta/(kB*T)
        if x > x_perp:
            return 0. # Phonon-Integral goes up to x_perp, it does not make sens with higher x, since would be negative.
        else:
            # 0.5 * x * (-x+x_perp) * (x**2 + (-x+x_perp)**2) /( (np.exp(x)-1) * (np.exp(-x+x_perp)-1) ) # numerically unstable for high x and x~0.
            dividend = 0.5 * np.exp(-x+x_perp) * x * (-x+x_perp) * (x**2 + (-x+x_perp)**2)
            divisor = np.exp(-x+x_perp) - 1 - np.exp(-2*x+x_perp) + np.exp(-x)
            return dividend/divisor if divisor!=0. else 0. # since the limit for x-> inf is 0.=0./0. here.

    cutoff = phononCutoffEnergy/(kB*T)
    x_perp = h*Delta/(kB*T)
    res = quad(DebyeIntegrandTwoEmissions, 0, min(x_perp, cutoff), args=(T, Delta))[0]
    
    kmix *= res
    return min(kmix, khoppLimit) if kmix > 1e0 else 0.
kmix2TwoEmissions = np.vectorize(kmix2TwoEmissions)

@jitOpt
def DetailedBallanceRatio_scalar(T, Delta):
    """
    Ratio of hopping rate up (E_y -> E_x in HF basis) over hopping rate
    down (E_x -> E_y in HF basis).
    See Plakhotnik2015.
    
    Units: T [K], Delta [Hz] is about 2*Eperp.
    
    return unit: 1
    """
    T = max(1e-6, T)
    return np.exp(-h*Delta/(kB*T))
DetailedBallanceRatio = np.vectorize(DetailedBallanceRatio_scalar)

#@jitOpt # gives a strange parser/optimization problem with float64.
def DebyeIntegrandFull_scalar(x, T, Delta):
    """
    See Plakhotnik2015 or Goldman2015 for derivation. See Fu2009 originally.
    Approximation contained here are the Debye model and standard
    deformation potential of long wavelength acoustic phonons.
    This means, that the polarization-specific E-phonon spectral density
    is approximated by simply elPhonCoup*energy**3.
    
    "This approximation is allowed in the linear dispersion regime, where the \
     wavelength of acoustic phonons is much larger than the lattice spacing". \
     [Plakhotnik2015 and Goldman2015, both literally]
    
    Units: T [K], Delta [Hz] is about 2*Eperp.
    
    x has unit: 1 and is energy in the Phonon-Integral divided by (kB*T).
    
    return unit: 1

    This integrand is for the 2-phonon Raman process for going down (E_x -> E_y in HF basis).
    """
    # required to obtain same result with numba as with native python:
    # x=np.float32(x)
    
    x_perp = h*Delta/(kB*T)
    if x <= x_perp:
        quotient = np.float64(0.) # Phonon-Integral starts at x_perp, it does not make sens with lower x, since would be negative.
    else:
        # 0.5 * np.exp(x) * x * (x-x_perp) * (x**2 + (x-x_perp)**2) /( (np.exp(x)-1) * (np.exp(x-x_perp)-1) ) # numerically unstable for high x and x~0.
        dividend = 0.5 * np.exp(-x) * x * (x-x_perp) * (x**2 + (x-x_perp)**2)
        divisor = np.exp(-x_perp) - np.exp(-x) - np.exp(-x_perp-x) + np.exp(-2*x)
        if divisor==0.:
            quotient = np.float64(0.) # return 0. # since the limit for x-> inf is 0.=0./0. here.
        else:
            quotient = dividend/divisor
    return quotient  
DebyeIntegrandFull = np.vectorize(DebyeIntegrandFull_scalar)

def PhononIntegralFull_scalar(T, Delta, phononCutoffEnergy=phonCutoff_default):
    """
    Phonon-Integral as used by Plakhotnik2015, where it is called I.
    
    Units: T [K], Delta [Hz] is about 2*Eperp, phononCutoffEnergy [eV]
    
    return unit: 1
    """
    #np.seterr(over='ignore')  #seterr to known value
    if T <= 0.:
        return 0. # to avoid zero division; at T=0 kmix2Full_scalar is 0 anyways.
    cutoff = phononCutoffEnergy/(kB*T)
    x_perp = h*Delta/(kB*T)
    res = quad(DebyeIntegrandFull_scalar, x_perp, cutoff, args=(T, Delta))[0]
    #np.seterr(over=None)  #seterr to known value
    return res
PhononIntegralFull = np.vectorize(PhononIntegralFull_scalar)

def loadPhononIntegralFull_LUT():
    pathandname = osjoin(PATH_LUT, 'PhononIntegralLUT', 'PhononIntegralLUT_noApprox.json')    
    PhononIntegralFull_LUT_dict = {}
    try:
        with open(pathandname, 'r') as f:
            LUTdict = load(f)
            for key in LUTdict.keys():
                x = np.array(LUTdict[key]["Deltas"])
                y = np.array(LUTdict[key]["Ts"])
                z = np.array(LUTdict[key]["LUT_Delta_T"])
                spline = RectBivariateSpline(x, y, z)
                PhononIntegralFull_LUT_dict[key] = spline
    except FileNotFoundError:
        pass
        # print(f"""Failed to load Debye-Integral LUT from {pathandname}. Please run 'updatePhononIntegralFullLUT()'.""")
        # Anyways, kmix2Full_scalar() will complain so not needed.
    return PhononIntegralFull_LUT_dict

PhononIntegralFull_LUT_dict = loadPhononIntegralFull_LUT()

def PhononIntegralFull_fromLUT_scalar(T, Delta,
                              phononCutoffEnergy=phonCutoff_default):
    """
    From Debye-Integral LUT obtain result via interpolation.
    
    NOTE: The LUT has a finite range, beyond which simply the edge value is used.
    """
    global PhononIntegralFull_LUT_dict
    LUT_dict=PhononIntegralFull_LUT_dict
    
    key = f'{phononCutoffEnergy:.5f}' # units: eV
    if key not in LUT_dict.keys():
        raise NotImplementedError
    I = float(LUT_dict[key](Delta, T))    
    return I
PhononIntegralFull_fromLUT = np.vectorize(PhononIntegralFull_fromLUT_scalar)

# Test accuracy of LUT:
if 0:
    T_test,Delta_test = 1, 1e9
    print('spline:\t\t', PhononIntegralFull_fromLUT(T_test,Delta_test))
    print('integral:\t', PhononIntegralFull_scalar(T_test,Delta_test))
    
def updatePhononIntegralFullLUT(
        phononCutoffEnergy=phonCutoff_default,
        steps=1000,
        ):
    """
    Update the LUT for the Phonon-Integral.
    Calculating the integral can take, dependent on Delta and T, most of the 
    time in determining e.g. the ssPL for a given situation.
    If the given phononCutoffEnergy is not yet existent in the LUT, it will
    be added. If it already exists, it will be overwritten.
    
    NOTE: The LUT only works up to strain Eperp 500GHz and for T below 500K.
    Also, T < 0.1K is not covered but simply 0.1K is used. But since T**5 is
    so small there, this does not matter.
    """
    r = 0.4 # set range that is covered logarithmically
    DeltaList = np.linspace(0.0e9, 2*49.9e9, num=int((1-r)*steps)) # units: Hz
    DeltaList = np.append(DeltaList,
                          np.logspace(np.log10(2*50e9), np.log10(2*500e9), num=int(r*steps))
                          ) # units: Hz
    TList = np.logspace(np.log10(0.1), np.log10(500), num=steps) # units: K
    
    pathandname = osjoin(PATH_LUT, 'PhononIntegralLUT', 'PhononIntegralLUT_noApprox.json')
    
    pathandname = ensure_dir(pathandname)
    try:
        with open(pathandname, 'r') as f:
            dic = load(f)
        print(f'Writing to {pathandname}...')
    except FileNotFoundError: # json.JSONDecodeError
        dic = {}
        print(f'Creating new {pathandname}...')
    
    dt = 1.5e-3 # units: s; check e.g. %%timeit    PhononIntegralFull_scalar(20, 100e9)
    print('This will take on the order of {:.0f}min.'.format(
        round(DeltaList.size*TList.size*dt/60)
        ))

    key = f'{phononCutoffEnergy:.5f}' # units: eV
    LUT = np.zeros((DeltaList.size, TList.size))
    for Delta_idx, Delta in enumerate(DeltaList):
        for T_idx, T in enumerate(TList):
            LUT[Delta_idx, T_idx] = PhononIntegralFull_scalar(
                T, Delta, phononCutoffEnergy=phononCutoffEnergy
                )
    thisdict = {
        "phononCutoffEnergy": phononCutoffEnergy,
        "Deltas": list(DeltaList),
        "Ts": list(TList),
        "LUT_Delta_T": [list(row) for row in LUT],
        }
    dic[key] = thisdict

    with open(pathandname, 'w') as f:
        dump(dic, f)
        
    global PhononIntegralFull_LUT_dict
    PhononIntegralFull_LUT_dict = loadPhononIntegralFull_LUT()

    name = f'Phonon-Integral - no approximation - phononCutoffEnergy {phononCutoffEnergy:.5f}'
    figMap = plt.figure(figsize=(9,7))
    figMap.suptitle(name, fontsize='medium')
    figMap.set_tight_layout(True)
    axes = figMap.add_subplot(111)
    im = axes.pcolormesh(TList, DeltaList/1e9, LUT,
                         cmap='pink', shading='nearest',
                         norm=LogNorm())
    plt.colorbar(im, label='LUT [1]')
    axes.set_ylabel(r'strain splitting $\Delta_{x,y}$ [GHz]')
    axes.set_xlabel(r'temperature $T$ [K]')
    axes.set_xlim((np.min(TList),np.max(TList)))
    axes.set_ylim((np.min(DeltaList[1:]/1e9),np.max(DeltaList[1:]/1e9)))
    axes.set_xscale('log')
    axes.set_yscale('log')
    # plt.show()
    pathandnamefig = osjoin(PATH_LUT, 'PhononIntegralLUT', name)
    figMap.savefig(ensure_dir('{}.png'.format(pathandnamefig)),
                   format='png', dpi=100,
                   )
    plt.close()
    print(f'Saved to {pathandname}')
 
LUT_WARNINGS_PRINTED_FOR = []
def kmix2Full_scalar(T, Delta, elPhonCoup=elPhonCoup_default, T0=0., 
                         phononCutoffEnergy=phonCutoff_default):
    """
    Rate of the 2-phonon Raman process for going down (E_x -> E_y in HF basis).
    
    This is second-order Fermi's golden rule for a 2-E-symmetric-phonon Raman process.
    See Fu2009 originally, and Plakhotnik2015 and Goldman2015.
    The full Phonon-Integral is used, more see DebyeIntegrandFull_scalar().
    
    Units: T [K], Delta [Hz] is about 2*Eperp, elPhonCoup [/us/meV**3], phononCutoffEnergy [eV]
    
    return Units: 1/s as all rates    
    """
    if T <= T0:
        return 0.
    else:
        T -= T0
    # To convert elPhonCoup to Hz/eV**3, we need *(1e15)**2.
    kmix = 64/np.pi * elPhonCoup**2 * (hbar*1e15) * (kB**3*1e15) * (T**5*kB**2)
    
    try:
        # much faster: use a LUT (but phononCutoffEnergy has to exist in it)
        kmix *= PhononIntegralFull_fromLUT_scalar(T, Delta, 
                                     phononCutoffEnergy=phononCutoffEnergy)
    except NotImplementedError:
        # calculate for given phononCutoffEnergy: 
        global LUT_WARNINGS_PRINTED_FOR
        if phononCutoffEnergy not in LUT_WARNINGS_PRINTED_FOR:
            print(f'NOTE: No LUT available for phonon cutoff energy phononCutoffEnergy={phononCutoffEnergy*1e3:.1f}meV. On the fly computation is slower.')
            print(f"Please run 'updatePhononIntegralFullLUT({phononCutoffEnergy:.5f})'. This notification will not be displayed again.")
            LUT_WARNINGS_PRINTED_FOR.append(phononCutoffEnergy)
        kmix *= PhononIntegralFull_scalar(T, Delta, phononCutoffEnergy=phononCutoffEnergy)
    return min(kmix, khoppLimit) if kmix > 1e0 else 0. # limit the rate at 50THz since otherwise the rate model becomes numerically unstable.
kmix2Full = np.vectorize(kmix2Full_scalar)

def getOrbitalRates_scalar(T, Delta,
                         elPhonCoup=elPhonCoup_default,
                         T0_2ph=0., # T0=4.4K in Goldman2015Phonon SI.
                         T0_1ph=0.,
                         phononCutoffEnergy=phonCutoff_default, # others: PlakhotnikCutoffEnergy, AbtewCutoffEnergy, diamondDebyeEnergy
                         ):
    """
    Get tuple of:
        - hopping rate down (E_x -> E_y in HF basis),
        - hopping rate up (E_y -> E_x in HF basis),
        - and orbital dephasing rate.
    Both processes are included for the hopping:
        - 1-phonon process
        - 2-phonon process
    Due to conservation of energy, dephasing can only happen
    via the 2-phonon process.
    
    Units: T [K], Delta [Hz] is about 2*Eperp.
    
    return unit: 1/s as all rates
    """
    kmix = kmix2Full_scalar(
        T, Delta, elPhonCoup=elPhonCoup,
        T0=T0_2ph, phononCutoffEnergy=phononCutoffEnergy
        )
    kT2orb = 0.#kmix # not considered for now.
    kmix += kmix1Full_scalar(T, Delta, elPhonCoup, T0_1ph)    
    R = DetailedBallanceRatio_scalar(T, Delta)
    return (kmix, R*kmix, kT2orb)
getOrbitalRates = np.vectorize(getOrbitalRates_scalar)


@jitOpt
def getBetas(laserpower, opticAlign, exRatio):
    """Get the kr normalized excitation rates into the Ex and Ey orbital branch."""
    betaSum = laserpower*opticAlign
    betaEx = betaSum*exRatio/(exRatio+1)
    betaEy = betaSum/(exRatio+1)
    return betaEx, betaEy

@jitOpt
def getPL(optically_active_population, kr, colExRatio, opticAlign, 
          background, laserpower, darkcounts):
    """
    The total photoluminescence PL is calculated as:
        
        emission_rate = optically_active_population * kr
        
        collection_efficiency = colExRatio * opticAlign
        
        background_total = background * laserpower + darkcounts
        
        PL = emission_rate * collection_efficiency + background_total
    """
    PL = optically_active_population*kr
    collectionEff = colExRatio * opticAlign
    PL = PL*collectionEff + background*laserpower + darkcounts
    return PL


######################
#%% ME model routines
######################

Ldummy = np.zeros((10, 10), dtype = np.complex128)

def makeIncoherentLindbladOpList(
                T = T_default,
                B = B_default,
                thetaB = thetaB_default,
                phiB = phiB_default,
                Eperp = Eperp_default,
                phiE = phiE_default,
                exRatio = exRatio_default,
                opticAlign = opticAlign_default,
                kr   = kr_default,
                kA1  = kA1_default,
                kE12 = kE12_default,
                kExy = kExy_default,
                SSTauDecay = SSTauDecay_default,
                SSPhonE = SSPhonE_default,
                SSRatio   = SSRatio_default,
                laserpower = laserpower_default,
                **kwargs, #they are simply ignored.
        ):
    """
    Note that one can provide a makeModelDict() as kwargs.
    
    Return a list of matrices that are the Lindblad operators for the given rates.
    Here, in contrast to makeCoherentLindbladOpList(), each decay is put in a
    separate operator, which causes a jump that destroys coherences.
    Also called collapse or jump operators.
    They are constructed from some jump operators that are initially in different
    bases, but the output is transformed, such that Llist_EZ is in EZ basis.
    
    Entries in Lindblad operators: Row is where we end up and Column is where 
    we come from.
    
    For more detail see https://arxiv.org/abs/2304.02521.
    """    
    betaEx, betaEy = getBetas(laserpower, opticAlign, exRatio)
    k71, k72 = getSSRatesFromSSL(lifetimeBoseEinsteinFunc_scalar(T, SSTauDecay, SSPhonE), SSRatio)
    T_HFtoEZ_withSS = get_T_HFtoEZ_withSS(B, thetaB, phiB, Eperp, phiE)
    T_EZtoHF_withSS = conjTransp(T_HFtoEZ_withSS)
    
    rates = { # structure: name, value, indices for rate matrix in the from: (from state, to state)
        'betaEx': (betaEx*kr, (
            (0, 3), (1, 4), (2, 5),
            ), 'HF'),
        'betaEy': (betaEy*kr, (
            (0, 6), (1, 7), (2, 8),
            ), 'HF'),
        'kr': (kr, (
            (3, 0), (4, 1), (5, 2), (6, 0), (7, 1), (8, 2),
            ), 'EZ'),
        'kExy': (kExy, (
            (5, 9), (6, 9),
            ), 'ZF'),
        'kA1': (kA1, (
            (7, 9),
            ), 'ZF'),
        'kE12': (kE12, (
            (3, 9), (4, 9),
            ), 'ZF'),
        'k71': (k71, (
            (9, 1),
            ), 'EZ'),
        'k72': (k72/2, (
            (9, 0), (9, 2),
            ), 'EZ'),
        }
    Llist_EZ = []
    for rate in rates.keys():
        if rates[rate][0] > 0.:
            for loc in rates[rate][1]:
                Lk = np.copy(Ldummy)
                Lk[loc[::-1]] = np.sqrt(rates[rate][0])
                # the -1 is needed since the tuples in rates are written as
                # 'from -> to'. But in the operators we need 'to <- from'.   
                if rates[rate][2] == 'ZF':
                    Lk = basisTrafo(Lk, T_EZtoZF_withSS, T_old_to_new=T_ZFtoEZ_withSS)
                elif rates[rate][2] == 'HF':
                    Lk = basisTrafo(Lk, T_EZtoHF_withSS, T_old_to_new=T_HFtoEZ_withSS)
                Llist_EZ.append(Lk)
    return Llist_EZ

LindbladOp_DecayOfEyToEx_HF = np.array([
        [ 0, 0, 0,   0, 0, 0,    0, 0, 0,    0 ],
        [ 0, 0, 0,   0, 0, 0,    0, 0, 0,    0 ],
        [ 0, 0, 0,   0, 0, 0,    0, 0, 0,    0 ],
        
        [ 0, 0, 0,   0, 0, 0,    1, 0, 0,    0 ],
        [ 0, 0, 0,   0, 0, 0,    0, 1, 0,    0 ],
        [ 0, 0, 0,   0, 0, 0,    0, 0, 1,    0 ],
        
        [ 0, 0, 0,   0, 0, 0,    0, 0, 0,    0 ],
        [ 0, 0, 0,   0, 0, 0,    0, 0, 0,    0 ],
        [ 0, 0, 0,   0, 0, 0,    0, 0, 0,    0 ],
        
        [ 0, 0, 0,   0, 0, 0,    0, 0, 0,    0 ],
        ], dtype = np.complex128)
LindbladOp_DecayOfExToEy_HF = np.array([
        [ 0, 0, 0,   0, 0, 0,    0, 0, 0,    0 ],
        [ 0, 0, 0,   0, 0, 0,    0, 0, 0,    0 ],
        [ 0, 0, 0,   0, 0, 0,    0, 0, 0,    0 ],
        
        [ 0, 0, 0,   0, 0, 0,    0, 0, 0,    0 ],
        [ 0, 0, 0,   0, 0, 0,    0, 0, 0,    0 ],
        [ 0, 0, 0,   0, 0, 0,    0, 0, 0,    0 ],
        
        [ 0, 0, 0,   1, 0, 0,    0, 0, 0,    0 ],
        [ 0, 0, 0,   0, 1, 0,    0, 0, 0,    0 ],
        [ 0, 0, 0,   0, 0, 1,    0, 0, 0,    0 ],
        
        [ 0, 0, 0,   0, 0, 0,    0, 0, 0,    0 ],
        ], dtype = np.complex128)

LindbladOp_DecayOfExToEx_HF = np.array([
        [ 0, 0, 0,   0, 0, 0,    0, 0, 0,    0 ],
        [ 0, 0, 0,   0, 0, 0,    0, 0, 0,    0 ],
        [ 0, 0, 0,   0, 0, 0,    0, 0, 0,    0 ],
        
        [ 0, 0, 0,   1, 0, 0,    0, 0, 0,    0 ],
        [ 0, 0, 0,   0, 1, 0,    0, 0, 0,    0 ],
        [ 0, 0, 0,   0, 0, 1,    0, 0, 0,    0 ],
        
        [ 0, 0, 0,   0, 0, 0,    0, 0, 0,    0 ],
        [ 0, 0, 0,   0, 0, 0,    0, 0, 0,    0 ],
        [ 0, 0, 0,   0, 0, 0,    0, 0, 0,    0 ],
        
        [ 0, 0, 0,   0, 0, 0,    0, 0, 0,    0 ],
        ], dtype = np.complex128)
LindbladOp_DecayOfEyToEy_HF = np.array([
        [ 0, 0, 0,   0, 0, 0,    0, 0, 0,    0 ],
        [ 0, 0, 0,   0, 0, 0,    0, 0, 0,    0 ],
        [ 0, 0, 0,   0, 0, 0,    0, 0, 0,    0 ],
        
        [ 0, 0, 0,   0, 0, 0,    0, 0, 0,    0 ],
        [ 0, 0, 0,   0, 0, 0,    0, 0, 0,    0 ],
        [ 0, 0, 0,   0, 0, 0,    0, 0, 0,    0 ],
        
        [ 0, 0, 0,   0, 0, 0,    1, 0, 0,    0 ],
        [ 0, 0, 0,   0, 0, 0,    0, 1, 0,    0 ],
        [ 0, 0, 0,   0, 0, 0,    0, 0, 1,    0 ],
        
        [ 0, 0, 0,   0, 0, 0,    0, 0, 0,    0 ],
        ], dtype = np.complex128)

LindbladOp_GS_msp1_ypiPulse_EZ = np.zeros((10,10), dtype = np.complex128)
LindbladOp_GS_msp1_ypiPulse_EZ[:2,:2] = sigma_y # as level1=0, level2=1 in piPulse()
LindbladOp_GS_msp1_xpiPulse_EZ = np.zeros((10,10), dtype = np.complex128)
LindbladOp_GS_msp1_xpiPulse_EZ[:2,:2] = sigma_x # as level1=0, level2=1 in piPulse()
# Meant for usage in applyJumpOp() in simulatePopTimeTrace():
# With D = getDissipator(LindbladOp_GS_msp1_ypiPulse_EZ) and U=1+D one can
# apply a pi pulse with rho_new = U*rho_0 in Fock-Liouville space.

def makeCoherentLindbladOpList(
                T = T_default,
                B = B_default,
                thetaB = thetaB_default,
                phiB = phiB_default,
                Eperp = Eperp_default, 
                phiE = phiE_default,
                elPhonCoup  = elPhonCoup_default,
                phonCutoff = phonCutoff_default,
                **kwargs, #they are simply ignored.
        ):
    """
    Note that one can provide a makeModelDict() as kwargs.
    
    Return a list of matrices that are the Lindblad operators for the given rates.
    Here, in contrast to makeIncoherentLindbladOpList(), decays are put in the 
    same operator in cases where they cause a jump that preserves coherences.
    Also called collapse or jump operators.
    They are constructed from some jump operators that are initially in different
    bases, but the output is transformed, such that Llist_EZ is in EZ basis.
    
    Entries in Lindblad operators: Row is where we end up and Column is where 
    we come from.

    For more detail see https://arxiv.org/abs/2304.02521.
    """
    Llist_EZ = []
    T_HFtoEZ_withSS, Delta = get_T_HFtoEZ_withSS_andDelta(
        B, thetaB, phiB, Eperp, phiE)
    T_EZtoHF_withSS = conjTransp(T_HFtoEZ_withSS)

    kmixDown, kmixUp, kT2orb = getOrbitalRates_scalar(T, Delta, 
                      elPhonCoup=elPhonCoup, phononCutoffEnergy=phonCutoff)

    if kmixDown > 0.:        
        LindbladOp_DecayOfExToEy_EZ = basisTrafo(
            LindbladOp_DecayOfExToEy_HF*np.sqrt(kmixDown),
            T_EZtoHF_withSS, T_old_to_new=T_HFtoEZ_withSS)
        
        LindbladOp_DecayOfEyToEx_EZ = basisTrafo(
            LindbladOp_DecayOfEyToEx_HF*np.sqrt(kmixUp),
            T_EZtoHF_withSS, T_old_to_new=T_HFtoEZ_withSS)
        
        Llist_EZ.append(LindbladOp_DecayOfExToEy_EZ)
        Llist_EZ.append(LindbladOp_DecayOfEyToEx_EZ)
        
    if kT2orb > 0.:
        LindbladOp_DecayOfExToEx_EZ = basisTrafo(
            LindbladOp_DecayOfExToEx_HF*np.sqrt(kT2orb),
            T_EZtoHF_withSS, T_old_to_new=T_HFtoEZ_withSS)
        
        LindbladOp_DecayOfEyToEy_EZ = basisTrafo(
            LindbladOp_DecayOfEyToEy_HF*np.sqrt(kT2orb),
            T_EZtoHF_withSS, T_old_to_new=T_HFtoEZ_withSS)
        
        Llist_EZ.append(LindbladOp_DecayOfExToEx_EZ)
        Llist_EZ.append(LindbladOp_DecayOfEyToEy_EZ)
        
    return Llist_EZ

@jitOpt
def spre(A):
    n = A.shape[0]
    Id = np.eye(n, dtype=A.dtype)
    S = np.kron(Id, A)
    return S

@jitOpt
def spost(A):
    n = A.shape[0]
    Id = np.eye(n, dtype=A.dtype)
    S = np.kron(A.transpose(), Id)
    return S

@jitOpt
def getDissipator(c):
    cdag = c.conjugate().transpose()
    cdc = np.dot(cdag, c)
    D = np.dot(spre(c), spost(cdag)) - spre(cdc)*0.5 - spost(cdc)*0.5
    return D

@jitOpt
def getLiovillian(H, *args):
    """
    Coded based on and effectively equivalent to qutip:
    https://qutip.org/docs/4.4/apidoc/functions.html?highlight=liouvillian#module-qutip.superoperator
    """    
    L = (spre(H) - spost(H))*(-1.0j)
    for i in range(len(args)):
        c = args[i]
        L += getDissipator(c)
    return L

@jitOpt
def calcPL(state, kr, colExRatio, opticAlign, 
                     background, laserpower, darkcounts):
    emittingLevelIdxs = [3, 4, 5, 6, 7, 8]
    x = np.sum(np.take(np.diag(state), emittingLevelIdxs))
    optically_active_population = np.real(x)
    return getPL(optically_active_population, kr, colExRatio, opticAlign, background, laserpower, darkcounts)

@jitOpt
def propagate(steps, U, state, kr, colExRatio, opticAlign, 
              background, laserpower, darkcounts):
    shape = state.shape
    states = np.zeros((steps,)+shape, dtype = np.complex128)
    states[0] = state # copies state   
    pls = np.zeros(steps)
    pls[0] = calcPL(
        state, kr, colExRatio, opticAlign, background, laserpower, darkcounts)
    state_vec = state.transpose().flatten()
    for i in range(1, steps):
        state_vec = np.dot(U, state_vec)
        state = state_vec.reshape(shape).transpose()
        states[i] = state
        pls[i] = calcPL(
            state, kr, colExRatio, opticAlign, background, laserpower, darkcounts)
    return states, pls

@jitOpt
def getsteadystate_byLin(L):
    """
    Coded based on qutip:
    https://qutip.org/docs/4.4/apidoc/functions.html?highlight=steadystate#module-qutip.steadystate
    """
    n = L.shape[0]
    m = int(np.sqrt(n))
    shape = (m,m)
    # to find a non-trivial solution to Lx=0, additionally include the condition that tr(state)=1.
    weight = np.mean(np.abs(L))
    b = np.zeros(n, dtype=np.complex128)
    b[0] = weight
    A = np.copy(L)
    A[0,:] += np.diag(weight*np.ones(m)).reshape(n)
    
    state_vec = np.linalg.solve(A, b)
    state =  state_vec.reshape(shape).transpose()
    
    state = state/np.real(np.trace(state)) # do not need to normalize since contained in solution. But does not slow it down and improves numerical value a bit.
    state = 0.5 * (state + conjTransp(state)) # make Hermitian
    return state

# @jitOpt # 10% slower with numba
def getsteadystate_byEig(L):
    n = L.shape[0]
    m = int(np.sqrt(n))
    shape = (m,m)
    
    ev, evec = eig(L)
    idx = np.argmin(np.abs(ev))
    state_vec = evec[:,idx]
    # state_vec =  np.ascontiguousarray(state_vec) # already done in eig but needed again if this function is jited.
    state =  state_vec.reshape(shape).transpose()

    state = state/np.real(np.trace(state)) # need to normalize
    state = 0.5 * (state + conjTransp(state)) # make Hermitian
    return state



##################################
#%% Classical rate model routines
##################################

@jitOpt
def getProb_EIGinX(B, thetaB, phiB, Eperp, phiE, T, T_XtoEZ_withSS,
                   avgedES=False,
                   highT_trf = False, # needed for avgedES only.
                   ):
    """
    Return probability matrix for finding eigenstates (EIG basis) in the
    basis X states. Eigenstates are of Hamiltonian with the given field 
    parameters. States are of the full space of GS, ES, SS here.
    The basis X is given by the trafo matrix T_XtoEZ_withSS.
    Be |eig_i> the ith eigenstate of H with |eig_i> = SUM_j a_i,j |j>,
    where |j> is the jth state of basis X.
    Then prob_EIGinX[i,j]=|a_i,j|^2.
        
    If avgedES=True, T_XtoEZ_withSS is ignored and T_XtoEZ_withSS=Id is assumed,
    i.e. X is automatically assumed to be EZ.
    """
    if not avgedES:
        H_EZ = get_H_EZ_withSS(B, thetaB, phiB, Eperp, phiE, T)
        T_EIGtoX_withSS = get_T_EIGtoX(basisTrafo(H_EZ, T_XtoEZ_withSS))
    else: # assume X=EZ
        if not highT_trf:
            H_EZ = get_avgH_EZ_withSS(B, thetaB, phiB, Eperp, phiE, T)
        else:
            H_EZ = get_avgHTRF_EZ_withSS(B, thetaB, phiB, Eperp, phiE, T)
        T_EIGtoX_withSS = get_T_EIGtoX(H_EZ)
    prob_EIGinX_withSS = np.abs(T_EIGtoX_withSS)**2
    return prob_EIGinX_withSS

@jitOpt
def getProb_YinX(T_YtoX_withSS):
    """
    Return probability matrix for finding Y basis states in the basis X states.
    T_YtoX_withSS is the trafo matrix of the basis change (with SS).
     
    More see getProb_EIGinX(), which is the same as
    getProb_YinX(get_T_EIGtoX(H_X)), where H_X is Hamiltonian in basis X
    with the SS and can be of both, the 10 or averaged 7 level system.
    """
    prob_YinX_withSS = (np.abs(T_YtoX_withSS)**2)
    return prob_YinX_withSS

@jitOpt
def averageRateMatrix_EZ(k_EZ):
    """Assume k_EZ as makeRateMatrix_EZ() gives and return the rate matrix for
    avgEZ basis."""
    kr = k_EZ[3,0]
    betaSum = (k_EZ[0,3]+k_EZ[0,6])/kr
    k57 = k_EZ[3,9]
    k47 = k_EZ[4,9]
    k71 = k_EZ[9,1]
    k72 = 2*k_EZ[9,0]
    # avgEZ basis: A+1  A0    A-1   E+1  E0  E-1  SS
    avgedk_EZ = np.array([
        [       0.,         0,         0,betaSum*kr,         0,         0,        0],
        [       0.,         0,         0,         0,betaSum*kr,         0,        0],
        [       0.,         0,         0,         0,         0,betaSum*kr,        0],
        [       kr,        0.,         0,         0,         0,         0,      k57],
        [       0.,        kr,         0,         0,         0,         0,      k47],
        [       0.,         0,        kr,         0,         0,         0,      k57],
        [    k72/2,       k71,     k72/2,        0.,         0,         0,        0]
        # the 0. are required for numba to understand the dtype of the rows.
        ], dtype=np.float64)
    return avgedk_EZ

@jitOpt
def getRateMatrix_EIG(B, thetaB, phiB, Eperp, phiE, T, T_XtoEZ_withSS, k_X,
                      avgedES=False,
                      highT_trf = False, # needed for avgedES only.
                      ):
    """
    Based on the classical rate matrix k_X in X basis, get the rate matrix k_EIG
    in the eigenstates (EIG) basis. Eigenstates are of Hamiltonian with the given
    field parameters.
    
    If avgedES=True, T_XtoEZ_withSS is ignored and T_XtoEZ_withSS=Id is assumed,
    i.e. X is automatically assumed to be EZ and k_X has to be in EZ basis.
    Further, k_X = k_EZ is for the 10 level model and averaged to its 7 level
    version.
    
    NOTE: this is not a basis change, it is not invertible and transitive.
    """
    prob_EIGinX_withSS = getProb_EIGinX(B, thetaB, phiB, Eperp, phiE, T,
                                        T_XtoEZ_withSS,
                                        avgedES=avgedES, highT_trf=highT_trf)
    if avgedES:
        # assume: rate matrix is in EZ: T_EZtoEZ=Id and k_EZ
        k_X = averageRateMatrix_EZ(k_X)
    k_EIG = np.dot(prob_EIGinX_withSS.transpose(), np.dot(k_X, prob_EIGinX_withSS))
    return k_EIG

@jitOpt
def getRateMatrix_Y(k_X, T_YtoX_withSS,
                    avgedES=False,
                    ):
    """
    Based on the rate matrix k_X in X basis, get the rate matrix k_Y in the
    Y basis. T_YtoX_withSS is the trafo matrix of the basis change (with SS).
    
    More see getRateMatrix_EIG(), which is the same as
    getRateMatrix_Y(k_X, get_T_EIGtoX(H_X)), where H_X is
    Hamiltonian in basis X with the SS and can be of both, the 10 or 
    averaged 7 level system (when avgedES=True).
    
    If avgedES=True, T_YtoX_withSS has to be for the averaged 7 level model
    with X = EZ. But the k_X = k_EZ is for the 10 level model and is averaged 
    to its 7 level version.

    NOTE: this is not a basis change, it is not invertible and transitive.
    """
    if avgedES:
        # assume: rate matrix is in EZ: T_EZtoEZ=Id and k_EZ
        k_X = averageRateMatrix_EZ(k_X)
    prob_YinX_withSS = getProb_YinX(T_YtoX_withSS)
    k_Y = np.dot(prob_YinX_withSS.transpose(), np.dot(k_X, prob_YinX_withSS))
    return k_Y

@jitOpt
def makeRateMatrix_LowTmodel(
        T_EIGtoEZ_withSS, # obtain from get_T_EIGtoX(H_EZ_withSS)
        T_HFtoEZ_withSS, # obtain from get_T_HFtoEZ_withSS(...)
        laserpower = laserpower_default,
        opticAlign = opticAlign_default, 
        exRatio = exRatio_default,
        kr   = kr_default,
        kA1  = kA1_default, # from A_1 to SS.
        kE12 = kE12_default, # from E_1,2 to SS.
        kExy = kExy_default,  # from E_x,y to SS.
        T = T_default,
        SSTauDecay = SSTauDecay_default,
        SSPhonE = SSPhonE_default,
        SSRatio = SSRatio_default,
        kmixUp = 0.,
        kmixDown = 0.,
        ):
    """
    Return the rate matrix for the classical rate model at low temperature.
    Up to around 30K, it shows a similar behavior as the ME model.
    For the given field conditions, return the rate matrix k_EIG of the 10 
    level model written in the eigenstate (EIG) basis of the Hamiltonian.
    
    NOTE: the rate matrix is only applicable to state vectors also written in 
    this field conditions' EIG basis. Such a state vector is usually generated 
    by the initState() function.
    """
    
    # NOTE: the calculation of a rate matrix at given conditions has to be 
    # determined right from the other conditions where the rate matrix is defined.
    # The transform from one condition to the other is not transitive.
    # Meaning that rates can only be put into a rate matrix if they are pure
    # rates of the basis states. Since the rates are known for different basis 
    # states, one has to separately transform them from the conditions where 
    # they are given to the desired ones.

    k71, k72 = getSSRatesFromSSL(lifetimeBoseEinsteinFunc_scalar(T, SSTauDecay, SSPhonE), SSRatio)
    betaEx, betaEy = getBetas(laserpower, opticAlign, exRatio)

    # HF basis: A_+1 A_0   A_-1  E_x,+1 E_x,0 E_x,-1  E_y,+1  E_y,0 E_y,-1 SS   
    #new:
    subk_HF = np.array([
            [        0,         0,         0, betaEx*kr,         0,         0, betaEy*kr,         0,         0,         0.],
            [        0,         0,         0,         0, betaEx*kr,         0,         0, betaEy*kr,         0,         0.],
            [        0,         0,         0,         0,         0, betaEx*kr,         0,         0, betaEy*kr,         0.],
            [       kr,         0,         0,         0,         0,         0,  kmixDown,         0,         0,         0.],
            [        0,        kr,         0,         0,         0,         0,         0,  kmixDown,         0,         0.],
            [        0,         0,        kr,         0,         0,         0,         0,         0,  kmixDown,         0.],
            [       kr,         0,         0,    kmixUp,         0,         0,         0,         0,         0,         0.],
            [        0,        kr,         0,         0,    kmixUp,         0,         0,         0,         0,         0.],
            [        0,         0,        kr,         0,         0,    kmixUp,         0,         0,         0,         0.],
            [    k72/2,       k71,     k72/2,         0,         0,         0,         0,         0,         0,         0.]
            # the 0. are required for numba to understand the dtype of the rows.
            ], dtype=np.float64)
    T_EIGtoHF_withSS = get_T_XtoY(T_EIGtoEZ_withSS, T_HFtoEZ_withSS)
    subkHF_EIG = getRateMatrix_Y(subk_HF, T_EIGtoHF_withSS)  
    
    # EZ basis: A+1  A0    A-1   Ex,+1  Ex,0  Ex,-1,  Ey,+1,  Ey,0  Ey,-1  SS
    # new:
    subk_ZF = np.array([
            [       0.,         0,         0,         0,         0,         0,         0,         0,         0,         0],
            [       0.,         0,         0,         0,         0,         0,         0,         0,         0,         0],
            [       0.,         0,         0,         0,         0,         0,         0,         0,         0,         0],
            [       0.,         0,         0,         0,         0,         0,         0,         0,         0,      kE12],
            [       0.,         0,         0,         0,         0,         0,         0,         0,         0,      kE12],
            [       0.,         0,         0,         0,         0,         0,         0,         0,         0,      kExy],
            [       0.,         0,         0,         0,         0,         0,         0,         0,         0,      kExy],
            [       0.,         0,         0,         0,         0,         0,         0,         0,         0,       kA1],
            [       0.,         0,         0,         0,         0,         0,         0,         0,         0,         0],
            [       0.,         0,         0,         0,         0,         0,         0,         0,         0,         0]
            # the 0. are required for numba to understand the dtype of the rows.
            ], dtype=np.float64)
    T_EIGtoZF_withSS = get_T_XtoY(T_EIGtoEZ_withSS, T_ZFtoEZ_withSS)
    subkZF_EIG = getRateMatrix_Y(subk_ZF, T_EIGtoZF_withSS)
    
    k_EIG = subkHF_EIG + subkZF_EIG
    return k_EIG

# for SZ model and high T model:
@jitOpt
def makeRateMatrix_EZ(
        betaEx,
        betaEy,
        kr,
        k47, # from ms=0 to SS.
        k57, # from ms=+-1 to SS.
        k71, # from SS to ms=0.
        k72, # from SS to ms=+-1.
        ):
    """
    Return rate matrix for the 10 level model written in the EZ basis.
    In general, this rate matrix is not correct at low temperature, correct is 
    makeRateMatrix_LowTmodel(). But it is used here for the HighT model, where
    it becomes correct when averaged. For more information, see SZmodel().
    """
    # EZ basis: A+1  A0    A-1   Ex,+1  Ex,0  Ex,-1,  Ey,+1,  Ey,0  Ey,-1  SS
    k_EZ = np.array([
            [       0.,         0,         0, betaEx*kr,         0,         0, betaEy*kr,         0,         0,         0],
            [       0.,         0,         0,         0, betaEx*kr,         0,         0, betaEy*kr,         0,         0],
            [       0.,         0,         0,         0,         0, betaEx*kr,         0,         0, betaEy*kr,         0],
            [       kr,        0.,         0,         0,         0,         0,         0,         0,         0,       k57],
            [       0.,        kr,         0,         0,         0,         0,         0,         0,         0,       k47],
            [       0.,         0,        kr,         0,         0,         0,         0,         0,         0,       k57],
            [       kr,        0.,         0,         0,         0,         0,         0,         0,         0,       k57],
            [       0.,        kr,         0,         0,         0,         0,         0,         0,         0,       k47],
            [       0.,         0,        kr,         0,         0,         0,         0,         0,         0,       k57],
            [    k72/2,       k71,     k72/2,        0.,         0,         0,         0,         0,         0,         0]
            # the 0. are required for numba to understand the dtype of the rows.
            ], dtype=np.float64)
    return k_EZ

# for SZ model and high T model (avgedES=True):
@jitOpt
def makeRateMatrix_EZmodel(T_EIGtoEZ_withSS, # obtain from get_T_EIGtoX(H_EZ_withSS)
        avgedES=False,
        laserpower = laserpower_default,
        opticAlign = opticAlign_default, 
        exRatio = exRatio_default,
        kr   = kr_default,
        kE12 = kE12_default,
        kA1 = kA1_default,
        kExy = kExy_default,
        T = T_default,
        SSTauDecay = SSTauDecay_default,
        SSPhonE = SSPhonE_default,
        SSRatio = SSRatio_default,
        **kwargs, #they are simply ignored.
        ):
    """
    Note that one can provide a makeModelDict() as kwargs, the only parameter
    that are additionally required are T_EIGtoEZ_withSS and avgedES.
    
    For the given field conditions, return the rate matrix k_EIG for the 10 
    level model written in the eigenstate (EIG) basis of the Hamiltonian.
    In general, this rate matrix is not correct at low temperature, correct is 
    makeRateMatrix_LowTmodel(). But it is used here for the HighT model, where
    it becomes correct when averaged. For more information, see SZmodel().
    
    NOTE: the rate matrix is only applicable to state vectors also written in 
    this field conditions' EIG basis. Such a state vector is usually generated 
    by the initState() function.
    
    If avgedES=True, T_EIGtoEZ_withSS has to be for the averaged 7 level model.
    """
    k71, k72 = getSSRatesFromSSL(lifetimeBoseEinsteinFunc_scalar(T, SSTauDecay, SSPhonE), SSRatio)
    betaEx, betaEy = getBetas(laserpower, opticAlign, exRatio)
    k57 = np.mean(np.array([kE12, kE12, kA1, 0.])) # otherwise the ME model does not become exactly the same as this one at high temperature.

    k_EZ = makeRateMatrix_EZ(
            betaEx,
            betaEy,
            kr,
            kExy, # from ms=0 to SS.
            k57, # from ms=+-1 to SS.
            k71, # from SS to ms=0.
            k72, # from SS to ms=+-1.
            )
    k_EIG = getRateMatrix_Y(k_EZ, T_EIGtoEZ_withSS, avgedES=avgedES)
    return k_EIG


@jitOpt
def getPropagationMatrix_classical(RM):
    S = np.diag(np.sum(RM, axis=1))
    M = (np.transpose(RM) - S)
    return M

@jitOpt
def getsteadystate_classical(RM):
    M = getPropagationMatrix_classical(RM)
    m = RM.shape[0]
    # to find a non-trivial solution to Mx=0, additionally include the condition that sum(P)=1.
    weight = np.mean(np.abs(M))
    b = np.zeros(m)
    b[0] = weight
    A = np.copy(M)
    A[0,:] += weight*np.ones(m)
    
    state = np.linalg.solve(A, b)
    state = state/np.sum(state) # do not need to normalize since contained in solution. But does not slow it down and improves numerical value a bit.
    return state

@jitOpt
def calcPL_classical(state, kr, colExRatio, opticAlign, 
                               background, laserpower, darkcounts):
    emittingLevelIdxs = [3, 4, 5, 6, 7, 8]
    optically_active_population = np.sum(np.take(state, emittingLevelIdxs))
    return getPL(optically_active_population, kr, colExRatio, opticAlign, 
                 background, laserpower, darkcounts)

@jitOpt
def propagate_classical(steps, U, state, kr, colExRatio, opticAlign, 
                        background, laserpower, darkcounts):
    shape = state.shape
    states = np.zeros((steps,)+shape, dtype = np.float64)
    states[0] = state # copies state 
    pls = np.zeros(steps)
    pls[0] = calcPL_classical(
        state, kr, colExRatio, opticAlign, background, laserpower, darkcounts)
    for i in range(1, steps):
        state = np.dot(U, state)
        states[i] = state
        pls[i] = calcPL_classical(
            state, kr, colExRatio, opticAlign, background, laserpower, darkcounts)
    return states, pls


@jitOpt
def calcPL_classicalHT(state, kr, colExRatio, opticAlign, 
                                 background, laserpower, darkcounts):
    emittingLevelIdxs = [3, 4, 5]
    optically_active_population = np.sum(np.take(state, emittingLevelIdxs))
    return getPL(optically_active_population, kr, colExRatio, opticAlign, 
                 background, laserpower, darkcounts)

@jitOpt
def propagate_classicalHT(steps, U, state, kr, colExRatio, opticAlign,
                          background, laserpower, darkcounts):
    shape = state.shape
    states = np.zeros((steps,)+shape, dtype = np.float64)
    states[0] = state # copies state 
    pls = np.zeros(steps)
    pls[0] = calcPL_classicalHT(
        state, kr, colExRatio, opticAlign, background, laserpower, darkcounts)
    for i in range(1, steps):
        state = np.dot(U, state)
        states[i] = state
        pls[i] = calcPL_classicalHT(
            state, kr, colExRatio, opticAlign, background, laserpower, darkcounts)
    return states, pls


##############
#%% modeldict
##############

def makeModelDict(
        Eperp      = Eperp_default, # unit: Hz
        phiE       = phiE_default, # unit: RAD
        B          = B_default, # unit: T
        thetaB     = thetaB_default, # unit: RAD
        phiB       = phiB_default, # unit: RAD
        kr         = kr_default, # unit: 1/s
        kE12       = kE12_default, # unit: 1/s
        kA1        = kA1_default, # unit: 1/s
        kExy       = kExy_default, # unit: 1/s
        exRatio    = exRatio_default, # unit: 1
        SSRatio    = SSRatio_default, # unit: 1
        SSPhonE    = SSPhonE_default, # unit: eV
        SSTauDecay = SSTauDecay_default, # unit: s
        T          = T_default, # unit: K
        elPhonCoup = elPhonCoup_default, # unit: 1/us*1/meV^3
        phonCutoff = phonCutoff_default, # unit: eV
        laserpower = laserpower_default, # unit: W
        background = background_default, # unit: cps/W
        colExRatio = colExRatio_default, # unit: cps*W*s
        opticAlign = opticAlign_default, # unit: 1/W
        darkcounts = darkcounts_default, # unit: cps
        piPulseFid = piPulseFid_default, # unit: 1
        highT_trf  = highT_trf_default, # unit: bool
        **kwargs, #they are simply ignored.
        ):
    
    """
    Generate a 'modeldict', which is used as kwargs to e.g. the NV models and
    contains all information about the parameter setting.
    For more information on the levels and states, see BasisStateNames().
    Further, see Tab. 1 in https://arxiv.org/abs/2304.02521.
    Use printModelDict(modeldict) to get an overview of parameters.
    
    Parameters
    ----------
    
    Eperp      : ES in-plane strain/el. field, unit: Hz
    
    phiE       : strain/el. field in-plane angle, unit: RAD
    
    B          : mag. field magnitude, unit: T
    
    thetaB     : mag. field misalignment angle, unit: RAD
    
    phiB       : mag. field in-plane angle, unit: RAD
    
    
    kr         : opt. emission rate, unit: 1/s
    
    kE12       : ISC rate for E_1 and E_2 levels, unit: 1/s (approx. avg. ISC rate for m_S = +-1)
    
    kA1        : ISC rate for A_1 level, unit: 1/s (ISC for A_2 is assumed to be 0)
    
    kExy       : ISC rate for m_S = 0, unit: 1/s
    
    exRatio    : beta_x/beta_y opt. excit. branching ratio, unit: 1
    
    SSRatio    : k_S0/k_S1 SS branching ratio, unit: 1
    
    SSPhonE    : SS emitted phonon energy, unit: eV
    
    SSTauDecay : SS decay time at 0K, unit: s
    
    T          : temperature, unit: K
    
    elPhonCoup : ES el.-phonon coup. strength, unit: 1/us*1/meV^3
    
    phonCutoff : phonon cutoff energy, unit: eV
    
    
    laserpower : laser power (*), unit: W
    
    background : background, unit: cps/W
    
    colExRatio : collection over excit. eff., unit: cps*W*s
    
    opticAlign : opt. alignment (excit. eff.), unit: 1/W
    
    darkcounts : dark counts independent of laserpower, unit: cps
    
    piPulseFid : pi pulse fidelity, unit: 1
    
    highT_trf  : is temperature reduction factor (trf) used? (**), unit: bool
    
    
    (*) The total photoluminescence PL is calculated as (see getPL() function):
        
        emission_rate = optically_active_population * kr
        
        collection_efficiency = colExRatio * opticAlign
        
        background_total = background * laserpower + darkcounts
        
        PL = emission_rate * collection_efficiency + background_total
    
    (**) For more information, see Equ. E1 in https://arxiv.org/abs/2304.02521.
    """
    dic = {
        'Eperp'     : Eperp,
        'phiE'      : phiE,
        'B'         : B,
        'thetaB'    : thetaB,
        'phiB'      : phiB,
        'kr'        : kr,
        'kA1'       : kA1,
        'kE12'      : kE12,
        'kExy'      : kExy,
        'exRatio'   : exRatio,
        'SSRatio'   : SSRatio,
        'SSPhonE'   : SSPhonE,
        'SSTauDecay': SSTauDecay,
        'T'         : T,
        'elPhonCoup': elPhonCoup,
        'phonCutoff': phonCutoff,
        'laserpower': laserpower,
        'background': background,
        'colExRatio': colExRatio,
        'opticAlign': opticAlign,
        'darkcounts': darkcounts,
        'piPulseFid': piPulseFid,
        'highT_trf' : highT_trf,
        }
    return dic

def loadModelDictFromFile(pathAndFile):
    """
    Load a 'modeldict' which is saved under the 'params' keyword in a .json
    file at pathAndFile.
    For more information, see makeModelDict() function.
    """
    with open(pathAndFile, 'r') as f:
          modeldict = load(f)["params"]
    return modeldict

def switchToReducedAngles(modeldict):
    """
    Reduce phiE to range [0, 60] and correspondingly adjust phiB.
    This effectively means that the x-Axis is defined in a certain direction 
    and the z-axis as well.
    """
    modeldict_reduced = deepcopy(modeldict)
    
    phiE = modeldict_reduced['phiE']
    phiB = modeldict_reduced['phiB']
    
    # define x-Axis such that phiE withing [0,120]:
    phiE_new = phiE % np.radians(120)
    rot = phiE - phiE_new
    phiB_new = phiB - rot
    
    # define z-Axis such that phiE withing [0,60]:
    if phiE_new > np.radians(60):
        phiE_new = np.radians(60) - (phiE_new - np.radians(60))
        phiB_new = np.radians(60) - (phiB_new - np.radians(60))
    
    phiB_new = (phiB_new + np.radians(360)) %  np.radians(360)
    modeldict_reduced['phiE'] = phiE_new 
    modeldict_reduced['phiB'] = phiB_new 
    return modeldict_reduced

def scaleParam(modeldict_item, remove1=False):
    """
    For an item (key, value) from a modeldict.items() list, return a tuple
    (unitName, value) with a string unitName and float value in common units.
    If value is a numpy.array, the whole scaled array is returned.
    For more information, see makeModelDict() function. 
    """
    key, value = modeldict_item
    if key[:len('T')] == 'T':
        unitName = 'K'
        scaledValue = value
    elif key[:len('B')] == 'B':
        unitName = 'mT'
        scaledValue = value*1e3
    elif key[:len('Eperp')] == 'Eperp':
        unitName = 'GHz'
        scaledValue = value/1e9
    elif key[:len('theta')] == 'theta' or key[:len('phi')] == 'phi':
        unitName = '°'
        scaledValue = np.degrees(value)
    elif (key[:len('exRatio')] == 'exRatio' or  
          key[:len('SSRatio')] == 'SSRatio' or 
          key[:len('piPulseFid')] == 'piPulseFid'):
        unitName = '1' if not remove1 else ''
        scaledValue = value
    elif key[:len('k')] == 'k':
        unitName = '1/us' if not remove1 else '/us'
        scaledValue = value/1e6
    elif key[:len('SSTauDecay')] == 'SSTauDecay':
        unitName = 'ns'
        scaledValue = value*1e9
    elif key[:len('opticAlign')] == 'opticAlign':
        unitName = '1/W' if not remove1 else '/W'
        scaledValue = value
    elif (key[:len('SSPhonE')] == 'SSPhonE' or 
          key[:len('phonCutoff')] == 'phonCutoff'):
        unitName = 'meV'
        scaledValue = value*1e3
    elif key[:len('colExRatio')] == 'colExRatio':
        unitName = 'kcts mW us'
        scaledValue = value*1e6
    elif key[:len('laserpower')] == 'laserpower':
        unitName = 'mW'
        scaledValue = value*1e3
    elif key[:len('elPhonCoup')] == 'elPhonCoup':
        unitName = '1/us meV^-3' if not remove1 else '/us meV^-3'
        scaledValue = value
    elif key[:len('background')] == 'background':
        unitName = 'kcts/mW'
        scaledValue = value/1e6
    elif key[:len('darkcounts')] == 'darkcounts':
        unitName = 'kcts'
        scaledValue = value/1e3
    else:
        if isinstance(value, (int, float)):
            unitName = '1' if not remove1 else ''
            scaledValue = value
        else:
            unitName = '1' if not remove1 else ''
            scaledValue = value

    return unitName, scaledValue

def formatParamValue(modeldict_item):
    """
    For an item (key, value) from a modeldict.items() list, return a string
    of the value with the common units.
    For more information, see makeModelDict() function.
    """    
    unitName, scaledValue = scaleParam(modeldict_item, remove1=True)
    key, value = modeldict_item
    if key[:len('T')] == 'T':
        string = f'{scaledValue:.1f}{unitName}'
    elif key[:len('B')] == 'B':
        string = f'{scaledValue:.1f}{unitName}'
    elif key[:len('Eperp')] == 'Eperp':
        string = f'{scaledValue:.1f}{unitName}'
    elif key[:len('theta')] == 'theta' or key[:len('phi')] == 'phi':
        string = f'{scaledValue:.1f}{unitName}'
    elif (key[:len('exRatio')] == 'exRatio' or  
          key[:len('SSRatio')] == 'SSRatio' or 
          key[:len('piPulseFid')] == 'piPulseFid'):
        string = f'{scaledValue:.2f}{unitName}'
    elif key[:len('k')] == 'k':
        string = f'{scaledValue:.1f}{unitName}'
    elif key[:len('SSTauDecay')] == 'SSTauDecay':
        string = f'{scaledValue:.0f}{unitName}'
    elif key[:len('opticAlign')] == 'opticAlign':
        string = f'{scaledValue:.1f}{unitName}'
    elif (key[:len('SSPhonE')] == 'SSPhonE' or
          key[:len('phonCutoff')] == 'phonCutoff'):
        string = f'{scaledValue:.1f}{unitName}'
    elif key[:len('colExRatio')] == 'colExRatio':
        string = f'{scaledValue:.1f}{unitName}'
    elif key[:len('laserpower')] == 'laserpower':
        string = f'{scaledValue:.2f}{unitName}'
    elif key[:len('elPhonCoup')] == 'elPhonCoup':
        string = f'{scaledValue:.0f}{unitName}'
    elif key[:len('background')] == 'background':
        string = f'{scaledValue:.1f}{unitName}'
    elif key[:len('darkcounts')] == 'darkcounts':
        string = f'{scaledValue:.1f}{unitName}'
    else:
        if isinstance(value, bool):
            string = f'{scaledValue}{unitName}'
        elif isinstance(value, (int, float)):
            string = f'{scaledValue:.2e}{unitName}'
        else:
            string = f'{scaledValue}{unitName}'
    return string

def getParamStr(modeldict_item):
    """
    For an item (key, value) from a modeldict.items() list, return a string
    'key = value' in the common units.
    For more information, see makeModelDict() function.
    """
    key, value = modeldict_item
    string = formatParamValue(modeldict_item)
    return f'{key} = {string}'

def printModelDict(modeldict, reducedAngles=True, printit=True):
    """
    Print a modeldict in common formatted units.
    For more information, see makeModelDict() function.
    """
    dic = deepcopy(modeldict)
    for key in deepcopy(dic):
        if dic[key]==None:
            dic.pop(key)
    if reducedAngles:
        items = switchToReducedAngles(dic).items()
    else:
        items = dic.items()
    string=''.join([
                   (f'{name:<10}' + '= ' + formatParamValue((name, val)) + '\n')
                   for name, val in items
                   ])[:-1]
    if printit:
        print(string)
        return None
    else:
        return string


##############
#%% NV Models
##############

class NVrateModel(object):
    """
    Base class for all NV models.
    To create an NV model (here at the example of the MEmodel), either use:
        
        * the makeModelDict() functions defaults by "MEmodel()",
        
        * provide specific parameters that should be different by e.g. \
        "MEmodel(Eperp=1e9, phiE=.1)",
        
        * or use a previously generated "modeldict=makeModelDict()" and e.g. \
        modified "modeldict['Eperp']=1e9" modeldict via "MEmodel(**modeldict)".
    
    All attributes like self.modeldict, self.H (the Hamiltonian in EZ basis),
    self.emittingLevelIdxs, self.basisNameOptions (see self.population())
    are only for information purposes and cannot be changed.
    
    Use printModelDict(self.modeldict) to get an overview of parameters.
    """
    def __init__(self, **modeldict):
        self.modeldict = makeModelDict(**modeldict) # generate from modeldict kwargs
        self.H = None # Hamiltonian (later should be of type Hamiltonian) # units: Hz
        self.emittingLevelIdxs = () # just for information purposes.
        self.basisNameOptions = [] # just for information purposes.

    def population(self, state, basisName='EZ'):
        """
        Given a state state of the model, return the population vector that has 
        in each component the probability that the state is in the respective 
        basisName basis state.
        
        A print 'state-name : population in it' can be obtained via
        printPop(basisName, self.population(state, basisName))
        
        Parameters
        ----------
        state : numpy.ndarray
            Depending on the model, this is a vector in EIG basis for classical 
            rate models or a density matrix in EZ basis for the MEmodel.
        basisName : str, optional
            Possible basisNames are self.basisNameOptions. For more information, see
            BasisStateNames().
        
        Returns
        -------
        pop : numpy.ndarray
            Probabilities to be in the basisName basis.

        """
        pass
    
    def getPureState(self, idx, idxInBasisName='EZ'):
        """
        Get a pure state (just population in index idx) of the basis 
        idxInBasisName.
        
        ATTENTION: Currently only works reliable for MEmodel due to inaccuracy
        of inverse() function used in classical rate models.        
        
        For more information, see BasisStateNames().       

        Parameters
        ----------
        idx : int
            index of the state in the basis idxInBasisName

        Returns
        -------
        state : numpy.ndarray
            Depending on the model, this is a vector in EIG basis for classical 
            rate models or a density matrix in EZ basis for the MEmodel.

        """
        pass
    
    def getGSPolarization_ms0(self, state):
        """
        Range of polarization: [0,1], where 0 means no m_S=0 in the GS,
        m_S=1/3 is the thermal polarization, and m_S=1 is a perfectly
        initialized state.
        
        Parameters
        ----------
        state : numpy.ndarray
            Depending on the model, this is a vector in EIG basis for classical 
            rate models or a density matrix in EZ basis for the MEmodel.

        Returns
        -------
        GSPolarization_ms0 : float
            Amount of population in the EZ basis m_S=0 state of the GS       
        """
        return self.population(state, basisName='avgEZ')[1]

    def getESPolarization_ms0(self, state):
        """
        Range of polarization: [0,1], where 0 means no m_S=0 in the ES,
        and m_S=1 is the result of perfectly initialized GS followed by a 
        complete excitation into the ES.
        
        Parameters
        ----------
        state : numpy.ndarray
            Depending on the model, this is a vector in EIG basis for classical 
            rate models or a density matrix in EZ basis for the MEmodel.

        Returns
        -------
        ESPolarization_ms0 : float
            Amount of population in the EZ basis m_S=0 state of the ES
        """
        return self.population(state, basisName='avgEZ')[4]
    
    def steadyState(self):
        """        
        Get the steady state under laser illumination.
        If modeldict['laserpower'] == 0, return the thermal state instead.        
        
        Returns
        -------
        state : numpy.ndarray
            Depending on the model, this is a vector in EIG basis for classical 
            rate models or a density matrix in EZ basis for the MEmodel.  
        """
        pass
    
    def PL(self, state=None):
        """
        Get the PL of state state.
        
        All self.emittingLevelIdxs (= excited state levels) have the same
        emission rate. Based on this and the setup parameters (for more
        information see makeModelDict()) calculate the detected PL (see getPL()).

        
        Parameters
        ----------
        state : numpy.ndarray
            Depending on the model, this is a vector in EIG basis for classical 
            rate models or a density matrix in EZ basis for the MEmodel. 
            Defaults to the steady state if not specified.
        
        Returns
        -------
        PL : float
            Units: cts/s
        """
        pass
    
    def propagateState(self, Deltat, state):
        """
        Propagate the state state by a timestep Deltat.
        
        NOTE: It is computationally inefficient to call
        self.propagateState() in loops. Use self.calcTimeTrace() instead.

        Parameters
        ----------
        Deltat : float
            Units: s
            
        state : numpy.ndarray
            Depending on the model, this is a vector in EIG basis for classical 
            rate models or a density matrix in EZ basis for the MEmodel.

        Returns
        -------
        state : numpy.ndarray
            Depending on the model, this is a vector in EIG basis for classical 
            rate models or a density matrix in EZ basis for the MEmodel.
        """
        return self.calcTimeTrace(
            np.linspace(0, Deltat, num=2),
            state, basisName=None
            )[1][-1]
    
    def calcTimeTrace(self, times, state, basisName='EZ'):
        """
        The time evolution of state state is evaluated for each t in times.
        basisName determines in which basis the returned population vectors
        are written.
        
        Parameters
        ----------
        times : numpy.array
            Must be evenly spaced and of size>= 2. Units: s
            
        state : numpy.ndarray
            Depending on the model, this is a vector in EIG basis for classical 
            rate models or a density matrix in EZ basis for the MEmodel.
            
        basisName : str, optional
            Possible basisNames are self.basisNameOptions. For more information,
            see self.population().

        Returns
        -------
        times : numpy.array
            Same object as input times. Unit: s
            
        states : numpy.ndarray
            A list of model specific states for each time in times: states[i] 
            is state for times[i].
            Depending on the model, a state is a vector in EIG basis for classical 
            rate models or a density matrix in EZ basis for the MEmodel.
            
        pls : numpy.ndarray
            PL values for each time in times. Units: cts/s
            
        pops_X : numpy.ndarray
            A list of population vectors for each time in times.
            The probability of being in state j of the X=basisName basis at 
            times[i] is element pops_X[i,j].
            If basisName=None, pops_X are not calculated to increase the speed,
            and pops_X=None is returned.
        """
        pass




class MEmodel(NVrateModel):
    """
    Master equation based model to simulate the NV center population dynamics
    over the full range of strain/el. field, magnetic field, and temperature
    from cryogenic to room temperature.
     
    Note that one can provide a makeModelDict() as kwargs.
    States of this model are density matrices in EZ basis.
    For information, see base class: NVrateModel()
    For more information, see https://arxiv.org/abs/2304.02521.
    """
    name = 'MEmodel'
    def __init__(self,
                Eperp      = Eperp_default, # unit: Hz
                phiE       = phiE_default, # unit: RAD
                B          = B_default, # unit: T
                thetaB     = thetaB_default, # unit: RAD
                phiB       = phiB_default, # unit: RAD
                kr         = kr_default, # unit: 1/s
                kE12       = kE12_default, # unit: 1/s
                kA1        = kA1_default, # unit: 1/s
                kExy       = kExy_default, # unit: 1/s
                exRatio    = exRatio_default, # unit: 1
                SSRatio    = SSRatio_default, # unit: 1
                SSPhonE    = SSPhonE_default, # unit: eV
                SSTauDecay = SSTauDecay_default, # unit: s
                T          = T_default, # unit: K
                elPhonCoup = elPhonCoup_default, # unit: 1/us*1/meV^3
                phonCutoff = phonCutoff_default, # unit: eV
                laserpower = laserpower_default, # unit: W
                background = background_default, # unit: cps/W
                colExRatio = colExRatio_default, # unit: cps*W*s
                opticAlign = opticAlign_default, # unit: 1/W
                darkcounts = darkcounts_default, # unit: cps
                piPulseFid = piPulseFid_default, # unit: 1
                highT_trf  = highT_trf_default, # unit: bool
                **kwargs, #they are simply ignored.
                ):
        self.modeldict = makeModelDict(                          
                          Eperp     = Eperp,
                          phiE      = phiE,
                          B         = B,
                          thetaB    = thetaB,
                          phiB      = phiB,
                          kr        = kr,
                          kA1       = kA1,
                          kE12      = kE12,
                          kExy      = kExy,
                          exRatio   = exRatio,
                          SSRatio   = SSRatio,
                          SSPhonE   = SSPhonE,
                          SSTauDecay= SSTauDecay,
                          T         = T,
                          elPhonCoup= elPhonCoup,
                          phonCutoff= phonCutoff,
                          laserpower= laserpower,
                          background= background,
                          colExRatio= colExRatio,
                          opticAlign= opticAlign,
                          darkcounts= darkcounts,
                          piPulseFid= piPulseFid,
                          highT_trf = highT_trf,
                          )
        
        self.H = get_H_EZ_withSS(B, thetaB, phiB, Eperp, phiE, T)
        # NOTE: when using self.H, a factor of *2*pi is needed to convert the units of the Hamiltonian from Hz to rad/s.
        self.emittingLevelIdxs = [3, 4, 5, 6, 7, 8] # just for information purposes.

        # compute on the fly once only when needed:
        self.T_HFtoEZ_withSS = None
        self.T_EZtoHF_withSS = None
        self.T_ZBtoEZ_withSS = None
        self.T_EZtoZB_withSS = None
        self.T_EIGtoEZ_withSS = None
        self.T_EZtoEIG_withSS = None
        
        self.basisNameOptions = ['EZ','ZF','ZB','HF','avgEZ','EIG']
        
        self.LindbladOp_List_EZ = []
        
        self.LindbladOp_List_EZ.extend(
            makeIncoherentLindbladOpList(**self.modeldict)
            )
        
        self.LindbladOp_List_EZ.extend(
            makeCoherentLindbladOpList(**self.modeldict)
            )
        
        self._generateLiovillian()
        self._ULmemory = {}
    
    def _generateLiovillian(self):
        self._Lnp = getLiovillian(self.H*2*np.pi, *self.LindbladOp_List_EZ)
    
    def _makeT_ZB(self):
        self.T_ZBtoEZ_withSS = get_T_ZBtoEZ_withSS(
            self.modeldict['B'], self.modeldict['thetaB'],
            self.modeldict['phiB'], self.modeldict['Eperp'],
            self.modeldict['phiE']
            )
        self.T_EZtoZB_withSS = conjTransp(self.T_ZBtoEZ_withSS)
    
    def _makeT_HF(self):
        self.T_HFtoEZ_withSS = get_T_HFtoEZ_withSS(
            self.modeldict['B'], self.modeldict['thetaB'],
            self.modeldict['phiB'], self.modeldict['Eperp'],
            self.modeldict['phiE']
            )
        self.T_EZtoHF_withSS = conjTransp(self.T_HFtoEZ_withSS)
        
    def _makeT_EIG(self):
        self.T_EIGtoEZ_withSS = get_T_EIGtoX(self.H)
        self.T_EZtoEIG_withSS = conjTransp(self.T_EIGtoEZ_withSS) 
    
    def population(self, state, basisName='EZ'):
        """
        For information, see base class method: NVrateModel.population()
        """
        if basisName=='EZ':
            diag = np.diag(state)
        elif basisName=='avgEZ':
            diag_EZ = np.diag(state)
            diag = np.array([diag_EZ[0],diag_EZ[1],diag_EZ[2],
                             diag_EZ[3]+diag_EZ[6],
                             diag_EZ[4]+diag_EZ[7],
                             diag_EZ[5]+diag_EZ[8],
                             diag_EZ[9],
                             ])
        elif basisName=='ZF':
            diag = np.diagonal(
                basisTrafo(state, T_ZFtoEZ_withSS, 
                           T_old_to_new=T_EZtoZF_withSS)
                )
        elif basisName=='ZB':
            if self.T_ZBtoEZ_withSS is None:
                self._makeT_ZB()
            diag = np.diagonal(
                basisTrafo(state, self.T_ZBtoEZ_withSS, 
                           T_old_to_new=self.T_EZtoZB_withSS)
                )
        elif basisName=='HF':
            if self.T_HFtoEZ_withSS is None:
                self._makeT_HF()
            diag = np.diagonal(
                basisTrafo(state, self.T_HFtoEZ_withSS, 
                           T_old_to_new=self.T_EZtoHF_withSS)
                )
        elif basisName=='EIG':
            if self.T_EIGtoEZ_withSS is None:
                self._makeT_EIG()
            diag = np.diagonal(
                basisTrafo(state, self.T_EIGtoEZ_withSS,
                           T_old_to_new=self.T_EZtoEIG_withSS)
                )
        else:
            errstr = f'Basis \'{basisName}\' unknown to {self.__class__.name}.\
                Options: {self.basisNameOptions}'
            raise NotImplementedError(errstr)
        pop = np.real(diag)
        return pop

    def getPureState(self, idx, idxInBasisName='EZ'):
        """
        For information, see base class method: NVrateModel.getPureState()
        """
        state_Basis = np.zeros((10,10), dtype = np.complex128)
        state_Basis[idx,idx] = 1.0
        if idxInBasisName=='EZ':
            state = state_Basis
        elif idxInBasisName=='ZF':
            state = basisTrafo(state_Basis, T_EZtoZF_withSS, 
                               T_old_to_new=T_ZFtoEZ_withSS)
        elif idxInBasisName=='ZB':
            if self.T_ZBtoEZ_withSS is None:
                self._makeT_ZB()
            state = basisTrafo(state_Basis, self.T_EZtoZB_withSS, 
                               T_old_to_new=self.T_ZBtoEZ_withSS)
        elif idxInBasisName=='HF':
            if self.T_HFtoEZ_withSS is None:
                self._makeT_HF()
            state = basisTrafo(state_Basis, self.T_EZtoHF_withSS,
                               T_old_to_new=self.T_HFtoEZ_withSS)
        elif idxInBasisName=='EIG':
            if self.T_EIGtoEZ_withSS is None:
                self._makeT_EIG()
            state = basisTrafo(state_Basis, self.T_EZtoEIG_withSS,
                               T_old_to_new=self.T_EIGtoEZ_withSS)
        else:
            errstr = f'Basis states of basis \'{idxInBasisName}\' \
have no state representation in {self.__class__.name}.'
            raise NotImplementedError(errstr)
        return state
    
    def steadyState(self):
        """
        For information, see base class method: NVrateModel.steadyState()
        """
        if self.modeldict['laserpower'] == 0:
            n = self._Lnp.shape[0]
            m = int(np.sqrt(n))
            shape = (m,m)
            thermalstate = np.zeros(shape, dtype=np.complex128)
            thermalstate[[0,1,2],[0,1,2]] = 1/3.
            return thermalstate

        # OPTION 1: use expm(self._Lnp*tstep) for a long tstep. But this is not a clean way.
        # works, 20% slower than qu.steadystate (without using numba)
        # tstep = 1e-3
        # UL = expm(self._Lnp*tstep)
        # n = self._Lnp.shape[0]
        # m = int(np.sqrt(n))
        # shape = (m,m)
        # rho0 = np.zeros(shape, dtype=(np.complex128))
        # rho0[0,0] = 1.0
        # states, pls = propagate(2, UL, rho0, 0, 0, 0, 0, 0, 0)
        # state = states[-1]
        # state = 0.5 * (state + conjTransp(state)) # make Hermitian
        
        # OPTION 2: use explicit eigenvalue solver.
        # works, 20% slower than qutip (without using numba, using numba with ascontiguousarray makes slower)
        # state = getsteadystate_byEig(self._Lnp)
        
        # OPTION 3: solve by linear matrix equation (dense matrix).
        # as fast as qutip (using numba does not help or hurt)
        state = getsteadystate_byLin(self._Lnp)

        return state

    def PL(self, state=None):
        """
        For information, see base class method: NVrateModel.PL()
        """
        if state is None:
            state = self.steadyState()
        PL = calcPL(
            state, self.modeldict['kr'], self.modeldict['colExRatio'],
            self.modeldict['opticAlign'], self.modeldict['background'],
            self.modeldict['laserpower'], self.modeldict['darkcounts'])
        return PL

    def calcTimeTrace(self, times, state, basisName='EZ'):
        """
        For information, see base class method: NVrateModel.calcTimeTrace()
        """
        tstep = (times[1] - times[0])
        
        loaded = False
        if self._ULmemory.keys():
            keys = list(self._ULmemory.keys())
            idx = getIndx(tstep, keys)
            if abs(tstep - keys[idx]) < 1e-20:
                UL = self._ULmemory[keys[idx]]
                loaded = True
        if not loaded:
            UL = expm(self._Lnp*tstep)
            self._ULmemory[tstep] = UL # no deep copy
        
        steps=times.size

        states, pls = propagate(
            steps, UL, state, self.modeldict['kr'], self.modeldict['colExRatio'],
            self.modeldict['opticAlign'], self.modeldict['background'],
            self.modeldict['laserpower'], self.modeldict['darkcounts'])
        if basisName is not None:
            pops_X = np.array([
                self.population(state, basisName=basisName) for state in states])
        else:
            pops_X = None       
        return times, states, pls, pops_X




class LowTmodel(NVrateModel):
    """
    Classical rate equation model to simulate the NV center population dynamics
    at cryogenic temperature.
    
    Up to around 30K, the classical LowTmodel() shows a similar behavior as the
    MEmodel() and might be useful as an approximation with significantly
    enhanced computational speed. The correct treatment of orbital hopping due 
    to electron-phonon coupling is the MEmodel(), though. Here, the orbital 
    hopping is introduced as a classical rate that also destroys the spin space
    coherences (which is wrong).    

    Note that one can provide a makeModelDict() as kwargs.
    States of this model are population vectors in EIG basis.
    For information, see base class: NVrateModel()
    """
    name = 'LowTmodel'
    def __init__(self,
                Eperp      = Eperp_default, # unit: Hz
                phiE       = phiE_default, # unit: RAD
                B          = B_default, # unit: T
                thetaB     = thetaB_default, # unit: RAD
                phiB       = phiB_default, # unit: RAD
                kr         = kr_default, # unit: 1/s
                kE12       = kE12_default, # unit: 1/s
                kA1        = kA1_default, # unit: 1/s
                kExy       = kExy_default, # unit: 1/s
                exRatio    = exRatio_default, # unit: 1
                SSRatio    = SSRatio_default, # unit: 1
                SSPhonE    = SSPhonE_default, # unit: eV
                SSTauDecay = SSTauDecay_default, # unit: s
                T          = T_default, # unit: K
                elPhonCoup = elPhonCoup_default, # unit: 1/us*1/meV^3
                phonCutoff = phonCutoff_default, # unit: eV
                laserpower = laserpower_default, # unit: W
                background = background_default, # unit: cps/W
                colExRatio = colExRatio_default, # unit: cps*W*s
                opticAlign = opticAlign_default, # unit: 1/W
                darkcounts = darkcounts_default, # unit: cps
                piPulseFid = piPulseFid_default, # unit: 1
                highT_trf  = highT_trf_default, # unit: bool
                **kwargs, #they are simply ignored.
                ):
        self.modeldict = makeModelDict( Eperp     = Eperp,
                                        phiE      = phiE,
                                        B         = B,
                                        thetaB    = thetaB,
                                        phiB      = phiB,
                                        kr        = kr,
                                        kA1       = kA1,
                                        kE12      = kE12,
                                        kExy      = kExy,
                                        exRatio   = exRatio,
                                        SSRatio   = SSRatio,
                                        SSPhonE   = SSPhonE,
                                        SSTauDecay= SSTauDecay,
                                        T         = T,
                                        elPhonCoup= elPhonCoup,
                                        phonCutoff= phonCutoff,
                                        laserpower= laserpower,
                                        background= background,
                                        colExRatio= colExRatio,
                                        opticAlign= opticAlign,
                                        darkcounts= darkcounts,
                                        piPulseFid= piPulseFid,
                                        highT_trf = highT_trf,
                                        )
        
        self.H = get_H_EZ_withSS(B, thetaB, phiB, Eperp, phiE, T)
        # NOTE: when using self.H, a factor of *2*pi is needed to convert the units of the Hamiltonian from Hz to rad/s.

        self.avgedES = False # indicates whether a classical rate model is based on the averaged Hamiltonian.
        
        self.emittingLevelIdxs = [3, 4, 5, 6, 7, 8] # just for information purposes.
        
        # compute on the fly once only when needed:
        self.prob_EIGinEZ_withSS = None
        self.prob_EIGinHF_withSS = None
        self.prob_EIGinZF_withSS = None
        self.prob_EIGinZB_withSS = None
        self.T_HFtoEZ_withSS, Delta = get_T_HFtoEZ_withSS_andDelta(
            B, thetaB, phiB,Eperp, phiE)
        self.T_EIGtoEZ_withSS = get_T_EIGtoX(self.H)
        
        self.basisNameOptions = ['EZ','ZF','ZB','HF','avgEZ']
            
        kmixDown, kmixUp, kT2orb = getOrbitalRates_scalar(
            T, Delta, elPhonCoup=elPhonCoup, phononCutoffEnergy=phonCutoff)
        
        self.RM = makeRateMatrix_LowTmodel(self.T_EIGtoEZ_withSS,
                                           self.T_HFtoEZ_withSS,
                                           laserpower = laserpower,
                                           opticAlign = opticAlign, 
                                           exRatio = exRatio,
                                           kr   = kr,
                                           kA1  = kA1,
                                           kE12 = kE12,
                                           kExy = kExy,
                                           T = T,
                                           SSTauDecay = SSTauDecay,
                                           SSPhonE = SSPhonE,
                                           SSRatio = SSRatio,
                                           kmixUp = kmixUp,
                                           kmixDown = kmixDown,
                                           )
                    
    def _makeprob_EZ(self):
        T_XtoEZ_withSS = Id10
        T_EIGtoX_withSS = get_T_XtoY(self.T_EIGtoEZ_withSS, T_XtoEZ_withSS)
        self.prob_EIGinEZ_withSS = getProb_YinX(T_EIGtoX_withSS)
    
    def _makeprob_ZF(self):
        T_XtoEZ_withSS = T_ZFtoEZ_withSS
        T_EIGtoX_withSS = get_T_XtoY(self.T_EIGtoEZ_withSS, T_XtoEZ_withSS)
        self.prob_EIGinZF_withSS = getProb_YinX(T_EIGtoX_withSS)
    
    def _makeprob_ZB(self):
        T_XtoEZ_withSS = get_T_ZBtoEZ_withSS(
                self.modeldict['B'], self.modeldict['thetaB'],
                self.modeldict['phiB'], self.modeldict['Eperp'],
                self.modeldict['phiE']
                )
        T_EIGtoX_withSS = get_T_XtoY(self.T_EIGtoEZ_withSS, T_XtoEZ_withSS)
        self.prob_EIGinZB_withSS = getProb_YinX(T_EIGtoX_withSS)
    
    def _makeprob_HF(self):
        T_XtoEZ_withSS = self.T_HFtoEZ_withSS 
        T_EIGtoX_withSS = get_T_XtoY(self.T_EIGtoEZ_withSS, T_XtoEZ_withSS)
        self.prob_EIGinHF_withSS = getProb_YinX(T_EIGtoX_withSS)
    
    def population(self, state, basisName='EZ'):
        """
        For information, see base class method: NVrateModel.population()
        """
        if basisName=='EIG':
            return np.copy(state).astype(np.float64) # required for python 2.7
        elif basisName=='EZ' or basisName=='avgEZ':
            if self.prob_EIGinEZ_withSS is None:
                self._makeprob_EZ()
            prob_EIGinX_withSS = self.prob_EIGinEZ_withSS
        elif basisName=='ZF':
            if self.prob_EIGinZF_withSS is None:
                self._makeprob_ZF()
            prob_EIGinX_withSS = self.prob_EIGinZF_withSS
        elif basisName=='ZB':
            if self.prob_EIGinZB_withSS is None:
                self._makeprob_ZB()
            prob_EIGinX_withSS = self.prob_EIGinZB_withSS
        elif basisName=='HF':
            if self.prob_EIGinHF_withSS is None:
                self._makeprob_HF()
            prob_EIGinX_withSS = self.prob_EIGinHF_withSS
        else:
            errstr = f'Basis \'{basisName}\' unknown to {self.__class__.name}. \
Options: {self.basisNameOptions}'
            raise NotImplementedError(errstr)
        
        state = np.copy(state).astype(np.float64) # required for python 2.7
        pop = np.dot(prob_EIGinX_withSS, state)

        if basisName=='avgEZ':
            pop_EZ = np.copy(pop)
            pop = np.array([pop_EZ[0],pop_EZ[1],pop_EZ[2],
                            pop_EZ[3]+pop_EZ[6],
                            pop_EZ[4]+pop_EZ[7],
                            pop_EZ[5]+pop_EZ[8],
                            pop_EZ[9],
                            ])
        return pop
    
    def getPureState(self, idx, idxInBasisName='EZ'):
        """
        For information, see base class method: NVrateModel.getPureState().
                
        ATTENTION: does not work for e.g. low strain or B.
        Currently only works reliable for MEmodel due to inaccuracy
        of inverse() function used in classical rate models.
        """
        state_X = np.zeros(10, dtype=np.float64)
        state_X[idx] = 1.0
        if idxInBasisName=='EIG':
            prob_EIGinX_withSS = np.eye(10, dtype=np.float64)
        elif idxInBasisName=='EZ':
            if self.prob_EIGinEZ_withSS is None:
                self._makeprob_EZ()
            prob_EIGinX_withSS = self.prob_EIGinEZ_withSS
        elif idxInBasisName=='ZF':
            if self.prob_EIGinZF_withSS is None:
                self._makeprob_ZF()
            prob_EIGinX_withSS = self.prob_EIGinZF_withSS
        elif idxInBasisName=='ZB':
            if self.prob_EIGinZB_withSS is None:
                self._makeprob_ZB()
            prob_EIGinX_withSS = self.prob_EIGinZB_withSS
        elif idxInBasisName=='HF':
            if self.prob_EIGinHF_withSS is None:
                self._makeprob_HF()
            prob_EIGinX_withSS = self.prob_EIGinHF_withSS
        else:
            errstr = f'Basis states of basis \'{idxInBasisName}\' \
have no state representation in {self.__class__.name}.'
            raise NotImplementedError(errstr)
        state = np.dot(inverse(prob_EIGinX_withSS), state_X)
        return state
        
    def steadyState(self):
        """
        For information, see base class method: NVrateModel.steadyState()
        """
        if self.modeldict['laserpower'] == 0:
            m = self.RM.shape[0]
            thermalstate = np.zeros(m, dtype=np.float64)
            thermalstate[[0,1,2]] = 1/3.
            return thermalstate
        state = getsteadystate_classical(self.RM)
        return state

    def PL(self, state=None):
        """
        For information, see base class method: NVrateModel.PL()
        """
        if state is None:
            state = self.steadyState()
        else:
            state = np.copy(state).astype(np.float64) # required for python 2.7
        PL = calcPL_classical(
            state, self.modeldict['kr'], self.modeldict['colExRatio'],
            self.modeldict['opticAlign'], self.modeldict['background'],
            self.modeldict['laserpower'], self.modeldict['darkcounts'])
        return PL

    def calcTimeTrace(self, times, state, basisName='EZ'):
        """
        For information, see base class method: NVrateModel.calcTimeTrace()
        """
        # VERSION 1: does not accumulate errors over steps but takes longer to compute.
        # S = np.diag(np.sum(self.RM, axis=1))
        # M = (np.transpose(self.RM) - S)
        # reltimes = times - times[0]
        # states = [expm(M * t).dot(state) for t in reltimes]

        # VERSION 2: faster but can accumulate errors just as the MEmodel version:
        tstep = (times[1] - times[0])
        steps=times.size
        M = getPropagationMatrix_classical(self.RM)
        UL = expm(M * tstep)
        state = np.copy(state).astype(np.float64) # required for python 2.7
        
        states, pls = propagate_classical(
            steps, UL, state, self.modeldict['kr'], self.modeldict['colExRatio'],
            self.modeldict['opticAlign'], self.modeldict['background'],
            self.modeldict['laserpower'], self.modeldict['darkcounts'])
        
        if basisName is not None:
            probs = np.array([
                self.population(state, basisName=basisName) for state in states])
        else:
            probs = None
        return times, states, pls, probs

class SZmodel(LowTmodel):
    """
    Classical rate equation model to simulate the NV center population dynamics
    at cryogenic temperature.
    This rate model is only applicable, and then similar to the
    LowTmodel(), when the eigenstates are approximately given by
    the EZ basis states, which is the case for high magnetic field and much 
    higher strain splitting caused by Eperp. This model was used in Happacher2022Low
    https://link.aps.org/doi/10.1103/PhysRevLett.128.177401.
    
    Use MEmodel or, if speed and not accuracy at low temperature matters, the 
    LowTmodel. The purpose of this model is only to compare its applicability.

    Note that one can provide a makeModelDict() as kwargs.
    States of this model are population vectors in EIG basis.
    For information, see base class: NVrateModel()
    """
    name = 'SZmodel'
    def __init__(self,
                Eperp      = Eperp_default, # unit: Hz
                phiE       = phiE_default, # unit: RAD
                B          = B_default, # unit: T
                thetaB     = thetaB_default, # unit: RAD
                phiB       = phiB_default, # unit: RAD
                kr         = kr_default, # unit: 1/s
                kE12       = kE12_default, # unit: 1/s
                kA1        = kA1_default, # unit: 1/s
                kExy       = kExy_default, # unit: 1/s
                exRatio    = exRatio_default, # unit: 1
                SSRatio    = SSRatio_default, # unit: 1
                SSPhonE    = SSPhonE_default, # unit: eV
                SSTauDecay = SSTauDecay_default, # unit: s
                T          = T_default, # unit: K
                elPhonCoup = elPhonCoup_default, # unit: 1/us*1/meV^3
                phonCutoff = phonCutoff_default, # unit: eV
                laserpower = laserpower_default, # unit: W
                background = background_default, # unit: cps/W
                colExRatio = colExRatio_default, # unit: cps*W*s
                opticAlign = opticAlign_default, # unit: 1/W
                darkcounts = darkcounts_default, # unit: cps
                piPulseFid = piPulseFid_default, # unit: 1
                highT_trf  = highT_trf_default, # unit: bool                
                **kwargs, #they are simply ignored.
                ):
        self.modeldict = makeModelDict( Eperp     = Eperp,
                                        phiE      = phiE,
                                        B         = B,
                                        thetaB    = thetaB,
                                        phiB      = phiB,
                                        kr        = kr,
                                        kA1       = kA1,
                                        kE12      = kE12,
                                        kExy      = kExy,
                                        exRatio   = exRatio,
                                        SSRatio   = SSRatio,
                                        SSPhonE   = SSPhonE,
                                        SSTauDecay= SSTauDecay,
                                        T         = T,
                                        elPhonCoup= elPhonCoup,
                                        phonCutoff= phonCutoff,
                                        laserpower= laserpower,
                                        background= background,
                                        colExRatio= colExRatio,
                                        opticAlign= opticAlign,
                                        darkcounts= darkcounts,
                                        piPulseFid= piPulseFid,
                                        highT_trf = highT_trf,
                                        )
        
        self.H = get_H_EZ_withSS(B, thetaB, phiB, Eperp, phiE, T)
        # NOTE: when using self.H, a factor of *2*pi is needed to convert the units of the Hamiltonian from Hz to rad/s.
        
        self.emittingLevelIdxs = [3, 4, 5, 6, 7, 8] # just for information purposes.

        self.avgedES = False # indicates whether a classical rate model is based on the averaged Hamiltonian.
        
        # compute on the fly once only when needed:
        self.prob_EIGinEZ_withSS = None
        self.prob_EIGinHF_withSS = None
        self.prob_EIGinZF_withSS = None
        self.prob_EIGinZB_withSS = None
        self.T_HFtoEZ_withSS, Delta = get_T_HFtoEZ_withSS_andDelta(
            B, thetaB, phiB,Eperp, phiE)
        self.T_EIGtoEZ_withSS = get_T_EIGtoX(self.H)
        
        self.basisNameOptions = ['EZ','ZF','ZB','HF','avgEZ']

        self.RM = makeRateMatrix_EZmodel(self.T_EIGtoEZ_withSS, avgedES=False,
                                         laserpower = laserpower,
                                         opticAlign = opticAlign, 
                                         exRatio = exRatio,
                                         kr   = kr,
                                         kE12 = kE12,
                                         kA1 = kA1,
                                         kExy = kExy,
                                         T = T,
                                         SSTauDecay = SSTauDecay,
                                         SSPhonE = SSPhonE,
                                         SSRatio = SSRatio,
                                         )


class HighTmodel(LowTmodel):
    """
    Classical rate equation model to simulate the NV center population dynamics
    around room temperature, as often employed in literature.
    
    As the orbital hopping rate of the MEmodel dominates the population dynamics,
    the MEmodel starts to resemble the behavior of this classical model.
    This HighTmodel model can thus be used for computations at high speed at 
    elevated temperature.
    
    Note that for high strain Eperp, some orbital properties remain at elevated
    temperature. These are only included in HighTmodel when highT_trf=True.

    Note that one can provide a makeModelDict() as kwargs.
    States of this model are population vectors in EIG basis of the orbitally
    averaged Hamiltonian.
    For information, see base class: NVrateModel()
    For more information, see https://arxiv.org/abs/2304.02521.
    """
    name = 'HighTmodel'
    def __init__(self,
                Eperp      = Eperp_default, # unit: Hz
                phiE       = phiE_default, # unit: RAD
                B          = B_default, # unit: T
                thetaB     = thetaB_default, # unit: RAD
                phiB       = phiB_default, # unit: RAD
                kr         = kr_default, # unit: 1/s
                kE12       = kE12_default, # unit: 1/s
                kA1        = kA1_default, # unit: 1/s
                kExy       = kExy_default, # unit: 1/s
                exRatio    = exRatio_default, # unit: 1
                SSRatio    = SSRatio_default, # unit: 1
                SSPhonE    = SSPhonE_default, # unit: eV
                SSTauDecay = SSTauDecay_default, # unit: s
                T          = T_default, # unit: K
                elPhonCoup = elPhonCoup_default, # unit: 1/us*1/meV^3
                phonCutoff = phonCutoff_default, # unit: eV
                laserpower = laserpower_default, # unit: W
                background = background_default, # unit: cps/W
                colExRatio = colExRatio_default, # unit: cps*W*s
                opticAlign = opticAlign_default, # unit: 1/W
                darkcounts = darkcounts_default, # unit: cps
                piPulseFid = piPulseFid_default, # unit: 1
                highT_trf  = highT_trf_default, # unit: bool
                **kwargs, #they are simply ignored.
                ):
        self.modeldict = makeModelDict( Eperp     = Eperp,
                                        phiE      = phiE,
                                        B         = B,
                                        thetaB    = thetaB,
                                        phiB      = phiB,
                                        kr        = kr,
                                        kA1       = kA1,
                                        kE12      = kE12,
                                        kExy      = kExy,
                                        exRatio   = exRatio,
                                        SSRatio   = SSRatio,
                                        SSPhonE   = SSPhonE,
                                        SSTauDecay= SSTauDecay,
                                        T         = T,
                                        elPhonCoup= elPhonCoup,
                                        phonCutoff= phonCutoff,
                                        laserpower= laserpower,
                                        background= background,
                                        colExRatio= colExRatio,
                                        opticAlign= opticAlign,
                                        darkcounts= darkcounts,
                                        piPulseFid= piPulseFid,
                                        highT_trf = highT_trf,
                                        )
        
        if not highT_trf:
            self.H = get_avgH_EZ_withSS(B, thetaB, phiB, Eperp, phiE, T)
        else:
            self.H = get_avgHTRF_EZ_withSS(B, thetaB, phiB, Eperp, phiE, T)
        # NOTE: when using self.H, a factor of *2*pi is needed to convert the units of the Hamiltonian from Hz to rad/s.     

        self.emittingLevelIdxs = [3, 4, 5] # just for information purposes.
        
        self.avgedES = True # indicates whether a classical rate model is based on the averaged Hamiltonian.
        
        # compute on the fly once only when needed:
        self.prob_EIGinEZ_withSS = None
        self.T_EIGtoEZ_withSS = get_T_EIGtoX(self.H)
        
        self.basisNameOptions = ['avgEZ']

        self.RM = makeRateMatrix_EZmodel(self.T_EIGtoEZ_withSS, avgedES=True,
                                         laserpower = laserpower,
                                         opticAlign = opticAlign, 
                                         exRatio = exRatio,
                                         kr   = kr,
                                         kE12 = kE12,
                                         kA1 = kA1,
                                         kExy = kExy,
                                         T = T,
                                         SSTauDecay = SSTauDecay,
                                         SSPhonE = SSPhonE,
                                         SSRatio = SSRatio,
                                         )
    
    def _makeprob_EZ(self):       
        T_XtoEZ_withSS = Id7
        T_EIGtoX_withSS = get_T_XtoY(self.T_EIGtoEZ_withSS, T_XtoEZ_withSS)
        self.prob_EIGinEZ_withSS = getProb_YinX(T_EIGtoX_withSS)
    
    def population(self, state, basisName='avgEZ'):
        """
        For information, see base class method: NVrateModel.population()
        """
        if basisName=='EIG':
            return np.copy(state).astype(np.float64) # required for python 2.7
        elif basisName=='avgEZ':
            if self.prob_EIGinEZ_withSS is None:
                self._makeprob_EZ()
            prob_EIGinX_withSS = self.prob_EIGinEZ_withSS
        else:
            errstr = f'Basis \'{basisName}\' unknown to {self.__class__.name}. \
Options: {self.basisNameOptions}'
            raise NotImplementedError(errstr)
            
        state = np.copy(state).astype(np.float64) # required for python 2.7
        pop = np.dot(prob_EIGinX_withSS, state)
        return pop
    
    def getPureState(self, idx, idxInBasisName='avgEZ'):
        """
        For information, see base class method: NVrateModel.getPureState()
        
        In this model, only idxInBasisName='EIG' is allowed.
        
        ATTENTION: does not work with e.g. B=50e-3, thetaB=1.
        Currently only works reliable for MEmodel due to inaccuracy
        of inverse() function used in classical rate models.
        """
        state_X = np.zeros(7, dtype=np.float64)
        state_X[idx] = 1.0
        if idxInBasisName=='EIG':
            prob_EIGinX_withSS = np.eye(7, dtype=np.float64)
        elif idxInBasisName=='avgEZ':
            if self.prob_EIGinEZ_withSS is None:
                self._makeprob_EZ()
            prob_EIGinX_withSS = self.prob_EIGinEZ_withSS
        else:
            errstr = f'Basis states of basis \'{idxInBasisName}\' \
have no state representation in {self.__class__.name}.'
            raise NotImplementedError(errstr)
        state = np.dot(inverse(prob_EIGinX_withSS), state_X)
        return state
        
    def PL(self, state=None):
        """
        For information, see base class method: NVrateModel.PL()
        """
        if state is None:
            state = self.steadyState()
        else:
            state = np.copy(state).astype(np.float64) # required for python 2.7
        PL = calcPL_classicalHT(
            state, self.modeldict['kr'], self.modeldict['colExRatio'],
            self.modeldict['opticAlign'], self.modeldict['background'],
            self.modeldict['laserpower'], self.modeldict['darkcounts'])
        return PL
    
    def calcTimeTrace(self, times, state, basisName='EZ'):     
        """
        For information, see base class method: NVrateModel.calcTimeTrace()
        """
        tstep = (times[1] - times[0])
        steps=times.size
        M = getPropagationMatrix_classical(self.RM)
        UL = expm(M * tstep)
        state = np.copy(state).astype(np.float64) # required for python 2.7
        
        states, pls = propagate_classicalHT(
            steps, UL, state, self.modeldict['kr'], self.modeldict['colExRatio'],
            self.modeldict['opticAlign'], self.modeldict['background'],
            self.modeldict['laserpower'], self.modeldict['darkcounts'])

        if basisName is not None:
            probs = np.array([
                self.population(state, basisName=basisName) for state in states])
        else:
            probs = None
        return times, states, pls, probs


##############################
#%% Operations with NV Models
##############################

def initState(k0, kbeta, twait=1e-3):
    """
    For given NV rate models, return the spin m_S=0 initialized state.
    Init sequence: steady state with laser on, then laser off for twait seconds.

    Parameters
    ----------
    k0 : NVrateModel
        NV rate models with laserpower = 0.
    kbeta : NVrateModel
        Same NV rate models type with laserpower > 0.

    Returns
    -------
    state : numpy.ndarray
        Depending on the NV model, this is a vector in EIG basis for classical 
        rate models or a density matrix in EZ basis for the MEmodel.
    """
    state = kbeta.steadyState() # laser on until steady state...
    state = k0.propagateState(twait, state) # ... then laser off for twait
    # kill all coherences? diag elements still have some small complex part...
    return state

def piPulse(state, level1=0, level2=1, piPulseFid=1.0):
    """
    Apply an instantaneous, pi-pulse between the levels with index level1 and
    level2.
    Which levels these are depends on the state. Thus, one should make sure that
    state is always in EIG basis. This is natively the case for classical models,
    but for MEmodel, one has to do a basis transformation first and after piPulse.
    
    The fidelity of the pi-pulse piPulseFid is in [0, 1.0].
    
    Parameters
    ----------
    state : numpy.ndarray
        Depending on the NV model, this is a vector in EIG basis for classical 
        rate models or a density matrix in (ideally) EIG basis for the MEmodel.   
    
    Returns
    -------
    state : numpy.ndarray
        Depending on the NV model, this is a vector in EIG basis for classical 
        rate models or a density matrix in (ideally) EIG basis for the MEmodel.
    """
    if state.ndim==2: # MEmodel in use:
        statenew = np.copy(state).astype(np.complex128) # required for python 2.7
        statenew[level1,level1] = (state[level2,level2]*piPulseFid 
                               + state[level1,level1]*(1-piPulseFid))
        statenew[level2,level2] = (state[level1,level1]*piPulseFid 
                               + state[level2,level2]*(1-piPulseFid))
    else: # 1D np.array, i.e. classical rate model in use:
        statenew = np.copy(state).astype(np.float64) # required for python 2.7
        statenew[level1] = (state[level2]*piPulseFid
                            + state[level1]*(1-piPulseFid))
        statenew[level2] = (state[level1]*piPulseFid
                            + state[level2]*(1-piPulseFid))
    return statenew


def makeTstepsFromDurations(tdurations, t0=0):
    """
    Take a list of time durations [s] that are parts of a sequence
    and return a list of times that mark the end of the
    respective part of the sequence.
    
    This format is used by the function calcTimeTrace().
    """
    t=t0
    tsteps=[]
    for i in range(len(tdurations)):
        t += tdurations[i]
        tsteps.append(t)
    return tsteps

def makeStepsForLaserRise(t0, tend, tauR=25e-9,
                          Delta_t=5e-9, N=4,
                          modelClass=MEmodel,
                          makePlot=False,
                          **modeldict):
    """
    Make a list of times tsteps and NV rate models to simulate the switch-on
    process of a laser.
    
    This format is used by the function calcTimeTrace().
    
    The laser power rises exponentially with a rise time tauR and in discrete
    steps of size Delta_t. All units are in [s].
    The laser turns on at t0, so first tstep is t0+Delta_t with the first
    kstep where the laser is a bit on. In the logic of ksteps,tsteps formalism
    (for details see calcTimeTrace()), this means that the laser is on 
    for all times t>t0. Use makePlot=True to view the behavior.
    The last tstep is tend with laser staying on the respective power level
    reached at that point.
    
    Assume constant laserpower after N*tauR.
    For tauR=0, simply a rectangular pulse is returned and Delta_t, N are ignored.
    In modeldict the laser has to be on: laserpower>0 (or default, which is on).
    
    Parameters
    ----------
    t0, tend, tauR, Delta_t : floats
        Units: s
    
    N : int
        see above
    
    modelClass : NVrateModel, optional
        Specify which rate model to use. Options:
        MEmode (default), HighTmodel, LowTmodel, SZmodel
        
    modeldict : dict, optional
        Optional keyword arguments can be provided by a modeldict or separately.
        For more details, see makeModelDict().
    
    
    Returns
    -------
    tsteps : list of float
        Monotonically increasing time steps of a sequence. Unit: s
    
    ksteps : list of NVrateModel
        ksteps[i] are the NVrateModel objects that are active until tsteps[i].
    """
    modeldict = makeModelDict(**modeldict) # generate from modeldict kwargs
    laserpowerFull = modelClass(**modeldict).modeldict['laserpower']
    if tauR == 0:
        tsteps = [t0,]
        modeldict['laserpower']=0.
        ksteps = [modelClass(**modeldict),]
        tsteps.append(tend)
        modeldict['laserpower']=laserpowerFull
        ksteps.append(modelClass(**modeldict))
    else:
        modeldict['laserpower']=0.
        ksteps = [modelClass(**modeldict),]
        tsteps = [t0,]
 
        if N*tauR < Delta_t:
            raise ValueError('N*tauR < Delta_t but has to be larger.')
        num = int( ( min(t0+N*tauR, tend) - (t0+Delta_t) )/Delta_t )
        times = np.linspace(t0+Delta_t, t0+num*Delta_t, num=num)
        
        if t0+N*tauR > tend and tend > times[-1]:
            times = np.append(times,tend)    
        laserpowerValues = laserpowerFull*(1-np.exp(-(times-t0)/tauR))
        
        for laserpower in laserpowerValues:
            modeldict['laserpower']=laserpower
            ksteps.append(modelClass(**modeldict))
        tsteps += list(times)
        if tend > tsteps[-1]:
            tsteps.append(tend)
            modeldict['laserpower']=laserpowerFull
            ksteps.append(modelClass(**modeldict))
        
        if makePlot:
            unit, scaling = scaleParam(('laserpower',1))
            name = f'Laser power with exponential rise time \
tauR={tauR*1e9:.2f}ns, Delta_t={Delta_t*1e9:.2f}ns, N={N:.0f}'
            fig = plt.figure(figsize=(8,5))
            fig.suptitle(name, fontsize='small')
            fig.set_tight_layout(True)
            axes = fig.add_subplot(111)
            laserpowers = np.array([kstep.modeldict['laserpower'] for kstep in ksteps])
            times = np.array(tsteps)
            axes.step(times*1e9,laserpowers*scaling,where='pre')
            axes.axvline(t0*1e9, color='black')
            axes.axvline(tend*1e9, color='black')
            axes.set_xlabel('time [ns]')
            axes.set_ylabel(f'laserpower [{unit}]')
            axes.grid(True)
            plt.show()
    return tsteps, ksteps

def calcTimeTrace(times, tsteps, ksteps, state0, basisName='EZ'):
    """    
    Propagate the initial state state0 at times[0] over all times in times
    under the sequence of conditions specified by tsteps, ksteps.
    
    The NV rate model specified by ksteps[i] is active during time t with
    tsteps[i-1] <= t < tsteps[i].
    
    This function is smart and can deal with any relation of times compared to
    tsteps and still yields the same physical result. Explicitly, this means the
    times sampling rate does not matter for the returned values at the same given 
    time.
 
    Note that in classical rate models, the Hamiltonian (self.H) of the ksteps
    should always be the same. For the MEmodel, this is not required.
    The reason is that the state_i after each time step in tsteps is used as the
    starting state for the next time step. But since in classical models the
    states are in the EIG basis, and the EIG basis changes with the Hamiltonian,
    the states change their meaning if self.H changes.
    
    For more details see NVrateModel().calcTimeTrace(), which is evaluated over
    the sequence tsteps, ksteps inside this function.
    Intentionally, the return of this function is the same as a return of 
    NVrateModel.calcTimeTrace().


    Note regarding computational speed:
    
    - ksteps usage: Only a fraction of the time spend here is the actual time \
    trace calculation. The rest is the construction of ksteps. \
    So if the same setting of times, tsteps, ksteps can be used again, \
    there is a gigantic speed-up. This is done in the \
    twoPointODMRTrace() and getContrast() functions.
    
    - For speed also consider that the relative position of tsteps and times can \
    make up to a factor 3 in speed since NVrateModel().propagateState() might \
    or might not be needed to fill the steps where the two are non-commensurate.
    
    
    Parameters
    ----------
    times : numpy.array
        Must be evenly spaced and of size>= 2. Units: s
    
    tsteps : list of float
        Monotonically increasing time steps of a sequence. Unit: s
    
    ksteps : list of NVrateModel
        ksteps[i] are the NVrateModel objects that are active until tsteps[i].
        
    state0 : numpy.ndarray
        Depending on the model, this is a vector in EIG basis for classical 
        rate models or a density matrix in EZ basis for the MEmodel.
        
    basisName : str, optional
        Possible basisNames are ksteps[i].basisNameOptions. For more information,
        see BasisStateNames().
        
        
    Returns
    -------
    times : numpy.array
        Same as input parameter. Units: s
    
    states : numpy.ndarray
        A list of model specific states for each time in times: states[i] 
        is state for times[i].
        Depending on the model, a state is a vector in EIG basis for classical 
        rate models or a density matrix in EZ basis for the MEmodel.
        If basisName=None, states are not returned to save memory,
        and states=None is returned.   
        
    pls : numpy.ndarray
        PL values for each time in times. Units: cts/s
        
    pops_X : numpy.ndarray
        A list of population vectors for each time in times.
        The probability of being in state j of the X=basisName basis at 
        times[i] is element pops_X[i,j]. For more information on basisNames,
        see NVrateModel.population().
        If basisName=None, pops_X are not calculated to increase the speed,
        and pops_X=None is returned.
    """
    if np.any(np.diff(tsteps) < 0):
        raise ValueError('tsteps are not monotonically increasing.')
    if times.size < 2:
        raise ValueError('times have size smaller than 2.')
    
    list_of_state = [] if basisName is not None else None
    pops_X = [] if basisName is not None else None
    PLs = []
    state = state0
    tstart = times[0]
    tend = times[-1]
    if tend >= tsteps[-1]:
        # extend last step (by appending the last step a second time) to cover all times:
        tsteps = tsteps + [times[-1]+1,] # just any time that is larger than all times elements.
        ksteps = ksteps + [ksteps[-1],]
    elif len(tsteps) > 1:
        if tend < tsteps[-2]:
            # remove all tsteps that are above the window in which tend lies:
            idx = np.argmax(np.array(tsteps)>tend) + 1
            tsteps = tsteps[:idx]
            ksteps = ksteps[:idx]
    # remove all steps that are before tstart:
    idx = np.argmax(np.array(tsteps)>tstart)
    tsteps = tsteps[idx:]
    ksteps = ksteps[idx:]
    for i in range(len(tsteps)):
        ti = tsteps[i]
        ki = ksteps[i]
        thistimes = times[(tstart <= times) & (times < ti)]
        if thistimes.size > 1:
            state = ki.propagateState(thistimes[0]-tstart, state)
            if basisName is not None:
                _, states, pls, pops = ki.calcTimeTrace(thistimes, state, basisName=basisName)
                pops_X.append(pops)
                list_of_state.append(states)
            else:
                _, states, pls, _ = ki.calcTimeTrace(thistimes, state, basisName=basisName)
            PLs.append(pls)
            state = states[-1]
        elif thistimes.size == 1:
            state = ki.propagateState(thistimes[0]-tstart, state)
            PLs.append(np.array([ki.PL(state=state)]))
            if basisName is not None:
                pops_X.append(np.expand_dims(ki.population(state, basisName=basisName),0))
                list_of_state.append(np.expand_dims(state,0))
        elif thistimes.size == 0:
            state = ki.propagateState(ti-tstart, state)
            tstart = ti
            continue
        state = ki.propagateState(ti-thistimes[-1], state)
        tstart = ti
    pls = np.concatenate(PLs)
    if basisName is not None:
        list_of_state = np.concatenate(list_of_state)
        pops_X = np.concatenate(pops_X)
    states = list_of_state
    return times, states, pls, pops_X

def twoPointODMRTrace(times, t1, t2, t3, t4,
                      state0 = None,
                      piPulseFirst=False, level1=0, level2=1,
                      tauR=0, Delta_t=5e-9, N=4,
                      modelClass=MEmodel,
                      **modeldict):
    """
    Return PL count rate [cts/s] at times times for a two-point pulsed ODMR 
    sequence.
    Sequence (for piPulseFirst=False): Starts with population vector state0 at
    t0=0 with Laser off, then on (at t1), then off (at t2),
    then pi-pulse(*) and laser on again (at t3), then off again (at t4).
    
    (*) NOTE: the second part is also computed for a start in state0 (see below).
    It is not a continuation of the population at the end of the first part.

    
    Parameters
    ----------
    times : numpy.array
        Must be evenly spaced and of size>= 2. Units: s
    
    t1, t2, t3, t4 : floats
        Monotonically increasing times of the sequence (see above). Unit: s
        
    state0 : numpy.ndarray, optional
        Depending on the modelClass, this is a vector in EIG basis for classical 
        rate models or a density matrix in EZ basis for the MEmodel.
        If state0=None (default), an initState() is done with the same laserpower 
        as the readout specified in modeldict.
        
    piPulseFirst : bool, optional
        Default False. If piPulseFirst=True, the pi pulse is applied right to
        state0 and not after the first laser pulse.
        For more information on level1, level2 see piPulse().
        
    level1, level2 : int, optional
        Indices of the levels in EIG basis between which the pi-pulse is applied
        to obtain the contrast. Which levels these are depends on the modeldict
        setting. You can use simulateEigenVsParam() to see the spin nature of
        the EIG levels.
        The pi-pulse fidelity is given by modeldict['piPulseFid'].
    
    tauR : float, optional
        Units: s. For optional parameters of the laser rise time tauR, Delta_t,
        and N, see makeStepsForLaserRise().
        By default, a simple rectangular pulse is used.
    
    modelClass : NVrateModel, optional
        Specify which rate model to use. Options:
        MEmode (default), HighTmodel, LowTmodel, SZmodel
        
    modeldict : dict, optional
        Optional keyword arguments can be provided by a modeldict or separately.
        For more details, see makeModelDict().
        NOTE: In modeldict the laser has to be on:
        'laserpower' > 0 (or default, which is on).
    
    
    Returns
    -------
    pls : numpy.ndarray
        PL values for each time in times. Units: cts/s
    """
    modeldict = makeModelDict(**modeldict) # generate from modeldict kwargs
    shift = 1e-16 # units: s; add this to ti since in the logic of tsteps/ksteps the condition
                  # should last until but excluding the next tstep. Here the times grid (e.g. 10ns)
                  # and the ti can coincide. This means the time element which is equal to t1s is
                  # already assigned to laser on, which then means that one propagates with laser
                  # on from the previous time step to this one. Which effectively means that the
                  # laser was already on a step before it is actually meant to be turned on.
                  # Thus shift to avoid this (also = is numerically unstable with floats especially when using np.arange).
    tsteps1, ksteps1 = makeStepsForLaserRise(t1+shift, t2+shift,
                                 tauR=tauR, Delta_t=Delta_t, N=N,
                                 modelClass=modelClass,
                                 **modeldict)
    kbeta = modelClass(**modeldict)
    modeldict0 = deepcopy(modeldict)
    modeldict0['laserpower']=0.
    k0 = modelClass(**modeldict0)
    if state0 is None:
        state0 = initState(k0, kbeta)
    else:
        state0 = state0
    
    # since makeStepsForLaserRise only turns laser on, we need to turn it off manually:
    extended = False
    if times[-1]>tsteps1[-1]:
        tsteps1.append(times[-1])
        ksteps1.append(k0)
        extended = True
    # Construction of ksteps2 takes almost half the time of this function call.
    # So use same ksteps1 and construct tsteps2 manually.
    # This might not behave as expected if t1/3+N*tauR > t2/4 but this is an experimentally undesired case in a normal pulsed-ODMR.
    if t1+N*tauR > t2 or t3+N*tauR > t4:
        tsteps2, ksteps2 = makeStepsForLaserRise(t3+shift, t4+shift,
                                      tauR=tauR, Delta_t=Delta_t, N=N,
                                      modelClass=modelClass,
                                      **modeldict)
        if times[-1]>tsteps2[-1]:
            tsteps2.append(times[-1])
            ksteps2.append(k0)
    else:
        ksteps2 = ksteps1 # same object
        tsteps2 = [t+(t3-t1) for t in tsteps1]
        if not extended:
            tsteps2[-1] = t4+shift
        else:
            tsteps2[-2] = t4+shift

    piPulseFid=kbeta.modeldict['piPulseFid']
    
    idx=np.argmax(times>t3)-1
    thistimes = times[:idx]
    
    if modelClass.name=='MEmodel':
        # convert the state (EZ basis for MEmodel) to the EIG basis:
        thismodel = ksteps1[0]
        _=thismodel.population(state0,'EIG') # to create thismodel.T_EIGtoEZ_withSS
        state0_EIG = basisTrafo(state0, thismodel.T_EIGtoEZ_withSS,
                                T_old_to_new=thismodel.T_EZtoEIG_withSS)
        # apply a state swap:
        state_EIG = piPulse(state0_EIG, level1=level1, level2=level2, 
                    piPulseFid=piPulseFid)
        # convert the EIG state back to the EZ basis of MEmodel states:
        state0_pi = basisTrafo(state_EIG, thismodel.T_EZtoEIG_withSS,
                               T_old_to_new=thismodel.T_EIGtoEZ_withSS)
    else: # classical models use EIG basis natively
        state0_pi = piPulse(state0, level1=level1, level2=level2,
                            piPulseFid=piPulseFid)
    
    _, _, PLs1, _ = calcTimeTrace(thistimes, tsteps1, ksteps1,
                                      state0 if not piPulseFirst else state0_pi,
                                      basisName=None,
                                      )
    thistimes = times[idx:]
    _, _, PLs2, _ = calcTimeTrace(thistimes, tsteps2, ksteps2,
                                      state0_pi if not piPulseFirst else state0,
                                      basisName=None,
                                      )
    pls = np.concatenate((PLs1, PLs2))
    return pls


@jitOpt
def getContrastOf2pointODMRTrace_Opt(times, pls, t1, t3,
                                 integrationMode,
                                 minLaserOnTime,
                                 piPulseFirst):
    t0, tpi = (t3, t1) if piPulseFirst else (t1, t3)
    
    tstep = times[1]-times[0] # times are equally spaced.
    
    mintInt = 50e-9 # at least that long the integration time has to be.
    minIdx = round(mintInt/tstep)
    epsilon = tstep*1e-6 # since ti are floats and might deviate slightly.

    csum = np.cumsum(pls)
    
    IntIdxRange = round(minLaserOnTime/tstep) # this index should not be included.
    
    startIntIdx = np.argmax(times > t0 + epsilon)
    refcsum = csum[startIntIdx:startIntIdx+IntIdxRange] - csum[startIntIdx-1]

    startIntIdx = np.argmax(times > tpi + epsilon)
    sigcsum = csum[startIntIdx:startIntIdx+IntIdxRange] - csum[startIntIdx-1]
    
    contrArray = 1 - sigcsum/refcsum
    
    if integrationMode == 'optC':
        idx = np.argmax(contrArray[minIdx:]) + minIdx  # idx starts at 0 and increases with increasing integration time.
    elif integrationMode == 'optSens':
        sensArray = np.sqrt(1-0.60653*contrArray)/np.sqrt(refcsum)/contrArray
        idx = np.argmin(sensArray[minIdx:]) + minIdx
    elif integrationMode == 'optSNR':
        sensArray = np.sqrt(1-0.5*contrArray)/np.sqrt(refcsum)/contrArray #*2 but irrelevant here
        idx = np.argmin(sensArray[minIdx:]) + minIdx
    else:
        raise NotImplementedError
    IntIdxRange = idx
    contrast = contrArray[idx]
    sig = sigcsum[idx]
    ref = refcsum[idx]
    tint =  tstep*idx
    
    sig = sig/float(IntIdxRange) # to convert summed countrates to average countrate during integration window
    ref = ref/float(IntIdxRange)
    return contrast, tint, sig, ref

@jitOpt
def getContrastOf2pointODMRTrace_Fix(times, pls, t1, t3,
                                 integrationTime,
                                 minLaserOnTime,
                                 piPulseFirst):
    t0, tpi = (t3, t1) if piPulseFirst else (t1, t3)
    
    tstep = times[1]-times[0] # times are equally spaced.
    
    # mintInt = 50e-9 # at least that long the integration time has to be.
    # minIdx = round(mintInt/tstep)
    epsilon = tstep*1e-6 # since ti are floats and might deviate slightly.

    IntIdxRange = round(integrationTime/tstep) + 1 # this index should be included.
    
    startIntIdx = np.argmax(times > t0 + epsilon)
    ref = np.sum(pls[ startIntIdx : startIntIdx + IntIdxRange ])

    startIntIdx = np.argmax(times > tpi + epsilon)
    sig = np.sum(pls[ startIntIdx : startIntIdx + IntIdxRange ])

    contrast = 1 - sig/ref
    tint = integrationTime
    
    sig = sig/float(IntIdxRange) # to convert summed countrates to average countrate during integration window
    ref = ref/float(IntIdxRange)
    return contrast, tint, sig, ref

def getContrastOf2pointODMRTrace(times, pls, t1, t3,
                                 integrationTime=300e-9,
                                 minLaserOnTime=1.5e-6,
                                 piPulseFirst=False):
    """
    Given a PL trace pls (Units [cts/s] or [cts]) at times, determine the contrast
    for a given integrationTime setting.
    The pls trace contains the PL signature of a signal and a reference readout.
    The integration for the contrast calculation starts at t1 and t3.
    
    For more context, see getContrast(makePlot=True), for generation of such a
    pls trace see twoPointODMRTrace().
    
    Parameters
    ----------
    times : numpy.array
        Must be evenly spaced. Units: s
    
    pls : numpy.ndarray
        PL values for each time in times, as returned by twoPointODMRTrace().
        Units: cts/s or cts
    
    t1, t3 : floats
        Start times of the readouts of the two states that give a readout contrast,
        as provided to twoPointODMRTrace(). Unit: s
        
    integrationTime : float or str, optional
        If a float [Unit: s] is provided, it is used as fix integration time.
        E.g. for integrationTime=250e-9 seconds, and 10ns time steps in times,
        a total of 25 points in pls is added together.
        
        If one of the strings 'optC'/ 'optSens'/ 'optSNR' is provided, the 
        integration time is chosen such to achieve maximal
        contrast/ sensitivity/ SNR.
        For sensitivity, a Gaussian pulsed ODMR is assumed.
        More see sensitivityGauss() and readoutSNR().
        
    piPulseFirst : bool, optional
        Specify whether the first pulse in the pls trace is the signal or the
        reference. Same argument as provided to twoPointODMRTrace().
    
    minLaserOnTime : float, optional
        The maximum integrationTime possible. Units: s
    

    Returns
    -------
    contrast : float
        Readout contrast. Units: 1
    
    tint : float
        integration time used by the routine. Units: s.
        The integration time is interpreted as follows: The first point of the
        sig or ref sum is the one which is larger than the start time (t1 or t3)
        - since laser only rises for t>t_start.
        
    sig, ref : floats
        Average signal and reference. If pls was in cts/s, then they are the
        average countrate during the tint integration windows. Else, the average
        counts per times bin.
    """
    if type(integrationTime) is float or type(integrationTime) is np.float_:
        return getContrastOf2pointODMRTrace_Fix(times, pls, t1, t3,
                                         integrationTime,
                                         minLaserOnTime,
                                         piPulseFirst)
    elif type(integrationTime) is str:
        return getContrastOf2pointODMRTrace_Opt(times, pls, t1, t3,
                                         integrationTime,
                                         minLaserOnTime,
                                         piPulseFirst)
    else:
        raise NotImplementedError(f'Got invalid argument integrationTime = {integrationTime}.')


def getContrast(integrationTime, minLaserOnTime=1.5e-6, tstepsize=1e-9,
                level1=0, level2=1, tauR=0, Delta_t=5e-9, N=4, 
                makePlot=False, state0=None,
                modelClass=MEmodel,
                **modeldict):
    """
    Get the readout contrast for a given integrationTime setting.
    
    If one just needs the contrast for given conditions, this function is more 
    handy and faster than using twoPointODMRTrace() and 
    getContrastOf2pointODMRTrace().

    Use makePlot=True to get a visual impression of whats happening inside this
    function.
    
    
    
    Parameters
    ----------        
    integrationTime : float or str, optional
        If a float [Unit: s] is provided, it is used as fix integration time.
        For more information see getContrastOf2pointODMRTrace().
        
        If one of the strings 'optC'/ 'optSens'/ 'optSNR' is provided, the 
        integration time is chosen such to achieve maximal
        contrast/ sensitivity/ SNR.
        For sensitivity, a Gaussian pulsed ODMR is assumed.
        More see sensitivityGauss() and readoutSNR().
        
    minLaserOnTime : float, optional
        The maximum integrationTime possible. Units: s

    tstepsize : float, optional
        Size of time steps used to evaluate the PL traces on. Unit: s.
        
    level1, level2 : int, optional
        Indices of the levels in EIG basis between which the pi-pulse is applied
        to obtain the contrast. Which levels these are depends on the modeldict
        setting. You can use simulateEigenVsParam() to see the spin nature of
        the EIG levels.
        The pi-pulse fidelity is given by modeldict['piPulseFid'].
        
    tauR : float, optional
        Units: s. For optional parameters of the laser rise time tauR, Delta_t,
        and N, see makeStepsForLaserRise().
        By default, a simple rectangular pulse is used.

    state0 : numpy.ndarray, optional
        Depending on the modelClass, this is a vector in EIG basis for classical 
        rate models or a density matrix in EZ basis for the MEmodel.
        If state0=None (default), an initState() is done with the same laserpower 
        as the readout specified in modeldict.
    
    modelClass : NVrateModel, optional
        Specify which rate model to use. Options:
        MEmode (default), HighTmodel, LowTmodel, SZmodel
        
    modeldict : dict, optional
        Optional keyword arguments can be provided by a modeldict or separately.
        For more details, see makeModelDict().
        NOTE: In modeldict the laser has to be on:
        'laserpower' > 0 (or default, which is on).
        
    
    Returns
    -------
    contrast : float
        Readout contrast. Units: 1
    
    tint : float
        integration time used by the routine. Units: s.
        For more information see getContrastOf2pointODMRTrace().
        
    ref : float
        Average countrate of the reference pulse during the tint integration 
        window. sig can be calculated from (1-contrast)*ref.
        For more information see getContrastOf2pointODMRTrace().
    """
    modeldict = makeModelDict(**modeldict) # generate from modeldict kwargs
    
    if type(integrationTime) is float or type(integrationTime) is np.float_:
        tend = integrationTime + 5*tstepsize # need to extend since t1s causes 2 laser off points in beginning and point in time at integrationTime is included.
    elif integrationTime in ('optC', 'optSens', 'optSNR'):
        tend = minLaserOnTime + 5*tstepsize # just assume that no integration window would ever be longer than minLaserOnTime.
    else:
        print('ERROR: got type(integrationTime)={} and integrationTime={}'.format(
            type(integrationTime), integrationTime))
        raise NotImplementedError

    shift = 1e-16 # more on this see twoPointODMRTrace()
    t1 = 0      # this is to make sure background and first NV light are in the same bin and the bin of t1 is still 0.
                # For more details see twoPointODMRTrace() where the same problem comes about.
    tsteps, ksteps = makeStepsForLaserRise(t1+shift, tend,
                                 tauR=tauR, Delta_t=Delta_t, N=N,
                                 modelClass=modelClass,
                                 **modeldict)
    kbeta = modelClass(**modeldict)
    if state0 is None:
        modeldict['laserpower']=0.
        k0 = modelClass(**modeldict)
        state0 = initState(k0, kbeta)
    else:
        state0 = state0

    piPulseFid=kbeta.modeldict['piPulseFid']
    
    PLs = []
    times = np.arange(0, tend, tstepsize)
    for piPulseFirst in [False, True]: # ms=0, ms=1 (assuming good init in state0)
        if not piPulseFirst:
            state = state0
        else:
            if modelClass.name=='MEmodel':
                # convert the state (EZ basis for MEmodel) to the EIG basis:
                thismodel = ksteps[0]
                _=thismodel.population(state0,'EIG') # to create thismodel.T_EIGtoEZ_withSS
                state0_EIG = basisTrafo(state0, thismodel.T_EIGtoEZ_withSS,
                                        T_old_to_new=thismodel.T_EZtoEIG_withSS)
                # apply a state swap:
                state_EIG = piPulse(state0_EIG, level1=level1, level2=level2, 
                            piPulseFid=piPulseFid)
                # convert the EIG state back to the EZ basis of MEmodel states:
                state = basisTrafo(state_EIG, thismodel.T_EZtoEIG_withSS,
                                        T_old_to_new=thismodel.T_EIGtoEZ_withSS)
            else: # classical models use EIG basis natively
                state = piPulse(state0, level1=level1, level2=level2, 
                            piPulseFid=piPulseFid)
        _, _, pls, _ = calcTimeTrace(times, tsteps, ksteps, state,
                                            basisName=None,
                                            )
        PLs.append(pls)

    # getContrastOf2pointODMRTrace expects a 2pointODMR trace. Generate one:
    # The concatenated traces are not physical, but the routine works on them.
    t3 = t1+times[-1]+tstepsize # +tstepsize since this is the value of the first time of the second part:
    times = np.concatenate((times, times+times[-1]+tstepsize))
    pls = np.concatenate(PLs)
    contrast, tint, sig, ref = getContrastOf2pointODMRTrace(times, pls, t1, t3,
                                                integrationTime=integrationTime,
                                                minLaserOnTime=tend-tstepsize,
                                                piPulseFirst=False)
    
    # the step in PL at t1 is caused by the background that turns on instantaneously.
    if makePlot:
        name = f'Found: tint={tint*1e9:.3f}ns, C={contrast*100:.2f}%\n\
tstepsize={tstepsize*1e9:.3f}ns, Delta_t={Delta_t*1e9:.3f}ns, tauR={tauR*1e9:.3f}ns'
        fig = plt.figure(figsize=(8,5))
        fig.suptitle(name, fontsize='small')
        fig.set_tight_layout(True)
        axes = fig.add_subplot(111)
        axes.plot(times*1e9,pls/1e3,marker='x',ls='-')
        axes.axvline(t1*1e9, color='black')
        axes.axvline(t3*1e9, color='black')
        axes.set_xlabel('time [ns]')
        axes.set_ylabel('PL [kcts/s]')
        axes.set_ylim(bottom=0)
        axes.grid(True)
        plt.show()
    return contrast, tint, ref


def getAmountMs0ForSequ(states, times, tsteps, ksteps):
    """
    Based on the equally named input and output parameters of
    calcTimeTrace(), get a numpy array of the amount of m_S=0
    present in the system (GS+ES), corresponding to the other outputs
    of calcTimeTrace().
    """
    amountMs0 = []
    for i in range(times.size):
        # ksteps[j] is active during tsteps[j-1] <= t < tsteps[j].
        # find index j that belongs to times[i]:
        t = times[i]
        if t > tsteps[-1]:
            j=-1
        else:
            j=np.argmax(tsteps >= t)
        k = ksteps[j]
        state = states[i]
        amountMs0.append(k.getGSPolarization_ms0(state) + k.getESPolarization_ms0(state))
    amountMs0 = np.array(amountMs0)
    return amountMs0


@jitOpt
def sensitivityGauss(contrast, tint, ref,
                     FWHM=1e6, sequDuration=1.5e-6+1e-6+1e-6):
    """
    Given the equally named results of a getContrast() or
    getContrastOf2pointODMRTrace() call, compute the sensitivity of a pulsed
    ODMR in units of T/sqrt(Hz). See Dreau2011Avoiding.
    
    For this, a full width at half maximum (FWHM) is assumed, and a duration of
    the entire sequence. The default is 1MHz=FWHM and the sequence is assumed to
    be 1.5us wait time, 1us pi-pulse, 1us laser pulse for readout and init.
    
    Roughly speaking: 1/sensitivity approx. sqrt(N)*C approx.
    In detail, see print(sensitivityEquation).
    """
    # See Dreau2011Avoiding: the maximum slope of Gaussian is at y=1-0.60653.
    dutyCyc = tint/sequDuration
    sens = np.sqrt(1-0.60653*contrast)/np.sqrt(ref)/contrast*0.700/28.03448e9*FWHM/np.sqrt(dutyCyc)
    return sens
sensitivityEquation = r'$\eta = \frac{2 \pi}{\gamma} \cdot \frac{0.7 \cdot FWHM \cdot \sqrt{1 - 0.6 \cdot C}}{C \cdot \sqrt{ref \cdot DC}}$'

@jitOpt
def sensitivityLor(contrast, tint, ref,
                   FWHM=1e6, sequDuration=1.5e-6+1e-6+1e-6):
    """
    Given the equally named results of a getContrast() or
    getContrastOf2pointODMRTrace() call, compute the sensitivity of a pulsed
    ODMR in units of T/sqrt(Hz). In contrast to sensitivityGauss(), a Lorentzian
    profile is assumed. See Dreau2011Avoiding.
    
    For this, a full width at half maximum (FWHM) is assumed, and a duration of
    the entire sequence. The default is 1MHz=FWHM and the sequence is assumed to
    be 1.5us wait time, 1us pi-pulse, 1us laser pulse for readout and init.
    
    Roughly speaking: 1/sensitivity approx. sqrt(N)*C approx.
    In detail, see print(sensitivityEquation_Lor).
    """
    # See Dreau2011Avoiding: the maximum slope of Lorentzian is at y=1-0.750.
    dutyCyc = tint/sequDuration
    sens = np.sqrt(1-0.750*contrast)/np.sqrt(ref)/contrast*0.770/28.03448e9*FWHM/np.sqrt(dutyCyc)
    return sens
sensitivityEquation_Lor = r'$\eta = \frac{2 \pi}{\gamma} \cdot \frac{0.8 \cdot FWHM \cdot \sqrt{1 - 0.8 \cdot C}}{C \cdot \sqrt{ref \cdot DC}}$'

@jitOpt
def readoutSNR(contrast, tint, ref):
    """
    Given the equally named results of a getContrast() or
    getContrastOf2pointODMRTrace() call, compute the signal to noise ratio (SNR)
    of a single readout. See Hopper2018Spin.
    
    In detail, see print(SNREquation).   
    The returned SNR has units 1.
    """
    SNR = np.sqrt(ref*tint)*contrast/np.sqrt(2-contrast)
    return SNR
SNREquation = r'$ SNR = \sqrt{PL \cdot t_{int}} \cdot \frac{C}{\sqrt{2-C}} $'


def getInitFidelity_ms0(modelClass=MEmodel, **modeldict):
    """
    Get the amount of m_S=0 (i.e. initialization fidelity) after a laser
    pulse as specified in modeldict, followed by a long waiting time
    (see initState()).
    
    Parameters
    ----------
    modelClass : NVrateModel, optional
        Specify which rate model to use. Options:
        MEmode (default), HighTmodel, LowTmodel, SZmodel
        
    modeldict : dict, optional
        Optional keyword arguments can be provided by a modeldict or separately.
        For more details, see makeModelDict().
        NOTE: In modeldict the laser has to be on:
        'laserpower' > 0 (or default, which is on).
        
    Returns
    -------
    initFidelity_ms0 : float
        Units: 1, range [0, 1].
    """
    modeldict = makeModelDict(**modeldict) # generate from modeldict kwargs
    kbeta = modelClass(**modeldict)
    modeldict['laserpower']=0.
    k0 = modelClass(**modeldict)
    state = initState(k0, kbeta)
    initFidelity_ms0 = kbeta.getGSPolarization_ms0(state)
    return initFidelity_ms0


def getReadoutFidelity_ms0(integrationTime='optSNR',
                tauR=0, Delta_t=5e-9, N=4, tstepsize=1e-9,
                level1=0, level2=1,
                modelClass=MEmodel,
                **modeldict):
    """
    Get the contrast, integration time, and SNR for a single readout when
    assuming a perfect 100% prior initialization into m_S=0.
    
    
    Parameters
    ---------- 
    For the optional parameters refer to getContrast(), which is the same as this
    function but without assuming a perfect initialization.
    
    Returns
    -------
    contrast : float
        Readout contrast. Units: 1
    
    integrationTime : float or str, optional
        If a float [Unit: s] is provided, it is used as fix integration time.
        For more information see getContrastOf2pointODMRTrace().
        
        If one of the strings 'optC'/ 'optSens'/ 'optSNR' is provided, the 
        integration time is chosen such to achieve maximal
        contrast/ sensitivity/ SNR.
        For sensitivity, a Gaussian pulsed ODMR is assumed.
        More see sensitivityGauss() and readoutSNR().
        
    SNR : float
        Signal to noise ratio (SNR) of the readout. For details see readoutSNR().
    """
    modeldict = makeModelDict(**modeldict) # generate from modeldict kwargs
    if modelClass == HighTmodel:
        P0 = modelClass(**modeldict).getPureState(1, idxInBasisName='avgEZ')
    else:
        P0 = modelClass(**modeldict).getPureState(1, idxInBasisName='EZ')
    
    contrast,tint,ref = getContrast(integrationTime,
                    tauR=tauR, Delta_t=Delta_t, N=N, tstepsize=tstepsize,
                    P0 = P0, modelClass=modelClass,
                    level1=level1, level2=level2,
                    **modeldict)
    SNR = readoutSNR(contrast, tint, ref)
    
    return contrast, tint, SNR
