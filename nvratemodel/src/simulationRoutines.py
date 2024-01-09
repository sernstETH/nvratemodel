from os.path import abspath, dirname
from os.path import join as osjoin
from scipy.optimize import minimize
from matplotlib.colors import LogNorm
import sys
sys.path.insert(0, abspath(osjoin(dirname(__file__), '..'))) # add root directory to import paths

from src.core import *


def simulateEigenVsParam(path=None, plot=True,
                         xParamName='B', x=np.linspace(0, 200e-3, 50),
                         plotES=True, plotGS=True, plotAvgES=True, plotOrbES=True,
                         **modeldict):
    """
    Compute the eigenvalues and eigenvectors (EV) of the Hamiltonian as a 
    function of xParamName.

    Parameters
    ----------
    path : str, optional
        Save result to this path if provided.
    plot : bool, optional
        Plot the result if desired.
    xParamName : str, optional
        Pick the x-axis parameter. Options are all keywords of modeldict,
        but only those that alter the Hamiltonian lead to an effect.
        For more information see makeModelDict().
    x : numpy.array, optional
        The values of xParamName to use as x-axis.
    plotES : bool, optional
        Include the excited state (ES) in the plot if desired.
    plotGS : bool, optional
        Include the ground state (GS) in the plot if desired.
    plotAvgES : bool, optional
        Include the averaged excited state (avgES) in the plot if desired.
    plotOrbES : bool, optional
        Include the orbital eigenenergies of the excited state in the plot if desired.
    modeldict : dict, optional
        Optional keyword arguments can be provided by a modeldict or separately.
        For more details, see makeModelDict().

    Returns
    -------
    dic : dict
        Contains the computation results and is also saved to file if path!=None.

    """
    modeldict = makeModelDict(**modeldict) # generate from modeldict kwargs
    thetaB, phiB, B = modeldict['thetaB'], modeldict['phiB'], modeldict['B']
    Eperp, phiE =  modeldict['Eperp'], modeldict['phiE']
    
    T = modeldict['T'] if 'T' in modeldict.keys() else 0.
    highT_trf = modeldict['highT_trf'] if 'highT_trf' in modeldict.keys() else False # used for avgES, then T from modeldict is used.

    xunit, xscaled = scaleParam((xParamName,x))
        
    ESenergies = []
    ESorbenergies = []
    ESeigVecsEZ = []
    ESeigVecsZF = []
    ESeigVecsHF = []
    ESeigVecsZB = []
    ESexpS_zOp = []
    ESexpOrbOp = []
    GSenergies = []
    GSresonances = []
    GSeigVecs = []
    GSexpS_zOp = []
    avgESenergies = []
    avgESeigVecs = []
    avgESexpS_zOp = []
    avgESresonances = []
    T_es_EZtoZF = T_EZtoZF[3:,3:]
    ESS_zOp = np.kron(Id2, S_z_opp)
    ESorbOp = np.kron(sigma_z, Id3)
    for i in range(x.size):
        if xParamName=='B':
            B = x[i]
        elif xParamName=='thetaB':
            thetaB = x[i]
        elif xParamName=='phiB':
            phiB = x[i]
        elif xParamName=='Eperp':
            Eperp = x[i]
        elif xParamName=='phiE':
            phiE = x[i]
        elif xParamName=='T':
            T = x[i]
        
        # ES:
        T_es_HFtoEZ = get_T_HFtoEZ_withSS(
            B, thetaB, phiB, Eperp, phiE
            )[3:-1,3:-1]
        T_es_EZtoHF = conjTransp(T_es_HFtoEZ)
        
        T_es_ZBtoEZ = get_T_ZBtoEZ_withSS(
            B, thetaB, phiB, Eperp, phiE
            )[3:-1,3:-1]
        T_es_EZtoZB = conjTransp(T_es_ZBtoEZ)
        
        ESorbenergies.append(get_HFev(B, thetaB, phiB, Eperp, phiE))
        
        eigvals, eigvecs = eigh(get_Hes_EZ(
            B, thetaB, phiB, Eperp, phiE
            ))
        ESenergies.append(eigvals)

        ESeigVecsEZ.append(eigvecs.transpose())
        
        ESeigVecsZF.append(np.array([
            np.dot(T_es_EZtoZF , np.array(vec)) for vec in ESeigVecsEZ[-1]
            ]).squeeze())
        ESeigVecsHF.append(np.array([
            np.dot(T_es_EZtoHF , np.array(vec)) for vec in ESeigVecsEZ[-1]
            ]).squeeze())
        ESeigVecsZB.append(np.array([
            np.dot(T_es_EZtoZB , np.array(vec)) for vec in ESeigVecsEZ[-1]
            ]).squeeze())
        
        ESexpS_zOp.append(np.array([
            expectationValue(ESS_zOp, vec).real for vec in ESeigVecsEZ[-1]
            ]))
        ESexpOrbOp.append(np.array([
            expectationValue(ESorbOp, vec).real for vec in ESeigVecsEZ[-1]
            ]))

        # GS:
        eigvals, eigvecs = eigh(get_Hgs_EZ(
            B, thetaB, phiB, Eperp, T
            ))
        GSenergies.append(eigvals)

        GSeigVecs.append(eigvecs.transpose())
        
        GSexpS_zOp.append(np.array([
            expectationValue(S_z_opp, vec).real for vec in GSeigVecs[-1]
            ]))
        
        energies = GSenergies[-1]
        spinvalues = GSexpS_zOp[-1]
        spinvalues_sorted, energies_sorted = sortarrays(
            spinvalues, energies, order='decr')
        resonances = np.abs(np.diff(energies_sorted))
        GSresonances.append(resonances) # f+, f- # alternative to core.getGSresonances()
        
        # avgES:
        if not highT_trf:
            avgHes_EZ = get_avgHes_EZ(
                B, thetaB, phiB, Eperp, phiE
                )
        else:
            avgHes_EZ = get_avgHesTRF_EZ(
                B, thetaB, phiB, Eperp, phiE, T
                )

        eigvals, eigvecs = eigh(avgHes_EZ)
        avgESenergies.append(eigvals)
        avgESeigVecs.append(eigvecs.transpose())
        
        avgESexpS_zOp.append(np.array([
            expectationValue(S_z_opp, vec).real for vec in avgESeigVecs[-1]
            ]))
        
        energies = avgESenergies[-1]
        spinvalues = avgESexpS_zOp[-1]
        spinvalues_sorted, energies_sorted = sortarrays(
            spinvalues, energies, order='decr')
        resonances = np.abs(np.diff(energies_sorted))
        avgESresonances.append(resonances) # f+, f-
        
        
    ESenergies = np.array(ESenergies)
    ESorbenergies = np.array(ESorbenergies)
    ESeigVecsEZ = np.array(ESeigVecsEZ)
    ESeigVecsZF = np.array(ESeigVecsZF)
    ESeigVecsHF = np.array(ESeigVecsHF)
    ESeigVecsZB = np.array(ESeigVecsZB)
    ESexpS_zOp = np.array(ESexpS_zOp)
    ESexpOrbOp = np.array(ESexpOrbOp)
    GSenergies = np.array(GSenergies)
    GSeigVecs = np.array(GSeigVecs)
    GSexpS_zOp = np.array(GSexpS_zOp)
    GSresonances = np.array(GSresonances)
    GSresonancesGrad = np.transpose([
        np.gradient(GSresonances[:,i], x)
        for i in range(GSresonances.shape[1])
        ]) # unit: Hz/T
    avgESenergies = np.array(avgESenergies)
    avgESeigVecs = np.array(avgESeigVecs)
    avgESexpS_zOp = np.array(avgESexpS_zOp)
    avgESresonances = np.array(avgESresonances)
    avgESresonancesGrad = np.transpose([
        np.gradient(avgESresonances[:,i], x)
        for i in range(avgESresonances.shape[1])
        ]) # unit: Hz/T

    if plot or path!=None:

        name = 'eigenenergies of Hamiltonian\n{}'.format(
            ''.join([
                getParamStr((key, value))+', '
                if key in ['thetaB', 'phiB', 'B', 'Eperp', 'phiE'] else ''
                for key, value in modeldict.items()
                ])[:-2]
            )
        figEnergies = plt.figure()
        figEnergies.suptitle(name)
        figEnergies.set_tight_layout(True)
        axes = figEnergies.add_subplot(111)
        if plotOrbES:
            for i in range(ESorbenergies.shape[1]):
                axes.plot(xscaled, ESorbenergies[:,i]/1e9,
                          label='{} orb. ES'.format([r'$E_x$', r'$E_y$'][i]), linestyle='-',
                          linewidth=4, color=['0.7', '0.85'][i])
        if plotES:
            for i in range(ESenergies.shape[1]):
                axes.plot(xscaled, ESenergies[:,i]/1e9,
                          label=f'{i+1}. ES', linestyle='-')
        if plotAvgES:
            for i in range(avgESenergies.shape[1]):
                axes.plot(xscaled, avgESenergies[:,i]/1e9,
                          label=f'{i+1}. avged ES', linestyle='dotted',
                          color='black')
        if plotGS:
            for i in range(GSenergies.shape[1]):
                axes.plot(xscaled, GSenergies[:,i]/1e9,
                          label=f'{i+1}. GS', linestyle='dashed')
        axes.set_xlabel(f'{xParamName} [{xunit}]')
        axes.set_ylabel(r'$E$ [GHz]')
        axes.set_xlim(xscaled.min())
        axes.grid(True)
        if plotOrbES or plotES or plotAvgES or plotGS:
            axes.legend(fontsize='small')
        
        name = 'resonances of Hamiltonian\n{}'.format(
            ''.join([
                getParamStr((key, value))+', '
                if key in ['thetaB', 'phiB', 'B', 'Eperp', 'phiE', 'T'] else ''
                for key, value in modeldict.items()
                ])[:-2]
            )
        figResonances = plt.figure(figsize=(8,5))
        figResonances.suptitle(name)
        figResonances.set_tight_layout(True)
        axes = figResonances.add_subplot(111)
        resNames = [r'$f_+$',r'$f_-$']
        if plotAvgES:
            for i in range(avgESresonances.shape[1]):
                axes.plot(xscaled, avgESresonances[:,i]/1e9,
                          label=f'{resNames[i]} of avged ES', linestyle='dotted')
        if plotGS:
            for i in range(GSresonances.shape[1]):
                axes.plot(xscaled, GSresonances[:,i]/1e9,
                          label=f'{resNames[i]} of GS', linestyle='dashed')
        axes.set_xlabel(f'{xParamName} [{xunit}]')
        axes.set_ylabel(r'$E$ [GHz]')
        axes.set_xlim(xscaled.min())
        axes.grid(True)
        if plotAvgES or plotGS:
            axes.legend(fontsize='small')
        
        name = 'Gradient of resonances of Hamiltonian\n{}'.format(
            ''.join([
                getParamStr((key, value))+', '
                if key in ['thetaB', 'phiB', 'B', 'Eperp', 'phiE', 'T'] else ''
                for key, value in modeldict.items()
                ])[:-2]
            )
        figResonancesGrad = plt.figure(figsize=(8,5))
        figResonancesGrad.suptitle(name)
        figResonancesGrad.set_tight_layout(True)
        axes = figResonancesGrad.add_subplot(111)
        resNames = [r'$f_+$',r'$f_-$']
        if plotAvgES:
            for i in range(avgESresonancesGrad.shape[1]):
                axes.plot(xscaled, avgESresonancesGrad[:,i]/1e9,
                          label=f'{resNames[i]} of avged ES', linestyle='dotted')
        if plotGS:
            for i in range(GSresonancesGrad.shape[1]):
                axes.plot(xscaled, GSresonancesGrad[:,i]/1e9,
                          label=f'{resNames[i]} of GS', linestyle='dashed')
        axes.set_xlabel(f'{xParamName} [{xunit}]')
        axes.set_ylabel(r'$\frac{d E}{d B}$ [GHz/T]')
        axes.set_xlim(xscaled.min())
        axes.grid(True)
        if plotAvgES or plotGS:
            axes.legend(fontsize='small')
        
        figExpS_z = plt.figure()
        figExpOrb = plt.figure()
        ops = ['S_z', 'Orb']
        for op in ops:
            ESexpOp = ESexpS_zOp if op=='S_z' else ESexpOrbOp
            OpName = r'$<\psi|\hat{S}_z|\psi>$' if op=='S_z' else r'$<\psi|\hat{\sigma}_z|\psi>$'
            figExp = figExpS_z if op=='S_z' else figExpOrb
            name = 'Expectation value {}\n{}'.format(
                OpName, ''.join([
                    getParamStr((key, value))+', '
                    if key in ['thetaB', 'phiB', 'B', 'Eperp', 'phiE'] else ''
                    for key, value in modeldict.items()
                    ])[:-2]
                )
            figExp.suptitle(name)
            figExp.set_tight_layout(True)
            axes = figExp.add_subplot(111)
            if plotES:
                for i in range(ESexpOp.shape[1]):
                    axes.plot(xscaled, ESexpOp[:,i],
                              label=f'{i+1}. ES', linestyle='-')
                    
            if op=='S_z':
                if plotAvgES:
                    for i in range(avgESexpS_zOp.shape[1]):
                        axes.plot(xscaled, avgESexpS_zOp[:,i],
                                  label=f'{i+1}. avged ES', linestyle='dotted')
                if plotGS:
                    for i in range(GSexpS_zOp.shape[1]):
                        axes.plot(xscaled, GSexpS_zOp[:,i],
                                  label=f'{i+1}. GS', linestyle='dashed')
                    
            axes.set_xlabel(f'{xParamName} [{xunit}]')
            axes.set_ylabel(OpName)
            axes.set_xlim(xscaled.min())
            axes.set_ylim((-1.1,1.1))
            axes.grid(True)
            if op=='S_z' and (plotES or plotAvgES or plotGS):
                axes.legend(fontsize='small')
            elif op=='Orb' and plotES:
                axes.legend(fontsize='small')
        
        NbrCols = 3
        NbrRows = 2
        figVecsEZ = plt.figure(figsize=(5*NbrCols, 4*NbrRows))
        figVecsZF = plt.figure(figsize=(5*NbrCols, 4*NbrRows))
        figVecsHF = plt.figure(figsize=(5*NbrCols, 4*NbrRows))
        figVecsZB = plt.figure(figsize=(5*NbrCols, 4*NbrRows))
        figVecsavgEZ = plt.figure(figsize=(5*NbrCols, 4*NbrRows))
        bases = ['EZ', 'ZF', 'HF', 'ZB', 'avgEZ']
        figVecss = [figVecsEZ, figVecsZF, figVecsHF, figVecsZB, figVecsavgEZ]
        ESeigVecss = [ESeigVecsEZ, ESeigVecsZF, ESeigVecsHF, ESeigVecsZB, avgESeigVecs]
        for j in range(len(bases)):
            figVecs = figVecss[j]
            basisName = bases[j]
            ESeigVecs = ESeigVecss[j]
            name = 'eigenvectors of {} in {}-basis, {}'.format(
                r'$H_{ES}$' if not basisName=='avgEZ' else r'$H$', basisName, ''.join([
                    getParamStr((key, value))+', '
                    if key in ['thetaB', 'phiB', 'B', 'Eperp', 'phiE'] else ''
                    for key, value in modeldict.items()
                    ])[:-2]
                )
            figVecs.suptitle(name)
            figVecs.set_tight_layout(True)
            for i in range(ESeigVecs.shape[1]):
                axes = figVecs.add_subplot(NbrRows, NbrCols, i+1)
                for idx in range(ESeigVecs.shape[2]):
                    axes.plot(xscaled, np.abs(ESeigVecs[:,i,idx])**2,
                              label=basisStateNames[bases[j]][idx+3],
                              linestyle=getmylinestyle(idx+3),
                              color=getmycolor(idx+3))
                axes.set_title('{}. ES {}'.format(i+1, r'$|\psi_{%i}>$'%(i+1)), fontsize='small')
                axes.set_xlabel(f'{xParamName} [{xunit}]')
                axes.set_ylabel(r'$\left|<basisstate|\psi_{%i}>\right|^2$'%(i+1))
                axes.set_xlim(xscaled.min())
                axes.set_ylim((-0.05,1.05))
                axes.legend(fontsize='small')
                axes.grid(True)
                
            if bases[j]=='avgEZ' and plotGS:
                for i2 in range(GSeigVecs.shape[1]):
                    axes = figVecs.add_subplot(NbrRows, NbrCols, i2+1+i+1)
                    for idx2 in range(GSeigVecs.shape[2]):
                        axes.plot(xscaled, np.abs(GSeigVecs[:,i2,idx2])**2,
                                  label=basisStateNames[bases[j]][idx2],
                                  linestyle=getmylinestyle(idx2),
                                  color=getmycolor(idx2))
                    axes.set_title('{}. GS {}'.format(i2+1, r'$|\psi_{%i}>$'%(i2+1)), fontsize='small')
                    axes.set_xlabel(f'{xParamName} [{xunit}]')
                    axes.set_ylabel(r'$\left|<basisstate|\psi_{%i}>\right|^2$'%(i2+1))
                    axes.set_xlim(xscaled.min())
                    axes.set_ylim((-0.05,1.05))
                    axes.legend(fontsize='small')
                    axes.grid(True)
    
    dic = {
        "xParamName":       xParamName,
        "x":                list(x),
        "ESenergies":       [list(row) for row in ESenergies],
        "avgedESenergies":  [list(row) for row in avgESenergies],
        "avgESresonances":  [list(row) for row in avgESresonances],
        "avgESresonancesGrad": [list(row) for row in avgESresonancesGrad],
        "orbESenergies":    [list(row) for row in ESorbenergies],
        "GSenergies":       [list(row) for row in GSenergies],
        "GSresonances":     [list(row) for row in GSresonances],
        "GSresonancesGrad": [list(row) for row in GSresonancesGrad],
        "ES_EZBasis_names": basisStateNames['EZ'][3:-1],
        "ESeigVecs_EZBasis": [[[str(val) for val in col] for col in row] for row in ESeigVecsEZ],
        "ES_ZFBasis_names": basisStateNames['ZF'][3:-1],        
        "ESeigVecs_ZFBasis": [[[str(val) for val in col] for col in row] for row in ESeigVecsZF],
        "ES_HFBasis_names": basisStateNames['HF'][3:-1],
        "ESeigVecs_HFBasis": [[[str(val) for val in col] for col in row] for row in ESeigVecsHF],
        "ES_ZBBasis_names": basisStateNames['ZB'][3:-1],
        "ESeigVecs_ZBBasis": [[[str(val) for val in col] for col in row] for row in ESeigVecsZB],
        "avgES_EZBasis_names": basisStateNames['avgEZ'][3:-1],
        "avgESeigVecs_EZBasis": [[[str(val) for val in col] for col in row] for row in avgESeigVecs],
        "GS_EZBasis_names": basisStateNames['EZ'][0:3],
        "GSeigVecs_EZBasis": [[[str(val) for val in col] for col in row] for row in GSeigVecs],
        "ESeigVecs_ExpectVal_S_z": [list(row) for row in ESexpS_zOp],
        "avgESeigVecs_ExpectVal_S_z": [list(row) for row in avgESexpS_zOp],
        "ESeigVecs_ExpectVal_Orb": [list(row) for row in ESexpOrbOp],
        "GSeigVecs_ExpectVal_S_z": [list(row) for row in GSexpS_zOp],
        "TRF":              highT_trf, # also in params['highT_trf']
        "params":           modeldict,
        }
    
    if path!=None:
        timestr = strftime("%Y-%m-%d_%H-%M-%S")
        explanation = '_EigenVsParam'
        path = osjoin(path, timestr+explanation)
        # save plots:
        name='Energies'
        pathandname = osjoin(path, name)
        figEnergies.savefig(ensure_dir('{}.png'.format(pathandname)),
                          format='png', dpi=100,
                          )
        name='Resonances'
        pathandname = osjoin(path, name)
        figResonances.savefig(ensure_dir('{}.png'.format(pathandname)),
                          format='png', dpi=100,
                          )
        name='Resonances_Gradient'
        pathandname = osjoin(path, name)
        figResonancesGrad.savefig(ensure_dir('{}.png'.format(pathandname)),
                          format='png', dpi=100,
                          )
        name='ESeigVecs_ExpectVal_S_z'
        pathandname = osjoin(path, name)
        figExpS_z.savefig(ensure_dir('{}.png'.format(pathandname)),
                          format='png', dpi=100,
                          )
        name='ESeigVecs_ExpectVal_Orb'
        pathandname = osjoin(path, name)
        figExpOrb.savefig(ensure_dir('{}.png'.format(pathandname)),
                          format='png', dpi=100,                          
                          )
        name='ESeigVecs_EZBasis'
        pathandname = osjoin(path, name)
        figVecsEZ.savefig(ensure_dir('{}.png'.format(pathandname)),
                          format='png', dpi=100,                          
                          )
        name='ESeigVecs_ZFBasis'
        pathandname = osjoin(path, name)
        figVecsZF.savefig(ensure_dir('{}.png'.format(pathandname)),
                          format='png', dpi=100,                          
                          )
        name='ESeigVecs_HFBasis'
        pathandname = osjoin(path, name)
        figVecsHF.savefig(ensure_dir('{}.png'.format(pathandname)),
                          format='png', dpi=100,                          
                          )
        name='ESeigVecs_ZBBasis'
        pathandname = osjoin(path, name)
        figVecsZB.savefig(ensure_dir('{}.png'.format(pathandname)),
                          format='png', dpi=100,                           
                          )
        name='avgESeigVecs_EZBasis'
        pathandname = osjoin(path, name)
        figVecsavgEZ.savefig(ensure_dir('{}.png'.format(pathandname)),
                          format='png', dpi=100,                           
                          )
        # save the data:
        name='EigenVsParam.json'
        pathandname = osjoin(path, name)
        with open(pathandname, 'w') as f:
            dump(dic, f)
        print(f'Saved to {path}')
    
    if plot:
        if plotES==False:
            for fig in [figExpOrb, figVecsEZ, figVecsZF, figVecsHF, figVecsZB]:
                plt.close(fig)
        
        if plotAvgES==False and plotGS==False:
            for fig in [figVecsavgEZ, figResonances]:
                plt.close(fig)
        plt.show()
    elif path!=None: # close all figures here if plot=False and saved, since too many.
        for fig in [figEnergies, figResonances, figExpS_z, figExpOrb, figVecsEZ, 
                    figVecsZF, figVecsHF, figVecsZB, figVecsavgEZ]:
            plt.close(fig)
    return dic



def simulatekmixvsT(path=None, plot=True, Tmax = 300, # unit: K
                num=300,
                EperpList=[3e9, 30e9, 200e9],
                T0=0., # T0=4.4K in Goldman2015Phonon SI.
                **modeldict,
                ):
    """
    Compute the orbital hopping/mixing rates (kmix) as a 
    function of temperature (T).

    Parameters
    ----------
    path : str, optional
        Save result to this path if provided.
    plot : bool, optional
        Plot the result if desired.
    T0 : float, optional
        Start value of temperature axis in units K.
    Tmax : float, optional
        End value of temperature axis in units K.
    EperpList : list, optional
        Evaluate for these values of Eperp. modeldict['Eperp'] is ignored.
    modeldict : dict, optional
        Optional keyword arguments can be provided by a modeldict or separately.
        For more details, see makeModelDict().

    Returns
    -------
    dic : dict
        Contains the computation results and is also saved to file if path!=None.

    """
    modeldict = makeModelDict(**modeldict) # generate from modeldict kwargs
    thetaB, phiB = modeldict['thetaB'], modeldict['phiB']
    phiE, B =  modeldict['phiE'], modeldict['B']
    elPhonCoup  = modeldict['elPhonCoup']
    modeldict['Eperp'] = EperpList[-1]
    modeldict['T'] = Tmax
    phononCutoffEnergy = modeldict['phonCutoff']
    
    temperaturePlot = np.logspace(0, np.log10(Tmax), num=num) # units: K
    
    kmix, kmix1, kmix2, upOVERdownRate = [], [], [], []
    for Eperp in EperpList:
        Delta = get_orbitalSplitting(B, thetaB, phiB, Eperp, phiE)
        kmix2.append(kmix2Full(temperaturePlot, Delta, elPhonCoup=elPhonCoup,
                                   T0=T0, phononCutoffEnergy=phononCutoffEnergy)) # based on kmix2Full_scalar.
        kmix1.append(kmix1Full(temperaturePlot, Delta, elPhonCoup=elPhonCoup)) # based on kmix1Full_scalar.
        kmix.append(getOrbitalRates(temperaturePlot, Delta,
                                         elPhonCoup=elPhonCoup, T0_2ph=T0, T0_1ph=T0,
                                         phononCutoffEnergy=phononCutoffEnergy,
                                         )[0]
                    )
        upOVERdownRate.append(DetailedBallanceRatio(temperaturePlot, Delta))
    kmix, kmix1 = np.array(kmix), np.array(kmix1)
    kmix2, upOVERdownRate = np.array(kmix2), np.array(upOVERdownRate)
    
    unit, scaling = scaleParam(('Eperp',1), remove1=True)
    unit_phonCutoff, scaling_phonCutoff = scaleParam(('phonCutoff',1), remove1=True)
    unit_elPhonCoup, scaling_elPhonCoup = scaleParam(('elPhonCoup',1), remove1=True)
   
    if plot or path!=None:
        name = 'orbital hopping/mixing rates down vs T'+'\n'
        name += 'elPhonCoup={:.1f}{}, phonCutoff={}{}, T0={:.1f}K'.format(
            elPhonCoup*scaling_elPhonCoup, unit_elPhonCoup,
            phononCutoffEnergy*scaling_phonCutoff, unit_phonCutoff, 
            T0
            )
        fig1 = plt.figure(figsize=(9,7))
        fig1.suptitle(name)
        fig1.set_tight_layout(True)
        axes = fig1.add_subplot(111)
        for j, Eperp in enumerate(EperpList):
            axes.plot(temperaturePlot, kmix[j]/1e6,
                  label='{} at {}={:.1f}{}'.format(r'$k_{\downarrow}$',
                                        r'$|\xi_\perp|$', Eperp*scaling, unit),
                  linestyle='-',
                  color=getmycolor(j*4),
                  )
            axes.plot(temperaturePlot, kmix1[j]/1e6,
                  label='{} at {}={:.1f}{}'.format(r'$k_{1-ph,\downarrow}$',
                                         r'$|\xi_\perp|$', Eperp*scaling, unit),
                  linestyle='dashed',
                  color=getmycolor(j*4),
                  )
            axes.plot(temperaturePlot, kmix2[j]/1e6,
                  label='{} at {}={:.1f}{}'.format(r'$k_{2-ph,\downarrow}$',
                                         r'$|\xi_\perp|$', Eperp*scaling, unit),
                  linestyle='dotted',
                  color=getmycolor(j*4),
                  )
        axes.legend(fontsize='medium')
        axes.set_yscale('log')
        axes.set_xscale('log')
        axes.set_xlabel('T [K]')
        axes.set_ylabel(r'orbital hopping rate [1/$\mu$s]')
        axes.grid(True)
    
        name = r'orbital hopping/mixing rate $E_y \rightarrow E_x$ over $E_x \leftarrow E_y$'+'\n'
        name += 'elPhonCoup={:.1f}{}, phonCutoff={}{}, T0={:.1f}K'.format(
            elPhonCoup*scaling_elPhonCoup, unit_elPhonCoup,
            phononCutoffEnergy*scaling_phonCutoff, unit_phonCutoff, 
            T0
            )
        fig2 = plt.figure(figsize=(8,6))
        fig2.suptitle(name)
        fig2.set_tight_layout(True)
        axes = fig2.add_subplot(111)
        for j, Eperp in enumerate(EperpList):
            axes.plot(temperaturePlot,
                      upOVERdownRate[j],
                      label='{}={:.1f}{}'.format(r'$|\xi_\perp|$', 
                                                 Eperp*scaling, unit),
                      linestyle='-',
                      color=getmycolor(j*4),
                  )
        axes.legend(fontsize='medium')
        axes.set_xscale('log')
        axes.set_xlabel('T [K]')
        axes.set_ylabel(r'$k_{\uparrow}/k_{\downarrow}$ [1]')
        axes.grid(True)
        
        name = 'orbital hopping/mixing rates up vs T'+'\n'
        name += 'elPhonCoup={:.1f}{}, phonCutoff={}{}, T0={:.1f}K'.format(
            elPhonCoup*scaling_elPhonCoup, unit_elPhonCoup,
            phononCutoffEnergy*scaling_phonCutoff, unit_phonCutoff, 
            T0
            )
        fig3 = plt.figure(figsize=(9,7))
        fig3.suptitle(name)
        fig3.set_tight_layout(True)
        axes = fig3.add_subplot(111)
        for j, Eperp in enumerate(EperpList):
            axes.plot(temperaturePlot, kmix[j]/1e6*upOVERdownRate[j],
                  label='{} at {}={:.1f}{}'.format(r'$k_{\uparrow}$',
                                        r'$|\xi_\perp|$', Eperp*scaling, unit),
                  linestyle='-',
                  color=getmycolor(j*4),
                  )
            axes.plot(temperaturePlot, kmix1[j]/1e6*upOVERdownRate[j],
                  label='{} at {}={:.1f}{}'.format(r'$k_{1-ph,\uparrow}$', 
                                        r'$|\xi_\perp|$', Eperp*scaling, unit),
                  linestyle='dashed',
                  color=getmycolor(j*4),
                  )
            axes.plot(temperaturePlot, kmix2[j]/1e6*upOVERdownRate[j],
                  label='{} at {}={:.1f}{}'.format(r'$k_{2-ph,\uparrow}$', 
                                        r'$|\xi_\perp|$', Eperp*scaling, unit),
                  linestyle='dotted',
                  color=getmycolor(j*4),
                  )
        axes.legend(fontsize='medium')
        axes.set_yscale('log')
        axes.set_xscale('log')
        axes.set_xlabel('T [K]')
        axes.set_ylabel(r'orbital hopping rate [1/$\mu$s]')
        axes.grid(True)
        
    dic = {
        "Ts":        list(temperaturePlot),
        "Eperps":    list(EperpList),
        "kmixDOWN":  [list(row) for row in kmix],  
        "kmix1":     [list(row) for row in kmix1],
        "kmix2":     [list(row) for row in kmix2],
        "upOVERdownRate": [list(row) for row in upOVERdownRate],
        "elPhonCoup":elPhonCoup, # also stored in modeldict['elPhonCoup']
        "phononCutoffEnergy": phononCutoffEnergy, # also stored in modeldict['phonCutoff']
        "T0":        T0,
        "params":    modeldict,
        }
    
    if path!=None:
        timestr = strftime("%Y-%m-%d_%H-%M-%S")
        explanation = '_kmixVsT'
        path = osjoin(path, timestr+explanation)
        
        # save plots:
        name='kmixVsT_down'
        pathandname = osjoin(path, name)
        fig1.savefig(ensure_dir('{}.png'.format(pathandname)),
                          format='png', dpi=100,
                          )
        name='kmixUpOverDown'
        pathandname = osjoin(path, name)
        fig2.savefig(ensure_dir('{}.png'.format(pathandname)),
                          format='png', dpi=100,                          
                          )
        name='kmixVsT_up'
        pathandname = osjoin(path, name)
        fig3.savefig(ensure_dir('{}.png'.format(pathandname)),
                          format='png', dpi=100,                          
                          )
        # save the data:
        name='kmixData.json'
        pathandname = osjoin(path, name)
        with open(pathandname, 'w') as f:
            dump(dic, f)
        print(f'Saved to {path}')
        
    if plot:
        plt.show()
    elif path!=None:
        for fig in [fig1, fig2, fig3]:
            plt.close(fig)
    return dic



def simulateIvsT(path=None, plot=True, Tmax=300, # unit: K
       num=300,
       cutoffEnergyList = [
       PlakhotnikCutoffEnergy, AbtewCutoffEnergy, diamondDebyeEnergy,
       ],
       **modeldict,
       ):
    """
    Compute the Debye Integral (I) as a 
    function of temperature (T) for different phonon cutoff energies.

    Parameters
    ----------
    path : str, optional
        Save result to this path if provided.
    plot : bool, optional
        Plot the result if desired.
    Tmax : float, optional
        End value of temperature axis in units K. It starts at 0K.
    cutoffEnergyList : list, optional
        Evaluate for these values of phonCutoff. modeldict['phonCutoff'] is ignored.
    modeldict : dict, optional
        Optional keyword arguments can be provided by a modeldict or separately.
        For more details, see makeModelDict().

    Returns
    -------
    dic : dict
        Contains the computation results and is also saved to file if path!=None.

    """
    modeldict = makeModelDict(**modeldict) # generate from modeldict kwargs
    temperaturePlot = np.logspace(0, np.log10(Tmax), num=num) # units: K
    Delta = get_orbitalSplitting(modeldict["B"], modeldict["thetaB"],
                                 modeldict["phiB"], modeldict["Eperp"],
                                 modeldict["phiE"])
    Eperp = modeldict["Eperp"]
    thresholdList = [phononCutoffEnergyi/kB*0.1 
                     for phononCutoffEnergyi in cutoffEnergyList]

    IList = []
    for i,phononCutoffEnergyi in enumerate(cutoffEnergyList):
        IList.append(DebyeIntegralFull(temperaturePlot, Delta, 
                                       phononCutoffEnergy=phononCutoffEnergyi))
    IList = np.array(IList)
    
    unit, scaling = scaleParam(('Eperp',1), remove1=True)
    unit_phonCutoff, scaling_phonCutoff = scaleParam(('phonCutoff',1), remove1=True)
    
    if plot or path!=None:
        name = 'Debye Integral (no approx.) at Eperp={:.3f}{}'.format(
            Eperp*scaling, unit
            )
        fig = plt.figure(figsize=(7,5))
        fig.suptitle(name)
        fig.set_tight_layout(True)
        axes = fig.add_subplot(111)
        for i,phononCutoffEnergyi in enumerate(cutoffEnergyList):
            axes.plot(temperaturePlot,
                      IList[i],
                      label='phonCutoff={:.1f}{}'.format(
                          phononCutoffEnergyi*scaling_phonCutoff, 
                          unit_phonCutoff
                          ),
                      linestyle=getmylinestyle(i),
                      color=getmycolor(i,4),
                      linewidth=2,
                  )
            axes.axvline(thresholdList[i],
                         label=r'phonCutoff$/k_B \cdot 0.1$',
                         color=getmycolor(i,4),
                         )
        axes.legend(fontsize='large')
        axes.set_xscale('log')
        axes.set_xlabel(r'$T$ [K]')
        axes.set_ylabel(r'Debye Integral $I$ [1]')
        axes.grid(True)
    
    dic = {
        "Ts":                   list(temperaturePlot),
        "phononCutoffEnergies": list(cutoffEnergyList),
        "Is":                   [list(row) for row in IList],
        "thresholds10percent":  list(thresholdList),
        "params":    modeldict, # NOTE: modeldict['phonCutoff'] is not used, see phononCutoffEnergies instead.
        }
    
    if path!=None:
        timestr = strftime("%Y-%m-%d_%H-%M-%S")
        explanation = '_DebyeIntegral'
        path = osjoin(path, timestr+explanation)
        # save plots:
        name='DebyeIntegral'
        pathandname = osjoin(path, name)
        fig.savefig(ensure_dir('{}.png'.format(pathandname)),
                          format='png', dpi=100,                          
                          )
        # save the data:
        name='DebyeIntegralData.json'
        pathandname = osjoin(path, name)
        with open(pathandname, 'w') as f:
            dump(dic, f)
        print(f'Saved to {path}')

    if plot:
        plt.show()
        
    return dic



def simulateReadoutVsParam(yAxis='C', path=None, plot=True,
             xParamName ='T', x=np.linspace(0, 300, 50),
             argsList = [(MEmodel, makeModelDict()),],
             integrationTime='optSNR', tauR=0, Delta_t=5e-9, N=4,
             FWHM=1e6, sequDuration=1.5e-6+1e-6+1e-6,
             level1=0, level2=1,
             ):
    """
    For a list of models/settings, compute a readout quantity on the y-axis as
    a function of a chosen parameter on the x-axis.

    Parameters
    ----------
    yAxis : str, optional
        Options:
            - 'C' for contrast. See getContrast().
            - 'SNR' for signal to noise ratio of a single readout. See readoutSNR().
            - 'PL' for the steady-state photoluminescence.
            - 'initFid' for the initialization fidelity. See getInitFidelity_ms0().
            - 'readFid' for the readout fidelity as SNR. See getReadoutFidelity_ms0().
            - 'Sens' ('normSens') for the (normalized) sensitivity of a pulsed \
            ODMR, assuming a FWHM (units Hz) and sequDuration (units s). \
            See sensitivityGauss().
    path : str, optional
        Save result to this path if provided.
    plot : bool, optional
        Plot the result if desired.
    xParamName : str, optional
        Pick the x-axis parameter. Options are all keywords of modeldict.
        For more information see makeModelDict().
    x : numpy.array, optional
        The values of xParamName to use as x-axis.
    argsList : list of tuples of (NVrateModel, modeldict), optional
        For each list element, the rate model NVrateModel with parameters as 
        specified in the dict modeldict (as returned by makeModelDict(), for 
        possible keywords see there) is computed over the x-axis.
    integrationTime : float or str, optional
        If a float [Unit: s] is provided, it is used as fix integration time.
        For more information see getContrastOf2pointODMRTrace().
        
        If one of the strings 'optC'/ 'optSens'/ 'optSNR' is provided, the 
        integration time is chosen such to achieve maximal
        contrast/ sensitivity/ SNR.
        For sensitivity, a Gaussian pulsed ODMR is assumed.
        More see sensitivityGauss() and readoutSNR().
    tauR : float, optional
        Units: s. For optional parameters of the laser rise time tauR, Delta_t,
        and N, see makeStepsForLaserRise().
        By default, a simple rectangular pulse is used.
    level1, level2 : int, optional
        Indices of the levels in EIG basis between which the pi-pulse is applied
        to obtain the contrast. Which levels these are depends on the modeldict
        setting. You can use simulateEigenVsParam() to see the spin nature of
        the EIG levels. For more information, see piPulse().
        The pi-pulse fidelity is given by modeldict['piPulseFid'].

    Returns
    -------
    dic : dict
        Contains the computation results and is also saved to file if path!=None.

    """
    if yAxis=='PL':
        y = np.zeros((len(argsList), x.size))
        y_alt = None
        for i in range(len(argsList)):
            modelClass = argsList[i][0]
            modeldict = deepcopy(argsList[i][1])
            for j in range(x.size):
                modeldict[xParamName] = x[j]
                y[i][j] = modelClass(**modeldict).PL()
        name = 'Steady state emission rate'
        ylabel = r'$PL$ [kcts/s]'
        yscale = lambda x: x/1e3
    elif yAxis=='initFid':
        y = np.zeros((len(argsList), x.size))
        y_alt = None
        for i in range(len(argsList)):
            modelClass = argsList[i][0]
            modeldict = deepcopy(argsList[i][1])
            for j in range(x.size):
                modeldict[xParamName] = x[j]
                y[i][j] = getInitFidelity_ms0(modelClass=modelClass,**modeldict)
        name = r'GS initialization fidelity $m_s=0$'
        ylabel = r'amount $m_s=0$ [1]'
        yscale = lambda x: x
    elif yAxis=='readFid':
        Cs = np.zeros((len(argsList), x.size))
        SNRs = np.zeros((len(argsList), x.size))
        tints = np.zeros((len(argsList), x.size))
        for i in range(len(argsList)):
            modelClass = argsList[i][0]
            modeldict = deepcopy(argsList[i][1])
            for j in range(x.size):
                modeldict[xParamName] = x[j]
                c,thistint,snr = getReadoutFidelity_ms0(integrationTime=integrationTime,
                    tauR=tauR, Delta_t=Delta_t, N=N,
                    modelClass=modelClass, level1=level1, level2=level2,
                    **modeldict)
                Cs[i][j] = c
                SNRs[i][j] = snr
                tints[i][j] = thistint
        y = SNRs
        y_alt = Cs
        name = f'readout SNR, t_int={integrationTime}, tau_R={tauR*1e9:.1f}ns, levels={level1}-{level2}'
        name += '\n'+SNREquation
        ylabel = r'SNR [1]'
        yscale = lambda x: x
    elif yAxis in ('normSens', 'Sens', 'C', 'SNR'): 
        Cs = np.zeros((len(argsList), x.size))
        Senss = np.zeros((len(argsList), x.size))
        SNRs = np.zeros((len(argsList), x.size))
        tints = np.zeros((len(argsList), x.size))
        for i in range(len(argsList)):
            modelClass = argsList[i][0]
            modeldict = deepcopy(argsList[i][1])
            for j in range(x.size):
                modeldict[xParamName] = x[j]
                c,thistint,ref = getContrast(
                    integrationTime, tauR=tauR, Delta_t=Delta_t, N=N,
                    modelClass=modelClass, level1=level1, level2=level2,
                    **modeldict)
                sens = sensitivityGauss(c,thistint,ref,FWHM=FWHM,
                                        sequDuration=sequDuration)
                snr = readoutSNR(c,thistint,ref)
                Cs[i][j] = c
                Senss[i][j] = sens
                SNRs[i][j] = snr
                tints[i][j] = thistint
        if yAxis=='Sens':
            y = Senss
            y_alt = Cs
            name = f'Sensitivity, t_int={integrationTime}, tau_R={tauR*1e9:.1f}ns, levels={level1}-{level2}'
            name += '\n'+sensitivityEquation+' at '
            name += rf'$FWHM$ = {FWHM/1e6:.1f}MHz, $t_s$={sequDuration*1e6:.1f}$\mu$s'
            ylabel = r'$\eta$ [$\frac{\mu T}{\sqrt{Hz}}$]'
            yscale = lambda x: x*1e6
        elif yAxis=='normSens':
            y = Senss
            y_alt = Cs
            name = f'Normalized sensitivity, t_int={integrationTime}, tau_R={tauR*1e9:.1f}ns, levels={level1}-{level2}'
            name += '\n'+sensitivityEquation
            ylabel = r'$\eta$ [norm.]'
            yscale = lambda x: x/np.min(y)
        elif yAxis=='C':
            y = Cs
            y_alt = Senss
            name = f'Contrast, t_int={integrationTime}, tau_R={tauR*1e9:.1f}ns, levels={level1}-{level2}'
            ylabel = r'$C$ [%]'
            yscale = lambda x: x*100
        elif yAxis=='SNR':
            y = SNRs
            y_alt = Cs
            name = f'SNR, t_int={integrationTime}, tau_R={tauR*1e9:.1f}ns, levels={level1}-{level2}'
            name += '\n'+SNREquation
            ylabel = r'SNR [1]'
            yscale = lambda x: x
    else:
        raise NotImplementedError
    
    tintsExist = True if yAxis in (
        'normSens', 'Sens', 'C', 'readFid', 'SNR'
        ) else False
    
    if path!=None or plot==True:
        xunit, xscaled = scaleParam((xParamName,x))

        fig = plt.figure(figsize=(8,8))
        fig.set_tight_layout(True)
        fig.suptitle(name)
        if tintsExist:
            gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])
            axes1 = plt.subplot(gs[0])
            axes2 = plt.subplot(gs[1])
        else:
            axes1 = fig.add_subplot(111)
        modeldictDefault = argsList[0][1]
        for i in range(len(argsList)):
            if i==0:
                string = ''.join([
                    getParamStr(item)+'\n'
                    for item in modeldictDefault.items()
                    ])[:-1]
            else:
                string = ''.join([
                    getParamStr((key, value))+'\n' 
                    if value!=modeldictDefault[key] else ''
                    for key, value in argsList[i][1].items()
                    ])[:-1]       
            axes1.plot(xscaled, yscale(y[i]),
                    linestyle=getmylinestyle(i), marker='.',
                    color=getmycolor(i, len(argsList)),
                    label='{} \n{}'.format(
                        argsList[i][0].name, string if string!=''
                        else 'see above')
                    )
            if tintsExist:
                axes2.plot(xscaled, tints[i]*1e9,
                        linestyle=getmylinestyle(i), marker='.',
                        color=getmycolor(i, len(argsList)))    
        axes1.set_ylabel(ylabel)
        if not tintsExist:
            axes1.set_xlabel(f'{xParamName} [{xunit}]')
        axes1.set_xlim(left=xscaled.min())
        axes1.legend(fontsize='x-small')
        axes1.grid(True)
        if tintsExist:
            axes2.set_ylabel(r'$t_{int}$ [ns]')
            axes2.set_xlabel(f'{xParamName} [{xunit}]')
            axes2.set_xlim(left=xscaled.min())
            axes2.grid(True)

    Nargs = len(argsList)
    dic = {
        "xParamName":       xParamName,
        "yParamName":       yAxis,
        "model_List":       [argsList[i][0].name for i in range(Nargs)],
        "params_List":      [argsList[i][1] for i in range(Nargs)],
        "x":                list(x), # same for all i
        "y_List":           [list(row) for row in y],
        "y_alt_List":       [list(row) for row in y_alt
                             ] if y_alt is not None else None,
        "tints_List":       [list(row) for row in tints
                             ]if tintsExist else None,
        "tauR":             tauR, # not relevant for 'PL'  and 'initFid'
        "Delta_t":          Delta_t, # not relevant for 'PL'  and 'initFid'
        "N":                N, # not relevant for 'PL'  and 'initFid'
        "tintMethod":       integrationTime, # not relevant for 'PL'  and 'initFid'
        "FWHM":             FWHM, # only relevant for 'Sens'
        "sequDuration":     sequDuration, # only relevant for 'Sens'
        "piPulseLevels":    [level1, level2], # not relevant for 'PL'  and 'initFid'
        }

    if path!=None:            
        timestr = strftime("%Y-%m-%d_%H-%M-%S")
        explanation = '_ReadoutvsParam'
        path = osjoin(path, timestr+explanation)
        # save plot:
        name='ReadoutvsParam'
        pathandname = osjoin(path, name)
        fig.savefig(ensure_dir('{}.png'.format(pathandname)),
                    format='png', dpi=100,                     
                    )
        # save the data:
        name='ReadoutvsParam.json'
        pathandname = osjoin(path, name)
        with open(pathandname, 'w') as f:
            dump(dic, f)
        print(f'Saved to {path}')

    if plot:
        plt.show()
        
    return dic



def simulate2DMap(zAxis='PL', path=None, plot=True,
             xParamName='B', x=np.linspace(0e-3, 200e-3, 50), 
             yParamName='T', y=np.linspace(0, 105, 50),
             integrationTime=250e-9, tauR=0, Delta_t=5e-9, N=4,
             FWHM=1e6, sequDuration=1.5e-6+1e-6+1e-6,
             level1=0, level2=1,
             modelClass = MEmodel,
             **modeldict):
    """
    For a given NV model and setting, compute a map of a readout quantity on 
    the z-axis as a function of a chosen parameter on the x-axis and y-axis.

    Parameters
    ----------
    zAxis : str, optional
        Options:
            - 'C' for contrast. See getContrast().
            - 'SNR' for signal to noise ratio of a single readout. See readoutSNR().
            - 'PL' for the steady-state photoluminescence.
            - 'initFid' for the initialization fidelity. See getInitFidelity_ms0().
            - 'readFid' for the readout fidelity as SNR. See getReadoutFidelity_ms0().
            - 'Sens' ('normSens') for the (normalized) sensitivity of a pulsed \
            ODMR, assuming a FWHM (units Hz) and sequDuration (units s). \
            See sensitivityGauss().
    path : str, optional
        Save result to this path if provided.
    plot : bool, optional
        Plot the result if desired.
    xParamName : str, optional
        Pick the x-axis parameter. Options are all keywords of modeldict.
        For more information see makeModelDict().
    x : numpy.array, optional
        The values of xParamName to use as x-axis.
    yParamName : str, optional
        Pick the y-axis parameter. Options are all keywords of modeldict.
        For more information see makeModelDict().
    y : numpy.array, optional
        The values of xParamName to use as y-axis.
    integrationTime : float or str, optional
        If a float [Unit: s] is provided, it is used as fix integration time.
        For more information see getContrastOf2pointODMRTrace().
        
        If one of the strings 'optC'/ 'optSens'/ 'optSNR' is provided, the 
        integration time is chosen such to achieve maximal
        contrast/ sensitivity/ SNR.
        For sensitivity, a Gaussian pulsed ODMR is assumed.
        More see sensitivityGauss() and readoutSNR().
    tauR : float, optional
        Units: s. For optional parameters of the laser rise time tauR, Delta_t,
        and N, see makeStepsForLaserRise().
        By default, a simple rectangular pulse is used.
    level1, level2 : int, optional
        Indices of the levels in EIG basis between which the pi-pulse is applied
        to obtain the contrast. Which levels these are depends on the modeldict
        setting. You can use simulateEigenVsParam() to see the spin nature of
        the EIG levels. For more information, see piPulse().
        The pi-pulse fidelity is given by modeldict['piPulseFid'].
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
    dic : dict
        Contains the computation results and is also saved to file if path!=None.

    """
    modeldict = makeModelDict(**modeldict) # generate from modeldict kwargs
    
    if zAxis=='PL':
        z = np.zeros((y.size, x.size))
        for xi,xval in enumerate(x):
            modeldict[xParamName] = xval
            for yi,yval in enumerate(y):
                modeldict[yParamName] = yval
                z[yi][xi] = modelClass(**modeldict).PL()
        z_alt = None
        name = 'Steady state emission rate'+f', {modelClass.name}'
        zlabel = r'$PL$ [kcts/s]'
        zscale = lambda x: x/1e3
    elif zAxis=='initFid':
        z = np.zeros((y.size, x.size))
        for xi,xval in enumerate(x):
            modeldict[xParamName] = xval
            for yi,yval in enumerate(y):
                modeldict[yParamName] = yval
                z[yi][xi] = getInitFidelity_ms0(modelClass=modelClass,**modeldict)
        z_alt = None
        name = r'GS initialization fidelity $m_s=0$'+f', {modelClass.name}'
        zlabel = r'amount $m_s=0$ [1]'
        zscale = lambda x: x
    elif zAxis=='readFid':
        Cs = np.zeros((y.size, x.size))
        SNRs = np.zeros((y.size, x.size))
        tints = np.zeros((y.size, x.size))
        for xi,xval in enumerate(x):
            modeldict[xParamName] = xval
            for yi,yval in enumerate(y):
                modeldict[yParamName] = yval
                c,thistint,snr = getReadoutFidelity_ms0(integrationTime=integrationTime,
                    tauR=tauR, Delta_t=Delta_t, N=N,
                    modelClass=modelClass, level1=level1, level2=level2,
                    **modeldict)
                Cs[yi][xi] = c
                SNRs[yi][xi] = snr
                tints[yi][xi] = thistint
        z = SNRs
        z_alt = Cs
        name = f'readout SNR, t_int={integrationTime}, tau_R={tauR*1e9:.1f}ns, levels={level1}-{level2}'+f', {modelClass.name}'
        name += '\n'+SNREquation
        zlabel = r'SNR [1]'
        zscale = lambda x: x
    elif zAxis in ('normSens', 'Sens', 'C', 'SNR'):
        Cs = np.zeros((y.size, x.size))
        Senss = np.zeros((y.size, x.size))
        SNRs = np.zeros((y.size, x.size))
        tints = np.zeros((y.size, x.size))
        for xi,xval in enumerate(x):
            modeldict[xParamName] = xval
            for yi,yval in enumerate(y):
                modeldict[yParamName] = yval
                c,thistint,ref = getContrast(
                    integrationTime, tauR=tauR, Delta_t=Delta_t, N=N,
                    modelClass=modelClass, level1=level1, level2=level2,
                    **modeldict)
                sens = sensitivityGauss(c,thistint,ref,FWHM=FWHM,
                                        sequDuration=sequDuration)
                snr = readoutSNR(c,thistint,ref)
                Senss[yi][xi] = sens
                Cs[yi][xi] = c
                SNRs[yi][xi] = snr
                tints[yi][xi] = thistint
        if zAxis=='Sens':
            z = Senss
            z_alt = Cs
            name = f'Sensitivity, t_int={integrationTime}, tau_R={tauR*1e9:.1f}ns, levels={level1}-{level2}'+f', {modelClass.name}'
            name += '\n'+sensitivityEquation+' at '
            name += rf'$FWHM$ = {FWHM/1e6:.1f}MHz, $t_s$={sequDuration*1e6:.1f}$\mu$s'
            zlabel = r'$\eta$ [$\frac{\mu T}{\sqrt{Hz}}$]'
            zscale = lambda x: x*1e6
        elif zAxis=='normSens':
            z = Senss
            z_alt = Cs
            name = f'Normalized sensitivity, t_int={integrationTime}, tau_R={tauR*1e9:.1f}ns, levels={level1}-{level2}'+f', {modelClass.name}'
            name += '\n'+sensitivityEquation
            zlabel = r'$\eta$ [norm.]'
            zscale = lambda x: x/np.min(z)
        elif zAxis=='C':
            z = Cs
            z_alt = Senss
            name = f'Contrast, t_int={integrationTime}, tau_R={tauR*1e9:.1f}ns, levels={level1}-{level2}'+f', {modelClass.name}'
            zlabel = r'$C$ [%]'
            zscale = lambda x: x*100
        elif zAxis=='SNR':
            z = SNRs
            z_alt = Cs
            name = f'SNR, t_int={integrationTime}, tau_R={tauR*1e9:.1f}ns, levels={level1}-{level2}'+f', {modelClass.name}'
            name += '\n'+SNREquation
            zlabel = r'SNR [1]'
            zscale = lambda x: x
    else:
        raise NotImplementedError
    
    tintsExist = True if zAxis in ('normSens', 'Sens', 'C', 'readFid', 'SNR'
                                   ) else False
    
    if path!=None or plot==True:
        xunit, xscaled = scaleParam((xParamName,x))
        yunit, yscaled = scaleParam((yParamName,y))
        
        color = {
            'PL': 'gray',
            'normSens': 'viridis',
            'Sens': 'viridis',
            'C': 'pink',
            'initFid': 'inferno',
            'readFid': 'Reds',
            'SNR': 'Reds',
            }
        
        fig = plt.figure(figsize=(8,9))
        fig.set_tight_layout(True)
        axes = fig.add_subplot(111)
        string = ''.join([
            getParamStr(item)+'\n'
            for item in modeldict.items()
            ])[:-1]
        im = axes.pcolormesh(xscaled, yscaled, zscale(z),
                             cmap=color[zAxis], shading='nearest')
        plt.colorbar(im, label=zlabel)
        axes.set_title(name+'\n'+string, fontsize='x-small')
        axes.set_xlabel(f'{xParamName} [{xunit}]')
        axes.set_ylabel(f'{yParamName} [{yunit}]')
        
        if tintsExist:            
            figTint = plt.figure(figsize=(8,9))
            figTint.set_tight_layout(True)
            axes = figTint.add_subplot(111)
            string = ''.join([
                getParamStr(item)+'\n'
                for item in modeldict.items()
                ])[:-1]
            im = axes.pcolormesh(xscaled, yscaled, tints*1e9,
                                 cmap='Blues_r', shading='nearest')
            plt.colorbar(im, label=r'$t_{int}$ [ns]')
            axes.set_title('Integration time for '+name+'\n'+string,
                           fontsize='x-small')
            axes.set_xlabel(f'{xParamName} [{xunit}]')
            axes.set_ylabel(f'{yParamName} [{yunit}]')
            
    dic = {
        "xParamName":       xParamName,
        "yParamName":       yParamName,
        "zParamName":       zAxis,
        "x":                list(x),
        "y":                list(y),
        "z":                [list(row) for row in z],
        "z_alt":            [list(row) for row in z_alt
                             ] if z_alt is not None else None,
        "tints":            [list(row) for row in tints
                             ] if tintsExist else None,
        "model":            modelClass.name,
        "params":           modeldict,
        "tauR":             tauR, # not relevant for 'PL'  and 'initFid'
        "Delta_t":          Delta_t, # not relevant for 'PL'  and 'initFid'
        "N":                N, # not relevant for 'PL'  and 'initFid'
        "tintMethod":       integrationTime, # not relevant for 'PL'  and 'initFid'
        "FWHM":             FWHM, # only relevant for 'Sens'
        "sequDuration":     sequDuration, # only relevant for 'Sens'
        "piPulseLevels":    [level1, level2], # not relevant for 'PL'  and 'initFid'
        }
    
    if path!=None:            
        timestr = strftime("%Y-%m-%d_%H-%M-%S")
        explanation = '_2Dmap'
        path = osjoin(path, timestr+explanation)
        # save plots:
        name='2Dmap'
        pathandname = osjoin(path, name)
        fig.savefig(ensure_dir('{}.png'.format(pathandname)),
                    format='png', dpi=100,                     
                    )
        if  tintsExist:
            name='2Dmap_tint'
            pathandname = osjoin(path, name)
            figTint.savefig(ensure_dir('{}.png'.format(pathandname)),
                        format='png', dpi=100,                        
                        )
        # save the data:
        name='2Dmap.json'
        pathandname = osjoin(path, name)
        with open(pathandname, 'w') as f:
            dump(dic, f)
        print(f'Saved to {path}')

    if plot:
        plt.show()
        
    return dic


def simulatePopTimeTrace(times, tsteps, ksteps, state0,
                     path=None,  plot=True, basisName='EZ',
                     jumpTimes=[], jumpOps=[],
                     explainStr='',
                     plotTrace=False, plotMs0=False,
                     plotGS=True, plotSS=True, plotES=True, plotPL=True,
                     specificEvalDictList=[],
                     ):
    """
    Propagate the initial state state0 at times[0] over all times in times
    under the sequence of conditions specified by tsteps, ksteps.
    
    For more details, see calcTimeTrace(), which this function wraps and extends
    in functionality.

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
    path : str, optional
        Save result to this path if provided.
    plot : bool, optional
        Plot the result if desired.
    basisName : str, optional
        Possible basisNames are ksteps[i].basisNameOptions. For more information,
        see BasisStateNames().
    jumpTimes : list of floats, optional
        Times at which to apply the respective jump operators jumpOps. Unit: s.
    jumpOps : list of numpy.array, optional
        Matrices that represent jump operators for the MEmodel. No unit.
        Available options are:
        LindbladOp_DecayOfEyToEx_HF, LindbladOp_DecayOfExToEy_HF:
            Makes a jump between the ES orbital state (like T1 relaxation).
            Only gives correct results if there are no orbital coherences ever!
        LindbladOp_DecayOfExToEx_HF, LindbladOp_DecayOfEyToEy_HF:
            Makes a complet T2 decay of the ES orbital state, i.e. kills coherences.
        LindbladOp_GS_msp1_xpiPulse_EZ, LindbladOp_GS_msp1_ypiPulse_EZ:
            Makes a perfect pi_x/y pulse on the GS spin.
    explainStr : str, optional
        String added to the figure plot title for clarity of the content.
    plotTrace : bool, optional
        Additionally plot the trace of the state (to check that it stays approx. 1).
    plotMs0 : bool, optional
        Additionally plot the amount of m_S=0 present in the system (GS+ES). See
        getAmountMs0ForSequ().
    plotGS : bool, optional
        Additionally plot the populations of the ground state (GS) states.
    plotSS : bool, optional
        Additionally plot the population of the shelving state (SS) states.
    plotES : bool, optional
        Additionally plot the populations of the excited state (ES) states.
    plotPL : bool, optional
        Additionally plot the current photoluminescence.
    specificEvalDictList : list of dicts, optional
        Add user-defined evaluations done at each times to the calculation.
        List elements have to be as follows: {"name": str, "func": python callable(*)}.
        (*) The function must take one argument state and return a float. The
        state is a state of the current ksteps NVrateModel.

    Returns
    -------
    dic : dict
        Contains the computation results and is also saved to file if path!=None.

    """
    def applyJumpOp(state, jumpOp):
        shape = state.shape
        D = getDissipator(jumpOp)
        state_vec = state.flatten('F')
        res_vec = state_vec + np.dot(D, state_vec) # D*state is the dstate/dt change of state. Thus the new state is old state + change.
        state_new = np.reshape(res_vec, shape, order='F')
        return state_new

    if jumpTimes != []:
        if False in [kstep.name=="MEmodel" for kstep in ksteps]:
            raise NotImplementedError("Cannot use jumpTimes/jumpOps in models other than MEmodel.")
        
        # adapted from calcTimeTrace algorithm.        
        jumpTimes, jumpOps = sortarrays(jumpTimes, jumpOps, order='incr')
        
        # remove all jumps that are after the last times element:
        if jumpTimes[-1] > times[-1]:
            idx = np.argmax(jumpTimes>times[-1])
            for i in range(idx, len(jumpTimes)):
                print(f'JumpOp at {jumpTimes[i]*1e9:.0f}ns is beyond times range.')
            jumpTimes = jumpTimes[:idx]
            jumpOps = jumpOps[:idx]
        
        # remove all jumps before the first times element:
        idx = np.argmax(jumpTimes>times[0])
        for i in range(idx):
            print(f'JumpOp at {jumpTimes[i]*1e9:.0f}ns is before times range.')
        jumpTimes = jumpTimes[idx:]
        jumpOps = jumpOps[idx:]
        
        def propagateState(end, start, state):
            """This correctly handles cases where tsteps occur between start
            and end."""
            thistimes = np.array([start, end])
            _, states, pls, pops = calcTimeTrace(
                thistimes, tsteps, ksteps, state, basisName=basisName)
            return states[-1], pls[-1], pops[-1]

        list_of_state = []
        pops_X = []
        PLs = []
        state = state0
        tstart = times[0]
        tend = times[-1]+1e-9 # simply a value that is larger than the last times
        for i in range(len(jumpTimes)+1):
            ti = jumpTimes[i] if i<len(jumpTimes) else tend
            thistimes = times[(tstart <= times) & (times < ti)]
            if thistimes.size > 1:
                state, _, _ = propagateState(thistimes[0], tstart, state)
                _, states, pls, pops = calcTimeTrace(
                    thistimes, tsteps, ksteps, state, basisName=basisName)
                pops_X.append(pops)
                list_of_state.append(states)
                PLs.append(pls)
                state = states[-1]
            elif thistimes.size == 1:
                state, PL, Pop = propagateState(thistimes[0], tstart, state)
                PLs.append(np.array([PL]))
                pops_X.append(np.expand_dims(Pop,0))
                list_of_state.append(np.expand_dims(state,0))
            elif thistimes.size == 0:
                state, _, _ = propagateState(ti, tstart, state)
                state = applyJumpOp(state, jumpOps[i]) if i<len(jumpTimes) else state
                tstart = ti
                continue
            state, _, _ = propagateState(ti, thistimes[-1], state)
            state = applyJumpOp(state, jumpOps[i]) if i<len(jumpTimes) else state
            tstart = ti
        PLs = np.concatenate(PLs)
        list_of_state = np.concatenate(list_of_state)
        pops_X = np.concatenate(pops_X)

    else:
        _, list_of_state, PLs, pops_X = calcTimeTrace(
            times, tsteps, ksteps, state0, basisName=basisName)
    
    
    specificEvalsList = []
    for evDict in specificEvalDictList:
        specificEvalsList.append(np.array([
            evDict["func"](state) for state in list_of_state]
            ))
        
    amountMs0 = getAmountMs0ForSequ(list_of_state, times, tsteps, ksteps)
    
    if plot or path!=None:
        firstmodeldict = ksteps[0].modeldict
        B = firstmodeldict['B']
        Eperp = firstmodeldict['Eperp']
        thetaB = firstmodeldict['thetaB']
        phiB = firstmodeldict['phiB']
        phiE = firstmodeldict['phiE']
        T = firstmodeldict['T']
        name = 'Time evolution in {}-basis - 1st step: {}={:.0f}K, {}={:.2f}mT, \
{}={:.1f}GHz, {}={:.2f}°, {}={:.0f}°, {}={:.0f}°\n{}'.format(
            basisName,
            r'$T$', T,
            r'$B$', B/1e-3,
            r'$\xi_\perp$', Eperp/1e9,
            r'$\theta_B$', np.degrees(thetaB),
            r'$\phi_B$', np.degrees(phiB),
            r'$\phi_\xi$',  np.degrees(phiE),
            explainStr,        
            )
        fig = plt.figure(figsize=(11,8))
        fig.set_tight_layout(True)
        fig.suptitle(name, fontsize='medium')
        if plotPL:
            gs = gridspec.GridSpec(2, 1, height_ratios=[4,1])
            axes2 = plt.subplot(gs[1])
        else:
            gs = gridspec.GridSpec(1, 1, height_ratios=[1])
            axes2 = None
        axes1 = plt.subplot(gs[0])
        for jumpTime in jumpTimes:
            axes1.axvline(jumpTime*1e9, color='aqua', linewidth=1)
            if plotPL:
                axes2.axvline(jumpTime*1e9, color='aqua', linewidth=1)
        if plotTrace:
            axes1.plot(times*1e9, np.sum(pops_X[:,:], axis=1),
                        linestyle='-',
                        color = '0.3',
                        marker='',
                        label=r'Tr($\rho$)',
                        )
        if plotMs0:
            axes1.plot(times*1e9, amountMs0,
                        linestyle='-',
                        color = 'skyblue',
                        marker='',
                        linewidth=3,
                        label=r'$<m_s=0>$',
                        )
        for idx in range(pops_X.shape[1]):
            if (not plotGS and idx<3) or (not plotSS and idx==9) or (
                    not plotES and idx in ksteps[0].emittingLevelIdxs):
                continue
            axes1.plot(times*1e9, pops_X[:,idx],
                        linestyle=getmylinestyle(idx),
                        color = getmycolor(idx),
                        marker='',
                        label='{}'.format(basisStateNames[basisName][idx]),
                        )
        for i,specificEval in enumerate(specificEvalsList):
            axes1.plot(times*1e9, specificEval,
                        linestyle=getmylinestyle(i+3),
                        color = 'black',
                        marker='',
                        label=specificEvalDictList[i]["name"],
                        )   
        axes1.set_ylabel('population [1]')
        if not plotPL:
            axes1.set_xlabel('time [ns]')
        axes1.set_xlim(left=times[0]*1e9, right=times[-1]*1e9)
        axes1.set_ylim(bottom=-0.01)
        axes1.legend(fontsize='large', loc='lower right')
        axes1.grid(True)
        if plotPL:
            axes2.plot(times*1e9, PLs/1e3, label='PL')
            axes2.set_ylabel('PL [kcts/s]')
            axes2.set_xlabel('time [ns]')
            axes2.set_xlim(left=times[0]*1e9, right=times[-1]*1e9)
            axes2.set_ylim(bottom=0/1e3, top=max(PLs.max()/1e3*1.1, 1))
            axes2.grid(True)        
    else:
        axes1, axes2 = None, None
    
    dic = {
        "basisName":        basisName,
        "basisStateNames":  basisStateNames[basisName],
        "tsteps":           list(tsteps),
        "ksteps":           [kstep.modeldict for kstep in ksteps],
        "jumpTimes":        list(jumpTimes),
        "times":            list(times),
        "PLs":              list(PLs),
        "Pops_inBasis":     [list(row) for row in pops_X],
        "amountMs0":        list(amountMs0),
        }
    for i,evDict in enumerate(specificEvalDictList):
        dic['specificEval_'+evDict["name"]] = list(specificEvalsList[i]) 

    if path!=None:
        timestr = strftime("%Y-%m-%d_%H-%M-%S")
        explanation = '_PopTimeTrace'
        path = osjoin(path, timestr+explanation)
        # save plots:
        name='PopTimeTrace'
        pathandname = osjoin(path, name)
        fig.savefig(ensure_dir('{}.png'.format(pathandname)),
                    format='png', dpi=100,                     
                    )
        # save the data:
        name='PopTimeTrace.json'
        pathandname = osjoin(path, name)
        with open(pathandname, 'w') as f:
            dump(dic, f)
        print(f'Saved to {path}')

    if plot:
        plt.show()
        
    return axes1, axes2, dic, path



def simulatePulseVsParam(path=None, plot=True, stackedPlot=True,
             argsList = [(MEmodel, makeModelDict()),],
             laserOnTime = 2.0e-6, dt=2e-9, integrationTime='optSNR',
             tauR=0, Delta_t=5e-9, N=4,
             level1=0, level2=1,
             ):
    """
    For a list of models/settings, compute a time-resolved pulsed readout
    sequence. Such sequences are the basis for the y/z-axis of simulateReadoutVsParam()
    and simulate2DMap().

    Parameters
    ----------
    path : str, optional
        Save result to this path if provided.
    plot : bool, optional
        Plot the result if desired.
    stackedPlot : bool, optional
        Plot the signal and readout parts of the sequence on top of each other.
    argsList : list of tuples of (NVrateModel, modeldict), optional
        For each list element, the rate model NVrateModel with parameters as 
        specified in the dict modeldict (as returned by makeModelDict(), for 
        possible keywords see there) is computed over the x-axis.
    laserOnTime : float, optional
        Duration of the laser pulse. Unit: s
    dt : float, optional
        Size of the time steps over which the sequence is evaluated. Unit: s
    integrationTime : float or str, optional
        If a float [Unit: s] is provided, it is used as fix integration time.
        For more information see getContrastOf2pointODMRTrace().
        
        If one of the strings 'optC'/ 'optSens'/ 'optSNR' is provided, the 
        integration time is chosen such to achieve maximal
        contrast/ sensitivity/ SNR.
        For sensitivity, a Gaussian pulsed ODMR is assumed.
        More see sensitivityGauss() and readoutSNR().
    tauR : float, optional
        Units: s. For optional parameters of the laser rise time tauR, Delta_t,
        and N, see makeStepsForLaserRise().
        By default, a simple rectangular pulse is used.
    level1, level2 : int, optional
        Indices of the levels in EIG basis between which the pi-pulse is applied
        to obtain the contrast. Which levels these are depends on the modeldict
        setting. You can use simulateEigenVsParam() to see the spin nature of
        the EIG levels. For more information, see piPulse().
        The pi-pulse fidelity is given by modeldict['piPulseFid'].

    Returns
    -------
    dic : dict
        Contains the computation results and is also saved to file if path!=None

    """
    t1, t2, t3, t4 = makeTstepsFromDurations(
        [0.1e-6, laserOnTime, 1e-6, laserOnTime])
    times=np.arange(0, t4+t1, dt)
    
    PLPlotList = np.zeros((len(argsList), times.size))
    CPlotList = np.zeros((len(argsList)))
    tintPlotList = np.zeros((len(argsList)))
    for i in range(len(argsList)):
        modeldict = argsList[i][1]
        modelClass = argsList[i][0]
        PLPlotList[i][:] = twoPointODMRTrace(times, t1, t2, t3, t4,
                                piPulseFirst=False,
                                level1=level1, level2=level2,
                                tauR=tauR, Delta_t=Delta_t, N=N,
                                modelClass=modelClass,
                                **modeldict)

        # this is a waste of time since the same twoPointODMRTrace is calculated 
        # twice but it makes the code easier:
        # should use getContrastOf2pointODMRTrace on PLPlotList[i][:] otherwise...                
        CPlotList[i],tintPlotList[i],_ = getContrast(integrationTime, minLaserOnTime=t2-t1,
                                   level1=level1, level2=level2,
                                   tauR=tauR, Delta_t=Delta_t, N=N,
                                   modelClass=modelClass,
                                   **modeldict)
            
    if plot or path!=None:       
        name = f'Pulsed laser Readout, tau_R={tauR*1e9:.1f}ns, levels={level1}-{level2}. Contrast for t_int={integrationTime}.'
        fig = plt.figure(figsize=(10,8))
        fig.set_tight_layout(True)
        axes = fig.add_subplot(111)        
        t1sIdx = getIndx(t1, times)
        t2sIdx = getIndx(t3, times)
        modeldictDefault = argsList[0][1]
        for i in range(len(argsList)):
            if i==0:
                string = ''.join([
                    getParamStr(item)+'\n'
                    for item in modeldictDefault.items()
                    ])
            else:
                string = ''.join([
                    getParamStr((key, value))+'\n' if value!=modeldictDefault[key] else ''
                    for key, value in argsList[i][1].items()
                    ])
            string += 'contrast @ t_int={:.0f}ns: {:.1f}%'.format(
                tintPlotList[i]*1e9, CPlotList[i]*100)
            if not stackedPlot:
                axes.plot(times*1e9, PLPlotList[i]/1e3,
                          linestyle=getmylinestyle(i),
                          color=getmycolor(i, len(argsList)),
                          label='{} model\n{}'.format(
                              argsList[i][0].name, string if string!='' else 'default'),
                          )
            else:
                axes.plot((times[0:t2sIdx-t1sIdx] - t1)*1e9,
                          PLPlotList[i][0:t2sIdx-t1sIdx]/1e3,
                          linestyle=getmylinestyle(i),
                          color=getmycolor(i, len(argsList)),
                          label='{} model\n{}'.format(
                              argsList[i][0].name, string if string!='' else 'default'),
                          )
                axes.plot((times[t2sIdx-t1sIdx:] - t3)*1e9,
                          PLPlotList[i][t2sIdx-t1sIdx:]/1e3,
                          linestyle=getmylinestyle(i),
                          color=getmycolor(i, len(argsList)),
                          label=r'$\pi$-pulse applied',
                          )
        axes.legend(fontsize='x-small')
        axes.set_title(name)
        axes.set_xlabel('t [ns]')
        axes.set_ylabel('PL [kcts/s]')
        axes.grid(True)
    
    Nargs = len(argsList)
    dic = {
        "model_List":       [argsList[i][0].name for i in range(Nargs)],
        "params_List":      [argsList[i][1] for i in range(Nargs)],
        "times":            list(times), # same for all i
        "PLs_List":         [list(PLPlotList[i]) for i in range(Nargs)],
        "C_List":           [CPlotList[i] for i in range(Nargs)],
        "tint_List":        [tintPlotList[i] for i in range(Nargs)],
        "tintMethod":       integrationTime,
        "timesteps":        [t1, t2, t3, t4],
        "tauR":             tauR,
        "Delta_t":          Delta_t,
        "N":                N,
        "piPulseLevels":    [level1, level2],
        }
    
    if path!=None:
        timestr = strftime("%Y-%m-%d_%H-%M-%S")
        name='PulseVsParam'
        explanation = '_'+name
        path = osjoin(path, timestr+explanation)
        # save plots:
        pathandname = osjoin(path, name)
        fig.savefig(ensure_dir('{}.png'.format(pathandname)),
                    format='png', dpi=100,                     
                    )
        # save the data:
        pathandname = osjoin(path, name+'.json')
        with open(pathandname, 'w') as f:
            dump(dic, f)
        print(f'Saved to {path}')  

    if plot:
        plt.show()

    return dic



def fitMagnetAlignment(GSresonances, path=None, plot=True,
                       B_min=0, B_max=10e-3,
                       **modeldict):
    """
    Fit observed spin resonances GSresonances = (f+, f-) to obtain the alignment
    of the bias magnetic field.
    Make sure you set the correct temperature in modeldict (or via T=X).
    The values for B and thetaB in modeldict (or via B= and theta=) are used as
    initial guess for the fitting.
    Note that a E-field/strain splitting of the GS is not included in the model.

    Parameters
    ----------
    GSresonances : tuple
        (f+, f-) spin resonances as they where observed. Unit: Hz
    path : str, optional
        Save result to this path if provided.
    plot : bool, optional
        Plot the result if desired.
    B_min : float
        Minimal magnetic field strength. Units: T
    B_max : float
        Maximal magnetic field strength. Units: T
    modeldict : dict, optional
        Optional keyword arguments can be provided by a modeldict or separately.
        For more details, see makeModelDict().

    Returns
    -------
    B_fit : float
        Magnetic field strength obtained in the fit. Units: T
    thetaB_fit : float
        Misalignment angle obtained in the fit. Units: rad
    """
    modeldict = makeModelDict(**modeldict) # generate from modeldict kwargs
    thetaB, B, T = modeldict['thetaB'], modeldict['B'], modeldict['T']
    resonances_user = GSresonances
    
    # full range to investigate:
    thetaB_min, thetaB_max = np.radians((0,90)) 
    
    # fit for minimum:
    def fitfunc(x):
        B, thetaB = x
        modeldict['B'] = B
        modeldict['thetaB'] = thetaB
        GSresonances = get_GSresonances(**modeldict) # f+, f-
        return np.linalg.norm((GSresonances-resonances_user)) # unit: error in Hz
    
    res = minimize(fitfunc, (B, thetaB), bounds=((B_min, B_max),(thetaB_min, thetaB_max)))
    print(res)
    B_fit, thetaB_fit = res.x
    modeldict_res = deepcopy(modeldict)
    modeldict_res['B'] = B_fit
    modeldict_res['thetaB'] = thetaB_fit
    resonances = get_GSresonances(**modeldict_res) # f+, f-
    print('fit result error [MHz]: ', (resonances-resonances_user)/1e6)
    
    
    if path!=None or plot==True:
        # compute:
        num = 100
        xParamName = 'B'
        yParamName = 'thetaB'
        x=np.linspace(B_min, B_max, num)
        y=np.linspace(thetaB_min, thetaB_max, num)
        
        z = np.zeros((y.size, x.size))
        for xi,xval in enumerate(x):
            for yi,yval in enumerate(y):
                z[yi][xi] = fitfunc((xval, yval))
        
        # plot:
        xunit, xscaled = scaleParam((xParamName,x))
        yunit, yscaled = scaleParam((yParamName,y))
        B_fitscaled = scaleParam((xParamName,B_fit))[1]
        thetaBscaled = scaleParam((yParamName,thetaB_fit))[1]
        zlabel = r'$\sqrt{(f_{+,err}^2+f_{-,err}^2}$ [$MHz$]'
        zscale = lambda x: x/1e6 # for MHz

        name = f'Deviation of observed resonances at ({zscale(resonances_user[0]):.2f}, {zscale(resonances_user[1]):.2f})MHz'
        name += ' from Simulation'
        name += '\n'+rf'Fit result: $B$ = {B_fitscaled:.2f}{xunit}, $thetaB$ = {thetaBscaled:.2f}{yunit}'
        name += '\n'+rf'$T$ = {T:.0f}K'
    
        
        fig = plt.figure(figsize=(8,7))
        fig.set_tight_layout(True)
        axes = fig.add_subplot(111)
        im = axes.pcolormesh(xscaled, yscaled, zscale(z),
                             shading='nearest', norm=LogNorm())
        plt.colorbar(im, label=zlabel)
        axes.set_title(name)
        axes.set_xlabel(f'{xParamName} [{xunit}]')
        axes.set_ylabel(f'{yParamName} [{yunit}]')

        axes.plot(B_fitscaled, thetaBscaled,
                  color='red', ls='', marker='*', ms=15)
        
        if path!=None:
            dic = {
                "xParamName":       xParamName,
                "yParamName":       yParamName,
                "zParamName":       zlabel,
                "x":                list(x),
                "y":                list(y),
                "z":                [list(row) for row in z],
                "params":           modeldict_res, # contains fit result
                "resonances_user":  list(resonances_user), # f+, f-
                "resonances_fit":   list(resonances), # f+, f-
                }

            timestr = strftime("%Y-%m-%d_%H-%M-%S")
            explanation = '_fitMagnetAlignment'
            path = osjoin(path, timestr+explanation)
            # save plots:
            name='fitMagnetAlignment'
            pathandname = osjoin(path, name)
            fig.savefig(ensure_dir('{}.png'.format(pathandname)),
                        format='png', dpi=100,                     
                        )
            # save the data:
            name='fitMagnetAlignment.json'
            pathandname = osjoin(path, name)
            with open(pathandname, 'w') as f:
                dump(dic, f)
            print(f'Saved to {path}')
        
        if plot:
            plt.show()
    
    return B_fit, thetaB_fit