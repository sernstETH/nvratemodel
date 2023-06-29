# nvratemodel
Numerical rate models to simulate the photo-physics of the Nitrogen-Vacancy (NV) center in diamond.

What is this?
--------------
What this library can do:
- For the first time, it is possible to simulate the NV center photo-physics over the commonly covered range of temperature (cryogenic to room temperature), strain/el. field, and magnetic field. For this, use the ```MEmodel``` (see below). The theory and details behind this model are provided in https://arxiv.org/abs/2304.02521.
- Simulate at high computational speed the NV center photo-physics at elevated temperature (e.g. room temperature). For this, use the ```HighTmodel``` (see below), which is a commonly used classical rate model. Even though this model is very common, the accessibility of simulating e.g. contrast or SNR vs. key parameters with just a few lines of code, as well as the inclusion of the "temperature reduction factor" (```highT_trf```) (see below), also make this part of the library a useful tool.

What this library cannot do (at this point):
- Simulate quantum gates on the spin other than instantaneous pi-pulses.
- Include the hyperfine structure in the simulation.
- Include the effects of strain/el. field on the ground state, and the effect of its on-axis components in general.



Installation
-------------
You need to install this library into a python environment. Consider the following steps:

- If you do not yet have a python environment (ideally >= python 3.8), first create one: For an example with Anaconda (recommended for unexperienced users) see https://conda.io/projects/conda/en/latest/user-guide/getting-started.html.
  Open the Anaconda Prompt console and create your ```myNVenv``` with
  ```
  conda create --name myNVenv python=3.9
  ```
- Activate the environment and install in the following order the required packages by running the following lines successively:
  ```
  activate myNVenv
  ```
  ```
  conda install numpy scipy matplotlib numba git
  ```
  If you use an already existing environment, and you are uncertain which packages you have installed already, check this via:
  ```
  conda list
  ```
  Please make sure all packages are **installed via conda, and not via pip**, as the performance of conda numpy with conda numba works best.

- Now we install this library into the environment by:
  ```
  pip install git+https://github.com/sernstETH/nvratemodel.git
  ```
  If you wish to uninstall it again later, use:
  ```
  pip uninstall nvratemodel
  ```

You can also use binder to directly use the library online without the need of any installation:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sernstETH/nvratemodel/HEAD)

But note that in its current state, it is extremely slow compared to an installation as given above, and should rather be seen as a tool to view the code.



First steps
-----------
Check the installation in python by entering in the Anaconda Prompt from above (with ```myNVenv``` active)
```
python
```
and import the library as:
```
import nvratemodel as nv
```
Then, run an example
```
nv.example_PRL_Fig3a.run(path=None)
```
which should look like the plot in our publication https://arxiv.org/abs/2301.05091.
You can further test the following examples
```
example_PRB_Fig7a
example_PRB_Fig4h
example_PRB_Fig3b
example_PRB_Fig7b
```
which should look like plots in our other, related publication https://arxiv.org/abs/2304.02521.

The code of these examples should serve as a guide on how to use the nvratemodel library. The various plot routines readily provided by the library, as well as the formalism of ```modeldict```s (dictionaries that contain a set of NV parameters) and ```NVrateModel```s (objects to simulate the behavior of a set of NV parameters) are further explained below.

There are two more things to note before you get started:

- Saving simulation results:
  
  In all examples above, as well as all plot routines (see below), the optional argument ```path``` can be set to ```PATH_DATA```, which defaults to ```your_user_folder/nvratemodel_data/```. If you wish to set a different path, modify the ```PATH_DATA``` parameter in the ```GLOBAL.py``` file located in the installed nvratemodel root directory. In the same file, you can also modify the default values for the  NV parameters used by the model. Of course, all these default values are only used, if no other values are explicitly provided by the user in the function calls.
  Test 
  ```
  nv.example_PRL_Fig3a.run(path=nv.PATH_DATA)
  ```
  to see whether it works and to have a look at the saved data structure (a human-readable ```.json``` file).

- Generating a LUT of the Debye Integral:
  
  As you probably noticed in the example above, the library printed a note about a missing lookup table (LUT) of the Debye Integral. If you just want to test the nvratemodel library a bit or if you just intend to use it at room temperature (```HighTmodel```), you can just ignore this. But if you want to use the ```MEmodel``` and are interested in a faster computational speed, do as the printed note tells you: generate (and automatically save in the root directory of the nvratemodel) a LUT, which will take on the order of 25min, but saves you time on all computations later on.
  Call 
  ```
  nv.updateDebyeIntegralFullLUT(0.168)
  ```
  for the default Debye cutoff energy of ```phonCutoff=0.168```eV. You can also set other values of ```modeldict['phonCutoff']``` (see below) - the library will then ask you again to update the LUT with the values requested.



Using modeldicts and NVrateModels
---------------------------------
This library has two core objects to handle simulations:

- ```modeldict```s are python dictionaries that contain a full set of NV parameters. Such parameters are e.g. the magnetic field, its direction, or also the ```phonCutoff```mentioned above. They are generated by the function:
  ```
  mymodeldict = nv.makeModelDict(**kwargs)
  ```
  For possible ```kwargs```, which at the same time are the keys of a ```modeldict```, have a look at the doc string:
  ```
  help(nv.makeModelDict)
  ```
  These ```kwargs``` also correspond to the parameters given in Tab. 1 of https://arxiv.org/abs/2304.02521. You can view the default values that are used if not all possible ```kwargs``` are provided by:
  ```
  nv.printModelDict(nv.makeModelDict())
  ```
  
- ```NVrateModel```s are objects to simulate the population dynamics, and thus photoluminescence (PL), of a set of NV parameters. Several ```NVrateModel```-types are available:
  - ```MEmodel```: Master equation based model to simulate the NV center population dynamics over the full range of strain/el. field, magnetic field, and temperature from cryogenic to room temperature.
  - ```LowTmodel```: Classical rate equation model to simulate the NV center population dynamics at cryogenic temperature. Up to around 30K, the classical ```LowTmodel``` shows a similar behavior as the ```MEmodel``` and might be useful as an approximation with significantly enhanced computational speed. The correct treatment of orbital hopping due to electron-phonon coupling is the ```MEmodel```, though. In the ```LowTmodel```, orbital hopping is introduced as a classical rate that also destroys the spin space coherences (which is wrong). In contrast to the ```SZmodel``` and https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.128.177401, the ```LowTmodel``` introduces the rates in the same bases as the ```MEmodel```, see Fig. 2(d) of https://arxiv.org/abs/2304.02521.
  - ```HighTmodel```: Classical rate equation model to simulate the NV center population dynamics around room temperature, as often employed in the literature. As the orbital hopping rate of the ```MEmodel``` dominates the population dynamics, the ```MEmodel``` starts to resemble the behavior of this classical ```HighTmodel```. It can thus be used for computations at high speeds at elevated temperatures. Note that for high strain ```modeldict['Eperp']```, some orbital properties remain at elevated temperatures. These are only included in ```HighTmodel``` when ```modeldict['highT_trf']=True```. For more details see https://arxiv.org/abs/2304.02521.
  
  The NV parameters can be provided in the instantiation of an ```NVrateModel``` object as kwargs based on a ```modeldict```. The following three code blocks generate the same ```NVrateModel``` object called ```myMEmodel```:
  ```
  myMEmodel = nv.MEmodel(B=10e-3, T=300)
  ```
  ```
  mymodeldict = nv.makeModelDict(B=10e-3, T=300)
  myMEmodel = nv.MEmodel(**mymodeldict)
  ```
  ```
  mymodeldict = nv.makeModelDict()
  mymodeldict['B'] = 10e-3
  mymodeldict['T'] = 300
  myMEmodel = nv.MEmodel(**mymodeldict)
  ```
  To obtain e.g. the steady-state PL we can call:
  ```
  myMEmodel.PL()
  ```
  Since the ```MEmodel``` should give the same result at room temperature (```T=300```K), we can compare this to the PL of the classical rate model with the same parameter set:
  ```
  nv.HighTmodel(**mymodeldict).PL()
  ```
  To view all methods that ```NVrateModel``` objects have, take a look at the doc strings of the methods of the base class ```NVrateModel```.


A word on the usage of the numba package
----------------------------------------
Maybe you noticed above, that calling e.g. ```nv.HighTmodel(**mymodeldict).PL()``` for the first time is dramatically slower than calling it for the second time. This is because the numba package has to translate python code to fast machine code when the required functions are called for the first time.
If you wish to disable the usage of numba, you can easily do this by setting
```
NUMBA_OPT = False
```
in the ```GLOBAL.py``` file located in the installed nvratemodel root directory. This might be helpful when you cannot make sense of cryptic error messages generated by numba.



Overview of readily available simulation routines
--------------------------------------------------
The code of the examples in the example folder of this library should serve as a guide on how to use the nvratemodel library.

... upload in progress


Programming simulations yourself
---------------------------------
... upload in progress





