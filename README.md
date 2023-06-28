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
  If you wish to uninstall it again later, use
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
Then run an example, e.g.
```
nv.example_PRL_Fig3a.run(path=None)
```
Which should look like the plot in our publication https://arxiv.org/abs/2301.05091.
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
  
  In all examples above, as well as all plot routines (see below), the optional argument ```path``` can be set to ```PATH_DATA```, which defaults to ```C:/nvratemodel_data```. If you wish to set a different path, modify the ```PATH_DATA``` parameter in the ```GLOBAL.py``` file located in the installed nvratemodel root directory. In the same file, you can also modify the default values for the  NV parameters used by the model. Of course, all these default values are only used, if no other values are explicitly provided by the user in the function calls.
  Test 
  ```
  nv.example_PRL_Fig3a.run(path=nv.PATH_DATA)
  ```
  to see whether it works and to have a look at the saved data structure (a human-readable ```.json``` file).

- Generating a LUT of the Debye Integral:
  
  As you probably noticed in the example above, the library printed a note about a missing look-up table (LUT) of the Debye Integral. If you just want to test the nvratemodel library a bit or if you just intend to use it at room temperature (```HighTmodel```), you can just ignore this. But if you want to use the ```MEmodel``` and are interested in a faster computational speed, do as the printed note tells you: generate (and automatically save in the root directory of the nvratemodel) a LUT, which will take some minutes, but saves you the time on all computations later on.
  Call 
  ```
  nv.updateDebyeIntegralFullLUT(0.168)
  ```
  for the default Debye cutoff energy of ```phonCutoff=0.168```eV. You can also set other values of ```modeldict['phonCutoff']``` (see below) - the library will then ask you again to update the LUT with the values requested.


Overview of the readily available plot routines
-----------------------------------------------
The code of the examples in the example folder of this library should serve as a guide on how to use the nvratemodel library.

... upload in progress



How to code with this library yourself
---------------------------------------
... upload in progress


