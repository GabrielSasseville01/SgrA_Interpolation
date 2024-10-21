This is a repository for the Python code used to generate the animation in Witzel et al. 2021. It also contains the animation (Figure 22 of Witzel et al.), the particle system with the posterior of the model parameters, and an example Jupyter notebook.

###################################################

The file included are:

run.particle : The "particle" file of the analysis decribed in Witzel et al. 2021, containing the model posterior that is the result of the ABC algorithm.

ssc_model.py : Python module with classes for generating synchrotron and synchrotron self-Compton spectra, as well as model their temporal evolution.

Animation demo.ipynb : Example Jupyter notebook that shows how to create an own animation.

SED_movie_demo.mp4 : SED Animation for particle 1418, as shown in Fig. 22 of  Witzel et al. 2021.

###################################################

Dependencies:

The python packages needed to execute ssc_model.py are: sys, math, decimal, numpy, mpmath, scipy, matplotlib, and mpl_toolkits. Furthermore, generating and saving the animation requires ffmpeg. Animation demo.ipynb requires Jupyter.

###################################################

Use:

Animation demo.ipynb demonstrates mainly the use of the class "SEDMovie". Its method "ModelSetup" calls all required classes and sets up the synchrotron/SSC model and the time series model with the input parameters. Of course, the other classes can be used independently of "SEDMovie", and basic documentation can be retrieved via: 

import ssc_model as model

help(model)

Note that the class "ElectronPlasma" requires as input instances of "SingleZoneSgrA" and '"constants", and "SyncSED" instances of "SingleZoneSgrA", '"constants", and "ElectronPlasma". However, several methods in "SyncSED" have keywords to override parameters provided by "SingleZoneSgrA" or "ElectronPlasma".


