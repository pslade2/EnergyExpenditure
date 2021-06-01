# EnergyExpenditure
[![DOI](https://zenodo.org/badge/372942363.svg)](https://zenodo.org/badge/latestdoi/372942363)

Code for the "Sensing leg movement enhances wearable monitoring of energy expenditure" paper. Additional data for replicating this study is available: https://simtk.org/projects/energy-est

This folder contains data, code, and results for validating the Wearable System. The software version, package dependencies, and installation instructions are listed at the bottom of this note.

The code folder contains python notebook files to process the raw validation data and produce energy expenditure estimates (compute_real_time_results.ipynb) and compute the figures from the paper (plots.ipynb). These files are Jupyter Notebook files, detailed instructions on this type of file and how to open them are available (https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Notebook%20Basics.html). Once the files are open select 'Run' and then 'Run all cells'. The output will appear below each cell. The compute_real_time_results.ipynb will plot the energy expenditure estimates of the Wearable System and raw metabolics measurements as well as the absolute percent error between the steady-state estimates of the Wearable System and metabolics. The plots.ipynb will produce replicates of the images shown in the manuscript for validating the processing of the different methods of estimating energy expenditure. The runtime is approximately 5 minutes on a "normal" desktop.

The real_time_model folder contains the weights for the linear regression model used by the Wearable System and the python file used to estimate energy expenditure in real time on the portable microcontroller (real_time_est.py). 

The real_time_validation_data folder contains the metabolics and raw inertial measurement data for one of the validation subjects. This folder will need to be unzipped before being used. Each subject folder contains the raw metabolics data as a .xlsx file and conditions folders. The conditions folders contain the raw inertial measurement data broken into five second increments, stored in sequential 'npy' files. The file_timestamp.csv contains the timestamps when each of the 'npy' files were saved. The energy_exp_estimates.csv contains columns of the time from the start of the condition, date, and energy expenditure in Watts.

The results folder contains the estimates computed from the compute_real_time_results.ipynb to replicate the real-time Wearable System estimates from the validation experiment. The full_data folder contain all the data for the compared methods across all subjects to be able to replicate the figures in the paper.

The full dataset is available to reviewers in a private repository linked in the paper, but was not included in this folder due to size constraints. Upon acceptence this will be published in a public repository. This includes all simulation models, all data from each of the experiments, code to train the energy expenditure models, and processing code to compute estimates from the compared methods (heart rate, smartwatch, etc). 

Python version 3.6.1
Modules:
pandas (0.25.3)
numpy (1.17.4)
scikit-learn (0.21.3)
scipy (1.3.2)
setuptools (27.2.0)
natsort (6.2.0)
matplotlib (2.0.2)
jupyter (1.0.0)
ipython (5.3.0)

The installation process for Python and related packages will depend on the users operating system, but should take approximately 10 minutes on a "normal" desktop. See the python package installation guide for instructions: https://packaging.python.org/tutorials/installing-packages/
