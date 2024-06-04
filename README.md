# 2D Finite Element modelling of fiber-reinforced composites with variable fiber volume fractions and fiber orientations

## Introduction
This method covers the process of acquireing material characteristics from 2D images of fiber-reinforced composites. These are subsequently used for a static compression analysis in a Finite Element (FE) model in the commercial FE-software, Abaqus<sup>TM</sup>. The method estimates the Fiber Volume Fraction (FVF) distribution and fiber orientation distribution based on image characteristics, and is used to study the effect on material behavior when loaded in compression. This work is based on a carbon fiber-reinforced Vinyl Ester composite, where images are aquired using optical microscopy of a plane parallel with the global fiber direction. The method is also applicable for other fiber-reinforced material systems and other image acquisition methods capable of capturing images with a clear contrast between constituents. 

## Installation guide

This program is based on the Anaconda Distribution, where a virtual environment is used for running the analysis. The file S0_main.sh is the main script from where all analysis steps are controlled. This file also contains a function that installes the virtual environment and necessary Python packages. To enable this function set the parameter Env_installed="no". This should only be done the first time the shell script is executed. After installation of the virtual environment, set Env_installed="yes". 

## Directories and files
- code
	* S0_main.sh
		- Main shell script for running the entire analysis. Datasets for analysis are defined in this script. 
	* S1_STanalysis.py
		- Script for estimating fiber volume fractions and material orientations and define FE model dimensions.
		- Input: 2D image data.
		- Output: Arrays with FVFs and fiber orientations . (MAP_Var.npz) and FE model dimensions (ImgDim.txt).
	* S2_Box.py
		- Script for generating FE model with mesh.
		- Input: Model dimensions (ImgDim.txt).
		- Output: Integration point coordinates (INTCOOR.dat).
	* S3_mapping.py
		- Script for mapping FVFs anf fiber orientations estimated in S1_STanalysis to FE mesh generated in S2_Box.
		- Input: Material orientation information (MAP_Var.npz) and integration point coordinates (INTCOOR.dat).
		- Output: Fortran files with orientation information for all integration points.
	* S4_Box_modified.py
		- Script for updating the FE model with the USDFLD and ORIENT function for loading FVFs and fiber orientation information, rotating local coordinate systems, and running the FE simulation.
		- Input: Abaqus CAE file generated in S2_Box.py and Fortran files generated in S3_mapping.py
		- Output: Field and history output for post processing. CAE and ODB files for post-processing in Abaqus.
	* S5_PostProcessing.py
		- Script for post processing simulation results from Abaqus.
		- Input: Abaqus result files saved as .out files and integration point coordinates (INTCOOR.dat).
		- Output: Result plots.
		
	* M1_ImgHandling.py
		- Python module with functions for handling image data.
	* M2_Alignment.py
		- Python module with functions for orientation calculations and plotting.
	* M3_AbqFunctions.py
		- Python module with Abaqus functions for part/assembly/mesh generation.
	* M4_IntegrationPoints
		- Python module with a function for plotting integration points with field variables.
		
	* I1_BoxIP.in
		- Input file for Abaqus. Used for exporting .dat file with integration point coordinates.
	* I2_user_subs.f
		- Fortran file with USDFLD and ORIENT functions. 
		
- data
	* CFPP.tif
		- Tiff file of optical microscopy image from carbon fiber-reinforced Vinyl Ester pultruded profile.
	
- results
	* Empty by default. 
	* Is used for saving all relevant data from an analysis of a sample including all result figures.
	* All result files will be saved to a folder of the same name as the sample name. 
		- Note that if an analysis is run for the same sample name, then the previous sample result folder is renamed with an appended counter. 

## Guidelines

- The complete analysis is executed by the shell script S0_main.sh
	* Define the sample names used for the analysis and set the crop_samples parameter for the tiff file.
- Running the individual python scripts without the shell script. 
	* Change the sample_name from 'shell_sample_name" to the name of the sample, and run the python script. 
		- Make sure that the imported python modules and data files are available in the working directory.

## Example of method
A precompiled version of this work is presented in a CodeOcean Notebook. This version is independent of Abaqus<sup>TM</sup>.
Link to CodeOcean: **Remember**

## Authors and acknowledgment
Lead and corresponding auther: Ole Villiam Ferguson (olen@dtu.dk) \
Co-auther: Lars Pilgaard Mikkelsen

## References
[1] O. V. Ferguson, L. P. Mikkelsen, "Three-Dimensional Finite Element Modeling of Anisotropic Materials using X-ray Computed Micro-Tomography Data", *Software Impacts*, https://doi.org/10.1016/j.simpa.2023.100523  **change to new SI journal**

