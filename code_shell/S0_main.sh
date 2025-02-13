#!/bin/bash

#Remember, no spaces between variable names and variables i.e., X = 1 will cause an error. 

## Install virtual environment for running the python scripts
## If Env_installed is set to "no", the environment is installed automatically
## Remember to change Env_installed to "yes" before running the script next time
Env_installed="yes"

## First time runnig the script we need to install a Python environment
if [ "$Env_installed" = "no" ]; then
	source /usr/local/apps/anaconda3/bin/activate
	conda create -n STenv
	conda activate  STenv
	conda install pip
	~/.conda/envs/STenv/bin/pip install numpy matplotlib structure-tensor jupyter pandas opencv-python scikit-learn
	echo "<> Python environment installed"
elif [ "$Env_installed" = "yes" ]; then
	echo "<> Python environment already installed"
else
	echo "<> Env_installed should be either yes or no"
fi

## Assign directory for Python environment and active it.
source /usr/local/apps/anaconda3/bin/activate
conda activate  STenv

# Declare cropping parameters for all datasets in samples_names
declare -a sample_names=("CFPP_1")

# Declare cropping parameters for all datasets in samples_names
crop_samples=([5500, 6350, 1020, 650])

# Declare kernel size parameters for all datasets in samples_names
kernel_multiplier_FVF=(1)

## Iterate over the data files defined in sample_name
i=0
for sample_name in ${sample_names[@]}; do
	echo index $i
	crop=${crop_samples[i]}
	k_mul=${kernel_multiplier_FVF[i]}
	result_dir=../results
	
	sample_dir="$result_dir/${sample_name}_files"
	if [ -d "$sample_dir" ]; then
		counter=$(find "${result_dir}/" -name "${sample_name}_files*" -type d | wc -l) 
		mv $sample_dir "${sample_dir}_${counter}"
		mkdir $sample_dir
		mkdir "${sample_dir}/figures"
	else
		echo "No folder available. Result folder is created"
		mkdir $sample_dir
		mkdir "${sample_dir}/figures"
	fi
	
	## Remove file used for stopping previous job
	[ -e "stopfile.txt" ] && rm "stopfile.txt"
	
	## Remove LCK file from previous job
	[ -e *lck ] && rm *lck
	
	## Analyse the image images and define model dimensions
	sed -e "s/shell_sample_name/$sample_name/;s/shell_crop/$crop/;s/shell_kernel/$k_mul/" S1_FVF_ST.py > "${sample_name}_S1_FVF_ST.py"
	python3 "${sample_name}_S1_FVF_ST.py"
	echo "<> CT image has been analysed and model dimensions are exported"
	
	## Generate FE-model and export integration points
	module load abaqus/2023
	sed -e "s/IP1/$sample_name\_IP1/" I1_BoxIP.in > "${sample_name}_IP2.inp"
	sed -e "s/shell_sample_name/$sample_name/" S2_Box.py > "${sample_name}_S2_Box.py"
	unset SLURM_GTIDS
	abq2023 cae noGUI="${sample_name}_S2_Box.py"
	wait
	echo "FE-model has been generated and integration points are exported"
	
	## Map fiber orientations to integration points
	sed -e "s/shell_sample_name/$sample_name/" S3_mapping.py > "${sample_name}_S3_mapping.py"
	python3 "${sample_name}_S3_mapping.py"
	echo "<> Fiber orientations has been mapped to the integration points"
	
	### Run simulation in Abaqus with orientation information mapped to integration points
	sed "s/InputOrientFortran1/$sample_name\_PHI/;s/InputOrientFortran2/$sample_name\_FVF/" I2_user_subs.f > "${sample_name}_user_subs.f"
	sed "s/InputModelCase/$sample_name/" S4_Box_modified.py > "${sample_name}_S4_Box_modified.py"
	abq2023 cae noGUI="${sample_name}_S4_Box_modified.py"
	
	### Postprocessing
	mv "${sample_name}_IP2.dat" "Out-${sample_name}_INTCOOR.dat"
	sed -e "s/shell_sample_name/$sample_name/" S5_PostProcessing.py > "${sample_name}_S5_PostProcessing.py"
	python3 "${sample_name}_S5_PostProcessing.py"
	echo "<> Postprocessing of Abaqus data complete."
	
	## Move files to sample result folder
	rm *.com *.prt *.jnl 
	*IP2* *.inp
	mv "${sample_name}"* "Out-${sample_name}"* "$result_dir/${sample_name}_files"/
	i=$(( i + 1 ))
	echo index $i
	
done
	
echo "<> Script complete"
