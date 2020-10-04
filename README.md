# Sleep Deprivation Classification using BOLD fMRI Images
This is the capstone project by team Data Knights (Alex Fung, Viswesh Krishnamurthy, Tony Lee, Patrick Osborne) for York University's Certificate in Machine Learning in collaboration with CAMH, with the oversight of Dr. Erin Dickie.

The blog post on the goals, methodology, data used and initial results of this project can be found on Medium here: [Sleep Deprivation classification using BOLD fMRI Data](https://medium.com/@visweshkris/sleep-deprivation-classification-using-bold-fmri-data-9cb762720131)

## Contact Us for Questions

The codebase for this project is complex and reflects multiple different approaches to solving this classification problem. Please feel free to contact us at DataKnightsAI@gmail.com. We would be happy to explain the code to you and point you in the right direction.

## Configuration files and Experiments
In order to facilitate cross-platform usage and to have one central location to keep track of all file locations and configurations, file paths and parameters of an experiment are stashed in a *configs.ini* file. These files are in the *experiments* folder where each experiment's config file is separated by a folder with the experiment's name. These *config.ini* files are read by the *configurations.py* module, which need to be imported into each script that would require the configurations. Test the configurations with *configprint.py*. To use the configurations, import *configurations.py* then instantiate a configurations class. For example: 

    import configurations
    configs = configurations.Config('experiment-name')

Please ensure that *configurations.py* has the correct path to the *experiments* folder or else it won't be able to find the config files.

## Getting Started

In order to get any of these scripts to run, you will need to place standard fMRIPrep output in the "01-data" folder. The scripts expect at least some .nii.gz (NIfTI files) that represent the fMRI images. For the masking section, pre-computed mask files are also required in .nii.gz format.

You will need to modify the experiments/configurations files to point to your local machine.

The folders are numbered in order that we approached the problem (suggested run order).

If you get stuck, please contact us.