# %% 
from nilearn import plotting
from nilearn import image

# %%
subjectDir = "../data/ds000201-download/sub-9001/"
sessionDir = "ses-1/"

# %%
# Plot anatomy image
plotting.plot_anat(subjectDir + sessionDir + "anat/sub-9001_ses-1_T1w.nii.gz")

# %%
# Plot glass brain
# TODO: Which image to use?
plotting.plot_glass_brain(subjectDir + sessionDir + "anat/sub-9001_ses-1_T1w.nii.gz")

# %%
# Read from and return Nifti image object that is smoothed.
smoothed_img = image.smooth_img(subjectDir + 
                                sessionDir + 
                                "anat/sub-9001_ses-1_T1w.nii.gz", fwhm=5)   
plotting.plot_anat(smoothed_img)

# %%
# Attempt to read 4D image
first_volume = image.index_img(subjectDir + 
                               sessionDir + 
                               "func/sub-9001_ses-1_task-arrows_bold.nii.gz", 0)

# %%
# The most basic plot, testing coords
plotting.plot_img(first_volume, cut_coords=[0,0,0])

# %%
# How many images in the time dimension?
iterable = image.iter_img(subjectDir + 
                               sessionDir + 
                               "func/sub-9001_ses-1_task-faces_bold.nii.gz")
sum(1 for _ in iterable)

# %%
