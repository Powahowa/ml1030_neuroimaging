# %% 
from nilearn import plotting
from nilearn import image
import nilearn
import nibabel as nib

# %%
subjectDir = "../../data/preprocessed/sub-9001/"
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
disp = plotting.plot_anat(smoothed_img)

# %%
# Read from and return Nifti image object that is smoothed.
smoothed_img = image.smooth_img(subjectDir + 
                                sessionDir + 
                                "anat/sub-9001_ses-1_T1w.nii.gz", fwhm='fast')   
disp = plotting.plot_anat(smoothed_img)
disp.add_contours(smoothed_img, levels=[0.5], colors='r')

# %%
# Attempt to read 4D image: MNI Asym file
i = 0
while i < 100:
    normalized_image = image.index_img(subjectDir + 
    sessionDir + 
    "func/sub-9001_ses-1_task-faces_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz", i)

    plt = plotting.plot_img(normalized_image, cut_coords=[0,0,0])
    plt.savefig('./plots/series/' + str(i) + '.png')
    plt.close()
    i += 1
    # i += 10

# %%
# Attempt to read 4D image: reg file
i = 0
while i < 100:
    nonnormalized_image = image.index_img(subjectDir + 
    sessionDir + 
    "func/sub-9001_ses-1_task-faces_space-T1w_desc-preproc_bold.nii.gz", i)

    plt = plotting.plot_img(nonnormalized_image, cut_coords=[0,0,0])
    plt.savefig('./plots/seriesT1/' + str(i) + '.png')
    plt.close()
    i += 1
    # i += 10

# %%
# How many images in the time dimension?
iterable = image.iter_img(subjectDir + 
    sessionDir + 
    "func/sub-9001_ses-1_task-arrows_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
sum(1 for _ in iterable)

# %%
# Nibabel load 4D image
img = nib.load(subjectDir + 
               sessionDir + 
               "func/sub-9001_ses-1_task-faces_bold.nii.gz")

# %%
# Examine the header of the 4D image
print(img.header)

# %%
# What is the shape of the 4D image?
img.header.get_data_shape()

# %%
# What type of data is the 4D image?
img.dataobj

# %%
# Is this a proxy array?
nib.is_proxy(img.dataobj)

# %%
# We first create a masker, and ask it to normalize the data to improve the
# decoding. The masker will extract a 2D array ready for machine learning
# with nilearn:
from nilearn.input_data import NiftiMasker
masker = NiftiMasker(mask_img=subjectDir + 
               sessionDir + 
               "func/sub-9001_ses-1_task-faces_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz", standardize=True)
fmri_masked = masker.fit(normalized_image)

# %%
report = masker.generate_report()
report

# %%
temp = masker.fit_transform(nonnormalized_image)
temp
temp.shape

# %%
fmri_masked_non_norm = masker.fit(nonnormalized_image)
report = masker.generate_report()
report
# %%
