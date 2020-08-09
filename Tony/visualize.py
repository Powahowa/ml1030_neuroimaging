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
plotting.plot_anat(subjectDir + 
    "anat/sub-9001_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz")

# %%
# Read from and return Nifti image object that is smoothed.
smoothed_img = image.smooth_img(subjectDir + 
    "anat/sub-9001_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz", fwhm=5)   
disp = plotting.plot_anat(smoothed_img)

# %%
# Read from and return Nifti image object that is smoothed.
smoothed_img = image.smooth_img(subjectDir + 
    "anat/sub-9001_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz", fwhm='fast')   
disp = plotting.plot_anat(smoothed_img)
disp.add_contours(smoothed_img, levels=[0.5], colors='r')

# %%
# Attempt to read and generate slideshow for 4D image: Normalized MNI Asym file
# NOTE: uncomment/comment stuff to actually generate the images to be used for slideshow.html
i = 0
while i < 100:
    normalized_image = image.index_img(subjectDir + 
    sessionDir + 
    "func/sub-9001_ses-1_task-faces_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz", i)

    # plt = plotting.plot_img(normalized_image, cut_coords=[0,0,0])
    # plt.savefig('./plots/series-normalized/' + str(i) + '.png')
    # plt.close()
    # i += 1
    i += 20

# %%
# Attempt to read generate slideshow for 4D image: Non-normalized file
# NOTE: uncomment/comment stuff to actually generate the images to be used for slideshow.html
i = 0
while i < 100:
    nonnormalized_image = image.index_img(subjectDir + 
    sessionDir + 
    "func/sub-9001_ses-1_task-faces_space-T1w_desc-preproc_bold.nii.gz", i)

    # plt = plotting.plot_img(nonnormalized_image, cut_coords=[0,0,0])
    # plt.savefig('./plots/series-nonnonmalized/' + str(i) + '.png')
    # plt.close()
    # i += 1
    i += 20

# %%
# How many images in the time dimension?
iterable = image.iter_img(subjectDir + 
    sessionDir + 
    "func/sub-9001_ses-1_task-faces_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
sum(1 for _ in iterable)

# %%
# Alternate method: Using Nibabel to load 4D image
img = nib.load(subjectDir + 
    sessionDir + 
    "func/sub-9001_ses-1_task-faces_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")

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
# Generate a report with the mask on normalized image
report = masker.generate_report()
report

# %%
# Check out the shape of normalized image with mask applied
norm_maskapplied = masker.fit_transform(normalized_image)
print(norm_maskapplied)
norm_maskapplied.shape

# %%
# Sanity check, generate a report with a mask on non-normalized image
fmri_masked_non_norm = masker.fit(nonnormalized_image)
report = masker.generate_report()
report

# %%
# Check out the shape of non-normalized image with mask applied
nonnorm_maskapplied = masker.fit_transform(nonnormalized_image)
print(nonnorm_maskapplied)
nonnorm_maskapplied.shape

# %%
