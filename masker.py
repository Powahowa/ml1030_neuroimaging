# %% 
from nilearn import plotting
from nilearn import image
import nilearn
import nibabel as nib
from nilearn.input_data import NiftiMasker


# %%
subjectDir = "./data/preprocessed/sub-9001/"

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
while i < 200:
    normalized_image = image.index_img(subjectDir + 
    sessionDir + 
    "func/sub-9001_ses-1_task-sleepiness_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz", i)

    # plt = plotting.plot_img(normalized_image, cut_coords=[0,0,0])
    # plt.savefig('./plots/series-normalized/' + str(i) + '.png')
    # plt.close()
    i += 1
    #i += 20

# %%
# How many images in the time dimension?

iterable = image.iter_img(subjectDir + 
sessionDir + 
"func/sub-9001_ses-1_task-faces_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
sum(1 for _ in iterable)

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
# Alternate method: Using Nibabel to load 4D image
img = nib.load(subjectDir + 
    sessionDir + 
    "func/sub-9001_ses-1_task-faces_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")




# %%

#plot mask (function)

def plotMask(subjectDir, sessionDir, normalized_image):

    # We first create a masker, and ask it to normalize the data to improve the
    # decoding. The masker will extract a 2D array ready for machine learning
    # with nilearn:
    masker = NiftiMasker(mask_img=subjectDir + 
        sessionDir + 
        "func/sub-9001_ses-1_task-sleepiness_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz", standardize=True)
    fmri_masked = masker.fit(normalized_image)

    # Generate a report with the mask on normalized image
    report = masker.generate_report()
    report

    # Check out the shape of normalized image with mask applied
    norm_maskapplied = masker.fit_transform(normalized_image)
    print(norm_maskapplied)
    norm_maskapplied.shape


# %%


sleepiness = image.load_img(subjectDir + 
sessionDir + 
"func/sub-9001_ses-1_task-sleepiness_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")

#%%
plotMask(subjectDir, sessionDir, sleepiness)

#%% test run before making function

# We first create a masker, and ask it to normalize the data to improve the
# decoding. The masker will extract a 2D array ready for machine learning
# with nilearn:
masker = NiftiMasker(mask_img=subjectDir + 
    sessionDir + 
    "func/sub-9001_ses-1_task-sleepiness_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz", standardize=True)
fmri_masked = masker.fit(sleepiness)

# Generate a report with the mask on normalized image
report = masker.generate_report()
report

# Check out the shape of normalized image with mask applied
norm_maskapplied = masker.fit_transform(sleepiness)
print(norm_maskapplied)
norm_maskapplied.shape

#%%

arrows = image.load_img(subjectDir + 
sessionDir + 
"func/sub-9001_ses-1_task-arrows_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")

faces = image.load_img(subjectDir + 
sessionDir + 
"func/sub-9001_ses-1_task-faces_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")

hands = image.load_img(subjectDir + 
sessionDir + 
"func/sub-9001_ses-1_task-hands_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")

rest = image.load_img(subjectDir + 
sessionDir + 
"func/sub-9001_ses-1_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")


# %%


# %%


# %%


# %%


# %%
# Check out the shape of non-normalized image with mask applied
nonnorm_maskapplied = masker.fit_transform(nonnormalized_image)
print(nonnorm_maskapplied)
nonnorm_maskapplied.shape

# %%
