# %%
import configurations

# File to print out the configs

# Call configurations with experiments folder as differentiator of .ini file
configs = configurations.Config('default')

print('Preprocessed data path: \n' + configs.dataDir + '\n')
print('Confounds file pattern: \n' + configs.confoundsFilePattern + '\n')
print('Preprocessed BOLD images file pattern: \n' + configs.preprocessedImagePattern + '\n')
print('Masked BOLD images file pattern: \n' + configs.maskedImagePattern + '\n')
print('Mask file and path: \n' + configs.maskDataFile + '\n')
print('Intermediate folder path: \n' + configs.intermediateDataPath + '\n')
print('Participants summary file: \n' + configs.participantsSummaryFile + '\n')
print('Raw voxel df filename: \n' + configs.rawVoxelFile + '\n')
print('Testing subject dir: \n' + configs.subjectDir + '\n')
print('Testing session dir: \n' + configs.sessionDir + '\n')
print('Testing save dir: \n' + configs.saveDir + '\n')
print('Start of time slice: \n' + configs.startSlice + '\n')
print('End of time slice: \n' + configs.endSlice + '\n')

# %%
