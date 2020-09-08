# %%
import configparser
import pathlib
import os.path
import os

class Config:
    # Class that contains all required configurations
    __doc__ = 'Class that contains all required configurations'
    def __init__(self, experimentName):

        self.experimentName = experimentName
        
        parser = configparser.ConfigParser()

        if os.name == 'nt':
            configFile = '../experiments/' + experimentName + '/configs.ini'
        else:
            configFile = '../experiments/' + experimentName + '/configsSCC.ini'
        parser.read(pathlib.Path(configFile))

        self.dataDir = parser['paths']['preprocessedDataPath']
        self.confoundsFilePattern = parser['filepatterns']['confoundsFilePattern']
        self.preprocessedImagePattern = parser['filepatterns']['preprocessedImagePattern']
        self.maskedImagePattern = parser['filepatterns']['maskedImagePattern']
        self.maskDataFile = parser['files']['maskDataFile']
        self.intermediateDataPath = parser['paths']['intermediateDataPath']
        self.participantsSummaryFile = parser['files']['participantsSummaryFile']
        self.rawVoxelFile = parser['files']['rawVoxelFile']
        self.subjectDir = parser['testing']['subjectDir']
        self.sessionDir = parser['testing']['sessionDir']
        self.saveDir = parser['testing']['saveDir']
        self.startSlice = int(parser['constants']['startSlice'])
        self.endSlice = int(parser['constants']['endSlice'])
# %%
