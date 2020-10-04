# %%
import pandas as pd
import numpy as np

# %%
subjectDir = "../data/ds000201-download/sub-9001/"
sessionDir = "ses-1/"

# %%
hands_df = pd.read_csv(subjectDir + sessionDir + 
                       "func/sub-9001_ses-1_task-hands_events.tsv", 
                       sep='\t')

# %%
hands_df

# %%
arrows_df = pd.read_csv(subjectDir + sessionDir + 
                       "func/sub-9001_ses-1_task-arrows_events.tsv", 
                       sep='\t')

# %%
arrows_df

# %%
faces_df = pd.read_csv(subjectDir + sessionDir + 
                       "func/sub-9001_ses-1_task-faces_events.tsv",
                       sep='\t')

# %%
faces_df

# %%
sleepiness_df = pd.read_csv(subjectDir + sessionDir + 
                       "func/sub-9001_ses-1_task-sleepiness_events.tsv",
                       sep='\t')

# %%
sleepiness_df

# %%
