#%%
import numpy as np 

a = np.array([5, 6, 7])

np.save('test12.npy', a, allow_pickle=True)

# %%
np.load('test12.npy', allow_pickle=True)
# %%
