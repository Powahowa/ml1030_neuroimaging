import pandas as pd

df = pd.read_pickle('./features/rawvoxelsdf-STCM_confoundsOut_43-103slice.pkl')

print(df)
print("DF Shape", df.shape)