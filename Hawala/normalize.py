import pandas as pd

df = pd.read_csv('fin-account.csv')

cols_to_normalize = ['pageRank','hiddenLink','blackHole']
for col in cols_to_normalize:
    df[col + '_z'] = (df[col] - df[col].mean()) / df[col].std()
df.to_csv("fin-account.csv",header=True,index=False)