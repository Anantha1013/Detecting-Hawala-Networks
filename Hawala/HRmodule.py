import pandas as pd

df = pd.read_csv('fin-account.csv')

cols = ['pageRank_z','hiddenLink_z','blackHole_z']
df['hawalaRanking'] = df[cols].mean(axis=1)
df.to_csv('fin-account.csv' ,header=True,index=False)