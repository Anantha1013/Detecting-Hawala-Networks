import pandas as pd

df = pd.read_csv('fin-account.csv')

top_10_hawalaRanking = df.nlargest(20, 'hawalaRanking')
top_10_anomalyRanking = df.nlargest(20, 'Anomaly Score')

# print("Top 10 HR",top_10_hawalaRanking)
# print("Top 10 AR",top_10_anomalyRanking)

df_view1 = df[['account_id','first_name','hawalaRanking','Anomaly Score']]
df1 = df_view1.nlargest(10, 'hawalaRanking')

df_view2 = df[['account_id','first_name','hawalaRanking','Anomaly Score']]
df2 = df_view2.nlargest(10,'Anomaly Score' )

df_view3 = pd.merge(df1, df2,on='account_id' ,how='inner')

# print("--- Top Hawala Ranking ---")
# print(df1)

# print("--- Top Anomaly Ranking ---")
# print(df2)

# print("--- High Hawala and Anomaly Ranking ---")
# print(df_view3)

df_view4 = pd.concat([df1, df2]).drop_duplicates().reset_index(drop=True)

print("Hawala Ranking and Anomaly Ranking")
print(df_view4)

trans = pd.read_csv("fin-transaction.csv")
ref_trans = trans[(trans['sender_id'] == 10) | (trans['receiver_id'] == 10)]

print(ref_trans.shape)