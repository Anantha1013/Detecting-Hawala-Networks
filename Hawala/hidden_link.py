import pandas as pd
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

G = nx.read_graphml("graph.graphml")
print(G.number_of_nodes())
print(G.number_of_edges())



def ip_to_binary_vector(ip):
  return [int(bit) for octet in ip.split('.') for bit in format(int(octet), '08b')]

def hidden_link_metric(df_account):
  hidden_link = {node: 0 for node in df_account['account_id']}
  for i in range(0,len(df_account)):
    for j in range(i+1,len(df_account)):

        vector1=np.array(ip_to_binary_vector(df_account.iloc[i,13]))
        vector2=np.array(ip_to_binary_vector(df_account.iloc[j,13]))

        vector1=vector1.reshape(1,-1)#converting to 2d as cosine similarity expects 2d array
        vector2=vector2.reshape(1,-1)

        similarity = cosine_similarity(vector1, vector2)
        print(similarity[0][0])
        if similarity[0][0]>0.88:
            print("similarity greater than 0.88")

            hidden_link[df_account.iloc[i,0]]=max(float(similarity[0][0]),hidden_link[df_account.iloc[j,0]])
            hidden_link[df_account.iloc[j,0]]=max(float(similarity[0][0]),hidden_link[df_account.iloc[j,0]])


  return hidden_link



df = pd.read_csv('fin-account.csv')
hiddenlink=hidden_link_metric(df)
print(hiddenlink)


for key, value in hiddenlink.items():
   df.loc[df["account_id"]==key,'hiddenLink']=value

df.to_csv("fin-account.csv",header=True,index=False)