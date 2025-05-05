import networkx as nx
import itertools
import pandas as pd

G = nx.read_graphml("graph.graphml")
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

#pagerank
def compute_pageRank(graph):
  pagerank = nx.pagerank(graph, alpha=0.85)
  return pagerank

pagerank = compute_pageRank(G)
print(pagerank.get(4500))

#write to account csv
df = pd.read_csv('fin-account.csv')

for key, value in pagerank.items():
   df.loc[df["account_id"]==key,'pageRank']=value

df.to_csv("fin-account.csv",header=True,index=False)