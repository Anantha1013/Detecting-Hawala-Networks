import networkx as nx
import itertools
import pandas as pd

G = nx.read_graphml("graph.graphml")
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())


#BlackHole
#to prune predessesors recursively
def prune_predesessors(graph,node,pList):
  # print("From the pruning function")
  # print("node:",node)
  preds=graph.predecessors(node)
  preds_in_plist=[]
  for i in preds:
    if i in pList:
      preds_in_plist.append(i)
  # print("Preds in plist:",preds_in_plist)
  if(len(preds_in_plist)==0):
    return pList
  for predecessor in preds_in_plist:
    pList.discard(predecessor)
    prune_predesessors(graph,predecessor,pList)
  return pList


def iblackHoleMetric(graph,n):
  V=graph.nodes()
  E=list(graph.edges())
  blackhole_metric = {node: 0 for node in V}
  for i in range(2,n):
    print("hi ",i)
    potential_list = {node for node in V if graph.out_degree(node)<i}

    print("-----------------------------------")
    # print("i = ",i)

    # print("Initial potential list")
    # print(potential_list)

    #prunning
    #modifying pruning as slight mistake
    for node in list(potential_list):
      if any(successor not in potential_list for successor in graph.successors(node)):
        # print('--------')
        # print('--------')
        # print("direct successors:",node)
        potential_list.discard(node)
        potential_list=prune_predesessors(graph,node,potential_list)
        # for predecessor in graph.predecessors(node):
        #   potential_list.discard(predecessor)
    # print('--------')
    # print('---------')
    candidate_list = potential_list.copy()

    # print("Candidate list")
    # print(candidate_list)

    #refine candidate_list
    for v in list(candidate_list):
      closure = {v}
      stack = [v]
      while stack:
        current = stack.pop()
        for successor in graph.successors(current):
          if successor not in closure:
            closure.add(successor)
            stack.append(successor)

      # print("Closure for node ",v)
      # print(closure)

      if len(closure) > i:
        # rec=v
        # #discard all its predecessors
        # for node in graph.predecessors(rec):
        #   candidate_list.discard(node)
        #   rec=node
        candidate_list.discard(v)
        candidate_list=prune_predesessors(graph,v,candidate_list)


      elif len(closure) == i:
        for node in closure:
          blackhole_metric[node] = max(len(closure),blackhole_metric[node])
        # rec=v
        # #discard all its predesessors
        # for node in graph.predecessors(rec):
        #   candidate_list.discard(node)
        #   rec=node
        candidate_list.discard(v)
        candidate_list=prune_predesessors(graph,v,candidate_list)
    final_list = candidate_list.copy()
    # print("Final list")
    # print(final_list)
    #final filtering


    subgraphs=get_all_subgraphs(final_list,graph,i)
    # for subgraph in subgraphs:
    #   print("------------")
    #   print(subgraph)
    for subgraph in subgraphs:
      if is_weakly_connected(graph, set(subgraph)) and is_blackhole(graph,set(subgraph)):
        for node in subgraph:
          blackhole_metric[node] = max(len(subgraph),blackhole_metric[node])
          print("Update")
  return blackhole_metric

def is_weakly_connected(graph, nodes):
  subgraph_graph = graph.subgraph(nodes)
  return nx.is_weakly_connected(subgraph_graph)

def is_blackhole(graph, nodes):
  for node in nodes:
    for neighbour in graph.neighbors(node):
      if neighbour not in nodes:
        return False
  return True


def get_all_subgraphs(nodes,graph,r):
    subgraphs = []
    # print(nodes)
    # Generate all possible subsets of nodes, excluding the empty set

    for subset in itertools.combinations(nodes, r):
      subgraph = graph.subgraph(subset)  # Create a subgraph for each subset
      subgraphs.append(subgraph)

    subgraph_final=[]
    for sub in subgraphs:
      # print(sub.nodes())
      subgraph_final.append(sub.nodes())


    return subgraph_final

blackHole_metric=iblackHoleMetric(G,5)
print(blackHole_metric)


for key, value in blackHole_metric.items():
   df.loc[df["account_id"]==int(key),'blackHole']=value

df.to_csv("fin-account.csv",header=True,index=False)