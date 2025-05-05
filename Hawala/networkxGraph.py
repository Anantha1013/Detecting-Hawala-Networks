from neo4j import GraphDatabase
import networkx as nx
import pandas as pd
import itertools

uri = "bolt://localhost:7687"  
user = "neo4j"
password = "07@April@2005"

driver = GraphDatabase.driver(uri, auth=(user, password))

def fetch_edges(tx):
    # Adjust query as per your relationship type and node labels
    query = """
    MATCH (a:Account)-[r:SENT]->(b:Account)
    RETURN a.account_id AS sender, b.account_id AS receiver
    """
    result = tx.run(query)
    return [(record["sender"], record["receiver"]) for record in result]

with driver.session() as session:
    edges = session.execute_read(fetch_edges)

G = nx.DiGraph()
G.add_edges_from(edges)
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())
nx.write_graphml(G, "graph.graphml")
