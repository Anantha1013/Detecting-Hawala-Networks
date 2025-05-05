import networkx as nx
import torch
from torch_geometric.utils import from_networkx # type: ignore
from torch_geometric.nn import GCNConv # type: ignore
import pandas as pd

#Features to consider for GNN
def compute_node_features(nx_graph,df_account,df_transaction):
    in_degrees = dict(nx_graph.in_degree())
    out_degrees = dict(nx_graph.out_degree())
    # pagerank = nx.pagerank(nx_graph)

    #removing withdrawal and deposit
    df_new=df_transaction.loc[(df_transaction['tx_type']!="DEPOSIT") & (df_transaction['tx_type']!="WITHDRAWAL")]
    df_new['receiver_id']=df_new['receiver_id'].astype(int)


    #amount of money sent by node
    sent_money=df_new.groupby('sender_id')['tx_amount'].sum().reset_index()
    money_sent=sent_money.set_index('sender_id')['tx_amount'].to_dict()
    # print(money_sent)

    #amount of money received by node
    received_money=df_new.groupby('receiver_id')['tx_amount'].sum().reset_index()
    money_received=received_money.set_index('receiver_id')['tx_amount'].to_dict()
    # print(money_received)

    #average money sent by node
    avg_sent_money=df_new.groupby('sender_id')['tx_amount'].mean().reset_index()
    avg_money_sent=avg_sent_money.set_index('sender_id')['tx_amount'].to_dict()
    # print(avg_money_sent)

    #average money received by node
    avg_received_money=df_new.groupby('receiver_id')['tx_amount'].mean().reset_index()
    avg_money_received=avg_received_money.set_index('receiver_id')['tx_amount'].to_dict()
    # print(avg_money_received)


    #no of unique ids this account sent to
    accounts=df_new.groupby('sender_id')['receiver_id'].nunique().reset_index()
    unique_accounts_sent=accounts.set_index('sender_id')['receiver_id'].to_dict()
    # print(unique_accounts_sent)

    #ipaddress
    df_account['ip_int'] = df_account['ip_address'].astype('category').cat.codes
    ip_add=df_account.set_index('account_id')['ip_int'].to_dict()
    # print(ip_add)


    accounts=df_account['account_id'].tolist()

    # for account_id in accounts:
    #   val=in_degrees.get(account_id)
    #   print(val)

    features = []
    for account_id in accounts:
      features.append([
             in_degrees.get(account_id,0),  # In-degree
             out_degrees.get(account_id,0),  # Out-degree
             money_sent.get(account_id, 0),  # Money sent
             money_received.get(account_id, 0),  # Money received
             avg_money_sent.get(account_id, 0),  # Average money sent
             avg_money_received.get(account_id, 0),  # Average money received
             unique_accounts_sent.get(account_id, 0),  # Unique accounts sent to
             ip_add.get(account_id,0)  # IP address
         ])
    return torch.tensor(features, dtype=torch.float)


#Defining GNN
class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 16)  # Hidden layer
        self.conv2 = GCNConv(16, out_channels)  # Output layer

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

#Hypersphere learning

def hypersphere_loss(embeddings, center, r2, nu):
    dist_sq = torch.sum((embeddings - center) ** 2, dim=1)
    penalty = torch.clamp(dist_sq - r2, min=0)
    loss = r2 + (1 / (nu * len(embeddings))) * torch.sum(penalty)
    return loss, dist_sq
#Aggregate features in pytorch tensor and call the model
def compute_node_embeddings(nx_graph,df_account,df_transaction, out_channels, epochs=100, nu=0.05):
    #converts into pytorch tensor of x and edge_index and other features
    pyg_graph = from_networkx(nx_graph)

    #Assign computed features
    #x has features
    #edge_index has edge information
    pyg_graph.x = compute_node_features(nx_graph,df_account,df_transaction)

    # number of classsification features
    #print(pyg_graph.x.size(1)) # number of classsification features

    #calling the model with inchannel=8 and outchannel=16
    #i.e 16 dimension embedding
    model = GCN(pyg_graph.x.size(1), out_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Compute initial center c from first pass (fixed)
    with torch.no_grad():
        init_embeddings = model(pyg_graph.x, pyg_graph.edge_index)
        center = torch.mean(init_embeddings, dim=0).to(device)

    # Start with a default radius
    r2 = torch.tensor(1.0, requires_grad=False).to(device)
    

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        embeddings = model(pyg_graph.x, pyg_graph.edge_index)

        loss, dist_sq = hypersphere_loss(embeddings, center, r2, nu)
        loss.backward()
        optimizer.step()

        # Update r2 to cover (1 - nu)% of the nodes
        with torch.no_grad():
            r2 = torch.quantile(dist_sq, 1 - nu)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}: Loss={loss.item():.4f}, Radius²={r2.item():.4f}")

    # Compute final embeddings and anomaly scores
    model.eval()
    with torch.no_grad():
        final_embeddings = model(pyg_graph.x, pyg_graph.edge_index)
        anomaly_scores = torch.sum((final_embeddings - center) ** 2, dim=1) / r2

    return final_embeddings.cpu(), anomaly_scores.cpu()
    #embeddings = model(pyg_graph.x, pyg_graph.edge_index)
    #return embeddings

# Example graph
# G=nx.DiGraph()
# G.add_nodes_from(['a','b','c','d','e','f','g','h','v'])
# G.add_edges_from([('a','b',),('b','c',),('c','h',),('h','v',),('h','g',),('e','a',),('e','v',),('d','e',),('d','h',),('f','e',),('f','g',),('v','a',),('v','c',)])
G = nx.read_graphml("graph.graphml")
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

df_account = pd.read_csv('fin-account.csv')
df_transaction = pd.read_csv('fin-transaction.csv')

node_embeddings, anomalies = compute_node_embeddings(G,df_account,df_transaction, out_channels=16)  # 3 output dimensions

#print(len(node_embeddings[0]))
print(node_embeddings)
print("Anamalies")
print(anomalies)
# for key, value in node_embeddings.items():
#    df_account.loc[df_account["account_id"]==int(key),'GCN']=value

# df_account.to_csv("fin-account.csv",header=True,index=False)
# for i, score in enumerate(anomalies):
#     print(i,"score",score)
anomaly_dict = {i: score.item() for i, score in enumerate(anomalies)}
print("Anamlaies dict")
print(len(anomaly_dict))
# threshold = 1.5
# anomalous_indices = torch.nonzero(anomalies > threshold, as_tuple=False).squeeze()

# # Convert to list if you want plain Python indices
# anomalous_nodes = anomalous_indices.tolist()

# print("Nodes with score > 1.5:", anomalous_nodes)

df = pd.read_csv("fin-account.csv")

for key, value in anomaly_dict.items():
   df.loc[df["account_id"]==key,'Anomaly Score']=value

df.to_csv("fin-account.csv",header=True,index=False)
