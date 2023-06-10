import networkx as nx
import matplotlib.pyplot as plt
import random
import pandas as pd
from haversine import haversine
from simQN_link import link_sim

G = nx.read_gml("archive\Chinanet.gml")
#print(G.nodes(data=True))
remove_node=[]
for node in G.nodes(data=True):
    if 'Longitude' not in node[1]:
        remove_node.append(node[0])
for node in remove_node:
    G.remove_node(node)
print(G.nodes(data=True))
node_data = {
    "node_name": [],
    "node_number": [],
    "node_Longitude":[],
    "node_Latitude":[],
}
for node in G.nodes(data=True):
    node_data['node_name'].append(node[0])
    node_data['node_number'].append(len(node_data['node_name']))
    node_data['node_Longitude'].append(node[1]['Longitude'])
    node_data['node_Latitude'].append(node[1]['Latitude'])
print(node_data)
origin = []
destination = []
capacity = []
aver = G.size()*2/(len(list(G))+1)
print(aver)
link_length=[]
for (i, j, data) in G.edges(data=True):
    dis = haversine((G.nodes[i]['Longitude'],G.nodes[i]['Latitude']),(G.nodes[j]['Longitude'],G.nodes[j]['Latitude']))
    link_length.append(dis)
    # origin.append(i)
    # destination.append(j)
    origin.append(list(G).index(i)+1)
    destination.append(list(G).index(j)+1)
    # if G.degree(i)>G.degree(j):
    #     capacity.append(link_sim(1000,dis,G.degree(j)))
    # else:
    #     capacity.append(link_sim(1000,dis,G.degree(i)))


# for (i, j, data) in G.edges(data=True):
#     print(data)
#     origin.append(list(G).index(i)+1)
#     destination.append(list(G).index(j)+1)
#     if G.degree(i) > aver and G.degree(j) > aver:
#         capacity.append(random.randint(220, 1840))
#     else:
#         capacity.append(random.randint(94, 372))
# dfdata = {
#     'origin': origin+destination,
#     'destination': destination+origin,
#     'capacity': capacity+capacity
# }
dfdata = {
    'origin': origin+destination,
    'destination': destination+origin,
    'capacity': link_length+link_length,
}
df = pd.DataFrame(dfdata)  # 创建DataFrame
node = pd.DataFrame(node_data)
writer = pd.ExcelWriter("data.xlsx")
df.to_excel(writer, index=False, sheet_name='node_data')
node.to_excel(writer, index=False, sheet_name='node')
writer.close()
# print(data)
# nx.draw_networkx_labels(G, pos, labels=node_labels)

# # no edge_labels parameter, default is showing all attributes of edges
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
# plt.show()  # 实时展示拓扑
# # plt.clf()
