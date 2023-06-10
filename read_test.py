from math import floor
from turtle import update
from urllib import request
from pulp import *
import pandas as pd
import networkx as nx
import os
import time
import matplotlib.pyplot as plt
import random
import copy
from scipy import stats
from gurobipy import *
import numpy as np
from creat_topology import creat_net

def draw_topology(G, fileName):
    pos = nx.spring_layout(G)

    ncolor=['r']+['y' for index in range(6)]+['r']+['y' for index in range(5)]+['r','r','r','r']+['y' for index in range(6)]+['r']+['y' for index in range(8)]+['r']+['y' for index in range(6)]+['r']+['y' for index in range(12)]+['r']+['y' for index in range(8)]+['r']+['y' for index in range(9)]
    nsize=[50]+[30 for index in range(6)]+[50]+[30 for index in range(5)]+[50,50,50,50]+[30 for index in range(6)]+[50]+[30 for index in range(8)]+[50]+[30 for index in range(6)]+[50]+[30 for index in range(12)]+[50]+[30 for index in range(8)]+[50]+[30 for index in range(9)]
    #fsize=[4 for i in range(1,72)]
    nx.draw(G, pos=pos,node_color=ncolor,node_size=nsize,width=0.5, arrowsize=0.5,arrowstyle='fancy')

    #nx.draw(G, pos=pos)
    # generate node_labels manually
    node_labels = {}
    edge_labels = {}
    for node in G.nodes:
        # node_labels[node] = str(node)+','+str(G.nodes[node]['node_type'])
        node_labels[node] = str(node)

    for edge in G.edges:
        edge_labels[edge] = G[edge[0]][edge[1]]['weight']

    nx.draw_networkx_labels(G, pos, labels=node_labels,font_size=4)

    # no edge_labels parameter, default is showing all attributes of edges
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,font_size=3)
    plt.savefig(fileName,dpi=1000,bbox_inches='tight')
    #plt.show() #实时展示拓扑
    plt.clf()


t_o=[1,1,14,14,15,16,17,17,24,17,17,40,53]+[1 for index in range(6)]+[8 for index in range(5)]+[17 for index in range(6)]+[24 for index in range(8)]+[33 for index in range(6)]+[40 for index in range(12)]+[53 for index in range(8)]+[62 for index in range(9)]
t_d=[14,8,8,15,16,17,24,33,33,40,53,53,62]+[index for index in range(2,8)]+[index for index in range(9,14)]+[index for index in range(18,24)]+[index for index in range(25,33)]+[index for index in range(34,40)]+[index for index in range(41,53)]+[index for index in range(54,62)]+[index for index in range(63,72)]
t_c=[155,155,155,620,620,620,205,205,205,205,205,205,205]+[155 for index in range(11)]+[205 for index in range(49)]
topology_data={
        'origin': t_o+t_d,
        'destination': t_d+t_o,
        'capacity': t_c+t_c,
    }
supplement_key={
    'origin':topology_data['origin'],
    'destination':topology_data['destination'],
    'val':topology_data['capacity'],
}
supplement_key0={
    'origin':topology_data['origin'],
    'destination':topology_data['destination'],
    'val':[0 for index in range(len(topology_data['capacity']))],
}
routes=topology_data
s_d={
    'node_name':range(1,max(topology_data['origin'])+1),
    'node_type':['unused' for index in range(max(topology_data['origin']))],
}

# print(request_data)

G = nx.DiGraph()  # 创建空的有向图

for i in range(len(s_d['node_name'])):
    G.add_node(s_d['node_name'][i], node_type=s_d['node_type'][i], node_in=0, node_out=0)

for i in range(len(routes['origin'])):
    G.add_edge(routes['origin'][i], routes['destination'][i], capacity=routes['capacity'][i])


b=np.load('rowdata/3.2jhgx/R1/1461_requestdata.npy',allow_pickle=True)
#d=nx.from_numpy_array(b)
print(b)
n1=np.load('rowdata/3.2jhgx/G1/1461_netdata.npy',allow_pickle=True)
n2=np.load('rowdata/3.2jhgx/G2/1461_netdata.npy',allow_pickle=True)
n3=np.load('rowdata/3.2jhgx/G3/1461_netdata.npy',allow_pickle=True)
G1=nx.from_numpy_array(n1)
G2=nx.from_numpy_array(n2)
G3=nx.from_numpy_array(n3)
# draw_topology(G1, 'topology1.jpg')
# draw_topology(G2, 'topology2.jpg')
# draw_topology(G3, 'topology3.jpg')
print(G1)
for (i,j) in G.edges():
    if G1.has_edge(i-1,j-1)==0 and G1.has_edge(j-1,i-1)==0:
        G1.add_edge(i-1,j-1, weight=0)
    if G2.has_edge(i-1,j-1)==0 and G2.has_edge(j-1,i-1)==0:
        G2.add_edge(i-1,j-1, weight=0)
    if G3.has_edge(i-1,j-1)==0 and G3.has_edge(j-1,i-1)==0:
        G3.add_edge(i-1,j-1, weight=0)
print(G1)
# draw_topology(G1, 'topology1.jpg')
# draw_topology(G2, 'topology2.jpg')
# draw_topology(G3, 'topology3.jpg')
alpha1=[]
c1=[]
c=[]
for (i,j) in G1.edges():
    c1.append(G1[i][j]['weight'])
    c.append(G[i+1][j+1]['capacity'])
    alpha1.append(1-G1[i][j]['weight']/G[i+1][j+1]['capacity'])

alpha2=[]
c2=[]
c=[]
for (i,j) in G2.edges():
    c2.append(G2[i][j]['weight'])
    c.append(G[i+1][j+1]['capacity'])
    alpha2.append(1-G2[i][j]['weight']/G[i+1][j+1]['capacity'])

alpha3=[]
c3=[]
c=[]
for (i,j) in G3.edges():
    c3.append(G3[i][j]['weight'])
    c.append(G[i+1][j+1]['capacity'])
    alpha3.append(1-G3[i][j]['weight']/G[i+1][j+1]['capacity'])

print(alpha1)
print(alpha2)
print(alpha3)
print(c1)
print(c2)
print(c3)
print(c)