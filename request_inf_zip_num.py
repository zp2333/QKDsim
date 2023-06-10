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

def creat_request(G, lam, miu, sigma):
    aver = G.size()/(len(list(G))+1)
    request_data = []
    request_source = []
    request_destination = []
    bottle_node=[]
    num = len(G.nodes)
    request_num = stats.poisson.rvs(mu=lam, size=num)
    for i in G.nodes:
        if G.degree(i)<=2:
            bottle_node.append(i)
    random.seed()
    flag_node=random.choice(bottle_node)
    for i in G.nodes:
        if G.degree(i)<=aver and i!=flag_node:
            # request_num[i-1]=0
            request_num[i-1]=stats.poisson.rvs(mu=lam)
        elif i==flag_node:
            request_num[i-1]=1
            print("successful")
        else:
            request_num[i-1]=stats.poisson.rvs(mu=G.degree(i)/aver*lam)
        for j in range(request_num[i-1]):
            if i!=flag_node:
                request_source.append(i)
                random.seed()
                m = random.choice(list(G))
                while m == i or m==flag_node:
                # while m == i :
                    random.seed()
                    m = random.choice(list(G))
                request_destination.append(m)
                request_data.append(abs(round(random.gauss(miu, sigma))))
            else:
                request_source.append(i)
                random.seed()
                m=random.choice(list(G[i]))
                request_destination.append(m)
                request_data.append(G[i][m]['capacity'])
    request_inf = {
        'request_data': request_data,
        'request_source': request_source,
        'request_destination': request_destination
    }
    return request_inf

# supplement_key = pd.read_excel("data_supplement.xlsx", sheet_name="node_data")
# supplement_key0 = pd.read_excel("data_supplement.xlsx", sheet_name="node_data0")
# fpath = os.path.join("data_copy.xlsx")
# routes = pd.read_excel(fpath, sheet_name="node_data")
# s_d = pd.read_excel(fpath, sheet_name="s_d")


#topology_data=creat_net("archive\Amres.gml")
t_o=[1,1,14,14,15,16,17,17,24,17,17,40,53]+[1 for index in range(6)]+[8 for index in range(5)]+[17 for index in range(6)]+[24 for index in range(8)]+[33 for index in range(6)]+[40 for index in range(12)]+[53 for index in range(8)]+[62 for index in range(9)]
t_d=[14,8,8,15,16,17,24,33,33,40,53,53,62]+[index for index in range(2,8)]+[index for index in range(9,14)]+[index for index in range(18,24)]+[index for index in range(25,33)]+[index for index in range(34,40)]+[index for index in range(41,53)]+[index for index in range(54,62)]+[index for index in range(63,72)]
t_c=[154,154,154,620,620,620,205,205,205,205,205,205,205]+[154 for index in range(11)]+[205 for index in range(49)]
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

request_inf_zip=[]
request_number=[5]#5,10,15,20,25,30,35,40
request_order=[0,0,0,0,0,0,0,0]
for i in range(3000):#300
    for j in range(3,23,8):#3,84,10
        request_inf = creat_request(G, j/200, 32, 0)#512bytes->32
        num=len(request_inf['request_data'])-1
        if num in request_number:
            request_inf_zip.append(request_inf)
            request_order[int(num/5-1)]+=1

np.save('request_inf_data_zip_5.npy', request_inf_zip)
for i in request_number:
    print(i,request_order[int(i/5-1)])
# a = np.load('request_inf_data_zip.npy', allow_pickle=True)
# for i in a:
#     print(i)