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
from creat_topology import read_net

def creat_request(G, lam, miu, sigma):
    aver = G.size()*2/(len(list(G))+1)
    request_data = []
    request_source = []
    request_destination = []
    num = len(G.nodes)
    request_num = stats.poisson.rvs(mu=lam, size=num)
    for i in G.nodes:
        if G.degree(i)<=aver:
            request_num[i-1]=0
            # print(i)
        else:
            request_num[i-1]=stats.poisson.rvs(mu=G.degree(i)/aver*lam)
        # request_num[i-1]=stats.poisson.rvs(mu=G.degree(i)/aver*lam)
        for j in range(request_num[i-1]):
            request_source.append(i)
            random.seed()
            m = random.choice(list(G))
            random.seed()
            flag=random.choice(list(range(50)))
            while m == i or (G.degree(m)<=aver and flag!=1):
                random.seed()
                m = random.choice(list(G))
            request_destination.append(m)
            request_data.append(abs(round(random.gauss(miu, sigma))))
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
def creat_rinf_zip(netfile,datafile,data_choice):
    topology_data=read_net(netfile)
    routes=topology_data
    s_d={
        'node_name':range(1,max(topology_data['origin'])+1),
        'node_type':['unused' for index in range(max(topology_data['origin']))],
    }
    G = nx.DiGraph()  # 创建空的有向图

    for i in range(len(s_d['node_name'])):
        G.add_node(s_d['node_name'][i], node_type=s_d['node_type'][i], node_in=0, node_out=0)

    for i in range(len(routes['origin'])):
        G.add_edge(routes['origin'][i], routes['destination'][i], capacity=routes['capacity'][i])

    request_inf_zip=[]
    request_number=[1,3,5,7,9,11,13,15,17,19]#3,5,10,15,20,25
    request_order=[0,0,0,0,0,0,0,0,0,0]
    for i in range(600):#300
        for j in range(3,100,10):#3,84,10
            request_inf = creat_request(G, j/100, 32, 0)#512bytes->32
            num=len(request_inf['request_data'])
            if num in request_number:
                request_inf_zip.append(request_inf)
                request_order[int(num/2)]+=1#/5

    np.save(datafile+data_choice+'_request_inf_data_zip.npy', request_inf_zip)
    for i in request_number:
        print(i,request_order[int(i/2)])
    # a = np.load('request_inf_data_zip.npy', allow_pickle=True)
    # for i in a:
    #     print(i)

store_path='result/'
topo_choice=['Dataxchange','Sprint','EliBackbone']#'Chinanet','Geant2012','Dataxchange','Sprint','EliBackbone'
data_choice='alpha'

for topo_c in topo_choice:
    creat_rinf_zip("archive/"+topo_c+'.gml',store_path+topo_c+'/',data_choice)