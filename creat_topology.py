from unittest import result
import networkx as nx
import matplotlib.pyplot as plt
import random
import pandas as pd
from haversine import haversine
from simQN_link import link_sim
def creat_net(file):
    G = nx.read_gml(file)
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
        origin.append(list(G).index(i)+1)
        destination.append(list(G).index(j)+1)
        if G.degree(i)>G.degree(j):
            capacity.append(int(link_sim(1000,dis*10,G.degree(j))*8/128))
        else:
            capacity.append(int(link_sim(1000,dis*10,G.degree(i))*8/128))
        
        # capacity.append(100)

    dfdata = {
        'origin': origin+destination,
        'destination': destination+origin,
        'capacity': capacity+capacity,
    }
    df = pd.DataFrame(dfdata)  # 创建DataFrame
    node = pd.DataFrame(node_data)
    writer = pd.ExcelWriter("data.xlsx")
    df.to_excel(writer, index=False, sheet_name='node_data')
    node.to_excel(writer, index=False, sheet_name='node')
    writer.close()
    result={
        'origin': origin+destination,
        'destination': destination+origin,
        'capacity': capacity+capacity,
    }
    return result

def read_net_degree(file):
    G = nx.read_gml(file)
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
    aver = G.size()*2/(len(list(G))+1)
    return aver,len(list(G)),G.size()

def read_net(file):
    G = nx.read_gml(file)
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
        origin.append(list(G).index(i)+1)
        destination.append(list(G).index(j)+1)
        if G.degree(i)>G.degree(j):
            capacity.append(1)
        else:
            capacity.append(1)
    result={
        'origin': origin+destination,
        'destination': destination+origin,
        'capacity': capacity+capacity,
    }
    return result

def creat_net2(file,path):
    G = nx.read_gml(file)
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
        origin.append(list(G).index(i)+1)
        destination.append(list(G).index(j)+1)
        if G.degree(i)>G.degree(j):
            capacity.append(int(link_sim(1000,dis*10,G.degree(j))*8/128))
        else:
            capacity.append(int(link_sim(1000,dis*10,G.degree(i))*8/128))
        
        # capacity.append(100)

    dfdata = {
        'origin': origin+destination,
        'destination': destination+origin,
        'capacity': capacity+capacity,
    }
    df = pd.DataFrame(dfdata)  # 创建DataFrame
    node = pd.DataFrame(node_data)
    writer = pd.ExcelWriter(path+"data.xlsx")
    df.to_excel(writer, index=False, sheet_name='node_data')
    node.to_excel(writer, index=False, sheet_name='node')
    writer.close()
    result={
        'origin': origin+destination,
        'destination': destination+origin,
        'capacity': capacity+capacity,
    }
    return result