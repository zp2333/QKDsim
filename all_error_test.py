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
import creat_topology
def pd_toExcel(G, fileName):  # pandas库储存数据到excel
    origin = []
    destination = []
    capacity = []
    node_name = []
    node_type = []
    for (i, j) in G.edges():
        if G.nodes[i]['node_type'] != 'unused' and G.nodes[j]['node_type'] != 'unused':
            origin.append(i)
            destination.append(j)
            capacity.append(G[i][j]['capacity'])
    for i in G.nodes:
        if G.nodes[i]['node_type'] != 'unused':
            node_name.append(i)
            node_type.append(G.nodes[i]['node_type'])
    dfData = {  # 用字典设置DataFrame所需数据
        'origin': origin,
        'destination': destination,
        'capacity': capacity
    }
    nodedata = {  # 用字典设置DataFrame所需数据
        'node_name': node_name,
        'node_type': node_type
    }
    df = pd.DataFrame(dfData)  # 创建DataFrame
    s_d = pd.DataFrame(nodedata)
    writer = pd.ExcelWriter(fileName)
    # 存表，去除原始索引列（0,1,2...）
    df.to_excel(writer, index=False, sheet_name='node_data')
    s_d.to_excel(writer, index=False, sheet_name='s_d')
    writer.close()


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
        edge_labels[edge] = G[edge[0]][edge[1]]['capacity']

    nx.draw_networkx_labels(G, pos, labels=node_labels,font_size=4)

    # no edge_labels parameter, default is showing all attributes of edges
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,font_size=3)
    plt.savefig(fileName,dpi=1000,bbox_inches='tight')
    #plt.show() #实时展示拓扑
    plt.clf()


def delay_constraint(G, request_inf, length_path):  # 根据延时约束确定路径长度约束   直接处理版
    for i in G.nodes():
        G.nodes[i]['node_type'] = 'unused'
    for i in range(len(request_inf['request_data'])):
        # cutoff传输经过的路径
        for path in nx.all_simple_paths(G, source=request_inf['request_source'][i], target=request_inf['request_destination'][i], cutoff=length_path):
            for i in path:
                if G.nodes[i]['node_type'] == 'unused':
                    G.nodes[i]['node_type'] = 'used'


def Min_consume(G, model,request_inf,length_path):
    '''最小密钥资源消耗量求解'''
    # solver =GUROBI_CMD()#调用gurobi求解
    # 建立变量
    # shipping_amount = pulp.LpVariable.dicts(
    #     'edge', G.edges(), lowBound=0, cat='Integer')

    # # 目标函数
    # model += lpSum(shipping_amount[(origin, destination)]
    #                for(origin, destination) in G.edges())
    # for name, data in G.nodes(data=True):
    #     model += lpSum([shipping_amount[(origin, name)]
    #                     for origin, _ in G.in_edges(name)]) >= G.nodes[name]['node_in']
    #     model += lpSum([shipping_amount[(name, destination)]
    #                     for _, destination in G.out_edges(name)]) >= G.nodes[name]['node_out']
    #     model += (-lpSum([shipping_amount[(origin, name)] for origin, _ in G.in_edges(name)])+lpSum(
    #         [shipping_amount[(name, destination)] for _, destination in G.out_edges(name)])) == (G.nodes[name]['node_out']-G.nodes[name]['node_in'])


    # # 满足密钥池约束
    # for (origin, destination, data) in G.edges(data=True):
    #     if G.nodes[origin]['node_type'] == 'used' and G.nodes[destination]['node_type'] == 'used':
    #         model += lpSum(shipping_amount[(origin, destination)] +
    #                        shipping_amount[(destination, origin)]) <= data['capacity']
    #     else:
    #         model += lpSum(shipping_amount[(origin, destination)] +
    #                        shipping_amount[(destination, origin)]) == 0

    shipping_amount=[]
    path_data=[]
    for i in range(len(request_inf['request_data'])):
        shipping_amount.append(pulp.LpVariable.dicts('edge'+str(i), G.edges(), lowBound=0, cat='Integer'))

    # 目标函数
    model += lpSum(lpSum([shipping_amount[i][(origin, destination)] for(origin, destination) in G.edges()]for i in range(len(request_inf['request_data']))))
    for i in range(len(request_inf['request_data'])):
        all_path=nx.all_simple_paths(G, source=request_inf['request_source'][i], target=request_inf['request_destination'][i], cutoff=length_path)
        number=0
        path_data.append([])
        path_list=[]
        # if request_inf['request_data'][i]==0:
        #     for (origin, destination, data) in G.edges(data=True):
        #          model+=shipping_amount[i][(origin, destination)]==0
        #     continue
        for path in all_path:
            path_data[i].append(pulp.LpVariable('path'+str(i)+'.'+str(number), lowBound=0, cat='Integer'))
            path_list.append(path)
            number+=1
        model+=lpSum(path_data[i][j] for j in range(len(path_data[i])))== request_inf['request_data'][i]

        for (origin, destination, data) in G.edges(data=True):
            re=fp(origin,destination,path_list)
            if re!=[]:
                model+=shipping_amount[i][(origin, destination)]==lpSum(path_data[i][j] for j in re)
            else:
                model+=shipping_amount[i][(origin, destination)]==0
    
    
    # 满足密钥池约束
    for (origin, destination, data) in G.edges(data=True):
            model += lpSum([shipping_amount[i][(origin, destination)] for i in range(len(request_inf['request_data']))])+lpSum([shipping_amount[i][(
                destination, origin)]for i in range(len(request_inf['request_data']))]) <= data['capacity']




    # 求解
    # start_time = time.perf_counter()
    #options=[('TimeLimit','1')]
    model.solve(GUROBI_CMD(options=[('TimeLimit','15')],msg=0))  # 调用gurobi求解 加入solver
    # print("用时：", time.perf_counter()-start_time, " s")

    # 组织计算结果
    rsl = []
    if LpStatus[model.status] == "Optimal":
        for origin, destination in G.edges():
            val=0
            for i in range(len(request_inf['request_data'])):
                val+=value(shipping_amount[i][(origin, destination)])
            rsl.append({"origin": origin, "destination": destination, "val": val})
    rsl_df = pd.DataFrame(rsl)
    return rsl_df

def fp(origin,destination,all_path):
    result=[]
    j=0
    for path in all_path:
        for i in range(len(path)-1):
            if path[i]==origin and path[i+1]==destination:
                result.append(j)
        j+=1
    return result


def Min_alpha(G, model,request_inf,length_path):
    '''最小化网络最大负载率求解'''
    # 建立变量
    shipping_amount=[]
    node_used=[]
    path_data=[]
    for i in range(len(request_inf['request_data'])):
        shipping_amount.append(pulp.LpVariable.dicts('edge'+str(i), G.edges(), lowBound=0, cat='Integer'))
    alpha = pulp.LpVariable('负载率', lowBound=0, upBound=1.0,cat='Continuous')  # 定义 alpha

    # 目标函数
    model += alpha
    for i in range(len(request_inf['request_data'])):
        all_path=nx.all_simple_paths(G, source=request_inf['request_source'][i], target=request_inf['request_destination'][i], cutoff=length_path)
        number=0
        path_list=[]
        path_data.append([])
        # if request_inf['request_data'][i]==0:
        #     for (origin, destination, data) in G.edges(data=True):
        #          model+=shipping_amount[i][(origin, destination)]==0
        #     continue
        for path in all_path:
            path_data[i].append(pulp.LpVariable('path'+str(i)+'.'+str(number), lowBound=0, cat='Integer'))
            path_list.append(path)
            number+=1
        model+=lpSum(path_data[i][j] for j in range(len(path_data[i])))== request_inf['request_data'][i]

        for (origin, destination, data) in G.edges(data=True):
            re=fp(origin,destination,path_list)
            if re!=[]:
                model+=shipping_amount[i][(origin, destination)]==lpSum(path_data[i][j] for j in re)
                node_used.append((origin,destination))
            else:
                model+=shipping_amount[i][(origin, destination)]==0
    
    
    # 满足密钥池约束
    for (origin, destination, data) in G.edges(data=True):
        model += lpSum([shipping_amount[i][(origin, destination)] for i in range(len(request_inf['request_data']))])+lpSum([shipping_amount[i][(
                destination, origin)]for i in range(len(request_inf['request_data']))]) <= data['capacity']
        if (origin, destination) in node_used:
            model += lpSum([shipping_amount[i][(origin, destination)] for i in range(len(request_inf['request_data']))])+lpSum([shipping_amount[i][(
                destination, origin)]for i in range(len(request_inf['request_data']))]) <= data['capacity']*alpha




    # 求解
    start_time = time.perf_counter()
    model.solve(GUROBI_CMD(options=[('TimeLimit','15')],msg=0))
    print("用时：", time.perf_counter()-start_time, " s")

    # 组织计算结果
    rsl = []
    if LpStatus[model.status] == "Optimal":
        for origin, destination in G.edges():
            val=0
            for i in range(len(request_inf['request_data'])):
                val+=value(shipping_amount[i][(origin, destination)])
            rsl.append({"origin": origin, "destination": destination, "val": val})
    rsl_df = pd.DataFrame(rsl)
    return rsl_df

def Min_alpha_iteration(G, model,request_inf,length_path,bottleneck_link):
    '''最小化网络最大负载率求解，迭代求解'''
    # 建立变量
    shipping_amount=[]
    node_used=[]
    path_data=[]
    bottleneck_link_temp=[]
    for i in range(len(request_inf['request_data'])):
        shipping_amount.append(pulp.LpVariable.dicts('edge'+str(i), G.edges(), lowBound=0, cat='Integer'))
    alpha = pulp.LpVariable('负载率', lowBound=0, upBound=1.0,cat='Continuous')  # 定义 alpha

    # 目标函数
    model += alpha
    for i in range(len(request_inf['request_data'])):
        all_path=nx.all_simple_paths(G, source=request_inf['request_source'][i], target=request_inf['request_destination'][i], cutoff=length_path)
        number=0
        path_list=[]
        path_data.append([])
        # if request_inf['request_data'][i]==0:
        #     for (origin, destination, data) in G.edges(data=True):
        #          model+=shipping_amount[i][(origin, destination)]==0
        #     continue
        for path in all_path:
            path_data[i].append(pulp.LpVariable('path'+str(i)+'.'+str(number), lowBound=0, cat='Integer'))
            path_list.append(path)
            number+=1
        model+=lpSum(path_data[i][j] for j in range(len(path_data[i])))== request_inf['request_data'][i]

        for (origin, destination, data) in G.edges(data=True):
            re=fp(origin,destination,path_list)
            if re!=[]:
                model+=shipping_amount[i][(origin, destination)]==lpSum(path_data[i][j] for j in re)
                node_used.append((origin,destination))
            else:
                model+=shipping_amount[i][(origin, destination)]==0
    
    
    # 满足密钥池约束
    for (origin, destination, data) in G.edges(data=True):
        model += lpSum([shipping_amount[i][(origin, destination)] for i in range(len(request_inf['request_data']))])+lpSum([shipping_amount[i][(
                destination, origin)]for i in range(len(request_inf['request_data']))]) <= data['capacity']
        if (origin, destination) in node_used:
            if(origin,destination) in bottleneck_link:
                model += lpSum([shipping_amount[i][(origin, destination)] for i in range(len(request_inf['request_data']))])+lpSum([shipping_amount[i][(
                destination, origin)]for i in range(len(request_inf['request_data']))]) == data['capacity']
            else:
                model += lpSum([shipping_amount[i][(origin, destination)] for i in range(len(request_inf['request_data']))])+lpSum([shipping_amount[i][(
                destination, origin)]for i in range(len(request_inf['request_data']))]) <= data['capacity']*alpha


    # 求解
    start_time = time.perf_counter()
    model.solve(GUROBI_CMD(options=[('TimeLimit','15'),('MipGap','0.003')],msg=0))
    print("用时：", time.perf_counter()-start_time, " s")
    
    # 组织计算结果
    rsl = []
    sd_temp=[]
    val_temp=[]
    if LpStatus[model.status] == "Optimal":
        temp_G=copy.deepcopy(G)
        for (origin, destination, data) in G.edges(data=True):
            val=0
            for i in range(len(request_inf['request_data'])):
                val+=value(shipping_amount[i][(origin, destination)])
            # if val==data['capacity']and ((origin, destination) not in bottleneck_link):
            #     bottleneck_link_temp.append((origin, destination))
            # if ((origin,destination) not in sd_temp) and ((destination,origin) not in sd_temp):
            #     sd_temp.append((origin,destination))
            #     val_temp.append(val)
            # elif (destination,origin) in sd_temp:
            #     val_temp[sd_temp.index((destination,origin))]+=val
            rsl.append({"origin": origin, "destination": destination, "val": val})
        # for (origin, destination, data) in G.edges(data=True):
        #     if (origin, destination) in sd_temp:
        #         if val_temp[sd_temp.index((origin, destination))]==data['capacity'] and ((origin, destination) not in bottleneck_link):
        #             bottleneck_link_temp.append((origin, destination))
    rsl_df = pd.DataFrame(rsl)
    # print(rsl_df)
    if len(rsl)!=0:
        rsl_df.val=-rsl_df.val
        update_key(temp_G,rsl_df)
        rsl_df.val=-rsl_df.val
        all_bottleneck_link=[]
        min_degree=1000
        for (i,j) in temp_G.edges():
            if temp_G[i][j]['capacity']==0 and (i,j) not in bottleneck_link:
                all_bottleneck_link.append((i,j))
                all_bottleneck_link.append((j,i))
                if min_degree>G.degree(i) or min_degree>G.degree(j):
                    min_degree=min(G.degree(i),G.degree(j))
                    min_degree_link=(i,j)
        if len(all_bottleneck_link)!=0:
            (i,j)=min_degree_link
            bottleneck_link_temp.append((i,j))
            bottleneck_link_temp.append((j,i))
            for sd in bottleneck_link_temp:
                bottleneck_link.append(sd)

    # for (origin, destination) in bottleneck_link_temp:#检查是否是真瓶颈
    #     all_path=nx.all_simple_paths(G, source=origin, target=destination, cutoff=length_path)
    #     flag=0
    #     for path in all_path:
    #         for i in range(len(path)-1):
    #             if (path[i],path[i-1]) not in bottleneck_link_temp and (path[i-1],path[i]) not in bottleneck_link_temp:
    #                 flag=1
    #     if flag==0:
    #         bottleneck_link.append((origin, destination))
    return rsl_df


def Max_alpha(G, model,request_inf,length_path):

    '''最大化网络最小负载率求解'''  
    shipping_amount=[]
    node_used=[]
    path_data=[]
    for i in range(len(request_inf['request_data'])):
        shipping_amount.append(pulp.LpVariable.dicts('edge'+str(i), G.edges(), lowBound=0, cat='Integer'))
    alpha = pulp.LpVariable('负载率', lowBound=0, upBound=1.0,cat='Continuous')  # 定义 alpha

    # 目标函数
    model += alpha
    for i in range(len(request_inf['request_data'])):
        all_path=nx.all_simple_paths(G, source=request_inf['request_source'][i], target=request_inf['request_destination'][i], cutoff=length_path)
        number=0
        path_data.append([])
        path_list=[]
        # if request_inf['request_data'][i]==0:
        #     for (origin, destination, data) in G.edges(data=True):
        #          model+=shipping_amount[i][(origin, destination)]==0
        #     continue
        for path in all_path:
            path_data[i].append(pulp.LpVariable('path'+str(i)+'.'+str(number), lowBound=0, cat='Integer'))
            path_list.append(path)
            number+=1
        model+=lpSum(path_data[i][j] for j in range(len(path_data[i])))== request_inf['request_data'][i]

        for (origin, destination, data) in G.edges(data=True):
            re=fp(origin,destination,path_list)
            if re!=[]:
                model+=shipping_amount[i][(origin, destination)]==lpSum(path_data[i][j] for j in re)
                node_used.append((origin,destination))
            else:
                model+=shipping_amount[i][(origin, destination)]==0
    
    
    # 满足密钥池约束
    for (origin, destination, data) in G.edges(data=True):
            model += lpSum([shipping_amount[i][(origin, destination)] for i in range(len(request_inf['request_data']))])+lpSum([shipping_amount[i][(
                destination, origin)]for i in range(len(request_inf['request_data']))]) <= data['capacity']
            if (origin, destination) in node_used:
                model += lpSum([shipping_amount[i][(origin, destination)] for i in range(len(request_inf['request_data']))])+lpSum([shipping_amount[i][(
                destination, origin)]for i in range(len(request_inf['request_data']))]) >= data['capacity']*alpha

    # 求解
    # start_time = time.perf_counter()
    model.solve(GUROBI_CMD(options=[('TimeLimit','45')],msg=0))
    # print("用时：", time.perf_counter()-start_time, " s")

    # 组织计算结果
    rsl = []
    if LpStatus[model.status] == "Optimal":
        for origin, destination in G.edges():
            val=0
            for i in range(len(request_inf['request_data'])):
                val+=value(shipping_amount[i][(origin, destination)])
            rsl.append({"origin": origin, "destination": destination, "val": val})
    rsl_df = pd.DataFrame(rsl)
    # print(rsl_df[rsl_df.val>0])
    return rsl_df


def read_request(fileName):
    request_data = []
    request_source = []
    request_destination = []
    fpath = os.path.join(fileName)
    request = pd.read_excel(fpath, sheet_name="request")
    for i, row in request.iterrows():
        request_data.append(row.data)
        request_source.append(row.source)
        request_destination.append(row.destination)

    request_inf = {
        'request_data': request_data,
        'request_source': request_source,
        'request_destination': request_destination
    }
    return request_inf


def update_key(G, update_data):
    for i in range(len(update_data['origin'])):
        if update_data['val'][i]<0:
            G[update_data['origin'][i]][update_data['destination'][i]]['capacity'] += update_data['val'][i]
            G[update_data['destination'][i]][update_data['origin'][i]]['capacity'] += update_data['val'][i]
        if update_data['val'][i]>0:
            G[update_data['origin'][i]][update_data['destination'][i]]['capacity'] += update_data['val'][i]
            if G[update_data['origin'][i]][update_data['destination'][i]]['capacity']>3*update_data['val'][i]:
                    G[update_data['origin'][i]][update_data['destination'][i]]['capacity']=3*update_data['val'][i]
            #G[update_data['destination'][i]][update_data['origin'][i]]['capacity'] += update_data['val'][i]


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
            request_num[i-1]=0
            # print(i)
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

def creat_request2(source,destination,num,data):
    request_data = []
    request_source = []
    request_destination = []
    for i in range(num):
        request_source.append(source)
        request_destination.append(destination)
        request_data.append(data)
    request_inf = {
        'request_data': request_data,
        'request_source': request_source,
        'request_destination': request_destination
    }
    return request_inf


def improve_route(G, length_path, request_inf, supplement_key,result):
    """非迭代式,带max"""
    G_temp=copy.deepcopy(G)
    # for i in range(len(request_inf['request_data'])):
    #     G.nodes[request_inf['request_source'][i]
    #             ]['node_out'] += request_inf['request_data'][i]
    #     G.nodes[request_inf['request_destination'][i]
    #             ]['node_in'] += request_inf['request_data'][i]
    # request_abandon = []

    # delay_constraint(G, request_inf, length_path)  # 延时约束更新节点列表
    request_abandon_num = 0
    # model = LpProblem("QKD_Model", LpMinimize)
    model = LpProblem("QKD_Model", LpMinimize)
    rsl_df = Min_alpha(G, model,request_inf,length_path)#####################
    while LpStatus[model.status] != "Optimal" and len(request_inf['request_data']) != 0:
        # print("ERROR,this result is not Optimal,please check again")
        random.seed(123)
        abandon_num = random.choice(range(len(request_inf['request_data'])))
        # G.nodes[request_inf['request_source'][abandon_num]
        #         ]['node_out'] -= request_inf['request_data'][abandon_num]
        # G.nodes[request_inf['request_destination'][abandon_num]
        #         ]['node_in'] -= request_inf['request_data'][abandon_num]

        # print(G.nodes('node_in'))
        # print(G.nodes('node_out'))
        # request_abandon.append(request_inf['request_data'][abandon_num])
        request_abandon_num = request_abandon_num+1
        request_inf['request_data'].pop(abandon_num)
        request_inf['request_source'].pop(abandon_num)
        request_inf['request_destination'].pop(abandon_num)
        model = LpProblem("QKD_Model", LpMinimize)
        if len(request_inf['request_data']) != 0:
            rsl_df = Min_alpha(G, model,request_inf,length_path)
    result['abandon2'].append(request_abandon_num)
    print('improve route abandon request number is:', request_abandon_num)
    if len(request_inf['request_data']) != 0:
        if value(model.objective) == 1:
            rsl_df_temp=copy.deepcopy(rsl_df)
            model = LpProblem("QKD_Model", LpMaximize)
            rsl_df = Max_alpha(G, model,request_inf,length_path)
            if LpStatus[model.status] != "Optimal":
                rsl_df=rsl_df_temp
            print("Max_alpha,the result is ", LpStatus[model.status])
            print("consume is ", sum(rsl_df.val))
            
        else:
            print("Min_alpha,the result is ", LpStatus[model.status])
            # print(rsl_df[rsl_df.val > 0])
            print("consume is ", sum(rsl_df.val))
            # print("alpha is ", value(model.objective))
            #result['alpha1'].append(value(model.objective))
        # result['consume1'].append(sum(rsl_df.val))
        #draw_topology(G, 'topology1.jpg')
        # 更新边的密钥容量
        rsl_df.val = -rsl_df.val
        update_key(G, rsl_df)
        alpha=[]
        link_use_num=0
        max_alpha=0
        for (i,j) in G.edges():
            x=(G_temp[i][j]['capacity']-G[i][j]['capacity'])/G_temp[i][j]['capacity']
            if x!=0:
                alpha.append(x)
            if x==1:
                link_use_num+=1
            elif x>max_alpha:
                max_alpha=x
        print("alpha is ",max(alpha))
        print("beta is",max_alpha)
        result['alpha2'].append(max(alpha))
        result['beta2'].append(max_alpha)
        result['link2'].append(link_use_num)
        print("link is",link_use_num)

    else :
        # result['consume1'].append(0)
        result['alpha2'].append(0)
        result['beta2'].append(0)
        result['link2'].append(0)
        #draw_topology(G, 'topology2.jpg')
    
    test=[]
    for (i,j) in G.edges():
        test.append(G[i][j]['capacity'])
    result['min2'].append(min(test))
    result['max2'].append(max(test))
    result['var2'].append(np.std(test,ddof=1))
    update_key(G, supplement_key)


    for i in G.nodes():
        G.nodes[i]['node_in'] = 0
        G.nodes[i]['node_out'] = 0
    #pd_toExcel(G, 'test.xlsx')
    #draw_topology(G, 'topology3.jpg')

def improve_route_iteration(G, length_path, request_inf, supplement_key,result):
    """迭代式"""
    G_temp=copy.deepcopy(G)
    request_abandon_num = 0
    model = LpProblem("QKD_Model", LpMinimize)
    bottleneck_link=[]
    start_time1 = time.perf_counter()
    rsl_df = Min_alpha_iteration(G, model,request_inf,length_path,bottleneck_link)
    time1=time.perf_counter()-start_time1
    while LpStatus[model.status] != "Optimal" and len(request_inf['request_data']) != 0:
        random.seed(123)
        abandon_num = random.choice(range(len(request_inf['request_data'])))
        request_abandon_num = request_abandon_num+1
        request_inf['request_data'].pop(abandon_num)
        request_inf['request_source'].pop(abandon_num)
        request_inf['request_destination'].pop(abandon_num)
        model = LpProblem("QKD_Model", LpMinimize)
        if len(request_inf['request_data']) != 0:
            bottleneck_link=[]
            rsl_df = Min_alpha_iteration(G, model,request_inf,length_path,bottleneck_link)
    result['abandon1'].append(request_abandon_num)
    print('improve route iteration abandon request number is:', request_abandon_num)
    start_time2=time.perf_counter()
    if len(request_inf['request_data']) != 0:
        interation_num=0
        while value(model.objective)==1 and interation_num <=10:
            interation_num+=1
            # if interation_num <=10:
            print("interating,the number is ", interation_num)
            print("bottleneck links is ",bottleneck_link)
            model = LpProblem("QKD_Model", LpMinimize)
            rsl_df = Min_alpha_iteration(G, model,request_inf,length_path,bottleneck_link)
            # print(rsl_df[rsl_df.val>0])    
            # print(value(model.objective))
            # else:
            #     break
        print("Min_alpha,the result is ", LpStatus[model.status])
        print("consume is ", sum(rsl_df.val))
            # print("alpha is ", value(model.objective))
            #result['alpha1'].append(value(model.objective))
        # result['consume1'].append(sum(rsl_df.val))
        #draw_topology(G, 'topology1.jpg')
        # 更新边的密钥容量
        rsl_df.val = -rsl_df.val
        update_key(G, rsl_df)
        alpha=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        alpha_temp=[]
        link_use_num=0
        for (i,j) in G.edges():
            x=(G_temp[i][j]['capacity']-G[i][j]['capacity'])/G_temp[i][j]['capacity']
            if x!=0:
                alpha_temp.append(x)
            alpha.append(x)
            if x==1:
                link_use_num+=1          
        alpha.sort(reverse=True)
        # if len(alpha)>=2
        print("alpha is ",alpha[0])
        print("beta is ",alpha[2])
        print("gamma is ",alpha[4])
        print("delta is ",alpha[6])
        print("epsilon is",(alpha[6]+alpha[8]+alpha[4])/3)
        result['alpha1'].append(alpha[0])
        result['beta1'].append(alpha[2])
        result['gamma1'].append(alpha[4])
        result['delta1'].append(alpha[6])
        result['epsilon1'].append((alpha[6]+alpha[8]+alpha[4])/3)
        result['link1'].append(link_use_num)
        print("link is",link_use_num)

    else :
        # result['consume1'].append(0)
        result['alpha1'].append(0)
        result['beta1'].append(0)
        result['link1'].append(0)
        result['gamma1'].append(0)
        result['delta1'].append(0)
        result['epsilon1'].append(0)
        alpha_temp=[0]
        #draw_topology(G, 'topology2.jpg')
    time1+=(time.perf_counter()-start_time2)
    test=[]
    for (i,j) in G.edges():
        test.append(G[i][j]['capacity'])
    result['min1'].append(min(test))
    result['max1'].append(max(test))
    result['var1'].append(np.std(alpha_temp,ddof=1))
    result['time1'].append(time1)
    update_key(G, supplement_key)


    for i in G.nodes():
        G.nodes[i]['node_in'] = 0
        G.nodes[i]['node_out'] = 0
    #pd_toExcel(G, 'test.xlsx')
    #draw_topology(G, 'topology3.jpg')
def improve_route_without_max(G, length_path, request_inf, supplement_key,result):
    """非迭代式,无最大优化"""
    G_temp=copy.deepcopy(G)
    # for i in range(len(request_inf['request_data'])):
    #     G.nodes[request_inf['request_source'][i]
    #             ]['node_out'] += request_inf['request_data'][i]
    #     G.nodes[request_inf['request_destination'][i]
    #             ]['node_in'] += request_inf['request_data'][i]
    # request_abandon = []

    # delay_constraint(G, request_inf, length_path)  # 延时约束更新节点列表
    request_abandon_num = 0
    # model = LpProblem("QKD_Model", LpMinimize)
    model = LpProblem("QKD_Model", LpMinimize)
    start_time1 = time.perf_counter()
    rsl_df = Min_alpha(G, model,request_inf,length_path)#####################
    time3=time.perf_counter()-start_time1
    while LpStatus[model.status] != "Optimal" and len(request_inf['request_data']) != 0:
        # print("ERROR,this result is not Optimal,please check again")
        random.seed(123)
        abandon_num = random.choice(range(len(request_inf['request_data'])))
        # G.nodes[request_inf['request_source'][abandon_num]
        #         ]['node_out'] -= request_inf['request_data'][abandon_num]
        # G.nodes[request_inf['request_destination'][abandon_num]
        #         ]['node_in'] -= request_inf['request_data'][abandon_num]

        # print(G.nodes('node_in'))
        # print(G.nodes('node_out'))
        # request_abandon.append(request_inf['request_data'][abandon_num])
        request_abandon_num = request_abandon_num+1
        request_inf['request_data'].pop(abandon_num)
        request_inf['request_source'].pop(abandon_num)
        request_inf['request_destination'].pop(abandon_num)
        model = LpProblem("QKD_Model", LpMinimize)
        if len(request_inf['request_data']) != 0:
            rsl_df = Min_alpha(G, model,request_inf,length_path)
    result['abandon3'].append(request_abandon_num)
    print('improve route without max abandon request number is:', request_abandon_num)
    start_time2 = time.perf_counter()
    if len(request_inf['request_data']) != 0:
        print("Min_alpha,the result is ", LpStatus[model.status])
        # print(rsl_df[rsl_df.val > 0])
        print("consume is ", sum(rsl_df.val))
        # print("alpha is ", value(model.objective))
        #result['alpha1'].append(value(model.objective))
        # result['consume1'].append(sum(rsl_df.val))
        #draw_topology(G, 'topology1.jpg')
        # 更新边的密钥容量
        rsl_df.val = -rsl_df.val
        update_key(G, rsl_df)
        alpha=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        alpha_temp=[]
        link_use_num=0
        for (i,j) in G.edges():
            x=(G_temp[i][j]['capacity']-G[i][j]['capacity'])/G_temp[i][j]['capacity']
            if x!=0:
                alpha_temp.append(x)
            alpha.append(x)
            if x==1:
                link_use_num+=1          
        alpha.sort(reverse=True)
        print("alpha is ",alpha[0])
        print("beta is ",alpha[2])
        print("gamma is ",alpha[4])
        print("delta is ",alpha[6])
        print("epsilon is",(alpha[6]+alpha[8]+alpha[4])/3)
        result['alpha3'].append(alpha[0])
        result['beta3'].append(alpha[2])
        result['gamma3'].append(alpha[4])
        result['delta3'].append(alpha[6])
        result['epsilon3'].append((alpha[6]+alpha[8]+alpha[4])/3)
        result['link3'].append(link_use_num)
        print("link is",link_use_num)

    else :
        # result['consume1'].append(0)
        result['alpha3'].append(0)
        result['beta3'].append(0)
        result['link3'].append(0)
        result['gamma3'].append(0)
        result['delta3'].append(0)
        result['epsilon3'].append(0)
        alpha_temp=[0]
        #draw_topology(G, 'topology2.jpg')
    time3 += time.perf_counter()-start_time2
    test=[]
    for (i,j) in G.edges():
        test.append(G[i][j]['capacity'])
    result['min3'].append(min(test))
    result['max3'].append(max(test))
    result['var3'].append(np.std(alpha_temp,ddof=1))
    result['time3'].append(time3)
    update_key(G, supplement_key)


    for i in G.nodes():
        G.nodes[i]['node_in'] = 0
        G.nodes[i]['node_out'] = 0

def takeSecond(elem):
    return len(elem)
def order(elem):
    path_capacity=[]
    for j in range(len(elem)-1):
        path_capacity.append(G[elem[j]][elem[j+1]]['capacity'])
    path_min=min(path_capacity)
    return path_min
def path_sort(G,path_list):
    length=[]
    all=[]
    final=[]
    for path in path_list:
        if len(path) not in length:
            length.append(len(path))
            all.append([])
        all[length.index(len(path))].append(path)
    for i in range(len(all)):
        all[i].sort(key=order)
        if i==0:
            final=all[0]
        else:
            final=final+all[i]
    return final
def compare_route1(G, length_path, request_inf0,request_inf, supplement_key,result):
    request_abandon_num  = len(request_inf0['request_data'])-len(request_inf['request_data'])
    G_temp=copy.deepcopy(G)
    G_change=copy.deepcopy(G)
    start_time1=time.perf_counter()
    for i in range(len(request_inf['request_data'])):
        G_copy=copy.deepcopy(G_change)
        flag=1
        path_list=list(nx.all_simple_paths(G_change,source=request_inf['request_source'][i],target=request_inf['request_destination'][i],cutoff=length_path))
        path_list.sort(key=takeSecond)
        path_test=path_sort(G_change,path_list)#按照最小容量大小排序
        for path in path_test:
            path_capacity=[]
            for j in range(len(path)-1):
                path_capacity.append(G_change[path[j]][path[j+1]]['capacity'])
            path_min=min(path_capacity)
            if path_min>=request_inf['request_data'][i]:
                flag=0
                for j in range(len(path)-1):
                    G_change[path[j]][path[j+1]]['capacity']-=request_inf['request_data'][i]
                    G_change[path[j+1]][path[j]]['capacity']-=request_inf['request_data'][i]
                break
            else:
                 request_inf['request_data'][i]-=path_min
                 for j in range(len(path)-1):
                    G_change[path[j]][path[j+1]]['capacity']-=path_min
                    G_change[path[j+1]][path[j]]['capacity']-=path_min
        if flag==1:
            request_abandon_num+=1
            G_change=copy.deepcopy(G_copy)
    result['abandon3'].append(request_abandon_num)
    print('compare route1 abandon request number is:', request_abandon_num)
    if request_abandon_num<len(request_inf['request_data']):
        alpha_temp=[]
        alpha=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        link_use_num=0
        consume=0
        for (i,j) in G_change.edges():
            if G_temp[i][j]['capacity']==0:
                x=0
            else:
                x=(G_temp[i][j]['capacity']-G_change[i][j]['capacity'])/G_temp[i][j]['capacity']
                consume+=G_temp[i][j]['capacity']-G_change[i][j]['capacity']
            if x!=0:
                alpha_temp.append(x)
            alpha.append(x)
            if x==1:
                link_use_num+=1
        alpha.sort(reverse=True)
        print("alpha is ",alpha[0])
        print("beta is ",alpha[2])
        print("gamma is ",alpha[4])
        print("delta is ",alpha[6])
        print("epsilon is",(alpha[6]+alpha[8]+alpha[4])/3)
        result['alpha3'].append(alpha[0])
        result['beta3'].append(alpha[2])
        result['gamma3'].append(alpha[4])
        result['delta3'].append(alpha[6])
        result['epsilon3'].append((alpha[6]+alpha[8]+alpha[4])/3)
        result['link3'].append(link_use_num)
        print("consume is ",consume/2)
        print("link is",link_use_num)

    else :
        # result['consume3'].append(0)
        result['alpha3'].append(0)
        result['beta3'].append(0)
        result['link3'].append(0)
        result['gamma3'].append(0)
        result['delta3'].append(0)
        result['epsilon3'].append(0)
        alpha_temp=[0]
    time1 =time.perf_counter()-start_time1
    test=[]
    for (i,j) in G_change.edges():
        test.append(G_change[i][j]['capacity'])
    result['min3'].append(min(test))
    result['max3'].append(max(test))
    result['var3'].append(np.std(alpha_temp,ddof=1))
    result['time3'].append(time1)
    update_key(G_change, supplement_key)
    for (i,j) in G_change.edges():
        G[i][j]['capacity']=G_change[i][j]['capacity']

def compare_route2(G, length_path, request_inf, supplement_key,result):
    # for i in range(len(request_inf['request_data'])):
    #     G.nodes[request_inf['request_source'][i]
    #             ]['node_out'] += request_inf['request_data'][i]
    #     G.nodes[request_inf['request_destination'][i]
    #             ]['node_in'] += request_inf['request_data'][i]
    # request_abandon = []

    #delay_constraint(G, request_inf, length_path)  # 延时约束更新节点列表
    request_abandon_num = 0
    model = LpProblem("QKD_Model", LpMinimize)
    start_time1 = time.perf_counter()
    rsl_df = Min_consume(G, model,request_inf,length_path)
    time2 = time.perf_counter()-start_time1
    while LpStatus[model.status] != "Optimal" and len(request_inf['request_data']) != 0:
        # print("ERROR,this result is not Optimal,please check again")
        random.seed(123)
        abandon_num = random.choice(range(len(request_inf['request_data'])))
        # G.nodes[request_inf['request_source'][abandon_num]
        #         ]['node_out'] -= request_inf['request_data'][abandon_num]
        # G.nodes[request_inf['request_destination'][abandon_num]
        #         ]['node_in'] -= request_inf['request_data'][abandon_num]

        # print(G.nodes('node_in'))
        # print(G.nodes('node_out'))
        # request_abandon.append(request_inf['request_data'][abandon_num])
        request_abandon_num = request_abandon_num+1
        request_inf['request_data'].pop(abandon_num)
        request_inf['request_source'].pop(abandon_num)
        request_inf['request_destination'].pop(abandon_num)
        model = LpProblem("QKD_Model", LpMinimize)
        if len(request_inf['request_data']) != 0:
            rsl_df = Min_consume(G, model,request_inf,length_path)
    result['abandon2'].append(request_abandon_num)
    print('compare route2 abandon request number is:', request_abandon_num)
    start_time2 = time.perf_counter()
    if len(request_inf['request_data']) != 0:
        print("Min_consume,the result is ", LpStatus[model.status])
        # print(rsl_df[rsl_df.val > 0])
        print("consume is ", sum(rsl_df.val))
        # result['consume2'].append(sum(rsl_df.val))
        #draw_topology(G, 'topology1.jpg')
        # 更新边的密钥容量
        rsl_df.val = -rsl_df.val
        G_temp=copy.deepcopy(G)
        update_key(G, rsl_df)
        alpha=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        alpha_temp=[]
        link_use_num=0
        for (i,j) in G.edges():
            if G_temp[i][j]['capacity']==0:
                x=0
            else:
                x=(G_temp[i][j]['capacity']-G[i][j]['capacity'])/G_temp[i][j]['capacity']
            if x!=0:
                alpha_temp.append(x)
            alpha.append(x)
            if x==1:
                link_use_num+=1        
        alpha.sort(reverse=True)
        print("alpha is ",alpha[0])
        print("beta is ",alpha[2])
        print("gamma is ",alpha[4])
        print("delta is ",alpha[6])
        print("epsilon is",(alpha[6]+alpha[8]+alpha[4])/3)
        result['alpha2'].append(alpha[0])
        result['beta2'].append(alpha[2])
        result['gamma2'].append(alpha[4])
        result['delta2'].append(alpha[6])
        result['epsilon2'].append((alpha[6]+alpha[8]+alpha[4])/3)
        result['link2'].append(link_use_num)
        print("link is",link_use_num)

    else :
        # result['consume2'].append(0)
        result['alpha2'].append(0)
        result['beta2'].append(0)
        result['link2'].append(0)
        result['gamma2'].append(0)
        result['delta2'].append(0)
        result['epsilon2'].append(0)
        alpha_temp=[0]
    time2 += time.perf_counter()-start_time2
    test=[]
    for (i,j) in G.edges():
        test.append(G[i][j]['capacity'])
    result['min2'].append(min(test))
    result['max2'].append(max(test))
    result['var2'].append(np.std(alpha_temp,ddof=1))
    result['time2'].append(time2)
    update_key(G, supplement_key)
    # flow_value, flow_dict = nx.maximum_flow(G, 29, 40)
    # result['out2'].append(flow_value)

    for i in G.nodes():
        G.nodes[i]['node_in'] = 0
        G.nodes[i]['node_out'] = 0
    #pd_toExcel(G, 'test.xlsx')
    #draw_topology(G, 'topology2.jpg')
def link_error(G1,G2,G3,num,max1_node,max2_node):
    for i in range(num):
        random.seed()
        m = random.choice(list(G1))
        while len(list(G1.neighbors(m)))==0 and len(list(G1))>2:
            if m!=max1_node and m!=max2_node:
                G1.remove_node(m)
                G2.remove_node(m)
                G3.remove_node(m)
            random.seed()
            m = random.choice(list(G1))
        if len(list(G1))>2:
            random.seed()
            j=random.choice(list(G1.neighbors(m)))
            G1.remove_edges_from([(m,j),(j,m)])
            G2.remove_edges_from([(m,j),(j,m)])
            G3.remove_edges_from([(m,j),(j,m)])


length_path_list = [5,6,6,6,6]
store_path='result/'
topo_choice=['Chinanet','Geant2012','Dataxchange','Sprint','EliBackbone']#'Chinanet','Geant2012','Dataxchange','Sprint','EliBackbone'
data_choice='error'
max1=[9,3,2,9,15]
max2=[36,5,3,11,10]
# supplement_key = pd.read_excel("data_supplement.xlsx", sheet_name="node_data")
# supplement_key0 = pd.read_excel("data_supplement.xlsx", sheet_name="node_data0")
# fpath = os.path.join("data_copy.xlsx")
# routes = pd.read_excel(fpath, sheet_name="node_data")
# s_d = pd.read_excel(fpath, sheet_name="s_d")

for topo_c in topo_choice:
    #topology_data=creat_topology.creat_net2("archive/"+topo_c+'.gml',store_path+topo_c+'/'+data_choice+'_')
    #writer = pd.ExcelWriter(path+"data.xlsx")
    topology_data = pd.read_excel(store_path+topo_c+'/'+data_choice+'_'+"data.xlsx")
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

    # max_f,maxflow = nx.maximum_flow(G, 21, 8)#最小费用最大流
    # print(max_f)
    degree_list=[]
    node_list=[]
    degree_re_list=[]
    for node in G.nodes():
        node_list.append(node)
        degree_list.append(G.degree(node))
        degree_re_list.append(G.degree(node))
    degree_re_list.sort(reverse=True)
    # max1=[]
    # max2=[]
    # for node in node_list:
    #     if G.degree(node)==degree_re_list[0]:
    #         max1.append(node)
    #     if G.degree(node)==degree_re_list[1]:
    #         max2.append(node)
    # max1_node=random.choice(max1)
    # random.seed()
    # max2_node=random.choice(max2)
    # while max2_node==max1_node:
    #     random.seed()
    #     max2_node=random.choice(max2)
    max1_node=max1[topo_choice.index(topo_c)]
    max2_node=max2[topo_choice.index(topo_c)]
    max_f,maxflow = nx.maximum_flow(G, max1_node, max2_node)#最小费用最大流
    max_f_int=int(0.95*max_f/32)
    request_inf=creat_request2(max1_node,max2_node,max_f_int,32)
    request_inf1 = copy.deepcopy(request_inf)
    request_inf2 = copy.deepcopy(request_inf)
    request_inf3 = copy.deepcopy(request_inf)
    G1 = copy.deepcopy(G)
    G2 = copy.deepcopy(G)
    G3 = copy.deepcopy(G)
    # 建立模型
    #request_inf = read_request('request.xlsx')

    #request_inf={'request_data': [11, 11, 11, 8, 10, 10, 10, 9, 10, 10], 'request_source': [1, 2, 2, 3, 4, 4, 5, 5, 6, 6], 'request_destination': [3, 6, 3, 6, 2, 1, 3, 4, 3, 5]}
    # request_inf1 = copy.deepcopy(request_inf)
    # request_inf2 = copy.deepcopy(request_inf)
    # improve_route(G1, length_path, request_inf1, supplement_key,result)
    # print("----------------------------------------------------------------")
    # compare_route(G2, length_path, request_inf2, supplement_key,result)
    temp={
            "request1":[],
            "abandon1":[],
            "min1":[],
            "max1":[],
            "alpha1":[],
            "beta1":[],
            "var1":[],
            "link1":[],
            "request2":[],
            "abandon2":[],
            "min2":[],
            "max2":[],
            "alpha2":[],
            "beta2":[],
            "var2":[],
            "link2":[],
            "request3":[],
            "abandon3":[],
            "min3":[],
            "max3":[],
            "alpha3":[],
            "beta3":[],
            "var3":[],
            "link3":[],
            "gamma1":[],
            "gamma2":[],
            "gamma3":[],
            "delta1":[],
            "delta2":[],
            "delta3":[],
            "epsilon1":[],
            "epsilon2":[],
            "epsilon3":[],
            "time1":[],
            "time2":[],
            "time3":[],
    }
    final={
            "error_link":[],
            "request1":[],
            "abandon1":[],
            "min1":[],
            "max1":[],
            "alpha1":[],
            "beta1":[],
            "var1":[],
            "link1":[],
            "request2":[],
            "abandon2":[],
            "min2":[],
            "max2":[],
            "alpha2":[],
            "beta2":[],
            "var2":[],
            "link2":[],
            "request3":[],
            "abandon3":[],
            "min3":[],
            "max3":[],
            "alpha3":[],
            "beta3":[],
            "var3":[],
            "link3":[],
            "gamma1":[],
            "gamma2":[],
            "gamma3":[],
            "delta1":[],
            "delta2":[],
            "delta3":[],
            "epsilon1":[],
            "epsilon2":[],
            "epsilon3":[],
            "time1":[],
            "time2":[],
            "time3":[],
    }
    final_aver1={
            "request1":[],
            # "abandon1":[],
            "min1":[],
            "max1":[],
            "alpha1":[],
            "beta1":[],
            "var1":[],
            "link1":[],
    }
    final_aver2={
            "request2":[],
            # "abandon2":[],
            "min2":[],
            "max2":[],
            "alpha2":[],
            "beta2":[],
            "var2":[],
            "link2":[],
    }
    final_aver3={
            "request3":[],
            # "abandon3":[],
            "min3":[],
            "max3":[],
            "alpha3":[],
            "beta3":[],
            "var3":[],
            "link3":[],
    }
    final_aver={
        "error_link":[],
        "abandon1":[],
        "abandon2":[],
        "abandon3":[],
        "alpha1":[],
        "beta1":[],
        "alpha2":[],
        "beta2":[],
        "alpha3":[],
        "beta3":[],
    }
    length_path=length_path_list[topo_choice.index(topo_c)]
    for i in range(10):
        request_inf1 = copy.deepcopy(request_inf)
        request_inf2 = copy.deepcopy(request_inf)
        request_inf3 = copy.deepcopy(request_inf)
        print("request is",request_inf)
        # print("request2 is",request_inf2)
        # print("request2 is",request_inf2)
        print("request_number is",len(request_inf['request_data']))
        print("This is ",i,"before test")
        temp['request1'].append(len(request_inf['request_data']))
        temp['request2'].append(len(request_inf['request_data']))
        temp['request3'].append(len(request_inf['request_data']))
        # for (i,j) in G1.edges():
        #     print((i,j),'->',G1[i][j]['capacity'])
        improve_route_iteration(G1, length_path, request_inf1, supplement_key,temp)
        # for (i,j) in G1.edges():
        #     print((i,j),'->',G1[i][j]['capacity'])
        print("----------------------------------------------------------------")
        compare_route1(G3, length_path,request_inf, request_inf3, supplement_key,temp)
        print("################################################################")
        compare_route2(G2, length_path, request_inf2, supplement_key,temp)
        print("****************************************************************")

    writer = pd.ExcelWriter(store_path+topo_c+'/'+data_choice+'_'+'result_3_keypool.xlsx')
    link_range=[[0,5,10,15,20,25,30,35,40,45,50,55,60,30,35,40,45,50,55,30,35,40,45,50,55],
    [0,5,10,15,20,25,30,35,40,45,50,55,30,35,40,45,50,25,30,35,40,45,50,25],
    [0,1,2,3,4,5,6,7,8,9,10,4,5,6,7,4,5,6,7],
    [0,2,4,6,8,10,12,14,16,8,10,12,14,8,10,12,14],
    [0,3,6,9,12,15,18,21,24,27,12,15,18,21,24,12,15,18,21,24]]
    for i in range(300):
        for j in link_range[topo_choice.index(topo_c)]:
            result={
                "error_link":[],
                "request1":[],
                "abandon1":[],
                "min1":[],
                "max1":[],
                "alpha1":[],
                "beta1":[],
                "var1":[],
                "link1":[],
                "request2":[],
                "abandon2":[],
                "min2":[],
                "max2":[],
                "alpha2":[],
                "beta2":[],
                "var2":[],
                "link2":[],
                "request3":[],
                "abandon3":[],
                "min3":[],
                "max3":[],
                "alpha3":[],
                "beta3":[],
                "var3":[],
                "link3":[],
                "gamma1":[],
                "gamma2":[],
                "gamma3":[],
                "delta1":[],
                "delta2":[],
                "delta3":[],
                "epsilon1":[],
                "epsilon2":[],
                "epsilon3":[],
                "time1":[],
                "time2":[],
                "time3":[],
            }
            request_inf=creat_request2(max1_node,max2_node,max_f_int,32)
            request_inf1 = copy.deepcopy(request_inf)
            request_inf2 = copy.deepcopy(request_inf)
            request_inf3 = copy.deepcopy(request_inf)
            G11 = copy.deepcopy(G1)
            G22 = copy.deepcopy(G2)
            G33 = copy.deepcopy(G3)
            link_error(G11,G22,G33,j,max1_node,max2_node)
            print("This is ",topo_c,':',i,'.',j,'/'," test")
            print("request is",request_inf)
            # print("request1 is",request_inf1)
            # print("request2 is",request_inf2)
            # print("request2 is",request_inf2)
            print("request_number is",len(request_inf['request_data']))
            result['error_link'].append(j)
            improve_route_iteration(G11, length_path, request_inf1, supplement_key0,result)
            result['request1'].append(len(request_inf['request_data']))
            print("----------------------------------------------------------------")
            compare_route2(G22, length_path, request_inf2, supplement_key0,result)
            result['request2'].append(len(request_inf['request_data']))
            print("****************************************************************")
            compare_route1(G33, length_path,request_inf,request_inf3,supplement_key0,result)
            result['request3'].append(len(request_inf['request_data']))
            print("################################################################")

            for key in final.keys():
                final[key]=final[key]+result[key]

            a1=nx.to_numpy_array(G11,weight='capacity')
            np.save(store_path+topo_c+'/'+data_choice+'/G1/'+str(i)+str(j)+'_netdata.npy', a1)
            np.save(store_path+topo_c+'/'+data_choice+'/R1/'+str(i)+str(j)+'_requestdata.npy', request_inf1)
            a2=nx.to_numpy_array(G22,weight='capacity')
            np.save(store_path+topo_c+'/'+data_choice+'/G2/'+str(i)+str(j)+'_netdata.npy', a2)
            np.save(store_path+topo_c+'/'+data_choice+'/R2/'+str(i)+str(j)+'_requestdata.npy', request_inf2)
            a3=nx.to_numpy_array(G33,weight='capacity')
            np.save(store_path+topo_c+'/'+data_choice+'/G3/'+str(i)+str(j)+'_netdata.npy', a3)
            np.save(store_path+topo_c+'/'+data_choice+'/R3/'+str(i)+str(j)+'_requestdata.npy', request_inf3)
            np.save(store_path+topo_c+'/'+data_choice+'/result/'+str(i)+str(j)+'_netlog.npy', result)

    res = pd.DataFrame(final)  # 创建DataFrame
        # 存表，去除原始索引列（0,1,2...）
    res.to_excel(writer, index=False, sheet_name='final')

    for i in final['error_link']:
        if i not in final_aver['error_link']:
            final_aver['error_link'].append(i)
            for j in range(len(final['error_link'])):
                if final['error_link'][j]==i:
                    if len(final_aver['abandon1'])==len(final_aver['error_link']):
                        final_aver['abandon1'][-1]=final_aver['abandon1'][-1]+final['abandon1'][j]
                        final_aver['abandon2'][-1]=final_aver['abandon2'][-1]+final['abandon2'][j]
                        final_aver['abandon3'][-1]=final_aver['abandon3'][-1]+final['abandon3'][j]
                        final_aver['alpha1'][-1]=final_aver['alpha1'][-1]+final['alpha1'][j]
                        final_aver['alpha2'][-1]=final_aver['alpha2'][-1]+final['alpha2'][j]
                        final_aver['alpha3'][-1]=final_aver['alpha3'][-1]+final['alpha3'][j]
                        final_aver['beta1'][-1]=final_aver['beta1'][-1]+final['beta1'][j]
                        final_aver['beta2'][-1]=final_aver['beta2'][-1]+final['beta2'][j]
                        final_aver['beta3'][-1]=final_aver['beta3'][-1]+final['beta3'][j]
                    else:
                        final_aver['abandon1'].append(final['abandon1'][j])
                        final_aver['abandon2'].append(final['abandon2'][j])
                        final_aver['abandon3'].append(final['abandon3'][j])
                        final_aver['alpha1'].append(final['alpha1'][j])
                        final_aver['alpha2'].append(final['alpha2'][j])
                        final_aver['alpha3'].append(final['alpha3'][j])
                        final_aver['beta1'].append(final['beta1'][j])
                        final_aver['beta2'].append(final['beta2'][j])
                        final_aver['beta3'].append(final['beta3'][j])
            final_aver['abandon1'][-1]=final_aver['abandon1'][-1]/final['error_link'].count(i)
            final_aver['abandon2'][-1]=final_aver['abandon2'][-1]/final['error_link'].count(i)
            final_aver['abandon3'][-1]=final_aver['abandon3'][-1]/final['error_link'].count(i)
            final_aver['beta1'][-1]=final_aver['beta1'][-1]/final['error_link'].count(i)
            final_aver['beta2'][-1]=final_aver['beta2'][-1]/final['error_link'].count(i)
            final_aver['beta3'][-1]=final_aver['beta3'][-1]/final['error_link'].count(i)
            final_aver['alpha1'][-1]=final_aver['alpha1'][-1]/final['error_link'].count(i)
            final_aver['alpha2'][-1]=final_aver['alpha2'][-1]/final['error_link'].count(i)
            final_aver['alpha3'][-1]=final_aver['alpha3'][-1]/final['error_link'].count(i)

    res_aver = pd.DataFrame(final_aver)  # 创建DataFrame
        # 存表，去除原始索引列（0,1,2...）
    res_aver.to_excel(writer, index=False, sheet_name='final_aver')
    writer.close()
    # draw_topology(G, 'topology0.jpg')
    # draw_topology(G1, 'topology1.jpg')
    # draw_topology(G2, 'topology2.jpg')
    # draw_topology(G3, 'topology3.jpg')