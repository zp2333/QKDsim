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

def pd_toExcel(G, fileName):  # pandas库储存数据到excel
    origin = []
    destination = []
    capacity = []
    node_name = []
    node_type = []
    for (i, j) in G.edges():
        origin.append(i)
        destination.append(j)
        capacity.append(G[i][j]['capacity'])
    for i in G.nodes:
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
    nx.draw(G, pos)
    # generate node_labels manually
    node_labels = {}
    edge_labels = {}
    for node in G.nodes:
        node_labels[node] = str(node)+','+str(G.nodes[node]['node_type'])

    for edge in G.edges:
        edge_labels[edge] = G[edge[0]][edge[1]]['capacity']

    nx.draw_networkx_labels(G, pos, labels=node_labels)

    # no edge_labels parameter, default is showing all attributes of edges
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.savefig(fileName)
    # plt.show() #实时展示拓扑
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
    model.solve(GUROBI_CMD(options=[('TimeLimit','45')],msg=0))  # 调用gurobi求解 加入solver
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
    model.solve(GUROBI_CMD(options=[('TimeLimit','45')],msg=0))
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
    for i, row in update_data.iterrows():
        if row['val']<0:
            G[row['origin']][row['destination']]['capacity'] += row['val']
            G[row['destination']][row['origin']]['capacity'] += row['val']
        elif row['val']>0:
             G[row['origin']][row['destination']]['capacity'] += row['val']


def creat_request(G, lam, miu, sigma):
    aver = G.size()*2/(len(list(G))+1)
    request_data = []
    request_source = []
    request_destination = []
    num = len(G.nodes)
    request_num = stats.poisson.rvs(mu=lam, size=num)
    for i in G.nodes:
        if G.degree(i)==1:
            request_num[i-1]=0
        else:
            request_num[i-1]=floor(G.degree(i)/aver*request_num[i-1])
        for j in range(request_num[i-1]):
            request_source.append(i)
            m = random.choice(list(G))
            while m == i or G.degree(m)<=aver:
                m = random.choice(list(G))
            request_destination.append(m)
            request_data.append(abs(round(random.gauss(miu, sigma))))
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

def creat_request3(G, num, miu, sigma):#在随机两个节点之间产生若干个请求
    aver = G.size()*2/(len(list(G))+1)
    request_data = []
    request_source = []
    request_destination = []
    random.seed()
    i=random.choice(list(G))
    while G.degree(i)<=aver:
        random.seed()
        i=random.choice(list(G))
    random.seed()
    m = random.choice(list(G))
    while m == i or G.degree(m)<=aver:
        random.seed()
        m = random.choice(list(G))
    for j in range(num):
        request_source.append(i)
        request_destination.append(m)
        request_data.append(abs(round(random.gauss(miu, sigma))))
    request_inf = {
        'request_data': request_data,
        'request_source': request_source,
        'request_destination': request_destination
    }
    return request_inf

def creat_request4(G, num, miu, sigma):#在随机两个节点之间产生若干个请求
    aver = G.size()*2/(len(list(G))+1)
    request_data = []
    request_source = []
    request_destination = []
    random.seed()
    i=random.choice(list(G))
    while G.degree(i)<=aver:
        random.seed()
        i=random.choice(list(G))
    random.seed()
    m = random.choice(list(G[i]))
    # while m == i or G.degree(m)<=aver:
    #     random.seed()
    #     m = random.choice(list(G))
    for j in range(num):
        request_source.append(i)
        request_destination.append(m)
        request_data.append(abs(round(random.gauss(miu, sigma))))
    request_inf = {
        'request_data': request_data,
        'request_source': request_source,
        'request_destination': request_destination
    }
    return request_inf

def improve_route(G, length_path, request_inf, supplement_key,result):
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
    result['abandon1'].append(request_abandon_num)
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
        for (i,j) in G.edges():
            x=(G_temp[i][j]['capacity']-G[i][j]['capacity'])/G_temp[i][j]['capacity']
            if x!=0:
                alpha.append(x)
        print("alpha is ",max(alpha))
        print("beta is",min(alpha))
        result['alpha1'].append(max(alpha))
        result['beta1'].append(min(alpha))

    else :
        # result['consume1'].append(0)
        result['alpha1'].append(0)
        result['beta1'].append(0)
        #draw_topology(G, 'topology2.jpg')
    
    test=[]
    for (i,j) in G.edges():
        test.append(G[i][j]['capacity'])
    result['min1'].append(min(test))
    result['max1'].append(max(test))
    update_key(G, supplement_key)


    for i in G.nodes():
        G.nodes[i]['node_in'] = 0
        G.nodes[i]['node_out'] = 0
    #pd_toExcel(G, 'test.xlsx')
    #draw_topology(G, 'topology3.jpg')



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
    result['abandon2'].append(request_abandon_num)
    print('compare route1 abandon request number is:', request_abandon_num)
    if request_abandon_num<len(request_inf['request_data']):
        alpha=[]
        consume=0
        for (i,j) in G_change.edges():
            x=(G_temp[i][j]['capacity']-G_change[i][j]['capacity'])/G_temp[i][j]['capacity']
            consume+=G_temp[i][j]['capacity']-G_change[i][j]['capacity']
            if x!=0:
                alpha.append(x)
        print("alpha is ",max(alpha))
        print("beta is ",min(alpha))
        print("consume is",consume/2)
        result['alpha2'].append(max(alpha))
        result['beta2'].append(min(alpha))
    else:
        result['alpha2'].append(0)
        result['beta2'].append(0)

    test=[]
    for (i,j) in G_change.edges():
        test.append(G_change[i][j]['capacity'])
    result['min2'].append(min(test))
    result['max2'].append(max(test))
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
    rsl_df = Min_consume(G, model,request_inf,length_path)
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
    print('compare route2 abandon request number is:', request_abandon_num)
    if len(request_inf['request_data']) != 0:
        print("Min_consume,the result is ", LpStatus[model.status])
        # print(rsl_df[rsl_df.val > 0])
        print("consume is ", sum(rsl_df.val))
        # result['consume3'].append(sum(rsl_df.val))
        #draw_topology(G, 'topology1.jpg')
        # 更新边的密钥容量
        rsl_df.val = -rsl_df.val
        G_temp=copy.deepcopy(G)
        update_key(G, rsl_df)
        alpha=[]
        for (i,j) in G.edges():
            x=(G_temp[i][j]['capacity']-G[i][j]['capacity'])/G_temp[i][j]['capacity']
            if x!=0:
                alpha.append(x)
        print("alpha is ",max(alpha))
        print("beta is",min(alpha))
        result['alpha3'].append(max(alpha))
        result['beta3'].append(min(alpha))
    else:
        # result['consume3'].append(0)
        result['alpha3'].append(0)
        result['beta3'].append(0)
        #draw_topology(G, 'topology2.jpg')

    test=[]
    for (i,j) in G.edges():
        test.append(G[i][j]['capacity'])
    result['min3'].append(min(test))
    result['max3'].append(max(test))
    update_key(G, supplement_key)
    # flow_value, flow_dict = nx.maximum_flow(G, 29, 40)
    # result['out3'].append(flow_value)

    for i in G.nodes():
        G.nodes[i]['node_in'] = 0
        G.nodes[i]['node_out'] = 0
    #pd_toExcel(G, 'test.xlsx')
    #draw_topology(G, 'topology3.jpg')

def link_error(G1,G2,G3,num):
    for i in range(num):
        random.seed()
        m = random.choice(list(G1))
        while len(list(G1.neighbors(m)))==0 and len(list(G1))>2:
            if m!=29 and m!=40:
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

length_path0 = 3
length_path = 3

supplement_key = pd.read_excel("data_supplement.xlsx", sheet_name="node_data")
supplement_key0 = pd.read_excel("data_supplement.xlsx", sheet_name="node_data0")
fpath = os.path.join("data_copy.xlsx")
routes = pd.read_excel(fpath, sheet_name="node_data")
s_d = pd.read_excel(fpath, sheet_name="s_d")

# print(request_data)

G = nx.DiGraph()  # 创建空的有向图

for i, row in s_d.iterrows():
    G.add_node(row.node_name, node_type=row.node_type, node_in=0, node_out=0)

for i, row in routes.iterrows():
    G.add_edge(row.origin, row.destination, capacity=row.capacity)
# 建立模型
#request_inf = read_request('request.xlsx')


# request_inf={'request_data': [11, 11, 11, 8, 10, 10, 10, 9, 10, 10], 'request_source': [1, 2, 2, 3, 4, 4, 5, 5, 6, 6], 'request_destination': [3, 6, 3, 6, 2, 1, 3, 4, 3, 5]}
# request_inf1 = copy.deepcopy(request_inf)
# request_inf2 = copy.deepcopy(request_inf)
# improve_route(G1, length_path, request_inf1, supplement_key,result)
# print("----------------------------------------------------------------")
# compare_route(G2, length_path, request_inf2, supplement_key,result)
final={
        "Brust_num":[],
        "abandon1":[],
        "min1":[],
        "max1":[],
        "alpha1":[],
        "beta1":[],
        "abandon2":[],
        "min2":[],
        "max2":[],
        "alpha2":[],
        "beta2":[],
        "abandon3":[],
        "min3":[],
        "max3":[],
        "alpha3":[],
        "beta3":[],
}
final_aver={
        "Brust_num":[],
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
writer = pd.ExcelWriter('result.xlsx')

for brust_num in range(10,70,10):
    temp={
        "request1":[],
        "abandon1":[],
        "min1":[],
        "max1":[],
        "alpha1":[],
        "beta1":[],
        "request2":[],
        "abandon2":[],
        "min2":[],
        "max2":[],
        "alpha2":[],
        "beta2":[],
        "request3":[],
        "abandon3":[],
        "min3":[],
        "max3":[],
        "alpha3":[],
        "beta3":[],
    }
    #request_inf = creat_request(G, 0.75, 320, 0)
    # max_flow,max_flow_dict=nx.maximum_flow(G,29,40)
    # print(max_flow)
    # flow_number=max_flow*0.9/128
    request_inf = creat_request2(29,40,54,128)
    G1 = copy.deepcopy(G)
    G2 = copy.deepcopy(G)
    G3 = copy.deepcopy(G)
    # request_inf = creat_request2(29,40,j+138,50)
    for i in range(10):
        #request_inf = creat_request(G, 1.4, 320, 0)
        #request_inf = creat_request2(29,40,210,32)
        #request_inf = creat_request(G, 0.5, 320, 0)
        request_inf1 = copy.deepcopy(request_inf)
        request_inf2 = copy.deepcopy(request_inf)
        request_inf3 = copy.deepcopy(request_inf)
        print("request is",request_inf)
        print("request_number is",len(request_inf['request_data']))
        print("This is ",i,"test")
        temp['request1'].append(len(request_inf['request_data']))
        temp['request2'].append(len(request_inf['request_data']))
        temp['request3'].append(len(request_inf['request_data']))
        improve_route(G1, length_path0, request_inf1, supplement_key,temp)
        print("----------------------------------------------------------------")
        compare_route1(G2, length_path0,request_inf, request_inf2, supplement_key,temp)#$******************
        print("################################################################")
        compare_route2(G3, length_path0, request_inf3, supplement_key,temp)
        print("****************************************************************")
    print(G.size())
    #request_inf = creat_request(G, 1.4, 320, 0)
    #request_inf = creat_request2(29,40,210,32)
    request_inf1 = copy.deepcopy(request_inf)
    request_inf2 = copy.deepcopy(request_inf)
    request_inf3 = copy.deepcopy(request_inf)
    print("request is",request_inf)
    print("request_number is",len(request_inf['request_data']))
    print("This is ",i,"test_begin")
    temp['request1'].append(len(request_inf['request_data']))
    temp['request2'].append(len(request_inf['request_data']))
    temp['request3'].append(len(request_inf['request_data']))
    improve_route(G1, length_path0, request_inf1, supplement_key0,temp)
    print("----------------------------------------------------------------")
    request_inf2 = copy.deepcopy(request_inf)
    compare_route1(G2, length_path0,request_inf, request_inf2, supplement_key0,temp)
    print("################################################################")
    compare_route2(G3, length_path0, request_inf3, supplement_key0,temp)
    print("****************************************************************")
    print(G1.edges(data='capacity'))
    print(G2.edges(data='capacity'))
    print(G3.edges(data='capacity'))
    pd_toExcel(G1,'G1.xlsx')
    pd_toExcel(G2,'G2.xlsx')
    pd_toExcel(G3,'G3.xlsx')
    for i in range(900):
            result={
                "Brust_num":[],
                "abandon1":[],
                "min1":[],
                "max1":[],
                "alpha1":[],
                "beta1":[],
                "abandon2":[],
                "min2":[],
                "max2":[],
                "alpha2":[],
                "beta2":[],
                "abandon3":[],
                "min3":[],
                "max3":[],
                "alpha3":[],
                "beta3":[],
            }
            request_inf=creat_request3(G,brust_num,96,0)
            request_inf1 = copy.deepcopy(request_inf)
            request_inf2 = copy.deepcopy(request_inf)
            request_inf3 = copy.deepcopy(request_inf)
            G11 = copy.deepcopy(G1)
            G22 = copy.deepcopy(G2)
            G33 = copy.deepcopy(G3)

            print(i,"test,Brust_num is ",brust_num)
            print("request is",request_inf)
            print("request_number is",len(request_inf['request_data']))
            result['Brust_num'].append(brust_num)
            improve_route(G11, length_path, request_inf1, supplement_key0,result)
            print("----------------------------------------------------------------")
            compare_route2(G33, length_path, request_inf3, supplement_key0,result)
            print("****************************************************************")
            compare_route1(G22, length_path, request_inf2,request_inf2, supplement_key0,result)
            print("################################################################")

            for key in final.keys():
                final[key]=final[key]+result[key]
    
   
   
res = pd.DataFrame(final)  # 创建DataFrame
    # 存表，去除原始索引列（0,1,2...）
res.to_excel(writer, index=False, sheet_name='final')

for i in final['Brust_num']:
    if i not in final_aver['Brust_num']:
        final_aver['Brust_num'].append(i)
        for j in range(len(final['Brust_num'])):
            if final['Brust_num'][j]==i:
                if len(final_aver['abandon1'])==len(final_aver['Brust_num']):
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
        final_aver['abandon1'][-1]=final_aver['abandon1'][-1]/final['Brust_num'].count(i)
        final_aver['abandon2'][-1]=final_aver['abandon2'][-1]/final['Brust_num'].count(i)
        final_aver['abandon3'][-1]=final_aver['abandon3'][-1]/final['Brust_num'].count(i)
        final_aver['beta1'][-1]=final_aver['beta1'][-1]/final['Brust_num'].count(i)
        final_aver['beta2'][-1]=final_aver['beta2'][-1]/final['Brust_num'].count(i)
        final_aver['beta3'][-1]=final_aver['beta3'][-1]/final['Brust_num'].count(i)
        final_aver['alpha1'][-1]=final_aver['alpha1'][-1]/final['Brust_num'].count(i)
        final_aver['alpha2'][-1]=final_aver['alpha2'][-1]/final['Brust_num'].count(i)
        final_aver['alpha3'][-1]=final_aver['alpha3'][-1]/final['Brust_num'].count(i)

res_aver = pd.DataFrame(final_aver)  # 创建DataFrame
    # 存表，去除原始索引列（0,1,2...）
res_aver.to_excel(writer, index=False, sheet_name='final_aver')
writer.close()
draw_topology(G1, 'topology1.jpg')
draw_topology(G2, 'topology2.jpg')
draw_topology(G3, 'topology3.jpg')