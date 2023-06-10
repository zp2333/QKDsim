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
import creat_topology as ct

def update_key(G, update_data):
    for i in range(len(update_data['origin'])):
        G[update_data['origin'][i]][update_data['destination'][i]]['capacity'] += update_data['val'][i]
        G[update_data['destination'][i]][update_data['origin'][i]]['capacity'] += update_data['val'][i]
def draw_topology(G, fileName):
    pos = nx.spring_layout(G)

    #ncolor=['r']+['y' for index in range(6)]+['r']+['y' for index in range(5)]+['r','r','r','r']+['y' for index in range(6)]+['r']+['y' for index in range(8)]+['r']+['y' for index in range(6)]+['r']+['y' for index in range(12)]+['r']+['y' for index in range(8)]+['r']+['y' for index in range(9)]
    #nsize=[50]+[30 for index in range(6)]+[50]+[30 for index in range(5)]+[50,50,50,50]+[30 for index in range(6)]+[50]+[30 for index in range(8)]+[50]+[30 for index in range(6)]+[50]+[30 for index in range(12)]+[50]+[30 for index in range(8)]+[50]+[30 for index in range(9)]
    #nx.draw(G, pos=pos,node_color=ncolor,node_size=nsize,width=0.5, arrowsize=0.5,arrowstyle='fancy')

    nx.draw(G, pos=pos,width=0.5, arrowsize=0.5,arrowstyle='fancy')
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
# def path_sort_alpha(alpha,path_list):
#     all=[]
#     alpha_list=[]
#     final=[]
#     for i in range(len(path_list)):
#         if alpha[i] not in alpha_list:
#             alpha_list.append(alpha[i])
#             all.append([])
def get_alpha(G,G_temp,path_list):
    alpha=[]
    for path in path_list:
        path_alpha=[]
        for j in range(len(path)-1):
            path_alpha.append((G[path[j]][path[j+1]]['capacity']-G_temp[path[j]][path[j+1]]['capacity'])/G[path[j]][path[j+1]]['capacity'])
        alpha_max=max(path_alpha)
        alpha.append(alpha_max)
    return alpha
def source_list_al(alpha,pathlist,request_inf_data):
    source_al=[0 for index in range(len(pathlist))]
    flag=0
    alpha_max=0
    list_one=0
    num=request_inf_data
    ch_list=[]
    beta=copy.deepcopy(alpha)
    beta.sort()
    if max(alpha)==0:
            source_al[0]+=1
            flag=1
    else:
        for x in alpha:
            if x>alpha_max and x!=1:
                alpha_max=x
            if x==1:
                list_one+=1
        if len(pathlist)>1 and alpha_max!=0:
            max_path_list=[]
            for j in range(len(alpha)):
                if alpha[j]>beta[0]:#int(len(beta)/20)
                    max_path_list.append(j)
            # random.seed()
            # max_path=random.choice(max_path_list)
            for i in range(len(pathlist)):
                if i not in max_path_list and alpha[i]!=1:
                    ch_list.append(i)
            for i in range(1):
                if num>0:
                    if len(ch_list)!=0:
                        random.seed()
                        ch_num=random.choice(ch_list)
                        source_al[ch_num]+=1
                        ch_list.remove(ch_num)
                        num-=1
                        flag=1
        elif alpha_max==0:
            flag=2
        else:
            for i in range(len(pathlist)):
                if alpha[i]!=1 and num>0:
                    source_al[i]+=1
                    num-=1
                    flag=1
    return source_al,flag
def update_HR(G,source_al,path_list):
    re_sum=0
    for i in range(len(path_list)):
        for j in range(len(path_list[i])-1):
            if source_al[i]>0:
                if G[path_list[i][j]][path_list[i][j+1]]['capacity']<source_al[i]:
                    re_sum+=source_al[i]
                else:
                    G[path_list[i][j]][path_list[i][j+1]]['capacity']-=source_al[i]
                    G[path_list[i][j+1]][path_list[i][j]]['capacity']-=source_al[i]
    return re_sum
def improve_route_Heuristic(G, length_path, request_inf, supplement_key,result):
    request_abandon_num=0
    G_change=copy.deepcopy(G)
    G_temp=copy.deepcopy(G)
    start_time1=time.perf_counter()
    for i in range(len(request_inf['request_data'])):
        G_copy=copy.deepcopy(G_change)
        path_shoest_lenth=nx.dijkstra_path_length(G_change,source=request_inf['request_source'][i],target=request_inf['request_destination'][i])
        path_test=list(nx.all_simple_paths(G_change,source=request_inf['request_source'][i],target=request_inf['request_destination'][i],cutoff=path_shoest_lenth+1))
        # path_test.sort(key=takeSecond)
        #path_list=path_sort(G_change,path_test)#按照最小容量大小排序
        path_list=path_test
        path_alpha=get_alpha(G,G_change,path_list)
        request_inf_data=request_inf['request_data'][i]
        source_flag=1
        while request_inf_data!=0 and source_flag==1:
            #time_test=time.perf_counter()
            source_al,source_flag=source_list_al(path_alpha,path_list,request_inf_data)
            re_sum=update_HR(G_change,source_al,path_list)
            if re_sum!=0:
                request_inf_data+=re_sum
            path_alpha=get_alpha(G,G_change,path_list)
            request_inf_data-=sum(source_al)
            #print("time_test is",time.perf_counter()-time_test)
        if request_inf_data!=0:
            request_abandon_num+=1
            G_change=copy.deepcopy(G_copy)
    result['abandon1'].append(request_abandon_num)
    print('Heuristic route abandon request number is:', request_abandon_num)
    if request_abandon_num<len(request_inf['request_data']):
        alpha_temp=[]
        alpha=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        link_use_num=0
        consume=0
        for (i,j) in G_change.edges():
            x=(G_temp[i][j]['capacity']-G_change[i][j]['capacity'])/G_temp[i][j]['capacity']
            consume+=G_temp[i][j]['capacity']-G_change[i][j]['capacity']
            if x!=0:
                # print(i,'----------->',j,'===',G_temp[i][j]['capacity']-G_change[i][j]['capacity'],'****',x)
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
        result['alpha1'].append(alpha[0])
        result['beta1'].append(alpha[2])
        result['gamma1'].append(alpha[4])
        result['delta1'].append(alpha[6])
        result['epsilon1'].append((alpha[6]+alpha[8]+alpha[4])/3)
        result['link1'].append(link_use_num)
        print("consume is ",consume/2)
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
    time1 =time.perf_counter()-start_time1
    print("time is",time1)
    test=[]
    for (i,j) in G_change.edges():
        test.append(G_change[i][j]['capacity'])
    result['min1'].append(min(test))
    result['max1'].append(max(test))
    result['var1'].append(np.std(alpha_temp,ddof=1))
    result['time1'].append(time1)
    update_key(G_change, supplement_key)
    for (i,j) in G_change.edges():
        G[i][j]['capacity']=G_change[i][j]['capacity']

def fp(origin,destination,all_path):
    result=[]
    j=0
    for path in all_path:
        for i in range(len(path)-1):
            if path[i]==origin and path[i+1]==destination:
                result.append(j)
        j+=1
    return result
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
    model.solve(GUROBI_CMD(options=[('TimeLimit','15')],msg=0))
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
length_path = 6
store_path='rowdata/3.2jhgx_H/'
# store_path='result/'
# topo_choice=['Chinanet','Geant2012','Dataxchange','Sprint','EliBackbone']#'Chinanet','Geant2012','Dataxchange','Sprint','EliBackbone'
# data_choice='alpha'
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

# max_f,maxflow = nx.maximum_flow(G, 21, 8)#最小费用最大流
# print(max_f)

# 建立模型
#request_inf = read_request('request.xlsx')

#request_inf={'request_data': [11, 11, 11, 8, 10, 10, 10, 9, 10, 10], 'request_source': [1, 2, 2, 3, 4, 4, 5, 5, 6, 6], 'request_destination': [3, 6, 3, 6, 2, 1, 3, 4, 3, 5]}
# request_inf1 = copy.deepcopy(request_inf)
# request_inf2 = copy.deepcopy(request_inf)
# improve_route(G1, length_path, request_inf1, supplement_key,result)
# print("----------------------------------------------------------------")
# compare_route(G2, length_path, request_inf2, supplement_key,result)
final={
        "request1":[],
        "abandon1":[],
        "min1":[],
        "max1":[],
        "alpha1":[],
        "beta1":[],
        "var1":[],
        "link1":[],
        "gamma1":[],
        "delta1":[],
        "epsilon1":[],
        "time1":[],
}
final_aver1={
        "request1":[],
        "min1":[],
        "max1":[],
        "alpha1":[],
        "beta1":[],
        "var1":[],
        "link1":[],
}

writer = pd.ExcelWriter(store_path+'result.xlsx')
a = np.load('request_inf_data_zip.npy', allow_pickle=True)
j=0
for request_inf in a:
    result={
        "request1":[],
        "abandon1":[],
        "min1":[],
        "max1":[],
        "alpha1":[],
        "beta1":[],
        "var1":[],
        "link1":[],
        "gamma1":[],
        "delta1":[],
        "epsilon1":[],
        "time1":[],
    }
    G1 = copy.deepcopy(G)
#         request_inf = {'request_data': [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10], 'request_source': [1, 1, 1, 1, 1, 2, 3, 5, 7, 8, 8, 8, 8, 8, 8, 9, 9, 11, 
# 11, 14, 14, 15, 15, 15, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 24, 26, 27, 29, 29, 29, 32, 33, 33, 33, 33, 33, 
# 33, 33, 33, 35, 37, 38, 38, 40, 40, 40, 40, 40, 41, 42, 42, 43, 43, 47, 48, 50, 51, 52, 52, 53, 53, 53, 53, 53, 
# 54, 54, 55, 56, 62, 62, 62, 63, 66, 70, 71, 71], 'request_destination': [8, 24, 62, 8, 40, 8, 17, 40, 40, 14, 62, 53, 14, 17, 14, 62, 17, 1, 8, 33, 33, 24, 17, 62, 1, 62, 24, 33, 62, 1, 33, 33, 40, 1, 17, 53, 53, 17, 1, 62, 
# 8, 40, 8, 8, 53, 1, 14, 40, 8, 1, 1, 24, 1, 53, 17, 8, 14, 33, 24, 40, 33, 62, 33, 14, 1, 8, 14, 24, 17, 24, 24, 24, 14, 24, 33, 62, 40, 17, 17, 40, 8, 24, 8, 17, 33, 53]}
    #request_inf = creat_request(G, j/200, 32, 0)#512bytes
    # request_inf = {
    #     'request_data':[620,160,200,200,155,155,155],
    #     'request_source':[15,1,17,17,1,1,1],
    #     'request_destination':[16,8,40,33,2,4,6],
    # }
    request_inf1 = copy.deepcopy(request_inf)
    print("This is ",j,'/',len(a)," test")
    j=j+1
    print("request is",request_inf)
    print("request_number is",len(request_inf['request_data']))
    improve_route_Heuristic(G1, length_path, request_inf1, supplement_key0,result)
    result['request1'].append(len(request_inf['request_data'])-result['abandon1'][-1])
    print("----------------------------------------------------------------")

    for key in final.keys():
        final[key]=final[key]+result[key]

    a1=nx.to_numpy_array(G1,weight='capacity')
    np.save(store_path+'G1/'+str(j)+'_netdata.npy', a1)
    np.save(store_path+'R1/'+str(j)+'_requestdata.npy', request_inf1)
    np.save(store_path+'result/'+str(j)+'_netlog.npy', result)

res = pd.DataFrame(final)  # 创建DataFrame
    # 存表，去除原始索引列（0,1,2...）
res.to_excel(writer, index=False, sheet_name='final')

for i in final['request1']:
    if i not in final_aver1['request1']:
        final_aver1['request1'].append(i)
        for j in range(len(final['request1'])):
            if final['request1'][j]==i:
                if len(final_aver1['min1'])==len(final_aver1['request1']):
                    final_aver1['min1'][-1]=final_aver1['min1'][-1]+final['min1'][j]
                    final_aver1['max1'][-1]=final_aver1['max1'][-1]+final['max1'][j]
                    final_aver1['alpha1'][-1]=final_aver1['alpha1'][-1]+final['alpha1'][j]
                    final_aver1['beta1'][-1]=final_aver1['beta1'][-1]+final['beta1'][j]
                    final_aver1['var1'][-1]=final_aver1['var1'][-1]+final['var1'][j]
                    final_aver1['link1'][-1]=final_aver1['link1'][-1]+final['link1'][j]
                else:
                    final_aver1['min1'].append(final['min1'][j])
                    final_aver1['max1'].append(final['max1'][j])
                    final_aver1['alpha1'].append(final['alpha1'][j])
                    final_aver1['beta1'].append(final['beta1'][j])
                    final_aver1['var1'].append(final['var1'][j])
                    final_aver1['link1'].append(final['link1'][j])
        final_aver1['min1'][-1]=final_aver1['min1'][-1]/final['request1'].count(i)
        final_aver1['max1'][-1]=final_aver1['max1'][-1]/final['request1'].count(i)
        final_aver1['alpha1'][-1]=final_aver1['alpha1'][-1]/final['request1'].count(i)
        final_aver1['beta1'][-1]=final_aver1['beta1'][-1]/final['request1'].count(i)
        final_aver1['var1'][-1]=final_aver1['var1'][-1]/final['request1'].count(i)
        final_aver1['link1'][-1]=final_aver1['link1'][-1]/final['request1'].count(i)

res_aver1 = pd.DataFrame(final_aver1)  # 创建DataFrame
    # 存表，去除原始索引列（0,1,2...）
res_aver1.to_excel(writer, index=False, sheet_name='final_aver1')
writer.close()
# draw_topology(G, 'topology0.jpg')
# draw_topology(G1, 'topology1.jpg')


#{'request_data': [32, 32, 32], 'request_source': [15, 24, 36], 'request_destination': [35, 17, 22]}
