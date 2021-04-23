#!python3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 21:27:58 2019

@author: kuokuo
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import csv





def inst_info(dic_info,key,data,op):  
    # key and data should be in order !!!!!!
    if op==1: #normalized
        key_info,key_info_lis=key
        info,lis_info1 = data
        n=[str(i+1) for i in range(len(info))] #['1','2','3','4','5','6','7']
        # store key_info
        for i,content in zip(n,info):
            cnt=0
            for value in content:
                dic_info[i][key_info[cnt]].append(value)
                cnt=cnt+1         
        # store key_info_lis
        for i,value in zip(key_info_lis,lis_info1):
            dic_info['1'][i].append(value)
    else: #original
        n=[str(i+1) for i in range(len(data))]
        for i,content in zip(n,data):
            cnt=0
            for value in content:
                dic_info[i][key[cnt]].append(value)
                cnt=cnt+1
    return dic_info

def test_inst_info(dic_info,key,data,op):  
    # key and data should be in order !!!!!!
    
    if op==1: #normalized
        key_info,key_info_lis=key
        info,lis_info = data
        n=[str(i+1) for i in range(len(info))]
        #lis_info1,lis_info2,lis_info3,lis_info4,lis_info5 = lis_info
        # store key_info
        for i,content in zip(n,info):
            cnt=0
            for value in content:
                dic_info[i][key_info[cnt]].append(value)
                cnt=cnt+1         
        # store key_info_lis
        for i,content in zip(n,lis_info):
            #cnt=0
            for j,value in zip(key_info_lis,content):
                dic_info[i][j].append(value)
    else: #original
        n=[str(i+1) for i in range(len(data))]
        for i,content in zip(n,data):
            cnt=0
            for value in content:
                dic_info[i][key[cnt]].append(value)
                cnt=cnt+1
    return dic_info


def train_avg_info(dic_info,key_avg,n):
    lis=[str(i+1) for i in range(1)] #['1','2','3','4','5','6','7']
    dic_avg_info={key_dic_info:{term: [] for term in key_avg} for key_dic_info in lis  }
    for key_dic_info in lis:
        dic = dic_info[key_dic_info]
        for key in key_avg:
            dic_avg_info[key_dic_info][key]=[sum(dic[key][i*n:(i+1)*n])/n for i in range(int(len(dic[key])/n))]
    return dic_avg_info

def test_avg_info(dic_info,key_avg,n):
    lis=[str(i+1) for i in range(len(dic_info))] #['1','2','3','4','5','6','7']
    dic_avg_info={key_dic_info:{term: [] for term in key_avg} for key_dic_info in lis  }
    for key_dic_info, dic in dic_info.items(): #key_dic_info=['1','2','3','4','5']
        for key in key_avg:
            dic_avg_info[key_dic_info][key]=[sum(dic[key][i*n:(i+1)*n])/n for i in range(int(len(dic[key])/n))]
    return dic_avg_info

def train_plot_avg(dic_avg_info,key_avg,realization,name,save_dir):
    title=['('+name+')Average '+i+' with '+str(realization)+' Realizations' for i in key_avg]   
    ylabel=['Average '+i for i in key_avg]
    xlabel='Training Steps (x'+str(realization)+')' 
    n=1
    label=[str(i+1) for i in range(n)]

    for i,key in enumerate(key_avg): 
        for j in range(n):
            plt.plot(np.arange(len(dic_avg_info[label[j]][key])), dic_avg_info[label[j]][key],label=label[j])
        plt.legend(loc='upper right')
        plt.title(title[i])
        plt.ylabel(ylabel[i])
        plt.xlabel(xlabel)
        plt.savefig(save_dir+title[i]+'.png') 
        plt.show() 

def test_plot_avg(dic_avg_info,key_avg,realization,name,save_dir):
    title=['('+name+')Average '+i+' with '+str(realization)+' Realizations' for i in key_avg]   
    ylabel=['Average '+i for i in key_avg]
    xlabel='Training Steps (x'+str(realization)+')' 
    n=len(dic_avg_info)
    label=[str(i+1) for i in range(n)]

    for i,key in enumerate(key_avg): 
        for j in range(n):
            plt.plot(np.arange(len(dic_avg_info[label[j]][key])), dic_avg_info[label[j]][key],label=label[j])
        plt.legend(loc='upper right')
        plt.title(title[i])
        plt.ylabel(ylabel[i])
        plt.xlabel(xlabel)
        plt.savefig(save_dir+title[i]+'.png') 
        plt.show()


def plot(start,lisRL,n,title,ylabel,xlabel):
    lisRL=lisRL[start::]
    avg = [sum(lisRL[i*n:(i+1*n)])/n for i in range(int(len(lisRL)/n))]
    plt.plot(np.arange(len(avg)), avg)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    #plt.savefig(title+'.png') 
    plt.show()

def plot_individual(env,dic_info,key_plot,n):    
    #key_plot=['Backhaul Difference','SINR','QoS Difference']
    #n=100
    #n=10
    nTerm = [env.nSBS, env.nUE, env.nUE]
    title=[i+' with '+str(n)+' Realizations' for i in key_plot]   
    ylabel=key_plot
    xlabel='Training Steps (x'+str(n)+')'   
    label1=['(DDPG1)SBS','(DDPG1)UE','(DDPG1)UE']
    label2=['(DDPG2)SBS','(DDPG2)UE','(DDPG2)UE']
    label3=['(DDPG3)SBS','(DDPG3)UE','(DDPG3)UE']
    label4=['(DDPG4)SBS','(DDPG4)UE','(DDPG4)UE']
    label5=['(DDPG5)SBS','(DDPG5)UE','(DDPG5)UE']
    label=[label1,label2,label3,label4,label5]
    color=['r','b','g','c','m']
    linestyle=['-','--',':',':','-']
    lis=['1']
    save={name:[[] for i in range(num)] for name,num in zip(key_plot,nTerm)}
    '''
    lis=['1','2','3','4','5']
    
    for i,key in enumerate(key_plot):#i:0-2 same term(backhaul,SINR,QoS) different methods
        difference_list=[dic_info[n][key] for n in lis] #[Method1 Method2 .. ]
        temp_list=[[]for i in range(5)] # number of Methods(5)
        for z,difference in enumerate(difference_list):
            for y in range(nTerm[i]):
                temp_list[z].append([difference[x][y] for x in range(len(difference))])
        for length in range(len(temp_list[0])):
            for k in range(5):
                #print(len(temp_list[k][length]))#400
                Bl=[sum(temp_list[k][length][j*n:(j+1)*n])/n for j in range(int(len(temp_list[k][length])/n))]
                plt.plot(np.arange(len(Bl)), Bl,label=label[k][i]+str(length),color=color[length],linestyle=linestyle[k])
    '''
    for i,key in enumerate(key_plot):#i:0-2 same term(backhaul,SINR,QoS) different methods
        difference_list=[dic_info[n][key] for n in lis] #[Method1 Method2 .. ]
        temp_list=[[]for i in range(1)] #----------------------JUST PLOT rl
        for z,difference in enumerate(difference_list):
            for y in range(nTerm[i]):
                temp_list[z].append([difference[x][y] for x in range(len(difference))])
        for length in range(len(temp_list[0])):
            for k in range(1):
                #print(len(temp_list[k][length]))#400
                Bl=[sum(temp_list[k][length][j*n:(j+1)*n])/n for j in range(int(len(temp_list[k][length])/n))]
                save[key][length]=Bl
                plt.plot(np.arange(len(Bl)), Bl,label=label[k][i]+str(length),color=color[length],linestyle=linestyle[k])

        plt.legend(loc='upper right')
        plt.title(title[i])
        plt.ylabel(ylabel[i])
        plt.xlabel(xlabel)
        #plt.savefig(title[i]+'.png')
        plt.show()
    return save 

def test_plot_individual(env,dic_info,method_index,key_plot,n,save_dir):    
    #key_plot=['Backhaul Difference','SINR','QoS Difference','Throughput']
    nTerm = [env.nSBS, env.nUE, env.nUE, env.nUE]
    title=['['+method_index+']'+i+' with '+str(n)+' Realizations' for i in key_plot]   
    ylabel=key_plot
    xlabel='Training Steps (x'+str(n)+')'   
    label=['SBS','UE','UE','UE']
    color=['r','b','g','c','m','y','k','b']
    linestyle=['-','--',':',':','-']

    save={name:[[] for i in range(num)] for name,num in zip(key_plot,nTerm)}
    save_ori={name:[[] for i in range(num)] for name,num in zip(key_plot,nTerm)}
    for i,key in enumerate(key_plot):#i:0-2 same term(backhaul,SINR,QoS) different methods
        difference = dic_info[key]
        temp_list=[]
        for y in range(nTerm[i]):
                temp_list.append([difference[x][y] for x in range(len(difference))])
        for length in range(len(temp_list)):
            Bl=[sum(temp_list[length][j*n:(j+1)*n])/n for j in range(int(len(temp_list[length])/n))]
            save[key][length]=Bl
            Bl_ori=[temp_list[length][j*n:(j+1)*n] for j in range(int(len(temp_list[length])/n))]
            for ori in Bl_ori:
                save_ori[key][length]=save_ori[key][length]+ori
            plt.plot(np.arange(len(Bl)), Bl,label=label[i]+str(length),color=color[length],linestyle=linestyle[0])

        plt.legend(loc='upper right')
        plt.title(title[i])
        plt.ylabel(ylabel[i])
        plt.xlabel(xlabel)
        plt.savefig(save_dir+title[i]+'.png')
        plt.show()
    return save,save_ori

        
def print_info(info,s):   
    c, P = info
    print('s ',s)
    print('c = ',c,' P in dBm=',10*np.log10(P*1000))
        
def test_print_info(info,s):     
    print('s ',s)    
    for i,data in enumerate(info):
        c, P = data
        print('[',i+1,'] c = ',c,' P in dBm=',10*np.log10(P*1000))


def writeCSV(dic_avg_info,dic_avg_info_ori,save,dic_info_ori_key,key_individual,key_avg,title,op):
    n=len(dic_avg_info)
    with open(title+'_history.csv','w',newline='') as csvfile:
        writer = csv.writer(csvfile) 
        # 1)normalized
        for key in key_avg:
            for i in [str(i+1) for i in range(n)] :             
                writer.writerow([i, '(normlaized)Average '+key ]+dic_avg_info[i][key])
        # 2)original
        for key in dic_info_ori_key:
            for i in [str(i+1) for i in range(n)] :             
                writer.writerow([i, '(original)Average '+key ]+dic_avg_info_ori[i][key])
        # 3)individual
        if op==1: #test mode, all baselines 
            for n_model in range(n):
                content=save[n_model]
                for key in key_individual:
                    for i,v in enumerate(content[key]):             
                        writer.writerow(['['+str(n_model+1)+']'+str(i),key]+v)
        else:
            for key in key_individual:
                for i,v in enumerate(save[key]):             
                    writer.writerow([i,key]+v)

def writeCSV_nobackhaul(dic_avg_info,dic_avg_info_ori,dic_info_ori_key,key_avg,title):
    n=len(dic_avg_info)
    with open(title+'_history.csv','w',newline='') as csvfile:
        writer = csv.writer(csvfile) 
        # 1)normalized
        for key in key_avg:
            for i in [str(i+1) for i in range(n)] :             
                writer.writerow([i, '(normlaized)Average '+key ]+dic_avg_info[i][key])
        # 2)original
        for key in dic_info_ori_key:
            for i in [str(i+1) for i in range(n)] :             
                writer.writerow([i, '(original)Average '+key ]+dic_avg_info_ori[i][key])
                
def writeEE(debug_dic_info_EE,debug_dic_info_EE_key,title):
    n=len(debug_dic_info_EE)
    with open(title+'_EE.csv','w',newline='') as csvfile:
        writer = csv.writer(csvfile) 
        # 1)normalized
        for key in debug_dic_info_EE_key:
            for iMethod in [str(i+1) for i in range(n)] :             
                writer.writerow([iMethod, key ]+debug_dic_info_EE[iMethod][key])
                
def writeI(debug_dic_info_I,debug_dic_info_I_key,nUE,title):
    n=len(debug_dic_info_I)
    with open(title+'_I.csv','w',newline='') as csvfile:
        writer = csv.writer(csvfile) 
        # 1)normalized
        for key in debug_dic_info_I_key:
            for iMETHOD in [str(i+1) for i in range(n)] :    
                for iUE in ['UE'+str(k) for k in range(nUE) ]:
                    writer.writerow([ key,iMETHOD,iUE ]+debug_dic_info_I[iMETHOD][key][iUE])

def writeAction(debug_dic_info_action,debug_dic_info_action_key,title):
        n_baseline=len(debug_dic_info_action)
        with open(title+'_Action.csv','w',newline='') as csvfile:
            writer = csv.writer(csvfile) 
            for iMethod in [str(i+1) for i in range(n_baseline)] :    
                for episdoe,content in enumerate (debug_dic_info_action[iMethod]['Association']):
                    for tstep,this in enumerate(content):
                        writer.writerow([iMethod, 'Association', episdoe, tstep ]+this)
                        writer.writerow([iMethod, 'Power Allocation', episdoe, tstep ]+list(debug_dic_info_action[iMethod]['Power Allocation'][episdoe][tstep]))
    

def findinf_list(lis):
    dic={'inf_index_list':[],'non_inf_list':[]}
    for i,value in enumerate(lis):
        if math.isinf(float(str(value))):
            dic['inf_index_list'].append(i)
        else:
            dic['non_inf_list'].append(value)
    return dic

def writeCSI(name,train_channel_episode):
    with open(name+'.csv','w',newline='') as csvfile:
        writer = csv.writer(csvfile)   
        for G in train_channel_episode:
            for i in list(G):
                writer.writerow(i)
                
def readCSI(name,nSBS,nUE,episode):
    with open(name+'.csv', newline='') as csvfile:
        rows = csv.reader(csvfile)
        rows = list(rows)
        lis_G=[]
        start=0
        for i in range(episode):
            G=np.array([float(i) for lis in rows[start:start+nSBS] for i in lis]).reshape(nSBS,-1)
            G=G[:,0:nUE+1]
            start=start+nSBS
            lis_G.append(G)
    return lis_G




'''
def writeBackhaulHistory(name,MAXepisode,debug_episode_back):
    back=np.zeros((MAXepisode,))
    back[debug_episode_back]=1
    with open(name+'.csv','w',newline='') as csvfile:
        writer = csv.writer(csvfile)  
        writer.writerow(back)

                
            
def readBackhaulHistory(name):
    with open('test_HistoryforBackhaulViolation.csv', newline='') as csvfile:
        rows = csv.reader(csvfile)
        rows = list(rows)[0]
        rows = [float(i) for i in rows]
        return rows

def plot_violateBackhaul(MAXepisode,debug_episode_back,name,save_dir):
    title='('+name+')History of Backhaul Constraint Violation'
    xlabel='Steps'
    back=np.zeros((MAXepisode,))
    back[debug_episode_back]=1
    plt.plot(np.arange(MAXepisode),back)
    plt.title(title)
    #plt.ylabel(ylabel[i])
    plt.xlabel(xlabel)
    plt.savefig(save_dir+title+'.png') 
    plt.show() 
'''    
def writeConstraintHistory(name,episode,debug_episode_back,mode):
    if mode == 0: # backhaul
        back=np.zeros((episode,))
        back[debug_episode_back]=1
        with open(name+'_HistoryforBackhaulViolation.csv','w',newline='') as csvfile:
            writer = csv.writer(csvfile)  
            writer.writerow(back)
    else:  #QoS
        with open(name+'_HistoryforQoSsatisfication.csv','w',newline='') as csvfile:
            writer = csv.writer(csvfile)  
            writer.writerow(debug_episode_back)
            
def writeConstraintHistory_v2(name,episode,debug_episode_back,mode):
    if mode == 0: # backhaul
        back=np.zeros((episode,))
        back[debug_episode_back]=1
        with open(name+'_HistoryforBackhaulViolation.csv','w',newline='') as csvfile:
            writer = csv.writer(csvfile)  
            writer.writerow(back)
    else:  #QoS
        with open(name+'_HistoryforQoSsatisfication.csv','w',newline='') as csvfile:
            writer = csv.writer(csvfile)  
            for i in debug_episode_back:
                writer.writerow(debug_episode_back[i])            
        
def readConstraintHistory(name,mode):
    if mode == 0: # backhaul
        filename=name+'_HistoryforBackhaulViolation'
    else:#QoS
        filename=name+'_HistoryforQoSsatisfication'
    with open(filename+'.csv', newline='') as csvfile:
        rows = csv.reader(csvfile)
        rows = list(rows)[0]
        rows = [float(i) for i in rows]
        return rows
        
def plot_constraint(episode,debug_episode_back,name,save_dir,mode):
    if mode ==0: # backhaul
        title='('+name+')History of Backhaul Constraint Violation'
        back=np.zeros((episode,))
        back[debug_episode_back]=1
        xlabel='Steps'
        plt.plot(np.arange(episode),back)
        plt.title(title)
    else:#QoS
        title='('+name+')History of QoS Satisfication'
        xlabel='Steps'
        plt.plot(np.arange(len(debug_episode_back)),debug_episode_back)
        plt.title(title)
    plt.xlabel(xlabel)
    plt.savefig(save_dir+title+'.png') 
    plt.show()     
    
    
def p_normalize(clip,P_NN):    
    # for sigmoid: have added noise -------------------------------------------
    P = np.array([  np.clip(power*clip, 0, clip)  for power in P_NN ])
    return P
    
