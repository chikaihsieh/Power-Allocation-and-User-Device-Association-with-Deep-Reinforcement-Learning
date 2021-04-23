#!python3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 02:02:15 2019

@author: kuo
"""


import numpy as np
#from itertools import combinations
import random
import matplotlib.pyplot as plt
import math
import csv
import scipy.stats as st
import copy


class env_PowerAllocation(object):
    def __init__(self,nMBS=1,lambda1=0,lambda2=0,lambda3=0,MAXepisode=1500,n_baseline=6):
        super(env_PowerAllocation, self).__init__()
        """setting---------------------------------------------------------------------"""
        self.nMBS = 1
        self.nSBS = 3 # 5  # or J
        self.nTP = self.nMBS+self.nSBS
        self.nUE = 5 #8 # or K
        self.rMBS = 500 # in m 
        self.dmin = 10
        self.P_Max_MBS = 10**(4.3-3) # in 19.95 W
        self.P_Max_SBS = 10**(2.4-3) # in 0.25 W
        self.Pc_MBS = 130 # in W
        self.Pc_SBS = 0.5 #6.8# in W
        self.NT = 100 
        self.Ng= 20  
        self.N = 3 #5#200 # number of subchannel
        #self.sigma_MBS = 10**(0.6) # in 6 dB
        #self.sigma_SBS = 10**(0.4) 
        self.subB = 15000 # in Hz
        #self.B = self.subB*self.N   
        #self.B_MBS2SBS = self.B/self.nSBS
        self.Noise = (10**(-17.4))*0.001 # W/Hz
        self.SINR_threshold = 1 # 0dB / 1dB / 2dB
        self.Throughput_UE_threshold = np.log2(1+self.SINR_threshold) # 1/1.1756/ 1.37
        self.lambda1=lambda1
        self.lambda2=lambda2
        self.lambda3=lambda3
        self.ori_sizeTable = self.nSBS**self.nUE      # all possible association    
        self.s_dim =  (self.nUE+1)*self.nSBS+self.nUE # state dimension
        #self.Throughput_SBS_threshold = [ 50 for i in range(self.nSBS)] 
        ######################################################################## limit of continuous parameters corresponding to each discrete action 
        self.parameter_min=[np.array([0 for i in range(self.nUE)]) ] #P_min_SBS
        self.parameter_max=[np.array([self.P_Max_SBS for i in range(self.nUE)]) ]
        ######################################################################## avoid inf or divide 0
        self.delta_min = 10**(-20)
        self.delta_max = 10**(20)
        self.SINR_min=10**(-5.5) #-20dB
        self.SINR_max=10**(5.5)  # 20dB
        ######################################################################## debug or analysis
        debug_I={str(i):{'UE'+str(j):[] for j in range(self.nUE)} for i in range(MAXepisode)} # I, intra-cluster, inter-cluster
        #self.debug_channel={str(i):[]for i in range(MAXepisode)}
        debug_UE_throughput={str(i):[]for i in range(MAXepisode)}  # each UE throughput
        debug_SBS_throughput={str(i):[]for i in range(MAXepisode)} 
        debug_SBS_threshold={str(i):[]for i in range(MAXepisode)}
        debug_c={str(i):[]for i in range(MAXepisode)}  # user association
        debug_p={str(i):[]for i in range(MAXepisode)}  # power allocation
        debug_backhaul={str(i):{}for i in range(MAXepisode)} # which episode and step violate backhaul constraint & SBS index
        debug_QoS={str(i):{}for i in range(MAXepisode)}      # which episode and step violate QoS constraint & UE index
        debug_system_throughput={str(i):[]for i in range(MAXepisode)}
        debug_system_energy={str(i):[]for i in range(MAXepisode)}
        #----------------------------------------------------------------------- for all n_baseline methods
        self.debug_I={str(i):copy.deepcopy(debug_I) for i in range(n_baseline+1)}
        self.debug_UE_throughput={str(i):copy.deepcopy(debug_UE_throughput) for i in range(n_baseline+1)}
        self.debug_SBS_throughput={str(i):copy.deepcopy(debug_SBS_throughput) for i in range(n_baseline+1)}
        self.debug_SBS_threshold={str(i):debug_SBS_threshold for i in range(n_baseline+1)}
        self.debug_c={str(i):copy.deepcopy(debug_c) for i in range(n_baseline+1)}
        self.debug_p={str(i):copy.deepcopy(debug_p) for i in range(n_baseline+1)}
        self.debug_backhaul={str(i):copy.deepcopy(debug_backhaul) for i in range(n_baseline+1)}
        self.debug_QoS={str(i):copy.deepcopy(debug_QoS) for i in range(n_baseline+1)}
        self.debug_system_throughput={str(i):copy.deepcopy(debug_system_throughput) for i in range(n_baseline+1)}
        self.debug_system_energy={str(i):copy.deepcopy(debug_system_energy) for i in range(n_baseline+1)}


        
    def new(self,name):
        # to create new "Network Geometry"
        # Uniform distribution of SBSs and UEs
        # SBS and UE at least far 10 meters from MBS --> if violate, print something and need to create a new SBS-UE distribution again 
        SBS_R,SBS_A = np.random.uniform(self.dmin,self.rMBS,self.nSBS), np.random.uniform(0,2*math.pi,self.nSBS)
        self.xSBS,self.ySBS = [ r*math.cos(a) for r,a in zip(SBS_R,SBS_A)],[ r*math.sin(a) for r,a in zip(SBS_R,SBS_A)]
        UE_R,UE_A = np.random.uniform(self.dmin,self.rMBS,self.nUE), np.random.uniform(0,2*math.pi,self.nUE)
        self.xUE,self.yUE = [ r*math.cos(a) for r,a in zip(UE_R,UE_A)],[ r*math.sin(a) for r,a in zip(UE_R,UE_A)]
        # for pathloss
        self.dSBS2UE=[ ((self.xUE-x)**2+(self.yUE-y)**2)**0.5 for x,y in zip(self.xSBS,self.ySBS)]
        self.dMBS2SBS=[((x)**2+(y)**2)**0.5 for x,y in zip(self.xSBS,self.ySBS)]
        print('dSBS2UE=',self.dSBS2UE,'\n')
        print('dMBS2SBS=',self.dMBS2SBS,'\n')
        # check distance     
        for iSBS,D in enumerate(self.dSBS2UE):
            for iUE,d in enumerate(list(D)):
                if d < 10:
                    print('SBS '+str(iSBS)+' UE '+str(iUE)+' too close')
                if d>2000:
                    print('SBS '+str(iSBS)+' UE '+str(iUE)+' too far')      
        for i,d in enumerate(self.dMBS2SBS):
            if d < 10:
                print('SBS '+str(i)+' MBS too close')
            if d > 5000:
                print('SBS '+str(i)+' MBS too far')
        
        """2)Plot"""
        self.plotNetwork(name)
        """3)Build Table"""
        self.build_table()
        chosen_c=np.random.choice([i for i in range(self.sizeTable)])
        self.chosen_TP2UE=self.TP2UE[chosen_c]
        self.chosen_UE2TP=self.UE2TP[chosen_c]
        """4)Initialize state"""
        self.channel()
        """5)Store location and channel gain """
        self.writeCSV(name)
 
    
    def load(self,name):
        # to build the used env
        # 1)load
        self.readCSV(name)
        # 2)plotNetwork
        self.plotNetwork(name)
        # 3)Build Table
        # 4)for pathloss
        self.dSBS2UE=[ ((np.array(self.xUE)-x)**2+(np.array(self.yUE)-y)**2)**0.5 for x,y in zip(np.array(self.xSBS),np.array(self.ySBS))]
        self.dMBS2SBS=[((x)**2+(y)**2)**0.5 for x,y in zip(np.array(self.xSBS),np.array(self.ySBS))]
        # 5)action_space 
        self.action_space=(self.sizeTable,[(self.parameter_min[0],self.parameter_max[0]) for i in range(self.sizeTable)])
        
    def reset(self):
        """1)Initialize channel/ state"""
        #self.channel()
        c = np.random.randint(low=0, high=self.sizeTable, size= 1)[0] #####################
        #P = np.array([ self.P_Max_SBS for i in range(self.nUE) ])
        P = np.random.uniform(0,self.P_Max_SBS*0.1,self.nUE).flatten()
        _, _,s,_,_ ,_,_= self.step(c,P,False,True,'0',0,0)
        #self.channel()
        #inits =  list(st.norm(0, 1).rvs(self.nUE))
        #initG = list(self.G.T.flatten())
        return s#inits,initG     
        
    def plotNetwork(self,name):
        # 1)plot TP & UE
        plt.figure(figsize=(5,5))
        plt.scatter([0],[0],s=80,c='red',marker='o',alpha=0.5,label='MBS')
        plt.scatter(self.xSBS,self.ySBS,s=50,c='green',marker='D',alpha=0.5,label='SBS')
        plt.scatter(self.xUE,self.yUE,s=50,c='blue',marker='*',alpha=0.5,label='UE')
        # 2)Display index
        plt.annotate("0", xy=(0,0), xytext=(0, 0))
        cnt=1
        for x,y in zip(self.xSBS,self.ySBS):
            plt.annotate("%s" % cnt, xy=(x,y), xytext=(x, y))
            cnt = cnt+1
        cnt=1
        for x,y in zip(self.xUE,self.yUE):
            plt.annotate("%s" % cnt, xy=(x,y), xytext=(x, y))
            cnt = cnt+1
        margin=50    
        plt.xlim((-self.rMBS-margin, self.rMBS+margin))
        plt.ylim((-self.rMBS-margin, self.rMBS+margin))
        plt.title('Network Geometry ')
        plt.xlabel('Distance(m)')
        plt.ylabel('Distance(m)')
        plt.legend(loc='upper right')
        plt.savefig(name+'.png')
        plt.show()
        print('SBS Location')
        for i in range(self.nSBS):
            print(i,' (',self.xSBS[i],',',self.ySBS[i],')')
        print('UE Location')
        for i in range(self.nUE):
            print(i,' (',self.xUE[i],',',self.yUE[i],')')
            
    def writeCSV(self,name):
        with open(name+'.csv','w',newline='') as csvfile:
            writer = csv.writer(csvfile) 
            # write SBS, UE location 
            writer.writerow(self.xSBS)
            writer.writerow(self.ySBS)
            writer.writerow(self.xUE)
            writer.writerow(self.yUE) 
            # write channel gain
            for i in list(self.G):                 
                writer.writerow(i)
            # write ori_TP2UE
            for key,lis in self.ori_TP2UE.items(): 
                for i in lis:
                    writer.writerow(i)
            # write ori_UE2TP        
            for key,lis in self.ori_UE2TP.items(): 
                    writer.writerow(lis)
            # write sizeTable ??  ##############################################      
            writer.writerow([self.sizeTable])
            # write TP2UE
            for key,lis in self.TP2UE.items():
                for i in lis:
                    writer.writerow(i)
            # write UE2TP
            for key,lis in self.UE2TP.items():
                writer.writerow(lis)
                
    def readCSV(self,FileName):
        self.ori_TP2UE={i:[] for i in range(self.ori_sizeTable)}
        self.ori_UE2TP={}
        with open(FileName+'.csv', newline='') as csvfile:
            rows = csv.reader(csvfile)
            rows = list(rows)
            # read SBS, UE location
            self.xSBS,self.ySBS=[ float(i) for i in rows[0]],[ float(i) for i in rows[1]]
            self.xUE,self.yUE=[ float(i) for i in rows[2]],[ float(i) for i in rows[3]] 
            # read channel gain
            self.G=np.array([float(i) for lis in rows[4:4+self.nSBS] for i in lis]).reshape(self.nSBS,self.nUE+1)
            # read ori_TP2UE
            cnt=4+self.nSBS
            for i in range(self.ori_sizeTable):
                for j in range(self.nSBS):
                    self.ori_TP2UE[i].append([int(v) for v in rows[cnt]])
                    cnt=cnt+1
            # read ori_UE2TP        
            for i in range(self.ori_sizeTable):
                self.ori_UE2TP[i]=[int(v) for v in rows[cnt]]
                cnt=cnt+1
            # read sizeTable ??    
            self.sizeTable=int(rows[cnt][0])
            self.TP2UE={i:[] for i in range(self.sizeTable)}
            self.UE2TP={}
            cnt=cnt+1
            # read TP2UE
            for i in range(self.sizeTable):
                for j in range(self.nSBS):
                    self.TP2UE[i].append([int(v) for v in rows[cnt]])
                    cnt=cnt+1
            # read UE2TP
            for i in range(self.sizeTable):
                self.UE2TP[i]=[int(v) for v in rows[cnt]]
                cnt=cnt+1 
        # action dimension               
        self.a_dim = self.nUE+ self.sizeTable
    
    def index_list(self,l):
        # l=[1,1,2,3,5,5]
        # L_list =[[], [0, 1], [2], [3], [], [4, 5]]
        # invalid: True, need to delete this action
        L_list=[]
        invalid= False
        for i in range(self.nSBS):
            loc=[]
            c=l.count(i)
            if c>self.N:
                invalid = True
            while c!=0:
                loc.append(l.index(i))
                l[l.index(i)]=self.nSBS+1
                c=l.count(i)
            L_list.append(loc)
        return L_list, invalid
    
    def build_table(self):
        # build table for 1) all possible associations  --> ori_UE2TP / ori_TP2UE / ori_sizeTable
        #                 2) those expect that violates cluster size constraint --> UE2TP / TP2UE / sizeTable
        """ori_UE2TP"""
        mask=[[i] for i in range(self.nSBS)]
        l2=[mask[i]+mask[j] for i in range(self.nSBS)for j in range(self.nSBS)]
        for cnt in range(self.nUE-2):
           l2=[l2[i]+mask[j] for i in range(len(l2))for j in range(self.nSBS)] 
        self.ori_UE2TP = {i:l2[i] for i in range(len(l2))}
        """ori_TP2UE"""       
        invalid_list=[]
        self.ori_TP2UE={}
        for key in self.ori_UE2TP:
            self.ori_TP2UE[key], invalid = self.index_list(self.ori_UE2TP[key].copy())
            if invalid:
                invalid_list.append(key)
        self.ori_sizeTable=len(self.ori_UE2TP)
        """Check if action is invalid"""
        self.TP2UE=self.ori_TP2UE.copy()
        self.UE2TP=self.ori_UE2TP.copy()
        for i in invalid_list:
            self.TP2UE.pop(i)
            self.UE2TP.pop(i)
        """Re-create """ 
        temp={}    
        for i,key in enumerate(self.TP2UE):    
            temp[i]=self.TP2UE[key]
        self.TP2UE=temp
        temp={}    
        for i,key in enumerate(self.UE2TP):    
            temp[i]=self.UE2TP[key]
        self.UE2TP=temp
        self.sizeTable=len(self.UE2TP)
            
    def channel(self):
        """ 1)Channel """    
        # 1)Rayleigh  
        mu=0
        sigma=1 #var=sigma**2
        #X = list(st.norm(mu, sigma/2).rvs(2*(self.nUE+1)*self.nSBS))
        #R=np.array([(X[i]**2+X[i+1]**2)**0.5 for i in range((self.nUE+1)*self.nSBS)]).reshape(self.nSBS,(self.nUE+1))
    
        # 2)Path loss                
        #Shadowing_UE=(st.norm(0, self.sigma_SBS).rvs(self.nSBS*self.nUE)).reshape(self.nSBS,self.nUE)
        #Shadowing_SBS=(st.norm(0, self.sigma_MBS).rvs(self.nSBS)).reshape(self.nSBS,1)
        Shadowing_UE=0
        Shadowing_SBS=0
        PL_UE=np.array([30.53+36.7*math.log10(d/1000) for dUE2SBS in self.dSBS2UE for d in dUE2SBS]).reshape(self.nSBS,self.nUE) + Shadowing_UE
        PL_SBS=np.array([19.77+3.91*math.log10(d/1000) for d in self.dMBS2SBS]).reshape(self.nSBS,1) + Shadowing_SBS      
        self.PL=np.concatenate((PL_UE,PL_SBS),axis=1) # in dB
        self.PL = 10**(-self.PL/10)
        
        # 3)Combination   
        #self.G=self.PL*(R**2)
        self.G=self.PL
        #print('G=',self.G,'\n')
    
    def SubchannelAllocation(self): 
        # uniformly allocate subchannel
        # if cluster size <= subchannel number --> no intra-cluster interference
        # if cluster size >  subchannel number --> intra-cluster interference
        self.B_TP2UE=[]
        for k in range(self.nSBS):
            if len(self.chosen_TP2UE[k])==0:
               self.B_TP2UE.append([]) 
            else:
               # Method 1.uniform allocation --> for N>= #UE in a cluster
               nUE_SBSk = len(self.chosen_TP2UE[k])
               if  nUE_SBSk> self.N:
                   temp=[]
                   for i in range(int(nUE_SBSk/self.N)):
                       temp.append(random.sample([i for i in range(self.N)],self.N))
                   temp.append(random.sample([i for i in range(self.N)],nUE_SBSk%self.N))
                   self.B_TP2UE.append([i for l in temp for i in l])
               else:
                   self.B_TP2UE.append(random.sample([i for i in range(self.N)],nUE_SBSk))  
               # Method 2.order allocation --> 0,1,..,(N-1),0,1..
               #self.B_TP2UE.append([i%self.N for i in range(len(self.chosen_TP2UE[k]))])
        self.B_UE2B={iUE:B for liUE,lB in zip(self.chosen_TP2UE,self.B_TP2UE) for iUE,B in zip(liUE,lB)}
        
    def mean_std(self,n,cflage,name):
        #    1) calculate mean and standard deviation and save 
        # or 2) load mean and standard deviation to use   
        key = ['Energy Efficiency','Backhaul Cost','QoS Gurantee','QoS Bad','QoS Good','System Throughput','QoS Squared Difference']
        self.dic_mean={i:0 for i in key}
        self.dic_std={i:0 for i in key}
        if cflage: # 1) calculate mean and standard deviation and save
            dic_data={i:[] for i in key}
            for k in range(n):
                print(k,' steps.......')
                c=np.random.choice([i for i in range(self.ori_sizeTable)])
                P=np.random.uniform(0,self.P_Max_SBS,self.nUE)             
                info_ori = self.step_mean_std(c,P)
                Energy_Efficiency_ori,Backhaul_cost_ori,QoS_good_ori,QoS_gurantee_ori,QoS_bad_ori,sum_c_Throughput_ori,QoS_squaredD_ori = info_ori
                dic_data['Energy Efficiency'].append(Energy_Efficiency_ori)
                dic_data['Backhaul Cost'].append(Backhaul_cost_ori)
                dic_data['QoS Gurantee'].append(QoS_gurantee_ori)
                dic_data['QoS Bad'].append(QoS_bad_ori)
                dic_data['QoS Good'].append(QoS_good_ori)
                dic_data['System Throughput'].append(sum_c_Throughput_ori)
                dic_data['QoS Squared Difference'].append(QoS_squaredD_ori)
            for i in key:
                self.dic_mean[i]=np.mean(np.array(dic_data[i]))
                self.dic_std[i]=np.std(np.array(dic_data[i]))
            with open(name,'w',newline='') as csvfile:
                writer = csv.writer(csvfile)  
                writer.writerow([self.dic_mean[i] for i in key])  
                writer.writerow([self.dic_std[i] for i in key])
        else:  # 2) load mean and standard deviation to use
            with open(name, newline='') as csvfile:
                rows = csv.reader(csvfile)
                rows = list(rows)
                for i,name in enumerate(key):
                    self.dic_mean[name]=float(rows[0][i])
                    self.dic_std[name]=float(rows[1][i])   
    
    
    def step_mean_std(self,chosen_c,P):
        # calculate mean and standard deviation
        self.chosen_TP2UE=self.ori_TP2UE[chosen_c]
        self.chosen_UE2TP=self.ori_UE2TP[chosen_c]        
        #1) channel------------------------------------------------------------
        self.channel()
        #2) SubchannelAllocation-----------------------------------------------
        self.SubchannelAllocation()
        #3) R------------------------------------------------------------------
        I = self._Interference(P)        
        SINR = self._SINR(I,P) #array
        SINR = np.clip(SINR,self.SINR_min,self.SINR_max)
        Throughput = self._Throughput(P,SINR)
        n=self.nSBS-sum([1 for i in self.chosen_TP2UE if len(i)==0])             
        # 3-1) check backhaul constraint
        Backhaul_difference = np.array(self.Throughput_BS)-np.array(self.Throughput_SBS_threshold) 
        dic_backhaul={i:dif for i,dif in enumerate(Backhaul_difference) if dif>0}
        Backhaul_cost_ori=0
        # 3-2) correct throughput when violating backhaul constraint --> divide backhaul capacity based on the ratio of transmit power     
        c_Throughput = Throughput.copy()
        for i in dic_backhaul:
            Backhaul_cost_ori=Backhaul_cost_ori+dic_backhaul[i]
            if dic_backhaul[i]>0:
                i_UE = self.chosen_TP2UE[i]
                for k in i_UE:
                    c_Throughput[k]=Throughput[k]*self.Throughput_SBS_threshold[i]/self.Throughput_BS[i]     
        c_Throughput=np.array(c_Throughput)
        Energy_Efficiency_ori = sum(c_Throughput)/(n*self.Pc_SBS+sum(P)) 
        Energy_Efficiency_ori = np.clip(Energy_Efficiency_ori,self.delta_min,self.delta_max)
        QoS_difference = c_Throughput-self.Throughput_UE_threshold       
        QoS_good_ori = sum([i for i in QoS_difference if i>0])
        QoS_bad_ori = sum([-i for i in QoS_difference if i<0])
        QoS_gurantee_ori=QoS_good_ori-QoS_bad_ori
        sum_c_Throughput = sum(c_Throughput)
        QoS_squaredD_ori = sum([i*i for i in QoS_difference])
        # 4)------------------------------------------------------------------- 
        info_ori=(Energy_Efficiency_ori,Backhaul_cost_ori,QoS_good_ori,QoS_gurantee_ori,QoS_bad_ori,sum_c_Throughput,QoS_squaredD_ori)     
        return info_ori 
        
    def step_train(self,chosen_c,P,f_ori_c,f_subc,f_debug,episode,timestep):
        # step for train
        done=False # True if violate backhaul constraint
        QoS_R=0    # 1 if satisfy all UEs' QoS requirement
        #0) Determine cluster--------------------------------------------------
        if f_ori_c ==False:
            self.chosen_TP2UE=self.TP2UE[chosen_c]
            self.chosen_UE2TP=self.UE2TP[chosen_c]  
        else:
            self.chosen_TP2UE=self.ori_TP2UE[chosen_c]
            self.chosen_UE2TP=self.ori_UE2TP[chosen_c]        
        #1) channel for same channel in 1 episode------------------------------
        #self.channel()
        #2) SubchannelAllocation for different cluster-------------------------
        if f_subc == True:
            self.SubchannelAllocation()
        #3) R------------------------------------------------------------------
        if f_debug:
            I = self.debug_Interference(P,episode,'1')  
        else:
            I = self._Interference(P)
        SINR = self._SINR(I,P) #array
        SINRdb = 10*np.log10(np.clip(SINR,self.delta_min,SINR)) 
        Throughput_ori = self._Throughput(P,SINR)
        n=self.nSBS-sum([1 for i in self.chosen_TP2UE if len(i)==0]) 
        # 3-1) check backhaul constraint
        Backhaul_difference = np.array(self.Throughput_BS)-np.array(self.Throughput_SBS_threshold) 
        dic_backhaul={i:dif for i,dif in enumerate(Backhaul_difference) if dif>0}
        Backhaul_cost_ori=0
        # 3-2) correct throughput when violating backhaul constraint --> divide backhaul capacity based on the ratio of transmit power
        c_Throughput_ori = copy.deepcopy(Throughput_ori)
        for i in dic_backhaul:
            Backhaul_cost_ori=Backhaul_cost_ori+dic_backhaul[i]   
            if dic_backhaul[i]>0:
                i_UE = self.chosen_TP2UE[i]
                for k in i_UE:
                    c_Throughput_ori[k]=Throughput_ori[k]*self.Throughput_SBS_threshold[i]/self.Throughput_BS[i]      
        c_Throughput_ori=np.array(c_Throughput_ori)
        sum_Throughput_ori =sum(c_Throughput_ori)
        Energy_Efficiency_ori = sum(c_Throughput_ori)/(n*self.Pc_SBS+sum(P)) 
        Energy_Efficiency_ori = np.clip(Energy_Efficiency_ori,-self.delta_max,self.delta_max)
        QoS_difference = c_Throughput_ori-self.Throughput_UE_threshold 
        QoS_good_ori = sum([i for i in QoS_difference if i>0])
        QoS_bad_ori = sum([-i for i in QoS_difference if i<0])
        QoS_gurantee_ori=QoS_good_ori-QoS_bad_ori 
        QoS_squaredD_ori = sum([i*i for i in QoS_difference])
        # 3-3) standardize
        Energy_Efficiency = (Energy_Efficiency_ori-self.dic_mean['Energy Efficiency'])/self.dic_std['Energy Efficiency']
        Backhaul_cost = Backhaul_cost_ori#(Backhaul_cost_ori-self.dic_mean['Backhaul Cost'])/self.dic_std['Backhaul Cost']
        QoS_gurantee= (QoS_gurantee_ori-self.dic_mean['QoS Gurantee'])/self.dic_std['QoS Gurantee']
        QoS_bad = (QoS_bad_ori-self.dic_mean['QoS Bad'])/self.dic_std['QoS Bad']  
        QoS_good =(QoS_good_ori-self.dic_mean['QoS Good'])/self.dic_std['QoS Good']
        sum_Throughput =(sum_Throughput_ori-self.dic_mean['System Throughput'])/self.dic_std['System Throughput']
        QoS_squaredD =(QoS_squaredD_ori-self.dic_mean['QoS Squared Difference'])/self.dic_std['QoS Squared Difference']
        # check QoS
        if QoS_bad_ori==0:
            QoS_R=1
        else:
            self.debug_QoS['1'][str(episode)][str(timestep)]=[i_UE for i_UE,i in enumerate(QoS_difference) if i>0]
        # check Backhaul
        if Backhaul_cost_ori>0:
            done = True
            self.debug_backhaul['1'][str(episode)][str(timestep)]=[i for i in dic_backhaul if dic_backhaul[i]>0]        
        # 3-4) reward  
        if Backhaul_cost_ori>0:
            done = True 
            R = self.lambda1*Energy_Efficiency-self.lambda2*QoS_squaredD  - 0.1
        else:
            R=self.lambda1*Energy_Efficiency-self.lambda2*QoS_squaredD          
        # 4) next state--------------------------------------------------------
        Ths_=copy.deepcopy(c_Throughput_ori)
        #Ths_=np.clip((Ths_-np.mean(Ths_))/(Ths_.var()**0.5),self.delta_min,self.delta_max)
        #Gs_ = self.G.T.flatten()
        #s_ = np.concatenate((Ths_,  Gs_),axis=0)
        s_ = Ths_
        # 5) info--------------------------------------------------------------        
        info = (R,Energy_Efficiency,Backhaul_cost,QoS_good,QoS_gurantee,QoS_bad,sum_Throughput,QoS_squaredD)
        info_lis=(list(Backhaul_difference),list(SINRdb),list(QoS_difference),list(c_Throughput_ori))
        info_ori=(Energy_Efficiency_ori,Backhaul_cost_ori,QoS_good_ori,QoS_gurantee_ori,QoS_bad_ori,sum_Throughput_ori,QoS_squaredD_ori)
        debug_info=(self.Throughput_SBS_threshold,self.Throughput_BS)
        # 6) debug--------------------------------------------------------------
        if f_debug:
            self.debug_UE_throughput['1'][str(episode)].append(c_Throughput_ori)
            self.debug_SBS_throughput['1'][str(episode)].append([ sum(c_Throughput_ori[BS]) for BS in self.chosen_TP2UE  ] )
            self.debug_SBS_threshold['1'][str(episode)].append(self.Throughput_SBS_threshold )
            self.debug_c['1'][str(episode)].append(self.chosen_UE2TP )
            self.debug_p['1'][str(episode)].append(10*np.log10(P*1000))
        return info,info_lis,s_,info_ori,done,debug_info,QoS_R
    
    def step(self,chosen_c,P,f_ori_c,f_subc,baseline,episode,timestep):
        # step for test
        done=False # True if violate backhaul constraint
        QoS_R=0    # 1 if satisfy all UEs' QoS requirement
        #0) Determine cluster--------------------------------------------------
        if f_ori_c ==False:
            self.chosen_TP2UE=self.TP2UE[chosen_c]
            self.chosen_UE2TP=self.UE2TP[chosen_c]  
        else:
            self.chosen_TP2UE=self.ori_TP2UE[chosen_c]
            self.chosen_UE2TP=self.ori_UE2TP[chosen_c]        
        #1) channel for same channel in 1 episode------------------------------
        #self.channel()
        #2) SubchannelAllocation for different cluster-------------------------
        if f_subc == True:
            self.SubchannelAllocation()
        #3) R------------------------------------------------------------------
        I = self.debug_Interference(P,episode,baseline)  
        SINR = self._SINR(I,P) #array
        SINRdb = 10*np.log10(np.clip(SINR,self.delta_min,SINR)) # cannot np.clip(SINR,self.delta_min,self.delta_max)       
        Throughput_ori = self._Throughput(P,SINR)
        n=self.nSBS-sum([1 for i in self.chosen_TP2UE if len(i)==0]) 
        # 3-1) backhaul
        Backhaul_difference = np.array(self.Throughput_BS)-np.array(self.Throughput_SBS_threshold) 
        dic_backhaul={i:dif for i,dif in enumerate(Backhaul_difference) if dif>0}
        Backhaul_cost_ori=0
        # 3-2) correct throughput
        c_Throughput_ori = copy.deepcopy(Throughput_ori)
        for i in dic_backhaul:
            Backhaul_cost_ori=Backhaul_cost_ori+dic_backhaul[i]   
            if dic_backhaul[i]>0:
                i_UE = self.chosen_TP2UE[i]
                for k in i_UE:
                    c_Throughput_ori[k]=Throughput_ori[k]*self.Throughput_SBS_threshold[i]/self.Throughput_BS[i]      
        c_Throughput_ori=np.array(c_Throughput_ori)
        sum_Throughput_ori =sum(c_Throughput_ori)
        Energy_Efficiency_ori = sum(c_Throughput_ori)/(n*self.Pc_SBS+sum(P)) 
        self.debug_system_throughput[baseline][str(episode)].append(sum(c_Throughput_ori))
        self.debug_system_energy[baseline][str(episode)].append([n*self.Pc_SBS+sum(P),n*self.Pc_SBS,sum(P)]) # overall,operation,transmit
        Energy_Efficiency_ori = np.clip(Energy_Efficiency_ori,-self.delta_max,self.delta_max)
        QoS_difference = c_Throughput_ori-self.Throughput_UE_threshold 
        QoS_good_ori = sum([i for i in QoS_difference if i>0])
        QoS_bad_ori = sum([-i for i in QoS_difference if i<0 ])
        
        QoS_gurantee_ori=QoS_good_ori-QoS_bad_ori 
        QoS_squaredD_ori = sum([i*i for i in QoS_difference])
        # 3-3) standardize
        Energy_Efficiency = (Energy_Efficiency_ori-self.dic_mean['Energy Efficiency'])/self.dic_std['Energy Efficiency']
        Backhaul_cost = Backhaul_cost_ori#(Backhaul_cost_ori-self.dic_mean['Backhaul Cost'])/self.dic_std['Backhaul Cost']
        QoS_gurantee= (QoS_gurantee_ori-self.dic_mean['QoS Gurantee'])/self.dic_std['QoS Gurantee']
        QoS_bad = (QoS_bad_ori-self.dic_mean['QoS Bad'])/self.dic_std['QoS Bad']  
        QoS_good =(QoS_good_ori-self.dic_mean['QoS Good'])/self.dic_std['QoS Good']
        sum_Throughput =(sum_Throughput_ori-self.dic_mean['System Throughput'])/self.dic_std['System Throughput']
        QoS_squaredD =(QoS_squaredD_ori-self.dic_mean['QoS Squared Difference'])/self.dic_std['QoS Squared Difference']
        # check QoS
        if QoS_bad_ori==0:
            QoS_R=1
        else:
            self.debug_QoS['1'][str(episode)][str(timestep)]=[i_UE for i_UE,i in enumerate(QoS_difference) if i>0]
        # check Backhaul
        if Backhaul_cost_ori>0:
            done = True
            self.debug_backhaul['1'][str(episode)][str(timestep)]=[i for i in dic_backhaul if dic_backhaul[i]>0]        
        # 3-4) reward  
        if Backhaul_cost_ori>0:
            done = True 
            R = self.lambda1*Energy_Efficiency-self.lambda2*QoS_squaredD  - 0.1
        else:
            R=self.lambda1*Energy_Efficiency-self.lambda2*QoS_squaredD          
        # 4) next state--------------------------------------------------------
        Ths_=copy.deepcopy(c_Throughput_ori)
        #Ths_=np.clip((Ths_-np.mean(Ths_))/(Ths_.var()**0.5),self.delta_min,self.delta_max)
        #Gs_ = self.G.T.flatten()
        #s_ = np.concatenate((Ths_,  Gs_),axis=0)
        s_ = Ths_
        # 5) info--------------------------------------------------------------        
        info = (R,Energy_Efficiency,Backhaul_cost,QoS_good,QoS_gurantee,QoS_bad,sum_Throughput,QoS_squaredD)
        info_lis=(list(Backhaul_difference),list(SINRdb),list(QoS_difference),list(c_Throughput_ori))
        info_ori=(Energy_Efficiency_ori,Backhaul_cost_ori,QoS_good_ori,QoS_gurantee_ori,QoS_bad_ori,sum_Throughput_ori,QoS_squaredD_ori)
        debug_info=(self.Throughput_SBS_threshold,self.Throughput_BS)
        # 6) debug--------------------------------------------------------------
        self.debug_UE_throughput[baseline][str(episode)].append(c_Throughput_ori)
        self.debug_SBS_throughput[baseline][str(episode)].append([ sum(c_Throughput_ori[BS]) for BS in self.chosen_TP2UE  ] )
        self.debug_SBS_threshold[baseline][str(episode)].append(self.Throughput_SBS_threshold )
        self.debug_c[baseline][str(episode)].append(self.chosen_UE2TP )
        self.debug_p[baseline][str(episode)].append(10*np.log10(P*1000))
        return info,info_lis,s_,info_ori,done,debug_info,QoS_R 
    
    def _pInterference(self,P,iUE,k):
        #UEs use the same subchannel
        iUE=[key for key in iUE if self.B_UE2B[key]==self.B_UE2B[k] ] 
        #3)iG
        iG=[self.G[self.chosen_UE2TP[j],k] for j in iUE]
        #4)iP
        iP=[P[i] for i in iUE]
        #5)I
        interference = np.sum( np.array(iG)*np.array(iP) )
        return interference
    
    def _Interference(self,P):    
        """Interference for ALL UEs"""
        I=[]
        for k in range(self.nUE):
            """1)inter-cell interference"""
            #1)iTP
            iTP=self.chosen_UE2TP[k]
            #2)iUE 
            #UEs in different clusters            
            inter_iUE=[i for i in range(self.nUE)]
            for i in self.chosen_TP2UE[iTP]:
                inter_iUE.remove(i)
            inter_interference=self._pInterference(P,inter_iUE,k)
            """2)intra-cell interference"""
            #UEs in same clusters   
            intra_iUE = self.chosen_TP2UE[iTP].copy()
            intra_iUE.remove(k)
            intra_interference=self._pInterference(P,intra_iUE,k)
            """3)interference"""
            interference = inter_interference+intra_interference
            I.append(interference)
        return I
    
    def debug_Interference(self,P,episode,baseline):    
        """Interference for ALL UEs"""
        I=[]
        for k in range(self.nUE):
            """1)inter-cell interference"""
            #1)iTP
            iTP=self.chosen_UE2TP[k]
            #2)iUE 
            #UEs in different clusters            
            inter_iUE=[i for i in range(self.nUE)]
            for i in self.chosen_TP2UE[iTP]:
                inter_iUE.remove(i)
            inter_interference=self._pInterference(P,inter_iUE,k)
            """2)intra-cell interference"""
            #UEs in same clusters   
            intra_iUE = self.chosen_TP2UE[iTP].copy()
            intra_iUE.remove(k)
            intra_interference=self._pInterference(P,intra_iUE,k)
            """3)interference"""
            interference = inter_interference+intra_interference
            I.append(interference)
            ##########################
            self.debug_I[baseline][str(episode)]['UE'+str(k)].append([interference,intra_interference,inter_interference])
        return I
    
    def _SINR(self,I,P):
        G_UE=[self.G[self.chosen_UE2TP[i],i] for i in range(self.nUE)]
        SINR=np.array(G_UE)*np.array(P)/(self.Noise*self.subB+np.array(I))#np.clip(np.array(G_UE)*np.array(P)/(self.Noise*self.subB+np.array(I)),-self.delta_max,self.delta_max)
        #signal_part=np.array(G_UE)*np.array(P)
        return SINR

    def _Throughput(self,P,SINR):
        # Method 2. ratio
        Throughput=np.log2(1+SINR)      
        self.Throughput_BS=[ sum(Throughput[BS]) for BS in self.chosen_TP2UE  ]        
        """ 1) Equal constraint
        #Throughput_SBS_threshold=np.log2(1+(self.G[:,-1]*self.P_Max_MBS)/(self.Noise*self.B_MBS2SBS))/self.nSBS####################
        #Throughput_SBS_threshold=self.B_MBS2SBS*np.log2(1+(self.G[:,-1]*self.P_Max_MBS)/(self.Noise*self.B_MBS2SBS))       
        """
        # 2) MIMO constraint
        self.Throughput_SBS_threshold = np.log2(1+((self.NT-self.Ng+1)/self.Ng)*( (self.G[:,-1]*self.P_Max_MBS)/(self.Noise*self.subB) ) )      
        return Throughput   

    def baseline1(self):
        # 1)UE choose the nearst SBS
        #the nearest SBS index
        dUE2SBS=[ ((np.array(self.xSBS)-x)**2+(np.array(self.ySBS)-y)**2)**0.5 for x,y in zip(self.xUE,self.yUE)]  
        chosen_UE2TP=[np.argmin(i) for i in dUE2SBS]  
        #the chosen_c
        for key,value in self.ori_UE2TP.items():
            if value==chosen_UE2TP:
                b1_chosen_c=key
        return b1_chosen_c
    
    def baseline2(self):
        # 2)UE choose the SBS with the best channel state  
        chosen_UE2TP=[np.argmax(self.G[:,i]) for i in range(self.nUE)]
        #the chosen_c
        for key,value in self.ori_UE2TP.items():
            if value==chosen_UE2TP:
                b2_chosen_c=key
        return b2_chosen_c
 
    def checkBackhaul(self,P):       
        I = self._Interference(P)        
        SINR = self._SINR(I,P) #array
        _ = self._Throughput(P,SINR)
        # calculate sum rate for all SBSs
        Backhaul_difference = np.array(self.Throughput_BS)-np.array(self.Throughput_SBS_threshold) 
        dic_backhaul={i:dif for i,dif in enumerate(Backhaul_difference) if dif>0}
        Backhaul_cost_ori=0
        # check backhaul constraint   
        for i in dic_backhaul:
            Backhaul_cost_ori=Backhaul_cost_ori+dic_backhaul[i]
        if Backhaul_cost_ori>0:
            violate = True
        else:
            violate = False
        return  violate
    
    def checkQoS(self,Throughput):
        violate = False
        QoS_difference = Throughput - self.Throughput_UE_threshold
        for i in QoS_difference:
            if i<0:
               violate=True
               break
        return violate
    
    def randomP(self,chosen_c,f_ori_c):
        # determine random power that satisfies backhaul constraint
        #0) Determine cluster--------------------------------------------------
        if f_ori_c ==False:
            self.chosen_TP2UE=self.TP2UE[chosen_c]
            self.chosen_UE2TP=self.UE2TP[chosen_c]  
        else:
            self.chosen_TP2UE=self.ori_TP2UE[chosen_c]
            self.chosen_UE2TP=self.ori_UE2TP[chosen_c]
        # 1) SubchannelAllocation----------------------------------------------
        self.SubchannelAllocation()
        # 2) checkBackhaul-----------------------------------------------------
        violate=True
        p_limit=1
        while violate:
            n_step=0   
            if p_limit<0:
                break
            while violate:
                n_step=n_step+1
                P=np.random.uniform(0,self.P_Max_SBS*p_limit,self.nUE)
                violate = self.checkBackhaul(P)
                if n_step >100:
                    break 
            p_limit=p_limit-0.1
        return P
    
    def randomC(self,P):
        # determine random association that satisfies backhaul constraint
        # 1) SubchannelAllocation----------------------------------------------
        self.SubchannelAllocation()
        # 2) checkBackhaul-----------------------------------------------------
        violate=True
        lis=[i for i in range(self.ori_sizeTable)]
        while violate:
            if len(lis)==0:
                return np.random.choice([i for i in range(self.ori_sizeTable)])
            chosen_c=np.random.choice(lis)
            lis.remove(chosen_c)
            self.chosen_TP2UE=self.ori_TP2UE[chosen_c]
            self.chosen_UE2TP=self.ori_UE2TP[chosen_c]
            violate = self.checkBackhaul(P)
        return chosen_c
#%%        
if __name__ == '__main__':


    #%% 1)  (a)create new SBS-UE distribution and (b) calculate mean and standard deviation  
    lambda1=0.43#0.53#1
    lambda2=0.16#0.05#0.42#0.8
    lambda3=0#0.1#0.3#0 
    mean_name='mean_std_cc_ct_0dB_s11_nv51_nobackhaul_new_N3_SBS3_UE5_3v3.csv'#'mean_std_cc_ct_0dB_s3_nv21_oldChannel_nobackhaul.csv'
    scenario_name = 'EnvInfo_3'
    mean_flage=True 
    env = env_PowerAllocation(lambda1=lambda1,lambda2=lambda2,lambda3=lambda3)
    env.load(name=scenario_name)
    #env.new(name=scenario_name)
    env.channel()
    env.writeCSV(scenario_name)
    env.mean_std(10**6,mean_flage,mean_name)
    #%% 2) load (a)the SBS-UE distribution and (b) mean and standard deviation  
    #lambda1=1#0.53#1
    #lambda2=0#0.05#0.42#0.8
    #lambda3=0#0.1#0.3#0 
    #mean_name='mean_std_cc_ct_0dB_s11_nv51_nobackhaul_new_N3_SBS3_UE5.csv'#'mean_std_cc_ct_0dB_s3_nv21_oldChannel_nobackhaul.csv'
    #scenario_name = 'EnvInfo_11'
    #mean_flage=False
    #env = env_PowerAllocation(lambda1=lambda1,lambda2=lambda2,lambda3=lambda3)
    #env.load(name=scenario_name)
    #env.mean_std(10**6,mean_flage,mean_name)

    