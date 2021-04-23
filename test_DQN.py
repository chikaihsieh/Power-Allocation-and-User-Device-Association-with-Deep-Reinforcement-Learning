#!python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import Counter
from torch.autograd import Variable
import time
import scipy.stats as st
import copy
import matplotlib.pyplot as plt
import os 
os.chdir('/home/chan/PDQN/') 
os.environ['CUDA_VISIBLE_DEVICES']='1'
from agent import Agent
from memory.memory import Memory
#from memory import Memory
from utils import soft_update_target_network, hard_update_target_network
from utils.noise import OrnsteinUhlenbeckActionNoise
from env import env_PowerAllocation
import tool as t
from pdqn import PDQNAgent
from DQN import DQNAgent






#%%

if __name__ == '__main__':
    # PDQN=====================================================================
    batch_size=128#32
    initial_memory_threshold=128#1000 # Number of transitions required to start learning.
    replay_memory_size=20000 # Replay memory transition capacity 
    epsilon_initial=1
    epsilon_steps=1000 # Number of episodes over which to linearly anneal epsilon
    epsilon_final=0.01 # Final epsilon value
    gamma=0.95
    clip_grad=1 # Parameter gradient clipping limit 
    use_ornstein_noise= False # False: Uniformly sample parameters & add noise to taken parameters / True: greedy parameters 
    inverting_gradients= True # Use inverting gradients scheme instead of squashing function
    seed=0 #Random seed
    save_freq = 100#0 # How often to save models (0 = never)
    # 1) ParamActor------------------------------------------------------------   
    learning_rate_actor_param=0.00001
    tau_actor_param=0.001
    """loss func for actor_parameter """
    average=False # Average weighted loss function  
    weighted=False # Naive weighted loss function
    random_weighted=False # Randomly weighted loss function
    indexed=False # Indexed loss function
    zero_index_gradients=False # Whether to zero all gradients for action-parameters not corresponding to the chosen action
    # 2) Actor-----------------------------------------------------------------
    tau_actor=0.1  
    learning_rate_actor=0.00001#0.0001#0.001 # reduce lr can avoid nan output
    action_input_layer=0# Which layer to input action parameters-- useless?  
    #--------------------------------------------------------------------------
    # Performance 
    dic_info_key = ['R','Energy Efficiency','Backhaul Cost','QoS Good','QoS Gurantee','QoS Bad','System Throughput','QoS Squared Difference','Backhaul Difference','SINRdb','QoS Difference','Throughput']
    dic_info={key_dic_info:{term: [] for term in dic_info_key} for key_dic_info in ['1','2','3','4','5','6']  }
    dic_info_no_back={key_dic_info:{term: [] for term in dic_info_key} for key_dic_info in ['1','2','3','4','5','6']  }
    dic_info_ori_key = ['Energy Efficiency','Backhaul Cost','QoS Good', 'QoS Gurantee', 'QoS Bad','System Throughput','QoS Squared Difference']
    dic_info_ori={key_dic_info:{term: [] for term in dic_info_ori_key} for key_dic_info in ['1','2','3','4','5','6'] }
    dic_info_ori_no_back={key_dic_info:{term: [] for term in dic_info_ori_key} for key_dic_info in ['1','2','3','4','5','6'] }
    a_info={'c':[],'P':[]}
    dic_store={'a':[],'ddpg_s':[],'r':[],'dqn_s':[],'dqn_Q':[]}
    dic_NN_output={'actor':[],'critic':[],'dqn_q_eval':[],'dqn_q_target':[]}
    num_back=0
    debug_QoSr={i:[] for i in ['1','2','3','4','5','6']}
    #--------------------------------------------------------------------------
    # debug
    debug_PNN=[]  
    debug_backhaul=[]
    debug_BSbackhaul=[]
    debug_channel_episode=[]
    debug_episode_back=[]
    debug_s=[]
    
    #%% Need to modify
    ###########################################################################
    scale_actions = True # True
    initialise_params = False#True#False # True:add pass-through layer to ActorParam and initilize them / False: not add pass-through layer to ActorParam
    MAXepisode = 100#1000
    MAXepisode_train = 1000
    MAXstep = 100#10#150
    realization=100#20
    title="PDQN1"#"PDQN_backhaul" # Prefix of output files
    #save_dir ="results" #Output directory 
    n_baseline=6
    load_dir ="results_PDQN_5v3/PDQN_cc_s11_r11_0dB_N3_10"#PDQN_cc_s3_r9_1dB_new4_rebuild40" #Output directory 
    load_num="_done"#"400"#
    load_dirDQN ="results_DQN_5v3/PDQN_cc_s11_r11_0dB_N3_10"#PDQN_cc_s3_r9_1dB_new4_rebuild40" #Output directory 
    load_numDQN="_done"#"400"#
    layers_actor=[512,128,16] # 1055-- --5  # # Hidden layers
    actor_kwargs={'hidden_layers': layers_actor, 'output_layer_init_std': 1e-5,'action_input_layer': action_input_layer,'activation': "relu"}
    layers_actor_param =[256]#[64,256] # 5-- --1050
    actor_param_kwargs={'hidden_layers': layers_actor_param, 'output_layer_init_std': 1e-5,'squashing_function': False,'activation': "relu"}
    name='mean_std_cc_ct_0dB_s11_nv51_nobackhaul_new_N3_SBS3_UE5_3v3.csv'#'mean_std_cc_nct.csv'
    scenario_name='EnvInfo_3'
    lambda1=0.43#0.53#1
    lambda2=0.16#0.05#0.42#0.8
    lambda3=0#0.1#0.3#0
    result_save=load_dirDQN+'/test_testChannel_block_fading'#'/test_all_'#'/test_testChannel'#'/test_last2000_'
    ###########################################################################
    #%% ENV
    env = env_PowerAllocation(lambda1=lambda1,lambda2=lambda2,lambda3=lambda3,MAXepisode=MAXepisode,n_baseline=n_baseline)
    #-------------------------------------------------------------------------- Choose Network Geometry
    #env.reset() # create a new one
    env.load(name=scenario_name) # use the previous one
    #-------------------------------------------------------------------------- mean_std   
    env.mean_std(10**5,False,name)#calculate(True) or load(False)
    num_actions = env.action_space[0]
    s_dim = env.nUE
    # use the same channel gain to test
    read_train_channel_episode = t.readCSI('CSI',env.nSBS,env.nUE,MAXepisode)

    #%% PDQN
    agent_class = PDQNAgent
    agent = agent_class(s_dim=s_dim, action_space=env.action_space,nUE=env.nUE,#observation_space=env.observation_space.spaces[0], action_space=env.action_space,
                        batch_size=batch_size,learning_rate_actor=learning_rate_actor,learning_rate_actor_param=learning_rate_actor_param,  # 0.001
                        epsilon_steps=epsilon_steps,epsilon_final=epsilon_final,gamma=gamma,
                        clip_grad=clip_grad,indexed=indexed,average=average,
                        random_weighted=random_weighted,tau_actor=tau_actor,weighted=weighted,
                        tau_actor_param=tau_actor_param,initial_memory_threshold=initial_memory_threshold,
                        use_ornstein_noise=use_ornstein_noise,replay_memory_size=replay_memory_size,inverting_gradients=inverting_gradients,
                        actor_kwargs=actor_kwargs,actor_param_kwargs=actor_param_kwargs,
                        zero_index_gradients=zero_index_gradients,seed=seed)

    power_level=5
    agent_classDQN = DQNAgent
    agentDQN = agent_classDQN(s_dim=s_dim, action_space=env.action_space,nUE=env.nUE,#observation_space=env.observation_space.spaces[0], action_space=env.action_space,
                        power_level=power_level,batch_size=batch_size,learning_rate_actor=learning_rate_actor,  # 0.001
                        epsilon_steps=epsilon_steps,epsilon_final=epsilon_final,gamma=gamma,
                        clip_grad=clip_grad,indexed=indexed,average=average,
                        random_weighted=random_weighted,tau_actor=tau_actor,weighted=weighted,
                        initial_memory_threshold=initial_memory_threshold,
                        use_ornstein_noise=use_ornstein_noise,replay_memory_size=replay_memory_size,inverting_gradients=inverting_gradients,
                        actor_kwargs=actor_kwargs,
                        zero_index_gradients=zero_index_gradients,seed=seed)
    # load the model 
    agent.load_models(prefix = os.path.join(load_dir, load_num))
    agentDQN.load_models(prefix = os.path.join(load_dirDQN, load_numDQN))
    start_time = time.time()
    total_step=0
    done1 = True
    s = env.reset()
    s = np.array(list(s), dtype=np.float32, copy=False)


    for episode in range(MAXepisode):
        print(episode, 'episode-----------')

        #env.G=read_train_channel_episode[episode]
    
        for timestep in range(MAXstep):
            total_step = total_step + 1
            print('Iteration '+str(total_step)+'=======================================')
            #==================================================================
            # 1
            """ 1) take an action--------------------------------------------"""
            c1, PNN1, all_action_parameters = agent._act(s) # array
            P1 = t.p_normalize(env.P_Max_SBS,PNN1)
            """ 2) step -- next state, reward, done--------------------------"""
            info1, lis_info1, s_, info_ori1, done1,debug_info1,QoS_R1 = env.step(c1,P1,False,True,'1',episode,timestep)
            debug_QoSr['1'].append(QoS_R1)
            s_ = np.array(list(s_), dtype=np.float32, copy=False)
            R1,Energy_Efficiency1,Backhaul_cost1,QoS_good1,QoS_gurantee1,QoS_bad1,sum_c_Throughput1,QoS_squaredD1, =info1  
            Energy_Efficiency_ori1,Backhaul_cost_ori1,QoS_good_ori1,QoS_gurantee_ori1,QoS_bad_ori1,sum_c_Throughput_ori1,QoS_squaredD1_ori1 =info_ori1
            Backhaul_difference1,SINRdb1,QoS_difference1,c_Throughput_ori1 = lis_info1
            Throughput_SBS_threshold,Throughput_BS = debug_info1
            debug_backhaul.append(Throughput_SBS_threshold)
            debug_BSbackhaul.append(Throughput_BS)
            # 2 the nearst SBS + random power allocation-------------------------------------------
            c2 = int(env.baseline1())
            a2 = env.randomP(c2,True)
            info2, lis_info2,_,info_ori2,_ ,_,QoS_R2 = env.step(c2,a2,True,False,'2',episode,timestep)
            debug_QoSr['2'].append(QoS_R2)
            # 3 the best channel + random power allocation ----------------------------------------
            c3 = int(env.baseline2())
            a3 = env.randomP(c3,True)
            info3, lis_info3,_,info_ori3,_,_,QoS_R3= env.step(c3,a3,True,False,'3',episode,timestep)
            debug_QoSr['3'].append(QoS_R3)
            # 4 RL clustering + random power allocatin--------------------------------------------
            c4 = copy.deepcopy(c1)
            a4 = env.randomP(c4,False)
            info4, lis_info4,_,info_ori4,_,_,QoS_R4 = env.step(c4,a4,False,False,'4',episode,timestep) 
            debug_QoSr['4'].append(QoS_R4)
            # 5 random clustering +  RL power---------------------------------------------------
            a5 = copy.deepcopy(P1)
            c5 = env.randomC(a5)
            info5, lis_info5,_,info_ori5,_,_,QoS_R5= env.step(c5,a5,True,False,'5',episode,timestep) 
            debug_QoSr['5'].append(QoS_R5)
            # 6 DQN
            a6 = agentDQN.act(s)   
            c6,P6=agentDQN.action_decoder(a6, env.P_Max_SBS)
            info6, lis_info6, _, info_ori6,_,_,QoS_R6 = env.step(c6,P6,False,True,'6',episode,timestep)
            debug_QoSr['6'].append(QoS_R6)
            #==================================================================
            
            """ 3) Print and store info--------------------------------------"""
            # info=(R,Energy_Efficiency,Backhaul_cost,QoS_good,QoS_gurantee,QoS_bad,SINRdb)
            # lis_info1=(list(Backhaul_difference),list(SINRdb),list(QoS_difference))          
            key_info=['R','Energy Efficiency','Backhaul Cost','QoS Good','QoS Gurantee','QoS Bad','System Throughput','QoS Squared Difference']           
            key_info_lis=['Backhaul Difference','SINRdb','QoS Difference','Throughput']
            
            dic_info = t.test_inst_info(dic_info,(key_info,key_info_lis),((info1,info2,info3,info4,info5,info6),(lis_info1,lis_info2,lis_info3,lis_info4,lis_info5,lis_info6)),1)
            dic_info_ori = t.test_inst_info(dic_info_ori,dic_info_ori_key,(info_ori1,info_ori2,info_ori3,info_ori4,info_ori5,info_ori6),0)
          
            t.test_print_info(((env.UE2TP[c1],P1),(env.ori_UE2TP[c2],a2),(env.ori_UE2TP[c3],a3),(env.UE2TP[c4],a4),(env.ori_UE2TP[c5],a5),(env.ori_UE2TP[c6],P6)),s) # print p in dBm 
            a_info['c'].append(env.UE2TP[c1])
            a_info['P'].append(10*np.log10(P1*1000))
            """ 4) update state ---------------------------------------------"""
            s = s_
            
            # not end the episode at the test phase
            if done1:
                num_back=num_back+1
                debug_episode_back.append(episode)
                print('violate backhaul')               
            else:
                dic_info_no_back = t.test_inst_info(dic_info_no_back,(key_info,key_info_lis),((info1,info2,info3,info4,info5,info6),(lis_info1,lis_info2,lis_info3,lis_info4,lis_info5,lis_info6)),1)
                dic_info_ori_no_back = t.test_inst_info(dic_info_ori_no_back,dic_info_ori_key,(info_ori1,info_ori2,info_ori3,info_ori4,info_ori5,info_ori6),0)

            
    end_time = time.time()
    print('num_back=',num_back,'/',total_step,' ',num_back/total_step*100,'%')
    print("Training took %.2f seconds" % (end_time - start_time))
    for i in debug_QoSr:
        num_QoS=sum([1 for k in debug_QoSr[i] if k==1 ])
        print('[',i,']satify Qos',num_QoS,'/',total_step,' ',num_QoS/total_step*100,'%')
    
    
    #%%  debug for constraints about backhaul   
    #t.plot_constraint(MAXepisode,debug_episode_back,'test',result_save,0)
    #t.writeConstraintHistory(result_save+'test_',MAXepisode,debug_episode_back,0)
    #t.plot_constraint(MAXepisode,debug_QoSr['1'],'test',result_save,1)
    #t.writeConstraintHistory_v2(result_save+'test_',MAXepisode,debug_QoSr,1)

    #%%    
    # 7) Average per realization steps and Save --------------------------------
    key_avg=['R','Energy Efficiency','Backhaul Cost','QoS Good','QoS Gurantee','QoS Bad','System Throughput','QoS Squared Difference']
    dic_avg_info = t.test_avg_info(dic_info,key_avg,realization)
    t.test_plot_avg(dic_avg_info,key_avg,realization,'normalize',result_save)
    #---------------------------------------------------------------------------  
    dic_avg_info_ori = t.test_avg_info(dic_info_ori,dic_info_ori_key,realization)
    t.test_plot_avg(dic_avg_info_ori,dic_info_ori_key,realization,'original',result_save)
    #---------------------------------------------------------------------------
    dic_avg_info_ori_no_back = t.test_avg_info(dic_info_ori_no_back,dic_info_ori_key,realization)
    t.test_plot_avg(dic_avg_info_ori_no_back,dic_info_ori_key,realization,'no_back_original',result_save)
    
    #%%  
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

    #realization=20
    # 8) plot results of each SBS or UE, e.g.Backhaul_difference-SBS, SINR-UE --
    key_individual=['Backhaul Difference','SINRdb','QoS Difference', 'Throughput']
    save=[]
    ori_save=[]
    for i in ['1','2','3','4','5','6']:
        save_avg,save_ori=test_plot_individual(env,dic_info[i],i,key_individual,realization,result_save)
        save.append(save_avg)
        ori_save.append(save_ori)
    # avg info
    t.writeCSV(dic_avg_info,dic_avg_info_ori,save,dic_info_ori_key,key_individual,key_avg,result_save,1)
    # original
    t.writeCSV(dic_info,dic_info_ori,ori_save,dic_info_ori_key,key_individual,key_avg,result_save+'_original',1)
    #%%
    # no_backhaul + original -- don't use individual info
    t.writeCSV_nobackhaul(dic_info_no_back,dic_avg_info_ori_no_back,dic_info_ori_key,key_avg,result_save+'_original_nobackhaul')
    #%%   write CSI
    #t.writeCSI('Rayleigh_CSIforTest_100episode_100timestep_s10',debug_channel_episode)
    #read = readBackhaulHistory('test_HistoryforBackhaulViolation')
    
    #%% debug
    test_debug_I = env.debug_I
    test_debug_UE_throughput = env.debug_UE_throughput # each UE throughput
    test_debug_SBS_throughput = env.debug_SBS_throughput
    test_debug_SBS_threshold = env.debug_SBS_threshold
    test_debug_c = env.debug_c
    test_debug_p = env.debug_p
    test_debug_backhaul = env.debug_backhaul
    test_debug_QoS = env.debug_QoS # which episode and step violate QoS & UE index
    test_debug_system_throughput = env.debug_system_throughput
    test_debug_system_energy = env.debug_system_energy
    # 1) EE    
    debug_dic_info_EE_key = ['System Power','Operational Power','Transmit Power','System Throughput']
    debug_dic_info_EE = {key_dic_info:{name_EE:[] for name_EE in debug_dic_info_EE_key} for key_dic_info in ['1','2','3','4','5','6'] }

    for iMETHOD in ['1','2','3','4','5','6']:
        for index,nameEE in enumerate(debug_dic_info_EE_key[:3]):
            debug_dic_info_EE[iMETHOD][nameEE]=[episode_EE[index] for episode in range(MAXepisode) for episode_EE in test_debug_system_energy[iMETHOD][str(episode)] ]          
        debug_dic_info_EE[iMETHOD]['System Throughput']=[episode_EE for episode in range(MAXepisode) for episode_EE in test_debug_system_throughput[iMETHOD][str(episode)] ]

    t.writeEE(debug_dic_info_EE,debug_dic_info_EE_key,result_save)
    
    # 2) Interference
    debug_dic_info_I_key = ['Interference','Intra-cluster Interference','Inter-cluster Interference']
    debug_I={i:{'UE'+str(j):[] for j in range(env.nUE)} for i in debug_dic_info_I_key} # I, intra-cluster, inter-cluster
    debug_dic_info_I = {key_dic_info:copy.deepcopy(debug_I) for key_dic_info in ['1','2','3','4','5','6'] }
    
    for index,name_I in enumerate(debug_dic_info_I_key):
        for iUE in ['UE'+str(i) for i in range(env.nUE) ]:
            for iMETHOD in [str(k+1) for k in range(6)]:
                debug_dic_info_I[iMETHOD][name_I][iUE]=[episode_I[index] for episode in range(MAXepisode) for episode_I in copy.deepcopy(test_debug_I[iMETHOD][str(episode)][iUE])]

    t.writeI(debug_dic_info_I,debug_dic_info_I_key,env.nUE,result_save)
    
    # 3) action
    debug_dic_info_action_key = ['Association','Power Allocation']
    debug_dic_info_action = {key_dic_info:{name_action:[] for name_action in debug_dic_info_action_key} for key_dic_info in [str(i+1) for i in range(n_baseline)]  }

    for iMETHOD in [str(i+1) for i in range(n_baseline)] :
        debug_dic_info_action[iMETHOD]['Association'] = [ test_debug_c[iMETHOD][str(episode)] for episode in range(MAXepisode) ]
        debug_dic_info_action[iMETHOD]['Power Allocation'] = [ test_debug_p[iMETHOD][str(episode)] for episode in range(MAXepisode) ]
                          
    t.writeAction(debug_dic_info_action,debug_dic_info_action_key,result_save)
