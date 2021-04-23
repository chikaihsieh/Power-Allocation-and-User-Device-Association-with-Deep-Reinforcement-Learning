#!python3
"""
Created on Sat Jun  1 16:54:41 2019

@author: kuo
"""

import time
import numpy as np
import os
import scipy.stats as st
import copy 
os.chdir('/home/chan/PDQN/') 
from pdqn import PDQNAgent
from env import env_PowerAllocation
import tool as t

#import tool
os.environ['CUDA_VISIBLE_DEVICES']='0'




#%%
if __name__ == '__main__':
 # PDQN=====================================================================
    batch_size=128
    initial_memory_threshold=128 #1000 # Number of transitions required to start learning.
    replay_memory_size=20000     # Replay memory transition capacity 
    epsilon_initial=1
    epsilon_steps=1000 # Number of episodes over which to linearly anneal epsilon
    epsilon_final=0.01 # Final epsilon value
    gamma=0.95
    clip_grad=1 # Parameter gradient clipping limit 
    inverting_gradients=True # Use inverting gradients scheme instead of squashing function
    seed=0 #0 #Random seed
    # 1) ParamActor------------------------------------------------------------
    layers_actor_param =[256]#[64,256]#(256,) # 5-- --1050
    actor_param_kwargs={'hidden_layers': layers_actor_param, 'output_layer_init_std': 1e-5,'squashing_function': False,'activation': "relu"}
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
    layers_actor=[512,128,16]#(256,)# # 1055-- --5  # # Hidden layers
    actor_kwargs={'hidden_layers': layers_actor, 'output_layer_init_std': 1e-5,'action_input_layer': action_input_layer,'activation': "relu"}
    #--------------------------------------------------------------------------
    # Performance
    dic_info_key = ['R','Energy Efficiency','Backhaul Cost','QoS Good','QoS Gurantee','QoS Bad','System Throughput','QoS Squared Difference','Backhaul Difference','SINRdb','QoS Difference','Throughput']
    dic_info={key_dic_info:{term: [] for term in dic_info_key} for key_dic_info in ['1','2','3','4','5','6','7','8']  }
    dic_info_ori_key = ['Energy Efficiency','Backhaul Cost','QoS Good', 'QoS Gurantee', 'QoS Bad','System Throughput','QoS Squared Difference']
    dic_info_ori={key_dic_info:{term: [] for term in dic_info_ori_key} for key_dic_info in ['1','2','3','4','5','6','7','8'] }
    
    a_info={'c':[],'P':[]}
    dic_store={'a':[],'ddpg_s':[],'r':[],'dqn_s':[],'dqn_Q':[]}
    dic_NN_output={'actor':[],'critic':[],'dqn_q_eval':[],'dqn_q_target':[]}
    num_back=0
    QoS_R=[]
    #--------------------------------------------------------------------------
    # debug
    debug_PNN=[]  
    debug_backhaul=[]
    debug_BSbackhaul=[]
    debug_episode_back=[]
    train_channel_episode=[]
    ############################################################################ change this    
    scale_actions = True
    initialise_params = False # True:add pass-through layer to ActorParam and initilize them / False: not add pass-through layer to ActorParam
    use_ornstein_noise=True#True # False: Uniformly sample parameters & add noise to taken parameters / True: greedy parameters 
    save_freq = 100#0 # How often to save models (0 = never)
    title="PDQN_cc_s11_r11_0dB_N3_1"#"PDQN2"#"PDQN_backhaul" # Prefix of output files
    save_dir ="results_PDQN_5v3" #Output directory
    load = False 
    load_dir ="results/"+title+"0"
    load_num="999"
    threshold = 0.005#1e-3
    start_episode=0
    MAXepisode = 100000#600#20000
    MAXstep = 100#150
    # evaluation_episodes=1000 # Episodes over which to evaluate after training
    realization=500#100
    lambda1=0.43#0.53#1
    lambda2=0.16#0.05#0.42#0.8
    lambda3=0#0.1#0.3#0
    mean_name='mean_std_cc_ct_0dB_s11_nv51_nobackhaul_new_N3_SBS3_UE5_3v3.csv'#'mean_std_cc_ct_0dB_s3_nv21_oldChannel_nobackhaul.csv'
    scenario_name = 'EnvInfo_3'
    mean_flage=False
    ###########################################################################
    #%% ENV
    env = env_PowerAllocation(lambda1=lambda1,lambda2=lambda2,lambda3=lambda3,MAXepisode=MAXepisode,n_baseline=1)
    #-------------------------------------------------------------------------- Choose Network Geometry
    env.load(name=scenario_name) # use the previous one 
    #-------------------------------------------------------------------------- mean_std
    env.mean_std(10**6,mean_flage,mean_name)#calculate(True) or load(False)
    num_actions = env.action_space[0]
    s_dim = env.nUE
    #%% PDQN
    # save model --------------------------------------------------------------
    if save_freq > 0 and save_dir:
        save_dir = os.path.join(save_dir, title + "{}".format(str(seed)))
        os.makedirs(save_dir, exist_ok=True)
        
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
    

    # 0) add bias to ActorParm by initialize bias of paaathrough --------------
    # desired bias 
    initial_params_ = list(np.random.uniform(0,env.P_Max_SBS,num_actions*5)) 
    # change the original parameter range to [-1,1]
    if scale_actions:
        for a in range (num_actions*5):
            initial_params_[a] = 2. * (initial_params_[a] - 0) / (env.P_Max_SBS - 0) - 1.
    # initilize bias
    if initialise_params:
        initial_weights = np.zeros((num_actions*5,s_dim))#np.zeros((env.action_space.spaces[0].n, env.observation_space.spaces[0].shape[0]))
        initial_bias = np.zeros(num_actions*5)#np.zeros(env.action_space.spaces[0].n)
        for a in range (num_actions*5):#(env.action_space.spaces[0].n):
            initial_bias[a] = initial_params_[a]
        agent.set_action_parameter_passthrough_weights(initial_weights, initial_bias)
    

    start_time = time.time()
    total_step=start_episode*MAXstep
    cont = True  
    episode=0
    episode_r_list=[]
    #=========================================================================== load existing model to train
    #load_dir='results_53/PDQN_cc_s11_r9_0dB_N3_20'
    #load_num='1_done'
    #agent.load_models(prefix = os.path.join(load_dir, load_num))
    #===========================================================================
    while cont: # episode
        
        episode=episode+1
        print(episode, 'episode--------------------------')
        # save model
        if save_freq > 0 and save_dir and episode % save_freq == 0:
            agent.save_models(os.path.join(save_dir, str(episode)))
        # reset  
        s = env.reset()
        s = np.array(list(s), dtype=np.float32, copy=False)
        # 1) take an action----------------------------------------------------
        c1, PNN1, all_action_parameters = agent.act(s)   
        P1 = t.p_normalize(env.P_Max_SBS,PNN1)

        train_channel_episode.append(env.G) 
        episode_r=[]
        tstep=0
        if total_step>50000:
              break
        while True:  # step 
            tstep = tstep + 1
            total_step = total_step + 1
            print('Iteration '+str(total_step)+'=======================================')
            # 2) step -- next state, reward, done------------------------------   
            #==================================================================
            info1, lis_info1, s_, info_ori1, done1,debug_info1,QoS_R1 = env.step_train(c1,P1,False,True,True,episode-1,tstep-1)
            s_ = np.array(list(s_), dtype=np.float32, copy=False)
            R1,Energy_Efficiency1,Backhaul_cost1,QoS_good1,QoS_gurantee1,QoS_bad1,sum_c_Throughput1,QoS_squaredD1 =info1  
            Energy_Efficiency_ori1,Backhaul_cost_ori1,QoS_good_ori1,QoS_gurantee_ori1,QoS_bad_ori1,sum_c_Throughput_ori1,QoS_squaredD_ori1=info_ori1
            Backhaul_difference1,SINRdb1,QoS_difference1,c_Throughput_ori1 = lis_info1
            Throughput_SBS_threshold,Throughput_BS = debug_info1
            debug_backhaul.append(Throughput_SBS_threshold)
            debug_BSbackhaul.append(Throughput_BS)
            QoS_R.append(QoS_R1)
            #==================================================================
            # 3) take an action------------------------------------------------
            c1_, PNN1_, all_action_parameters_ = agent.act(s_)
            P1_ = t.p_normalize(env.P_Max_SBS,PNN1_)
            # 4) learn---------------------------------------------------------
            agent.step(s, (c1, all_action_parameters), R1, s_,  (c1_, all_action_parameters_), done1 )
            dic_store['a'].append([c1]+P1)
            dic_store['r'].append(R1)
            dic_store['ddpg_s'].append(s)
            debug_PNN.append(PNN1)
            # 5) Print and store info ------------------------------------------
            key_info=['R','Energy Efficiency','Backhaul Cost','QoS Good','QoS Gurantee','QoS Bad','System Throughput','QoS Squared Difference']           
            key_info_lis=['Backhaul Difference','SINRdb','QoS Difference','Throughput']
            
            dic_info = t.inst_info(dic_info,(key_info,key_info_lis),((info1,info1),lis_info1),1)
            dic_info_ori = t.inst_info(dic_info_ori,dic_info_ori_key,(info_ori1,info_ori1),0)
          
            key_inst=['R','Energy Efficiency','Backhaul Cost','QoS Good','QoS Gurantee','QoS Bad','System Throughput','QoS Squared Difference']
            t.print_info((env.UE2TP[c1],P1),s) # print p in dB 
            a_info['c'].append(env.UE2TP[c1])
            a_info['P'].append(10*np.log10(P1*1000))
            
            episode_r.append(R1)
            
            # 6) update --------------------------------------------------------
            c1, P1, all_action_parameters = c1_, P1_, all_action_parameters_
            s = s_
            # number of backhaul constraint violation
            if done1:
                num_back=num_back+1
                debug_episode_back.append(episode)
            # check if end the episode
            if (tstep>=MAXstep) or done1:
                break
            
        agent.end_episode()
        episode_r_list.append(np.mean([episode_r])) 
        # check if end the training   
        if (episode>=MAXepisode) :
            print('MAXepisode')
            cont=False
        if (episode>100):
            m = np.mean([episode_r_list[episode-100:episode-1]])
            not_convergence = [1 for i in episode_r_list[episode-100:episode-1] if abs(i-m)> threshold]
            if sum(not_convergence)==0:
                print('Convergence')
                cont=False
    #%% end training        
    end_time = time.time()
    if episode>=MAXepisode:
        print('MAXepisode')
    else:
        print('episode=',episode)
       
    print("Training took %.2f seconds" % (end_time - start_time))
    print('(violate)num_back=',num_back,'/',total_step,' ',num_back/total_step*100,'%')
    num_QoS=sum([1 for k in QoS_R if k==1 ])
    print('(follow) Qos',num_QoS,'/',total_step,' ',num_QoS/total_step*100,'%')
    
    # debug for constraints about backhaul 
    #debug_episode_back = [i-1 for i in debug_episode_back]
    #t.plot_constraint(MAXepisode,debug_episode_back,'train',save_dir+'/',0)
    #t.writeConstraintHistory(save_dir+'/train_',MAXepisode,debug_episode_back,0)
    #t.plot_constraint(MAXepisode,QoS_R,'train',save_dir+'/',1)
    #t.writeConstraintHistory(save_dir+'/train_',MAXepisode,QoS_R,1)
    
    #%%
    # save model
    if save_freq > 0 and save_dir:
        agent.save_models(os.path.join(save_dir, '_done'))   
   
    #%%
    # 7) Average per realization steps and Save --------------------------------
    key_avg=['R','Energy Efficiency','Backhaul Cost','QoS Good','QoS Gurantee','QoS Bad','System Throughput','QoS Squared Difference']
    dic_avg_info = t.train_avg_info(dic_info,key_avg,realization)
    t.train_plot_avg(dic_avg_info,key_avg,realization,'normalize',save_dir+'/train_')
    #-------------------------------------------------------------------------  
    dic_avg_info_ori = t.train_avg_info(dic_info_ori,dic_info_ori_key,realization)
    t.train_plot_avg(dic_avg_info_ori,dic_info_ori_key,realization,'original',save_dir+'/train_')
    
    

    #%%  
    import matplotlib.pyplot as plt
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
    
    # 8) plot results of each SBS or UE, e.g.Backhaul_difference-SBS, SINR-UE--------------
    key_individual=['Backhaul Difference','SINRdb','QoS Difference', 'Throughput']
    save,save_ori= test_plot_individual(env,dic_info['1'],'1',key_individual,realization,save_dir+'/train_')
    # 9) write info ------------------------------------------------------------
    # average info
    t.writeCSV(dic_avg_info,dic_avg_info_ori,save,dic_info_ori_key,key_individual,key_avg,save_dir+'/train',0)
    # original
    #t.writeCSV(dic_info,dic_info_ori,save_ori,dic_info_ori_key,key_individual,key_avg,save_dir+'/train_original',0)
      
    #%% debug
    t.writeCSI(save_dir+'/CSI',train_channel_episode)
    debug_I = env.debug_I
    debug_UE_throughput = env.debug_UE_throughput # each UE throughput
    debug_SBS_throughput = env.debug_SBS_throughput
    debug_SBS_threshold = env.debug_SBS_threshold
    debug_c = env.debug_c
    debug_p = env.debug_p
    debug_backhaul = env.debug_backhaul
    debug_QoS = env.debug_QoS # which episode and step violate QoS & UE index
    #%% test actual converage range
    threshold=7.5
    m = np.mean([episode_r_list[episode-100:episode-1]])
    not_convergence = [1 for i in episode_r_list[episode-100:episode-1] if abs(i-m)> threshold]
    if sum(not_convergence)==0:
        print('Convergence')
    else:
        print('not')