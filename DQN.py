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



class DQNActor(nn.Module):

    def __init__(self, state_size, action_size, power_level, hidden_layers=(100,),
                 output_layer_init_std=None, activation="relu", **kwargs):
        super(DQNActor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.activation = activation
        self.power_level=power_level
        #self.state_size-----------------------
        # version 1 (hidden layer >= 2)
        # create layers -------------------------------------------------------
        self.layers = nn.ModuleList()
        # 1-0) state input layer - 1st hidden layer    
        self.state_input_layer = nn.Linear(self.state_size , hidden_layers[0])      
        # 1-1) action input layer - 2nd hidden layer 

        # 1-2) all hidden layer
        nh = len(hidden_layers)
        for i in range(1,nh):
            self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
        # 1-3) the last hidden layer - output layer (action_size) -- A(s,a)
        self.layers.append(nn.Linear(hidden_layers[nh-1], self.action_size))
        # 1-4) the last hidden layer - output layer (1) -- V(s)
        self.value_layer = nn.Linear(hidden_layers[nh-1], self.action_size)
        
        # initialise layer weights --------------------------------------------
        # 1-0) all layers except the last layer -- He initialization / zero initialzation
        nn.init.kaiming_normal_(self.state_input_layer.weight, nonlinearity=activation)
        nn.init.zeros_(self.state_input_layer.bias)
        for i in range(0, len(self.layers) - 1):
            nn.init.kaiming_normal_(self.layers[i].weight, nonlinearity=activation)
            nn.init.zeros_(self.layers[i].bias)
        # 1-1) the last layer for A(s,a) -- normal initialzation / zero initialzation
        nn.init.normal_(self.layers[-1].weight, mean=0., std=output_layer_init_std)
        nn.init.zeros_(self.layers[-1].bias)
        # 1-2) the last layer for V(s) -- normal initialzation / zero initialzation
        nn.init.normal_(self.value_layer.weight, mean=0., std=output_layer_init_std)
        nn.init.zeros_(self.value_layer.bias)
        
        '''
        # version 0
        # create layers -------------------------------------------------------
        self.layers = nn.ModuleList()
        inputSize = self.state_size + self.action_parameter_size #5+210*5
        if hidden_layers is not None:
            nh = len(hidden_layers)
            # 1-0) input layer (inputSize) - 1st hidden layer
            self.layers.append(nn.Linear(inputSize, hidden_layers[0]))
            # 1-1) all hidden layer
            for i in range(1, nh):
                self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            # 1-2) the last hidden layer - output layer (action_size)  
            lastHiddenLayerSize = hidden_layers[nh - 1]
        self.layers.append(nn.Linear(lastHiddenLayerSize, self.action_size))

        # initialise layer weights --------------------------------------------
        # 1-0) all layers except the last layer -- He initialization / zero initialzation
        for i in range(0, len(self.layers) - 1):
            nn.init.kaiming_normal_(self.layers[i].weight, nonlinearity=activation)
            nn.init.zeros_(self.layers[i].bias)
        # 1-1) the last layer -- normal initialzation / zero initialzation
        nn.init.normal_(self.layers[-1].weight, mean=0., std=output_layer_init_std)
        nn.init.zeros_(self.layers[-1].bias)
        '''
    def forward(self, state):
        negative_slope = 0.01 # slope for leaky_relu

        # version 1
        num_layers = len(self.layers)
        if self.activation == "relu":
            # 0-0) state input layer - 1st hidden layer
            x= F.relu(self.state_input_layer(state))
            x = F.relu(self.layers[0](x))
            # 0-1) action input layer + 1st hidden layer
        elif self.activation == "leaky_relu":
            # 0-0) state input layer - 1st hidden layer
            x= F.leaky_relu(self.state_input_layer(state),negative_slope)
            x = F.leaky_relu(self.layers[0](x),negative_slope)
            # 0-1) action input layer + 1st hidden layer
        else:
            raise ValueError("Unknown activation function "+str(self.activation))
        # 0-2) (action input layer + 1st hidden layer) - other hidden layers        
        for i in range(1, num_layers - 1):
            if self.activation == "relu":
                x = F.relu(self.layers[i](x))
            elif self.activation == "leaky_relu":
                x = F.leaky_relu(self.layers[i](x), negative_slope)
            else:
                raise ValueError("Unknown activation function "+str(self.activation))
        # 0-3)  the last hidden layer - output layer ( not pass through activation function  ) 
        V = self.value_layer(x)
        """
        # version 0
        # 1-0) all layers except the last layer -- pass through activation function
        x = torch.cat((state, action_parameters), dim=1)
        num_layers = len(self.layers)
        for i in range(0, num_layers - 1):
            if self.activation == "relu":
                x = F.relu(self.layers[i](x))
            elif self.activation == "leaky_relu":
                x = F.leaky_relu(self.layers[i](x), negative_slope)
            else:
                raise ValueError("Unknown activation function "+str(self.activation))
        # 1-1) the last layer -- not pass through activation function
        Q = self.layers[-1](x)
        """
        return V

#%%

"""   
num_actions=210
action_parameter_size=210*5
s_dim=5
action_input_layer=0# Which layer to input action parameters
layers=[32,16]#(256,)# # Hidden layers 
actor_param_kwargs={'hidden_layers': layers, 'output_layer_init_std': 1e-5,'squashing_function': False}
actor_param = ParamActor(s_dim, num_actions, action_parameter_size, **actor_param_kwargs)
print(actor_param)
"""
#%%
class DQNAgent(Agent):
    #DDPG actor-critic agent for parameterised action spaces [Hausknecht and Stone 2016]

    NAME = "DQN Agent"

    def __init__(self,
                 s_dim,#observation_space,
                 action_space,
                 nUE, power_level,
                 actor_class=DQNActor,
                 actor_kwargs={},
                 epsilon_initial=1.0,
                 epsilon_final=0.05,
                 epsilon_steps=10000,
                 batch_size=64,
                 gamma=0.99,
                 tau_actor=0.01,  # Polyak averaging factor for copying target weights
                 replay_memory_size=1000000,
                 learning_rate_actor=0.0001,
                 initial_memory_threshold=0,
                 use_ornstein_noise=False,  # if false, uses epsilon-greedy with uniform-random action-parameter exploration
                 loss_func=F.mse_loss, # F.mse_loss
                 clip_grad=10,
                 inverting_gradients=False,
                 zero_index_gradients=False,
                 indexed=False,
                 weighted=False,
                 average=False,
                 random_weighted=False,
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 seed=None):
        super(DQNAgent, self).__init__(s_dim, action_space)#observation_space, action_space)
        self.device = torch.device(device)
        self.nUE=nUE
        """
        parameter_min[i] -- np.array
        action_space=(num_action, [(parameter_min[i],parameter_max[i]) for i in range(num_action)])
        """
        self.power_level=power_level
        self.num_actions = self.action_space[0]*(self.power_level**self.nUE) # number of discrete actions
        self.action_max = torch.from_numpy(np.ones((self.num_actions,))).float().to(device)##
        self.action_min = -self.action_max.detach()##
        self.action_range = (self.action_max-self.action_min).detach()##
        #print([self.action_space.spaces[i].high for i in range(1,self.num_actions+1)])
        self.epsilon = epsilon_initial
        self.epsilon_initial = epsilon_initial
        self.epsilon_final = epsilon_final
        self.epsilon_steps = epsilon_steps
        self.indexed = indexed
        self.weighted = weighted
        self.average = average
        self.random_weighted = random_weighted
        assert (weighted ^ average ^ random_weighted) or not (weighted or average or random_weighted)
        #??
        self.batch_size = batch_size
        self.gamma = gamma
        self.replay_memory_size = replay_memory_size
        self.initial_memory_threshold = initial_memory_threshold
        self.learning_rate_actor = learning_rate_actor
        self.inverting_gradients = inverting_gradients
        self.tau_actor = tau_actor
        self._step = 0
        self._episode = 0
        self.updates = 0
        self.clip_grad = clip_grad
        self.zero_index_gradients = zero_index_gradients

        self.np_random = None
        self.seed = seed
        self._seed(seed)
        #??
        self.use_ornstein_noise = use_ornstein_noise

        #print(self.num_actions+self.action_parameter_size)
        """
        observation_space=np.array([Qos_difference of UE 0])
        """
        # 0) Memory
        self.replay_memory =  Memory(replay_memory_size, (s_dim,), (1,), next_actions=False)## #Memory(replay_memory_size, observation_space.shape, (1+self.action_parameter_size,), next_actions=False)
        # 1-1) Actor-eval
        self.actor = actor_class(s_dim, self.num_actions, power_level , **actor_kwargs).to(device)#self.actor = actor_class(self.observation_space.shape[0], self.num_actions, self.action_parameter_size, **actor_kwargs).to(device)
        # 2-2) Actor-target
        self.actor_target = actor_class(s_dim, self.num_actions, power_level, **actor_kwargs).to(device)#self.actor_target = actor_class(self.observation_space.shape[0], self.num_actions, self.action_parameter_size, **actor_kwargs).to(device)
        hard_update_target_network(self.actor, self.actor_target) # directly copy without ratio
        self.actor_target.eval()
        # 2-3) Actor parameter
        # 2-4) Actor Loss Function
        self.loss_func = loss_func  # l1_smooth_loss performs better but original paper used MSE

        # Original DDPG paper [Lillicrap et al. 2016] used a weight decay of 0.01 for Q (critic)
        # but setting weight_decay=0.01 on the critic_optimiser seems to perform worse...
        # using AMSgrad ("fixed" version of Adam, amsgrad=True) doesn't seem to help either...
        self.actor_optimiser = optim.Adam(self.actor.parameters(), lr=self.learning_rate_actor) #, betas=(0.95, 0.999))

    def __str__(self):
        desc = super().__str__() + "\n"
        desc += "Actor Network {}\n".format(self.actor) + \
                "Actor Alpha: {}\n".format(self.learning_rate_actor) + \
                "Gamma: {}\n".format(self.gamma) + \
                "Tau (actor): {}\n".format(self.tau_actor) + \
                "Inverting Gradients: {}\n".format(self.inverting_gradients) + \
                "Replay Memory: {}\n".format(self.replay_memory_size) + \
                "Batch Size: {}\n".format(self.batch_size) + \
                "Initial memory: {}\n".format(self.initial_memory_threshold) + \
                "epsilon_initial: {}\n".format(self.epsilon_initial) + \
                "epsilon_final: {}\n".format(self.epsilon_final) + \
                "epsilon_steps: {}\n".format(self.epsilon_steps) + \
                "Clip Grad: {}\n".format(self.clip_grad) + \
                "Ornstein Noise?: {}\n".format(self.use_ornstein_noise) + \
                "Zero Index Grads?: {}\n".format(self.zero_index_gradients) + \
                "Seed: {}\n".format(self.seed)
        return desc
    
    # initialize parameter(passthrough layer of ActorParam) by user


    def _seed(self, seed=None):
        """
        NOTE: this will not reset the randomly initialised weights; use the seed parameter in the constructor instead.

        :param seed:
        :return:
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.np_random = np.random.RandomState(seed=seed)
        if seed is not None:
            torch.manual_seed(seed)
            if self.device == torch.device("cuda"):
                torch.cuda.manual_seed(seed)

    def start_episode(self):
        pass

    def end_episode(self):
        # adjust epsilon for epsilon-greedy
        self._episode += 1
        ep = self._episode
        if ep < self.epsilon_steps:
            self.epsilon = self.epsilon_initial - (self.epsilon_initial - self.epsilon_final) * (
                    ep / self.epsilon_steps)
        else:
            self.epsilon = self.epsilon_final
            
    # take an action for train =================================================
    def act(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).to(self.device)
            # 0) get action parameters-----------------------------------------
            
            # 1) get discrete action-------------------------------------------
            # Hausknecht and Stone [2016] use epsilon greedy actions with uniform random action-parameter exploration
            rnd = self.np_random.uniform()
            if rnd < self.epsilon:
                action = self.np_random.choice(self.num_actions)

            else:
                # select maximum action
                Q_a = self.actor.forward(state.unsqueeze(0))
                Q_a = Q_a.detach().cpu().data.numpy()
                action = np.argmax(Q_a)
            # 3) add noise-----------------------------------------------------
            # add noise only to parameters of chosen action
            #print('action=',action)
            #print('all_action_parameters=',all_action_parameters.shape)

                #noise = self.noise.sample().reshape(self.num_actions,5)[action,:]
                #action_parameters = action_parameters + noise
                
            
        return action
    
    # take the deterministic action for test ===================================
    def _act(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).to(self.device)
            # 0) get all action parameters-------------------------------------
            # 1) get discrete action (select maximum action)-------------------
            Q_a = self.actor.forward(state.unsqueeze(0))
            Q_a = Q_a.detach().cpu().data.numpy()
            action = np.argmax(Q_a)
            # 3) get action parameters-----------------------------------------          

            #print('act all_action_parameters.shape=',action_parameters.shape)
        return action

    def action_decoder(self, action, max_power):
        cluster=int(action/(self.power_level**self.nUE))
        power=[0 for i in range(self.nUE)]
        temppower=action%(self.power_level**self.nUE)
        idx=self.nUE-1
        while True:
          if idx>0:
            #power[idx]=(temppower%self.power_level+1)/self.power_level*max_power
            power[idx]=1/10**(temppower%self.power_level)*max_power       
            temppower=int(temppower/self.power_level)
            idx=idx-1
          else:
            #power[idx]=(temppower/self.power_level+1)/self.power_level*max_power
            power[idx]=1/10**(temppower/self.power_level)*max_power
            break
            
        power=np.array(power)
        
        return cluster,power



    def step(self, state, action, reward, next_state, next_action, terminal):
        #c1,P1
        act = action
        self._step += 1 # number of agent.step
        #self._step = _step
        # self._add_sample(state, np.concatenate((all_actions.data, all_action_parameters.data)).ravel(), reward, next_state, terminal)
        # 1) Memory -----------------------------------------------------------
        self._add_sample(state, np.array([act]), reward, next_state, np.array([next_action]), terminal=terminal)
        # 2) Update -----------------------------------------------------------
        if self._step >= self.batch_size and self._step >= self.initial_memory_threshold:
            self._optimize_td_loss()
            self.updates += 1
            #self.update = update

    def _add_sample(self, state, action, reward, next_state, next_action, terminal):
        assert len(action) == 1 
        self.replay_memory.append(state, action, reward, next_state, terminal=terminal)

    def _optimize_td_loss(self):
        if self._step < self.batch_size or self._step < self.initial_memory_threshold:
            return
        # 2-1) Sample a batch from replay memory
        states, actions, rewards, next_states, terminals = self.replay_memory.sample(self.batch_size, random_machine=self.np_random)
        # 2-2) form
        states = torch.from_numpy(states).to(self.device)
        actions_combined = torch.from_numpy(actions).to(self.device)  # make sure to separate actions and parameters
        actions = actions_combined.long()
        rewards = torch.from_numpy(rewards).to(self.device).squeeze()
        next_states = torch.from_numpy(next_states).to(self.device)
        terminals = torch.from_numpy(terminals).to(self.device).squeeze()
        # 2-3) Update parameters
        # ---------------------- optimize actor ----------------------
        with torch.no_grad():
            pred_Q_a = self.actor_target(next_states)
            Qprime = torch.max(pred_Q_a, 1, keepdim=True)[0].squeeze()
            # Compute the TD error
            target = rewards + (1 - terminals) * self.gamma * Qprime

        # Compute current Q-values using policy network
        q_values = self.actor(states)
        y_predicted = q_values.gather(1, actions.view(-1, 1)).squeeze()
        y_expected = target
        loss_Q = self.loss_func(y_predicted, y_expected)

        self.actor_optimiser.zero_grad() # 1
        loss_Q.backward() # 2 
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm(self.actor.parameters(), self.clip_grad)
        self.actor_optimiser.step() # 3
        # ---------------------- optimize actor-parameter ----------------------


        # ---------------------- update target-network ------------------------
        soft_update_target_network(self.actor, self.actor_target, self.tau_actor)

    def save_models(self, prefix):
        """
        saves the target actor and critic models
        :param prefix: the count of episodes iterated
        :return:
        """
        torch.save(self.actor.state_dict(), prefix + '_actor.pt')
        print('Models saved successfully')

    def load_models(self, prefix):
        """
        loads the target actor and critic models, and copies them onto actor and critic models
        :param prefix: the count of episodes iterated (used to find the file name)
        :param target: whether to load the target newtwork too (not necessary for evaluation)
        :return:
        """
        # also try load on CPU if no GPU available?
        self.actor.load_state_dict(torch.load(prefix + '_actor.pt', map_location='cpu'))
        print('Models loaded successfully')     

if __name__ == '__main__':
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
    dic_info={key_dic_info:{term: [] for term in dic_info_key} for key_dic_info in ['1','2','3','4','5']  }
    dic_info_no_back={key_dic_info:{term: [] for term in dic_info_key} for key_dic_info in ['1','2','3','4','5']  }
    dic_info_ori_key = ['Energy Efficiency','Backhaul Cost','QoS Good', 'QoS Gurantee', 'QoS Bad','System Throughput','QoS Squared Difference']
    dic_info_ori={key_dic_info:{term: [] for term in dic_info_ori_key} for key_dic_info in ['1','2','3','4','5'] }
    dic_info_ori_no_back={key_dic_info:{term: [] for term in dic_info_ori_key} for key_dic_info in ['1','2','3','4','5'] }
    a_info={'c':[],'P':[]}
    dic_store={'a':[],'ddpg_s':[],'r':[],'dqn_s':[],'dqn_Q':[]}
    dic_NN_output={'actor':[],'critic':[],'dqn_q_eval':[],'dqn_q_target':[]}
    num_back=0
    debug_QoSr={i:[] for i in ['1','2','3','4','5']}
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
    n_baseline=5
    load_dir ="results_53/PDQN_cc_s11_r11_0dB_N3_10"#PDQN_cc_s3_r9_1dB_new4_rebuild40" #Output directory 
    load_num="_done"#"400"#
    layers_actor=[512,128,16] # 1055-- --5  # # Hidden layers
    actor_kwargs={'hidden_layers': layers_actor, 'output_layer_init_std': 1e-5,'action_input_layer': action_input_layer,'activation': "relu"}
    layers_actor_param =[256]#[64,256] # 5-- --1050
    actor_param_kwargs={'hidden_layers': layers_actor_param, 'output_layer_init_std': 1e-5,'squashing_function': False,'activation': "relu"}
    name='mean_std_cc_ct_0dB_s11_nv51_nobackhaul_new_N3_SBS3_UE8.csv'#'mean_std_cc_nct.csv'
    scenario_name='EnvInfo_11'
    lambda1=0.2#0.53#1
    lambda2=0.8#0.05#0.42#0.8
    lambda3=0#0.1#0.3#0
    result_save=load_dir+'/test_testChannel_block_fading'#'/test_all_'#'/test_testChannel'#'/test_last2000_'
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
    read_train_channel_episode = t.readCSI('Rayleigh_CSIforTest_100episode_100timestep_s11',env.nSBS,env.nUE,MAXepisode)

    #%% DQN
    power_level=2
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
    agentDQN.action_decoder(5, env.P_Max_SBS)

#%%


