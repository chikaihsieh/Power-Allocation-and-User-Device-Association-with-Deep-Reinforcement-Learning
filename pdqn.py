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



class QActor(nn.Module):

    def __init__(self, state_size, action_size, action_parameter_size, hidden_layers=(100,), action_input_layer=0,
                 output_layer_init_std=None, activation="relu", **kwargs):
        super(QActor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.action_parameter_size = action_parameter_size
        self.activation = activation
        
        #self.state_size-----------------------
        # version 1 (hidden layer >= 2)
        # create layers -------------------------------------------------------
        self.layers = nn.ModuleList()
        # 1-0) state input layer - 1st hidden layer    
        self.state_input_layer = nn.Linear(self.state_size , hidden_layers[0])      
        # 1-1) action input layer - 2nd hidden layer 
        self.action_input_layer = nn.Linear(self.action_parameter_size , hidden_layers[1])
        # 1-2) all hidden layer
        nh = len(hidden_layers)
        for i in range(1,nh):
            self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
        # 1-3) the last hidden layer - output layer (action_size) -- A(s,a)
        self.layers.append(nn.Linear(hidden_layers[nh-1], self.action_size))
        # 1-4) the last hidden layer - output layer (1) -- V(s)
        self.value_layer = nn.Linear(hidden_layers[nh-1], 1)
        
        # initialise layer weights --------------------------------------------
        # 1-0) all layers except the last layer -- He initialization / zero initialzation
        nn.init.kaiming_normal_(self.state_input_layer.weight, nonlinearity=activation)
        nn.init.zeros_(self.state_input_layer.bias)
        nn.init.kaiming_normal_(self.action_input_layer.weight, nonlinearity=activation)
        nn.init.zeros_(self.action_input_layer.bias)
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
    def forward(self, state, action_parameters):
        negative_slope = 0.01 # slope for leaky_relu

        # version 1
        num_layers = len(self.layers)
        if self.activation == "relu":
            # 0-0) state input layer - 1st hidden layer
            x= F.relu(self.state_input_layer(state))
            x = F.relu(self.layers[0](x))
            # 0-1) action input layer + 1st hidden layer
            x = F.relu(self.action_input_layer(action_parameters)) + x 
        elif self.activation == "leaky_relu":
            # 0-0) state input layer - 1st hidden layer
            x= F.leaky_relu(self.state_input_layer(state),negative_slope)
            x = F.leaky_relu(self.layers[0](x),negative_slope)
            # 0-1) action input layer + 1st hidden layer
            x = F.leaky_relu(self.action_input_layer(action_parameters),negative_slope) + x
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
        A = self.layers[-1](x)
        Q =  V + A
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
        return Q

#%%
class ParamActor(nn.Module):

    def __init__(self, state_size, action_size, action_parameter_size, hidden_layers, squashing_function=False,
                 output_layer_init_std=None, init_type="kaiming", activation="relu", init_std=None):
        super(ParamActor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.action_parameter_size = action_parameter_size
        self.squashing_function = squashing_function
        self.activation = activation
        if init_type == "normal":
            assert init_std is not None and init_std > 0
        assert self.squashing_function is False  # unsupported, cannot get scaling right yet

        # create layers -------------------------------------------------------        
        self.layers = nn.ModuleList()
        inputSize = self.state_size # 5
        if hidden_layers is not None:
            nh = len(hidden_layers)
            # 0-0) input layer (inputSize) - 1st hidden layer
            self.layers.append(nn.Linear(inputSize, hidden_layers[0]))
            # 0-1) all hidden layer
            for i in range(1, nh):
                self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            # 0-2) the last hidden layer - output layer (action_size)
            lastHiddenLayerSize = hidden_layers[nh - 1]
        self.action_parameters_output_layer = nn.Linear(lastHiddenLayerSize, self.action_parameter_size) 
        # 1-0)  why need?
        self.action_parameters_passthrough_layer = nn.Linear(self.state_size, self.action_parameter_size) 

        # initialise layer weights --------------------------------------------
        for i in range(0, len(self.layers)):
            # 0-0) all layers except the last layer -- He initialization or normal / zero initialzation
            if init_type == "kaiming":
                nn.init.kaiming_normal_(self.layers[i].weight, nonlinearity=activation)
            elif init_type == "normal":
                nn.init.normal_(self.layers[i].weight, std=init_std)
            else:
                raise ValueError("Unknown init_type "+str(init_type))
            nn.init.zeros_(self.layers[i].bias)
        # 0-1) the last layer -- normal initialzation / zero initialzation
        if output_layer_init_std is not None:
            nn.init.normal_(self.action_parameters_output_layer.weight, std=output_layer_init_std)
        else:
            nn.init.zeros_(self.action_parameters_output_layer.weight)
        nn.init.zeros_(self.action_parameters_output_layer.bias)
        # 1-0) initialize zero, can use "set_action_parameter_passthrough_weights" to initialize again
        nn.init.zeros_(self.action_parameters_passthrough_layer.weight)
        nn.init.zeros_(self.action_parameters_passthrough_layer.bias)

        # fix passthrough layer to avoid instability, rest of network can compensate ??
        self.action_parameters_passthrough_layer.requires_grad = False
        self.action_parameters_passthrough_layer.weight.requires_grad = False
        self.action_parameters_passthrough_layer.bias.requires_grad = False

    def forward(self, state):
        # forward -------------------------------------------------------------
        x = state
        negative_slope = 0.01# slope for leaky_relu
        # 0-0) all layers except the last layer -- pass through activation function
        num_hidden_layers = len(self.layers) 
        for i in range(0, num_hidden_layers):
            if self.activation == "relu":
                x = F.relu(self.layers[i](x))
            elif self.activation == "leaky_relu":
                x = F.leaky_relu(self.layers[i](x), negative_slope)
            else:
                raise ValueError("Unknown activation function "+str(self.activation))
        # 0-1) the last layer -- not pass through activation function
        action_params = self.action_parameters_output_layer(x)
        # 1-0) 
        action_params += self.action_parameters_passthrough_layer(state)
        
        # limit action_params -------------------------------------------------
        # 0) use tanh
        if self.squashing_function:
            assert False  # scaling not implemented yet
            action_params = action_params.tanh()
            action_params = action_params * self.action_param_lim
        # 1) use sigmoid
        #action_params = F.sigmoid(action_params)###############
        action_params = torch.sigmoid(action_params)
        
        return action_params 
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
class PDQNAgent(Agent):
    #DDPG actor-critic agent for parameterised action spaces [Hausknecht and Stone 2016]

    NAME = "P-DQN Agent"

    def __init__(self,
                 s_dim,#observation_space,
                 action_space,
                 nUE,
                 actor_class=QActor,
                 actor_kwargs={},
                 actor_param_class=ParamActor,
                 actor_param_kwargs={},
                 epsilon_initial=1.0,
                 epsilon_final=0.05,
                 epsilon_steps=10000,
                 batch_size=64,
                 gamma=0.99,
                 tau_actor=0.01,  # Polyak averaging factor for copying target weights
                 tau_actor_param=0.001,
                 replay_memory_size=1000000,
                 learning_rate_actor=0.0001,
                 learning_rate_actor_param=0.00001,
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
        super(PDQNAgent, self).__init__(s_dim, action_space)#observation_space, action_space)
        self.device = torch.device(device)
        self.nUE=nUE
        """
        parameter_min[i] -- np.array
        action_space=(num_action, [(parameter_min[i],parameter_max[i]) for i in range(num_action)])
        """
        
        self.num_actions = self.action_space[0] # number of discrete actions
        self.action_parameter_sizes = np.array([self.action_space[1][i][0].shape for i in range(self.num_actions) ])#np.array([self.action_space.spaces[i].shape[0] for i in range(1,self.num_actions+1)])
        self.action_parameter_size = int(self.action_parameter_sizes.sum()) #210-----------------------
        self.action_max = torch.from_numpy(np.ones((self.num_actions,))).float().to(device)
        self.action_min = -self.action_max.detach()
        self.action_range = (self.action_max-self.action_min).detach()
        #print([self.action_space.spaces[i].high for i in range(1,self.num_actions+1)])
        self.action_parameter_max_numpy = np.concatenate([self.action_space[1][i][1] for i in range(self.num_actions)]).ravel()#np.concatenate([self.action_space.spaces[i].high for i in range(1,self.num_actions+1)]).ravel()
        self.action_parameter_min_numpy = np.concatenate([self.action_space[1][i][0] for i in range(self.num_actions)]).ravel()#np.concatenate([self.action_space.spaces[i].low for i in range(1,self.num_actions+1)]).ravel()
        self.action_parameter_range_numpy = (self.action_parameter_max_numpy - self.action_parameter_min_numpy)
        self.action_parameter_max = torch.from_numpy(self.action_parameter_max_numpy).float().to(device)
        self.action_parameter_min = torch.from_numpy(self.action_parameter_min_numpy).float().to(device)
        self.action_parameter_range = torch.from_numpy(self.action_parameter_range_numpy).float().to(device)
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
        self.action_parameter_offsets = self.action_parameter_sizes.cumsum()
        self.action_parameter_offsets = np.insert(self.action_parameter_offsets, 0, 0)

        self.batch_size = batch_size
        self.gamma = gamma
        self.replay_memory_size = replay_memory_size
        self.initial_memory_threshold = initial_memory_threshold
        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_actor_param = learning_rate_actor_param
        self.inverting_gradients = inverting_gradients
        self.tau_actor = tau_actor
        self.tau_actor_param = tau_actor_param
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
        self.noise = OrnsteinUhlenbeckActionNoise(self.action_parameter_size, random_machine=self.np_random, mu=0., theta=0.15, sigma=0.0001) #, theta=0.01, sigma=0.01)

        #print(self.num_actions+self.action_parameter_size)
        """
        observation_space=np.array([Qos_difference of UE 0])
        """
        # 0) Memory
        self.replay_memory =  Memory(replay_memory_size, (s_dim,), (1+self.action_parameter_size,), next_actions=False)#Memory(replay_memory_size, observation_space.shape, (1+self.action_parameter_size,), next_actions=False)
        # 1-1) Actor-eval
        self.actor = actor_class(s_dim, self.num_actions, self.action_parameter_size, **actor_kwargs).to(device)#self.actor = actor_class(self.observation_space.shape[0], self.num_actions, self.action_parameter_size, **actor_kwargs).to(device)
        # 2-2) Actor-target
        self.actor_target = actor_class(s_dim, self.num_actions, self.action_parameter_size, **actor_kwargs).to(device)#self.actor_target = actor_class(self.observation_space.shape[0], self.num_actions, self.action_parameter_size, **actor_kwargs).to(device)
        hard_update_target_network(self.actor, self.actor_target) # directly copy without ratio
        self.actor_target.eval()
        # 2-3) Actor parameter
        self.actor_param = actor_param_class(s_dim, self.num_actions, self.action_parameter_size, **actor_param_kwargs).to(device)#self.actor_param = actor_param_class(self.observation_space.shape[0], self.num_actions, self.action_parameter_size, **actor_param_kwargs).to(device)
        self.actor_param_target = actor_param_class(s_dim, self.num_actions, self.action_parameter_size, **actor_param_kwargs).to(device)#self.actor_param_target = actor_param_class(self.observation_space.shape[0], self.num_actions, self.action_parameter_size, **actor_param_kwargs).to(device)
        hard_update_target_network(self.actor_param, self.actor_param_target)
        self.actor_param_target.eval()
        # 2-4) Actor Loss Function
        self.loss_func = loss_func  # l1_smooth_loss performs better but original paper used MSE

        # Original DDPG paper [Lillicrap et al. 2016] used a weight decay of 0.01 for Q (critic)
        # but setting weight_decay=0.01 on the critic_optimiser seems to perform worse...
        # using AMSgrad ("fixed" version of Adam, amsgrad=True) doesn't seem to help either...
        self.actor_optimiser = optim.Adam(self.actor.parameters(), lr=self.learning_rate_actor) #, betas=(0.95, 0.999))
        self.actor_param_optimiser = optim.Adam(self.actor_param.parameters(), lr=self.learning_rate_actor_param) #, betas=(0.95, 0.999)) #, weight_decay=critic_l2_reg)

    def __str__(self):
        desc = super().__str__() + "\n"
        desc += "Actor Network {}\n".format(self.actor) + \
                "Param Network {}\n".format(self.actor_param) + \
                "Actor Alpha: {}\n".format(self.learning_rate_actor) + \
                "Actor Param Alpha: {}\n".format(self.learning_rate_actor_param) + \
                "Gamma: {}\n".format(self.gamma) + \
                "Tau (actor): {}\n".format(self.tau_actor) + \
                "Tau (actor-params): {}\n".format(self.tau_actor_param) + \
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
    def set_action_parameter_passthrough_weights(self, initial_weights, initial_bias=None):
        # 0) check size is the same
        passthrough_layer = self.actor_param.action_parameters_passthrough_layer
        print('weight shape of passthrough_layer',passthrough_layer.weight.data.size())
        print('initial_weights shape ',initial_weights.shape)
        print('bias shape of passthrough_layer',passthrough_layer.bias.data.size())
        print('initial_bias shape ',initial_bias.shape)       
        assert initial_weights.shape == passthrough_layer.weight.data.size()
        # 1) initialize weight
        passthrough_layer.weight.data = torch.Tensor(initial_weights).float().to(self.device)
        # 2) initialize bias
        if initial_bias is not None:
            assert initial_bias.shape == passthrough_layer.bias.data.size()
            passthrough_layer.bias.data = torch.Tensor(initial_bias).float().to(self.device)
        # 3) not optimize parameters of passthrough layer of ActorParam
        passthrough_layer.requires_grad = False
        passthrough_layer.weight.requires_grad = False
        passthrough_layer.bias.requires_grad = False
        hard_update_target_network(self.actor_param, self.actor_param_target)

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

    def _ornstein_uhlenbeck_noise(self, all_action_parameters):
        """ Continuous action exploration using an  ornstein_uhlenbeck_noise"""
        return all_action_parameters.data.numpy() + (self.noise.sample() * self.action_parameter_range_numpy)

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
            all_action_parameters = self.actor_param.forward(state)
            # 1) get discrete action-------------------------------------------
            # Hausknecht and Stone [2016] use epsilon greedy actions with uniform random action-parameter exploration
            rnd = self.np_random.uniform()
            if rnd < self.epsilon:
                action = self.np_random.choice(self.num_actions)
                if not self.use_ornstein_noise:
                    all_action_parameters = torch.from_numpy(np.random.uniform(self.action_parameter_min_numpy,self.action_parameter_max_numpy))
            else:
                # select maximum action
                Q_a = self.actor.forward(state.unsqueeze(0), all_action_parameters.unsqueeze(0))
                Q_a = Q_a.detach().cpu().data.numpy()
                action = np.argmax(Q_a)
            # 3) add noise-----------------------------------------------------
            # add noise only to parameters of chosen action
            all_action_parameters = all_action_parameters.cpu().data.numpy()
            #print('action=',action)
            #print('all_action_parameters=',all_action_parameters.shape)
            
            if self.use_ornstein_noise and self.noise is not None:
                all_action_parameters[action*self.nUE:(action+1)*self.nUE]=all_action_parameters[action*self.nUE:(action+1)*self.nUE]+self.noise.sample()[action*self.nUE:(action+1)*self.nUE]
                #noise = self.noise.sample().reshape(self.num_actions,5)[action,:]
                #action_parameters = action_parameters + noise
                
            action_parameters=all_action_parameters.reshape(self.num_actions,self.nUE)[action,:]
            
        return action, action_parameters, all_action_parameters
    
    # take the deterministic action for test ===================================
    def _act(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).to(self.device)
            # 0) get all action parameters-------------------------------------
            all_action_parameters = self.actor_param.forward(state)
            # 1) get discrete action (select maximum action)-------------------
            Q_a = self.actor.forward(state.unsqueeze(0), all_action_parameters.unsqueeze(0))
            Q_a = Q_a.detach().cpu().data.numpy()
            action = np.argmax(Q_a)
            # 3) get action parameters-----------------------------------------          
            all_action_parameters = all_action_parameters.cpu().data.numpy()
            action_parameters=all_action_parameters.reshape(self.num_actions,self.nUE)[action,:]
            #print('act all_action_parameters.shape=',action_parameters.shape)
        return action, action_parameters, all_action_parameters

    def _zero_index_gradients(self, grad, batch_action_indices, inplace=True):
        assert grad.shape[0] == batch_action_indices.shape[0]
        grad = grad.cpu()

        if not inplace:
            grad = grad.clone()
        with torch.no_grad():
            ind = torch.zeros(self.action_parameter_size, dtype=torch.long)
            for a in range(self.num_actions):
                ind[self.action_parameter_offsets[a]:self.action_parameter_offsets[a+1]] = a
            # ind_tile = np.tile(ind, (self.batch_size, 1))
            ind_tile = ind.repeat(self.batch_size, 1).to(self.device)
            actual_index = ind_tile != batch_action_indices[:, np.newaxis]
            grad[actual_index] = 0.
        return grad

    def _invert_gradients(self, grad, vals, grad_type, inplace=True):
        # 5x faster on CPU (for Soccer, slightly slower for Goal, Platform?)
        if grad_type == "actions":
            max_p = self.action_max
            min_p = self.action_min
            rnge = self.action_range
        elif grad_type == "action_parameters":
            max_p = self.action_parameter_max
            min_p = self.action_parameter_min
            rnge = self.action_parameter_range
        else:
            raise ValueError("Unhandled grad_type: '"+str(grad_type) + "'")

        max_p = max_p.cpu()
        min_p = min_p.cpu()
        rnge = rnge.cpu()
        grad = grad.cpu()
        vals = vals.cpu()

        assert grad.shape == vals.shape

        if not inplace:
            grad = grad.clone()
        with torch.no_grad():
            # index = grad < 0  # actually > but Adam minimises, so reversed (could also double negate the grad)
            index = grad > 0
            grad[index] *= (index.float() * (max_p - vals) / rnge)[index]
            grad[~index] *= ((~index).float() * (vals - min_p) / rnge)[~index]

        return grad

    def step(self, state, action, reward, next_state, next_action, terminal):
        #c1,P1
        act, all_action_parameters = action
        self._step += 1 # number of agent.step
        #self._step = _step
        # self._add_sample(state, np.concatenate((all_actions.data, all_action_parameters.data)).ravel(), reward, next_state, terminal)
        # 1) Memory -----------------------------------------------------------
        self._add_sample(state, np.concatenate(([act],all_action_parameters)).ravel(), reward, next_state, np.concatenate(([next_action[0]],next_action[1])).ravel(), terminal=terminal)
        # 2) Update -----------------------------------------------------------
        if self._step >= self.batch_size and self._step >= self.initial_memory_threshold:
            self._optimize_td_loss()
            self.updates += 1
            #self.update = update

    def _add_sample(self, state, action, reward, next_state, next_action, terminal):
        assert len(action) == 1 + self.action_parameter_size
        self.replay_memory.append(state, action, reward, next_state, terminal=terminal)

    def _optimize_td_loss(self):
        if self._step < self.batch_size or self._step < self.initial_memory_threshold:
            return
        # 2-1) Sample a batch from replay memory
        states, actions, rewards, next_states, terminals = self.replay_memory.sample(self.batch_size, random_machine=self.np_random)
        # 2-2) form
        states = torch.from_numpy(states).to(self.device)
        actions_combined = torch.from_numpy(actions).to(self.device)  # make sure to separate actions and parameters
        actions = actions_combined[:, 0].long()
        action_parameters = actions_combined[:, 1:]
        rewards = torch.from_numpy(rewards).to(self.device).squeeze()
        next_states = torch.from_numpy(next_states).to(self.device)
        terminals = torch.from_numpy(terminals).to(self.device).squeeze()
        # 2-3) Update parameters
        # ---------------------- optimize actor ----------------------
        with torch.no_grad():
            pred_next_action_parameters = self.actor_param_target.forward(next_states)
            pred_Q_a = self.actor_target(next_states, pred_next_action_parameters)
            Qprime = torch.max(pred_Q_a, 1, keepdim=True)[0].squeeze()
            # Compute the TD error
            target = rewards + (1 - terminals) * self.gamma * Qprime

        # Compute current Q-values using policy network
        q_values = self.actor(states, action_parameters)
        y_predicted = q_values.gather(1, actions.view(-1, 1)).squeeze()
        y_expected = target
        loss_Q = self.loss_func(y_predicted, y_expected)

        self.actor_optimiser.zero_grad() # 1
        loss_Q.backward() # 2 
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm(self.actor.parameters(), self.clip_grad)
        self.actor_optimiser.step() # 3
        # ---------------------- optimize actor-parameter ----------------------
        with torch.no_grad():
            action_params = self.actor_param(states)
        action_params.requires_grad = True
        assert (self.weighted ^ self.average ^ self.random_weighted) or \
               not (self.weighted or self.average or self.random_weighted)
        Q = self.actor(states, action_params)
        Q_val = Q
        if self.weighted:
            # approximate categorical probability density (i.e. counting)
            counts = Counter(actions.cpu().numpy())
            weights = torch.from_numpy(
                np.array([counts[a] / actions.shape[0] for a in range(self.num_actions)])).float().to(self.device)
            Q_val = weights * Q
        elif self.average:
            Q_val = Q / self.num_actions
        elif self.random_weighted:
            weights = np.random.uniform(0, 1., self.num_actions)
            weights /= np.linalg.norm(weights)
            weights = torch.from_numpy(weights).float().to(self.device)
            Q_val = weights * Q
        if self.indexed:
            Q_indexed = Q_val.gather(1, actions.unsqueeze(1))
            Q_loss = torch.mean(Q_indexed)
        else:
            Q_loss = torch.mean(torch.sum(Q_val, 1))
        self.actor.zero_grad() # 1
        Q_loss.backward() #2
        from copy import deepcopy
        delta_a = deepcopy(action_params.grad.data)
        # step 2
        action_params = self.actor_param(Variable(states))
        delta_a[:] = self._invert_gradients(delta_a, action_params, grad_type="action_parameters", inplace=True)
        if self.zero_index_gradients:
            delta_a[:] = self._zero_index_gradients(delta_a, batch_action_indices=actions, inplace=True)

        out = -torch.mul(delta_a, action_params)
        self.actor_param.zero_grad()
        out.backward(torch.ones(out.shape).to(self.device))
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm(self.actor_param.parameters(), self.clip_grad)
        self.actor_param_optimiser.step() #3
        # ---------------------- update target-network ------------------------
        soft_update_target_network(self.actor, self.actor_target, self.tau_actor)
        soft_update_target_network(self.actor_param, self.actor_param_target, self.tau_actor_param)

    def save_models(self, prefix):
        """
        saves the target actor and critic models
        :param prefix: the count of episodes iterated
        :return:
        """
        torch.save(self.actor.state_dict(), prefix + '_actor.pt')
        torch.save(self.actor_param.state_dict(), prefix + '_actor_param.pt')
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
        self.actor_param.load_state_dict(torch.load(prefix + '_actor_param.pt', map_location='cpu'))
        print('Models loaded successfully')     




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
    load_dir ="results_PDQN_5v3/PDQN_cc_s11_r11_0dB_N3_10"#PDQN_cc_s3_r9_1dB_new4_rebuild40" #Output directory 
    load_num="_done"#"400"#
    layers_actor=[512,128,16] # 1055-- --5  # # Hidden layers
    actor_kwargs={'hidden_layers': layers_actor, 'output_layer_init_std': 1e-5,'action_input_layer': action_input_layer,'activation': "relu"}
    layers_actor_param =[256]#[64,256] # 5-- --1050
    actor_param_kwargs={'hidden_layers': layers_actor_param, 'output_layer_init_std': 1e-5,'squashing_function': False,'activation': "relu"}
    name='mean_std_cc_ct_0dB_s11_nv51_nobackhaul_new_N3_SBS3_UE5_3v3.csv'#'mean_std_cc_nct.csv'
    scenario_name='EnvInfo_3'
    lambda1=0.43#0.53#1
    lambda2=0.16#0.05#0.42#0.8
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
    #read_train_channel_episode = t.readCSI('CSI',env.nSBS,env.nUE,MAXepisode)

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

    # load the model 
    agent.load_models(prefix = os.path.join(load_dir, load_num))
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
            a3 = env.randomP(c2,True)
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
            #==================================================================
            
            """ 3) Print and store info--------------------------------------"""
            # info=(R,Energy_Efficiency,Backhaul_cost,QoS_good,QoS_gurantee,QoS_bad,SINRdb)
            # lis_info1=(list(Backhaul_difference),list(SINRdb),list(QoS_difference))          
            key_info=['R','Energy Efficiency','Backhaul Cost','QoS Good','QoS Gurantee','QoS Bad','System Throughput','QoS Squared Difference']           
            key_info_lis=['Backhaul Difference','SINRdb','QoS Difference','Throughput']
            
            dic_info = t.test_inst_info(dic_info,(key_info,key_info_lis),((info1,info2,info3,info4,info5),(lis_info1,lis_info2,lis_info3,lis_info4,lis_info5)),1)
            dic_info_ori = t.test_inst_info(dic_info_ori,dic_info_ori_key,(info_ori1,info_ori2,info_ori3,info_ori4,info_ori5),0)
          
            t.test_print_info(((env.UE2TP[c1],P1),(env.ori_UE2TP[c2],a2),(env.ori_UE2TP[c3],a3),(env.UE2TP[c4],a4),(env.ori_UE2TP[c5],a5)),s) # print p in dBm 
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
                dic_info_no_back = t.test_inst_info(dic_info_no_back,(key_info,key_info_lis),((info1,info2,info3,info4,info5),(lis_info1,lis_info2,lis_info3,lis_info4,lis_info5)),1)
                dic_info_ori_no_back = t.test_inst_info(dic_info_ori_no_back,dic_info_ori_key,(info_ori1,info_ori2,info_ori3,info_ori4,info_ori5),0)

            
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
    for i in ['1','2','3','4','5']:
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
    debug_dic_info_EE = {key_dic_info:{name_EE:[] for name_EE in debug_dic_info_EE_key} for key_dic_info in ['1','2','3','4','5'] }

    for iMETHOD in ['1','2','3','4','5']:
        for index,nameEE in enumerate(debug_dic_info_EE_key[:3]):
            debug_dic_info_EE[iMETHOD][nameEE]=[episode_EE[index] for episode in range(MAXepisode) for episode_EE in test_debug_system_energy[iMETHOD][str(episode)] ]          
        debug_dic_info_EE[iMETHOD]['System Throughput']=[episode_EE for episode in range(MAXepisode) for episode_EE in test_debug_system_throughput[iMETHOD][str(episode)] ]

    t.writeEE(debug_dic_info_EE,debug_dic_info_EE_key,result_save)
    
    # 2) Interference
    debug_dic_info_I_key = ['Interference','Intra-cluster Interference','Inter-cluster Interference']
    debug_I={i:{'UE'+str(j):[] for j in range(env.nUE)} for i in debug_dic_info_I_key} # I, intra-cluster, inter-cluster
    debug_dic_info_I = {key_dic_info:copy.deepcopy(debug_I) for key_dic_info in ['1','2','3','4','5'] }
    
    for index,name_I in enumerate(debug_dic_info_I_key):
        for iUE in ['UE'+str(i) for i in range(env.nUE) ]:
            for iMETHOD in [str(k+1) for k in range(5)]:
                debug_dic_info_I[iMETHOD][name_I][iUE]=[episode_I[index] for episode in range(MAXepisode) for episode_I in copy.deepcopy(test_debug_I[iMETHOD][str(episode)][iUE])]

    t.writeI(debug_dic_info_I,debug_dic_info_I_key,env.nUE,result_save)
    
    # 3) action
    debug_dic_info_action_key = ['Association','Power Allocation']
    debug_dic_info_action = {key_dic_info:{name_action:[] for name_action in debug_dic_info_action_key} for key_dic_info in [str(i+1) for i in range(n_baseline)]  }

    for iMETHOD in [str(i+1) for i in range(n_baseline)] :
        debug_dic_info_action[iMETHOD]['Association'] = [ test_debug_c[iMETHOD][str(episode)] for episode in range(MAXepisode) ]
        debug_dic_info_action[iMETHOD]['Power Allocation'] = [ test_debug_p[iMETHOD][str(episode)] for episode in range(MAXepisode) ]
                          
    t.writeAction(debug_dic_info_action,debug_dic_info_action_key,result_save)
