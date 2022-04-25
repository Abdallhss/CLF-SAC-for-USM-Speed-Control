# -*- coding: utf-8 -*-
"""SAC_USM_summary015082021.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dkO9jeq8YQvj1qT2-Jm-qK8GoS3Ih7ow
"""

# Commented out IPython magic to ensure Python compatibility.
from SAC_Agent import SAC
#import numpy as np
#import matplotlib.pyplot as plt
#from evaluate import run_evals
import torch.nn as nn
import time
#import numpy as np
obs_dim = 5
act_dim = 1
act_limit= [1]
buffer_size = int(1e6)
hidden_sizes = ([64]*2,[64]*4,[5])
activation=(nn.ReLU,nn.ReLU,nn.ReLU6)
agent_save = 'exp'#'sim_noise'
agent_load = 'exp' #sim_noise' #'sim_speednoise'
agent_train = False
seed = 0

path = ""

env_type = 'EXP' #'EXP'
agent = SAC(env_type=env_type,obs_dim = obs_dim, act_dim = act_dim, act_limit= act_limit,  polyak=0.99,
            alpha=0.1, alpha_rate = 1,lr=5e-3,hidden_sizes = hidden_sizes, gamma=0.5,gamma_rate=1, activation=activation,seed=seed)

if env_type == 'EXP':
    agent.env.start(amp=3.0,freq=42)

if agent_load != None:
    agent.load_network(agent_load)
if agent_train:
    start_time =time.time()
    LOGS,losses = agent.train_agent(num_eps=2000-1, max_ep_len = 10, batch_size=64,start_steps = -1,update_every=10,eval_every=500)
    print("training time: {}".format(time.time()-start_time))
    agent.plot_training_logs(LOGS,losses)
    if agent_save != None:
        agent.save_network(agent_save)
        
agent.run_evals_ep('trained_39khz',n=100,freq=39,temp=20)
agent.run_evals_ep('trained_45khz',n=100,freq=45,temp=20)
agent.run_evals_full()

if env_type == 'EXP':
    agent.env.stop()



