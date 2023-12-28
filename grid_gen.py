import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import sys
import numpy as np
import grid2op
from grid2op.PlotGrid import PlotMatplot
from grid2op.Runner import Runner
from grid2op.Agent import DoNothingAgent, PowerLineSwitch, TopologyGreedy
import networkx as nx
import matplotlib.pyplot as plt 
import matplotlib

# env_name = 'rte_case5_example'
env_name = 'rte_case14_realistic'
# env_name = 'rte_case118_example'
# env_name = 'educ_case14_storage'

env = grid2op.make(env_name, test=False) # default = "rte_case14_realistic" environment 

path_to_save = os.path.abspath("testing_TG_{}".format(env_name))
runner = Runner(**env.get_params_for_runner(), agentClass=TopologyGreedy) 
res=runner.run(nb_episode=1,max_iter=50,path_save=path_to_save, pbar=True) 
