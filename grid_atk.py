import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import sys
import numpy as np
import grid2op
from grid2op.PlotGrid import PlotMatplot
from grid2op.Runner import Runner
from grid2op.Agent import DoNothingAgent, PowerLineSwitch, TopologyGreedy
from grid2op.Action import PowerlineSetAction
from grid2op.Opponent import RandomLineOpponent, BaseActionBudget, GeometricOpponent
from grid2op.Episode import EpisodeReplay
from tqdm import tqdm

env_name = 'rte_case14_realistic'


env = grid2op.make(env_name, test = False)

env_with_opponent = grid2op.make(env_name,
                                  opponent_attack_cooldown= 12*3, # 12 * Hours
                                  opponent_attack_duration= 12*0.5, # 12 * Hours 
                                  opponent_budget_per_ts= 0.1,
                                  opponent_init_budget= 0,
                                  opponent_action_class=PowerlineSetAction,
                                  opponent_class=RandomLineOpponent,
                                  opponent_budget_class=BaseActionBudget,
                                  kwargs_opponent={"lines_attacked":
                                   ["1_3_3", "1_4_4", "3_6_15", "9_10_12", "11_12_13", "12_13_14"]}                                  
                                  )

agentClass = PowerLineSwitch  
path_name = f"Dataset/Atk/{env_name}/PLS/" 

if not os.path.exists(path_name):
   os.makedirs(path_name)
path_to_save = os.path.abspath(path_name)

nb_episode = 1
max_iter = 12*24 # 12 * Hours 

runner = Runner(**env_with_opponent.get_params_for_runner(), agentClass=agentClass) 

res=runner.run(nb_episode = nb_episode, max_iter = max_iter, path_save = path_to_save, pbar = True) 

for i in range(nb_episode):
    plot_epi = EpisodeReplay(path_to_save)
    plot_epi.replay_episode(res[i][1], gif_name='this_episode')
    