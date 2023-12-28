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

# env_name = 'rte_case5_example'
# env_name = 'rte_case14_realistic'
env_name = 'l2rpn_wcci_2022'
# env_name = 'educ_case14_storage'

env = grid2op.make(env_name, test = False)
line_list = ['91_101_55',
 '31_113_76',
 '87_88_29',
 '11_13_57',
 '7_29_100',
 '48_68_170',
 '62_63_160',
 '11_15_83',
 '55_57_148',
 '2_4_96',
 '63_64_163',
 '8_9_140',
 '83_84_25',
 '104_107_66',
 '27_28_99',
 '36_38_115',
 '15_16_86',
 '13_14_79',
 '102_104_61',
 '26_27_98'] 

env_with_opponent = grid2op.make(env_name,
                                  opponent_attack_cooldown= 12*3, # 12 * Hours
                                  opponent_attack_duration= 12*0.5, # 12 * Hours 
                                  opponent_budget_per_ts= 0.1,
                                  opponent_init_budget= 0,
                                  opponent_action_class=PowerlineSetAction,
                                  opponent_class=RandomLineOpponent,
                                  opponent_budget_class=BaseActionBudget,
                                  # kwargs_opponent={"lines_attacked":
                                  #    ["1_3_3", "1_4_4", "3_6_15", "9_10_12", "11_12_13", "12_13_14"]}
                                  kwargs_opponent={"lines_attacked": line_list}
                                  )


agentClass = PowerLineSwitch  
path_name = f"Dataset/Attack_dataset/{env_name}/PLS/" 
# path_name = f"Dataset/Normal_dataset/{env_name}/PLS/" 

# agentClass = TopologyGreedy 
# path_name = f"Dataset/Attack_dataset/{env_name}/TG/" 
# path_name = f"Dataset/Normal_dataset/{env_name}/TG/" 

if not os.path.exists(path_name):
   os.makedirs(path_name)
path_to_save = os.path.abspath(path_name)

nb_episode = 1000
max_iter = 12*24 # 12 * Hours 

# pbar = tqdm(total=nb_episode, disable=False)

runner = Runner(**env_with_opponent.get_params_for_runner(), agentClass=agentClass) 
# runner = Runner(**env.get_params_for_runner(), agentClass=agentClass) 

res=runner.run(nb_episode = nb_episode, max_iter = max_iter, path_save = path_to_save, pbar = True) 

# for i in range(nb_episode):
#     plot_epi = EpisodeReplay(path_to_save)
#     plot_epi.replay_episode(res[i][1], gif_name='this_episode')


    