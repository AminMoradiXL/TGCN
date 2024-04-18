import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import grid2op
from grid2op.Runner import Runner
from grid2op.Agent import DoNothingAgent, PowerLineSwitch, TopologyGreedy
from grid2op.Reward import L2RPNReward

env_name = 'l2rpn_case14_sandbox'

env = grid2op.make(env_name,
                   reward_class=L2RPNReward)

path_to_save = os.path.abspath("Dataset/testing_TG_{}".format(env_name))
runner = Runner(**env.get_params_for_runner(), agentClass=PowerLineSwitch) 
res=runner.run(nb_episode=10, max_iter=-1,path_save=path_to_save, pbar=True) 
