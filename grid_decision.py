import grid2op
from grid2op.Agent import RecoPowerlineAgent
import numpy as np 

env_name = 'rte_case14_realistic'
env = grid2op.make(env_name, test=False)

###############
attack_probability = 0.01; 

# the gym loops
agent = RecoPowerlineAgent(env.action_space)
nb_episodes = 5
for i in range(nb_episodes):
    obs = env.reset()
    done = False
    reward = env.reward_range[0]
    while not done:
        act = agent.act(obs, reward, done)
        obs, reward, done, info = env.step(act)
        print(obs.rho)
        
        attack_occurance = np.random.rand() < attack_probability
        if attack_occurance:
            act = env.action_space.disconnect_powerline(line_name='4_5_17')
        
    

# act = env.action_space.disconnect_powerline(line_name='4_5_17')
# obs_after, reward, done, info = env.step(act)
line_attacked = ["1_3_3", "1_4_4", "3_6_15", "9_10_12", "11_12_13", "12_13_14"]
env.name_line[random]