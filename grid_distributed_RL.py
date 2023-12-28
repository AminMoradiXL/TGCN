import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import grid2op
from grid2op.Agent import RecoPowerlineAgent, PowerLineSwitch, TopologyGreedy
from grid2op.Runner import Runner
import tensorflow as tf
import random
import numpy as np 
from ttictoc import tic, toc
import os

test_number = 1
attack_time = 24
decision_interval = 10
nb_episode = 50
max_iter = 50
costly  = 0.7

env = grid2op.make('rte_case14_realistic')
attack_list = ["1_3_3", "1_4_4", "3_6_15", "9_10_12", "11_12_13", "12_13_14"]

TG_model = tf.keras.models.load_model(f'Results/Attack/rte_case14_realistic/TG/test_{test_number}/my_model')
PLS_model = tf.keras.models.load_model(f'Results/Attack/rte_case14_realistic/PLS/test_{test_number}/my_model')

def extreme_finder(pred, th):
    pred_extreme = np.where(pred>th,1,0)    
    extreme_degree = sum(sum(pred_extreme))
    return extreme_degree   

def agent_selection(PLS_pred, TG_pred, th):
    PLS_ex_deg = extreme_finder(PLS_pred, th)
    TG_ex_deg = extreme_finder(TG_pred, th)
    if costly * PLS_ex_deg > TG_ex_deg:
        selected_agent = 'TopologyGreedy'
    else:
        selected_agent = 'PowerLineSwitch'
    if PLS_ex_deg == 0 and PLS_ex_deg == 0:
        selected_agent = 'RecoPowerlineAgent'
    return selected_agent 
 
###############
# the gym loop

selected_agent = 'RecoPowerlineAgent'
rho = np.zeros((nb_episode, max_iter + 1, env.n_line))
agent_list = []

for th in np.arange(0.7,1,0.1):
    th = round(th,2)
    file_name = f'Results/CompareTH/reults_{th}.txt'
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    for i in range(nb_episode):
        message = f'episde {i} started ...\n'
        print(message)
        with open(file_name,'a+') as f:
            f.write(message)
            f.close()
        t = 0 
        obs = env.reset()
        rho[i,t,:] = obs.rho 
        done = False
        reward = env.reward_range[0]
        while not done:
            if selected_agent == 'RecoPowerlineAgent':
                agent = RecoPowerlineAgent(env.action_space)
            elif selected_agent == 'TopologyGreedy': 
                agent = TopologyGreedy(env.action_space)
            elif selected_agent == 'PowerLineSwitch': 
                agent = PowerLineSwitch(env.action_space)

            act = agent.act(obs, reward, done)
            obs, reward, done, info = env.step(act)
            t = t + 1
            rho[i,t,:] = obs.rho
            if t == attack_time :
                attacked_line = random.choice(attack_list)
                act = env.action_space.disconnect_powerline(line_name=attacked_line)
            if t == attack_time + decision_interval :
                window = rho[i,t-23:t+1,:]
                window = np.expand_dims(window, axis = 0)
                window = np.expand_dims(window, axis = 3)
                TG_pred = TG_model.predict(window)
                TG_pred = TG_pred[0,:,:]
                PLS_pred = PLS_model.predict(window)
                PLS_pred = PLS_pred[0,:,:]
                selected_agent  = agent_selection(PLS_pred, TG_pred, th)
                agent_change = True
                tic()
            if t == max_iter:
                done = True
        if done:
            agent_list.append(selected_agent)
            message = f'episde {i} ended. Selected agent was {selected_agent}.\n' 
            print(message)
            with open(file_name,'a+') as f:
                f.write(message)
                f.close()
            if agent_change:
                each_step_comp = toc()/(t-(attack_time + decision_interval))
                message = f'it costed {each_step_comp} seconds per step\n'
                print(message)
                with open(file_name,'a+') as f:
                    f.write(message)
                    f.close()
                agent_change = False
            

        

    
            
        
    