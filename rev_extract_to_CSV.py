from grid2op.Episode import EpisodeData
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

agent_name = 'redisp'
path_ = f'test_results/eval_logs/{agent_name}_118'
path_agent = os.path.abspath(path_)
episode_studied = EpisodeData.list_episode(path_agent)
n_episodes = len(episode_studied)

this_episode = EpisodeData.from_disk(*episode_studied[0])

for i in range(n_episodes):
    print(f'Episode {i}/{n_episodes}')
    this_episode = EpisodeData.from_disk(*episode_studied[i])
    nstep = len(this_episode.observations)
    for t, obs in enumerate(this_episode.observations):
        nlines = obs.n_line
        break

    rho = np.zeros((nstep, nlines))

    for t, obs in enumerate(this_episode.observations):
        rho[t, :] = obs.rho

    rho = rho[:-1,:]
    if i==0:
        rho_Total=rho 
    else:
        rho_Total=np.concatenate((rho_Total,rho),axis=0)

df = pd.DataFrame(rho_Total)
df.to_csv(f'test_results/eval_csv/rho_{agent_name}.csv')



    
