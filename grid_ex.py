from grid2op.Episode import EpisodeData
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

env_name = 'rte_case14_realistic'
path_ = f'Dataset/Normal_dataset/{env_name}/TG/'
path_agent = os.path.abspath(path_)
episode_studied = EpisodeData.list_episode(path_agent)
n_episodes = len(episode_studied)

this_episode = EpisodeData.from_disk(*episode_studied[0])

for i in range(n_episodes):
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


fig = plt.figure(figsize=(8, 8))
plt.matshow(np.corrcoef(rho_Total.T), 0)
plt.clim(-1,1)
plt.colorbar()
plt.xlabel("powerline number")
plt.ylabel("powerline number")
fig.savefig(f'rho_normal_{env_name}_TG_correlation',dpi=600)

df = pd.DataFrame(rho_Total)
df.to_csv(f'rho_normal_{env_name}_TG.csv')



    
