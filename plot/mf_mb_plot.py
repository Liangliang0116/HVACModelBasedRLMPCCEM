import pickle
import numpy as np
import pandas as pd
import matplotlib
# matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
# plt.rcParams.update({'font.size': 12})

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 12})


city = 'SF'
plt.figure(figsize=(8, 8), dpi=80)
fig, (ax1, ax2) = plt.subplots(2)

reward_ppo = []
for idx_year in range(20):
    df = pd.read_csv('results/{}/ppo/episode-{}.csv'.format(city, idx_year))
    reward_np = df['reward'].to_numpy()[:-1]
    reward_ppo.append(reward_np)
        
reward_np_ppo = np.asarray(reward_ppo)
reward_np_ppo_shape = reward_np_ppo.shape
reward_np_ppo_reshaped = reward_np_ppo.reshape((reward_np_ppo_shape[1], reward_np_ppo_shape[0]))
reward_ave_ppo = np.mean(reward_np_ppo_reshaped, axis=-1)

data_idxes_ppo = np.arange(reward_np_ppo_shape[1]) * reward_np_ppo_shape[0]

reward_cem_rl = []
df_1 = pd.read_csv('results/{}/cem_rl/episode-0.csv'.format(city))
df_2 = pd.read_csv('results/{}/cem_rl/episode-1.csv'.format(city))
reward_np_1 = df_1['reward'].to_numpy()[:-1]
reward_np_2 = df_2['reward'].to_numpy()[:-1]

reward_np_cem_rl = np.concatenate((reward_np_1, reward_np_2), axis=0)
reward_np_cem_rl_shape = reward_np_cem_rl.shape

data_idxes_cem_rl = np.arange(reward_np_cem_rl_shape[0])

ax1.plot(data_idxes_cem_rl[:20000], reward_np_cem_rl[:20000])
ax1.set_title('MBRL-MC', fontsize=12)
ax2.plot(data_idxes_ppo[:10000], reward_ave_ppo[:10000])
ax2.set_title('PPO', fontsize=12)
plt.xlabel('Number of Environment Data')
ax1.set_xlim([0, 20000])
ax1.set_yticks(np.arange(-6, 3, 2)) 
ax2.set_xlim([0, 200000])
ax2.set_yticks(np.arange(-6, 3, 2)) 
plt.tight_layout()
fig.text(0.03, 0.45, 'Reward', ha='center', rotation='vertical')
plt.savefig('figures/Fig5.pdf', format='pdf')
fig.show()

