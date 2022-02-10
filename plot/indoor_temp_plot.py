import pandas as pd
import numpy as np
import matplotlib
# matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
# matplotlib.rcParams.update({'font.size': 14})

plt.figure(figsize=(8, 12), dpi=80)


cities = ['Golden', 'SF', 'Sterling', 'Chicago', 'Tampa']

# cities = ['Golden']

# indoor_temp_west = {}
# indoor_temp_east = {}

# for city in cities:
    
#     _, ax0 = plt.subplots(figsize=(8, 4.2))
    
#     csv_file = '../runs/new_mem_size/' + city + '_23.5_1.5_20_5_8192_10_50_False_model_based_cemrl_real_time_mem_size_100000_protected/episode-1.csv'
#     df = pd.read_csv(csv_file)
    
#     indoor_temp_west['%s' % city] = df['west_temperature'][:34560]
#     indoor_temp_east['%s' % city] = df['east_temperature'][:34560]
        
#     indoor_temp_west[city].plot(label='west zone')
#     indoor_temp_east[city].plot(label='east zone')
    
    
# plt.legend(loc=4)
# plt.xlim((0, 34560))
# plt.xticks(np.arange(0, 34560, step=5760))
# plt.xticks([0, 2976, 5664, 8640, 11520, 14496, 17376, 20352, 23328, 26208, 29184, 32064], 
#            ['Jan 1', 'Feb 1', 'Mar 1', 'Apr 1', 'May 1', 'Jun 1', 'Jul 1', 'Aug 1', 'Sep 1', 'Oct 1', 'Nov 1', 'Dec 1'])
# plt.hlines(y=22, xmin=0, xmax=34560, linestyles='dotted', colors='C0', zorder=2.5)

# ax0.set_xlabel('Month')
# ax0.set_ylabel('Indoor Air Temperature ($^{\circ}$C)')

# plt.show()


indoor_temp_west = {}
indoor_temp_east = {}
ax = {}
num_cities = len(cities)

for count, city in enumerate(cities):
    
    pos_fig = str(num_cities) + '1' + str(count + 1)
    
    if count == 0:
        ax["ax{0}".format(count + 1)] = plt.subplot(pos_fig)
    else:
        ax["ax{0}".format(count + 1)] = plt.subplot(pos_fig, sharex=ax["ax1"])
    
    csv_file = 'results/' + city + '/cem_rl/episode-1.csv'
    df = pd.read_csv(csv_file)
    
    indoor_temp_west['%s' % city] = df['west_temperature'][17376:20352]
    indoor_temp_east['%s' % city] = df['east_temperature'][17376:20352]
        
    indoor_temp_west[city].plot(label='West zone')
    indoor_temp_east[city].plot(label='East zone')
    
    plt.hlines(y=22, xmin=17376, xmax=20352, linestyles='dashed', colors='r', zorder=2.5)
    plt.hlines(y=25, xmin=17376, xmax=20352, linestyles='dashed', colors='r', zorder=2.5)
    plt.hlines(y=23.5, xmin=17376, xmax=20352, linestyles='dashed', colors='k', zorder=2.5)
    
    ax['ax' + str(count + 1)].set_title(city, fontsize=14)
    plt.yticks([20.5, 22.0, 23.5, 25.0, 26.5, 28.0])
    
 
plt.legend(loc=2, ncol=2, frameon=False)
plt.xlim((17376, 20352))
# plt.xticks(np.arange(17376, 20352, step=480))
plt.xticks([17376, 17856, 18336, 18816, 19296, 19776, 20256], 
           ['Jul 1', 'Jul 5', 'Jul 10', 'Jul 15', 'Jul 20', 'Jul 25', 'Jul 30'])


ax["ax5"].set_xlabel('Date')
ax["ax3"].set_ylabel('Indoor Air Temperature ($^{\circ}$C)')
plt.tight_layout()
plt.savefig('figures/Fig2.pdf', format='pdf')
plt.show()


plt.figure(figsize=(8, 12), dpi=80)

# cities = ['Golden']

ite_power = {}
hvac_power = {}
total_power = {}
outside_temperature = {}
ax = {}
ax_twin = {}
num_cities = len(cities)

for count, city in enumerate(cities):
    
    pos_fig = str(num_cities) + '1' + str(count + 1)
    
    if count == 0:
        ax["ax{0}".format(count + 1)] = plt.subplot(pos_fig)
        ax_twin["ax{0}".format(count + 1)] = ax["ax{0}".format(count + 1)].twinx()
    else:
        ax["ax{0}".format(count + 1)] = plt.subplot(pos_fig, sharex=ax["ax1"])
        ax_twin["ax{0}".format(count + 1)] = ax["ax{0}".format(count + 1)].twinx()
    
    csv_file = 'results/' + city + '/cem_rl/episode-1.csv'
    df = pd.read_csv(csv_file)
    
    ite_power['%s' % city] = df['ite_power'][:34560] / 1000
    hvac_power['%s' % city] = df['hvac_power'][:34560] / 1000
    total_power['%s' % city] = ite_power['%s' % city] + hvac_power['%s' % city]
    
    outside_temperature['%s' % city] = df['outside_temperature'][:34560]
    
    ax["ax{0}".format(count + 1)].plot(total_power[city], color='C0')
    ax_twin["ax{0}".format(count + 1)].plot(outside_temperature[city], color='C1')
    
    ax["ax{0}".format(count + 1)].tick_params(axis='y', labelcolor='C0')
    ax_twin["ax{0}".format(count + 1)].tick_params(axis='y', labelcolor='C1')
    
    ax['ax' + str(count + 1)].set_title(city, fontsize=14)
    
    if count == num_cities - 1:
        plt.setp(ax['ax' + str(count + 1)].get_xticklabels(), visible=True)
        plt.setp(ax_twin['ax' + str(count + 1)].get_xticklabels(), visible=True)
    else:
        plt.setp(ax['ax' + str(count + 1)].get_xticklabels(), visible=False)
        plt.setp(ax_twin['ax' + str(count + 1)].get_xticklabels(), visible=False)
        
   
plt.xlim((0, 34560))
plt.xticks(np.arange(0, 34560, step=5760))
plt.xticks([0, 2976, 5664, 8640, 11520, 14496, 17376, 20352, 23328, 26208, 29184, 32064], 
        ['Jan 1', 'Feb 1', 'Mar 1', 'Apr 1', 'May 1', 'Jun 1', 'Jul 1', 'Aug 1', 'Sep 1', 'Oct 1', 'Nov 1', 'Dec 1'])

# plt.xticks([0, 5664, 11520, 17376, 23328, 29184], 
#            ['Jan 1', 'Mar 1', 'May 1', 'Jul 1', 'Sep 1', 'Nov 1'])

ax["ax5"].set_xlabel('Date')
ax["ax3"].set_ylabel('Total Power Consumption (kW)', color='C0')
ax_twin["ax3"].set_ylabel('Outdoor Air Temperature ($^{\circ}$C)', color='C1')
plt.tight_layout()
plt.savefig('figures/Fig3.pdf', format='pdf')
plt.show()

