import pickle
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
# plt.rcParams.update({'font.size': 12})

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 14})


cities = ['Golden', 'SF', 'Sterling', 'Chicago', 'Tampa']
# mu_scores = {}

# for city in cities:
    
#     with open('EnergyPlusSplitEpisodeWrapper-run_' + city + '/log.pkl', 'rb') as f:
#         data = pickle.load(f)
#         mu_scores['%s' % city] = data['mu_score']
        
#     mu_scores[city].plot(label=city)
   
# plt.legend() 
# plt.xlabel('# Evaluation')
# plt.ylabel('mu_score')
# plt.show()
        

# average_scores = {}

# for city in cities:
    
#     with open('EnergyPlusSplitEpisodeWrapper-run_' + city + '/log.pkl', 'rb') as f:
#         data = pickle.load(f)
#         average_scores['%s' % city] = data['average_score']
        
#     average_scores[city].plot(label=city)
   
# plt.legend() 
# plt.xlabel('# Evaluation')
# plt.ylabel('average_score')
# plt.show()

average_scores_half = {}

_, ax = plt.subplots(figsize=(8, 4))

for city in cities:
    
    with open('EnergyPlusSplitEpisodeWrapper-run_' + city + '/log.pkl', 'rb') as f:
        data = pickle.load(f)
        average_scores_half['%s' % city] = data['average_score_half']
        
    average_scores_half[city].plot(label=city)

plt.xlim((-1, 138))
plt.legend(loc=4) 
plt.xticks(np.arange(0, 138, step=17.25))
# plt.xticks([0, 17.25, 34.5, 51.75, 69, 86.25, 103.5, 120.75], 
#            ['Jan', 'Apr', 'Jul', 'Oct', 'Jan', 'Apr', 'Jul', 'Oct'])
plt.xticks([0, 11.5, 23, 34.5, 46, 57.5, 69, 80.5, 92, 103.5, 115, 126.5], 
           ['Jan', 'Mar', 'May', 'Jul', 'Sep', 'Nov', 'Jan', 'Mar', 'May', 'Jul', 'Sep', 'Nov'])
plt.xlabel('Month')
plt.ylabel('Average fitness of the top 5 actors')
plt.tight_layout()
plt.savefig('Fig4_.pdf', format='pdf')
plt.show()


