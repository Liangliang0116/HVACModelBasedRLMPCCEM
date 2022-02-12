# MBRL-MC: An HVAC Control Approach via Combining Model-based Deep Reinforcement Learning and Model Predictive Control

This repository is for the paper "MBRL-MC: An HVAC Control Approach via Combining Model-based Deep Reinforcement Learning and Model Predictive Control". We run it with the Python version 3.8.10 on Ubuntu 20.04. The codes are mainly adapted from https://github.com/vermouth1992/mbrl-hvac and 
https://github.com/apourchot/CEM-RL. 

## Requirements

The packages installed in the virtual environment are listed in ```requirements.txt```. You can install them by running

### Python Setup

```setup
pip install -r requirements.txt
```

### EnergyPlus
Please follow https://github.com/IBM/rl-testbed-for-energyplus and install EnergyPlus version 9.2.0. Note that this repository is only tested on EnergyPlus version 9.2.0. There mighe be some unexpected issues with other EnergyPlus versions for this repository. 

## Before Training
Before training, the city weather file information in ```.bashrc``` should be changed to the one corresponding to the city whose weather data that are to be used. 

## Training

To train MBRL-MC, run
```train
python train_model_based.py --city SF --mpc_horizon 5 --num_days_on_policy 10 --training_epochs 50 --num_years 2 --algorithm cem_rl --n_grad 5 --max_steps 19200 --start_steps 2000 --n_episode 960 --verbose
```

To train the model-based RL with random shooting, run

```train
python train_model_based.py --city SF --mpc_horizon 5 --num_days_on_policy 10 --training_epochs 50 --num_years 2 --algorithm random_shooting --n_grad 5 --max_steps 19200 --start_steps 2000 --n_episode 960 --verbose
```

To train the model-based RL with imitation learning, run

```train
python train_model_based.py --city SF --mpc_horizon 5 --num_days_on_policy 10 --training_epochs 50 --num_years 2 --algorithm imitation_learning --n_grad 5 --max_steps 19200 --start_steps 2000 --n_episode 960 --verbose
```

To train the PPO algorithm, run 
```train
python train_ppo.py --city SF
```

To run the PID, run
```train
python train_pid.py --city SF
```

The file ```.vscode/launch.json``` also shows the above training configurations. 

All the results will be saved in ```results``` folder. 

## After Training 

Use ```plot/indoor_temp_plot.py```, ```plot/score_plot.py```, and ```plot/mf_mb_plot.py``` to visualize the results. 

### Available cities
- SF
- Golden
- Chicago
- Sterling
- Tampa 


## Citation
Our codes are based the codes of the following papers. Please also cite them when possible. 

```bib
@article{Zhang2019BuildingHS,
  title={Building HVAC Scheduling Using Reinforcement Learning via Neural Network Based Model Approximation},
  author={Chi Zhang and S. Kuppannagari and R. Kannan and V. Prasanna},
  journal={Proceedings of the 6th ACM International Conference on Systems for Energy-Efficient Buildings, Cities, and Transportation},
  year={2019}
}
```

```bib
@article{pourchot2018cem,
  title={CEM-RL: Combining evolutionary and gradient-based methods for policy search},
  author={Pourchot, Alo{\"\i}s and Sigaud, Olivier},
  journal={arXiv preprint arXiv:1810.01222},
  year={2018}
}
```

Please also cite the paper that introduces the environment.

```bib
@InProceedings{10.1007/978-981-13-2853-4_4,
author="Moriyama, Takao and De Magistris, Giovanni and Tatsubori, Michiaki and Pham, Tu-Hoa and Munawar, Asim and Tachibana, Ryuki",
title="Reinforcement Learning Testbed for Power-Consumption Optimization",
booktitle="Methods and Applications for Modeling and Simulation of Complex Systems",
year="2018",
publisher="Springer Singapore",
address="Singapore",
pages="45--59",
isbn="978-981-13-2853-4"
}
```
