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

To train the model-based RL with MPC and CEM (the proposed algorithm in this paper), run
```train
python train_model_based.py --city SF --mpc_horizon 5 --num_days_on_policy 10 --training_epochs 50 --num_years 2 --cem_rl --n_grad 5 --max_steps 19200 --start_steps 2000 --n_episode 960
```

To train the model-based RL with random shooting (RS), run

```train
python train_model_based.py --city SF --mpc_horizon 5 --num_days_on_policy 10 --training_epochs 50 --num_years 2 --n_grad 5 --max_steps 19200 --start_steps 2000 --n_episode 960
```

To train the model-based RL with dagger, run

```train
python train_model_based.py --city SF --mpc_horizon 5 --num_days_on_policy 10 --training_epochs 50 --num_years 2 --dagger --n_grad 5 --max_steps 19200 --start_steps 2000 --n_episode 960
```

It will create a folder called ``runs`` that includes all the state, action and rewards during the training.
The EnergyPlus generated files will be in the ``log`` folder. 
The generated files associated with the actors and critic will be in the ```results``` folder. 

## After Training 

Run ```csv_compute.py``` to compute some further information for each obtained result. 
Run ```indoor_temp_plot.py``` and ```score_plot.py``` to visualize the results. 

### Available cities
- SF
- Golden
- Chicago
- Sterling
- Tampa 

We also provide shell script file in case you want to run everything. Checkout
- run_pid.sh
- run_ppo.sh
- run_model_based_plan.sh
- run_model_based_dagger.sh

## Citation
Please cite the papers that our codes are based on. 

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
