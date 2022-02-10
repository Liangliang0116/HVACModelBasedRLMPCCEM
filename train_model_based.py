import numpy as np
import time

from torchlib.deep_rl import RandomAgent
from torchlib.utils.random.sampler import UniformSampler, MultivariateGaussianSampler
from agent import ModelBasedHistoryPlanAgent, ModelBasedHistoryDaggerAgent, EnergyPlusDynamicsModel, \
    BestRandomActionHistoryPlanner, BestRandomActionHistoryAdaptivePlanner
from agent.utils import EpisodicHistoryDataset
from agent.sampler import Sampler
from gym_energyplus import make_env, ALL_CITIES
from outdoor_temp_extract import outdoor_temp_interpolate_and_extract
from agent.cem_rl.cem_rl import ModelBasedCEMRLAgent
from logger import logger
from postprocessing_data import postprocessing_data


def train(args, checkpoint_path=None):
    
    dataset_maxlen = 96 * args['num_days_per_episodes'] * args['num_dataset_maxlen_days']
    max_rollout_length = 96 * args['num_days_per_episodes']
    num_on_policy_iters = (365 * args['num_years'] // args['num_days_per_episodes'] -
                           args['num_init_random_rollouts']) // args['num_days_on_policy']

    if args['new_log_dir']:
        if args['adaptive_sampling']:
            log_dir = args['log_dir'] + '{}/{}_adaptive/{}'.format('_'.join(args['city']), args['algorithm'], time.strftime("%Y%m%d-%H%M%S"))
        else:
            log_dir = args['log_dir'] + '{}/{}/{}'.format('_'.join(args['city']), args['algorithm'], time.strftime("%Y%m%d-%H%M%S"))
    else:
        if args['adaptive_sampling']:
            log_dir = args['log_dir'] + '{}/{}_adaptive'.format('_'.join(args['city']), args['algorithm'])
        else:
            log_dir = args['log_dir'] + '{}/{}'.format('_'.join(args['city']), args['algorithm'])

    env = make_env(cities=args['city'], 
                   temperature_center=args['temp_center'], 
                   temp_tolerance=args['temp_tolerance'], 
                   obs_normalize=True,
                   action_normalize=True,
                   num_days_per_episode=1, 
                   log_dir=log_dir)
    
    logger.configure(dir=log_dir+'/logger', 
                     format_strs=['stdout', 'log', 'csv'], 
                     snapshot_mode='last')
    
    outdoor_temp_interpolate_and_extract(city=args['city'],
                                         weather_file=env.weather_files[0],
                                         num_years=args['num_years'],
                                         temperature_center=args['temp_center'])

    baseline_agent = RandomAgent(env.action_space)
    dataset = EpisodicHistoryDataset(maxlen=dataset_maxlen, 
                                     window_length=args['window_length'])
    
    """ -------------- Obtain initial random samples from the environment -------------- """
    logger.log("Obtaining random samples from the environment...")
    sampler = Sampler(env=env, 
                      window_length=args['window_length'], 
                      mpc_horizon=args['mpc_horizon'])

    initial_dataset = sampler.sample(policy=baseline_agent, 
                                     num_rollouts=args['num_init_random_rollouts'],
                                     max_rollout_length=np.inf)
    dataset.append(initial_dataset) 

    model = EnergyPlusDynamicsModel(state_dim=env.observation_space.shape[0],
                                    action_dim=env.action_space.shape[0],
                                    hidden_size=32,
                                    learning_rate=1e-3,
                                    log_dir=log_dir)

    if args['algorithm'] == 'cem_rl':
        agent = ModelBasedCEMRLAgent(env=env, 
                                     model=model, 
                                     actor_lr=args['actor_lr'],
                                     critic_lr=args['critic_lr'], 
                                     mpc_horizon=args['mpc_horizon'], 
                                     window_length=args['window_length'], 
                                     layer_norm=args['layer_norm'], 
                                     temperature_center=args['temp_center'],
                                     pop_size=args['pop_size'], 
                                     mem_size=args['mem_size'], 
                                     gauss_sigma=args['gauss_sigma'], 
                                     sigma_init=args['sigma_init'], 
                                     damp=args['damp'], 
                                     damp_limit=args['damp_limit'], 
                                     elitism=args['elitism'],
                                     period=args['period'], 
                                     log_dir=log_dir, 
                                     save_all_models=args['save_all_models'])

    else:
        if args['adaptive_sampling']:
            action_sampler = MultivariateGaussianSampler(mu=np.zeros(env.action_space.shape[0]), 
                                                         sigma=np.ones(env.action_space.shape[0]))
            planner = BestRandomActionHistoryAdaptivePlanner(model=model, 
                                                             action_space=env.action_space,
                                                             action_sampler=action_sampler, 
                                                             cost_fn=env.cost_fn, 
                                                             city=env.city,
                                                             horizon=args['mpc_horizon'],
                                                             num_random_action_selection=args['num_random_action_selection'],
                                                             gamma=args['gamma'],
                                                             damp=args['damp'], 
                                                             damp_limit=args['damp_limit'])
        else:
            action_sampler = UniformSampler(low=env.action_space.low, high=env.action_space.high)
            planner = BestRandomActionHistoryPlanner(model=model, 
                                                     action_sampler=action_sampler, 
                                                     cost_fn=env.cost_fn, 
                                                     city=env.city,
                                                     horizon=args['mpc_horizon'],
                                                     num_random_action_selection=args['num_random_action_selection'],
                                                     ratio_elite=0.5,
                                                     gamma=args['gamma'])

        if args['algorithm'] == 'imitation_learning':
            agent = ModelBasedHistoryDaggerAgent(model=model, 
                                                 planner=planner, 
                                                 policy_data_size=10000,
                                                 window_length=args['window_length'], 
                                                 baseline_agent=baseline_agent,
                                                 state_dim=env.observation_space.shape[0],
                                                 action_dim=env.action_space.shape[0])
        else:
            agent = ModelBasedHistoryPlanAgent(model=model, 
                                               planner=planner, 
                                               window_length=args['window_length'], 
                                               baseline_agent=baseline_agent)

    start_time = time.time()
    for num_iter in range(num_on_policy_iters):
        itr_start_time = time.time()
        logger.log("\n ---------------- Iteration %d/%d ----------------" % (num_iter + 1, num_on_policy_iters))
        agent.set_statistics(dataset)
        
        """ --------------- Fit the dynamics model --------------- """
        logger.log("Training dynamics model for %i epochs..." % args['model_training_epochs'])
        time_fit_model_start = time.time()
        agent.fit_dynamic_model(dataset=dataset, 
                                epoch=args['model_training_epochs'], 
                                batch_size=args['batch_size'], 
                                verbose=args['verbose'])
        logger.record_tabular('Time-ModelFit', time.time() - time_fit_model_start)
        
        """ --------------- Fit the policy --------------- """
        logger.log("Training policy...")
        time_fit_policy_start = time.time()
        if args['algorithm'] == 'cem_rl':
            agent.fit_policy(batch_size=args['batch_size'],
                             n_grad=args['n_grad'],
                             max_steps=args['max_steps'], 
                             start_steps=args['start_steps'], 
                             n_episodes=args['n_episodes'], 
                             n_noisy=args['n_noisy'], 
                             n_eval=args['n_eval'])
        elif args['algorithm'] == 'imitation_learning':
            agent.fit_policy(epoch=args['policy_training_epochs'], 
                             batch_size=args['batch_size'], 
                             verbose=args['verbose'])
        logger.record_tabular('Time-PolicyFit', time.time() - time_fit_policy_start)
        
        """ -------------- Obtain samples from the environment -------------- """
        logger.log("Continuing obtaining samples from the environment...")
        time_env_sampling_start = time.time()
        on_policy_dataset = sampler.sample(policy=agent, 
                                           num_rollouts=args['num_days_on_policy'], 
                                           max_rollout_length=max_rollout_length)
        logger.record_tabular('Time-EnvSampling', time.time() - time_env_sampling_start)
            
        """ ------------------- Other loggings ------------------- """
        logger.logkv('Iteration Index', num_iter + 1)
        logger.logkv('Time', time.time() - start_time)
        logger.logkv('Iteration Time', time.time() - itr_start_time)
        logger.logkv('Size of Dataset', len(dataset))
        logger.logkv('Number of Trajectories', dataset.num_trajectories)
        
        on_policy_stats = on_policy_dataset.log()
        for key, value in on_policy_stats.items():
            logger.logkv(key, value)
            
        logger.dumpkvs()
        dataset.append(on_policy_dataset)
        
    logger.log("Training finished")
    
    """ ------------------- Postprocessing data ------------------- """
    logger.log("Postprocessing data...")
    postprocessing_data(log_dir=log_dir, 
                        num_years=args['num_years'],
                        temperature_center=args['temp_center'],
                        tolerance=args['temp_tolerance'])
    logger.log("Done")

    if checkpoint_path:
        agent.save_checkpoint(checkpoint_path)


def make_parser():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--city', type=str, choices=ALL_CITIES, nargs='+', help='city of which the weather file is used')
    parser.add_argument('--algorithm', type=str, default='cem_rl', 
                        choices=['cem_rl', 'random_shooting', 'imitation_learning'], help='algorithm to be trained')
    parser.add_argument('--adaptive_sampling', dest="adaptive_sampling", action="store_true")
    parser.add_argument('--temp_center', type=float, default=23.5, help='temperature center that is perferred')
    parser.add_argument('--temp_tolerance', type=float, default=1.5, help='temperature deviation from center to avoid uncomfortableness')
    parser.add_argument('--window_length', type=int, default=20, help='window length of historical data')
    parser.add_argument('--num_years', type=int, default=2, help='number of years considered in EnergyPlus simulation')
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--num_init_random_rollouts', type=int, default=65, help='number of rollouts for the initial random dataset')
    parser.add_argument('--num_dataset_maxlen_days', type=int, default=120)
    parser.add_argument('--num_random_action_selection', type=int, default=8192)
    parser.add_argument('--num_days_per_episodes', type=int, default=1, help='number of days in simulation per episode')
    parser.add_argument('--mpc_horizon', type=int, default=5, help='mpc prediction horizon')
    parser.add_argument('--num_days_on_policy', type=int, default=15)
    
    parser.add_argument('--model_training_epochs', type=int, default=50, help='training epochs for dynamics model')
    parser.add_argument('--policy_training_epochs', type=int, default=50, help='training epochs for policy')
    parser.add_argument('--batch_size', default=128, type=int, help='training batch size for both dynamics model and policy')

    # DDPG parameters
    parser.add_argument('--actor_lr', default=0.001, type=float, help='learning rate for actor')
    parser.add_argument('--critic_lr', default=0.001, type=float, help='learning rate for critic')
    parser.add_argument('--layer_norm', dest='layer_norm', action='store_true', help='indicator wheather to use layer normalization')
    parser.add_argument('--gauss_sigma', default=0.1, type=float)

    # Evolutionary strategy parameters
    parser.add_argument('--pop_size', default=10, type=int, help='size of the evolutionary population')
    parser.add_argument('--elitism', dest="elitism",  action='store_true')
    parser.add_argument('--n_grad', default=5, type=int, help='number of actors that use gradient steps to optimize')
    parser.add_argument('--sigma_init', default=1e-3, type=float)
    parser.add_argument('--damp', default=1e-3, type=float)
    parser.add_argument('--damp_limit', default=1e-5, type=float)

    # Training parameters
    parser.add_argument('--n_episodes', default=100, type=int, help='number of evaluation episodes in each time')
    parser.add_argument('--start_steps', default=10000, type=int, 
                        help='step index after which the actors and critics will be updated when training the cem_rl policy')
    parser.add_argument('--max_steps', default=1000000, type=int, help='maximum number of iteration steps when training the cem_rl policy')
    parser.add_argument('--mem_size', default=100000, type=int)
    parser.add_argument('--n_noisy', default=0, type=int, help='number of noisy actors')
    parser.add_argument('--n_eval', default=10, type=int, 
                        help='number of evaluation episodes in when evaluating the actors after training when training the cem_rl policy')

    # misc
    parser.add_argument('--log_dir', default='results/', type=str)
    parser.add_argument('--new_log_dir', dest='new_log_dir', action='store_true')
    parser.add_argument('--period', default=5000, type=int)
    parser.add_argument('--save_all_models', dest="save_all_models", action="store_true")
    parser.add_argument('--verbose', dest="verbose", action="store_true")

    return parser


if __name__ == '__main__':
    parser = make_parser()
    args = vars(parser.parse_args())
    
    train(args=args, checkpoint_path=None)
