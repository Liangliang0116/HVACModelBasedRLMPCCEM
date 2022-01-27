import numpy as np

from torchlib.deep_rl import RandomAgent
from torchlib.utils.random.sampler import UniformSampler
from agent import ModelBasedHistoryPlanAgent, ModelBasedHistoryDaggerAgent, EnergyPlusDynamicsModel, \
    BestRandomActionHistoryPlanner
from agent.utils import EpisodicHistoryDataset
from agent.sampler import Sampler
from gym_energyplus import make_env, ALL_CITIES
from outdoor_temp_extract import outdoor_temp_interpolate_and_extract
from cem_rl.es_grad_im import ModelBasedCEMRLAgent


def train(args, checkpoint_path=None):
    
    dataset_maxlen = 96 * args['num_days_per_episodes'] * args['num_dataset_maxlen_days']
    max_rollout_length = 96 * args['num_days_per_episodes']
    num_on_policy_iters = (365 * args['num_years'] // args['num_days_per_episodes'] -
                           args['num_init_random_rollouts']) // args['num_days_on_policy']

    log_dir = 'runs/{}/{}'.format('_'.join(args['city']), args['algorithm'])

    env = make_env(cities=args['city'], 
                   temperature_center=args['temp_center'], 
                   temp_tolerance=args['temp_tolerance'], 
                   obs_normalize=True,
                   action_normalize=True,
                   num_days_per_episode=1, 
                   log_dir=log_dir)
    
    outdoor_temp_interpolate_and_extract(city=args['city'],
                                         weather_file=env.weather_files[0],
                                         num_years=args['num_years'],
                                         temperature_center=args['temp_center'])

    baseline_agent = RandomAgent(env.action_space)
    dataset = EpisodicHistoryDataset(maxlen=dataset_maxlen, 
                                     window_length=args['window_length'])

    print('Gathering initial dataset...')
    sampler = Sampler(env=env, 
                      window_length=args['window_length'], 
                      mpc_horizon=args['mpc_horizon'])

    initial_dataset = sampler.sample(policy=baseline_agent, 
                                     num_rollouts=args['num_init_random_rollouts'],
                                     max_rollout_length=np.inf)
    dataset.append(initial_dataset) # TODO: check the size

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
                                     batch_size=args['batch_size'], 
                                     mpc_horizon=args['mpc_horizon'], 
                                     window_length=args['window_length'], 
                                     hidden_size=32, 
                                     layer_norm=args['layer_norm'], 
                                     temperature_center=args['temp_center'],
                                     pop_size=args['pop_size'], 
                                     mem_size=args['mem_size'], 
                                     gauss_sigma=args['gauss_sigma'], 
                                     sigma_init=args['sigma_init'], 
                                     damp=args['damp'], 
                                     damp_limit=args['damp_limit'], 
                                     elitism=args['elitism'],
                                     max_steps=args['max_steps'], 
                                     start_steps=args['start_steps'], 
                                     n_grad=args['n_grad'], 
                                     n_noisy=args['n_noisy'], 
                                     n_episodes=args['n_episodes'], 
                                     period=args['period'], 
                                     n_eval=args['n_eval'], 
                                     output=args['output'], 
                                     save_all_models=args['save_all_models']
                                     )

    else:
        action_sampler = UniformSampler(low=env.action_space.low, high=env.action_space.high)
        planner = BestRandomActionHistoryPlanner(model=model, 
                                                 action_sampler=action_sampler, 
                                                 cost_fn=env.cost_fn, 
                                                 horizon=args['mpc_horizon'],
                                                 num_random_action_selection=args['num_random_action_selection'],
                                                 gamma=args['gamma']) # TODO: gamma 
        if args['algorithm'] == 'imitation_learning':
            agent = ModelBasedHistoryDaggerAgent(model=model, 
                                                 planner=planner, 
                                                 policy_data_size=10000,
                                                 window_length=args['window_length'], 
                                                 baseline_agent=baseline_agent,
                                                 state_dim=env.observation_space.shape[0],
                                                 action_dim=env.action_space.shape[0],
                                                 hidden_size=32)
        else:
            agent = ModelBasedHistoryPlanAgent(model=model, 
                                               planner=planner, 
                                               window_length=args['window_length'], 
                                               baseline_agent=baseline_agent)

    for num_iter in range(num_on_policy_iters):
        if args['verbose']:
            print('On policy iteration {}/{}. Size of dataset: {}. Number of trajectories: {}'.format(
                  num_iter + 1, num_on_policy_iters, len(dataset), dataset.num_trajectories))
        agent.set_statistics(dataset)
        agent.fit_dynamic_model(dataset=dataset, 
                                epoch=args['training_epochs'], 
                                batch_size=args['batch_size'], 
                                verbose=args['verbose'])
        
        if args['algorithm'] == 'cem_rl':
            agent.fit_policy()
        else: 
            agent.fit_policy(dataset=dataset, 
                             epoch=args['training_epochs'], 
                             batch_size=args['batch_size'], 
                             verbose=args['verbose'])  # If not dagger, this line does nothing. 
        on_policy_dataset = sampler.sample(policy=agent, 
                                           num_rollouts=args['num_days_on_policy'], 
                                           max_rollout_length=max_rollout_length)

        # record on policy dataset statistics
        if args['verbose']:
            stats = on_policy_dataset.log()
            strings = []
            for key, value in stats.items():
                strings.append(key + ": {:.4f}".format(value))
            strings = " - ".join(strings)
            print(strings)

        dataset.append(on_policy_dataset)

    if checkpoint_path:
        agent.save_checkpoint(checkpoint_path)


def make_parser():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--city', type=str, choices=ALL_CITIES, nargs='+', help='city of which the weather file is used')
    parser.add_argument('--algorithm', type=str, default='cem_rl', 
                        choices=['cem_rl', 'random_shooting', 'imitation_learning'], help='the algorithm to be trained')
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
    
    parser.add_argument('--training_epochs', type=int, default=60, help='training epochs')
    parser.add_argument('--batch_size', default=128, type=int, help='training batch size for both dynamics model and policy')
    
    parser.add_argument('--start_steps', default=10000, type=int)

    # DDPG parameters
    parser.add_argument('--actor_lr', default=0.001, type=float, help='learning rate for actor')
    parser.add_argument('--critic_lr', default=0.001, type=float, help='learning rate for critic')
    parser.add_argument('--layer_norm', dest='layer_norm', action='store_true', help='indicator wheather to use layer normalization')
    parser.add_argument('--gauss_sigma', default=0.1, type=float)

    # Evolutionary strategy parameters
    parser.add_argument('--pop_size', default=10, type=int, help='size of the evolutionary population')
    parser.add_argument('--elitism', dest="elitism",  action='store_true')
    parser.add_argument('--n_grad', default=5, type=int)
    parser.add_argument('--sigma_init', default=1e-3, type=float)
    parser.add_argument('--damp', default=1e-3, type=float)
    parser.add_argument('--damp_limit', default=1e-5, type=float)

    # Training parameters
    parser.add_argument('--n_episodes', default=100, type=int)
    parser.add_argument('--max_steps', default=1000000, type=int)
    parser.add_argument('--mem_size', default=20000, type=int)
    parser.add_argument('--n_noisy', default=0, type=int)

    # misc
    parser.add_argument('--output', default='results/', type=str)
    parser.add_argument('--period', default=5000, type=int)
    parser.add_argument('--n_eval', default=10, type=int)
    parser.add_argument('--save_all_models', dest="save_all_models", action="store_true")
    parser.add_argument('--verbose', dest="verbose", action="store_true")

    return parser


if __name__ == '__main__':
    
    parser = make_parser()
    args = vars(parser.parse_args())
    
    train(args=args, checkpoint_path=None)
