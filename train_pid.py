import numpy as np

from torchlib.deep_rl import BaseAgent
from gym_energyplus import make_env, ALL_CITIES
from postprocessing_data import postprocessing_data


class PIDAgent(BaseAgent):
    def __init__(self, target, sensitivity=1.0, alpha=0.5):
        self.sensitivity = sensitivity
        self.act_west_prev = target
        self.act_east_prev = target
        self.alpha = alpha
        self.target = target

        self.lo = 10.0
        self.hi = 40.0
        self.flow_hi = 7.0
        self.flow_lo = self.flow_hi * 0.25

        self.default_flow = 7.0

        self.low = np.array([self.lo, self.lo, self.flow_lo, self.flow_lo])
        self.high = np.array([self.hi, self.hi, self.flow_hi, self.flow_hi])

    def predict(self, state):
        delta_west = state[1] - self.target
        act_west = self.target - delta_west * self.sensitivity
        act_west = act_west * self.alpha + self.act_west_prev * (1 - self.alpha)
        self.act_west_prev = act_west

        delta_east = state[2] - self.target
        act_east = self.target - delta_east * self.sensitivity
        act_east = act_east * self.alpha + self.act_east_prev * (1 - self.alpha)
        self.act_east_prev = act_east

        act_west = max(self.lo, min(act_west, self.hi))
        act_east = max(self.lo, min(act_east, self.hi))
        action = np.array([act_west, act_east, self.default_flow, self.default_flow])
        return action


def make_parser():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--city', type=str, choices=ALL_CITIES, nargs='+')
    parser.add_argument('--temp_center', type=float, default=23.5)
    parser.add_argument('--temp_tolerance', type=float, default=1.5)
    parser.add_argument('--sensitivity', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--new_log_dir', dest='new_log_dir', action='store_true')
    parser.add_argument('--log_dir', default='results/', type=str)
    return parser


if __name__ == '__main__':

    parser = make_parser()
    args = vars(parser.parse_args())

    city = args['city']
    temperature_center = args['temp_center']
    temperature_tolerance = args['temp_tolerance']
    sensitivity = args['sensitivity']
    alpha = args['alpha']
    
    if args['new_log_dir']:
        log_dir = args['log_dir'] + '{}/pid/{}'.format('_'.join(args['city']), time.strftime("%Y%m%d-%H%M%S"))
    else:
        log_dir = args['log_dir'] + '{}/pid'.format('_'.join(args['city']))

    env = make_env(cities=city, 
                   temperature_center=temperature_center, 
                   temp_tolerance=temperature_tolerance, 
                   obs_normalize=False, 
                   action_normalize=False,
                   num_days_per_episode=1, 
                   log_dir=log_dir)

    true_done = False
    day_index = 1

    agent = PIDAgent(target=temperature_center-3.5, sensitivity=sensitivity, alpha=alpha)

    while not true_done:
        obs = env.reset()
        print('Day {}'.format(day_index))
        done = False
        info = None
        r = 0.
        while not done:
            action = agent.predict(obs)
            obs, reward, done, info = env.step(action)
            r += reward
        print('Total reward: {:.4f}'.format(r))
        day_index += 1
        true_done = info['true_done']
        
    postprocessing_data(log_dir=log_dir, 
                        num_years=1,
                        temperature_center=args['temp_center'],
                        tolerance=args['temp_tolerance'])
