import pandas as pd
import numpy as np


def postprocessing_data(log_dir, 
                        num_years=2, 
                        temperature_center=23.5,
                        tolerance=1.5,
                        lambda_1=0.5,
                        lambda_2=0.1,
                        energy_weight=1/100000):
    """ Calculate the reward related values from the raw data saved when training. 

    Args:
        log_dir (str): log directory of the raw data
        num_years (int, optional): number of simulation years in EnergyPlus. Defaults to 2. 
                                   Typically, only the raw data in the last simulation year is considered. 
        temperature_center (float, optional): temperature center that is preferred. Defaults to 23.5.
        tolerance (float, optional): temperature deviation from the center to avoid uncomfortableness. Defaults to 1.5.
        lambda_1 (float, optional): coefficient to combine different reward terms. Defaults to 0.5.
        lambda_2 (float, optional): coefficient to combine different reward terms. Defaults to 0.1.
        energy_weight ([type], optional): coefficient to combine different reward terms. Defaults to 1/100000.
    """
    
    csv_file = log_dir + '/episode-{}.csv'.format(num_years-1)
    csv_file_processed = log_dir + '/episode-{}_processed.csv'.format(num_years-1)

    df = pd.read_csv(csv_file)
    data = df.to_numpy()
    total_data_num = data.shape[0]
    
    # violation_num = 0
    # for i in range(total_data_num):
    #     if (i % 96 >= 24 and abs(data[i, 1] - 23.5) > 1.5) or (i % 96 < 24 and abs(data[i, 1] - 23.5) > 1.5):
    #         violation_num += 1
    #     if (i % 96 >= 24 and abs(data[i, 2] - 23.5) > 1.5) or (i % 96 < 24 and abs(data[i, 2] - 23.5) > 1.5):
    #         violation_num += 1

    reward_center_temp = []
    reward_trapezoid = []
    reward_energy = []
    temp_dev_square_zone1 = []
    temp_dev_square_zone2 = []

    for i in range(total_data_num):
        reward_center_temp_i = np.exp(- lambda_1 * (data[i, 1] - temperature_center) ** 2 
                                      - lambda_1 * (data[i, 2] - temperature_center) ** 2)
        reward_trapezoid_i = lambda_2 * (min(0, data[i, 1] - temperature_center + tolerance) + 
                                         min(0, temperature_center + tolerance - data[i, 1]) +
                                         min(0, data[i, 2] - temperature_center + tolerance) + 
                                         min(0, temperature_center + tolerance - data[i, 2]))
        reward_energy_i = - energy_weight * (data[i, 3] + data[i, 4])
        reward_center_temp.append(reward_center_temp_i)
        reward_trapezoid.append(reward_trapezoid_i)
        reward_energy.append(reward_energy_i)
        temp_dev_square_zone1_i = (data[i, 1] - temperature_center) ** 2
        temp_dev_square_zone1.append(temp_dev_square_zone1_i)
        temp_dev_square_zone2_i = (data[i, 2] - temperature_center) ** 2
        temp_dev_square_zone2.append(temp_dev_square_zone2_i)

    reward_center_temp = np.asarray(reward_center_temp)
    reward_trapezoid = np.asarray(reward_trapezoid)
    reward_energy = np.asarray((reward_energy))
    reward_total = reward_center_temp + reward_trapezoid + reward_energy
    temp_dev_square_zone1 = np.asarray(temp_dev_square_zone1)
    temp_dev_square_zone2 = np.asarray(temp_dev_square_zone2)

    df['reward_center_temp'] = reward_center_temp
    df['reward_trapezoid'] = reward_trapezoid
    df['reward_energy'] = reward_energy
    df['reward_total'] = reward_total
    df['temp_dev_square_zone1'] = temp_dev_square_zone1
    df['temp_dev_square_zone2'] = temp_dev_square_zone2
    df.to_csv(csv_file_processed, index=False)
    
    
if __name__ == '__main__':
    """
    I do not know why implementing the function postprocessing_data from train_model_based.py 
    cannot generate correct results in my computer. This is quite strange. 
    
    If this issue happens, you need to run this file independently with manually typed log_dir. 
    
    In addition, one direct method to check if the issue above happens is to check if the 
    number of rows in the generated csv file is the same as that in the original csv file. 
    
    """
    postprocessing_data(log_dir='results/SF/ppo',
                        num_years=20)
    