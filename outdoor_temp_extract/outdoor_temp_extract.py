import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from ladybug.epw import EPW


def outdoor_temp_interpolate_and_extract(city,
                                         weather_file, 
                                         num_years,
                                         temperature_center=23.5, 
                                         obs_max=30.0,
                                         num_timesteps_interpolate=4):

    epw = EPW(weather_file)

    data = [
        'dry_bulb_temperature'
    ]

    df = pd.DataFrame({d:getattr(epw, d).values for d in data})
    df.to_csv('outdoor_temp_extract/original_outdoor_temp_{}.csv'.format('_'.join(city)))
    df_numpy = df.to_numpy()
    df_numpy = np.squeeze(df_numpy, axis=1)
    num_total_hours = df_numpy.shape[0]

    t = np.arange(num_total_hours)
    interp_func = interp1d(t, df_numpy)
    t_new = np.arange(0, num_total_hours - 1, 1 / num_timesteps_interpolate)
    df_new = interp_func(t_new)
    df_new = np.tile(df_new, num_years+1)
    df_new_scaled = np.zeros(df_new.shape[0])
    df_new_scaled[:] = (df_new[:] - temperature_center) / obs_max
    pd.DataFrame(df_new_scaled).to_csv('outdoor_temp_extract/interpolated_outdoor_temp_{}.csv'.format('_'.join(city)))

