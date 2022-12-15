import pandas as pd
import numpy as np

def dt64_to_float(dt64):
    """Converts numpy.datetime64 to year as float.

    Rounded to days

    Parameters
    ----------
    dt64 : np.datetime64 or np.ndarray(dtype='datetime64[X]')
        date data

    Returns
    -------
    float or np.ndarray(dtype=float)
        Year in floating point representation
    """
    # print(dt64)
    year = dt64.astype('M8[Y]')
    # print('year:', year)
    days = (dt64 - year).astype('timedelta64[D]')
    # print('days:', days)
    year_next = year + np.timedelta64(1, 'Y')
    # print('year_next:', year_next)
    days_of_year = (year_next.astype('M8[D]') - year.astype('M8[D]')
                    ).astype('timedelta64[D]')
    # print('days_of_year:', days_of_year)
    dt_float = 1970 + year.astype(float) + days / (days_of_year)
    # print('dt_float:', dt_float)
    return dt_float


def extract_dataframe(name):

    folder = "./data/{}.csv".format(name)
    df =  pd.read_csv(folder)
    return df


def exctract_dataset_array(name):

    df = extract_dataframe(name)

    time_col = df.columns[0]
    temperature_cols = list(df.columns)
    temperature_cols.pop(0)

    temperature = df[temperature_cols].astype(float)
    time = df[time_col]

    array_temperature = temperature.to_numpy()
    array_time_dt64 = np.array(time,dtype=np.datetime64)
    array_time_float = dt64_to_float(np.array(time,dtype=np.datetime64))

    return array_temperature, array_time_float, array_time_dt64


def timeseries_split_dataset(temperature, time_float, time_dt64 = None, proportion_test = 0.8, proportion_train = 0.2):
    
    if proportion_test + proportion_train > 1:
        raise Exception("The cumulative proportion of training and testing set can't be over 1") 

    len = temperature.shape[0]

    training_set = []
    testing_set = []

    training_set.append( temperature[:int( len * proportion_test )] )
    training_set.append( time_float[:int( len * proportion_test )] )
    
    testing_set.append( temperature[int( len * proportion_test ):int( len * (proportion_test + proportion_train) )] )
    testing_set.append( time_float[int( len * proportion_test ):int( len * (proportion_test + proportion_train) )] )


    if time_dt64 is not None:
        training_set.append( time_dt64[:int( len * proportion_test )] )
        testing_set.append( time_dt64[int( len * proportion_test ):int( len * (proportion_test + proportion_train) )] )

    return (training_set, testing_set)


def rescale_time(time, initial_time = 1981.6657534246576, final_time = 2017):
    return (time - initial_time)/(final_time - initial_time)


def normalize_temperature(temperature):
    max = np.max(temperature)
    min = np.min(temperature)
    return (temperature-min)/(max-min)-1/2, max, min

def denormalize_temperature(temperature_normalized, max, min):
    return (temperature_normalized+1/2)*(max-min) + min

def generate_basic_timeseries_splitted_normalized_dataset(name, proportion_test = 0.8, rescale = True):

    temperature, time_float, time_dt64 = exctract_dataset_array(name)
    temperature, max, min = normalize_temperature(temperature)

    if rescale:
        time_float = rescale_time(time_float)
    
    return timeseries_split_dataset(temperature, time_float, time_dt64 = time_dt64, proportion_test = proportion_test, proportion_train = 1 - proportion_test), max, min


