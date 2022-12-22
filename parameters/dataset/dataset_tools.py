import pandas as pd
import numpy as np
import torch


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


def get_month(dt64):
    month = np.array([dt64[i].astype('datetime64[M]').astype(
        int) % 12 for i in range(len(dt64))])

    month_indexed = np.zeros((month.shape[0], 12)).astype(float)
    for i in range(month.shape[0]):
        month_indexed[i, month[i]] = 1

    return month_indexed


def get_month_from_scaled_float(float, initial_time=1981.6657534246576, final_time=2017):
    if not torch.is_tensor(float):
        raise NotImplementedError
    float = float*(final_time - initial_time) + final_time
    month = float - float.type(torch.IntTensor)
    month = (month * 12).type(torch.IntTensor)
    return month


def extract_dataframe(name):

    folder = "./data/{}.csv".format(name)
    df = pd.read_csv(folder)
    return df


def extract_dataset_array(name):

    df = extract_dataframe(name)

    time_col = df.columns[0]
    temperature_cols = list(df.columns)
    temperature_cols.pop(0)

    temperature = df[temperature_cols].astype(float)
    time = df[time_col]

    array_temperature = temperature.to_numpy()
    array_time_dt64 = np.array(time, dtype=np.datetime64)
    array_time_float = dt64_to_float(np.array(time, dtype=np.datetime64))

    return array_temperature, array_time_float, array_time_dt64


def timeseries_split_dataset(temperature, time_float, time_dt64=None, proportion_test=0.8, proportion_train=0.2):

    if proportion_test + proportion_train > 1:
        raise Exception(
            "The cumulative proportion of training and testing set can't be over 1")

    len = temperature.shape[0]

    training_set = []
    testing_set = []

    training_set.append(temperature[:int(len * proportion_test)])
    training_set.append(time_float[:int(len * proportion_test)])

    testing_set.append(temperature[int(
        len * proportion_test):int(len * (proportion_test + proportion_train))])
    testing_set.append(time_float[int(
        len * proportion_test):int(len * (proportion_test + proportion_train))])

    if time_dt64 is not None:
        training_set.append(time_dt64[:int(len * proportion_test)])
        testing_set.append(time_dt64[int(
            len * proportion_test):int(len * (proportion_test + proportion_train))])

    return (training_set, testing_set)


def rescale_time(time, initial_time=1981.6657534246576, final_time=2017):
    time[:, 0] = (time[:, 0] - initial_time)/(final_time - initial_time)
    return time


def normalize_temperature(temperature):
    max = np.max(temperature)
    min = np.min(temperature)
    return (temperature-min)/(max-min)-1/2, max, min


def denormalize_temperature(temperature_normalized, max, min):
    return (temperature_normalized+1/2)*(max-min) + min


def generate_basic_timeseries_splitted_normalized_dataset(name, proportion_test=0.8, rescale=True):

    temperature, time_float, time_dt64 = extract_dataset_array(name)
    temperature, max, min = normalize_temperature(temperature)

    time_float = np.expand_dims(time_float, axis=1)

    if rescale:
        time_float = rescale_time(time_float)

    return timeseries_split_dataset(temperature, time_float, time_dt64=time_dt64, proportion_test=proportion_test, proportion_train=1 - proportion_test), max, min


def generate_basic_timeseries_splitted_normalized_dataset_with_month_info(name, proportion_test=0.8, rescale=True):

    temperature, time_float, time_dt64 = extract_dataset_array(name)
    time_month = get_month(time_dt64)
    time_float = np.concatenate(
        (np.expand_dims(time_float, axis=1), time_month), axis=1)

    temperature, max, min = normalize_temperature(temperature)

    if rescale:
        time_float = rescale_time(time_float)

    return timeseries_split_dataset(temperature, time_float, time_dt64=time_dt64, proportion_test=proportion_test, proportion_train=1 - proportion_test), max, min
