# Hackathon for Generative modeling - MAP670U

## Authors

| LAST NAME       | First name |
| --------------- | ---------- |
| COTTART         | Kellian    |
| BERSANI--VERONI | Thomas     |
| WASIK           | Thomas     |

## Description

The purpose of this challenge is to make a simulation of global warming sea surface temperatures.

We have access to 9618 days with one temperature for each day around 10 stations.

This data corresponds to all the days between `1981-09-01` and `2007-12-31`. It is our **training set**. We now have to generate data between `2008-01-01` and `2016-12-31`.

## Details

By splitting this training set into a train set and a test set with a proportionnality of 85% - 15%, we gather the data to extrapolate and generate the data of the following years.

Using a conditional NICE, we keep the months informations to train the model following a time series behavior, without properly being one.

## model.py

The file `model.py` contains the creation of a uniformly selected month system for the noise to imitate a time series.

As the data of our model is normalized to process it, we need to denormalize it to have a proper output for the simulation.
