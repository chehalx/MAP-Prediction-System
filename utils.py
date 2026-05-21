#!/usr/bin/env python3

import time
import math
import subprocess
import sys
import random
import webbrowser
import numpy as np
from datetime import datetime
from dateutil import tz
import pytz
import os, sys

import operator
import copy
from collections import Counter

from scipy.stats import norm
from scipy.special import softmax
import pandas as pd

import matplotlib.pyplot as plt
import re


# get_ipython().system(' pip install tsfel # installing TSFEL for feature extraction')
def str2bool(v):
  return v.lower() in ("true", "1", "https", "load")

def mac_to_int(mac):
    res = re.match('^((?:(?:[0-9a-f]{2}):){5}[0-9a-f]{2})$', mac.lower())
    if res is None:
        raise ValueError('invalid mac address')
    return int(res.group(0).replace(':', ''), 16)

def int_to_mac(macint):
    # if type(macint) != int:
    #     raise ValueError('invalid integer')
    newint = int(macint)
    return ':'.join(['{}{}'.format(a, b)
                     for a, b
                     in zip(*[iter('{:012x}'.format(newint))]*2)])

# This function converts the time string to epoch time xxx.xxx (second.ms).
# Example: time = "2020-08-13T02:03:00.200", zone = "UTC" or "America/New_York"
# If time = "2020-08-13T02:03:00.200Z" in UTC time, then call timestamp = local_time_epoch(time[:-1], "UTC"), which removes 'Z' in the string end
def local_time_epoch(time, zone):
    local_tz = pytz.timezone(zone)
    localTime = datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%f")
    local_dt = local_tz.localize(localTime, is_dst=None)
    # utc_dt = local_dt.astimezone(pytz.utc)
    epoch = local_dt.timestamp()
    # print("epoch time:", epoch) # this is the epoch time in seconds, times 1000 will become epoch time in milliseconds
    # print(type(epoch)) # float
    return epoch

# This function converts the epoch time xxx.xxx (second.ms) to time string.
# Example: time = "2020-08-13T02:03:00.200", zone = "UTC" or "America/New_York"
# If time = "2020-08-13T02:03:00.200Z" in UTC time, then call timestamp = local_time_epoch(time[:-1], "UTC"), which removes 'Z' in the string end
def epoch_time_local(epoch, zone):
    local_tz = pytz.timezone(zone)
    time = datetime.fromtimestamp(epoch).astimezone(local_tz).strftime("%Y-%m-%dT%H:%M:%S.%f")
    return time

# This function converts the grafana URL time to epoch time. For exmaple, given below URL
# https://sensorweb.us:3000/grafana/d/OSjxFKvGk/caretaker-vital-signs?orgId=1&var-mac=b8:27:eb:6c:6e:22&from=1612293741993&to=1612294445244
# 1612293741993 means epoch time 1612293741.993; 1612294445244 means epoch time 1612294445.244
def grafana_time_epoch(time):
    return time/1000

def influx_time_epoch(time):
    return time/10e8


def load_data_file(data_file):
    if data_file.endswith('.csv'):
        data_set = pd.read_csv(data_file).to_numpy()
    elif data_file.endswith('.npy'):
        data_set = np.load(data_file)
    return data_set


def calc_mae(gt, pred):
    return np.mean(abs(np.array(gt)-np.array(pred)))
    
# list1: label; list2: prediction
def plot_2vectors(label, pred, name):
    list1 = label
    list2 = np.array(pred)
    if len(list2.shape) == 2:
        mae = calc_mae(list1, list2[:,0])
    else:
        mae = calc_mae(list1, list2)

    # zipped_lists = zip(list1, list2)
    # sorted_pairs = sorted(zipped_lists)

    # tuples = zip(*sorted_pairs)
    # list1, list2 = np.array([ list(tuple) for tuple in  tuples])

    # print(list1.shape)
    # print(list2.shape)
    
    sorted_id = sorted(range(len(list1)), key=lambda k: list1[k])

    plt.clf()
    plt.text(0,np.min(list2),f'MAE={mae}')

    # plt.plot(range(num_rows), list2, label=name + ' prediction')
    plt.scatter(np.arange(list2.shape[0]),list2[sorted_id],s = 1, alpha=0.5,label=f'{name} prediction', color='blue')

    plt.scatter(np.arange(list1.shape[0]),list1[sorted_id],s = 1, alpha=0.5,label=f'{name} label', color='red')

    # plt.plot(range(num_rows), list1, 'r.', label=name + ' label')

    plt.legend()
    plt.savefig(f'{name}.png')
    print(f'Saved plot to {name}.png')
    plt.show()