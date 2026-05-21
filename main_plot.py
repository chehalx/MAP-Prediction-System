# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from utils import * 

# # data_path = './data/all_good_train.1000_6.npy'
# data_file = './data/all_good_test.1000_6.npy'
# prediction_file = './data/prediction.npy'

# # hr_prediction, hr_MAE, rr_prediction, rr_MAE, 
# sp_prediction, sp_MAE, dp_prediction, dp_MAE = prediction(data_file)
# np.save(prediction_file, [hr_prediction, rr_prediction, sp_prediction, dp_prediction] )
# # print('===============')
# # print('hear rate MAE:', hr_MAE)
# # print('===============')
# # print('respiratory rate MAE:', rr_MAE)
# print('===============')
# print('Systolic MAE:', sp_MAE)
# print('===============')
# print('Diastolic MAE:', dp_MAE)
# print('===============')


# data_set = np.load(data_path)
# # hr_label = data_set[:, -4]
# # rr_label = data_set[:, -3]
# sp_label = data_set[:, -2]
# dp_label = data_set[:, -1]

# # [hr_prediction, rr_prediction, sp_prediction, dp_prediction] = np.load(prediction_path)
# [sp_prediction, dp_prediction] = np.load(prediction_path)


# plot_2vectors(hr_label, hr_prediction, 'hr')
# plot_2vectors(rr_label, rr_prediction, 'rr')
plot_2vectors(sp_label, sp_prediction, 'sp')
plot_2vectors(dp_label, dp_prediction, 'dp')




