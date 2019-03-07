# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 18:04:35 2018

"""

import os
import random

import numpy as np
import pandas as pd


def __append_data(data, file_path_list, label_list, img_dir_path, label_dict):
    for sample in data:
        file_path_list.append(os.path.join(img_dir_path, sample[1]))
        label_list.append(label_dict[sample[2].lower()])


def load_dataset(img_dir_path, label_file_path, valid_rate=0.1):
    """
    This function is intended to split the dataset into training, validation, test data.
    
    Input:
        img_dir_path: directory path of 'images/';
        label_file_path: label file path e.g. 'data/legend.csv';
        valid_rate: validation data rate (0 - 1), 0.1 by default.
        
    Output:
        train_file_paths: training image file path list;
        train_labels: training label array (numpy.ndarray);
        valid_file_paths: validation image file path list;
        valid_labels: validation label array (numpy.ndarray);
        test_file_paths: test image file path list;
        test_labels: test label array (numpy.ndarray);
        label_dict: label dictionary (key: str, value: int).
    """
    
    data_frame = pd.read_csv(label_file_path).sample(frac=1, random_state=11)
    label_dict = dict()
    for label_name in set([key.lower() for key in data_frame['emotion'].unique()]):
        label_dict[label_name] = len(label_dict.keys())

    label_data_dict = dict()
    for sample in data_frame.values:
        label = sample[2]
        if label not in label_data_dict:
            label_data_dict[label] = list()
        label_data_dict[label].append(sample)

    train_file_path_list = list()
    train_label_list = list()
    valid_file_path_list = list()
    valid_label_list = list()
    test_file_path_list = list()
    test_label_list = list()
    for label in label_data_dict.keys():
        data = label_data_dict[label]
        test_size = int(len(data) * 0.1)
        train_data = data[:-test_size]
        valid_size = int(len(train_data) * valid_rate)
        __append_data(train_data[:-valid_size], train_file_path_list, train_label_list, img_dir_path, label_dict)
        __append_data(train_data[-valid_size:], valid_file_path_list, valid_label_list, img_dir_path, label_dict)
        __append_data(data[-test_size:], test_file_path_list, test_label_list, img_dir_path, label_dict)

    zipped_train_list = list(zip(train_file_path_list, train_label_list))
    random.shuffle(zipped_train_list)
    train_file_paths, train_labels = zip(*zipped_train_list)
    return list(train_file_paths), np.array(list(train_labels)), valid_file_path_list, np.array(valid_label_list),\
           test_file_path_list, np.array(test_label_list), label_dict
           
