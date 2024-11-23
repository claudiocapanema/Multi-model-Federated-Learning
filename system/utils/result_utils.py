# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import pandas as pd
import numpy as np
import os


def average_data(algorithm="", dataset="", goal="", times=10, args={}):

    for m in range(len(dataset)):
        test_acc = get_all_results_for_one_algo(algorithm, dataset[m], goal, times, args)

        max_accurancy = []
        for i in range(times):
            max_accurancy.append(test_acc[i].max())

        print("std for best accuracy:", np.std(max_accurancy))
        print("mean for best accuracy:", np.mean(max_accurancy))


def get_all_results_for_one_algo(algorithm="", dataset="", goal="", times=10, args={}):
    test_acc = []
    algorithms_list = [algorithm] * times
    for i in range(times):
        file_name = """../results/clients_{}/alpha_{}/fc_{}/rounds_{}/epochs_{}/""".format(args.num_clients, args.alpha, args.join_ratio, args.global_rounds, args.local_epochs) + dataset + "_" + algorithms_list[i] + "_" + goal + "_" + str(i)
        test_acc.append(np.array(read_data_then_delete(file_name, delete=False)))

    return test_acc


def read_data_then_delete(file_name, delete=False):
    file_path = "../results/" + file_name + ".csv"

    rs_test_acc =pd.read_csv(file_path)['Accuracy'].to_numpy()

    if delete:
        os.remove(file_path)
    print("Length: ", len(rs_test_acc))

    return rs_test_acc