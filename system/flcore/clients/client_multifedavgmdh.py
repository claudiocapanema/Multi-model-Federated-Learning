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

import random
import sys
import copy
import os
import torch
import numpy as np
from .client_multifedavg import MultiFedAvgClient
from .utils.models_utils import load_model, get_weights, load_data, set_weights, test, train



import logging
logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module


class MultiFedAvgMDHClient(MultiFedAvgClient):
    def __init__(self, args, id, model):
        try:
            super().__init__(args, id, model)
            self.p_ME = [None] * self.ME
            self.p_ME_list = {me: [] for me in range(self.ME)}
            self.fc_ME = [0] * self.ME
            self.il_ME = [0] * self.ME
            self.num_examples = [0] * self.ME
            self.similarity_ME = [[]] * self.ME
            self.mean_p_ME = [None] * self.ME
            self.NT = [None] * self.ME
            self.previous_alpha = self.alpha

            self.num_examples, self.p_ME, self.fc_ME, self.il_ME = self._get_datasets_metrics_meh(self.trainloader, self.ME, self.client_id,
                                                                           self.n_classes)
        except Exception as e:
            logger.error("MultiFedAvgMDHClient __init__ error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))


    def _get_datasets_metrics_meh(self, trainloader, ME, client_id, n_classes, concept_drift_window=None):

        try:
            p_ME = []
            fc_ME = []
            il_ME = []
            num_examples = [0] * self.ME
            for me in range(ME):
                labels_me = []
                n_classes_me = n_classes[me]
                p_me = {i: 0 for i in range(n_classes_me)}
                with (torch.no_grad()):
                    for batch in trainloader[me]:
                        labels = batch["label"]
                        num_examples[me] += labels.shape[0]

                        if concept_drift_window is not None:
                            labels = (labels + concept_drift_window[me])
                            labels = labels % n_classes[me]
                        labels = labels.detach().cpu().numpy()
                        labels_me += labels.tolist()
                    unique, count = np.unique(labels_me, return_counts=True)
                    data_unique_count_dict = dict(zip(np.array(unique).tolist(), np.array(count).tolist()))
                    for label in data_unique_count_dict:
                        p_me[label] = data_unique_count_dict[label]
                    p_me = np.array(list(p_me.values()))
                    fc_me = len(np.argwhere(p_me > 0)) / n_classes_me
                    il_me = len(np.argwhere(p_me < np.sum(p_me) / n_classes_me)) / n_classes_me
                    p_me = p_me / np.sum(p_me)
                    p_ME.append(p_me)
                    fc_ME.append(fc_me)
                    il_ME.append(il_me)
                    logger.info(f"p_me {p_me} fc_me {fc_me} il_me {il_me} model {me} client {client_id}")
            return num_examples, p_ME, fc_ME, il_ME
        except Exception as e:
            logger.error("_get_datasets_metrics error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))


        



