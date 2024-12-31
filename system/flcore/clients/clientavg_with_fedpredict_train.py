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

import copy
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client
from fedpredict import fedpredict_client_torch
from sklearn.metrics.pairwise import cosine_similarity
import os
import sys


class clientAVGWithFedPredictTrain(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.last_training_round = 0

    def train(self, m, global_model, t, global_model_training_class_count):
        train_class_count = self.train_class_count[m] / np.sum(self.train_class_count[m])
        print(global_model_training_class_count.shape, train_class_count.shape)
        s = cosine_similarity([global_model_training_class_count], [train_class_count])
        trainloader = self.trainloader[m]
        nt = t - self.last_training_round
        self.set_parameters_combined_model(m, global_model, t, nt, s)
        self.last_training_round = t
        self.model[m].to(self.device)
        self.model[m].train()
        
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                if type(y) == tuple:
                    y = torch.from_numpy(np.array(list(y), dtype=np.int32))
                y = y.type(torch.LongTensor).to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model[m](x).to(self.device)
                loss = self.loss(output, y)
                self.optimizer[m].zero_grad()
                loss.backward()
                self.optimizer[m].step()

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler[m].step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_parameters_combined_model(self, m, global_model, t, nt, s):

        combined_model = self.fedpredict_client_dynamic_torch(local_model=self.model[m], global_model=global_model,
                                                 t=t, T=100, nt=nt, M=[], similarity=s)

        print("combinado: ", combined_model)
        for new_param, old_param in zip(self.model[m].parameters(), combined_model.parameters()):
            old_param.data = new_param.data.clone()

    def fedpredict_client_dynamic_torch(self,
                                        local_model,
                                        global_model,
                                        t,
                                        T,
                                        nt,
                                        M,
                                        similarity,
                                        filename = '',
                                        knowledge_distillation=False,
                                        decompress=False):
        """

        Args:
            local_model: torch.nn.Module, required
            global_model: torch.nn.Module or List[NDArrays], required
            current_proportion: List[float], required
            t: int, required
            T: int, required
            nt: int, required
            M: list, required
                The list of the indexes of the shared global model layers
            local_client_information: dict, required
            filename: str, optional. Default=''
            knowledge_distillation: bool, optional. Default=False
                If the model has knowledge distillation, then set True, to indicate that the global model parameters have
                to be combined with the student model
            decompress: bool, optional. Default=False
               Whether or not to decompress global model parameters in case a previous compression was applied. Only set
                   True if using "FedPredict_server" and compressing the shared parameters.

        Returns: torch.nn.Module
            The combined model

        """
        # Using 'torch.load'
        try:

            if knowledge_distillation:
                model_shape = [i.detach().cpu().numpy().shape for i in local_model.student.parameters()]
            else:
                model_shape = [i.detach().cpu().numpy().shape for i in local_model.parameters()]
            print("a d: ", type(global_model))
            M = 0
            for new_param, old_param in zip(global_model.parameters(), local_model.parameters()):
                # old_param.data = new_param.data.clone()
                print("camada")
                M = M +1
            M = [i for i in range(M)]
            # if len(M) == 0:
            #     # M = [i for i in range(len(global_model))]
            #     M = 0
            #     for server_param in global_model.parameters():
            #         M += 1

            # if len(global_model) != len(M):
            #     raise Exception(
            #         """Lenght of parameters of the global model is {} and is different from the M {}""".format(
            #             len(global_model), len(M)))

            local_model = self.fedpredict_dynamic_combine_models(global_model, local_model, t, T, nt, M,
                                                                 similarity)

            if os.path.exists(filename):
                # Load local parameters to 'self.model'
                # print("existe modelo local")
                local_model.load_state_dict(torch.load(filename))
                local_model = self.fedpredict_dynamic_combine_models(global_model, local_model, t, T, nt, M,
                                                                similarity)
            # else:
            #     # print("usar modelo global: ", cid)
            #     if not knowledge_distillation:
            #         for old_param, new_param in zip(local_model.parameters(), global_model.parameters()):
            #             old_param.data = new_param.data.clone()
            #     else:
            #         local_model.new_client = True
            #         for old_param, new_param in zip(local_model.student.parameters(), global_model):
            #             old_param.data = new_param.data.clone()

            return local_model

        except Exception as e:
            print("FedPredict dynamic client")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    def fedpredict_dynamic_combine_models(self, global_parameters, model, t, T, nt, M, similarity):
        try:

            local_model_weights, global_model_weight = self.fedpredict_dynamic_core(t, T, nt, similarity)
            count = 0
            global_parameters.to(self.device)
            model.to(self.device)
            for new_param, old_param in zip(global_parameters.parameters(), model.parameters()):
                if count in M:
                    if new_param.shape == old_param.shape:
                        # percent = (count) / len(M)
                        # global_model_weight = -percent + 1
                        # local_model_weights = percent
                        # print("peso global: ", global_model_weight, " peso local", local_model_weights, " camada: ", count)
                        old_param.data = (
                                global_model_weight * new_param.data.clone() + local_model_weights * old_param.data.clone())
                    else:
                        print("Não combinou, CNN student: ", new_param.shape, " CNN 3 proto: ", old_param.shape)
                count += 1

            return model

        except Exception as e:
            print("FedPredict dynamic combine models")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    def fedpredict_dynamic_core(self, t, T, nt, similarity):
        try:
            print("fedpredict_dynamic_core rodada: ", t, "local classes: ", similarity)
            similarity = float(np.round(similarity, 1))

            if nt == 0:
                global_model_weight = 0
            elif nt == t:
                global_model_weight = 1
            # elif similarity != 1:
            #     global_model_weight = 1
            else:
                # update_level = 1 / nt
                evolution_level = t / int(T)
                eq1 = (- evolution_level) * similarity
                eq2 = round(np.exp(eq1), 6)
                global_model_weight = eq2
                # global_model_weight = 0

            # global_model_weight = similarity

            local_model_weights = 1 - global_model_weight

            print("rodada: ", t, " rounds sem fit: ", nt, "\npeso global: ", global_model_weight, " peso local: ",
                  local_model_weights)

            return local_model_weights, global_model_weight

        except Exception as e:
            print("fedpredict core")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)


