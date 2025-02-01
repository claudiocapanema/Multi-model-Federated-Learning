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
from utils.privacy import *


class clientFedNome(Client):
    def __init__(self, args, id, **kwargs):
        super().__init__(args, id,  **kwargs)

    def train(self, m, global_model, client_cosine_similarity):
        trainloader = self.trainloader[m]
        self.set_parameters(m, global_model)
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
                self.optimizer[m].zero_grad()
                output = self.model[m](x).to(self.device)
                loss = self.loss(output, y)

                loss.backward()
                self.optimizer[m].step()

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler[m].step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    # def set_parameters(self, m, model, cosine_similarity):
    #     self.model[m].to('cpu')
    #     model.to('cpu')
    #     size = len([i.detach().cpu() for i in self.model[m].parameters()])
    #     count = 1
    #     for new_param, old_param in zip(model.parameters(), self.model[m].parameters()):
    #         if count >= size -1 and m == 0:
    #             continue
    #             old_param.data = (1-cosine_similarity) * old_param.data.clone() + cosine_similarity * new_param.data.clone()
    #         else:
    #             old_param.data = new_param.data.clone()
    #         count += 1
