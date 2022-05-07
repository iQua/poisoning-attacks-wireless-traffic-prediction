# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: attacks.py
-----------------------------------------------
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn import metrics
import copy

import torch.nn.functional as F
import torch.autograd as autograd
from collections import OrderedDict
from functools import partial
import warnings
from utils.models import LSTM, CNNLSTM
from redundants.train_utils import normal_weight_init

import higher
import tqdm

torch.manual_seed(2021)
np.random.seed(2021)

'''Basic Model Poison
'''

class NegMSE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()
    def forward(self, y, pred):
        return -self.criterion(y, pred)


class ModelPoison(object):
    def __init__(self, args, train, test, historic_global_model=None, target=None, scale_up=1.0):
        self.args = args

        self.train_loader = self.process_data(train)
        self.test_loader = self.process_data(test)

        self.historic_global_model = historic_global_model

        self.device = 'cuda' if args.gpu else 'cpu'
        self.scale_up = scale_up

        if target is None:
            target = copy.deepcopy(train)
            self.criterion = NegMSE().to(self.device)
        else:
            self.criterion = nn.MSELoss().to(self.device)

        self.target_loader = self.process_data(target)

    def process_data(self, dataset):
        data = list(zip(*dataset))
        if self.args.fedsgd == 1:
            loader = DataLoader(data, shuffle=True, batch_size=len(data))
        else:
            loader = DataLoader(data, shuffle=False, batch_size=self.args.local_bs)
        return loader

    def update_weights(self, model, global_round):
        global_model = copy.deepcopy(model)
        model.train()
        epoch_loss = []
        lr = self.args.lr

        if self.args.opt == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif self.args.opt == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=self.args.momentum)

        for iter in range(self.args.local_epoch):
            batch_loss = []

            for batch_idx, (xc, xp, y) in enumerate(self.target_loader):
                xc, xp = xc.float().to(self.device), xp.float().to(self.device)
                y = y.float().to(self.device)

                model.zero_grad()
                pred = model(xc, xp)

                loss = self.criterion(y, pred)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))



        ### scale up
        attack_weights = self.scaleup_attack_weights(model.state_dict(), global_model.state_dict(), self.scale_up)

        return attack_weights, sum(epoch_loss)/len(epoch_loss), epoch_loss

    def scaleup_attack_weights(self, local_weights, global_weights, scale_up):
        result = copy.deepcopy(local_weights)
        for key in local_weights.keys():
            result[key] = scale_up * (local_weights[key] - global_weights[key]) + global_weights[key]
        return result

    def clamp_weights(self, local_weights, global_weights, historic_global_weights, scale_up):
        result = copy.deepcopy(local_weights)
        for key in local_weights.keys():
            max_updates = scale_up * torch.abs(historic_global_weights[key] - global_weights[key])
            result[key] = torch.min(torch.max(local_weights[key] - global_weights[key], -max_updates), max_updates) + global_weights[key]
        return result

    def flatten_all_updates(self, global_weights, user_local_weights):
        all_user_updates = [[] for i in range(len(user_local_weights))]
        for key in global_weights.keys():
            for i in range(len(all_user_updates)):
                all_user_updates[i] = (user_local_weights[i][key] - global_weights[key]).data.view(-1) \
                                      if not len(all_user_updates[i]) else torch.cat((all_user_updates[i], \
                                                        (user_local_weights[i][key] - global_weights[key]).view(-1)))
        return all_user_updates





'''Adaptive Model Poison
'''

class AdaptiveModelPoison(ModelPoison):
    def __init__(self, args, train, test, historic_global_model=None,
                                            target=None, defense=None):
        self.args = args

        self.train_loader = self.process_data(train)
        self.test_loader = self.process_data(test)
        self.historic_global_model = historic_global_model
        self.device = 'cuda' if args.gpu else 'cpu'

        if target is None:
            target = copy.deepcopy(train)
            self.criterion = NegMSE().to(self.device)
        else:
            self.criterion = nn.MSELoss().to(self.device)

        self.target_loader = self.process_data(target)
        self.defense = defense


    def update_weights(self, model, global_round):
        global_model = copy.deepcopy(model)

        if self.historic_global_model is None:
            attack_model = copy.deepcopy(model)
        else:
            attack_model = copy.deepcopy(self.historic_global_model)

        lr = self.args.lr/10

        if self.args.opt == 'adam':
            optimizer = torch.optim.Adam(attack_model.parameters(), lr=lr)
        elif self.args.opt == 'sgd':
            optimizer = torch.optim.SGD(attack_model.parameters(), lr=lr,
                                            momentum=self.args.momentum)


        for batch_idx, (xc, xp, y) in enumerate(self.target_loader):
            xc, xp = xc.float().to(self.device), xp.float().to(self.device)
            y = y.float().to(self.device)

            attack_model.zero_grad()
            pred = attack_model(xc, xp)

            loss = self.criterion(y, pred)
            loss.backward()
            optimizer.step()


        attack_weights = attack_model.state_dict()
        global_weights = global_model.state_dict()


        if self.historic_global_model is not None:
            historic_flatten_updates = self.flatten_all_updates(global_weights,
                                                [self.historic_global_model.state_dict()])[0]
            historic_updates_norm = torch.norm(historic_flatten_updates)
        else:
            historic_updates_norm = torch.tensor(1.0)



        attack_updates_norm = torch.norm(self.flatten_all_updates(global_weights,
                                                                [attack_weights])[0])
        if self.args.apply_detection:
            if global_round == 1:
                historic_updates_norm = torch.tensor(1.0)
            gamma = np.random.uniform(10.0, 11.0)
            beta = np.random.uniform(1.0, 1.1)

            scale_up = min(gamma*historic_updates_norm.item(),
                (beta + historic_updates_norm.item()))/attack_updates_norm.item()
        else:

            gamma = 1e3
            if self.args.apply_defense is None:
                bound = np.random.uniform(1e4, 1e5)
            elif self.args.apply_defense == 'trimmed_mean':
                bound = np.random.uniform(1e5, 1e6)
            elif self.args.apply_defense == 'multi_krum' and self.args.file == 'trento.h5':
                gamma = 2000
                bound = np.random.uniform(1e6, 1e7)
            else:
                bound = np.random.uniform(1e6, 1e7)

            scale_up = min(gamma, bound/attack_updates_norm.item())


        for key in attack_weights:
            attack_weights[key] = scale_up * (attack_weights[key] - global_weights[key]) + global_weights[key]

        return attack_weights, 0.0, 0.0



'''
Data poisoning
'''


class DataPoison(object):
    def __init__(self, args, train, test, target=None, num_ensemble=4):
        self.args = args
        # self.epsilon = self.args.attack_epsilon
        self.epsilon = self.args.attack_epsilon*(train[0].max() - train[0].min())
        self.alpha = 0.1

        ## device and criterion
        self.device = 'cuda' if args.gpu else 'cpu'
        self.criterion = nn.MSELoss().to(self.device)

        ## initialize train_data, train_perts, train_loader (train_loader will load xc, xp, y, dc, dp, y)
        self.train_data = list(train)
        self.train_perts = [np.zeros_like(train[i]) for i in range(len(train))]
        self.train_loader = self.process_data(tuple(self.train_data[::]) + \
                                    tuple(self.train_perts[::]), is_train=True)

        self.train_max = [train[i].max() for i in range(len(train))]
        self.train_min = [train[i].min() for i in range(len(train))]


        ## load test and target data
        self.test_loader = self.process_data(test, is_train=False)
        if target is None:
            target = copy.deepcopy(train)
            self.target_criterion = NegMSE().to(self.device)

        self.target_loader = self.process_data(target, is_target=True)

        self.num_ensemble = num_ensemble

    def process_data(self, dataset, is_target=False, is_train=True):
        data = list(zip(*dataset))
        if is_target:
            return DataLoader(data, shuffle=False, batch_size=len(data))
        if is_train:
            return DataLoader(data, shuffle=True,
                                    batch_size=self.args.batch_size)
        else:
            return DataLoader(data, shuffle=False,
                                    batch_size=self.args.batch_size)

    '''
    Initialize surrogate model and corresponding optimizer
    '''
    def initialize_attack_models(self):
        ## attack models
        self.attack_models = {}
        for model_idx in range(self.num_ensemble):
            self.attack_models[model_idx] = self.model_initialization(model_idx)

    def model_initialization(self, pre_train_epochs=0):

        model = LSTM(self.args).to(self.device)
        model = CNNLSTM(self.args).to(self.device)

        lr = self.args.lr
        if self.args.opt == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif self.args.opt == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                        momentum=self.args.momentum)

        for epoch in range(pre_train_epochs):
            for batch_idx, (xc, xp, y, _, _, _) in enumerate(self.train_loader):
                xc, xp = xc.float().to(self.device), xp.float().to(self.device)
                y = y.float().to(self.device)
                model.zero_grad()
                pred = model(xc, xp)
                loss = self.criterion(y, pred)
                loss.backward()
                optimizer.step()

        return (model, optimizer)

    def mask_generation(self, data_size, outsize, p=0.5):
        '''
            return masks for dc and dp, and dy
            data_mask is for dc and dp; out_mask is for dy
        '''
        out_mask = np.random.binomial(size=outsize, n=1, p=p)
        out_mask = torch.from_numpy(out_mask).float().to(self.device)
        data_mask = torch.repeat_interleave(out_mask, data_size[1], dim=1).unsqueeze(2)

        out_mask.requires_grad = False
        data_mask.requires_grad = False

        return data_mask, out_mask


    def modify_data(self, num_rounds=50):
        '''
            Implemented by the child classes
        '''
        pass

    def clamp(self, X, lower_limit, upper_limit):
        return torch.max(torch.min(X, upper_limit), lower_limit)

    def checkpoint_target(self):
        ## evaluate the current surrogate models on the data
        for model_idx in self.attack_models:
            attack_model, optimizer = self.attack_models[model_idx]
            attack_model.eval()
            for target_idx, (xct, xpt, yt) in enumerate(self.target_loader):
                xct, xpt = xct.float().to(self.device), xpt.float().to(self.device)
                yt = yt.float().to(self.device)
                pred = attack_model(xct, xpt)
                mse = self.criterion(yt, pred)
                print('current MSE on the surrogate model: {}'.format(mse.item()))

            attack_model.train()

class UniformPoison(DataPoison):
    def modify_data(self, num_rounds=1):
        for i in range(len(self.train_data)):
            self.train_perts = np.random.uniform(-self.epsilon, self.epsilon,
                                                      self.train_data[i].shape)

        return [self.train_data[i] + self.train_perts[i] \
                        for i in range(len(self.train_data))]



class DataMetaPoisonAdam(DataPoison):
    def __init__(self, args, train, test, target=None, num_ensemble=2, attack_lr=10.0, mask_prob=0.8):
        super().__init__(args, train, test, target, num_ensemble)
        self.perts_momentum = [np.zeros_like(train[i]) for i in range(len(train))]
        self.perts_velocity = [np.zeros_like(train[i]) for i in range(len(train))]

        self.train_loader = self.process_data(tuple(self.train_data[::]) + \
                                    tuple(self.train_perts[::]) + \
                                    tuple(self.perts_momentum[::]) + \
                                    tuple(self.perts_velocity[::]), is_train=True)
        self.alpha = attack_lr
        self.gamma = 1.0
        self.beta1, self.beta2 = 0.9, 0.999
        self.num_ensemble = num_ensemble
        self.initialize_attack_models()

        self.mask_prob = mask_prob

    def model_initialization(self, pre_train_epochs=0):
        if self.args.surrogate_model == 'lstm':
            model = LSTM(self.args).to(self.device)
        elif self.args.surrogate_model == 'cnnlstm':
            model = CNNLSTM(self.args).to(self.device)
        else:
            raise NotImplementedError
        model.train()

        lr = self.args.lr
        if self.args.opt == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif self.args.opt == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                        momentum=self.args.momentum)

        for epoch in range(pre_train_epochs):
            for batch_idx, (xc, xp, y, _, _, _, _, _, _, _, _, _) in enumerate(self.train_loader):
                xc, xp = xc.float().to(self.device), xp.float().to(self.device)
                y = y.float().to(self.device)
                model.zero_grad()
                pred = model(xc, xp)
                loss = self.criterion(y, pred)
                loss.backward()
                optimizer.step()

        return (model, optimizer)



    def modify_data(self, num_rounds=20):

        print('---------------meta poison on the training data-------------------')

        lr = self.args.lr
        batch_size = self.args.batch_size
        epsilon = self.epsilon

        for round in tqdm.tqdm(range(num_rounds)):

            for batch_idx, (xc, xp, y, dc, dp, dy, mc, mp, my, vc, vp, vy) in enumerate(self.train_loader):
                xc, xp = xc.float().to(self.device), xp.float().to(self.device)
                y = y.float().to(self.device)
                dc, dp = dc.float().to(self.device), dp.float().to(self.device)
                dy = dy.float().to(self.device)

                mc, mp = mc.float().to(self.device), mp.float().to(self.device)
                my = my.float().to(self.device)
                vc, vp = vc.float().to(self.device), vp.float().to(self.device)
                vy = vy.float().to(self.device)

                data_mask, out_mask = self.mask_generation(dc.size(), dy.size(),
                                                               1-self.mask_prob)


                ### compute the index to store the train data and perturbations
                srt, end = batch_idx*batch_size, batch_idx*batch_size+batch_size

                ### optimize the perturbations
                dc.requires_grad = True
                dp.requires_grad = True
                dy.requires_grad = True

                grad_dc, grad_dp, grad_dy = 0, 0, 0

                for model_idx in self.attack_models:
                    attack_model, optimizer = self.attack_models[model_idx]
                    attack_model.zero_grad()
                    with torch.backends.cudnn.flags(enabled=False):
                        with higher.innerloop_ctx(attack_model, optimizer) as (metamodel, diffopt):
                            pred = metamodel(xc+data_mask*dc, xp+data_mask*dp)
                            loss = self.criterion(y+out_mask*dy, pred)
                            diffopt.step(loss)

                            metamodel.eval()
                            for target_idx, (xct, xpt, yt) in enumerate(self.target_loader):
                                xct, xpt = xct.float().to(self.device), xpt.float().to(self.device)
                                yt = yt.float().to(self.device)
                                target_loss = self.target_criterion(yt, metamodel(xct, xpt))

                            grads = autograd.grad(target_loss, [dc, dp, dy])

                            grad_dc, grad_dp, grad_dy = grad_dc + grads[0].detach(), \
                                                        grad_dp + grads[1].detach(), \
                                                        grad_dy + grads[2].detach()
                            metamodel.train()

                grad_dc, grad_dp, grad_dy = grad_dc/self.num_ensemble, \
                                            grad_dp/self.num_ensemble, \
                                            grad_dy/self.num_ensemble
                ### update perturbations
                n_iters = num_rounds*len(self.train_loader) + batch_idx
                mc = self.beta1*mc + (1-self.beta1)*grad_dc.detach()
                mp = self.beta1*mp + (1-self.beta1)*grad_dp.detach()
                my = self.beta1*my + (1-self.beta1)*grad_dy.detach()

                vc = self.beta2*vc + (1-self.beta2)*grad_dc.detach()**2
                vp = self.beta2*vp + (1-self.beta2)*grad_dp.detach()**2
                vy = self.beta2*vy + (1-self.beta2)*grad_dy.detach()**2

                update_dc = self.alpha * mc/(1-self.beta1**n_iters)/ \
                                (torch.sqrt(vc/(1-self.beta2**n_iters)) + 1e-8)

                update_dp = self.alpha * mp/(1-self.beta1**n_iters)/ \
                                (torch.sqrt(vp/(1-self.beta2**n_iters)) + 1e-8)

                update_dy = self.alpha * my/(1-self.beta1**n_iters)/ \
                                (torch.sqrt(vy/(1-self.beta2**n_iters)) + 1e-8)

                dc.data = torch.clamp(dc.data - update_dc.detach(), -epsilon, epsilon)
                dp.data = torch.clamp(dp.data - update_dp.detach(), -epsilon, epsilon)
                dy.data = torch.clamp(dy.data - update_dy.detach(), -epsilon, epsilon)



                self.train_data[0][srt:end] = xc.detach().cpu().numpy()
                self.train_data[1][srt:end] = xp.detach().cpu().numpy()
                self.train_data[2][srt:end] = y.detach().cpu().numpy()

                self.train_perts[0][srt:end] = dc.data.detach().cpu().numpy()
                self.train_perts[1][srt:end] = dp.data.detach().cpu().numpy()
                self.train_perts[2][srt:end] = dy.data.detach().cpu().numpy()


                self.perts_momentum[0][srt:end] = mc.detach().cpu().numpy()
                self.perts_momentum[1][srt:end] = mp.detach().cpu().numpy()
                self.perts_momentum[2][srt:end] = my.detach().cpu().numpy()

                self.perts_velocity[0][srt:end] = vc.detach().cpu().numpy()
                self.perts_velocity[1][srt:end] = vp.detach().cpu().numpy()
                self.perts_velocity[2][srt:end] = vy.detach().cpu().numpy()


                ## train the model on the perturbed samples
                for model_idx in self.attack_models:
                    attack_model, optimizer = self.attack_models[model_idx]
                    attack_model.zero_grad()
                    pred = attack_model(xc+data_mask*dc.detach(), xp+data_mask*dp.detach())
                    loss = self.criterion(y+out_mask*dy.detach(), pred)
                    loss.backward()
                    optimizer.step()

            # self.checkpoint_target()
            if round % 10 == 0 and round != 0:
                print('------reinitialize attack models------')
                self.initialize_attack_models()
            ## update train_loader
            self.train_loader = self.process_data(tuple(self.train_data[::]) + \
                                        tuple(self.train_perts[::]) + \
                                        tuple(self.perts_momentum[::]) + \
                                        tuple(self.perts_velocity[::]), is_train=True)
            self.alpha = self.gamma * self.alpha

        print('max perturation: {}'.format(max(np.abs(self.train_perts[0]).max(),
                                          np.abs(self.train_perts[1]).max(),
                                          np.abs(self.train_perts[2]).max())))

        return [self.train_data[i] + self.train_perts[i] \
                        for i in range(len(self.train_data))]






'''
SSGD meta learning
'''
class DataMetaPoisonSSGD(DataPoison):
    def __init__(self, args, train, test, target=None, num_ensemble=4,
                                            attack_lr=0.2, mask_prob=0.8):
        super().__init__(args, train, test, target, num_ensemble)
        self.alpha = attack_lr
        self.gamma = 0.9
        self.mask_prob = mask_prob
        self.initialize_attack_models()


    def modify_data(self, num_rounds=20):

        print('---------------meta poison on the training data-------------------')

        lr = self.args.lr
        batch_size = self.args.batch_size
        epsilon = self.epsilon
        alpha = self.alpha


        for round in tqdm.tqdm(range(num_rounds)):

            for batch_idx, (xc, xp, y, dc, dp, dy) in enumerate(self.train_loader):
                xc, xp = xc.float().to(self.device), xp.float().to(self.device)
                y = y.float().to(self.device)
                dc, dp = dc.float().to(self.device), dp.float().to(self.device)
                dy = dy.float().to(self.device)

                data_mask, out_mask = self.mask_generation(dc.size(), dy.size(),
                                                              1-self.mask_prob)


                ### compute the index to store the train data and perturbations
                srt, end = batch_idx*batch_size, batch_idx*batch_size+batch_size

                ### optimize the perturbations
                dc.requires_grad = True
                dp.requires_grad = True
                dy.requires_grad = True

                grad_dc, grad_dp, grad_dy = 0, 0, 0

                for model_idx in self.attack_models:
                    attack_model, optimizer = self.attack_models[model_idx]
                    # print(id(attack_model), id(self.attack_models[model_idx][0]))
                    attack_model.zero_grad()
                    with torch.backends.cudnn.flags(enabled=False):
                        with higher.innerloop_ctx(attack_model, optimizer) as (metamodel, diffopt):
                            pred = metamodel(xc+data_mask*dc, xp+data_mask*dp)
                            loss = self.criterion(y+out_mask*dy, pred)
                            diffopt.step(loss)


                            for target_idx, (xct, xpt, yt) in enumerate(self.target_loader):
                                xct, xpt = xct.float().to(self.device), xpt.float().to(self.device)
                                yt = yt.float().to(self.device)
                                target_loss = self.criterion(yt, metamodel(xct, xpt))

                            grads = autograd.grad(target_loss, [dc, dp, dy])
                        grad_dc, grad_dp, grad_dy = grad_dc + grads[0].detach(), \
                                                    grad_dp + grads[1].detach(), \
                                                    grad_dy + grads[2].detach()

                ### update perturbations
                dc.data = torch.clamp(dc.data - alpha*torch.sign(grad_dc.detach()), -epsilon, epsilon)
                dp.data = torch.clamp(dp.data - alpha*torch.sign(grad_dp.detach()), -epsilon, epsilon)
                dy.data = torch.clamp(dy.data - alpha*torch.sign(grad_dy.detach()), -epsilon, epsilon)


                self.train_data[0][srt:end] = xc.detach().cpu().numpy()
                self.train_data[1][srt:end] = xp.detach().cpu().numpy()
                self.train_data[2][srt:end] = y.detach().cpu().numpy()

                self.train_perts[0][srt:end] = dc.data.detach().cpu().numpy()
                self.train_perts[1][srt:end] = dp.data.detach().cpu().numpy()
                self.train_perts[2][srt:end] = dy.data.detach().cpu().numpy()


                ## train the model on the perturbed samples
                for model_idx in self.attack_models:
                    attack_model, optimizer = self.attack_models[model_idx]
                    attack_model.zero_grad()
                    pred = attack_model(xc+data_mask*dc.detach(), xp+data_mask*dp.detach())
                    loss = self.criterion(y+out_mask*dy.detach(), pred)
                    loss.backward()
                    optimizer.step()

            # self.checkpoint_target()
            if round % 10 == 0 and round != 0:
                print('------reinitialize attack models------')
                self.initialize_attack_models()
            ## update train_loader
            self.train_loader = self.process_data(tuple(self.train_data[::]) + \
                                     tuple(self.train_perts[::]), is_train=True)

            alpha = self.gamma * alpha

        
        return [self.train_data[i] + self.train_perts[i] \
                        for i in range(len(self.train_data))]




'''
Tianhang's first order algorithm
'''
class FirstOrderPoisonSSGD(DataPoison):
    def __init__(self, args, train, test, target=None, num_ensemble=4):
        super().__init__(args, train, test, target, num_ensemble)
        self.alpha = 0.2
        self.initialize_attack_models()

    def update_weights(self, model, grads):
        lr = self.args.lr
        with torch.no_grad():
            counter = 0
            for name, param in model.parameters():
                param.add_(-lr*grads[counter].detach())
                counter += 1

    def model_initialization(self, pre_train_epochs=0):
        model = LSTM(self.args).to(self.device)
        model.train()

        lr = self.args.lr
        if self.args.opt == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif self.args.opt == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                        momentum=self.args.momentum)

        optsgd = torch.optim.SGD(model.parameters(), lr=lr)

        for epoch in range(pre_train_epochs):
            for batch_idx, (xc, xp, y, _, _, _) in enumerate(self.train_loader):
                xc, xp = xc.float().to(self.device), xp.float().to(self.device)
                y = y.float().to(self.device)
                model.zero_grad()
                pred = model(xc, xp)
                loss = self.criterion(y, pred)
                loss.backward()
                optimizer.step()

        return (model, optimizer, optsgd)

    def checkpoint_target(self):
        ## evaluate the current surrogate models on the data
        for model_idx in self.attack_models:
            attack_model, _, _ = self.attack_models[model_idx]
            attack_model.eval()
            for target_idx, (xct, xpt, yt) in enumerate(self.target_loader):
                xct, xpt = xct.float().to(self.device), xpt.float().to(self.device)
                yt = yt.float().to(self.device)
                pred = attack_model(xct, xpt)
                mse = self.criterion(yt, pred)
                print('current MSE on the surrogate model: {}'.format(mse.item()))

            attack_model.train()


    def modify_data(self, num_rounds=20):

        lr = self.args.lr
        batch_size = self.args.batch_size
        epsilon = self.epsilon
        alpha = self.alpha


        for round in tqdm.tqdm(range(num_rounds)):

            for batch_idx, (xc, xp, y, dc, dp, dy) in enumerate(self.train_loader):
                xc, xp = xc.float().to(self.device), xp.float().to(self.device)
                y = y.float().to(self.device)
                dc, dp = dc.float().to(self.device), dp.float().to(self.device)
                dy = dy.float().to(self.device)

                data_mask, out_mask = self.mask_generation(dc.size(), dy.size())


                ### compute the index to store the train data and perturbations
                srt, end = batch_idx*batch_size, batch_idx*batch_size+batch_size

                ### optimize the perturbations
                dc.requires_grad = True
                dp.requires_grad = True
                dy.requires_grad = True

                grad_dc, grad_dp, grad_dy = 0, 0, 0

                for model_idx in self.attack_models:
                    attack_model, _, optsgd = self.attack_models[model_idx]

                    original_pred = attack_model(xc+data_mask*dc, xp+data_mask*dp)
                    original_loss = self.criterion(y+out_mask*dy, original_pred)
                    original_grads = autograd.grad(original_loss, [dc, dp, dy])

                    temp_state = copy.deepcopy(attack_model.state_dict())

                    for target_idx, (xct, xpt, yt) in enumerate(self.target_loader):
                        xct, xpt = xct.float().to(self.device), xpt.float().to(self.device)
                        yt = yt.float().to(self.device)

                        attack_model.zero_grad()
                        target_pred = attack_model(xct, xpt)
                        target_loss = self.criterion(yt, target_pred)
                        target_loss.backward()
                        optsgd.step()

                    desired_pred = attack_model(xc+data_mask*dc, xp+data_mask*dp)
                    desired_loss = self.criterion(y+out_mask*dy, desired_pred)
                    desired_grads = autograd.grad(desired_loss, [dc, dp, dy])

                    attack_model.load_state_dict(temp_state)


                    grad_dc, grad_dp, grad_dy = grad_dc - (desired_grads[0].detach() - original_grads[0].detach()), \
                                                grad_dp - (desired_grads[1].detach() - original_grads[1].detach()), \
                                                grad_dy - (desired_grads[2].detach() - original_grads[2].detach()),

                ### update perturbations
                dc.data = torch.clamp(dc.data + alpha*torch.sign(grad_dc.detach()), -epsilon, epsilon)
                dp.data = torch.clamp(dp.data + alpha*torch.sign(grad_dp.detach()), -epsilon, epsilon)
                dy.data = torch.clamp(dy.data + alpha*torch.sign(grad_dy.detach()), -epsilon, epsilon)


                self.train_data[0][srt:end] = xc.detach().cpu().numpy()
                self.train_data[1][srt:end] = xp.detach().cpu().numpy()
                self.train_data[2][srt:end] = y.detach().cpu().numpy()

                self.train_perts[0][srt:end] = dc.data.detach().cpu().numpy()
                self.train_perts[1][srt:end] = dp.data.detach().cpu().numpy()
                self.train_perts[2][srt:end] = dy.data.detach().cpu().numpy()


                ## train the model on the perturbed samples
                # if np.random.uniform(0, 1) > 0.8:
                for model_idx in self.attack_models:
                    attack_model, optimizer, _ = self.attack_models[model_idx]
                    attack_model.zero_grad()
                    pred = attack_model(xc+data_mask*dc.detach(), xp+data_mask*dp.detach())
                    loss = self.criterion(y+out_mask*dy.detach(), pred)
                    loss.backward()
                    optimizer.step()

            self.checkpoint_target()
            if round % 10 == 0 and round != 0:
                print('------reinitialize attack models------')
                self.initialize_attack_models()
            ## update train_loader
            self.train_loader = self.process_data(tuple(self.train_data[::]) + \
                                     tuple(self.train_perts[::]), is_train=True)

        return [self.train_data[i] + self.train_perts[i] \
                        for i in range(len(self.train_data))]







class ComparisonMetaPoison(DataPoison):
    def __init__(self, args, train, test, target=None, num_ensemble=4):
        super().__init__(args, train, test, target, num_ensemble)
        self.perts_momentum = [np.zeros_like(train[i]) for i in range(len(train))]
        self.perts_velocity = [np.zeros_like(train[i]) for i in range(len(train))]

        self.train_loader = self.process_data(tuple(self.train_data[::]) + \
                                    tuple(self.train_perts[::]) + \
                                    tuple(self.perts_momentum[::]) + \
                                    tuple(self.perts_velocity[::]), is_train=True)
        self.alpha = 10.0
        self.beta1, self.beta2 = 0.9, 0.999
        self.num_ensemble = num_ensemble
        self.initialize_attack_models()

    def model_initialization(self, pre_train_epochs=0):
        model = LSTM(self.args).to(self.device)
        model.train()

        lr = self.args.lr
        if self.args.opt == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif self.args.opt == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                        momentum=self.args.momentum)

        for epoch in range(pre_train_epochs):
            for batch_idx, (xc, xp, y, _, _, _, _, _, _, _, _, _) in enumerate(self.train_loader):
                xc, xp = xc.float().to(self.device), xp.float().to(self.device)
                y = y.float().to(self.device)
                model.zero_grad()
                pred = model(xc, xp)
                loss = self.criterion(y, pred)
                loss.backward()
                optimizer.step()

        return (model, optimizer)



    def modify_data(self, num_rounds=20):

        print('---------------meta poison on the training data-------------------')

        lr = self.args.lr
        batch_size = self.args.batch_size
        epsilon = self.epsilon

        for round in tqdm.tqdm(range(num_rounds)):

            for batch_idx, (xc, xp, y, dc, dp, dy, mc, mp, my, vc, vp, vy) in enumerate(self.train_loader):
                xc, xp = xc.float().to(self.device), xp.float().to(self.device)
                y = y.float().to(self.device)
                dc, dp = dc.float().to(self.device), dp.float().to(self.device)
                dy = dy.float().to(self.device)

                mc, mp = mc.float().to(self.device), mp.float().to(self.device)
                my = my.float().to(self.device)
                vc, vp = vc.float().to(self.device), vp.float().to(self.device)
                vy = vy.float().to(self.device)

                data_mask, out_mask = self.mask_generation(dc.size(), dy.size(), p=0.0)


                ### compute the index to store the train data and perturbations
                srt, end = batch_idx*batch_size, batch_idx*batch_size+batch_size

                ### optimize the perturbations
                dc.requires_grad = True
                dp.requires_grad = True
                dy.requires_grad = True

                grad_dc, grad_dp, grad_dy = 0, 0, 0

                for model_idx in self.attack_models:
                    attack_model, optimizer = self.attack_models[model_idx]
                    attack_model.zero_grad()
                    with torch.backends.cudnn.flags(enabled=False):
                        with higher.innerloop_ctx(attack_model, optimizer) as (metamodel, diffopt):
                            pred = metamodel(xc+data_mask*dc, xp+data_mask*dp)
                            loss = self.criterion(y+out_mask*dy, pred)
                            diffopt.step(loss)

                            metamodel.eval()
                            for target_idx, (xct, xpt, yt) in enumerate(self.target_loader):
                                xct, xpt = xct.float().to(self.device), xpt.float().to(self.device)
                                yt = yt.float().to(self.device)
                                target_loss = self.criterion(yt, metamodel(xct, xpt))

                            grads = autograd.grad(target_loss, [dc, dp, dy])

                            grad_dc, grad_dp, grad_dy = grad_dc + grads[0].detach(), \
                                                        grad_dp + grads[1].detach(), \
                                                        grad_dy + grads[2].detach()
                            metamodel.train()

                grad_dc, grad_dp, grad_dy = grad_dc/self.num_ensemble, \
                                            grad_dp/self.num_ensemble, \
                                            grad_dy/self.num_ensemble
                ### update perturbations
                n_iters = num_rounds*len(self.train_loader) + batch_idx
                mc = self.beta1*mc + (1-self.beta1)*grad_dc.detach()
                mp = self.beta1*mp + (1-self.beta1)*grad_dp.detach()
                my = self.beta1*my + (1-self.beta1)*grad_dy.detach()

                vc = self.beta2*vc + (1-self.beta2)*grad_dc.detach()**2
                vp = self.beta2*vp + (1-self.beta2)*grad_dp.detach()**2
                vy = self.beta2*vy + (1-self.beta2)*grad_dy.detach()**2

                update_dc = self.alpha * mc/(1-self.beta1**n_iters)/ \
                                (torch.sqrt(vc/(1-self.beta2**n_iters)) + 1e-8)

                update_dp = self.alpha * mp/(1-self.beta1**n_iters)/ \
                                (torch.sqrt(vp/(1-self.beta2**n_iters)) + 1e-8)

                update_dy = self.alpha * my/(1-self.beta1**n_iters)/ \
                                (torch.sqrt(vy/(1-self.beta2**n_iters)) + 1e-8)

                dc.data = torch.clamp(dc.data - update_dc.detach(), -epsilon, epsilon)
                dp.data = torch.clamp(dp.data - update_dp.detach(), -epsilon, epsilon)
                # dp.data = self.clamp(dp.data + update_dp.detach(),
                #                         self.train_min[1]-xp.data, self.train_max[1]-xp.data)
                dy.data = torch.clamp(dy.data - update_dy.detach(), -epsilon, epsilon)



                self.train_data[0][srt:end] = xc.detach().cpu().numpy()
                self.train_data[1][srt:end] = xp.detach().cpu().numpy()
                self.train_data[2][srt:end] = y.detach().cpu().numpy()

                self.train_perts[0][srt:end] = dc.data.detach().cpu().numpy()
                self.train_perts[1][srt:end] = dp.data.detach().cpu().numpy()
                self.train_perts[2][srt:end] = dy.data.detach().cpu().numpy()


                self.perts_momentum[0][srt:end] = mc.detach().cpu().numpy()
                self.perts_momentum[1][srt:end] = mp.detach().cpu().numpy()
                self.perts_momentum[2][srt:end] = my.detach().cpu().numpy()

                self.perts_velocity[0][srt:end] = vc.detach().cpu().numpy()
                self.perts_velocity[1][srt:end] = vp.detach().cpu().numpy()
                self.perts_velocity[2][srt:end] = vy.detach().cpu().numpy()


                ## train the model on the perturbed samples
                for model_idx in self.attack_models:
                    attack_model, optimizer = self.attack_models[model_idx]
                    attack_model.zero_grad()
                    pred = attack_model(xc+data_mask*dc.detach(), xp+data_mask*dp.detach())
                    loss = self.criterion(y+out_mask*dy.detach(), pred)
                    loss.backward()
                    optimizer.step()

            # self.checkpoint_target()
            if round % 10 == 0 and round != 0:
                print('------reinitialize attack models------')
                self.initialize_attack_models()
            ## update train_loader
            self.train_loader = self.process_data(tuple(self.train_data[::]) + \
                                        tuple(self.train_perts[::]) + \
                                        tuple(self.perts_momentum[::]) + \
                                        tuple(self.perts_velocity[::]), is_train=True)

        print('max perturation: {}'.format(max(np.abs(self.train_perts[0]).max(),
                                          np.abs(self.train_perts[1]).max(),
                                          np.abs(self.train_perts[2]).max())))

        return [self.train_data[i] + self.train_perts[i] \
                        for i in range(len(self.train_data))]


class NoAttack(DataPoison):

    def modify_data(self, num_rounds=10):

        lr = self.args.lr
        batch_size = self.args.batch_size
        epsilon = self.epsilon
        alpha = self.alpha

        for round in tqdm.tqdm(range(num_rounds)):

            for batch_idx, (xc, xp, y, dc, dp, dy) in enumerate(self.train_loader):
                xc, xp = xc.float().to(self.device), xp.float().to(self.device)
                y = y.float().to(self.device)

                for model_idx in self.attack_models:
                    attack_model, optimizer = self.attack_models[model_idx]
                    attack_model.zero_grad()
                    pred = attack_model(xc, xp)
                    loss = self.criterion(y, pred)
                    loss.backward()
                    optimizer.step()

            # self.checkpoint_target()

            ## update train_loader
            self.train_loader = self.process_data(tuple(self.train_data[::]) + \
                                     tuple(self.train_perts[::]), is_train=True)

        
        return [self.train_data[i] + self.train_perts[i] \
                        for i in range(len(self.train_data))]
