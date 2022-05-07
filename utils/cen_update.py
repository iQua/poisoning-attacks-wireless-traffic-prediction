# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: fed_update.py
-----------------------------------------------
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import tqdm
from sklearn import metrics

torch.manual_seed(2020)
np.random.seed(2020)


class CentralUpdate(object):
    def __init__(self, args, train, test):
        self.args = args
        self.train_loader = self.process_data(train, is_train=True)
        self.test_loader = self.process_data(test, is_train=False)
        self.device = 'cuda' if args.gpu else 'cpu'
        self.criterion = nn.MSELoss().to(self.device)

    def process_data(self, dataset, is_train=True):
        data = list(zip(*dataset))
        if is_train:
            loader = DataLoader(data, shuffle=True, batch_size=self.args.batch_size)
        else:
            loader = DataLoader(data, shuffle=False, batch_size=self.args.batch_size)
        return loader

    def update_weights(self, model, central_epochs):
        model.train()
        epoch_loss = []
        lr = self.args.lr

        if self.args.opt == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif self.args.opt == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=self.args.momentum)

        for epoch in tqdm.tqdm(range(central_epochs)):
            batch_loss = []

            for batch_idx, (xc, xp, y) in enumerate(self.train_loader):
                xc, xp = xc.float().to(self.device), xp.float().to(self.device)
                y = y.float().to(self.device)

                model.zero_grad()
                pred = model(xc, xp)

                loss = self.criterion(y, pred)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), epoch_loss


def test_inference(args, model, dataset):
    model.eval()
    loss, mse = 0.0, 0.0
    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.MSELoss().to(device)
    data_loader = DataLoader(list(zip(*dataset)), batch_size=args.local_bs, shuffle=False)
    pred_list, truth_list = [], []

    with torch.no_grad():
        for batch_idx, (xc, xp, y) in enumerate(data_loader):
            xc, xp = xc.float().to(device), xp.float().to(device)
            y = y.float().to(device)
            pred = model(xc, xp)

            batch_loss = criterion(y, pred)
            loss += batch_loss.item()

            batch_mse = torch.mean((pred - y) ** 2)
            mse += batch_mse.item()

            pred_list.append(pred.detach().cpu())
            truth_list.append(y.detach().cpu())

    final_prediction = np.concatenate(pred_list).ravel()
    final_truth = np.concatenate(truth_list).ravel()
    nrmse= (metrics.mean_squared_error(final_truth, final_prediction) ** 0.5) / (max(final_truth) - min(final_truth))
    avg_loss = loss / len(data_loader)
    avg_mse = mse / len(data_loader)

    return avg_loss, avg_mse, nrmse, final_prediction, final_truth
