# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: defenses.py
-----------------------------------------------
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn import metrics
import copy

torch.manual_seed(2020)
np.random.seed(2020)


class DataPoisonDefense(object):
    def __init__(self, args):
        self.args = args

    def sphere_remove_outliers(self, train, proportion=0.3):
        '''
            Use the sphere method to remove outliers
        '''
        print('-------remove {} data from the dataset----'.format(proportion))
        train = list(train)
        l2_square_dis = 0
        for idx in range(len(train)):
            idx_mean = np.mean(train[idx], axis=0, keepdims=True)
            reduce_axis = tuple([i for i in range(1, len(idx_mean.shape))])
            l2_square_dis += np.sum(np.square(train[idx] - idx_mean), axis=reduce_axis)

        removal_partition_idx = int((1-proportion)*l2_square_dis.shape[0])
        remaining_indices = np.argpartition(l2_square_dis, removal_partition_idx)[0:removal_partition_idx]


        for idx in range(len(train)):
            train[idx] = train[idx][remaining_indices]

        return tuple(train)


    def adj_remove_outliers(self, train, proportion=0.3):
        '''
            Use the our method to remove outliers
        '''
        print('-------remove {} data from the dataset----'.format(proportion))
        train = list(train)
        adj_dis = 0
        adj_dis = np.abs(train[0][:,0,:] - train[2])
        for idx in range(1, train[0].shape[1]):
            adj_dis += np.abs(train[0][:,idx,:] - train[0][:,idx-1,:])

        adj_dis = np.squeeze(adj_dis)

        removal_partition_idx = int((1-proportion)*adj_dis.shape[0])
        remaining_indices = np.argpartition(adj_dis, removal_partition_idx)[0:removal_partition_idx]


        for idx in range(len(train)):
            train[idx] = train[idx][remaining_indices]

        return tuple(train)

    def add_noise(self, train, sigma=0.5):
        '''
            Add random noise to corrupt the patterns of poisoned data
        '''
        train = list(train)
        for idx in range(len(train)):
            train[idx] += sigma*np.random.randn(*train[idx].shape)

        return tuple(train)

class AdversaryError(Exception):
    def __init__(self):
        self.message = 'malicious client detected'


class RobustAggregation(object):
    def __init__(self, args):
        self.args = args


    def flatten_all_updates(self, global_weights, user_local_weights):
        all_user_updates = [[] for i in range(len(user_local_weights))]
        for key in global_weights.keys():
            for i in range(len(all_user_updates)):
                all_user_updates[i] = (user_local_weights[i][key] - global_weights[key]).data.view(-1) \
                                      if not len(all_user_updates[i]) else torch.cat((all_user_updates[i], \
                                                        (user_local_weights[i][key] - global_weights[key]).view(-1)))
        return all_user_updates

    def simple_detection(self, global_weights, user_local_weights):
        user_updates = self.flatten_all_updates(global_weights, user_local_weights)
        update_norms = torch.Tensor([torch.norm(model_updates)
                                        for model_updates in user_updates]).float()
        update_norms = torch.sort(update_norms)[0]

        normal_length = update_norms.size(0)

        median = torch.median(update_norms[0:normal_length])

        sigma = torch.sqrt(torch.median(torch.square(update_norms[0:normal_length] - median)))

        if update_norms.max() > 40 * median or \
                update_norms.max() > median + 400*sigma:
            raise AdversaryError



    def add_mean_updates(self, avg_updates, global_weights):
        updated_weights = copy.deepcopy(global_weights)
        start_idx = 0
        for key in updated_weights:
            updated_weights[key] += avg_updates[start_idx:start_idx+len(updated_weights[key].data.view(-1))].reshape(updated_weights[key].data.shape)
            start_idx += len(updated_weights[key].data.view(-1))

        return updated_weights


    def multi_krum(self, global_weights, user_local_weights, n_attackers, m=None):

        candidates_local_weights = []
        remaining_updates = self.flatten_all_updates(global_weights, user_local_weights)
        # print([torch.norm(model_updates) for model_updates in remaining_updates])
        all_indices = np.arange(len(remaining_updates))

        if m is None:
            m = len(remaining_updates) - 2 - n_attackers

        torch.cuda.empty_cache()
        distances = []
        for update in remaining_updates:
            distance = []
            for update_ in remaining_updates:
                distance.append(torch.norm((update - update_)) ** 2)
            distance = torch.Tensor(distance).float()
            ## distance[None, :] will expand a dim along None (10, ) -> (1, 10)
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

        distances = torch.sort(distances, dim=1)[0]
        scores = torch.sum(distances[:, :len(remaining_updates) - 2 - n_attackers], dim=1)
        indices = torch.argsort(scores)[:m]

        for idx in indices.cpu().numpy():
            candidates_local_weights.append(user_local_weights[idx])

        return candidates_local_weights

    def trimmed_mean(self, global_weights, user_local_weights, n_attackers):
        all_updates = self.flatten_all_updates(global_weights, user_local_weights)
        converted_updates = torch.cat([all_updates[i].unsqueeze(0) for i in range(len(all_updates))], dim=0)

        sorted_updates = torch.sort(converted_updates, 0)[0][n_attackers:-n_attackers]
        mean_updates = torch.mean(sorted_updates, 0)

        trimmed_mean_weights = self.add_mean_updates(mean_updates, global_weights)

        return trimmed_mean_weights

    def median(self, global_weights, user_local_weights):
        all_updates = self.flatten_all_updates(global_weights, user_local_weights)
        converted_updates = torch.cat([all_updates[i].unsqueeze(0) for i in range(len(all_updates))], dim=0)

        median_updates = torch.median(converted_updates, 0)[0]

        median_weights = self.add_mean_updates(median_updates, global_weights)

        return median_weights





def test_state_dict(model_state_dict):
    for name in model_state_dict:
        print(model_state_dict[name][0], model_state_dict[name][1])
        break


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


def scaleup_attack_weights(local_model, global_model, scale_up):
    result = copy.deepcopy(local_model)
    for key in local_model.keys():
        result[key] = scale_up * (local_model[key] - global_model[key]) + global_model[key]
    return result
