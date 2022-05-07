# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: centralized.py
-----------------------------------------------
"""
import numpy as np
import torch
import pandas as pd
import sys
import random

sys.path.append('../')
from utils.misc import args_parser
from utils.misc import get_data, process_isolated, set_logger, save_model
from utils.models import LSTM
from utils.cen_update import CentralUpdate, test_inference
from utils.attacks import (DataMetaPoisonSSGD, DataMetaPoisonAdam, 
        NoAttack, FirstOrderPoisonSSGD, UniformPoison)
from utils.defenses import DataPoisonDefense
from sklearn import metrics

import logging
import setGPU



torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    args = args_parser()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    data, _, selected_cells, mean, std, _, _ = get_data(args)

    device = 'cuda' if args.gpu else 'cpu'

    parameter_list = 'Centrailized-data-{:}-type-{:}-'.format(args.file, args.type)
    parameter_list += '-frac-{:.2f}-epoch-{:}-batch-{:}-seed-{:}'.format(args.frac,
                                                                   args.epochs,
                                                                   args.batch_size,
                                                                   args.seed)
    log_id = parameter_list
    logger = set_logger(log_id=log_id, args=args, logger_name=__name__)

    train, val, test = process_isolated(args, data)

    global_model = LSTM(args).to(device)
    global_model.train()

    global_weights = global_model.state_dict()

    best_val_loss = None
    val_loss = []
    val_acc = []
    cell_loss = []
    loss_hist = []


    m = max(int(args.adversary_frac * args.bs), 1)
    potential_adversary_idx = random.sample(selected_cells, m)

    logger.info('Potential Adversarial ID: {}'.format(' '.join(map(str, potential_adversary_idx))))

    ## aggregate training and testing data
    '''
    The structure of train[cell] is a tuple (close, period, target)
    '''
    train_data, test_data = [[], [], []], [[], [], []]
    target_data = None

    if not args.collision:
        for cell in selected_cells:

            if args.poison and cell in potential_adversary_idx:

                if args.attack_optimizer == 'adam':
                    adversary = DataMetaPoisonAdam(args, train[cell], test[cell], target_data,
                                                    args.num_ensemble, attack_lr = args.attack_lr,
                                                                        mask_prob = args.mask_prob)
                elif args.attack_optimizer == 'ssgd':
                    adversary = DataMetaPoisonSSGD(args, train[cell], test[cell], target_data, args.num_ensemble,
                                                                                        attack_lr = args.attack_lr,
                                                                                        mask_prob = args.mask_prob)
                elif args.attack_optimizer == 'fssgd':
                    adversary = FirstOrderPoisonSSGD(args, train[cell], test[cell], target_data, args.num_ensemble)
                elif args.attack_optimizer == 'uniform':
                    adversary = UniformPoison(args, train[cell], test[cell])
                else:
                    adversary = NoAttack(args, train[cell], test[cell], target_data, args.num_ensemble)
                    print('Not implemented')
                    # raise
                train[cell] = adversary.modify_data(num_rounds=args.attack_rounds)

            for idx in range(len(train_data)):
                train_data[idx].append(train[cell][idx])
                test_data[idx].append(test[cell][idx])

        for idx in range(len(train_data)):
            train_data[idx] = np.concatenate(train_data[idx], axis=0)
            test_data[idx] = np.concatenate(test_data[idx], axis=0)

        train_data, test_data = tuple(train_data), tuple(test_data)

    else:
        if args.poison:
            ## aggregate the poisoning data
            poison_train_data, poison_test_data = [[], [], []], [[], [], []]
            poison_train_idx = [0]
            train_end_idx = 0
            for cell in potential_adversary_idx:
                for idx in range(len(poison_train_data)):
                    poison_train_data[idx].append(train[cell][idx].copy())
                    poison_test_data[idx].append(test[cell][idx].copy())
                train_end_idx += train[cell][idx].shape[0]
                poison_train_idx.append(train_end_idx)

            for idx in range(len(poison_train_data)):
                poison_train_data[idx] = np.concatenate(poison_train_data[idx], axis=0)
                poison_test_data[idx] = np.concatenate(poison_test_data[idx], axis=0)


            if args.attack_optimizer == 'adam':
                adversary = DataMetaPoisonAdam(args, tuple(poison_train_data),
                                                        tuple(poison_test_data),
                                                        target_data, args.num_ensemble,
                                                        attack_lr = args.attack_lr,
                                                        mask_prob = args.mask_prob)
            elif args.attack_optimizer == 'ssgd':
                adversary = DataMetaPoisonSSGD(args, tuple(poison_train_data),
                                                        tuple(poison_test_data),
                                                        target_data, args.num_ensemble)
            elif args.attack_optimizer == 'comparison':
                adversary = DataMetaPoisonSSGD(args, tuple(poison_train_data),
                                                        tuple(poison_test_data),
                                                        target_data, args.num_ensemble)
            else:
                adversary = NoAttack(args, train[cell], test[cell], target_data, args.num_ensemble)
                print('Not implemented')
                # raise
            poison_train_data = adversary.modify_data(num_rounds=args.attack_rounds)


            for cell_idx in range(len(potential_adversary_idx)):
                cell = potential_adversary_idx[cell_idx]
                train[cell] = list(train[cell])
                for idx in range(len(poison_train_data)):
                    train[cell][idx] = poison_train_data[idx][poison_train_idx[cell_idx]: \
                                                            poison_train_idx[cell_idx+1]]
                train[cell] = tuple(train[cell])


        for cell in selected_cells:
            for idx in range(len(train_data)):
                train_data[idx].append(train[cell][idx])
                test_data[idx].append(test[cell][idx])

        for idx in range(len(train_data)):
            train_data[idx] = np.concatenate(train_data[idx], axis=0)
            test_data[idx] = np.concatenate(test_data[idx], axis=0)

        train_data, test_data = tuple(train_data), tuple(test_data)

    if args.apply_defense is None:
        pass
    elif args.apply_defense == 'sphere_sani':
        defense = DataPoisonDefense(args)
        train_data = defense.sphere_remove_outliers(train_data, args.removal_proportion)
    elif args.apply_defense == 'adj_sani':
        defense = DataPoisonDefense(args)
        train_data = defense.adj_remove_outliers(train_data, args.removal_proportion)
    elif 'rand' in args.apply_defense:
        defense = DataPoisonDefense(args)
        train_data = defense.add_noise(train_data, args.sigma)
    else:
        raise NotImplementedError




    ## train the model
    '''
    Directly use the data to update the central model
    '''

    model_operation = CentralUpdate(args, train_data, test_data)
    global_weights, _ = model_operation.update_weights(model=global_model, central_epochs=args.epochs)

    if args.poison:
        save_model(log_id, args, global_weights, selected_cells, potential_adversary_idx)
    else:
        save_model(log_id, args, global_weights, selected_cells)

    # Test model accuracy
    pred, truth = {}, {}
    potential_adv_pred, potential_adv_truth = {}, {}
    test_loss_list = []
    test_mse_list = []
    nrmse = 0.0

    global_model.load_state_dict(global_weights)

    for cell in selected_cells:
        cell_test = test[cell]
        test_loss, test_mse, test_nrmse, pred[cell], truth[cell] = test_inference(args, global_model, cell_test)
        if cell in potential_adversary_idx:
            # logger.info('Potential Adversarial ID: {} Target MSE: {} NRMSE: {}'.format(cell, test_mse, test_nrmse))
            potential_adv_pred[cell], potential_adv_truth = pred[cell], truth[cell]
        nrmse += test_nrmse

        test_loss_list.append(test_loss)
        test_mse_list.append(test_mse)

    df_pred = pd.DataFrame.from_dict(pred)
    df_truth = pd.DataFrame.from_dict(truth)

    mse = metrics.mean_squared_error(df_pred.values.ravel(), df_truth.values.ravel())
    mae = metrics.mean_absolute_error(df_pred.values.ravel(), df_truth.values.ravel())
    nrmse = nrmse / len(selected_cells)
    
    logger.info('Centrailized File: {:} Type: {:} MSE: {:.4f} MAE: {:.4f}, NRMSE: {:.4f}'.format(args.file, args.type, mse, mae,
                                                                                     nrmse))
