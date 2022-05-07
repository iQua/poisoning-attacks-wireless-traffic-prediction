# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: fed_avg.py
-----------------------------------------------
"""
import numpy as np
import h5py
import tqdm
import copy
import torch
import pandas as pd
import sys
import random

sys.path.append('../')
from utils.misc import args_parser, average_weights
from utils.misc import get_data, process_isolated, set_logger
from utils.models import LSTM
from utils.fed_update import LocalUpdate, test_inference
from utils.attacks import ModelPoison, AdaptiveModelPoison
from utils.defenses import RobustAggregation
from sklearn import metrics
import setGPU

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def reset_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

if __name__ == '__main__':
    args = args_parser()
    reset_random_seed(args.seed)

    data, _, selected_cells, mean, std, _, _ = get_data(args)

    device = 'cuda' if args.gpu else 'cpu'

    parameter_list = 'FedAvg-data-{:}-type-{:}-'.format(args.file, args.type)
    parameter_list += '-frac-{:.2f}-le-{:}-lb-{:}-seed-{:}'.format(args.frac, args.local_epoch,
                                                                   args.local_bs,
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

    if args.poison:
        m = max(int(args.adversary_frac * args.bs), 1)
        adversary_idx = [random.sample(selected_cells, 1)[0] for i in range(m)]
        historic_global_model = None

        logger.info(str(adversary_idx))
        reset_random_seed(args.seed)

    defense = RobustAggregation(args)


    ## attack model availability
    target = None

    for epoch in tqdm.tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        # print(f'\n | Global Training Round: {epoch+1} |\n')
        global_model.train()

        m = max(int(args.frac * args.bs), 1)
        cell_idx = random.sample(selected_cells, m)
        # print(cell_idx)
        historic_global_model_temp = copy.deepcopy(global_model)

        for cell in cell_idx:
            cell_train, cell_test = train[cell], test[cell]

            if args.poison and cell in adversary_idx:
                local_model = AdaptiveModelPoison(args, cell_train, cell_test,
                                                    historic_global_model, target, args.apply_defense)
            else:
                # print('normal cell: {}'.format(cell))
                local_model = LocalUpdate(args, cell_train, cell_test)

            global_model.load_state_dict(global_weights)
            global_model.train()

            w, loss, epoch_loss = local_model.update_weights(model=copy.deepcopy(global_model),
                                                             global_round=epoch)

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            cell_loss.append(loss)

        loss_hist.append(sum(cell_loss)/len(cell_loss))

        if args.apply_detection:
            defense.simple_detection(global_weights, local_weights)


        n_attackers = int(m * args.adversary_frac)

        if args.apply_defense == 'multi_krum':
            local_weights = defense.multi_krum(global_weights, local_weights, n_attackers=n_attackers)
            global_weights = average_weights(local_weights)

        elif args.apply_defense == 'trimmed_mean':
            global_weights = defense.trimmed_mean(global_weights, local_weights, n_attackers=n_attackers)
            # print('number of remaining local users for update: {}'.format(len(local_weights)))

        elif args.apply_defense == 'median':
            global_weights = defense.median(global_weights, local_weights)

        else:
            global_weights = average_weights(local_weights)

        # Update global model
        global_model.load_state_dict(global_weights)
        historic_global_model = historic_global_model_temp


    # Test model accuracy
    pred, truth = {}, {}
    test_loss_list = []
    test_mse_list = []
    nrmse = 0.0

    global_model.load_state_dict(global_weights)


    for cell in selected_cells:
        cell_test = test[cell]
        test_loss, test_mse, test_nrmse, pred[cell], truth[cell] = test_inference(args, global_model, cell_test)
        nrmse += test_nrmse

        test_loss_list.append(test_loss)
        test_mse_list.append(test_mse)

    df_pred = pd.DataFrame.from_dict(pred)
    df_truth = pd.DataFrame.from_dict(truth)

    mse = metrics.mean_squared_error(df_pred.values.ravel(), df_truth.values.ravel())
    mae = metrics.mean_absolute_error(df_pred.values.ravel(), df_truth.values.ravel())
    nrmse = nrmse / len(selected_cells)
    logger.info('FedAvg File: {:} Type: {:} MSE: {:.4f} MAE: {:.4f}, NRMSE: {:.4f}'.format(args.file, args.type, mse, mae,
                                                                                            nrmse))
