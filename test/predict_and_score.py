#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('./')
sys.path.append('../')

import os
import numpy as np
import pandas as pd
import torch
import pickle
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix    
import re

import config
from libs.nn import BaselineModel
from libs.utils import user_select
import libs.plots as myplt
import matplotlib.pyplot as plt

if __name__ == '__main__':
    ## logger 셋팅
    import logging
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s %(levelname)s \t %(message)s")
    logger = logging.getLogger(__name__)

    ## CLI 셋팅
    import argparse
    from argparse import BooleanOptionalAction
    ap = argparse.ArgumentParser()
    ap.add_argument('-v', '--verbose', type=int, default=0, help='verbose level')
    ap.add_argument('--debug', default=False, action=BooleanOptionalAction, help='debug message')

    args = vars(ap.parse_args())
    if args['verbose']:
        logger.setLevel(logging.DEBUG)

    logger.info("Started...")
    logger.debug(f"Argument: {args}")

    # # import freeze_support
    # from multiprocessing import freeze_support
    # freeze_support()

    cfg = config.BASELINE_MODEL
    # if args['wide']:
    #     cfg = config.BASELINE_WIDE_MODEL

    ROOT_DIR = '.' if os.path.exists('config') else '..' 


    logger.info("Choose test file: ")
    dir_path = os.path.join(ROOT_DIR, 'dataset')
    file = user_select('^(test|train).local.csv$', dir_path)
    # file = user_select('^test(.)+csv$', dir_path)
    logger.info(file)
    # file = 'test.local.csv'

    logger.info(f"Load test data")
    test_csv = os.path.join(ROOT_DIR, 'dataset', file)
    test_df = pd.read_csv(test_csv)
    test_id = test_df.pop('ID')
    y_true = test_df['y'].values
    # test_df['y'] = np.ones_like(test_id, dtype=np.float32) # dummy y

    logger.info(f"Load power transformer")
    if 'local' in file:
        pt_file = 'power_transformer.train.local.pkl'
    else:
        pt_file = 'power_transformer.train.pkl'

    pt_file = os.path.join(ROOT_DIR, 'outputs', pt_file)
    with open(pt_file, 'rb') as f:
        pt = pickle.load(f)

    logger.info(f"Apply power transformer to test data")
    X_test = pt.transform(test_df.values)
    X_test = X_test[:, :-1] # remove the y

    logger.info("Choose model:")
    dir_path = os.path.join(ROOT_DIR, 'models')
    checkpoint_file = user_select('ckpt$', dir_path)
    logger.info(checkpoint_file)

    logger.info(f"Load checkpoint")
    checkpoint = os.path.join(ROOT_DIR, 'models', checkpoint_file)
    # checkpoint = os.path.join(ROOT_DIR, 'models', 'baseline_model.ckpt')

    layers = re.search(r'L((.)+)-d', checkpoint_file).group(1) 
    layers = list(map(int, layers.split('_')))

    model = BaselineModel.load_from_checkpoint(
        checkpoint,
        num_input=layers[0], # cfg['num_input'],
        num_output=cfg['num_output'],
        layers=layers[1:], # cfg['layers'],
        dropout=cfg['dropout'],
        learning_rate=cfg['learning_rate']
    ).to('cpu')

    # X_test = torch.tensor(X_test, dtype=torch.float32).to('mps')
    # logger.debug(f"Random test")
    # x_test = X_test[torch.randint(0, X_test.size(0), (1,))] 
    # model.eval()
    # with torch.no_grad():
    #     y_pred = model(x_test)
    #
    logger.info(f"Make prediction for test data")
    X_test = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        y_pred = model(X_test)

    if 'y_scaled' in checkpoint_file:
        logger.info(f"Inverse the power transformer to the prediction")
        y_pred = y_pred.reshape(-1, 1)
        X_test = torch.cat((X_test, y_pred), axis=1)
        X_test = pt.inverse_transform(X_test)
        y_pred = X_test[:, -1]
        # y_pred = y_pred.cpu().numpy()
    else:
        y_pred = y_pred.numpy()

    logger.info('Review the prediction:')
    y_pred_sorted = np.sort(y_pred)[::-1]
    print(*y_pred_sorted[:10], sep='\n', end='\n...\n')
    print(*y_pred_sorted[-10:], sep='\n')

    logger.info(f"Make top 10% of the prediction to 1")
    y_pred_bin = (y_pred > np.percentile(y_pred, 90)).astype(np.int32)
    y_true_bin = (y_true > np.percentile(y_true, 95)).astype(np.int32)

    y_pred_percentile = np.percentile(y_pred, 90)
    y_true_percentile = np.percentile(y_true, 90)
    logger.debug(f'Percenile: {y_pred_percentile} vs {y_true_percentile}')

    cr = classification_report(y_true_bin, y_pred_bin)
    logger.info(f"Classification Report:\n{cr}\n")

    cm = confusion_matrix(y_true_bin, y_pred_bin)
    logger.info(f"Confusion Matrix:\n{cm}\n")

    df_y = pd.DataFrame({'ID': test_id, 'y': y_pred})
    myplt.plot_feature_distribution(df_y, 'y')
    myplt.plot_feature_distribution(test_df, 'y')
    plt.show()
