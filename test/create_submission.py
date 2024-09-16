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
import re

import config
from libs.nn import BaselineModel
from libs.utils import user_select

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
    # ap.add_argument('--wide', default=False, action=BooleanOptionalAction, help='select wide model')
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

    # logger.info(f"Select model configuration")
    cfg = config.BASELINE_MODEL
    # if args['wide']:
    #     cfg = config.BASELINE_WIDE_MODEL

    ROOT_DIR = '.' if os.path.exists('config') else '..' 

    # logger.info(f"Choose test file: ")
    # dir_path = os.path.join(ROOT_DIR, 'dataset')
    # file = user_select('^test(.)+csv$', dir_path)
    # logger.info(f"{file}")
    file = 'test.csv'

    logger.info(f"Load test data")
    test_csv = os.path.join(ROOT_DIR, 'dataset', file)
    test_df = pd.read_csv(test_csv)
    test_id = test_df.pop('ID')
    test_df['y'] = np.ones_like(test_id, dtype=np.float32) # dummy y

    logger.info(f"Load power transformer")
    if 'local' in file:
        pt_file = 'power_transformer.train.local.pkl'
    else:
        pt_file = 'power_transformer.train.pkl'

    # logger.info(f"Load power transformer")
    # dir_path = os.path.join(ROOT_DIR, 'outputs')
    # pt_file = user_select('^power(.)+pkl$', dir_path)
    # logger.info(f"{pt_file}")
    pt_file = os.path.join(ROOT_DIR, 'outputs', pt_file)
    # pt_file = os.path.join(ROOT_DIR, 'dataset', 'power_transformer.pkl')
    with open(pt_file, 'rb') as f:
        pt = pickle.load(f)

    logger.info(f"Apply power transformer to test data")
    X_test = pt.transform(test_df.values)
    X_test = X_test[:, :-1] # remove the y

    logger.info(f"Select model:")
    dir_path = os.path.join(ROOT_DIR, 'models')
    checkpoint_file = user_select('ckpt$', dir_path)
    logger.info(f"{checkpoint_file}")
    checkpoint = os.path.join(ROOT_DIR, 'models', checkpoint_file)
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
    X_test = torch.tensor(X_test, dtype=torch.float32)

    logger.info(f"Make a prediction")
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

    logger.info(f"Convert top 10% of the prediction to 1")
    y_pred_bin = (y_pred > np.percentile(y_pred, 90)).astype(np.int32)

    logger.info(f"Review submission:")
    submission = pd.DataFrame({'ID': test_id, 'y': y_pred_bin, 'y_pred': y_pred})
    print(submission.sort_values(by='y_pred', ascending=False))

    logger.info(f"Save submission as submission.csv")
    submission.pop('y_pred')
    submission_file = f'submission-{checkpoint_file[:-5]}.csv'
    submission.to_csv(os.path.join(ROOT_DIR, 'outputs', submission_file), index=False)
    # submission.to_csv(os.path.join(ROOT_DIR, 'outputs', 'submission.csv'), index=False)


