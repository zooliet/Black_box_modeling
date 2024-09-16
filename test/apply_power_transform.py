#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('./')
sys.path.append('../')

import os
import pandas as pd
from sklearn.preprocessing import PowerTransformer
import pickle
import matplotlib.pyplot as plt
import libs.plots as myplt
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
    ap.add_argument('-v', '--verbose', type=int, default=0, help='verbose level')
    ap.add_argument('--debug', default=False, action=BooleanOptionalAction, help='debug message')

    args = vars(ap.parse_args())
    if args['verbose']:
        logger.setLevel(logging.DEBUG)

    logger.info("Started...")
    logger.debug(f"Argument: {args}")

    ROOT_DIR = '.' if os.path.exists('config') else '..' 

    logger.info("Choose train file: ")
    dir_path = os.path.join(ROOT_DIR, 'dataset')
    pattern = '^train((?!pt).)+csv$'
    file = user_select(pattern, dir_path)
    # file = user_select('^train(.)+csv$', dir_path)
    logger.info(f'{file}')
    train_csv = os.path.join(ROOT_DIR, 'dataset', file)

    logger.info(f"Load train data")
    train_df = pd.read_csv(train_csv)

    logger.debug("Save ID and y columns temporarily")
    train_id = train_df['ID']
    train_y = train_df['y'] 

    logger.debug(f"Opt out ID")
    train_df.pop('ID')

    logger.info(f"Apply power transformer to train data")
    pt = PowerTransformer()
    X_pt = pt.fit_transform(train_df.values)
    train_df_pt = pd.DataFrame(X_pt, columns=train_df.columns)
    logger.debug("lambda_ : ", pt.lambdas_)

    logger.info(f"Save power transformer as power_transformer.pkl")
    pickle_file = f'power_transformer.{'.'.join(file.split('.')[:-1])}.pkl'
    pickle.dump(
        pt, 
        open(os.path.join(ROOT_DIR, 'outputs', pickle_file), 'wb')
        # open(os.path.join(ROOT_DIR, 'outputs', 'power_transformer.pkl'), 'wb')
    )

    if args['debug']:
        logger.debug("Reload power transformer")
        pt = pickle.load(open(os.path.join(ROOT_DIR, 'output', 'power_transformer.train.pkl'), 'rb'))
        logger.debug("lambda_ : ", pt.lambdas_)

    logger.debug("Restore ID")
    train_df.insert(loc=0, column='ID', value=train_id.values)
    train_df_pt.insert(loc=0, column='ID', value=train_id.values)

    # y column: power transformed
    logger.info(f"Save power transformed data as train*.pt.csv")
    transformed_file = f"{'.'.join(file.split('.')[:-1])}.pt.y_scaled.csv"
    train_df_pt.to_csv(os.path.join(ROOT_DIR, 'dataset', transformed_file), index=False)

    # y column: original
    train_df_pt_y = train_df_pt.copy()
    train_df_pt_y['y'] = train_y 
    transformed_file = f"{'.'.join(file.split('.')[:-1])}.pt.csv"
    train_df_pt_y.to_csv(os.path.join(ROOT_DIR, 'dataset', transformed_file), index=False)

    logger.info("Comapred the original and power transformed data")
    # columns = train_df_pt.columns[1:] # except 'ID'
    columns = ['x_0', 'x_1', 'x_2', 'x_3', 'x_4', 'y']

    for col in columns:
        myplt.plot_feature_distribution(train_df, col)
        myplt.plot_feature_distribution(train_df_pt, col)
        myplt.plot_feature_distribution(train_df_pt_y, col)

    plt.show()

