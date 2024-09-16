#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('./')
sys.path.append('../')

import os
import pandas as pd
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
    file = user_select('^(train|test)(.)+csv$', dir_path)
    logger.info(f'{file}')

    logger.info(f"Load train data")
    train_csv = os.path.join(ROOT_DIR, 'dataset', file)
    train_df = pd.read_csv(train_csv)
    train_df = train_df.drop(columns=['ID'])

    logger.info("Select features to inspect")
    # columns = train_df.columns[1:-1] # exclude ID and y 
    # columns = ['x_0', 'x_1', 'x_2', 'x_3', 'x_4', 'y']
    columns = ['y']

    logger.info("Plot the features")
    for col in columns:
        myplt.plot_feature(train_df, col)
        myplt.plot_feature_distribution(train_df, col)
        myplt.plot_feature_to_target(train_df, col, 'y')
        myplt.plot_feature_contribution(train_df, col, 'y')
        plt.show()

    myplt.plot_heatmap(train_df, columns)
    plt.show()



