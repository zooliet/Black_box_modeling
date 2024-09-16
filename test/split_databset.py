#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('./')
sys.path.append('../')

import os
import pandas as pd
from sklearn.model_selection import train_test_split

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

    logger.info("Load train data")
    ROOT_DIR = '.' if os.path.exists('config') else '..' 
    original_file = os.path.join(ROOT_DIR, 'dataset', 'train.csv')
    df = pd.read_csv(original_file)
    logger.info(f"Raw dataset: {df.shape}")
    # Split the dataset into training and testing sets
    train_df, test_df = train_test_split(df, test_size=4000, random_state=42, shuffle=True)

    train_csv = os.path.join(ROOT_DIR, 'dataset', 'train.local.csv')
    train_df.to_csv(train_csv, index=False)

    test_csv = os.path.join(ROOT_DIR, 'dataset', 'test.local.csv')
    test_df.to_csv(test_csv, index=False)

    logger.info(f"Train dataset: {train_df.shape}")
    logger.info(f"Test dataset: {test_df.shape}")
    
