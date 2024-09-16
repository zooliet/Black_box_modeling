#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('./')
sys.path.append('../')

import pandas as pd
import numpy as np
import os
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

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

    logger.info("Select the first submission file: ")
    dir_path = os.path.join(ROOT_DIR, 'outputs')
    submission_1 = user_select('^submission(.)+csv$', dir_path)
    logger.info(submission_1)

    logger.info("Select the second submission file: ")
    submission_2 = user_select('^submission(.)+csv$', dir_path)
    logger.info(submission_2)

    logger.info(f"Compare {submission_1} and {submission_2}")

    submission_1 = pd.read_csv(os.path.join(dir_path, submission_1))
    submission_2 = pd.read_csv(os.path.join(dir_path, submission_2))

    submissions = np.stack([submission_1['y'], submission_2['y']], axis=1)
    submission_1 = submissions[:, 0]
    submission_2 = submissions[:, 1]

    cr = classification_report(submission_1, submission_2)
    cm = confusion_matrix(submission_1, submission_2)

    logger.info(f'\n{cr}\n')
    logger.info(f'\n{cm}\n')
    
