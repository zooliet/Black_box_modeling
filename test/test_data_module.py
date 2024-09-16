#!/usr/bin/env python
# coding: utf-8

import os
import sys
sys.path.append('./')
sys.path.append('../')

from libs.data_loader import BBDataset, BBDataModule

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

    logger.info(f"Load the train data")
    ROOT_DIR = '.' if os.path.exists('config') else '..' 
    csv_file = os.path.join(ROOT_DIR, 'dataset', 'train.csv')

    ## BBDataset instance 
    logger.info("Create BBDataset instance")
    dataset = BBDataset(csv_file)
    print(dataset.df.head())

    logger.info("BBDataset#__len__() test")
    print(len(dataset))

    logger.info("BBDataset#__getitem__() test")
    print(dataset[0])
    print(dataset[100])


    logger.info("Create BBDataModule instance")
    data_dir = os.path.join(ROOT_DIR, 'dataset')
    data_module = BBDataModule(csv_file=csv_file, batch_size=32, num_workers=4)
    print(type(data_module))

    logger.info("BBDataModule#setup() test")
    data_module.setup()
    print(data_module.train_ds)
    print(data_module.val_ds)
    print(data_module.test_ds)

    print(len(data_module.train_ds))
    print(len(data_module.val_ds))
    print(len(data_module.test_ds))

    logger.info("BBDataModule#train_dataloader() test")
    train_loader = data_module.train_dataloader()
    print(type(train_loader))
    for X, y in train_loader:
        print(X.shape, y.shape)

 
