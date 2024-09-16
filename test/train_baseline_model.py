#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('./')
sys.path.append('../')

import os
import numpy as np
import pandas as pd

import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data.dataloader import default_collate

import config
from libs.data_loader import BBDataModule, BBDataset
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

    logger.info("Choose train file: ")
    dir_path = os.path.join(ROOT_DIR, 'dataset')
    file = user_select('^train(.)+csv$', dir_path)
    logger.info(f'{file}')

    logger.info(f"Load train data")
    train_csv = os.path.join(ROOT_DIR, 'dataset', file)
    df = pd.read_csv(train_csv)
    np.testing.assert_equal(df.shape[1]-2, cfg['num_input'])

    logger.info(f"Create BaselineModel instance")
    model = BaselineModel(
        num_input=cfg['num_input'], 
        num_output=cfg['num_output'], 
        layers=cfg['layers'],
        dropout=cfg['dropout'],
        learning_rate=cfg['learning_rate']
    ) 

    logger.debug(f"Model:\n{model}")
    if args['debug']:
        testset = BBDataset(csv_file=train_csv, transform=None)
        X, y = default_collate([testset[0]])
        model.eval()
        with torch.no_grad():
            y_pred = model(X)
        logger.debug(f"y_pred: {y_pred.detach().numpy()}")
        model.train()

    logger.info(f"Create BBDataModule instance")
    data_module = BBDataModule(
        csv_file=train_csv, 
        batch_size=cfg['batch_size'], 
        num_workers=cfg['num_workers']
    )

    logger.info(f"Make TensorBoardLogger at 'tb_logs/'")
    log_dir = os.path.join(ROOT_DIR, 'tb_logs')
    tb_logger = TensorBoardLogger(log_dir, name=cfg['label'])

    logger.info("Make checkpoint directory at 'models/'")
    checkpoint_dir = os.path.join(ROOT_DIR, 'models')
    checkpoint_file = f'{cfg['label']}-v{tb_logger.version}-L{'_'.join(map(str,cfg['layers']))}-d{cfg['dropout']}-' + '{epoch:02d}-{val_rmse:.4f}'
    if 'y_scaled' in train_csv:
        checkpoint_file = checkpoint_file + '-y_scaled'

    logger.info("Create ModelCheckpoint and EarlyStopping callbacks")
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=checkpoint_dir,
        filename=checkpoint_file,
        save_top_k=3,
        mode='min',
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=False,
        mode='min'
    )

    logger.info("Create Trainer instance")
    trainer = pl.Trainer(
        # limit_train_batches=0.1, # use only 10% of the training data
        min_epochs=1,
        max_epochs=cfg['num_epochs'],
        # precision='bf16-mixed',
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=tb_logger,
        # profiler=profiler,
        # profiler='simple'
    )

    logger.info("Start fitting the model...")
    trainer.fit(model, data_module)

    logger.info("Validate the model")
    trainer.validate(model, data_module)

