

BASELINE_MODEL = {
    'train_csv_file': 'train.csv', # 'train_pt.csv', 'train_pt_excl_y.csv', 
    'test_csv_file': 'test.csv',
    'num_input': 11,
    'num_output': 1,
    'layers': [512, 32, 8],
    # 'layers': [25, 15, 8],
    # 'layers': [16, 8, 4],
    'dropout': 0.0,
    'loss_fn': 'MSELoss',
    'num_epochs': 200,
    'batch_size': 32,
    'learning_rate': 0.0001,
    'num_workers': 4,
    'label': 'baseline',
}

