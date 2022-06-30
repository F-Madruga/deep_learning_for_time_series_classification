import os
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from torchvision import transforms
from inceptiontime import InceptionTime
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from utils import load_image_datasets, get_dataloaders, plotting_time_normalization, black_and_white_image_to_binary
from constants import TRAIN_NEW_MODELS, DATASETS_WITH_BAD_IMAGE_RESOLUTIONS, DATASETS_PATH, GPUS, RESULTS, FAST_DEV_RUN


def train_inceptiontime(datasets_parameters, data_type, results):
    # transformers
    train_transform = transforms.Compose([lambda x: black_and_white_image_to_binary(x), transforms.ToTensor(), lambda x: torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))])
    val_transform = transforms.Compose([lambda x: black_and_white_image_to_binary(x), transforms.ToTensor(), lambda x: torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))])
    test_transform = transforms.Compose([lambda x: black_and_white_image_to_binary(x), transforms.ToTensor(), lambda x: torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))])
    if data_type == 'line_plots':
        train_transform = transforms.Compose([lambda x: plotting_time_normalization(x), transforms.ToTensor(), lambda x: torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))])
        val_transform = transforms.Compose([lambda x: plotting_time_normalization(x), transforms.ToTensor(), lambda x: torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))])
        test_transform = transforms.Compose([lambda x: plotting_time_normalization(x), transforms.ToTensor(), lambda x: torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))])
    # add column for model results
    if f'inceptiontime_{data_type}_train_accuracy' not in results.columns or TRAIN_NEW_MODELS:
        print('Never trained this model')
        results[f'inceptiontime_{data_type}_train_accuracy'] = np.full((len(results)), -1).tolist()
        results[f'inceptiontime_{data_type}_val_accuracy'] = np.full((len(results)), -1).tolist()
        results[f'inceptiontime_{data_type}_test_accuracy'] = np.full((len(results)), -1).tolist()
        results.to_csv(RESULTS, index=False)
    model_parameters = pd.read_excel(open('datasets_and_model_parameters.xlsx', 'rb'), sheet_name='inceptiontime_parameters')
    # training model
    for i, dataset in datasets_parameters.iterrows():
        # model parameters
        num_epochs = int(model_parameters.iloc[i]['epochs'])
        batch_size = int(model_parameters.iloc[i]['batch_size'])
        learning_rate = float(model_parameters.iloc[i]['learning_rate'])
        num_inception_blocks = int(model_parameters.iloc[i]['num_inception_blocks'])
        n_filters = []
        for i in range(num_inception_blocks):
            n_filters.append(int(model_parameters.iloc[i]['num_filters_' + str(i + 1)]))
        use_residual = bool(model_parameters.iloc[i]['use_residual'])
        print(f'[{i + 1}/{len(datasets_parameters)}] INCEPTIONTIME {dataset["name"]}: num_epochs = {num_epochs}, batch_size = {batch_size}, learning_rate = {learning_rate}, num_inception_blocks = {num_inception_blocks}, n_filters = {n_filters}, use_residual = {use_residual}')
        if dataset["name"] not in DATASETS_WITH_BAD_IMAGE_RESOLUTIONS:
            dataset_path = os.path.join(DATASETS_PATH, dataset['name'], data_type)
            train_dataset = None
            test_dataset = None
            val_dataset = None
            train_loader = None
            test_loader = None
            val_loader = None
            has_val_dataset = os.path.isdir(os.path.join(dataset_path, 'val'))
            if has_val_dataset:
                train_dataset, val_dataset, test_dataset = load_image_datasets(dataset_path, train_transform, test_transform, val_transform)
                train_loader, val_loader, test_loader = get_dataloaders(batch_size, train_dataset, test_dataset, val_dataset)
            else:
                train_dataset, test_dataset = load_image_datasets(dataset_path, train_transform, test_transform)
                train_loader, test_loader = get_dataloaders(batch_size, train_dataset, test_dataset)
            model = None
            trainer = Trainer(gpus=GPUS, max_epochs=num_epochs, fast_dev_run=FAST_DEV_RUN)
            if results.iloc[i][f'inceptiontime_{data_type}_train_accuracy'] == -1:
                model = InceptionTime(1, int(dataset['num_classes']), learning_rate, n_filters=[32, 32, 32, 32, 32], use_residual=use_residual)
                if has_val_dataset:
                    trainer.fit(model, train_loader, val_loader)
                else:
                    trainer.fit(model, train_loader)
                train_accuracy = trainer.test(model, train_loader)[0]['test_accuracy']
                test_accuracy = trainer.test(model, train_loader)[0]['test_accuracy']
                results.loc[i,f'inceptiontime_{data_type}_train_accuracy'] = train_accuracy
                results.loc[i,f'inceptiontime_{data_type}_test_accuracy'] = test_accuracy
                if has_val_dataset:
                    val_accuracy = trainer.test(model, val_loader)[0]['test_accuracy']
                    results.loc[i,f'inceptiontime_{data_type}_val_accuracy'] = val_accuracy
                    print(f'train_accuracy = {train_accuracy}, test_accuracy = {test_accuracy}, val_accuracy = {val_accuracy}')
                else:
                    print(f'train_accuracy = {train_accuracy}, test_accuracy = {test_accuracy}')
                results.to_csv(RESULTS, index=False)
            else:
                print('Model already trained')
        else:
            print('Bad image dimension')
    results.to_csv(RESULTS, index=False)
    return results